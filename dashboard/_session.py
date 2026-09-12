"""Shared Streamlit session-state helpers: LLM client, cached dataframe, session logger.

One instance of each lives per browser session, reused across every dashboard
tab that needs to call an agent or write to the audit log (the feature
workbench, the GBM retrain control, and — later — the GLM distillation
workbench), so a single dashboard visit produces one coherent session log.
"""

from datetime import datetime
from pathlib import Path

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data_loader import load_dataset
from core.llm_client import LLMClient
from core.run_scope import sessions_dir as run_sessions_dir
from core.session_logger import SessionLogger


def init_state() -> None:
    st.session_state.setdefault("dash_df", None)
    st.session_state.setdefault("dash_llm", None)
    st.session_state.setdefault("dash_logger", None)
    st.session_state.setdefault("dash_session_id", None)


LLM_NOT_CONFIGURED_MSG = (
    "ANTHROPIC_API_KEY is not set — agent actions (Regenerate from scratch, "
    "Update/Finalize with a remark) are unavailable this session. Provide it "
    "via a `.env` file locally, or a Secret on a hosted deploy, to enable "
    "them. Read-only tabs and existing checkpoints still work."
)


def llm_available() -> bool:
    """Cheap, side-effect-free peek at whether `get_llm` would actually
    succeed — lets a caller show a proactive notice (same idea as the
    dataset-missing sidebar warning) instead of only failing the moment an
    agent action is clicked."""
    load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_llm(cfg: dict) -> LLMClient | None:
    init_state()
    if st.session_state.dash_llm is None:
        if not llm_available():
            st.error(LLM_NOT_CONFIGURED_MSG)
            return None
        llm_cfg = cfg["llm"]
        st.session_state.dash_llm = LLMClient(
            api_key=os.environ["ANTHROPIC_API_KEY"], model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
        )
    return st.session_state.dash_llm


def get_df(cfg: dict) -> pd.DataFrame | None:
    """Returns None (after a friendly st.error) if the raw dataset file itself
    isn't available — e.g. a hosted demo deployment, where data/*.csv is
    deliberately gitignored and never committed (see .gitignore). Every caller
    must check for None: the read-only card/locked-view rendering path already
    tolerates it (see `_column_kind`/`_stats_line`/`_render_leakage_warning` in
    feature_workbench.py), but anything that actually needs real rows (agent
    proposal generation, GBM training, GLM fitting) must bail out cleanly
    instead, same shape as the `if llm is None: return` guard on `get_llm`.
    """
    init_state()
    if st.session_state.dash_df is None:
        try:
            st.session_state.dash_df = load_dataset(cfg["data"])
        except (FileNotFoundError, OSError):
            st.error(
                f"Dataset file not found at `{cfg['data'].get('path')}`. Expected in a "
                "hosted demo — the raw data is never committed to the repo."
            )
            return None
    return st.session_state.dash_df


def demo_mode_notices(cfg: dict) -> list[str]:
    """Runtime-availability check for the two things a hosted demo
    deliberately ships without — the raw dataset file and
    `ANTHROPIC_API_KEY` — so a visitor sees a clear banner above the main
    tabs before picking one, rather than only after clicking into a tab that
    needs either. Cheap and side-effect-free (no `st.error`/`st.stop`); the
    caller decides how to render the result. Distinct from
    `validate_config`'s own sidebar warnings, which cover config-shape
    issues (e.g. a stale seed-file column reference) rather than runtime
    environment availability.
    """
    notices = []
    dataset_path = Path(cfg["data"]["path"])
    if not dataset_path.exists():
        notices.append(
            f"Dataset file not found at `{dataset_path}` — Feature Selection, "
            "GBM training, and GLM fitting are unavailable this session. "
            "Read-only tabs (Overview, GLM Results, Audit Trail) and existing "
            "checkpoints still work."
        )
    if not llm_available():
        notices.append(LLM_NOT_CONFIGURED_MSG)
    return notices


def get_logger(config_path: Path) -> SessionLogger:
    init_state()
    if st.session_state.dash_logger is None:
        st.session_state.dash_logger = SessionLogger(sessions_dir=run_sessions_dir(config_path))
        st.session_state.dash_session_id = datetime.now().strftime("%Y-%m-%d %H:%M")
    return st.session_state.dash_logger


def get_session_id(config_path: Path) -> str:
    get_logger(config_path)  # ensures dash_session_id is set
    return st.session_state.dash_session_id
