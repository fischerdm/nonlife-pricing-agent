"""Shared Streamlit session-state helpers: LLM client, cached dataframe, session logger.

One instance of each lives per browser session, reused across every dashboard
tab that needs to call an agent or write to the audit log (the feature
workbench, the GBM retrain control, and — later — the GLM distillation
workbench), so a single dashboard visit produces one coherent session log.
"""

from datetime import datetime

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data_loader import load_dataset
from core.llm_client import LLMClient
from core.session_logger import SessionLogger


def init_state() -> None:
    st.session_state.setdefault("dash_df", None)
    st.session_state.setdefault("dash_llm", None)
    st.session_state.setdefault("dash_logger", None)
    st.session_state.setdefault("dash_session_id", None)


def get_llm(cfg: dict) -> LLMClient | None:
    init_state()
    if st.session_state.dash_llm is None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Set ANTHROPIC_API_KEY in your .env file to use the agent.")
            return None
        llm_cfg = cfg["llm"]
        st.session_state.dash_llm = LLMClient(
            api_key=api_key, model=llm_cfg["model"], temperature=llm_cfg["temperature"],
        )
    return st.session_state.dash_llm


def get_df(cfg: dict) -> pd.DataFrame:
    init_state()
    if st.session_state.dash_df is None:
        st.session_state.dash_df = load_dataset(cfg["data"])
    return st.session_state.dash_df


def get_logger() -> SessionLogger:
    init_state()
    if st.session_state.dash_logger is None:
        st.session_state.dash_logger = SessionLogger()
        st.session_state.dash_session_id = datetime.now().strftime("%Y-%m-%d %H:%M")
    return st.session_state.dash_logger


def get_session_id() -> str:
    get_logger()  # ensures dash_session_id is set
    return st.session_state.dash_session_id
