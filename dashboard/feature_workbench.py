"""Interactive Feature Selection + Grouping Workbench (Streamlit Layer 2).

Combines feature selection and categorical grouping into one agent-driven
draft. The actuary reviews the whole draft in one screen — select/deselect
variables (including ones the agent excluded), leave comments — then
re-runs the agent with all feedback at once, looping until finalised.
"""

from datetime import datetime
from pathlib import Path

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data_loader import load_dataset
from core.feature_pipeline import (
    generate_draft,
    proposal_from_config,
    refine_draft,
    save_feature_checkpoint,
)
from core.llm_client import LLMClient
from core.schemas import CategoryCluster, FeatureProposal, GroupingResponse
from core.session_logger import SessionLogger
from dashboard.approval_gate import _save_feature_decisions, _save_grouping_decisions


def render_feature_workbench(cfg: dict, config_path: Path) -> None:
    _init_state()

    if st.session_state.wb_draft is None:
        _render_readonly_summary(cfg)
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("Revise current selection", use_container_width=True):
            with st.spinner("Loading dataset..."):
                df = _get_df(cfg)
            st.session_state.wb_draft = proposal_from_config(cfg, df=df)
            st.session_state.wb_iteration += 1
            st.rerun()
        if c2.button("Start fresh proposal", use_container_width=True, type="primary"):
            _generate_fresh_draft(cfg)
            st.rerun()
        return

    _render_edit_form(cfg, config_path)
    if st.button("Discard draft and start over"):
        st.session_state.wb_draft = None
        st.rerun()


# ── State helpers ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("wb_draft", None)
    st.session_state.setdefault("wb_iteration", 0)
    st.session_state.setdefault("wb_df", None)
    st.session_state.setdefault("wb_llm", None)
    st.session_state.setdefault("wb_logger", None)
    st.session_state.setdefault("wb_session_id", None)


def _get_llm(cfg: dict) -> LLMClient | None:
    if st.session_state.wb_llm is None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Set ANTHROPIC_API_KEY in your .env file to use the workbench.")
            return None
        llm_cfg = cfg["llm"]
        st.session_state.wb_llm = LLMClient(
            api_key=api_key, model=llm_cfg["model"], temperature=llm_cfg["temperature"],
        )
    return st.session_state.wb_llm


def _get_df(cfg: dict) -> pd.DataFrame:
    if st.session_state.wb_df is None:
        st.session_state.wb_df = load_dataset(cfg["data"])
    return st.session_state.wb_df


def _get_logger() -> SessionLogger:
    if st.session_state.wb_logger is None:
        st.session_state.wb_logger = SessionLogger()
        st.session_state.wb_session_id = datetime.now().strftime("%Y-%m-%d %H:%M")
    return st.session_state.wb_logger


# ── Draft generation ───────────────────────────────────────────────────────────

def _generate_fresh_draft(cfg: dict) -> None:
    llm = _get_llm(cfg)
    if llm is None:
        return
    with st.spinner("Generating feature selection + grouping draft..."):
        df = _get_df(cfg)
        draft = generate_draft(llm, df, cfg["data"], cfg.get("grouping", {}))
    st.session_state.wb_draft = draft
    st.session_state.wb_iteration += 1
    _get_logger().log(
        "feature_proposal", stage="feature_selection", iteration=st.session_state.wb_iteration,
        numeric=[f.model_dump() for f in draft.numeric],
        categorical=[f.model_dump() for f in draft.categorical],
        excluded=list(draft.excluded), exclusion_rationale=draft.exclusion_rationale,
    )


# ── Read-only summary (shown when no draft is in progress) ────────────────────

def _render_readonly_summary(cfg: dict) -> None:
    features = cfg.get("features", {})
    numeric = features.get("numeric", [])
    categorical = features.get("categorical", [])

    if not numeric and not categorical:
        st.info("No feature selection checkpoint yet. Generate a first draft below.")
        return

    st.subheader("Approved Numeric Features")
    rows = [{"Feature": f["name"], "Description": f.get("description", "")} for f in numeric]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Approved Categorical Features")
    for f in categorical:
        grouping = f.get("grouping") or {}
        with st.expander(f"**{f['name']}** — {len(grouping)} clusters"):
            st.caption(f.get("description", ""))
            crows = [
                {"Cluster": k, "Original Values": "  |  ".join(str(e) for e in v), "# Values": len(v)}
                for k, v in grouping.items()
            ]
            st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)


# ── Edit form ───────────────────────────────────────────────────────────────────

def _column_kind(df: pd.DataFrame | None, col: str) -> str:
    if df is None or col not in df.columns:
        return "unknown"
    return "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"


def _feature_card(
    name: str,
    kind: str,
    description: str,
    data_quality_note: str | None,
    default_checked: bool,
    actuary_note: str | None,
    iteration: int,
    grouping: dict[str, list[str]] | None = None,
    exclusion_note: str | None = None,
) -> tuple[bool, str]:
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        checked = c1.checkbox(
            "Include", value=default_checked, key=f"iter{iteration}_include_{name}",
        )
        c2.markdown(f"**{name}**  ·  _{kind}_")
        if description:
            c2.markdown(f"**Rationale:** {description}")
        if exclusion_note:
            c2.caption(f"Why not proposed: {exclusion_note}")
        if data_quality_note:
            c2.caption(f"⚠️ {data_quality_note}")
        if grouping:
            with st.expander(f"{len(grouping)} clusters"):
                rows = [
                    {"Cluster": k, "Original Values": "  |  ".join(str(e) for e in v), "# Values": len(v)}
                    for k, v in grouping.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        comment = st.text_area(
            "Comment for agent", value=actuary_note or "",
            key=f"iter{iteration}_comment_{name}", height=68,
        )
    return checked, comment


def _render_edit_form(cfg: dict, config_path: Path) -> None:
    draft: FeatureProposal = st.session_state.wb_draft
    it = st.session_state.wb_iteration
    st.caption(f"Draft iteration {it} — review, comment, then re-run or finalize.")

    checkbox_state: dict[str, bool] = {}
    comment_state: dict[str, str] = {}
    excluded_state: dict[str, tuple[bool, str]] = {}

    with st.form("workbench_form"):
        tab_numeric, tab_categorical, tab_excluded = st.tabs([
            f"Numerical ({len(draft.numeric)})",
            f"Categorical ({len(draft.categorical)})",
            f"Not Proposed ({len(draft.excluded)})",
        ])

        with tab_numeric:
            for feat in draft.numeric:
                checked, comment = _feature_card(
                    feat.name, "numeric", feat.description, feat.data_quality_note,
                    bool(feat.approved), feat.actuary_note, it,
                )
                checkbox_state[feat.name] = checked
                comment_state[feat.name] = comment

        with tab_categorical:
            for feat in draft.categorical:
                checked, comment = _feature_card(
                    feat.name, "categorical", feat.description, feat.data_quality_note,
                    bool(feat.approved), feat.actuary_note, it, grouping=feat.grouping,
                )
                checkbox_state[feat.name] = checked
                comment_state[feat.name] = comment

        with tab_excluded:
            if not draft.excluded:
                st.caption("Nothing excluded — every dataset column is currently proposed.")
            df_for_kind = st.session_state.wb_df
            for col in draft.excluded:
                checked, comment = _feature_card(
                    col, _column_kind(df_for_kind, col), draft.excluded_description.get(col, ""),
                    None, False, "", it,
                    exclusion_note=draft.exclusion_rationale.get(col, ""),
                )
                excluded_state[col] = (checked, comment)

        col_rerun, col_finalize = st.columns(2)
        submit_rerun = col_rerun.form_submit_button("Save & Re-run agent", use_container_width=True)
        submit_finalize = col_finalize.form_submit_button(
            "Finalize", use_container_width=True, type="primary",
        )

    if submit_rerun or submit_finalize:
        _handle_submit(
            cfg, config_path, draft, checkbox_state, comment_state, excluded_state,
            finalize=submit_finalize,
        )


def _handle_submit(
    cfg: dict,
    config_path: Path,
    draft: FeatureProposal,
    checkbox_state: dict[str, bool],
    comment_state: dict[str, str],
    excluded_state: dict[str, tuple[bool, str]],
    finalize: bool,
) -> None:
    all_feats = list(draft.numeric) + list(draft.categorical)
    remarks: dict[str, str] = {}

    for feat in all_feats:
        new_approved = checkbox_state[feat.name]
        if new_approved != bool(feat.approved):
            feat.approved = new_approved
        new_comment = comment_state[feat.name].strip()
        if new_comment:
            feat.actuary_note = new_comment
            remarks[feat.name] = new_comment

    for col, (checked, comment) in excluded_state.items():
        if checked:
            remarks[col] = comment.strip() or "Actuary requests including this variable in the model."

    logger = _get_logger()
    session_id = st.session_state.wb_session_id or datetime.now().strftime("%Y-%m-%d %H:%M")
    if remarks:
        logger.log(
            "feature_remarks", stage="feature_selection",
            iteration=st.session_state.wb_iteration, remarks=remarks,
        )

    if remarks:
        llm = _get_llm(cfg)
        if llm is None:
            return
        spinner_msg = (
            "Sending feedback to agent for one final revision..." if finalize
            else "Sending feedback to agent for a revised draft..."
        )
        with st.spinner(spinner_msg):
            df = _get_df(cfg)
            draft = refine_draft(llm, df, cfg["data"], cfg.get("grouping", {}), draft, remarks)
        st.session_state.wb_iteration += 1
        logger.log(
            "feature_proposal", stage="feature_selection", iteration=st.session_state.wb_iteration,
            numeric=[f.model_dump() for f in draft.numeric],
            categorical=[f.model_dump() for f in draft.categorical],
            excluded=list(draft.excluded), exclusion_rationale=draft.exclusion_rationale,
        )

    if not finalize:
        st.session_state.wb_draft = draft
        if not remarks:
            st.info("Approval flags updated — no comments to send to the agent.")
        st.rerun()
        return

    save_feature_checkpoint(config_path, cfg, draft)

    approved_names = [f.name for f in (list(draft.numeric) + list(draft.categorical)) if f.approved is True]
    logger.log(
        "feature_selection_complete", stage="feature_selection",
        iterations=st.session_state.wb_iteration, approved=approved_names,
    )
    for cat in draft.categorical:
        if cat.approved and cat.grouping:
            logger.log(
                "grouping_complete", stage="grouping", col_name=cat.name,
                iterations=st.session_state.wb_iteration, final_clusters=cat.grouping,
            )

    _save_feature_decisions(draft, session_id)
    for cat in draft.categorical:
        if cat.grouping:
            response = GroupingResponse(clusters=[
                CategoryCluster(cluster_name=k, elements=v, rationale="")
                for k, v in cat.grouping.items()
            ])
            _save_grouping_decisions(cat.name, response, session_id)

    st.session_state.wb_draft = None
    st.cache_data.clear()
    st.success("Checkpoint saved — downstream GBM/GLM stages will use this feature set.")
    st.rerun()
