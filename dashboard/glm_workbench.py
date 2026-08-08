"""Interactive GLM Distillation Workbench (Streamlit Layer 2).

Mirrors the Feature & Grouping Workbench: the distillation agent proposes GLM
terms (main effects for every approved feature + pairwise interactions ranked
by the GBM's H-statistics), the actuary reviews every term as a card —
include/exclude, leave a comment — then re-runs the agent with all feedback
at once, looping until finalized. Finalizing writes glm_config.yaml, which
the GLM fitting step reads from.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.distillation_pipeline import generate_glm_draft, refine_glm_draft
from core.glm_pipeline import proposal_from_glm_config, save_glm_checkpoint
from core.schemas import GLMProposal, GLMTerm
from core.seed_config import DISTILLATION_SEED_FILENAME, load_distillation_seed
from dashboard import _session
from dashboard.approval_gate import _save_glm_decisions


def render_glm_workbench(cfg: dict, glm_config_path: Path) -> None:
    _session.init_state()
    _init_state()

    if not cfg.get("gbm_output", {}).get("interactions"):
        st.info("Train the GBM first (GBM tab) — GLM distillation reads its H-statistics.")
        return

    if st.session_state.glm_draft is None:
        existing = proposal_from_glm_config(glm_config_path)
        _render_readonly_summary(existing)
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button(
            "Revise current selection", use_container_width=True, disabled=existing is None,
            key="glm_revise_btn",
        ):
            st.session_state.glm_draft = existing
            st.session_state.glm_seed = load_distillation_seed(
                glm_config_path.parent / DISTILLATION_SEED_FILENAME,
            )
            st.session_state.glm_iteration += 1
            st.rerun()
        if c2.button("Start fresh proposal", use_container_width=True, type="primary", key="glm_fresh_btn"):
            _generate_fresh_draft(cfg, glm_config_path)
            st.rerun()
        return

    _render_edit_form(cfg, glm_config_path)
    if st.button("Discard draft and start over", key="glm_discard_btn"):
        st.session_state.glm_draft = None
        st.rerun()


# ── State helpers ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("glm_draft", None)
    st.session_state.setdefault("glm_iteration", 0)
    st.session_state.setdefault("glm_seed", None)


def _approved_feature_names(cfg: dict) -> list[str]:
    features = cfg.get("features", {})
    return (
        [f["name"] for f in features.get("numeric", [])]
        + [f["name"] for f in features.get("categorical", [])]
    )


# ── Draft generation ───────────────────────────────────────────────────────────

def _generate_fresh_draft(cfg: dict, glm_config_path: Path) -> None:
    llm = _session.get_llm(cfg)
    if llm is None:
        return
    data_cfg = cfg["data"]
    seed = load_distillation_seed(glm_config_path.parent / DISTILLATION_SEED_FILENAME)
    st.session_state.glm_seed = seed
    with st.spinner("Proposing GLM terms from GBM H-statistics..."):
        draft = generate_glm_draft(
            llm, cfg["gbm_output"]["interactions"], _approved_feature_names(cfg), data_cfg, seed=seed,
        )
    st.session_state.glm_draft = draft
    st.session_state.glm_iteration += 1
    _session.get_logger().log(
        "glm_term_proposal", stage="glm_distillation", iteration=st.session_state.glm_iteration,
        terms=[t.model_dump() for t in draft.terms],
    )


# ── Read-only summary (shown when no draft is in progress) ────────────────────

def _render_readonly_summary(proposal: GLMProposal | None) -> None:
    if proposal is None:
        st.info("No GLM distillation checkpoint yet. Generate a first draft below.")
        return

    approved = [t for t in proposal.terms if t.approved is True]
    st.subheader("Approved GLM Terms")
    rows = [{"Term": t.name, "Type": t.term_type, "Rationale": t.rationale} for t in approved]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if proposal.formula:
        st.caption("Patsy formula")
        st.code(proposal.formula, language=None)


# ── Edit form ───────────────────────────────────────────────────────────────────

def _term_card(term: GLMTerm, iteration: int) -> tuple[bool, str]:
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        checked = c1.checkbox(
            "Include", value=bool(term.approved), key=f"iter{iteration}_include_{term.name}",
        )
        c2.markdown(f"**{term.name}**  ·  _{term.term_type}_")
        if term.h_statistic is not None:
            c2.caption(f"📊 H-statistic: {term.h_statistic:.4f}")
        if term.rationale:
            c2.markdown(f"**Rationale:** {term.rationale}")
        comment = st.text_area(
            "Comment for agent", value=term.actuary_note or "",
            key=f"iter{iteration}_comment_{term.name}", height=68,
        )
    return checked, comment


def _render_edit_form(cfg: dict, glm_config_path: Path) -> None:
    draft: GLMProposal = st.session_state.glm_draft
    it = st.session_state.glm_iteration
    st.caption(f"Draft iteration {it} — review, comment, then re-run or finalize.")

    main_terms = [t for t in draft.terms if t.term_type == "main"]
    interaction_terms = [t for t in draft.terms if t.term_type != "main"]

    checkbox_state: dict[str, bool] = {}
    comment_state: dict[str, str] = {}

    with st.form("glm_workbench_form"):
        tab_main, tab_interactions = st.tabs([
            f"Main Effects ({len(main_terms)})",
            f"Interactions ({len(interaction_terms)})",
        ])

        with tab_main:
            for term in main_terms:
                checked, comment = _term_card(term, it)
                checkbox_state[term.name] = checked
                comment_state[term.name] = comment

        with tab_interactions:
            st.caption(
                "⚠️ Standard actuarial practice: an interaction term should only stay in the "
                "model alongside the main effects of its constituent variables — those are "
                "proposed for every approved feature in the Main Effects tab, not just the "
                "ones involved in an interaction. Excluding a main effect while keeping its "
                "interaction is rarely defensible and should have an explicit rationale."
            )
            for term in interaction_terms:
                checked, comment = _term_card(term, it)
                checkbox_state[term.name] = checked
                comment_state[term.name] = comment

        col_rerun, col_finalize = st.columns(2)
        submit_rerun = col_rerun.form_submit_button(
            "Save & Re-run agent", use_container_width=True, key="glm_rerun_btn",
        )
        submit_finalize = col_finalize.form_submit_button(
            "Finalize", use_container_width=True, type="primary", key="glm_finalize_btn",
        )

    if submit_rerun or submit_finalize:
        _handle_submit(cfg, glm_config_path, draft, checkbox_state, comment_state, finalize=submit_finalize)


def _handle_submit(
    cfg: dict,
    glm_config_path: Path,
    draft: GLMProposal,
    checkbox_state: dict[str, bool],
    comment_state: dict[str, str],
    finalize: bool,
) -> None:
    data_cfg = cfg["data"]
    remarks: dict[str, str] = {}

    for term in draft.terms:
        new_approved = checkbox_state[term.name]
        if new_approved != bool(term.approved):
            term.approved = new_approved
        new_comment = comment_state[term.name].strip()
        if new_comment:
            term.actuary_note = new_comment
            remarks[term.name] = new_comment

    logger = _session.get_logger()
    session_id = _session.get_session_id()
    if remarks:
        logger.log(
            "glm_term_remarks", stage="glm_distillation",
            iteration=st.session_state.glm_iteration, remarks=remarks,
        )

    if remarks:
        llm = _session.get_llm(cfg)
        if llm is None:
            return
        spinner_msg = (
            "Sending feedback to agent for one final revision..." if finalize
            else "Sending feedback to agent for a revised draft..."
        )
        with st.spinner(spinner_msg):
            draft = refine_glm_draft(
                llm, draft, remarks, data_cfg, seed=st.session_state.glm_seed,
            )
        st.session_state.glm_iteration += 1
        logger.log(
            "glm_term_proposal", stage="glm_distillation", iteration=st.session_state.glm_iteration,
            terms=[t.model_dump() for t in draft.terms],
        )

    if not finalize:
        st.session_state.glm_draft = draft
        if not remarks:
            st.info("Approval flags updated — no comments to send to the agent.")
        st.rerun()
        return

    formula = save_glm_checkpoint(glm_config_path, data_cfg, draft)

    approved_terms = [t.name for t in draft.terms if t.approved is True]
    logger.log(
        "glm_distillation_complete", stage="glm_distillation",
        iterations=st.session_state.glm_iteration, approved_terms=approved_terms,
    )
    _save_glm_decisions(draft, session_id)

    st.session_state.glm_draft = None
    st.cache_data.clear()
    st.success(f"GLM checkpoint saved — formula: `{formula}`")
    st.rerun()
