"""In-dashboard GLM coefficient review gate (Streamlit Layer 2).

Fits the GLM from the approved glm_config.yaml formula, then lets the actuary
review each term's coefficients (sign, significance, CI) and reject terms —
the same reject-and-auto-refit loop as dashboard.approval_gate.run_glm_coef_gate,
as Streamlit cards instead of a CLI prompt. Logs the same event names/shapes
as that CLI gate so the GLM Results and Audit Trail tabs need no changes to
pick up a dashboard-driven fit.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.feature_pipeline import apply_groupings, proposal_from_config
from core.glm_pipeline import proposal_from_glm_config
from core.schemas import GLMTerm
from dashboard import _session
from dashboard.approval_gate import _save_glm_coef_decisions
from tools.glm_tools import build_formula, coef_summary, fit_glm, param_to_term


def render_glm_coef_review(cfg: dict, glm_config_path: Path) -> None:
    _session.init_state()
    _init_state()

    approved_proposal = proposal_from_glm_config(glm_config_path)
    if approved_proposal is None:
        st.info("Finalize the GLM Distillation Workbench first — no approved terms yet.")
        return

    approved_names = frozenset(t.name for t in approved_proposal.terms if t.approved is True)
    if (
        st.session_state.coef_active_terms is not None
        and st.session_state.coef_source_terms != approved_names
    ):
        st.warning("The GLM Distillation checkpoint changed since this review started — resetting.")
        _reset_state()

    if st.session_state.coef_active_terms is None:
        st.caption(
            f"{len(approved_proposal.terms)} approved term(s) from distillation. "
            "Fit the GLM to begin coefficient review."
        )
        if st.button("📐 Fit GLM"):
            _run_initial_fit(cfg, approved_proposal.terms)
            st.session_state.coef_source_terms = approved_names
            st.rerun()
        return

    _render_review_form(cfg)


# ── State helpers ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("coef_result", None)
    st.session_state.setdefault("coef_active_terms", None)
    st.session_state.setdefault("coef_source_terms", None)
    st.session_state.setdefault("coef_iteration", 0)
    st.session_state.setdefault("coef_df", None)


def _reset_state() -> None:
    st.session_state.coef_result = None
    st.session_state.coef_active_terms = None
    st.session_state.coef_source_terms = None
    st.session_state.coef_iteration = 0
    st.session_state.coef_df = None


def _grouped_df(cfg: dict) -> pd.DataFrame:
    """Cache the grouped dataframe across refits within one review session."""
    if st.session_state.coef_df is None:
        df = _session.get_df(cfg)
        proposal = proposal_from_config(cfg)
        st.session_state.coef_df = apply_groupings(df, proposal)
    return st.session_state.coef_df


# ── Fitting ─────────────────────────────────────────────────────────────────────

def _fit(cfg: dict, terms: list[GLMTerm]):
    data_cfg = cfg["data"]
    formula = build_formula(data_cfg["target_col"], terms)
    df = _grouped_df(cfg)
    return fit_glm(
        df=df, formula=formula,
        target_col=data_cfg["target_col"], exposure_col=data_cfg["exposure_col"],
        family=data_cfg["objective"],
    ), formula


def _run_initial_fit(cfg: dict, terms: list[GLMTerm]) -> None:
    with st.spinner("Fitting GLM..."):
        result, formula = _fit(cfg, terms)
    st.session_state.coef_result = result
    st.session_state.coef_active_terms = list(terms)
    st.session_state.coef_iteration = 0

    summary_df = coef_summary(result)
    _session.get_logger().log(
        "glm_fit", stage="glm",
        formula=formula, aic=float(result.aic),
        deviance_explained=float(1 - result.deviance / result.null_deviance),
        coefficients=summary_df.to_dict(orient="records"),
    )


# ── Review form ─────────────────────────────────────────────────────────────────

def _render_review_form(cfg: dict) -> None:
    result = st.session_state.coef_result
    active_terms: list[GLMTerm] = st.session_state.coef_active_terms
    it = st.session_state.coef_iteration + 1

    summary = coef_summary(result)
    summary["term"] = summary["parameter"].map(param_to_term)

    _session.get_logger().log(
        "glm_coef_review", stage="glm_coefficient_review", iteration=it,
        coefficients=summary.to_dict(orient="records"),
        active_terms=[t.name for t in active_terms],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deviance Explained", f"{1 - result.deviance / result.null_deviance:.2%}")
    c2.metric("AIC", f"{result.aic:,.0f}")
    c3.metric("Terms under review", len(active_terms))
    c4.metric("Review pass", it)
    st.caption("Keep or reject each term below, based on sign, significance, and CI.")

    keep_state: dict[str, bool] = {}
    note_state: dict[str, str] = {}

    with st.form("glm_coef_review_form"):
        for term in active_terms:
            term_params = summary[summary["term"] == term.name]
            keep_state[term.name], note_state[term.name] = _term_coef_card(term, term_params, it)

        submit = st.form_submit_button("Submit review", type="primary", use_container_width=True)

    if submit:
        _handle_review_submit(cfg, result, active_terms, keep_state, note_state, it)


def _term_coef_card(term: GLMTerm, params_df: pd.DataFrame, iteration: int) -> tuple[bool, str]:
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        keep = c1.checkbox("Keep", value=True, key=f"coef_iter{iteration}_keep_{term.name}")
        c2.markdown(f"**{term.name}**  ·  _{term.term_type}_")
        if term.rationale:
            c2.caption(term.rationale)

        rows = [
            {
                "Parameter": row["parameter"],
                "Coef": round(row["coef"], 4),
                "exp(Coef)": round(row["exp_coef"], 4),
                "p-value": round(row["p_value"], 6),
                "95% CI (exp)": f"[{row['ci_lower_exp']:.4f}, {row['ci_upper_exp']:.4f}]",
                "Sig.": "*" if row["p_value"] < 0.05 else "",
            }
            for _, row in params_df.iterrows()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        note = st.text_area(
            "Note (optional — recorded for audit, e.g. reason for rejection)",
            value="", key=f"coef_iter{iteration}_note_{term.name}", height=60,
        )
    return keep, note


def _handle_review_submit(
    cfg: dict,
    result,
    active_terms: list[GLMTerm],
    keep_state: dict[str, bool],
    note_state: dict[str, str],
    iteration: int,
) -> None:
    logger = _session.get_logger()
    session_id = _session.get_session_id()
    rejected = {name for name, kept in keep_state.items() if not kept}

    for term in active_terms:
        logger.log(
            "glm_coef_decision", stage="glm_coefficient_review",
            term=term.name, decision="rejected" if term.name in rejected else "kept",
            note=note_state.get(term.name, ""),
        )

    remaining_terms = [t for t in active_terms if t.name not in rejected]

    if not rejected:
        logger.log(
            "glm_coef_review_complete", stage="glm_coefficient_review",
            iterations=iteration, final_terms=[t.name for t in active_terms],
        )
        _save_glm_coef_decisions(active_terms, session_id)
        _log_rating_factors(result)
        _reset_state()
        st.cache_data.clear()
        st.success("Coefficient review complete — all terms accepted. See results below.")
        st.rerun()
        return

    if not remaining_terms:
        st.warning("All remaining terms were rejected — nothing left to fit.")
        logger.log(
            "glm_coef_review_complete", stage="glm_coefficient_review",
            iterations=iteration, final_terms=[],
        )
        _reset_state()
        st.rerun()
        return

    with st.spinner(f"Removing {len(rejected)} rejected term(s) and refitting..."):
        result, _ = _fit(cfg, remaining_terms)
    st.session_state.coef_result = result
    st.session_state.coef_active_terms = remaining_terms
    st.session_state.coef_iteration = iteration
    st.rerun()


def _log_rating_factors(result) -> None:
    final_summary = coef_summary(result)
    _session.get_logger().log(
        "rating_factors", stage="glm",
        aic=float(result.aic),
        deviance_explained=float(1 - result.deviance / result.null_deviance),
        rating_factors=final_summary.to_dict(orient="records"),
    )
