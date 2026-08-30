"""Interactive GLM Distillation Workbench (Streamlit Layer 2).

Mirrors the Feature & Grouping Workbench: the distillation agent proposes GLM
terms (main effects for every approved feature + pairwise interactions ranked
by the GBM's H-statistics), the actuary reviews every term as a card —
include/exclude, leave a comment — then re-runs the agent with all feedback
at once, looping until finalized. Finalizing writes glm_config.yaml, which
the GLM fitting step reads from.

Tab placement (Main Effects vs. Interactions) is always actuary/data-owned:
`core.distillation_pipeline.reconcile_terms` recomputes each term's `term_type`
structurally from its own name (":"-joined => interaction, else main) on every
Update/Finalize, independent of what the agent's refine response says — the one
exception is "polynomial", which only survives for a term the actuary explicitly
remarked on this round. `approved` is likewise always taken from the submitted
checkbox state. Finalizing shows the same card layout locked (checkboxes/
comments disabled) instead of a plain table, and "Re-open" loads it back into an
editable draft.

Comments accumulate the same way as the Feature Workbench: each term keeps a
`comment_history` (see `core.schemas.CommentEntry`), shown above its comment box
via `dashboard._comments.render_comment_history`. "💾" saves a comment
immediately (no LLM call); Update/Finalize send every not-yet-sent entry as
that term's remark.

Two ways to introduce an interaction the agent hasn't proposed: a "➕ Add a new
interaction" card at the bottom of the Interactions tab (two dropdowns over
currently-included main effects, appends instantly via
`core.distillation_pipeline.add_manual_interaction` — no LLM call, same as
unchecking a feature costs nothing in the Feature Workbench), and a general
"Message to the agent" box at the top of the form for anything more open-ended
("consider something with region and vehicle_age") — sent as `general_remark`
on the next Update/Finalize, distinct from the per-term remarks dict.

Every draft is snapshotted to disk under `reports/drafts/` (same three kinds,
same directories, as the Feature Workbench — distinguished by a "glm_draft_"
filename prefix so the two pickers never mix up each other's snapshots).
"""

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from core.distillation_pipeline import (
    add_manual_interaction,
    generate_glm_draft,
    list_glm_draft_snapshots,
    load_glm_draft_snapshot,
    reconcile_terms,
    refine_glm_draft,
    save_glm_draft_snapshot,
)
from core.glm_pipeline import proposal_from_glm_config, save_glm_checkpoint
from core.schemas import CommentEntry, GLMProposal
from core.seed_config import DISTILLATION_SEED_FILENAME, load_distillation_seed
from dashboard import _session
from dashboard._comments import render_comment_history
from dashboard.approval_gate import _save_glm_decisions

_LOCKED_ITERATION = -1  # stable widget-key namespace for the locked (post-finalize) view


def render_glm_workbench(cfg: dict, glm_config_path: Path) -> None:
    _session.init_state()
    _init_state()

    if not cfg.get("gbm_output", {}).get("interactions"):
        st.info("Train the GBM first (GBM tab) — GLM distillation reads its H-statistics.")
        return

    if st.session_state.glm_draft is None:
        _render_locked_view(cfg, glm_config_path)
    else:
        _render_edit_form(cfg, glm_config_path)
        if st.button("Discard draft and start over", key="glm_discard_btn"):
            st.session_state.glm_draft = None
            st.rerun()

    st.divider()
    _render_snapshot_loader(glm_config_path)


# ── State helpers ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("glm_draft", None)
    st.session_state.setdefault("glm_iteration", 0)
    st.session_state.setdefault("glm_seed", None)
    st.session_state.setdefault("glm_pending_snapshot_load", None)
    st.session_state.setdefault("glm_comment_round", {})  # per-term comment-box key generation


def _approved_feature_names(cfg: dict) -> list[str]:
    features = cfg.get("features", {})
    return (
        [f["name"] for f in features.get("numeric", [])]
        + [f["name"] for f in features.get("categorical", [])]
    )


# ── Draft generation ───────────────────────────────────────────────────────────

def _generate_fresh_draft(cfg: dict, glm_config_path: Path) -> None:
    """'Regenerate from scratch': always calls the LLM for a brand-new,
    actuary-untouched draft and snapshots it as kind="initial"."""
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
    save_glm_draft_snapshot(draft, kind="initial")
    st.session_state.glm_draft = draft
    st.session_state.glm_iteration += 1
    st.session_state.glm_comment_round = {}
    _session.get_logger().log(
        "glm_term_proposal", stage="glm_distillation", iteration=st.session_state.glm_iteration,
        terms=[t.model_dump() for t in draft.terms],
    )


# ── Locked view (shown when no draft is in progress) ───────────────────────────

def _render_locked_view(cfg: dict, glm_config_path: Path) -> None:
    proposal = proposal_from_glm_config(glm_config_path)
    has_checkpoint = proposal is not None

    if not has_checkpoint:
        st.info("No GLM distillation checkpoint yet. Generate a first draft below.")
    else:
        _render_cards(proposal, iteration=_LOCKED_ITERATION, locked=True)

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button(
        "Re-open", use_container_width=True, disabled=not has_checkpoint, key="glm_revise_btn",
    ):
        st.session_state.glm_draft = proposal
        st.session_state.glm_seed = load_distillation_seed(
            glm_config_path.parent / DISTILLATION_SEED_FILENAME,
        )
        st.session_state.glm_iteration += 1
        st.session_state.glm_comment_round = {}
        st.rerun()
    if c2.button("Regenerate from scratch", use_container_width=True, type="primary", key="glm_regen_btn"):
        _generate_fresh_draft(cfg, glm_config_path)
        st.rerun()


# ── Shared card rendering (edit form + locked view) ────────────────────────────

def _term_card(term, iteration: int, locked: bool) -> tuple[bool, str, bool]:
    """Render one term's card. Returns (checked, comment_box_text, saved)."""
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        checked = c1.checkbox(
            "Include", value=term.approved is not False,
            key=f"iter{iteration}_include_{term.name}", disabled=locked,
        )
        c2.markdown(f"**{term.name}**  ·  _{term.term_type}_")
        if term.h_statistic is not None:
            c2.caption(f"📊 H-statistic: {term.h_statistic:.4f}")
        if term.rationale:
            c2.markdown(f"**Rationale:** {term.rationale}")

        render_comment_history(term.comment_history)

        round_ = st.session_state.glm_comment_round.get(term.name, 0)
        cc1, cc2 = st.columns([5, 1])
        comment = cc1.text_area(
            "Comment for agent", value="", key=f"comment_{term.name}_{round_}", height=68,
            disabled=locked, label_visibility="collapsed",
        )
        saved = False
        if not locked:
            saved = cc2.form_submit_button(
                "💾", key=f"iter{iteration}_save_{term.name}", help="Save this comment",
            )
    return checked, comment, saved


def _render_add_interaction_card(
    main_effect_names: list[str], iteration: int,
) -> tuple[str, str, str, bool] | None:
    """A card for manually adding an interaction — no LLM call, appends instantly.
    Returns None when there aren't at least two included main effects to pick from."""
    with st.container(border=True):
        st.markdown("**➕ Add a new interaction**")
        if len(main_effect_names) < 2:
            st.caption("Need at least two included main effects to build an interaction from.")
            return None
        c1, c2 = st.columns(2)
        feature_a = c1.selectbox("Feature A", main_effect_names, key=f"iter{iteration}_add_a")
        feature_b = c2.selectbox(
            "Feature B", main_effect_names, key=f"iter{iteration}_add_b",
            index=min(1, len(main_effect_names) - 1),
        )
        rationale = st.text_input(
            "Rationale (optional)", key=f"iter{iteration}_add_rationale",
            placeholder="Why does this interaction make actuarial sense?",
        )
        submitted = st.form_submit_button("Add interaction", key=f"iter{iteration}_add_submit")
    return feature_a, feature_b, rationale, submitted


def _render_cards(
    proposal: GLMProposal, iteration: int, locked: bool,
) -> tuple[dict[str, bool], dict[str, str], dict[str, bool], tuple[str, str, str, bool] | None]:
    """Render the two-tab card layout shared by the edit form and the locked view.

    Returns (checkbox_state, comment_state, save_clicks, add_state) — the latter
    three are unused by callers when `locked` (nothing gets submitted), but
    harmless to collect either way.
    """
    main_terms = [t for t in proposal.terms if t.term_type != "interaction"]
    interaction_terms = [t for t in proposal.terms if t.term_type == "interaction"]

    checkbox_state: dict[str, bool] = {}
    comment_state: dict[str, str] = {}
    save_clicks: dict[str, bool] = {}
    add_state: tuple[str, str, str, bool] | None = None

    tab_main, tab_interactions = st.tabs([
        f"Main Effects ({len(main_terms)})",
        f"Interactions ({len(interaction_terms)})",
    ])

    with tab_main:
        if not main_terms:
            st.caption("No main effects proposed yet.")
        for term in main_terms:
            checked, comment, saved = _term_card(term, iteration, locked)
            checkbox_state[term.name] = checked
            comment_state[term.name] = comment
            save_clicks[term.name] = saved

    with tab_interactions:
        st.caption(
            "⚠️ Standard actuarial practice: an interaction term should only stay in the "
            "model alongside the main effects of its constituent variables — those are "
            "proposed for every approved feature in the Main Effects tab, not just the "
            "ones involved in an interaction. Excluding a main effect while keeping its "
            "interaction is rarely defensible and should have an explicit rationale."
        )
        if not interaction_terms:
            st.caption("No interactions proposed yet.")
        for term in interaction_terms:
            checked, comment, saved = _term_card(term, iteration, locked)
            checkbox_state[term.name] = checked
            comment_state[term.name] = comment
            save_clicks[term.name] = saved

        if not locked:
            included_main_names = [t.name for t in main_terms if t.approved is not False]
            add_state = _render_add_interaction_card(included_main_names, iteration)

    return checkbox_state, comment_state, save_clicks, add_state


# ── Edit form ───────────────────────────────────────────────────────────────────

def _render_edit_form(cfg: dict, glm_config_path: Path) -> None:
    draft: GLMProposal = st.session_state.glm_draft
    it = st.session_state.glm_iteration
    st.caption(f"Draft iteration {it} — review, comment, then Update or Finalize.")

    with st.form("glm_workbench_form"):
        general_remark = st.text_area(
            "💬 Message to the agent (optional — not tied to a specific term)",
            value="", key=f"glm_general_{it}", height=70,
            placeholder='e.g. "Consider an interaction between region and vehicle_age"',
        )
        st.divider()

        checkbox_state, comment_state, save_clicks, add_state = _render_cards(draft, it, locked=False)

        col_rerun, col_finalize = st.columns(2)
        submit_rerun = col_rerun.form_submit_button(
            "Update", use_container_width=True, key="glm_rerun_btn",
        )
        submit_finalize = col_finalize.form_submit_button(
            "Finalize", use_container_width=True, type="primary", key="glm_finalize_btn",
        )

    if add_state is not None and add_state[3]:
        _handle_add_interaction(draft, add_state[0], add_state[1], add_state[2])
        return

    if submit_rerun or submit_finalize:
        _handle_submit(
            cfg, glm_config_path, draft, checkbox_state, comment_state, general_remark,
            finalize=submit_finalize,
        )
        return

    saved_name = next((name for name, clicked in save_clicks.items() if clicked), None)
    if saved_name is not None:
        _handle_save_comment(draft, saved_name, comment_state[saved_name])


def _handle_add_interaction(draft: GLMProposal, feature_a: str, feature_b: str, rationale: str) -> None:
    try:
        add_manual_interaction(draft, feature_a, feature_b, rationale.strip())
    except ValueError as e:
        st.session_state.glm_draft = draft
        st.error(str(e))
        return
    save_glm_draft_snapshot(draft, kind="modified")
    st.session_state.glm_draft = draft
    st.success(f"Added interaction `{feature_a}:{feature_b}`.")
    st.rerun()


def _handle_save_comment(draft: GLMProposal, name: str, text: str) -> None:
    """Save one card's comment immediately — appends to history and clears the
    box, without touching any other card or costing an LLM call."""
    text = text.strip()
    if text:
        term = next((t for t in draft.terms if t.name == name), None)
        if term is not None:
            term.comment_history.append(CommentEntry(
                author="actuary", text=text, ts=datetime.now(timezone.utc).isoformat(),
            ))
            save_glm_draft_snapshot(draft, kind="modified")
    st.session_state.glm_comment_round[name] = st.session_state.glm_comment_round.get(name, 0) + 1
    st.session_state.glm_draft = draft
    st.rerun()


def _handle_submit(
    cfg: dict,
    glm_config_path: Path,
    draft: GLMProposal,
    checkbox_state: dict[str, bool],
    comment_state: dict[str, str],
    general_remark: str,
    finalize: bool,
) -> None:
    data_cfg = cfg["data"]
    remarks: dict[str, str] = {}
    now = datetime.now(timezone.utc).isoformat()

    for term in draft.terms:
        box_text = comment_state.get(term.name, "").strip()
        if box_text:
            # Implicit save: typed but never clicked 💾 — don't discard it.
            term.comment_history.append(CommentEntry(author="actuary", text=box_text, ts=now))
            st.session_state.glm_comment_round[term.name] = (
                st.session_state.glm_comment_round.get(term.name, 0) + 1
            )
        unsent = [e.text for e in term.comment_history if not e.sent]
        if unsent:
            remarks[term.name] = "\n---\n".join(unsent)

    general_remark = general_remark.strip()

    # Term placement (main/interaction) and approval are actuary/data-owned:
    # recompute unconditionally before any agent call.
    draft = reconcile_terms(draft, checkbox_state, remarked=set(remarks))

    logger = _session.get_logger()
    session_id = _session.get_session_id()
    if remarks:
        logger.log(
            "glm_term_remarks", stage="glm_distillation",
            iteration=st.session_state.glm_iteration, remarks=remarks,
        )
    if general_remark:
        logger.log(
            "glm_general_remark", stage="glm_distillation",
            iteration=st.session_state.glm_iteration, remark=general_remark,
        )

    should_refine = bool(remarks) or bool(general_remark)

    if should_refine:
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
                general_remark=general_remark or None,
            )
        # Defense-in-depth: the agent's response can't move a term or flip its
        # approval even if it tried to — re-apply the actuary's true state.
        draft = reconcile_terms(draft, checkbox_state, remarked=set(remarks))
        st.session_state.glm_iteration += 1
        logger.log(
            "glm_term_proposal", stage="glm_distillation", iteration=st.session_state.glm_iteration,
            terms=[t.model_dump() for t in draft.terms],
        )

    if not finalize:
        save_glm_draft_snapshot(draft, kind="modified")
        st.session_state.glm_draft = draft
        if not should_refine:
            st.info("Approval flags updated — no comments to send to the agent.")
        st.rerun()
        return

    formula = save_glm_checkpoint(glm_config_path, data_cfg, draft)
    save_glm_draft_snapshot(draft, kind="finalized")

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


# ── Snapshot loader (always visible, regardless of edit/locked state) ──────────

def _snapshot_label(path: Path) -> str:
    stem = path.stem.removeprefix("glm_draft_")
    try:
        label = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        label = stem
    try:
        proposal = load_glm_draft_snapshot(path)
        n_main = sum(1 for t in proposal.terms if t.term_type != "interaction")
        n_inter = len(proposal.terms) - n_main
        label += f" — {n_main} main, {n_inter} interactions"
    except Exception:
        pass
    return label


def _load_snapshot_into_draft(path: Path, glm_config_path: Path) -> None:
    st.session_state.glm_draft = load_glm_draft_snapshot(path)
    st.session_state.glm_seed = load_distillation_seed(glm_config_path.parent / DISTILLATION_SEED_FILENAME)
    st.session_state.glm_iteration += 1
    st.session_state.glm_comment_round = {}


def _request_snapshot_load(path: Path, glm_config_path: Path) -> None:
    """Load immediately if nothing's at risk; otherwise defer to the confirm step
    below, which is the only place `glm_draft` actually gets overwritten in that case."""
    if st.session_state.glm_draft is None:
        _load_snapshot_into_draft(path, glm_config_path)
    else:
        st.session_state.glm_pending_snapshot_load = path


def _render_snapshot_picker(col, kind: str, label: str, glm_config_path: Path) -> None:
    snapshots = list_glm_draft_snapshots(kind)
    with col:
        st.caption(f"{label} ({len(snapshots)})")
        pick = st.selectbox(
            label, snapshots, format_func=_snapshot_label, key=f"glm_pick_{kind}",
            index=None, placeholder="Select a snapshot…", label_visibility="collapsed",
        )
        if st.button(
            "Load", key=f"glm_load_{kind}_btn", disabled=pick is None, use_container_width=True,
        ):
            _request_snapshot_load(pick, glm_config_path)
            st.rerun()


def _render_snapshot_loader(glm_config_path: Path) -> None:
    with st.expander("📂 Load a saved snapshot"):
        c1, c2, c3 = st.columns(3)
        _render_snapshot_picker(c1, "initial", "Initial agent proposals", glm_config_path)
        _render_snapshot_picker(c2, "modified", "Modified drafts", glm_config_path)
        _render_snapshot_picker(c3, "finalized", "Finalized checkpoints", glm_config_path)

    pending = st.session_state.glm_pending_snapshot_load
    if pending is not None:
        if st.session_state.glm_draft is not None:
            st.warning(f"Loading **{pending.name}** will discard your current unsaved draft. Continue?")
        else:
            st.info(f"Load **{pending.name}**?")
        cc1, cc2 = st.columns(2)
        if cc1.button("Yes, load it", key="glm_confirm_load_btn", type="primary", use_container_width=True):
            _load_snapshot_into_draft(pending, glm_config_path)
            st.session_state.glm_pending_snapshot_load = None
            st.rerun()
        if cc2.button("Cancel", key="glm_cancel_load_btn", use_container_width=True):
            st.session_state.glm_pending_snapshot_load = None
            st.rerun()
