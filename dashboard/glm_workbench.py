"""Interactive GLM Distillation Workbench (Streamlit Layer 2).

Mirrors the Feature & Grouping Workbench: the distillation agent proposes GLM
terms (main effects for every approved feature + pairwise interactions ranked
by the GBM's H-statistics), the actuary reviews every term as a card —
include/exclude, leave a comment — then re-runs the agent with all feedback
at once, looping until finalized. Finalizing writes glm_config.yaml, which
the GLM fitting step reads from.

Tab placement is always actuary/data-owned. A term's Main Effects vs.
Interactions split is structural — `core.distillation_pipeline.reconcile_terms`
recomputes `term_type` from its own name (":"-joined => interaction, else main)
on every Update/Finalize, independent of what the agent's refine response says
— the one exception is "polynomial", which only survives for a term the
actuary explicitly remarked on this round. Unchecking a term moves it to the
Not Proposed tab rather than leaving it, still unchecked, in its original tab
— `_render_cards` sorts by `approved` before `term_type`, so a term only ever
appears in exactly one of the three tabs; re-checking it there (no LLM call)
restores it to its proper tab on the next Update. Unlike the Feature
Workbench's excluded tab, the full term (rationale, comment history,
H-statistic) survives the move intact — there's a real `GLMTerm` object to
keep, not just a bare column name. Finalizing shows the same card layout
locked (checkboxes/comments disabled) instead of a plain table, and "Re-open"
loads it back into an editable draft.

Comments accumulate the same way as the Feature Workbench: each term keeps a
`comment_history` (see `core.schemas.CommentEntry`), shown above its comment box
via `dashboard._comments.render_comment_history`. "💾" saves a comment
immediately (no LLM call); Update/Finalize send every not-yet-sent entry as
that term's remark.

Three ways to introduce a term the agent hasn't proposed at all: a "➕ Add a new
interaction" card at the bottom of the Interactions tab (two dropdowns over
currently-included main effects, appends instantly via
`core.distillation_pipeline.add_manual_interaction` — no LLM call, same as
unchecking a feature costs nothing in the Feature Workbench); the Not Proposed
tab's "never proposed" sections, listing approved features with no main-effect
term at all and GBM-ranked pairs (sourced from the `gbm_output` checkpoint, not
recomputed) the agent didn't propose as an interaction — checking one and
hitting Update adds it (`add_manual_main_effect`/`add_manual_interaction`)
*and* sends it to the agent as a remark, same "promote and let the agent weigh
in" pattern as the Feature Workbench's excluded tab; and a general "Message to
the agent" box at the top of the form for anything more open-ended ("consider
something with region and vehicle_age") — sent as `general_remark` on the next
Update/Finalize, distinct from the per-term remarks dict. The Not Proposed
tab's third section, "Previously proposed, now excluded", is the rejected
terms described above, not a way to introduce something new.

A GBM retrain (or a Feature Selection re-finalize) can leave an existing
draft/checkpoint referencing a feature that's no longer approved — Re-open,
Regenerate, loading a snapshot, and every Update/Finalize all run
`core.distillation_pipeline.reconcile_feature_membership` against the
*current* approved feature list, forcing any such term's `approved` to
`False` (it lands in Not Proposed's "Previously proposed, now excluded",
with an auto-added comment-history note explaining why) — never silently
carried forward as if still valid, and never something a re-checked box or
an agent refine response can override while the feature stays unapproved.
A feature that was removed and later re-approved needs no special
handling: it simply stops being "missing" on the next reconcile, and its
old term can be re-checked normally. `st.warning` surfaces which terms just
got auto-excluded, once, right where it happens.

Each Main Effects card notes which interactions (if any) currently use it;
each Interactions card always names its two constituent main effects
("🔗 Considered as main effects: ...") and additionally flags one that's
currently excluded ("🚫 ...") — all computed fresh every render from the
draft's own state, not live mid-form.

Every draft is snapshotted to disk under `reports/drafts/` (same three kinds,
same directories, as the Feature Workbench — distinguished by a "glm_draft_"
filename prefix so the two pickers never mix up each other's snapshots).

The locked view also carries a "GBM run to distill from" picker, mirroring the
GBM tab's own feature-set picker — sourced from `core.gbm_pipeline.
list_gbm_runs()` (GBM's run history, read back from `gbm_complete` session-log
events rather than a snapshot file, since GBM has no actuary review loop of its
own). It only affects "Regenerate from scratch" — Update/Finalize never touch
GBM interactions again once a draft exists. Which run fed the current draft is
recorded in `st.session_state.glm_gbm_source` and logged on both the initial
`glm_term_proposal` and the terminal `glm_distillation_complete` event, for the
Audit Trail's Model Lineage view to read back — `None` if the draft came from
Re-open or a loaded snapshot instead of a fresh regenerate.
"""

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from core.distillation_pipeline import (
    add_manual_interaction,
    add_manual_main_effect,
    generate_glm_draft,
    list_glm_draft_snapshots,
    load_glm_draft_snapshot,
    reconcile_feature_membership,
    reconcile_terms,
    refine_glm_draft,
    save_glm_draft_snapshot,
)
from core.gbm_pipeline import list_gbm_runs, restore_gbm_run
from core.glm_pipeline import proposal_from_glm_config, save_glm_checkpoint
from core.run_scope import decisions_log_path as run_decisions_log_path
from core.run_scope import drafts_dir as run_drafts_dir
from core.run_scope import sessions_dir as run_sessions_dir
from core.schemas import CommentEntry, GLMProposal, GLMTerm
from core.seed_config import DISTILLATION_SEED_FILENAME, load_distillation_seed
from core.snapshot_utils import format_ts
from dashboard import _session
from dashboard._comments import render_comment_history
from dashboard.approval_gate import _save_glm_decisions

_LOCKED_ITERATION = -1  # stable widget-key namespace for the locked (post-finalize) view
_CURRENT_GBM_OPTION = "Current (project_config.yaml)"


def render_glm_workbench(cfg: dict, glm_config_path: Path) -> None:
    _session.init_state()
    _init_state()

    notice = st.session_state.pop("glm_feature_membership_notice", None)
    if notice:
        st.warning(notice)

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
    _render_snapshot_loader(cfg, glm_config_path)


# ── State helpers ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("glm_draft", None)
    st.session_state.setdefault("glm_iteration", 0)
    st.session_state.setdefault("glm_seed", None)
    st.session_state.setdefault("glm_pending_snapshot_load", None)
    st.session_state.setdefault("glm_comment_round", {})  # per-term comment-box key generation
    st.session_state.setdefault("glm_gbm_source", None)  # lineage: which GBM run seeded this draft
    st.session_state.setdefault("glm_feature_membership_notice", None)  # queued auto-exclude warning


def _note_auto_excluded(excluded: list[str], lead_in: str) -> None:
    """Queue a warning about terms `reconcile_feature_membership` just auto-
    excluded, for display at the top of the *next* render.

    Calling `st.warning()` directly here would be pointless: every call site
    that needs this immediately follows with `st.rerun()`, which raises an
    exception that aborts the current script run before that message ever
    reaches the browser — a message queued in an aborted run does not
    survive into the fresh one Streamlit starts next. Stashing it in
    `session_state` and popping it in `render_glm_workbench` (the one entry
    point every subsequent render passes through, whichever view ends up
    showing) is what actually gets it in front of the actuary.
    """
    if not excluded:
        return
    st.session_state.glm_feature_membership_notice = (
        f"{lead_in} auto-excluded {len(excluded)} term(s) whose feature is no longer "
        f"approved: {', '.join(sorted(excluded))}. Review them in the Not Proposed tab."
    )


def _approved_feature_names(cfg: dict) -> list[str]:
    """Names of currently-approved features only — `approved is not False`,
    matching the default-approved-unless-explicitly-rejected convention used
    everywhere else (`feature_workbench.py`'s checkbox default, etc.). Every
    entry reaching `project_config.yaml`'s checkpoint has already been through
    the Feature Workbench's Finalize gate, so `approved` here is always a
    real `True`/`False`, never the mid-review `None` — but the `is not False`
    form is used anyway for the same defense-in-depth reason as elsewhere.

    Previously returned *every* name in `features.numeric`/`features.categorical`
    regardless of `approved`, silently including rejected features — fed to
    the LLM as candidates in `generate_glm_draft`, offered for promotion in
    the "Not Proposed" tab's missing-main-effects section, and (the bug that
    surfaced this one) making `reconcile_feature_membership` a no-op for any
    term whose feature had actually been rejected, since a rejected feature's
    name was still "in the approved list" as far as this function was
    concerned.
    """
    features = cfg.get("features", {})
    return (
        [f["name"] for f in features.get("numeric", []) if f.get("approved") is not False]
        + [f["name"] for f in features.get("categorical", []) if f.get("approved") is not False]
    )


# ── Draft generation ───────────────────────────────────────────────────────────

def _gbm_run_label(run: dict) -> str:
    """Rich, self-describing label for the picker dropdown only — includes the
    feature snapshot it was trained on, to help the actuary tell runs apart."""
    ts = format_ts(run.get("ts"))
    src = (run.get("feature_source") or {}).get("label", "?")
    n_int = len(run.get("interactions") or [])
    return f"{ts} — trained on {src} ({n_int} interactions)"




def _render_gbm_source_picker(glm_config_path: Path) -> str | dict:
    """A run picker mirroring the GBM tab's own feature-set picker. Returns
    `_CURRENT_GBM_OPTION` or a specific run dict from `list_gbm_runs()`.
    Only matters for a brand-new proposal — refine/Update calls never touch
    GBM interactions again once a draft exists."""
    runs = list_gbm_runs(run_sessions_dir(glm_config_path))
    if not runs:
        return _CURRENT_GBM_OPTION
    options = [_CURRENT_GBM_OPTION, *runs]
    return st.selectbox(
        "GBM run to distill from (only used by Regenerate from scratch)",
        options, format_func=lambda o: o if isinstance(o, str) else _gbm_run_label(o),
        key="glm_gbm_pick",
    )


def _resolve_gbm_source(cfg: dict, project_config_path: Path, gbm_pick: str | dict) -> tuple[list[dict], dict]:
    """Return (interactions, gbm_source) for the picked run, restoring it as the
    active checkpoint first if it's a historical one — mirrors `gbm_workbench`'s
    own restore-then-use pattern for feature snapshots."""
    if isinstance(gbm_pick, dict):
        restore_gbm_run(project_config_path, cfg, gbm_pick)
        return gbm_pick["interactions"], {"kind": "historical_run", "ts": format_ts(gbm_pick.get("ts"))}

    # "Current" always coincides with the newest gbm_complete run (GBM's only
    # writers are Train/Retrain and this restore branch, always kept in sync).
    runs = list_gbm_runs(run_sessions_dir(project_config_path))
    ts = format_ts(runs[0].get("ts")) if runs else None
    return cfg["gbm_output"]["interactions"], {"kind": "current", "ts": ts}


def _generate_fresh_draft(cfg: dict, glm_config_path: Path, gbm_pick: str | dict) -> None:
    """'Regenerate from scratch': always calls the LLM for a brand-new,
    actuary-untouched draft and snapshots it as kind="initial"."""
    llm = _session.get_llm(cfg)
    if llm is None:
        return
    data_cfg = cfg["data"]
    seed = load_distillation_seed(glm_config_path.parent / DISTILLATION_SEED_FILENAME)
    st.session_state.glm_seed = seed
    project_config_path = glm_config_path.parent / "project_config.yaml"
    interactions, gbm_source = _resolve_gbm_source(cfg, project_config_path, gbm_pick)
    st.session_state.glm_gbm_source = gbm_source
    with st.spinner("Proposing GLM terms from GBM H-statistics..."):
        draft = generate_glm_draft(
            llm, interactions, _approved_feature_names(cfg), data_cfg, seed=seed,
        )
    # Defense-in-depth only — the agent is only ever given approved feature
    # names to propose from, so this should be a no-op; a non-empty result
    # here would mean the agent proposed a term for a feature it wasn't
    # offered, which is worth surfacing rather than silently swallowing.
    excluded = reconcile_feature_membership(draft, _approved_feature_names(cfg))
    _note_auto_excluded(excluded, "Regenerating")
    save_glm_draft_snapshot(draft, kind="initial", drafts_dir=run_drafts_dir(glm_config_path))
    st.session_state.glm_draft = draft
    st.session_state.glm_iteration += 1
    st.session_state.glm_comment_round = {}
    _session.get_logger(glm_config_path).log(
        "glm_term_proposal", stage="glm_distillation", iteration=st.session_state.glm_iteration,
        terms=[t.model_dump() for t in draft.terms], gbm_source=gbm_source,
    )


# ── Locked view (shown when no draft is in progress) ───────────────────────────

def _render_locked_view(cfg: dict, glm_config_path: Path) -> None:
    proposal = proposal_from_glm_config(glm_config_path)
    has_checkpoint = proposal is not None

    if not has_checkpoint:
        st.info("No GLM distillation checkpoint yet. Generate a first draft below.")
    else:
        _render_cards(proposal, cfg, iteration=_LOCKED_ITERATION, locked=True)

    st.divider()
    gbm_pick = _render_gbm_source_picker(glm_config_path)
    c1, c2 = st.columns(2)
    if c1.button(
        "Re-open", use_container_width=True, disabled=not has_checkpoint, key="glm_revise_btn",
    ):
        excluded = reconcile_feature_membership(proposal, _approved_feature_names(cfg))
        st.session_state.glm_draft = proposal
        st.session_state.glm_seed = load_distillation_seed(
            glm_config_path.parent / DISTILLATION_SEED_FILENAME,
        )
        st.session_state.glm_iteration += 1
        st.session_state.glm_comment_round = {}
        _note_auto_excluded(excluded, "Re-opening")
        st.rerun()
    if c2.button("Regenerate from scratch", use_container_width=True, type="primary", key="glm_regen_btn"):
        _generate_fresh_draft(cfg, glm_config_path, gbm_pick)
        st.rerun()


# ── Shared card rendering (edit form + locked view) ────────────────────────────

def _term_card(
    term, iteration: int, locked: bool, related_notes: list[str] | None = None, show_history: bool = True,
) -> tuple[bool, str, bool]:
    """Render one term's card. Returns (checked, comment_box_text, saved).

    `related_notes` renders as one caption per entry — e.g. an interaction shows
    both which main effects it's built from and, if applicable, that one of them
    is currently excluded, as two separate lines rather than one combined string.

    `show_history=False` renders a bare comment box with no save button and no
    history section — used for "Not Proposed" cards, which are virtual `GLMTerm`s
    that don't actually exist on the draft yet, so there's nothing to persist a
    history against (same reasoning as the Feature Workbench's `comment_history=
    None` branch for its own "Not Proposed" cards).
    """
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        checked = c1.checkbox(
            "Include", value=term.approved is not False,
            key=f"glm_iter{iteration}_include_{term.name}", disabled=locked,
        )
        c2.markdown(f"**{term.name}**  ·  _{term.term_type}_")
        if term.h_statistic is not None:
            c2.caption(f"📊 H-statistic: {term.h_statistic:.4f}")
        if term.rationale:
            c2.markdown(f"**Rationale:** {term.rationale}")
        for note in related_notes or []:
            c2.caption(note)

        if not show_history:
            with c2:
                comment = st.text_area(
                    "Comment for agent", value="", key=f"glm_comment_{term.name}_{iteration}", height=68,
                    disabled=locked, label_visibility="collapsed",
                )
            return checked, comment, False

        with c2:
            render_comment_history(term.comment_history)

            round_ = st.session_state.glm_comment_round.get(term.name, 0)
            cc1, cc2 = st.columns([5, 1])
            comment = cc1.text_area(
                "Comment for agent", value="", key=f"glm_comment_{term.name}_{round_}", height=68,
                disabled=locked, label_visibility="collapsed",
            )
            saved = False
            if not locked:
                saved = cc2.form_submit_button(
                    "💾", key=f"glm_iter{iteration}_save_{term.name}", help="Save this comment",
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
        feature_a = c1.selectbox("Feature A", main_effect_names, key=f"glm_iter{iteration}_add_a")
        feature_b = c2.selectbox(
            "Feature B", main_effect_names, key=f"glm_iter{iteration}_add_b",
            index=min(1, len(main_effect_names) - 1),
        )
        rationale = st.text_input(
            "Rationale (optional)", key=f"glm_iter{iteration}_add_rationale",
            placeholder="Why does this interaction make actuarial sense?",
        )
        submitted = st.form_submit_button("Add interaction", key=f"glm_iter{iteration}_add_submit")
    return feature_a, feature_b, rationale, submitted


def _missing_main_effects(proposal: GLMProposal, cfg: dict) -> list[str]:
    """Approved features with no term at all on the draft yet (any term_type) —
    normally empty (a main effect is proposed per approved feature), but a
    feature approved after distillation last ran could land here."""
    proposed_names = {t.name for t in proposal.terms}
    return [f for f in _approved_feature_names(cfg) if f not in proposed_names]


def _missing_interactions(proposal: GLMProposal, cfg: dict) -> list[dict]:
    """GBM-ranked pairs (from the checkpoint the agent was seeded with) that
    aren't on the draft as an interaction under either name order — e.g. ones
    the agent dropped for a low H-statistic. Sourced from the checkpoint, not
    recomputed, so this is exactly what the agent could have chosen from."""
    proposed_names = {t.name for t in proposal.terms}
    all_pairs = cfg.get("gbm_output", {}).get("interactions", [])
    return [
        i for i in all_pairs
        if f"{i['feature_a']}:{i['feature_b']}" not in proposed_names
        and f"{i['feature_b']}:{i['feature_a']}" not in proposed_names
    ]


def _render_cards(
    proposal: GLMProposal, cfg: dict, iteration: int, locked: bool,
) -> tuple[
    dict[str, bool], dict[str, str], dict[str, bool],
    tuple[str, str, str, bool] | None, dict[str, tuple[bool, str, str]],
]:
    """Render the three-tab card layout shared by the edit form and the locked view.

    Returns (checkbox_state, comment_state, save_clicks, add_state,
    not_proposed_state) — all but `checkbox_state`/`comment_state` are unused by
    callers when `locked` (nothing gets submitted), but harmless to collect either
    way. `not_proposed_state` maps a not-yet-proposed name to
    (checked, comment, kind) where kind is "main" or "interaction".
    """
    # Rejected (unchecked) terms move to the Not Proposed tab rather than
    # lingering, still unchecked, in their original tab — the actuary-owned
    # tab-placement rule applies to approval, not just structural term_type.
    # Unlike the Feature Workbench's excluded tab, the full GLMTerm (rationale,
    # comment_history, h_statistic) survives the move intact, since there's a
    # real object to keep rather than a bare column name.
    approved_terms = [t for t in proposal.terms if t.approved is not False]
    rejected_terms = [t for t in proposal.terms if t.approved is False]
    main_terms = [t for t in approved_terms if t.term_type != "interaction"]
    interaction_terms = [t for t in approved_terms if t.term_type == "interaction"]
    included_main_names = {t.name for t in main_terms}

    # Cross-references for point-in-time context on each card (recomputed fresh
    # every render from the draft as currently checked — not live mid-form).
    interaction_usage: dict[str, list[str]] = {}
    for t in interaction_terms:
        for feat in t.name.split(":"):
            interaction_usage.setdefault(feat, []).append(t.name)

    def _interaction_notes(term) -> list[str]:
        parts = term.name.split(":")
        notes = [f"🔗 Considered as main effects: {' and '.join(parts)}"]
        missing_parts = [p for p in parts if p not in included_main_names]
        if missing_parts:
            notes.append(f"🚫 Main effect(s) currently excluded: {', '.join(missing_parts)}")
        return notes

    def _main_effect_notes(term) -> list[str] | None:
        used_in = interaction_usage.get(term.name, [])
        return [f"🔗 Used in {len(used_in)} interaction(s): {', '.join(used_in)}"] if used_in else None

    def _term_notes(term) -> list[str] | None:
        return _interaction_notes(term) if term.term_type == "interaction" else _main_effect_notes(term)

    checkbox_state: dict[str, bool] = {}
    comment_state: dict[str, str] = {}
    save_clicks: dict[str, bool] = {}
    add_state: tuple[str, str, str, bool] | None = None
    not_proposed_state: dict[str, tuple[bool, str, str]] = {}

    missing_main = _missing_main_effects(proposal, cfg)
    missing_interactions = _missing_interactions(proposal, cfg)
    n_missing = len(rejected_terms) + len(missing_main) + len(missing_interactions)

    tab_main, tab_interactions, tab_not_proposed = st.tabs([
        f"Main Effects ({len(main_terms)})",
        f"Interactions ({len(interaction_terms)})",
        f"Not Proposed ({n_missing})",
    ])

    with tab_main:
        if not main_terms:
            st.caption("No main effects proposed yet.")
        for term in main_terms:
            checked, comment, saved = _term_card(term, iteration, locked, related_notes=_main_effect_notes(term))
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
            checked, comment, saved = _term_card(term, iteration, locked, related_notes=_interaction_notes(term))
            checkbox_state[term.name] = checked
            comment_state[term.name] = comment
            save_clicks[term.name] = saved

        if not locked:
            add_state = _render_add_interaction_card(sorted(included_main_names), iteration)

    with tab_not_proposed:
        if n_missing == 0:
            st.caption(
                "Nothing left — every approved feature has a main effect, and every "
                "GBM-ranked pair is already proposed as an interaction."
            )
        if rejected_terms:
            st.markdown("**Previously proposed, now excluded** — check to bring back")
            for term in rejected_terms:
                checked, comment, saved = _term_card(term, iteration, locked, related_notes=_term_notes(term))
                checkbox_state[term.name] = checked
                comment_state[term.name] = comment
                save_clicks[term.name] = saved
        if missing_main:
            st.markdown("**Main effects the agent never proposed**")
            for name in missing_main:
                virtual = GLMTerm(name=name, term_type="main", rationale="", approved=False)
                checked, comment, _ = _term_card(virtual, iteration, locked, show_history=False)
                not_proposed_state[name] = (checked, comment, "main")
        if missing_interactions:
            st.markdown("**GBM-ranked pairs the agent didn't propose as an interaction**")
            for i in sorted(missing_interactions, key=lambda x: -x["h_statistic"]):
                name = f"{i['feature_a']}:{i['feature_b']}"
                virtual = GLMTerm(
                    name=name, term_type="interaction", rationale="",
                    approved=False, h_statistic=i["h_statistic"],
                )
                checked, comment, _ = _term_card(virtual, iteration, locked, show_history=False)
                not_proposed_state[name] = (checked, comment, "interaction")

    return checkbox_state, comment_state, save_clicks, add_state, not_proposed_state


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

        checkbox_state, comment_state, save_clicks, add_state, not_proposed_state = _render_cards(
            draft, cfg, it, locked=False,
        )

        col_rerun, col_finalize = st.columns(2)
        submit_rerun = col_rerun.form_submit_button(
            "Update", use_container_width=True, key="glm_rerun_btn",
        )
        submit_finalize = col_finalize.form_submit_button(
            "Finalize", use_container_width=True, type="primary", key="glm_finalize_btn",
        )

    if add_state is not None and add_state[3]:
        _handle_add_interaction(draft, add_state[0], add_state[1], add_state[2], glm_config_path)
        return

    if submit_rerun or submit_finalize:
        _handle_submit(
            cfg, glm_config_path, draft, checkbox_state, comment_state, general_remark,
            not_proposed_state, finalize=submit_finalize,
        )
        return

    saved_name = next((name for name, clicked in save_clicks.items() if clicked), None)
    if saved_name is not None:
        _handle_save_comment(draft, saved_name, comment_state[saved_name], glm_config_path)


def _handle_add_interaction(
    draft: GLMProposal, feature_a: str, feature_b: str, rationale: str, glm_config_path: Path,
) -> None:
    try:
        add_manual_interaction(draft, feature_a, feature_b, rationale.strip())
    except ValueError as e:
        st.session_state.glm_draft = draft
        st.error(str(e))
        return
    save_glm_draft_snapshot(draft, kind="modified", drafts_dir=run_drafts_dir(glm_config_path))
    st.session_state.glm_draft = draft
    st.success(f"Added interaction `{feature_a}:{feature_b}`.")
    st.rerun()


def _handle_save_comment(draft: GLMProposal, name: str, text: str, glm_config_path: Path) -> None:
    """Save one card's comment immediately — appends to history and clears the
    box, without touching any other card or costing an LLM call."""
    text = text.strip()
    if text:
        term = next((t for t in draft.terms if t.name == name), None)
        if term is not None:
            term.comment_history.append(CommentEntry(
                author="actuary", text=text, ts=datetime.now(timezone.utc).isoformat(),
            ))
            save_glm_draft_snapshot(draft, kind="modified", drafts_dir=run_drafts_dir(glm_config_path))
    st.session_state.glm_comment_round[name] = st.session_state.glm_comment_round.get(name, 0) + 1
    st.session_state.glm_draft = draft
    st.rerun()


def _promote_not_proposed(
    draft: GLMProposal, not_proposed_state: dict[str, tuple[bool, str, str]],
) -> dict[str, str]:
    """Add every checked "Not Proposed" item directly to the draft — no LLM call,
    same reasoning as the "+ Add interaction" card. Returns a name -> comment map
    for whatever the actuary typed alongside a promoted item, folded into this
    round's remarks so the agent sees why it was added, same as a promoted
    "excluded" column in the Feature Workbench.
    """
    promoted_comments: dict[str, str] = {}
    for name, (checked, comment, kind) in not_proposed_state.items():
        if not checked:
            continue
        note = comment.strip() or "Actuary promoted this from Not Proposed."
        try:
            if kind == "main":
                add_manual_main_effect(draft, name, rationale=note)
            else:
                feature_a, feature_b = name.split(":", 1)
                add_manual_interaction(draft, feature_a, feature_b, rationale=note)
        except ValueError:
            continue  # already got added another way this round — nothing to do
        promoted_comments[name] = note
    return promoted_comments


def _handle_submit(
    cfg: dict,
    glm_config_path: Path,
    draft: GLMProposal,
    checkbox_state: dict[str, bool],
    comment_state: dict[str, str],
    general_remark: str,
    not_proposed_state: dict[str, tuple[bool, str, str]],
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

    remarks.update(_promote_not_proposed(draft, not_proposed_state))

    general_remark = general_remark.strip()

    # Term placement (main/interaction) and approval are actuary/data-owned:
    # recompute unconditionally before any agent call. Feature-membership runs
    # right after, on the same cadence — it can override a just-re-checked box
    # back to excluded if the box's feature is still de-approved (e.g. a
    # "Not Proposed" GBM-ranked pair just promoted by _promote_not_proposed
    # above can itself reference a feature that's no longer approved).
    draft = reconcile_terms(draft, checkbox_state, remarked=set(remarks))
    auto_excluded = set(reconcile_feature_membership(draft, _approved_feature_names(cfg)))

    logger = _session.get_logger(glm_config_path)
    session_id = _session.get_session_id(glm_config_path)
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
        auto_excluded |= set(reconcile_feature_membership(draft, _approved_feature_names(cfg)))
        st.session_state.glm_iteration += 1
        logger.log(
            "glm_term_proposal", stage="glm_distillation", iteration=st.session_state.glm_iteration,
            terms=[t.model_dump() for t in draft.terms],
        )

    _note_auto_excluded(sorted(auto_excluded), "This round")

    if not finalize:
        save_glm_draft_snapshot(draft, kind="modified", drafts_dir=run_drafts_dir(glm_config_path))
        st.session_state.glm_draft = draft
        if not should_refine:
            st.info("Approval flags updated — no comments to send to the agent.")
        st.rerun()
        return

    formula = save_glm_checkpoint(glm_config_path, data_cfg, draft)
    save_glm_draft_snapshot(draft, kind="finalized", drafts_dir=run_drafts_dir(glm_config_path))

    approved_terms = [t.name for t in draft.terms if t.approved is True]
    logger.log(
        "glm_distillation_complete", stage="glm_distillation",
        iterations=st.session_state.glm_iteration, approved_terms=approved_terms,
        gbm_source=st.session_state.glm_gbm_source,
    )
    _save_glm_decisions(draft, session_id, run_decisions_log_path(glm_config_path))

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


def _load_snapshot_into_draft(path: Path, glm_config_path: Path, cfg: dict) -> None:
    draft = load_glm_draft_snapshot(path)
    # A snapshot can be arbitrarily old — same feature-membership reconcile as
    # Re-open, so a snapshot from before a feature was de-approved doesn't
    # silently reintroduce an orphaned term.
    excluded = reconcile_feature_membership(draft, _approved_feature_names(cfg))
    st.session_state.glm_draft = draft
    st.session_state.glm_seed = load_distillation_seed(glm_config_path.parent / DISTILLATION_SEED_FILENAME)
    st.session_state.glm_iteration += 1
    st.session_state.glm_comment_round = {}
    _note_auto_excluded(excluded, "Loading this snapshot")


def _request_snapshot_load(path: Path, glm_config_path: Path, cfg: dict) -> None:
    """Load immediately if nothing's at risk; otherwise defer to the confirm step
    below, which is the only place `glm_draft` actually gets overwritten in that case."""
    if st.session_state.glm_draft is None:
        _load_snapshot_into_draft(path, glm_config_path, cfg)
    else:
        st.session_state.glm_pending_snapshot_load = path


def _render_snapshot_picker(col, kind: str, label: str, glm_config_path: Path, cfg: dict) -> None:
    snapshots = list_glm_draft_snapshots(kind, run_drafts_dir(glm_config_path))
    with col:
        st.caption(f"{label} ({len(snapshots)})")
        pick = st.selectbox(
            label, snapshots, format_func=_snapshot_label, key=f"glm_pick_{kind}",
            index=None, placeholder="Select a snapshot…", label_visibility="collapsed",
        )
        if st.button(
            "Load", key=f"glm_load_{kind}_btn", disabled=pick is None, use_container_width=True,
        ):
            _request_snapshot_load(pick, glm_config_path, cfg)
            st.rerun()


def _render_snapshot_loader(cfg: dict, glm_config_path: Path) -> None:
    with st.expander("📂 Load a saved snapshot"):
        c1, c2, c3 = st.columns(3)
        _render_snapshot_picker(c1, "initial", "Initial agent proposals", glm_config_path, cfg)
        _render_snapshot_picker(c2, "modified", "Modified drafts", glm_config_path, cfg)
        _render_snapshot_picker(c3, "finalized", "Finalized checkpoints", glm_config_path, cfg)

    pending = st.session_state.glm_pending_snapshot_load
    if pending is not None:
        if st.session_state.glm_draft is not None:
            st.warning(f"Loading **{pending.name}** will discard your current unsaved draft. Continue?")
        else:
            st.info(f"Load **{pending.name}**?")
        cc1, cc2 = st.columns(2)
        if cc1.button("Yes, load it", key="glm_confirm_load_btn", type="primary", use_container_width=True):
            _load_snapshot_into_draft(pending, glm_config_path, cfg)
            st.session_state.glm_pending_snapshot_load = None
            st.rerun()
        if cc2.button("Cancel", key="glm_cancel_load_btn", use_container_width=True):
            st.session_state.glm_pending_snapshot_load = None
            st.rerun()
