"""Interactive Feature Selection + Grouping Workbench (Streamlit Layer 2).

Combines feature selection and categorical grouping into one agent-driven
draft. The actuary reviews the whole draft in one screen — select/deselect
variables (including ones the agent excluded), leave comments — then
re-runs the agent with all feedback at once, looping until finalised.

Tab placement (which of Numerical/Categorical/Not Proposed a variable is in) is
always actuary-owned: `core.feature_pipeline.reconcile_membership` recomputes it
from the submitted checkbox state on every Update/Finalize, independent of what
the agent's refine response says. Finalizing shows the same card layout locked
(checkboxes/comments disabled) rather than a plain table, and "Re-open" loads it
back into an editable draft.

Every draft is snapshotted to disk under `reports/drafts/` as one of three kinds,
browsable via the "Load a saved snapshot" section: "initial" (a genuinely fresh,
actuary-untouched LLM proposal, written only by "Regenerate from scratch" — LLM
output isn't deterministic, so this is the only way back to a specific past take),
"modified" (an actuary-edited draft, snapshotted after every Update round or saved
comment), and "finalized" (one per Finalize — project_config.yaml only ever holds
the *current* checkpoint, this is the full history). Loading a snapshot while a
draft is already in progress warns before discarding it.

Comments accumulate rather than overwrite: each numeric/categorical variable keeps
a `comment_history` (see `core.schemas.CommentEntry`) shown above its comment box,
newest first, labeled "👤 Actuary" / "Claude" (with the logo at
`icons/claude-ai.svg` if present, inlined next to the text for tight baseline
alignment; else an orange text fallback via Streamlit's :orange[] markdown
directive — see `_claude_logo_data_uri`). "💾" saves a comment immediately
(appends + clears the box, no LLM call); Update/Finalize send every not-yet-sent
entry as that variable's remark. `actuary_note` on the underlying models is now
purely transient — the agent's reply for the current round, merged into history
and cleared by `core.feature_pipeline.refine_draft` — nothing in this file should
read it directly.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from core.data_quality import detect_target_leakage
from core.feature_pipeline import (
    generate_draft,
    list_draft_snapshots,
    load_draft_snapshot,
    proposal_from_config,
    reconcile_membership,
    refine_draft,
    save_draft_snapshot,
    save_feature_checkpoint,
)
from core.run_scope import decisions_log_path as run_decisions_log_path
from core.run_scope import drafts_dir as run_drafts_dir
from core.schemas import CategoryCluster, CommentEntry, FeatureProposal, GroupingResponse
from core.seed_config import FEATURE_SEED_FILENAME, load_feature_seed
from dashboard import _session
from dashboard._comments import render_comment_history
from dashboard.approval_gate import _save_feature_decisions, _save_grouping_decisions

_LOCKED_ITERATION = -1  # stable widget-key namespace for the locked (post-finalize) view


def render_feature_workbench(cfg: dict, config_path: Path) -> None:
    _session.init_state()
    _init_state()

    if st.session_state.wb_draft is None:
        _render_locked_view(cfg, config_path)
    else:
        _render_edit_form(cfg, config_path)
        if st.button("Discard draft and start over", key="wb_discard_btn"):
            st.session_state.wb_draft = None
            st.rerun()

    st.divider()
    _render_snapshot_loader(config_path)


# ── State helpers ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("wb_draft", None)
    st.session_state.setdefault("wb_iteration", 0)
    st.session_state.setdefault("wb_seed", None)
    st.session_state.setdefault("wb_pending_snapshot_load", None)
    st.session_state.setdefault("wb_comment_round", {})  # per-variable comment-box key generation


# ── Draft generation ───────────────────────────────────────────────────────────

def _regenerate_draft(cfg: dict, config_path: Path) -> None:
    """'Regenerate from scratch': always calls the LLM for a brand-new,
    actuary-untouched draft and snapshots it as kind="initial" — never reuses a
    cached draft, never overwrites a prior snapshot."""
    llm = _session.get_llm(cfg)
    if llm is None:
        return
    seed = load_feature_seed(config_path.parent / FEATURE_SEED_FILENAME)
    st.session_state.wb_seed = seed
    with st.spinner("Generating feature selection + grouping draft..."):
        df = _session.get_df(cfg)
        if df is None:
            return
        draft = generate_draft(llm, df, cfg["data"], cfg.get("grouping", {}), seed=seed)
    save_draft_snapshot(draft, kind="initial", drafts_dir=run_drafts_dir(config_path))
    st.session_state.wb_draft = draft
    st.session_state.wb_iteration += 1
    st.session_state.wb_comment_round = {}
    _session.get_logger(config_path).log(
        "feature_proposal", stage="feature_selection", iteration=st.session_state.wb_iteration,
        numeric=[f.model_dump() for f in draft.numeric],
        categorical=[f.model_dump() for f in draft.categorical],
        excluded=list(draft.excluded), exclusion_rationale=draft.exclusion_rationale,
    )


# ── Locked view (shown when no draft is in progress) ───────────────────────────

def _render_locked_view(cfg: dict, config_path: Path) -> None:
    features = cfg.get("features", {})
    has_checkpoint = bool(features.get("numeric") or features.get("categorical"))

    if not has_checkpoint:
        st.info("No feature selection checkpoint yet. Generate a first draft below.")
    else:
        # Force-loaded, not merely opportunistic — this view renders on every
        # script rerun (Streamlit executes every tab's body regardless of which
        # one is visible), so this pays the dataset-load cost once per browser
        # session (`_session.get_df` caches into session state) rather than on
        # every rerun. Deliberately changed from an opportunistic peek at
        # `st.session_state.get("dash_df")`: the target-leakage banner below
        # needs real data to check anything, and a checkpoint carrying leaked
        # features is exactly the case where the actuary must be warned before
        # touching a checkbox, not only after some unrelated action (Re-open,
        # GBM training, ...) happens to have already loaded the dataset.
        with st.spinner("Loading dataset..."):
            df = _session.get_df(cfg)
        proposal = proposal_from_config(cfg, df=df)
        _render_cards(
            proposal, df, iteration=_LOCKED_ITERATION, locked=True,
            target_col=cfg.get("data", {}).get("target_col"),
        )

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button(
        "Re-open", use_container_width=True, disabled=not has_checkpoint, key="wb_revise_btn",
    ):
        with st.spinner("Loading dataset..."):
            df = _session.get_df(cfg)
        st.session_state.wb_draft = proposal_from_config(cfg, df=df)
        st.session_state.wb_seed = load_feature_seed(config_path.parent / FEATURE_SEED_FILENAME)
        st.session_state.wb_iteration += 1
        st.session_state.wb_comment_round = {}
        st.rerun()
    if c2.button("Regenerate from scratch", use_container_width=True, type="primary", key="wb_regen_btn"):
        _regenerate_draft(cfg, config_path)
        st.rerun()


# ── Shared card rendering (edit form + locked view) ────────────────────────────

def _column_kind(df: pd.DataFrame | None, col: str) -> str:
    if df is None or col not in df.columns:
        return "unknown"
    return "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"


def _stats_line(df: pd.DataFrame | None, col: str, kind: str) -> str | None:
    """One-line summary stats: mean/std for numeric, n observations for categorical."""
    if df is None or col not in df.columns:
        return None
    series = df[col]
    n = int(series.notna().sum())
    if kind == "numeric":
        return f"📊 n={n:,}  ·  mean={series.mean():,.2f}  ·  std={series.std():,.2f}"
    n_unique = int(series.nunique())
    return f"📊 n={n:,} observations  ·  {n_unique} unique values"


def _feature_card(
    name: str,
    kind: str,
    description: str,
    data_quality_note: str | None,
    default_checked: bool,
    comment_history: list[CommentEntry] | None,
    iteration: int,
    df: pd.DataFrame | None = None,
    grouping: dict[str, list[str]] | None = None,
    exclusion_note: str | None = None,
    locked: bool = False,
) -> tuple[bool, str, bool]:
    """Render one variable's card. `comment_history=None` means there's nowhere to
    persist one (Not Proposed cards are bare column names, not full Pydantic
    objects) — falls back to a plain scratch comment box with no save button, same
    as before comment history existed. Returns (checked, comment_box_text, saved).
    """
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        checked = c1.checkbox(
            "Include", value=default_checked, key=f"iter{iteration}_include_{name}", disabled=locked,
        )
        c2.markdown(f"**{name}**  ·  _{kind}_")
        stats_line = _stats_line(df, name, kind)
        if stats_line:
            c2.caption(stats_line)
        if description:
            c2.markdown(f"**Rationale:** {description}")
        if exclusion_note:
            c2.caption(f"Why not proposed: {exclusion_note}")
        if data_quality_note:
            c2.caption(f"⚠️ {data_quality_note}")
        if grouping:
            with st.expander(f"{len(grouping)} clusters"):
                rows = [
                    {
                        "Cluster": k,
                        "Original Values": "  |  ".join(str(e) for e in v),
                        "# Values": len(v),
                        "# Observations": (
                            int(df[name].isin(v).sum()) if df is not None and name in df.columns else None
                        ),
                    }
                    for k, v in grouping.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if comment_history is None:
            with c2:
                comment = st.text_area(
                    "Comment for agent", value="", key=f"iter{iteration}_comment_{name}",
                    height=68, disabled=locked,
                )
            return checked, comment, False

        with c2:
            render_comment_history(comment_history)

            round_ = st.session_state.wb_comment_round.get(name, 0)
            cc1, cc2 = st.columns([5, 1])
            comment = cc1.text_area(
                "Comment for agent", value="", key=f"wb_comment_{name}_{round_}", height=68,
                disabled=locked, label_visibility="collapsed",
            )
            saved = False
            if not locked:
                saved = cc2.form_submit_button(
                    "💾", key=f"iter{iteration}_save_{name}", help="Save this comment",
                )
    return checked, comment, saved


def _render_leakage_warning(proposal: FeatureProposal, df: pd.DataFrame | None, target_col: str | None) -> None:
    """Code-computed target-leakage check, shown alongside (never instead of) the
    agent's own per-column `data_quality_note` — see `core.data_quality.
    detect_target_leakage` for why a single-feature correlation check alone would
    have missed the actual bug this closes (seven premium sub-components whose
    *sum*, not any one of them, reconstructs the target)."""
    if df is None or not target_col or target_col not in df.columns:
        return
    approved_numeric = [f.name for f in proposal.numeric if f.approved is not False]
    flagged = detect_target_leakage(df, approved_numeric, target_col)
    if flagged is None:
        return

    lines = []
    if flagged["individual"]:
        items = ", ".join(f":red[**{name}**] (|r|={corr:.3f})" for name, corr in flagged["individual"].items())
        lines.append(f"Individually correlated with `{target_col}`: {items}.")
    if flagged["combined_flag"]:
        contributors = ", ".join(f":red[{name}]" for name in flagged["top_contributors"])
        lines.append(
            f"Combined, the approved numeric features explain **{flagged['combined_r2']:.6f}** "
            f"of `{target_col}`'s variance (R²) — check for a subset that sums to or otherwise "
            f"reconstructs the target. Most-correlated individually: {contributors}."
        )
    st.warning(
        "⚠️ **Possible target leakage** among approved numeric features:\n\n" + "\n\n".join(lines)
    )


def _render_cards(
    proposal: FeatureProposal, df: pd.DataFrame | None, iteration: int, locked: bool,
    target_col: str | None = None,
) -> tuple[dict[str, bool], dict[str, str], dict[str, tuple[bool, str]], dict[str, bool]]:
    """Render the three-tab card layout shared by the edit form and the locked view.

    Returns (checkbox_state, comment_state, excluded_state, save_clicks) — unused
    by callers when `locked` (nothing gets submitted), but harmless to collect
    either way. `save_clicks` only ever has entries for numeric/categorical cards.
    """
    _render_leakage_warning(proposal, df, target_col)

    checkbox_state: dict[str, bool] = {}
    comment_state: dict[str, str] = {}
    excluded_state: dict[str, tuple[bool, str]] = {}
    save_clicks: dict[str, bool] = {}

    # Tab placement within numeric/categorical is by `approved`, not list
    # membership — an unchecked feature stays a full NumericFeatureConfig/
    # CategoricalFeatureConfig object (see reconcile_membership) so it renders
    # in "Not Proposed" via the same full-card path (comment history, Claude
    # icon) as any other card, instead of collapsing to a bare column name.
    # Only `proposal.excluded` (columns the agent never proposed a card for at
    # all) uses that bare-name fallback.
    numeric_shown = [f for f in proposal.numeric if f.approved is not False]
    numeric_hidden = [f for f in proposal.numeric if f.approved is False]
    categorical_shown = [f for f in proposal.categorical if f.approved is not False]
    categorical_hidden = [f for f in proposal.categorical if f.approved is False]
    not_proposed_count = len(proposal.excluded) + len(numeric_hidden) + len(categorical_hidden)

    tab_numeric, tab_categorical, tab_excluded = st.tabs([
        f"Numerical ({len(numeric_shown)})",
        f"Categorical ({len(categorical_shown)})",
        f"Not Proposed ({not_proposed_count})",
    ])

    with tab_numeric:
        for feat in numeric_shown:
            checked, comment, saved = _feature_card(
                feat.name, "numeric", feat.description, feat.data_quality_note,
                True, feat.comment_history, iteration, df=df, locked=locked,
            )
            checkbox_state[feat.name] = checked
            comment_state[feat.name] = comment
            save_clicks[feat.name] = saved

    with tab_categorical:
        for feat in categorical_shown:
            checked, comment, saved = _feature_card(
                feat.name, "categorical", feat.description, feat.data_quality_note,
                True, feat.comment_history, iteration, df=df,
                grouping=feat.grouping, locked=locked,
            )
            checkbox_state[feat.name] = checked
            comment_state[feat.name] = comment
            save_clicks[feat.name] = saved

    with tab_excluded:
        if not_proposed_count == 0:
            st.caption("Nothing excluded — every dataset column is currently proposed.")

        for feat in numeric_hidden:
            checked, comment, saved = _feature_card(
                feat.name, "numeric", feat.description, feat.data_quality_note,
                False, feat.comment_history, iteration, df=df, locked=locked,
            )
            checkbox_state[feat.name] = checked
            comment_state[feat.name] = comment
            save_clicks[feat.name] = saved

        for feat in categorical_hidden:
            checked, comment, saved = _feature_card(
                feat.name, "categorical", feat.description, feat.data_quality_note,
                False, feat.comment_history, iteration, df=df,
                grouping=feat.grouping, locked=locked,
            )
            checkbox_state[feat.name] = checked
            comment_state[feat.name] = comment
            save_clicks[feat.name] = saved

        for col in proposal.excluded:
            checked, comment, _saved = _feature_card(
                col, _column_kind(df, col), proposal.excluded_description.get(col, ""),
                None, False, None, iteration, df=df,
                exclusion_note=proposal.exclusion_rationale.get(col, ""), locked=locked,
            )
            excluded_state[col] = (checked, comment)

    return checkbox_state, comment_state, excluded_state, save_clicks


# ── Edit form ───────────────────────────────────────────────────────────────────

def _render_edit_form(cfg: dict, config_path: Path) -> None:
    draft: FeatureProposal = st.session_state.wb_draft
    it = st.session_state.wb_iteration
    st.caption(f"Draft iteration {it} — review, comment, then Update or Finalize.")

    df_for_stats = _session.get_df(cfg)

    with st.form("workbench_form"):
        checkbox_state, comment_state, excluded_state, save_clicks = _render_cards(
            draft, df_for_stats, it, locked=False, target_col=cfg.get("data", {}).get("target_col"),
        )

        col_rerun, col_finalize = st.columns(2)
        submit_rerun = col_rerun.form_submit_button(
            "Update", use_container_width=True, key="wb_rerun_btn",
        )
        submit_finalize = col_finalize.form_submit_button(
            "Finalize", use_container_width=True, type="primary", key="wb_finalize_btn",
        )

    if submit_rerun or submit_finalize:
        _handle_submit(
            cfg, config_path, draft, checkbox_state, comment_state, excluded_state,
            finalize=submit_finalize,
        )
        return

    saved_name = next((name for name, clicked in save_clicks.items() if clicked), None)
    if saved_name is not None:
        _handle_save_comment(draft, saved_name, comment_state[saved_name], config_path)


def _handle_save_comment(draft: FeatureProposal, name: str, text: str, config_path: Path) -> None:
    """Save one card's comment immediately — appends to history and clears the
    box, without touching any other card or costing an LLM call. Everything else
    on screen (other cards' checkboxes/comments) is untouched Streamlit widget
    state, unapplied until a real Update/Finalize — nothing is lost."""
    text = text.strip()
    if text:
        feat = next((f for f in list(draft.numeric) + list(draft.categorical) if f.name == name), None)
        if feat is not None:
            feat.comment_history.append(CommentEntry(
                author="actuary", text=text, ts=datetime.now(timezone.utc).isoformat(),
            ))
            save_draft_snapshot(draft, kind="modified", drafts_dir=run_drafts_dir(config_path))
    st.session_state.wb_comment_round[name] = st.session_state.wb_comment_round.get(name, 0) + 1
    st.session_state.wb_draft = draft
    st.rerun()


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
    now = datetime.now(timezone.utc).isoformat()

    for feat in all_feats:
        box_text = comment_state[feat.name].strip()
        if box_text:
            # Implicit save: typed but never clicked 💾 — don't discard it.
            feat.comment_history.append(CommentEntry(author="actuary", text=box_text, ts=now))
            st.session_state.wb_comment_round[feat.name] = st.session_state.wb_comment_round.get(feat.name, 0) + 1
        unsent = [e.text for e in feat.comment_history if not e.sent]
        if unsent:
            remarks[feat.name] = "\n---\n".join(unsent)

    excluded_comments: dict[str, str] = {}
    for col, (checked, comment) in excluded_state.items():
        if checked:
            note = comment.strip() or "Actuary requests including this variable in the model."
            remarks[col] = note
            excluded_comments[col] = note

    # Tab placement is actuary-owned: recompute it from the checkbox state alone,
    # unconditionally — this is what makes an uncommented uncheck move a variable
    # to "Not Proposed" with zero LLM calls. Merge the two checkbox dicts into one.
    merged_checkbox_state = {**checkbox_state, **{col: checked for col, (checked, _) in excluded_state.items()}}
    df = _session.get_df(cfg)
    draft = reconcile_membership(draft, merged_checkbox_state, df, comments=excluded_comments)

    logger = _session.get_logger(config_path)
    session_id = _session.get_session_id(config_path)
    if remarks:
        logger.log(
            "feature_remarks", stage="feature_selection",
            iteration=st.session_state.wb_iteration, remarks=remarks,
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
            draft = refine_draft(
                llm, df, cfg["data"], cfg.get("grouping", {}), draft, remarks,
                seed=st.session_state.wb_seed,
            )
        # Defense-in-depth: the agent's response can't move a variable even if it
        # tried to — re-apply the actuary's true membership on top of it, same
        # checkbox state as above.
        draft = reconcile_membership(draft, merged_checkbox_state, df, comments=excluded_comments)
        st.session_state.wb_iteration += 1
        logger.log(
            "feature_proposal", stage="feature_selection", iteration=st.session_state.wb_iteration,
            numeric=[f.model_dump() for f in draft.numeric],
            categorical=[f.model_dump() for f in draft.categorical],
            excluded=list(draft.excluded), exclusion_rationale=draft.exclusion_rationale,
        )

    if not finalize:
        save_draft_snapshot(draft, kind="modified", drafts_dir=run_drafts_dir(config_path))
        st.session_state.wb_draft = draft
        if not remarks:
            st.info("Selection updated — no comments to send to the agent.")
        st.rerun()
        return

    invalidated = save_feature_checkpoint(config_path, cfg, draft)
    save_draft_snapshot(draft, kind="finalized", drafts_dir=run_drafts_dir(config_path))

    approved_names = [f.name for f in (list(draft.numeric) + list(draft.categorical)) if f.approved is True]
    logger.log(
        "feature_selection_complete", stage="feature_selection",
        iterations=st.session_state.wb_iteration, approved=approved_names,
    )
    if invalidated:
        logger.log(
            "checkpoints_invalidated", stage="feature_selection",
            reason="Approved feature set changed on finalize; GBM/GLM checkpoints cleared.",
        )
    for cat in draft.categorical:
        if cat.approved and cat.grouping:
            logger.log(
                "grouping_complete", stage="grouping", col_name=cat.name,
                iterations=st.session_state.wb_iteration, final_clusters=cat.grouping,
            )

    decisions_log_path = run_decisions_log_path(config_path)
    _save_feature_decisions(draft, session_id, decisions_log_path)
    for cat in draft.categorical:
        if cat.grouping:
            response = GroupingResponse(clusters=[
                CategoryCluster(cluster_name=k, elements=v, rationale="")
                for k, v in cat.grouping.items()
            ])
            _save_grouping_decisions(cat.name, response, session_id, decisions_log_path)

    st.session_state.wb_draft = None
    st.cache_data.clear()
    if invalidated:
        st.warning(
            "Checkpoint saved — the approved feature set changed, so the existing "
            "GBM/GLM checkpoints were cleared. Retrain before reviewing GLM results."
        )
    else:
        st.success("Checkpoint saved — downstream GBM/GLM stages will use this feature set.")
    st.rerun()


# ── Snapshot loader (always visible, regardless of edit/locked state) ──────────

def _snapshot_label(path: Path) -> str:
    stem = path.stem.removeprefix("feature_draft_")
    try:
        label = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        label = stem
    try:
        proposal = load_draft_snapshot(path)
        label += f" — {len(proposal.numeric)} numeric, {len(proposal.categorical)} categorical"
    except Exception:
        pass
    return label


def _load_snapshot_into_draft(path: Path, config_path: Path) -> None:
    st.session_state.wb_draft = load_draft_snapshot(path)
    st.session_state.wb_seed = load_feature_seed(config_path.parent / FEATURE_SEED_FILENAME)
    st.session_state.wb_iteration += 1
    st.session_state.wb_comment_round = {}


def _request_snapshot_load(path: Path, config_path: Path) -> None:
    """Load immediately if nothing's at risk; otherwise defer to the confirm step
    below, which is the only place `wb_draft` actually gets overwritten in that case."""
    if st.session_state.wb_draft is None:
        _load_snapshot_into_draft(path, config_path)
    else:
        st.session_state.wb_pending_snapshot_load = path


def _render_snapshot_picker(col, kind: str, label: str, config_path: Path) -> None:
    snapshots = list_draft_snapshots(kind, run_drafts_dir(config_path))
    with col:
        st.caption(f"{label} ({len(snapshots)})")
        pick = st.selectbox(
            label, snapshots, format_func=_snapshot_label, key=f"wb_pick_{kind}",
            index=None, placeholder="Select a snapshot…", label_visibility="collapsed",
        )
        if st.button(
            "Load", key=f"wb_load_{kind}_btn", disabled=pick is None, use_container_width=True,
        ):
            _request_snapshot_load(pick, config_path)
            st.rerun()


def _render_snapshot_loader(config_path: Path) -> None:
    with st.expander("📂 Load a saved snapshot"):
        c1, c2, c3 = st.columns(3)
        _render_snapshot_picker(c1, "initial", "Initial agent proposals", config_path)
        _render_snapshot_picker(c2, "modified", "Modified drafts", config_path)
        _render_snapshot_picker(c3, "finalized", "Finalized checkpoints", config_path)

    pending = st.session_state.wb_pending_snapshot_load
    if pending is not None:
        if st.session_state.wb_draft is not None:
            st.warning(f"Loading **{pending.name}** will discard your current unsaved draft. Continue?")
        else:
            st.info(f"Load **{pending.name}**?")
        cc1, cc2 = st.columns(2)
        if cc1.button("Yes, load it", key="wb_confirm_load_btn", type="primary", use_container_width=True):
            _load_snapshot_into_draft(pending, config_path)
            st.session_state.wb_pending_snapshot_load = None
            st.rerun()
        if cc2.button("Cancel", key="wb_cancel_load_btn", use_container_width=True):
            st.session_state.wb_pending_snapshot_load = None
            st.rerun()
