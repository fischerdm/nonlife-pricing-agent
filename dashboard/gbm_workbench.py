"""In-dashboard GBM retrain trigger (Streamlit Layer 2).

Runs GBM training + H-statistics synchronously in the Streamlit process, the
same way the feature workbench calls its agents directly, rather than
requiring a trip to the CLI orchestrator.

Training always runs against one *finalized* feature checkpoint — picked from
the same `reports/drafts/finalized/` history the Feature & Grouping Workbench
snapshots into on every Finalize (see `core.feature_pipeline.save_draft_
snapshot`). Defaults to whichever checkpoint is currently active in
project_config.yaml; picking an older finalized snapshot instead restores it
as the active checkpoint first — same `save_feature_checkpoint` invalidation
semantics as re-finalizing the Feature Workbench itself — and then trains on it.

Every `gbm_complete` session-log event records which feature snapshot fed the
run (`feature_source`: `{"kind", "label", "ts"[, "path"]}` — `ts` is the plain
timestamp for lineage display, `label` the richer picker-dropdown description)
— this doubles as GBM's own run history for `core.gbm_pipeline.list_gbm_runs()`
(GBM has no actuary review loop, so there's no separate draft/finalized
snapshot file for it) and as the provenance the Audit Trail's Model Lineage
view reads back.
"""

from pathlib import Path

import streamlit as st

from core.feature_pipeline import (
    apply_groupings,
    list_draft_snapshots,
    load_draft_snapshot,
    proposal_from_config,
    save_feature_checkpoint,
)
from core.gbm_pipeline import save_gbm_checkpoint, train_gbm
from core.snapshot_utils import snapshot_ts
from dashboard import _session

_CURRENT_OPTION = "Current checkpoint (project_config.yaml)"


def render_gbm_control(cfg: dict, config_path: Path) -> None:
    """Render the finalized-version picker + (re)train button."""
    _session.init_state()

    finalized = list_draft_snapshots("finalized")
    features = cfg.get("features", {})
    has_active_features = bool(features.get("numeric") or features.get("categorical"))

    if not finalized and not has_active_features:
        st.info("No finalized feature selection yet — finalize the Feature & Grouping Workbench first.")
        return

    options = [_CURRENT_OPTION, *finalized]
    pick = st.selectbox(
        "Feature set to train on", options, format_func=_option_label, key="gbm_snapshot_pick",
    )

    has_checkpoint = bool(cfg.get("gbm_output", {}).get("interactions"))
    label = "🔁 Retrain GBM" if has_checkpoint else "🔁 Train GBM"

    if not has_checkpoint:
        st.info(
            "No GBM checkpoint yet — either this is the first run, or the feature "
            "set changed since the last GBM training. Train it to refresh the "
            "H-statistics the GLM distillation step reads from."
        )

    if st.button(label):
        invalidated = False
        if pick == _CURRENT_OPTION:
            if not has_active_features:
                st.error("No approved features yet — finalize the Feature & Grouping Workbench first.")
                return
            proposal = proposal_from_config(cfg)
            # "Current" always coincides with the newest finalized snapshot — the
            # only two writers of project_config.yaml's features (Finalize, and
            # this control's own restore branch below) always keep them in sync —
            # so it's a safe, concrete stand-in for lineage purposes.
            label = _snapshot_label(finalized[0]) if finalized else "current (no finalized snapshot on disk)"
            ts = snapshot_ts(finalized[0], "feature_draft_") if finalized else None
            feature_source = {"kind": "current", "label": label, "ts": ts}
        else:
            with st.spinner("Restoring the selected finalized feature set..."):
                proposal = load_draft_snapshot(pick)
                invalidated = save_feature_checkpoint(config_path, cfg, proposal)
            feature_source = {
                "kind": "finalized_snapshot", "label": _snapshot_label(pick),
                "path": str(pick), "ts": snapshot_ts(pick, "feature_draft_"),
            }

        with st.spinner("Training GBM and computing H-statistics — this can take a minute..."):
            df = _session.get_df(cfg)
            grouped_df = apply_groupings(df, proposal)
            agent, interactions = train_gbm(grouped_df, proposal, cfg["data"], cfg.get("gbm", {}))

        save_gbm_checkpoint(config_path, cfg, agent, interactions)
        _session.get_logger().log(
            "gbm_complete", stage="gbm",
            feature_importances=agent.feature_importances, interactions=interactions,
            feature_source=feature_source,
        )
        st.cache_data.clear()
        if invalidated:
            st.warning(
                "GBM trained on the restored feature set — its GLM distillation terms "
                "were cleared since the feature set changed. Re-run distillation."
            )
        else:
            st.success("GBM trained — checkpoint saved.")
        st.rerun()


def _option_label(opt: str | Path) -> str:
    return opt if isinstance(opt, str) else _snapshot_label(opt)


def _snapshot_label(path: Path) -> str:
    """Rich, picker-dropdown label. `core.snapshot_utils.snapshot_ts` gives the
    plain, single-level timestamp used for lineage logging instead."""
    label = snapshot_ts(path, "feature_draft_")
    try:
        proposal = load_draft_snapshot(path)
        label += f" — {len(proposal.numeric)} numeric, {len(proposal.categorical)} categorical"
    except Exception:
        pass
    return label
