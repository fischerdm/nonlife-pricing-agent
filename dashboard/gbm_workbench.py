"""In-dashboard GBM retrain trigger (Streamlit Layer 2).

Runs GBM training + H-statistics synchronously in the Streamlit process, the
same way the feature workbench calls its agents directly, rather than
requiring a trip to the CLI orchestrator.
"""

from pathlib import Path

import streamlit as st

from core.feature_pipeline import apply_groupings, proposal_from_config
from core.gbm_pipeline import save_gbm_checkpoint, train_gbm
from dashboard import _session


def render_gbm_control(cfg: dict, config_path: Path) -> None:
    """Render the (re)train button. Call before reading cfg['gbm_output'] for display."""
    _session.init_state()

    has_checkpoint = bool(cfg.get("gbm_output", {}).get("interactions"))
    label = "🔁 Retrain GBM" if has_checkpoint else "🔁 Train GBM"

    if not has_checkpoint:
        st.info(
            "No GBM checkpoint yet — either this is the first run, or the feature "
            "set changed since the last GBM training. Train it to refresh the "
            "H-statistics the GLM distillation step reads from."
        )

    if st.button(label):
        approved = cfg.get("features", {}).get("numeric", []) + cfg.get("features", {}).get("categorical", [])
        if not approved:
            st.error("No approved features yet — finalize the Feature & Grouping Workbench first.")
            return

        with st.spinner("Training GBM and computing H-statistics — this can take a minute..."):
            df = _session.get_df(cfg)
            proposal = proposal_from_config(cfg)
            grouped_df = apply_groupings(df, proposal)
            agent, interactions = train_gbm(grouped_df, proposal, cfg["data"], cfg.get("gbm", {}))

        save_gbm_checkpoint(config_path, cfg, agent, interactions)
        _session.get_logger().log(
            "gbm_complete", stage="gbm",
            feature_importances=agent.feature_importances, interactions=interactions,
        )
        st.cache_data.clear()
        st.success("GBM trained — checkpoint saved.")
        st.rerun()
