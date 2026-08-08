"""Shared GBM training + checkpoint logic, used by both the Orchestrator and the dashboard.

`train_gbm` expects `df` to already have approved-categorical groupings applied
(see `core.feature_pipeline.apply_groupings`) — it does not apply them itself,
so a caller that fits both GBM and GLM on the same dataframe only groups once.
"""
from pathlib import Path

import pandas as pd
import yaml

from agents.gbm_agent import GBMAgent
from core.schemas import FeatureProposal


def train_gbm(
    df: pd.DataFrame,
    proposal: FeatureProposal,
    data_cfg: dict,
    gbm_cfg: dict,
) -> tuple[GBMAgent, list[dict]]:
    """Train a GBM on the approved feature set and return (agent, ranked H-statistics)."""
    feature_cols = (
        [f.name for f in proposal.numeric if f.approved]
        + [f.name for f in proposal.categorical if f.approved]
    )
    agent = GBMAgent(gbm_cfg)
    interactions = agent.run(
        df=df,
        feature_cols=feature_cols,
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
    )
    return agent, interactions


def save_gbm_checkpoint(
    config_path: Path, config: dict, agent: GBMAgent, interactions: list[dict],
) -> None:
    """Persist interactions + feature importances into project_config.yaml.

    Feature importances are checkpointed here (not just logged to the session
    JSONL) so the dashboard can render a fresh retrain without depending on
    session-log history, which may be stale or absent for a dashboard-only run.
    """
    config["gbm_output"] = {
        "interactions": interactions,
        "feature_importances": agent.feature_importances,
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"GBM checkpoint saved to {config_path}")
