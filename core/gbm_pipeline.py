"""Shared GBM training + checkpoint logic, used by both the Orchestrator and the dashboard.

`train_gbm` expects `df` to already have approved-categorical groupings applied
(see `core.feature_pipeline.apply_groupings`) — it does not apply them itself,
so a caller that fits both GBM and GLM on the same dataframe only groups once.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from agents.gbm_agent import GBMAgent
from core.run_scope import gbm_model_path
from core.schemas import FeatureProposal


def train_gbm(
    df: pd.DataFrame,
    proposal: FeatureProposal,
    data_cfg: dict,
    gbm_cfg: dict,
    config_path: Path,
) -> tuple[GBMAgent, list[dict]]:
    """Train a GBM on the approved feature set and return (agent, ranked H-statistics).

    `config_path` is used only to resolve `gbm_cfg["model_path"]` under the
    active run's own reports/ dir (see `core.run_scope.gbm_model_path`) —
    without it, a bare relative filename in project_config.yaml would resolve
    against the process's CWD instead of the run that's actually training.
    """
    feature_cols = (
        [f.name for f in proposal.numeric if f.approved]
        + [f.name for f in proposal.categorical if f.approved]
    )
    resolved_cfg = dict(gbm_cfg)
    if gbm_cfg.get("model_path"):
        resolved_cfg["model_path"] = str(gbm_model_path(config_path, gbm_cfg["model_path"]))
    agent = GBMAgent(resolved_cfg)
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


# ── Run history (no actuary review loop, so no draft/finalized distinction —
# every Train/Retrain is immediately "final". Rather than a parallel snapshot-
# file system like the Feature/GLM Distillation workbenches have, GBM's run
# history is read back from the session log's own `gbm_complete` events, which
# already carry everything a later stage would need (interactions,
# feature_importances) — see CLAUDE.md for the reasoning.) ──────────────────

def list_gbm_runs(sessions_dir: Path) -> list[dict]:
    """Every `gbm_complete` event across all session logs in `sessions_dir`,
    newest first (see `core.run_scope.sessions_dir`)."""
    runs: list[dict] = []
    if not sessions_dir.exists():
        return runs
    for path in sorted(sessions_dir.glob("session_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("event") == "gbm_complete":
                    event["_session"] = path.stem
                    runs.append(event)
    runs.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return runs


def restore_gbm_run(config_path: Path, config: dict, run: dict) -> None:
    """Make a historical GBM run (from `list_gbm_runs()`) the active `gbm_output`
    checkpoint — e.g. so GLM Distillation can be re-seeded from an older run
    without retraining, the same "restore, then use" pattern as
    `feature_pipeline.save_feature_checkpoint` for an older feature snapshot.
    """
    config["gbm_output"] = {
        "interactions": run["interactions"],
        "feature_importances": run["feature_importances"],
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"GBM checkpoint restored to run from {run.get('ts')} at {config_path}")
