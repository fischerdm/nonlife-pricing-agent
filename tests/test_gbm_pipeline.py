"""Unit tests for core.gbm_pipeline. No LLM calls, small synthetic model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from core.gbm_pipeline import list_gbm_runs, restore_gbm_run, save_gbm_checkpoint, train_gbm
from core.schemas import CategoricalFeatureConfig, FeatureProposal, NumericFeatureConfig


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 1_000
    age = rng.integers(18, 80, n).astype(float)
    region = rng.choice(["NORTH", "SOUTH"], n)
    exposure = rng.uniform(0.5, 1.0, n)
    premium = np.exp(-3.0 + 0.02 * age) * exposure * np.exp(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "driver_age": age, "region": region, "total_exposure": exposure, "total_premium": premium,
    })


@pytest.fixture
def data_cfg():
    return {"target_col": "total_premium", "exposure_col": "total_exposure"}


@pytest.fixture
def gbm_cfg():
    return {
        "n_rounds": 20, "early_stopping_rounds": 5, "val_fraction": 0.2,
        "num_leaves": 7, "top_n_features": 2, "h_stat_n_sample": 50, "h_stat_grid_size": 5,
    }


@pytest.fixture
def proposal():
    return FeatureProposal(
        numeric=[NumericFeatureConfig(name="driver_age", description="d", approved=True)],
        categorical=[CategoricalFeatureConfig(name="region", description="d", approved=True)],
    )


def test_train_gbm_uses_only_approved_features(synthetic_df, proposal, data_cfg, gbm_cfg, tmp_path):
    proposal.numeric.append(NumericFeatureConfig(name="unapproved", description="d", approved=False))
    synthetic_df = synthetic_df.assign(unapproved=1.0)
    config_path = tmp_path / "config" / "project_config.yaml"

    agent, interactions = train_gbm(synthetic_df, proposal, data_cfg, gbm_cfg, config_path)

    feature_names = {f["feature"] for f in agent.feature_importances}
    assert feature_names == {"driver_age", "region"}
    assert isinstance(interactions, list)


def test_save_gbm_checkpoint_persists_interactions_and_importances(synthetic_df, proposal, data_cfg, gbm_cfg, tmp_path):
    config_path = tmp_path / "config" / "project_config.yaml"
    config_path.parent.mkdir()
    agent, interactions = train_gbm(synthetic_df, proposal, data_cfg, gbm_cfg, config_path)
    config = {"features": {}}

    save_gbm_checkpoint(config_path, config, agent, interactions)

    assert config["gbm_output"]["interactions"] == interactions
    assert config["gbm_output"]["feature_importances"] == agent.feature_importances

    saved = yaml.safe_load(config_path.read_text())
    assert saved["gbm_output"]["interactions"] == interactions


# ── GBM run history (read back from session logs) ─────────────────────────────

def _write_session(path, events: list[dict]) -> None:
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_list_gbm_runs_returns_empty_when_no_sessions_dir(tmp_path):
    assert list_gbm_runs(tmp_path / "sessions") == []


def test_list_gbm_runs_filters_to_gbm_complete_events(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    _write_session(sessions_dir / "session_a.jsonl", [
        {"ts": "2026-01-01T00:00:00", "event": "feature_selection_complete"},
        {"ts": "2026-01-01T01:00:00", "event": "gbm_complete", "interactions": [1], "feature_importances": [2]},
    ])

    runs = list_gbm_runs(sessions_dir)
    assert len(runs) == 1
    assert runs[0]["interactions"] == [1]
    assert runs[0]["_session"] == "session_a"


def test_list_gbm_runs_sorted_newest_first_across_sessions(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    _write_session(sessions_dir / "session_a.jsonl", [
        {"ts": "2026-01-01T00:00:00", "event": "gbm_complete", "interactions": [], "feature_importances": []},
    ])
    _write_session(sessions_dir / "session_b.jsonl", [
        {"ts": "2026-02-01T00:00:00", "event": "gbm_complete", "interactions": [], "feature_importances": []},
    ])

    runs = list_gbm_runs(sessions_dir)
    assert [r["ts"] for r in runs] == ["2026-02-01T00:00:00", "2026-01-01T00:00:00"]


def test_restore_gbm_run_overwrites_checkpoint_and_persists(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    config = {"features": {}, "gbm_output": {"interactions": ["stale"], "feature_importances": ["stale"]}}
    run = {"ts": "2026-01-01T00:00:00", "interactions": ["fresh"], "feature_importances": ["fresh"]}

    restore_gbm_run(config_path, config, run)

    assert config["gbm_output"] == {"interactions": ["fresh"], "feature_importances": ["fresh"]}
    saved = yaml.safe_load(config_path.read_text())
    assert saved["gbm_output"]["interactions"] == ["fresh"]
