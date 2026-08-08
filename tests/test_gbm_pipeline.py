"""Unit tests for core.gbm_pipeline. No LLM calls, small synthetic model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from core.gbm_pipeline import save_gbm_checkpoint, train_gbm
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


def test_train_gbm_uses_only_approved_features(synthetic_df, proposal, data_cfg, gbm_cfg):
    proposal.numeric.append(NumericFeatureConfig(name="unapproved", description="d", approved=False))
    synthetic_df = synthetic_df.assign(unapproved=1.0)

    agent, interactions = train_gbm(synthetic_df, proposal, data_cfg, gbm_cfg)

    feature_names = {f["feature"] for f in agent.feature_importances}
    assert feature_names == {"driver_age", "region"}
    assert isinstance(interactions, list)


def test_save_gbm_checkpoint_persists_interactions_and_importances(synthetic_df, proposal, data_cfg, gbm_cfg, tmp_path):
    agent, interactions = train_gbm(synthetic_df, proposal, data_cfg, gbm_cfg)
    config_path = tmp_path / "project_config.yaml"
    config = {"features": {}}

    save_gbm_checkpoint(config_path, config, agent, interactions)

    assert config["gbm_output"]["interactions"] == interactions
    assert config["gbm_output"]["feature_importances"] == agent.feature_importances

    saved = yaml.safe_load(config_path.read_text())
    assert saved["gbm_output"]["interactions"] == interactions
