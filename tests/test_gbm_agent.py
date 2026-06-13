"""Unit tests for GBMAgent and compute_h_statistics. No LLM calls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.gbm_agent import GBMAgent
from tools.shap_tools import compute_h_statistics, _h_stat


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 2_000
    age = rng.integers(18, 80, n).astype(float)
    vehicle_age = rng.integers(0, 20, n).astype(float)
    region = rng.choice(["NORTH", "SOUTH", "EAST", "WEST"], n)
    exposure = rng.uniform(0.5, 1.0, n)
    # Log-rate has a real interaction between age and vehicle_age
    log_rate = -3.0 + 0.02 * age + 0.05 * vehicle_age + 0.001 * age * vehicle_age
    premium = np.exp(log_rate) * exposure * np.exp(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "driver_age": age,
        "vehicle_age": vehicle_age,
        "region": region,
        "total_exposure": exposure,
        "total_premium": premium,
    })


@pytest.fixture
def gbm_config():
    return {
        "n_rounds": 50,
        "early_stopping_rounds": 10,
        "val_fraction": 0.2,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "top_n_features": 3,
        "h_stat_n_sample": 100,
        "h_stat_grid_size": 8,
    }


# ── _prepare_data ─────────────────────────────────────────────────────────────

def test_prepare_data_log_rate_target(synthetic_df, gbm_config):
    agent = GBMAgent(gbm_config)
    X, y = agent._prepare_data(
        synthetic_df, ["driver_age", "vehicle_age", "region"], "total_premium", "total_exposure"
    )
    expected_y = np.log(synthetic_df["total_premium"] / synthetic_df["total_exposure"])
    np.testing.assert_allclose(y, expected_y.to_numpy(), rtol=1e-6)


def test_prepare_data_encodes_categoricals(synthetic_df, gbm_config):
    agent = GBMAgent(gbm_config)
    X, _ = agent._prepare_data(
        synthetic_df, ["driver_age", "vehicle_age", "region"], "total_premium", "total_exposure"
    )
    assert pd.api.types.is_integer_dtype(X["region"])
    assert "region" in agent._cat_cols


def test_prepare_data_numeric_cols_unchanged(synthetic_df, gbm_config):
    agent = GBMAgent(gbm_config)
    X, _ = agent._prepare_data(
        synthetic_df, ["driver_age", "vehicle_age"], "total_premium", "total_exposure"
    )
    pd.testing.assert_series_equal(X["driver_age"], synthetic_df["driver_age"], check_names=False)


# ── GBMAgent.run ──────────────────────────────────────────────────────────────

def test_run_returns_sorted_interactions(synthetic_df, gbm_config):
    agent = GBMAgent(gbm_config)
    interactions = agent.run(
        df=synthetic_df,
        feature_cols=["driver_age", "vehicle_age", "region"],
        target_col="total_premium",
        exposure_col="total_exposure",
    )
    assert isinstance(interactions, list)
    assert len(interactions) == 3  # C(3,2) = 3 pairs
    h_vals = [r["h_statistic"] for r in interactions]
    assert h_vals == sorted(h_vals, reverse=True), "Interactions must be sorted descending"


def test_run_interaction_keys(synthetic_df, gbm_config):
    agent = GBMAgent(gbm_config)
    interactions = agent.run(
        df=synthetic_df,
        feature_cols=["driver_age", "vehicle_age", "region"],
        target_col="total_premium",
        exposure_col="total_exposure",
    )
    for row in interactions:
        assert {"feature_a", "feature_b", "h_statistic"} == set(row.keys())
        assert 0.0 <= row["h_statistic"] <= 1.0 + 1e-6


def test_run_detects_planted_interaction(synthetic_df, gbm_config):
    """The planted age * vehicle_age interaction should rank in the top pair."""
    agent = GBMAgent(gbm_config)
    interactions = agent.run(
        df=synthetic_df,
        feature_cols=["driver_age", "vehicle_age", "region"],
        target_col="total_premium",
        exposure_col="total_exposure",
    )
    top = interactions[0]
    top_pair = {top["feature_a"], top["feature_b"]}
    assert top_pair == {"driver_age", "vehicle_age"}


# ── _h_stat ───────────────────────────────────────────────────────────────────

def test_h_stat_zero_when_additive():
    """If F_ij = F_i + F_j exactly, H-stat should be ~0."""
    g = 5
    pd_i = np.arange(g, dtype=float)
    pd_j = np.arange(g, dtype=float) * 0.5
    pd2 = pd_i[:, None] + pd_j[None, :]  # perfectly additive
    h = _h_stat(pd2, pd_i, pd_j)
    assert h < 1e-6


def test_h_stat_bounded():
    rng = np.random.default_rng(1)
    pd2 = rng.standard_normal((8, 8))
    pd_i = rng.standard_normal(8)
    pd_j = rng.standard_normal(8)
    h = _h_stat(pd2, pd_i, pd_j)
    assert 0.0 <= h


# ── compute_h_statistics ──────────────────────────────────────────────────────

def test_compute_h_statistics_top_n_cutoff(synthetic_df, gbm_config):
    """top_n_features=2 → only 1 pair evaluated."""
    agent = GBMAgent({**gbm_config, "top_n_features": 2})
    interactions = agent.run(
        df=synthetic_df,
        feature_cols=["driver_age", "vehicle_age", "region"],
        target_col="total_premium",
        exposure_col="total_exposure",
    )
    assert len(interactions) == 1
