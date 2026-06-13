import numpy as np
import pandas as pd
import pytest

from core.schemas import InteractionHypothesis, ValidationResult
from core.validator import Validator


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n = 2000
    driver_age = rng.uniform(18, 80, n)
    vehicle_power_kw = rng.uniform(50, 200, n)
    return pd.DataFrame({
        "driver_age": driver_age,
        "vehicle_power_kw": vehicle_power_kw,
        "claim_count": rng.poisson(0.1, n).astype(float),
        "exposure_years": rng.uniform(0.1, 1.0, n),
    })


@pytest.fixture
def hypothesis():
    return InteractionHypothesis(
        feature_a="driver_age",
        feature_b="vehicle_power_kw",
        operation="multiply",
        new_feature_name="age_x_kw",
        rationale="Test interaction",
    )


def test_validator_returns_validation_result(synthetic_df, hypothesis):
    config = {"objective": "poisson", "n_rounds": 50, "early_stopping_rounds": 10}
    validator = Validator(config)

    result = validator.validate(
        df=synthetic_df,
        feature_cols=["driver_age", "vehicle_power_kw"],
        target_col="claim_count",
        exposure_col="exposure_years",
        hypothesis=hypothesis,
        new_feature_col="age_x_kw",
    )

    assert isinstance(result, ValidationResult)
    assert result.hypothesis == hypothesis
    assert isinstance(result.deviance_delta_pct, float)
    assert isinstance(result.gain_rank, int)
    assert result.gain_rank >= 1
    assert result.approved is None


def test_validator_deviance_delta_is_bounded(synthetic_df, hypothesis):
    config = {"objective": "poisson", "n_rounds": 50, "early_stopping_rounds": 10}
    validator = Validator(config)

    result = validator.validate(
        df=synthetic_df,
        feature_cols=["driver_age", "vehicle_power_kw"],
        target_col="claim_count",
        exposure_col="exposure_years",
        hypothesis=hypothesis,
        new_feature_col="age_x_kw",
    )
    # Delta should be within a reasonable range (not orders of magnitude off)
    assert -50 < result.deviance_delta_pct < 50
