import numpy as np
import pandas as pd

from core.data_quality import detect_target_leakage


def _rng_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })


def test_returns_none_when_nothing_is_correlated():
    df = _rng_df()
    df["target"] = np.random.default_rng(1).normal(size=len(df))

    assert detect_target_leakage(df, ["a", "b", "c"], "target") is None


def test_flags_a_single_near_duplicate_feature():
    df = _rng_df()
    df["target"] = df["a"] * 2 + 1  # exact linear duplicate of "a"

    result = detect_target_leakage(df, ["a", "b", "c"], "target")

    assert result is not None
    assert "a" in result["individual"]
    assert result["individual"]["a"] > 0.99


def test_flags_a_combination_that_sums_to_the_target_even_though_no_single_feature_does():
    """The real 2026-09-08 bug: several sub-components individually correlate with
    the target at only ~0.5-0.75, but their *sum* reconstructs it almost exactly."""
    rng = np.random.default_rng(2)
    n = 1000
    parts = pd.DataFrame({f"p{i}": rng.uniform(1, 10, size=n) for i in range(5)})
    df = parts.copy()
    df["target"] = parts.sum(axis=1)

    result = detect_target_leakage(df, list(parts.columns), "target")

    assert result is not None
    assert result["combined_flag"] is True
    assert result["combined_r2"] > 0.999
    # None of the individual components should look suspicious on its own.
    assert result["individual"] == {}
    for corr in result["top_contributors"].values():
        assert corr < 0.95


def test_ignores_features_not_present_in_the_dataframe():
    df = _rng_df()
    df["target"] = df["a"] * 2

    result = detect_target_leakage(df, ["a", "not_a_real_column"], "target")

    assert result is not None
    assert "not_a_real_column" not in result["individual"]


def test_returns_none_for_missing_target_column():
    df = _rng_df()
    assert detect_target_leakage(df, ["a", "b"], "missing_target") is None


def test_returns_none_with_too_few_usable_rows():
    df = pd.DataFrame({"a": [1.0, 2.0, None], "target": [1.0, None, 3.0]})
    assert detect_target_leakage(df, ["a"], "target") is None
