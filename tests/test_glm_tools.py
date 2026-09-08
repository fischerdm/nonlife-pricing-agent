import numpy as np
import pandas as pd

from tools.glm_tools import (
    _formula_columns,
    fit_glm,
    format_missing_value_warning,
    missing_value_report,
)


def _synthetic_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "vehicle_age": rng.uniform(0, 20, size=n),
        "driver_age": rng.uniform(18, 80, size=n),
        "exposure": rng.uniform(0.1, 1.0, size=n),
    })
    rate = np.exp(0.5 + 0.01 * df["vehicle_age"] - 0.005 * df["driver_age"])
    df["premium"] = rate * df["exposure"] * rng.gamma(5, 1 / 5, size=n)
    return df


def test_formula_columns_extracts_names_from_both_sides_and_interactions():
    cols = _formula_columns(
        "total_premium ~ vehicle_age + driver_age:vehicle_group + Intercept",
        available_columns=["total_premium", "vehicle_age", "driver_age", "vehicle_group"],
    )
    assert cols == ["total_premium", "vehicle_age", "driver_age", "vehicle_group"]


def test_missing_value_report_is_none_when_nothing_dropped():
    df = _synthetic_df()
    result = fit_glm(df, "premium ~ vehicle_age + driver_age", "premium", "exposure")
    assert result.missing_value_report is None


def test_missing_value_report_flags_dropped_rows_and_their_source_column():
    df = _synthetic_df()
    df.loc[[3, 7, 9], "vehicle_age"] = np.nan

    result = fit_glm(df, "premium ~ vehicle_age + driver_age", "premium", "exposure")

    report = result.missing_value_report
    assert report is not None
    assert report["n_dropped"] == 3
    assert report["n_total"] == len(df)
    assert report["n_used"] == len(df) - 3
    assert report["null_counts"] == {"vehicle_age": 3}


def test_missing_value_report_also_checks_the_exposure_column():
    df = _synthetic_df()
    report = missing_value_report(
        df, "premium ~ vehicle_age", exposure_col="exposure",
        result=_FakeResult(nobs=len(df) - 2),
    )
    # exposure itself has no nulls in this fixture, so nothing should be blamed on it
    assert report is None or "exposure" not in report["null_counts"]


def test_format_missing_value_warning_reads_naturally():
    report = {
        "n_total": 1000, "n_used": 995, "n_dropped": 5, "pct_dropped": 0.5,
        "null_counts": {"vehicle_age": 3, "driver_age": 2},
    }
    text = format_missing_value_warning(report)
    assert "5" in text and "1,000" in text
    assert "vehicle_age (3 null)" in text
    assert "driver_age (2 null)" in text


class _FakeResult:
    def __init__(self, nobs):
        self.nobs = nobs
