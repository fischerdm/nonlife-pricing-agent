"""Phase 3 — statsmodels GLM wrapper for the distillation output."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rich.console import Console
from rich.table import Table
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from core.schemas import GLMTerm

_console = Console()


_FAMILIES = {
    "gamma": sm.families.Gamma(link=sm.families.links.Log()),
    "poisson": sm.families.Poisson(link=sm.families.links.Log()),
    "tweedie": sm.families.Tweedie(link=sm.families.links.Log()),
}


def build_formula(target_col: str, approved_terms: list[GLMTerm]) -> str:
    """Build a patsy formula string from approved GLM terms.

    Main-effect term names are used directly; interaction terms already carry
    the colon notation (e.g. 'driver_age:vehicle_age') that patsy expects.
    """
    rhs_parts = [t.name for t in approved_terms if t.approved is True]
    rhs = " + ".join(rhs_parts) if rhs_parts else "1"
    return f"{target_col} ~ {rhs}"


def fit_glm(
    df: pd.DataFrame,
    formula: str,
    target_col: str,
    exposure_col: str,
    family: str = "gamma",
) -> GLMResultsWrapper:
    """Fit a GLM with log-exposure offset and return the fitted result.

    The exposure offset (log(exposure_col)) accounts for pro-rata earned premium.
    Gamma with log link is the standard choice for severity / pure premium.

    Patsy silently drops any row with a NaN in a formula variable before
    fitting — standard, harmless behavior for a handful of rows, but
    invisible unless someone compares row counts themselves. Rather than
    leave that to each caller to remember, the fitted result always carries
    a `missing_value_report` attribute (None if nothing was dropped) so any
    caller can surface it without recomputing anything.
    """
    fam = _FAMILIES[family.lower()]
    offset = np.log(df[exposure_col])
    model = smf.glm(formula=formula, data=df, family=fam, offset=offset)
    result = model.fit()
    result.missing_value_report = missing_value_report(df, formula, exposure_col, result)
    return result


def missing_value_report(
    df: pd.DataFrame, formula: str, exposure_col: str, result: GLMResultsWrapper,
) -> dict | None:
    """Diagnose rows dropped between `df` and the fitted result's `nobs`.

    Returns None if nothing was dropped. Otherwise a dict with the drop
    count/percentage and a per-column null count (only for formula variables
    plus the exposure column, and only those that actually have nulls) so
    the actuary can see which feature(s) drove it, not just that some rows
    silently vanished.
    """
    n_total = len(df)
    n_used = int(result.nobs)
    n_dropped = n_total - n_used
    if n_dropped <= 0:
        return None

    cols = _formula_columns(formula, df.columns)
    if exposure_col in df.columns:
        cols = list(dict.fromkeys([*cols, exposure_col]))
    null_counts = {c: int(df[c].isnull().sum()) for c in cols if df[c].isnull().any()}

    return {
        "n_total": n_total,
        "n_used": n_used,
        "n_dropped": n_dropped,
        "pct_dropped": n_dropped / n_total * 100 if n_total else 0.0,
        "null_counts": dict(sorted(null_counts.items(), key=lambda kv: -kv[1])),
    }


def _formula_columns(formula: str, available_columns) -> list[str]:
    """Distinct column names referenced on either side of a patsy formula,
    matched against real dataframe columns since patsy syntax (~, +, :,
    [T.]) isn't itself a column name."""
    tokens = re.split(r"[~+:\s]+", formula)
    available = set(available_columns)
    return [t for t in dict.fromkeys(tokens) if t in available]


def format_missing_value_warning(report: dict) -> str:
    """Plain-text warning line, shared by the CLI and dashboard callers."""
    breakdown = ", ".join(f"{col} ({n:,} null)" for col, n in report["null_counts"].items())
    return (
        f"{report['n_dropped']:,} of {report['n_total']:,} rows "
        f"({report['pct_dropped']:.2f}%) were silently dropped from the fit due to "
        f"missing values. Affected column(s): {breakdown or 'unknown'}."
    )


def print_glm_summary(result: GLMResultsWrapper) -> None:
    """Print coefficient table and key diagnostics."""
    print(result.summary())
    print(f"\nDeviance:       {result.deviance:.4f}")
    print(f"Null deviance:  {result.null_deviance:.4f}")
    print(f"% explained:    {100 * (1 - result.deviance / result.null_deviance):.2f}%")
    print(f"AIC:            {result.aic:.2f}")


def coef_summary(result: GLMResultsWrapper) -> pd.DataFrame:
    """Return a tidy DataFrame of coefficients, relativities, p-values, and exp-scale CIs."""
    ci = result.conf_int()
    tbl = pd.DataFrame({
        "coef": result.params,
        "exp_coef": np.exp(result.params),
        "p_value": result.pvalues,
        "ci_lower_exp": np.exp(ci[0]),
        "ci_upper_exp": np.exp(ci[1]),
    })
    tbl.index.name = "parameter"
    return tbl.reset_index()


def param_to_term(param: str) -> str:
    """Map a patsy parameter name back to its formula term name.

    Strips [T.level] suffixes from each colon-separated factor so that
    'vehicle_group[T.suv]' → 'vehicle_group' and
    'driver_age:vehicle_group[T.suv]' → 'driver_age:vehicle_group'.
    """
    if param == "Intercept":
        return "Intercept"
    parts = param.split(":")
    return ":".join(re.sub(r"\[T\..*\]$", "", p) for p in parts)


def _pvalue_str(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e} ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f}"


def _parse_level(param: str) -> str:
    """Extract a display-friendly level label from a patsy parameter name."""
    if param == "Intercept":
        return "—"
    if ":" in param:
        return param  # interaction: show the full parameter name
    m = re.search(r"\[T\.(.*?)\]", param)
    if m:
        return m.group(1)
    return "(per unit +1)"  # numeric main effect


def print_rating_factors(result: GLMResultsWrapper) -> None:
    """Print exp(coef) relativities grouped by term; reference levels omitted (= 1.000)."""
    summary = coef_summary(result)
    summary["term"] = summary["parameter"].map(param_to_term)

    # Preserve term order from the coefficient table
    seen: list[str] = []
    for t in summary["term"]:
        if t not in seen:
            seen.append(t)

    table = Table(title="Rating Factors (Relativities)", show_header=True)
    table.add_column("Term", style="cyan bold", no_wrap=True)
    table.add_column("Level / Parameter")
    table.add_column("Relativity", justify="right")
    table.add_column("95% CI (exp scale)", justify="right")
    table.add_column("p-value", justify="right")

    for term in seen:
        group = summary[summary["term"] == term]
        first = True
        for _, row in group.iterrows():
            p_color = "red" if row["p_value"] > 0.05 else "green"
            table.add_row(
                term if first else "",
                _parse_level(row["parameter"]),
                f"{row['exp_coef']:.4f}",
                f"[{row['ci_lower_exp']:.4f}, {row['ci_upper_exp']:.4f}]",
                f"[{p_color}]{_pvalue_str(row['p_value'])}[/{p_color}]",
            )
            first = False

    _console.print(table)
    _console.print(
        "[dim]Note: reference level for each categorical factor is not shown "
        "and has relativity 1.000[/dim]"
    )
