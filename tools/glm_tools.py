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
    """
    fam = _FAMILIES[family.lower()]
    offset = np.log(df[exposure_col])
    model = smf.glm(formula=formula, data=df, family=fam, offset=offset)
    return model.fit()


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
