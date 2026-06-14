"""Phase 3 — statsmodels GLM wrapper for the distillation output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from core.schemas import GLMTerm


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
