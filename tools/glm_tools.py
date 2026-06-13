"""Phase 3 — statsmodels GLM wrapper for the distillation output."""

from __future__ import annotations

import pandas as pd


def fit_glm(
    df: pd.DataFrame,
    formula: str,
    target_col: str,
    exposure_col: str,
    family: str = "poisson",
):
    """Fit a GLM using the given patsy formula and return the fitted model."""
    raise NotImplementedError("Phase 3 — implement GLM fitting with statsmodels")


def build_formula(base_features: list[str], interaction_terms: list[str]) -> str:
    """Build a patsy formula string from main effects and interaction terms."""
    raise NotImplementedError("Phase 3 — implement formula builder")
