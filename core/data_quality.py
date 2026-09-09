"""Structural target-leakage check for approved numeric features.

Closes part of the `target_leakage_validation_deferred` memory note: on
2026-09-08 a fresh Feature Selection pass approved seven premium
sub-component columns whose *sum* correlates with the target `total_premium`
at 0.9999999999999999 (the target literally is their sum), crashing the GLM
fit with an IRLS `NaN, inf or invalid value detected in weights` error.
Individually, none of those seven columns is suspicious — each correlates
with the target at only ~0.47-0.75 — so a naive single-feature correlation
check would have missed exactly the case that broke a real fit. This module
checks both: a single feature that's an obvious near-duplicate of the
target, and the *combined* linear fit of every approved numeric feature
against the target, which catches a leakage relationship that only shows up
once several features are considered together.

Code-computed, not LLM-guessed — same rationale as every other "structural
guarantee, not just a prompt-level note" decision in this project (seed-config
locks, `reconcile_feature_membership`): the agent's own `data_quality_note` is
prose the LLM writes about a column it's looked at in isolation, and has no
way to notice a joint relationship like this one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R2_LEAKAGE_THRESHOLD = 0.999
CORR_LEAKAGE_THRESHOLD = 0.95
_TOP_N_CONTRIBUTORS = 5


def detect_target_leakage(
    df: pd.DataFrame,
    numeric_feature_names: list[str],
    target_col: str,
    r2_threshold: float = R2_LEAKAGE_THRESHOLD,
    corr_threshold: float = CORR_LEAKAGE_THRESHOLD,
) -> dict | None:
    """Flag approved numeric features that look like they leak the target.

    Returns None if nothing is flagged. Otherwise a dict with:
    - `individual`: {feature: |corr|} for any feature whose own correlation
      with the target exceeds `corr_threshold`, sorted descending.
    - `combined_r2`: R² of an OLS fit of the target on every approved numeric
      feature together (None if fewer than two features are present).
    - `combined_flag`: True if `combined_r2` exceeds `r2_threshold`.
    - `top_contributors`: the features with the highest individual |corr|
      with the target (up to `_TOP_N_CONTRIBUTORS`) — a hint, not a proof,
      of which features are driving a `combined_flag`, since no subset
      search is performed.
    """
    present = [c for c in dict.fromkeys(numeric_feature_names) if c in df.columns and c != target_col]
    if not present or target_col not in df.columns:
        return None

    usable = df[[*present, target_col]].dropna()
    if len(usable) < 10:
        return None

    correlations = usable[present].corrwith(usable[target_col]).abs().sort_values(ascending=False)
    individual_flags = correlations[correlations >= corr_threshold]

    combined_r2 = None
    if len(present) >= 2:
        X = np.column_stack([usable[present].to_numpy(dtype=float), np.ones(len(usable))])
        y = usable[target_col].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ coef
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        combined_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    combined_flag = combined_r2 is not None and combined_r2 >= r2_threshold
    if individual_flags.empty and not combined_flag:
        return None

    return {
        "individual": individual_flags.to_dict(),
        "combined_r2": combined_r2,
        "combined_flag": combined_flag,
        "top_contributors": correlations.head(_TOP_N_CONTRIBUTORS).to_dict(),
    }
