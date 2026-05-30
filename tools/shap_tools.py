"""Phase 3 — SHAP interaction values and H-statistic computation."""

from __future__ import annotations


def compute_shap_interaction_values(model, X):
    """Return SHAP interaction value matrix (n_samples, n_features, n_features)."""
    raise NotImplementedError("Phase 3 — implement SHAP interaction values")


def compute_h_statistics(model, X, feature_names: list[str]) -> list[dict]:
    """Compute Friedman H-statistics for all pairwise interactions, ranked descending."""
    raise NotImplementedError("Phase 3 — implement H-statistic computation")
