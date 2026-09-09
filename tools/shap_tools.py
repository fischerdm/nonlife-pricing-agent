"""Friedman H-statistics for pairwise interaction ranking."""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd


def _feature_grid(X: pd.DataFrame, feature: str, grid_size: int, categorical: bool = False) -> np.ndarray:
    col = X[feature].dropna()
    if categorical:
        # `col` holds label-encoded category codes (see GBMAgent._prepare_data).
        # Quantile-interpolating between codes (the branch below) would
        # synthesize a fractional "category" that doesn't exist — meaningless
        # for a partial dependence, and it crashes when written back into the
        # codes' integer-dtype column (pandas raises rather than upcasting).
        # Use the actual codes present instead, evenly subsampled down to
        # grid_size if there are more of them than that.
        codes = np.sort(col.unique())
        if len(codes) > grid_size:
            idx = np.unique(np.linspace(0, len(codes) - 1, grid_size).round().astype(int))
            codes = codes[idx]
        return codes
    quantiles = np.linspace(0, 1, grid_size)
    return np.unique(np.quantile(col, quantiles))


def _pd_1d(model: lgb.Booster, X: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    out = np.empty(len(grid))
    X_tmp = X.copy()
    # Widen to float so a quantile-interpolated grid value (e.g. a numeric
    # feature that happens to be integer-dtype in the source data) can always
    # be written back — LightGBM predicts on the values, not the pandas
    # dtype, so this is safe regardless of the column's original dtype.
    if not pd.api.types.is_float_dtype(X_tmp[feature]):
        X_tmp[feature] = X_tmp[feature].astype(float)
    for k, v in enumerate(grid):
        X_tmp.loc[:, feature] = v
        out[k] = model.predict(X_tmp).mean()
    return out


def _pd_2d(
    model: lgb.Booster,
    X: pd.DataFrame,
    feat_i: str,
    feat_j: str,
    grid_i: np.ndarray,
    grid_j: np.ndarray,
) -> np.ndarray:
    mat = np.empty((len(grid_i), len(grid_j)))
    X_tmp = X.copy()
    for feat in (feat_i, feat_j):
        if not pd.api.types.is_float_dtype(X_tmp[feat]):
            X_tmp[feat] = X_tmp[feat].astype(float)
    for k, vi in enumerate(grid_i):
        X_tmp.loc[:, feat_i] = vi
        for l, vj in enumerate(grid_j):
            X_tmp.loc[:, feat_j] = vj
            mat[k, l] = model.predict(X_tmp).mean()
    return mat


def _h_stat(pd2: np.ndarray, pd_i: np.ndarray, pd_j: np.ndarray) -> float:
    """Friedman H-statistic (sqrt form, range [0, 1])."""
    pd2_c = pd2 - pd2.mean()
    residual = pd2_c - (pd_i - pd_i.mean())[:, None] - (pd_j - pd_j.mean())[None, :]
    den = float((pd2_c**2).sum())
    if den < 1e-10:
        return 0.0
    return float(np.sqrt((residual**2).sum() / den))


def compute_h_statistics(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature_names: list[str],
    cat_cols: list[str] | None = None,
    top_n_features: int = 15,
    n_sample: int = 500,
    grid_size: int = 20,
) -> list[dict]:
    """
    Friedman H-statistics for all pairs among the top-N most important features.

    Features are ranked by LightGBM gain importance (fast, no extra SHAP call needed).
    Partial dependences are averaged over a random sample of n_sample rows to keep
    computation tractable on large datasets.

    `cat_cols` names the features that are label-encoded categoricals (see
    `GBMAgent._prepare_data`) — their partial-dependence grid is built from the
    actual category codes present rather than quantile-interpolated, see `_feature_grid`.

    Returns list of {"feature_a", "feature_b", "h_statistic"} dicts, sorted descending.
    """
    cat_col_set = set(cat_cols or [])
    importance = model.feature_importance(importance_type="gain")
    n_top = min(top_n_features, len(feature_names))
    top_idx = np.argsort(importance)[::-1][:n_top]
    top_features = [feature_names[i] for i in top_idx]

    X_sample = X.sample(min(n_sample, len(X)), random_state=42).reset_index(drop=True)

    grids = {
        f: _feature_grid(X_sample, f, grid_size, categorical=f in cat_col_set) for f in top_features
    }
    pd1 = {f: _pd_1d(model, X_sample, f, grids[f]) for f in top_features}

    results = []
    for feat_i, feat_j in itertools.combinations(top_features, 2):
        pd2 = _pd_2d(model, X_sample, feat_i, feat_j, grids[feat_i], grids[feat_j])
        h = _h_stat(pd2, pd1[feat_i], pd1[feat_j])
        results.append({"feature_a": feat_i, "feature_b": feat_j, "h_statistic": round(h, 6)})

    return sorted(results, key=lambda x: x["h_statistic"], reverse=True)
