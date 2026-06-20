"""Phase 3 — GBM Agent: train model, compute H-statistics."""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tools.shap_tools import compute_h_statistics

logger = logging.getLogger(__name__)

_LGB_BASE_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_jobs": -1,
}


class GBMAgent:
    """Train a LightGBM model on log-rate target and rank pairwise interactions by H-statistic."""

    def __init__(self, config: dict):
        self.config = config
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] = []
        self._cat_cols: list[str] = []

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        exposure_col: str,
    ) -> list[dict]:
        """Train LightGBM, save model, and return ranked pairwise H-statistics."""
        self._feature_names = list(feature_cols)
        X, y = self._prepare_data(df, feature_cols, target_col, exposure_col)
        self._model = self._train(X, y)

        if model_path := self.config.get("model_path"):
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self._model.save_model(model_path)
            logger.info("GBM model saved to %s", model_path)

        return compute_h_statistics(
            model=self._model,
            X=X,
            feature_names=self._feature_names,
            top_n_features=self.config.get("top_n_features", 15),
            n_sample=self.config.get("h_stat_n_sample", 500),
            grid_size=self.config.get("h_stat_grid_size", 20),
        )

    @property
    def feature_importances(self) -> list[dict]:
        """Return gain-importance per feature as a sorted list, normalised to sum to 1."""
        if self._model is None:
            return []
        raw = self._model.feature_importance(importance_type="gain").astype(float)
        total = raw.sum() or 1.0
        return sorted(
            [
                {"feature": name, "importance": float(val / total)}
                for name, val in zip(self._feature_names, raw)
            ],
            key=lambda x: -x["importance"],
        )

    def _prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        exposure_col: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Label-encode categoricals; target = log(premium / exposure)."""
        X = df[feature_cols].copy()
        self._cat_cols = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype("category").cat.codes
                self._cat_cols.append(col)

        exposure = df[exposure_col].clip(lower=1e-6)
        y = np.log(df[target_col] / exposure)
        return X, y.to_numpy()

    def _train(self, X: pd.DataFrame, y: np.ndarray) -> lgb.Booster:
        params = {
            **_LGB_BASE_PARAMS,
            "num_leaves": self.config.get("num_leaves", 63),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "feature_fraction": self.config.get("feature_fraction", 0.8),
            "bagging_fraction": self.config.get("bagging_fraction", 0.8),
            "bagging_freq": 5,
        }

        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        split = int(len(X) * (1 - self.config.get("val_fraction", 0.2)))
        train_idx, val_idx = idx[:split], idx[split:]

        cat_feature = self._cat_cols if self._cat_cols else "auto"
        dtrain = lgb.Dataset(
            X.iloc[train_idx], label=y[train_idx], categorical_feature=cat_feature
        )
        dval = lgb.Dataset(
            X.iloc[val_idx], label=y[val_idx], reference=dtrain, categorical_feature=cat_feature
        )

        callbacks = [
            lgb.early_stopping(self.config.get("early_stopping_rounds", 50), verbose=False),
            lgb.log_evaluation(period=50),
        ]
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.config.get("n_rounds", 300),
            valid_sets=[dval],
            callbacks=callbacks,
        )
        logger.info("GBM trained: %d trees", model.num_trees())
        return model
