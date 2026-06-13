"""Phase 3 — GBM Agent: train model, compute SHAP values and H-statistics."""

from __future__ import annotations

import pandas as pd


class GBMAgent:
    """Train a LightGBM model and expose SHAP-based interaction diagnostics."""

    def __init__(self, config: dict):
        self.config = config

    def train(self, df: pd.DataFrame, feature_cols: list[str], target_col: str, exposure_col: str):
        raise NotImplementedError("Phase 3 — implement GBM training in gbm_agent.py")

    def compute_shap_interactions(self):
        raise NotImplementedError("Phase 3 — implement SHAP interaction values")

    def compute_h_statistics(self) -> list[dict]:
        raise NotImplementedError("Phase 3 — implement H-statistic ranking")
