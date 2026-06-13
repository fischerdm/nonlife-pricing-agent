import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.feature_builder import build_interaction_feature
from core.schemas import InteractionHypothesis, ValidationResult


class Validator:
    def __init__(self, config: dict):
        objective = config.get("objective", "poisson")
        self.lgb_params = {
            "objective": objective,
            "metric": objective,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        self.n_rounds = config.get("n_rounds", 300)
        self.early_stopping_rounds = config.get("early_stopping_rounds", 50)

    def validate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        exposure_col: str,
        hypothesis: InteractionHypothesis,
        new_feature_col: str,
    ) -> ValidationResult:
        df = df.copy()
        df[new_feature_col] = build_interaction_feature(df, hypothesis)

        X_full = df[feature_cols + [new_feature_col]]
        y = df[target_col]
        w = df[exposure_col]

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_full, y, w, test_size=0.2, random_state=42
        )

        base_dev = self._train_and_score(
            X_train[feature_cols], y_train, w_train,
            X_val[feature_cols], y_val, w_val,
        )

        cand_dev, gain_rank = self._train_and_score(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            return_importance=True,
            target_feature=new_feature_col,
        )

        delta_pct = (cand_dev - base_dev) / abs(base_dev) * 100

        return ValidationResult(
            hypothesis=hypothesis,
            deviance_delta_pct=delta_pct,
            gain_rank=gain_rank,
            baseline_deviance=base_dev,
            new_deviance=cand_dev,
        )

    def _train_and_score(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        w_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        w_val: pd.Series,
        return_importance: bool = False,
        target_feature: str | None = None,
    ) -> float | tuple[float, int]:
        evals_result: dict = {}
        dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train.values)
        dval = lgb.Dataset(X_val, label=y_val, weight=w_val.values, reference=dtrain)

        model = lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=self.n_rounds,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
                lgb.record_evaluation(evals_result),
            ],
        )

        metric_name = self.lgb_params["metric"]
        score = model.best_score["val"][metric_name]

        if return_importance:
            feature_names = model.feature_name()
            importance = model.feature_importance(importance_type="gain")
            feat_importance = dict(zip(feature_names, importance))
            sorted_feats = sorted(feat_importance, key=feat_importance.get, reverse=True)
            rank = (sorted_feats.index(target_feature) + 1) if target_feature in sorted_feats else -1
            return score, rank

        return score
