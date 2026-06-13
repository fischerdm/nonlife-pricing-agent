import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from agents.grouping_agent import GroupingAgent
from agents.hypothesis_agent import HypothesisAgent
from core.llm_client import LLMClient
from core.schemas import FeatureMetadata, ValidationResult
from core.validator import Validator
from dashboard.approval_gate import run_approval_gate


class Orchestrator:
    def __init__(self, config_path: str = "config/project_config.yaml"):
        load_dotenv()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        llm_cfg = self.config["llm"]
        self.llm = LLMClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
        )
        val_cfg = self.config["validation"]
        self.validator = Validator(val_cfg)
        self.min_improvement = val_cfg["min_deviance_improvement_pct"]

    def run(self) -> list[ValidationResult]:
        df = pd.read_parquet(self.config["data"]["path"])
        target_col = self.config["data"]["target_col"]
        exposure_col = self.config["data"]["exposure_col"]

        numeric_features = [
            FeatureMetadata(name=f["name"], dtype="numeric", description=f["description"])
            for f in self.config["features"].get("numeric", [])
        ]
        categorical_features = [
            FeatureMetadata(name=f["name"], dtype="categorical", description=f["description"])
            for f in self.config["features"].get("categorical", [])
        ]
        all_features = numeric_features + categorical_features

        # Phase 2: apply categorical groupings before validation
        df = self._apply_groupings(df, exposure_col)

        feature_cols = [f.name for f in all_features]

        # Phase 1: hypothesis generation + validation
        agent = HypothesisAgent(self.llm)
        response = agent.generate(
            features=all_features,
            target_col=target_col,
            n_hypotheses=self.config["llm"]["max_hypotheses"],
        )

        results: list[ValidationResult] = []
        for hypothesis in response.hypotheses:
            if hypothesis.feature_a not in df.columns or hypothesis.feature_b not in df.columns:
                continue

            result = self.validator.validate(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                exposure_col=exposure_col,
                hypothesis=hypothesis,
                new_feature_col=hypothesis.new_feature_name,
            )

            if result.deviance_delta_pct > -self.min_improvement:
                result.approved = False

            results.append(result)

        candidates = [r for r in results if r.approved is None]
        approved = run_approval_gate(candidates) if candidates else []

        self._log_summary(results)
        return approved

    def _apply_groupings(self, df: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
        df = df.copy()
        cat_configs = self.config["features"].get("categorical", [])
        if not cat_configs:
            return df

        grouping_agent = GroupingAgent(
            llm_client=self.llm,
            min_exposure=self.config["validation"]["min_exposure_per_cluster"],
        )
        for feat_cfg in cat_configs:
            col = feat_cfg["name"]
            if col not in df.columns:
                continue
            mapping = grouping_agent.group(
                df=df,
                col_name=col,
                exposure_col=exposure_col,
                n_clusters=feat_cfg["n_clusters"],
            )
            df[col] = grouping_agent.apply_grouping(df, col, mapping)

        return df

    def _log_summary(self, results: list[ValidationResult]) -> None:
        approved = sum(1 for r in results if r.approved is True)
        rejected = sum(1 for r in results if r.approved is False)
        skipped = sum(1 for r in results if r.approved is None)
        print(
            f"\nSummary: {len(results)} hypotheses tested — "
            f"{approved} approved, {rejected} rejected, {skipped} skipped"
        )
