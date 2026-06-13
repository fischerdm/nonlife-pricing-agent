import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from agents.feature_selection_agent import FeatureSelectionAgent
from core.llm_client import LLMClient
from core.schemas import CategoricalFeatureConfig, FeatureProposal, NumericFeatureConfig
from dashboard.approval_gate import run_feature_gate


class Orchestrator:
    def __init__(self, config_path: str = "config/project_config.yaml"):
        load_dotenv()
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        llm_cfg = self.config["llm"]
        self.llm = LLMClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
        )

    def run(self) -> None:
        data_cfg = self.config["data"]
        df = pd.read_csv(
            data_cfg["path"],
            sep=data_cfg.get("sep", ","),
            low_memory=False,
        )
        if data_cfg.get("filter_zeros"):
            target_col = data_cfg["target_col"]
            df = df[df[target_col] > 0].copy()

        # ── Stage 1: feature selection ─────────────────────────────────────────
        proposal = self._load_or_run_feature_selection(df)

        # ── Stage 2: grouping agent for approved categoricals ──────────────────
        # TODO: implement grouping stage (calls GroupingAgent per categorical)

        # ── Stage 3: GBM training ──────────────────────────────────────────────
        # TODO: implement GBMAgent (Phase 3)

        # ── Stage 4: GLM distillation ──────────────────────────────────────────
        # TODO: implement DistillationAgent + run_glm_gate (Phase 3)

        raise NotImplementedError(
            "Pipeline past feature selection not yet implemented — Phase 3."
        )

    # ── Feature selection checkpoint ───────────────────────────────────────────

    def _load_or_run_feature_selection(self, df: pd.DataFrame) -> FeatureProposal:
        """Return saved proposal if all features are approved; otherwise run the gate."""
        if self._features_fully_approved():
            print("Feature selection: loading from checkpoint.")
            return self._proposal_from_config()

        print("Feature selection: running agent + actuary gate.")
        data_cfg = self.config["data"]
        agent = FeatureSelectionAgent(self.llm)
        proposal = agent.propose(
            df=df,
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            objective=data_cfg["objective"],
        )
        proposal = run_feature_gate(
            proposal=proposal,
            agent=agent,
            objective=data_cfg["objective"],
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
        )
        self._save_proposal_to_config(proposal)
        return proposal

    def _features_fully_approved(self) -> bool:
        """True only if the config has a feature list with every entry approved=true."""
        features = self.config.get("features")
        if not features:
            return False
        all_feats = features.get("numeric", []) + features.get("categorical", [])
        return bool(all_feats) and all(f.get("approved") is True for f in all_feats)

    def _proposal_from_config(self) -> FeatureProposal:
        features = self.config["features"]
        numeric = [NumericFeatureConfig(**f) for f in features.get("numeric", [])]
        categorical = [CategoricalFeatureConfig(**f) for f in features.get("categorical", [])]
        return FeatureProposal(numeric=numeric, categorical=categorical)

    def _save_proposal_to_config(self, proposal: FeatureProposal) -> None:
        """Write approved features back to project_config.yaml as the checkpoint."""
        self.config["features"] = {
            "numeric": [
                _feature_to_dict(f) for f in proposal.numeric if f.approved is True
            ],
            "categorical": [
                _feature_to_dict(f) for f in proposal.categorical if f.approved is True
            ],
        }
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        print(f"Feature checkpoint saved to {self.config_path}")


def _feature_to_dict(feat: NumericFeatureConfig | CategoricalFeatureConfig) -> dict:
    return {k: v for k, v in feat.model_dump().items() if v is not None}
