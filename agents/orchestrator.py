import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from agents.distillation_agent import DistillationAgent
from agents.feature_selection_agent import FeatureSelectionAgent
from agents.gbm_agent import GBMAgent
from agents.grouping_agent import OTHER_RESIDUAL, GroupingAgent
from core.data_loader import load_dataset
from core.feature_pipeline import proposal_from_config, save_feature_checkpoint
from core.llm_client import LLMClient
from core.schemas import (
    FeatureProposal,
    GLMProposal,
    GLMTerm,
)
from core.session_logger import SessionLogger
from dashboard.approval_gate import run_feature_gate, run_glm_coef_gate, run_glm_gate, run_grouping_gate
from tools.glm_tools import build_formula, coef_summary, fit_glm, print_glm_summary, print_rating_factors


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
        self.logger = SessionLogger()

    def run(self) -> None:
        data_cfg = self.config["data"]
        self.logger.log(
            "session_start",
            config={
                "target_col": data_cfg["target_col"],
                "exposure_col": data_cfg["exposure_col"],
                "objective": data_cfg["objective"],
                "model": self.config["llm"]["model"],
            },
        )

        try:
            df = load_dataset(data_cfg)

            # ── Stage 1: feature selection ─────────────────────────────────────
            proposal = self._load_or_run_feature_selection(df)

            # ── Stage 2: grouping agent for approved categoricals ──────────────
            proposal = self._load_or_run_grouping(df, proposal)
            df = self._apply_groupings(df, proposal)

            # ── Stage 3: GBM training + H-statistics ──────────────────────────
            interactions = self._load_or_run_gbm(df, proposal)

            # ── Stage 4: GLM distillation ──────────────────────────────────────
            glm_proposal = self._load_or_run_distillation(df, proposal, interactions)

            # ── Stage 5: GLM fitting ───────────────────────────────────────────
            self._fit_and_report_glm(df, glm_proposal)

            self.logger.log("session_complete")
        finally:
            self.logger.close()

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
            df=df,
            objective=data_cfg["objective"],
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            logger=self.logger,
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
        return proposal_from_config(self.config)

    def _save_proposal_to_config(self, proposal: FeatureProposal) -> None:
        """Write approved features back to project_config.yaml as the checkpoint."""
        save_feature_checkpoint(self.config_path, self.config, proposal)

    # ── Grouping checkpoint ────────────────────────────────────────────────────

    def _load_or_run_grouping(self, df: pd.DataFrame, proposal: FeatureProposal) -> FeatureProposal:
        """Run GroupingAgent for each approved categorical that lacks a grouping checkpoint."""
        cats_needing_grouping = [
            f for f in proposal.categorical
            if f.approved and f.grouping is None
        ]
        if not cats_needing_grouping:
            print("Grouping: all categoricals have checkpointed groupings.")
            return proposal

        print(f"Grouping: running agent for {len(cats_needing_grouping)} categorical(s).")
        data_cfg = self.config["data"]
        grouping_cfg = self.config.get("grouping", {})
        agent = GroupingAgent(self.llm, min_exposure=grouping_cfg.get("min_exposure", 500))

        for cat_feat in cats_needing_grouping:
            response = agent.group(
                df=df,
                col_name=cat_feat.name,
                exposure_col=data_cfg["exposure_col"],
                n_clusters=cat_feat.n_clusters,
                claim_freq_col=data_cfg.get("claim_freq_col"),
            )
            response = run_grouping_gate(
                col_name=cat_feat.name,
                response=response,
                agent=agent,
                df=df,
                exposure_col=data_cfg["exposure_col"],
                n_clusters=cat_feat.n_clusters,
                claim_freq_col=data_cfg.get("claim_freq_col"),
                logger=self.logger,
            )
            cat_feat.grouping = {c.cluster_name: c.elements for c in response.clusters}

        self._save_proposal_to_config(proposal)
        return proposal

    def _apply_groupings(self, df: pd.DataFrame, proposal: FeatureProposal) -> pd.DataFrame:
        """Replace each approved categorical column with its grouped values."""
        df = df.copy()
        for cat_feat in proposal.categorical:
            if cat_feat.approved and cat_feat.grouping:
                mapping = {
                    val: cluster_name
                    for cluster_name, values in cat_feat.grouping.items()
                    for val in values
                }
                df[cat_feat.name] = df[cat_feat.name].map(mapping).fillna(OTHER_RESIDUAL)
        return df

    # ── GBM checkpoint ─────────────────────────────────────────────────────────

    def _load_or_run_gbm(self, df: pd.DataFrame, proposal: FeatureProposal) -> list[dict]:
        """Return saved H-statistics if checkpoint exists; otherwise train and compute."""
        if self.config.get("gbm_output", {}).get("interactions"):
            print("GBM: loading from checkpoint.")
            return self.config["gbm_output"]["interactions"]

        print("GBM: training model and computing H-statistics.")
        data_cfg = self.config["data"]
        feature_cols = (
            [f.name for f in proposal.numeric if f.approved]
            + [f.name for f in proposal.categorical if f.approved]
        )
        agent = GBMAgent(self.config.get("gbm", {}))
        interactions = agent.run(
            df=df,
            feature_cols=feature_cols,
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
        )
        self.logger.log(
            "gbm_complete",
            stage="gbm",
            feature_importances=agent.feature_importances,
            interactions=interactions,
        )
        self._save_gbm_to_config(interactions)
        return interactions

    def _save_gbm_to_config(self, interactions: list[dict]) -> None:
        self.config["gbm_output"] = {"interactions": interactions}
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        print(f"GBM checkpoint saved to {self.config_path}")

    # ── GLM distillation checkpoint ────────────────────────────────────────────

    def _load_or_run_distillation(
        self,
        df: pd.DataFrame,
        proposal: FeatureProposal,
        interactions: list[dict],
    ) -> GLMProposal:
        if self._glm_fully_approved():
            print("GLM distillation: loading from checkpoint.")
            return self._proposal_from_glm_config()

        print("GLM distillation: running agent + actuary gate.")
        data_cfg = self.config["data"]
        approved_features = (
            [f.name for f in proposal.numeric if f.approved]
            + [f.name for f in proposal.categorical if f.approved]
        )
        agent = DistillationAgent(self.llm, lob=data_cfg.get("lob", "motor"))
        glm_proposal = agent.propose(
            h_stat_interactions=interactions,
            approved_features=approved_features,
            objective=data_cfg["objective"],
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
        )
        glm_proposal = run_glm_gate(
            proposal=glm_proposal,
            agent=agent,
            objective=data_cfg["objective"],
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            logger=self.logger,
        )
        self._save_glm_to_config(glm_proposal, data_cfg)
        return glm_proposal

    def _glm_fully_approved(self) -> bool:
        glm_cfg_path = self.config_path.parent / "glm_config.yaml"
        if not glm_cfg_path.exists():
            return False
        with open(glm_cfg_path) as f:
            glm_cfg = yaml.safe_load(f)
        terms = glm_cfg.get("glm", {}).get("terms", [])
        return bool(terms) and all(t.get("approved") is True for t in terms)

    def _proposal_from_glm_config(self) -> GLMProposal:
        glm_cfg_path = self.config_path.parent / "glm_config.yaml"
        with open(glm_cfg_path) as f:
            glm_cfg = yaml.safe_load(f)
        terms = [GLMTerm(**t) for t in glm_cfg["glm"]["terms"]]
        formula = glm_cfg["glm"].get("formula")
        return GLMProposal(terms=terms, formula=formula)

    def _save_glm_to_config(self, proposal: GLMProposal, data_cfg: dict) -> None:
        glm_cfg_path = self.config_path.parent / "glm_config.yaml"
        approved_terms = [t for t in proposal.terms if t.approved is True]
        formula = build_formula(data_cfg["target_col"], approved_terms)
        proposal.formula = formula

        glm_cfg: dict = {}
        if glm_cfg_path.exists():
            with open(glm_cfg_path) as f:
                glm_cfg = yaml.safe_load(f) or {}

        glm_cfg.setdefault("glm", {})
        glm_cfg["glm"]["objective"] = data_cfg["objective"]
        glm_cfg["glm"]["link"] = "log"
        glm_cfg["glm"]["terms"] = [
            {k: v for k, v in t.model_dump().items() if v is not None}
            for t in proposal.terms
            if t.approved is True
        ]
        glm_cfg["glm"]["formula"] = formula

        with open(glm_cfg_path, "w") as f:
            yaml.dump(glm_cfg, f, allow_unicode=True, sort_keys=False)
        print(f"GLM checkpoint saved to {glm_cfg_path}")

    # ── GLM fitting ────────────────────────────────────────────────────────────

    def _fit_and_report_glm(self, df: pd.DataFrame, glm_proposal: GLMProposal) -> None:
        data_cfg = self.config["data"]
        active_terms = [t for t in glm_proposal.terms if t.approved is True]
        if not active_terms:
            print("No approved GLM terms — skipping GLM fit.")
            return

        formula = glm_proposal.formula or build_formula(data_cfg["target_col"], active_terms)
        print(f"\nFitting GLM: {formula}\n")
        result = fit_glm(
            df=df,
            formula=formula,
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            family=data_cfg["objective"],
        )
        print_glm_summary(result)

        summary_df = coef_summary(result)
        self.logger.log(
            "glm_fit",
            stage="glm",
            formula=formula,
            aic=float(result.aic),
            deviance_explained=float(1 - result.deviance / result.null_deviance),
            coefficients=summary_df.to_dict(orient="records"),
        )

        # ── Coefficient review gate: reject terms, refit until satisfied ──────
        result, active_terms = run_glm_coef_gate(
            result=result,
            active_terms=active_terms,
            df=df,
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            family=data_cfg["objective"],
            logger=self.logger,
        )

        # ── Rating factors table ───────────────────────────────────────────────
        if active_terms:
            print_rating_factors(result)
            final_summary = coef_summary(result)
            self.logger.log(
                "rating_factors",
                stage="glm",
                aic=float(result.aic),
                deviance_explained=float(1 - result.deviance / result.null_deviance),
                rating_factors=final_summary.to_dict(orient="records"),
            )
