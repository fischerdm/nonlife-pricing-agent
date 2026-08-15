import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from agents.distillation_agent import DistillationAgent
from agents.feature_selection_agent import FeatureSelectionAgent
from agents.grouping_agent import GroupingAgent
from core.data_loader import load_dataset
from core.distillation_pipeline import generate_glm_draft
from core.feature_pipeline import apply_groupings, proposal_from_config, save_feature_checkpoint
from core.gbm_pipeline import save_gbm_checkpoint, train_gbm
from core.glm_pipeline import proposal_from_glm_config, save_glm_checkpoint
from core.llm_client import LLMClient
from core.schemas import FeatureProposal, GLMProposal
from core.seed_config import (
    DISTILLATION_SEED_FILENAME,
    FEATURE_SEED_FILENAME,
    load_distillation_seed,
    load_feature_seed,
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
            df = apply_groupings(df, proposal)

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
        seed = load_feature_seed(self.config_path.parent / FEATURE_SEED_FILENAME)
        agent = FeatureSelectionAgent(self.llm)
        proposal = agent.propose(
            df=df,
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            objective=data_cfg["objective"],
            seed=seed,
        )
        proposal = run_feature_gate(
            proposal=proposal,
            agent=agent,
            df=df,
            objective=data_cfg["objective"],
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            logger=self.logger,
            seed=seed,
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

    # ── GBM checkpoint ─────────────────────────────────────────────────────────

    def _load_or_run_gbm(self, df: pd.DataFrame, proposal: FeatureProposal) -> list[dict]:
        """Return saved H-statistics if checkpoint exists; otherwise train and compute.

        `df` must already have approved-categorical groupings applied.
        """
        if self.config.get("gbm_output", {}).get("interactions"):
            print("GBM: loading from checkpoint.")
            return self.config["gbm_output"]["interactions"]

        print("GBM: training model and computing H-statistics.")
        data_cfg = self.config["data"]
        agent, interactions = train_gbm(df, proposal, data_cfg, self.config.get("gbm", {}))
        self.logger.log(
            "gbm_complete",
            stage="gbm",
            feature_importances=agent.feature_importances,
            interactions=interactions,
        )
        save_gbm_checkpoint(self.config_path, self.config, agent, interactions)
        return interactions

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
        seed = load_distillation_seed(self.config_path.parent / DISTILLATION_SEED_FILENAME)
        approved_features = (
            [f.name for f in proposal.numeric if f.approved]
            + [f.name for f in proposal.categorical if f.approved]
        )
        glm_proposal = generate_glm_draft(
            self.llm, interactions, approved_features, data_cfg, seed=seed,
        )
        agent = DistillationAgent(self.llm, lob=data_cfg.get("lob", "motor"))
        glm_proposal = run_glm_gate(
            proposal=glm_proposal,
            agent=agent,
            objective=data_cfg["objective"],
            target_col=data_cfg["target_col"],
            exposure_col=data_cfg["exposure_col"],
            logger=self.logger,
            seed=seed,
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
        proposal = proposal_from_glm_config(glm_cfg_path)
        assert proposal is not None, "_glm_fully_approved() already confirmed terms exist"
        return proposal

    def _save_glm_to_config(self, proposal: GLMProposal, data_cfg: dict) -> None:
        glm_cfg_path = self.config_path.parent / "glm_config.yaml"
        save_glm_checkpoint(glm_cfg_path, data_cfg, proposal)

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
