"""Phase 3 — Distillation Agent: ranked H-statistics → GLM term proposal."""

from __future__ import annotations

import json

from core.llm_client import LLMClient
from core.schemas import GLMProposal


class DistillationAgent:
    """Propose GLM main effects and pairwise interaction terms from GBM H-statistics."""

    def __init__(self, llm_client: LLMClient, lob: str = "motor"):
        self.llm = llm_client
        self.lob = lob

    def propose(
        self,
        h_stat_interactions: list[dict],
        approved_features: list[str],
        objective: str,
        target_col: str,
        exposure_col: str,
    ) -> GLMProposal:
        """Initial proposal: main effects for all approved features + selected interactions."""
        return self.llm.call_template(
            agent_name="distillation",
            section="proposal",
            response_model=GLMProposal,
            objective=objective,
            lob=self.lob,
            target_col=target_col,
            exposure_col=exposure_col,
            features_json=json.dumps(approved_features, indent=2),
            interactions_json=json.dumps(h_stat_interactions, indent=2),
        )

    def refine(
        self,
        previous_proposal: GLMProposal,
        actuary_remarks: dict[str, str],
        objective: str,
        target_col: str,
        exposure_col: str,
    ) -> GLMProposal:
        """Revise the proposal incorporating actuary remarks."""
        return self.llm.call_template(
            agent_name="distillation",
            section="refinement",
            response_model=GLMProposal,
            previous_proposal_json=json.dumps(previous_proposal.model_dump(), indent=2),
            actuary_remarks_json=json.dumps(actuary_remarks, indent=2),
        )
