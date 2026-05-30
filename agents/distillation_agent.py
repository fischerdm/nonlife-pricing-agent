"""Phase 3 — Distillation Agent: GBM insights → GLM interaction extraction."""

from __future__ import annotations

from core.llm_client import LLMClient
from core.schemas import DistillationResponse

DISTILLATION_PROMPT = """
You are a senior actuary reviewing the output of a Gradient Boosting Machine
for a {lob} pricing model.

The model identified the following pairwise interactions (ranked by H-statistic):
{interactions_json}

Your tasks:
1. Select the interactions that are actuarially justifiable and regulatorily defensible.
2. For each selected interaction, specify whether it should enter the GLM as:
   - A product term ("product")
   - A ratio term ("ratio")
   - A binned interaction ("binned")
3. Flag any interaction that appears spurious or data-driven only (set "spurious": true).

Respond ONLY with valid JSON:
{{
  "approved_interactions": [...],
  "rejected_interactions": [...],
  "glm_formula_terms": ["feature_a:feature_b", ...]
}}
"""


class DistillationAgent:
    """Extract interpretable GLM structure from a trained GBM."""

    def __init__(self, llm_client: LLMClient, lob: str = "motor"):
        self.llm = llm_client
        self.lob = lob

    def distill(self, h_stat_interactions: list[dict]) -> DistillationResponse:
        raise NotImplementedError(
            "Phase 3 — implement distillation pipeline in distillation_agent.py"
        )
