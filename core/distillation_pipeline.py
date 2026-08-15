"""Shared GLM distillation draft-generation logic, used by both the Orchestrator
and the dashboard's GLM Distillation Workbench — mirrors core/feature_pipeline.py.
"""
from core.llm_client import LLMClient
from core.schemas import DistillationSeed, GLMProposal
from agents.distillation_agent import DistillationAgent


def generate_glm_draft(
    llm: LLMClient,
    h_stat_interactions: list[dict],
    approved_features: list[str],
    data_cfg: dict,
    seed: DistillationSeed | None = None,
) -> GLMProposal:
    agent = DistillationAgent(llm, lob=data_cfg.get("lob", "motor"))
    return agent.propose(
        h_stat_interactions=h_stat_interactions,
        approved_features=approved_features,
        objective=data_cfg["objective"],
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
        seed=seed,
    )


def refine_glm_draft(
    llm: LLMClient,
    previous: GLMProposal,
    remarks: dict[str, str],
    data_cfg: dict,
    seed: DistillationSeed | None = None,
) -> GLMProposal:
    agent = DistillationAgent(llm, lob=data_cfg.get("lob", "motor"))
    return agent.refine(
        previous_proposal=previous,
        actuary_remarks=remarks,
        objective=data_cfg["objective"],
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
        seed=seed,
    )
