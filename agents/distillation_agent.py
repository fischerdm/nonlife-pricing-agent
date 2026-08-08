"""Phase 3 — Distillation Agent: ranked H-statistics → GLM term proposal."""

from __future__ import annotations

import json

from core.llm_client import LLMClient
from core.schemas import DistillationSeed, GLMProposal


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
        seed: DistillationSeed | None = None,
    ) -> GLMProposal:
        """Initial proposal: main effects for all approved features + selected interactions."""
        locked = _locked_excluded_names(seed)
        usable_features = [f for f in approved_features if f not in locked]
        usable_interactions = _drop_locked_interactions(h_stat_interactions, locked)
        proposal = self.llm.call_template(
            agent_name="distillation",
            section="proposal",
            response_model=GLMProposal,
            objective=objective,
            lob=self.lob,
            target_col=target_col,
            exposure_col=exposure_col,
            features_json=json.dumps(usable_features, indent=2),
            interactions_json=json.dumps(usable_interactions, indent=2),
            seed_context_json=json.dumps(_seed_context(seed), indent=2),
        )
        return _strip_locked_terms(proposal, locked)

    def refine(
        self,
        previous_proposal: GLMProposal,
        actuary_remarks: dict[str, str],
        objective: str,
        target_col: str,
        exposure_col: str,
        seed: DistillationSeed | None = None,
    ) -> GLMProposal:
        """Revise the proposal incorporating actuary remarks."""
        locked = _locked_excluded_names(seed)
        proposal = self.llm.call_template(
            agent_name="distillation",
            section="refinement",
            response_model=GLMProposal,
            previous_proposal_json=json.dumps(previous_proposal.model_dump(), indent=2),
            actuary_remarks_json=json.dumps(actuary_remarks, indent=2),
            seed_context_json=json.dumps(_seed_context(seed), indent=2),
        )
        return _strip_locked_terms(proposal, locked, actuary_remarks=actuary_remarks)


# ── Seed-lock helpers ──────────────────────────────────────────────────────────

def _locked_excluded_names(seed: DistillationSeed | None) -> set[str]:
    """Names with temperature 0.0 — never sent to the LLM, structurally can't be proposed."""
    if seed is None:
        return set()
    return {e.name for e in seed.commercially_excluded if e.temperature <= 0.0}


def _seed_context(seed: DistillationSeed | None) -> list[dict]:
    """Flexible (temperature > 0.0) commercially-sensitive entries, given as context."""
    if seed is None:
        return []
    return [
        {"name": e.name, "rationale": e.rationale, "temperature": e.temperature}
        for e in seed.commercially_excluded
        if e.temperature > 0.0
    ]


def _drop_locked_interactions(h_stat_interactions: list[dict], locked: set[str]) -> list[dict]:
    if not locked:
        return h_stat_interactions
    return [
        i for i in h_stat_interactions
        if i["feature_a"] not in locked and i["feature_b"] not in locked
    ]


def _term_features(term_name: str) -> set[str]:
    """A main effect's own name, or both sides of an "a:b" interaction."""
    return set(term_name.split(":"))


def _strip_locked_terms(
    proposal: GLMProposal, locked: set[str], actuary_remarks: dict[str, str] | None = None,
) -> GLMProposal:
    """Defense-in-depth: drop any term referencing a locked name the LLM shouldn't have seen.

    A term the actuary explicitly remarked on this round is left alone — that's the
    actuary's own override, not the agent deviating unprompted.
    """
    if not locked:
        return proposal
    remarked = set(actuary_remarks or {})
    proposal.terms = [
        t for t in proposal.terms
        if t.name in remarked or not (_term_features(t.name) & locked)
    ]
    return proposal
