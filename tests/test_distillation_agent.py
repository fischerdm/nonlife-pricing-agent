from unittest.mock import MagicMock

import pytest

from agents.distillation_agent import DistillationAgent
from core.schemas import CommerciallyExcludedEntry, DistillationSeed, GLMProposal, GLMTerm

APPROVED_FEATURES = ["driver_age", "vehicle_age", "driver_occupation"]
INTERACTIONS = [
    {"feature_a": "driver_age", "feature_b": "vehicle_age", "h_statistic": 0.3},
    {"feature_a": "driver_occupation", "feature_b": "vehicle_age", "h_statistic": 0.2},
]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call_template.return_value = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="vehicle_age", term_type="main", rationale="r"),
        GLMTerm(name="driver_age:vehicle_age", term_type="interaction", rationale="r", h_statistic=0.3),
    ])
    return llm


def _locked_seed() -> DistillationSeed:
    return DistillationSeed(commercially_excluded=[
        CommerciallyExcludedEntry(name="driver_occupation", rationale="not at quote time", temperature=0.0),
    ])


def test_propose_drops_locked_feature_from_prompt(mock_llm):
    agent = DistillationAgent(mock_llm)
    agent.propose(
        h_stat_interactions=INTERACTIONS, approved_features=APPROVED_FEATURES,
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=_locked_seed(),
    )

    kwargs = mock_llm.call_template.call_args.kwargs
    assert "driver_occupation" not in kwargs["features_json"]
    assert "driver_occupation" not in kwargs["interactions_json"]


def test_propose_strips_hallucinated_locked_term_defense_in_depth(mock_llm):
    # Simulate the LLM proposing a term for a feature it was never shown.
    mock_llm.call_template.return_value = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="driver_occupation", term_type="main", rationale="hallucinated"),
        GLMTerm(name="driver_occupation:vehicle_age", term_type="interaction", rationale="r"),
    ])

    agent = DistillationAgent(mock_llm)
    proposal = agent.propose(
        h_stat_interactions=INTERACTIONS, approved_features=APPROVED_FEATURES,
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=_locked_seed(),
    )

    names = {t.name for t in proposal.terms}
    assert names == {"driver_age"}


def test_propose_keeps_flexible_feature_and_adds_seed_context(mock_llm):
    seed = DistillationSeed(commercially_excluded=[
        CommerciallyExcludedEntry(name="driver_occupation", rationale="sensitive", temperature=0.4),
    ])
    agent = DistillationAgent(mock_llm)
    agent.propose(
        h_stat_interactions=INTERACTIONS, approved_features=APPROVED_FEATURES,
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=seed,
    )

    kwargs = mock_llm.call_template.call_args.kwargs
    assert "driver_occupation" in kwargs["features_json"]
    assert "driver_occupation" in kwargs["seed_context_json"]


def _reintroduced_response() -> GLMProposal:
    return GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="driver_occupation", term_type="main", rationale="reintroduced"),
    ])


def test_refine_strips_locked_term_without_a_remark(mock_llm):
    previous = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])
    mock_llm.call_template.return_value = _reintroduced_response()

    agent = DistillationAgent(mock_llm)
    stripped = agent.refine(
        previous_proposal=previous, actuary_remarks={"driver_age": "keep"},
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=_locked_seed(),
    )
    assert {t.name for t in stripped.terms} == {"driver_age"}


def test_refine_keeps_locked_term_when_actuary_explicitly_remarks_it(mock_llm):
    previous = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])
    mock_llm.call_template.return_value = _reintroduced_response()

    agent = DistillationAgent(mock_llm)
    kept = agent.refine(
        previous_proposal=previous, actuary_remarks={"driver_occupation": "actually please add this"},
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=_locked_seed(),
    )
    assert "driver_occupation" in {t.name for t in kept.terms}
