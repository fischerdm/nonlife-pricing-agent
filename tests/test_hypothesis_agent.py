from unittest.mock import MagicMock

import pytest

from agents.hypothesis_agent import HypothesisAgent
from core.schemas import FeatureMetadata, HypothesisResponse, InteractionHypothesis


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.call.return_value = HypothesisResponse(
        hypotheses=[
            InteractionHypothesis(
                feature_a="driver_age",
                feature_b="vehicle_power_kw",
                operation="multiply",
                new_feature_name="driver_age_x_kw",
                rationale="Young drivers in powerful vehicles have higher risk",
            )
        ]
    )
    return mock


@pytest.fixture
def sample_features():
    return [
        FeatureMetadata(name="driver_age", dtype="numeric", description="Age of driver"),
        FeatureMetadata(name="vehicle_power_kw", dtype="numeric", description="Power in kW"),
    ]


def test_hypothesis_agent_generates_response(mock_llm, sample_features):
    agent = HypothesisAgent(mock_llm)
    result = agent.generate(sample_features, target_col="claim_count", n_hypotheses=1)

    assert isinstance(result, HypothesisResponse)
    assert len(result.hypotheses) == 1


def test_hypothesis_agent_calls_llm_once(mock_llm, sample_features):
    agent = HypothesisAgent(mock_llm)
    agent.generate(sample_features, target_col="claim_count", n_hypotheses=3)

    mock_llm.call.assert_called_once()


def test_hypothesis_agent_passes_n_to_prompt(mock_llm, sample_features):
    agent = HypothesisAgent(mock_llm)
    agent.generate(sample_features, target_col="claim_count", n_hypotheses=7)

    call_args = mock_llm.call.call_args
    prompt = call_args[0][0]
    assert "7" in prompt


def test_hypothesis_agent_returns_correct_feature_names(mock_llm, sample_features):
    agent = HypothesisAgent(mock_llm)
    result = agent.generate(sample_features, target_col="claim_count")

    h = result.hypotheses[0]
    assert h.feature_a == "driver_age"
    assert h.feature_b == "vehicle_power_kw"
    assert h.operation == "multiply"
