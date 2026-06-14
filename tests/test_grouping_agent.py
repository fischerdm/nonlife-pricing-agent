from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.grouping_agent import OTHER_RESIDUAL, GroupingAgent
from core.schemas import CategoryCluster, GroupingResponse


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.call.return_value = GroupingResponse(
        clusters=[
            CategoryCluster(
                cluster_name="HIGH_RISK",
                elements=["delivery_driver", "taxi_driver"],
                rationale="Professional drivers with high exposure",
            ),
            CategoryCluster(
                cluster_name="LOW_RISK",
                elements=["office_worker", "teacher"],
                rationale="Predictable commute, low mileage",
            ),
        ]
    )
    return mock


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    occupations = (
        ["delivery_driver"] * 600
        + ["taxi_driver"] * 700
        + ["office_worker"] * 1200
        + ["teacher"] * 900
        + ["rare_job"] * 50   # below min_exposure threshold
    )
    return pd.DataFrame({
        "occupation": occupations,
        "exposure_years": rng.uniform(0.5, 1.0, len(occupations)),
        "claim_count": rng.poisson(0.1, len(occupations)),
    })


def test_grouping_agent_returns_response(mock_llm, sample_df):
    agent = GroupingAgent(mock_llm, min_exposure=500)
    response = agent.group(sample_df, "occupation", "exposure_years", n_clusters=2)

    assert isinstance(response, GroupingResponse)
    mapping = agent.build_mapping(response)
    assert mapping["delivery_driver"] == "HIGH_RISK"
    assert mapping["office_worker"] == "LOW_RISK"


def test_grouping_agent_apply_fills_residual(mock_llm, sample_df):
    agent = GroupingAgent(mock_llm, min_exposure=500)
    response = agent.group(sample_df, "occupation", "exposure_years", n_clusters=2)
    mapping = agent.build_mapping(response)
    grouped = agent.apply_grouping(sample_df, "occupation", mapping)

    assert grouped[sample_df["occupation"] == "rare_job"].eq(OTHER_RESIDUAL).all()


def test_grouping_agent_calls_llm_once(mock_llm, sample_df):
    agent = GroupingAgent(mock_llm, min_exposure=500)
    agent.group(sample_df, "occupation", "exposure_years", n_clusters=2)

    mock_llm.call.assert_called_once()


def test_grouping_agent_refine_calls_llm(mock_llm, sample_df):
    agent = GroupingAgent(mock_llm, min_exposure=500)
    initial = agent.group(sample_df, "occupation", "exposure_years", n_clusters=2)
    agent.refine(
        df=sample_df,
        col_name="occupation",
        exposure_col="exposure_years",
        n_clusters=2,
        previous_response=initial,
        actuary_remarks={"HIGH_RISK": "split delivery from taxi, different risk profile"},
    )

    assert mock_llm.call.call_count == 2
