from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from core.feature_pipeline import generate_draft, refine_draft
from core.schemas import (
    CategoricalFeatureConfig,
    CategoryCluster,
    FeatureProposal,
    GroupingResponse,
    NumericFeatureConfig,
)

DATA_CFG = {
    "target_col": "premium",
    "exposure_col": "exposure_years",
    "objective": "gamma",
}
GROUPING_CFG = {"min_exposure": 500}


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    occupations = ["delivery_driver"] * 600 + ["office_worker"] * 1200
    return pd.DataFrame({
        "occupation": occupations,
        "vehicle_age": rng.uniform(0, 20, len(occupations)),
        "exposure_years": rng.uniform(0.5, 1.0, len(occupations)),
        "premium": rng.uniform(100, 1000, len(occupations)),
    })


def _grouping_response(cluster_name="HIGH_RISK") -> GroupingResponse:
    return GroupingResponse(clusters=[
        CategoryCluster(cluster_name=cluster_name, elements=["delivery_driver"], rationale="r"),
        CategoryCluster(cluster_name="LOW_RISK", elements=["office_worker"], rationale="r"),
    ])


@pytest.fixture
def mock_llm():
    return MagicMock()


def test_generate_draft_groups_every_categorical(mock_llm, sample_df):
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d")],
        categorical=[CategoricalFeatureConfig(name="occupation", description="d", n_clusters=2)],
    )
    mock_llm.call.return_value = _grouping_response()

    proposal = generate_draft(mock_llm, sample_df, DATA_CFG, GROUPING_CFG)

    mock_llm.call_template.assert_called_once()
    assert mock_llm.call.call_count == 1
    assert proposal.categorical[0].grouping == {
        "HIGH_RISK": ["delivery_driver"],
        "LOW_RISK": ["office_worker"],
    }


def test_refine_draft_carries_forward_untouched_grouping(mock_llm, sample_df):
    previous = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[CategoricalFeatureConfig(
            name="occupation", description="d", n_clusters=2, approved=True,
            grouping={"HIGH_RISK": ["delivery_driver"], "LOW_RISK": ["office_worker"]},
        )],
    )
    # Simulate the LLM not echoing back the grouping field on refine.
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[CategoricalFeatureConfig(
            name="occupation", description="d", n_clusters=2, approved=True,
        )],
    )

    updated = refine_draft(mock_llm, sample_df, DATA_CFG, GROUPING_CFG, previous, remarks={})

    mock_llm.call.assert_not_called()
    assert updated.categorical[0].grouping == {
        "HIGH_RISK": ["delivery_driver"],
        "LOW_RISK": ["office_worker"],
    }


def test_refine_draft_groups_newly_promoted_categorical(mock_llm, sample_df):
    previous = FeatureProposal(
        numeric=[],
        categorical=[],
        excluded=["occupation"],
        exclusion_rationale={"occupation": "not useful"},
    )
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[],
        categorical=[CategoricalFeatureConfig(name="occupation", description="d", n_clusters=2)],
    )
    mock_llm.call.return_value = _grouping_response()

    updated = refine_draft(
        mock_llm, sample_df, DATA_CFG, GROUPING_CFG, previous,
        remarks={"occupation": "please include this"},
    )

    assert mock_llm.call.call_count == 1
    assert updated.categorical[0].grouping == {
        "HIGH_RISK": ["delivery_driver"],
        "LOW_RISK": ["office_worker"],
    }


def test_refine_draft_refines_remarked_categorical_with_existing_grouping(mock_llm, sample_df):
    previous = FeatureProposal(
        numeric=[],
        categorical=[CategoricalFeatureConfig(
            name="occupation", description="d", n_clusters=2, approved=True,
            grouping={"OLD_HIGH": ["delivery_driver"], "OLD_LOW": ["office_worker"]},
        )],
    )
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[],
        categorical=[CategoricalFeatureConfig(
            name="occupation", description="d", n_clusters=2, approved=True,
        )],
    )
    mock_llm.call.return_value = _grouping_response("NEW_HIGH")

    updated = refine_draft(
        mock_llm, sample_df, DATA_CFG, GROUPING_CFG, previous,
        remarks={"occupation": "split further"},
    )

    assert mock_llm.call.call_count == 1
    assert updated.categorical[0].grouping == {
        "NEW_HIGH": ["delivery_driver"],
        "LOW_RISK": ["office_worker"],
    }
