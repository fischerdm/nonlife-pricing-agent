from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.feature_selection_agent import FeatureSelectionAgent
from core.schemas import (
    CategoricalFeatureConfig,
    CategoricalFeatureSeed,
    FeatureProposal,
    FeatureSeed,
    NumericFeatureConfig,
    NumericFeatureSeed,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "vehicle_age": rng.uniform(0, 20, n),
        "vehicle_value": rng.uniform(1000, 50000, n),
        "vehicle_brand": rng.choice(["RENAULT", "BMW"], n),
        "premium": rng.uniform(100, 1000, n),
        "exposure_years": rng.uniform(0.5, 1.0, n),
    })


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call_template.return_value = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d")],
        categorical=[],
    )
    return llm


def test_propose_hides_locked_column_from_prompt_and_merges_it_back(mock_llm, sample_df):
    seed = FeatureSeed(categorical=[CategoricalFeatureSeed(
        name="vehicle_brand", description="seeded", n_clusters=2, approved=True,
        temperature=0.0, grouping={"A": ["RENAULT"], "B": ["BMW"]},
    )])

    agent = FeatureSelectionAgent(mock_llm)
    proposal = agent.propose(
        df=sample_df, target_col="premium", exposure_col="exposure_years",
        objective="gamma", seed=seed,
    )

    sent_profiles = mock_llm.call_template.call_args.kwargs["column_profiles_json"]
    assert "vehicle_brand" not in sent_profiles

    names = {c.name for c in proposal.categorical}
    assert "vehicle_brand" in names
    merged = next(c for c in proposal.categorical if c.name == "vehicle_brand")
    assert merged.grouping == {"A": ["RENAULT"], "B": ["BMW"]}
    assert merged.approved is True


def test_propose_keeps_flexible_column_in_prompt_and_seed_context(mock_llm, sample_df):
    seed = FeatureSeed(numeric=[NumericFeatureSeed(
        name="vehicle_value", description="seeded prior", approved=True, temperature=0.5,
    )])

    agent = FeatureSelectionAgent(mock_llm)
    agent.propose(
        df=sample_df, target_col="premium", exposure_col="exposure_years",
        objective="gamma", seed=seed,
    )

    kwargs = mock_llm.call_template.call_args.kwargs
    assert "vehicle_value" in kwargs["column_profiles_json"]
    assert "vehicle_value" in kwargs["seed_context_json"]
    assert '"temperature": 0.5' in kwargs["seed_context_json"]


def test_propose_without_seed_passes_empty_seed_context(mock_llm, sample_df):
    agent = FeatureSelectionAgent(mock_llm)
    agent.propose(
        df=sample_df, target_col="premium", exposure_col="exposure_years", objective="gamma",
    )

    assert mock_llm.call_template.call_args.kwargs["seed_context_json"] == "[]"


def test_refine_remark_on_locked_column_lets_it_through_this_round(mock_llm, sample_df):
    seed = FeatureSeed(categorical=[CategoricalFeatureSeed(
        name="vehicle_brand", description="seeded", n_clusters=2, approved=True,
        temperature=0.0, grouping={"A": ["RENAULT"], "B": ["BMW"]},
    )])
    previous = FeatureProposal(
        numeric=[],
        categorical=[CategoricalFeatureConfig(
            name="vehicle_brand", description="seeded", n_clusters=2, approved=True,
            grouping={"A": ["RENAULT"], "B": ["BMW"]},
        )],
    )
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[],
        categorical=[CategoricalFeatureConfig(
            name="vehicle_brand", description="revised per actuary", n_clusters=3, approved=True,
        )],
    )

    agent = FeatureSelectionAgent(mock_llm)
    updated = agent.refine(
        df=sample_df, previous_proposal=previous,
        actuary_remarks={"vehicle_brand": "split into 3 groups instead"},
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=seed,
    )

    sent_profiles = mock_llm.call_template.call_args.kwargs["column_profiles_json"]
    assert "vehicle_brand" in sent_profiles  # remark overrides the lock for this round
    assert updated.categorical[0].description == "revised per actuary"  # LLM's answer trusted


def test_refine_without_remark_keeps_locked_column_hidden_and_unchanged(mock_llm, sample_df):
    seed = FeatureSeed(categorical=[CategoricalFeatureSeed(
        name="vehicle_brand", description="seeded", n_clusters=2, approved=True,
        temperature=0.0, grouping={"A": ["RENAULT"], "B": ["BMW"]},
    )])
    previous = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[CategoricalFeatureConfig(
            name="vehicle_brand", description="seeded", n_clusters=2, approved=True,
            grouping={"A": ["RENAULT"], "B": ["BMW"]},
        )],
    )
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[],  # LLM never saw vehicle_brand, so it can't return it
    )

    agent = FeatureSelectionAgent(mock_llm)
    updated = agent.refine(
        df=sample_df, previous_proposal=previous, actuary_remarks={"vehicle_age": "keep it"},
        objective="gamma", target_col="premium", exposure_col="exposure_years", seed=seed,
    )

    sent_profiles = mock_llm.call_template.call_args.kwargs["column_profiles_json"]
    sent_previous = mock_llm.call_template.call_args.kwargs["previous_proposal_json"]
    assert "vehicle_brand" not in sent_profiles
    assert "vehicle_brand" not in sent_previous
    merged = next(c for c in updated.categorical if c.name == "vehicle_brand")
    assert merged.grouping == {"A": ["RENAULT"], "B": ["BMW"]}
