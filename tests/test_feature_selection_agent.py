from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.feature_selection_agent import FeatureSelectionAgent
from core.schemas import (
    CategoricalFeatureConfig,
    CategoricalFeatureSeed,
    CommentEntry,
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


def test_refine_omits_comment_history_key_entirely_from_prompt(mock_llm, sample_df):
    """Regression test: an earlier version only emptied comment_history to []
    instead of omitting the key, and seeing that key at all was enough for the
    LLM to try filling it with a flat string, which fails CommentEntry
    validation. The key must not appear in previous_proposal_json at all."""
    previous = FeatureProposal(
        numeric=[NumericFeatureConfig(
            name="vehicle_age", description="d", approved=True,
            comment_history=[CommentEntry(author="actuary", text="why keep this?", ts="t1")],
        )],
        categorical=[],
    )

    agent = FeatureSelectionAgent(mock_llm)
    agent.refine(
        df=sample_df, previous_proposal=previous, actuary_remarks={"vehicle_age": "why keep this?"},
        objective="gamma", target_col="premium", exposure_col="exposure_years",
    )

    sent_previous = mock_llm.call_template.call_args.kwargs["previous_proposal_json"]
    assert "comment_history" not in sent_previous


def test_refine_demoted_feature_sent_as_excluded_and_restored_with_history(mock_llm, sample_df):
    """A numeric/categorical feature the actuary unchecked (approved=False) stays
    a full config object in reconcile_membership's output (see
    test_feature_pipeline.py) so its comment_history survives — but the
    refinement prompt's "excluded" shape was never designed to see an
    approved:false entry sitting inside "numeric". `refine` must represent it as
    a bare `excluded` name for the LLM round-trip, then restore the real object
    (comment_history intact) afterward regardless of what the LLM echoed back."""
    previous = FeatureProposal(
        numeric=[NumericFeatureConfig(
            name="vehicle_age", description="d", approved=False,
            comment_history=[CommentEntry(author="actuary", text="target leakage", ts="t1")],
        )],
        categorical=[],
    )
    mock_llm.call_template.return_value = FeatureProposal(numeric=[], categorical=[])

    agent = FeatureSelectionAgent(mock_llm)
    updated = agent.refine(
        df=sample_df, previous_proposal=previous, actuary_remarks={},
        objective="gamma", target_col="premium", exposure_col="exposure_years",
    )

    sent_previous = mock_llm.call_template.call_args.kwargs["previous_proposal_json"]
    assert '"name": "vehicle_age"' not in sent_previous.split('"excluded"')[0]  # not inside "numeric"
    assert "vehicle_age" in sent_previous  # present as a bare excluded name instead

    assert [f.name for f in updated.numeric] == ["vehicle_age"]
    assert updated.numeric[0].approved is False
    assert [e.text for e in updated.numeric[0].comment_history] == ["target leakage"]
    assert "vehicle_age" not in updated.excluded


def test_refine_demoted_feature_unremarked_does_not_duplicate_note(mock_llm, sample_df):
    """The LLM is instructed to carry an unremarked excluded entry's rationale
    forward unchanged — if it echoes back the actuary's own prior comment text,
    that must NOT be re-appended to comment_history mislabeled as a Claude
    reply."""
    previous = FeatureProposal(
        numeric=[NumericFeatureConfig(
            name="vehicle_age", description="d", approved=False,
            comment_history=[CommentEntry(author="actuary", text="target leakage", ts="t1")],
        )],
        categorical=[],
    )
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[], categorical=[],
        excluded=["vehicle_age"], exclusion_rationale={"vehicle_age": "target leakage"},
    )

    agent = FeatureSelectionAgent(mock_llm)
    updated = agent.refine(
        df=sample_df, previous_proposal=previous, actuary_remarks={},
        objective="gamma", target_col="premium", exposure_col="exposure_years",
    )

    history = updated.numeric[0].comment_history
    assert len(history) == 1
    assert history[0].author == "actuary"


def test_refine_demoted_feature_remarked_folds_reply_into_history(mock_llm, sample_df):
    """A remark on a still-unchecked feature can't change its placement, but the
    LLM's reply should still reach the actuary — same as `actuary_note` does for
    any other feature."""
    previous = FeatureProposal(
        numeric=[NumericFeatureConfig(
            name="vehicle_age", description="d", approved=False,
            comment_history=[CommentEntry(author="actuary", text="is this target leakage?", ts="t1")],
        )],
        categorical=[],
    )
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[], categorical=[],
        excluded=["vehicle_age"],
        exclusion_rationale={"vehicle_age": "Confirmed: correlates 0.98 with the target."},
    )

    agent = FeatureSelectionAgent(mock_llm)
    updated = agent.refine(
        df=sample_df, previous_proposal=previous,
        actuary_remarks={"vehicle_age": "is this target leakage?"},
        objective="gamma", target_col="premium", exposure_col="exposure_years",
    )

    # agent.refine alone only sets actuary_note — refine_draft (core/feature_pipeline.py)
    # is what folds it into comment_history, same as for any other feature.
    assert updated.numeric[0].actuary_note == "Confirmed: correlates 0.98 with the target."
    assert [e.text for e in updated.numeric[0].comment_history] == ["is this target leakage?"]
