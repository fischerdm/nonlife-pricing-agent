from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from core.feature_pipeline import generate_draft, refine_draft, save_feature_checkpoint
from core.schemas import (
    CategoricalFeatureConfig,
    CategoricalFeatureSeed,
    CategoryCluster,
    FeatureProposal,
    FeatureSeed,
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


def test_generate_draft_merges_locked_seed_and_skips_its_grouping(mock_llm, sample_df):
    seed = FeatureSeed(categorical=[CategoricalFeatureSeed(
        name="occupation", description="seeded", n_clusters=2, approved=True, temperature=0.0,
        grouping={"HIGH_RISK": ["delivery_driver"], "LOW_RISK": ["office_worker"]},
    )])
    # The agent never sees `occupation` (it's locked), so it can't propose it itself.
    mock_llm.call_template.return_value = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d")],
        categorical=[],
    )

    proposal = generate_draft(mock_llm, sample_df, DATA_CFG, GROUPING_CFG, seed=seed)

    sent_profiles = mock_llm.call_template.call_args.kwargs["column_profiles_json"]
    assert "occupation" not in sent_profiles
    mock_llm.call.assert_not_called()  # no GroupingAgent call for the seeded grouping
    names = {c.name for c in proposal.categorical}
    assert "occupation" in names
    merged = next(c for c in proposal.categorical if c.name == "occupation")
    assert merged.grouping == {"HIGH_RISK": ["delivery_driver"], "LOW_RISK": ["office_worker"]}


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


# ── save_feature_checkpoint: downstream invalidation ───────────────────────────

def _proposal(numeric_names, categorical_name=None, grouping=None):
    return FeatureProposal(
        numeric=[NumericFeatureConfig(name=n, description="d", approved=True) for n in numeric_names],
        categorical=(
            [CategoricalFeatureConfig(
                name=categorical_name, description="d", approved=True, grouping=grouping,
            )] if categorical_name else []
        ),
    )


def test_save_feature_checkpoint_first_save_is_never_flagged_invalidated(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    config = {}  # no prior checkpoint at all

    invalidated = save_feature_checkpoint(config_path, config, _proposal(["vehicle_age"]))

    assert invalidated is False
    assert config["features"]["numeric"][0]["name"] == "vehicle_age"


def test_save_feature_checkpoint_no_invalidation_when_feature_set_unchanged(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    config = {
        "features": {"numeric": [{"name": "vehicle_age"}], "categorical": []},
        "gbm_output": {"interactions": [{"feature_a": "a", "feature_b": "b", "h_statistic": 0.1}]},
    }

    invalidated = save_feature_checkpoint(config_path, config, _proposal(["vehicle_age"]))

    assert invalidated is False
    assert config["gbm_output"]  # untouched


def test_save_feature_checkpoint_invalidates_gbm_output_when_feature_set_changes(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    config = {
        "features": {"numeric": [{"name": "vehicle_age"}], "categorical": []},
        "gbm_output": {"interactions": [{"feature_a": "a", "feature_b": "b", "h_statistic": 0.1}]},
    }

    invalidated = save_feature_checkpoint(config_path, config, _proposal(["driver_age"]))

    assert invalidated is True
    assert "gbm_output" not in config


def test_save_feature_checkpoint_clears_glm_terms_when_feature_set_changes(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    glm_config_path = tmp_path / "glm_config.yaml"
    glm_config_path.write_text(yaml.dump({
        "glm": {"terms": [{"name": "vehicle_age", "term_type": "main", "rationale": "r", "approved": True}],
                 "formula": "premium ~ vehicle_age"},
    }))
    config = {"features": {"numeric": [{"name": "vehicle_age"}], "categorical": []}}

    invalidated = save_feature_checkpoint(config_path, config, _proposal(["driver_age"]))

    assert invalidated is True
    saved_glm = yaml.safe_load(glm_config_path.read_text())
    assert saved_glm["glm"]["terms"] == []
    assert saved_glm["glm"]["formula"] is None


def test_save_feature_checkpoint_invalidates_on_grouping_change(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    config = {
        "features": {
            "numeric": [],
            "categorical": [{"name": "occupation", "grouping": {"A": ["x"], "B": ["y"]}}],
        },
        "gbm_output": {"interactions": []},
    }
    new_proposal = _proposal([], categorical_name="occupation", grouping={"A": ["x", "y"]})

    invalidated = save_feature_checkpoint(config_path, config, new_proposal)

    assert invalidated is True
    assert "gbm_output" not in config
