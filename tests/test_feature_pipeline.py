from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

import core.feature_pipeline as feature_pipeline
from core.feature_pipeline import (
    generate_draft,
    list_draft_snapshots,
    load_draft_snapshot,
    reconcile_membership,
    refine_draft,
    save_draft_snapshot,
    save_feature_checkpoint,
)
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


# ── reconcile_membership ────────────────────────────────────────────────────────

def test_reconcile_membership_unchecked_moves_to_excluded(sample_df):
    draft = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[],
    )

    updated = reconcile_membership(draft, {"vehicle_age": False}, sample_df)

    assert updated.numeric == []
    assert "vehicle_age" in updated.excluded
    assert updated.excluded_description["vehicle_age"] == "d"
    assert updated.exclusion_rationale["vehicle_age"] == "Actuary excluded this round."


def test_reconcile_membership_unchecked_uses_actuary_note_as_rationale(sample_df):
    draft = FeatureProposal(
        numeric=[NumericFeatureConfig(
            name="vehicle_age", description="d", approved=True, actuary_note="Too collinear with driver_age.",
        )],
        categorical=[],
    )

    updated = reconcile_membership(draft, {"vehicle_age": False}, sample_df)

    assert updated.exclusion_rationale["vehicle_age"] == "Too collinear with driver_age."


def test_reconcile_membership_checked_excluded_promotes_by_dtype(sample_df):
    draft = FeatureProposal(
        numeric=[], categorical=[],
        excluded=["vehicle_age", "occupation"],
        exclusion_rationale={"vehicle_age": "r", "occupation": "r"},
        excluded_description={"vehicle_age": "d", "occupation": "d"},
    )

    updated = reconcile_membership(draft, {"vehicle_age": True, "occupation": True}, sample_df)

    assert [f.name for f in updated.numeric] == ["vehicle_age"]
    assert [f.name for f in updated.categorical] == ["occupation"]
    assert updated.excluded == []
    assert "vehicle_age" not in updated.exclusion_rationale
    assert "occupation" not in updated.excluded_description


def test_reconcile_membership_sets_approved_true_for_kept_and_promoted(sample_df):
    draft = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=None)],
        categorical=[],
        excluded=["occupation"],
        exclusion_rationale={"occupation": "r"}, excluded_description={"occupation": "d"},
    )

    updated = reconcile_membership(draft, {"vehicle_age": True, "occupation": True}, sample_df)

    assert updated.numeric[0].approved is True
    assert updated.categorical[0].approved is True


def test_reconcile_membership_carries_comment_into_promoted_actuary_note(sample_df):
    draft = FeatureProposal(
        numeric=[], categorical=[],
        excluded=["occupation"], exclusion_rationale={"occupation": "r"}, excluded_description={"occupation": "d"},
    )

    updated = reconcile_membership(
        draft, {"occupation": True}, sample_df, comments={"occupation": "please include this"},
    )

    assert updated.categorical[0].actuary_note == "please include this"


def test_reconcile_membership_missing_checkbox_state_defaults_to_excluded(sample_df):
    """A feature the caller forgot to include a checkbox_state entry for is treated
    as unchecked (demoted), never silently kept — avoids stale approvals surviving
    by accident."""
    draft = FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[],
    )

    updated = reconcile_membership(draft, {}, sample_df)

    assert updated.numeric == []
    assert "vehicle_age" in updated.excluded


# ── Draft snapshots ──────────────────────────────────────────────────────────────

def _sample_proposal() -> FeatureProposal:
    return FeatureProposal(
        numeric=[NumericFeatureConfig(name="vehicle_age", description="d", approved=True)],
        categorical=[],
    )


def test_list_draft_snapshots_returns_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    assert list_draft_snapshots("initial") == []
    assert list_draft_snapshots("modified") == []


def test_save_and_load_draft_snapshot_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    path = save_draft_snapshot(_sample_proposal(), kind="initial")
    loaded = load_draft_snapshot(path)

    assert loaded.numeric[0].name == "vehicle_age"


def test_save_draft_snapshot_never_overwrites_prior_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    first = save_draft_snapshot(_sample_proposal(), kind="initial")
    second = save_draft_snapshot(FeatureProposal(
        numeric=[NumericFeatureConfig(name="driver_age", description="d", approved=True)],
        categorical=[],
    ), kind="initial")

    assert first != second
    assert first.exists() and second.exists()


def test_list_draft_snapshots_returns_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    first = save_draft_snapshot(_sample_proposal(), kind="initial")
    second = save_draft_snapshot(_sample_proposal(), kind="initial")

    assert list_draft_snapshots("initial") == [second, first]


def test_list_draft_snapshots_keeps_kinds_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    save_draft_snapshot(_sample_proposal(), kind="initial")
    save_draft_snapshot(_sample_proposal(), kind="modified")
    save_draft_snapshot(_sample_proposal(), kind="modified")

    assert len(list_draft_snapshots("initial")) == 1
    assert len(list_draft_snapshots("modified")) == 2


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
