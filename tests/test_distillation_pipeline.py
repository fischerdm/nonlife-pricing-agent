from unittest.mock import MagicMock

import pytest

from core.distillation_pipeline import (
    add_manual_interaction,
    add_manual_main_effect,
    generate_glm_draft,
    list_glm_draft_snapshots,
    load_glm_draft_snapshot,
    reconcile_terms,
    refine_glm_draft,
    save_glm_draft_snapshot,
)
from core.schemas import CommentEntry, CommerciallyExcludedEntry, DistillationSeed, GLMProposal, GLMTerm

DATA_CFG = {"objective": "gamma", "target_col": "premium", "exposure_col": "exposure_years"}


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call_template.return_value = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
    ])
    return llm


def test_generate_glm_draft_forwards_seed_to_agent(mock_llm):
    seed = DistillationSeed(commercially_excluded=[
        CommerciallyExcludedEntry(name="driver_occupation", rationale="r"),
    ])

    generate_glm_draft(
        mock_llm, [{"feature_a": "driver_occupation", "feature_b": "driver_age", "h_statistic": 0.1}],
        ["driver_age", "driver_occupation"], DATA_CFG, seed=seed,
    )

    kwargs = mock_llm.call_template.call_args.kwargs
    assert "driver_occupation" not in kwargs["features_json"]


def test_refine_glm_draft_forwards_seed_to_agent(mock_llm):
    seed = DistillationSeed(commercially_excluded=[
        CommerciallyExcludedEntry(name="driver_occupation", rationale="r"),
    ])
    mock_llm.call_template.return_value = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="driver_occupation", term_type="main", rationale="hallucinated"),
    ])
    previous = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])

    updated = refine_glm_draft(mock_llm, previous, {"driver_age": "keep"}, DATA_CFG, seed=seed)

    assert {t.name for t in updated.terms} == {"driver_age"}


def test_refine_glm_draft_forwards_general_remark_to_agent(mock_llm):
    previous = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])

    refine_glm_draft(
        mock_llm, previous, {}, DATA_CFG, general_remark="consider region:vehicle_age",
    )

    kwargs = mock_llm.call_template.call_args.kwargs
    assert kwargs["general_remark"] == "consider region:vehicle_age"


# ── refine_glm_draft: comment history accumulation ──────────────────────────────

def test_refine_glm_draft_appends_agent_reply_and_clears_transient_note(mock_llm):
    previous = GLMProposal(terms=[GLMTerm(
        name="driver_age", term_type="main", rationale="r",
        comment_history=[CommentEntry(author="actuary", text="why keep this?", ts="t1")],
    )])
    mock_llm.call_template.return_value = GLMProposal(terms=[GLMTerm(
        name="driver_age", term_type="main", rationale="r", actuary_note="Kept: strong predictor.",
    )])

    updated = refine_glm_draft(mock_llm, previous, {"driver_age": "why keep this?"}, DATA_CFG)

    term = updated.terms[0]
    assert term.actuary_note is None  # transient field cleared, history is the source of truth
    assert [e.text for e in term.comment_history] == ["why keep this?", "Kept: strong predictor."]
    assert term.comment_history[0].sent is True  # included in this round's remarks
    assert term.comment_history[1].author == "agent"


def test_refine_glm_draft_leaves_unremarked_history_unsent(mock_llm):
    previous = GLMProposal(terms=[
        GLMTerm(
            name="driver_age", term_type="main", rationale="r",
            comment_history=[CommentEntry(author="actuary", text="pending", ts="t1")],
        ),
        GLMTerm(name="vehicle_age", term_type="main", rationale="r"),
    ])
    mock_llm.call_template.return_value = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="vehicle_age", term_type="main", rationale="r", actuary_note="unrelated reply"),
    ])

    updated = refine_glm_draft(mock_llm, previous, {"vehicle_age": "some remark"}, DATA_CFG)

    assert updated.terms[0].comment_history[0].sent is False


# ── reconcile_terms ──────────────────────────────────────────────────────────────

def test_reconcile_terms_forces_approved_from_checkbox_state():
    draft = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r", approved=True)])

    updated = reconcile_terms(draft, checkbox_state={"driver_age": False})

    assert updated.terms[0].approved is False


def test_reconcile_terms_derives_term_type_structurally_from_name():
    draft = GLMProposal(terms=[
        GLMTerm(name="driver_age:vehicle_age", term_type="main", rationale="r"),  # agent mislabeled it
        GLMTerm(name="driver_age", term_type="interaction", rationale="r"),        # agent mislabeled it
    ])

    updated = reconcile_terms(draft, checkbox_state={})

    by_name = {t.name: t.term_type for t in updated.terms}
    assert by_name["driver_age:vehicle_age"] == "interaction"
    assert by_name["driver_age"] == "main"


def test_reconcile_terms_drops_unprompted_polynomial_reclassification():
    draft = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="polynomial", rationale="r")])

    updated = reconcile_terms(draft, checkbox_state={}, remarked=set())

    assert updated.terms[0].term_type == "main"


def test_reconcile_terms_keeps_polynomial_when_actuary_remarked_it():
    draft = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="polynomial", rationale="r")])

    updated = reconcile_terms(draft, checkbox_state={}, remarked={"driver_age"})

    assert updated.terms[0].term_type == "polynomial"


# ── add_manual_interaction ────────────────────────────────────────────────────────

def test_add_manual_interaction_appends_approved_term():
    draft = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="vehicle_age", term_type="main", rationale="r"),
    ])

    updated = add_manual_interaction(draft, "driver_age", "vehicle_age", "actuarially sound")

    added = updated.terms[-1]
    assert added.name == "driver_age:vehicle_age"
    assert added.term_type == "interaction"
    assert added.approved is True
    assert added.rationale == "actuarially sound"


def test_add_manual_interaction_rejects_self_interaction():
    draft = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])

    with pytest.raises(ValueError):
        add_manual_interaction(draft, "driver_age", "driver_age")


def test_add_manual_main_effect_appends_approved_term():
    draft = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])

    updated = add_manual_main_effect(draft, "vehicle_brand", "actuarially sound")

    added = updated.terms[-1]
    assert added.name == "vehicle_brand"
    assert added.term_type == "main"
    assert added.approved is True
    assert added.rationale == "actuarially sound"


def test_add_manual_main_effect_rejects_existing_name():
    draft = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])

    with pytest.raises(ValueError):
        add_manual_main_effect(draft, "driver_age")


def test_add_manual_interaction_rejects_duplicate_regardless_of_order():
    draft = GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r"),
        GLMTerm(name="vehicle_age", term_type="main", rationale="r"),
        GLMTerm(name="vehicle_age:driver_age", term_type="interaction", rationale="r"),
    ])

    with pytest.raises(ValueError):
        add_manual_interaction(draft, "driver_age", "vehicle_age")


# ── draft snapshots ────────────────────────────────────────────────────────────────

def test_save_and_load_glm_draft_snapshot_round_trips(tmp_path, monkeypatch):
    import core.distillation_pipeline as distillation_pipeline
    monkeypatch.setattr(distillation_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    proposal = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])
    path = save_glm_draft_snapshot(proposal, kind="initial")
    loaded = load_glm_draft_snapshot(path)

    assert loaded == proposal
    assert list_glm_draft_snapshots("initial") == [path]


def test_glm_draft_snapshots_keep_kinds_separate_and_dont_collide_with_feature_snapshots(tmp_path, monkeypatch):
    import core.distillation_pipeline as distillation_pipeline
    import core.feature_pipeline as feature_pipeline
    monkeypatch.setattr(distillation_pipeline, "DRAFTS_DIR", tmp_path / "drafts")
    monkeypatch.setattr(feature_pipeline, "DRAFTS_DIR", tmp_path / "drafts")

    proposal = GLMProposal(terms=[GLMTerm(name="driver_age", term_type="main", rationale="r")])
    save_glm_draft_snapshot(proposal, kind="modified")
    feature_pipeline.save_draft_snapshot(
        feature_pipeline.FeatureProposal(numeric=[], categorical=[]), kind="modified",
    )

    assert len(list_glm_draft_snapshots("modified")) == 1
    assert len(feature_pipeline.list_draft_snapshots("modified")) == 1
