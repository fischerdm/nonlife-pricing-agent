from unittest.mock import MagicMock

import pytest

from core.distillation_pipeline import generate_glm_draft, refine_glm_draft
from core.schemas import CommerciallyExcludedEntry, DistillationSeed, GLMProposal, GLMTerm

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
