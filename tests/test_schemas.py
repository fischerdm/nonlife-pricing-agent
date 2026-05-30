import pytest
from pydantic import ValidationError

from core.schemas import (
    CategoryCluster,
    FeatureMetadata,
    GroupingResponse,
    HypothesisResponse,
    InteractionHypothesis,
    ValidationResult,
)


def test_interaction_hypothesis_valid():
    h = InteractionHypothesis(
        feature_a="driver_age",
        feature_b="vehicle_power_kw",
        operation="multiply",
        new_feature_name="age_x_kw",
        rationale="Young driver / high power synergy",
    )
    assert h.feature_a == "driver_age"
    assert h.operation == "multiply"


def test_interaction_hypothesis_invalid_operation():
    with pytest.raises(ValidationError):
        InteractionHypothesis(
            feature_a="a",
            feature_b="b",
            operation="subtract",
            new_feature_name="a_minus_b",
            rationale="invalid",
        )


def test_hypothesis_response_empty():
    r = HypothesisResponse(hypotheses=[])
    assert r.hypotheses == []


def test_validation_result_default_approved_is_none():
    h = InteractionHypothesis(
        feature_a="a", feature_b="b", operation="multiply",
        new_feature_name="a_x_b", rationale="test",
    )
    v = ValidationResult(
        hypothesis=h,
        deviance_delta_pct=-0.5,
        gain_rank=3,
        baseline_deviance=1.0,
        new_deviance=0.995,
    )
    assert v.approved is None


def test_grouping_response_clusters():
    r = GroupingResponse(clusters=[
        CategoryCluster(cluster_name="LOW_RISK", elements=["a", "b"], rationale="safe"),
        CategoryCluster(cluster_name="HIGH_RISK", elements=["c"], rationale="risky"),
    ])
    assert len(r.clusters) == 2
    assert r.clusters[0].cluster_name == "LOW_RISK"


def test_feature_metadata_dtype_constraint():
    with pytest.raises(ValidationError):
        FeatureMetadata(name="x", dtype="ordinal", description="bad dtype")
