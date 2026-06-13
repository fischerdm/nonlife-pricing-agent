from pydantic import BaseModel
from typing import Literal


class FeatureMetadata(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    description: str


class InteractionHypothesis(BaseModel):
    feature_a: str
    feature_b: str
    operation: Literal["multiply", "divide", "ratio_a_over_b", "ratio_b_over_a"]
    new_feature_name: str
    rationale: str


class HypothesisResponse(BaseModel):
    hypotheses: list[InteractionHypothesis]


class CategoryCluster(BaseModel):
    cluster_name: str
    elements: list[str]
    rationale: str


class GroupingResponse(BaseModel):
    clusters: list[CategoryCluster]


class ValidationResult(BaseModel):
    hypothesis: InteractionHypothesis
    deviance_delta_pct: float
    gain_rank: int
    baseline_deviance: float
    new_deviance: float
    approved: bool | None = None


class DistillationInteraction(BaseModel):
    feature_a: str
    feature_b: str
    h_statistic: float
    glm_term: Literal["product", "ratio", "binned"]
    rationale: str
    spurious: bool = False


class DistillationResponse(BaseModel):
    approved_interactions: list[DistillationInteraction]
    rejected_interactions: list[DistillationInteraction]
    glm_formula_terms: list[str]
