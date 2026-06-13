from pydantic import BaseModel
from typing import Literal


# ── Phase 1 schemas (hypothesis generation) ───────────────────────────────────

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


class ValidationResult(BaseModel):
    hypothesis: InteractionHypothesis
    deviance_delta_pct: float
    gain_rank: int
    baseline_deviance: float
    new_deviance: float
    approved: bool | None = None


# ── Feature selection schemas ─────────────────────────────────────────────────

class NumericFeatureConfig(BaseModel):
    name: str
    description: str
    data_quality_note: str | None = None
    approved: bool | None = None
    actuary_note: str | None = None


class CategoricalFeatureConfig(BaseModel):
    name: str
    description: str
    ordinal: bool = False
    order: list[str] | None = None         # ordinal level order, lowest → highest risk
    n_clusters: int = 5
    data_quality_note: str | None = None
    approved: bool | None = None
    actuary_note: str | None = None
    grouping: dict[str, list[str]] | None = None   # filled by grouping agent


class FeatureProposal(BaseModel):
    numeric: list[NumericFeatureConfig]
    categorical: list[CategoricalFeatureConfig]
    excluded: list[str] = []
    exclusion_rationale: dict[str, str] = {}


# ── Grouping schemas ──────────────────────────────────────────────────────────

class CategoryCluster(BaseModel):
    cluster_name: str
    elements: list[str]
    rationale: str


class GroupingResponse(BaseModel):
    clusters: list[CategoryCluster]


# ── GLM / distillation schemas ────────────────────────────────────────────────

class GLMTerm(BaseModel):
    name: str                                       # "driver_age" or "driver_age:vehicle_age"
    term_type: Literal["main", "interaction", "polynomial"]
    h_statistic: float | None = None                # SHAP H-stat, interactions only
    rationale: str
    approved: bool | None = None
    actuary_note: str | None = None


class GLMProposal(BaseModel):
    terms: list[GLMTerm]
    formula: str | None = None                      # patsy formula, built after approval


# ── Legacy distillation (Phase 3 placeholder) ─────────────────────────────────

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
