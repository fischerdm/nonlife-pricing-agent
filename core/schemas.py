from pydantic import BaseModel, field_validator
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

class CommentEntry(BaseModel):
    """One entry in a variable's accumulated comment history.

    Distinct from `actuary_note` below, which is transient — the LLM's single
    reply for the current round only. History is code-managed: the agent is
    never asked to produce it, only to reply, and the calling code appends that
    reply on top of what's already there (see `refine_draft`).
    """
    author: Literal["actuary", "agent"]
    text: str
    ts: str                                # ISO datetime, same convention as SessionLogger
    sent: bool = False                     # already included in a remarks payload sent to the agent


def _discard_malformed_comment_history(v: object) -> list:
    """Defensive backstop, not just a prompt nicety: comment_history is entirely
    code-managed (see refine_draft) — the agent is never asked to populate it and
    shouldn't see it at all in what it's sent — but if a future prompt tweak or an
    off-script response still includes something here (e.g. a flat string instead
    of a CommentEntry dict), silently drop it rather than fail validation for the
    whole proposal. Real, code-appended history is unaffected either way.
    """
    if not isinstance(v, list):
        return []
    return [e for e in v if isinstance(e, (dict, CommentEntry))]


class NumericFeatureConfig(BaseModel):
    name: str
    description: str
    data_quality_note: str | None = None
    approved: bool | None = None
    actuary_note: str | None = None        # transient: the LLM's reply for the current round only
    comment_history: list[CommentEntry] = []

    @field_validator("comment_history", mode="before")
    @classmethod
    def _validate_comment_history(cls, v: object) -> list:
        return _discard_malformed_comment_history(v)


class CategoricalFeatureConfig(BaseModel):
    name: str
    description: str
    ordinal: bool = False
    order: list[str] | None = None         # ordinal level order, lowest → highest risk
    n_clusters: int = 5
    data_quality_note: str | None = None
    approved: bool | None = None
    actuary_note: str | None = None        # transient: the LLM's reply for the current round only
    comment_history: list[CommentEntry] = []
    grouping: dict[str, list[str]] | None = None   # filled by grouping agent

    @field_validator("comment_history", mode="before")
    @classmethod
    def _validate_comment_history(cls, v: object) -> list:
        return _discard_malformed_comment_history(v)


class FeatureProposal(BaseModel):
    numeric: list[NumericFeatureConfig]
    categorical: list[CategoricalFeatureConfig]
    excluded: list[str] = []
    exclusion_rationale: dict[str, str] = {}          # why the agent left it out
    excluded_description: dict[str, str] = {}         # what the column represents


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


# ── GBM output ───────────────────────────────────────────────────────────────

class PairwiseInteraction(BaseModel):
    feature_a: str
    feature_b: str
    h_statistic: float


# ── Seed config schemas (actuary-authored priors, see config/*_seed.example.yaml) ──
#
# Seeds are optional, hand-edited, rarely-changing inputs — distinct from
# project_config.yaml/glm_config.yaml, which are session checkpoints that
# accumulate and get invalidated. `temperature` governs how much license the
# *agent* has to deviate from an actuary-specified entry (0.0 is enforced in
# code, never even sent to the LLM; >0.0 is a prompt-level suggestion) — it
# never limits the actuary's own always-full override power.

class NumericFeatureSeed(NumericFeatureConfig):
    temperature: float = 0.3
    updated_at: str | None = None


class CategoricalFeatureSeed(CategoricalFeatureConfig):
    temperature: float = 0.3
    updated_at: str | None = None


class FeatureSeed(BaseModel):
    numeric: list[NumericFeatureSeed] = []
    categorical: list[CategoricalFeatureSeed] = []


class CommerciallyExcludedEntry(BaseModel):
    name: str
    rationale: str
    temperature: float = 0.0
    updated_at: str | None = None


class DistillationSeed(BaseModel):
    commercially_excluded: list[CommerciallyExcludedEntry] = []


