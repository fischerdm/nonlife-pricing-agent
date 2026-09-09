import json

import pandas as pd

from core.llm_client import LLMClient
from core.schemas import (
    CategoricalFeatureConfig,
    CategoricalFeatureSeed,
    FeatureProposal,
    FeatureSeed,
    NumericFeatureConfig,
    NumericFeatureSeed,
)

_EXCLUDE_ALWAYS = {"insured_id", "year"}


class FeatureSelectionAgent:
    """Profile dataset columns, propose a feature list, refine based on actuary remarks."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def propose(
        self,
        df: pd.DataFrame,
        target_col: str,
        exposure_col: str,
        objective: str,
        seed: FeatureSeed | None = None,
    ) -> FeatureProposal:
        exclude = {target_col, exposure_col} | _EXCLUDE_ALWAYS
        locked = _locked_seed_names(seed)
        profiles = self._profile_columns(df, exclude | locked)
        proposal = self.llm.call_template(
            agent_name="feature_selection",
            section="proposal",
            response_model=FeatureProposal,
            objective=objective,
            target_col=target_col,
            exposure_col=exposure_col,
            column_profiles_json=json.dumps(profiles, indent=2),
            seed_context_json=json.dumps(_seed_context(seed), indent=2),
        )
        return _merge_locked_entries(proposal, seed)

    def refine(
        self,
        df: pd.DataFrame,
        previous_proposal: FeatureProposal,
        actuary_remarks: dict[str, str],
        objective: str,
        target_col: str,
        exposure_col: str,
        seed: FeatureSeed | None = None,
    ) -> FeatureProposal:
        exclude = {target_col, exposure_col} | _EXCLUDE_ALWAYS
        # A locked entry the actuary left a remark on this round is the actuary's
        # own override, not the agent deviating unprompted — let it through.
        locked = _locked_seed_names(seed) - set(actuary_remarks)
        profiles = self._profile_columns(df, exclude | locked)
        stripped, demoted = _flatten_demoted(_strip_locked(previous_proposal, locked))
        sendable_previous = _proposal_dict_for_prompt(stripped)
        proposal = self.llm.call_template(
            agent_name="feature_selection",
            section="refinement",
            response_model=FeatureProposal,
            objective=objective,
            target_col=target_col,
            exposure_col=exposure_col,
            column_profiles_json=json.dumps(profiles, indent=2),
            previous_proposal_json=json.dumps(sendable_previous, indent=2),
            actuary_remarks_json=json.dumps(actuary_remarks, indent=2),
            seed_context_json=json.dumps(_seed_context(seed, exclude_names=set(actuary_remarks)), indent=2),
        )
        proposal = _restore_demoted(proposal, demoted, remarked=set(actuary_remarks))
        return _merge_locked_entries(proposal, seed, already_handled=set(actuary_remarks))

    def _profile_columns(self, df: pd.DataFrame, exclude: set[str]) -> list[dict]:
        profiles = []
        for col in df.columns:
            if col in exclude:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            profile: dict = {
                "name": col,
                "dtype": "numeric" if is_numeric else "categorical",
                "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                "n_unique": int(df[col].nunique()),
            }
            if is_numeric:
                profile.update({
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                })
            else:
                profile["top_values"] = {
                    str(k): int(v)
                    for k, v in df[col].value_counts().head(10).items()
                }
            profiles.append(profile)
        return profiles


# ── Seed-lock helpers ──────────────────────────────────────────────────────────

def _locked_seed_names(seed: FeatureSeed | None) -> set[str]:
    """Names of seed entries with temperature 0.0 — never sent to the LLM at all."""
    if seed is None:
        return set()
    return {e.name for e in seed.numeric if e.temperature <= 0.0} | {
        e.name for e in seed.categorical if e.temperature <= 0.0
    }


def _seed_context(seed: FeatureSeed | None, exclude_names: set[str] = frozenset()) -> list[dict]:
    """Flexible (temperature > 0.0) seed entries, given to the LLM as fixed priors."""
    if seed is None:
        return []
    entries: list[NumericFeatureSeed | CategoricalFeatureSeed] = [
        e for e in list(seed.numeric) + list(seed.categorical)
        if e.temperature > 0.0 and e.name not in exclude_names
    ]
    return [
        {
            "name": e.name,
            "temperature": e.temperature,
            "proposed_description": e.description,
            "proposed_approved": e.approved,
            "proposed_grouping": getattr(e, "grouping", None),
            "actuary_note": e.actuary_note,
        }
        for e in entries
    ]


def _strip_locked(proposal: FeatureProposal, locked: set[str]) -> FeatureProposal:
    """Copy of `proposal` with locked entries removed, for sending to the LLM."""
    if not locked:
        return proposal
    return proposal.model_copy(update={
        "numeric": [f for f in proposal.numeric if f.name not in locked],
        "categorical": [f for f in proposal.categorical if f.name not in locked],
    })


def _flatten_demoted(
    proposal: FeatureProposal,
) -> tuple[FeatureProposal, dict[str, NumericFeatureConfig | CategoricalFeatureConfig]]:
    """Represent every actuary-unchecked (`approved is False`) numeric/categorical
    feature as a bare `excluded` entry for the prompt, matching the shape the
    refinement prompt expects ("excluded" = not currently included) — while
    keeping the real object aside (comment_history and all) to restore verbatim
    after the LLM responds, via `_restore_demoted`.

    `core.feature_pipeline.reconcile_membership` deliberately keeps a demoted
    feature's full config object in place (with `approved=False`) rather than
    collapsing it to a bare name, so its comment_history survives for the
    actuary to see. But the refinement prompt is told the "numeric"/"categorical"/
    "excluded" shape it's sent must be echoed back unchanged, and was never
    designed to see an `approved: false` entry sitting inside "numeric"/
    "categorical" — trusting a non-deterministic model to route around that
    correctly on every round is exactly the kind of prompt-only enforcement this
    codebase avoids elsewhere (see "Minimal-diff refinement" in CLAUDE.md). Never
    actually sending the object through the round-trip removes the risk instead.
    """
    demoted: dict[str, NumericFeatureConfig | CategoricalFeatureConfig] = {}
    kept_numeric: list[NumericFeatureConfig] = []
    kept_categorical: list[CategoricalFeatureConfig] = []
    excluded = list(proposal.excluded)
    exclusion_rationale = dict(proposal.exclusion_rationale)
    excluded_description = dict(proposal.excluded_description)

    def _demote(feat: NumericFeatureConfig | CategoricalFeatureConfig) -> None:
        demoted[feat.name] = feat
        excluded.append(feat.name)
        excluded_description[feat.name] = feat.description
        latest = feat.comment_history[-1].text if feat.comment_history else None
        exclusion_rationale[feat.name] = latest or "Actuary excluded this round."

    for feat in proposal.numeric:
        if feat.approved is not False:
            kept_numeric.append(feat)
        else:
            _demote(feat)
    for feat in proposal.categorical:
        if feat.approved is not False:
            kept_categorical.append(feat)
        else:
            _demote(feat)

    flattened = proposal.model_copy(update={
        "numeric": kept_numeric, "categorical": kept_categorical,
        "excluded": excluded, "exclusion_rationale": exclusion_rationale,
        "excluded_description": excluded_description,
    })
    return flattened, demoted


def _restore_demoted(
    proposal: FeatureProposal,
    demoted: dict[str, NumericFeatureConfig | CategoricalFeatureConfig],
    remarked: set[str],
) -> FeatureProposal:
    """Undo `_flatten_demoted`: re-insert each demoted feature's real object back
    into "numeric"/"categorical" (comment_history intact, still `approved=False`),
    removing the bare placeholder the LLM echoed for it in "excluded". A reply
    the LLM wrote for a feature actually remarked on this round lands in
    `exclusion_rationale` (the field an "excluded" entry has instead of
    `actuary_note`) — folded back onto `actuary_note` here so `refine_draft`'s
    existing fold-into-history logic treats it identically to a reply for any
    other feature. An *unremarked* demoted feature's `exclusion_rationale` is
    just the actuary's own prior comment carried forward unchanged (per the
    refinement prompt's instruction) — never fold that back into `actuary_note`,
    or it would re-append the actuary's own words to their history mislabeled as
    a Claude reply.
    """
    for name, feat in demoted.items():
        reply = proposal.exclusion_rationale.pop(name, None)
        proposal.excluded_description.pop(name, None)
        if name in proposal.excluded:
            proposal.excluded.remove(name)
        if name in remarked and reply:
            feat.actuary_note = reply
        if isinstance(feat, NumericFeatureConfig):
            proposal.numeric.append(feat)
        else:
            proposal.categorical.append(feat)
    return proposal


def _proposal_dict_for_prompt(proposal: FeatureProposal) -> dict:
    """Dump `proposal` for the prompt with `comment_history` omitted entirely —
    not just emptied to `[]`. It only needs this round's `actuary_remarks_json`,
    not the full accumulated feed (the refine prompt's payload size is already a
    latency concern on longer sessions) — and critically, even an *empty*
    `comment_history: []` key on every variable was enough for the model to try
    "helpfully" filling it in with a hallucinated flat string once it had an
    actuary remark to record, which fails validation (it expects structured
    CommentEntry objects, not a string). Omitting the key removes the temptation;
    `NumericFeatureConfig`/`CategoricalFeatureConfig` also discard any malformed
    `comment_history` a response still includes, as a code-level backstop."""
    return proposal.model_dump(exclude={
        "numeric": {"__all__": {"comment_history"}},
        "categorical": {"__all__": {"comment_history"}},
    })


def _merge_locked_entries(
    proposal: FeatureProposal, seed: FeatureSeed | None, already_handled: set[str] = frozenset(),
) -> FeatureProposal:
    """Append locked seed entries the LLM never saw back onto its response.

    `already_handled` names (typically ones with an actuary remark this round) were
    deliberately sent to the LLM, so its returned value for them is trusted as-is.
    """
    if seed is None:
        return proposal
    present = {f.name for f in proposal.numeric} | {f.name for f in proposal.categorical}

    for entry in seed.numeric:
        if entry.temperature > 0.0 or entry.name in already_handled or entry.name in present:
            continue
        proposal.numeric.append(NumericFeatureConfig(**_drop_seed_fields(entry)))

    for entry in seed.categorical:
        if entry.temperature > 0.0 or entry.name in already_handled or entry.name in present:
            continue
        proposal.categorical.append(CategoricalFeatureConfig(**_drop_seed_fields(entry)))

    return proposal


def _drop_seed_fields(entry: NumericFeatureSeed | CategoricalFeatureSeed) -> dict:
    return {k: v for k, v in entry.model_dump().items() if k not in ("temperature", "updated_at")}
