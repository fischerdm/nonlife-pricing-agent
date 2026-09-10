"""Shared GLM distillation draft-generation logic, used by both the Orchestrator
and the dashboard's GLM Distillation Workbench — mirrors core/feature_pipeline.py.
"""
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agents.distillation_agent import DistillationAgent
from core.llm_client import LLMClient
from core.overrides import force_field_from_authority, pin_unchanged_fields, ratchet_exclude
from core.schemas import CommentEntry, DistillationSeed, GLMProposal, GLMTerm

_SNAPSHOT_KINDS = ("initial", "modified", "finalized")


def generate_glm_draft(
    llm: LLMClient,
    h_stat_interactions: list[dict],
    approved_features: list[str],
    data_cfg: dict,
    seed: DistillationSeed | None = None,
) -> GLMProposal:
    agent = DistillationAgent(llm, lob=data_cfg.get("lob", "motor"))
    return agent.propose(
        h_stat_interactions=h_stat_interactions,
        approved_features=approved_features,
        objective=data_cfg["objective"],
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
        seed=seed,
    )


def refine_glm_draft(
    llm: LLMClient,
    previous: GLMProposal,
    remarks: dict[str, str],
    data_cfg: dict,
    seed: DistillationSeed | None = None,
    general_remark: str | None = None,
) -> GLMProposal:
    """Refine a draft with actuary remarks, then pin unremarked content and
    reconcile comment history.

    Comment history is code-owned, not the LLM's: carry it forward per term (the
    agent is never asked to echo it back), mark whatever was just sent as sent,
    append the agent's transient reply for this round (if any) as a new entry,
    then clear the transient field — mirrors
    `core.feature_pipeline.refine_draft`'s tail block exactly.
    """
    agent = DistillationAgent(llm, lob=data_cfg.get("lob", "motor"))
    updated = agent.refine(
        previous_proposal=previous,
        actuary_remarks=remarks,
        objective=data_cfg["objective"],
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
        seed=seed,
        general_remark=general_remark,
    )

    prev_by_name = {t.name: t for t in previous.terms}

    # Minimal-diff refinement (see CLAUDE.md and core/overrides.py): a term
    # the actuary didn't remark on this round keeps its previous rationale/
    # h_statistic pinned exactly, regardless of what this round's LLM call
    # returned for it. `approved`/`term_type` have their own narrower
    # structural guarantees (`reconcile_terms`) and aren't touched here.
    pin_unchanged_fields(
        updated.terms, prev_by_name, set(remarks), fields=("rationale", "h_statistic"),
    )

    now = datetime.now(timezone.utc).isoformat()
    for term in updated.terms:
        prev_term = prev_by_name.get(term.name)
        history = [e.model_copy() for e in prev_term.comment_history] if prev_term else []
        if term.name in remarks:
            for entry in history:
                entry.sent = True
        if term.actuary_note:
            history.append(CommentEntry(author="agent", text=term.actuary_note, ts=now))
        term.comment_history = history
        term.actuary_note = None

    return updated


def reconcile_terms(
    draft: GLMProposal,
    checkbox_state: dict[str, bool],
    remarked: set[str] = frozenset(),
) -> GLMProposal:
    """Recompute each term's `approved` flag and structural `term_type`, purely
    from the actuary's current checkbox state, independent of what the agent's
    refine response carries — same defense-in-depth pattern as
    `feature_pipeline.reconcile_membership`. Call this before any agent refine
    call (so its previous_proposal_json reflects true state) and again after.

    `term_type` is derived from the term's own name (":"-joined => interaction,
    else main) except "polynomial", which only survives for a term the actuary
    explicitly remarked on this round — the same re-specification request that
    would have prompted it; an agent can't reclassify a term unprompted.
    `approved` is always the actuary's checkbox value for any term the checkbox
    state covers. A term the agent proposed this round that the actuary hasn't
    seen yet (not in `checkbox_state`) is left as returned — that's the point of
    a refine call, to propose something new for the *next* review pass. The
    `approved`-forcing itself is `core.overrides.force_field_from_authority`.
    """
    for term in draft.terms:
        structural_type = "interaction" if ":" in term.name else "main"
        if not (term.term_type == "polynomial" and term.name in remarked):
            term.term_type = structural_type
    force_field_from_authority(draft.terms, checkbox_state, "approved", only_if_present=True)
    return draft


def reconcile_feature_membership(draft: GLMProposal, approved_features: list[str]) -> list[str]:
    """Force `approved=False` for any term whose constituent feature(s) —
    split on ':' the same way an interaction's name is joined — are no
    longer in the *currently* approved feature set. A correctness guarantee,
    not just a staleness nudge: a term referencing a feature that's since
    been de-approved at Feature Selection must be structurally impossible to
    fit, not merely discouraged. Same defense-in-depth pattern as
    `reconcile_terms` — call it right after, both before and after any agent
    refine call, so nothing (agent output or a re-checked box) can slip an
    orphaned term back to `approved=True`.

    Mutates `draft` in place (like `reconcile_terms`) and returns the names
    of terms it just flipped to excluded — empty if nothing changed. Only
    ever *forces off*, never forces on: a feature that was removed and later
    re-approved simply stops being "missing" on the next call, and its old
    term (however the actuary or a prior call last left `approved`) is free
    to be re-checked normally again — no separate handling needed for the
    delete-then-re-add case.

    Only flags a term the moment it actually transitions from
    approved-or-undecided to excluded here — a term already sitting at
    `approved=False` from a previous call is left alone and not re-reported,
    so Re-open, Update, and Finalize don't each surface the same exclusion
    as if it were new. The explanatory `comment_history` entry is likewise
    skipped if the term's last entry already says the same thing, so an
    actuary repeatedly re-checking a still-orphaned term's box doesn't build
    up a wall of identical notes — just the one already there confirming why
    it won't stick.

    The transition/one-way-lock mechanics are `core.overrides.ratchet_exclude`;
    this function only supplies the domain-specific validity check (a term's
    constituent features, split on ":") and how to record the note.
    """
    approved_set = set(approved_features)

    def invalid_reason(term) -> str | None:
        missing = [f for f in term.name.split(":") if f not in approved_set]
        if not missing:
            return None
        return (
            f"Automatically excluded: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} no longer an approved feature."
        )

    def append_note(term, reason: str) -> None:
        if not term.comment_history or term.comment_history[-1].text != reason:
            term.comment_history.append(
                CommentEntry(author="agent", text=reason, ts=datetime.now(timezone.utc).isoformat())
            )

    return ratchet_exclude(draft.terms, invalid_reason, append_note)


def add_manual_interaction(
    draft: GLMProposal, feature_a: str, feature_b: str, rationale: str = "",
) -> GLMProposal:
    """Actuary-authored interaction term, added directly with no LLM round-trip —
    mirrors how unchecking a feature in the Feature Workbench costs zero LLM calls.

    Raises ValueError for a self-interaction or one already on the draft
    (order-insensitive: "a:b" and "b:a" name the same term to patsy).
    """
    if feature_a == feature_b:
        raise ValueError("An interaction needs two distinct features.")
    name = f"{feature_a}:{feature_b}"
    existing = {t.name for t in draft.terms}
    if name in existing or f"{feature_b}:{feature_a}" in existing:
        raise ValueError(f"An interaction between {feature_a} and {feature_b} is already on the draft.")
    draft.terms.append(GLMTerm(
        name=name, term_type="interaction",
        rationale=rationale or "Actuary-added interaction.", approved=True,
    ))
    return draft


def add_manual_main_effect(draft: GLMProposal, feature_name: str, rationale: str = "") -> GLMProposal:
    """Actuary-promoted main effect, added directly with no LLM round-trip — same
    reasoning as `add_manual_interaction`. Used by the "Not Proposed" tab, for an
    approved feature the agent never proposed a main effect for.

    Raises ValueError if a term with this name already exists on the draft.
    """
    if feature_name in {t.name for t in draft.terms}:
        raise ValueError(f"A term named {feature_name} is already on the draft.")
    draft.terms.append(GLMTerm(
        name=feature_name, term_type="main",
        rationale=rationale or "Actuary-added main effect.", approved=True,
    ))
    return draft


# ── Draft snapshots (disk-backed, never overwritten, three kinds) ───────────────
#
# Same convention as core/feature_pipeline.py, sharing DRAFTS_DIR's kind
# subfolders but with a distinct "glm_draft_" filename prefix so the two
# workbenches' snapshots never collide or get mixed up in either's picker.

def save_glm_draft_snapshot(proposal: GLMProposal, kind: str, drafts_dir: Path) -> Path:
    """See `core.feature_pipeline.save_draft_snapshot` — same `drafts_dir`
    convention (they share the same run-scoped directory's kind subfolders,
    distinguished only by the `glm_draft_` filename prefix below)."""
    assert kind in _SNAPSHOT_KINDS, f"unknown snapshot kind: {kind!r}"
    kind_dir = drafts_dir / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = kind_dir / f"glm_draft_{timestamp}.yaml"
    with open(path, "w") as f:
        yaml.dump(proposal.model_dump(), f, allow_unicode=True, sort_keys=False)
    print(f"GLM draft snapshot ({kind}) saved to {path}")
    return path


def list_glm_draft_snapshots(kind: str, drafts_dir: Path) -> list[Path]:
    """All saved snapshots of one kind under `drafts_dir`, newest first."""
    assert kind in _SNAPSHOT_KINDS, f"unknown snapshot kind: {kind!r}"
    kind_dir = drafts_dir / kind
    if not kind_dir.exists():
        return []
    return sorted(kind_dir.glob("glm_draft_*.yaml"), reverse=True)


def load_glm_draft_snapshot(path: Path) -> GLMProposal:
    """Load one specific snapshot file (from `list_glm_draft_snapshots`)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return GLMProposal(**data)
