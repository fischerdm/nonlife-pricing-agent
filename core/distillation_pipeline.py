"""Shared GLM distillation draft-generation logic, used by both the Orchestrator
and the dashboard's GLM Distillation Workbench — mirrors core/feature_pipeline.py.
"""
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agents.distillation_agent import DistillationAgent
from core.llm_client import LLMClient
from core.schemas import CommentEntry, DistillationSeed, GLMProposal, GLMTerm

DRAFTS_DIR = Path("reports/drafts")
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
    """Refine a draft with actuary remarks, then reconcile comment history.

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
    a refine call, to propose something new for the *next* review pass.
    """
    for term in draft.terms:
        structural_type = "interaction" if ":" in term.name else "main"
        if not (term.term_type == "polynomial" and term.name in remarked):
            term.term_type = structural_type
        if term.name in checkbox_state:
            term.approved = checkbox_state[term.name]
    return draft


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


# ── Draft snapshots (disk-backed, never overwritten, three kinds) ───────────────
#
# Same convention as core/feature_pipeline.py, sharing DRAFTS_DIR's kind
# subfolders but with a distinct "glm_draft_" filename prefix so the two
# workbenches' snapshots never collide or get mixed up in either's picker.

def save_glm_draft_snapshot(proposal: GLMProposal, kind: str) -> Path:
    assert kind in _SNAPSHOT_KINDS, f"unknown snapshot kind: {kind!r}"
    kind_dir = DRAFTS_DIR / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = kind_dir / f"glm_draft_{timestamp}.yaml"
    with open(path, "w") as f:
        yaml.dump(proposal.model_dump(), f, allow_unicode=True, sort_keys=False)
    print(f"GLM draft snapshot ({kind}) saved to {path}")
    return path


def list_glm_draft_snapshots(kind: str) -> list[Path]:
    """All saved snapshots of one kind, newest first."""
    assert kind in _SNAPSHOT_KINDS, f"unknown snapshot kind: {kind!r}"
    kind_dir = DRAFTS_DIR / kind
    if not kind_dir.exists():
        return []
    return sorted(kind_dir.glob("glm_draft_*.yaml"), reverse=True)


def load_glm_draft_snapshot(path: Path) -> GLMProposal:
    """Load one specific snapshot file (from `list_glm_draft_snapshots`)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return GLMProposal(**data)
