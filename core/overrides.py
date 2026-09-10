"""Agent-output overrides — one place for every function whose job is to make
sure a field the LLM regenerates every round can never be the field's real
source of truth.

Every refine call in this codebase (`FeatureSelectionAgent.refine`,
`DistillationAgent.refine`, `GroupingAgent.refine`) sends the entire previous
proposal to the LLM and gets back a complete, freshly-generated one, not a
diff. Nothing about that call guarantees an untouched item's content stays
byte-for-byte the same, or that a flag the actuary controls (approved,
placement) comes back the way the actuary last set it. This module is the
three structural mechanisms this codebase uses to enforce that anyway,
without relying on the prompt alone:

- `pin_unchanged_fields` — a field stays exactly as it was unless the round's
  authoritative "this changed" signal names the item. Resistance to
  unprompted drift.
- `force_field_from_authority` — a field is re-set from an external,
  human-owned source on *every* call, whether or not anything changed.
  There's nothing to resist here; the agent never had a say in the first
  place.
- `ratchet_exclude` — an item is forced out the moment it becomes invalid,
  with a note explaining why, and — the ratchet part — never silently
  re-included by a later call even if it becomes valid again. Restoring it
  is always a deliberate, separate action.

These are deliberately a different species from the more common "guardrail"
meaning (Guardrails AI, NeMo Guardrails, moderation/schema validation): those
check one output against a fixed rule, stateless with respect to prior
rounds. Everything here is stateful across rounds — it compares this
round's output against what came before, or against a live external source,
and overrides accordingly. Don't reach for this module to validate a single
response in isolation; that's a different job.

Callers own their own domain shape (what a "field" or an "item" means, how
an item's validity is computed, how a note gets recorded) — these functions
only know about `.name` and whatever attributes/dict keys they're told to
touch, so the same three primitives work for a dataset feature, a GLM term,
or anything else with a stable name across rounds.

See CLAUDE.md's "Agent-output overrides" key design decision for the full
rationale and the concrete callers (`core/feature_pipeline.py`,
`core/distillation_pipeline.py`).
"""

from typing import Any, Callable, Optional


def pin_unchanged_fields(
    items: list,
    prev_by_name: dict,
    changed: set[str],
    fields: tuple[str, ...],
) -> list[str]:
    """For every item in `items` that both existed before (`prev_by_name`)
    and isn't named in `changed` this round, overwrite each of `fields` with
    its exact value from the previous round — discarding whatever this
    round's response returned for it, no matter how small the difference.
    `fields` not present on a given item's type (e.g. `ordinal` on a
    purely-numeric feature) are silently skipped.

    Two things are deliberately exempt from pinning: an item named in
    `changed` this round (revising it is the entire point of naming it), and
    a brand-new item absent from `prev_by_name` — there's nothing to pin it
    to, so it keeps this round's content as-is. Placement/approval-style
    fields are out of scope here; see `force_field_from_authority` and
    `ratchet_exclude` for those.

    Mutates `items` in place. Returns the names of items actually pinned —
    empty if nothing needed pinning this round.
    """
    pinned: list[str] = []
    for item in items:
        prev = prev_by_name.get(item.name)
        if prev is None or item.name in changed:
            continue
        pinned.append(item.name)
        for field in fields:
            if hasattr(item, field):
                setattr(item, field, getattr(prev, field))
    return pinned


def force_field_from_authority(
    items: list,
    authority: dict[str, Any],
    field: str,
    *,
    only_if_present: bool = False,
    default: Any = None,
) -> None:
    """For every item in `items`, set `field` from `authority[item.name]` —
    called on *every* round, not just when something changed, because the
    agent never owns this field to begin with.

    An item whose name isn't a key in `authority` gets `default`, unless
    `only_if_present` is True, in which case it's left exactly as it arrived
    (the item is something the authority hasn't reviewed yet — e.g. a term
    the agent proposed this round that the actuary hasn't seen, so there's
    no verdict to force).

    Mutates `items` in place.
    """
    for item in items:
        if item.name in authority:
            setattr(item, field, authority[item.name])
        elif not only_if_present:
            setattr(item, field, default)


def ratchet_exclude(
    items: list,
    invalid_reason: Callable[[Any], Optional[str]],
    append_note: Callable[[Any, str], None],
    *,
    approved_field: str = "approved",
) -> list[str]:
    """For every item, call `invalid_reason(item)` — a human-readable reason
    string if the item is no longer valid, `None` if it's fine. An item that
    just became invalid gets `approved_field` forced to `False` and
    `append_note(item, reason)` called so the caller can record why (in
    whatever note/history shape it uses — this function doesn't assume one).

    The ratchet: only ever forces `approved_field` *off*, never back on, and
    only on the actual transition — an item already sitting at
    `approved_field is False` is left alone and not re-reported, so calling
    this repeatedly (e.g. once before and once after an agent call) doesn't
    re-flag the same item or spam its notes. An item that becomes valid
    again later simply stops being reported here; nothing about this
    function restores it, that's always a separate, deliberate action by
    the caller.

    Mutates `items` in place. Returns the names of items that just
    transitioned to excluded this call — empty if nothing changed.
    """
    just_excluded: list[str] = []
    for item in items:
        reason = invalid_reason(item)
        if reason is None:
            continue
        if getattr(item, approved_field) is False:
            continue
        setattr(item, approved_field, False)
        just_excluded.append(item.name)
        append_note(item, reason)
    return just_excluded
