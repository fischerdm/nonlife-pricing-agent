"""Minimal-diff refinement — one shared primitive used by every refine
pipeline (`core/feature_pipeline.py::refine_draft`,
`core/distillation_pipeline.py::refine_glm_draft`).

Every refine call sends the *entire* previous proposal to the LLM and gets
back a complete, freshly-generated one in return, regardless of how narrow
the actuary's remark was. At a non-zero temperature (0.2 throughout this
project), nothing guarantees that regeneration reproduces an untouched
term's or feature's own descriptive content byte-for-byte between rounds.
Left unchecked, an actuary who only remarked on one item could see
everything else quietly reword itself round to round — corrosive
specifically because this review workflow depends on the actuary re-reading
largely-the-same content across many iterations to spot what's actually new.

See CLAUDE.md's "Minimal-diff refinement" key design decision for the full
rationale, and why a prompt instruction alone (both refine prompts already
say some version of "leave everything else exactly as given") was not
considered sufficient — same doctrine as every other structural-guarantee
decision in this codebase.
"""


def pin_unremarked_fields(
    items: list,
    prev_by_name: dict,
    remarked: set[str],
    fields: tuple[str, ...],
) -> list[str]:
    """For every item in `items` that both existed before (`prev_by_name`)
    and wasn't named in `remarked` this round, overwrite each of `fields`
    with its exact value from the previous round — discarding whatever this
    round's LLM response returned for it, no matter how small the
    difference. `fields` not present on a given item's type (e.g. `ordinal`
    on a purely-numeric feature) are silently skipped.

    Two things are deliberately exempt from pinning: an item the actuary
    *did* remark on this round (revising it is the entire point of asking),
    and a brand-new item absent from `prev_by_name` (a newly-promoted
    feature, an interaction added via a general remark, a newly-ranked
    pair) — there's nothing to pin it to, so it keeps the LLM's fresh
    content as-is. `approved`/`term_type`/`grouping`-style placement fields
    are out of scope here; they have their own narrower structural
    guarantees elsewhere (`reconcile_terms`, `reconcile_membership`,
    `reconcile_feature_membership`) and are not touched by this function.

    Mutates `items` in place. Returns the names of items actually pinned
    (i.e. that existed before and had at least one field forced back) —
    empty if nothing needed pinning this round.
    """
    pinned: list[str] = []
    for item in items:
        prev = prev_by_name.get(item.name)
        if prev is None or item.name in remarked:
            continue
        pinned.append(item.name)
        for field in fields:
            if hasattr(item, field):
                setattr(item, field, getattr(prev, field))
    return pinned
