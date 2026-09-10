from core.overrides import force_field_from_authority, pin_unchanged_fields, ratchet_exclude
from core.schemas import CommentEntry, GLMTerm, NumericFeatureConfig


# ── pin_unchanged_fields ─────────────────────────────────────────────────────────

def test_pins_fields_for_unchanged_existing_item():
    prev = NumericFeatureConfig(name="vehicle_age", description="original", data_quality_note="note")
    updated = [NumericFeatureConfig(name="vehicle_age", description="reworded", data_quality_note="reworded note")]

    pinned = pin_unchanged_fields(
        updated, {"vehicle_age": prev}, changed=set(), fields=("description", "data_quality_note"),
    )

    assert updated[0].description == "original"
    assert updated[0].data_quality_note == "note"
    assert pinned == ["vehicle_age"]


def test_leaves_changed_item_untouched():
    prev = NumericFeatureConfig(name="vehicle_age", description="original")
    updated = [NumericFeatureConfig(name="vehicle_age", description="revised per remark")]

    pinned = pin_unchanged_fields(
        updated, {"vehicle_age": prev}, changed={"vehicle_age"}, fields=("description",),
    )

    assert updated[0].description == "revised per remark"
    assert pinned == []


def test_leaves_brand_new_item_untouched():
    updated = [NumericFeatureConfig(name="new_feature", description="fresh content")]

    pinned = pin_unchanged_fields(updated, prev_by_name={}, changed=set(), fields=("description",))

    assert updated[0].description == "fresh content"
    assert pinned == []


def test_skips_fields_not_present_on_the_item_type():
    """GLMTerm has no `description`/`ordinal` fields — pinning those must not error."""
    prev = GLMTerm(name="driver_age", term_type="main", rationale="original")
    updated = [GLMTerm(name="driver_age", term_type="main", rationale="reworded")]

    pinned = pin_unchanged_fields(
        updated, {"driver_age": prev}, changed=set(), fields=("rationale", "description", "ordinal"),
    )

    assert updated[0].rationale == "original"
    assert pinned == ["driver_age"]


def test_mutates_items_in_place_and_returns_nothing_extra():
    prev = GLMTerm(name="driver_age", term_type="main", rationale="original", h_statistic=0.1)
    item = GLMTerm(name="driver_age", term_type="main", rationale="reworded", h_statistic=0.9)

    pin_unchanged_fields([item], {"driver_age": prev}, changed=set(), fields=("rationale", "h_statistic"))

    assert item.rationale == "original"
    assert item.h_statistic == 0.1


# ── force_field_from_authority ───────────────────────────────────────────────────

def test_force_field_sets_from_authority():
    items = [NumericFeatureConfig(name="vehicle_age", description="d", approved=None)]

    force_field_from_authority(items, {"vehicle_age": True}, "approved", default=False)

    assert items[0].approved is True


def test_force_field_defaults_when_absent_from_authority():
    items = [NumericFeatureConfig(name="vehicle_age", description="d", approved=True)]

    force_field_from_authority(items, {}, "approved", default=False)

    assert items[0].approved is False


def test_force_field_only_if_present_leaves_unreviewed_item_alone():
    items = [GLMTerm(name="driver_age", term_type="main", rationale="r", approved=True)]

    force_field_from_authority(items, {}, "approved", only_if_present=True)

    assert items[0].approved is True


def test_force_field_only_if_present_still_overrides_when_named():
    items = [GLMTerm(name="driver_age", term_type="main", rationale="r", approved=True)]

    force_field_from_authority(items, {"driver_age": False}, "approved", only_if_present=True)

    assert items[0].approved is False


# ── ratchet_exclude ───────────────────────────────────────────────────────────────

def _term(name: str, approved: bool | None) -> GLMTerm:
    return GLMTerm(name=name, term_type="main", rationale="r", approved=approved)


def _reason_if_named(bad_name: str):
    def invalid_reason(term):
        return f"{term.name} is no longer valid." if term.name == bad_name else None
    return invalid_reason


def _append_note_dedup(term, reason: str) -> None:
    if not term.comment_history or term.comment_history[-1].text != reason:
        term.comment_history.append(CommentEntry(author="agent", text=reason, ts="2026-09-10T00:00:00Z"))


def test_ratchet_excludes_newly_invalid_item():
    terms = [_term("vehicle_age", True)]

    excluded = ratchet_exclude(terms, _reason_if_named("vehicle_age"), _append_note_dedup)

    assert terms[0].approved is False
    assert excluded == ["vehicle_age"]
    assert terms[0].comment_history[0].text == "vehicle_age is no longer valid."


def test_ratchet_leaves_valid_items_alone():
    terms = [_term("driver_age", True)]

    excluded = ratchet_exclude(terms, _reason_if_named("vehicle_age"), _append_note_dedup)

    assert terms[0].approved is True
    assert excluded == []
    assert terms[0].comment_history == []


def test_ratchet_never_restores_an_excluded_item():
    """Only ever forces off; a later call with the item now valid must not flip it back on."""
    terms = [_term("vehicle_age", True)]
    ratchet_exclude(terms, _reason_if_named("vehicle_age"), _append_note_dedup)

    excluded_again = ratchet_exclude(terms, _reason_if_named("something_else"), _append_note_dedup)

    assert terms[0].approved is False
    assert excluded_again == []


def test_ratchet_does_not_reflag_an_already_excluded_item():
    terms = [_term("vehicle_age", True)]
    ratchet_exclude(terms, _reason_if_named("vehicle_age"), _append_note_dedup)

    excluded_again = ratchet_exclude(terms, _reason_if_named("vehicle_age"), _append_note_dedup)

    assert excluded_again == []
    assert len(terms[0].comment_history) == 1
