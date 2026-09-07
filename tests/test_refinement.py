from core.refinement import pin_unremarked_fields
from core.schemas import GLMTerm, NumericFeatureConfig


def test_pins_fields_for_unremarked_existing_item():
    prev = NumericFeatureConfig(name="vehicle_age", description="original", data_quality_note="note")
    updated = [NumericFeatureConfig(name="vehicle_age", description="reworded", data_quality_note="reworded note")]

    pinned = pin_unremarked_fields(
        updated, {"vehicle_age": prev}, remarked=set(), fields=("description", "data_quality_note"),
    )

    assert updated[0].description == "original"
    assert updated[0].data_quality_note == "note"
    assert pinned == ["vehicle_age"]


def test_leaves_remarked_item_untouched():
    prev = NumericFeatureConfig(name="vehicle_age", description="original")
    updated = [NumericFeatureConfig(name="vehicle_age", description="revised per remark")]

    pinned = pin_unremarked_fields(
        updated, {"vehicle_age": prev}, remarked={"vehicle_age"}, fields=("description",),
    )

    assert updated[0].description == "revised per remark"
    assert pinned == []


def test_leaves_brand_new_item_untouched():
    updated = [NumericFeatureConfig(name="new_feature", description="fresh content")]

    pinned = pin_unremarked_fields(updated, prev_by_name={}, remarked=set(), fields=("description",))

    assert updated[0].description == "fresh content"
    assert pinned == []


def test_skips_fields_not_present_on_the_item_type():
    """GLMTerm has no `description`/`ordinal` fields — pinning those must not error."""
    prev = GLMTerm(name="driver_age", term_type="main", rationale="original")
    updated = [GLMTerm(name="driver_age", term_type="main", rationale="reworded")]

    pinned = pin_unremarked_fields(
        updated, {"driver_age": prev}, remarked=set(), fields=("rationale", "description", "ordinal"),
    )

    assert updated[0].rationale == "original"
    assert pinned == ["driver_age"]


def test_mutates_items_in_place_and_returns_nothing_extra():
    prev = GLMTerm(name="driver_age", term_type="main", rationale="original", h_statistic=0.1)
    item = GLMTerm(name="driver_age", term_type="main", rationale="reworded", h_statistic=0.9)

    pin_unremarked_fields([item], {"driver_age": prev}, remarked=set(), fields=("rationale", "h_statistic"))

    assert item.rationale == "original"
    assert item.h_statistic == 0.1
