import pytest

from reconcile_tiers._core.ids import make_locator_id
from reconcile_tiers._core.lineage import STEP_EXTRACT_WALLS, record


def test_make_locator_id_uses_tier_scope_and_colon_joined_parts():
    locator = make_locator_id("abc-123", "wall", 0, 3)

    assert locator == "abc-123::tier-wall::0:3"
    assert make_locator_id("abc-123", "tier-wall", 0, 3) == locator


@pytest.mark.parametrize(
    "args",
    [
        ("", "wall", 0),
        ("abc-123", "", 0),
        ("abc-123", "wall"),
        ("abc-123", "wall", ""),
        ("abc-123", "wall", "bad::part"),
    ],
)
def test_make_locator_id_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        make_locator_id(*args)


def test_record_returns_new_lineage_tuple_without_mutating_input():
    lineage = ({"step": "previous", "action": "kept"},)

    updated = record(lineage, STEP_EXTRACT_WALLS, "created", "from scan-cache")

    assert lineage == ({"step": "previous", "action": "kept"},)
    assert updated == (
        {"step": "previous", "action": "kept"},
        {"step": STEP_EXTRACT_WALLS, "action": "created", "detail": "from scan-cache"},
    )


def test_record_accepts_empty_lineage_and_omits_blank_detail():
    assert record(None, STEP_EXTRACT_WALLS, "created") == (
        {"step": STEP_EXTRACT_WALLS, "action": "created"},
    )
