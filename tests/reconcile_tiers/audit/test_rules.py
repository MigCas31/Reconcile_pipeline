"""Unit tests for soft-defect rules in reconcile_tiers/audit/rules.py.

Each test builds a minimal payload-shaped dict that exercises one rule's gate.
"""

from __future__ import annotations

from reconcile_tiers.audit.rules import (
    rule_ceiling_yspan_excessive,
    rule_story_ceiling_overcount,
    rule_wall_cutout_outside_outline,
)


def _payload_with_one_wall(
    *, wall_y: tuple[float, float], cutout_y: tuple[float, float]
):
    wy_lo, wy_hi = wall_y
    cy_lo, cy_hi = cutout_y
    return {
        "rooms": [
            {
                "story": 0,
                "walls": [
                    {
                        "locator_id": "uuid::tier-wall::0:WALL/0",
                        "corners": [
                            {"x": 0.0, "y": wy_lo, "z": 0.0},
                            {"x": 0.0, "y": wy_hi, "z": 0.0},
                            {"x": 0.0, "y": wy_hi, "z": 4.0},
                            {"x": 0.0, "y": wy_lo, "z": 4.0},
                        ],
                        "cutouts": [
                            {
                                "corners": [
                                    {"x": 0.0, "y": cy_lo, "z": 1.0},
                                    {"x": 0.0, "y": cy_hi, "z": 1.0},
                                    {"x": 0.0, "y": cy_hi, "z": 2.0},
                                    {"x": 0.0, "y": cy_lo, "z": 2.0},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_audit_rule_passes_when_cutout_inside_wall():
    payload = _payload_with_one_wall(wall_y=(0.0, 3.0), cutout_y=(1.0, 2.0))
    assert rule_wall_cutout_outside_outline(payload) == []


def test_audit_rule_flags_cutout_above_wall_top():
    payload = _payload_with_one_wall(wall_y=(0.0, 1.5), cutout_y=(0.5, 2.0))
    flags = rule_wall_cutout_outside_outline(payload)
    assert len(flags) == 1
    assert flags[0]["rule"] == "wall_cutout_outside_outline"
    assert flags[0]["evidence"]["excess_m"] > 0.4
    assert flags[0]["evidence"]["wall_y_range"] == [0.0, 1.5]
    assert flags[0]["evidence"]["cutout_y_range"] == [0.5, 2.0]


def test_audit_rule_flags_cutout_below_wall_bottom():
    payload = _payload_with_one_wall(wall_y=(1.0, 3.0), cutout_y=(0.5, 2.0))
    flags = rule_wall_cutout_outside_outline(payload)
    assert len(flags) == 1
    assert flags[0]["evidence"]["excess_m"] > 0.4


def test_audit_rule_tolerates_1mm_floating_point_drift():
    payload = _payload_with_one_wall(wall_y=(0.0, 2.0), cutout_y=(-1e-4, 2.0 + 1e-4))
    assert rule_wall_cutout_outside_outline(payload) == []


# ---- ceiling_yspan_excessive -------------------------------------------------


def _ceiling(yspan: float, *, locator: str = "uuid::tier-ceiling-flat::0"):
    """A unit square ceiling whose corners span `yspan` vertically."""
    return {
        "locator_id": locator,
        "source": "computed_oblique",
        "role": "ceiling",
        "corners": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": yspan, "z": 0.0},
            {"x": 1.0, "y": yspan, "z": 1.0},
            {"x": 0.0, "y": 0.0, "z": 1.0},
        ],
    }


def test_ceiling_yspan_passes_when_under_threshold():
    payload = {"ceiling": [_ceiling(0.5), _ceiling(1.9)]}
    assert rule_ceiling_yspan_excessive(payload) == []


def test_ceiling_yspan_low_severity_just_over_threshold():
    payload = {"ceiling": [_ceiling(2.5)]}
    flags = rule_ceiling_yspan_excessive(payload)
    assert len(flags) == 1
    assert flags[0]["rule"] == "ceiling_yspan_excessive"
    assert flags[0]["severity"] == "low"
    assert flags[0]["evidence"]["y_span_m"] == 2.5
    assert flags[0]["evidence"]["threshold_m"] == 2.0


def test_ceiling_yspan_medium_severity_at_3m():
    payload = {"ceiling": [_ceiling(4.0)]}
    flags = rule_ceiling_yspan_excessive(payload)
    assert flags[0]["severity"] == "medium"


def test_ceiling_yspan_high_severity_at_5m_plus():
    payload = {"ceiling": [_ceiling(8.0)]}
    flags = rule_ceiling_yspan_excessive(payload)
    assert flags[0]["severity"] == "high"
    assert flags[0]["evidence"]["y_span_m"] == 8.0


def test_ceiling_yspan_skips_pieces_with_no_corners():
    payload = {"ceiling": [{"locator_id": "uuid::tier-ceiling-flat::0", "corners": []}]}
    assert rule_ceiling_yspan_excessive(payload) == []


# ---- story_ceiling_overcount -------------------------------------------------


def _room(*, story: int, room_index: int, y_lo: float, y_hi: float):
    """A 4x4 room polygon at `story`, with floor + walls Y range [y_lo, y_hi]."""
    return {
        "locator_id": f"uuid::tier-room::{room_index}",
        "story": story,
        "floor": [
            {
                "corners": [
                    {"x": 0.0, "y": y_lo, "z": 0.0},
                    {"x": 4.0, "y": y_lo, "z": 0.0},
                    {"x": 4.0, "y": y_lo, "z": 4.0},
                    {"x": 0.0, "y": y_lo, "z": 4.0},
                ]
            }
        ],
        "walls": [
            {
                "corners": [
                    {"x": 0.0, "y": y_lo, "z": 0.0},
                    {"x": 0.0, "y": y_hi, "z": 0.0},
                    {"x": 4.0, "y": y_hi, "z": 0.0},
                    {"x": 4.0, "y": y_lo, "z": 0.0},
                ]
            }
        ],
    }


def _story_ceiling(*, mid_y: float, locator: str):
    return {
        "locator_id": locator,
        "corners": [
            {"x": 0.0, "y": mid_y, "z": 0.0},
            {"x": 1.0, "y": mid_y, "z": 0.0},
            {"x": 1.0, "y": mid_y, "z": 1.0},
            {"x": 0.0, "y": mid_y, "z": 1.0},
        ],
    }


def test_story_overcount_passes_when_ratio_sane():
    # 2 stories, both with 4 rooms / 5 ceilings — ratio 1.25, well under 3.0.
    rooms = [_room(story=0, room_index=i, y_lo=0.0, y_hi=2.5) for i in range(4)] + [
        _room(story=1, room_index=4 + i, y_lo=2.5, y_hi=5.0) for i in range(4)
    ]
    ceilings = [
        _story_ceiling(mid_y=2.4, locator=f"uuid::tier-ceiling-flat::{i}")
        for i in range(5)
    ] + [
        _story_ceiling(mid_y=4.9, locator=f"uuid::tier-ceiling-flat::{5 + i}")
        for i in range(5)
    ]
    payload = {"rooms": rooms, "ceiling": ceilings}
    assert rule_story_ceiling_overcount(payload) == []


def test_story_overcount_flags_oversaturated_story():
    # Story 0 has 1 room but 6 ceilings — oversaturated trigger.
    rooms = [_room(story=0, room_index=0, y_lo=0.0, y_hi=2.5)] + [
        _room(story=1, room_index=1 + i, y_lo=2.5, y_hi=5.0) for i in range(4)
    ]
    ceilings = [
        _story_ceiling(mid_y=2.4, locator=f"uuid::tier-ceiling-flat::{i}")
        for i in range(6)
    ]
    payload = {"rooms": rooms, "ceiling": ceilings}
    flags = rule_story_ceiling_overcount(payload)
    assert len(flags) == 1
    assert flags[0]["rule"] == "story_ceiling_overcount"
    assert flags[0]["evidence"]["sub_pattern"] == "oversaturated"
    assert flags[0]["evidence"]["story"] == 0
    assert flags[0]["evidence"]["rooms_in_story"] == 1
    assert flags[0]["evidence"]["ceilings_in_story"] == 6
    assert flags[0]["locator"] == "uuid::tier-room::0"


def test_story_overcount_flags_high_ratio_on_non_top_story():
    # Non-top story 0 with 4 rooms / 14 ceilings -> ratio 3.5.
    rooms = [_room(story=0, room_index=i, y_lo=0.0, y_hi=2.5) for i in range(4)] + [
        _room(story=1, room_index=4 + i, y_lo=2.5, y_hi=5.0) for i in range(4)
    ]
    ceilings = [
        _story_ceiling(mid_y=2.4, locator=f"uuid::tier-ceiling-flat::{i}")
        for i in range(14)
    ]
    payload = {"rooms": rooms, "ceiling": ceilings}
    flags = rule_story_ceiling_overcount(payload)
    assert len(flags) == 1
    assert flags[0]["evidence"]["sub_pattern"] == "high_ratio"
    assert flags[0]["evidence"]["story"] == 0
    assert abs(flags[0]["evidence"]["ratio"] - 3.5) < 1e-6


def test_story_overcount_does_not_flag_top_story_high_ratio():
    # Top story 1 with 1 room / 14 ceilings hits the ratio test but top stories
    # legitimately have many roof obliques per room. Only oversaturated should
    # fire on top story; high_ratio should not.
    # Setup: 4 rooms in story 0, 1 room in story 1 (top), 14 ceilings in story 1.
    rooms = [_room(story=0, room_index=i, y_lo=0.0, y_hi=2.5) for i in range(4)] + [
        _room(story=1, room_index=4, y_lo=2.5, y_hi=5.0)
    ]
    ceilings = [
        _story_ceiling(mid_y=4.9, locator=f"uuid::tier-ceiling-flat::{i}")
        for i in range(14)
    ]
    payload = {"rooms": rooms, "ceiling": ceilings}
    flags = rule_story_ceiling_overcount(payload)
    # Oversaturated still triggers (1 room, 14 ceilings >= 5) but only once;
    # high_ratio rule is gated on non-top story, so we must not get a duplicate.
    assert len(flags) == 1
    assert flags[0]["evidence"]["sub_pattern"] == "oversaturated"


def test_story_overcount_returns_empty_when_no_rooms():
    assert rule_story_ceiling_overcount({"rooms": [], "ceiling": []}) == []
