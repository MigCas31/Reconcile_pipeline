"""Tests for segment-cycle classification with verbatim tier geometry."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.segment_tier_room_payload import (
    SEGMENT_TIER_SHELL,
    build_segment_tier_room_payload,
)
from tests.reconcile_tiers.room_postprocessing.test_segment_room_cycles import (
    _four_wall_room_payload,
    _two_adjacent_rooms_payload,
)


def test_four_wall_preserves_tier_wall_corners() -> None:
    tier_payload = _four_wall_room_with_floor()
    original_south = tier_payload["rooms"][0]["walls"][0]
    out = build_segment_tier_room_payload(tier_payload, corner_tol=0.05)
    assert len(out["rooms"]) == 1
    assert out["room_postprocessing_source"]["shell"] == SEGMENT_TIER_SHELL
    assert out["room_postprocessing_source"]["geometry_source"] == "tier"

    room = out["rooms"][0]
    south = next(w for w in room["walls"] if w["locator_id"] == "w-south")
    assert south["corners"] == original_south["corners"]
    assert room["floor"][0]["corners"] == tier_payload["rooms"][0]["floor"][0]["corners"]


def test_two_adjacent_cycles_get_tier_walls_per_room() -> None:
    payload = _two_adjacent_rooms_with_floors()
    out = build_segment_tier_room_payload(payload, corner_tol=0.05)
    rooms = out["rooms"]
    assert len(rooms) >= 2

    left = next(r for r in rooms if "w-left-west" in {w["locator_id"] for w in r["walls"]})
    right = next(r for r in rooms if "w-right-east" in {w["locator_id"] for w in r["walls"]})
    assert "w-shared" in {w["locator_id"] for w in left["walls"]}
    assert "w-shared" in {w["locator_id"] for w in right["walls"]}
    assert len(left["floor"]) >= 1
    assert len(right["floor"]) >= 1


def test_assign_skips_cycle_without_tier_floor() -> None:
    graph = build_segment_tier_room_payload(
        _two_adjacent_rooms_payload(),
        corner_tol=0.05,
    )
    assert graph["room_postprocessing_source"]["segment_room_count"] == 0


def _four_wall_room_with_floor() -> dict[str, Any]:
    payload = _four_wall_room_payload()
    payload["rooms"][0]["floor"] = [
        {
            "locator_id": "floor-0",
            "corners": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 4.0, "y": 0.0, "z": 0.0},
                {"x": 4.0, "y": 0.0, "z": 3.0},
                {"x": 0.0, "y": 0.0, "z": 3.0},
            ],
        }
    ]
    return payload


def _two_adjacent_rooms_with_floors() -> dict[str, Any]:
    payload = _two_adjacent_rooms_payload()
    payload["rooms"][0]["floor"] = [
        {
            "locator_id": "floor-left",
            "corners": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 2.0, "y": 0.0, "z": 0.0},
                {"x": 2.0, "y": 0.0, "z": 2.0},
                {"x": 0.0, "y": 0.0, "z": 2.0},
            ],
        },
        {
            "locator_id": "floor-right",
            "corners": [
                {"x": 2.0, "y": 0.0, "z": 0.0},
                {"x": 4.0, "y": 0.0, "z": 0.0},
                {"x": 4.0, "y": 0.0, "z": 2.0},
                {"x": 2.0, "y": 0.0, "z": 2.0},
            ],
        },
    ]
    return payload
