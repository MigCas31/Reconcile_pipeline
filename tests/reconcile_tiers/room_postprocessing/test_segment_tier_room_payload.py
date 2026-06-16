"""Tests for segment-cycle classification with perimeter walls + tier floors."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.segment_tier_room_payload import (
    SEGMENT_TIER_GEOMETRY_SOURCE,
    SEGMENT_TIER_SHELL,
    build_segment_tier_room_payload,
)
from tests.reconcile_tiers.room_postprocessing.test_segment_room_cycles import (
    _four_wall_room_payload,
    _two_adjacent_rooms_payload,
)


def test_four_wall_walls_use_trimmed_tier_geometry() -> None:
    tier_payload = _four_wall_room_with_floor()
    out = build_segment_tier_room_payload(tier_payload, corner_tol=0.05)
    assert len(out["rooms"]) == 1
    assert out["room_postprocessing_source"]["shell"] == SEGMENT_TIER_SHELL
    assert (
        out["room_postprocessing_source"]["geometry_source"]
        == SEGMENT_TIER_GEOMETRY_SOURCE
    )

    room = out["rooms"][0]
    graph_room = out["segment_room_graph"]["nodes"][0]
    assert len(room["walls"]) == len(graph_room["perimeter_sides"])
    assert {w["locator_id"] for w in room["walls"]} == set(graph_room["wall_ids"])
    assert room["floor"][0]["corners"] == tier_payload["rooms"][0]["floor"][0]["corners"]

    tier_walls = {w["locator_id"]: w for w in tier_payload["rooms"][0]["walls"]}
    for wall in room["walls"]:
        orig = tier_walls[wall["locator_id"]]
        orig_ys = {c["y"] for c in orig["corners"]}
        wall_ys = {c["y"] for c in wall["corners"]}
        assert wall_ys == orig_ys
        assert wall["corners"] == orig["corners"]
        assert wall.get("source_wall_id") == wall["locator_id"]


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
