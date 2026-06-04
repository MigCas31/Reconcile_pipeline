"""Tests for tier_payload built from segment-room cycles."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.segment_room_payload import (
    _postprocess_elements,
    build_segment_room_tier_payload,
)
from tests.reconcile_tiers.room_postprocessing.test_segment_room_cycles import (
    _four_wall_room_payload,
    _two_adjacent_rooms_payload,
)
from tests.reconcile_tiers.room_postprocessing.test_wall_segment_graph import (
    _passing_wall_junction_payload,
)


def test_four_wall_yields_one_segment_room_with_four_walls() -> None:
    out = build_segment_room_tier_payload(_four_wall_room_payload(), corner_tol=0.05)
    rooms = out["rooms"]
    assert len(rooms) == 1
    room = rooms[0]
    assert len(room["walls"]) == 4
    assert len(room["floor"][0]["corners"]) >= 3
    wall_ids = {w["locator_id"] for w in room["walls"]}
    assert wall_ids == {"w-south", "w-east", "w-north", "w-west"}
    assert out["room_postprocessing_source"]["segment_room_count"] == 1


def test_two_adjacent_rooms_yield_at_least_two_segment_rooms() -> None:
    out = build_segment_room_tier_payload(_two_adjacent_rooms_payload(), corner_tol=0.05)
    rooms = out["rooms"]
    assert len(rooms) >= 2
    all_walls: set[str] = set()
    for room in rooms:
        assert len(room["walls"]) >= 3
        for w in room["walls"]:
            all_walls.add(w["locator_id"])
    assert "w-shared" in all_walls


def test_postprocess_splits_walls_for_junction_payload() -> None:
    """Junction split runs before segment rooms; split ids must exist on elements."""

    elements, _ = _postprocess_elements(
        _passing_wall_junction_payload(),
        corner_tol=0.05,
        adjacency_tol=0.5,
    )
    wall_ids = {el.id for el in elements if el.kind == "wall"}
    assert "w-long::split::0" in wall_ids
    assert "w-long::split::1" in wall_ids
