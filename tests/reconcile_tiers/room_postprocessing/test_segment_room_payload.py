"""Tests for tier_payload built from segment-room cycles."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.segment_group_representative import (
    base_wall_id,
)
from reconcile_tiers.room_postprocessing.segment_room_payload import (
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
    assert out["room_postprocessing_source"]["shell"] == "half_closed_floor_walls"


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


def test_junction_split_room_one_tile_per_physical_wall() -> None:
    """Split pieces along one long wall must not duplicate tiles in tier payload."""

    out = build_segment_room_tier_payload(
        _passing_wall_junction_payload(),
        corner_tol=0.05,
        adjacency_tol=0.5,
    )
    rooms = out["rooms"]
    if not rooms:
        return
    room = rooms[0]
    locator_ids = [w["locator_id"] for w in room["walls"]]
    assert len(locator_ids) == len(set(locator_ids))
    long_count = sum(1 for lid in locator_ids if lid == "w-long")
    assert long_count <= 1
    bases = {base_wall_id(lid) for lid in locator_ids}
    assert len(locator_ids) == len(bases)


def test_floor_ring_uses_cycle_polygon_not_clipped_scan() -> None:
    """Shell floor follows wall-delineated cycle, matching wall bottom corners."""

    out = build_segment_room_tier_payload(_four_wall_room_payload(), corner_tol=0.05)
    room_node = next(
        n
        for n in out["segment_room_graph"]["nodes"]
        if n.get("kind") == "segment_room"
    )
    tier_room = out["rooms"][0]
    floor_xz = {(c["x"], c["z"]) for c in tier_room["floor"][0]["corners"]}
    cycle_xz = {
        (float(p["x"]), float(p["z"])) for p in room_node["polygon_xz"]
    }
    assert floor_xz == cycle_xz
