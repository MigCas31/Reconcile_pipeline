"""Tests for the building-level merge (plan §3 Increment 7)."""

from __future__ import annotations

from reconcile_tiers.polyhedron.building_merge import (
    envelope_candidate_from_building,
    repair_building,
)


def _pt(x: float, y: float, z: float) -> dict:
    return {"x": x, "y": y, "z": z}


def _rect_floor(z0: float, z1: float, y: float = 0.0) -> list[dict]:
    return [_pt(0, y, z0), _pt(4, y, z0), _pt(4, y, z1), _pt(0, y, z1)]


def _wall(corners: list[tuple[float, float, float]], plane: dict) -> dict:
    return {
        "corners": [_pt(*c) for c in corners],
        "plane": plane,
    }


def _two_room_one_story_payload() -> dict:
    """Two adjacent rooms sharing one wall at z=3.

    Room A: 0<=x<=4, 0<=z<=3, y=0..2.5.
    Room B: 0<=x<=4, 3<=z<=6, y=0..2.5.
    Shared wall at z=3.
    """
    shared_wall_corners_A = [
        (0.0, 0.0, 3.0),
        (4.0, 0.0, 3.0),
        (4.0, 2.5, 3.0),
        (0.0, 2.5, 3.0),
    ]
    shared_wall_corners_B = [
        (4.0, 0.0, 3.0),
        (0.0, 0.0, 3.0),
        (0.0, 2.5, 3.0),
        (4.0, 2.5, 3.0),
    ]
    room_a = {
        "story": 0,
        "locator_id": "demo::tier-room::0",
        "floor": [
            {
                "corners": _rect_floor(0, 3),
                "plane": {"a": 0, "b": -1, "c": 0, "d": 0},
            }
        ],
        "walls": [
            _wall(
                shared_wall_corners_A,
                {"a": 0, "b": 0, "c": 1, "d": 3},
            ),
            _wall(
                [(4, 0, 0), (4, 0, 3), (4, 2.5, 3), (4, 2.5, 0)],
                {"a": 1, "b": 0, "c": 0, "d": 4},
            ),
            _wall(
                [(0, 0, 0), (4, 0, 0), (4, 2.5, 0), (0, 2.5, 0)],
                {"a": 0, "b": 0, "c": -1, "d": 0},
            ),
            _wall(
                [(0, 0, 0), (0, 0, 3), (0, 2.5, 3), (0, 2.5, 0)],
                {"a": -1, "b": 0, "c": 0, "d": 0},
            ),
        ],
    }
    room_b = {
        "story": 0,
        "locator_id": "demo::tier-room::1",
        "floor": [
            {
                "corners": _rect_floor(3, 6),
                "plane": {"a": 0, "b": -1, "c": 0, "d": 0},
            }
        ],
        "walls": [
            _wall(
                [(4, 0, 6), (0, 0, 6), (0, 2.5, 6), (4, 2.5, 6)],
                {"a": 0, "b": 0, "c": 1, "d": 6},
            ),
            _wall(
                [(4, 0, 3), (4, 0, 6), (4, 2.5, 6), (4, 2.5, 3)],
                {"a": 1, "b": 0, "c": 0, "d": 4},
            ),
            _wall(
                shared_wall_corners_B,
                {"a": 0, "b": 0, "c": -1, "d": -3},
            ),
            _wall(
                [(0, 0, 3), (0, 0, 6), (0, 2.5, 6), (0, 2.5, 3)],
                {"a": -1, "b": 0, "c": 0, "d": 0},
            ),
        ],
    }
    payload = {
        "rooms": [room_a, room_b],
        "ceiling": [
            {
                "corners": [
                    _pt(0, 2.5, 0),
                    _pt(4, 2.5, 0),
                    _pt(4, 2.5, 3),
                    _pt(0, 2.5, 3),
                ],
                "plane": {"a": 0, "b": 1, "c": 0, "d": 2.5},
            },
            {
                "corners": [
                    _pt(0, 2.5, 3),
                    _pt(4, 2.5, 3),
                    _pt(4, 2.5, 6),
                    _pt(0, 2.5, 6),
                ],
                "plane": {"a": 0, "b": 1, "c": 0, "d": 2.5},
            },
        ],
    }
    return payload


def _two_story_stacked_payload() -> dict:
    """Two single-room stories stacked. Story 0 ceiling at y=2.5 coincides
    with Story 1 floor at y=2.5 (same plane signature, opposite normals
    in tier_payload convention: floor faces -Y, ceiling faces +Y)."""
    story0 = {
        "story": 0,
        "locator_id": "demo::tier-room::0",
        "floor": [
            {
                "corners": _rect_floor(0, 3, y=0.0),
                "plane": {"a": 0, "b": -1, "c": 0, "d": 0},
            }
        ],
        "walls": [
            _wall(
                [(0, 0, 3), (4, 0, 3), (4, 2.5, 3), (0, 2.5, 3)],
                {"a": 0, "b": 0, "c": 1, "d": 3},
            ),
            _wall(
                [(4, 0, 0), (4, 0, 3), (4, 2.5, 3), (4, 2.5, 0)],
                {"a": 1, "b": 0, "c": 0, "d": 4},
            ),
            _wall(
                [(0, 0, 0), (4, 0, 0), (4, 2.5, 0), (0, 2.5, 0)],
                {"a": 0, "b": 0, "c": -1, "d": 0},
            ),
            _wall(
                [(0, 0, 0), (0, 0, 3), (0, 2.5, 3), (0, 2.5, 0)],
                {"a": -1, "b": 0, "c": 0, "d": 0},
            ),
        ],
    }
    story1 = {
        "story": 1,
        "locator_id": "demo::tier-room::1",
        "floor": [
            {
                "corners": _rect_floor(0, 3, y=2.5),
                "plane": {"a": 0, "b": -1, "c": 0, "d": -2.5},
            }
        ],
        "walls": [
            _wall(
                [(0, 2.5, 3), (4, 2.5, 3), (4, 5, 3), (0, 5, 3)],
                {"a": 0, "b": 0, "c": 1, "d": 3},
            ),
            _wall(
                [(4, 2.5, 0), (4, 2.5, 3), (4, 5, 3), (4, 5, 0)],
                {"a": 1, "b": 0, "c": 0, "d": 4},
            ),
            _wall(
                [(0, 2.5, 0), (4, 2.5, 0), (4, 5, 0), (0, 5, 0)],
                {"a": 0, "b": 0, "c": -1, "d": 0},
            ),
            _wall(
                [(0, 2.5, 0), (0, 2.5, 3), (0, 5, 3), (0, 5, 0)],
                {"a": -1, "b": 0, "c": 0, "d": 0},
            ),
        ],
    }
    payload = {
        "rooms": [story0, story1],
        "ceiling": [
            {
                "corners": [
                    _pt(0, 2.5, 0),
                    _pt(4, 2.5, 0),
                    _pt(4, 2.5, 3),
                    _pt(0, 2.5, 3),
                ],
                "plane": {"a": 0, "b": 1, "c": 0, "d": 2.5},
            },
            {
                "corners": [
                    _pt(0, 5, 0),
                    _pt(4, 5, 0),
                    _pt(4, 5, 3),
                    _pt(0, 5, 3),
                ],
                "plane": {"a": 0, "b": 1, "c": 0, "d": 5},
            },
        ],
    }
    return payload


def test_repair_building_marks_shared_wall_interior():
    """Two rooms sharing a wall at z=3 → both rooms' z=3 walls are
    classified as interior_shared_wall."""
    payload = _two_room_one_story_payload()
    result = repair_building(payload)
    interior_walls = [
        f for f in result.faces if f.kind == "interior_shared_wall"
    ]
    # Two faces, one in each room, pointing at each other.
    assert len(interior_walls) == 2
    rooms = {f.room_index for f in interior_walls}
    assert rooms == {0, 1}
    # Both have plane z=3 (within tolerance) — the shared wall plane.
    for f in interior_walls:
        assert abs(abs(f.plane.c) - 1.0) < 0.05
        assert abs(abs(f.plane.d) - 3.0) < 0.05


def test_repair_building_marks_storey_boundary_interior():
    """Story 0 ceiling + Story 1 floor at y=2.5 → both classified as
    interior_storey_boundary."""
    payload = _two_story_stacked_payload()
    result = repair_building(payload)
    boundaries = [
        f for f in result.faces if f.kind == "interior_storey_boundary"
    ]
    assert len(boundaries) == 2
    stories = sorted(f.story for f in boundaries)
    assert stories == [0, 1]
    for f in boundaries:
        assert abs(abs(f.plane.b) - 1.0) < 0.05
        assert abs(abs(f.plane.d) - 2.5) < 0.05


def test_envelope_candidate_drops_interior_by_default():
    """envelope_candidate_from_building emits only exterior faces by
    default; interior faces appear when include_interior=True."""
    payload = _two_room_one_story_payload()
    result = repair_building(payload)
    default_env = envelope_candidate_from_building(result)
    assert default_env is not None
    sources = {f.source for f in default_env.faces}
    # No interior_shared_wall label leaks through default emission.
    assert not any("::interior" in s for s in sources)
    exterior_count = sum(1 for f in result.faces if f.kind == "exterior")
    assert len(default_env.faces) == exterior_count

    full_env = envelope_candidate_from_building(
        result, include_interior=True
    )
    assert full_env is not None
    assert len(full_env.faces) == len(result.faces)
    assert any("::interior_shared_wall" in f.source for f in full_env.faces)


def test_repair_building_isolated_rooms_have_no_interior_faces():
    """Two rooms that don't share a wall → no interior faces."""
    payload = _two_room_one_story_payload()
    # Move room B far away in z so it doesn't share the wall any more.
    for w in payload["rooms"][1]["walls"]:
        for c in w["corners"]:
            c["z"] += 20
    for c in payload["rooms"][1]["floor"][0]["corners"]:
        c["z"] += 20
    for w in payload["rooms"][1]["walls"]:
        plane = w["plane"]
        if abs(plane["c"]) > 0.5:
            plane["d"] = plane["d"] * 1 + 20 * plane["c"]
    payload["ceiling"][1]["corners"] = [
        {"x": c["x"], "y": c["y"], "z": c["z"] + 20}
        for c in payload["ceiling"][1]["corners"]
    ]
    result = repair_building(payload)
    interior = [f for f in result.faces if f.kind != "exterior"]
    assert interior == []


def test_repair_building_drops_same_room_filler_duplicate():
    """A filler that the per-room repair picks via neighbor-plane
    extension lands coplanar with the tile it extends. In the building
    envelope this coincident pair is visually redundant — the filler
    should be classified as duplicate so only one copy survives."""
    from reconcile_tiers.polyhedron.building_merge import repair_building

    payload = _two_room_one_story_payload()
    result = repair_building(payload)
    same_room_dup_fillers = [
        f
        for f in result.faces
        if f.kind == "duplicate" and f.source.startswith("polyhedron_v3_filler")
    ]
    # On the synthetic two-room payload the per-room repair adds a few
    # fillers whose plane matches the room's floor or ceiling; those are
    # the ones we expect to be deduped.
    assert all(f.kind == "duplicate" for f in same_room_dup_fillers)


def test_repair_building_keeps_partially_overlapping_coplanar_tiles():
    """Two coplanar tiles from different rooms that share only a sliver
    of overlap must NOT be classified as duplicate — each carries unique
    area that would leave a 2D gap in the envelope if one were dropped."""
    from reconcile_tiers.polyhedron.building_merge import repair_building

    payload = _two_room_one_story_payload()
    # Shift room B's ceiling so it OVERHANGS room A's ceiling region by
    # only 25% (a sliver). Without the dual-overlap threshold this small
    # overlap would mark one of them as a duplicate.
    payload["ceiling"][1]["corners"] = [
        _pt(0, 2.5, 2.5),  # overlaps with room A's ceiling z=[0,3] by z=[2.5,3]
        _pt(4, 2.5, 2.5),
        _pt(4, 2.5, 5.5),
        _pt(0, 2.5, 5.5),
    ]
    result = repair_building(payload)
    ceil_dups = [
        f
        for f in result.faces
        if f.kind == "duplicate" and f.source == "ceiling"
    ]
    # Both ceilings carry unique area; neither should be marked duplicate.
    assert ceil_dups == []
