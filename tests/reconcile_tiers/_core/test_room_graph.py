"""Room-adjacency graph construction from doors and shared walls."""

from __future__ import annotations

from reconcile_tiers._core.room_graph import build_room_graph
from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)


def _wall(
    wid: str,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    y_low: float = 0.0,
    y_high: float = 2.5,
) -> ExtractedWall:
    """Build a 4-corner wall quad spanning (x0,z0)→(x1,z1) at floor-to-ceiling y."""
    return ExtractedWall(
        id=wid,
        corners=[
            [x0, y_low, z0],
            [x1, y_low, z1],
            [x1, y_high, z1],
            [x0, y_high, z0],
        ],
        source="merged-room",
    )


def _room(
    *,
    index: int,
    story: int,
    floor_polygon: list[list[float]],
    walls: list[ExtractedWall],
    doors: list[ExtractedElement] | None = None,
) -> ExtractedRoom:
    return ExtractedRoom(
        index=index,
        story=story,
        floor_polygon=floor_polygon,
        walls_merged=walls,
        walls_computed=walls,
        doors=doors or [],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )


def _two_rooms_sharing_a_wall(*, with_door: bool):
    """Two rectangular rooms sharing a 3-m wall along x ≈ 4.0 with 0.10 m thickness.

    Room A: footprint (0,0)-(4,3); east wall at x=4.0 (interior face).
    Room B: footprint (4.10,0)-(8,3); west wall at x=4.10 (interior face).

    Optionally place a door on Room B's west wall, parented by it.
    """
    a_walls = [
        _wall("A_north", 0.0, 0.0, 4.0, 0.0),
        _wall("A_east", 4.0, 0.0, 4.0, 3.0),
        _wall("A_south", 4.0, 3.0, 0.0, 3.0),
        _wall("A_west", 0.0, 3.0, 0.0, 0.0),
    ]
    b_walls = [
        _wall("B_west", 4.10, 0.0, 4.10, 3.0),
        _wall("B_north", 4.10, 0.0, 8.0, 0.0),
        _wall("B_east", 8.0, 0.0, 8.0, 3.0),
        _wall("B_south", 8.0, 3.0, 4.10, 3.0),
    ]
    b_doors: list[ExtractedElement] = []
    if with_door:
        b_doors.append(
            ExtractedElement(
                id="door1",
                # Door spans 1.0 m on B's west wall at z in [1.0, 2.0].
                corners=[
                    [4.10, 0.0, 1.0],
                    [4.10, 0.0, 2.0],
                    [4.10, 2.1, 2.0],
                    [4.10, 2.1, 1.0],
                ],
                source="merged-room",
                parent_wall_id="B_west",
            )
        )
    a = _room(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
        ],
        walls=a_walls,
    )
    b = _room(
        index=1,
        story=0,
        floor_polygon=[
            [4.10, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 3.0],
            [4.10, 0.0, 3.0],
        ],
        walls=b_walls,
        doors=b_doors,
    )
    return [a, b]


def test_two_rooms_with_shared_wall_emit_shared_wall_edge():
    rooms = _two_rooms_sharing_a_wall(with_door=False)
    g = build_room_graph(rooms)
    assert frozenset({0, 1}) in g.shared_wall_edges
    assert g.shared_wall_edges[frozenset({0, 1})] > 2.0  # ~3 m of overlap
    assert g.door_edges == frozenset()


def test_two_rooms_with_door_emit_door_edge():
    rooms = _two_rooms_sharing_a_wall(with_door=True)
    g = build_room_graph(rooms)
    assert frozenset({0, 1}) in g.door_edges
    assert frozenset({0, 1}) in g.shared_wall_edges


def test_neighbours_returns_door_and_wall_partners():
    rooms = _two_rooms_sharing_a_wall(with_door=True)
    g = build_room_graph(rooms)
    assert g.neighbours(0) == {1}
    assert g.neighbours(1) == {0}


def test_isolated_rooms_have_no_edges():
    """Two rooms far apart, no shared walls."""
    a = _room(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
        ],
        walls=[
            _wall("A_n", 0.0, 0.0, 4.0, 0.0),
            _wall("A_e", 4.0, 0.0, 4.0, 3.0),
            _wall("A_s", 4.0, 3.0, 0.0, 3.0),
            _wall("A_w", 0.0, 3.0, 0.0, 0.0),
        ],
    )
    # Room B placed 5 m away (well outside MAX_WALL_THICKNESS_M = 0.70).
    b = _room(
        index=1,
        story=0,
        floor_polygon=[
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [14.0, 0.0, 3.0],
            [10.0, 0.0, 3.0],
        ],
        walls=[
            _wall("B_n", 10.0, 0.0, 14.0, 0.0),
            _wall("B_e", 14.0, 0.0, 14.0, 3.0),
            _wall("B_s", 14.0, 3.0, 10.0, 3.0),
            _wall("B_w", 10.0, 3.0, 10.0, 0.0),
        ],
    )
    g = build_room_graph([a, b])
    assert g.shared_wall_edges == {}
    assert g.door_edges == frozenset()
    assert g.neighbours(0) == set()


def test_per_story_partition_is_recorded():
    rooms = _two_rooms_sharing_a_wall(with_door=False)
    # Add a third room on story 1.
    upper = _room(
        index=2,
        story=1,
        floor_polygon=[
            [0.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 3.0],
            [0.0, 3.0, 3.0],
        ],
        walls=[],
    )
    g = build_room_graph([*rooms, upper])
    assert g.rooms_by_story[0] == frozenset({0, 1})
    assert g.rooms_by_story[1] == frozenset({2})


def test_door_with_unknown_parent_wall_is_skipped():
    rooms = _two_rooms_sharing_a_wall(with_door=False)
    # Add a stray door on Room A whose parent_wall_id refers to a nonexistent wall.
    bad_door = ExtractedElement(
        id="bogus",
        corners=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        source="merged-room",
        parent_wall_id="NOT_A_REAL_WALL",
    )
    rooms[0] = ExtractedRoom(
        index=rooms[0].index,
        story=rooms[0].story,
        floor_polygon=rooms[0].floor_polygon,
        walls_merged=rooms[0].walls_merged,
        walls_computed=rooms[0].walls_computed,
        doors=[bad_door],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    g = build_room_graph(rooms)
    # Bad door does not produce a door edge; shared-wall edge still detected.
    assert g.door_edges == frozenset()
    assert frozenset({0, 1}) in g.shared_wall_edges


def test_empty_rooms_return_empty_graph():
    g = build_room_graph([])
    assert g.door_edges == frozenset()
    assert g.shared_wall_edges == {}
    assert g.rooms_by_story == {}
