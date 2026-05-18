from collections import Counter
from dataclasses import replace

import pytest
from shapely.geometry import Polygon

from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
    RawCeilingPlane,
)
from reconcile_tiers.extract.wall_pairs import compute_slab_walls


def _wall(
    wall_id: str,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    floor_y: float = -1.6069,
    top_y: float = 0.7631,
) -> ExtractedWall:
    return ExtractedWall(
        id=wall_id,
        source="test",
        corners=[
            [a[0], top_y, a[1]],
            [b[0], top_y, b[1]],
            [b[0], floor_y, b[1]],
            [a[0], floor_y, a[1]],
        ],
    )


def _room(index: int, wall: ExtractedWall) -> ExtractedRoom:
    floor_y = min(corner[1] for corner in wall.corners)
    top_y = max(corner[1] for corner in wall.corners)
    return ExtractedRoom(
        index=index,
        story=0,
        floor_polygon=[
            [wall.corners[3][0], floor_y, wall.corners[3][2]],
            [wall.corners[2][0], floor_y, wall.corners[2][2]],
            [wall.corners[1][0], floor_y, wall.corners[1][2]],
            [wall.corners[0][0], floor_y, wall.corners[0][2]],
        ],
        walls_merged=[],
        walls_computed=[wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[
            [wall.corners[3][0], top_y, wall.corners[3][2]],
            [wall.corners[2][0], top_y, wall.corners[2][2]],
            [wall.corners[1][0], top_y, wall.corners[1][2]],
            [wall.corners[0][0], top_y, wall.corners[0][2]],
        ],
        ceiling_type="flat",
        ceiling_eave_height=top_y,
        ceiling_ridge_height=top_y,
    )


def test_wall_pair_slab_closes_reported_42cm_room_gap():
    # Regression for 1825a812-09d0-4407-9265-182a07053cfc:
    # rooms 1 and 5 have facing wall scans 0.4225 m apart. That is a
    # plausible room-to-room wall cavity, not an opening in the building.
    room_1_wall = _wall(
        "AD89BAD9-517E-4E0C-BC9F-0F0804BDD0EF",
        (-9.131575990041876, -3.5724685796832527),
        (-8.790712754250041, -0.13754000925902554),
    )
    room_5_wall = _wall(
        "2C73E698-DB49-48DA-AC04-251FA1791893",
        (-8.387496624593833, -0.3527692418979935),
        (-8.72378880843358, -3.741631462007667),
    )

    walls = compute_slab_walls([_room(1, room_1_wall), _room(5, room_5_wall)])

    assert Counter(wall.type for wall in walls) == {
        "gap_floor": 1,
        "gap_ceiling": 1,
        "within_story": 2,
    }
    for wall in walls:
        assert "slab:r0w0-r1w0" in wall.id


def test_wall_pair_slab_closes_sub_room_width_cavity():
    # A 69 cm strip between two long facing room walls is narrower than a
    # plausible scanned room in the corpus; treat it as wall/gap volume.
    room_a = _room(0, _wall("room-a-wall", (0.0, 0.0), (3.0, 0.0)))
    room_b = _room(1, _wall("room-b-wall", (0.0, 0.69), (3.0, 0.69)))

    walls = compute_slab_walls([room_a, room_b])

    assert Counter(wall.type for wall in walls) == {
        "gap_floor": 1,
        "gap_ceiling": 1,
        "within_story": 2,
    }


def test_bounded_recess_slab_closes_three_wall_ceiling_gap():
    # Regression for 6e8a252f-fc38-4ffa-8691-3f43938a0a16:
    # two facing external wall runs are ~83 cm apart, which is wider than a
    # wall-thickness slab. A third wall visibly caps one end, so this is a
    # narrow wall-bounded recess that needs a ceiling cap.
    left = _wall("left-recess-wall", (0.0, 0.0), (0.0, 2.0))
    right = _wall("right-recess-wall", (0.83, 0.0), (0.83, 2.0))
    cap = _wall("cap-recess-wall", (0.0, 0.0), (0.83, 0.0))
    door = ExtractedElement(
        id="left-recess-door",
        source="test",
        parent_wall_id=left.id,
        corners=[
            [0.0, -1.6069, 0.0],
            [0.0, -1.6069, 0.7],
            [0.0, 0.2, 0.7],
            [0.0, 0.2, 0.0],
        ],
    )

    walls = compute_slab_walls(
        [
            replace(_room(0, left), doors=[door]),
            _room(1, right),
            _room(2, cap),
        ]
    )

    assert Counter(wall.type for wall in walls) == {
        "gap_floor": 1,
        "gap_ceiling": 1,
        "within_story": 2,
    }
    ceiling = next(wall for wall in walls if wall.type == "gap_ceiling")
    assert Polygon(
        [(corner[0], corner[2]) for corner in ceiling.corners]
    ).area == pytest.approx(1.66)


def test_bounded_recess_requires_opening_on_paired_wall():
    left = _wall("left-recess-wall", (0.0, 0.0), (0.0, 2.0))
    right = _wall("right-recess-wall", (0.83, 0.0), (0.83, 2.0))
    cap = _wall("cap-recess-wall", (0.0, 0.0), (0.83, 0.0))

    assert (
        compute_slab_walls(
            [
                _room(0, left),
                _room(1, right),
                _room(2, cap),
            ]
        )
        == []
    )


def test_wide_wall_pair_without_cap_is_not_treated_as_wall_thickness():
    room_a = _room(0, _wall("room-a-wall", (0.0, 0.0), (0.0, 2.0)))
    room_b = _room(1, _wall("room-b-wall", (0.83, 0.0), (0.83, 2.0)))

    assert compute_slab_walls([room_a, room_b]) == []


def test_wall_pair_side_caps_extend_to_descent_strip_bottoms():
    floor_y = -1.4
    room_a_wall = replace(
        _wall("room-a-wall", (0.0, 0.0), (3.0, 0.0), floor_y=floor_y, top_y=1.0),
        descent_strip=[
            [
                [0.0, -1.8, 0.0],
                [3.0, -1.8, 0.0],
                [3.0, floor_y, 0.0],
                [0.0, floor_y, 0.0],
            ]
        ],
    )
    room_a_attached = replace(
        _wall("room-a-attached", (3.0, 0.0), (3.0, -1.0), floor_y=floor_y, top_y=1.0),
        descent_strip=[
            [
                [3.0, -1.85, 0.0],
                [3.0, -1.85, -1.0],
                [3.0, floor_y, -1.0],
                [3.0, floor_y, 0.0],
            ]
        ],
    )
    room_b_wall = replace(
        _wall("room-b-wall", (0.0, 0.2), (3.0, 0.2), floor_y=floor_y, top_y=1.0),
        descent_strip=[
            [
                [0.0, -1.9, 0.2],
                [3.0, -1.9, 0.2],
                [3.0, floor_y, 0.2],
                [0.0, floor_y, 0.2],
            ]
        ],
    )
    room_b_attached = replace(
        _wall("room-b-attached", (3.0, 0.2), (3.0, 1.2), floor_y=floor_y, top_y=1.0),
        descent_strip=[
            [
                [3.0, -1.95, 0.2],
                [3.0, -1.95, 1.2],
                [3.0, floor_y, 1.2],
                [3.0, floor_y, 0.2],
            ]
        ],
    )
    room_a = replace(
        _room(0, room_a_wall), walls_computed=[room_a_wall, room_a_attached]
    )
    room_b = replace(
        _room(1, room_b_wall), walls_computed=[room_b_wall, room_b_attached]
    )

    walls = compute_slab_walls([room_a, room_b])

    side_walls = [wall for wall in walls if wall.type == "within_story"]
    assert len(side_walls) == 2
    end_side = max(
        side_walls, key=lambda wall: max(corner[0] for corner in wall.corners)
    )
    ys = sorted(corner[1] for corner in end_side.corners[:2])
    assert ys == pytest.approx([-1.95, -1.85])


def test_wall_pair_gap_ceiling_follows_adjacent_sloped_room_ceiling():
    # The room-to-room slab is a wall-thickness closure, but its top must
    # still respect the room ceiling. A flat wall top above the roof plane
    # would otherwise create a horizontal strip protruding through the slant.
    room_a_wall = _wall(
        "sloped-side",
        (0.0, 0.0),
        (2.0, 0.0),
        floor_y=0.0,
        top_y=2.0,
    )
    room_b_wall = _wall(
        "no-ceiling-side",
        (0.0, 0.2),
        (2.0, 0.2),
        floor_y=0.0,
        top_y=2.0,
    )
    room_a = _room(0, room_a_wall)
    room_a = replace(
        room_a,
        floor_polygon=[
            [0.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        ceiling_polygon=[
            [0.0, 0.5, -1.0],
            [2.0, 0.5, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        ceiling_type="sloped",
        ceiling_eave_height=0.5,
        ceiling_ridge_height=1.0,
    )
    room_b = _room(1, room_b_wall)
    room_b = replace(
        room_b,
        floor_polygon=[
            [0.0, 0.0, 0.2],
            [2.0, 0.0, 0.2],
            [2.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
        ],
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )

    walls = compute_slab_walls([room_a, room_b])

    ceiling = next(wall for wall in walls if wall.type == "gap_ceiling")
    ys = [corner[1] for corner in ceiling.corners]
    assert max(ys) < 1.2
    assert max(ys) - min(ys) > 0.09


def test_wall_pair_gap_ceiling_prefers_local_ceiling_over_lower_wall_top():
    room_a_wall = _wall(
        "high-side",
        (0.0, 0.0),
        (2.0, 0.0),
        floor_y=0.0,
        top_y=1.0,
    )
    room_b_wall = _wall(
        "short-scanned-side",
        (0.0, 0.2),
        (2.0, 0.2),
        floor_y=0.0,
        top_y=0.4,
    )
    room_a = replace(
        _room(0, room_a_wall),
        ceiling_polygon=[
            [0.0, 1.0, -1.0],
            [2.0, 1.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        ceiling_eave_height=1.0,
        ceiling_ridge_height=1.0,
    )
    room_b = replace(
        _room(1, room_b_wall),
        floor_polygon=[
            [0.0, 0.0, 0.2],
            [2.0, 0.0, 0.2],
            [2.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
        ],
        ceiling_polygon=[
            [0.0, 0.9, 0.2],
            [2.0, 0.9, 0.2],
            [2.0, 0.9, 1.2],
            [0.0, 0.9, 1.2],
        ],
        ceiling_eave_height=0.9,
        ceiling_ridge_height=0.9,
    )

    walls = compute_slab_walls([room_a, room_b])

    ceiling = next(wall for wall in walls if wall.type == "gap_ceiling")
    assert {round(corner[1], 3) for corner in ceiling.corners} == {0.9}
    side_tops = [
        corner[1]
        for wall in walls
        if wall.type == "within_story"
        for corner in wall.corners[2:]
    ]
    assert min(side_tops) == pytest.approx(0.9)


def test_wall_pair_gap_ceiling_uses_local_raw_ceiling_facets():
    room_a_wall = _wall(
        "mixed-ceiling-side",
        (0.0, 0.0),
        (4.0, 0.0),
        floor_y=0.0,
        top_y=3.0,
    )
    room_b_wall = _wall(
        "opposite-side",
        (0.0, 0.2),
        (4.0, 0.2),
        floor_y=0.0,
        top_y=3.0,
    )
    room_a = replace(
        _room(0, room_a_wall),
        floor_polygon=[
            [0.0, 0.0, -1.0],
            [4.0, 0.0, -1.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        ceiling_polygon=[
            [0.0, 2.0, -1.0],
            [4.0, 2.0, -1.0],
            [4.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        ceiling_type="sloped",
        ceiling_eave_height=1.0,
        ceiling_ridge_height=2.0,
        raw_ceiling_planes=[
            RawCeilingPlane(
                corners=[
                    [0.0, 1.0, -1.0],
                    [2.0, 1.0, -1.0],
                    [2.0, 1.5, 0.0],
                    [0.0, 1.5, 0.0],
                ]
            ),
            RawCeilingPlane(
                corners=[
                    [2.0, 2.0, -1.0],
                    [4.0, 1.0, -1.0],
                    [4.0, 1.0, 0.0],
                    [2.0, 2.0, 0.0],
                ]
            ),
        ],
    )
    room_b = replace(
        _room(1, room_b_wall),
        floor_polygon=[
            [0.0, 0.0, 0.2],
            [4.0, 0.0, 0.2],
            [4.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
        ],
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )

    walls = compute_slab_walls([room_a, room_b])

    ceiling = next(wall for wall in walls if wall.type == "gap_ceiling")
    y_by_x = {
        round(corner[0], 1): corner[1]
        for corner in ceiling.corners
        if round(corner[2], 1) == 0.0
    }
    assert y_by_x[0.0] == pytest.approx(1.5)
    assert y_by_x[4.0] == pytest.approx(1.0)
