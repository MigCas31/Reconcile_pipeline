from __future__ import annotations

import copy

import numpy as np
from shapely.geometry import MultiLineString

from reconcile.extract3d.overlaps import (
    _project_line_interval,
    clip_floor_overlaps,
    floor_polygon_to_shapely,
)


def _wall(
    wall_id: str,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    y0: float = 0.0,
    y1: float = 2.4,
) -> dict:
    return {
        "id": wall_id,
        "corners": [
            [x0, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z0],
        ],
    }


def _room(
    floor_bounds: tuple[float, float, float, float],
    walls: list[dict],
    *,
    story: int = 0,
    ceiling_y: float = 2.4,
) -> dict:
    x0, z0, x1, z1 = floor_bounds
    return {
        "story": story,
        "floor_polygon": [
            [x0, 0.0, z0],
            [x1, 0.0, z0],
            [x1, 0.0, z1],
            [x0, 0.0, z1],
        ],
        "ceiling_polygon": [
            [x0, ceiling_y, z0],
            [x1, ceiling_y, z0],
            [x1, ceiling_y, z1],
            [x0, ceiling_y, z1],
        ],
        "walls_computed": walls,
        "doors": [],
        "windows": [],
    }


def _poly_area(room: dict, key: str = "floor_polygon") -> float:
    poly = floor_polygon_to_shapely(room[key])
    return 0.0 if poly is None else float(poly.area)


def test_clip_overlaps_long_shared_wall_outside() -> None:
    rooms_out = [
        _room(
            (0.0, 0.0, 4.0, 2.0),
            [_wall("winner-wall", 0.0, 0.0, 4.0, 0.0)],
        ),
        _room(
            (3.0, 0.0, 7.0, 2.0),
            [_wall("loser-wall", -3.0, 0.0, 7.0, 0.0)],
        ),
    ]

    clipped = copy.deepcopy(rooms_out)
    metrics = clip_floor_overlaps(clipped)

    assert metrics
    assert clipped[1]["floor_clipped"] is True
    assert [wall["id"] for wall in clipped[1]["walls_computed"]] == []
    assert [wall["id"] for wall in clipped[1]["walls_removed_overlap"]] == [
        "loser-wall"
    ]
    assert any(
        decision["wall_id"] == "loser-wall" and decision["decision"] == "removed"
        for decision in metrics[0]["wall_decisions"]
    )
    assert _poly_area(clipped[0]) == 8.0
    assert _poly_area(clipped[1]) == 6.0
    assert (
        floor_polygon_to_shapely(clipped[0]["floor_polygon"])
        .intersection(floor_polygon_to_shapely(clipped[1]["floor_polygon"]))
        .area
        == 0.0
    )


def test_clip_overlaps_keeps_parallel_offset() -> None:
    rooms_out = [
        _room(
            (0.0, 0.0, 4.0, 2.0),
            [_wall("winner-wall", 0.0, 0.0, 4.0, 0.0)],
        ),
        _room(
            (3.0, 0.0, 7.0, 2.0),
            [_wall("offset-wall", -3.0, 0.35, 7.0, 0.35)],
        ),
    ]

    clipped = copy.deepcopy(rooms_out)
    metrics = clip_floor_overlaps(clipped)

    assert metrics
    assert [wall["id"] for wall in clipped[1]["walls_computed"]] == ["offset-wall"]
    assert clipped[1]["walls_removed_overlap"] == []
    assert any(
        decision["wall_id"] == "offset-wall" and decision["decision"] == "kept"
        for decision in metrics[0]["wall_decisions"]
    )


def test_clip_overlaps_keeps_touching_no_winner() -> None:
    rooms_out = [
        _room(
            (0.0, 0.0, 4.0, 2.0),
            [_wall("winner-wall", 0.0, 0.0, 4.0, 0.0)],
        ),
        _room(
            (3.0, 0.0, 7.0, 2.0),
            [_wall("exterior-wall", 2.9, 2.0, 7.0, 2.0)],
        ),
    ]

    clipped = copy.deepcopy(rooms_out)
    metrics = clip_floor_overlaps(clipped)

    assert metrics
    assert [wall["id"] for wall in clipped[1]["walls_computed"]] == ["exterior-wall"]
    assert clipped[1]["walls_removed_overlap"] == []
    assert any(
        decision["wall_id"] == "exterior-wall" and decision["decision"] == "kept"
        for decision in metrics[0]["wall_decisions"]
    )


def test_clipped_floor_owns_ceiling() -> None:
    rooms_out = [
        _room(
            (0.0, 0.0, 4.0, 2.0),
            [_wall("winner-wall", 0.0, 0.0, 4.0, 0.0)],
            ceiling_y=2.7,
        ),
        _room(
            (3.0, 0.0, 7.0, 2.0),
            [_wall("loser-wall", -3.0, 0.0, 7.0, 0.0)],
            ceiling_y=2.7,
        ),
    ]

    clipped = copy.deepcopy(rooms_out)
    clip_floor_overlaps(clipped)

    derived_ceilings = []
    for room in clipped:
        ceiling_y = room["ceiling_polygon"][0][1]
        derived_ceiling = [
            [corner[0], ceiling_y, corner[2]] for corner in room["floor_polygon"]
        ]
        derived_ceilings.append(derived_ceiling)

    ceiling_a = floor_polygon_to_shapely(derived_ceilings[0])
    ceiling_b = floor_polygon_to_shapely(derived_ceilings[1])
    assert ceiling_a is not None
    assert ceiling_b is not None
    assert ceiling_a.intersection(ceiling_b).area == 0.0


def test_project_line_interval_accepts_multiline_geometry() -> None:
    line = MultiLineString([[(1.0, 0.0), (2.0, 0.0)], [(4.0, 0.0), (5.0, 0.0)]])

    start, end = _project_line_interval(
        line, np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )

    assert start == 1.0
    assert end == 5.0
