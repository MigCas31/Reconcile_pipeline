"""Unit tests for ``reconcile.extract3d.height_alignment``.

The module snaps coplanar room groups on the same story to a common floor and
ceiling Y so neighbouring rooms stop showing a 2-5 cm step that scan noise had
left behind. The gate is a combined 6 cm floor-slab delta *and* 5 cm wall-height
delta; intentional outliers (knee walls, split levels) must survive.
"""

from __future__ import annotations

import pytest

from reconcile.extract3d.height_alignment import (
    CORNER_SNAP_TOL_M,
    FLOOR_ALIGN_TOL_M,
    WALL_HEIGHT_ALIGN_TOL_M,
    align_room_heights,
)
from reconcile.extract3d.lineage import STEP_ALIGN_ROOM_HEIGHTS


def _wall_quad(
    x0: float, z0: float, x1: float, z1: float, bot_y: float, top_y: float
) -> dict:
    return {
        "corners": [
            [x0, bot_y, z0],
            [x1, bot_y, z1],
            [x1, top_y, z1],
            [x0, top_y, z0],
        ],
        "id": f"wall-{x0:.2f}-{z0:.2f}-{x1:.2f}-{z1:.2f}-{bot_y:.3f}",
    }


def _rect_room(
    *,
    story: int,
    floor_y: float,
    wall_height: float,
    x0: float = 0.0,
    z0: float = 0.0,
    x1: float = 4.0,
    z1: float = 3.0,
) -> dict:
    """
    Rectangular room with 4 walls along each edge, all at
    floor_y..floor_y+wall_height.
    """
    top_y = floor_y + wall_height
    floor_polygon = [
        [x0, floor_y, z0],
        [x1, floor_y, z0],
        [x1, floor_y, z1],
        [x0, floor_y, z1],
    ]
    walls = [
        _wall_quad(x0, z0, x1, z0, floor_y, top_y),
        _wall_quad(x1, z0, x1, z1, floor_y, top_y),
        _wall_quad(x1, z1, x0, z1, floor_y, top_y),
        _wall_quad(x0, z1, x0, z0, floor_y, top_y),
    ]
    return {
        "story": story,
        "floor_polygon": floor_polygon,
        "walls_computed": walls,
    }


def _room_floor_y(room: dict) -> float:
    return sum(c[1] for c in room["floor_polygon"]) / len(room["floor_polygon"])


def _wall_min_max(room: dict, wall_idx: int) -> tuple[float, float]:
    corners = room["walls_computed"][wall_idx]["corners"]
    ys = [c[1] for c in corners]
    return min(ys), max(ys)


def test_snaps_two_rooms_with_small_floor_delta():
    """4 cm floor delta + equal wall heights → both rooms snap to the median."""
    a = _rect_room(story=0, floor_y=0.00, wall_height=2.50, x0=0.0, x1=4.0)
    b = _rect_room(story=0, floor_y=0.04, wall_height=2.50, x0=5.0, x1=9.0)

    metrics = align_room_heights([a, b])

    assert metrics["aligned_groups"] == 1
    assert metrics["aligned_rooms"] == 2
    assert metrics["aligned_walls"] == 8

    target_floor = 0.02
    assert _room_floor_y(a) == pytest.approx(target_floor, abs=1e-6)
    assert _room_floor_y(b) == pytest.approx(target_floor, abs=1e-6)
    for room in (a, b):
        for wall_idx in range(4):
            bot_y, top_y = _wall_min_max(room, wall_idx)
            assert bot_y == pytest.approx(target_floor, abs=1e-6)
            assert top_y == pytest.approx(target_floor + 2.50, abs=1e-6)


def test_rejects_pair_with_floor_delta_over_threshold():
    """7 cm floor delta → no snap, even with matching wall heights."""
    a = _rect_room(story=0, floor_y=0.00, wall_height=2.50)
    b = _rect_room(story=0, floor_y=0.07, wall_height=2.50, x0=5.0, x1=9.0)

    metrics = align_room_heights([a, b])
    assert metrics["aligned_groups"] == 0
    assert _room_floor_y(a) == pytest.approx(0.00)
    assert _room_floor_y(b) == pytest.approx(0.07)


def test_rejects_pair_with_wall_height_delta_over_threshold():
    """Floor slabs close, but wall heights differ by > 5 cm → no snap."""
    a = _rect_room(story=0, floor_y=0.00, wall_height=2.50)
    b = _rect_room(
        story=0, floor_y=0.03, wall_height=2.60, x0=5.0, x1=9.0
    )  # 10 cm taller

    metrics = align_room_heights([a, b])
    assert metrics["aligned_groups"] == 0
    assert _room_floor_y(a) == pytest.approx(0.00)
    assert _room_floor_y(b) == pytest.approx(0.03)


def test_no_chaining_across_spread_cap():
    """Three rooms at 0.00 / 0.05 / 0.11 must not chain into one 11 cm group.

    The first two are within 6 cm (5 cm delta) and should group; adding the
    third would push the spread to 11 cm, so the sweep must start a fresh group.
    """
    a = _rect_room(story=0, floor_y=0.00, wall_height=2.50, x0=0.0, x1=4.0)
    b = _rect_room(story=0, floor_y=0.05, wall_height=2.50, x0=5.0, x1=9.0)
    c = _rect_room(story=0, floor_y=0.11, wall_height=2.50, x0=10.0, x1=14.0)

    metrics = align_room_heights([a, b, c])
    assert metrics["aligned_groups"] == 1
    assert metrics["aligned_rooms"] == 2

    target_ab = 0.025
    assert _room_floor_y(a) == pytest.approx(target_ab, abs=1e-6)
    assert _room_floor_y(b) == pytest.approx(target_ab, abs=1e-6)
    assert _room_floor_y(c) == pytest.approx(0.11, abs=1e-6)


def test_knee_wall_preserved():
    """A short interior wall (top 0.5 m below siblings) must not get lifted.

    Its top is more than ``CORNER_SNAP_TOL_M`` below the wall's own max Y, so
    the per-corner snap skips it even when the room as a whole is re-slabbed.
    """
    a = _rect_room(story=0, floor_y=0.00, wall_height=2.50)
    b = _rect_room(story=0, floor_y=0.04, wall_height=2.50, x0=5.0, x1=9.0)
    # Knee wall in room A: tall enough to contribute a Y range, but its own
    # max Y is 2.00 — so CORNER_SNAP_TOL_M (5 cm) from max is 1.95..2.05,
    # far below the room-level ceiling of 2.50.
    knee = _wall_quad(1.0, 1.0, 2.0, 1.0, bot_y=0.00, top_y=2.00)
    knee["id"] = "knee"
    a["walls_computed"].append(knee)

    align_room_heights([a, b])

    assert a["walls_computed"][-1]["id"] == "knee"
    knee_bot, knee_top = _wall_min_max(a, wall_idx=len(a["walls_computed"]) - 1)
    # Bottom sat at the room's floor line, so it follows the snap to 0.02.
    assert knee_bot == pytest.approx(0.02, abs=1e-6)
    # Top at 2.00 is far below the room's 2.50 ceiling line, so it stays put.
    assert knee_top == pytest.approx(2.00, abs=1e-6)


def test_lineage_records_on_modified_elements():
    a = _rect_room(story=0, floor_y=0.00, wall_height=2.50)
    b = _rect_room(story=0, floor_y=0.04, wall_height=2.50, x0=5.0, x1=9.0)

    align_room_heights([a, b])

    for room in (a, b):
        entries = [
            e for e in room.get("lineage", []) if e["step"] == STEP_ALIGN_ROOM_HEIGHTS
        ]
        assert len(entries) == 1
        assert "floor_y=" in entries[0]["detail"]
        for wall in room["walls_computed"]:
            wentries = [
                e
                for e in wall.get("lineage", [])
                if e["step"] == STEP_ALIGN_ROOM_HEIGHTS
            ]
            assert len(wentries) == 1
            assert "bot=" in wentries[0]["detail"]
            assert "top=" in wentries[0]["detail"]


def test_different_stories_never_group():
    """Two rooms at identical Ys on different stories must stay independent."""
    a = _rect_room(story=0, floor_y=3.00, wall_height=2.50)
    b = _rect_room(story=1, floor_y=3.00, wall_height=2.50, x0=5.0, x1=9.0)

    metrics = align_room_heights([a, b])
    assert metrics["aligned_groups"] == 0


def test_constants_match_plan():
    """Guard against accidental drift of the 6/5 cm thresholds."""
    assert FLOOR_ALIGN_TOL_M == pytest.approx(0.06)
    assert WALL_HEIGHT_ALIGN_TOL_M == pytest.approx(0.05)
    assert CORNER_SNAP_TOL_M == pytest.approx(0.05)
