import pytest
from shapely.geometry import Polygon

from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)
from reconcile_tiers.extract.overlaps import clip_floor_overlaps


def _wall(
    wall_id: str,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    floor_y: float = 0.0,
    top_y: float = 2.5,
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


def _room(
    index: int,
    floor: list[tuple[float, float]],
    walls: list[ExtractedWall],
    *,
    doors: list[ExtractedElement] | None = None,
    windows: list[ExtractedElement] | None = None,
) -> ExtractedRoom:
    return ExtractedRoom(
        index=index,
        story=0,
        floor_polygon=[[x, 0.0, z] for x, z in floor],
        walls_merged=[],
        walls_computed=walls,
        doors=doors or [],
        windows=windows or [],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )


def _door(
    door_id: str,
    center: tuple[float, float],
    *,
    parent_wall_id: str | None,
) -> ExtractedElement:
    x, z = center
    return ExtractedElement(
        id=door_id,
        source="test",
        parent_wall_id=parent_wall_id,
        corners=[
            [x - 0.35, 0.0, z],
            [x + 0.35, 0.0, z],
            [x + 0.35, 2.0, z],
            [x - 0.35, 2.0, z],
        ],
    )


def test_overlap_clipping_preserves_wall_on_surviving_floor_boundary():
    # The large room claims a thin overlap strip first. The smaller room's
    # wall lies on the post-clip boundary, so removing the whole wall leaves
    # an impossible floor edge with no boundary surface.
    large = _room(
        0,
        [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        [_wall("winner", (4.0, 0.0), (4.0, 4.0))],
    )
    small = _room(
        1,
        [(3.9, 0.0), (6.0, 0.0), (6.0, 3.0), (3.9, 3.0)],
        [_wall("surviving-boundary", (4.0, 0.0), (4.0, 3.0))],
    )

    clipped = clip_floor_overlaps([large, small])

    small_after = clipped[1]
    floor_poly = Polygon([(point[0], point[2]) for point in small_after.floor_polygon])

    assert floor_poly.area == pytest.approx(6.0)
    assert [wall.id for wall in small_after.walls_computed] == ["surviving-boundary"]


def test_overlap_clipping_keeps_door_when_parent_wall_stays():
    # The smaller room loses its left strip to the larger room. The door's
    # midpoint is inside the winner polygon, but its parent wall remains in
    # the losing room, so the door stays with its wall.
    large = _room(
        0,
        [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        [_wall("winner-wall", (4.0, 0.0), (4.0, 4.0))],
    )
    small = _room(
        1,
        [(3.0, 1.0), (5.0, 1.0), (5.0, 3.0), (3.0, 3.0)],
        [_wall("door-parent", (3.5, 1.0), (3.5, 3.0))],
        doors=[_door("door", (3.5, 2.0), parent_wall_id="door-parent")],
    )

    clipped = clip_floor_overlaps([large, small])

    assert clipped[0].doors == []
    assert [door.id for door in clipped[1].doors] == ["door"]


def test_overlap_clipping_does_not_move_door_without_parent_wall():
    # If the door has a stale parent id and no room owns that wall after
    # clipping, midpoint containment alone must not move it to the winner.
    # It remains in the source room so openings are never silently lost.
    large = _room(
        0,
        [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        [_wall("winner-wall", (4.0, 0.0), (4.0, 4.0))],
    )
    small = _room(
        1,
        [(3.0, 1.0), (5.0, 1.0), (5.0, 3.0), (3.0, 3.0)],
        [_wall("other-wall", (5.0, 1.0), (5.0, 3.0))],
        doors=[_door("door", (3.5, 2.0), parent_wall_id="missing-parent")],
    )

    clipped = clip_floor_overlaps([large, small])

    assert clipped[0].doors == []
    assert [door.id for door in clipped[1].doors] == ["door"]


def test_overlap_clipping_moves_door_when_winner_owns_parent_wall():
    # Same geometry, but the winner already owns the door's parent wall. In
    # that case moving the door with that wall is physically coherent.
    large = _room(
        0,
        [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        [_wall("door-parent", (3.5, 1.0), (3.5, 3.0))],
    )
    small = _room(
        1,
        [(3.0, 1.0), (5.0, 1.0), (5.0, 3.0), (3.0, 3.0)],
        [_wall("other-wall", (5.0, 1.0), (5.0, 3.0))],
        doors=[_door("door", (3.5, 2.0), parent_wall_id="door-parent")],
    )

    clipped = clip_floor_overlaps([large, small])

    assert [door.id for door in clipped[0].doors] == ["door"]
    assert clipped[1].doors == []
