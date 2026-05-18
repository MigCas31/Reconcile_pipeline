from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import (
    make_valid,
    oriented_rectangle_side_lengths,
)
from reconcile_tiers.extract.building import ExtractedRoom, ExtractedWall
from reconcile_tiers.extract.gaps import decompose_polys, floor_polygon_to_shapely

MIN_INFERRED_ROOM_AREA_M2 = 0.50
MIN_INFERRED_ROOM_WIDTH_M = 0.50
MIN_ADJACENT_ROOMS = 2
MIN_TOUCH_LENGTH_M = 0.05


def _floor_y(room: ExtractedRoom) -> float | None:
    if not room.floor_polygon:
        return None
    return float(np.mean([corner[1] for corner in room.floor_polygon]))


def _wall_top_y(room: ExtractedRoom) -> float | None:
    tops: list[float] = []
    for wall in [*room.walls_computed, *room.walls_merged]:
        if wall.corners:
            tops.append(max(float(corner[1]) for corner in wall.corners))
    return float(np.median(tops)) if tops else None


def _dominant_heating(rooms: list[ExtractedRoom]) -> str | None:
    values = [room.heating for room in rooms if room.heating]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _is_room_scale_hole(hole: Polygon) -> bool:
    if hole.is_empty or not hole.is_valid or hole.area < MIN_INFERRED_ROOM_AREA_M2:
        return False
    sides = oriented_rectangle_side_lengths(hole)
    if not sides or min(sides) < MIN_INFERRED_ROOM_WIDTH_M:
        return False
    return True


def _polygon_to_floor(poly: Polygon, floor_y: float) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(x), float(floor_y), float(z)] for x, z in coords]


def _walls_for_floor(
    floor: list[list[float]],
    *,
    wall_y_min: float,
    wall_y_max: float,
    room_index: int,
) -> list[ExtractedWall]:
    walls: list[ExtractedWall] = []
    if len(floor) < 3:
        return walls
    for idx, corner in enumerate(floor):
        nxt = floor[(idx + 1) % len(floor)]
        walls.append(
            ExtractedWall(
                id=f"inferred-void-room-{room_index}-wall-{idx}",
                source="inferred-enclosed-void",
                synthetic=True,
                corners=[
                    [corner[0], wall_y_min, corner[2]],
                    [nxt[0], wall_y_min, nxt[2]],
                    [nxt[0], wall_y_max, nxt[2]],
                    [corner[0], wall_y_max, corner[2]],
                ],
            )
        )
    return walls


def infer_enclosed_void_rooms(rooms: list[ExtractedRoom]) -> list[ExtractedRoom]:
    """Create synthetic rooms for fully enclosed unscanned same-story voids.

    A room-scale interior ring in the union of same-story floor polygons is
    physical evidence of an omitted closet/small room, not external air. Adding
    it as a room lets downstream adjacency, gap, and stitch logic classify the
    surrounding walls as internal instead of synthesising closure artifacts.
    """

    if not rooms:
        return rooms

    next_room_index = max((room.index for room in rooms), default=-1) + 1
    rooms_by_story: dict[int, list[tuple[ExtractedRoom, Polygon]]] = defaultdict(list)
    for room in rooms:
        poly = floor_polygon_to_shapely(room.floor_polygon)
        if poly is None:
            continue
        rooms_by_story[room.story].append((room, poly))

    inferred: list[ExtractedRoom] = []
    for story, story_rooms in sorted(rooms_by_story.items()):
        if len(story_rooms) < MIN_ADJACENT_ROOMS:
            continue
        footprint = make_valid(unary_union([poly for _room, poly in story_rooms]))
        for part in decompose_polys(footprint):
            for ring in part.interiors:
                hole = Polygon(ring)
                if not hole.is_valid:
                    hole = make_valid(hole)
                if not isinstance(hole, Polygon) or not _is_room_scale_hole(hole):
                    continue

                adjacent = [
                    room
                    for room, room_poly in story_rooms
                    if hole.boundary.intersection(room_poly.boundary).length
                    >= MIN_TOUCH_LENGTH_M
                ]
                if len(adjacent) < MIN_ADJACENT_ROOMS:
                    continue

                floor_ys = [
                    value for room in adjacent if (value := _floor_y(room)) is not None
                ]
                if not floor_ys:
                    continue
                floor_y = float(np.median(floor_ys))
                wall_tops = [
                    value
                    for room in adjacent
                    if (value := _wall_top_y(room)) is not None
                ]
                wall_y_max = float(np.median(wall_tops)) if wall_tops else floor_y + 2.5
                if wall_y_max - floor_y < 0.5:
                    wall_y_max = floor_y + 2.5

                room_index = next_room_index
                next_room_index += 1
                floor = _polygon_to_floor(hole, floor_y)
                inferred.append(
                    ExtractedRoom(
                        index=room_index,
                        story=story,
                        floor_polygon=floor,
                        walls_merged=[],
                        walls_computed=_walls_for_floor(
                            floor,
                            wall_y_min=floor_y,
                            wall_y_max=wall_y_max,
                            room_index=room_index,
                        ),
                        doors=[],
                        windows=[],
                        openings=[],
                        storages=[],
                        raw_ceiling_planes=[],
                        raw_ceiling_source="inferred-enclosed-void",
                        ceiling_polygon=[
                            [corner[0], wall_y_max, corner[2]] for corner in floor
                        ],
                        ceiling_type="flat",
                        ceiling_eave_height=wall_y_max,
                        ceiling_ridge_height=wall_y_max,
                        synthetic_walls=[],
                        heating=_dominant_heating(adjacent),
                    )
                )

    return [*rooms, *inferred] if inferred else rooms


def maybe_infer_enclosed_void_rooms(rooms: list[ExtractedRoom]) -> list[ExtractedRoom]:
    """Back-compat wrapper for call sites that prefer an opt-out hook."""

    return infer_enclosed_void_rooms(rooms)
