from __future__ import annotations

from typing import TYPE_CHECKING

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from reconcile_tiers.extract.building import BuildingModel, ExtractedRoom

STITCH_FOOTPRINT_HULL_BUFFER_M = 0.03
STITCH_MAX_ROOM_SUPPORT_DISTANCE_M = 0.50


def story_footprint_context(
    rooms: list[ExtractedRoom] | BuildingModel,
) -> dict[int, dict]:
    """Per-story support polygons for stitch validation.

    Accepts either a BuildingModel (uses ``.rooms``) or a plain list of rooms,
    so the same helper serves the construction stage (``stitch_wall_gaps`` —
    rooms only) and the assembly stage (``assemble_gap_pieces`` — full model)."""

    room_iter = getattr(rooms, "rooms", rooms)
    by_story: dict[int, list] = {}
    for room in room_iter:
        if len(room.floor_polygon) < 3:
            continue
        try:
            poly = Polygon(
                [(float(point[0]), float(point[2])) for point in room.floor_polygon]
            )
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-9:
            continue
        by_story.setdefault(room.story, []).append(poly)

    context: dict[int, dict] = {}
    for story, polys in by_story.items():
        union = unary_union(polys)
        if union.is_empty:
            continue
        context[story] = {
            "union": union,
            "hull": union.convex_hull.buffer(STITCH_FOOTPRINT_HULL_BUFFER_M),
        }
    return context


def is_xz_supported_by_story_footprint(
    x: float, z: float, footprint_context: dict | None
) -> bool:
    if footprint_context is None:
        return True
    point = Point(float(x), float(z))
    if not footprint_context["hull"].covers(point):
        return False
    return (
        float(point.distance(footprint_context["union"]))
        <= STITCH_MAX_ROOM_SUPPORT_DISTANCE_M
    )


def is_stitch_edge_unsupported_by_story_footprint(
    corners: list[list[float]], footprint_context: dict | None
) -> bool:
    if footprint_context is None or len(corners) < 2:
        return False
    line = LineString(
        [
            (float(corners[0][0]), float(corners[0][2])),
            (float(corners[1][0]), float(corners[1][2])),
        ]
    )
    if line.length <= 1e-9:
        return False
    midpoint = line.interpolate(0.5, normalized=True)
    if not footprint_context["hull"].covers(midpoint):
        return True
    return (
        float(midpoint.distance(footprint_context["union"]))
        > STITCH_MAX_ROOM_SUPPORT_DISTANCE_M
    )
