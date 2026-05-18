"""Last-step adjacency tagging -- splits envelope elements at exposure
boundaries and tags each uniform region with an `AdjacencyKind`.

Boundaries handled in v1:
- Basement walls split horizontally at terrain_y (above-grade vs below-grade).
- Walls split along their length where adjacent same-story rooms share part
  of the wall's footprint (interior partition vs exterior).
- Top-story flat-lid ceilings split at the XZ outline of overlying obliques
  (under-sky vs under-unheated-attic).
- Story-1 floors above a partly unheated basement split at the basement
  footprint outline.

Locator IDs of split pieces gain a `/<index>` suffix when more than one
piece is emitted from a single source element.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.assemble.walls_to_rooms import dedup_room_walls
from reconcile_tiers.payload.schema import (
    AdjacencyKind,
    CeilingPiece,
    CeilingSource,
    GapKind,
    GapPiece,
    GapScope,
    HorizontalLid,
    Room,
    TierPayload,
    Vec3,
    Wall,
)

LOGGER = logging.getLogger(__name__)

BASEMENT_DEEP_THRESHOLD_M = 2.0
WALL_OUTWARD_STEP_M = 0.10
STORY_BRACKET_TOL_M = 0.10
SPLIT_AREA_EPS_M2 = 1e-4
UFH_HEATING_VALUES = frozenset({"floorHeating", "radiatorsAndFloorHeating"})
UNHEATED_HEATING_VALUE = "unheated"
FLAT_LID_SOURCES = frozenset(
    {
        CeilingSource.FLAT_EMIT,
        CeilingSource.FLAT_CEILING,
        CeilingSource.ATTIC_FLAT_LID,
        CeilingSource.ROOF_ARRANGEMENT_ATTIC,
        CeilingSource.FLAT_GAP_CLOSURE,
        CeilingSource.THERMAL_CAP,
        CeilingSource.RAW_FALLBACK,
        CeilingSource.RAW_SCAN,
    }
)
SLOPED_LID_SOURCES = frozenset(
    {
        CeilingSource.ROOF_ARRANGEMENT,
        CeilingSource.COMPUTED_OBLIQUE,
    }
)


def _is_ufh(room: Room) -> bool:
    return room.heating in UFH_HEATING_VALUES


def _is_unheated(room: Room) -> bool:
    return room.heating == UNHEATED_HEATING_VALUE


def _xz_polygon(corners: list[Vec3]) -> Polygon | None:
    if len(corners) < 3:
        return None
    poly = Polygon([(c.x, c.z) for c in corners])
    if not poly.is_valid:
        poly = make_valid(poly)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _xz_polygon_from_xyz(corners: list[list[float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
    if not poly.is_valid:
        poly = make_valid(poly)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _xz_centroid(corners: list[Vec3]) -> Point | None:
    if not corners:
        return None
    n = len(corners)
    return Point(
        sum(c.x for c in corners) / n,
        sum(c.z for c in corners) / n,
    )


def _mean_y(corners: list[Vec3]) -> float:
    return sum(c.y for c in corners) / len(corners) if corners else 0.0


def _closest_story(y: float, story_mean_ys: list[float]) -> int | None:
    if not story_mean_ys:
        return None
    return min(range(len(story_mean_ys)), key=lambda s: abs(story_mean_ys[s] - y))


def _polys_to_polygon_list(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > SPLIT_AREA_EPS_M2 else []
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            if isinstance(g, Polygon) and g.area > SPLIT_AREA_EPS_M2:
                out.append(g)
        return out
    return []


def _polygon_to_xz_vec3(poly: Polygon, y: float) -> list[Vec3]:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [Vec3(float(x), float(y), float(z)) for x, z in coords]


def _polygon_to_plane_vec3(poly: Polygon, plane) -> list[Vec3]:
    """Lift an XZ polygon onto the supplied `Plane` so each corner satisfies
    a*x + b*y + c*z + d = 0 within numerical tolerance.
    """
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if abs(plane.b) < 1e-9:
        # Degenerate: fall back to a flat lid at z=0.
        return [Vec3(float(x), 0.0, float(z)) for x, z in coords]
    return [
        Vec3(float(x), float((plane.d - plane.a * x - plane.c * z) / plane.b), float(z))
        for x, z in coords
    ]


# ---------------------------------------------------------------------------
# Per-region tag evaluation (no geometry -- given a representative point and the
# host context, returns the AdjacencyKind that applies there).
# ---------------------------------------------------------------------------


def _eval_wall_tag_at(
    *,
    x: float,
    z: float,
    y: float,
    host_room: Room,
    has_basement: bool,
    terrain_y: float,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
) -> AdjacencyKind:
    if _is_unheated(host_room):
        return AdjacencyKind.INTERNAL_TO_UNHEATED_HOST

    point = Point(x, z)
    for poly, other in room_polys_by_story.get(host_room.story, []):
        if other is host_room:
            continue
        if poly.covers(point):
            return (
                AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
                if _is_unheated(other)
                else AdjacencyKind.INTERNAL_TO_HEATED
            )

    if host_room.story == 0 and has_basement and y < terrain_y - 1e-6:
        depth = max(0.0, terrain_y - y)
        return (
            AdjacencyKind.BASEMENT_WALL_GROUND_SHALLOW
            if depth <= BASEMENT_DEEP_THRESHOLD_M
            else AdjacencyKind.BASEMENT_WALL_GROUND_DEEP
        )

    return AdjacencyKind.EXTERNAL_AIR


def _eval_floor_tag_at(
    *,
    x: float,
    z: float,
    host_room: Room,
    has_basement: bool,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
) -> AdjacencyKind:
    if _is_unheated(host_room):
        return AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
    if host_room.story == 0:
        return (
            AdjacencyKind.GROUND_SLAB_UFH
            if _is_ufh(host_room)
            else AdjacencyKind.GROUND_SLAB
        )
    if host_room.story == 1 and has_basement:
        point = Point(x, z)
        for poly, basement_room in room_polys_by_story.get(0, []):
            if poly.covers(point):
                return (
                    AdjacencyKind.UNHEATED_BASEMENT_FLOOR
                    if _is_unheated(basement_room)
                    else AdjacencyKind.INTERNAL_TO_HEATED
                )
        # Floor footprint extends beyond any basement room: it's on grade.
        return (
            AdjacencyKind.GROUND_SLAB_UFH
            if _is_ufh(host_room)
            else AdjacencyKind.GROUND_SLAB
        )
    return AdjacencyKind.INTERNAL_TO_HEATED


def _eval_ceiling_tag_at(
    *,
    x: float,
    z: float,
    piece_source: CeilingSource,
    host_room: Room | None,
    top_story_idx: int,
    oblique_xz_polys: list[Polygon],
) -> AdjacencyKind:
    if host_room is not None and _is_unheated(host_room):
        return AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
    if host_room is not None and host_room.story < top_story_idx:
        return AdjacencyKind.INTERNAL_TO_HEATED
    point = Point(x, z)
    is_oblique_above = any(p.covers(point) for p in oblique_xz_polys)
    if is_oblique_above:
        if piece_source in SLOPED_LID_SOURCES:
            return AdjacencyKind.EXTERNAL_AIR
        return AdjacencyKind.UNHEATED_ATTIC
    return AdjacencyKind.EXTERNAL_AIR


# ---------------------------------------------------------------------------
# Splitting primitives.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gap pieces (no geometric splitting in v1; uniform tag per piece).
# ---------------------------------------------------------------------------


def _gap_inter_story_tag(
    centroid: Point,
    gap_y: float,
    story_mean_ys: list[float],
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
) -> AdjacencyKind:
    above_idx: int | None = None
    below_idx: int | None = None
    for s, mean_y in enumerate(story_mean_ys):
        if mean_y > gap_y + STORY_BRACKET_TOL_M:
            if above_idx is None or mean_y < story_mean_ys[above_idx]:
                above_idx = s
        elif mean_y < gap_y - STORY_BRACKET_TOL_M:
            if below_idx is None or mean_y > story_mean_ys[below_idx]:
                below_idx = s

    def covered_by_heated(story_idx: int | None) -> bool:
        if story_idx is None:
            return False
        for poly, room in room_polys_by_story.get(story_idx, []):
            if not _is_unheated(room) and poly.covers(centroid):
                return True
        return False

    if covered_by_heated(above_idx) and covered_by_heated(below_idx):
        return AdjacencyKind.INTERNAL_TO_HEATED
    return AdjacencyKind.EXTERNAL_AIR


def _tag_gap(
    gap: GapPiece,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
    story_mean_ys: list[float],
    oblique_xz_polys: list[Polygon],
) -> AdjacencyKind:
    centroid = _xz_centroid(gap.corners)
    gap_y = _mean_y(gap.corners)

    if gap.kind == GapKind.EXTERIOR_SIDE:
        return AdjacencyKind.EXTERNAL_AIR

    if gap.kind == GapKind.EXTERIOR_FLOOR:
        story_idx = _closest_story(gap_y, story_mean_ys)
        return (
            AdjacencyKind.GROUND_SLAB if story_idx == 0 else AdjacencyKind.EXTERNAL_AIR
        )

    if gap.kind == GapKind.EXTERIOR_CEIL:
        if centroid is not None and any(p.covers(centroid) for p in oblique_xz_polys):
            return AdjacencyKind.UNHEATED_ATTIC
        return AdjacencyKind.EXTERNAL_AIR

    if gap.kind in (GapKind.GAP_FLOOR, GapKind.GAP_CEILING):
        if centroid is None:
            return AdjacencyKind.UNKNOWN
        if gap.scope == GapScope.INTER_STORY:
            return _gap_inter_story_tag(
                centroid, gap_y, story_mean_ys, room_polys_by_story
            )
        if gap.scope == GapScope.INTRA_STORY:
            story_idx = _closest_story(gap_y, story_mean_ys)
            if story_idx is not None:
                for poly, room in room_polys_by_story.get(story_idx, []):
                    if poly.covers(centroid):
                        return (
                            AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
                            if _is_unheated(room)
                            else AdjacencyKind.INTERNAL_TO_HEATED
                        )
            return AdjacencyKind.EXTERNAL_AIR
        return AdjacencyKind.UNKNOWN

    if gap.kind == GapKind.SIDE:
        if gap.scope == GapScope.INTER_STORY:
            return AdjacencyKind.EXTERNAL_AIR
        if gap.scope == GapScope.INTRA_STORY:
            if centroid is None:
                return AdjacencyKind.UNKNOWN
            coords = [[c.x, c.y, c.z] for c in gap.corners]
            nx, _ny, nz = newell_normal(coords)
            nxz_len = (nx * nx + nz * nz) ** 0.5
            story_idx = _closest_story(gap_y, story_mean_ys)
            if story_idx is None:
                return AdjacencyKind.UNKNOWN
            candidates: list[Point] = []
            if nxz_len > 1e-9:
                for sign in (1.0, -1.0):
                    candidates.append(
                        Point(
                            centroid.x + sign * (nx / nxz_len) * WALL_OUTWARD_STEP_M,
                            centroid.y + sign * (nz / nxz_len) * WALL_OUTWARD_STEP_M,
                        )
                    )
            for stepped in candidates:
                for poly, room in room_polys_by_story.get(story_idx, []):
                    if poly.covers(stepped):
                        return (
                            AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
                            if _is_unheated(room)
                            else AdjacencyKind.INTERNAL_TO_HEATED
                        )
            return AdjacencyKind.EXTERNAL_AIR
        return AdjacencyKind.UNKNOWN

    return AdjacencyKind.UNKNOWN


# ---------------------------------------------------------------------------
# Ceiling host-room resolution (centroid-based, story-aware).
# ---------------------------------------------------------------------------


def _resolve_ceiling_host(
    piece: CeilingPiece,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
    story_mean_ys: list[float],
) -> Room | None:
    centroid = _xz_centroid(piece.corners)
    if centroid is None:
        return None
    ceiling_y = _mean_y(piece.corners)
    candidates = sorted(
        (
            s
            for s, mean_y in enumerate(story_mean_ys)
            if mean_y <= ceiling_y + STORY_BRACKET_TOL_M
        ),
        key=lambda s: abs(story_mean_ys[s] - ceiling_y),
    )
    for s in candidates:
        for poly, room in room_polys_by_story.get(s, []):
            if poly.covers(centroid):
                return room
    return None


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def _suffix_locator(base: str, idx: int, total: int) -> str:
    return base if total <= 1 else f"{base}/{idx}"


def _room_xz_union(room: Room) -> Polygon | None:
    polys: list[Polygon] = []
    for piece in room.floor:
        poly = _xz_polygon(piece.corners)
        if poly is not None:
            polys.append(poly)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    union = unary_union(polys)
    if isinstance(union, Polygon):
        return union
    if hasattr(union, "geoms"):
        # MultiPolygon -- return convex hull as a fallback (rare; keeps
        # downstream point-in-polygon tests well-defined).
        return union.convex_hull if not union.is_empty else None
    return None


def tag_payload(
    payload: TierPayload,
    *,
    has_basement: bool,
    oblique_surface_corners: list[list[list[float]]],
) -> TierPayload:
    """Split each envelope element at exposure boundaries and tag every
    resulting region with an `AdjacencyKind`. Pure: returns a new payload.
    """
    rooms = payload.rooms
    if not rooms:
        return payload

    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]] = {}
    for room in rooms:
        poly = _room_xz_union(room)
        if poly is not None:
            room_polys_by_story.setdefault(room.story, []).append((poly, room))

    max_story = max(r.story for r in rooms)
    story_mean_ys: list[float] = []
    for s in range(max_story + 1):
        ys = [
            _mean_y(piece.corners)
            for r in rooms
            if r.story == s
            for piece in r.floor
            if piece.corners
        ]
        story_mean_ys.append(sum(ys) / len(ys) if ys else 0.0)

    if has_basement and len(story_mean_ys) >= 2:
        terrain_y = story_mean_ys[1]
    elif story_mean_ys:
        terrain_y = story_mean_ys[0]
    else:
        terrain_y = 0.0

    oblique_xz_polys = [
        poly
        for corners in oblique_surface_corners
        if (poly := _xz_polygon_from_xyz(corners)) is not None
    ]

    top_story_idx = max_story

    new_rooms: list[Room] = []
    for room in rooms:
        # Floors
        floor_pieces: list[HorizontalLid] = []
        for orig_floor in room.floor:
            splits = _split_floor(
                orig_floor,
                room,
                has_basement=has_basement,
                room_polys_by_story=room_polys_by_story,
            )
            for corners, tag in splits:
                if len(corners) < 3:
                    continue
                floor_pieces.append(replace(orig_floor, corners=corners, adjacency=tag))
        if not floor_pieces:
            floor_pieces = list(room.floor)

        # Walls
        wall_pieces: list[Wall] = []
        for orig_wall in room.walls:
            splits = _split_wall(
                orig_wall,
                room,
                has_basement=has_basement,
                terrain_y=terrain_y,
                room_polys_by_story=room_polys_by_story,
            )
            for idx, (corners, tag, band) in enumerate(splits):
                if len(corners) < 3:
                    continue
                piece_cutouts = _clip_cutouts_to_band(orig_wall.cutouts, band)
                wall_pieces.append(
                    replace(
                        orig_wall,
                        corners=corners,
                        adjacency=tag,
                        cutouts=piece_cutouts,
                        locator_id=_suffix_locator(
                            orig_wall.locator_id, idx, len(splits)
                        ),
                    )
                )

        new_rooms.append(replace(room, floor=floor_pieces, walls=wall_pieces))

    new_rooms = dedup_room_walls(new_rooms, cross_room=False)

    new_ceilings: list[CeilingPiece] = []
    for orig_piece in payload.ceiling:
        host = _resolve_ceiling_host(orig_piece, room_polys_by_story, story_mean_ys)
        splits = _split_ceiling(
            orig_piece,
            host,
            top_story_idx=top_story_idx,
            oblique_xz_polys=oblique_xz_polys,
        )
        for idx, (corners, tag) in enumerate(splits):
            if len(corners) < 3:
                continue
            new_ceilings.append(
                replace(
                    orig_piece,
                    corners=corners,
                    holes=_holes_contained_by_corners(orig_piece, corners),
                    adjacency=tag,
                    locator_id=_suffix_locator(orig_piece.locator_id, idx, len(splits)),
                )
            )

    new_knee_walls = [
        replace(kw, adjacency=AdjacencyKind.UNHEATED_ATTIC) for kw in payload.knee_walls
    ]
    new_dormer_faces = [
        replace(df, adjacency=AdjacencyKind.EXTERNAL_AIR) for df in payload.dormer_faces
    ]
    new_gable_closures = [
        replace(gc, adjacency=AdjacencyKind.EXTERNAL_AIR)
        for gc in payload.gable_closures
    ]

    new_gaps = [
        replace(
            gap,
            adjacency=_tag_gap(
                gap, room_polys_by_story, story_mean_ys, oblique_xz_polys
            ),
        )
        for gap in payload.gaps
    ]

    return replace(
        payload,
        rooms=new_rooms,
        gaps=new_gaps,
        ceiling=new_ceilings,
        knee_walls=new_knee_walls,
        dormer_faces=new_dormer_faces,
        gable_closures=new_gable_closures,
    )


# Re-exports — wall/floor/ceiling splitting at adjacency boundaries lives in
# `_adjacency_split`. `tag_payload` (above) calls these.
from reconcile_tiers.payload._adjacency_split import (  # noqa: E402, F401
    _clip_against_y,
    _clip_cutouts_to_band,
    _dedup_consecutive_vec3,
    _holes_contained_by_corners,
    _slice_wall_by_y_band,
    _split_ceiling,
    _split_floor,
    _split_wall,
    _tag_whole_wall,
)
