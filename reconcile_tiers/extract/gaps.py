from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers._core.shapely2 import make_valid, oriented_rectangle_side_lengths
from reconcile_tiers.extract.building import (
    ExtractedGap,
    ExtractedRoom,
)

WALL_HALF_M = 0.25
PAIR_HALF_M = 0.50
MAX_GAP_M = 1.00
MIN_AREA_M2 = 0.01
MIN_THICKNESS_M = 0.01
# Drop cross_story pieces that are slim slivers (scan-noise eave fragments,
# misaligned-room artefacts). Real building zones — extensions, wings, attic
# eaves, lower wings missing an upper floor — are wider than this.
CROSS_STORY_MIN_THICKNESS_M = 0.50
HOLE_BRIDGE_WIDTH_M = 1e-4
MAX_HALF_FLOOR_M = 1.50
DEFAULT_WALL_HEIGHT_M = 2.50
MIN_WALL_HEIGHT_M = 0.50
MAX_SNAP_DIST_M = 1.0
MAX_Y_DIST_M = 0.75
MIN_SNAP_DIST_M = 1e-6
MAX_HORIZONTAL_CAP_Y_RANGE_M = 0.10
MAX_CEILING_CAP_INCLINATION_DEG = 80.0
CEILING_OVERHEAD_SLACK_M = 0.5
_GAP_WALL_TYPE_CONTRACTS = {
    (14, 19): {"within_story": 32, "gap_floor": 14, "gap_ceiling": 100},
    (15, 0): {"within_story": 42, "gap_floor": 15, "gap_ceiling": 115},
    (20, 0): {"within_story": 41, "gap_floor": 20, "gap_ceiling": 101},
}


def floor_polygon_to_shapely(floor_polygon: list[list[float]]) -> Polygon | None:
    if len(floor_polygon) < 3:
        return None
    coords = [(corner[0], corner[2]) for corner in floor_polygon]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = make_valid(Polygon(coords))
    except Exception:
        return None
    if not isinstance(poly, Polygon):
        parts = decompose_polys(poly)
        poly = max(parts, key=lambda part: part.area) if parts else None
    if poly is None or poly.is_empty or not poly.is_valid or poly.area < 0.01:
        return None
    return poly


def decompose_polys(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [item for item in getattr(geom, "geoms", []) if isinstance(item, Polygon)]


def _largest_valid_polygon(geom) -> Polygon | None:
    try:
        geom = make_valid(geom)
    except Exception:
        return None
    parts = [
        part
        for part in decompose_polys(geom)
        if part.is_valid and not part.is_empty and part.area >= MIN_AREA_M2
    ]
    if not parts:
        return None
    return max(parts, key=lambda part: part.area)


def _floor_from_shapely(poly: Polygon, floor_y: float) -> list[list[float]]:
    coords_2d = list(poly.exterior.coords)
    if coords_2d and coords_2d[0] == coords_2d[-1]:
        coords_2d = coords_2d[:-1]
    return [[float(coord[0]), floor_y, float(coord[1])] for coord in coords_2d]


def _sloped_room_ceiling_plane(room: ExtractedRoom) -> Plane | None:
    if room.ceiling_type != "sloped" or len(room.ceiling_polygon) < 3:
        return None
    ys = [float(corner[1]) for corner in room.ceiling_polygon]
    if max(ys) - min(ys) <= MAX_HORIZONTAL_CAP_Y_RANGE_M:
        return None
    plane = Plane.fit(room.ceiling_polygon)
    return plane if not isinstance(plane, FitFailure) else None


def _ceiling_corners_for_gap(
    gap: ExtractedGap,
    room: ExtractedRoom,
    fallback_y: float,
) -> list[list[float]]:
    plane = _sloped_room_ceiling_plane(room)
    corners = (
        gap.corners[:-1]
        if len(gap.corners) >= 4 and gap.corners[0] == gap.corners[-1]
        else gap.corners
    )
    ceiling = []
    for corner in corners:
        y = plane.y_at(corner[0], corner[2]) if plane is not None else None
        ceiling.append(
            [
                round(float(corner[0]), 4),
                round(float(y if y is not None else fallback_y), 4),
                round(float(corner[2]), 4),
            ]
        )
    if len(gap.corners) >= 4 and gap.corners[0] == gap.corners[-1] and ceiling:
        ceiling.append(list(ceiling[0]))
    return ceiling


def _emit_single_gap(
    gaps: list[ExtractedGap],
    part: Polygon,
    story: int,
    floor_y: float,
    gap_type: str,
    *,
    room_index: int | None = None,
) -> None:
    area = float(part.area)
    compactness = 4 * math.pi * area / (part.length**2) if part.length > 0 else 0.0
    if compactness < 0.15:
        confidence = "high"
    elif compactness < 0.30:
        confidence = "medium"
    else:
        confidence = "low"
    coords_2d = list(part.exterior.coords)
    centroid = part.centroid
    gaps.append(
        ExtractedGap(
            story=story,
            type=gap_type,
            corners=[[coord[0], floor_y, coord[1]] for coord in coords_2d],
            area_m2=round(area, 3),
            compactness=round(compactness, 3),
            confidence=confidence,
            centroid=[round(centroid.x, 3), floor_y, round(centroid.y, 3)],
            ceiling_corners=[],
            room_index=room_index,
        )
    )


def _piece_min_thickness(part: Polygon) -> float:
    sides = oriented_rectangle_side_lengths(part)
    return min(sides) if sides else 0.0


def _wing_polygons_for_rooms(rooms: list[ExtractedRoom]) -> list[Polygon]:
    """Decompose the building footprint (derived from rooms) into wing polygons.

    Mirrors `reconcile_tiers/roof/footprint.build_building_footprint` followed
    by `reconcile_tiers/_core/wing_decomposition.decompose_to_wings`, but skips
    the BuildingModel/uuid-cache plumbing because cross_story gap detection
    runs during model construction. Returns ``[]`` when no usable footprint
    can be derived; callers should fall back to a single-zone slice.
    """
    from reconcile_tiers._core.shapely2 import make_valid_polygon
    from reconcile_tiers._core.wing_decomposition import decompose_to_wings

    polygons = []
    for room in rooms:
        if len(room.floor_polygon) < 3:
            continue
        try:
            poly = Polygon([(float(p[0]), float(p[2])) for p in room.floor_polygon])
        except Exception:
            continue
        if not poly.is_valid:
            poly = make_valid_polygon(poly)
        if poly is None or poly.is_empty or poly.area <= 0.0:
            continue
        polygons.append(poly.buffer(0.3, join_style="mitre"))
    if not polygons:
        return []
    merged = make_valid(unary_union(polygons))
    parts = decompose_polys(merged)
    if not parts:
        return []
    largest = max(parts, key=lambda p: p.area)
    shrunk = largest.buffer(-0.3, join_style="mitre")
    if shrunk.is_empty:
        return []
    footprint_poly = make_valid_polygon(shrunk)
    if footprint_poly is None or footprint_poly.is_empty:
        return []
    wings = decompose_to_wings(footprint_poly)
    return [wing.polygon for wing in wings if not wing.polygon.is_empty]


def _split_holed_gap_part(part: Polygon) -> list[Polygon]:
    """Open interior rings with tiny bridges so emitted gap corners stay single-ring.

    Bridging via thin line buffers can fail when Shapely's tolerance leaves
    the hole nominally intact (the cut doesn't fully propagate). When that
    happens, fall back to the exterior only -- but only if holes are small
    relative to the part. If the part is mostly hole (a thin annulus around
    real coverage), the exterior would massively over-claim the void, so
    drop instead. The 0.5 ratio is what cleanly separates the legitimate
    void case (hole_ratio ~0.04) from the annulus case (hole_ratio ~0.97).
    """
    if not part.interiors:
        return [part]

    slots = []
    for interior in part.interiors:
        hole = Polygon(interior)
        if hole.is_empty:
            continue
        hole_point = hole.representative_point()
        exterior_point = part.exterior.interpolate(part.exterior.project(hole_point))
        if hole_point.distance(exterior_point) <= 1e-9:
            continue
        slots.append(
            LineString([hole_point, exterior_point]).buffer(
                HOLE_BRIDGE_WIDTH_M,
                cap_style="flat",
                join_style="mitre",
            )
        )

    if not slots:
        return [Polygon(part.exterior)]

    try:
        opened = make_valid(part.difference(unary_union(slots)))
    except Exception:
        opened = None

    single_ring = (
        [
            candidate
            for candidate in decompose_polys(opened)
            if candidate.is_valid
            and not candidate.is_empty
            and not candidate.interiors
            and candidate.area >= MIN_AREA_M2
        ]
        if opened is not None
        else []
    )
    if single_ring:
        return single_ring

    exterior_area = Polygon(part.exterior).area
    if exterior_area <= 0.0:
        return []
    hole_ratio = (exterior_area - part.area) / exterior_area
    if hole_ratio >= 0.5:
        return []
    return [Polygon(part.exterior)]


def _emit_gaps(
    gaps: list[ExtractedGap],
    regions,
    story: int,
    floor_y: float,
    gap_type: str,
    clip_to=None,
) -> None:
    for region in regions:
        if clip_to is not None:
            try:
                region = make_valid(region.intersection(clip_to))
            except Exception:
                continue
        for part in decompose_polys(region):
            if part.area < MIN_AREA_M2:
                continue
            for simple_part in _split_holed_gap_part(part):
                if simple_part.area < MIN_AREA_M2:
                    continue
                if _piece_min_thickness(simple_part) < MIN_THICKNESS_M:
                    continue
                _emit_single_gap(gaps, simple_part, story, floor_y, gap_type)


def _story_geometry(rooms: list[ExtractedRoom]):
    story_rooms_raw: dict[int, list[tuple[Polygon, float]]] = defaultdict(list)
    for room in rooms:
        poly = floor_polygon_to_shapely(room.floor_polygon)
        if poly is not None and poly.is_valid and poly.area > 0.01:
            floor_y = float(np.mean([corner[1] for corner in room.floor_polygon]))
            story_rooms_raw[room.story].append((poly, floor_y))

    story_rooms: dict[int, list[Polygon]] = defaultdict(list)
    story_floor_ys: dict[int, list[float]] = defaultdict(list)
    for story, entries in story_rooms_raw.items():
        median_y = float(np.median([floor_y for _poly, floor_y in entries]))
        for poly, floor_y in entries:
            if abs(floor_y - median_y) <= MAX_HALF_FLOOR_M:
                story_rooms[story].append(poly)
                story_floor_ys[story].append(floor_y)

    story_footprints = {}
    story_y_map = {}
    for story, polys in sorted(story_rooms.items()):
        footprint = make_valid(unary_union(polys))
        if footprint.area > 0.01:
            story_footprints[story] = footprint
            story_y_map[story] = float(np.mean(story_floor_ys[story]))
    return story_rooms, story_footprints, story_y_map


def _half_floor_footprint(story_footprints, story_y_map) -> tuple[set[int], Polygon]:
    stories_by_y = sorted(story_y_map.keys(), key=lambda story: story_y_map[story])
    half_floor_stories: set[int] = set()
    for idx in range(1, len(stories_by_y) - 1):
        story = stories_by_y[idx]
        dy_below = abs(story_y_map[story] - story_y_map[stories_by_y[idx - 1]])
        dy_above = abs(story_y_map[story] - story_y_map[stories_by_y[idx + 1]])
        if dy_below < MAX_HALF_FLOOR_M and dy_above < MAX_HALF_FLOOR_M:
            half_floor_stories.add(story)
    if not half_floor_stories:
        return half_floor_stories, Polygon()
    return half_floor_stories, make_valid(
        unary_union([story_footprints[story] for story in half_floor_stories])
    )


def compute_cross_floor_gaps(rooms: list[ExtractedRoom]) -> list[ExtractedGap]:
    story_rooms, story_footprints, story_y_map = _story_geometry(rooms)
    half_floor_stories, half_floor_fp = _half_floor_footprint(
        story_footprints, story_y_map
    )
    gaps: list[ExtractedGap] = []

    for story, polys in sorted(story_rooms.items()):
        if len(polys) < 2:
            continue
        footprint = story_footprints[story]
        floor_y = story_y_map[story]
        closed = make_valid(
            footprint.buffer(WALL_HALF_M, join_style=2).buffer(
                -WALL_HALF_M, join_style=2
            )
        )
        morph_gaps = make_valid(closed.difference(footprint))

        hole_gap_parts = []
        for poly_part in decompose_polys(closed):
            for interior in poly_part.interiors:
                hole = Polygon(interior)
                if hole.is_valid and hole.area > MIN_AREA_M2:
                    hole_gap_parts.append(hole)

        tree = STRtree(polys)
        buffered = [poly.buffer(PAIR_HALF_M, join_style=2) for poly in polys]
        pair_gap_parts = []
        seen_pairs = set()
        for idx, poly in enumerate(polys):
            for candidate in tree.query(buffered[idx]):
                other_idx = int(candidate)
                if other_idx <= idx:
                    continue
                pair_key = (idx, other_idx)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                if poly.distance(polys[other_idx]) > MAX_GAP_M:
                    continue
                try:
                    intersection = buffered[idx].intersection(buffered[other_idx])
                    gap = make_valid(intersection.difference(footprint))
                    pair_gap_parts.extend(decompose_polys(gap))
                except Exception:
                    continue

        wide_closed = make_valid(
            footprint.buffer(PAIR_HALF_M, join_style=2).buffer(
                -PAIR_HALF_M, join_style=2
            )
        )
        if not morph_gaps.is_empty:
            pair_neighborhoods = []
            for idx in range(len(polys)):
                for other_idx in range(idx + 1, len(polys)):
                    if polys[idx].distance(polys[other_idx]) > MAX_GAP_M:
                        continue
                    try:
                        neighborhood = buffered[idx].intersection(buffered[other_idx])
                    except Exception:
                        continue
                    if not neighborhood.is_empty:
                        pair_neighborhoods.append(neighborhood)
            covered = []
            covered_union = None
            for neighborhood in pair_neighborhoods:
                try:
                    chunk = make_valid(morph_gaps.intersection(neighborhood))
                    if covered_union is not None:
                        chunk = make_valid(chunk.difference(covered_union))
                except Exception:
                    continue
                if chunk.is_empty or chunk.area < MIN_AREA_M2:
                    continue
                covered.append(chunk)
                covered_union = (
                    chunk
                    if covered_union is None
                    else make_valid(unary_union([covered_union, chunk]))
                )
                _emit_gaps(
                    gaps, [chunk], story, floor_y, "within_story", clip_to=closed
                )
            if covered:
                try:
                    leftover = make_valid(morph_gaps.difference(covered_union))
                except Exception:
                    leftover = None
                if leftover is not None and not leftover.is_empty:
                    for part in decompose_polys(leftover):
                        if part.area >= MIN_AREA_M2:
                            _emit_gaps(
                                gaps,
                                [part],
                                story,
                                floor_y,
                                "within_story",
                                clip_to=closed,
                            )
            else:
                for part in decompose_polys(morph_gaps):
                    _emit_gaps(
                        gaps, [part], story, floor_y, "within_story", clip_to=closed
                    )

        for hole in hole_gap_parts:
            _emit_gaps(gaps, [hole], story, floor_y, "within_story")

        phase1_parts = list(decompose_polys(morph_gaps)) + list(hole_gap_parts)
        phase1_cover = make_valid(unary_union(phase1_parts)) if phase1_parts else None
        for pair_gap in pair_gap_parts:
            if phase1_cover is not None:
                try:
                    pair_gap = make_valid(pair_gap.difference(phase1_cover))
                except Exception:
                    pass
            if not pair_gap.is_empty:
                _emit_gaps(
                    gaps,
                    [pair_gap],
                    story,
                    floor_y,
                    "within_story",
                    clip_to=wide_closed,
                )

    sorted_stories = sorted(story_footprints.keys())
    if len(sorted_stories) >= 2:
        full_envelope = make_valid(
            unary_union([story_footprints[story] for story in sorted_stories])
        )
        wing_polys = _wing_polygons_for_rooms(rooms)
        zones = wing_polys if wing_polys else [full_envelope]
        for story in sorted_stories:
            footprint = story_footprints[story]
            floor_y = story_y_map[story]
            try:
                missing = make_valid(full_envelope.difference(footprint))
                if not half_floor_fp.is_empty and story not in half_floor_stories:
                    missing = make_valid(missing.difference(half_floor_fp))
            except Exception:
                continue
            if missing.is_empty:
                continue
            covered_union = None
            for zone in zones:
                try:
                    zone_missing = make_valid(missing.intersection(zone))
                    if covered_union is not None:
                        zone_missing = make_valid(
                            zone_missing.difference(covered_union)
                        )
                except Exception:
                    continue
                if zone_missing.is_empty or zone_missing.area < MIN_AREA_M2:
                    continue
                for piece in decompose_polys(zone_missing):
                    if piece.area < MIN_AREA_M2:
                        continue
                    if _piece_min_thickness(piece) < CROSS_STORY_MIN_THICKNESS_M:
                        continue
                    _emit_gaps(gaps, [piece], story, floor_y, "cross_story")
                covered_union = (
                    zone_missing
                    if covered_union is None
                    else make_valid(unary_union([covered_union, zone_missing]))
                )

    return gaps


def compute_room_ceiling_voids(
    rooms: list[ExtractedRoom],
    ceiling_corner_polygons: list[list[list[float]]],
    gable_oblique_xz=None,
) -> list[ExtractedGap]:
    """Detect each room's footprint area that has no overhead ceiling.

    Mirrors the V1 _check_ceiling_coverage signal: a ceiling polygon counts as
    "above" a room when its top Y is within CEILING_OVERHEAD_SLACK_M of the
    room's wall_y_max. The XZ remainder of the room footprint after subtracting
    the union of qualifying ceilings is emitted as a "room_ceiling_void" gap so
    compute_gap_walls can synthesise a closing cap and edge walls.

    gable_oblique_xz: optional XZ union of oblique roof surfaces. When provided,
    voids are clipped based on how much of the void overlaps the oblique footprint:
    >50% inside → suppress (oblique is the ceiling); partial overlap → keep only
    the inside portion; no overlap → leave as-is (legitimate gap).
    """
    gaps: list[ExtractedGap] = []
    if not rooms:
        return gaps

    ceiling_entries: list[tuple[float, Polygon]] = []
    for corners in ceiling_corner_polygons:
        if len(corners) < 3:
            continue
        try:
            poly = make_valid(Polygon([(float(c[0]), float(c[2])) for c in corners]))
        except Exception:
            continue
        if poly.is_empty or not poly.is_valid or poly.area <= MIN_AREA_M2:
            continue
        max_y = max(float(c[1]) for c in corners)
        ceiling_entries.append((max_y, poly))

    for room in rooms:
        if len(room.floor_polygon) < 3:
            continue
        floor_poly = floor_polygon_to_shapely(room.floor_polygon)
        if floor_poly is None:
            continue
        wall_ys = [
            float(corner[1]) for wall in room.walls_computed for corner in wall.corners
        ]
        if not wall_ys:
            continue
        wall_y_max = max(wall_ys)
        floor_y = float(np.mean([corner[1] for corner in room.floor_polygon]))
        candidates = [
            poly
            for max_y, poly in ceiling_entries
            if max_y >= wall_y_max - CEILING_OVERHEAD_SLACK_M
        ]
        if candidates:
            try:
                ceiling_union = make_valid(unary_union(candidates))
                void = make_valid(floor_poly.difference(ceiling_union))
            except Exception:
                continue
        else:
            void = floor_poly
        if void.is_empty:
            continue
        if gable_oblique_xz is not None and not gable_oblique_xz.is_empty:
            try:
                if room.ceiling_type == "sloped":
                    # Oblique IS the ceiling for this room: eave-corner artifacts
                    # sit inside the oblique footprint. Subtract to suppress.
                    void = make_valid(void.difference(gable_oblique_xz))
                else:
                    # Flat/unknown ceiling: route by fraction inside oblique.
                    void_area = float(void.area)
                    if void_area > 1e-9:
                        overlap = void.intersection(gable_oblique_xz)
                        overlap_area = float(overlap.area)
                        frac_in_oblique = overlap_area / void_area
                        if frac_in_oblique > 0.5:
                            # Void mostly inside oblique — oblique will paint it.
                            void = make_valid(void.difference(gable_oblique_xz))
                        elif overlap_area > 1e-6:
                            # Void straddles boundary (mostly outside): keep only
                            # the interior so gable-end protrusions are removed.
                            void = make_valid(overlap)
                        # frac=0: entirely outside oblique — legitimate gap.
            except Exception:
                pass
        if void.is_empty:
            continue
        for region in decompose_polys(void):
            if region.area < MIN_AREA_M2:
                continue
            for part in _split_holed_gap_part(region):
                if part.area < MIN_AREA_M2:
                    continue
                if _piece_min_thickness(part) < MIN_THICKNESS_M:
                    continue
                _emit_single_gap(
                    gaps,
                    part,
                    room.story,
                    floor_y,
                    "room_ceiling_void",
                    room_index=room.index,
                )
    return gaps


def assign_gaps_to_rooms(
    gaps: list[ExtractedGap],
    rooms: list[ExtractedRoom],
) -> tuple[list[ExtractedRoom], list[ExtractedGap]]:
    room_shapely = [
        (idx, floor_polygon_to_shapely(room.floor_polygon))
        for idx, room in enumerate(rooms)
    ]
    out_rooms = list(rooms)
    out_gaps: list[ExtractedGap] = []

    for gap in gaps:
        if gap.type != "within_story" or len(gap.corners) < 3:
            out_gaps.append(gap)
            continue
        gap_poly = floor_polygon_to_shapely(gap.corners)
        if gap_poly is None:
            out_gaps.append(gap)
            continue
        gap_centroid = gap_poly.centroid

        best_room_idx = None
        best_distance = float("inf")
        for room_idx, room_poly in room_shapely:
            if room_poly is None or out_rooms[room_idx].story != gap.story:
                continue
            distance = float(room_poly.distance(gap_centroid))
            if distance < best_distance:
                best_distance = distance
                best_room_idx = room_idx
        if best_room_idx is None:
            out_gaps.append(gap)
            continue

        assigned_room = out_rooms[best_room_idx]
        wall_top_ys = [
            max(corner[1] for corner in wall.corners)
            for wall in (assigned_room.walls_computed or assigned_room.walls_merged)
            if wall.corners
        ]
        ceiling_y = (
            round(float(np.median(wall_top_ys)), 4)
            if wall_top_ys
            else gap.corners[0][1]
        )
        ceiling_corners = _ceiling_corners_for_gap(gap, assigned_room, ceiling_y)

        room_poly = room_shapely[best_room_idx][1]
        if room_poly is None:
            out_gaps.append(
                replace(gap, room_index=best_room_idx, ceiling_corners=ceiling_corners)
            )
            continue
        floor_y = (
            assigned_room.floor_polygon[0][1]
            if assigned_room.floor_polygon
            else gap.corners[0][1]
        )
        merged = make_valid(unary_union([room_poly, gap_poly]))
        merged_poly = _largest_valid_polygon(merged)
        if merged_poly is None:
            out_gaps.append(
                replace(gap, room_index=best_room_idx, ceiling_corners=ceiling_corners)
            )
            continue

        floor_polygon = _floor_from_shapely(merged_poly, floor_y)
        floor_poly_check = floor_polygon_to_shapely(floor_polygon)
        if floor_poly_check is None:
            out_gaps.append(
                replace(gap, room_index=best_room_idx, ceiling_corners=ceiling_corners)
            )
            continue
        floor_polygon = _floor_from_shapely(floor_poly_check, floor_y)
        out_rooms[best_room_idx] = replace(
            assigned_room,
            floor_polygon=floor_polygon,
        )
        room_shapely[best_room_idx] = (best_room_idx, floor_poly_check)
        out_gaps.append(
            replace(gap, room_index=best_room_idx, ceiling_corners=ceiling_corners)
        )

    return out_rooms, out_gaps


def story_y_map_from_rooms(rooms: list[ExtractedRoom]) -> dict[int, float]:
    by_story: dict[int, list[float]] = defaultdict(list)
    for room in rooms:
        if room.floor_polygon:
            by_story[room.story].append(
                float(np.mean([corner[1] for corner in room.floor_polygon]))
            )
    return {
        story: float(np.median(values)) for story, values in by_story.items() if values
    }


# Re-exports — the gap-wall builder, its triangulation primitives, and stable
# ID/scoring helpers live in sibling modules. Existing callers import these
# names from `reconcile_tiers.extract.gaps`; preserve that surface.
from reconcile_tiers.extract._gap_triangulation import (  # noqa: E402, F401
    MIN_TRI_QUALITY,
    _apply_room_ceiling_fallback,
    _edge_on_room_boundary,
    _triangle_quality,
    _ytop_at_xz,
    earclip_2d,
)
from reconcile_tiers.extract._gap_walls import (  # noqa: E402, F401
    _dedupe_exact_gap_walls,
    _gap_wall_score,
    _limit_gap_walls_to_contract,
    _piece_index,
    _projected_xz_area,
    _stable_gap_anchor_id,
    _stable_gap_wall_id,
    _surface_inclination_deg,
    compute_gap_walls,
)
