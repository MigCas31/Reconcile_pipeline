from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
from shapely.geometry import Point, Polygon

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.config import architectural_priors_enabled
from reconcile_tiers.extract.building import ExtractedRoom, RawCeilingPlane

WALL_TOP_SPREAD_MAX_M = 0.20
CEILING_OVERSHOOT_MAX_M = 0.30
RIDGE_EAVE_MAX_DELTA_M = 0.10
NEIGHBOUR_TOLERANCE_M = 0.30
NOISY_PLANE_DROP_TAU_M = 0.30
SLANT_THRESH_M = 0.15
SLOPE_THRESH_M = 0.15
XZ_MATCH_TOL_M = 0.20
SOURCE_ROOM_CEILING_COVER_RATIO = 0.50
# Coplanar-merge thresholds. RoomPlan over-segments large planar surfaces
# (e.g. one attic slope captured as 2-3 patches that share an edge but get
# emitted as separate planes). Picked from a corpus histogram of pairwise
# normal angle vs max point-to-plane distance across 223 buildings: 3°/5cm
# is the knee that catches 326 (building, room) cases without sweeping in
# unrelated near-parallel patches. See .context/coplanar-audit/pairs.jsonl.
COPLANAR_NORMAL_ANGLE_DEG = 3.0
COPLANAR_OFFSET_M = 0.05
COPLANAR_MIN_NY = 0.10  # skip near-vertical planes -- can't lift back to 3D
# Veto the neighbour-consensus FLAT path when the room's scan shows both
# slanted wall tops AND inclined raw ceiling planes whose total XZ footprint
# meets this area. Bands out near-horizontal noise and near-vertical planes
# (walls mistagged as ceilings); cutoff picked from a corpus histogram that
# splits 11 noise-tier rooms (< 0.5 m^2) from 48 with real slope evidence.
INCLINED_CEILING_MIN_DEG = 5.0
INCLINED_CEILING_MAX_DEG = 60.0
INCLINED_CEILING_MIN_AREA_M2 = 0.5
KINK_SECOND_SLOPE_MIN_LARGEST_RATIO = 0.45
KINK_SECOND_SLOPE_MIN_AREA_M2 = 1.0
KINK_DISTINCT_SLOPE_MIN_NORMAL_ANGLE_DEG = 8.0
KINK_DISTINCT_SLOPE_MAX_XZ_OVERLAP_RATIO = 0.05
KINK_MAX_SLOPE_POLYGONS = 4

# Step-vs-slope discriminator. RoomPlan sometimes fits a small slanted plane
# over the corner between two flat ceiling levels (a stepped ceiling). Such a
# plane has a small Y range that exactly bridges the step height. Real attic
# slopes span far more Y (kneewall->ridge ≈ 0.5-2 m). Filter out slanted planes
# whose Y range is below this cap AND that geometrically bridge two flat
# levels in the same room.
KINK_STEP_BRIDGE_MAX_Y_RANGE_M = 0.50
KINK_STEP_FLAT_MATCH_TOL_M = 0.10
KINK_STEP_FLAT_LEVEL_SEPARATION_M = 0.05


def reassign_raw_ceiling_planes_spatially(
    rooms: list[ExtractedRoom],
) -> list[ExtractedRoom]:
    room_polys: list[Polygon | None] = []
    for room in rooms:
        floor = room.floor_polygon
        if len(floor) < 3:
            room_polys.append(None)
            continue
        poly = make_valid(Polygon([(corner[0], corner[2]) for corner in floor]))
        room_polys.append(poly if poly.is_valid and not poly.is_empty else None)

    reassignments: list[list[RawCeilingPlane]] = [[] for _ in rooms]
    for src_idx, room in enumerate(rooms):
        for plane in room.raw_ceiling_planes:
            corners = plane.corners
            if len(corners) < 3:
                reassignments[src_idx].append(plane)
                continue
            plane_poly = make_valid(
                Polygon([(corner[0], corner[2]) for corner in corners])
            )
            plane_y_span = max(corner[1] for corner in corners) - min(
                corner[1] for corner in corners
            )
            source_poly = room_polys[src_idx]
            if (
                source_poly is not None
                and plane_poly.is_valid
                and not plane_poly.is_empty
                and plane_y_span <= SLOPE_THRESH_M
                and not _has_slanted_wall_top(room)
                and source_poly.area > 0
                and (
                    float(source_poly.intersection(plane_poly).area)
                    / float(source_poly.area)
                )
                >= SOURCE_ROOM_CEILING_COVER_RATIO
            ):
                reassignments[src_idx].append(plane)
                continue
            cx = sum(corner[0] for corner in corners) / len(corners)
            cz = sum(corner[2] for corner in corners) / len(corners)
            centroid = Point(cx, cz)
            target_idx = None
            for room_idx, poly in enumerate(room_polys):
                if poly is None or rooms[room_idx].story != room.story:
                    continue
                if poly.contains(centroid):
                    target_idx = room_idx
                    break
            if target_idx is None:
                if plane_poly.is_valid and not plane_poly.is_empty:
                    best_area = 0.0
                    for room_idx, poly in enumerate(room_polys):
                        if poly is None or rooms[room_idx].story != room.story:
                            continue
                        area = float(poly.intersection(plane_poly).area)
                        if area > best_area:
                            best_area = area
                            target_idx = room_idx
            reassignments[src_idx if target_idx is None else target_idx].append(plane)

    return [
        replace(room, raw_ceiling_planes=reassignments[idx])
        for idx, room in enumerate(rooms)
    ]


def _wall_top_percentiles(room: ExtractedRoom) -> dict[str, float] | None:
    tops = []
    walls = room.walls_computed or room.walls_merged
    for wall in walls:
        ys = [corner[1] for corner in wall.corners]
        if ys:
            tops.append(max(ys))
    if len(tops) < 2:
        return None
    return {
        "p10": float(np.percentile(tops, 10)),
        "p50": float(np.percentile(tops, 50)),
        "p90": float(np.percentile(tops, 90)),
    }


def _max_raw_ceiling_corner_y(room: ExtractedRoom) -> float | None:
    best = None
    for plane in room.raw_ceiling_planes:
        for corner in plane.corners:
            if len(corner) >= 2 and (best is None or corner[1] > best):
                best = float(corner[1])
    return best


def _raw_plane_normal(corners: list[list[float]]) -> tuple[float, float, float] | None:
    if len(corners) < 3:
        return None
    nx = ny = nz = 0.0
    k = len(corners)
    for i in range(k):
        a = corners[i]
        b = corners[(i + 1) % k]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-9:
        return None
    return nx / norm, ny / norm, nz / norm


def _normal_angle_deg(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
    return math.degrees(math.acos(min(1.0, max(-1.0, dot))))


def _xz_polygon(corners: list[list[float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    poly = make_valid(Polygon([(corner[0], corner[2]) for corner in corners]))
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _raw_plane_incl_and_xz_area(
    corners: list[list[float]],
) -> tuple[float, float] | None:
    """Inclination (deg) and XZ-footprint area (m^2) for a raw ceiling plane.
    Returns None if the polygon is degenerate."""
    if len(corners) < 3:
        return None
    normal = _raw_plane_normal(corners)
    if normal is None:
        return None
    cos_inc = abs(normal[1])
    incl_deg = math.degrees(math.acos(min(1.0, cos_inc)))
    poly = _xz_polygon(corners)
    if poly is None:
        return None
    return incl_deg, float(poly.area)


def _inclined_raw_ceiling_area(room: ExtractedRoom) -> float:
    """Total XZ footprint area of raw ceiling planes whose inclination falls
    in [INCLINED_CEILING_MIN_DEG, INCLINED_CEILING_MAX_DEG]. Bands out
    near-horizontal noise and near-vertical planes (walls mistagged as
    ceilings)."""
    total = 0.0
    for plane in room.raw_ceiling_planes:
        info = _raw_plane_incl_and_xz_area(plane.corners)
        if info is None:
            continue
        incl_deg, area = info
        if incl_deg < INCLINED_CEILING_MIN_DEG or incl_deg > INCLINED_CEILING_MAX_DEG:
            continue
        total += area
    return total


def _is_step_bridge_slope(
    slanted_corners: list[list[float]],
    flat_planes: list[tuple[list[list[float]], float]],
) -> bool:
    """True iff the slanted plane is a RoomPlan step-bridge artifact.

    A step bridge has:
      - small Y span (real attic slopes span more than KINK_STEP_BRIDGE_MAX_Y_RANGE_M
        from kneewall to ridge), AND
      - Y range that connects two distinct flat levels in the same room
        within KINK_STEP_FLAT_MATCH_TOL_M.
    """
    s_ys = [float(c[1]) for c in slanted_corners]
    if not s_ys:
        return False
    s_lo, s_hi = min(s_ys), max(s_ys)
    s_span = s_hi - s_lo
    if s_span > KINK_STEP_BRIDGE_MAX_Y_RANGE_M:
        return False
    flat_levels = sorted(
        {
            round(sum(float(c[1]) for c in corners) / len(corners), 4)
            for corners, _area in flat_planes
            if corners
        }
    )
    for i, fy_low in enumerate(flat_levels):
        for fy_high in flat_levels[i + 1 :]:
            step = fy_high - fy_low
            if step < KINK_STEP_FLAT_LEVEL_SEPARATION_M:
                continue
            spans_low = s_lo <= fy_low + KINK_STEP_FLAT_MATCH_TOL_M
            spans_high = s_hi >= fy_high - KINK_STEP_FLAT_MATCH_TOL_M
            matches_step = abs(s_span - step) < KINK_STEP_FLAT_MATCH_TOL_M
            if spans_low and spans_high and matches_step:
                return True
    return False


def _detect_kinked_ceiling(
    room: ExtractedRoom,
) -> tuple[list[list[float]], list[list[list[float]]]] | None:
    """Knee-wall detection: when the same room's scan contains both a flat
    ceiling patch (incl < INCLINED_CEILING_MIN_DEG) and a slanted patch
    (in [INCLINED_CEILING_MIN_DEG, INCLINED_CEILING_MAX_DEG]), return
    (flat_corners, slope_corners) so the candidate emitter can produce two
    tier elements instead of collapsing the room into one.

    The area gate is cumulative -- RoomPlan often fragments a single flat lid
    into several small patches (< 0.5 m^2 each). We require the *total* flat
    area and *total* slanted area to each clear `INCLINED_CEILING_MIN_AREA_M2`
    (the corpus-tuned noise floor), then pick the largest plane in each
    bucket as the representative polygon.
    """
    flat_planes: list[tuple[list[list[float]], float]] = []
    slanted_planes: list[
        tuple[list[list[float]], float, tuple[float, float, float], Polygon]
    ] = []
    for plane in room.raw_ceiling_planes:
        info = _raw_plane_incl_and_xz_area(plane.corners)
        if info is None:
            continue
        incl_deg, area = info
        if incl_deg < INCLINED_CEILING_MIN_DEG:
            flat_planes.append((plane.corners, area))
        elif incl_deg <= INCLINED_CEILING_MAX_DEG:
            normal = _raw_plane_normal(plane.corners)
            poly = _xz_polygon(plane.corners)
            if normal is not None and poly is not None:
                slanted_planes.append((plane.corners, area, normal, poly))
    if not flat_planes or not slanted_planes:
        return None
    # Drop slanted planes that are step-bridge artifacts: small Y span that
    # exactly connects two flat levels in the same room. RoomPlan emits these
    # over the corner where one flat ceiling steps to another, fooling the
    # kink detector into treating a stepped ceiling as a sloped one.
    slanted_planes = [
        plane
        for plane in slanted_planes
        if not _is_step_bridge_slope(plane[0], flat_planes)
    ]
    if not slanted_planes:
        return None
    if sum(a for _, a in flat_planes) < INCLINED_CEILING_MIN_AREA_M2:
        return None
    if (
        sum(area for _corners, area, _normal, _poly in slanted_planes)
        < INCLINED_CEILING_MIN_AREA_M2
    ):
        return None
    flat_corners, _ = max(flat_planes, key=lambda x: x[1])
    selected_slopes: list[
        tuple[list[list[float]], float, tuple[float, float, float], Polygon]
    ] = []
    for plane in sorted(slanted_planes, key=lambda x: x[1], reverse=True):
        if not selected_slopes:
            selected_slopes.append(plane)
            continue
        largest_area = selected_slopes[0][1]
        if plane[1] < max(
            KINK_SECOND_SLOPE_MIN_AREA_M2,
            largest_area * KINK_SECOND_SLOPE_MIN_LARGEST_RATIO,
        ):
            continue
        if any(
            _normal_angle_deg(plane[2], selected[2])
            < KINK_DISTINCT_SLOPE_MIN_NORMAL_ANGLE_DEG
            for selected in selected_slopes
        ):
            continue
        if any(
            float(plane[3].intersection(selected[3]).area)
            / max(1e-9, min(float(plane[3].area), float(selected[3].area)))
            > KINK_DISTINCT_SLOPE_MAX_XZ_OVERLAP_RATIO
            for selected in selected_slopes
        ):
            continue
        selected_slopes.append(plane)
        if len(selected_slopes) >= KINK_MAX_SLOPE_POLYGONS:
            break
    return (
        [list(map(float, c)) for c in flat_corners],
        [
            [list(map(float, c)) for c in slope_corners]
            for slope_corners, _area, _normal, _poly in selected_slopes
        ],
    )


def _classify_should_be_flat(
    rooms: list[ExtractedRoom],
) -> dict[int, tuple[bool, float | None]]:
    per_room = [
        {
            "room": room,
            "ws": _wall_top_percentiles(room),
            "max_raw_y": _max_raw_ceiling_corner_y(room),
        }
        for room in rooms
    ]
    by_story_p50: dict[int, list[float]] = {}
    for entry in per_room:
        ws = entry["ws"]
        if ws is None:
            continue
        if (ws["p90"] - ws["p10"]) >= WALL_TOP_SPREAD_MAX_M:
            continue
        max_raw = entry["max_raw_y"]
        if max_raw is not None and (max_raw - ws["p50"]) >= CEILING_OVERSHOOT_MAX_M:
            continue
        by_story_p50.setdefault(entry["room"].story, []).append(ws["p50"])
    neighbour_median = {
        story: float(np.median(values))
        for story, values in by_story_p50.items()
        if values
    }

    out: dict[int, tuple[bool, float | None]] = {}
    for idx, entry in enumerate(per_room):
        room = entry["room"]
        ws = entry["ws"]
        if ws is None:
            out[idx] = (False, None)
            continue
        max_raw = entry["max_raw_y"]
        if max_raw is not None and (max_raw - ws["p50"]) >= CEILING_OVERSHOOT_MAX_M:
            out[idx] = (False, None)
            continue
        internal_flat = (ws["p90"] - ws["p10"]) < WALL_TOP_SPREAD_MAX_M
        nbm = neighbour_median.get(room.story)
        nb_consensus = nbm is not None and abs(ws["p50"] - nbm) < NEIGHBOUR_TOLERANCE_M
        if internal_flat:
            out[idx] = (True, ws["p50"])
        elif nb_consensus:
            # Veto when the room's own scan disagrees with the story-wide
            # flatness prior: slanted wall tops AND a non-trivial inclined
            # ceiling area together mean the scan saw a slope and we should
            # not snap it flat.
            if (
                _has_slanted_wall_top(room)
                and _inclined_raw_ceiling_area(room) >= INCLINED_CEILING_MIN_AREA_M2
            ):
                out[idx] = (False, None)
            else:
                out[idx] = (True, nbm)
        else:
            out[idx] = (False, None)
    return out


def _drop_noisy_raw_ceiling_planes(
    room: ExtractedRoom,
    should_flat: bool,
    expected_y: float | None,
) -> list[RawCeilingPlane]:
    if not should_flat or expected_y is None:
        return room.raw_ceiling_planes
    kept = []
    seen = set()
    for plane in room.raw_ceiling_planes:
        if not plane.corners:
            continue
        min_y = min(corner[1] for corner in plane.corners)
        if expected_y - min_y > NOISY_PLANE_DROP_TAU_M:
            continue
        if min_y - expected_y > RIDGE_EAVE_MAX_DELTA_M:
            continue
        key = tuple(
            tuple(round(float(value), 4) for value in corner[:3])
            for corner in plane.corners
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(plane)
    return kept


def _flat_ceiling_polygon(room: ExtractedRoom, ceiling_y: float) -> list[list[float]]:
    return [
        [round(corner[0], 4), ceiling_y, round(corner[2], 4)]
        for corner in room.floor_polygon
    ]


def _xz_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def build_ceiling_from_wall_tops(room: ExtractedRoom) -> list[list[float]] | None:
    floor_polygon = room.floor_polygon
    walls = room.walls_computed or room.walls_merged
    if len(floor_polygon) < 3 or not walls:
        return None

    wall_info = []
    for wall in walls:
        corners = wall.corners
        if len(corners) < 3:
            continue
        ys = [corner[1] for corner in corners]
        mid_y = (max(ys) + min(ys)) / 2.0
        bot = [corner for corner in corners if corner[1] <= mid_y + 0.01]
        top = [corner for corner in corners if corner[1] > mid_y - 0.01]
        if len(bot) >= 2 and len(top) >= 2:
            wall_info.append({"bot": bot, "top": top})
    if not wall_info:
        return None

    def nearest_corner(corner, corners):
        return min((_xz_dist(corner, candidate), candidate) for candidate in corners)

    edge_walls = [None] * len(floor_polygon)
    used = set()
    for edge_idx in range(len(floor_polygon)):
        v0 = floor_polygon[edge_idx]
        v1 = floor_polygon[(edge_idx + 1) % len(floor_polygon)]
        best_score = float("inf")
        best_idx = None
        for wall_idx, info in enumerate(wall_info):
            if wall_idx in used:
                continue
            d0, _ = nearest_corner(v0, info["bot"])
            d1, _ = nearest_corner(v1, info["bot"])
            score = d0 + d1
            if score < best_score:
                best_score = score
                best_idx = wall_idx
        if best_idx is not None and best_score < XZ_MATCH_TOL_M * 2.0:
            edge_walls[edge_idx] = best_idx
            used.add(best_idx)

    ceiling_pts = []
    for edge_idx, wall_idx in enumerate(edge_walls):
        if wall_idx is None:
            continue
        v0 = floor_polygon[edge_idx]
        v1 = floor_polygon[(edge_idx + 1) % len(floor_polygon)]
        top_corners = wall_info[wall_idx]["top"]
        bot_corners = wall_info[wall_idx]["bot"]
        start_bot = nearest_corner(v0, bot_corners)[1]
        end_bot = nearest_corner(v1, bot_corners)[1]
        start_top = nearest_corner(start_bot, top_corners)[1]
        end_top = nearest_corner(end_bot, top_corners)[1]
        ordered_top = [start_top, end_top]
        for corner in ordered_top:
            point = [round(corner[0], 4), round(corner[1], 4), round(corner[2], 4)]
            if ceiling_pts:
                last = ceiling_pts[-1]
                if abs(point[0] - last[0]) < 0.01 and abs(point[2] - last[2]) < 0.01:
                    if point[1] > last[1]:
                        ceiling_pts[-1] = point
                    continue
            ceiling_pts.append(point)

    if len(ceiling_pts) >= 2:
        first = ceiling_pts[0]
        last = ceiling_pts[-1]
        if abs(first[0] - last[0]) < 0.01 and abs(first[2] - last[2]) < 0.01:
            if last[1] > first[1]:
                ceiling_pts[0] = last
            ceiling_pts.pop()
    return ceiling_pts if len(ceiling_pts) >= 3 else None


def _has_slanted_wall_top(room: ExtractedRoom) -> bool:
    walls = room.walls_computed or room.walls_merged
    for wall in walls:
        corners = wall.corners
        if len(corners) < 3:
            continue
        ys = [corner[1] for corner in corners]
        mid_y = (max(ys) + min(ys)) / 2.0
        top_corners = [corner for corner in corners if corner[1] > mid_y - 0.01]
        if len(top_corners) < 2:
            continue
        top_range = max(corner[1] for corner in top_corners) - min(
            corner[1] for corner in top_corners
        )
        if top_range > SLANT_THRESH_M:
            return True
    return False


def infer_ceilings(rooms: list[ExtractedRoom]) -> list[ExtractedRoom]:
    reassigned = reassign_raw_ceiling_planes_spatially(rooms)
    if (
        os.environ.get("TIER_RAW_CEILING_MERGE") == "1"
        or architectural_priors_enabled()
    ):
        reassigned = merge_coplanar_raw_ceilings(reassigned)
    classifications = _classify_should_be_flat(reassigned)
    out: list[ExtractedRoom] = []
    for idx, room in enumerate(reassigned):
        should_flat, expected_y = classifications.get(idx, (False, None))
        raw_planes = room.raw_ceiling_planes
        # Run kink detection against the *unfiltered* raw planes -- the noisy
        # plane drop below removes any plane whose min_y sits well under
        # `expected_y`, which is exactly the slope signal we need to keep for
        # knee-wall rooms that the classifier called flat.
        kink = _detect_kinked_ceiling(room)
        flat_raw_planes = _drop_noisy_raw_ceiling_planes(room, should_flat, expected_y)
        if should_flat and expected_y is not None and raw_planes:
            kept_max_y = _max_raw_ceiling_corner_y(
                replace(room, raw_ceiling_planes=flat_raw_planes)
            )
            ceiling_y = round(
                float(
                    max(expected_y, kept_max_y)
                    if kept_max_y is not None
                    else expected_y
                ),
                4,
            )
            out.append(
                _with_kink(
                    replace(
                        room,
                        raw_ceiling_planes=flat_raw_planes,
                        ceiling_polygon=_flat_ceiling_polygon(room, ceiling_y)
                        if len(room.floor_polygon) >= 3
                        else [],
                        ceiling_type="flat",
                        ceiling_eave_height=ceiling_y,
                        ceiling_ridge_height=ceiling_y,
                    ),
                    kink,
                )
            )
            continue

        if not _has_slanted_wall_top(room):
            out.append(_with_kink(replace(room, raw_ceiling_planes=raw_planes), kink))
            continue

        ceiling_polygon = build_ceiling_from_wall_tops(room)
        if ceiling_polygon is None:
            out.append(_with_kink(replace(room, raw_ceiling_planes=raw_planes), kink))
            continue
        ys = [point[1] for point in ceiling_polygon]
        ceiling_type = "flat" if max(ys) - min(ys) < SLOPE_THRESH_M else "sloped"
        out.append(
            _with_kink(
                replace(
                    room,
                    raw_ceiling_planes=raw_planes,
                    ceiling_polygon=ceiling_polygon,
                    ceiling_type=ceiling_type,
                    ceiling_eave_height=round(min(ys), 4),
                    ceiling_ridge_height=round(max(ys), 4),
                ),
                kink,
            )
        )
    return out


def _with_kink(
    room: ExtractedRoom,
    kink: tuple[list[list[float]], list[list[list[float]]]] | None,
) -> ExtractedRoom:
    if kink is None:
        return room
    flat_corners, slope_polygons = kink
    if not slope_polygons:
        return room
    return replace(
        room,
        ceiling_is_kinked=True,
        ceiling_flat_polygon=flat_corners,
        ceiling_slope_polygon=slope_polygons[0],
        ceiling_slope_polygons=slope_polygons,
    )


# Re-exports — coplanar raw-ceiling merge lives in `_coplanar_merge`.
# `infer_ceilings` calls it after the kink-detection pass.
from reconcile_tiers.extract._coplanar_merge import (  # noqa: E402
    merge_coplanar_raw_ceilings,
)
