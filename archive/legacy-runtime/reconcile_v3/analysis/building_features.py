"""Per-building context features from ``reconcile/buildings_3d.json``.

Each labeled segment sits inside one building. This module computes a
per-UUID dict (footprint shape, scan-quality priors, wall-orientation stats)
that the feature-expansion layer joins onto every row.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_v3.constants import (
    PART_RECTANGULARITY_MIN,
    PART_SHAPE_L_SOLIDITY_MAX,
    PART_SHAPE_T_SOLIDITY_MAX,
    PART_SHAPE_U_SOLIDITY_MAX,
)

_EPS = 1e-9


def _room_polygon(fp: list) -> Polygon | None:
    try:
        poly = Polygon([(float(c[0]), float(c[2])) for c in fp])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= _EPS:
        return None
    return poly


def _room_polygons(building: dict) -> list[Polygon]:
    polys: list[Polygon] = []
    for room in building.get("rooms", []) or []:
        fp = room.get("floor_polygon") or []
        if len(fp) < 3:
            continue
        p = _room_polygon(fp)
        if p is not None:
            polys.append(p)
    return polys


def _iter_all_ys(building: dict) -> Iterable[float]:
    for room in building.get("rooms", []) or []:
        for c in room.get("floor_polygon", []) or []:
            yield float(c[1])
        for wall in (room.get("walls_merged") or []) + (
            room.get("walls_computed") or []
        ):
            for c in wall.get("corners") or []:
                yield float(c[1])


def _compactness(area: float, perimeter: float) -> float | None:
    if perimeter <= _EPS:
        return None
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def _entropy(weights: list[float]) -> float | None:
    if not weights:
        return None
    arr = np.asarray(weights, dtype=float)
    total = float(arr.sum())
    if total <= _EPS:
        return None
    probs = arr / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(probs > 0.0, probs * np.log(probs), 0.0))
    denom = math.log(len(arr)) if len(arr) > 1 else 1.0
    return float(ent / denom) if denom > 0 else 0.0


def _bbox_aspect(poly: Polygon) -> float | None:
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if min(w, h) <= _EPS:
        return None
    return float(max(w, h) / min(w, h))


def _mrr_coords(poly: Polygon) -> list:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mrr = poly.minimum_rotated_rectangle
        return list(mrr.exterior.coords)[:-1] if hasattr(mrr, "exterior") else []
    except Exception:
        return []


def _elongation_ratio(poly: Polygon) -> float | None:
    coords = _mrr_coords(poly)
    if len(coords) < 4:
        return None
    p0, p1, p2 = (np.asarray(coords[i], dtype=float) for i in range(3))
    e1 = float(np.linalg.norm(p1 - p0))
    e2 = float(np.linalg.norm(p2 - p1))
    if min(e1, e2) <= _EPS:
        return None
    return float(max(e1, e2) / min(e1, e2))


def _principal_axis_deg(poly: Polygon) -> float | None:
    """Azimuth (0-180°) of the footprint's minimum-rotated-rectangle major axis."""
    coords = _mrr_coords(poly)
    if len(coords) < 4:
        return None
    p0, p1, p2 = (np.asarray(coords[i], dtype=float) for i in range(3))
    e1v = p1 - p0
    e2v = p2 - p1
    e1 = float(np.linalg.norm(e1v))
    e2 = float(np.linalg.norm(e2v))
    if min(e1, e2) <= _EPS:
        return None
    major = e1v if e1 >= e2 else e2v
    az = math.degrees(math.atan2(float(major[0]), float(major[1])))
    return float(az % 180.0)


def _polygon_reflex_corner_count(poly: Polygon) -> int:
    try:
        coords = list(poly.exterior.coords)[:-1]
    except Exception:
        return 0
    if len(coords) < 3:
        return 0
    pts = np.asarray(coords, dtype=float)
    signed_area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        signed_area += x1 * y2 - x2 * y1
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    reflex = 0
    for i in range(len(pts)):
        a = pts[(i - 1) % len(pts)]
        b = pts[i]
        c = pts[(i + 1) % len(pts)]
        ab = b - a
        bc = c - b
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        if cross * orientation < 0.0:
            reflex += 1
    return reflex


def _shape_flags(poly: Polygon) -> dict[str, bool | None]:
    area = float(poly.area)
    hull_area = float(poly.convex_hull.area) if poly.convex_hull.area > _EPS else None
    solidity = area / hull_area if hull_area else None
    reflex = _polygon_reflex_corner_count(poly)
    rectangularity = None
    coords = _mrr_coords(poly)
    if len(coords) >= 4:
        p0, p1, p2 = (np.asarray(coords[i], dtype=float) for i in range(3))
        e1 = float(np.linalg.norm(p1 - p0))
        e2 = float(np.linalg.norm(p2 - p1))
        mrr_area = e1 * e2
        rectangularity = area / mrr_area if mrr_area > _EPS else None
    return {
        "bld_footprint_is_L_shape": bool(
            reflex == 1 and (solidity or 0.0) < PART_SHAPE_L_SOLIDITY_MAX
        ),
        "bld_footprint_is_T_shape": bool(
            reflex == 2 and (solidity or 0.0) < PART_SHAPE_T_SOLIDITY_MAX
        ),
        "bld_footprint_is_U_shape": bool(
            reflex >= 2 and (solidity or 0.0) < PART_SHAPE_U_SOLIDITY_MAX
        ),
        "bld_footprint_is_rectangle": bool(
            rectangularity is not None
            and rectangularity > PART_RECTANGULARITY_MIN
            and (solidity or 0.0) > PART_RECTANGULARITY_MIN
        ),
    }


def _wall_orientation_stats(building: dict) -> dict[str, Any]:
    bins: dict[float, float] = {}
    for room in building.get("rooms", []) or []:
        for wall in (room.get("walls_merged") or []) + (
            room.get("walls_computed") or []
        ):
            corners = wall.get("corners") or []
            if len(corners) < 2:
                continue
            pts = np.asarray(corners, dtype=float)
            xz = pts[:, [0, 2]]
            centered = xz - xz.mean(axis=0)
            principal = None
            try:
                cov = np.cov(centered.T)
                _, eigvecs = np.linalg.eigh(cov)
                principal = eigvecs[:, -1]
            except Exception:
                principal = None
            if principal is None:
                continue
            az = (
                math.degrees(math.atan2(float(principal[0]), float(principal[1])))
                % 180.0
            )
            length = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    length = max(
                        length,
                        math.hypot(
                            float(pts[i, 0] - pts[j, 0]), float(pts[i, 2] - pts[j, 2])
                        ),
                    )
            bins[round(az / 10.0) * 10.0] = bins.get(
                round(az / 10.0) * 10.0, 0.0
            ) + max(length, 1e-3)
    if not bins:
        return {
            "bld_dominant_wall_azimuth_deg": None,
            "bld_wall_azimuth_entropy": None,
        }
    dominant = max(bins.items(), key=lambda kv: kv[1])[0] % 180.0
    return {
        "bld_dominant_wall_azimuth_deg": float(dominant),
        "bld_wall_azimuth_entropy": _entropy(list(bins.values())),
    }


def compute_building_features(building: dict) -> dict[str, Any]:
    """Return per-building features keyed by ``bld_*``."""
    rooms = building.get("rooms", []) or []
    story_set = {r.get("story") for r in rooms if r.get("story") is not None}
    polys = _room_polygons(building)
    out: dict[str, Any] = {
        "bld_classification": building.get("classification"),
        "bld_stories_found": building.get("stories_found"),
        "bld_story_count_rooms": len(story_set),
        "bld_story_count": len(story_set),
        "bld_room_count": len(rooms),
        "bld_wall_count": sum(
            len(r.get("walls_merged") or []) + len(r.get("walls_computed") or [])
            for r in rooms
        ),
        "bld_door_count": sum(len(r.get("doors") or []) for r in rooms),
        "bld_window_count": sum(len(r.get("windows") or []) for r in rooms),
        "bld_cross_floor_gap_count": len(building.get("cross_floor_gaps") or []),
        "bld_gap_wall_count": len(building.get("gap_walls") or []),
        "bld_stitch_wall_count": len(building.get("stitch_walls") or []),
        "bld_footprint_area_m2": None,
        "bld_footprint_perimeter_m": None,
        "bld_footprint_compactness": None,
        "bld_footprint_bbox_aspect": None,
        "bld_footprint_convex_hull_ratio": None,
        "bld_footprint_solidity": None,
        "bld_footprint_convexity_deficiency": None,
        "bld_footprint_elongation_ratio": None,
        "bld_footprint_centroid_x": None,
        "bld_footprint_centroid_z": None,
        "bld_footprint_part_count": None,
        "bld_footprint_principal_axis_deg": None,
        "bld_footprint_interior_ring_count": None,
        "bld_footprint_is_L_shape": None,
        "bld_footprint_is_T_shape": None,
        "bld_footprint_is_U_shape": None,
        "bld_footprint_is_rectangle": None,
        "bld_height_m": None,
        "bld_y_min_m": None,
        "bld_y_max_m": None,
        "bld_typical_story_height_m": None,
        "bld_has_basement": None,
        "bld_scan_quality_score": None,
        "bld_dominant_wall_azimuth_deg": None,
        "bld_wall_azimuth_entropy": None,
        "scan_quality_overlap_fraction": None,
        "scan_quality_cross_floor_gap_density": None,
    }
    if polys:
        union = None
        try:
            union = unary_union(polys)
        except Exception:
            union = None
        if union is not None and not union.is_empty:
            area = float(union.area)
            perimeter = float(union.length)
            hull = union.convex_hull
            hull_area = float(hull.area) if hull.area > _EPS else None
            if union.geom_type == "MultiPolygon":
                geoms = list(union.geoms)
                largest = max(geoms, key=lambda p: p.area)
                part_count = len(geoms)
            else:
                geoms = [union]
                largest = union
                part_count = 1
            solidity = area / hull_area if hull_area else None
            out.update(
                {
                    "bld_footprint_area_m2": area,
                    "bld_footprint_perimeter_m": perimeter,
                    "bld_footprint_compactness": _compactness(area, perimeter),
                    "bld_footprint_bbox_aspect": _bbox_aspect(largest),
                    "bld_footprint_convex_hull_ratio": solidity,
                    "bld_footprint_solidity": solidity,
                    "bld_footprint_convexity_deficiency": (1.0 - solidity)
                    if solidity is not None
                    else None,
                    "bld_footprint_elongation_ratio": _elongation_ratio(largest),
                    "bld_footprint_centroid_x": float(union.centroid.x),
                    "bld_footprint_centroid_z": float(union.centroid.y),
                    "bld_footprint_part_count": part_count,
                    "bld_footprint_principal_axis_deg": _principal_axis_deg(largest),
                    "bld_footprint_interior_ring_count": sum(
                        len(getattr(p, "interiors", [])) for p in geoms
                    ),
                }
            )
            out.update(_shape_flags(largest))
    ys = list(_iter_all_ys(building))
    if ys:
        out["bld_y_min_m"] = float(min(ys))
        out["bld_y_max_m"] = float(max(ys))
        out["bld_height_m"] = float(max(ys) - min(ys))
    if out["bld_height_m"] is not None and out["bld_story_count"] > 0:
        out["bld_typical_story_height_m"] = float(
            out["bld_height_m"] / out["bld_story_count"]
        )
    out["bld_has_basement"] = bool(
        any(
            (room.get("story") or 0) < 0
            for room in rooms
            if room.get("story") is not None
        )
    )

    overlap = building.get("overlap_metrics") or {}
    room_count = float(len(rooms) or 0.0)
    footprint_area = out.get("bld_footprint_area_m2")
    overlap_fraction = (
        float(overlap.get("floor_overlap_count") or 0.0) / room_count
        if room_count > 0
        else None
    )
    wall_clip_fraction = (
        float(overlap.get("walls_clipped") or 0.0)
        / float(overlap.get("walls_checked") or 0.0)
        if float(overlap.get("walls_checked") or 0.0) > 0.0
        else None
    )
    out["scan_quality_overlap_fraction"] = overlap_fraction
    out["scan_quality_cross_floor_gap_density"] = (
        float(len(building.get("cross_floor_gaps") or [])) / footprint_area
        if footprint_area and footprint_area > _EPS
        else None
    )
    scan_terms = [x for x in (overlap_fraction, wall_clip_fraction) if x is not None]
    out["bld_scan_quality_score"] = (
        float(sum(scan_terms) / len(scan_terms)) if scan_terms else None
    )
    out.update(_wall_orientation_stats(building))
    return out


def load_building_features(buildings_3d_path: Path) -> dict[str, dict[str, Any]]:
    """Return ``uuid → per-building feature dict``."""
    with buildings_3d_path.open() as f:
        buildings = json.load(f)
    if not isinstance(buildings, list):
        raise ValueError(
            f"Expected a list of buildings in {buildings_3d_path}, got "
            f"{type(buildings).__name__}"
        )
    return {b["uuid"]: compute_building_features(b) for b in buildings if "uuid" in b}
