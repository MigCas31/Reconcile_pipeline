"""Band 3 features derived from the streamed V3 context cache.

For each labeled segment we need to resolve:

    - which V3 part it sits in (by XZ centroid vs part footprint)
    - how many kneewall extensions overlap its XZ projection
    - whether any dormer overlaps it
    - whether its merged plane survived to a final ``slanted_roofs`` entry

All inputs come from ``v3_context.json`` (written by ``v3_context.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from ..audit import note_geom_skip

_EPS = 1e-9


def _xz_poly(corners_xyz: Sequence[Sequence[float]]) -> Polygon | None:
    if not corners_xyz or len(corners_xyz) < 3:
        return None
    try:
        p = Polygon([(float(c[0]), float(c[2])) for c in corners_xyz])
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return None
        return p
    except Exception:
        return None


def _plane_cos(a: Sequence[float], b: Sequence[float]) -> float:
    """Abs cosine of the angle between two plane normals."""
    na = np.asarray(a[:3], dtype=float)
    nb = np.asarray(b[:3], dtype=float)
    la, lb = float(np.linalg.norm(na)), float(np.linalg.norm(nb))
    if la < _EPS or lb < _EPS:
        return 0.0
    return float(abs(np.dot(na / la, nb / lb)))


def _part_for_segment(
    segment_poly: Polygon | None, parts: Sequence[dict]
) -> dict | None:
    """Return the part whose footprint contains the segment centroid."""
    if segment_poly is None or not parts:
        return None
    centroid = segment_poly.centroid
    best_part = None
    best_area = -1.0
    for p in parts:
        fp = p.get("footprint_xz") or []
        poly = _xz_poly(fp) if fp and len(fp[0]) == 3 else None
        if poly is None:
            # some parts store 2D footprint as (x, z) pairs
            try:
                poly = Polygon([(float(c[0]), float(c[1])) for c in fp])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area <= 0:
                    poly = None
            except Exception:
                poly = None
        if poly is None:
            continue
        if poly.contains(centroid) and poly.area > best_area:
            best_part = p
            best_area = poly.area
    return best_part


_GABLE_STATUSES = (
    "not_gable",
    "gable_complete",
    "gable_along_extend",
    "gable_cross_review",
    "gable_ambiguous",
)

_GABLE_METRIC_KEYS = (
    "n_slanted_roofs",
    "elong",
    "coverage",
    "n_dormers",
    "n_arch_flats",
    "ridge_y_abs",
    "ridge_vs_expected",
    "ridge_vs_major",
    "dincl",
    "daz180",
    "major_m",
    "minor_m",
    "az0",
    "az1",
    "incl0",
    "incl1",
    "ridge_az",
    "major_az",
)


def _part_features(part: dict | None) -> dict[str, Any]:
    status = None
    tier1_count = None
    tier2_count = None
    metrics: dict = {}
    if part is not None:
        ge = part.get("gable_extension") or {}
        status = ge.get("status")
        tier1_count = len(ge.get("tier1_reasons") or [])
        tier2_count = len(ge.get("tier2_reasons") or [])
        metrics = ge.get("metrics") or {}
    out: dict[str, Any] = {
        "part_gable_status": status,
        "part_gable_tier1_reason_count": tier1_count,
        "part_gable_tier2_reason_count": tier2_count,
    }
    for s in _GABLE_STATUSES:
        out[f"part_gable_is_{s}"] = (status == s) if status is not None else None
    for k in _GABLE_METRIC_KEYS:
        out[f"part_gable_metric_{k}"] = (
            float(metrics[k]) if k in metrics and metrics[k] is not None else None
        )
    # Part footprint geometry (recomputed from the context slice).
    fp = part.get("footprint_xz") if part else None
    part_poly = None
    if fp:
        first = fp[0]
        try:
            if len(first) == 3:
                part_poly = Polygon([(float(c[0]), float(c[2])) for c in fp])
            else:
                part_poly = Polygon([(float(c[0]), float(c[1])) for c in fp])
            if not part_poly.is_valid:
                part_poly = part_poly.buffer(0)
            if part_poly.is_empty or part_poly.area <= 0:
                part_poly = None
        except Exception:
            part_poly = None
    if part_poly is not None:
        out["part_footprint_area_m2"] = float(part_poly.area)
        out["part_footprint_perimeter_m"] = float(part_poly.length)
        out["part_room_count"] = len(part.get("room_ids") or []) if part else None
    else:
        out["part_footprint_area_m2"] = None
        out["part_footprint_perimeter_m"] = None
        out["part_room_count"] = len(part.get("room_ids") or []) if part else None
    return out


def _kneewall_features(
    segment_poly: Polygon | None,
    room_ids_in_labels: set[str],
    wall_extensions: Sequence[dict],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kneewall_count_overlapping_xz": 0,
        "kneewall_count_in_same_room": 0,
        "kneewall_overlap_area_m2": 0.0,
        "kneewall_nearest_distance_m": None,
        "any_kneewall_in_building": False,
    }
    if not wall_extensions:
        return out
    knee = [w for w in wall_extensions if w.get("behind_knee_wall")]
    out["any_kneewall_in_building"] = bool(knee)
    if segment_poly is None or not knee:
        return out
    overlap_polys = []
    overlap_area = 0.0
    min_dist = None
    overlap_count = 0
    same_room_count = 0
    for w in knee:
        sc = w.get("strip_corners") or []
        poly = _xz_poly(sc) if sc and len(sc[0]) == 3 else None
        if poly is None:
            continue
        if poly.intersects(segment_poly):
            overlap_count += 1
            try:
                area = float(poly.intersection(segment_poly).area)
            except Exception:
                area = 0.0
            overlap_area += area
            overlap_polys.append(poly)
        d = float(segment_poly.distance(poly))
        if min_dist is None or d < min_dist:
            min_dist = d
        if w.get("room_id") and w.get("room_id") in room_ids_in_labels:
            same_room_count += 1
    out["kneewall_count_overlapping_xz"] = overlap_count
    out["kneewall_count_in_same_room"] = same_room_count
    out["kneewall_overlap_area_m2"] = overlap_area
    out["kneewall_nearest_distance_m"] = min_dist
    return out


def _dormer_features(
    segment_poly: Polygon | None, dormers: Sequence[dict]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dormer_count_in_building": int(len(dormers) if dormers else 0),
        "dormer_count_overlapping_xz": 0,
        "dormer_overlap_area_m2": 0.0,
        "dormer_nearest_distance_m": None,
    }
    if segment_poly is None or not dormers:
        return out
    min_dist = None
    overlap_count = 0
    overlap_area = 0.0
    for d in dormers:
        poly = None
        # Try common polygon fields.
        for k in ("footprint_xz", "corners", "polygon"):
            if d.get(k):
                first = d[k][0] if d[k] else None
                if first is None:
                    continue
                try:
                    if len(first) == 3:
                        poly = Polygon([(float(c[0]), float(c[2])) for c in d[k]])
                    else:
                        poly = Polygon([(float(c[0]), float(c[1])) for c in d[k]])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_empty or poly.area <= 0:
                        poly = None
                    else:
                        break
                except Exception:
                    poly = None
        if poly is None:
            continue
        if poly.intersects(segment_poly):
            overlap_count += 1
            try:
                overlap_area += float(poly.intersection(segment_poly).area)
            except Exception as exc:
                note_geom_skip(exc, "context_features.dormer_overlap")
        dist = float(segment_poly.distance(poly))
        if min_dist is None or dist < min_dist:
            min_dist = dist
    out["dormer_count_overlapping_xz"] = overlap_count
    out["dormer_overlap_area_m2"] = overlap_area
    out["dormer_nearest_distance_m"] = min_dist
    return out


def _survival_features(
    merged_plane: Sequence[float] | None,
    segment_poly: Polygon | None,
    slanted_roofs: Sequence[dict],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "final_slanted_roof_count": int(len(slanted_roofs) if slanted_roofs else 0),
        "survived_to_final_plane": None,
        "best_final_plane_cos": None,
        "best_final_overlap_area_m2": None,
        "best_final_overlap_fraction": None,
    }
    if merged_plane is None or not slanted_roofs:
        return out
    best_cos = 0.0
    best_overlap_frac = 0.0
    best_overlap_area = 0.0
    for sr in slanted_roofs:
        pln = sr.get("plane")
        if not pln or len(pln) < 3:
            continue
        cos = _plane_cos(merged_plane, pln)
        overlap_area = 0.0
        overlap_frac = 0.0
        if segment_poly is not None:
            sr_poly = _xz_poly(sr.get("corners") or [])
            if sr_poly is not None:
                try:
                    inter = segment_poly.intersection(sr_poly)
                    overlap_area = float(inter.area)
                    overlap_frac = (
                        overlap_area / float(segment_poly.area)
                        if segment_poly.area > _EPS
                        else 0.0
                    )
                except Exception as exc:
                    note_geom_skip(exc, "context_features.survival_overlap")
        score = cos * (overlap_frac if overlap_frac > 0 else 1.0)
        if score > best_cos * (best_overlap_frac if best_overlap_frac > 0 else 1.0):
            best_cos = cos
            best_overlap_area = overlap_area
            best_overlap_frac = overlap_frac
    out["best_final_plane_cos"] = float(best_cos)
    out["best_final_overlap_area_m2"] = float(best_overlap_area)
    out["best_final_overlap_fraction"] = float(best_overlap_frac)
    out["survived_to_final_plane"] = bool(
        best_cos >= 0.97 and best_overlap_frac >= 0.25
    )
    return out


def context_features(record: dict, building_context: dict | None) -> dict[str, Any]:
    """Return Band 3 features for one label record given its building context."""
    corners = record.get("segment_corners_xyz") or []
    plane = record.get("merged_plane")
    segment_poly = _xz_poly(corners)
    room_refs = record.get("room_boundary_refs") or []
    room_ids = {
        f"room:{rr.get('room_index')}"
        for rr in room_refs
        if rr.get("kind") == "room" and rr.get("room_index") is not None
    }

    if building_context is None:
        part = None
        wall_ext = []
        dormers = []
        slanted_roofs = []
        meta_counts = {
            "ctx_roof_proposals_count": None,
            "ctx_merged_roof_segments_count": None,
            "ctx_part_count": None,
        }
    else:
        parts = building_context.get("parts") or []
        part = _part_for_segment(segment_poly, parts)
        wall_ext = building_context.get("wall_extensions") or []
        dormers = building_context.get("dormers") or []
        slanted_roofs = building_context.get("slanted_roofs") or []
        meta_counts = {
            "ctx_roof_proposals_count": building_context.get("roof_proposals_count"),
            "ctx_merged_roof_segments_count": building_context.get(
                "merged_roof_segments_count"
            ),
            "ctx_part_count": len(parts),
        }

    out: dict[str, Any] = dict(meta_counts)
    out.update(_part_features(part))
    out.update(_kneewall_features(segment_poly, room_ids, wall_ext))
    out.update(_dormer_features(segment_poly, dormers))
    out.update(_survival_features(plane, segment_poly, slanted_roofs))
    return out
