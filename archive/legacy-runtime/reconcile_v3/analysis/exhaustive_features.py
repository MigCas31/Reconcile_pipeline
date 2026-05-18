"""Additional slanted-roof features beyond the baseline analysis bands.

The baseline stack (`feature_expansion`, `context_features`, `advanced_features`)
already covers the first few catalogue waves. This module layers on the
remaining in-scope, V3-native geometry/context features that are derivable from:

    - the label/inference record itself,
    - per-building features from ``buildings_3d.json``,
    - compact V3 context streamed from ``reconcile_v3_results.json``.

Everything stays pure: no network calls, no hidden global state. That keeps
training and inference aligned.
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polylabel, unary_union

from reconcile_v3.audit import note_geom_skip
from reconcile_v3.constants import (
    COPLANAR_D_TOL_M,
    EAVE_OVERHANG_MIN_M,
    FLAT_CEILING_DOMINANCE_RATIO,
    FLAT_CEILING_SPREAD_M,
    GABLE_UNIMODAL_RESULTANT_MIN,
    GRID_BUCKET_COARSE_M,
    GRID_BUCKET_FINE_M,
    GRID_BUCKET_MEDIUM_M,
    GRID_SNAP_COARSE_TOL_M,
    GRID_SNAP_FINE_TOL_M,
    PLANE_INTERIOR_CLIP_MARGIN_M,
    PYRAMID_ASPECT_MAX,
    RIDGE_HEIGHT_REASONABLE_MARGIN_M,
    ROOF_REDUNDANT_AREA_M2,
    SCAN_NOISE_M,
    THIN_SLIVER_WIDTH_M,
    TINY_AREA_M2,
    VERTEX_Y_BAND_M,
    VERTEX_Y_CLUSTER_SPLIT_M,
    WALL_EDGE_ALIGN_DIST_M,
    WALL_SUPPORT_BUFFER_M,
)

from . import advanced_features as adv
from . import context_features as ctx
from . import feature_expansion as fe

_EPS = 1e-9
_GRID_BUCKETS = (GRID_BUCKET_FINE_M, GRID_BUCKET_MEDIUM_M, GRID_BUCKET_COARSE_M)
_INCL_BUCKETS = (0.0, 5.0, 10.0, 15.0, 22.5, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _xz_poly(points: Sequence[Sequence[float]]) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    try:
        first = points[0]
        if len(first) >= 3:
            poly = Polygon([(float(p[0]), float(p[2])) for p in points])
        else:
            poly = Polygon([(float(p[0]), float(p[1])) for p in points])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= _EPS:
            return None
        return poly
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_distance(poly: Polygon, other: Polygon) -> float | None:
    try:
        return float(poly.distance(other))
    except Exception:
        return None


def _safe_intersects(left: Any, right: Any) -> bool:
    try:
        return bool(left.intersects(right))
    except Exception:
        return False


def _safe_unary_union(polys: Sequence[Polygon]) -> Polygon | None:
    clean = [poly for poly in polys if poly is not None and not poly.is_empty]
    if not clean:
        return None
    try:
        return unary_union(clean)
    except Exception:
        repaired = []
        for poly in clean:
            try:
                fixed = poly if poly.is_valid else poly.buffer(0)
            except Exception as exc:
                note_geom_skip(exc, "exhaustive_features.repair_buffer0")
                continue
            if fixed is not None and not fixed.is_empty:
                repaired.append(fixed)
        if not repaired:
            return None
        try:
            return unary_union(repaired)
        except Exception:
            return None


def _iter_geom_points(geom: Any) -> Iterable[Point]:
    if geom is None:
        return []
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Point":
        return [geom]
    if gtype in {"MultiPoint", "GeometryCollection", "MultiLineString"}:
        pts: list[Point] = []
        for sub in getattr(geom, "geoms", []):
            pts.extend(_iter_geom_points(sub))
        return pts
    if gtype == "LineString":
        coords = list(getattr(geom, "coords", []))
        if not coords:
            return []
        return [Point(*coords[0]), Point(*coords[-1])]
    return []


def _ring_points_xy(corners_xyz: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray([(float(c[0]), float(c[2])) for c in corners_xyz], dtype=float)


def _gini(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray([abs(float(v)) for v in values], dtype=float)
    if not len(arr) or np.all(arr < _EPS):
        return 0.0
    arr.sort()
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * arr)) / (n * np.sum(arr)))


def _iqr(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _skewness(values: Sequence[float]) -> float | None:
    if len(values) < 3:
        return None
    arr = np.asarray(values, dtype=float)
    std = arr.std(ddof=0)
    if std <= _EPS:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / std) ** 3))


def _kurtosis(values: Sequence[float]) -> float | None:
    if len(values) < 4:
        return None
    arr = np.asarray(values, dtype=float)
    std = arr.std(ddof=0)
    if std <= _EPS:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / std) ** 4))


def _entropy_from_counts(counter: Counter[Any]) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    probs = np.asarray(list(counter.values()), dtype=float) / float(total)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(probs > 0.0, probs * np.log(probs), 0.0))
    denom = math.log(len(counter)) if len(counter) > 1 else 1.0
    return float(ent / denom) if denom > 0 else 0.0


def _footprint_corners(poly: Polygon | None) -> list[tuple[float, float]]:
    if poly is None or not hasattr(poly, "exterior"):
        return []
    return [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]


def _interior_angles_xy(
    corners_xyz: Sequence[Sequence[float]],
) -> tuple[list[float], int]:
    pts = _ring_points_xy(corners_xyz)
    if len(pts) < 3:
        return ([], 0)
    signed_area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        signed_area += x1 * y2 - x2 * y1
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    angles: list[float] = []
    reflex = 0
    for i in range(len(pts)):
        a = pts[(i - 1) % len(pts)]
        b = pts[i]
        c = pts[(i + 1) % len(pts)]
        v1 = a - b
        v2 = c - b
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        if l1 <= _EPS or l2 <= _EPS:
            continue
        cosv = max(-1.0, min(1.0, float(np.dot(v1, v2) / (l1 * l2))))
        interior = math.degrees(math.acos(cosv))
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if cross * orientation < 0.0:
            interior = 360.0 - interior
            reflex += 1
        angles.append(float(interior))
    return (angles, reflex)


def _edge_records(corners_xyz: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    if not corners_xyz or len(corners_xyz) < 2:
        return []
    pts = np.asarray(corners_xyz, dtype=float)
    out: list[dict[str, Any]] = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        vec = b - a
        length3 = float(np.linalg.norm(vec))
        length2 = float(math.hypot(float(vec[0]), float(vec[2])))
        dy = float(vec[1])
        az = (
            math.degrees(math.atan2(float(vec[0]), float(vec[2]))) % 180.0
            if length2 > _EPS
            else None
        )
        out.append(
            {
                "a": a,
                "b": b,
                "length3": length3,
                "length2": length2,
                "dy": dy,
                "mid": (a + b) / 2,
                "azimuth": az,
                "horizontal": abs(dy) < GRID_BUCKET_MEDIUM_M and length2 > _EPS,
                "descending": dy < -SCAN_NOISE_M,
                "ascending": dy > SCAN_NOISE_M,
            }
        )
    return out


def _bucket8(az: float | None) -> str | None:
    if az is None:
        return None
    names = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return names[int(((az % 360.0) + 22.5) // 45) % 8]


def _bucket16(az: float | None) -> int | None:
    if az is None:
        return None
    return int(((az % 360.0) + 11.25) // 22.5) % 16


def _incl_bucket(incl: float | None) -> str | None:
    if incl is None:
        return None
    if incl < 5.0:
        return "flat"
    if incl < 20.0:
        return "shallow"
    if incl < 50.0:
        return "medium"
    if incl < 80.0:
        return "steep"
    return "vertical"


def _grid_snap_fraction(
    corners_xyz: Sequence[Sequence[float]], grid: float
) -> float | None:
    if not corners_xyz:
        return None
    snapped = 0
    for c in corners_xyz:
        dx = abs((float(c[0]) / grid) - round(float(c[0]) / grid)) * grid
        dz = abs((float(c[2]) / grid) - round(float(c[2]) / grid)) * grid
        snap_tol = (
            GRID_SNAP_FINE_TOL_M
            if grid <= GRID_BUCKET_MEDIUM_M
            else GRID_SNAP_COARSE_TOL_M
        )
        if dx <= grid / 2 and dz <= grid / 2 and max(dx, dz) <= snap_tol:
            snapped += 1
    return float(snapped / len(corners_xyz))


def _signed_log10(x: float) -> float | None:
    if not math.isfinite(x):
        return None
    if abs(x) <= _EPS:
        return 0.0
    return float(math.copysign(math.log10(abs(x)), x))


def _fourier_boundary_features(
    corners_xyz: Sequence[Sequence[float]], n_descriptors: int = 16
) -> dict[str, float | None]:
    pts = _ring_points_xy(corners_xyz)
    if len(pts) < 4:
        return {
            **{f"fourier_phase_{i}": None for i in range(1, n_descriptors + 1)},
            **{f"fourier_mag_{i}": None for i in range(1, n_descriptors + 1)},
            **{f"fourier_log_{i}": None for i in range(1, n_descriptors + 1)},
            "fourier_energy_fraction_top_5": None,
        }
    closed = np.vstack([pts, pts[:1]])
    seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(seg_lens))
    if total <= _EPS:
        return {
            **{f"fourier_phase_{i}": None for i in range(1, n_descriptors + 1)},
            **{f"fourier_mag_{i}": None for i in range(1, n_descriptors + 1)},
            **{f"fourier_log_{i}": None for i in range(1, n_descriptors + 1)},
            "fourier_energy_fraction_top_5": None,
        }
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    samples = 64
    t = np.linspace(0.0, total, samples, endpoint=False)
    resampled = np.empty((samples, 2))
    for i, ti in enumerate(t):
        idx = np.searchsorted(cum, ti, side="right") - 1
        idx = max(0, min(idx, len(seg_lens) - 1))
        seg = seg_lens[idx]
        if seg <= _EPS:
            resampled[i] = closed[idx]
            continue
        alpha = (ti - cum[idx]) / seg
        resampled[i] = closed[idx] + alpha * (closed[idx + 1] - closed[idx])
    signal = resampled[:, 0] + 1j * resampled[:, 1]
    signal -= signal.mean()
    coeffs = np.fft.fft(signal)
    mags = np.abs(coeffs)
    energy = np.square(mags[1:])
    total_energy = float(np.sum(energy))
    mag_norm = float(mags[1]) if len(mags) > 1 else 0.0
    out = {
        f"fourier_phase_{i}": (float(np.angle(coeffs[i])) if i < len(coeffs) else None)
        for i in range(1, n_descriptors + 1)
    }
    out.update(
        {
            f"fourier_mag_{i}": (
                float(mags[i] / mag_norm) if i < len(mags) and mag_norm > _EPS else None
            )
            for i in range(1, n_descriptors + 1)
        }
    )
    out.update(
        {
            f"fourier_log_{i}": _signed_log10(out[f"fourier_mag_{i}"] or 0.0)
            if out[f"fourier_mag_{i}"] is not None
            else None
            for i in range(1, n_descriptors + 1)
        }
    )
    out["fourier_energy_fraction_top_5"] = (
        float(np.sum(np.sort(energy)[-5:]) / total_energy)
        if total_energy > _EPS
        else None
    )
    return out


def _projection_xy(
    points_xyz: Sequence[Sequence[float]], axes: tuple[int, int]
) -> np.ndarray:
    if not points_xyz:
        return np.empty((0, 2), dtype=float)
    return np.asarray(
        [(float(p[axes[0]]), float(p[axes[1]])) for p in points_xyz], dtype=float
    )


def _hu_projection_features(
    points_xyz: Sequence[Sequence[float]],
) -> dict[str, float | None]:
    projections = {
        "hu2_xz": _projection_xy(points_xyz, (0, 2)),
        "hu2_xy": _projection_xy(points_xyz, (0, 1)),
        "hu2_zy": _projection_xy(points_xyz, (2, 1)),
    }
    out: dict[str, float | None] = {f"hu_{i}": None for i in range(1, 8)}
    out.update({f"{prefix}_{i}": None for prefix in projections for i in range(1, 8)})
    xz = projections["hu2_xz"]
    if len(xz) >= 3:
        hu = adv._hu_moments(adv._central_moments(xz))
        for i, value in enumerate(hu, start=1):
            out[f"hu_{i}"] = value
            out[f"hu2_xz_{i}"] = value
    for prefix, xy in projections.items():
        if prefix == "hu2_xz" or len(xy) < 3:
            continue
        hu = adv._hu_moments(adv._central_moments(xy))
        for i, value in enumerate(hu, start=1):
            out[f"{prefix}_{i}"] = value
    return out


def _point_cloud_shape_features(
    points_xyz: Sequence[Sequence[float]],
) -> dict[str, float | None]:
    out = {
        "hackel_linearity": None,
        "hackel_planarity": None,
        "hackel_scatter": None,
        "hackel_omnivariance": None,
        "hackel_anisotropy": None,
        "hackel_eigenentropy": None,
        "hackel_curvature_change": None,
        "hackel_sum": None,
        **{f"hu3_xyz_{i}": None for i in range(1, 8)},
    }
    pts = np.asarray(points_xyz, dtype=float)
    if len(pts) < 3:
        return out
    centered = pts - pts.mean(axis=0)
    try:
        cov = np.cov(centered.T)
        eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
    except np.linalg.LinAlgError:
        return out
    if len(eigvals) != 3 or eigvals[0] <= _EPS:
        return out
    l1, l2, l3 = (float(v) for v in eigvals)
    total = l1 + l2 + l3
    out["hackel_linearity"] = float((l1 - l2) / l1)
    out["hackel_planarity"] = float((l2 - l3) / l1)
    out["hackel_scatter"] = float(l3 / l1)
    out["hackel_omnivariance"] = float((max(l1 * l2 * l3, 0.0)) ** (1.0 / 3.0))
    out["hackel_anisotropy"] = float((l1 - l3) / l1)
    if total > _EPS:
        probs = np.asarray([l1, l2, l3], dtype=float) / total
        valid = probs > 0.0
        out["hackel_eigenentropy"] = float(-np.sum(probs[valid] * np.log(probs[valid])))
        out["hackel_curvature_change"] = float(l3 / total)
    out["hackel_sum"] = float(total)
    invariants = [
        total,
        l1 * l2 + l1 * l3 + l2 * l3,
        l1 * l2 * l3,
        l1 * l1 + l2 * l2 + l3 * l3,
        (l1 - l2) * (l1 - l2) + (l2 - l3) * (l2 - l3) + (l1 - l3) * (l1 - l3),
        l1 / max(l3, _EPS),
        (l1 - l2 - l3) / max(total, _EPS),
    ]
    for i, value in enumerate(invariants, start=1):
        out[f"hu3_xyz_{i}"] = _signed_log10(value)
    return out


def _symmetry_iou(poly: Polygon | None, transformed: Polygon | None) -> float | None:
    if poly is None or transformed is None or poly.is_empty or transformed.is_empty:
        return None
    try:
        inter = float(poly.intersection(transformed).area)
        union = float(poly.union(transformed).area)
    except Exception:
        return None
    if union <= _EPS:
        return None
    return float(inter / union)


def _lag1_autocorr(values: Sequence[float]) -> float | None:
    if len(values) < 3:
        return None
    arr = np.asarray(values, dtype=float)
    x0 = arr - arr.mean()
    denom = float(np.dot(x0, x0))
    if denom <= _EPS:
        return 0.0
    return float(np.dot(x0, np.roll(x0, -1)) / denom)


def _parse_room_index(token: Any) -> int | None:
    if token is None:
        return None
    if isinstance(token, int):
        return token
    text = str(token)
    try:
        if ":" in text:
            return int(text.split(":", 1)[1])
        return int(text)
    except Exception:
        return None


def _parse_topology_room_index(source_id: Any) -> int | None:
    direct = _parse_room_index(source_id)
    if direct is not None:
        return direct
    text = str(source_id or "")
    marker = "merged_room_"
    if marker not in text:
        return None
    try:
        return int(text.rsplit(marker, 1)[1])
    except Exception:
        return None


def _poly_from_xz_points(points: Sequence[Sequence[float]]) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    try:
        poly = Polygon([(float(p[0]), float(p[1])) for p in points])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= _EPS:
        return None
    return poly


def _poly_iou(left: Polygon | None, right: Polygon | None) -> float | None:
    if left is None or right is None:
        return None
    try:
        inter = float(left.intersection(right).area)
        union = float(left.union(right).area)
    except Exception:
        return None
    if union <= _EPS:
        return None
    return float(inter / union)


def _segment_match_by_overlap(
    seg_poly: Polygon | None,
    candidates: Sequence[dict[str, Any]],
    *,
    poly_getter,
) -> tuple[dict[str, Any] | None, Polygon | None, float]:
    best = None
    best_poly = None
    best_score = 0.0
    if seg_poly is None:
        return (None, None, 0.0)
    for cand in candidates:
        poly = poly_getter(cand)
        if poly is None:
            continue
        try:
            score = float(seg_poly.intersection(poly).area)
        except Exception as exc:
            note_geom_skip(exc, "exhaustive_features.cand_score")
            continue
        if score > best_score:
            best = cand
            best_poly = poly
            best_score = score
    return (best, best_poly, best_score)


def _part_for_poly(
    segment_poly: Polygon | None, building_context: dict | None
) -> dict | None:
    if segment_poly is None or building_context is None:
        return None
    return ctx._part_for_segment(segment_poly, building_context.get("parts") or [])


def _room_polys_from_refs(room_refs: Sequence[dict]) -> list[Polygon]:
    polys = []
    for rr in room_refs or []:
        poly = _xz_poly(rr.get("footprint_xz") or rr.get("polygon") or [])
        if poly is not None:
            polys.append(poly)
    return polys


def _nearest_distance_to_polys(
    poly: Polygon | None, others: Iterable[Polygon]
) -> float | None:
    if poly is None:
        return None
    best = None
    for other in others:
        d = _safe_distance(poly, other)
        if d is None:
            continue
        if best is None or d < best:
            best = d
    return best


def _segment_peers(record: dict, building_context: dict | None) -> list[dict]:
    if building_context is None:
        return []
    my_id = record.get("proposal_id")
    return [
        s
        for s in building_context.get("merged_roof_segments") or []
        if s.get("id") != my_id
    ]


def _same_part(peer: dict, part: dict | None, building_context: dict | None) -> bool:
    if part is None or building_context is None:
        return False
    peer_poly = _xz_poly(peer.get("corners") or peer.get("footprint_xz") or [])
    peer_part = _part_for_poly(peer_poly, building_context)
    return bool(peer_part and part.get("id") == peer_part.get("id"))


def _flow_vector_xz(plane: Sequence[float]) -> np.ndarray | None:
    if not plane or len(plane) < 3:
        return None
    vec = np.asarray([-float(plane[0]), -float(plane[2])], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        return None
    return vec / norm


def _ray_exit_distance(
    origin: Point | None,
    direction: np.ndarray | None,
    boundary_poly: Polygon | None,
) -> float | None:
    if origin is None or direction is None or boundary_poly is None:
        return None
    minx, miny, maxx, maxy = boundary_poly.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    far = Point(
        origin.x + float(direction[0]) * span * 8.0,
        origin.y + float(direction[1]) * span * 8.0,
    )
    ray = LineString([(origin.x, origin.y), (far.x, far.y)])
    try:
        inter = boundary_poly.boundary.intersection(ray)
    except Exception:
        return None
    dists = [
        float(origin.distance(pt))
        for pt in _iter_geom_points(inter)
        if float(origin.distance(pt)) > SCAN_NOISE_M
    ]
    return float(min(dists)) if dists else None


def _story_indices(record: dict) -> list[int]:
    vals: list[int] = []
    for rr in record.get("room_boundary_refs") or []:
        story = rr.get("story")
        if story is not None:
            coerced = _coerce_int(story)
            if coerced is not None:
                vals.append(coerced)
    for m in record.get("cluster_members") or []:
        fs = m.get("features") or {}
        if fs.get("segment_story") is not None:
            coerced = _coerce_int(fs["segment_story"])
            if coerced is not None:
                vals.append(coerced)
    return sorted(set(vals))


@lru_cache(maxsize=1)
def _git_sha(repo_root: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def exhaustive_features(
    record: dict,
    row: dict[str, Any],
    *,
    building_context: dict | None,
    wall_index: dict[str, dict[str, Any]] | None,
    aux_context: dict | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    corners = record.get("segment_corners_xyz") or []
    plane = record.get("merged_plane") or []
    seg_poly = _xz_poly(corners)
    boundary_poly = _xz_poly(record.get("building_boundary_xz") or [])
    edges = _edge_records(corners)
    ys = [float(c[1]) for c in corners] if corners else []
    angles, reflex_count = _interior_angles_xy(corners)
    az, incl = (
        fe._plane_to_azimuth_incl(plane) if plane else (float("nan"), float("nan"))
    )
    plane_az = None if math.isnan(az) else float(az)
    plane_incl = None if math.isnan(incl) else float(incl)
    part = _part_for_poly(seg_poly, building_context)
    peers = _segment_peers(record, building_context)
    peer_polys = [
        (peer, _xz_poly(peer.get("corners") or peer.get("footprint_xz") or []))
        for peer in peers
    ]
    own_area = float(seg_poly.area) if seg_poly is not None else None
    centroid = seg_poly.centroid if seg_poly is not None else None

    out: dict[str, Any] = {}

    # Vertex-level signals.
    out["poly_vertex_count"] = len(corners)
    out["poly_vertex_count_after_simplify"] = None
    out["poly_unique_xy_count"] = (
        len({(round(float(c[0]), 2), round(float(c[2]), 2)) for c in corners})
        if corners
        else 0
    )
    out["poly_reflex_corner_count"] = reflex_count
    out["poly_acute_corner_count"] = int(sum(1 for a in angles if a < 60.0))
    out["poly_right_angle_fraction"] = (
        float(sum(1 for a in angles if abs(a - 90.0) <= 5.0) / len(angles))
        if angles
        else None
    )
    if seg_poly is not None:
        simp = seg_poly.simplify(SCAN_NOISE_M, preserve_topology=True)
        out["poly_vertex_count_after_simplify"] = len(list(simp.exterior.coords)) - 1
    if ys:
        y_arr = np.asarray(ys, dtype=float)
        y_min = float(y_arr.min())
        y_max = float(y_arr.max())
        out["vtx_y_range_m"] = float(y_max - y_min)
        out["vtx_ridge_band_count"] = int(np.sum(y_arr >= y_max - VERTEX_Y_BAND_M))
        out["vtx_eave_band_count"] = int(np.sum(y_arr <= y_min + VERTEX_Y_BAND_M))
        out["vtx_mid_band_count"] = int(
            len(y_arr) - out["vtx_ridge_band_count"] - out["vtx_eave_band_count"]
        )
        out["vtx_ridge_band_fraction"] = float(out["vtx_ridge_band_count"] / len(y_arr))
        out["vtx_eave_band_fraction"] = float(out["vtx_eave_band_count"] / len(y_arr))
        out["vtx_y_iqr_m"] = _iqr(ys)
        out["vtx_y_skewness"] = _skewness(ys)
        out["vtx_y_kurtosis"] = _kurtosis(ys)
        out["vtx_y_cluster_count_k2"] = (
            1 if out["vtx_y_range_m"] < VERTEX_Y_CLUSTER_SPLIT_M else 2
        )
        out["vtx_y_gap_at_median_m"] = (
            float(
                np.median(y_arr[y_arr >= np.median(y_arr)])
                - np.median(y_arr[y_arr <= np.median(y_arr)])
            )
            if len(y_arr) >= 2
            else None
        )
    else:
        for key in (
            "vtx_y_range_m",
            "vtx_ridge_band_count",
            "vtx_eave_band_count",
            "vtx_mid_band_count",
            "vtx_ridge_band_fraction",
            "vtx_eave_band_fraction",
            "vtx_y_iqr_m",
            "vtx_y_skewness",
            "vtx_y_kurtosis",
            "vtx_y_cluster_count_k2",
            "vtx_y_gap_at_median_m",
        ):
            out[key] = None
    for grid in _GRID_BUCKETS:
        key = f"vtx_grid_snap_fraction_{str(grid).replace('.', 'p')}"
        out[key] = _grid_snap_fraction(corners, grid)
    out["vtx_grid_axis_alignment_deg"] = row.get("poly_min_rect_azimuth_deg")
    if corners:
        offsets = []
        for c in corners:
            gx = round(float(c[0]) / 0.1) * 0.1
            gz = round(float(c[2]) / 0.1) * 0.1
            offsets.append(math.hypot(float(c[0]) - gx, float(c[2]) - gz))
        out["vtx_grid_snap_std_m"] = float(np.std(offsets)) if offsets else None
    wall_top_ys = []
    for rr in record.get("room_boundary_refs") or []:
        for wall in rr.get("walls") or []:
            for pt in wall.get("corners") or []:
                wall_top_ys.append(float(pt[1]))
    out["vtx_on_wall_top_count"] = (
        int(sum(1 for y in ys if any(abs(y - wy) <= 0.1 for wy in wall_top_ys)))
        if ys and wall_top_ys
        else 0
    )
    out["vtx_on_wall_top_fraction"] = (
        float(out["vtx_on_wall_top_count"] / len(ys)) if ys else None
    )
    if boundary_poly is not None and corners:
        outside = []
        boundary_line = boundary_poly.boundary
        for c in corners:
            p = Point(float(c[0]), float(c[2]))
            if not boundary_poly.buffer(1e-6).contains(p):
                outside.append(float(p.distance(boundary_line)))
        out["vtx_outside_building_footprint_count"] = len(outside)
        out["vtx_outside_building_footprint_max_m"] = (
            float(max(outside)) if outside else 0.0
        )
    else:
        out["vtx_outside_building_footprint_count"] = 0
        out["vtx_outside_building_footprint_max_m"] = None
    slabs = building_context.get("slabs") or [] if building_context else []
    slab_ys = []
    for slab in slabs:
        poly = slab.get("polygon") or []
        if poly:
            slab_ys.extend(float(pt[1]) for pt in poly if len(pt) >= 3)
    out["vtx_on_floor_slab_count"] = (
        int(sum(1 for y in ys if any(abs(y - sy) <= 0.1 for sy in slab_ys)))
        if ys and slab_ys
        else 0
    )
    wall_ext = building_context.get("wall_extensions") or [] if building_context else []
    knee_wall_top_ys = []
    for ext in wall_ext:
        if not ext.get("behind_knee_wall"):
            continue
        strip = ext.get("strip_corners") or []
        if strip:
            knee_wall_top_ys.append(max(float(pt[1]) for pt in strip if len(pt) >= 3))
    out["vtx_on_kneewall_top_count"] = (
        int(sum(1 for y in ys if any(abs(y - ky) <= 0.1 for ky in knee_wall_top_ys)))
        if ys and knee_wall_top_ys
        else 0
    )
    opposing_edge_lines = []
    for peer, peer_poly in peer_polys:
        if peer_poly is None:
            continue
        for pedge in _edge_records(peer.get("corners") or []):
            try:
                opposing_edge_lines.append(
                    LineString(
                        [
                            (float(pedge["a"][0]), float(pedge["a"][2])),
                            (float(pedge["b"][0]), float(pedge["b"][2])),
                        ]
                    )
                )
            except Exception as exc:
                note_geom_skip(exc, "exhaustive_features.opposing_edge")
                continue
    if corners and opposing_edge_lines:
        out["vtx_on_opposing_segment_edge_count"] = int(
            sum(
                1
                for c in corners
                if min(
                    float(Point(float(c[0]), float(c[2])).distance(line))
                    for line in opposing_edge_lines
                )
                <= 0.05
            )
        )
    else:
        out["vtx_on_opposing_segment_edge_count"] = 0
    if corners:
        all_seg_corners = []
        if building_context is not None:
            for seg in building_context.get("merged_roof_segments") or []:
                all_seg_corners.append(
                    np.asarray(seg.get("corners") or [], dtype=float)
                )
        valences = []
        for c in np.asarray(corners, dtype=float):
            count = 0
            for seg_pts in all_seg_corners:
                if seg_pts.size == 0:
                    continue
                if np.any(
                    np.linalg.norm(seg_pts - c, axis=1) <= GRID_SNAP_COARSE_TOL_M
                ):
                    count += 1
            valences.append(count)
        counter = Counter(valences)
        out["vtx_valence_max"] = max(valences) if valences else None
        out["vtx_valence_min"] = min(valences) if valences else None
        out["vtx_valence_distribution_entropy"] = _entropy_from_counts(counter)
    else:
        out["vtx_valence_max"] = None
        out["vtx_valence_min"] = None
        out["vtx_valence_distribution_entropy"] = None

    # Edge-level signals.
    edge_lengths = [e["length3"] for e in edges if e["length3"] > _EPS]
    edge_az = [e["azimuth"] for e in edges if e["azimuth"] is not None]
    out["edge_length_m_sum"] = float(sum(edge_lengths)) if edge_lengths else None
    out["edge_length_m_min"] = float(min(edge_lengths)) if edge_lengths else None
    out["edge_length_m_max"] = float(max(edge_lengths)) if edge_lengths else None
    out["edge_length_m_mean"] = float(np.mean(edge_lengths)) if edge_lengths else None
    out["edge_length_m_median"] = (
        float(np.median(edge_lengths)) if edge_lengths else None
    )
    out["edge_length_m_std"] = float(np.std(edge_lengths)) if edge_lengths else None
    out["edge_length_m_iqr"] = _iqr(edge_lengths)
    out["edge_length_m_gini"] = _gini(edge_lengths)
    out["edge_shortest_to_longest_ratio"] = (
        float(min(edge_lengths) / max(edge_lengths))
        if edge_lengths and max(edge_lengths) > _EPS
        else None
    )
    if edge_lengths:
        logs = np.log1p(np.asarray(edge_lengths, dtype=float))
        hist, _ = np.histogram(logs, bins=min(5, max(2, len(edge_lengths))))
        out["edge_length_entropy_log"] = _entropy_from_counts(
            Counter({i: int(v) for i, v in enumerate(hist) if v > 0})
        )
    else:
        out["edge_length_entropy_log"] = None
    out["edge_azimuth_deg_std"] = float(np.std(edge_az)) if edge_az else None
    if edge_az:
        bins, _ = np.histogram(edge_az, bins=np.arange(0.0, 181.0, 20.0))
        out["edge_azimuth_deg_modes_count"] = int(np.sum(bins > 0))
    else:
        out["edge_azimuth_deg_modes_count"] = None
    turns = [180.0 - a for a in angles]
    out["edge_turning_angle_sum_deg"] = (
        float(sum(abs(t) for t in turns)) if turns else None
    )
    out["edge_turning_angle_std_deg"] = float(np.std(turns)) if turns else None
    parallel = 0
    perpendicular = 0
    for i in range(len(edge_az)):
        for j in range(i + 1, len(edge_az)):
            diff = fe._azimuth_diff_deg(edge_az[i], edge_az[j])
            if diff < 3.0 or abs(diff - 180.0) < 3.0:
                parallel += 1
            if abs(diff - 90.0) < 3.0:
                perpendicular += 1
    pair_count = max(len(edge_az) * (len(edge_az) - 1) / 2, 1.0)
    out["edge_parallel_pair_count"] = parallel
    out["edge_parallel_pair_fraction"] = (
        float(parallel / pair_count) if edge_az else None
    )
    out["edge_perpendicular_pair_count"] = perpendicular
    out["edge_rectangularity_score"] = row.get("poly_right_angle_fraction")

    y_min = min(ys) if ys else None
    y_max = max(ys) if ys else None
    ridge_len = eave_len = hip_len = valley_len = rake_len = free_len = 0.0
    ridge_count = eave_count = hip_count = valley_count = rake_count = free_count = 0
    boundary_lines = []
    if part and part.get("footprint_xz"):
        part_poly = _xz_poly(part.get("footprint_xz") or [])
        if part_poly is not None:
            boundary_lines.append(part_poly.boundary)
    wall_centroids: list[tuple[float, float]] = []
    for _wid, info in (wall_index or {}).items():
        pts = info.get("corners") or []
        if not pts:
            continue
        wall_centroids.append(
            (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[2] for p in pts])),
            )
        )
    for e in edges:
        length = e["length3"]
        mid = e["mid"]
        touch_peer = False
        if (
            y_max is not None
            and e["horizontal"]
            and float(mid[1]) >= y_max - PLANE_INTERIOR_CLIP_MARGIN_M
        ):
            ridge_len += length
            ridge_count += 1
        elif (
            y_min is not None
            and e["horizontal"]
            and float(mid[1]) <= y_min + PLANE_INTERIOR_CLIP_MARGIN_M
        ):
            eave_len += length
            eave_count += 1
        elif e["descending"]:
            if (
                e["azimuth"] is not None
                and plane_az is not None
                and abs(fe._azimuth_diff_deg(e["azimuth"], plane_az) - 90.0) < 15.0
            ):
                hip_len += length
                hip_count += 1
            else:
                rake_len += length
                rake_count += 1
        elif e["ascending"]:
            valley_len += length
            valley_count += 1
        for _, peer_poly in peer_polys:
            if peer_poly is None:
                continue
            line = LineString(
                [
                    (float(e["a"][0]), float(e["a"][2])),
                    (float(e["b"][0]), float(e["b"][2])),
                ]
            )
            if _safe_intersects(line.buffer(SCAN_NOISE_M), peer_poly.boundary):
                touch_peer = True
                break
        if not touch_peer:
            free_len += length
            free_count += 1
    out["edge_is_ridge_count"] = ridge_count
    out["edge_is_hip_count"] = hip_count
    out["edge_is_valley_count"] = valley_count
    out["edge_is_eave_count"] = eave_count
    out["edge_is_rake_count"] = rake_count
    out["edge_is_free_count"] = free_count
    out["edge_ridge_length_m"] = ridge_len
    out["edge_eave_length_m"] = eave_len
    out["edge_free_length_m"] = free_len
    out["edge_free_length_fraction"] = (
        float(free_len / out["edge_length_m_sum"])
        if out["edge_length_m_sum"] and out["edge_length_m_sum"] > _EPS
        else None
    )
    out["edge_ridge_to_eave_length_ratio"] = (
        float(ridge_len / max(eave_len, _EPS))
        if (ridge_len > 0 or eave_len > 0)
        else None
    )
    out["edge_role_entropy"] = _entropy_from_counts(
        Counter(
            {
                "ridge": ridge_count,
                "hip": hip_count,
                "valley": valley_count,
                "eave": eave_count,
                "rake": rake_count,
                "free": free_count,
            }
        )
    )
    ridge_az = None
    eave_az = None
    for e in edges:
        if (
            ridge_az is None
            and e["horizontal"]
            and y_max is not None
            and float(e["mid"][1]) >= y_max - PLANE_INTERIOR_CLIP_MARGIN_M
        ):
            ridge_az = e["azimuth"]
        if (
            eave_az is None
            and e["horizontal"]
            and y_min is not None
            and float(e["mid"][1]) <= y_min + PLANE_INTERIOR_CLIP_MARGIN_M
        ):
            eave_az = e["azimuth"]

    dihedrals = []
    nm = np.asarray(plane[:3], dtype=float) if len(plane) >= 3 else None
    if nm is not None and np.linalg.norm(nm) > _EPS:
        nm = nm / np.linalg.norm(nm)
        for op in record.get("opposing_planes") or []:
            if len(op) < 3:
                continue
            on = np.asarray(op[:3], dtype=float)
            if np.linalg.norm(on) <= _EPS:
                continue
            on = on / np.linalg.norm(on)
            dihedrals.append(
                float(
                    math.degrees(
                        math.acos(max(-1.0, min(1.0, abs(float(np.dot(nm, on))))))
                    )
                )
            )
    out["edge_shared_dihedral_deg_min"] = float(min(dihedrals)) if dihedrals else None
    out["edge_shared_dihedral_deg_max"] = float(max(dihedrals)) if dihedrals else None
    out["edge_shared_dihedral_deg_mean"] = (
        float(np.mean(dihedrals)) if dihedrals else None
    )
    out["edge_shared_dihedral_deg_std"] = (
        float(np.std(dihedrals)) if dihedrals else None
    )
    out["edge_shared_dihedral_deg_count_under_120"] = int(
        sum(1 for d in dihedrals if d < 120.0)
    )
    out["edge_shared_dihedral_deg_count_over_150"] = int(
        sum(1 for d in dihedrals if d > 150.0)
    )

    wall_top_lines = []
    wall_azimuths = []
    resolved_wall_lengths = []
    resolved_wall_top_ys = []
    source_wall_ids = {
        m.get("source_wall_id")
        for m in record.get("cluster_members") or []
        if m.get("source_wall_id")
    }
    source_wall_stats: list[dict[str, Any]] = []
    source_wall_materials: Counter[str] = Counter()
    for wid, info in (wall_index or {}).items():
        pts = info.get("corners") or []
        if len(pts) < 2:
            continue
        top_y = max(float(p[1]) for p in pts)
        wall_id = info.get("wall_id") or wid
        if wall_id in source_wall_ids:
            resolved_wall_top_ys.append(top_y)
        top_pts = [
            (float(p[0]), float(p[2]))
            for p in pts
            if abs(float(p[1]) - top_y) <= SCAN_NOISE_M
        ]
        if len(top_pts) >= 2:
            top_line = None
            try:
                top_line = LineString(top_pts)
            except Exception:
                top_line = None
            if top_line is not None:
                wall_top_lines.append(top_line)
        az_guess = adv._wall_azimuth_incl_length_top_bottom(pts).get("azimuth_deg")
        if az_guess is not None:
            wall_azimuths.append(float(az_guess))
        if wall_id in source_wall_ids:
            wall_stats = adv._wall_azimuth_incl_length_top_bottom(pts)
            resolved_wall_lengths.append(float(wall_stats.get("length_m") or 0.0))
            source_wall_stats.append(info)
            material = (
                info.get("material")
                or info.get("wall_material")
                or info.get("classification")
            )
            if isinstance(material, str) and material:
                source_wall_materials[material] += 1
    midpoint_dists = []
    aligned_count = 0
    aligned_len = 0.0
    orth_count = 0
    for e in edges:
        if not wall_top_lines:
            break
        p = Point(float(e["mid"][0]), float(e["mid"][2]))
        d = min(float(p.distance(line)) for line in wall_top_lines)
        midpoint_dists.append(d)
        if e["azimuth"] is not None and wall_azimuths:
            nearest_wall_az = min(
                wall_azimuths, key=lambda waz: fe._azimuth_diff_deg(waz, e["azimuth"])
            )
            diff = fe._azimuth_diff_deg(nearest_wall_az, e["azimuth"])
            if diff <= 5.0 and d <= WALL_EDGE_ALIGN_DIST_M:
                aligned_count += 1
                aligned_len += e["length3"]
            if abs(diff - 90.0) <= 10.0:
                orth_count += 1
    out["edge_to_nearest_wall_top_m_min"] = (
        float(min(midpoint_dists)) if midpoint_dists else None
    )
    out["edge_to_nearest_wall_top_m_median"] = (
        float(np.median(midpoint_dists)) if midpoint_dists else None
    )
    out["edge_aligned_with_wall_count"] = aligned_count
    out["edge_aligned_with_wall_length_fraction"] = (
        float(aligned_len / out["edge_length_m_sum"])
        if out["edge_length_m_sum"] and out["edge_length_m_sum"] > _EPS
        else None
    )
    out["edge_orthogonal_to_wall_count"] = orth_count
    out["swall_length_median"] = (
        float(np.median(resolved_wall_lengths)) if resolved_wall_lengths else None
    )
    out["swall_top_y_min"] = (
        float(min(resolved_wall_top_ys)) if resolved_wall_top_ys else None
    )
    out["swall_top_y_max"] = (
        float(max(resolved_wall_top_ys)) if resolved_wall_top_ys else None
    )

    # Plane-fit / member distributions.
    out["plane_tilt_deg"] = plane_incl
    out["plane_slope_ratio"] = row.get("plane_rise_over_run")
    out["plane_pitch_is_medium"] = (
        bool(20.0 <= plane_incl < 50.0) if plane_incl is not None else None
    )
    out["plane_pitch_matches_dk_norm"] = (
        bool(25.0 <= plane_incl <= 50.0) if plane_incl is not None else None
    )
    out["plane_d_m"] = float(plane[3]) if len(plane) >= 4 else None
    member_residuals = []
    member_norm_cos = []
    member_ds = []
    for m in record.get("cluster_members") or []:
        mp = m.get("plane") or []
        if len(mp) >= 4 and corners:
            residual = fe._plane_residual_rms(corners, mp)
            if residual is not None:
                member_residuals.append(residual)
        if len(mp) >= 3 and len(plane) >= 3:
            mn = np.asarray(mp[:3], dtype=float)
            nn = np.asarray(plane[:3], dtype=float)
            if np.linalg.norm(mn) > _EPS and np.linalg.norm(nn) > _EPS:
                member_norm_cos.append(
                    float(abs(np.dot(mn / np.linalg.norm(mn), nn / np.linalg.norm(nn))))
                )
        if len(mp) >= 4:
            member_ds.append(float(mp[3]))
    out["slant_residual_mae_m"] = (
        float(np.mean(np.abs(member_residuals))) if member_residuals else None
    )
    out["slant_residual_max_m"] = (
        float(max(member_residuals)) if member_residuals else None
    )
    out["slant_residual_iqr_m"] = _iqr(member_residuals)
    out["slant_residual_skewness"] = _skewness(member_residuals)
    out["plane_fit_r2"] = (
        float(1.0 - (np.var(member_residuals) / (np.var(ys) + _EPS)))
        if member_residuals and ys
        else None
    )
    out["plane_top_y_m"] = max(ys) if ys else None
    out["plane_mid_y_m"] = float(np.mean(ys)) if ys else None
    out["plane_bottom_y_m"] = min(ys) if ys else None
    out["plane_y_extent_m"] = (max(ys) - min(ys)) if ys else None
    story_h = row.get("bld_typical_story_height_m")
    out["plane_y_extent_vs_story_height"] = (
        float(out["plane_y_extent_m"] / story_h)
        if out["plane_y_extent_m"] is not None and story_h not in (None, 0)
        else None
    )
    out["plane_normal_consistency_index"] = (
        float(np.mean(member_norm_cos)) if member_norm_cos else None
    )
    out["plane_curvature_rms_m_per_m"] = (
        float(
            np.sqrt(np.mean(np.square(member_residuals)))
            / max(math.sqrt(own_area or 0.0), _EPS)
        )
        if member_residuals and own_area
        else None
    )
    out["plane_waviness_amplitude_m"] = (
        float(max(member_residuals) - min(member_residuals))
        if len(member_residuals) >= 2
        else None
    )
    if member_ds:
        arr = np.asarray(member_ds, dtype=float)
        out["member_plane_d_m_mean"] = float(arr.mean())
        out["member_plane_d_m_std"] = float(arr.std(ddof=0))
        out["member_plane_d_m_min"] = float(arr.min())
        out["member_plane_d_m_max"] = float(arr.max())
        out["member_plane_d_m_range"] = float(arr.max() - arr.min())
        out["member_plane_d_m_median"] = float(np.median(arr))
    else:
        for key in (
            "member_plane_d_m_mean",
            "member_plane_d_m_std",
            "member_plane_d_m_min",
            "member_plane_d_m_max",
            "member_plane_d_m_range",
            "member_plane_d_m_median",
        ):
            out[key] = None
    story_counter = Counter(_story_indices(record))
    out["member_unique_source_stories"] = len(story_counter)
    out["member_story_entropy"] = _entropy_from_counts(story_counter)
    if record.get("cluster_members"):
        member_azs = []
        heur = Counter()
        trace_rules = Counter()
        for m in record.get("cluster_members") or []:
            fs = m.get("features") or {}
            if fs.get("segment_azimuth_deg") is not None:
                member_azs.append(float(fs["segment_azimuth_deg"]))
            label = m.get("heuristic_label")
            if label:
                heur[label] += 1
            trace = m.get("trace") or {}
            rule = trace.get("rule")
            if rule:
                trace_rules[rule] += 1
        if member_azs:
            rad = np.radians(member_azs)
            cmean = float(np.mean(np.cos(rad)))
            smean = float(np.mean(np.sin(rad)))
            resultant = math.sqrt(cmean * cmean + smean * smean)
            mean_az = math.degrees(math.atan2(smean, cmean)) % 360.0
            out["member_azimuth_circular_mean_deg"] = mean_az
            out["member_azimuth_circular_variance"] = float(1.0 - resultant)
            out["member_azimuth_resultant_length"] = float(resultant)
            out["member_azimuth_is_unimodal"] = bool(
                resultant > GABLE_UNIMODAL_RESULTANT_MIN
            )
            hist, _ = np.histogram(
                np.mod(member_azs, 180.0), bins=np.arange(0.0, 181.0, 15.0)
            )
            peak_counts = sorted((int(v) for v in hist if v > 0), reverse=True)
            if len(peak_counts) >= 2 and peak_counts[0] > 0:
                out["member_azimuth_bimodality_index"] = float(
                    peak_counts[1] / peak_counts[0]
                )
            else:
                out["member_azimuth_bimodality_index"] = 0.0 if peak_counts else None
        else:
            out["member_azimuth_circular_mean_deg"] = None
            out["member_azimuth_circular_variance"] = None
            out["member_azimuth_resultant_length"] = None
            out["member_azimuth_is_unimodal"] = None
            out["member_azimuth_bimodality_index"] = None
        out["member_heuristic_accepted_count"] = heur.get("accepted", 0)
        out["member_heuristic_rejected_count"] = heur.get("rejected", 0)
        total_heur = sum(heur.values())
        out["member_heuristic_rejected_fraction"] = (
            float(heur.get("rejected", 0) / total_heur) if total_heur else None
        )
        out["member_heuristic_uncertain_fraction"] = (
            float(
                (heur.get("uncertain", 0) + heur.get("not_evaluated", 0)) / total_heur
            )
            if total_heur
            else None
        )
        out["member_heuristic_unanimous"] = (
            bool(len([k for k, v in heur.items() if v > 0]) <= 1)
            if total_heur
            else None
        )
        out["member_trace_rule_entropy"] = _entropy_from_counts(trace_rules)
        out["member_trace_top_rule"] = (
            max(trace_rules.items(), key=lambda kv: kv[1])[0] if trace_rules else None
        )
        out["member_trace_rules_count"] = len(trace_rules)
    else:
        for key in (
            "member_azimuth_circular_mean_deg",
            "member_azimuth_circular_variance",
            "member_azimuth_resultant_length",
            "member_azimuth_is_unimodal",
            "member_azimuth_bimodality_index",
            "member_heuristic_accepted_count",
            "member_heuristic_rejected_count",
            "member_heuristic_rejected_fraction",
            "member_heuristic_uncertain_fraction",
            "member_heuristic_unanimous",
            "member_trace_rule_entropy",
            "member_trace_top_rule",
            "member_trace_rules_count",
        ):
            out[key] = (
                None
                if key
                not in (
                    "member_heuristic_accepted_count",
                    "member_heuristic_rejected_count",
                    "member_trace_rules_count",
                )
                else 0
            )
    out["cluster_confidence_score"] = _safe_float(
        (record.get("cluster_params") or {}).get("confidence")
    )
    normal_dot_min = _safe_float(
        (record.get("cluster_params") or {}).get("normal_dot_min")
    )
    out["cluster_plane_incl_tolerance_deg"] = (
        float(math.degrees(math.acos(max(-1.0, min(1.0, normal_dot_min)))))
        if normal_dot_min is not None
        else None
    )
    out["cluster_plane_azimuth_tolerance_deg"] = out["cluster_plane_incl_tolerance_deg"]
    out["cluster_plane_d_tolerance_m"] = _safe_float(
        (record.get("cluster_params") or {}).get("d_abs_max")
    )

    # Position/orientation.
    out["poly_centroid_x"] = float(centroid.x) if centroid is not None else None
    out["poly_centroid_y"] = float(np.mean(ys)) if ys else None
    out["poly_centroid_z"] = float(centroid.y) if centroid is not None else None
    out["poly_top_y_m_abs"] = out["plane_top_y_m"]
    out["poly_bottom_y_m_abs"] = out["plane_bottom_y_m"]
    out["poly_centroid_x_rel_bld_center"] = (
        float(out["poly_centroid_x"] - row["bld_footprint_centroid_x"])
        if out["poly_centroid_x"] is not None
        and row.get("bld_footprint_centroid_x") is not None
        else None
    )
    out["poly_centroid_z_rel_bld_center"] = (
        float(out["poly_centroid_z"] - row["bld_footprint_centroid_z"])
        if out["poly_centroid_z"] is not None
        and row.get("bld_footprint_centroid_z") is not None
        else None
    )
    if (
        centroid is not None
        and row.get("bld_footprint_centroid_x") is not None
        and row.get("bld_footprint_centroid_z") is not None
    ):
        d_center = math.hypot(
            float(centroid.x - row["bld_footprint_centroid_x"]),
            float(centroid.y - row["bld_footprint_centroid_z"]),
        )
        out["distance_to_footprint_center_m"] = float(d_center)
        out["distance_to_footprint_center_normalised"] = (
            float(d_center / math.sqrt(row["bld_footprint_area_m2"]))
            if row.get("bld_footprint_area_m2")
            else None
        )
    else:
        out["distance_to_footprint_center_m"] = None
        out["distance_to_footprint_center_normalised"] = None
    out["poly_in_footprint"] = row.get("inside_building_footprint")
    major_az = row.get("bld_footprint_principal_axis_deg")
    if (
        centroid is not None
        and major_az is not None
        and row.get("bld_footprint_area_m2")
    ):
        major_vec = np.asarray(
            [math.sin(math.radians(major_az)), math.cos(math.radians(major_az))],
            dtype=float,
        )
        rel = np.asarray(
            [
                centroid.x - (row.get("bld_footprint_centroid_x") or 0.0),
                centroid.y - (row.get("bld_footprint_centroid_z") or 0.0),
            ],
            dtype=float,
        )
        out["poly_long_axis_projection_m"] = float(np.dot(rel, major_vec))
        out["poly_short_axis_projection_m"] = float(
            np.dot(rel, np.asarray([major_vec[1], -major_vec[0]], dtype=float))
        )
        out["poly_long_axis_projection_normalised"] = float(
            out["poly_long_axis_projection_m"]
            / max(math.sqrt(row["bld_footprint_area_m2"]), _EPS)
        )
    else:
        out["poly_long_axis_projection_m"] = None
        out["poly_short_axis_projection_m"] = None
        out["poly_long_axis_projection_normalised"] = None
    out["height_above_ground_m"] = (
        float(out["poly_centroid_y"] - row["bld_y_min_m"])
        if out["poly_centroid_y"] is not None and row.get("bld_y_min_m") is not None
        else None
    )
    out["height_above_ground_fraction"] = (
        float(out["height_above_ground_m"] / row["bld_height_m"])
        if out["height_above_ground_m"] is not None
        and row.get("bld_height_m") not in (None, 0)
        else None
    )
    out["plane_azimuth_sin"] = (
        float(math.sin(math.radians(plane_az))) if plane_az is not None else None
    )
    out["plane_azimuth_cos"] = (
        float(math.cos(math.radians(plane_az))) if plane_az is not None else None
    )
    out["plane_azimuth_bucket_8"] = _bucket8(plane_az)
    out["plane_azimuth_bucket_16"] = _bucket16(plane_az)
    if plane_az is not None and major_az is not None:
        major_diff = fe._azimuth_diff_deg(plane_az, major_az)
        out["plane_az_vs_bld_minor_deg"] = float(abs(90.0 - major_diff))
        out["plane_az_parallel_to_major"] = bool(major_diff < 15.0)
        out["plane_az_perpendicular_to_major"] = bool(major_diff > 75.0)
        out["plane_az_diagonal_to_major"] = bool(35.0 <= major_diff <= 55.0)
    else:
        out["plane_az_vs_bld_minor_deg"] = None
        out["plane_az_parallel_to_major"] = None
        out["plane_az_perpendicular_to_major"] = None
        out["plane_az_diagonal_to_major"] = None
    out["plane_az_vs_nearest_wall_deg"] = (
        min(fe._azimuth_diff_deg(plane_az, waz) for waz in wall_azimuths)
        if plane_az is not None and wall_azimuths
        else None
    )
    out["plane_incl_sin"] = (
        float(math.sin(math.radians(plane_incl))) if plane_incl is not None else None
    )
    out["plane_incl_cos"] = (
        float(math.cos(math.radians(plane_incl))) if plane_incl is not None else None
    )
    out["plane_incl_bucket_5"] = _incl_bucket(plane_incl)
    out["plane_incl_is_dk_typical"] = (
        bool(25.0 <= plane_incl <= 50.0) if plane_incl is not None else None
    )
    out["plane_incl_too_steep_for_residential"] = (
        bool(plane_incl > 60.0) if plane_incl is not None else None
    )
    out["plane_incl_too_shallow_for_oblique"] = (
        bool(plane_incl < 5.0) if plane_incl is not None else None
    )
    normals = []
    ds_norm = []
    for m in record.get("cluster_members") or []:
        p = m.get("plane") or []
        if len(p) >= 4:
            normals.append(np.asarray(p[:3], dtype=float))
            ds_norm.append(float(p[3]))
    for idx, key in enumerate(("a", "b", "c")):
        vals = [float(n[idx]) for n in normals if np.linalg.norm(n) > _EPS]
        out[f"normals_{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"normals_{key}_std"] = float(np.std(vals)) if vals else None
    out["normals_d_mean"] = float(np.mean(ds_norm)) if ds_norm else None
    out["normals_d_std"] = float(np.std(ds_norm)) if ds_norm else None

    # Scale-invariant, neighbor, room/story, part/building.
    out["area_vs_footprint_area_ratio"] = (
        float(own_area / row["bld_footprint_area_m2"])
        if own_area and row.get("bld_footprint_area_m2")
        else None
    )
    out["perimeter_vs_footprint_perimeter_ratio"] = (
        float(row["poly_perimeter_xz_m"] / row["bld_footprint_perimeter_m"])
        if row.get("poly_perimeter_xz_m") and row.get("bld_footprint_perimeter_m")
        else None
    )
    part_poly = _xz_poly(part.get("footprint_xz") or []) if part else None
    out["area_vs_part_area_ratio"] = (
        float(own_area / part_poly.area)
        if own_area and part_poly is not None and part_poly.area > _EPS
        else None
    )
    out["perimeter_vs_part_perimeter_ratio"] = (
        float(row["poly_perimeter_xz_m"] / part_poly.length)
        if row.get("poly_perimeter_xz_m")
        and part_poly is not None
        and part_poly.length > _EPS
        else None
    )
    out["plane_top_y_vs_bld_height_ratio"] = (
        float(out["plane_top_y_m"] / row["bld_height_m"])
        if out.get("plane_top_y_m") is not None
        and row.get("bld_height_m") not in (None, 0)
        else None
    )
    out["plane_y_extent_vs_story_height_ratio"] = out["plane_y_extent_vs_story_height"]
    out["opposing_cluster_unique_count"] = len(
        set(record.get("opposing_cluster_canonicals") or [])
    )
    if record.get("opposing_planes"):
        op_incls = []
        op_diffs = []
        op_az_diffs = []
        for op in record.get("opposing_planes") or []:
            opz, opi = fe._plane_to_azimuth_incl(op)
            if not math.isnan(opi):
                op_incls.append(float(opi))
                if plane_incl is not None:
                    op_diffs.append(abs(float(opi) - plane_incl))
            if not math.isnan(opz) and plane_az is not None:
                op_az_diffs.append(fe._azimuth_diff_deg(float(opz), plane_az))
        out["opposing_incl_min_deg"] = float(min(op_incls)) if op_incls else None
        out["opposing_incl_max_deg"] = float(max(op_incls)) if op_incls else None
        out["opposing_incl_diff_mean_deg"] = (
            float(np.mean(op_diffs)) if op_diffs else None
        )
        out["opposing_is_gable_pair"] = bool(
            len(record.get("opposing_planes") or []) == 1
            and op_az_diffs
            and 160.0 <= op_az_diffs[0] <= 200.0
            and op_diffs
            and op_diffs[0] <= 5.0
        )
        out["opposing_gable_incl_asymmetry_deg"] = (
            op_diffs[0] if len(op_diffs) == 1 else None
        )
        out["opposing_gable_azimuth_offset_deg"] = (
            abs(op_az_diffs[0] - 180.0) if len(op_az_diffs) == 1 else None
        )
        out["opposing_is_hip_trio"] = bool(
            len(record.get("opposing_planes") or []) == 3
        )
        out["opposing_is_hip_quartet"] = bool(
            len(record.get("opposing_planes") or []) == 4
        )
        out["opposing_hip_closure_angle_deg"] = (
            float(sum(op_az_diffs)) if op_az_diffs else None
        )
    else:
        for key in (
            "opposing_incl_min_deg",
            "opposing_incl_max_deg",
            "opposing_incl_diff_mean_deg",
            "opposing_is_gable_pair",
            "opposing_gable_incl_asymmetry_deg",
            "opposing_gable_azimuth_offset_deg",
            "opposing_is_hip_trio",
            "opposing_is_hip_quartet",
            "opposing_hip_closure_angle_deg",
        ):
            out[key] = None
    same_part_count = 0
    cross_part_count = 0
    coplanar = []
    for peer, peer_poly in peer_polys:
        if peer_poly is None:
            continue
        if _same_part(peer, part, building_context):
            same_part_count += 1
        else:
            cross_part_count += 1
        pplane = peer.get("merged_plane") or []
        paz, pincl = (
            fe._plane_to_azimuth_incl(pplane)
            if pplane
            else (float("nan"), float("nan"))
        )
        if (
            plane_az is not None
            and plane_incl is not None
            and not math.isnan(paz)
            and not math.isnan(pincl)
        ):
            pdiff = (
                abs(float(pplane[3]) - float(plane[3]))
                if len(pplane) >= 4 and len(plane) >= 4
                else 999.0
            )
            if (
                fe._azimuth_diff_deg(plane_az, float(paz)) < 5.0
                and abs(plane_incl - float(pincl)) < 3.0
                and pdiff < COPLANAR_D_TOL_M
            ):
                coplanar.append(peer_poly)
    out["opposing_same_part_count"] = same_part_count
    out["opposing_cross_part_count"] = cross_part_count
    out["opposing_cross_part_fraction"] = (
        float(
            cross_part_count
            / max(
                (record.get("opposing_planes") and len(record.get("opposing_planes")))
                or 0,
                1,
            )
        )
        if record.get("opposing_planes")
        else None
    )
    out["coplanar_peer_count"] = len(coplanar)
    if seg_poly is not None and coplanar:
        union = _safe_unary_union(coplanar)
        out["coplanar_peer_union_area_m2"] = (
            float(union.area) if union is not None else None
        )
        out["coplanar_peer_overlap_area_m2"] = float(
            sum(seg_poly.intersection(p).area for p in coplanar)
        )
        out["coplanar_peer_is_rank_1_by_area"] = bool(
            own_area is not None and own_area >= max(float(p.area) for p in coplanar)
        )
    else:
        out["coplanar_peer_union_area_m2"] = None
        out["coplanar_peer_overlap_area_m2"] = None
        out["coplanar_peer_is_rank_1_by_area"] = None

    out["swall_unresolved_count"] = max(
        (row.get("cluster_member_count") or 0) - (row.get("swall_resolved_count") or 0),
        0,
    )
    exterior_count = int(
        sum(
            1
            for info in source_wall_stats
            if info.get("is_exterior") is True
            or info.get("exterior") is True
            or str(info.get("kind") or "").lower().startswith("exterior")
        )
    )
    out["swall_is_exterior_count"] = exterior_count
    out["swall_is_exterior_fraction"] = (
        float(exterior_count / len(source_wall_stats)) if source_wall_stats else None
    )
    out["swall_is_interior_count"] = max(len(source_wall_stats) - exterior_count, 0)
    out["swall_is_interior_fraction"] = (
        float(out["swall_is_interior_count"] / len(source_wall_stats))
        if source_wall_stats
        else None
    )
    out["swall_supports_only_interior"] = (
        bool(source_wall_stats) and exterior_count == 0
    )
    out["swall_supports_mostly_interior"] = (
        bool(source_wall_stats) and (out["swall_is_exterior_fraction"] or 0.0) <= 0.25
    )
    out["swall_has_door_count"] = int(
        sum(
            1
            for info in source_wall_stats
            if bool(info.get("has_door")) or bool(info.get("doors"))
        )
    )
    out["swall_has_window_count"] = int(
        sum(
            1
            for info in source_wall_stats
            if bool(info.get("has_window")) or bool(info.get("windows"))
        )
    )
    out["swall_material_entropy"] = _entropy_from_counts(source_wall_materials)
    if plane_az is not None and wall_azimuths:
        diffs = [fe._azimuth_diff_deg(plane_az, waz) for waz in wall_azimuths]
        out["swall_azimuth_alignment_mean_deg"] = float(np.mean(diffs))
    else:
        out["swall_azimuth_alignment_mean_deg"] = None

    room_ids = {
        rr.get("room_id")
        for rr in record.get("room_boundary_refs") or []
        if rr.get("room_id")
    }
    touched_room_ids = set(room_ids)
    touched_room_ids.update(
        m.get("source_room_id")
        for m in (record.get("cluster_members") or [])
        if m.get("source_room_id")
    )
    touched_room_ids.update(
        m.get("slab_room_id")
        for m in (record.get("cluster_members") or [])
        if m.get("slab_room_id")
    )
    out["room_id_count"] = len(room_ids)
    room_slab_polys = [_xz_poly(s.get("polygon") or []) for s in slabs]
    room_slab_pairs = [
        (slab, poly)
        for slab, poly in zip(slabs, room_slab_polys, strict=False)
        if poly is not None and slab.get("room_id") in touched_room_ids
    ]
    room_floor_areas = [
        float(poly.area) for _, poly in room_slab_pairs if poly.area > _EPS
    ]
    out["room_floor_area_total_m2"] = (
        float(sum(room_floor_areas)) if room_floor_areas else None
    )
    out["room_floor_area_mean_m2"] = (
        float(np.mean(room_floor_areas)) if room_floor_areas else None
    )
    if seg_poly is not None and room_slab_pairs:
        room_coverages = [
            float(seg_poly.intersection(poly).area / max(float(poly.area), _EPS))
            for _, poly in room_slab_pairs
        ]
        out["seg_room_coverage_fraction_min"] = float(min(room_coverages))
        out["seg_room_coverage_fraction_mean"] = float(np.mean(room_coverages))
        out["seg_room_coverage_fraction_max"] = float(max(room_coverages))
        total_room_area = sum(float(poly.area) for _, poly in room_slab_pairs)
        total_overlap = float(
            sum(seg_poly.intersection(poly).area for _, poly in room_slab_pairs)
        )
        out["seg_local_slant_fraction_of_touched_rooms"] = float(
            total_overlap / max(total_room_area, _EPS)
        )
    else:
        out["seg_room_coverage_fraction_min"] = None
        out["seg_room_coverage_fraction_mean"] = None
        out["seg_room_coverage_fraction_max"] = None
        out["seg_local_slant_fraction_of_touched_rooms"] = None
    out["room_has_dormer_count"] = 0
    if part is not None:
        out["part_story_count"] = len(part.get("stories") or [])
        part_story_indices = sorted(
            story
            for story in (part.get("stories") or [])
            if _coerce_int(story) is not None
        )
        part_story_indices = [_coerce_int(story) for story in part_story_indices]
        part_story_indices = [
            story for story in part_story_indices if story is not None
        ]
        out["part_story_index_min"] = (
            min(part_story_indices) if part_story_indices else None
        )
        out["part_story_index_max"] = (
            max(part_story_indices) if part_story_indices else None
        )
        out["part_story_index_range"] = (
            max(part_story_indices) - min(part_story_indices)
            if part_story_indices
            else None
        )
        out["part_has_basement"] = bool(
            part_story_indices and min(part_story_indices) < 0
        )
    else:
        out["part_story_count"] = None
        out["part_story_index_min"] = None
        out["part_story_index_max"] = None
        out["part_story_index_range"] = None
        out["part_has_basement"] = None
    room_story_indices = _story_indices(record)
    out["story_count_touched"] = len(room_story_indices)
    out["story_is_top_only"] = bool(
        len(room_story_indices) == 1
        and room_story_indices
        and row.get("bld_story_count")
        and room_story_indices[0] >= row["bld_story_count"] - 1
    )
    out["story_index_max"] = max(room_story_indices) if room_story_indices else None
    out["story_index_min"] = min(room_story_indices) if room_story_indices else None
    out["story_index_range"] = (
        (max(room_story_indices) - min(room_story_indices))
        if room_story_indices
        else None
    )
    out["segment_crosses_story_boundary"] = bool(len(room_story_indices) > 1)

    if part is not None:
        ge = part.get("gable_extension") or {}
        ridge_line = ge.get("ridge_line")
        uncovered = _xz_poly(ge.get("uncovered_region_xz") or [])
        out["part_gable_has_ridge_line"] = bool(ridge_line)
        if ridge_line and len(ridge_line) == 2:
            out["part_gable_ridge_length_m"] = float(
                math.hypot(
                    float(ridge_line[1][0] - ridge_line[0][0]),
                    float(ridge_line[1][2] - ridge_line[0][2]),
                )
            )
        else:
            out["part_gable_ridge_length_m"] = None
        out["part_gable_uncovered_region_area_m2"] = (
            float(uncovered.area) if uncovered is not None else None
        )
        if seg_poly is not None and uncovered is not None and own_area:
            out["seg_covers_gable_uncovered_region_fraction"] = float(
                seg_poly.intersection(uncovered).area / own_area
            )
        else:
            out["seg_covers_gable_uncovered_region_fraction"] = None
        if ridge_line and seg_poly is not None:
            ridge = LineString(
                [
                    (float(ridge_line[0][0]), float(ridge_line[0][2])),
                    (float(ridge_line[1][0]), float(ridge_line[1][2])),
                ]
            )
            out["seg_intersects_gable_ridge_line"] = _safe_intersects(
                seg_poly.buffer(SCAN_NOISE_M), ridge
            )
        else:
            out["seg_intersects_gable_ridge_line"] = None
        out["part_footprint_aspect_ratio"] = row.get("part_footprint_perimeter_m")
    else:
        for key in (
            "part_gable_has_ridge_line",
            "part_gable_ridge_length_m",
            "part_gable_uncovered_region_area_m2",
            "seg_covers_gable_uncovered_region_fraction",
            "seg_intersects_gable_ridge_line",
        ):
            out[key] = None

    # V3 idle/context features.
    slabs = building_context.get("slabs") or [] if building_context else []
    slab_polys = [_xz_poly(s.get("polygon") or []) for s in slabs]
    slab_polys = [p for p in slab_polys if p is not None]
    out["bld_slab_count"] = (
        building_context.get("slab_count") if building_context else None
    )
    out["bld_flat_ceiling_count"] = (
        building_context.get("flat_ceiling_count") if building_context else None
    )
    out["bld_slanted_roof_count"] = (
        building_context.get("slanted_roof_count") if building_context else None
    )
    out["bld_roof_proposal_count"] = (
        building_context.get("roof_proposals_count") if building_context else None
    )
    out["bld_merged_roof_segment_count"] = (
        building_context.get("merged_roof_segments_count") if building_context else None
    )
    out["bld_dormer_count"] = (
        building_context.get("dormer_count") if building_context else None
    )
    out["bld_unresolved_region_count"] = (
        building_context.get("unresolved_region_count") if building_context else None
    )
    out["bld_part_count"] = (
        building_context.get("part_count") if building_context else None
    )
    out["bld_proposal_density"] = (
        float(out["bld_merged_roof_segment_count"] / row["bld_footprint_area_m2"])
        if out.get("bld_merged_roof_segment_count") is not None
        and row.get("bld_footprint_area_m2")
        else None
    )
    out["bld_complexity_index"] = (
        float(
            (out.get("bld_part_count") or 0)
            * (row.get("bld_story_count") or 0)
            * (row.get("bld_footprint_bbox_aspect") or 0)
        )
        if out.get("bld_part_count") is not None
        else None
    )
    out["seg_nearest_slab_distance_m"] = _nearest_distance_to_polys(
        seg_poly, slab_polys
    )
    out["seg_is_above_slab_count"] = int(
        sum(1 for p in slab_polys if seg_poly is not None and p.intersects(seg_poly))
    )
    out["bld_slab_area_total_m2"] = (
        float(sum(p.area for p in slab_polys)) if slab_polys else 0.0
    )
    slanted_polys = (
        [
            _xz_poly(s.get("corners") or s.get("footprint_xz") or [])
            for s in (building_context.get("slanted_roofs") or [])
        ]
        if building_context
        else []
    )
    slanted_polys = [p for p in slanted_polys if p is not None]
    out["bld_slanted_area_total_m2"] = (
        float(sum(p.area for p in slanted_polys)) if slanted_polys else 0.0
    )
    out["bld_slanted_area_fraction"] = (
        float(out["bld_slanted_area_total_m2"] / row["bld_footprint_area_m2"])
        if row.get("bld_footprint_area_m2")
        else None
    )
    out["seg_is_small_partial_room_slant"] = bool(
        out.get("room_id_count") == 1
        and (out.get("seg_room_coverage_fraction_max") or 1.0) <= 0.35
        and (out.get("bld_slanted_area_fraction") or 1.0) <= 0.35
    )
    out["seg_small_partial_room_slant_score"] = (
        float(
            max(0.0, 1.0 - min(out.get("seg_room_coverage_fraction_mean") or 1.0, 1.0))
            * max(
                0.0,
                1.0 - min((out.get("bld_slanted_area_fraction") or 1.0) / 0.35, 1.0),
            )
        )
        if out.get("seg_room_coverage_fraction_mean") is not None
        and out.get("bld_slanted_area_fraction") is not None
        else None
    )

    wall_ext = building_context.get("wall_extensions") or [] if building_context else []
    ext_polys = [_xz_poly(w.get("strip_corners") or []) for w in wall_ext]
    ext_pairs = [
        (w, p) for w, p in zip(wall_ext, ext_polys, strict=False) if p is not None
    ]
    if seg_poly is not None:
        touching = [(w, p) for w, p in ext_pairs if p.intersects(seg_poly)]
        out["seg_wall_extension_contact_count"] = len(touching)
        out["seg_wall_extension_total_length_m"] = (
            float(sum(p.length for _, p in touching)) if touching else 0.0
        )
        out["seg_has_behind_knee_wall_extension"] = bool(
            any(w.get("behind_knee_wall") for w, _ in touching)
        )
    else:
        out["seg_wall_extension_contact_count"] = 0
        out["seg_wall_extension_total_length_m"] = 0.0
        out["seg_has_behind_knee_wall_extension"] = None
    out["bld_wall_extension_count"] = len(wall_ext)
    out["bld_knee_wall_count"] = sum(1 for w in wall_ext if w.get("behind_knee_wall"))
    knee_pairs = [(w, p) for w, p in ext_pairs if w.get("behind_knee_wall")]
    out["bld_knee_wall_total_length_m"] = (
        float(sum(p.length for _, p in knee_pairs)) if knee_pairs else 0.0
    )
    support_terms = []
    for _w, p in knee_pairs:
        shell_len = float(p.length)
        if shell_len > _EPS:
            support_terms.append(
                float(p.intersection(seg_poly).length / shell_len)
                if seg_poly is not None
                else 0.0
            )
    out["bld_knee_wall_occupied_shell_support_mean"] = (
        float(np.mean(support_terms)) if support_terms else None
    )
    out["bld_knee_wall_dropped_count"] = int(
        sum(
            1
            for w, _ in knee_pairs
            if str((w.get("trace") or {}).get("decision", "")).lower() == "dropped"
        )
    )
    if seg_poly is not None and knee_pairs:
        nearest_knee = min(
            knee_pairs, key=lambda item: float(seg_poly.distance(item[1]))
        )
        nearest_poly = nearest_knee[1]
        nearest_wall = nearest_knee[0]
        out["seg_has_kneewall_below"] = bool(seg_poly.distance(nearest_poly) <= 1.0)
        out["seg_kneewall_distance_m"] = float(seg_poly.distance(nearest_poly))
        out["seg_kneewall_span_fraction"] = float(
            seg_poly.intersection(nearest_poly.buffer(WALL_SUPPORT_BUFFER_M)).area
            / max(seg_poly.area, _EPS)
        )
        strip = nearest_wall.get("strip_corners") or []
        out["seg_kneewall_height_m"] = (
            float(max(float(pt[1]) for pt in strip) - min(float(pt[1]) for pt in strip))
            if strip
            else None
        )
        out["seg_kneewall_eave_consistency"] = (
            float(
                1.0
                / (
                    1.0
                    + abs(
                        (row.get("plane_bottom_y_m") or 0.0)
                        - max(float(pt[1]) for pt in strip)
                    )
                )
            )
            if strip and row.get("plane_bottom_y_m") is not None
            else None
        )
    else:
        out["seg_has_kneewall_below"] = False if wall_ext else None
        out["seg_kneewall_distance_m"] = None
        out["seg_kneewall_span_fraction"] = None
        out["seg_kneewall_height_m"] = None
        out["seg_kneewall_eave_consistency"] = None

    flats = building_context.get("flat_ceilings") or [] if building_context else []
    flat_polys = [_xz_poly(c.get("footprint_xz") or []) for c in flats]
    flat_pairs = [
        (c, p) for c, p in zip(flats, flat_polys, strict=False) if p is not None
    ]
    if seg_poly is not None and own_area:
        overlap = sum(float(seg_poly.intersection(p).area) for _, p in flat_pairs)
        out["seg_flat_ceiling_overlap_m2"] = overlap
        out["seg_flat_ceiling_overlap_fraction"] = float(overlap / own_area)
        out["seg_is_mostly_a_flat_ceiling"] = bool(
            overlap / own_area > FLAT_CEILING_DOMINANCE_RATIO
            and (plane_incl or 0.0) < 10.0
        )
    else:
        out["seg_flat_ceiling_overlap_m2"] = None
        out["seg_flat_ceiling_overlap_fraction"] = None
        out["seg_is_mostly_a_flat_ceiling"] = None
    out["bld_flat_ceiling_area_total_m2"] = (
        float(sum(p.area for _, p in flat_pairs)) if flat_pairs else 0.0
    )

    gaps = building_context.get("gaps") or [] if building_context else []
    gap_polys = [_xz_poly(g.get("footprint_xz") or []) for g in gaps]
    gap_pairs = [(g, p) for g, p in zip(gaps, gap_polys, strict=False) if p is not None]
    out["seg_gap_proximity_m"] = _nearest_distance_to_polys(
        seg_poly, [p for _, p in gap_pairs]
    )
    out["seg_gap_adjacent_count"] = int(
        sum(
            1
            for _, p in gap_pairs
            if seg_poly is not None and seg_poly.distance(p) <= 0.5
        )
    )
    out["bld_gap_floor_area_m2"] = (
        float(sum(p.area for _, p in gap_pairs)) if gap_pairs else 0.0
    )
    out["bld_gap_to_footprint_ratio"] = (
        float(out["bld_gap_floor_area_m2"] / row["bld_footprint_area_m2"])
        if row.get("bld_footprint_area_m2")
        else None
    )
    out["bld_gap_status_entropy"] = _entropy_from_counts(
        Counter(g.get("status") for g, _ in gap_pairs if g.get("status"))
    )

    unresolved = building_context.get("unresolved") or [] if building_context else []
    unresolved_polys = [_xz_poly(u.get("footprint_xz") or []) for u in unresolved]
    unresolved_pairs = [
        (u, p)
        for u, p in zip(unresolved, unresolved_polys, strict=False)
        if p is not None
    ]
    out["seg_unresolved_region_proximity_m"] = _nearest_distance_to_polys(
        seg_poly, [p for _, p in unresolved_pairs]
    )
    if seg_poly is not None:
        out["seg_unresolved_region_overlap_m2"] = float(
            sum(seg_poly.intersection(p).area for _, p in unresolved_pairs)
        )
    else:
        out["seg_unresolved_region_overlap_m2"] = None
    out["bld_unresolved_region_area_total_m2"] = (
        float(sum(p.area for _, p in unresolved_pairs)) if unresolved_pairs else 0.0
    )
    if unresolved_pairs:
        reasons = Counter(
            u.get("reason") for u, _ in unresolved_pairs if u.get("reason")
        )
        out["bld_unresolved_reason_mode"] = (
            max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else None
        )
    else:
        out["bld_unresolved_reason_mode"] = None

    trace = None
    if building_context:
        for seg in building_context.get("merged_roof_segments") or []:
            if seg.get("id") == record.get("proposal_id"):
                trace = seg.get("trace")
                break
    trace = trace or {}
    out["trace_stage"] = trace.get("stage")
    out["trace_rule"] = trace.get("rule")
    reason_tokens = str(trace.get("decision_reason") or "").split()
    out["trace_decision_reason_token_entropy"] = (
        _entropy_from_counts(Counter(reason_tokens)) if reason_tokens else None
    )

    # Shape aliases and additional polygon descriptors.
    if seg_poly is not None:
        hull = seg_poly.convex_hull
        minx, miny, maxx, maxy = seg_poly.bounds
        out["poly_convex_hull_perimeter_m"] = float(hull.length)
        out["poly_bbox_aligned_length_m"] = float(max(maxx - minx, maxy - miny))
        out["poly_bbox_aligned_width_m"] = float(min(maxx - minx, maxy - miny))
        out["poly_interior_ring_count"] = len(getattr(seg_poly, "interiors", []))
        out["poly_outside_footprint_area_m2"] = row.get("overshoot_area_m2")
        out["poly_inside_footprint_area_m2"] = (
            float(max((own_area or 0.0) - (row.get("overshoot_area_m2") or 0.0), 0.0))
            if own_area is not None
            else None
        )
        out["poly_outside_footprint_fraction"] = (
            float((row.get("overshoot_area_m2") or 0.0) / own_area)
            if own_area
            else None
        )
        out["distance_to_footprint_edge_normalised"] = (
            float(
                (row.get("distance_to_footprint_edge_m") or 0.0)
                / math.sqrt(row.get("bld_footprint_area_m2") or 1.0)
            )
            if row.get("distance_to_footprint_edge_m") is not None
            and row.get("bld_footprint_area_m2")
            else None
        )
        try:
            shell_core = polylabel(seg_poly)
            out["plane_interior_crossing_depth_m"] = float(
                shell_core.distance(seg_poly.boundary)
            )
        except Exception:
            out["plane_interior_crossing_depth_m"] = None
        footprint_corners = _footprint_corners(boundary_poly)
        if centroid is not None and footprint_corners:
            corner_d = min(
                math.hypot(centroid.x - x, centroid.y - y) for x, y in footprint_corners
            )
            out["poly_is_near_footprint_corner"] = bool(corner_d <= 1.0)
        else:
            out["poly_is_near_footprint_corner"] = None
        out["poly_overhang_exceeds_0p5m"] = bool(
            (out.get("poly_outside_footprint_fraction") or 0.0) > SCAN_NOISE_M
            and (out.get("vtx_outside_building_footprint_max_m") or 0.0) > 0.5
        )
    else:
        for key in (
            "poly_convex_hull_perimeter_m",
            "poly_bbox_aligned_length_m",
            "poly_bbox_aligned_width_m",
            "poly_interior_ring_count",
            "poly_inside_footprint_area_m2",
            "poly_outside_footprint_area_m2",
            "poly_outside_footprint_fraction",
            "distance_to_footprint_edge_normalised",
            "plane_interior_crossing_depth_m",
            "poly_is_near_footprint_corner",
            "poly_overhang_exceeds_0p5m",
        ):
            out[key] = None
    flow_vec = _flow_vector_xz(plane)
    if centroid is not None and boundary_poly is not None and flow_vec is not None:
        fwd_exit = _ray_exit_distance(centroid, flow_vec, boundary_poly)
        rev_exit = _ray_exit_distance(centroid, -flow_vec, boundary_poly)
        out["plane_downslope_exit_distance_to_footprint_m"] = fwd_exit
        out["plane_upslope_exit_distance_to_footprint_m"] = rev_exit
        out["plane_downslope_points_outside"] = (
            bool(fwd_exit <= rev_exit + SCAN_NOISE_M)
            if fwd_exit is not None and rev_exit is not None
            else None
        )
        out["plane_downslope_exit_vs_reverse_ratio"] = (
            float(fwd_exit / max(rev_exit, _EPS))
            if fwd_exit is not None and rev_exit is not None
            else None
        )
    else:
        out["plane_downslope_exit_distance_to_footprint_m"] = None
        out["plane_upslope_exit_distance_to_footprint_m"] = None
        out["plane_downslope_points_outside"] = None
        out["plane_downslope_exit_vs_reverse_ratio"] = None
    if boundary_poly is not None and edges:
        boundary_line = boundary_poly.boundary
        exterior_edge_len = 0.0
        eave_shell_dists: list[float] = []
        eave_contact_len = 0.0
        total_edge_len = 0.0
        for e in edges:
            try:
                line = LineString(
                    [
                        (float(e["a"][0]), float(e["a"][2])),
                        (float(e["b"][0]), float(e["b"][2])),
                    ]
                )
            except Exception as exc:
                note_geom_skip(exc, "exhaustive_features.edge_line")
                continue
            total_edge_len += float(e["length3"])
            dist_shell = float(line.distance(boundary_line))
            if dist_shell <= WALL_SUPPORT_BUFFER_M:
                exterior_edge_len += float(e["length3"])
            is_eave = bool(
                y_min is not None
                and e["horizontal"]
                and float(e["mid"][1]) <= y_min + PLANE_INTERIOR_CLIP_MARGIN_M
            )
            if is_eave:
                eave_shell_dists.append(dist_shell)
                if dist_shell <= WALL_SUPPORT_BUFFER_M:
                    eave_contact_len += float(e["length3"])
        out["plane_exterior_edge_contact_fraction"] = (
            float(exterior_edge_len / max(total_edge_len, _EPS))
            if total_edge_len > _EPS
            else None
        )
        out["plane_eave_edge_to_exterior_shell_m"] = (
            float(min(eave_shell_dists)) if eave_shell_dists else None
        )
        out["plane_eave_exterior_contact_fraction"] = (
            float(eave_contact_len / max(eave_len, _EPS)) if eave_len > _EPS else None
        )
    else:
        out["plane_exterior_edge_contact_fraction"] = None
        out["plane_eave_edge_to_exterior_shell_m"] = None
        out["plane_eave_exterior_contact_fraction"] = None
    out["poly_mrr_length_m"] = row.get("poly_min_rect_major_m")
    out["poly_mrr_width_m"] = row.get("poly_min_rect_minor_m")
    out["rectangularity"] = row.get("poly_min_rect_fill_ratio")
    out["form_factor"] = (
        float((own_area or 0.0) / row["poly_perimeter_xz_m"])
        if own_area is not None and row.get("poly_perimeter_xz_m") not in (None, 0)
        else None
    )
    out["poly_reflex_count"] = out.get("poly_reflex_corner_count")
    out["reock"] = row.get("reock_compactness")
    out["schwartzberg"] = row.get("schwartzberg_compactness")
    out["convex_deficiency"] = (
        1.0 - row["poly_convex_hull_ratio"]
        if row.get("poly_convex_hull_ratio") is not None
        else None
    )
    out["circularity"] = row.get("polsby_popper")
    out["elongation"] = (
        float(1.0 - 1.0 / row["poly_min_rect_aspect"])
        if row.get("poly_min_rect_aspect") not in (None, 0)
        else None
    )
    out["waviness"] = (
        float(row["poly_perimeter_xz_m"] / out["poly_convex_hull_perimeter_m"])
        if row.get("poly_perimeter_xz_m") and out.get("poly_convex_hull_perimeter_m")
        else None
    )
    out["roundness_ratio"] = (
        float(
            4.0
            * (own_area or 0.0)
            / (math.pi * (row.get("poly_min_rect_major_m") or 1.0) ** 2)
        )
        if own_area is not None and row.get("poly_min_rect_major_m")
        else None
    )
    out["radial_distance_mean_m"] = row.get("radial_mean_m")
    out["radial_distance_std_m"] = row.get("radial_std_m")
    out["radial_distance_cv"] = row.get("radial_cv")
    radial_sig = (
        adv._radial_signature(_ring_points_xy(corners), n_samples=16)
        if len(corners) >= 3
        else []
    )
    out["radial_distance_autocorr_lag1"] = _lag1_autocorr(radial_sig)
    out.update(_fourier_boundary_features(corners))
    out.update(_hu_projection_features(corners))
    out.update(_point_cloud_shape_features(corners))
    if corners:
        vx = float(np.mean([float(c[0]) for c in corners]))
        vy = float(np.mean([float(c[1]) for c in corners]))
        vz = float(np.mean([float(c[2]) for c in corners]))
        out["poly_vertex_centroid_x"] = vx
        out["poly_vertex_centroid_y"] = vy
        out["poly_vertex_centroid_z"] = vz
        cx = row.get("poly_centroid_x")
        cy = row.get("poly_centroid_y")
        cz = row.get("poly_centroid_z")
        out["poly_centroid_divergence_m"] = (
            float(math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2 + (vz - cz) ** 2))
            if None not in (cx, cy, cz)
            else None
        )
    else:
        out["poly_vertex_centroid_x"] = None
        out["poly_vertex_centroid_y"] = None
        out["poly_vertex_centroid_z"] = None
        out["poly_centroid_divergence_m"] = None
    out["poly_centroid_utm_e"] = row.get("poly_centroid_x")
    out["poly_centroid_utm_n"] = row.get("poly_centroid_z")
    out["poly_centroid_utm_utm_n"] = row.get("poly_centroid_z")
    if seg_poly is not None and not seg_poly.is_empty:
        try:
            pole = polylabel(seg_poly, tolerance=0.01)
            radius = float(pole.distance(seg_poly.boundary))
        except Exception:
            pole = None
            radius = None
        out["poly_min_inscribed_circle_radius_m"] = radius
        out["poly_pole_of_inaccessibility_m"] = radius
        out["poly_is_sliver"] = bool(
            (row.get("poly_min_width_m") or 99.0) < THIN_SLIVER_WIDTH_M
            and (row.get("poly_bbox_aspect") or 0.0) > 10.0
        )
        pts = np.asarray(corners, dtype=float) if corners else np.empty((0, 3))
        pinch = False
        if len(pts) >= 4:
            for i in range(len(pts)):
                for j in range(i + 2, len(pts)):
                    if i == 0 and j == len(pts) - 1:
                        continue
                    if float(np.linalg.norm(pts[i] - pts[j])) < 0.1:
                        pinch = True
                        break
                if pinch:
                    break
        out["poly_has_pinch_point"] = pinch
        try:
            rot180 = affinity.rotate(seg_poly, 180.0, origin="centroid")
            out["symmetry_rot180_iou"] = _symmetry_iou(seg_poly, rot180)
            cx = float(seg_poly.centroid.x)
            cy = float(seg_poly.centroid.y)
            refl_x = affinity.scale(seg_poly, xfact=-1.0, yfact=1.0, origin=(cx, cy))
            refl_z = affinity.scale(seg_poly, xfact=1.0, yfact=-1.0, origin=(cx, cy))
            out["symmetry_reflect_x_iou"] = _symmetry_iou(seg_poly, refl_x)
            out["symmetry_reflect_z_iou"] = _symmetry_iou(seg_poly, refl_z)
            principal = row.get("poly_min_rect_azimuth_deg") or 0.0
            aligned = affinity.rotate(seg_poly, -principal, origin="centroid")
            refl_principal = affinity.rotate(
                affinity.scale(aligned, xfact=-1.0, yfact=1.0, origin="centroid"),
                principal,
                origin="centroid",
            )
            out["symmetry_reflect_principal_iou"] = _symmetry_iou(
                seg_poly, refl_principal
            )
            candidates = {
                0.0: out["symmetry_reflect_x_iou"],
                90.0: out["symmetry_reflect_z_iou"],
                float(principal): out["symmetry_reflect_principal_iou"],
            }
            best = max(
                (
                    (angle, score)
                    for angle, score in candidates.items()
                    if score is not None
                ),
                key=lambda item: item[1],
                default=(None, None),
            )
            out["symmetry_best_reflection_deg"] = best[0]
        except Exception:
            out["symmetry_rot180_iou"] = None
            out["symmetry_reflect_x_iou"] = None
            out["symmetry_reflect_z_iou"] = None
            out["symmetry_reflect_principal_iou"] = None
            out["symmetry_best_reflection_deg"] = None
    else:
        for key in (
            "poly_min_inscribed_circle_radius_m",
            "poly_pole_of_inaccessibility_m",
            "poly_is_sliver",
            "poly_has_pinch_point",
            "symmetry_rot180_iou",
            "symmetry_reflect_x_iou",
            "symmetry_reflect_z_iou",
            "symmetry_reflect_principal_iou",
            "symmetry_best_reflection_deg",
        ):
            out[key] = None

    # Coverage / relational ranks.
    same_part_peers = [
        peer_poly
        for peer, peer_poly in peer_polys
        if peer_poly is not None and _same_part(peer, part, building_context)
    ]
    all_seg_polys = [p for _, p in peer_polys if p is not None]
    if seg_poly is not None and part_poly is not None:
        union = (
            _safe_unary_union([seg_poly, *same_part_peers])
            if same_part_peers
            else seg_poly
        )
        if union is None:
            union = seg_poly
        out["part_coverage_union_area_m2"] = float(union.area)
        out["part_coverage_union_to_footprint_ratio"] = (
            float(union.area / part_poly.area) if part_poly.area > _EPS else None
        )
        out["part_coverage_gap_area_m2"] = float(max(part_poly.area - union.area, 0.0))
        out["part_coverage_gap_fraction"] = (
            float(out["part_coverage_gap_area_m2"] / part_poly.area)
            if part_poly.area > _EPS
            else None
        )
        cover_sum = own_area + sum(float(p.area) for p in same_part_peers)
        out["part_coverage_over_cover_area_m2"] = float(
            max(cover_sum - union.area, 0.0)
        )
        out["part_coverage_over_cover_fraction"] = float(
            out["part_coverage_over_cover_area_m2"] / max(union.area, _EPS)
        )
        contrib_union = _safe_unary_union(same_part_peers) if same_part_peers else None
        if contrib_union is not None:
            out["this_seg_coverage_contribution_m2"] = float(
                max(union.area - contrib_union.area, 0.0)
            )
            out["this_seg_is_redundant"] = bool(
                out["this_seg_coverage_contribution_m2"] < ROOF_REDUNDANT_AREA_M2
            )
        else:
            out["this_seg_coverage_contribution_m2"] = own_area
            out["this_seg_is_redundant"] = False
        out["this_seg_coverage_contribution_fraction"] = (
            float(out["this_seg_coverage_contribution_m2"] / union.area)
            if union.area > _EPS
            else None
        )
        gap_region = part_poly.difference(union)
        out["this_seg_covers_gap_region"] = bool(
            not gap_region.is_empty and seg_poly.intersects(gap_region)
        )
    else:
        for key in (
            "part_coverage_union_area_m2",
            "part_coverage_union_to_footprint_ratio",
            "part_coverage_gap_area_m2",
            "part_coverage_gap_fraction",
            "part_coverage_over_cover_area_m2",
            "part_coverage_over_cover_fraction",
            "this_seg_coverage_contribution_m2",
            "this_seg_is_redundant",
            "this_seg_coverage_contribution_fraction",
            "this_seg_covers_gap_region",
        ):
            out[key] = None
    if seg_poly is not None and boundary_poly is not None:
        all_union = (
            _safe_unary_union([seg_poly, *all_seg_polys]) if all_seg_polys else seg_poly
        )
        if all_union is None:
            all_union = seg_poly
        out["bld_coverage_union_area_m2"] = float(all_union.area)
        out["bld_coverage_union_to_footprint_ratio"] = (
            float(all_union.area / boundary_poly.area)
            if boundary_poly.area > _EPS
            else None
        )
        out["bld_coverage_gap_fraction"] = (
            float(max(boundary_poly.area - all_union.area, 0.0) / boundary_poly.area)
            if boundary_poly.area > _EPS
            else None
        )
        out["bld_coverage_over_cover_fraction"] = float(
            max(
                (own_area or 0.0)
                + sum(float(p.area) for p in all_seg_polys)
                - all_union.area,
                0.0,
            )
            / max(all_union.area, _EPS)
        )
    else:
        out["bld_coverage_union_area_m2"] = None
        out["bld_coverage_union_to_footprint_ratio"] = None
        out["bld_coverage_gap_fraction"] = None
        out["bld_coverage_over_cover_fraction"] = None
    top_ys = [row.get("plane_top_y_m")] + [
        max(float(c[1]) for c in (peer.get("corners") or []))
        for peer, _ in peer_polys
        if peer.get("corners")
    ]
    bottom_ys = [row.get("plane_bottom_y_m")] + [
        min(float(c[1]) for c in (peer.get("corners") or []))
        for peer, _ in peer_polys
        if peer.get("corners")
    ]
    top_vals = [v for v in top_ys if v is not None]
    bottom_vals = [v for v in bottom_ys if v is not None]
    if top_vals and bottom_vals and row.get("plane_y_extent_m") is not None:
        part_range = max(top_vals) - min(bottom_vals)
        out["part_y_coverage_range_m"] = float(part_range)
        out["this_seg_y_range_fraction"] = float(
            row["plane_y_extent_m"] / max(part_range, _EPS)
        )
        out["this_seg_is_above_part_median_y"] = bool(
            (row.get("plane_mid_y_m") or 0.0) >= float(np.median(top_vals))
        )
    else:
        out["part_y_coverage_range_m"] = None
        out["this_seg_y_range_fraction"] = None
        out["this_seg_is_above_part_median_y"] = None

    # Building/part/cluster ranks.
    peer_areas = [float(p.area) for _, p in peer_polys if p is not None]
    if own_area is not None:
        sorted([own_area, *peer_areas], reverse=True)
        out["seg_rank_in_bld_by_area"] = 1 + sum(1 for a in peer_areas if a > own_area)
        out["seg_percentile_area_in_bld"] = float(
            sum(1 for a in peer_areas if a <= own_area) / max(len(peer_areas), 1)
        )
    else:
        out["seg_rank_in_bld_by_area"] = None
        out["seg_percentile_area_in_bld"] = None
    peer_tops = [
        max(float(c[1]) for c in (peer.get("corners") or []))
        for peer, _ in peer_polys
        if peer.get("corners")
    ]
    if row.get("plane_top_y_m") is not None:
        out["seg_rank_in_bld_by_top_y"] = 1 + sum(
            1 for y in peer_tops if y > row["plane_top_y_m"]
        )
        out["segment_height_vs_max_segment_height"] = (
            float(row["plane_top_y_m"] / max([row["plane_top_y_m"], *peer_tops]))
            if [row["plane_top_y_m"], *peer_tops]
            else None
        )
    else:
        out["seg_rank_in_bld_by_top_y"] = None
        out["segment_height_vs_max_segment_height"] = None
    peer_incls = []
    for peer, _ in peer_polys:
        paz, pincl = fe._plane_to_azimuth_incl(peer.get("merged_plane") or [])
        if not math.isnan(pincl):
            peer_incls.append(float(pincl))
    if plane_incl is not None:
        out["seg_rank_in_bld_by_incl"] = 1 + sum(
            1 for x in peer_incls if x > plane_incl
        )
        out["seg_percentile_incl_in_bld"] = float(
            sum(1 for x in peer_incls if x <= plane_incl) / max(len(peer_incls), 1)
        )
        out["seg_is_bld_unique_by_incl"] = bool(
            sum(1 for x in peer_incls if abs(x - plane_incl) <= 5.0) == 0
        )
        peer_azs = []
        for peer, _ in peer_polys:
            paz, _ = fe._plane_to_azimuth_incl(peer.get("merged_plane") or [])
            if not math.isnan(paz):
                peer_azs.append(float(paz))
        out["seg_is_bld_unique_by_az"] = (
            bool(
                sum(
                    1
                    for x in peer_azs
                    if fe._azimuth_diff_deg(x, plane_az or 0.0) <= 15.0
                )
                == 0
            )
            if plane_az is not None
            else None
        )
    else:
        out["seg_rank_in_bld_by_incl"] = None
        out["seg_percentile_incl_in_bld"] = None
        out["seg_is_bld_unique_by_incl"] = None
        out["seg_is_bld_unique_by_az"] = None
    same_cluster = [
        peer
        for peer, _ in peer_polys
        if peer.get("cluster_canonical_id") == record.get("cluster_canonical_id")
    ]
    same_cluster_areas = [
        float(_xz_poly(peer.get("corners") or peer.get("footprint_xz") or []).area)
        for peer in same_cluster
        if _xz_poly(peer.get("corners") or peer.get("footprint_xz") or []) is not None
    ]
    out["seg_rank_in_cluster_by_area"] = (
        1 + sum(1 for a in same_cluster_areas if own_area is not None and a > own_area)
        if own_area is not None
        else None
    )
    same_part_area_peers = [
        float(peer_poly.area)
        for peer, peer_poly in peer_polys
        if peer_poly is not None and _same_part(peer, part, building_context)
    ]
    same_part_top_peers = [
        max(float(c[1]) for c in (peer.get("corners") or []))
        for peer, peer_poly in peer_polys
        if peer_poly is not None
        and _same_part(peer, part, building_context)
        and peer.get("corners")
    ]
    same_part_incl_peers = []
    for peer, peer_poly in peer_polys:
        if peer_poly is None or not _same_part(peer, part, building_context):
            continue
        _, pincl = fe._plane_to_azimuth_incl(peer.get("merged_plane") or [])
        if not math.isnan(pincl):
            same_part_incl_peers.append(float(pincl))
    out["seg_rank_in_part_by_area"] = (
        1
        + sum(1 for a in same_part_area_peers if own_area is not None and a > own_area)
        if own_area is not None
        else None
    )
    out["seg_rank_in_part_by_height"] = (
        1
        + sum(
            1
            for y in same_part_top_peers
            if row.get("plane_top_y_m") is not None and y > row["plane_top_y_m"]
        )
        if row.get("plane_top_y_m") is not None
        else None
    )
    out["seg_rank_in_part_by_incl"] = (
        1
        + sum(
            1 for x in same_part_incl_peers if plane_incl is not None and x > plane_incl
        )
        if plane_incl is not None
        else None
    )
    out["seg_fraction_of_part_roof_area"] = (
        float((own_area or 0.0) / ((own_area or 0.0) + sum(same_part_area_peers)))
        if own_area is not None
        and ((own_area or 0.0) + sum(same_part_area_peers)) > _EPS
        else None
    )
    out["seg_is_part_primary_roof"] = (
        bool(out.get("seg_rank_in_part_by_area") == 1 and (plane_incl or 0.0) > 20.0)
        if out.get("seg_rank_in_part_by_area") is not None
        else None
    )
    out["area_rank_in_part_normalised"] = (
        float(out["seg_rank_in_part_by_area"] / max(len(same_part_area_peers) + 1, 1))
        if out.get("seg_rank_in_part_by_area") is not None
        else None
    )
    out["area_rank_in_building_normalised"] = (
        float(out["seg_rank_in_bld_by_area"] / max(len(peer_areas) + 1, 1))
        if out.get("seg_rank_in_bld_by_area") is not None
        else None
    )
    out["incl_vs_part_median_incl_ratio"] = (
        float(plane_incl / np.median([plane_incl, *same_part_incl_peers]))
        if plane_incl is not None and [plane_incl, *same_part_incl_peers]
        else None
    )
    out["member_count_vs_part_member_count"] = (
        float(
            (row.get("cluster_member_count") or 0)
            / max(
                (row.get("cluster_member_count") or 0)
                + sum(
                    (peer.get("features") or {}).get("member_count", 0)
                    for peer, peer_poly in peer_polys
                    if peer_poly is not None
                    and _same_part(peer, part, building_context)
                ),
                1,
            )
        )
        if row.get("cluster_member_count") is not None
        else None
    )
    out["edge_count_vs_median_in_part"] = (
        float(
            (row.get("edge_count") or 0)
            / max(
                np.median(
                    [row.get("edge_count") or 0]
                    + [
                        len(_edge_records(peer.get("corners") or []))
                        for peer, peer_poly in peer_polys
                        if peer_poly is not None
                        and _same_part(peer, part, building_context)
                    ]
                ),
                1.0,
            )
        )
        if row.get("edge_count") is not None
        else None
    )
    out["edges_vs_part_edges_ratio"] = (
        float(
            (row.get("edge_count") or 0)
            / max(
                (row.get("edge_count") or 0)
                + sum(
                    len(_edge_records(peer.get("corners") or []))
                    for peer, peer_poly in peer_polys
                    if peer_poly is not None
                    and _same_part(peer, part, building_context)
                ),
                1,
            )
        )
        if row.get("edge_count") is not None
        else None
    )
    out["hull_efficiency_vs_part_median"] = (
        float(
            (row.get("poly_convex_hull_ratio") or 0.0)
            / max(
                np.median(
                    [row.get("poly_convex_hull_ratio") or 0.0]
                    + [
                        (peer.get("features") or {}).get(
                            "poly_convex_hull_ratio",
                            row.get("poly_convex_hull_ratio") or 0.0,
                        )
                        for peer, peer_poly in peer_polys
                        if peer_poly is not None
                        and _same_part(peer, part, building_context)
                    ]
                ),
                _EPS,
            )
        )
        if row.get("poly_convex_hull_ratio") is not None
        else None
    )
    if plane_az is not None:
        same_part_peer_az = []
        for peer, peer_poly in peer_polys:
            if peer_poly is None or not _same_part(peer, part, building_context):
                continue
            paz, _ = fe._plane_to_azimuth_incl(peer.get("merged_plane") or [])
            if not math.isnan(paz):
                same_part_peer_az.append(float(paz))
        if same_part_peer_az:
            out["azimuth_spread_relative_to_part"] = float(
                (row.get("member_plane_azimuth_spread_deg") or 0.0)
                / max(
                    max([*same_part_peer_az, plane_az])
                    - min([*same_part_peer_az, plane_az]),
                    _EPS,
                )
            )
        else:
            out["azimuth_spread_relative_to_part"] = None
    else:
        out["azimuth_spread_relative_to_part"] = None

    # Drainage / support / counter-evidence.
    if seg_poly is not None and boundary_poly is not None and plane_az is not None:
        nearest_edge = None
        nearest_dist = None
        coords = _footprint_corners(boundary_poly)
        if centroid is not None and len(coords) >= 2:
            for i in range(len(coords)):
                a = coords[i]
                b = coords[(i + 1) % len(coords)]
                line = LineString([a, b])
                d = float(Point(centroid.x, centroid.y).distance(line))
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = d
                    nearest_edge = (
                        (a[0] + b[0]) / 2 - centroid.x,
                        (a[1] + b[1]) / 2 - centroid.y,
                    )
        if nearest_edge is not None:
            flow = np.asarray(
                [math.sin(math.radians(plane_az)), math.cos(math.radians(plane_az))],
                dtype=float,
            )
            edge_vec = np.asarray(nearest_edge, dtype=float)
            if np.linalg.norm(edge_vec) > _EPS:
                edge_vec = edge_vec / np.linalg.norm(edge_vec)
                out["drainage_to_footprint_edge_cos"] = float(np.dot(flow, edge_vec))
            else:
                out["drainage_to_footprint_edge_cos"] = None
        else:
            out["drainage_to_footprint_edge_cos"] = None
    else:
        out["drainage_to_footprint_edge_cos"] = None
    out["drainage_azimuth_deg"] = row.get("drainage_flow_azimuth_deg")
    if (
        plane_incl is not None
        and row.get("height_above_ground_m") is not None
        and plane_incl > 1.0
    ):
        out["drainage_to_ground_distance_m"] = float(
            row["height_above_ground_m"] / max(math.tan(math.radians(plane_incl)), _EPS)
        )
    else:
        out["drainage_to_ground_distance_m"] = None
    if seg_poly is not None and wall_top_lines:
        support_lengths = 0.0
        support_count = 0
        overhang = 0.0
        for e in edges:
            line = LineString(
                [
                    (float(e["a"][0]), float(e["a"][2])),
                    (float(e["b"][0]), float(e["b"][2])),
                ]
            )
            touching = any(
                _safe_intersects(line.buffer(WALL_SUPPORT_BUFFER_M), wl)
                for wl in wall_top_lines
            )
            if touching:
                support_count += 1
                support_lengths += e["length3"]
            elif eave_count and e["horizontal"]:
                overhang += e["length2"]
        out["seg_supported_by_wall_count"] = support_count
        out["seg_supported_by_wall_length_m"] = support_lengths
        out["seg_supported_by_wall_fraction"] = float(
            support_lengths / max(out["edge_length_m_sum"] or 0.0, _EPS)
        )
        out["seg_has_eave_overhang"] = bool(overhang > EAVE_OVERHANG_MIN_M)
        out["seg_overhang_length_m"] = float(overhang)
        out["seg_is_cantilevered"] = bool(
            support_lengths < 0.5 * max(out["edge_length_m_sum"] or 0.0, _EPS)
        )
    else:
        out["seg_supported_by_wall_count"] = 0
        out["seg_supported_by_wall_length_m"] = None
        out["seg_supported_by_wall_fraction"] = None
        out["seg_has_eave_overhang"] = None
        out["seg_overhang_length_m"] = None
        out["seg_is_cantilevered"] = None
    out["seg_area_moment_about_centroid_m4"] = (
        float(
            (own_area or 0.0)
            * (
                (row.get("poly_min_rect_major_m") or 0.0) ** 2
                + (row.get("poly_min_rect_minor_m") or 0.0) ** 2
            )
            / 12.0
        )
        if own_area is not None
        else None
    )
    out["seg_torque_about_building_center_m3"] = (
        float((own_area or 0.0) * (row.get("distance_to_footprint_center_m") or 0.0))
        if own_area is not None
        else None
    )
    out["seg_aspect_is_unstable"] = bool(
        (row.get("poly_bbox_aspect") or 0.0) > 20.0
        and (row.get("poly_min_width_m") or 1.0) < FLAT_CEILING_SPREAD_M
    )
    out["plane_top_y_below_br18_cap"] = (
        bool((row.get("plane_top_y_m") or 0.0) < 12.0)
        if row.get("plane_top_y_m") is not None
        else None
    )
    out["plane_top_y_above_building_physics_limit"] = (
        bool((row.get("plane_top_y_m") or 0.0) > 20.0)
        if row.get("plane_top_y_m") is not None
        else None
    )
    out["plane_bottom_y_below_slab"] = (
        bool((row.get("plane_bottom_y_m") or 0.0) < (row.get("bld_y_min_m") or 0.0))
        if row.get("plane_bottom_y_m") is not None
        and row.get("bld_y_min_m") is not None
        else None
    )
    out["plane_normal_dot_gravity"] = (
        float(abs(plane[1]) / np.linalg.norm(np.asarray(plane[:3], dtype=float)))
        if len(plane) >= 3 and np.linalg.norm(np.asarray(plane[:3], dtype=float)) > _EPS
        else None
    )

    out["seg_has_ridge"] = bool(ridge_len > 0.0)
    out["seg_has_eave"] = bool(eave_len > 0.0)
    out["seg_has_hip"] = bool(hip_len > 0.0)
    out["seg_has_valley"] = bool(valley_len > 0.0)
    out["seg_has_rake"] = bool(rake_len > 0.0)
    out["seg_has_free_edge"] = bool(free_len > 0.0)
    out["hip_edge_length_m"] = float(hip_len)
    out["valley_edge_length_m"] = float(valley_len)
    out["rake_edge_length_m"] = float(rake_len)
    out["ridge_to_eave_length_ratio"] = out["edge_ridge_to_eave_length_ratio"]
    out["seg_ridge_is_at_part_top"] = (
        bool(
            ridge_len > 0.0
            and row.get("plane_top_y_m") is not None
            and top_vals
            and abs(row["plane_top_y_m"] - max(top_vals))
            <= PLANE_INTERIOR_CLIP_MARGIN_M
        )
        if top_vals
        else None
    )
    out["seg_eave_is_at_wall_top"] = (
        bool(
            eave_len > 0.0
            and y_min is not None
            and wall_top_ys
            and min(abs(y_min - wy) for wy in wall_top_ys)
            <= PLANE_INTERIOR_CLIP_MARGIN_M
        )
        if y_min is not None and wall_top_ys
        else None
    )
    out["seg_ridge_colinear_with_part_major"] = (
        bool(
            ridge_len > 0.0
            and part is not None
            and (part.get("gable_extension") or {}).get("metrics", {}).get("major_az")
            is not None
            and plane_az is not None
            and fe._azimuth_diff_deg(
                plane_az,
                float(
                    (part.get("gable_extension") or {})
                    .get("metrics", {})
                    .get("major_az")
                ),
            )
            < 15.0
        )
        if ridge_len > 0.0
        else None
    )
    out["seg_ridge_colinear_with_bld_major"] = (
        bool(
            ridge_len > 0.0
            and plane_az is not None
            and major_az is not None
            and fe._azimuth_diff_deg(plane_az, major_az) < 15.0
        )
        if ridge_len > 0.0
        else None
    )

    # Part/building relation helpers that the catalogue names explicitly.
    if part_poly is not None:
        part_hull = part_poly.convex_hull
        part_coords = list(part_poly.minimum_rotated_rectangle.exterior.coords)[:-1]
        if len(part_coords) >= 3:
            p0, p1, p2 = (np.asarray(part_coords[i], dtype=float) for i in range(3))
            pe1 = float(np.linalg.norm(p1 - p0))
            pe2 = float(np.linalg.norm(p2 - p1))
            major = max(pe1, pe2)
            minor = min(pe1, pe2)
            out["part_footprint_elongation"] = (
                float(1.0 - (minor / major)) if major > _EPS else None
            )
        else:
            out["part_footprint_elongation"] = None
        out["part_footprint_solidity"] = (
            float(part_poly.area / part_hull.area) if part_hull.area > _EPS else None
        )
    else:
        out["part_footprint_elongation"] = None
        out["part_footprint_solidity"] = None
    out["part_has_kneewall"] = bool(out.get("seg_has_behind_knee_wall_extension"))
    out["bld_has_attic_story"] = bool(
        out.get("bld_knee_wall_count") or out.get("typ_knee_extended_candidate")
    )
    out["bld_footprint_aspect_ratio"] = row.get("bld_footprint_bbox_aspect")
    out["bld_footprint_elongation"] = (
        float(1.0 - 1.0 / row["bld_footprint_elongation_ratio"])
        if row.get("bld_footprint_elongation_ratio") not in (None, 0)
        else None
    )
    out["bld_footprint_solidity"] = row.get("bld_footprint_solidity")
    out["bld_footprint_convexity_deficiency"] = row.get(
        "bld_footprint_convexity_deficiency"
    )
    out["bld_footprint_interior_ring_count"] = row.get(
        "bld_footprint_interior_ring_count"
    )
    out["bld_footprint_is_L_shape"] = row.get("bld_footprint_is_L_shape")
    out["bld_footprint_is_T_shape"] = row.get("bld_footprint_is_T_shape")
    out["bld_footprint_is_U_shape"] = row.get("bld_footprint_is_U_shape")
    out["bld_footprint_is_rectangle"] = row.get("bld_footprint_is_rectangle")
    out["bld_has_basement"] = row.get("bld_has_basement")
    out["bld_room_count"] = row.get("bld_room_count")
    out["bld_wall_count"] = row.get("bld_wall_count")
    out["bld_door_count"] = row.get("bld_door_count")
    out["bld_window_count"] = row.get("bld_window_count")
    out["bld_cross_floor_gap_count"] = row.get("bld_cross_floor_gap_count")
    out["bld_scan_quality_score"] = row.get("bld_scan_quality_score")
    out["bld_dominant_wall_azimuth_deg"] = row.get("bld_dominant_wall_azimuth_deg")
    out["bld_wall_azimuth_entropy"] = row.get("bld_wall_azimuth_entropy")
    out["bld_footprint_orientation_rank_cluster"] = (
        round((row.get("bld_footprint_principal_axis_deg") or 0.0) / 15.0)
        if row.get("bld_footprint_principal_axis_deg") is not None
        else None
    )

    # Room and dormer relations.
    if room_ids:
        out["room_has_simple_slant_count"] = 0
        out["room_is_top_story_count"] = int(
            sum(
                1
                for s in room_story_indices
                if row.get("bld_story_count") is not None
                and s >= row["bld_story_count"] - 1
            )
        )
        out["room_is_top_story_fraction"] = float(
            out["room_is_top_story_count"] / max(len(room_story_indices), 1)
        )
    else:
        out["room_has_simple_slant_count"] = None
        out["room_is_top_story_count"] = None
        out["room_is_top_story_fraction"] = None
    dormers = building_context.get("dormers") or [] if building_context else []
    dormer_polys = [
        _xz_poly(d.get("corners") or d.get("footprint_xz") or []) for d in dormers
    ]
    dormer_pairs = [
        (d, p) for d, p in zip(dormers, dormer_polys, strict=False) if p is not None
    ]
    out["bld_dormer_total_front_wall_length_m"] = (
        float(sum(p.length for _, p in dormer_pairs)) if dormer_pairs else 0.0
    )
    out["bld_dormer_total_surface_area_m2"] = (
        float(sum(p.area for _, p in dormer_pairs)) if dormer_pairs else 0.0
    )
    out["bld_dormer_per_slanted_roof_ratio"] = (
        float(len(dormer_pairs) / max(out.get("bld_slanted_roof_count") or 0, 1))
        if out.get("bld_slanted_roof_count") is not None
        else None
    )
    out["room_has_dormer_count"] = int(
        sum(1 for d, _ in dormer_pairs if d.get("roof_surface_id"))
    )
    out["seg_is_dormer_front"] = bool(
        any(
            d.get("front_wall_id")
            in (m.get("source_wall_id") for m in record.get("cluster_members") or [])
            for d, _ in dormer_pairs
        )
    )
    out["seg_is_dormer_top"] = bool(
        any(
            d.get("roof_surface_id") == record.get("proposal_id")
            for d, _ in dormer_pairs
        )
    )
    out["seg_dormer_proximity_m"] = _nearest_distance_to_polys(
        seg_poly, [p for _, p in dormer_pairs]
    )
    out["seg_contains_dormer_locus_count"] = int(
        sum(
            1
            for _, p in dormer_pairs
            if seg_poly is not None and _safe_intersects(seg_poly, p)
        )
    )
    out["seg_has_dormer_hole"] = bool((out.get("poly_interior_ring_count") or 0) > 0)
    out["seg_requires_dormer_second_pass"] = bool(
        dormer_pairs
        and (
            out["seg_is_dormer_front"]
            or out["seg_is_dormer_top"]
            or (out["seg_contains_dormer_locus_count"] or 0) > 0
            or (
                out.get("seg_dormer_proximity_m") is not None
                and out["seg_dormer_proximity_m"] <= 1.5
            )
        )
    )
    out["seg_is_in_gable_tier1_reasons"] = None
    out["seg_is_in_gable_tier2_reasons"] = None

    # Neighbor-specific gable helpers.
    opp_areas = [
        float(peer_poly.area)
        for peer, peer_poly in peer_polys
        if peer_poly is not None
        and peer.get("cluster_canonical_id")
        in set(record.get("opposing_cluster_canonicals") or [])
    ]
    out["opposing_gable_area_ratio"] = (
        float((own_area or 0.0) / max(opp_areas[0], _EPS))
        if own_area is not None and len(opp_areas) == 1
        else None
    )
    out["opposing_gable_eave_y_asymmetry_m"] = (
        float(
            abs(
                (row.get("plane_bottom_y_m") or 0.0)
                - min(float(c[1]) for c in (peers[0].get("corners") or []))
            )
        )
        if len(peers) == 1 and peers[0].get("corners")
        else None
    )
    if len(peers) == 1 and ridge_az is not None:
        peer_edges = _edge_records(peers[0].get("corners") or [])
        peer_ridge_az = None
        peer_ys = [float(c[1]) for c in (peers[0].get("corners") or [])]
        peer_ymax = max(peer_ys) if peer_ys else None
        for e in peer_edges:
            if (
                e["horizontal"]
                and peer_ymax is not None
                and float(e["mid"][1]) >= peer_ymax - PLANE_INTERIOR_CLIP_MARGIN_M
            ):
                peer_ridge_az = e["azimuth"]
                break
        out["opposing_gable_ridge_colinearity"] = (
            float(1.0 - fe._azimuth_diff_deg(ridge_az, peer_ridge_az) / 180.0)
            if peer_ridge_az is not None
            else None
        )
    else:
        out["opposing_gable_ridge_colinearity"] = None

    # Part drainage aggregates.
    same_part_drain = []
    if plane_az is not None and own_area is not None:
        same_part_drain.append((plane_az, own_area))
    for peer, peer_poly in peer_polys:
        if peer_poly is None or not _same_part(peer, part, building_context):
            continue
        paz, _ = fe._plane_to_azimuth_incl(peer.get("merged_plane") or [])
        if math.isnan(paz):
            continue
        same_part_drain.append((float(paz), float(peer_poly.area)))
    if same_part_drain:
        vec = np.asarray(
            [
                [math.sin(math.radians(azv)) * area, math.cos(math.radians(azv)) * area]
                for azv, area in same_part_drain
            ]
        )
        resultant = vec.sum(axis=0)
        spread = [azv for azv, _ in same_part_drain]
        out["part_drainage_azimuth_resultant"] = float(
            math.degrees(math.atan2(resultant[0], resultant[1])) % 360.0
        )
        total_area = sum(area for _, area in same_part_drain)
        out["part_drainage_balance"] = float(
            np.linalg.norm(resultant) / max(total_area, _EPS)
        )
        out["part_drainage_azimuths_spread_deg"] = (
            float(max(spread) - min(spread)) if len(spread) >= 2 else 0.0
        )
        quadrants = {
            int(((azv % 360.0) + 45.0) // 90.0) % 4 for azv, _ in same_part_drain
        }
        out["part_drainage_has_4way_split"] = bool(len(quadrants) == 4)
    else:
        out["part_drainage_azimuth_resultant"] = None
        out["part_drainage_balance"] = None
        out["part_drainage_azimuths_spread_deg"] = None
        out["part_drainage_has_4way_split"] = None
    if centroid is not None and wall_centroids and plane_az is not None:
        gutter = min(
            wall_centroids,
            key=lambda pt: math.hypot(pt[0] - centroid.x, pt[1] - centroid.y),
        )
        gutter_vec = np.asarray(
            [gutter[0] - centroid.x, gutter[1] - centroid.y], dtype=float
        )
        flow = np.asarray(
            [math.sin(math.radians(plane_az)), math.cos(math.radians(plane_az))],
            dtype=float,
        )
        if np.linalg.norm(gutter_vec) > _EPS:
            gutter_vec = gutter_vec / np.linalg.norm(gutter_vec)
            out["drainage_to_nearest_gutter_cos"] = float(np.dot(flow, gutter_vec))
        else:
            out["drainage_to_nearest_gutter_cos"] = None
    else:
        out["drainage_to_nearest_gutter_cos"] = None
    if centroid is not None and plane_az is not None:
        flow = np.asarray(
            [math.sin(math.radians(plane_az)), math.cos(math.radians(plane_az))],
            dtype=float,
        )
        upstream = 0
        downstream = 0
        for _peer, peer_poly in peer_polys:
            if peer_poly is None:
                continue
            peer_centroid = peer_poly.centroid
            vec = np.asarray(
                [peer_centroid.x - centroid.x, peer_centroid.y - centroid.y],
                dtype=float,
            )
            along = float(np.dot(vec, flow))
            if along > WALL_SUPPORT_BUFFER_M:
                downstream += 1
            elif along < -WALL_SUPPORT_BUFFER_M:
                upstream += 1
        out["seg_watershed_downstream_count"] = downstream
        out["seg_watershed_upstream_count"] = upstream
        out["seg_watershed_is_ridge_origin"] = bool(upstream == 0 and ridge_len > 0.0)
        out["seg_watershed_is_gutter_sink"] = bool(downstream == 0 and eave_len > 0.0)
        prevailing = None
        out["clim_prevailing_wind_az_deg"] = prevailing
        out["clim_snow_load_kpa"] = None
        out["clim_rain_mm_yr"] = None
        out["seg_windward_score"] = (
            float(
                math.cos(math.radians(abs(fe._azimuth_diff_deg(plane_az, prevailing))))
            )
            if prevailing is not None
            else None
        )
    else:
        out["seg_watershed_downstream_count"] = None
        out["seg_watershed_upstream_count"] = None
        out["seg_watershed_is_ridge_origin"] = None
        out["seg_watershed_is_gutter_sink"] = None
        out["clim_prevailing_wind_az_deg"] = None
        out["clim_snow_load_kpa"] = None
        out["clim_rain_mm_yr"] = None
        out["seg_windward_score"] = None

    # Typology + counter-evidence + scan artefact.
    out["typ_gable_candidate"] = out.get("opposing_is_gable_pair")
    out["typ_gable_incl_symmetry_deg"] = out.get("opposing_gable_incl_asymmetry_deg")
    out["typ_gable_leg_azimuth_match"] = row.get("derived_plane_az_vs_bld_major_deg")
    out["typ_hip_candidate"] = bool(
        out.get("opposing_is_hip_trio") or out.get("opposing_is_hip_quartet")
    )
    out["typ_hip_closure_deg"] = out.get("opposing_hip_closure_angle_deg")
    out["typ_shed_candidate"] = bool(
        (record.get("opposing_planes") and len(record.get("opposing_planes")) <= 1)
        or not record.get("opposing_planes")
    ) and bool(plane_incl is not None and 5.0 <= plane_incl <= 30.0)
    out["typ_shed_is_architectural"] = bool(
        (plane_incl or 0.0) > 10.0 and (own_area or 0.0) > 5.0
    )
    out["typ_shed_is_outbuilding_scale"] = bool(
        (own_area or 0.0) < 20.0
        and (out.get("area_vs_footprint_area_ratio") or 0.0) < FLAT_CEILING_SPREAD_M
    )
    out["typ_flat_candidate"] = (
        bool((plane_incl or 90.0) < 5.0) if plane_incl is not None else None
    )
    out["typ_flat_is_ceiling_not_roof"] = out.get("seg_is_mostly_a_flat_ceiling")
    out["typ_pyramid_candidate"] = bool(
        out.get("opposing_is_hip_quartet")
        and (row.get("poly_bbox_aspect") or 99.0) < PYRAMID_ASPECT_MAX
    )
    out["typ_lplan_candidate"] = bool(
        row.get("bld_footprint_is_L_shape") and out.get("typ_gable_candidate")
    )
    out["typ_tplan_candidate"] = bool(
        row.get("bld_footprint_is_T_shape") and out.get("typ_gable_candidate")
    )
    out["typ_uplan_candidate"] = bool(
        row.get("bld_footprint_is_U_shape") and out.get("typ_gable_candidate")
    )
    out["typ_knee_extended_candidate"] = bool(
        out.get("seg_has_behind_knee_wall_extension")
    )
    out["typ_complex_candidate"] = bool(
        (out.get("bld_slanted_roof_count") or 0) >= 5
        and not any(
            out.get(k)
            for k in (
                "typ_gable_candidate",
                "typ_hip_candidate",
                "typ_shed_candidate",
                "typ_flat_candidate",
            )
        )
    )
    out["typ_gable_ridge_horizontality"] = (
        float(ridge_len / max(out.get("edge_ridge_length_m") or 0.0, _EPS))
        if ridge_len > 0.0
        else None
    )
    ridge_az = None
    eave_az = None
    for e in edges:
        if (
            ridge_az is None
            and e["horizontal"]
            and y_max is not None
            and float(e["mid"][1]) >= y_max - PLANE_INTERIOR_CLIP_MARGIN_M
        ):
            ridge_az = e["azimuth"]
        if (
            eave_az is None
            and e["horizontal"]
            and y_min is not None
            and float(e["mid"][1]) <= y_min + PLANE_INTERIOR_CLIP_MARGIN_M
        ):
            eave_az = e["azimuth"]
    out["typ_gable_eave_parallelism_to_ridge_deg"] = (
        fe._azimuth_diff_deg(ridge_az, eave_az)
        if ridge_az is not None and eave_az is not None
        else None
    )
    out["typ_hip_has_apex_point"] = bool(
        out.get("typ_hip_candidate") and out.get("vtx_ridge_band_count", 0) <= 1
    )
    out["typ_hip_aspect_ratio"] = (
        row.get("poly_bbox_aspect") if out.get("typ_hip_candidate") else None
    )
    out["typ_hip_apex_height_above_eave_m"] = (
        float((row.get("plane_top_y_m") or 0.0) - (row.get("plane_bottom_y_m") or 0.0))
        if out.get("typ_hip_candidate")
        and row.get("plane_top_y_m") is not None
        and row.get("plane_bottom_y_m") is not None
        else None
    )
    lower = [
        float(x) for x in peer_incls if plane_incl is not None and x > plane_incl + 5.0
    ]
    upper = [
        float(x) for x in peer_incls if plane_incl is not None and x < plane_incl - 5.0
    ]
    out["typ_mansard_candidate"] = bool(
        lower and upper and out.get("seg_has_behind_knee_wall_extension")
    )
    out["typ_mansard_lower_leg_incl_deg"] = float(max(lower)) if lower else None
    out["typ_mansard_upper_leg_incl_deg"] = float(min(upper)) if upper else None
    out["typ_mansard_knuckle_y_m"] = (
        row.get("plane_mid_y_m") if out.get("typ_mansard_candidate") else None
    )
    out["typ_mansard_is_lower"] = (
        bool(
            plane_incl is not None
            and out.get("typ_mansard_lower_leg_incl_deg") is not None
            and plane_incl >= out["typ_mansard_lower_leg_incl_deg"] - 1.0
        )
        if out.get("typ_mansard_candidate")
        else None
    )
    out["typ_gambrel_candidate"] = bool(
        lower and upper and out.get("typ_gable_candidate")
    )
    out["typ_pyramid_apex_distance_m"] = (
        float(
            math.hypot(
                out.get("poly_long_axis_projection_m") or 0.0,
                out.get("poly_short_axis_projection_m") or 0.0,
            )
        )
        if out.get("typ_pyramid_candidate")
        else None
    )
    out["typ_lplan_leg_index"] = (
        0
        if out.get("typ_lplan_candidate")
        and (out.get("poly_long_axis_projection_m") or 0.0) < 0
        else 1
        if out.get("typ_lplan_candidate")
        else None
    )
    out["typ_lplan_elbow_distance_m"] = (
        float(abs(out.get("poly_short_axis_projection_m") or 0.0))
        if out.get("typ_lplan_candidate")
        else None
    )
    out["typ_half_hip_candidate"] = bool(
        out.get("typ_gable_candidate") and out.get("seg_has_hip")
    )
    out["typ_hipped_gable_candidate"] = bool(
        out.get("typ_hip_candidate") and out.get("seg_has_rake")
    )
    out["typ_tower_candidate"] = bool(
        (row.get("poly_bbox_aspect") or 99.0) <= 1.5
        and (own_area or 0.0) < 25.0
        and (row.get("plane_top_y_m") or 0.0) > (row.get("plane_bottom_y_m") or 0.0)
    )
    out["typ_tower_is_conical"] = bool(
        out.get("typ_tower_candidate") and out.get("poly_vertex_count", 0) >= 6
    )

    out["adv_plane_clips_into_wall_interior"] = bool(
        (out.get("edge_to_nearest_wall_top_m_min") or 99.0) > 0.5
        and (row.get("plane_bottom_y_m") or 0.0)
        < (row.get("bld_y_min_m") or 0.0) + PLANE_INTERIOR_CLIP_MARGIN_M
    )
    out["adv_plane_clips_through_floor"] = bool(
        (out.get("seg_is_above_slab_count") or 0) > 0 and (plane_incl or 0.0) > 5.0
    )
    out["adv_eave_below_floor"] = bool(
        y_min is not None
        and row.get("bld_y_min_m") is not None
        and y_min < row["bld_y_min_m"]
    )
    out["adv_ridge_above_reasonable"] = (
        bool(
            (row.get("plane_top_y_m") or 0.0)
            > (row.get("bld_height_m") or 0.0) + RIDGE_HEIGHT_REASONABLE_MARGIN_M
        )
        if row.get("plane_top_y_m") is not None and row.get("bld_height_m") is not None
        else None
    )
    out["adv_plane_is_vertical"] = (
        bool((plane_incl or 0.0) > 85.0) if plane_incl is not None else None
    )
    out["adv_plane_is_horizontal"] = (
        bool((plane_incl or 90.0) < 2) if plane_incl is not None else None
    )
    out["adv_is_thin_sliver"] = bool(
        (own_area or 1.0) < 1.0
        and (row.get("poly_min_width_m") or 1.0) < THIN_SLIVER_WIDTH_M
    )
    out["adv_high_min_inscribed_circle"] = bool(
        (out.get("poly_min_inscribed_circle_radius_m") or 0.0) > 1.5
    )
    out["adv_is_tiny_area"] = bool((own_area or 1.0) < TINY_AREA_M2)
    out["adv_has_single_member"] = bool((row.get("cluster_member_count") or 0) == 1)
    out["adv_high_reflex_count"] = bool((out.get("poly_reflex_corner_count") or 0) >= 3)
    out["adv_low_grid_snap"] = bool(
        (out.get("vtx_grid_snap_fraction_0p05") or 0.0) < 0.1
    )
    out["adv_duplicates_accepted_peer"] = bool(
        len(coplanar) > 0
        and (
            row.get("heuristic_label") == "accepted"
            or any(
                (peer.get("heuristic_label") == "accepted") for peer, _ in peer_polys
            )
        )
    )
    out["adv_duplicate_cluster_canonical"] = bool(len(same_cluster) > 0)
    out["adv_overlap_with_accepted_peer_fraction"] = (
        float(
            max(
                (seg_poly.intersection(p).area / max(seg_poly.union(p).area, _EPS))
                for p in coplanar
            )
        )
        if seg_poly is not None and coplanar
        else None
    )
    out["adv_is_orphan_oblique"] = (
        bool(
            (plane_incl or 0.0) >= 15.0
            and (
                record.get("opposing_planes") is None
                or len(record.get("opposing_planes") or []) == 0
            )
        )
        if plane_incl is not None
        else None
    )
    out["adv_gable_has_no_opposing"] = bool(
        out.get("typ_gable_candidate") and not record.get("opposing_planes")
    )
    out["adv_hip_has_wrong_peer_count"] = bool(
        out.get("typ_hip_candidate") and (len(record.get("opposing_planes") or []) < 3)
    )
    out["adv_pitch_outside_dk_norm"] = bool(
        plane_incl is not None
        and not (25.0 <= plane_incl <= 50.0)
        and plane_incl >= 5.0
    )
    out["adv_cross_story_without_part_support"] = bool(
        (row.get("member_story_delta_max") or 0.0) > 0.0
        and (out.get("part_story_count") or 1) == 1
    )
    out["adv_above_top_story_ceiling"] = (
        bool(
            (row.get("plane_bottom_y_m") or 0.0) > (row.get("bld_y_max_m") or 0.0) + 0.5
        )
        if row.get("plane_bottom_y_m") is not None
        and row.get("bld_y_max_m") is not None
        else None
    )

    out["artefact_vertex_on_grid_fraction"] = out.get("vtx_grid_snap_fraction_0p1")
    out["artefact_incl_near_discrete_bucket"] = bool(
        plane_incl is not None
        and min(abs(plane_incl - b) for b in _INCL_BUCKETS) <= 1.0
    )
    out["artefact_azimuth_near_cardinal"] = bool(
        plane_az is not None
        and min(fe._azimuth_diff_deg(plane_az, b) for b in (0.0, 90.0, 180.0, 270.0))
        <= 3.0
    )
    out["artefact_small_segment_from_doorway"] = bool(
        (own_area or 99.0) < 2.0 and (out.get("swall_has_door_count") or 0) > 0
    )
    stair_core_parts = []
    interior_support = out.get("swall_is_interior_fraction")
    if interior_support is not None:
        stair_core_parts.append(float(interior_support))
    dist_edge_norm = out.get("distance_to_footprint_edge_normalised")
    if dist_edge_norm is not None:
        stair_core_parts.append(float(min(dist_edge_norm / 0.2, 1.0)))
    interior_depth = out.get("plane_interior_crossing_depth_m")
    if interior_depth is not None:
        stair_core_parts.append(float(min(interior_depth / 1.5, 1.0)))
    down_exit = out.get("plane_downslope_exit_distance_to_footprint_m")
    if down_exit is not None:
        stair_core_parts.append(float(min(down_exit / 2.0, 1.0)))
    exterior_contact = out.get("plane_exterior_edge_contact_fraction")
    if exterior_contact is not None:
        stair_core_parts.append(float(max(0.0, 1.0 - min(exterior_contact, 1.0))))
    eave_shell_dist = out.get("plane_eave_edge_to_exterior_shell_m")
    if eave_shell_dist is not None:
        stair_core_parts.append(float(min(eave_shell_dist / 1.5, 1.0)))
    downslope_outside = out.get("plane_downslope_points_outside")
    if downslope_outside is not None:
        stair_core_parts.append(0.0 if downslope_outside else 1.0)
    story_count = row.get("bld_story_count")
    if story_count is not None:
        stair_core_parts.append(1.0 if story_count >= 3 else 0.0)
    room_count = row.get("bld_room_count")
    if room_count is not None:
        stair_core_parts.append(1.0 if room_count >= 10 else 0.0)
    out["artefact_internal_staircase_score"] = (
        float(np.mean(stair_core_parts)) if stair_core_parts else None
    )
    out["artefact_internal_staircase_candidate"] = bool(
        out.get("artefact_internal_staircase_score") is not None
        and out["artefact_internal_staircase_score"] >= 0.7
        and bool(out.get("swall_supports_mostly_interior"))
        and not bool(out.get("seg_has_behind_knee_wall_extension"))
        and not bool(out.get("seg_requires_dormer_second_pass"))
    )
    out["artefact_duplicated_wall_signature"] = bool(
        len(source_wall_ids) > 0
        and len(source_wall_ids)
        < len(
            [m for m in record.get("cluster_members") or [] if m.get("source_wall_id")]
        )
    )
    out["artefact_from_scan_cache_only"] = bool(
        str(trace.get("stage") or "").lower() == "cache"
    )
    out["scan_quality_overlap_fraction"] = row.get("scan_quality_overlap_fraction")
    out["scan_quality_cross_floor_gap_density"] = row.get(
        "scan_quality_cross_floor_gap_density"
    )

    # Optional ontology / V2 / cross-modal band.
    aux_out: dict[str, Any] = {
        "swall_thickness_mean_m": None,
        "swall_thickness_std_m": None,
        "part_knee_wall_count": None,
        "ont_part_family_guess": None,
        "ont_part_slant_ratio": None,
        "ont_part_max_room_slant_delta_m": None,
        "ont_part_articulation_room_count": None,
        "ont_part_hypothesis_count": None,
        "ont_cell_complex_flat_cell_count": None,
        "ont_cell_complex_oblique_cell_count": None,
        "ont_cell_complex_mixed_cell_count": None,
        "seg_cell_id": None,
        "seg_cell_is_pure_flat": None,
        "seg_cell_is_pure_oblique": None,
        "seg_crosses_cell_boundary_count": None,
        "ont_coverage_confirmed_sloped_atom_count": None,
        "ont_coverage_partial_sloped_atom_count": None,
        "ont_coverage_subpart_count": None,
        "ont_coverage_gable_run_count": None,
        "ont_coverage_l_t_subpart_count": None,
        "seg_atom_sloped_state": None,
        "seg_subpart_semantic_kind": None,
        "seg_subpart_member_count": None,
        "ont_evidence_edge_tier_mean": None,
        "ont_evidence_supports_oblique": None,
        "seg_evidence_tier": None,
        "ont_top_boundary_ceiling_plane_count": None,
        "seg_projects_onto_ceiling_plane": None,
        "seg_ceiling_plane_azimuth_match_deg": None,
        "ont_hypothesis_total_count": None,
        "ont_hypothesis_selected_count": None,
        "seg_hypothesis_match_selected": None,
        "seg_hypothesis_match_score": None,
        "ont_room_is_simple_slant": None,
        "seg_all_rooms_simple_slant": None,
        "ont_room_is_mixed": None,
        "seg_any_room_mixed": None,
        "ont_part_mixed_room_fraction": None,
        "ont_thermal_ceiling_is_flat": None,
        "ont_thermal_ceiling_count": None,
        "ont_flat_intermediate_candidate_count": None,
        "seg_overlaps_flat_intermediate": None,
        "ont_partition_region_count": None,
        "seg_partition_id": None,
        "seg_partition_area_m2": None,
        "v2_node_count_adjacent_to_seg": None,
        "v2_seg_is_in_gable_run_chain": None,
        "v2_seg_chain_length": None,
        "v2_source_wall_thickness_m_mean": None,
        "v2_source_wall_thickness_m_std": None,
        "v2_source_wall_is_exterior_fraction": None,
        "v2_ifc_class_mode": None,
        "v2_ifc_class_entropy": None,
        "xm_v1_oblique_match_exists": None,
        "xm_v1_oblique_match_azimuth_diff_deg": None,
        "xm_v1_oblique_match_incl_diff_deg": None,
        "xm_v1_oblique_match_centroid_distance_m": None,
        "xm_v1_match_count": None,
        "xm_v1_flat_match_exists": None,
        "xm_v1_is_v3_orphan": None,
        "xm_ontology_family_agrees": None,
        "xm_ontology_disagreement_mode": None,
        "xm_hypothesis_solver_selects_seg": None,
        "xm_cell_complex_agrees_with_ridge_class": None,
        "xm_v2_chain_membership_agrees": None,
        "xm_v2_gable_run_endpoint_matches_seg": None,
    }
    if aux_context is not None and seg_poly is not None:
        topo = aux_context.get("topology") or {}
        onto = aux_context.get("ontology") or {}
        touched_room_indices = sorted(
            {
                idx
                for idx in (
                    rr.get("room_index")
                    if rr.get("room_index") is not None
                    else _parse_room_index(rr.get("room_id"))
                    for rr in (record.get("room_boundary_refs") or [])
                )
                if idx is not None
            }
        )
        touched_room_graph_ids = {
            f"room:merged_room_{idx}" for idx in touched_room_indices
        }

        topo_cells = topo.get("cell_complex_cells") or []
        nearby_cells = []
        for cell in topo_cells:
            fp = (cell.get("properties") or {}).get("xz_footprint") or []
            cell_poly = _poly_from_xz_points(fp)
            if cell_poly is not None and seg_poly.intersects(
                cell_poly.buffer(WALL_SUPPORT_BUFFER_M)
            ):
                nearby_cells.append(cell)
        aux_out["v2_node_count_adjacent_to_seg"] = len(nearby_cells)

        adj_edges = [
            edge
            for edge in (topo.get("adjacency_edges") or [])
            if edge.get("from_id") in touched_room_graph_ids
            or edge.get("to_id") in touched_room_graph_ids
        ]
        thickness_vals = [
            float((edge.get("evidence") or {}).get("thickness_cm")) / 100.0
            for edge in adj_edges
            if (edge.get("evidence") or {}).get("thickness_cm") is not None
        ]
        thickness_std_vals = [
            float((edge.get("evidence") or {}).get("thickness_std_cm")) / 100.0
            for edge in adj_edges
            if (edge.get("evidence") or {}).get("thickness_std_cm") is not None
        ]
        if thickness_vals:
            aux_out["swall_thickness_mean_m"] = float(np.mean(thickness_vals))
            aux_out["v2_source_wall_thickness_m_mean"] = aux_out[
                "swall_thickness_mean_m"
            ]
        if thickness_std_vals:
            aux_out["swall_thickness_std_m"] = float(np.mean(thickness_std_vals))
            aux_out["v2_source_wall_thickness_m_std"] = aux_out["swall_thickness_std_m"]
        surface_roles = [
            str((node.get("properties") or {}).get("surface_role") or "")
            for node in (topo.get("surface_nodes") or [])
        ]
        if surface_roles:
            aux_out["v2_ifc_class_mode"] = Counter(
                str(node.get("ifc_class") or "")
                for node in (topo.get("surface_nodes") or [])
                if node.get("ifc_class")
            ).most_common(1)[0][0]
            aux_out["v2_ifc_class_entropy"] = _entropy_from_counts(
                Counter(
                    str(node.get("ifc_class") or "")
                    for node in (topo.get("surface_nodes") or [])
                    if node.get("ifc_class")
                )
            )
            exterior_count = sum(1 for role in surface_roles if role == "exterior_wall")
            aux_out["v2_source_wall_is_exterior_fraction"] = float(
                exterior_count / len(surface_roles)
            )

        building_parts = onto.get("building_parts") or []
        best_part, _best_part_poly, _ = _segment_match_by_overlap(
            seg_poly,
            building_parts,
            poly_getter=lambda part_rec: _poly_from_xz_points(
                part_rec.get("polygon_xz") or []
            ),
        )
        roof_cells = onto.get("full_model_roof_cells") or []
        best_cell, _, _ = _segment_match_by_overlap(
            seg_poly,
            roof_cells,
            poly_getter=lambda cell_rec: _xz_poly(
                next(
                    (
                        face.get("corners") or []
                        for face in (cell_rec.get("faces") or [])
                        if str(face.get("role") or "") in {"roof", "slab"}
                    ),
                    [],
                )
            ),
        )
        if best_cell is not None:
            aux_out["seg_cell_id"] = best_cell.get("id")
            aux_out["seg_cell_is_pure_flat"] = bool(
                best_cell.get("roof_surface_kind") == "flat"
            )
            aux_out["seg_cell_is_pure_oblique"] = bool(
                best_cell.get("roof_surface_kind") == "oblique"
            )
        aux_out["seg_crosses_cell_boundary_count"] = int(
            sum(
                1
                for cell in roof_cells
                if _xz_poly(
                    next(
                        (
                            face.get("corners") or []
                            for face in (cell.get("faces") or [])
                            if str(face.get("role") or "") in {"roof", "slab"}
                        ),
                        [],
                    )
                )
                is not None
                and seg_poly.intersects(
                    _xz_poly(
                        next(
                            (
                                face.get("corners") or []
                                for face in (cell.get("faces") or [])
                                if str(face.get("role") or "") in {"roof", "slab"}
                            ),
                            [],
                        )
                    )
                )
            )
        )

        semantic_atoms = onto.get("semantic_atoms") or []
        best_atom, _, _atom_overlap = _segment_match_by_overlap(
            seg_poly,
            semantic_atoms,
            poly_getter=lambda atom: _xz_poly(atom.get("poly") or []),
        )
        if best_atom is not None:
            aux_out["seg_atom_sloped_state"] = best_atom.get("sloped_coverage_state")
            aux_out["seg_evidence_tier"] = best_atom.get("support_evidence_score")
            aux_out["seg_partition_id"] = best_atom.get("id")
            aux_out["seg_partition_area_m2"] = best_atom.get("area_m2")
        aux_out["ont_partition_region_count"] = len(semantic_atoms)
        aux_out["ont_coverage_confirmed_sloped_atom_count"] = int(
            sum(
                1
                for atom in semantic_atoms
                if str(atom.get("sloped_coverage_state") or "") == "confirmed"
            )
        )
        aux_out["ont_coverage_partial_sloped_atom_count"] = int(
            sum(
                1
                for atom in semantic_atoms
                if str(atom.get("sloped_coverage_state") or "") == "partial"
            )
        )
        aux_out["ont_evidence_edge_tier_mean"] = (
            float(
                np.mean(
                    [
                        float(atom.get("support_evidence_score"))
                        for atom in semantic_atoms
                        if atom.get("support_evidence_score") is not None
                    ]
                )
            )
            if semantic_atoms
            else None
        )
        aux_out["ont_evidence_supports_oblique"] = bool(
            any(
                (atom.get("support_evidence_score") or 0) > 0 for atom in semantic_atoms
            )
        )
        aux_out["ont_top_boundary_ceiling_plane_count"] = int(
            sum(
                1
                for surface in (onto.get("full_model_renderable_surfaces") or [])
                if str(surface.get("category") or "").startswith("room_ceiling")
            )
        )
        aux_out["seg_projects_onto_ceiling_plane"] = best_atom is not None

        coverage_subparts = onto.get("coverage_subparts") or []
        best_subpart, _, _ = _segment_match_by_overlap(
            seg_poly,
            coverage_subparts,
            poly_getter=lambda sub: _poly_from_xz_points(sub.get("polygon_xz") or []),
        )
        if best_subpart is not None:
            aux_out["seg_subpart_semantic_kind"] = best_subpart.get("semantic_kind")
            aux_out["seg_subpart_member_count"] = len(
                best_subpart.get("room_indices") or []
            )
        aux_out["ont_coverage_subpart_count"] = len(coverage_subparts)
        aux_out["ont_coverage_gable_run_count"] = int(
            sum(
                1
                for sub in coverage_subparts
                if str(sub.get("semantic_kind") or "") == "gable_run"
            )
        )
        aux_out["ont_coverage_l_t_subpart_count"] = int(
            sum(
                1
                for sub in coverage_subparts
                if any(
                    tag in str(sub.get("semantic_kind") or "")
                    for tag in ("l_", "t_", "lplan", "tplan")
                )
            )
        )
        aux_out["v2_seg_is_in_gable_run_chain"] = bool(
            best_subpart is not None
            and str(best_subpart.get("semantic_kind") or "") == "gable_run"
        )
        aux_out["v2_seg_chain_length"] = (
            int(
                sum(
                    1
                    for sub in coverage_subparts
                    if sub.get("roof_hypothesis_id")
                    == (best_subpart or {}).get("roof_hypothesis_id")
                )
            )
            if best_subpart is not None
            else None
        )

        if best_part is not None:
            room_indices = [
                idx
                for idx in (best_part.get("room_indices") or [])
                if isinstance(idx, int)
            ]
            room_summaries = onto.get("room_summaries") or {}
            room_records = [
                room_summaries.get(f"room:{idx}", {}) for idx in room_indices
            ]
            aux_out["ont_part_family_guess"] = best_part.get("roof_family_guess")
            hypothesis_ids = list(best_part.get("hypothesis_ids") or [])
            selected_ids = list(best_part.get("oblique_hypothesis_ids") or []) + list(
                best_part.get("flat_hypothesis_ids") or []
            )
            aux_out["ont_part_hypothesis_count"] = len(hypothesis_ids)
            aux_out["ont_hypothesis_total_count"] = len(hypothesis_ids)
            aux_out["ont_hypothesis_selected_count"] = len(selected_ids)
            aux_out["ont_part_articulation_room_count"] = len(
                best_part.get("articulation_room_ids") or []
            )
            aux_out["ont_part_slant_ratio"] = (
                float(
                    len(best_part.get("oblique_hypothesis_ids") or [])
                    / max(len(hypothesis_ids), 1)
                )
                if hypothesis_ids
                else 0.0
            )
            deltas = [
                float(room_rec.get("slant_delta_m"))
                for room_rec in room_records
                if room_rec.get("slant_delta_m") is not None
            ]
            aux_out["ont_part_max_room_slant_delta_m"] = (
                float(max(deltas)) if deltas else None
            )
            part_id = best_part.get("id")
            aux_out["part_knee_wall_count"] = int(
                sum(
                    1
                    for wall in (onto.get("full_model_knee_walls") or [])
                    if wall.get("part_id") == part_id
                )
            )
            part_roof_cells = [
                cell for cell in roof_cells if cell.get("part_id") == part_id
            ]
            aux_out["ont_cell_complex_flat_cell_count"] = int(
                sum(
                    1
                    for cell in part_roof_cells
                    if cell.get("roof_surface_kind") == "flat"
                )
            )
            aux_out["ont_cell_complex_oblique_cell_count"] = int(
                sum(
                    1
                    for cell in part_roof_cells
                    if cell.get("roof_surface_kind") == "oblique"
                )
            )
            aux_out["ont_cell_complex_mixed_cell_count"] = int(
                sum(
                    1
                    for cell in part_roof_cells
                    if len(
                        {
                            str(face.get("role") or "")
                            for face in (cell.get("faces") or [])
                            if face.get("role")
                        }
                    )
                    > 2
                )
            )
            simple_flags = [
                bool(room_rec.get("covered_by_sloped_roof"))
                and not bool(room_rec.get("mixed"))
                for room_rec in room_records
                if room_rec
            ]
            mixed_flags = [
                bool(room_rec.get("mixed")) for room_rec in room_records if room_rec
            ]
            aux_out["ont_room_is_simple_slant"] = (
                bool(simple_flags[0]) if len(simple_flags) == 1 else None
            )
            aux_out["seg_all_rooms_simple_slant"] = bool(simple_flags) and all(
                simple_flags
            )
            aux_out["ont_room_is_mixed"] = (
                bool(mixed_flags[0]) if len(mixed_flags) == 1 else None
            )
            aux_out["seg_any_room_mixed"] = bool(mixed_flags) and any(mixed_flags)
            aux_out["ont_part_mixed_room_fraction"] = (
                float(sum(1 for flag in mixed_flags if flag) / len(mixed_flags))
                if mixed_flags
                else None
            )
            flat_flags = [
                "flat_ceiling" in (room_rec.get("roles") or [])
                for room_rec in room_records
                if room_rec
            ]
            aux_out["ont_thermal_ceiling_is_flat"] = (
                bool(flat_flags[0]) if len(flat_flags) == 1 else None
            )
            aux_out["ont_thermal_ceiling_count"] = int(
                sum(1 for flag in flat_flags if flag)
            )

        oblique_surfaces = onto.get("roof_surfaces_oblique") or []
        best_oblique, _best_oblique_poly, best_oblique_overlap = (
            _segment_match_by_overlap(
                seg_poly,
                oblique_surfaces,
                poly_getter=lambda surf: _xz_poly(surf.get("corners") or []),
            )
        )
        aux_out["xm_v1_oblique_match_exists"] = bool(
            best_oblique is not None and best_oblique_overlap > 0.0
        )
        aux_out["xm_v1_match_count"] = int(
            sum(
                1
                for surf in oblique_surfaces
                if _xz_poly(surf.get("corners") or []) is not None
                and seg_poly.intersects(
                    _xz_poly(surf.get("corners") or []).buffer(SCAN_NOISE_M)
                )
            )
        )
        flat_surfaces = onto.get("roof_surfaces_flat") or []
        best_flat, _, best_flat_overlap = _segment_match_by_overlap(
            seg_poly,
            flat_surfaces,
            poly_getter=lambda surf: _xz_poly(surf.get("corners") or []),
        )
        aux_out["xm_v1_flat_match_exists"] = bool(
            best_flat is not None and best_flat_overlap > 0.0
        )
        aux_out["ont_flat_intermediate_candidate_count"] = int(
            sum(
                1
                for surf in flat_surfaces
                if str(surf.get("kind") or "") == "intermediate"
            )
        )
        aux_out["seg_overlaps_flat_intermediate"] = bool(
            any(
                str(surf.get("kind") or "") == "intermediate"
                and _xz_poly(surf.get("corners") or []) is not None
                and seg_poly.intersects(
                    _xz_poly(surf.get("corners") or []).buffer(SCAN_NOISE_M)
                )
                for surf in flat_surfaces
            )
        )
        aux_out["xm_v1_is_v3_orphan"] = bool(
            not aux_out["xm_v1_oblique_match_exists"]
            and not aux_out["xm_v1_flat_match_exists"]
        )
        if best_oblique is not None:
            surf_az = _safe_float(best_oblique.get("avg_azimuth_deg"))
            surf_incl = _safe_float(best_oblique.get("avg_incl_deg"))
            surf_centroid = (
                _xz_poly(best_oblique.get("corners") or []).centroid
                if _xz_poly(best_oblique.get("corners") or []) is not None
                else None
            )
            aux_out["xm_v1_oblique_match_azimuth_diff_deg"] = (
                fe._azimuth_diff_deg(plane_az, surf_az)
                if plane_az is not None and surf_az is not None
                else None
            )
            aux_out["xm_v1_oblique_match_incl_diff_deg"] = (
                float(abs(plane_incl - surf_incl))
                if plane_incl is not None and surf_incl is not None
                else None
            )
            aux_out["xm_v1_oblique_match_centroid_distance_m"] = (
                float(seg_poly.centroid.distance(surf_centroid))
                if surf_centroid is not None
                else None
            )
            aux_out["seg_ceiling_plane_azimuth_match_deg"] = aux_out[
                "xm_v1_oblique_match_azimuth_diff_deg"
            ]
            selected_ids = set(
                (best_part or {}).get("oblique_hypothesis_ids") or []
            ) | set((best_part or {}).get("flat_hypothesis_ids") or [])
            aux_out["seg_hypothesis_match_selected"] = bool(
                best_oblique.get("roof_hypothesis_id") in selected_ids
            )
            aux_out["seg_hypothesis_match_score"] = (
                float(best_oblique_overlap / max(own_area or 0.0, _EPS))
                if own_area
                else None
            )
        family = str(aux_out.get("ont_part_family_guess") or "")
        typ_guess = None
        if out.get("typ_gable_candidate"):
            typ_guess = "gable_or_multi_slope"
        elif out.get("typ_flat_candidate"):
            typ_guess = "flat_or_capped"
        elif out.get("typ_hip_candidate") or out.get("typ_complex_candidate"):
            typ_guess = "mixed_or_partial"
        aux_out["xm_ontology_family_agrees"] = (
            bool(typ_guess is not None and family == typ_guess) if family else None
        )
        aux_out["xm_ontology_disagreement_mode"] = (
            None
            if aux_out["xm_ontology_family_agrees"] or typ_guess is None or not family
            else f"{typ_guess}_vs_{family}"
        )
        aux_out["xm_hypothesis_solver_selects_seg"] = aux_out.get(
            "seg_hypothesis_match_selected"
        )
        aux_out["xm_cell_complex_agrees_with_ridge_class"] = (
            bool(out.get("seg_has_ridge"))
            == bool(
                best_cell is not None
                and best_cell.get("cell_kind") in {"attic", "upper_void"}
            )
            if best_cell is not None
            else None
        )
        aux_out["xm_v2_chain_membership_agrees"] = (
            aux_out.get("v2_seg_is_in_gable_run_chain")
            == (aux_out.get("seg_subpart_semantic_kind") == "gable_run")
            if aux_out.get("v2_seg_is_in_gable_run_chain") is not None
            else None
        )
        aux_out["xm_v2_gable_run_endpoint_matches_seg"] = (
            bool(aux_out.get("v2_seg_is_in_gable_run_chain"))
            and bool(out.get("seg_has_ridge") or out.get("seg_has_rake"))
            if aux_out.get("v2_seg_is_in_gable_run_chain") is not None
            else None
        )
    out.update(aux_out)

    # Temporal / versioning / interactions.
    repo = Path(repo_root or Path(__file__).resolve().parents[2])
    sha = _git_sha(str(repo))
    out["pipeline_v3_git_sha"] = sha
    out["pipeline_v2_git_sha"] = sha
    out["pipeline_v1_git_sha"] = sha
    out["cluster_algorithm_version"] = (record.get("cluster_params") or {}).get(
        "algorithm_version"
    )
    out["model_train_timestamp"] = None
    out["model_hash"] = None
    out["label_age_days_mean"] = None

    out["interaction_covered_side_x_member_accept"] = float(
        (row.get("covered_side_count") or 0.0)
        * (row.get("member_heuristic_accepted_fraction") or 0.0)
    )
    out["interaction_drainage_center_cos_x_incl"] = float(
        (row.get("drainage_to_building_center_cos") or 0.0) * (plane_incl or 0.0)
    )
    out["interaction_story_delta_x_part_story_count"] = float(
        (row.get("member_story_delta_max") or 0.0)
        * (out.get("part_story_count") or 0.0)
    )
    out["interaction_ridge_len_x_part_n_slanted"] = (
        float(
            (ridge_len or 0.0)
            * (
                (part.get("gable_extension") or {})
                .get("metrics", {})
                .get("n_slanted_roofs")
                or 0.0
            )
        )
        if part is not None
        else None
    )
    out["interaction_swall_distance_x_member_count"] = float(
        (row.get("swall_centroid_to_seg_mean_m") or 0.0)
        * (row.get("cluster_member_count") or 0.0)
    )
    out["interaction_area_x_interiority"] = (
        float(
            (own_area or 0.0)
            * max(
                0.0,
                1.0
                - (
                    (row.get("distance_to_footprint_edge_m") or 0.0)
                    / max(math.sqrt(row.get("bld_footprint_area_m2") or 1.0), _EPS)
                ),
            )
        )
        if own_area is not None
        else None
    )
    out["interaction_opposing_incl_diff_x_count"] = float(
        (out.get("opposing_incl_diff_max_deg") or 0.0)
        * (row.get("opposing_count") or 0.0)
    )
    out["interaction_normals_entropy_x_cluster_member_count"] = float(
        (row.get("normals_d_entropy") or 0.0) * (row.get("cluster_member_count") or 0.0)
    )

    return out
