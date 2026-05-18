"""Geometry audit for tier-6 (gable) buildings in viewer-tiers.

The user reports that the rendered roof oblique polygons for tier-6 buildings
"make no architectural sense": faces overshoot the ridge into thin air, stop
short of where they should sit, or cover the wrong (x,z) region. The plane
parameters look right, but the polygon clipped out of each plane does not.

This script does NOT change anything. It re-classifies every building into
its tier (using `complexity_tiers.classify_building`), filters to tier 6,
and for each one computes per-face geometry diagnostics anchored to the
analytic ridge (intersection of the two best-opposing planes) and the
building footprint stored in `roof_results.ceiling.footprint`.

Outputs:
    .context/tier6_audit/summary.json   — per-building records
    .context/tier6_audit/summary.md     — human-readable bucketed table
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconcile.complexity_tiers import classify_building

REPO = Path(__file__).resolve().parents[1]
BUILDINGS_3D_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
V3_RESULTS_PATH = REPO / "reconcile" / "reconcile_v3_results.json"

OUT_DIR = REPO / ".context" / "tier6_audit"

# Tolerances for bucketing (metres). Tuned for ~30 cm scan noise + half a
# wall thickness; anything larger is a real geometric defect, not measurement
# slop.
RIDGE_TOL_M = 0.3
LOW_INCL_DEG = 10.0
HIGH_INCL_DEG = 70.0
OUTSIDE_FOOTPRINT_TOL_M2 = 1.0
EAVE_ABOVE_WALL_TOP_TOL_M = 0.5
EAVE_BELOW_WALL_TOP_TOL_M = 0.7
RIDGE_DISCONTINUITY_TOL_M = 0.3
FACADE_ASYMMETRY_TOL_M = 1.0
UNCOVERED_FOOTPRINT_TOL_M2 = 1.0
OVER_COVERED_TOL_M2 = 1.0


# ---------------------------------------------------------------------------
# Vector / plane helpers
# ---------------------------------------------------------------------------


def newell_normal(corners: list[list[float]]) -> tuple[float, float, float]:
    """Newell's method — robust to nearly collinear leading triples."""
    nx = ny = nz = 0.0
    n = len(corners)
    for i in range(n):
        x0, y0, z0 = corners[i]
        x1, y1, z1 = corners[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return nx, ny, nz


def fit_plane(corners: list[list[float]]) -> dict | None:
    """Return a dict with unit normal + d for ax+by+cz+d=0, oriented n_y>=0."""
    if len(corners) < 3:
        return None
    nx, ny, nz = newell_normal(corners)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-9:
        return None
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    if ny < 0:
        nx, ny, nz = -nx, -ny, -nz
    cx = sum(c[0] for c in corners) / len(corners)
    cy = sum(c[1] for c in corners) / len(corners)
    cz = sum(c[2] for c in corners) / len(corners)
    d = -(nx * cx + ny * cy + nz * cz)
    return {"n": (nx, ny, nz), "d": d}


def plane_y_at(plane: dict, x: float, z: float) -> float:
    """Solve ax + by + cz + d = 0 for y."""
    nx, ny, nz = plane["n"]
    if abs(ny) < 1e-9:
        return float("nan")
    return -(nx * x + nz * z + plane["d"]) / ny


def polygon_area_3d(corners: list[list[float]]) -> float:
    nx, ny, nz = newell_normal(corners)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def plane_intersection_xz(plane_a: dict, plane_b: dict) -> dict | None:
    """
    Ridge line as ``{dir_xz, point_xz, dir_3d, point_3d}``; ``None`` if degenerate.
    """
    ax, ay, az = plane_a["n"]
    bx, by, bz = plane_b["n"]
    # Cross product of the normals = ridge direction.
    rx = ay * bz - az * by
    ry = az * bx - ax * bz
    rz = ax * by - ay * bx
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rmag < 1e-6:
        return None
    rx, ry, rz = rx / rmag, ry / rmag, rz / rmag

    # A point on the ridge: minimum-norm solution of the 2x3 plane system,
    # mirrors V3 `_plane_intersection_point`.
    try:
        import numpy as np

        A = np.array([[ax, ay, az], [bx, by, bz]], dtype=float)
        b = np.array([-plane_a["d"], -plane_b["d"]], dtype=float)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        px, py, pz = float(sol[0]), float(sol[1]), float(sol[2])
    except Exception:
        return None

    # Project ridge into (x, z); some pathological gables can have a near
    # vertical ridge — flag and bail.
    horiz = math.sqrt(rx * rx + rz * rz)
    if horiz < 1e-6:
        return None
    return {
        "dir_xz": (rx / horiz, rz / horiz),
        "point_xz": (px, pz),
        "dir_3d": (rx, ry, rz),
        "point_3d": (px, py, pz),
        "ridge_y_abs": abs(ry),
    }


def signed_uphill_distance_xz(
    point_xz: tuple[float, float],
    plane: dict,
    ridge_pt_xz: tuple[float, float],
) -> float:
    """Signed (x,z) distance from ridge to point along the plane's UPHILL direction.

    Positive = uphill of ridge (i.e. past it, into the air on the opposing
    plane's territory). Negative = downhill of ridge (the correct side for
    this plane's polygon to sit on).
    """
    nx, _, nz = plane["n"]
    horiz = math.sqrt(nx * nx + nz * nz)
    if horiz < 1e-9:
        return 0.0
    # Uphill in (x,z) for a plane with n_y > 0 is opposite to (n_x, n_z).
    ux, uz = -nx / horiz, -nz / horiz
    return (point_xz[0] - ridge_pt_xz[0]) * ux + (point_xz[1] - ridge_pt_xz[1]) * uz


# ---------------------------------------------------------------------------
# Pair selection — mirrors complexity_tiers.detect_gable
# ---------------------------------------------------------------------------


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def best_opposing_pair(obliques: list[dict]) -> tuple[int, int, float] | None:
    """Return ``(i, j, area_fraction)`` for the highest-area opposing pair.

    Mirrors the selection in `complexity_tiers.detect_gable` (180° ± 30°,
    incl within 10°) so the audit measures the same pair the tier classifier
    chose.
    """
    metas = []
    for s in obliques:
        cl = s.get("cluster") or {}
        az = cl.get("avgAzimuth")
        incl = cl.get("avgIncl")
        area = polygon_area_3d(s.get("corners") or [])
        metas.append((az, incl, area))

    total_area = sum(m[2] for m in metas) or 0.0
    if total_area <= 0:
        return None

    best = None
    for i in range(len(metas)):
        az_i, incl_i, area_i = metas[i]
        if az_i is None or incl_i is None:
            continue
        for j in range(i + 1, len(metas)):
            az_j, incl_j, area_j = metas[j]
            if az_j is None or incl_j is None:
                continue
            if abs(_angle_diff_deg(az_i, az_j) - 180.0) > 30.0:
                continue
            if abs(incl_i - incl_j) > 10.0:
                continue
            pair_area = area_i + area_j
            if best is None or pair_area > best[2]:
                best = (i, j, pair_area)
    if best is None:
        return None
    return (best[0], best[1], best[2] / total_area)


def second_best_opposing_pair_fraction(
    obliques: list[dict], used: tuple[int, int]
) -> float:
    """Area-fraction of the best opposing pair NOT using either index in ``used``.

    Flags hipped / two-ridge buildings that pass detect_gable because the
    primary pair cleared the 70 % threshold while a second perpendicular
    pair also exists.
    """
    remaining = [(i, s) for i, s in enumerate(obliques) if i not in used]
    if len(remaining) < 2:
        return 0.0
    sub = [s for _, s in remaining]
    pair = best_opposing_pair(sub)
    if pair is None:
        return 0.0
    total_area = sum(polygon_area_3d(s.get("corners") or []) for s in obliques)
    return (pair[2] / total_area) if total_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Building reference heights — for eave anchoring checks
# ---------------------------------------------------------------------------


def building_reference_heights(building: dict) -> dict:
    """Return ``{floor_y_min, wall_top_y_max}`` from raw scanned geometry.

    floor_y_min: lowest floor-polygon y across all rooms — the building's
    ground reference. Eaves should not drop below this by more than scan noise.

    wall_top_y_max: highest computed-wall corner y across all rooms — the
    eave height you'd expect for the top story. Eaves should sit at or
    slightly below this; floating well above means the roof hangs in air.
    """
    floor_ys: list[float] = []
    wall_ys: list[float] = []
    for room in building.get("rooms") or []:
        for c in room.get("floor_polygon") or []:
            try:
                floor_ys.append(float(c[1]))
            except (TypeError, ValueError, IndexError):
                pass
        for wall in room.get("walls_computed") or []:
            for c in wall.get("corners") or []:
                try:
                    wall_ys.append(float(c[1]))
                except (TypeError, ValueError, IndexError):
                    pass
    return {
        "floor_y_min": min(floor_ys) if floor_ys else None,
        "wall_top_y_max": max(wall_ys) if wall_ys else None,
    }


# ---------------------------------------------------------------------------
# Top / eave edge extraction and ridge continuity
# ---------------------------------------------------------------------------


def _signed_uphill_per_corner(face: dict, plane: dict, ridge_pt_xz) -> list[float]:
    return [
        signed_uphill_distance_xz((c[0], c[2]), plane, ridge_pt_xz)
        for c in (face.get("corners") or [])
    ]


def top_and_eave_corners(
    face: dict, plane: dict, ridge_pt_xz, *, band_m: float = 0.3
) -> dict:
    """Group corners into ridge-side (top) and eave-side (bottom) bands.

    The "top edge" is the corners within ``band_m`` of the most-uphill corner;
    the "eave edge" is the corners within ``band_m`` of the most-downhill one.
    For a clean trapezoidal gable face this splits 4 corners into 2+2.
    """
    corners = face.get("corners") or []
    if len(corners) < 3:
        return {"top": [], "eave": []}
    signed = _signed_uphill_per_corner(face, plane, ridge_pt_xz)
    s_max, s_min = max(signed), min(signed)
    top = [corners[i] for i in range(len(corners)) if signed[i] >= s_max - band_m]
    eave = [corners[i] for i in range(len(corners)) if signed[i] <= s_min + band_m]
    return {"top": top, "eave": eave}


def _project_onto_axis(point_3d, axis_origin, axis_dir) -> float:
    return (
        (point_3d[0] - axis_origin[0]) * axis_dir[0]
        + (point_3d[1] - axis_origin[1]) * axis_dir[1]
        + (point_3d[2] - axis_origin[2]) * axis_dir[2]
    )


def facade_continuity_metrics(eave_a: list, eave_b: list, ridge: dict) -> dict:
    """Measure whether A's and B's eaves run the same length along the ridge.

    Both faces of a clean gable should run the full length of the long-side
    facade. If face A's eave projects onto a wider parametric interval along
    the ridge axis than face B's, the building has a stretch of long-side
    wall that has slope-A above it but no slope-B — a visible gap on the
    B-side facade. Reported as ``facade_asymmetry_m`` (max one-sided
    overhang) and ``coverage_ratio`` (shorter eave / longer eave).
    """
    if not eave_a or not eave_b:
        return {
            "facade_asymmetry_m": None,
            "coverage_ratio": None,
            "eave_extent_a_m": None,
            "eave_extent_b_m": None,
        }
    rdir = ridge["dir_3d"]
    rorg = ridge["point_3d"]
    ts_a = [_project_onto_axis(p, rorg, rdir) for p in eave_a]
    ts_b = [_project_onto_axis(p, rorg, rdir) for p in eave_b]
    a_lo, a_hi = min(ts_a), max(ts_a)
    b_lo, b_hi = min(ts_b), max(ts_b)
    overhang_lo = abs(a_lo - b_lo)
    overhang_hi = abs(a_hi - b_hi)
    asymmetry = max(overhang_lo, overhang_hi)
    extent_a = a_hi - a_lo
    extent_b = b_hi - b_lo
    longer = max(extent_a, extent_b)
    coverage = (min(extent_a, extent_b) / longer) if longer > 1e-6 else 1.0
    return {
        "facade_asymmetry_m": round(asymmetry, 2),
        "coverage_ratio": round(coverage, 3),
        "eave_extent_a_m": round(extent_a, 2),
        "eave_extent_b_m": round(extent_b, 2),
    }


def ridge_continuity_metrics(top_a: list, top_b: list, ridge: dict) -> dict:
    """Measure overlap and 3D gap between A's and B's ridge-side corners.

    Both top edges should sit on the same line (the analytic ridge). We
    project each face's top corners onto the ridge axis to get a parametric
    interval, then compare:

    - ``overlap_ratio``: shared parametric extent / total parametric extent.
      1.0 means the two ridge edges fully coincide; 0 means they are
      disjoint along the ridge.
    - ``gap_3d_m``: max perpendicular distance from any top corner to the
      analytic ridge line. Captures the case where face A's top edge sits
      higher (or lower) than face B's even though the planes intersect at
      the ridge — a continuity break.
    """
    if not top_a or not top_b:
        return {"overlap_ratio": None, "gap_3d_m": None}

    rdir = ridge["dir_3d"]
    rorg = ridge["point_3d"]

    def perp_dist(p):
        # |((p - rorg) cross rdir)| since rdir is unit length.
        dx, dy, dz = p[0] - rorg[0], p[1] - rorg[1], p[2] - rorg[2]
        cx = dy * rdir[2] - dz * rdir[1]
        cy = dz * rdir[0] - dx * rdir[2]
        cz = dx * rdir[1] - dy * rdir[0]
        return math.sqrt(cx * cx + cy * cy + cz * cz)

    ts_a = [_project_onto_axis(p, rorg, rdir) for p in top_a]
    ts_b = [_project_onto_axis(p, rorg, rdir) for p in top_b]
    a_lo, a_hi = min(ts_a), max(ts_a)
    b_lo, b_hi = min(ts_b), max(ts_b)
    overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    total = max(a_hi, b_hi) - min(a_lo, b_lo)
    overlap_ratio = (overlap / total) if total > 1e-6 else 1.0
    gap_3d = max(perp_dist(p) for p in (top_a + top_b))
    return {
        "overlap_ratio": round(overlap_ratio, 3),
        "gap_3d_m": round(gap_3d, 3),
        "ridge_extent_a_m": round(a_hi - a_lo, 2),
        "ridge_extent_b_m": round(b_hi - b_lo, 2),
    }


# ---------------------------------------------------------------------------
# Footprint helpers
# ---------------------------------------------------------------------------


def shapely_polygon_or_none(coords_xz: list):
    try:
        from shapely.geometry import Polygon

        poly = Polygon([(float(p[0]), float(p[1])) for p in coords_xz])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def face_xz_polygon(face: dict):
    corners = face.get("corners") or []
    if len(corners) < 3:
        return None
    return shapely_polygon_or_none([(c[0], c[2]) for c in corners])


# ---------------------------------------------------------------------------
# Per-face metrics
# ---------------------------------------------------------------------------


def face_metrics(face: dict, plane: dict, ridge: dict) -> dict:
    corners = face.get("corners") or []
    signed = [
        signed_uphill_distance_xz((c[0], c[2]), plane, ridge["point_xz"])
        for c in corners
    ]
    if not signed:
        return {
            "overshoot_m": 0.0,
            "shortfall_m": 0.0,
            "top_edge_dy_m": 0.0,
            "y_min": None,
            "y_max": None,
            "eave_y_min": None,
            "eave_y_max": None,
            "top_y_avg": None,
        }

    max_s = max(signed)
    min_s = min(signed)
    overshoot_m = max(0.0, max_s)
    shortfall_m = max(0.0, -max_s) if max_s < 0 else 0.0

    top_idx = [i for i in range(len(corners)) if signed[i] >= max_s - 0.3]
    eave_idx = [i for i in range(len(corners)) if signed[i] <= min_s + 0.3]
    top_corner_ys = [corners[i][1] for i in top_idx]
    eave_corner_ys = [corners[i][1] for i in eave_idx]
    top_edge_dy_m = (
        max(top_corner_ys) - min(top_corner_ys) if len(top_corner_ys) >= 2 else 0.0
    )
    all_ys = [c[1] for c in corners]
    return {
        "overshoot_m": round(overshoot_m, 3),
        "shortfall_m": round(shortfall_m, 3),
        "top_edge_dy_m": round(top_edge_dy_m, 3),
        "y_min": round(min(all_ys), 3),
        "y_max": round(max(all_ys), 3),
        "eave_y_min": round(min(eave_corner_ys), 3) if eave_corner_ys else None,
        "eave_y_max": round(max(eave_corner_ys), 3) if eave_corner_ys else None,
        "top_y_avg": (
            round(sum(top_corner_ys) / len(top_corner_ys), 3) if top_corner_ys else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-building audit
# ---------------------------------------------------------------------------


def audit_building(
    uuid: str, building: dict, roof: dict, v3_entry: dict | None
) -> dict:
    obliques = (roof.get("roof_surfaces") or {}).get("oblique") or []
    n_oblique = len(obliques)

    # Detection sanity
    n_low_incl = 0
    n_high_incl = 0
    for s in obliques:
        cl = s.get("cluster") or {}
        incl = cl.get("avgIncl")
        if incl is None:
            continue
        if incl < LOW_INCL_DEG:
            n_low_incl += 1
        elif incl > HIGH_INCL_DEG:
            n_high_incl += 1

    pair = best_opposing_pair(obliques)
    if pair is None:
        return {
            "uuid": uuid,
            "address": building.get("address"),
            "n_oblique": n_oblique,
            "n_low_incl": n_low_incl,
            "n_high_incl": n_high_incl,
            "pair_area_fraction": None,
            "second_pair_area_fraction": None,
            "ridge_y_abs": None,
            "faces": [],
            "footprint_area_m2": None,
            "outside_footprint_area_m2": None,
            "v3_status": (v3_entry.get("parts") or [{}])[0]
            .get("gable_extension", {})
            .get("status")
            if v3_entry
            else None,
            "buckets": ["no_pair"],
        }

    a_idx, b_idx, pair_frac = pair
    second_frac = second_best_opposing_pair_fraction(obliques, (a_idx, b_idx))

    face_a = obliques[a_idx]
    face_b = obliques[b_idx]
    plane_a = fit_plane(face_a.get("corners") or [])
    plane_b = fit_plane(face_b.get("corners") or [])

    ridge = None
    if plane_a is not None and plane_b is not None:
        ridge = plane_intersection_xz(plane_a, plane_b)

    faces_out: list[dict] = []
    top_corners_per_face: dict[str, list] = {}
    eave_corners_per_face: dict[str, list] = {}
    if ridge is not None:
        for label, idx, face, plane in (
            ("A", a_idx, face_a, plane_a),
            ("B", b_idx, face_b, plane_b),
        ):
            metrics = face_metrics(face, plane, ridge)
            cl = face.get("cluster") or {}
            faces_out.append(
                {
                    "label": label,
                    "index": idx,
                    "az": round(float(cl.get("avgAzimuth", 0.0)), 1),
                    "incl": round(float(cl.get("avgIncl", 0.0)), 1),
                    "n_corners": len(face.get("corners") or []),
                    "area_m2": round(polygon_area_3d(face.get("corners") or []), 2),
                    **metrics,
                }
            )
            grouped = top_and_eave_corners(face, plane, ridge["point_xz"])
            top_corners_per_face[label] = grouped["top"]
            eave_corners_per_face[label] = grouped["eave"]

    # Footprint containment — use the pipeline's stored 2D footprint.
    footprint = (roof.get("ceiling") or {}).get("footprint") or []
    fp_poly = shapely_polygon_or_none(footprint)
    footprint_area = float(fp_poly.area) if fp_poly is not None else None

    outside_area_total = None
    outside_per_face: list[float] = []
    if fp_poly is not None:
        outside_area_total = 0.0
        for face in (face_a, face_b):
            fpoly = face_xz_polygon(face)
            if fpoly is None:
                outside_per_face.append(0.0)
                continue
            try:
                outside_area = max(0.0, fpoly.difference(fp_poly).area)
            except Exception:
                outside_area = 0.0
            outside_per_face.append(round(outside_area, 2))
            outside_area_total += outside_area
        outside_area_total = round(outside_area_total, 2)
    for i, oa in enumerate(outside_per_face):
        if i < len(faces_out):
            faces_out[i]["outside_footprint_area_m2"] = oa

    # Coverage prioritisation — for every (x,z) inside the footprint, how
    # many oblique faces from the *full* obliques list cover that point?
    #   Uncovered (0 faces) = "gap": no roof at this (x,z).
    #   Over-covered (≥ 2 faces) = "fight": two faces both claim this
    #   (x,z), only one should win.
    # Both directly probe whether the pipeline's per-plane prioritisation
    # is correct.
    uncovered_m2 = None
    over_covered_m2 = None
    if fp_poly is not None:
        try:
            from shapely.ops import unary_union

            face_polys = []
            for s in obliques:
                fp = face_xz_polygon(s)
                if fp is not None:
                    face_polys.append(fp)
            if face_polys:
                union = unary_union(face_polys)
                inside_union = union.intersection(fp_poly)
                uncovered_m2 = round(max(0.0, fp_poly.area - inside_union.area), 2)
                # Sum of individual face areas inside the footprint, minus
                # the union area inside the footprint = double-counted overlap.
                inside_sum = sum(p.intersection(fp_poly).area for p in face_polys)
                over_covered_m2 = round(max(0.0, inside_sum - inside_union.area), 2)
            else:
                uncovered_m2 = round(fp_poly.area, 2)
                over_covered_m2 = 0.0
        except Exception:
            uncovered_m2 = None
            over_covered_m2 = None

    # V3 cross-check
    v3_status = None
    v3_ridge_line = None
    if v3_entry is not None:
        parts = v3_entry.get("parts") or []
        # tier 6 buildings are typically 1-part — read the first part's status.
        if parts:
            ge = parts[0].get("gable_extension") or {}
            v3_status = ge.get("status")
            v3_ridge_line = ge.get("ridge_line")

    # Height anchoring — eave vs floor and wall top.
    refs = building_reference_heights(building)
    floor_y_min = refs["floor_y_min"]
    wall_top_y_max = refs["wall_top_y_max"]

    eave_below_floor_max_m = 0.0
    eave_above_wall_top_max_m = 0.0
    eave_far_below_wall_top_max_m = 0.0
    for f in faces_out:
        ey = f.get("eave_y_min")
        if ey is None:
            continue
        if floor_y_min is not None:
            below = floor_y_min - ey
            if below > eave_below_floor_max_m:
                eave_below_floor_max_m = below
        if wall_top_y_max is not None:
            above = ey - wall_top_y_max
            if above > eave_above_wall_top_max_m:
                eave_above_wall_top_max_m = above
            far_below = wall_top_y_max - ey
            if far_below > eave_far_below_wall_top_max_m:
                eave_far_below_wall_top_max_m = far_below

    # Ridge continuity between A and B (do the top edges meet?).
    continuity = {
        "overlap_ratio": None,
        "gap_3d_m": None,
        "ridge_extent_a_m": None,
        "ridge_extent_b_m": None,
    }
    if (
        ridge is not None
        and "A" in top_corners_per_face
        and "B" in top_corners_per_face
    ):
        continuity = ridge_continuity_metrics(
            top_corners_per_face["A"], top_corners_per_face["B"], ridge
        )

    # Facade continuity between A and B (do the eaves run the same length?).
    facade = {
        "facade_asymmetry_m": None,
        "coverage_ratio": None,
        "eave_extent_a_m": None,
        "eave_extent_b_m": None,
    }
    if (
        ridge is not None
        and eave_corners_per_face.get("A")
        and eave_corners_per_face.get("B")
    ):
        facade = facade_continuity_metrics(
            eave_corners_per_face["A"], eave_corners_per_face["B"], ridge
        )

    # Bucketing
    buckets: list[str] = []
    overshoot_max = max((f["overshoot_m"] for f in faces_out), default=0.0)
    shortfall_max = max((f["shortfall_m"] for f in faces_out), default=0.0)
    top_dy_max = max((f["top_edge_dy_m"] for f in faces_out), default=0.0)
    if overshoot_max > RIDGE_TOL_M:
        buckets.append("overshoot")
    if shortfall_max > RIDGE_TOL_M:
        buckets.append("shortfall")
    if top_dy_max > RIDGE_TOL_M:
        buckets.append("ridge_not_horizontal")
    if outside_area_total is not None and outside_area_total > OUTSIDE_FOOTPRINT_TOL_M2:
        buckets.append("outside_footprint")
    if n_low_incl >= 1 or n_oblique > 2 or second_frac > 0.10:
        buckets.append("extra_obliques")
    # Eaves should sit at-or-near the wall top. Far below = the (x,z)
    # extension of the polygon past the wall has dragged the plane down to
    # an absurd y. Far above = roof floats over the wall.
    if eave_far_below_wall_top_max_m > EAVE_BELOW_WALL_TOP_TOL_M:
        buckets.append("eave_below_wall_top")
    if eave_above_wall_top_max_m > EAVE_ABOVE_WALL_TOP_TOL_M:
        buckets.append("eave_floats")
    # Discontinuity along the ridge: top edges don't share parametric extent
    # (overlap < 50 %) or sit at different perpendicular distances from the
    # analytic ridge (>30 cm).
    if (
        continuity.get("overlap_ratio") is not None
        and continuity["overlap_ratio"] < 0.5
    ):
        buckets.append("ridge_discontinuity")
    if (
        continuity.get("gap_3d_m") is not None
        and continuity["gap_3d_m"] > RIDGE_DISCONTINUITY_TOL_M
    ):
        if "ridge_discontinuity" not in buckets:
            buckets.append("ridge_discontinuity")
    if (
        facade.get("facade_asymmetry_m") is not None
        and facade["facade_asymmetry_m"] > FACADE_ASYMMETRY_TOL_M
    ):
        buckets.append("facade_gap")
    if uncovered_m2 is not None and uncovered_m2 > UNCOVERED_FOOTPRINT_TOL_M2:
        buckets.append("uncovered_footprint")
    if over_covered_m2 is not None and over_covered_m2 > OVER_COVERED_TOL_M2:
        buckets.append("over_covered")
    if not buckets:
        buckets.append("clean")

    return {
        "uuid": uuid,
        "address": building.get("address"),
        "n_oblique": n_oblique,
        "n_low_incl": n_low_incl,
        "n_high_incl": n_high_incl,
        "pair_area_fraction": round(pair_frac, 3),
        "second_pair_area_fraction": round(second_frac, 3),
        "ridge_y_abs": round(ridge["ridge_y_abs"], 3) if ridge else None,
        "faces": faces_out,
        "footprint_area_m2": round(footprint_area, 2) if footprint_area else None,
        "outside_footprint_area_m2": outside_area_total,
        "floor_y_min": round(floor_y_min, 3) if floor_y_min is not None else None,
        "wall_top_y_max": round(wall_top_y_max, 3)
        if wall_top_y_max is not None
        else None,
        "eave_below_floor_max_m": round(eave_below_floor_max_m, 3),
        "eave_above_wall_top_max_m": round(eave_above_wall_top_max_m, 3),
        "eave_far_below_wall_top_max_m": round(eave_far_below_wall_top_max_m, 3),
        "ridge_continuity": continuity,
        "facade_continuity": facade,
        "uncovered_footprint_area_m2": uncovered_m2,
        "over_covered_area_m2": over_covered_m2,
        "v3_status": v3_status,
        "v3_ridge_line": v3_ridge_line,
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _format_address(addr) -> str:
    if isinstance(addr, dict):
        parts = [
            str(addr.get(k, ""))
            for k in ("street", "number", "postcode", "city")
            if addr.get(k)
        ]
        return " ".join(parts).strip() or "(no address)"
    return str(addr) if addr else "(no address)"


def main() -> int:
    with BUILDINGS_3D_PATH.open() as f:
        raw = json.load(f)
    buildings: dict[str, dict] = {}
    if isinstance(raw, list):
        for b in raw:
            uid = b.get("building_uuid") or b.get("uuid")
            if uid:
                buildings[uid] = b
    elif isinstance(raw, dict):
        buildings = raw

    with ROOF_RESULTS_PATH.open() as f:
        roof_results: dict[str, dict] = json.load(f)

    v3_lookup: dict[str, dict] = {}
    if V3_RESULTS_PATH.exists():
        with V3_RESULTS_PATH.open() as f:
            v3_data = json.load(f)
        if isinstance(v3_data, list):
            for entry in v3_data:
                uid = entry.get("building_uuid")
                if uid:
                    v3_lookup[uid] = entry

    # Find tier-6 set.
    tier6_uuids: list[str] = []
    for uid, building in buildings.items():
        roof = roof_results.get(uid)
        result = classify_building(building, roof)
        if result["tier"] == 6:
            tier6_uuids.append(uid)
    tier6_uuids.sort()

    print(f"[tier6_audit] {len(tier6_uuids)} tier-6 buildings")
    records: list[dict] = []
    for uid in tier6_uuids:
        building = buildings[uid]
        roof = roof_results.get(uid) or {}
        v3_entry = v3_lookup.get(uid)
        rec = audit_building(uid, building, roof, v3_entry)
        records.append(rec)
        fac = rec.get("facade_continuity") or {}
        print(
            f"  {uid[:8]} n_obl={rec['n_oblique']:>2} "
            f"facade asym={fac.get('facade_asymmetry_m')} "
            f"cov={fac.get('coverage_ratio')} "
            f"unc_fp={rec.get('uncovered_footprint_area_m2')} "
            f"over={rec.get('over_covered_area_m2')} "
            f"v3={rec['v3_status']!s:>20} "
            f"buckets={rec['buckets']}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = OUT_DIR / "summary.json"
    summary_json.write_text(json.dumps(records, indent=2))
    print(f"[tier6_audit] wrote {summary_json}")

    # Bucket distribution
    from collections import Counter

    bucket_counter: Counter[str] = Counter()
    for r in records:
        for b in r["buckets"]:
            bucket_counter[b] += 1
    print("[tier6_audit] bucket counts (a building can be in multiple):")
    for name, count in bucket_counter.most_common():
        print(f"  {name:>22}: {count}")

    # Markdown report
    lines: list[str] = []
    lines.append("# Tier-6 gable audit\n")
    lines.append(f"Audited {len(records)} tier-6 buildings.\n")
    lines.append(
        "Tolerances: ridge overshoot/shortfall flagged > "
        f"{RIDGE_TOL_M} m, outside-footprint > {OUTSIDE_FOOTPRINT_TOL_M2} m², "
        f"low-inclination < {LOW_INCL_DEG}°.\n"
    )
    lines.append("## Bucket counts\n")
    lines.append("| Bucket | Count |")
    lines.append("|---|---|")
    for name, count in bucket_counter.most_common():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("## Per-building")
    lines.append("")
    lines.append(
        "| UUID | Address | n_obl | A area | B area | "
        "eave<wall_top (m) | ridge ovl | facade asym (m) | facade cov | "
        "uncovered fp (m²) | over-covered (m²) | outside (m²) | v3 | buckets |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: (sorted(x["buckets"]), x["uuid"])):
        addr = _format_address(r.get("address"))
        a_area = b_area = "—"
        if len(r["faces"]) >= 1:
            f = r["faces"][0]
            a_area = f"{f['area_m2']:.1f}"
            f"{f.get('y_min', 0.0):+.2f}..{f.get('y_max', 0.0):+.2f}"
        if len(r["faces"]) >= 2:
            f = r["faces"][1]
            b_area = f"{f['area_m2']:.1f}"
            f"{f.get('y_min', 0.0):+.2f}..{f.get('y_max', 0.0):+.2f}"
        outside = (
            f"{r['outside_footprint_area_m2']:.2f}"
            if r.get("outside_footprint_area_m2") is not None
            else "—"
        )
        cont = r.get("ridge_continuity") or {}
        overlap = (
            f"{cont['overlap_ratio']:.2f}"
            if cont.get("overlap_ratio") is not None
            else "—"
        )
        f"{cont['gap_3d_m']:.2f}" if cont.get("gap_3d_m") is not None else "—"
        fac = r.get("facade_continuity") or {}
        f_asym = (
            f"{fac['facade_asymmetry_m']:.2f}"
            if fac.get("facade_asymmetry_m") is not None
            else "—"
        )
        f_cov = (
            f"{fac['coverage_ratio']:.2f}"
            if fac.get("coverage_ratio") is not None
            else "—"
        )
        (
            f"{fac['eave_extent_a_m']:.2f}"
            if fac.get("eave_extent_a_m") is not None
            else "—"
        )
        (
            f"{fac['eave_extent_b_m']:.2f}"
            if fac.get("eave_extent_b_m") is not None
            else "—"
        )
        unc = r.get("uncovered_footprint_area_m2")
        ovc = r.get("over_covered_area_m2")
        unc_s = f"{unc:.2f}" if unc is not None else "—"
        ovc_s = f"{ovc:.2f}" if ovc is not None else "—"
        lines.append(
            f"| `{r['uuid'][:8]}` | {addr} | {r['n_oblique']} | "
            f"{a_area} | {b_area} | "
            f"{r.get('eave_far_below_wall_top_max_m', 0.0):+.2f} | "
            f"{overlap} | "
            f"{f_asym} | {f_cov} | "
            f"{unc_s} | {ovc_s} | {outside} | "
            f"{r.get('v3_status') or '—'} | {', '.join(r['buckets'])} |"
        )
    summary_md = OUT_DIR / "summary.md"
    summary_md.write_text("\n".join(lines))
    print(f"[tier6_audit] wrote {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
