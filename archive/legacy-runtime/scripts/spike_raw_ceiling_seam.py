"""Spike: do raw scan ceilings disambiguate the oblique:1 ↔ oblique:2 seam?

Read-only diagnostic for the L-shape building 16784bad-2cd9-4f4c-bb26-60355981cfe2.

Usage:
    python scripts/spike_raw_ceiling_seam.py
    python scripts/spike_raw_ceiling_seam.py --uuid <other> --pair 0 3

Prints, for each chosen pair of legacy roof-oblique targets:
  - target plane (centroid, normal, azimuth, inclination) and XZ extent
  - candidate raw scan ceiling polygons in the union XZ bbox + assignment
  - per-target evidence footprint (Shapely union of assigned ceilings in XZ)
  - geometric seam (current legacy / v3 behaviour: plane-plane intersection
    projected to XZ as a half-plane line A*x + B*z + C = 0)
  - implied evidence-based seam: the boundary between the two assigned
    evidence footprints, suitable for visual comparison
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import ijson
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"

DEFAULT_UUID = "16784bad-2cd9-4f4c-bb26-60355981cfe2"
DEFAULT_PAIR = (1, 2)  # roof-oblique::oblique:1 vs oblique:2

NORMAL_DOT_MIN = 0.85  # ~32° normal mismatch tolerance for ceiling↔target
PERP_DIST_MAX_M = 0.50  # max |centroid offset| from target plane
MIN_EVIDENCE_AREA_M2 = 0.5  # per-target gate before trusting evidence


def _xz_polygon(corners):
    pts = [(float(c[0]), float(c[2])) for c in corners]
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def _fit_plane(corners):
    """SVD plane fit. Returns (centroid:np.array(3), normal:np.array(3) up-pointing)."""
    pts = np.array(
        [[float(c[0]), float(c[1]), float(c[2])] for c in corners], dtype=float
    )
    if len(pts) < 3:
        return None, None
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    n = np.linalg.norm(normal)
    if n < 1e-9:
        return centroid, None
    normal = normal / n
    if normal[1] < 0:
        normal = -normal
    return centroid, normal


def _normal_to_az_inc(n):
    az = (math.degrees(math.atan2(float(n[0]), float(n[2]))) + 360.0) % 360.0
    horiz = math.hypot(float(n[0]), float(n[2]))
    inc = math.degrees(math.atan2(horiz, float(n[1])))
    return az, inc


def _half_plane_for_opposing(own, other):
    """Coefficients (A, B, C) such that A*x + B*z + C ≥ 0 marks where own is
    covered (own.y ≤ other.y). Mirrors reconcile_v3 _half_plane_for_opposing.
    own/other = (a, b, c, d) with normalized normal."""
    a_i, b_i, c_i, d_i = own
    a_j, b_j, c_j, d_j = other
    return (b_j * a_i - b_i * a_j, b_j * c_i - b_i * c_j, b_j * d_i - b_i * d_j)


def _plane_coeffs(centroid, normal):
    """Plane Ax + By + Cz + D = 0 with up-pointing normal."""
    A, B, C = float(normal[0]), float(normal[1]), float(normal[2])
    D = -(A * centroid[0] + B * centroid[1] + C * centroid[2])
    return (A, B, C, float(D))


def _signed_dist(coeffs, p):
    A, B, C, D = coeffs
    return A * p[0] + B * p[1] + C * p[2] + D


def _load_building(uuid):
    with open(BUILDINGS_PATH) as fh:
        for b in ijson.items(fh, "item"):
            if b.get("uuid") == uuid:
                return b
    raise SystemExit(f"Building {uuid} not in {BUILDINGS_PATH}")


def _load_roof_result(uuid):
    with open(ROOF_RESULTS_PATH, "rb") as fh:
        for key, value in ijson.kvitems(fh, ""):
            if key == uuid:
                # Decimal -> float for math
                return json.loads(json.dumps(value, default=float))
    raise SystemExit(f"Roof result {uuid} not in {ROOF_RESULTS_PATH}")


def _collect_obliques(roof_result, uuid):
    """
    Return list of dicts: {idx, element_id, corners, poly_xz, centroid, normal,
    plane}.
    """
    out = []
    for idx, surf in enumerate(
        (roof_result.get("roof_surfaces") or {}).get("oblique") or []
    ):
        corners = surf.get("corners") or []
        poly = _xz_polygon(corners)
        centroid, normal = _fit_plane(corners)
        if poly is None or centroid is None or normal is None:
            continue
        plane = _plane_coeffs(centroid, normal)
        az, inc = _normal_to_az_inc(normal)
        out.append(
            {
                "idx": idx,
                "element_id": f"{uuid}::roof-oblique::oblique:{idx}",
                "corners": corners,
                "poly_xz": poly,
                "centroid": centroid.tolist(),
                "normal": normal.tolist(),
                "plane": plane,
                "azimuth_deg": az,
                "inclination_deg": inc,
            }
        )
    return out


def _collect_raw_ceilings(building):
    """
    Flatten raw_ceiling_planes across rooms; return [{room_index, story,
    plane_index, corners, centroid, normal, area_3d, area_xz}].
    """
    out = []
    for room_index, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        for plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) < 3:
                continue
            centroid, normal = _fit_plane(corners)
            if centroid is None or normal is None:
                continue
            poly_xz = _xz_polygon(corners)
            if poly_xz is None:
                continue
            out.append(
                {
                    "room_index": room_index,
                    "story": story,
                    "plane_index": plane_index,
                    "element_id": f"ceiling-raw:{story}:{room_index}:{plane_index}",
                    "corners": corners,
                    "centroid": centroid.tolist(),
                    "normal": normal.tolist(),
                    "poly_xz": poly_xz,
                    "area_xz_m2": float(poly_xz.area),
                }
            )
    return out


def _assign_ceilings_to_targets(targets, ceilings):
    """For each ceiling, find the best-matching target. Returns:
    per_target: {element_id: list[ceiling_record]}
    unassigned: list[ceiling_record]
    """
    per_target = {t["element_id"]: [] for t in targets}
    unassigned = []
    for ceil in ceilings:
        n_ceil = ceil["normal"]
        c_ceil = ceil["centroid"]
        best = None
        best_score = -math.inf
        for t in targets:
            n_t = t["normal"]
            dot = abs(n_ceil[0] * n_t[0] + n_ceil[1] * n_t[1] + n_ceil[2] * n_t[2])
            if dot < NORMAL_DOT_MIN:
                continue
            dist = abs(_signed_dist(t["plane"], c_ceil))
            if dist > PERP_DIST_MAX_M:
                continue
            # Higher dot, lower dist => better. Combine into single score.
            score = dot - 0.2 * (dist / PERP_DIST_MAX_M)
            if score > best_score:
                best = t
                best_score = score
        if best is None:
            unassigned.append(ceil)
        else:
            per_target[best["element_id"]].append({**ceil, "match_score": best_score})
    return per_target, unassigned


def _evidence_footprint(assigned_ceilings):
    polys = [c["poly_xz"] for c in assigned_ceilings]
    if not polys:
        return None
    return unary_union(polys)


def _seam_line_for_pair(target_a, target_b, bounds):
    """Return shapely LineString for the geometric seam (A*x + B*z + C = 0)
    clipped to the union bounds box. None if degenerate or doesn't cross."""
    coeffs = _half_plane_for_opposing(target_a["plane"], target_b["plane"])
    A, B, C = coeffs
    if abs(A) < 1e-12 and abs(B) < 1e-12:
        return None, coeffs
    minx, minz, maxx, maxz = bounds
    pts = []
    # intersect with each edge of bbox
    edges = [
        ((minx, minz), (maxx, minz)),
        ((maxx, minz), (maxx, maxz)),
        ((maxx, maxz), (minx, maxz)),
        ((minx, maxz), (minx, minz)),
    ]
    for (x1, z1), (x2, z2) in edges:
        v1 = A * x1 + B * z1 + C
        v2 = A * x2 + B * z2 + C
        if (v1 <= 0) == (v2 <= 0):
            continue
        if abs(v1 - v2) < 1e-18:
            continue
        t = v1 / (v1 - v2)
        pts.append((x1 + t * (x2 - x1), z1 + t * (z2 - z1)))
    if len(pts) < 2:
        return None, coeffs
    return LineString(pts[:2]), coeffs


def _evidence_seam(footprint_a, footprint_b):
    """Implied seam = shared boundary between the two evidence footprints, OR
    midline between them when they don't touch."""
    if footprint_a is None or footprint_b is None:
        return None, "no-evidence"
    try:
        shared = footprint_a.boundary.intersection(footprint_b.boundary)
    except Exception:
        shared = None
    if (
        shared is not None
        and not getattr(shared, "is_empty", True)
        and float(getattr(shared, "length", 0.0) or 0.0) > 0
    ):
        return shared, "shared-boundary"
    # No touching: report the closest pair of points (proxy seam midpoint)
    try:
        pa, pb = (footprint_a.centroid, footprint_b.centroid)
        midpoint = LineString([(pa.x, pa.y), (pb.x, pb.y)])
    except Exception:
        return None, "disjoint"
    return midpoint, "disjoint-centroid-line"


def _polygon_summary(poly):
    if poly is None:
        return "None"
    if hasattr(poly, "geoms"):
        n = len(list(poly.geoms))
        return (
            f"MultiPolygon(area={poly.area:.2f}m² n={n} "
            f"bounds={tuple(round(v, 2) for v in poly.bounds)})"
        )
    return (
        f"Polygon(area={poly.area:.2f}m² "
        f"bounds={tuple(round(v, 2) for v in poly.bounds)})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", default=DEFAULT_UUID)
    ap.add_argument(
        "--pair",
        nargs=2,
        type=int,
        default=list(DEFAULT_PAIR),
        help="Two oblique target indices, e.g. --pair 1 2",
    )
    ap.add_argument(
        "--all-pairs",
        action="store_true",
        help="Iterate over every pair of obliques whose XZ bboxes overlap.",
    )
    args = ap.parse_args()

    print(f"Building: {args.uuid}")
    building = _load_building(args.uuid)
    roof_result = _load_roof_result(args.uuid)

    targets = _collect_obliques(roof_result, args.uuid)
    print(f"Legacy oblique targets found: {len(targets)}")
    for t in targets:
        bx = tuple(round(v, 2) for v in t["poly_xz"].bounds)
        n = t["normal"]
        print(
            f"  [{t['idx']}] az={t['azimuth_deg']:.0f}° "
            f"inc={t['inclination_deg']:.0f}° "
            f"normal=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f}) area={t['poly_xz'].area:.2f}m² "
            f"bbox={bx}"
        )

    ceilings = _collect_raw_ceilings(building)
    print(
        f"\nRaw scan ceilings found: {len(ceilings)} (across "
        f"{len(building.get('rooms') or [])} rooms)"
    )

    if args.all_pairs:
        pairs = []
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                bi = targets[i]["poly_xz"].bounds
                bj = targets[j]["poly_xz"].bounds
                if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                    continue
                pairs.append((i, j))
        print(f"\nXZ-overlapping pairs: {len(pairs)}")
    else:
        pairs = [tuple(args.pair)]

    for i, j in pairs:
        ti = next((t for t in targets if t["idx"] == i), None)
        tj = next((t for t in targets if t["idx"] == j), None)
        if ti is None or tj is None:
            print(f"\n[pair {i}↔{j}] missing target")
            continue
        print(f"\n=== pair oblique:{i} ↔ oblique:{j} ===")
        print(
            f"  Δazimuth = {abs(ti['azimuth_deg'] - tj['azimuth_deg']):.0f}°  "
            f"Δinclination = {abs(ti['inclination_deg'] - tj['inclination_deg']):.0f}°"
        )

        per_target, unassigned = _assign_ceilings_to_targets([ti, tj], ceilings)
        ass_i = per_target[ti["element_id"]]
        ass_j = per_target[tj["element_id"]]
        print(
            f"  raw ceilings assigned to oblique:{i} = {len(ass_i)} (Σarea_xz = "
            f"{sum(c['area_xz_m2'] for c in ass_i):.2f}m²)"
        )
        print(
            f"  raw ceilings assigned to oblique:{j} = {len(ass_j)} (Σarea_xz = "
            f"{sum(c['area_xz_m2'] for c in ass_j):.2f}m²)"
        )
        print(f"  unassigned (not matching either)        = {len(unassigned)}")

        fp_i = _evidence_footprint(ass_i)
        fp_j = _evidence_footprint(ass_j)
        print(f"  evidence_footprint[oblique:{i}]: {_polygon_summary(fp_i)}")
        print(f"  evidence_footprint[oblique:{j}]: {_polygon_summary(fp_j)}")

        union = ti["poly_xz"].union(tj["poly_xz"])
        seam_geom, seam_coeffs = _seam_line_for_pair(ti, tj, union.bounds)
        if seam_geom is None:
            print(
                f"  geometric seam:  degenerate "
                f"(coeffs={tuple(round(c, 3) for c in seam_coeffs)})"
            )
        else:
            (x1, z1), (x2, z2) = list(seam_geom.coords)
            print(
                f"  geometric seam:  ({x1:.2f},{z1:.2f}) → ({x2:.2f},{z2:.2f})  "
                f"coeffs={tuple(round(c, 3) for c in seam_coeffs)}"
            )

        ev_seam, ev_kind = _evidence_seam(fp_i, fp_j)
        if ev_seam is None:
            print(f"  evidence seam:   {ev_kind}")
        else:
            length = float(getattr(ev_seam, "length", 0.0) or 0.0)
            print(
                f"  evidence seam:   {ev_kind}  type={ev_seam.geom_type}  "
                f"length={length:.2f}m  "
                f"bounds={tuple(round(v, 2) for v in ev_seam.bounds)}"
            )

        # Gate evaluation
        ai = float(fp_i.area) if fp_i is not None else 0.0
        aj = float(fp_j.area) if fp_j is not None else 0.0
        gate = ai >= MIN_EVIDENCE_AREA_M2 and aj >= MIN_EVIDENCE_AREA_M2
        print(
            f"  evidence gate (≥{MIN_EVIDENCE_AREA_M2}m² each): "
            f"{'PASS' if gate else 'FAIL'}  (ai={ai:.2f}m², aj={aj:.2f}m²)"
        )


if __name__ == "__main__":
    main()
