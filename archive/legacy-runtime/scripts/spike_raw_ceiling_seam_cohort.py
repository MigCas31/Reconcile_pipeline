"""Cohort sweep for the raw-ceiling intersection-seam idea.

For every building in buildings_3d.json that has ≥2 legacy roof-oblique
surfaces with overlapping XZ bounding boxes, classify each overlapping pair
by:

  - Δazimuth bucket (mirror ≈180° vs perpendicular ≈90° vs other)
  - per-side raw-ceiling evidence area (m²)
  - shared-boundary length between the two evidence footprints (m)
  - centroid distance between evidence footprints (m)
  - 2D distance between the geometric seam line and the evidence seam line

Use the aggregated table to pick the trigger gate for the new piece_role.

Output: a single CSV-style report streamed to stdout (one row per pair)
plus a per-building summary at the end.

Read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import ijson
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"

NORMAL_DOT_MIN = 0.85
PERP_DIST_MAX_M = 0.50


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
    a_i, b_i, c_i, d_i = own
    a_j, b_j, c_j, d_j = other
    return (b_j * a_i - b_i * a_j, b_j * c_i - b_i * c_j, b_j * d_i - b_i * d_j)


def _plane_coeffs(centroid, normal):
    A, B, C = float(normal[0]), float(normal[1]), float(normal[2])
    D = -(A * centroid[0] + B * centroid[1] + C * centroid[2])
    return (A, B, C, float(D))


def _signed_dist(coeffs, p):
    A, B, C, D = coeffs
    return A * p[0] + B * p[1] + C * p[2] + D


def _angle_diff(a, b):
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def _collect_obliques(roof_result, uuid):
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
    out = []
    for _room_index, room in enumerate(building.get("rooms") or []):
        int(room.get("story", 0))
        for _plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
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
                    "centroid": centroid.tolist(),
                    "normal": normal.tolist(),
                    "poly_xz": poly_xz,
                    "area_xz_m2": float(poly_xz.area),
                }
            )
    return out


def _assign(targets, ceilings):
    per_target = {t["element_id"]: [] for t in targets}
    for ceil in ceilings:
        n = ceil["normal"]
        c = ceil["centroid"]
        best = None
        best_score = -math.inf
        for t in targets:
            n_t = t["normal"]
            dot = abs(n[0] * n_t[0] + n[1] * n_t[1] + n[2] * n_t[2])
            if dot < NORMAL_DOT_MIN:
                continue
            dist = abs(_signed_dist(t["plane"], c))
            if dist > PERP_DIST_MAX_M:
                continue
            score = dot - 0.2 * (dist / PERP_DIST_MAX_M)
            if score > best_score:
                best = t
                best_score = score
        if best is not None:
            per_target[best["element_id"]].append(ceil)
    return per_target


def _evidence(assigned):
    polys = [c["poly_xz"] for c in assigned]
    if not polys:
        return None
    return unary_union(polys)


def _seam_geom(target_a, target_b, bounds):
    coeffs = _half_plane_for_opposing(target_a["plane"], target_b["plane"])
    A, B, C = coeffs
    if abs(A) < 1e-12 and abs(B) < 1e-12:
        return None
    minx, minz, maxx, maxz = bounds
    pts = []
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
        return None
    return LineString(pts[:2])


def _safe_length(g):
    if g is None or getattr(g, "is_empty", True):
        return 0.0
    return float(getattr(g, "length", 0.0) or 0.0)


def _shared_boundary_length(a, b):
    if a is None or b is None:
        return 0.0
    try:
        return _safe_length(a.boundary.intersection(b.boundary))
    except Exception:
        return 0.0


def _centroid_distance(a, b):
    if a is None or b is None:
        return float("nan")
    try:
        ca = a.centroid
        cb = b.centroid
    except Exception:
        return float("nan")
    return math.hypot(ca.x - cb.x, ca.y - cb.y)


def _seam_distance(geom_seam, evidence_geom_a, evidence_geom_b):
    """Approx 'how far is the geometric seam from the evidence picture': max
    distance from each evidence centroid to the geometric seam line. Larger
    means evidence disagrees more strongly with the geometric seam."""
    if geom_seam is None or evidence_geom_a is None or evidence_geom_b is None:
        return float("nan")
    try:
        return max(
            float(geom_seam.distance(evidence_geom_a.centroid)),
            float(geom_seam.distance(evidence_geom_b.centroid)),
        )
    except Exception:
        return float("nan")


def _bucket_azimuth_delta(d):
    if d <= 30:
        return "≤30 (same)"
    if 60 <= d <= 120:
        return "60-120 (perp)"
    if d >= 150:
        return "≥150 (mirror)"
    return "other"


def _process_building(uuid, building, roof_result, writer):
    targets = _collect_obliques(roof_result, uuid)
    if len(targets) < 2:
        return 0
    ceilings = _collect_raw_ceilings(building)
    pairs_emitted = 0
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            a = targets[i]
            b = targets[j]
            ba = a["poly_xz"].bounds
            bb = b["poly_xz"].bounds
            if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
                continue
            azd = _angle_diff(a["azimuth_deg"], b["azimuth_deg"])
            inc_d = abs(a["inclination_deg"] - b["inclination_deg"])
            assigned = _assign([a, b], ceilings)
            ass_a = assigned[a["element_id"]]
            ass_b = assigned[b["element_id"]]
            ev_a = _evidence(ass_a)
            ev_b = _evidence(ass_b)
            ev_area_a = float(ev_a.area) if ev_a is not None else 0.0
            ev_area_b = float(ev_b.area) if ev_b is not None else 0.0
            shared = _shared_boundary_length(ev_a, ev_b)
            cdist = _centroid_distance(ev_a, ev_b)
            union = a["poly_xz"].union(b["poly_xz"])
            geom_seam = _seam_geom(a, b, union.bounds)
            seam_disp = _seam_distance(geom_seam, ev_a, ev_b)
            writer(
                {
                    "uuid": uuid,
                    "i": a["idx"],
                    "j": b["idx"],
                    "az_delta_deg": round(azd, 1),
                    "az_bucket": _bucket_azimuth_delta(azd),
                    "inc_delta_deg": round(inc_d, 1),
                    "ev_area_a": round(ev_area_a, 2),
                    "ev_area_b": round(ev_area_b, 2),
                    "ev_count_a": len(ass_a),
                    "ev_count_b": len(ass_b),
                    "shared_m": round(shared, 2),
                    "centroid_dist_m": round(cdist, 2) if cdist == cdist else None,
                    "seam_disp_m": round(seam_disp, 2)
                    if seam_disp == seam_disp
                    else None,
                    "overlap_m2": round(
                        float(a["poly_xz"].intersection(b["poly_xz"]).area), 2
                    ),
                }
            )
            pairs_emitted += 1
    return pairs_emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Max buildings to process")
    ap.add_argument(
        "--out", type=Path, default=REPO / ".context" / "intersection_seam_cohort.jsonl"
    )
    args = ap.parse_args()

    print(f"Reading {ROOF_RESULTS_PATH.name} (large; streaming)...", file=sys.stderr)

    # Step 1: stream roof_results to extract (uuid -> oblique surfaces). Keep only
    # the oblique payload to limit memory.
    roof_obliques_by_uuid: dict[str, list[dict]] = {}
    with open(ROOF_RESULTS_PATH, "rb") as fh:
        for uuid, value in ijson.kvitems(fh, ""):
            obs = (value.get("roof_surfaces") or {}).get("oblique") or []
            if len(obs) < 2:
                continue
            # Decimal -> float
            kept = []
            for surf in obs:
                corners = [[float(x) for x in c] for c in (surf.get("corners") or [])]
                if len(corners) >= 3:
                    kept.append({"corners": corners})
            if len(kept) >= 2:
                roof_obliques_by_uuid[uuid] = kept

    print(
        f"Buildings with ≥2 oblique roofs: {len(roof_obliques_by_uuid)}",
        file=sys.stderr,
    )

    # Step 2: stream buildings_3d.json, process those that have a roof entry.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILDINGS_PATH) as fh, open(args.out, "w") as out_fh:
        processed = 0
        total_pairs = 0

        def writer(row):
            out_fh.write(json.dumps(row) + "\n")

        for building in ijson.items(fh, "item"):
            uuid = building.get("uuid")
            if uuid not in roof_obliques_by_uuid:
                continue
            roof_result = {"roof_surfaces": {"oblique": roof_obliques_by_uuid[uuid]}}
            n = _process_building(uuid, building, roof_result, writer)
            processed += 1
            total_pairs += n
            if args.limit and processed >= args.limit:
                break
            if processed % 25 == 0:
                print(
                    f"  ... {processed} buildings, {total_pairs} pairs", file=sys.stderr
                )
    print(
        f"Done. {processed} buildings, {total_pairs} pairs → {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
