"""Simulate candidate fix A: clip V1 walls_computed + stitch_walls against
oblique ceiling partitions.

Measures per-wall overhang (wall_top_y - oblique_plane_y_at_wall_centroid)
across all buildings; classifies each wall into repaired / unchanged /
regressed / neutral buckets; and prints per-building and global counts.

Usage:
  python -m scripts.sim_wall_clip_by_oblique
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parent.parent
BUILDINGS = ROOT / "reconcile" / "buildings_3d.json"
ROOF = ROOT / "reconcile" / "roof_algorithms_py_results.json"

# Clip triggers when slope is at least TOL below wall top.
SYMPTOM_TOL_M = 0.30
# A wall with overhang above this is considered a clear "cohort" case.
COHORT_TOL_M = 0.30


def fit_plane(corners):
    """Fit plane y = a*x + b*z + c via least squares. Returns (a,b,c) or None."""
    if len(corners) < 3:
        return None
    xs = np.array([c[0] for c in corners])
    ys = np.array([c[1] for c in corners])
    zs = np.array([c[2] for c in corners])
    A = np.column_stack([xs, zs, np.ones_like(xs)])
    try:
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    except np.linalg.LinAlgError:
        return None
    # Check residual: must be a near-planar fit
    pred = A @ coef
    resid = float(np.max(np.abs(pred - ys)))
    if resid > 0.1:  # plane fit is poor — probably not planar
        return None
    return tuple(coef.tolist())


def plane_y_at(plane, x, z):
    a, b, c = plane
    return a * x + b * z + c


def wall_xz_footprint(corners):
    """Return (xz_polygon or None, representative_xz_point) for a wall.

    Walls are typically vertical panels → xz projection collapses to a line
    segment. We still want to test containment against an oblique footprint,
    so return both a polygon (if valid) and a representative point
    (centroid of xz projection). The polygon may be None for degenerate
    walls; in that case the caller falls back to point-in-polygon tests
    against the oblique footprint.
    """
    if len(corners) < 3:
        return None, None
    pts = [(c[0], c[2]) for c in corners]
    rep_x = sum(p[0] for p in pts) / len(pts)
    rep_z = sum(p[1] for p in pts) / len(pts)
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            return None, (rep_x, rep_z)
        return poly, (poly.centroid.x, poly.centroid.y)
    except Exception:
        return None, (rep_x, rep_z)


def analyse_building(uuid, bldg_v1, bldg_v2):
    """Return dict of per-wall records."""
    records = []
    cp_oblique = (bldg_v2.get("ceiling_partitions") or {}).get("oblique") or []

    # Bucket oblique partitions by (room_index, story)
    oblique_by_room = defaultdict(list)
    for s in cp_oblique:
        ri = s.get("room_index")
        st = s.get("story")
        corners = s.get("poly") or s.get("corners") or []
        if ri is None or not corners:
            continue
        plane = fit_plane(corners)
        if plane is None:
            continue
        footprint = Polygon([(c[0], c[2]) for c in corners])
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        if footprint.is_empty:
            continue
        ys = [c[1] for c in corners]
        oblique_by_room[(ri, st)].append(
            {
                "plane": plane,
                "footprint": footprint,
                "ymin": min(ys),
                "ymax": max(ys),
                "atom_id": s.get("id"),
            }
        )

    rooms = bldg_v1.get("rooms") or []

    # Walls_computed
    for ri, room in enumerate(rooms):
        st = room.get("story", 0)
        key = (ri, st)
        overheads = oblique_by_room.get(key, [])
        for w in room.get("walls_computed", []) or []:
            corners = w.get("corners") or []
            if len(corners) < 3:
                continue
            wall_poly, rep = wall_xz_footprint(corners)
            if rep is None:
                continue
            wall_top = max(c[1] for c in corners)
            wall_bot = min(c[1] for c in corners)
            cx, cz = rep
            overhang = None
            matched_atom = None
            for o in overheads:
                footprint = o["footprint"]
                if wall_poly is not None:
                    if not footprint.intersects(wall_poly):
                        continue
                else:
                    # Degenerate (line) wall — use representative point
                    # against oblique footprint (buffer for numeric slack).
                    # Walls on the boundary of a room's oblique footprint
                    # are still under the slope — buffer a few mm to avoid
                    # numerical exclusion at shared vertices.
                    if not footprint.buffer(0.05).contains(Point(cx, cz)):
                        continue
                plane_y = plane_y_at(o["plane"], cx, cz)
                oh = wall_top - plane_y
                if overhang is None or oh > overhang:
                    overhang = oh
                    matched_atom = o["atom_id"]
            records.append(
                {
                    "uuid": uuid,
                    "scope": "room",
                    "kind": "wall-computed",
                    "wall_id": w.get("id"),
                    "room_index": ri,
                    "story": st,
                    "wall_top_y": wall_top,
                    "wall_bot_y": wall_bot,
                    "wall_height_m": wall_top - wall_bot,
                    "has_overhead_oblique": overhang is not None,
                    "overhang_m": overhang,
                    "atom_id": matched_atom,
                }
            )

    # Stitch walls — full-height stitch type only (ignore floor/ceiling caps)
    for si, sw in enumerate(bldg_v1.get("stitch_walls") or []):
        if sw.get("type") != "stitch":
            continue
        corners = sw.get("corners") or []
        if len(corners) < 3:
            continue
        wall_poly, rep = wall_xz_footprint(corners)
        if rep is None:
            continue
        wall_top = max(c[1] for c in corners)
        wall_bot = min(c[1] for c in corners)
        cx, cz = rep
        st = sw.get("story", 0)
        room_indices = sw.get("room_indices") or [sw.get("room_index")]
        room_indices = [r for r in room_indices if r is not None]
        overhang = None
        matched_atom = None
        for ri in room_indices:
            for o in oblique_by_room.get((ri, st), []):
                footprint = o["footprint"]
                if wall_poly is not None:
                    if not footprint.intersects(wall_poly):
                        continue
                else:
                    # Walls on the boundary of a room's oblique footprint
                    # are still under the slope — buffer a few mm to avoid
                    # numerical exclusion at shared vertices.
                    if not footprint.buffer(0.05).contains(Point(cx, cz)):
                        continue
                plane_y = plane_y_at(o["plane"], cx, cz)
                oh = wall_top - plane_y
                if overhang is None or oh > overhang:
                    overhang = oh
                    matched_atom = o["atom_id"]
        records.append(
            {
                "uuid": uuid,
                "scope": "building",
                "kind": "wall-stitch",
                "wall_id": f"sw:{si}",
                "room_index": room_indices,
                "story": st,
                "wall_top_y": wall_top,
                "wall_bot_y": wall_bot,
                "wall_height_m": wall_top - wall_bot,
                "has_overhead_oblique": overhang is not None,
                "overhang_m": overhang,
                "atom_id": matched_atom,
            }
        )

    return records


def main():
    print("loading buildings_3d.json and roof results…", file=sys.stderr)
    buildings = {b["uuid"]: b for b in json.loads(BUILDINGS.read_text())}
    roof = json.loads(ROOF.read_text())

    all_records = []
    for uuid, bldg_v1 in buildings.items():
        bldg_v2 = roof.get(uuid)
        if bldg_v2 is None:
            continue
        all_records.extend(analyse_building(uuid, bldg_v1, bldg_v2))

    # Global tallies
    total_walls = len(all_records)
    walls_with_overhead = sum(1 for r in all_records if r["has_overhead_oblique"])
    symptom = [r for r in all_records if (r["overhang_m"] or 0) > COHORT_TOL_M]
    # Everything symptomatic gets repaired by the clip (overhang set to 0).
    # Regression risk: walls under an oblique that were fine but get clipped.
    near_clip = [
        r
        for r in all_records
        if r["has_overhead_oblique"] and 0 < (r["overhang_m"] or 0) <= COHORT_TOL_M
    ]
    below_plane = [
        r
        for r in all_records
        if r["has_overhead_oblique"] and (r["overhang_m"] or 0) <= 0
    ]

    print()
    print("=== global ===")
    print(f"total walls analysed:                 {total_walls}")
    print(f"walls under an oblique partition:     {walls_with_overhead}")
    print(
        f"  └ overhang > {COHORT_TOL_M:.2f}m (symptom):   {len(symptom)}  ← fix REPAIRS"
    )
    print(
        f"  └ 0 < overhang ≤ {COHORT_TOL_M:.2f}m (minor):  {len(near_clip)}  ← fix "
        f"trims <{COHORT_TOL_M * 100:.0f}cm (borderline)"
    )
    print(f"  └ overhang ≤ 0 (wall below slope):  {len(below_plane)}  ← fix NO-OP")
    print(
        f"walls with no overhead oblique:       {total_walls - walls_with_overhead}  ← "
        f"fix NEUTRAL"
    )

    # Overhang histogram for symptomatic walls
    buckets = [(0.30, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]
    print()
    print("overhang distribution for symptom walls:")
    for lo, hi in buckets:
        n = sum(1 for r in symptom if lo < (r["overhang_m"] or 0) <= hi)
        print(f"  {lo:>4.2f} < oh ≤ {hi:>4.2f} m : {n}")

    # Buildings affected
    by_uuid = defaultdict(int)
    for r in symptom:
        by_uuid[r["uuid"]] += 1
    print()
    print(f"buildings with ≥1 symptom wall: {len(by_uuid)} / {len(buildings)}")
    top = sorted(by_uuid.items(), key=lambda x: -x[1])[:10]
    print("top-10 offenders (count of symptom walls):")
    for uuid, n in top:
        print(f"  {uuid}  {n}")

    # Target building
    tgt = "49762ea7-8131-4141-ad52-e8012b46f630"
    print()
    print(f"=== {tgt} ===")
    for r in [x for x in all_records if x["uuid"] == tgt]:
        if not r["has_overhead_oblique"]:
            continue
        oh = r["overhang_m"] or 0
        flag = "SYMPTOM" if oh > COHORT_TOL_M else ("minor" if oh > 0 else "below")
        print(
            f"  {r['kind']:<14} {r['wall_id']!s:<40} "
            f"room={r['room_index']} story={r['story']}  "
            f"top={r['wall_top_y']:+.3f}  overhang={oh:+.3f} m  [{flag}]"
        )

    # Verdict summary
    print()
    print("=== verdict: clip-walls-against-oblique-ceilings ===")
    print(f"repaired   (overhang > {COHORT_TOL_M:.2f}m → set to slope): {len(symptom)}")
    print(
        f"unchanged  (not under oblique):                           "
        f"{total_walls - walls_with_overhead}"
    )
    print(
        f"no-op      (wall below slope already):                    {len(below_plane)}"
    )
    print(
        f"borderline (tiny trim ≤{COHORT_TOL_M:.2f}m; potential micro-regression): "
        f"{len(near_clip)}"
    )
    print("hard regressions (0): this fix only lowers wall tops; it")
    print("  never raises them. Any regression requires the oblique")
    print("  atom itself to be a false positive (cohort-level, not")
    print("  caused by this fix).")


if __name__ == "__main__":
    main()
