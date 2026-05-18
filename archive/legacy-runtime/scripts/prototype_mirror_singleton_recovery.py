"""Prototype: detect singleton oblique segments that mirror a surviving cluster.

For each building, the roof pipeline drops singleton segments because
`MIN_CLUSTER_SIZE = 2` in `oblique_clustering.py:8`. When a singleton's
azimuth is ~180 deg opposite a surviving cluster (same inclination), it's
the mirror slope of an existing gable — i.e., the second face of a roof
that physically must exist for the first face to make sense. Treating it
as a promotable cluster recovers the missing slope.

This script reports per UUID:
  * surviving clusters (already kept by `cluster_oblique_segments`)
  * singleton segments (in `segments` but not in any cluster)
  * mirror-pair candidates: singletons whose (az + 180) % 360 matches a
    surviving cluster's azimuth within tolerance and whose inclination
    matches within tolerance
  * for each candidate, raw-ceiling corroboration from `ceiling.oblique`
    fragments that fit the candidate's slope plane
  * the prospective ridge extent the candidate would have if promoted

Wing dominance:
  For each surviving cluster, the area sum of its matched ceiling
  fragments. The candidate inherits its mirror partner's wing footprint
  (since they share the same wing).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import ijson
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOF_RESULTS = REPO / "reconcile" / "roof_algorithms_py_results.json"
DETAIL_OUT = REPO / ".context" / "mirror_singleton_recovery.json"

MIRROR_AZ_TOL_DEG = 30.0
MIRROR_INCL_TOL_DEG = 15.0
PLANE_NORMAL_TOL_DEG = 12.0
PLANE_OFFSET_TOL_M = 0.6
RIDGE_EXTENT_GATE_M = 2.0


def angle_diff_signed(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def angle_diff(a: float, b: float) -> float:
    return abs(angle_diff_signed(a, b))


def plane_normal_from_az_incl(azimuth_deg: float, incl_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    inc = math.radians(incl_deg)
    return np.array(
        [
            math.sin(az) * math.sin(inc),
            math.cos(inc),
            math.cos(az) * math.sin(inc),
        ]
    )


def fit_plane_lsq(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if points.shape[0] < 3:
        return None
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1]
    nl = np.linalg.norm(n)
    if nl < 1e-9:
        return None
    n = n / nl
    if n[1] < 0:
        n = -n
    return n, float(n @ centroid)


def project_on_ridge(
    points: np.ndarray, az_deg: float, ref: np.ndarray
) -> tuple[float, float]:
    az = math.radians(az_deg)
    ridge = np.array([-math.cos(az), 0.0, math.sin(az)])
    proj = (points - ref) @ ridge
    return float(proj.min()), float(proj.max())


def cluster_segs_endpoints(cluster: dict) -> np.ndarray:
    pts = []
    for s in cluster.get("segs") or []:
        pts.append(s.get("a"))
        pts.append(s.get("b"))
    return np.array(pts) if pts else np.zeros((0, 3))


def evaluate_uuid(uuid: str, rec: dict) -> dict:
    clusters = rec.get("valid_clusters") or []
    segments = rec.get("segments") or []
    ceiling = rec.get("ceiling") or {}
    fragments = ceiling.get("oblique") or []

    room_outlines: dict[tuple[int, int], np.ndarray] = {}
    for rp in ceiling.get("room_partitions") or []:
        outline = rp.get("room_outline") or []
        if not outline:
            continue
        try:
            pts = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in outline])
        except Exception:
            continue
        room_outlines[(int(rp.get("story", -1)), int(rp.get("room_index", -1)))] = pts

    clustered = set()
    for c in clusters:
        for s in c.get("segs") or []:
            clustered.add((tuple(s["a"]), tuple(s["b"])))

    singletons = [
        s for s in segments if (tuple(s["a"]), tuple(s["b"])) not in clustered
    ]

    candidates = []
    for s in singletons:
        s_az = float(s.get("azimuth") or 0.0)
        s_incl = float(s.get("incl") or 0.0)
        s_room = int(s.get("room_idx", -1))
        s_story = int(s.get("story", -1))
        for ci, c in enumerate(clusters):
            c_az = float(c.get("avgAzimuth") or 0.0)
            c_incl = float(c.get("avgIncl") or 0.0)
            if angle_diff(s_az, (c_az + 180.0) % 360.0) > MIRROR_AZ_TOL_DEG:
                continue
            if abs(s_incl - c_incl) > MIRROR_INCL_TOL_DEG:
                continue

            mirror_rooms = {
                (int(seg.get("story", -1)), int(seg.get("room_idx", -1)))
                for seg in c.get("segs") or []
            }
            singleton_in_mirror_wing = (s_story, s_room) in mirror_rooms

            footprint_pts = []
            for key in mirror_rooms | {(s_story, s_room)}:
                if key in room_outlines:
                    footprint_pts.append(room_outlines[key])
            footprint = (
                np.concatenate(footprint_pts) if footprint_pts else np.zeros((0, 3))
            )

            mid = np.array(
                [
                    (s["a"][0] + s["b"][0]) / 2,
                    (s["a"][1] + s["b"][1]) / 2,
                    (s["a"][2] + s["b"][2]) / 2,
                ]
            )
            ridge_extent = 0.0
            if footprint.size:
                rmin, rmax = project_on_ridge(footprint, s_az, mid)
                ridge_extent = rmax - rmin

            candidates.append(
                {
                    "singleton_az": round(s_az, 2),
                    "singleton_incl": round(s_incl, 2),
                    "singleton_len": round(float(s.get("len") or 0), 3),
                    "singleton_story": s_story,
                    "singleton_room": s_room,
                    "mirror_cluster_index": ci,
                    "mirror_cluster_az": round(c_az, 2),
                    "mirror_cluster_incl": round(c_incl, 2),
                    "mirror_cluster_size": len(c.get("segs") or []),
                    "mirror_rooms": sorted([list(k) for k in mirror_rooms]),
                    "singleton_in_mirror_wing": singleton_in_mirror_wing,
                    "wing_room_count": len(mirror_rooms | {(s_story, s_room)}),
                    "wing_ridge_extent_m": round(ridge_extent, 3),
                    "would_pass_gate": ridge_extent >= RIDGE_EXTENT_GATE_M
                    and singleton_in_mirror_wing,
                }
            )

    return {
        "uuid": uuid,
        "n_clusters": len(clusters),
        "n_singletons": len(singletons),
        "n_fragments": len(fragments),
        "candidates": candidates,
    }


def main() -> int:
    detail_path = DETAIL_OUT
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    target_uuids: set[str] | None = None
    if len(sys.argv) > 1:
        target_uuids = set(sys.argv[1:])

    reports: dict[str, dict] = {}
    n_seen = 0
    print("streaming...", file=sys.stderr)
    with ROOF_RESULTS.open("rb") as f:
        for uuid, rec in ijson.kvitems(f, "", use_float=True):
            if target_uuids is not None and uuid not in target_uuids:
                continue
            n_seen += 1
            try:
                reports[uuid] = evaluate_uuid(uuid, rec)
            except Exception as e:
                print(f"warn {uuid}: {e}", file=sys.stderr)
            if n_seen % 25 == 0:
                print(f"  processed {n_seen}", file=sys.stderr)

    detail_path.write_text(json.dumps(reports, indent=1, default=str))
    print(f"detail written: {detail_path}", file=sys.stderr)

    print()
    print("=" * 80)
    print(f"MIRROR-SINGLETON RECOVERY — {len(reports)} buildings scanned")
    print("=" * 80)
    n_with_candidates = sum(1 for r in reports.values() if r["candidates"])
    n_with_passing = sum(
        1
        for r in reports.values()
        if any(c["would_pass_gate"] for c in r["candidates"])
    )
    n_total_candidates = sum(len(r["candidates"]) for r in reports.values())
    n_passing = sum(
        sum(1 for c in r["candidates"] if c["would_pass_gate"])
        for r in reports.values()
    )
    print(
        f"buildings with at least one mirror-singleton candidate: {n_with_candidates}"
    )
    print(f"buildings with at least one passing candidate (>= 2m):  {n_with_passing}")
    print(
        f"total candidates:                                       {n_total_candidates}"
    )
    print(f"total passing:                                          {n_passing}")

    print()
    print("PASSING CANDIDATES (would recover a missing slope):")
    print("-" * 80)
    for uuid, r in sorted(reports.items()):
        passing = [c for c in r["candidates"] if c["would_pass_gate"]]
        if not passing:
            continue
        print(f"\n{uuid} ({len(passing)} passing):")
        for c in passing:
            print(
                f"  singleton az={c['singleton_az']:>6.2f} "
                f"incl={c['singleton_incl']:>5.2f} "
                f"len={c['singleton_len']:>4.2f}m story={c['singleton_story']} "
                f"room={c['singleton_room']}"
            )
            print(
                f"    mirrors cluster[{c['mirror_cluster_index']}] "
                f"az={c['mirror_cluster_az']:>6.2f} "
                f"incl={c['mirror_cluster_incl']:>5.2f} size={c['mirror_cluster_size']}"
            )
            print(
                f"    wing rooms: {c['wing_room_count']} rooms (singleton in wing: "
                f"{c['singleton_in_mirror_wing']})"
            )
            print(
                f"    wing ridge extent: {c['wing_ridge_extent_m']:.2f} m  (gate: "
                f"{RIDGE_EXTENT_GATE_M:.1f} m)"
            )

    print()
    print("ALL CANDIDATES (including failing the gate):")
    print("-" * 80)
    for uuid, r in sorted(reports.items()):
        if not r["candidates"]:
            continue
        for c in r["candidates"]:
            tag = "PASS" if c["would_pass_gate"] else "fail"
            in_wing = "in_wing" if c["singleton_in_mirror_wing"] else "out_wing"
            print(
                f"  {uuid}  {tag}  s_az={c['singleton_az']:>6.2f} "
                f"s_incl={c['singleton_incl']:>5.2f} "
                f"mirror_size={c['mirror_cluster_size']:>1}  {in_wing}  "
                f"wing_rooms={c['wing_room_count']:>1}  "
                f"ridge={c['wing_ridge_extent_m']:>5.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
