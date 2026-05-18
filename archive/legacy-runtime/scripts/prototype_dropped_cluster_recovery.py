"""Prototype: recover dropped oblique clusters using ceiling-polygon plane fits.

For each cluster that survived `cluster_oblique_segments` but produced no
surface in `roof_surfaces.oblique` (typically because of the 2 m ridge-extent
gate at `ceiling_plane_generation.py:305`), this script:

1. Computes the slope plane from the cluster (azimuth + inclination + segment
   midpoint).
2. Inspects every `ceiling.oblique` polygon (the slanted ceiling fragments
   inferred from the raw scan) and fits a plane through it via least squares.
3. Compares each fitted plane's normal + offset to the cluster's slope plane.
   Fragments that match within tolerance are counted as raw-scan evidence
   that the cluster's slope plane is real.
4. Computes the prospective ridge extent from the union of those fragments'
   polygon footprints, projected onto the ridge axis. If >= 2.0 m, the cluster
   would pass the gate when measured against rooms-under-slope rather than
   segment endpoints.

Wing dominance:
  For each cluster, the area sum of matched ceiling fragments is the dominance
  weight. Compared across clusters this tells us which slope covers more of
  the building (i.e., which wing is dominant in T or L footprints).

The script does NOT modify the pipeline. It writes a per-UUID JSON report.
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
DETAIL_OUT = REPO / ".context" / "dropped_cluster_recovery.json"

PLANE_NORMAL_TOL_DEG = 12.0
PLANE_OFFSET_TOL_M = 0.6
RIDGE_EXTENT_GATE_M = 2.0


def plane_normal(azimuth_deg: float, incl_deg: float) -> np.ndarray:
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
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    if normal[1] < 0:
        normal = -normal
    offset = float(normal @ centroid)
    return normal, offset


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cos_a = float(np.clip(a @ b, -1.0, 1.0))
    return math.degrees(math.acos(abs(cos_a)))


def cluster_segment_midpoint(cluster: dict) -> np.ndarray:
    pts = []
    for s in cluster.get("segs") or []:
        a = s.get("a")
        b = s.get("b")
        if a and b:
            pts.append([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2])
    return np.array(pts).mean(axis=0) if pts else np.zeros(3)


def project_on_ridge(
    points: np.ndarray, az_deg: float, ref: np.ndarray
) -> tuple[float, float]:
    az = math.radians(az_deg)
    ridge = np.array([-math.cos(az), 0.0, math.sin(az)])
    proj = (points - ref) @ ridge
    return float(proj.min()), float(proj.max())


def project_on_slope(
    points: np.ndarray, az_deg: float, ref: np.ndarray
) -> tuple[float, float]:
    az = math.radians(az_deg)
    slope = np.array([math.sin(az), 0.0, math.cos(az)])
    proj = (points - ref) @ slope
    return float(proj.min()), float(proj.max())


def cluster_to_slope(cluster: dict) -> dict:
    az = float(cluster["avgAzimuth"])
    incl = float(cluster["avgIncl"])
    n = plane_normal(az, incl)
    ref = cluster_segment_midpoint(cluster)
    return {
        "az": az,
        "incl": incl,
        "n": n,
        "ref": ref,
        "offset": float(n @ ref),
    }


def fragment_plane(fragment: dict) -> tuple[np.ndarray, float, np.ndarray] | None:
    poly = fragment.get("poly") or []
    if len(poly) < 3:
        return None
    pts = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in poly])
    fit = fit_plane_lsq(pts)
    if fit is None:
        return None
    normal, offset = fit
    return normal, offset, pts


def evaluate_uuid(uuid: str, rec: dict) -> dict:
    clusters = rec.get("valid_clusters") or []
    obliques = rec.get("roof_surfaces", {}).get("oblique") or []
    committed_keys = set()
    for o in obliques:
        cl = o.get("cluster") or {}
        committed_keys.add(
            (
                round(float(cl.get("avgAzimuth") or 0), 2),
                round(float(cl.get("avgIncl") or 0), 2),
            )
        )

    ceiling = rec.get("ceiling") or {}
    fragments = ceiling.get("oblique") or []
    fragment_planes = []
    for i, f in enumerate(fragments):
        fp = fragment_plane(f)
        if fp is None:
            continue
        normal, offset, pts = fp
        fragment_planes.append(
            {
                "index": i,
                "id": f.get("id"),
                "story": f.get("story"),
                "room_index": f.get("room_index"),
                "area_m2": float(f.get("area_m2") or 0.0),
                "normal": normal,
                "offset": offset,
                "pts": pts,
                "current_hypothesis": f.get("roof_hypothesis_id"),
            }
        )

    cluster_reports = []
    for ci, cl in enumerate(clusters):
        slope = cluster_to_slope(cl)
        committed = (round(slope["az"], 2), round(slope["incl"], 2)) in committed_keys
        seg_pts = []
        for s in cl.get("segs") or []:
            seg_pts.append(s.get("a"))
            seg_pts.append(s.get("b"))
        seg_pts = np.array(seg_pts) if seg_pts else np.zeros((0, 3))
        if seg_pts.size:
            seg_min_r, seg_max_r = project_on_ridge(seg_pts, slope["az"], slope["ref"])
            seg_ridge_extent = seg_max_r - seg_min_r
        else:
            seg_ridge_extent = 0.0

        matched_fragments = []
        for fp in fragment_planes:
            d_angle = angle_between(slope["n"], fp["normal"])
            d_offset = abs(slope["offset"] - fp["offset"])
            if d_angle <= PLANE_NORMAL_TOL_DEG and d_offset <= PLANE_OFFSET_TOL_M:
                matched_fragments.append(
                    {
                        **fp,
                        "delta_angle": d_angle,
                        "delta_offset": d_offset,
                    }
                )

        matched_pts = (
            np.concatenate([m["pts"] for m in matched_fragments])
            if matched_fragments
            else np.zeros((0, 3))
        )
        if matched_pts.size:
            frag_min_r, frag_max_r = project_on_ridge(
                matched_pts, slope["az"], slope["ref"]
            )
            frag_ridge_extent = frag_max_r - frag_min_r
            frag_min_s, frag_max_s = project_on_slope(
                matched_pts, slope["az"], slope["ref"]
            )
            frag_slope_extent = frag_max_s - frag_min_s
        else:
            frag_ridge_extent = 0.0
            frag_slope_extent = 0.0

        matched_area_m2 = sum(m["area_m2"] for m in matched_fragments)

        cluster_reports.append(
            {
                "cluster_index": ci,
                "az": round(slope["az"], 2),
                "incl": round(slope["incl"], 2),
                "segs_count": len(cl.get("segs") or []),
                "segments_ridge_extent_m": round(seg_ridge_extent, 3),
                "currently_committed": committed,
                "matched_fragments": [
                    {
                        "index": m["index"],
                        "id": m["id"],
                        "story": m["story"],
                        "room_index": m["room_index"],
                        "area_m2": round(m["area_m2"], 2),
                        "delta_angle_deg": round(m["delta_angle"], 2),
                        "delta_offset_m": round(m["delta_offset"], 3),
                        "current_hypothesis": m["current_hypothesis"],
                    }
                    for m in matched_fragments
                ],
                "matched_fragment_count": len(matched_fragments),
                "matched_area_m2": round(matched_area_m2, 2),
                "fragment_ridge_extent_m": round(frag_ridge_extent, 3),
                "fragment_slope_extent_m": round(frag_slope_extent, 3),
                "would_pass_gate": frag_ridge_extent >= RIDGE_EXTENT_GATE_M,
            }
        )

    total_matched_area = sum(c["matched_area_m2"] for c in cluster_reports)
    for c in cluster_reports:
        c["dominance_pct"] = (
            round(100.0 * c["matched_area_m2"] / total_matched_area, 1)
            if total_matched_area > 0
            else 0.0
        )

    return {
        "uuid": uuid,
        "fragment_count": len(fragments),
        "fragment_with_plane": len(fragment_planes),
        "clusters": cluster_reports,
        "fragments_total_area_m2": round(
            sum(f.get("area_m2") or 0 for f in fragments), 2
        ),
        "fragments_matched_area_m2": round(total_matched_area, 2),
    }


def main() -> int:
    detail_path = DETAIL_OUT
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    target_uuids = set()
    if len(sys.argv) > 1:
        target_uuids = set(sys.argv[1:])

    summary_path = REPO / ".context" / "roof_signal_availability.json"
    if not target_uuids and summary_path.exists():
        prior = json.loads(summary_path.read_text())
        for s in prior:
            if s.get("dropped_clusters", 0) > 0 and s.get("concavity", 0) >= 0.10:
                target_uuids.add(s["uuid"])
        target_uuids.add("74e87bcd-3989-4d5c-8f16-f7782dc3afbd")
    print(f"target uuids: {len(target_uuids)}", file=sys.stderr)

    reports: dict[str, dict] = {}
    with ROOF_RESULTS.open("rb") as f:
        for uuid, rec in ijson.kvitems(f, "", use_float=True):
            if uuid not in target_uuids:
                continue
            try:
                reports[uuid] = evaluate_uuid(uuid, rec)
            except Exception as e:
                print(f"warn {uuid}: {e}", file=sys.stderr)

    detail_path.write_text(json.dumps(reports, indent=1, default=str))
    print(f"detail written: {detail_path}", file=sys.stderr)

    print()
    print("=" * 70)
    print("DROPPED CLUSTER RECOVERY — per-uuid summary")
    print("=" * 70)
    for uuid, r in sorted(reports.items()):
        [c for c in r["clusters"] if c["currently_committed"]]
        dropped = [c for c in r["clusters"] if not c["currently_committed"]]
        recoverable = [c for c in dropped if c["would_pass_gate"]]
        print(
            f"\n{uuid}  fragments={r['fragment_count']} (matched_area={
                r[('fragments_matched_area_m2')]:.1f}/{
                r['fragments_total_area_m2']:.1f} m²)"
        )
        for c in r["clusters"]:
            tag = (
                "COMMITTED"
                if c["currently_committed"]
                else ("RECOVERABLE" if c["would_pass_gate"] else "STILL DROPPED")
            )
            print(
                f"  [{c['cluster_index']}] az={c['az']:>6.2f} incl={c['incl']:>5.2f}  "
                f"segs={c['segs_count']:>1}  "
                f"seg_ridge={c['segments_ridge_extent_m']:>5.2f}m  "
                f"frag_match={c['matched_fragment_count']:>2} "
                f"(area={c['matched_area_m2']:>5.1f} m², "
                f"dom={c['dominance_pct']:>4.1f}%)  "
                f"frag_ridge={c['fragment_ridge_extent_m']:>5.2f}m  "
                f"frag_slope={c['fragment_slope_extent_m']:>5.2f}m  "
                f"{tag}"
            )
        if recoverable:
            print(f"  -> {len(recoverable)} dropped cluster(s) RECOVERABLE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
