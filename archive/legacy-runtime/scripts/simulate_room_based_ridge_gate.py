"""Simulate the proposed fix: replace segment-endpoint ridge extent with
room-footprint ridge extent in `build_ceiling_planes`.

For each cluster (kept or dropped), compute:
  * segment-based ridge extent (current rule)
  * room-based ridge extent: union of floor polygons of rooms whose segments
    are in the cluster, projected onto the ridge axis
  * verdict under each rule against the 2.0 m gate

This is a dry-run only — no pipeline changes. The intent is to confirm that
the proposed change recovers the right clusters on 74e87bcd and a list of
peer buildings, without surfacing rake-edge artifacts elsewhere.
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
DETAIL_OUT = REPO / ".context" / "room_based_ridge_simulation.json"

GATE_M = 2.0


def project_on_ridge(
    points: np.ndarray, az_deg: float, ref: np.ndarray
) -> tuple[float, float]:
    az = math.radians(az_deg)
    ridge = np.array([-math.cos(az), 0.0, math.sin(az)])
    proj = (points - ref) @ ridge
    return float(proj.min()), float(proj.max())


def evaluate_uuid(uuid: str, rec: dict) -> dict:
    clusters = rec.get("valid_clusters") or []
    obliques = rec.get("roof_surfaces", {}).get("oblique") or []
    committed_keys = {
        (
            round(float((o.get("cluster") or {}).get("avgAzimuth") or 0), 2),
            round(float((o.get("cluster") or {}).get("avgIncl") or 0), 2),
        )
        for o in obliques
    }

    room_outlines: dict[tuple[int, int], np.ndarray] = {}
    for rp in rec.get("ceiling", {}).get("room_partitions") or []:
        outline = rp.get("room_outline") or []
        if not outline:
            continue
        try:
            pts = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in outline])
        except Exception:
            continue
        room_outlines[(int(rp.get("story", -1)), int(rp.get("room_index", -1)))] = pts

    cluster_results = []
    for ci, cl in enumerate(clusters):
        az = float(cl.get("avgAzimuth") or 0.0)
        incl = float(cl.get("avgIncl") or 0.0)
        segs = cl.get("segs") or []
        seg_pts = []
        rooms_seen = set()
        for s in segs:
            seg_pts.append(s.get("a"))
            seg_pts.append(s.get("b"))
            rooms_seen.add((int(s.get("story", -1)), int(s.get("room_idx", -1))))
        seg_pts = np.array(seg_pts) if seg_pts else np.zeros((0, 3))

        ref = (
            np.array(
                [
                    sum((s["a"][0] + s["b"][0]) / 2 for s in segs) / max(1, len(segs)),
                    sum((s["a"][1] + s["b"][1]) / 2 for s in segs) / max(1, len(segs)),
                    sum((s["a"][2] + s["b"][2]) / 2 for s in segs) / max(1, len(segs)),
                ]
            )
            if segs
            else np.zeros(3)
        )

        if seg_pts.size:
            seg_min, seg_max = project_on_ridge(seg_pts, az, ref)
            seg_extent = seg_max - seg_min
        else:
            seg_extent = 0.0

        room_pts = []
        for key in rooms_seen:
            if key in room_outlines:
                room_pts.append(room_outlines[key])
        room_pts = np.concatenate(room_pts) if room_pts else np.zeros((0, 3))
        if room_pts.size:
            r_min, r_max = project_on_ridge(room_pts, az, ref)
            room_extent = r_max - r_min
        else:
            room_extent = 0.0

        seg_pass = seg_extent >= GATE_M
        room_pass = room_extent >= GATE_M
        currently_kept = (round(az, 2), round(incl, 2)) in committed_keys

        cluster_results.append(
            {
                "cluster_index": ci,
                "az": round(az, 2),
                "incl": round(incl, 2),
                "seg_count": len(segs),
                "rooms": sorted([list(k) for k in rooms_seen]),
                "seg_ridge_extent_m": round(seg_extent, 3),
                "room_ridge_extent_m": round(room_extent, 3),
                "current_rule_pass": seg_pass,
                "proposed_rule_pass": room_pass,
                "currently_kept_in_results": currently_kept,
                "verdict_change": (
                    "recovered"
                    if room_pass and not seg_pass
                    else (
                        "kept"
                        if room_pass and seg_pass
                        else (
                            "still_dropped"
                            if not room_pass and not seg_pass
                            else "would_drop"
                        )
                    )
                ),
            }
        )

    return {
        "uuid": uuid,
        "n_clusters": len(clusters),
        "clusters": cluster_results,
    }


def main() -> int:
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)

    target_uuids: set[str] | None = None
    if len(sys.argv) > 1:
        target_uuids = set(sys.argv[1:])

    reports: dict[str, dict] = {}
    print("streaming...", file=sys.stderr)
    with ROOF_RESULTS.open("rb") as f:
        for uuid, rec in ijson.kvitems(f, "", use_float=True):
            if target_uuids is not None and uuid not in target_uuids:
                continue
            try:
                reports[uuid] = evaluate_uuid(uuid, rec)
            except Exception as e:
                print(f"warn {uuid}: {e}", file=sys.stderr)

    DETAIL_OUT.write_text(json.dumps(reports, indent=1, default=str))
    print(f"detail written: {DETAIL_OUT}", file=sys.stderr)

    n = len(reports)
    n_recovered = 0
    n_recovered_buildings = 0
    n_would_drop = 0
    for r in reports.values():
        recovered = sum(1 for c in r["clusters"] if c["verdict_change"] == "recovered")
        would_drop = sum(
            1 for c in r["clusters"] if c["verdict_change"] == "would_drop"
        )
        n_recovered += recovered
        n_would_drop += would_drop
        if recovered:
            n_recovered_buildings += 1

    print()
    print("=" * 80)
    print(f"ROOM-BASED RIDGE GATE — simulation across {n} buildings")
    print("=" * 80)
    print(f"clusters RECOVERED (failed seg gate, would pass room gate):  {n_recovered}")
    print(
        f"buildings with at least one recovered cluster:               "
        f"{n_recovered_buildings}"
    )
    print(
        f"clusters that would NOW DROP (passed seg, fail room):        {n_would_drop}  "
        f"(regression risk)"
    )

    target = "74e87bcd-3989-4d5c-8f16-f7782dc3afbd"
    if target in reports:
        print()
        print(f"TARGET {target}")
        print("-" * 80)
        for c in reports[target]["clusters"]:
            print(
                f"  cluster[{c['cluster_index']}] az={c['az']:>6.2f} "
                f"incl={c['incl']:>5.2f} "
                f"segs={c['seg_count']}  seg_ridge={c['seg_ridge_extent_m']:>5.2f}m  "
                f"room_ridge={c['room_ridge_extent_m']:>5.2f}m  "
                f"verdict: {c['verdict_change']}"
            )

    print()
    print("=" * 80)
    print("RECOVERED CLUSTERS (per uuid)")
    print("=" * 80)
    for uuid, r in sorted(reports.items()):
        recovered = [c for c in r["clusters"] if c["verdict_change"] == "recovered"]
        if not recovered:
            continue
        print(f"\n{uuid}:")
        for c in recovered:
            print(
                f"  [{c['cluster_index']}] az={c['az']:>6.2f} incl={c['incl']:>5.2f} "
                f"segs={c['seg_count']}  seg={c['seg_ridge_extent_m']:>5.2f}m  "
                f"room={c['room_ridge_extent_m']:>5.2f}m  rooms={c['rooms']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
