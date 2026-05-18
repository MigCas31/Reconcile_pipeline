"""Scan-trust audit (H5): does the raw ceiling perimeter sit on the wall-tops?

For each room in ``buildings_3d.json`` with at least one ``raw_ceiling_planes``
entry, compute the "perimeter-to-wall-top distance": for every raw plane
corner, find the nearest 3D point on any ``walls_computed`` corner polygon
top-edge and record the distance. A room whose p95 distance is large is
untrustworthy scan data — H1/H3/H4 conclusions for that room should be
down-weighted.

Walls-computed corners from ``room.walls_computed[].corners`` define 3D wall
polygons. The wall-top is the set of corner points with the maximum y-value
for that wall (within a small tolerance to handle leaning walls).

Outputs ``reports/raw_ceiling_wall_top_trust/`` with:

* ``per_plane.csv`` — per-plane p50/p95 distance to wall-tops, number of
  corners that land within TRUSTED_DIST_M (0.30 m).
* ``per_room.csv`` — per-room trust score (fraction of corners trusted across
  all planes in the room).
* ``summary.json`` — corpus distribution, worst rooms.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
OUT_DIR = REPO / "reports" / "raw_ceiling_wall_top_trust"

TRUSTED_DIST_M = 0.30
WALL_TOP_TOL_M = 0.10


def _wall_top_segments(
    room: dict,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    segs = []
    for wall in room.get("walls_computed") or []:
        corners = wall.get("corners") or []
        if len(corners) < 3:
            continue
        ys = [float(c[1]) for c in corners]
        top_y = max(ys)
        tops = [
            tuple(float(v) for v in c)
            for c in corners
            if abs(float(c[1]) - top_y) <= WALL_TOP_TOL_M
        ]
        if len(tops) < 2:
            continue
        tops.sort(key=lambda p: (p[0], p[2]))
        for i in range(len(tops) - 1):
            segs.append((tops[i], tops[i + 1]))
        segs.append((tops[-1], tops[0]))
    return segs


def _dist_point_to_segment_3d(p, a, b) -> float:
    ap = np.array([p[0] - a[0], p[1] - a[1], p[2] - a[2]])
    ab = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, float(np.dot(ap, ab)) / denom))
    closest = np.array(a) + t * ab
    return float(np.linalg.norm(np.array(p) - closest))


def _min_dist_to_wall_tops(p, wall_segs) -> float:
    if not wall_segs:
        return float("inf")
    return min(_dist_point_to_segment_3d(p, a, b) for a, b in wall_segs)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BUILDINGS_PATH.open() as handle:
        buildings = json.load(handle)

    plane_rows: list[dict] = []
    room_rows: list[dict] = []

    for building in buildings:
        uuid = building["uuid"]
        for ri, room in enumerate(building.get("rooms") or []):
            planes = room.get("raw_ceiling_planes") or []
            if not planes:
                continue
            wall_segs = _wall_top_segments(room)
            if not wall_segs:
                continue
            story = int(room.get("story", 0))

            room_dists: list[float] = []
            room_trusted = 0
            room_total_corners = 0
            for pi, plane in enumerate(planes):
                corners = plane.get("corners") or []
                if len(corners) < 3:
                    continue
                dists = [
                    _min_dist_to_wall_tops(
                        (float(c[0]), float(c[1]), float(c[2])), wall_segs
                    )
                    for c in corners
                ]
                trusted = sum(1 for d in dists if d <= TRUSTED_DIST_M)
                p50 = float(np.median(dists))
                p95 = float(np.percentile(dists, 95))
                plane_rows.append(
                    {
                        "uuid": uuid,
                        "element_id": f"{uuid}::ceiling-raw::{story}:{ri}:{pi}",
                        "story": story,
                        "room_index": ri,
                        "plane_index": pi,
                        "n_corners": len(corners),
                        "p50_dist_to_wall_top_m": round(p50, 3),
                        "p95_dist_to_wall_top_m": round(p95, 3),
                        "trusted_corner_fraction": round(trusted / len(corners), 3),
                    }
                )
                room_dists.extend(dists)
                room_trusted += trusted
                room_total_corners += len(corners)

            if room_total_corners == 0:
                continue
            room_rows.append(
                {
                    "uuid": uuid,
                    "story": story,
                    "room_index": ri,
                    "n_planes": len(planes),
                    "n_corners_total": room_total_corners,
                    "trust_score": round(room_trusted / room_total_corners, 3),
                    "p50_dist_m": round(float(np.median(room_dists)), 3),
                    "p95_dist_m": round(float(np.percentile(room_dists, 95)), 3),
                }
            )

    _write_csv(OUT_DIR / "per_plane.csv", plane_rows)
    _write_csv(OUT_DIR / "per_room.csv", room_rows)

    trust_scores = [r["trust_score"] for r in room_rows]
    p95s = [r["p95_dist_m"] for r in room_rows]

    buckets = [0, 0.25, 0.5, 0.75, 0.9, 1.0001]
    trust_hist: Counter = Counter()
    for v in trust_scores:
        for i in range(len(buckets) - 1):
            if buckets[i] <= v < buckets[i + 1]:
                trust_hist[f"{buckets[i]:g}-{buckets[i + 1]:g}"] += 1
                break

    worst_rooms = sorted(room_rows, key=lambda r: r["trust_score"])[:20]
    summary = {
        "thresholds": {
            "trusted_dist_m": TRUSTED_DIST_M,
            "wall_top_tol_m": WALL_TOP_TOL_M,
        },
        "n_rooms_scored": len(room_rows),
        "n_planes_scored": len(plane_rows),
        "trust_score_hist": dict(trust_hist),
        "trust_score_p50": round(float(np.median(trust_scores)), 3)
        if trust_scores
        else 0.0,
        "trust_score_p25": round(float(np.percentile(trust_scores, 25)), 3)
        if trust_scores
        else 0.0,
        "p95_dist_m_p50": round(float(np.median(p95s)), 3) if p95s else 0.0,
        "p95_dist_m_p95": round(float(np.percentile(p95s, 95)), 3) if p95s else 0.0,
        "worst_rooms": [
            {
                k: r[k]
                for k in (
                    "uuid",
                    "story",
                    "room_index",
                    "trust_score",
                    "p95_dist_m",
                    "n_planes",
                )
            }
            for r in worst_rooms
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Rooms scored: {len(room_rows)}  planes: {len(plane_rows)}")
    print("Trust score hist:", dict(trust_hist))
    print(f"Trust p50={summary['trust_score_p50']}  p25={summary['trust_score_p25']}")
    print(
        f"p95 distance to wall-top — median={summary['p95_dist_m_p50']} m, "
        f"p95={summary['p95_dist_m_p95']} m"
    )


if __name__ == "__main__":
    main()
