"""Edge analysis of raw RoomPlan ceiling polygons (H3).

For each ``raw_ceiling_plane`` polygon edge, classify:

* ``horizontal`` — |Δy| <= HORIZONTAL_DY_M (0.05 m)
* ``sloped``    — otherwise

Then, horizontal edges get a second label:

* ``ridge_or_hip`` — endpoints match another raw plane's edge in XZ+y (within
  tolerance) AND the two planes' normals differ by >= RIDGE_MIN_ANGLE_DEG.
* ``eave``        — near the building XZ footprint (from
  ``roof_algorithms_py_results.json::ceiling.footprint``) within EAVE_FP_TOL_M.
* ``isolated``    — neither of the above.

Outputs per building:

* ``reports/raw_ceiling_edges/per_building.csv`` — edge counts by label, ridge
  candidate count vs. pipeline ``valid_clusters`` count (sanity check for
  over-/under-clustering).
* ``reports/raw_ceiling_edges/per_edge.csv`` — one row per horizontal edge with
  its label, shareable viewer element id (of the parent raw plane), azimuth of
  the edge in XZ, and any matched partner plane id.
* ``reports/raw_ceiling_edges/summary.json`` — corpus rollups plus top
  buildings whose raw-ceiling ridge count exceeds their pipeline oblique
  cluster count by the largest margin.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
OUT_DIR = REPO / "reports" / "raw_ceiling_edges"

HORIZONTAL_DY_M = 0.05
MIN_EDGE_LEN_M = 0.2
RIDGE_MIN_ANGLE_DEG = 10.0
RIDGE_MATCH_XZ_TOL_M = 0.30
RIDGE_MATCH_Y_TOL_M = 0.20
EAVE_FP_TOL_M = 0.50
NEAR_VERTICAL_EXCLUDE_DEG = 80.0
OBLIQUE_MIN_INCL_DEG = 5.0


def _fit_plane(corners):
    pts = np.asarray(corners, dtype=float)
    if pts.shape[0] < 3 or pts.shape[1] < 3:
        return None
    centroid = pts.mean(axis=0)
    diffs = pts - centroid
    try:
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    n = float(np.linalg.norm(normal))
    if n < 1e-9:
        return None
    return centroid, normal / n


def _inclination_deg(normal):
    ny = max(-1.0, min(1.0, float(normal[1])))
    return math.degrees(math.acos(abs(ny)))


def _normals_angle_deg(n1, n2) -> float:
    dot = float(np.dot(n1, n2))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(abs(dot)))


def _edge_azimuth_xz(a, b) -> float:
    dx = float(b[0] - a[0])
    dz = float(b[2] - a[2])
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return float("nan")
    return math.degrees(math.atan2(dx, dz)) % 180.0


def _footprint_lines(roof_result: dict):
    fp = (roof_result.get("ceiling") or {}).get("footprint") or []
    if len(fp) < 3:
        return []
    lines: list[tuple[float, float, float, float]] = []
    n = len(fp)
    for i in range(n):
        ax, az = (
            float(fp[i][0]),
            float(fp[i][2]) if len(fp[i]) >= 3 else float(fp[i][1]),
        )
        bx, bz = (
            float(fp[(i + 1) % n][0]),
            float(fp[(i + 1) % n][2])
            if len(fp[(i + 1) % n]) >= 3
            else float(fp[(i + 1) % n][1]),
        )
        lines.append((ax, az, bx, bz))
    return lines


def _distance_point_to_segment(px, pz, ax, az, bx, bz) -> float:
    vx, vz = bx - ax, bz - az
    wx, wz = px - ax, pz - az
    denom = vx * vx + vz * vz
    if denom < 1e-12:
        return math.hypot(wx, wz)
    t = max(0.0, min(1.0, (vx * wx + vz * wz) / denom))
    cx, cz = ax + t * vx, az + t * vz
    return math.hypot(px - cx, pz - cz)


def _edge_near_footprint(a, b, fp_lines) -> bool:
    if not fp_lines:
        return False
    ax_, az_ = a[0], a[2]
    bx_, bz_ = b[0], b[2]
    mx, mz = (ax_ + bx_) * 0.5, (az_ + bz_) * 0.5
    for ax, az, bx, bz in fp_lines:
        for px, pz in ((ax_, az_), (bx_, bz_), (mx, mz)):
            if _distance_point_to_segment(px, pz, ax, az, bx, bz) <= EAVE_FP_TOL_M:
                return True
    return False


def _endpoints_match(a1, b1, a2, b2) -> bool:
    def close(p, q) -> bool:
        return (
            math.hypot(p[0] - q[0], p[2] - q[2]) <= RIDGE_MATCH_XZ_TOL_M
            and abs(p[1] - q[1]) <= RIDGE_MATCH_Y_TOL_M
        )

    return (close(a1, a2) and close(b1, b2)) or (close(a1, b2) and close(b1, a2))


def _collect_edges(building, roof_result):
    building["uuid"]
    fp_lines = _footprint_lines(roof_result)

    planes = []  # (story, ri, pi, corners, normal, incl)
    for ri, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        for pi, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) < 3:
                continue
            fit = _fit_plane(corners)
            if fit is None:
                continue
            _, normal = fit
            incl = _inclination_deg(normal)
            if incl >= NEAR_VERTICAL_EXCLUDE_DEG:
                continue
            planes.append((story, ri, pi, corners, normal, incl))

    edge_rows: list[dict] = []
    edge_records = []  # ((story, ri, pi), a, b, dy, normal) for ridge matching

    for story, ri, pi, corners, normal, incl in planes:
        n = len(corners)
        for j in range(n):
            a = corners[j]
            b = corners[(j + 1) % n]
            dy = abs(float(b[1] - a[1]))
            length = math.dist(a, b)
            if length < MIN_EDGE_LEN_M:
                continue
            horizontal = dy <= HORIZONTAL_DY_M
            edge_records.append(
                (
                    story,
                    ri,
                    pi,
                    a,
                    b,
                    dy,
                    horizontal,
                    normal,
                    incl,
                    length,
                )
            )

    # Match horizontal edges across different planes (ridge/hip candidates).
    by_story: dict[int, list] = defaultdict(list)
    for rec in edge_records:
        by_story[rec[0]].append(rec)

    for story, recs in by_story.items():
        for i, rec in enumerate(recs):
            (_, ri, pi, a, b, dy, horiz, normal, incl, length) = rec
            if not horiz:
                continue
            match_partner = None
            match_angle = 0.0
            for j in range(len(recs)):
                if i == j:
                    continue
                (_, ri2, pi2, a2, b2, _dy2, horiz2, normal2, _incl2, _len2) = recs[j]
                if ri2 == ri and pi2 == pi:
                    continue
                if not horiz2:
                    continue
                if not _endpoints_match(a, b, a2, b2):
                    continue
                angle = _normals_angle_deg(normal, normal2)
                if angle >= RIDGE_MIN_ANGLE_DEG and angle > match_angle:
                    match_partner = (ri2, pi2, angle)
                    match_angle = angle

            if match_partner is not None:
                label = "ridge_or_hip"
                partner_id = (
                    f"{building['uuid']}::ceiling-raw::{story}:"
                    f"{match_partner[0]}:{match_partner[1]}"
                )
            elif _edge_near_footprint(a, b, fp_lines):
                label = "eave"
                partner_id = ""
            else:
                label = "isolated"
                partner_id = ""

            edge_rows.append(
                {
                    "uuid": building["uuid"],
                    "plane_element_id": (
                        f"{building['uuid']}::ceiling-raw::{story}:{ri}:{pi}"
                    ),
                    "story": story,
                    "room_index": ri,
                    "plane_index": pi,
                    "length_m": round(length, 3),
                    "edge_azimuth_xz_deg": round(_edge_azimuth_xz(a, b), 2)
                    if not math.isnan(_edge_azimuth_xz(a, b))
                    else "",
                    "y_mid": round((a[1] + b[1]) * 0.5, 4),
                    "dy": round(dy, 4),
                    "plane_inclination_deg": round(incl, 2),
                    "label": label,
                    "partner_plane_id": partner_id,
                    "partner_angle_deg": round(match_angle, 2) if match_partner else "",
                }
            )

    bldg_totals = {
        "total_edges": sum(1 for r in edge_records),
        "horizontal_edges": sum(1 for r in edge_records if r[6]),
        "sloped_edges": sum(1 for r in edge_records if not r[6]),
        "ridge_or_hip": sum(1 for r in edge_rows if r["label"] == "ridge_or_hip"),
        "eave": sum(1 for r in edge_rows if r["label"] == "eave"),
        "isolated_horizontal": sum(1 for r in edge_rows if r["label"] == "isolated"),
    }
    return edge_rows, bldg_totals


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
    with ROOF_RESULTS_PATH.open() as handle:
        roof_results = json.load(handle)

    all_edge_rows: list[dict] = []
    bldg_rows: list[dict] = []
    corpus = Counter()

    for building in buildings:
        uuid = building["uuid"]
        roof_result = roof_results.get(uuid, {})
        edges, totals = _collect_edges(building, roof_result)
        if not edges and not totals["total_edges"]:
            continue
        all_edge_rows.extend(edges)
        n_oblique_clusters = sum(
            1
            for cl in roof_result.get("valid_clusters") or []
            if cl.get("avgIncl", 0) > 0
        )
        n_oblique_surfaces = len(
            (roof_result.get("roof_surfaces") or {}).get("oblique") or []
        )
        bldg_rows.append(
            {
                "uuid": uuid,
                "total_edges": totals["total_edges"],
                "horizontal_edges": totals["horizontal_edges"],
                "sloped_edges": totals["sloped_edges"],
                "ridge_or_hip_edges": totals["ridge_or_hip"],
                "eave_edges": totals["eave"],
                "isolated_horizontal_edges": totals["isolated_horizontal"],
                "pipeline_oblique_clusters": n_oblique_clusters,
                "pipeline_oblique_surfaces": n_oblique_surfaces,
                "ridge_minus_clusters": totals["ridge_or_hip"] - n_oblique_clusters,
            }
        )
        for k, v in totals.items():
            corpus[k] += v

    _write_csv(OUT_DIR / "per_edge.csv", all_edge_rows)
    _write_csv(OUT_DIR / "per_building.csv", bldg_rows)

    top_under_clustered = sorted(
        bldg_rows, key=lambda r: r["ridge_minus_clusters"], reverse=True
    )[:15]
    top_over_clustered = sorted(bldg_rows, key=lambda r: r["ridge_minus_clusters"])[:15]
    summary = {
        "thresholds": {
            "horizontal_dy_m": HORIZONTAL_DY_M,
            "min_edge_len_m": MIN_EDGE_LEN_M,
            "ridge_min_angle_deg": RIDGE_MIN_ANGLE_DEG,
            "ridge_match_xz_tol_m": RIDGE_MATCH_XZ_TOL_M,
            "ridge_match_y_tol_m": RIDGE_MATCH_Y_TOL_M,
            "eave_fp_tol_m": EAVE_FP_TOL_M,
        },
        "n_buildings": len(bldg_rows),
        "corpus_totals": dict(corpus),
        "top_under_clustered": [
            {
                k: r[k]
                for k in (
                    "uuid",
                    "ridge_or_hip_edges",
                    "pipeline_oblique_clusters",
                    "ridge_minus_clusters",
                )
            }
            for r in top_under_clustered
        ],
        "top_over_clustered": [
            {
                k: r[k]
                for k in (
                    "uuid",
                    "ridge_or_hip_edges",
                    "pipeline_oblique_clusters",
                    "ridge_minus_clusters",
                )
            }
            for r in top_over_clustered
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"Edges total: {corpus['total_edges']}  horizontal: "
        f"{corpus['horizontal_edges']}"
        f"  sloped: {corpus['sloped_edges']}"
    )
    print(
        f"Horizontal labels — ridge_or_hip: {corpus['ridge_or_hip']}"
        f"  eave: {corpus['eave']}  isolated: {corpus['isolated_horizontal']}"
    )


if __name__ == "__main__":
    main()
