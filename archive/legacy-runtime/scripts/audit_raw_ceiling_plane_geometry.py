"""Baseline distribution audit of raw RoomPlan ceiling planes.

Read-only. For every ``raw_ceiling_planes`` entry in ``reconcile/buildings_3d.json``
fits an SVD plane, derives ``(azimuth, inclination, area, y_range, n_corners,
centroid)`` and writes:

* ``reports/raw_ceiling_geometry/per_plane.csv`` — one row per plane with the
  shareable viewer element id so rows can be jumped to via
  ``python -m reconcile.element_locator --element-id ...``.
* ``reports/raw_ceiling_geometry/per_building.csv`` — per-building plane counts
  and oblique share.
* ``reports/raw_ceiling_geometry/summary.json`` — corpus histograms for
  inclination, azimuth, area, n_corners, y_range.

This is the baseline for later H1/H3 work: it quantifies how many oblique raw
planes exist before we try to use them to validate or split roof surfaces.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
OUT_DIR = REPO / "reports" / "raw_ceiling_geometry"

OBLIQUE_MIN_INCL_DEG = 5.0
STEEP_MIN_INCL_DEG = 30.0


def _fit_plane(corners: list[list[float]]):
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


def _inclination_deg(normal: np.ndarray) -> float:
    ny = max(-1.0, min(1.0, float(normal[1])))
    return math.degrees(math.acos(abs(ny)))


def _azimuth_deg(normal: np.ndarray) -> float:
    # Downhill direction in the XZ plane: negative horizontal component of the
    # upward normal. Returns 0..360 with 0 = +Z (north), 90 = +X (east).
    nx, nz = float(normal[0]), float(normal[2])
    if abs(nx) < 1e-9 and abs(nz) < 1e-9:
        return float("nan")
    az = math.degrees(math.atan2(-nx, -nz)) % 360.0
    return az


def _polygon_xz_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    s = 0.0
    n = len(corners)
    for i in range(n):
        x1, z1 = corners[i][0], corners[i][2]
        x2, z2 = corners[(i + 1) % n][0], corners[(i + 1) % n][2]
        s += x1 * z2 - x2 * z1
    return abs(s) * 0.5


def _polygon_3d_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    pts = np.asarray(corners, dtype=float)
    origin = pts[0]
    total = 0.0
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - origin
        v2 = pts[i + 1] - origin
        total += float(np.linalg.norm(np.cross(v1, v2)))
    return total * 0.5


def _collect_rows() -> tuple[list[dict], list[dict]]:
    with BUILDINGS_PATH.open() as handle:
        data = json.load(handle)

    plane_rows: list[dict] = []
    bldg_rows: list[dict] = []

    for building in data:
        uuid = building["uuid"]
        n_planes = 0
        n_oblique = 0
        n_steep = 0
        incls: list[float] = []
        area_sum = 0.0

        for room_index, room in enumerate(building.get("rooms") or []):
            story = int(room.get("story", 0))
            planes = room.get("raw_ceiling_planes") or []
            for plane_index, plane in enumerate(planes):
                corners = plane.get("corners") or []
                if len(corners) < 3:
                    continue
                fit = _fit_plane(corners)
                if fit is None:
                    continue
                centroid, normal = fit
                incl = _inclination_deg(normal)
                az = _azimuth_deg(normal)
                area_xz = _polygon_xz_area(corners)
                area_3d = _polygon_3d_area(corners)
                ys = [float(c[1]) for c in corners]

                n_planes += 1
                if incl >= OBLIQUE_MIN_INCL_DEG:
                    n_oblique += 1
                if incl >= STEEP_MIN_INCL_DEG:
                    n_steep += 1
                incls.append(incl)
                area_sum += area_xz

                plane_rows.append(
                    {
                        "uuid": uuid,
                        "element_id": (
                            f"{uuid}::ceiling-raw::{story}:{room_index}:{plane_index}"
                        ),
                        "story": story,
                        "room_index": room_index,
                        "plane_index": plane_index,
                        "n_corners": len(corners),
                        "inclination_deg": round(incl, 3),
                        "azimuth_deg": round(az, 3) if not math.isnan(az) else "",
                        "centroid_x": round(float(centroid[0]), 4),
                        "centroid_y": round(float(centroid[1]), 4),
                        "centroid_z": round(float(centroid[2]), 4),
                        "y_min": round(min(ys), 4),
                        "y_max": round(max(ys), 4),
                        "y_range": round(max(ys) - min(ys), 4),
                        "area_xz_m2": round(area_xz, 4),
                        "area_3d_m2": round(area_3d, 4),
                    }
                )

        bldg_rows.append(
            {
                "uuid": uuid,
                "address": building.get("address"),
                "n_rooms": len(building.get("rooms") or []),
                "n_planes": n_planes,
                "n_oblique_ge_5deg": n_oblique,
                "n_steep_ge_30deg": n_steep,
                "oblique_share": round(n_oblique / n_planes, 3) if n_planes else 0.0,
                "incl_p50": round(float(np.median(incls)), 2) if incls else "",
                "incl_p95": round(float(np.percentile(incls, 95)), 2) if incls else "",
                "area_xz_sum_m2": round(area_sum, 3),
            }
        )

    return plane_rows, bldg_rows


def _hist(values: list[float], edges: list[float]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for v in values:
        placed = False
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo <= v < hi:
                counter[f"{lo:g}_{hi:g}"] += 1
                placed = True
                break
        if not placed:
            counter[f">={edges[-1]:g}"] += 1
    return dict(counter)


def _summary(plane_rows: list[dict], bldg_rows: list[dict]) -> dict:
    incls = [float(r["inclination_deg"]) for r in plane_rows]
    azs = [float(r["azimuth_deg"]) for r in plane_rows if r["azimuth_deg"] != ""]
    areas = [float(r["area_xz_m2"]) for r in plane_rows]
    y_ranges = [float(r["y_range"]) for r in plane_rows]
    n_corners = [int(r["n_corners"]) for r in plane_rows]

    incl_edges = [0, 1, 2.5, 5, 10, 20, 30, 45, 60, 80]
    area_edges = [0, 0.5, 1, 2, 5, 10, 20, 50]
    y_range_edges = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    az_edges = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    return {
        "n_buildings": len(bldg_rows),
        "n_buildings_with_any_raw": sum(1 for r in bldg_rows if r["n_planes"] > 0),
        "n_planes_total": len(plane_rows),
        "n_oblique_ge_5deg": sum(1 for v in incls if v >= OBLIQUE_MIN_INCL_DEG),
        "n_steep_ge_30deg": sum(1 for v in incls if v >= STEEP_MIN_INCL_DEG),
        "oblique_share_corpus": round(
            sum(1 for v in incls if v >= OBLIQUE_MIN_INCL_DEG) / len(incls), 3
        )
        if incls
        else 0.0,
        "inclination_hist_deg": _hist(incls, incl_edges),
        "azimuth_hist_deg": _hist(azs, az_edges),
        "area_xz_hist_m2": _hist(areas, area_edges),
        "y_range_hist_m": _hist(y_ranges, y_range_edges),
        "n_corners_counter": dict(Counter(n_corners)),
        "buildings_with_most_oblique": [
            {
                "uuid": r["uuid"],
                "n_oblique": r["n_oblique_ge_5deg"],
                "n_planes": r["n_planes"],
            }
            for r in sorted(
                bldg_rows, key=lambda b: b["n_oblique_ge_5deg"], reverse=True
            )[:20]
        ],
    }


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
    plane_rows, bldg_rows = _collect_rows()
    _write_csv(OUT_DIR / "per_plane.csv", plane_rows)
    _write_csv(OUT_DIR / "per_building.csv", bldg_rows)

    summary = _summary(plane_rows, bldg_rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"Wrote {len(plane_rows)} plane rows across "
        f"{summary['n_buildings_with_any_raw']}"
        f"/{summary['n_buildings']} buildings."
    )
    print(
        f"Oblique (>={OBLIQUE_MIN_INCL_DEG} deg): {summary['n_oblique_ge_5deg']}"
        f"  Steep (>={STEEP_MIN_INCL_DEG} deg): {summary['n_steep_ge_30deg']}"
    )
    print("Inclination histogram:", summary["inclination_hist_deg"])
    print("Area XZ histogram:", summary["area_xz_hist_m2"])


if __name__ == "__main__":
    main()
