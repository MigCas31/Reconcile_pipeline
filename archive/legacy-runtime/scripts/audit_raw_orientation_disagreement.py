"""Raw-ceiling orientation disagreement over shared (x,z).

Hypothesis (2026-04-21 pivot from the user):
    For a given (x,z), the computed pipeline plane is usually more reliable
    than noisy scan fragments. BUT when **two raw planes cover the same
    (x,z) with different orientations**, that's a slope-split (dormer, hip,
    ridge transition) the pipeline may have flattened into a single
    computed surface.

This audit measures exactly that signal, per pair of raw ceiling planes in
the same building and story:

    overlap_area_m2  - XZ intersection area of the two raw polygons
    angle_deg        - angle between the two fitted-plane normals

Pairs with overlap >= ``MIN_OVERLAP_AREA_M2`` and angle >=
``MIN_DISAGREEMENT_DEG`` are emitted as viewer-consumable polygons, lifted
onto the higher of the two planes so they render at roof level.

Near-vertical raw planes (inclination > ``MAX_INCLINATION_DEG``) are dropped
first — those are RoomPlan wall fragments mislabelled as ceilings, not
real ceiling orientation evidence.

Output::

    reports/raw_orientation_disagreement/
        per_pair.csv
        summary.json
        disagreement_polygons.json   # viewer-consumable, keyed by UUID
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
OUT_DIR = REPO / "reports" / "raw_orientation_disagreement"

MIN_PLANE_AREA_M2 = 0.5
MIN_OVERLAP_AREA_M2 = 0.3
MAX_INCLINATION_DEG = 80.0
MIN_DISAGREEMENT_DEG = 20.0


def _xz_polygon(corners) -> Polygon | None:
    if not corners or len(corners) < 3:
        return None
    p = Polygon([(float(c[0]), float(c[2])) for c in corners])
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or not p.is_valid or p.area < MIN_PLANE_AREA_M2:
        return None
    return p


def _fit_plane(corners):
    pts = np.asarray(corners, dtype=float)
    if pts.shape[0] < 3:
        return None
    centroid = pts.mean(axis=0)
    diffs = pts - centroid
    try:
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vt[-1]
    n = float(np.linalg.norm(normal))
    if n < 1e-9:
        return None
    normal = normal / n
    if normal[1] < 0:
        normal = -normal
    incl = float(np.degrees(np.arccos(min(1.0, abs(float(normal[1]))))))
    az = float(
        (np.degrees(np.arctan2(float(normal[0]), float(normal[2]))) + 360.0) % 360.0
    )
    return centroid, normal, az, incl


def _angle_between(n1, n2) -> float:
    d = float(np.clip(abs(float(np.dot(n1, n2))), 0.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def _lift_polygon_y(poly: Polygon, centroid, normal) -> list[list[float]]:
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
    if abs(ny) < 1e-6:
        return []
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.is_empty:
        return []

    def y_fn(x, z):
        return cy - (nx * (x - cx) + nz * (z - cz)) / ny

    return [
        [float(x), float(y_fn(x, z)), float(z)] for x, z in poly.exterior.coords[:-1]
    ]


def _collect_raw_planes(building: dict) -> list[dict]:
    out: list[dict] = []
    for ri, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        for pi, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            poly = _xz_polygon(corners)
            if poly is None:
                continue
            fit = _fit_plane(corners)
            if fit is None:
                continue
            centroid, normal, az, incl = fit
            if incl > MAX_INCLINATION_DEG:
                continue
            out.append(
                {
                    "story": story,
                    "room_index": ri,
                    "plane_index": pi,
                    "corners": corners,
                    "xz": poly,
                    "centroid": centroid,
                    "normal": normal,
                    "azimuth": az,
                    "inclination": incl,
                }
            )
    return out


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

    rows: list[dict] = []
    disagreement_buildings: dict[str, list[dict]] = {}

    for building in buildings:
        uuid = building.get("uuid")
        if not uuid:
            continue
        planes = _collect_raw_planes(building)
        if len(planes) < 2:
            continue

        per_building: list[dict] = []
        pair_idx = 0
        n = len(planes)
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = planes[i], planes[j]
                if pi["story"] != pj["story"]:
                    continue
                try:
                    inter = pi["xz"].intersection(pj["xz"])
                except Exception:
                    continue
                if inter.is_empty or inter.area < MIN_OVERLAP_AREA_M2:
                    continue
                angle = _angle_between(pi["normal"], pj["normal"])
                if angle < MIN_DISAGREEMENT_DEG:
                    continue
                hi = pi if float(pi["centroid"][1]) > float(pj["centroid"][1]) else pj
                lifted = _lift_polygon_y(inter, hi["centroid"], hi["normal"])
                if not lifted:
                    continue

                pair_id_i = f"{pi['story']}:{pi['room_index']}:{pi['plane_index']}"
                pair_id_j = f"{pj['story']}:{pj['room_index']}:{pj['plane_index']}"
                element_id = f"{uuid}::raw-disagreement::{pair_idx}"
                rows.append(
                    {
                        "uuid": uuid,
                        "pair_index": pair_idx,
                        "story": pi["story"],
                        "plane_i": pair_id_i,
                        "plane_j": pair_id_j,
                        "same_room": int(pi["room_index"] == pj["room_index"]),
                        "overlap_area_m2": round(float(inter.area), 3),
                        "angle_deg": round(angle, 2),
                        "az_i": round(pi["azimuth"], 1),
                        "incl_i": round(pi["inclination"], 1),
                        "az_j": round(pj["azimuth"], 1),
                        "incl_j": round(pj["inclination"], 1),
                        "element_id": element_id,
                    }
                )
                per_building.append(
                    {
                        "pair_index": pair_idx,
                        "element_id": element_id,
                        "plane_i_element_id": f"{uuid}::ceiling-raw::{pair_id_i}",
                        "plane_j_element_id": f"{uuid}::ceiling-raw::{pair_id_j}",
                        "story": pi["story"],
                        "same_room": bool(pi["room_index"] == pj["room_index"]),
                        "corners": lifted,
                        "overlap_area_m2": round(float(inter.area), 3),
                        "angle_deg": round(angle, 2),
                        "azimuth_i": round(pi["azimuth"], 1),
                        "inclination_i": round(pi["inclination"], 1),
                        "azimuth_j": round(pj["azimuth"], 1),
                        "inclination_j": round(pj["inclination"], 1),
                    }
                )
                pair_idx += 1
        if per_building:
            disagreement_buildings[uuid] = per_building

    _write_csv(OUT_DIR / "per_pair.csv", rows)
    (OUT_DIR / "disagreement_polygons.json").write_text(
        json.dumps({"buildings": disagreement_buildings}, indent=2)
    )

    def _bucket(value: float, edges: list[float]) -> str:
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                return f"{edges[i]:g}-{edges[i + 1]:g}"
        return f">={edges[-1]:g}"

    angle_edges = [20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.001]
    area_edges = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
    angle_hist: Counter = Counter()
    area_hist: Counter = Counter()
    for r in rows:
        angle_hist[_bucket(r["angle_deg"], angle_edges)] += 1
        area_hist[_bucket(r["overlap_area_m2"], area_edges)] += 1

    summary = {
        "thresholds": {
            "min_plane_area_m2": MIN_PLANE_AREA_M2,
            "min_overlap_area_m2": MIN_OVERLAP_AREA_M2,
            "max_inclination_deg": MAX_INCLINATION_DEG,
            "min_disagreement_deg": MIN_DISAGREEMENT_DEG,
        },
        "n_buildings_scanned": len(buildings),
        "n_buildings_with_disagreement": len(disagreement_buildings),
        "n_pairs": len(rows),
        "n_same_room_pairs": sum(1 for r in rows if r["same_room"] == 1),
        "angle_deg_hist": dict(angle_hist),
        "overlap_area_m2_hist": dict(area_hist),
        "top_buildings_by_pair_count": sorted(
            [(uuid, len(v)) for uuid, v in disagreement_buildings.items()],
            key=lambda x: -x[1],
        )[:15],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Pairs flagged: {len(rows)}")
    print(
        f"Buildings with disagreement: {len(disagreement_buildings)} / {len(buildings)}"
    )
    print(f"Same-room pairs: {summary['n_same_room_pairs']}")
    print(f"Angle histogram: {dict(angle_hist)}")
    print(f"Area histogram: {dict(area_hist)}")


if __name__ == "__main__":
    main()
