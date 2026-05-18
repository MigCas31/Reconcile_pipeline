"""Correlate raw-ceiling pipeline disagreement against room attributes.

Consumes ``reports/raw_ceiling_pipeline_gap/per_plane.csv`` (the per-plane
classification produced by ``audit_raw_ceiling_pipeline_gap.py``) and pulls
room-level attributes from ``reconcile/buildings_3d.json`` plus dormer
annotations from ``reconcile/roof_algorithms_py_results.json``.

For each room with scored raw planes, computes:

* ``n_planes`` and ``n_disagree`` where "disagree" means any of
  ``oblique_wrong_orientation``, ``flat_covered_by_oblique``,
  ``oblique_covered_by_flat``.
* Structural features: story, n_walls_computed, n_doors, n_windows,
  n_openings, n_storages, floor_area_m2, n_floor_corners,
  raw_ceiling_source, ceiling_type.
* Raw-plane features: incl_p50, incl_range, incl_std, azimuth_range,
  n_oblique, n_near_vertical.
* Boolean flags: has_dormer, has_simple_slant_ceiling_type, exposed_room.

Writes ``reports/raw_ceiling_disagreement_features/per_room.csv`` and a
``summary.json`` that bins rooms by each feature and prints the disagreement
rate per bin. Highlights the features with the largest between-bin spread.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
PER_PLANE_CSV = REPO / "reports" / "raw_ceiling_pipeline_gap" / "per_plane.csv"
OUT_DIR = REPO / "reports" / "raw_ceiling_disagreement_features"

DISAGREE_CLASSES = {
    "oblique_wrong_orientation",
    "flat_covered_by_oblique",
    "oblique_covered_by_flat",
}
OBLIQUE_MIN_INCL_DEG = 5.0
NEAR_VERTICAL_DEG = 80.0
MIN_ROOM_PLANES = 1


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


def _incl_az(normal):
    ny = max(-1.0, min(1.0, float(normal[1])))
    incl = math.degrees(math.acos(abs(ny)))
    nx, nz = float(normal[0]), float(normal[2])
    if abs(nx) < 1e-9 and abs(nz) < 1e-9:
        az = None
    else:
        az = math.degrees(math.atan2(-nx, -nz)) % 360.0
    return incl, az


def _angle_spread(values):
    if not values:
        return 0.0
    # circular range
    vals = sorted(values)
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)] + [
        360.0 - vals[-1] + vals[0]
    ]
    return 360.0 - max(gaps)


def _floor_area(polygon) -> float:
    if not polygon or len(polygon) < 3:
        return 0.0
    pts = [(p[0], p[2]) for p in polygon]
    return float(Polygon(pts).area)


def _load_dormer_rooms(roof_results, buildings):
    out = set()
    for uuid, r in roof_results.items():
        dms = r.get("dormers") or []
        if not dms:
            continue
        bldg = buildings.get(uuid)
        if not bldg:
            continue
        for d in dms:
            ri = d.get("room_index")
            if ri is None or ri >= len(bldg.get("rooms") or []):
                continue
            st = int(bldg["rooms"][ri].get("story", 0))
            out.add((uuid, st, ri))
    return out


def _exposed_keys(roof_result):
    keys = set()
    for e in (roof_result.get("ceiling") or {}).get("exposed_rooms") or []:
        ri, st = e.get("room_index"), e.get("story")
        if ri is None or st is None:
            continue
        keys.add((int(st), int(ri)))
    return keys


def _load_per_plane_rows():
    by_room = defaultdict(list)
    with PER_PLANE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["uuid"], int(row["story"]), int(row["room_index"]))
            by_room[key].append(row)
    return by_room


def _build_room_rows():
    with BUILDINGS_PATH.open() as f:
        buildings_list = json.load(f)
    buildings = {b["uuid"]: b for b in buildings_list}
    with ROOF_RESULTS_PATH.open() as f:
        roof_results = json.load(f)
    dormer_rooms = _load_dormer_rooms(roof_results, buildings)
    per_plane_by_room = _load_per_plane_rows()

    rows: list[dict] = []
    for uuid, bldg in buildings.items():
        roof_result = roof_results.get(uuid, {})
        exposed = _exposed_keys(roof_result)
        for ri, room in enumerate(bldg.get("rooms") or []):
            story = int(room.get("story", 0))
            key = (uuid, story, ri)
            if key not in per_plane_by_room:
                continue
            plane_rows = per_plane_by_room[key]
            if len(plane_rows) < MIN_ROOM_PLANES:
                continue

            # room structural features
            n_walls = len(room.get("walls_computed") or [])
            n_doors = len(room.get("doors") or [])
            n_windows = len(room.get("windows") or [])
            n_openings = len(room.get("openings") or [])
            n_storages = len(room.get("storages") or [])
            floor_area = _floor_area(room.get("floor_polygon"))
            n_floor_corners = len(room.get("floor_polygon") or [])
            ceiling_type = room.get("ceiling_type") or "none"
            raw_source = room.get("raw_ceiling_source") or "none"

            # raw-plane features
            incls: list[float] = []
            azs: list[float] = []
            n_near_vertical = 0
            n_oblique = 0
            areas: list[float] = []
            for plane in room.get("raw_ceiling_planes") or []:
                fit = _fit_plane(plane.get("corners") or [])
                if fit is None:
                    continue
                _, normal = fit
                incl, az = _incl_az(normal)
                incls.append(incl)
                if az is not None and incl >= OBLIQUE_MIN_INCL_DEG:
                    azs.append(az)
                if incl >= NEAR_VERTICAL_DEG:
                    n_near_vertical += 1
                if incl >= OBLIQUE_MIN_INCL_DEG:
                    n_oblique += 1
                corners = plane.get("corners") or []
                if len(corners) >= 3:
                    pts_xz = [(c[0], c[2]) for c in corners]
                    areas.append(float(Polygon(pts_xz).area))

            incl_p50 = float(np.median(incls)) if incls else 0.0
            incl_range = (max(incls) - min(incls)) if incls else 0.0
            incl_std = float(np.std(incls)) if incls else 0.0
            azimuth_range = _angle_spread(azs) if len(azs) >= 2 else 0.0
            raw_area = float(sum(areas))

            n_planes_scored = len(plane_rows)
            n_disagree = sum(
                1 for r in plane_rows if r["classification"] in DISAGREE_CLASSES
            )
            disagreement_rate = n_disagree / n_planes_scored

            rows.append(
                {
                    "uuid": uuid,
                    "story": story,
                    "room_index": ri,
                    "has_dormer": key in dormer_rooms,
                    "exposed": (story, ri) in exposed,
                    "n_planes_scored": n_planes_scored,
                    "n_disagree": n_disagree,
                    "disagreement_rate": round(disagreement_rate, 3),
                    "ceiling_type": ceiling_type,
                    "raw_ceiling_source": raw_source,
                    "n_walls_computed": n_walls,
                    "n_doors": n_doors,
                    "n_windows": n_windows,
                    "n_openings": n_openings,
                    "n_storages": n_storages,
                    "n_floor_corners": n_floor_corners,
                    "floor_area_m2": round(floor_area, 2),
                    "raw_area_xz_m2": round(raw_area, 2),
                    "raw_area_vs_floor": round(raw_area / floor_area, 3)
                    if floor_area > 0
                    else 0.0,
                    "n_raw_planes_room": len(room.get("raw_ceiling_planes") or []),
                    "n_raw_oblique": n_oblique,
                    "n_raw_near_vertical": n_near_vertical,
                    "incl_p50_deg": round(incl_p50, 2),
                    "incl_range_deg": round(incl_range, 2),
                    "incl_std_deg": round(incl_std, 2),
                    "azimuth_range_deg": round(azimuth_range, 2),
                }
            )
    return rows


# --- binning helpers ---------------------------------------------------------


def _bucket_numeric(value, edges, labels):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return labels[i]
    return labels[-1]


def _weighted_rate(rows: list[dict]) -> tuple[int, int, float]:
    n = sum(r["n_planes_scored"] for r in rows)
    d = sum(r["n_disagree"] for r in rows)
    return d, n, (d / n if n else 0.0)


def _analyze_bins(rows: list[dict]) -> dict[str, Any]:
    corpus_d, corpus_n, corpus_rate = _weighted_rate(rows)

    bool_features = ["has_dormer", "exposed"]
    cat_features = ["ceiling_type", "raw_ceiling_source"]
    num_specs = {
        "story": ([0, 1, 2, 3, 99], ["0", "1", "2", "3+"]),
        "n_walls_computed": ([0, 4, 7, 12, 999], ["<=3", "4-6", "7-11", "12+"]),
        "n_windows": ([0, 1, 3, 6, 999], ["0", "1-2", "3-5", "6+"]),
        "n_doors": ([0, 1, 2, 999], ["0", "1", "2+"]),
        "n_openings": ([0, 1, 999], ["0", "1+"]),
        "n_storages": ([0, 1, 999], ["0", "1+"]),
        "floor_area_m2": ([0, 10, 20, 35, 999], ["<10", "10-20", "20-35", "35+"]),
        "raw_area_vs_floor": (
            [0, 0.5, 0.9, 1.1, 1.5, 999],
            ["<0.5", "0.5-0.9", "0.9-1.1", "1.1-1.5", "1.5+"],
        ),
        "n_raw_planes_room": ([0, 2, 4, 7, 999], ["1", "2-3", "4-6", "7+"]),
        "n_raw_oblique": ([0, 1, 3, 6, 999], ["0", "1-2", "3-5", "6+"]),
        "n_raw_near_vertical": ([0, 1, 3, 999], ["0", "1-2", "3+"]),
        "incl_p50_deg": ([0, 1, 10, 30, 180], ["<1", "1-10", "10-30", "30+"]),
        "incl_range_deg": (
            [0, 5, 20, 45, 90, 180],
            ["<5", "5-20", "20-45", "45-90", "90+"],
        ),
        "incl_std_deg": ([0, 2, 10, 25, 180], ["<2", "2-10", "10-25", "25+"]),
        "azimuth_range_deg": (
            [0, 30, 90, 180, 361],
            ["<30", "30-90", "90-180", "180+"],
        ),
        "n_floor_corners": ([0, 5, 7, 10, 99], ["<=4", "5-6", "7-9", "10+"]),
    }

    report: dict[str, Any] = {
        "corpus": {
            "n_planes": corpus_n,
            "n_disagree": corpus_d,
            "rate": round(corpus_rate, 4),
        },
        "bool_features": {},
        "categorical_features": {},
        "numeric_features": {},
    }

    def _rows_for(filter_fn):
        return [r for r in rows if filter_fn(r)]

    for feat in bool_features:
        pos = _rows_for(lambda r, f=feat: r[f])
        neg = _rows_for(lambda r, f=feat: not r[f])
        d_p, n_p, rate_p = _weighted_rate(pos)
        d_n, n_n, rate_n = _weighted_rate(neg)
        report["bool_features"][feat] = {
            "true": {
                "rooms": len(pos),
                "planes": n_p,
                "disagree": d_p,
                "rate": round(rate_p, 4),
            },
            "false": {
                "rooms": len(neg),
                "planes": n_n,
                "disagree": d_n,
                "rate": round(rate_n, 4),
            },
            "relative_risk": round(rate_p / rate_n, 2) if rate_n else None,
        }

    for feat in cat_features:
        bins = defaultdict(list)
        for r in rows:
            bins[r[feat]].append(r)
        cat_report = {}
        for label, bucket in bins.items():
            d, n, rate = _weighted_rate(bucket)
            cat_report[label] = {
                "rooms": len(bucket),
                "planes": n,
                "disagree": d,
                "rate": round(rate, 4),
            }
        report["categorical_features"][feat] = cat_report

    for feat, (edges, labels) in num_specs.items():
        bins = defaultdict(list)
        for r in rows:
            b = _bucket_numeric(r[feat], edges, labels)
            bins[b].append(r)
        num_report = {}
        for label in labels:
            bucket = bins.get(label, [])
            d, n, rate = _weighted_rate(bucket)
            num_report[label] = {
                "rooms": len(bucket),
                "planes": n,
                "disagree": d,
                "rate": round(rate, 4),
            }
        rates = [v["rate"] for v in num_report.values() if v["planes"] >= 20]
        num_report["_spread_max_minus_min"] = (
            round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0.0
        )
        report["numeric_features"][feat] = num_report

    # Rank numeric features by spread for quick reading.
    ranked = sorted(
        report["numeric_features"].items(),
        key=lambda kv: kv[1].get("_spread_max_minus_min", 0.0),
        reverse=True,
    )
    report["numeric_features_ranked_by_spread"] = [k for k, _ in ranked]
    return report


def _print_table(report: dict[str, Any]) -> None:
    corpus = report["corpus"]
    print(f"Corpus: {corpus['n_planes']} planes, rate={corpus['rate']:.3f}\n")

    print("Bool features:")
    for feat, data in report["bool_features"].items():
        t, f = data["true"], data["false"]
        rr = data["relative_risk"]
        print(
            f"  {feat:<18}  true:  rooms={t['rooms']:>4}  planes={t['planes']:>4}  "
            f"rate={t['rate']:.3f}"
            f"   false: rooms={f['rooms']:>4}  planes={f['planes']:>4}  "
            f"rate={f['rate']:.3f}"
            f"   RR={rr}"
        )
    print()
    print("Categorical features:")
    for feat, data in report["categorical_features"].items():
        print(f"  {feat}:")
        for label, info in sorted(data.items(), key=lambda kv: -kv[1]["planes"]):
            print(
                f"    {label:<20}  rooms={info['rooms']:>4}  planes={info['planes']:>4}"
                f"  disagree={info['disagree']:>4}  rate={info['rate']:.3f}"
            )
    print()
    print("Numeric features (ranked by bin-spread):")
    for feat in report["numeric_features_ranked_by_spread"]:
        data = report["numeric_features"][feat]
        spread = data.get("_spread_max_minus_min", 0.0)
        print(f"  {feat} (spread={spread:.3f}):")
        for label, info in data.items():
            if label.startswith("_"):
                continue
            print(
                f"    {label:<10}  rooms={info['rooms']:>4}  planes={info['planes']:>4}"
                f"  disagree={info['disagree']:>4}  rate={info['rate']:.3f}"
            )


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
    rows = _build_room_rows()
    _write_csv(OUT_DIR / "per_room.csv", rows)
    report = _analyze_bins(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    _print_table(report)


if __name__ == "__main__":
    main()
