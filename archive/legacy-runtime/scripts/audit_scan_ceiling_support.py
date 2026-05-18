"""Audit how well raw scan ceilings support roof/ceiling plane candidates.

Read-only: runs no pipeline, edits nothing. Consumes the existing
``reconcile/buildings_3d.json`` (scan-derived geometry, including
``raw_ceiling_planes`` per room) and ``reconcile/roof_algorithms_py_results.json``
(roof pipeline output, including the pre-selection ``ceiling.planes`` list and
the committed ``roof_surfaces.oblique`` / ``roof_surfaces.flat`` lists).

For each roof / ceiling plane surface, compares its XZ polygon to the union of
XZ polygons of the raw scanned ceiling planes in the rooms it covers on the
same story, and reports:

* XZ coverage fraction of the surface by raw ceilings.
* Vertical residual (median / p95 |dy|) between the surface's plane-y and raw
  ceiling centroid y-values where the centroid falls inside the surface.
* Ridge / eave agreement: does the surface's y-range overlap raw ceiling
  y-range in the overlapping XZ region.
* Normal agreement: SVD-fitted raw ceiling plane normal vs. surface normal.

Classification per surface: ``strong`` (scan ceilings agree), ``weak`` (scan
ceilings disagree beyond tolerance), ``none`` (no scan coverage in any covered
room — can't judge).

Outputs ``reports/scan_ceiling_support.json`` (per-surface rows with
viewer-clickable element IDs for both the surface and the matched
``ceiling-raw`` planes) and ``reports/scan_ceiling_support_summary.csv``
(per-building counts). Prints a stdout table with the top-20 buildings by
``weak`` committed-surface count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
REPORTS_DIR = REPO / "reports"

# Classification thresholds — deliberately generous so scan noise doesn't flag
# genuine agreement as disagreement.
WEAK_ABS_DY_M = 0.5
WEAK_NORMAL_DOT = 0.80
MIN_XZ_COVERAGE_FOR_JUDGEMENT = 0.10


@dataclass
class RawCeiling:
    room_index: int
    story: int
    plane_index: int
    poly_xz: Polygon
    centroid_xz: tuple[float, float]
    centroid_y: float
    y_min: float
    y_max: float
    corners: list[list[float]]


def _xz_polygon(corners: list[list[float]]) -> Polygon | None:
    pts = [(c[0], c[2]) for c in corners if len(c) >= 3]
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _fit_plane_svd(corners: list[list[float]]):
    pts = np.array(corners, dtype=float)
    if len(pts) < 3:
        return None, None
    centroid = pts.mean(axis=0)
    diffs = pts - centroid
    try:
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    return centroid, normal


def _y_at_from_plane_eq(
    centroid: np.ndarray, normal: np.ndarray
) -> Callable[[float, float], float]:
    if abs(normal[1]) < 1e-6:
        c1 = float(centroid[1])
        return lambda x, z, _c1=c1: _c1
    c0, c1, c2 = float(centroid[0]), float(centroid[1]), float(centroid[2])
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    return lambda x, z: c1 - (nx * (x - c0) + nz * (z - c2)) / ny


def _candidate_polygon(plane: dict) -> Polygon | None:
    rx, rz = float(plane["ridgeX"]), float(plane["ridgeZ"])
    sx, sz = float(plane["slopeX"]), float(plane["slopeZ"])
    ref = plane["ref"]
    ref_x, ref_z = float(ref["x"]), float(ref["z"])
    bounds = [
        (plane["minRidge"], plane["minSlope"]),
        (plane["maxRidge"], plane["minSlope"]),
        (plane["maxRidge"], plane["maxSlope"]),
        (plane["minRidge"], plane["maxSlope"]),
    ]
    corners_xz = []
    for r_, s_ in bounds:
        x = ref_x + r_ * rx + s_ * sx
        z = ref_z + r_ * rz + s_ * sz
        corners_xz.append((x, z))
    poly = Polygon(corners_xz)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _candidate_y_fn(plane: dict) -> Callable[[float, float], float]:
    n = plane["n"]
    ref = plane["ref"]
    nx, ny, nz = float(n["x"]), float(n["y"]), float(n["z"])
    rx, ry, rz = float(ref["x"]), float(ref["y"]), float(ref["z"])
    if abs(ny) < 1e-6:
        return lambda x, z, _ry=ry: _ry
    return lambda x, z: ry - (nx * (x - rx) + nz * (z - rz)) / ny


def _candidate_normal(plane: dict) -> np.ndarray:
    n = plane["n"]
    vec = np.array([float(n["x"]), float(n["y"]), float(n["z"])])
    if vec[1] < 0:
        vec = -vec
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return vec / norm


def _exposed_room_keys(roof_result: dict) -> set[tuple[int, int]] | None:
    """Return the set of (story, room_index) pairs that are roof-exposed.

    These are the rooms whose ceiling scan actually sees sky — ceilings in
    rooms with another room above them are interior caps between stories and
    shouldn't be compared against roof surface candidates. The roof pipeline
    already produces this set as ``ceiling.exposed_rooms`` (every entry there
    has ``is_roof_candidate: True``).

    Returns ``None`` if the roof result has no exposed_rooms block — the
    caller should then fall back to accepting all raw ceilings.
    """
    exposed = (roof_result.get("ceiling") or {}).get("exposed_rooms")
    if not exposed:
        return None
    keys: set[tuple[int, int]] = set()
    for entry in exposed:
        room_index = entry.get("room_index")
        story = entry.get("story")
        if room_index is None or story is None:
            continue
        keys.add((int(story), int(room_index)))
    return keys


def _collect_raw_ceilings(
    building: dict, allowed_keys: set[tuple[int, int]] | None
) -> list[RawCeiling]:
    out: list[RawCeiling] = []
    for room_index, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        if allowed_keys is not None and (story, room_index) not in allowed_keys:
            continue
        for plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) < 3:
                continue
            poly = _xz_polygon(corners)
            if poly is None:
                continue
            y_vals = [float(c[1]) for c in corners if len(c) >= 3]
            if not y_vals:
                continue
            cx, cz = poly.centroid.x, poly.centroid.y
            out.append(
                RawCeiling(
                    room_index=room_index,
                    story=story,
                    plane_index=plane_index,
                    poly_xz=poly,
                    centroid_xz=(float(cx), float(cz)),
                    centroid_y=float(np.mean(y_vals)),
                    y_min=float(min(y_vals)),
                    y_max=float(max(y_vals)),
                    corners=corners,
                )
            )
    return out


def _score_surface(
    surface_poly: Polygon,
    surface_y_at: Callable[[float, float], float],
    surface_normal: np.ndarray | None,
    raw_candidates: Iterable[RawCeiling],
) -> dict[str, Any]:
    surface_area = float(surface_poly.area)
    matches: list[RawCeiling] = []
    for rc in raw_candidates:
        cx, cz = rc.centroid_xz
        if surface_poly.contains(Point(cx, cz)) or surface_poly.intersects(rc.poly_xz):
            matches.append(rc)

    if not matches or surface_area < 1e-6:
        return {
            "match_count": 0,
            "matches": [],
            "xz_coverage": 0.0,
            "median_dy": None,
            "p95_abs_dy": None,
            "normal_dot_median": None,
            "raw_y_min": None,
            "raw_y_max": None,
            "surface_y_min_at_match": None,
            "surface_y_max_at_match": None,
        }

    union = unary_union([rc.poly_xz for rc in matches])
    try:
        covered = float(surface_poly.intersection(union).area)
    except Exception:
        covered = 0.0
    xz_coverage = covered / surface_area if surface_area > 0 else 0.0

    dys: list[float] = []
    normal_dots: list[float] = []
    sampled_surface_ys: list[float] = []
    raw_ys: list[float] = []
    for rc in matches:
        cx, cz = rc.centroid_xz
        if not surface_poly.contains(Point(cx, cz)):
            continue
        try:
            y_surf = float(surface_y_at(cx, cz))
        except Exception:
            continue
        dys.append(y_surf - rc.centroid_y)
        sampled_surface_ys.append(y_surf)
        raw_ys.append(rc.centroid_y)
        _, rc_normal = _fit_plane_svd(rc.corners)
        if rc_normal is not None and surface_normal is not None:
            snorm = float(np.linalg.norm(surface_normal))
            rnorm = float(np.linalg.norm(rc_normal))
            if snorm > 1e-9 and rnorm > 1e-9:
                dot = abs(float(np.dot(rc_normal / rnorm, surface_normal / snorm)))
                normal_dots.append(min(1.0, dot))

    median_dy = float(np.median(dys)) if dys else None
    p95_abs_dy = float(np.percentile(np.abs(dys), 95)) if dys else None
    normal_dot_median = float(np.median(normal_dots)) if normal_dots else None

    return {
        "match_count": len(matches),
        "matches": [(rc.story, rc.room_index, rc.plane_index) for rc in matches],
        "xz_coverage": xz_coverage,
        "median_dy": median_dy,
        "p95_abs_dy": p95_abs_dy,
        "normal_dot_median": normal_dot_median,
        "raw_y_min": float(min(raw_ys)) if raw_ys else None,
        "raw_y_max": float(max(raw_ys)) if raw_ys else None,
        "surface_y_min_at_match": float(min(sampled_surface_ys))
        if sampled_surface_ys
        else None,
        "surface_y_max_at_match": float(max(sampled_surface_ys))
        if sampled_surface_ys
        else None,
    }


def _classify(scores: dict[str, Any]) -> str:
    if (
        scores["xz_coverage"] < MIN_XZ_COVERAGE_FOR_JUDGEMENT
        or scores["median_dy"] is None
    ):
        return "none"
    if abs(scores["median_dy"]) > WEAK_ABS_DY_M:
        return "weak"
    nd = scores["normal_dot_median"]
    if nd is not None and nd < WEAK_NORMAL_DOT:
        return "weak"
    return "strong"


def _emit_raw_ids(uuid: str, matches: list[tuple[int, int, int]]) -> list[str]:
    return [
        f"{uuid}::ceiling-raw::{story}:{room}:{plane}"
        for (story, room, plane) in matches
    ]


def _audit_committed_oblique(
    uuid: str, ob: dict, idx: int, raw_by_story: dict[int, list[RawCeiling]]
) -> dict | None:
    corners = ob.get("corners") or []
    poly = _xz_polygon(corners)
    if poly is None:
        return None
    centroid, normal = _fit_plane_svd(corners)
    if centroid is None or normal is None:
        return None
    y_at = _y_at_from_plane_eq(centroid, normal)
    story = ob.get("dominant_story")
    story_raw = raw_by_story.get(int(story), []) if story is not None else []
    scores = _score_surface(poly, y_at, normal, story_raw)
    return {
        "uuid": uuid,
        "group": "committed_oblique",
        "element_id": f"{uuid}::roof-oblique::oblique:{idx}",
        "story": story,
        "avg_incl": (ob.get("cluster") or {}).get("avgIncl"),
        "avg_azimuth": (ob.get("cluster") or {}).get("avgAzimuth"),
        "surface_area_xz": float(poly.area),
        "match_count": scores["match_count"],
        "xz_coverage": scores["xz_coverage"],
        "median_dy": scores["median_dy"],
        "p95_abs_dy": scores["p95_abs_dy"],
        "normal_dot_median": scores["normal_dot_median"],
        "raw_y_min": scores["raw_y_min"],
        "raw_y_max": scores["raw_y_max"],
        "surface_y_min_at_match": scores["surface_y_min_at_match"],
        "surface_y_max_at_match": scores["surface_y_max_at_match"],
        "ceiling_raw_ids": _emit_raw_ids(uuid, scores["matches"]),
        "classification": _classify(scores),
    }


def _audit_committed_flat(
    uuid: str, fl: dict, idx: int, raw_by_story: dict[int, list[RawCeiling]]
) -> dict | None:
    corners = fl.get("corners") or []
    poly = _xz_polygon(corners)
    if poly is None:
        return None
    ys = [float(c[1]) for c in corners if len(c) >= 3]
    if not ys:
        return None
    y_const = float(np.mean(ys))
    normal = np.array([0.0, 1.0, 0.0])

    def y_at(_x: float, _z: float, _y=y_const) -> float:
        return _y

    story = fl.get("dominant_story")
    story_raw = raw_by_story.get(int(story), []) if story is not None else []
    scores = _score_surface(poly, y_at, normal, story_raw)
    return {
        "uuid": uuid,
        "group": "committed_flat",
        "element_id": f"{uuid}::roof-flat::flat:{idx}",
        "story": story,
        "avg_incl": 0.0,
        "avg_azimuth": None,
        "surface_area_xz": float(poly.area),
        "match_count": scores["match_count"],
        "xz_coverage": scores["xz_coverage"],
        "median_dy": scores["median_dy"],
        "p95_abs_dy": scores["p95_abs_dy"],
        "normal_dot_median": scores["normal_dot_median"],
        "raw_y_min": scores["raw_y_min"],
        "raw_y_max": scores["raw_y_max"],
        "surface_y_min_at_match": scores["surface_y_min_at_match"],
        "surface_y_max_at_match": scores["surface_y_max_at_match"],
        "ceiling_raw_ids": _emit_raw_ids(uuid, scores["matches"]),
        "classification": _classify(scores),
    }


def _audit_candidate_plane(
    uuid: str, plane: dict, idx: int, raw_by_story: dict[int, list[RawCeiling]]
) -> dict | None:
    poly = _candidate_polygon(plane)
    if poly is None:
        return None
    y_at = _candidate_y_fn(plane)
    normal = _candidate_normal(plane)
    story = plane.get("dominantStory")
    story_raw = raw_by_story.get(int(story), []) if story is not None else []
    scores = _score_surface(poly, y_at, normal, story_raw)
    cl = plane.get("cl") or {}
    return {
        "uuid": uuid,
        "group": "candidate_oblique",
        "element_id": f"{uuid}::ceiling-oblique::ceiling-oblique:{idx}",
        "story": story,
        "avg_incl": cl.get("avgIncl"),
        "avg_azimuth": cl.get("avgAzimuth"),
        "surface_area_xz": float(poly.area),
        "match_count": scores["match_count"],
        "xz_coverage": scores["xz_coverage"],
        "median_dy": scores["median_dy"],
        "p95_abs_dy": scores["p95_abs_dy"],
        "normal_dot_median": scores["normal_dot_median"],
        "raw_y_min": scores["raw_y_min"],
        "raw_y_max": scores["raw_y_max"],
        "surface_y_min_at_match": scores["surface_y_min_at_match"],
        "surface_y_max_at_match": scores["surface_y_max_at_match"],
        "ceiling_raw_ids": _emit_raw_ids(uuid, scores["matches"]),
        "classification": _classify(scores),
    }


def _audit_building(
    building: dict, roof_result: dict, exposed_only: bool = True
) -> list[dict]:
    uuid = building["uuid"]
    allowed_keys = _exposed_room_keys(roof_result) if exposed_only else None
    raw = _collect_raw_ceilings(building, allowed_keys)
    raw_by_story: dict[int, list[RawCeiling]] = {}
    for rc in raw:
        raw_by_story.setdefault(rc.story, []).append(rc)

    rows: list[dict] = []

    roof_surfaces = roof_result.get("roof_surfaces") or {}
    for idx, ob in enumerate(roof_surfaces.get("oblique") or []):
        row = _audit_committed_oblique(uuid, ob, idx, raw_by_story)
        if row is not None:
            rows.append(row)
    for idx, fl in enumerate(roof_surfaces.get("flat") or []):
        row = _audit_committed_flat(uuid, fl, idx, raw_by_story)
        if row is not None:
            rows.append(row)

    ceiling = roof_result.get("ceiling") or {}
    for idx, plane in enumerate(ceiling.get("planes") or []):
        row = _audit_candidate_plane(uuid, plane, idx, raw_by_story)
        if row is not None:
            rows.append(row)

    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    by_uuid: dict[str, dict[str, Counter]] = {}
    for row in rows:
        bucket = by_uuid.setdefault(row["uuid"], {})
        group_counts = bucket.setdefault(row["group"], Counter())
        group_counts[row["classification"]] += 1

    summary: list[dict] = []
    for uuid, groups in by_uuid.items():
        ob = groups.get("committed_oblique", Counter())
        fl = groups.get("committed_flat", Counter())
        cand = groups.get("candidate_oblique", Counter())
        summary.append(
            {
                "uuid": uuid,
                "committed_oblique_strong": ob.get("strong", 0),
                "committed_oblique_weak": ob.get("weak", 0),
                "committed_oblique_none": ob.get("none", 0),
                "committed_flat_strong": fl.get("strong", 0),
                "committed_flat_weak": fl.get("weak", 0),
                "committed_flat_none": fl.get("none", 0),
                "candidate_oblique_strong": cand.get("strong", 0),
                "candidate_oblique_weak": cand.get("weak", 0),
                "candidate_oblique_none": cand.get("none", 0),
                "candidate_minus_committed_weak": cand.get("weak", 0)
                - ob.get("weak", 0),
            }
        )
    return summary


def _print_top_table(summary: list[dict], n: int = 20) -> None:
    ranked = sorted(summary, key=lambda s: s["committed_oblique_weak"], reverse=True)
    print("\nTop buildings by committed_oblique weak count:")
    print(
        f"{'uuid':<40} {'ob_w':>5} {'ob_s':>5} {'ob_n':>5} {'cand_w':>7} {'cand_s':>7} "
        f"{'cand_n':>7}"
    )
    for row in ranked[:n]:
        if row["committed_oblique_weak"] == 0 and row["committed_oblique_strong"] == 0:
            continue
        print(
            f"{row['uuid']:<40} "
            f"{row['committed_oblique_weak']:>5} "
            f"{row['committed_oblique_strong']:>5} "
            f"{row['committed_oblique_none']:>5} "
            f"{row['candidate_oblique_weak']:>7} "
            f"{row['candidate_oblique_strong']:>7} "
            f"{row['candidate_oblique_none']:>7}"
        )


def _write_reports(
    rows: list[dict],
    summary: list[dict],
    out_dir: Path,
    exposed_only: bool,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scan_ceiling_support.json"
    csv_path = out_dir / "scan_ceiling_support_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "thresholds": {
                    "weak_abs_dy_m": WEAK_ABS_DY_M,
                    "weak_normal_dot": WEAK_NORMAL_DOT,
                    "min_xz_coverage_for_judgement": MIN_XZ_COVERAGE_FOR_JUDGEMENT,
                },
                "filter": {"exposed_rooms_only": exposed_only},
                "rows": rows,
            },
            handle,
            indent=2,
        )

    if summary:
        fieldnames = list(summary[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary:
                writer.writerow(row)
    else:
        csv_path.write_text("", encoding="utf-8")

    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buildings",
        type=Path,
        default=BUILDINGS_PATH,
        help="Path to buildings_3d.json (default: %(default)s)",
    )
    parser.add_argument(
        "--roof-results",
        type=Path,
        default=ROOF_RESULTS_PATH,
        help="Path to roof_algorithms_py_results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--building",
        type=str,
        default=None,
        help="Restrict to a single building UUID (smoke test mode).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Output directory for JSON+CSV reports (default: %(default)s).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many buildings to print in the stdout table.",
    )
    parser.add_argument(
        "--include-covered",
        action="store_true",
        help=(
            "Include raw ceilings from rooms covered by another room above. "
            "Default: only use raw ceilings from roof-exposed rooms "
            "(ceiling.exposed_rooms) so interior story caps don't create "
            "spurious disagreements."
        ),
    )
    args = parser.parse_args(argv)

    buildings = json.loads(args.buildings.read_text(encoding="utf-8"))
    roof_results = json.loads(args.roof_results.read_text(encoding="utf-8"))

    if args.building is not None:
        buildings = [b for b in buildings if b.get("uuid") == args.building]
        if not buildings:
            print(
                f"No building with uuid {args.building} in {args.buildings}",
                file=sys.stderr,
            )
            return 2

    rows: list[dict] = []
    skipped_no_roof = 0
    for building in buildings:
        uuid = building.get("uuid")
        if not uuid:
            continue
        roof_result = roof_results.get(uuid)
        if roof_result is None:
            skipped_no_roof += 1
            continue
        rows.extend(
            _audit_building(
                building, roof_result, exposed_only=not args.include_covered
            )
        )

    summary = _summarize(rows)
    json_path, csv_path = _write_reports(
        rows, summary, args.out_dir, exposed_only=not args.include_covered
    )

    print(
        f"Audited {len(summary)} buildings ({skipped_no_roof} skipped: no roof "
        f"results)."
    )
    print(f"  rows: {len(rows)}")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    _print_top_table(summary, n=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
