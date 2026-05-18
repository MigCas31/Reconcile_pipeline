"""Complement to ``scripts/audit_scan_ceiling_support.py``.

``scan_ceiling_support`` walks roof surfaces and scores how well each one is
supported by the raw scan ceilings below it (answers "is this surface real?").
This script inverts that: it walks every raw ceiling plane in an exposed room
and asks "is this raw plane covered by a roof/ceiling surface with the right
orientation?".

Outputs per raw plane:

* coverage fraction from committed oblique/flat roof surfaces (separately)
* best-matching surface element id
* normal dot against the best oblique surface
* whether the pipeline's assignment class (oblique/flat) matches the raw
  plane's class (oblique if incl >= 5 deg, flat otherwise)

Per-room roll-up answers H4: does the per-room majority class of raw planes
agree with the pipeline's per-room class?

Reports written to ``reports/raw_ceiling_pipeline_gap/``:

* ``per_plane.csv`` — one row per raw plane with classification bucket
* ``per_room.csv`` — per (story, room_index) majority class vs pipeline class
* ``summary.json`` — corpus cohort counts per bucket plus top outliers
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
OUT_DIR = REPO / "reports" / "raw_ceiling_pipeline_gap"

OBLIQUE_MIN_INCL_DEG = 5.0
MIN_COVERAGE_FOR_JUDGEMENT = 0.10
STRONG_COVERAGE = 0.50
NORMAL_DOT_MATCH = 0.80
NEAR_VERTICAL_EXCLUDE_DEG = 80.0  # skip raw planes that look like walls


@dataclass
class RawPlane:
    uuid: str
    story: int
    room_index: int
    plane_index: int
    corners: list[list[float]]
    poly: Polygon
    centroid: np.ndarray
    normal: np.ndarray
    inclination: float
    area_xz: float

    @property
    def element_id(self) -> str:
        return (
            f"{self.uuid}::ceiling-raw::{self.story}:"
            f"{self.room_index}:{self.plane_index}"
        )

    @property
    def is_oblique(self) -> bool:
        return self.inclination >= OBLIQUE_MIN_INCL_DEG


@dataclass
class SurfaceEntry:
    element_id: str
    kind: str  # "roof-oblique" or "roof-flat"
    story: int
    poly: Polygon
    normal: np.ndarray | None = None
    room_indices: set[int] = field(default_factory=set)


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


def _xz_poly(corners: list[list[float]]) -> Polygon | None:
    pts = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    if not p.is_valid or p.is_empty:
        return None
    return p


def _exposed_keys(roof_result: dict) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for entry in (roof_result.get("ceiling") or {}).get("exposed_rooms") or []:
        ri, st = entry.get("room_index"), entry.get("story")
        if ri is None or st is None:
            continue
        keys.add((int(st), int(ri)))
    return keys


def _collect_raw_planes(
    building: dict, exposed_keys: set[tuple[int, int]]
) -> list[RawPlane]:
    uuid = building["uuid"]
    out: list[RawPlane] = []
    for room_index, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        if exposed_keys and (story, room_index) not in exposed_keys:
            continue
        for plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) < 3:
                continue
            poly = _xz_poly(corners)
            if poly is None:
                continue
            fit = _fit_plane(corners)
            if fit is None:
                continue
            centroid, normal = fit
            incl = _inclination_deg(normal)
            out.append(
                RawPlane(
                    uuid=uuid,
                    story=story,
                    room_index=room_index,
                    plane_index=plane_index,
                    corners=corners,
                    poly=poly,
                    centroid=centroid,
                    normal=normal,
                    inclination=incl,
                    area_xz=float(poly.area),
                )
            )
    return out


def _collect_surfaces(uuid: str, roof_result: dict) -> list[SurfaceEntry]:
    out: list[SurfaceEntry] = []
    roof_surfaces = roof_result.get("roof_surfaces") or {}

    for idx, ob in enumerate(roof_surfaces.get("oblique") or []):
        corners = ob.get("corners") or []
        poly = _xz_poly(corners)
        if poly is None:
            continue
        fit = _fit_plane(corners)
        normal = fit[1] if fit else None
        out.append(
            SurfaceEntry(
                element_id=f"{uuid}::roof-oblique::oblique:{idx}",
                kind="roof-oblique",
                story=int(ob.get("dominant_story", 0)),
                poly=poly,
                normal=normal,
            )
        )

    for idx, fl in enumerate(roof_surfaces.get("flat") or []):
        corners = fl.get("corners") or []
        poly = _xz_poly(corners)
        if poly is None:
            continue
        story = int(fl.get("story", fl.get("dominant_story", 0)))
        ri = fl.get("room_index")
        entry = SurfaceEntry(
            element_id=f"{uuid}::roof-flat::flat:{idx}",
            kind="roof-flat",
            story=story,
            poly=poly,
            normal=np.array([0.0, 1.0, 0.0]),
        )
        if ri is not None:
            entry.room_indices.add(int(ri))
        out.append(entry)

    return out


def _score_plane_against_surfaces(
    plane: RawPlane, surfaces: list[SurfaceEntry]
) -> dict[str, Any]:
    area = plane.area_xz
    if area < 1e-6:
        return {}

    matches_ob: list[tuple[SurfaceEntry, float]] = []
    matches_fl: list[tuple[SurfaceEntry, float]] = []
    for s in surfaces:
        if s.story != plane.story:
            continue
        if not s.poly.intersects(plane.poly):
            continue
        try:
            covered = float(plane.poly.intersection(s.poly).area)
        except Exception:
            continue
        frac = covered / area
        if frac <= 0:
            continue
        if s.kind == "roof-oblique":
            matches_ob.append((s, frac))
        else:
            matches_fl.append((s, frac))

    matches_ob.sort(key=lambda x: x[1], reverse=True)
    matches_fl.sort(key=lambda x: x[1], reverse=True)
    cov_ob = sum(f for _, f in matches_ob)
    cov_fl = sum(f for _, f in matches_fl)
    best_ob = matches_ob[0] if matches_ob else None
    best_fl = matches_fl[0] if matches_fl else None

    normal_dot = None
    if best_ob and best_ob[0].normal is not None:
        nd = float(np.dot(plane.normal, best_ob[0].normal))
        normal_dot = round(abs(max(-1.0, min(1.0, nd))), 3)

    return {
        "coverage_oblique": round(min(cov_ob, 1.0), 3),
        "coverage_flat": round(min(cov_fl, 1.0), 3),
        "best_oblique_id": best_ob[0].element_id if best_ob else "",
        "best_oblique_frac": round(best_ob[1], 3) if best_ob else 0.0,
        "best_flat_id": best_fl[0].element_id if best_fl else "",
        "best_flat_frac": round(best_fl[1], 3) if best_fl else 0.0,
        "normal_dot_best_oblique": normal_dot,
    }


def _classify(plane: RawPlane, scores: dict[str, Any]) -> str:
    if plane.inclination >= NEAR_VERTICAL_EXCLUDE_DEG:
        return "near_vertical_skip"
    cov_ob = scores.get("coverage_oblique", 0.0)
    cov_fl = scores.get("coverage_flat", 0.0)
    total = cov_ob + cov_fl
    if total < MIN_COVERAGE_FOR_JUDGEMENT:
        return "uncovered"

    if plane.is_oblique:
        if cov_ob >= STRONG_COVERAGE:
            nd = scores.get("normal_dot_best_oblique")
            if nd is not None and nd < NORMAL_DOT_MATCH:
                return "oblique_wrong_orientation"
            return "oblique_matched"
        if cov_fl >= STRONG_COVERAGE:
            return "oblique_covered_by_flat"  # missed oblique roof
        return "oblique_partial"
    else:
        if cov_fl >= STRONG_COVERAGE:
            return "flat_matched"
        if cov_ob >= STRONG_COVERAGE:
            return "flat_covered_by_oblique"  # roof pipeline extended too far
        return "flat_partial"


def _per_room_rollup(plane_rows: list[dict]) -> list[dict]:
    by_room: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in plane_rows:
        by_room[(row["uuid"], row["story"], row["room_index"])].append(row)

    out: list[dict] = []
    for (uuid, story, ri), rows in by_room.items():
        incls = [float(r["inclination_deg"]) for r in rows]
        ob_count = sum(1 for v in incls if v >= OBLIQUE_MIN_INCL_DEG)
        n = len(rows)
        majority = "oblique" if ob_count > n - ob_count else "flat"
        classes = Counter(r["classification"] for r in rows)
        pipeline_ob = any(float(r["coverage_oblique"]) >= STRONG_COVERAGE for r in rows)
        pipeline_fl = any(float(r["coverage_flat"]) >= STRONG_COVERAGE for r in rows)
        if pipeline_ob and not pipeline_fl:
            pipeline_class = "oblique"
        elif pipeline_fl and not pipeline_ob:
            pipeline_class = "flat"
        elif pipeline_ob and pipeline_fl:
            pipeline_class = "mixed"
        else:
            pipeline_class = "none"
        out.append(
            {
                "uuid": uuid,
                "story": story,
                "room_index": ri,
                "n_planes": n,
                "n_oblique_raw": ob_count,
                "raw_majority_class": majority,
                "pipeline_class": pipeline_class,
                "raw_vs_pipeline_agrees": (
                    majority == pipeline_class
                    if pipeline_class in {"oblique", "flat"}
                    else False
                ),
                "incl_p50_deg": round(float(np.median(incls)), 2) if incls else 0.0,
                "classification_counts": json.dumps(dict(classes)),
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


def _summary(plane_rows: list[dict], room_rows: list[dict]) -> dict:
    class_counts = Counter(r["classification"] for r in plane_rows)
    agreement_counts = Counter(r["pipeline_class"] for r in room_rows)
    raw_vs_pipe = Counter()
    for r in room_rows:
        key = f"{r['raw_majority_class']}_vs_{r['pipeline_class']}"
        raw_vs_pipe[key] += 1

    top_uncovered_oblique = sorted(
        [
            r
            for r in plane_rows
            if r["classification"] == "uncovered"
            and float(r["inclination_deg"]) >= OBLIQUE_MIN_INCL_DEG
        ],
        key=lambda r: float(r["area_xz_m2"]),
        reverse=True,
    )[:20]
    top_oblique_covered_by_flat = sorted(
        [r for r in plane_rows if r["classification"] == "oblique_covered_by_flat"],
        key=lambda r: float(r["area_xz_m2"]),
        reverse=True,
    )[:20]
    top_wrong_orientation = sorted(
        [r for r in plane_rows if r["classification"] == "oblique_wrong_orientation"],
        key=lambda r: float(r["area_xz_m2"]),
        reverse=True,
    )[:20]

    def _slim(row: dict) -> dict:
        return {
            k: row[k]
            for k in (
                "element_id",
                "inclination_deg",
                "azimuth_deg",
                "area_xz_m2",
                "coverage_oblique",
                "coverage_flat",
                "best_oblique_id",
                "best_flat_id",
                "normal_dot_best_oblique",
                "classification",
            )
        }

    return {
        "thresholds": {
            "oblique_min_incl_deg": OBLIQUE_MIN_INCL_DEG,
            "min_coverage_for_judgement": MIN_COVERAGE_FOR_JUDGEMENT,
            "strong_coverage": STRONG_COVERAGE,
            "normal_dot_match": NORMAL_DOT_MATCH,
            "near_vertical_exclude_deg": NEAR_VERTICAL_EXCLUDE_DEG,
        },
        "n_planes_scored": len(plane_rows),
        "classification_counts": dict(class_counts),
        "n_rooms": len(room_rows),
        "pipeline_class_counts": dict(agreement_counts),
        "raw_vs_pipeline_class_counts": dict(raw_vs_pipe),
        "top_uncovered_oblique": [_slim(r) for r in top_uncovered_oblique],
        "top_oblique_covered_by_flat": [_slim(r) for r in top_oblique_covered_by_flat],
        "top_wrong_orientation": [_slim(r) for r in top_wrong_orientation],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BUILDINGS_PATH.open() as handle:
        buildings = json.load(handle)
    with ROOF_RESULTS_PATH.open() as handle:
        roof_results = json.load(handle)

    plane_rows: list[dict] = []
    for building in buildings:
        uuid = building["uuid"]
        roof_result = roof_results.get(uuid)
        if not roof_result:
            continue
        exposed = _exposed_keys(roof_result)
        raw_planes = _collect_raw_planes(building, exposed)
        if not raw_planes:
            continue
        surfaces = _collect_surfaces(uuid, roof_result)
        for plane in raw_planes:
            scores = _score_plane_against_surfaces(plane, surfaces)
            if not scores:
                continue
            cls = _classify(plane, scores)
            # Compute azimuth for reporting
            nx, nz = float(plane.normal[0]), float(plane.normal[2])
            if abs(nx) < 1e-9 and abs(nz) < 1e-9:
                az_deg = ""
            else:
                az_deg = round(math.degrees(math.atan2(-nx, -nz)) % 360.0, 2)
            plane_rows.append(
                {
                    "uuid": uuid,
                    "element_id": plane.element_id,
                    "story": plane.story,
                    "room_index": plane.room_index,
                    "plane_index": plane.plane_index,
                    "inclination_deg": round(plane.inclination, 2),
                    "azimuth_deg": az_deg,
                    "area_xz_m2": round(plane.area_xz, 3),
                    "coverage_oblique": scores["coverage_oblique"],
                    "coverage_flat": scores["coverage_flat"],
                    "best_oblique_id": scores["best_oblique_id"],
                    "best_oblique_frac": scores["best_oblique_frac"],
                    "best_flat_id": scores["best_flat_id"],
                    "best_flat_frac": scores["best_flat_frac"],
                    "normal_dot_best_oblique": scores["normal_dot_best_oblique"],
                    "classification": cls,
                }
            )

    room_rows = _per_room_rollup(plane_rows)

    _write_csv(OUT_DIR / "per_plane.csv", plane_rows)
    _write_csv(OUT_DIR / "per_room.csv", room_rows)
    summary = _summary(plane_rows, room_rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"Scored {len(plane_rows)} raw planes across {summary['n_rooms']} exposed rooms"
    )
    print("Classification counts:", summary["classification_counts"])
    print(
        "Raw-vs-pipeline per-room agreement:", summary["raw_vs_pipeline_class_counts"]
    )


if __name__ == "__main__":
    main()
