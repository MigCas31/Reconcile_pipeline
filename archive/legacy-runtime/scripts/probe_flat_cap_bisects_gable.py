"""Diagnostic — cohort scan for flat ceiling partitions that bisect a gable.

Symptom matched: a flat cap atom is almost entirely covered (>95%) in 2D by
an oblique roof surface whose y-range extends BELOW the cap (the slope's
eave reaches down past cap_y) AND substantially ABOVE the cap (ridge rises
at least `rise_threshold` metres above cap_y). That is the geometry the user
flagged on 72122129 / ceiling-partition:a40fe4e6...: the flat partition sits
inside the gable triangle, visually splitting a correct sloped ceiling.

This script is read-only and produces a JSON report under /tmp.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

RESULTS_PATH = Path("reconcile/roof_algorithms_py_results.json")
OUT_PATH = Path("/tmp/flat_cap_bisects_gable.json")

OVERLAP_FRACTION_MIN = 0.95
EAVE_MAX_ABOVE_CAP_M = 0.15  # slope must reach *down* to within 15 cm of cap_y
RIDGE_MIN_ABOVE_CAP_M = 0.40  # slope must rise at least 40 cm above cap_y
MIN_AREA_M2 = 0.10  # ignore numerical slivers — they're a different symptom


def _poly_xz(corners: list[list[float]]) -> ShapelyPolygon | None:
    if not corners or len(corners) < 3:
        return None
    ring = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
    if len(ring) < 3:
        return None
    poly = ShapelyPolygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _y_range(corners: list[list[float]]) -> tuple[float, float] | None:
    ys = [float(c[1]) for c in corners if len(c) >= 3]
    if not ys:
        return None
    return min(ys), max(ys)


def scan() -> dict[str, Any]:
    results = json.loads(RESULTS_PATH.read_text())
    cohort: list[dict[str, Any]] = []
    by_building: dict[str, int] = defaultdict(int)

    for uuid, b in results.items():
        oblique_surfaces = (b.get("roof_surfaces") or {}).get("oblique") or []
        oblique_records: list[dict[str, Any]] = []
        for surf in oblique_surfaces:
            corners = surf.get("corners") or surf.get("poly") or []
            poly = _poly_xz(corners)
            yr = _y_range(corners)
            if poly is None or yr is None:
                continue
            oblique_records.append(
                {
                    "poly": poly,
                    "area": poly.area,
                    "y_min": yr[0],
                    "y_max": yr[1],
                    "hypothesis_id": surf.get("roof_hypothesis_id"),
                    "cluster": surf.get("cluster"),
                }
            )

        if not oblique_records:
            continue

        tb_nodes = {
            str(n.get("id")): n
            for n in ((b.get("top_boundary_graph") or {}).get("nodes") or [])
            if isinstance(n, dict) and n.get("type") == "TopBoundaryAtom"
        }

        flat_partitions = (b.get("ceiling_partitions") or {}).get("flat") or []
        for partition in flat_partitions:
            atom_id = str(partition.get("id") or "")
            if not atom_id:
                continue
            cap_y = partition.get("top_y_m")
            if cap_y is None:
                continue
            cap_y = float(cap_y)
            area = float(partition.get("area_m2") or 0.0)
            if area < MIN_AREA_M2:
                continue
            cap_poly = _poly_xz(partition.get("poly") or [])
            if cap_poly is None:
                continue
            cap_area = cap_poly.area
            if cap_area <= 0:
                continue

            node = tb_nodes.get(atom_id) or {}
            role = str(node.get("role") or "")
            if role not in {
                "flat_transition_cap",
                "flat_transition_cap_inferred",
                "flat_ceiling",
            }:
                continue

            # Find any oblique surface that bisects the cap from below+above.
            best: dict[str, Any] | None = None
            for ob in oblique_records:
                inter: BaseGeometry = cap_poly.intersection(ob["poly"])
                if inter.is_empty:
                    continue
                ovr = inter.area / cap_area
                if ovr < OVERLAP_FRACTION_MIN:
                    continue
                eave_above_cap = ob["y_min"] - cap_y
                ridge_above_cap = ob["y_max"] - cap_y
                if eave_above_cap > EAVE_MAX_ABOVE_CAP_M:
                    continue
                if ridge_above_cap < RIDGE_MIN_ABOVE_CAP_M:
                    continue
                candidate = {
                    "overlap_fraction": ovr,
                    "eave_above_cap_m": eave_above_cap,
                    "ridge_above_cap_m": ridge_above_cap,
                    "hypothesis_id": ob["hypothesis_id"],
                }
                if (
                    best is None
                    or candidate["ridge_above_cap_m"] > best["ridge_above_cap_m"]
                ):
                    best = candidate

            if best is None:
                continue

            cohort.append(
                {
                    "uuid": uuid,
                    "atom_id": atom_id,
                    "role": role,
                    "flat_role": partition.get("flat_role"),
                    "cap_y": cap_y,
                    "area_m2": area,
                    "room_index": partition.get("room_index"),
                    "story": partition.get("story"),
                    "roof_hypothesis_id": partition.get("roof_hypothesis_id"),
                    **best,
                }
            )
            by_building[uuid] += 1

    report = {
        "thresholds": {
            "OVERLAP_FRACTION_MIN": OVERLAP_FRACTION_MIN,
            "EAVE_MAX_ABOVE_CAP_M": EAVE_MAX_ABOVE_CAP_M,
            "RIDGE_MIN_ABOVE_CAP_M": RIDGE_MIN_ABOVE_CAP_M,
            "MIN_AREA_M2": MIN_AREA_M2,
        },
        "total_buildings_scanned": len(results),
        "buildings_hit": len(by_building),
        "atoms_hit": len(cohort),
        "top_buildings": sorted(by_building.items(), key=lambda kv: -kv[1])[:15],
        "sample_atoms": cohort[:25],
    }
    return report


def main() -> None:
    report = scan()
    OUT_PATH.write_text(json.dumps(report, indent=2, default=float))
    print(f"buildings scanned: {report['total_buildings_scanned']}")
    print(f"buildings hit:     {report['buildings_hit']}")
    print(f"atoms hit:         {report['atoms_hit']}")
    print("top buildings:")
    for uuid, cnt in report["top_buildings"]:
        print(f"  {uuid}: {cnt}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
