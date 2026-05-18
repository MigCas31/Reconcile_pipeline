"""How often do computed roof surfaces over-extend beyond raw scan evidence?

Phase 2 (post-pivot) audit. Computed pipeline owns orientation/inclination;
raw ceilings are trusted only as a bound on extent (XZ footprint + Y). This
script measures, per roof surface in ``roof_algorithms_py_results.json``,
how far the computed surface reaches past the union of overlapping raw
ceiling polygons.

For every ``roof_surfaces.oblique[i]`` and ``roof_surfaces.flat[i]``:

* ``overextend_fraction_xz`` — fraction of the surface's XZ footprint that
  sits outside the union of XZ polygons of the raw ceiling planes that
  overlap it.
* ``overextend_y_m`` — surface top-y minus the highest raw-plane corner
  y among overlapping raw planes (None when there's no overlap).

Overlap is purely spatial: any raw plane in the building whose XZ polygon
intersects the surface XZ polygon with area >= ``MIN_OVERLAP_AREA_M2`` is
counted. That keeps the audit independent of the pipeline's room-membership
decisions, which may themselves be what we're trying to correct.

Output::

    reports/computed_extent_vs_raw/
        per_surface.csv
        summary.json
        overextend_polygons.json   # viewer-consumable, keyed by element_id
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
OUT_DIR = REPO / "reports" / "computed_extent_vs_raw"

MIN_SURFACE_AREA_M2 = 0.5
MIN_OVERLAP_AREA_M2 = 0.1
# Surfaces whose top-y is this far BELOW the highest overlapping raw-ceiling
# corner are skipped — they can't physically be roofs. Corpus-wide this removes
# ~30% of pipeline flat surfaces (whole-building envelope fallbacks with
# room_index=None, interior horizontal fragments, etc.). Overextend is a
# "computed reaches past raw" signal, not a "pipeline emitted a phantom flat"
# signal; the latter is a separate bug worth tracking but doesn't belong in
# this overlay.
MAX_NEGATIVE_GAP_M = -0.1


def _xz_polygon(corners) -> Polygon | None:
    if not corners or len(corners) < 3:
        return None
    p = Polygon([(float(c[0]), float(c[2])) for c in corners])
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or not p.is_valid or p.area < 1e-6:
        return None
    return p


def _collect_raw_planes(building: dict) -> list[dict]:
    """Flatten raw_ceiling_planes across all rooms; keep XZ poly + y range."""
    out: list[dict] = []
    for ri, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        for pi, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            poly = _xz_polygon(corners)
            if poly is None:
                continue
            ys = [float(c[1]) for c in corners]
            out.append(
                {
                    "story": story,
                    "room_index": ri,
                    "plane_index": pi,
                    "xz": poly,
                    "y_min": min(ys),
                    "y_max": max(ys),
                }
            )
    return out


def _fit_plane(corners):
    """SVD plane fit; return (centroid, normal) or None."""
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
    n = float(np.linalg.norm(normal))
    if n < 1e-9:
        return None
    return centroid, normal / n


def _lift_polygon_to_surface(xz_poly: Polygon, surface_corners) -> list[list[float]]:
    """Assign Y to each vertex of an XZ polygon using the surface's fitted plane.

    Falls back to mean surface Y if the plane is (near-)vertical.
    """
    mean_y = float(np.mean([c[1] for c in surface_corners]))
    fit = _fit_plane(surface_corners)
    if fit is None:

        def y_fn(x, z):
            return mean_y

    else:
        centroid, normal = fit
        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
        cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
        if abs(ny) < 1e-6:

            def y_fn(x, z):
                return mean_y

        else:

            def y_fn(x, z):
                return cy - (nx * (x - cx) + nz * (z - cz)) / ny

    poly = xz_poly
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.is_empty:
        return []
    return [
        [float(x), float(y_fn(x, z)), float(z)] for x, z in poly.exterior.coords[:-1]
    ]


def _measure_surface(
    surface_corners, raw_planes: list[dict]
) -> tuple[dict, list[list[float]]] | None:
    surface_xz = _xz_polygon(surface_corners)
    if surface_xz is None or surface_xz.area < MIN_SURFACE_AREA_M2:
        return None
    surface_y_max = max(float(c[1]) for c in surface_corners)

    overlapping: list[dict] = []
    for plane in raw_planes:
        try:
            inter = surface_xz.intersection(plane["xz"])
        except Exception:
            continue
        if inter.is_empty or inter.area < MIN_OVERLAP_AREA_M2:
            continue
        overlapping.append(plane)

    overextend_corners: list[list[float]] = []
    raw_y_max: float | None = None
    if overlapping:
        try:
            raw_union = unary_union([p["xz"].buffer(0) for p in overlapping])
        except Exception:
            raw_union = None
        if raw_union is not None and not raw_union.is_empty:
            covered = surface_xz.intersection(raw_union).area
            try:
                over_xz = surface_xz.difference(raw_union)
            except Exception:
                over_xz = None
            if (
                over_xz is not None
                and not over_xz.is_empty
                and over_xz.area >= MIN_OVERLAP_AREA_M2
            ):
                overextend_corners = _lift_polygon_to_surface(over_xz, surface_corners)
        else:
            covered = 0.0
        raw_y_max = max(p["y_max"] for p in overlapping)
        overextend_y = surface_y_max - raw_y_max
        if overextend_y < MAX_NEGATIVE_GAP_M:
            return None
    else:
        covered = 0.0
        overextend_y = None
        overextend_corners = _lift_polygon_to_surface(surface_xz, surface_corners)

    overextend_frac = max(0.0, 1.0 - covered / surface_xz.area)
    rooms = {(p["story"], p["room_index"]) for p in overlapping}
    measurement = {
        "surface_area_m2": round(float(surface_xz.area), 3),
        "surface_y_max": round(surface_y_max, 3),
        "overextend_fraction_xz": round(float(overextend_frac), 3),
        "overextend_y_m": (
            round(float(overextend_y), 3) if overextend_y is not None else None
        ),
        "raw_y_max": round(float(raw_y_max), 3) if raw_y_max is not None else None,
        "n_overlapping_planes": len(overlapping),
        "n_overlapping_rooms": len(rooms),
    }
    return measurement, overextend_corners


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bucket(value: float, edges: list[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return f"{edges[i]:g}-{edges[i + 1]:g}"
    return f">={edges[-1]:g}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BUILDINGS_PATH.open() as handle:
        buildings = json.load(handle)
    with ROOF_RESULTS_PATH.open() as handle:
        roof_results = json.load(handle)

    buildings_by_uuid = {b["uuid"]: b for b in buildings}

    rows: list[dict] = []
    overextend_buildings: dict[str, list[dict]] = {}

    def _emit(uuid, kind, idx, surface, measurement, overextend_corners, story):
        element_kind = "roof-oblique" if kind == "oblique" else "roof-flat"
        sub = "oblique" if kind == "oblique" else "flat"
        element_id = f"{uuid}::{element_kind}::{sub}:{idx}"
        rows.append(
            {
                "uuid": uuid,
                "surface_kind": kind,
                "surface_index": idx,
                "story": story,
                "element_id": element_id,
                **measurement,
            }
        )
        if overextend_corners:
            overextend_buildings.setdefault(uuid, []).append(
                {
                    "source_element_id": element_id,
                    "overextend_element_id": (f"{uuid}::roof-overextend::{sub}:{idx}"),
                    "surface_kind": kind,
                    "surface_index": idx,
                    "story": story,
                    "corners": overextend_corners,
                    "overextend_fraction_xz": measurement["overextend_fraction_xz"],
                    "overextend_y_m": measurement["overextend_y_m"],
                    "raw_y_max": measurement["raw_y_max"],
                    "surface_area_m2": measurement["surface_area_m2"],
                }
            )

    for uuid, result in roof_results.items():
        building = buildings_by_uuid.get(uuid)
        if building is None:
            continue
        raw_planes = _collect_raw_planes(building)
        if not raw_planes:
            continue
        roof_surfaces = result.get("roof_surfaces") or {}

        for idx, surface in enumerate(roof_surfaces.get("oblique") or []):
            out = _measure_surface(surface.get("corners") or [], raw_planes)
            if out is None:
                continue
            measurement, overextend_corners = out
            story = (
                surface.get("dominant_story")
                if surface.get("dominant_story") is not None
                else surface.get("story")
            )
            _emit(uuid, "oblique", idx, surface, measurement, overextend_corners, story)

        for idx, surface in enumerate(roof_surfaces.get("flat") or []):
            # Skip whole-building envelope fallbacks (room_index=None). These
            # are stitched from leftover segments and routinely produce
            # hundred-square-metre rectangles below the true ceiling plane,
            # which dominate the overlay without carrying any real overextend
            # signal.
            if surface.get("room_index") is None:
                continue
            out = _measure_surface(surface.get("corners") or [], raw_planes)
            if out is None:
                continue
            measurement, overextend_corners = out
            _emit(
                uuid,
                "flat",
                idx,
                surface,
                measurement,
                overextend_corners,
                surface.get("story"),
            )

    _write_csv(OUT_DIR / "per_surface.csv", rows)
    (OUT_DIR / "overextend_polygons.json").write_text(
        json.dumps({"buildings": overextend_buildings}, indent=2)
    )

    def _percentiles(values: list[float]) -> dict:
        if not values:
            return {}
        arr = np.asarray(values, dtype=float)
        return {
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "max": round(float(arr.max()), 3),
        }

    frac_edges = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
    y_edges = [-10.0, -0.5, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 100.0]

    def _cohort(kind: str) -> dict:
        cohort = [r for r in rows if r["surface_kind"] == kind]
        if not cohort:
            return {"n": 0}
        frac_hist: Counter = Counter()
        for r in cohort:
            frac_hist[_bucket(r["overextend_fraction_xz"], frac_edges)] += 1
        y_hist: Counter = Counter()
        for r in cohort:
            y = r["overextend_y_m"]
            if y is None:
                y_hist["no_overlap"] += 1
            else:
                y_hist[_bucket(y, y_edges)] += 1
        no_overlap = sum(1 for r in cohort if r["n_overlapping_planes"] == 0)
        return {
            "n": len(cohort),
            "n_no_overlap": no_overlap,
            "overextend_fraction_xz": _percentiles(
                [r["overextend_fraction_xz"] for r in cohort]
            ),
            "overextend_fraction_xz_hist": dict(frac_hist),
            "overextend_y_m": _percentiles(
                [r["overextend_y_m"] for r in cohort if r["overextend_y_m"] is not None]
            ),
            "overextend_y_m_hist": dict(y_hist),
            "worst_fraction": sorted(
                cohort, key=lambda r: -r["overextend_fraction_xz"]
            )[:10],
            "worst_y": sorted(
                cohort,
                key=lambda r: -(r["overextend_y_m"] or 0.0),
            )[:10],
        }

    summary = {
        "thresholds": {
            "min_surface_area_m2": MIN_SURFACE_AREA_M2,
            "min_overlap_area_m2": MIN_OVERLAP_AREA_M2,
        },
        "n_buildings_with_raw": len({r["uuid"] for r in rows}),
        "oblique": _cohort("oblique"),
        "flat": _cohort("flat"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    oblique = summary["oblique"]
    flat = summary["flat"]
    print(f"Surfaces scored: oblique={oblique.get('n', 0)}  flat={flat.get('n', 0)}")
    print(f"Buildings with raw ceilings: {summary['n_buildings_with_raw']}")
    print("Oblique overextend_fraction_xz:", oblique.get("overextend_fraction_xz"))
    print("Oblique overextend_y_m:", oblique.get("overextend_y_m"))
    print("Flat overextend_fraction_xz:", flat.get("overextend_fraction_xz"))
    print("Flat overextend_y_m:", flat.get("overextend_y_m"))


if __name__ == "__main__":
    main()
