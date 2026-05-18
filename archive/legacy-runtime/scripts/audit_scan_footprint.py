#!/usr/bin/env python3
"""D4 — Scan-footprint robustness audit.

The reconstruction BIP (reconcile_v3/reconstruction/, Phases A-B) uses the
scan-derived footprint as its coverage target. If that footprint is wrong
— missing an L-wing, collapsed to a convex hull, self-intersecting — the
solver will silently pick the wrong faces. This script computes quality
signals for the footprint of every building in pipeline-outputs/ and
ranks the most suspicious ones for manual adjudication.

Usage:
    python scripts/audit_scan_footprint.py \
        --pipeline-dir pipeline-outputs \
        --out-dir reports/scan_footprint_audit_20260419

Outputs:
    <out>/metrics.json         per-building quality signals
    <out>/suspects.md          ranked top-N ordered list for review
    <out>/geojson/<uuid>.geojson  per-suspect viz (footprint + hull + rooms)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile.extract3d import extract_building
from reconcile.roof_algorithms_py.ceiling_plane_generation import collect_exposed_rooms
from reconcile.roof_algorithms_py.footprint_derivation import build_building_footprint
from reconcile.roof_algorithms_py.simple_slant import identify_simple_slant_rooms
from reconcile.roof_algorithms_py.story_index import (
    build_story_index,
    build_story_stats,
)

_ROOM_BUFFER_M = 0.3
_DEFAULT_TOP_N = 20


@dataclass
class FootprintMetrics:
    uuid: str
    status: str  # "ok" | "no_exposed_rooms" | "no_footprint" | "error"
    n_exposed_rooms: int
    n_vertices: int
    area_m2: float
    is_valid: bool
    is_simple: bool
    hull_area_m2: float
    convexity: float  # area / hull_area; 1.0 => convex
    obb_aspect_ratio: float
    unbuffered_union_area_m2: float
    buffer_bridged_ratio: float  # (buffered_area - unbuffered_area) / buffered_area
    fell_back_to_hull: bool
    error: str | None = None


def _polygon_shapely(coords):
    from shapely.geometry import Polygon

    return Polygon(coords)


def _convex_hull_area(coords):
    try:
        from shapely.geometry import MultiPoint

        hull = MultiPoint(list(coords)).convex_hull
        return float(hull.area) if hull.is_valid else 0.0
    except Exception:
        return 0.0


def _obb_aspect_ratio(coords) -> float:
    try:
        from shapely.geometry import Polygon

        poly = Polygon(coords)
        if not poly.is_valid or poly.area <= 0:
            return float("nan")
        mbr = poly.minimum_rotated_rectangle
        pts = list(mbr.exterior.coords)[:-1]
        if len(pts) < 4:
            return float("nan")
        sides = [math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        long_side = max(sides)
        short_side = min(sides)
        if short_side < 1e-6:
            return float("inf")
        return long_side / short_side
    except Exception:
        return float("nan")


def _unbuffered_union_area(floor_polys_2d) -> float:
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polys = []
        for pts in floor_polys_2d:
            if len(pts) < 3:
                continue
            p = Polygon(pts)
            if p.is_valid and not p.is_empty:
                polys.append(p)
        if not polys:
            return 0.0
        merged = unary_union(polys)
        return float(merged.area)
    except Exception:
        return 0.0


def _buffered_union_area(floor_polys_2d) -> float:
    """Area that ``_union_room_footprint`` would see before the -buffer shrink."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polys = []
        for pts in floor_polys_2d:
            if len(pts) < 3:
                continue
            p = Polygon(pts).buffer(_ROOM_BUFFER_M, join_style="mitre")
            if p.is_valid and not p.is_empty:
                polys.append(p)
        if not polys:
            return 0.0
        merged = unary_union(polys)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        return float(merged.area)
    except Exception:
        return 0.0


def _compute_metrics(uuid: str, bldg: dict) -> tuple[FootprintMetrics, dict]:
    """Returns (metrics, debug) where debug carries polygons for GeoJSON dump."""
    story_index = build_story_index(bldg)
    story_stats = build_story_stats(bldg)
    simple_slant = identify_simple_slant_rooms(
        bldg, story_index["has_floor_above"], story_index["all_stories"]
    )
    exclude_rooms = simple_slant["simple_slant_rooms"]
    exposed_rooms = collect_exposed_rooms(
        bldg, story_index["has_floor_above"], exclude_room_indices=exclude_rooms
    )

    floor_polys_2d = [
        [(p[0], p[2]) for p in er["fp"]]
        for er in exposed_rooms
        if er.get("fp") and len(er["fp"]) >= 3
    ]

    if not exposed_rooms:
        m = FootprintMetrics(
            uuid=uuid,
            status="no_exposed_rooms",
            n_exposed_rooms=0,
            n_vertices=0,
            area_m2=0.0,
            is_valid=False,
            is_simple=False,
            hull_area_m2=0.0,
            convexity=float("nan"),
            obb_aspect_ratio=float("nan"),
            unbuffered_union_area_m2=0.0,
            buffer_bridged_ratio=float("nan"),
            fell_back_to_hull=False,
        )
        return m, {"floor_polys_2d": []}

    footprint_info = build_building_footprint(
        exposed_rooms, story_stats["story_floor_polys"]
    )
    footprint = footprint_info.get("building_footprint")

    if not footprint or len(footprint) < 3:
        m = FootprintMetrics(
            uuid=uuid,
            status="no_footprint",
            n_exposed_rooms=len(exposed_rooms),
            n_vertices=0,
            area_m2=0.0,
            is_valid=False,
            is_simple=False,
            hull_area_m2=0.0,
            convexity=float("nan"),
            obb_aspect_ratio=float("nan"),
            unbuffered_union_area_m2=_unbuffered_union_area(floor_polys_2d),
            buffer_bridged_ratio=float("nan"),
            fell_back_to_hull=False,
        )
        return m, {"floor_polys_2d": floor_polys_2d, "footprint": None}

    poly = _polygon_shapely(footprint)
    area = float(poly.area) if poly.is_valid else 0.0
    hull_area = _convex_hull_area(footprint)
    convexity = area / hull_area if hull_area > 1e-6 else float("nan")
    unbuffered = _unbuffered_union_area(floor_polys_2d)
    buffered = _buffered_union_area(floor_polys_2d)
    bridged_ratio = (
        (buffered - unbuffered) / buffered if buffered > 1e-6 else float("nan")
    )
    # Heuristic: if Shapely-union path fell through, the function returns the
    # convex hull of room floor points. That hull is (by construction)
    # strictly convex, so convexity ~ 1.0 AND the footprint vertex count
    # equals the hull vertex count. We only flag when both hold AND the
    # unbuffered-union path failed or collapsed.
    fell_back = abs(convexity - 1.0) < 1e-3 and unbuffered <= 1e-6

    metrics = FootprintMetrics(
        uuid=uuid,
        status="ok",
        n_exposed_rooms=len(exposed_rooms),
        n_vertices=len(footprint),
        area_m2=area,
        is_valid=bool(poly.is_valid),
        is_simple=bool(poly.is_simple),
        hull_area_m2=hull_area,
        convexity=convexity,
        obb_aspect_ratio=_obb_aspect_ratio(footprint),
        unbuffered_union_area_m2=unbuffered,
        buffer_bridged_ratio=bridged_ratio,
        fell_back_to_hull=fell_back,
    )
    debug = {"floor_polys_2d": floor_polys_2d, "footprint": footprint}
    return metrics, debug


def _suspicion_score(m: FootprintMetrics) -> float:
    """Higher => more suspicious. Criteria:

    * invalid / non-simple footprint       +5
    * fell back to convex hull              +3
    * area < 20 m² or > 1500 m²             +2
    * convexity > 0.995 (possibly convex)   +1
    * obb aspect > 8                        +2
    * buffer bridged > 30 % of area         +1
    """
    if m.status != "ok":
        return 10.0
    score = 0.0
    if not m.is_valid or not m.is_simple:
        score += 5.0
    if m.fell_back_to_hull:
        score += 3.0
    if m.area_m2 < 20.0 or m.area_m2 > 1500.0:
        score += 2.0
    if not math.isnan(m.convexity) and m.convexity > 0.995:
        score += 1.0
    if not math.isnan(m.obb_aspect_ratio) and m.obb_aspect_ratio > 8.0:
        score += 2.0
    if not math.isnan(m.buffer_bridged_ratio) and m.buffer_bridged_ratio > 0.30:
        score += 1.0
    return score


def _polygon_to_feature(coords_xz, name, props=None) -> dict:
    ring = [[x, z] for (x, z) in coords_xz]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"name": name, **(props or {})},
    }


def _write_geojson(out_path: Path, uuid: str, debug: dict, metrics: FootprintMetrics):
    features = []
    fp = debug.get("footprint")
    if fp:
        features.append(_polygon_to_feature(fp, "footprint", asdict(metrics)))
    for i, pts in enumerate(debug.get("floor_polys_2d", [])):
        features.append(_polygon_to_feature(pts, f"room_{i}"))
    out_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", default="pipeline-outputs", type=Path)
    ap.add_argument("--scan-cache", default=".scan-cache", type=Path)
    ap.add_argument("--out-dir", default="reports/scan_footprint_audit", type=Path)
    ap.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many buildings (for smoke tests).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir = args.out_dir / "geojson"
    geojson_dir.mkdir(exist_ok=True)

    uuids = sorted(
        entry.name
        for entry in args.pipeline_dir.iterdir()
        if entry.is_dir() and (entry / "merged.json").exists()
    )
    if args.limit:
        uuids = uuids[: args.limit]

    print(f"Auditing {len(uuids)} buildings...")

    results: list[tuple[FootprintMetrics, dict]] = []
    for i, uuid in enumerate(uuids):
        try:
            bldg = extract_building(uuid, args.pipeline_dir, args.scan_cache)
            if not bldg:
                metrics = FootprintMetrics(
                    uuid=uuid,
                    status="error",
                    n_exposed_rooms=0,
                    n_vertices=0,
                    area_m2=0.0,
                    is_valid=False,
                    is_simple=False,
                    hull_area_m2=0.0,
                    convexity=float("nan"),
                    obb_aspect_ratio=float("nan"),
                    unbuffered_union_area_m2=0.0,
                    buffer_bridged_ratio=float("nan"),
                    fell_back_to_hull=False,
                    error="extract_building returned None",
                )
                results.append((metrics, {}))
                continue
            metrics, debug = _compute_metrics(uuid, bldg)
        except Exception as exc:
            metrics = FootprintMetrics(
                uuid=uuid,
                status="error",
                n_exposed_rooms=0,
                n_vertices=0,
                area_m2=0.0,
                is_valid=False,
                is_simple=False,
                hull_area_m2=0.0,
                convexity=float("nan"),
                obb_aspect_ratio=float("nan"),
                unbuffered_union_area_m2=0.0,
                buffer_bridged_ratio=float("nan"),
                fell_back_to_hull=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            debug = {}
            traceback.print_exc()
        results.append((metrics, debug))
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(uuids)}")

    # Rank
    scored = sorted(results, key=lambda pair: _suspicion_score(pair[0]), reverse=True)

    # Write per-building metrics
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps([asdict(m) for m, _ in results], indent=2, default=lambda x: None)
    )

    # Write GeoJSON for top suspects
    top = scored[: args.top_n]
    for metrics, debug in top:
        if debug:
            _write_geojson(
                geojson_dir / f"{metrics.uuid}.geojson", metrics.uuid, debug, metrics
            )

    # Aggregates
    ok = [m for m, _ in results if m.status == "ok"]
    no_fp = [m for m, _ in results if m.status == "no_footprint"]
    no_rooms = [m for m, _ in results if m.status == "no_exposed_rooms"]
    errors = [m for m, _ in results if m.status == "error"]
    fell_back = [m for m in ok if m.fell_back_to_hull]
    invalid = [m for m in ok if not m.is_valid or not m.is_simple]

    # Markdown summary
    md_lines = [
        "# Scan-footprint robustness audit",
        "",
        f"Corpus: `{args.pipeline_dir}` — {len(results)} buildings",
        "",
        "## Aggregates",
        "",
        f"- OK: **{len(ok)}**",
        f"- No exposed rooms: {len(no_rooms)}",
        f"- No footprint produced: {len(no_fp)}",
        f"- Pipeline errors: {len(errors)}",
        f"- Fell back to convex hull (union path failed): **{len(fell_back)}**",
        f"- Invalid / self-intersecting footprint: **{len(invalid)}**",
        "",
        "## Top suspects for manual adjudication",
        "",
        "| rank | uuid | score | status | area m² | convexity | aspect | fallback | "
        "valid | simple | note |",
        "|-----:|------|------:|--------|--------:|----------:|-------:|:--------:|:-----:|:------:|------|",
    ]
    for rank, (m, _) in enumerate(top, 1):
        note = m.error or ""
        md_lines.append(
            f"| {rank} | `{m.uuid}` | {_suspicion_score(m):.1f} | {m.status} | "
            f"{m.area_m2:.1f} | {m.convexity:.3f} | "
            f"{m.obb_aspect_ratio:.2f} | "
            f"{'Y' if m.fell_back_to_hull else '.'} | "
            f"{'Y' if m.is_valid else 'N'} | "
            f"{'Y' if m.is_simple else 'N'} | {note} |"
        )

    md_lines += [
        "",
        "## Adjudication protocol",
        "",
        "For each top-N building:",
        "",
        "1. Open `geojson/<uuid>.geojson` in QGIS or https://geojson.io "
        "(this is XZ-plane scan-local coordinates, not geographic).",
        "2. Judge: is the derived footprint (blue) a faithful outline of "
        "   the union of the room polygons? Flag if:",
        "   - it misses an L- or T-shape wing",
        "   - it collapses to a convex hull over a clearly concave building",
        "   - it is self-intersecting or has absurd aspect ratio",
        "3. Record verdict and failure mode in `adjudications.csv` "
        "   (columns: uuid, verdict=ok|wrong|uncertain, failure_mode, notes).",
        "",
        "## Gate",
        "",
        "Per the revised plan: if precision/recall of the footprint across "
        "the 20-building sample falls below 0.95, fix "
        "`reconcile/roof_algorithms_py/footprint_derivation.py` before "
        "starting Phase A (candidate face generation).",
    ]

    md_path = args.out_dir / "suspects.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {metrics_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {len(top)} GeoJSON files to {geojson_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
