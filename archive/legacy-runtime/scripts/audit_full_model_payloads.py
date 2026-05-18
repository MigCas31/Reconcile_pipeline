from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile.extract3d.builder import extract_building
from reconcile.roof_algorithms_py import run_roof_algorithms
from reconcile.viewer_server import (
    FULL_BUILDING_PART_ID,
    PIPELINE_ROOT,
    SCAN_CACHE_ROOT,
    _build_ontology_part_payloads,
    _build_ontology_summary,
)
from reconcile_v2.graph_builder import build_topology_graph

TARGET_CATEGORIES = (
    "base_exterior_wall",
    "base_interior_wall",
    "base_room_floor",
    "base_room_ceiling",
    "base_window",
    "base_door",
    "base_opening",
    "exterior_roof",
    "knee_wall",
    "unresolved_region",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ontology full-model payloads for building-part:full-building "
            "across pipeline-outputs/."
        )
    )
    parser.add_argument(
        "--output-json",
        default=".context/full_model_full_building_audit.json",
        help="Path to write the JSON audit report.",
    )
    parser.add_argument(
        "--output-md",
        default=".context/full_model_full_building_audit.md",
        help="Path to write the Markdown audit summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of buildings to audit.",
    )
    parser.add_argument(
        "--uuid",
        action="append",
        default=[],
        help="Specific building UUID(s) to audit. May be repeated.",
    )
    return parser.parse_args()


def _select_uuids(root: Path, requested: list[str], limit: int) -> list[str]:
    if requested:
        return requested
    uuids = sorted(path.name for path in root.iterdir() if path.is_dir())
    if limit > 0:
        return uuids[:limit]
    return uuids


def _classify_building(
    *,
    counts: Counter[str],
    part_payload: dict,
    summary: dict,
) -> list[str]:
    categories: list[str] = []
    has_base_shell = any(
        counts.get(name, 0) > 0
        for name in (
            "base_exterior_wall",
            "base_interior_wall",
            "base_room_floor",
            "base_room_ceiling",
        )
    )
    has_roof = counts.get("exterior_roof", 0) > 0
    has_knee = counts.get("knee_wall", 0) > 0
    has_unresolved = counts.get("unresolved_region", 0) > 0
    has_fenestration = any(
        counts.get(name, 0) > 0 for name in ("base_window", "base_door", "base_opening")
    )
    roof_cells = int(
        (part_payload.get("metadata") or {}).get("roof_cell_count", 0) or 0
    )
    occupied_cells = int(
        (part_payload.get("metadata") or {}).get("occupied_room_cell_count", 0) or 0
    )
    summary_counts = Counter(
        str(surface.get("category") or "")
        for surface in (summary.get("renderable_surfaces") or [])
        if isinstance(surface, dict)
    )

    if has_base_shell:
        categories.append("has_base_shell")
    else:
        categories.append("missing_base_shell")
    if has_roof:
        categories.append("has_roof_surface")
    else:
        categories.append("missing_roof_surface")
    if has_knee:
        categories.append("has_knee_wall")
    if has_unresolved:
        categories.append("has_unresolved_region")
    if has_fenestration:
        categories.append("has_fenestration")
    if roof_cells > 0:
        categories.append("has_roof_cells")
    else:
        categories.append("missing_roof_cells")
    if occupied_cells > 0:
        categories.append("has_occupied_room_cells")
    else:
        categories.append("missing_occupied_room_cells")
    if summary_counts and not any(
        key not in {"room_ceiling_flat", "room_ceiling_sloped", "attic_floor"}
        for key in summary_counts
    ):
        categories.append("summary_is_ceiling_only")
    return categories


def _roof_evidence_room_count(roof: dict) -> int:
    room_summaries = (roof.get("top_boundary_graph") or {}).get("room_summaries") or {}
    count = 0
    for summary in room_summaries.values():
        if not isinstance(summary, dict):
            continue
        if (
            bool(summary.get("partially_covered_by_sloped_roof"))
            or bool(summary.get("strong_perimeter_sloped"))
            or bool(summary.get("strong_knee_wall_signal"))
            or bool(summary.get("has_candidate_attic_relation"))
            or bool(summary.get("has_candidate_upper_void_relation"))
            or int(summary.get("roof_evidence_score", 0) or 0) >= 4
        ):
            count += 1
    return count


def _classify_missing_roof_payload(
    *,
    counts: Counter[str],
    roof: dict,
    summary: dict,
    part_payload: dict,
) -> str:
    if counts.get("exterior_roof", 0) > 0:
        return "has_roof_surface"
    roof_surfaces = roof.get("roof_surfaces") or {}
    flat_count = len(roof_surfaces.get("flat") or [])
    oblique_count = len(roof_surfaces.get("oblique") or [])
    roof_cells = int(
        (part_payload.get("metadata") or {}).get("roof_cell_count", 0) or 0
    )
    evidence_rooms = _roof_evidence_room_count(roof)
    unresolved_count = int(
        (summary.get("metadata") or {}).get("unresolved_region_count", 0) or 0
    )
    if (
        flat_count > 0
        and oblique_count == 0
        and roof_cells == 0
        and evidence_rooms == 0
    ):
        return "flat_surface_only_no_graph_roof"
    if oblique_count > 0 and roof_cells == 0:
        return "roof_surface_present_but_no_cells"
    if evidence_rooms > 0 and flat_count == 0 and oblique_count == 0:
        return "roof_evidence_without_surfaces"
    if unresolved_count > 0:
        return "summary_unresolved_without_part_roof"
    return "no_roof_signal"


def _roof_mode(payload_metadata: dict) -> str:
    roof_cells = int(payload_metadata.get("roof_cell_count", 0) or 0)
    exact_flat = int(payload_metadata.get("roof_exact_flat_surface_count", 0) or 0)
    coverage_patch = int(
        payload_metadata.get("roof_coverage_patch_surface_count", 0) or 0
    )
    fallback = int(payload_metadata.get("roof_fallback_surface_count", 0) or 0)
    parts: list[str] = []
    if roof_cells > 0:
        parts.append("exact_cells")
    if exact_flat > 0:
        parts.append("exact_flat")
    if coverage_patch > 0:
        parts.append("coverage_patch")
    if fallback > 0:
        parts.append("fallback")
    return "_plus_".join(parts) if parts else "no_roof"


def _surface_corners(surface: dict) -> list[list[float]]:
    for key in ("corners", "poly", "polygon"):
        value = surface.get(key)
        if isinstance(value, list):
            return value
    return []


def _polygon_area_3d(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    nx = ny = nz = 0.0
    count = len(corners)
    for index in range(count):
        x1, y1, z1 = (float(value) for value in corners[index][:3])
        x2, y2, z2 = (float(value) for value in corners[(index + 1) % count][:3])
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _total_surface_area(surfaces: list[dict]) -> float:
    return round(
        sum(
            _polygon_area_3d(_surface_corners(surface))
            for surface in surfaces
            if isinstance(surface, dict)
        ),
        6,
    )


def _area_metrics(
    summary: dict, part_payload: dict, roof: dict
) -> dict[str, float | None]:
    payload_surfaces = [
        surface
        for surface in (part_payload.get("renderable_surfaces") or [])
        if isinstance(surface, dict)
    ]
    summary_surfaces = [
        surface
        for surface in (summary.get("renderable_surfaces") or [])
        if isinstance(surface, dict)
    ]
    roof_surfaces = roof.get("roof_surfaces") or {}
    heuristic_ceiling = roof.get("ceiling") or {}
    ontology_roof_surfaces = [
        surface
        for surface in payload_surfaces
        if str(surface.get("category") or "") == "exterior_roof"
    ]
    unresolved_surfaces = [
        surface
        for surface in payload_surfaces
        if str(surface.get("category") or "") == "unresolved_region"
    ]
    ontology_semantic_ceiling_surfaces = [
        surface
        for surface in summary_surfaces
        if str(surface.get("category") or "")
        in {"room_ceiling_flat", "room_ceiling_sloped", "attic_floor"}
    ]
    ontology_shell_ceiling_surfaces = [
        surface
        for surface in payload_surfaces
        if str(surface.get("category") or "") == "base_room_ceiling"
    ]
    heuristic_roof_surfaces = [
        surface
        for key in ("flat", "oblique")
        for surface in (roof_surfaces.get(key) or [])
        if isinstance(surface, dict)
    ]
    heuristic_ceiling_surfaces = [
        surface
        for key in ("flat", "oblique", "simple_slant")
        for surface in (heuristic_ceiling.get(key) or [])
        if isinstance(surface, dict)
    ]

    ontology_roof_area = _total_surface_area(ontology_roof_surfaces)
    unresolved_area = _total_surface_area(unresolved_surfaces)
    heuristic_roof_area = _total_surface_area(heuristic_roof_surfaces)
    ontology_ceiling_area = _total_surface_area(ontology_semantic_ceiling_surfaces)
    ontology_shell_ceiling_area = _total_surface_area(ontology_shell_ceiling_surfaces)
    heuristic_ceiling_area = _total_surface_area(heuristic_ceiling_surfaces)

    def _ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 1e-6:
            return None
        return round(numerator / denominator, 6)

    roof_overlap = _union_projected_area_and_overlap(
        ontology_roof_surfaces, heuristic_roof_surfaces
    )
    semantic_ceiling_overlap = _union_projected_area_and_overlap(
        ontology_semantic_ceiling_surfaces,
        heuristic_ceiling_surfaces,
    )
    shell_ceiling_overlap = _union_projected_area_and_overlap(
        ontology_shell_ceiling_surfaces,
        heuristic_ceiling_surfaces,
    )

    return {
        "ontology_roof_area_m2": ontology_roof_area,
        "heuristic_roof_area_m2": heuristic_roof_area,
        "roof_area_ratio": _ratio(ontology_roof_area, heuristic_roof_area),
        "roof_area_delta_m2": round(ontology_roof_area - heuristic_roof_area, 6),
        "unresolved_area_m2": unresolved_area,
        "ontology_ceiling_area_m2": ontology_ceiling_area,
        "ontology_shell_ceiling_area_m2": ontology_shell_ceiling_area,
        "heuristic_ceiling_area_m2": heuristic_ceiling_area,
        "ceiling_area_ratio": _ratio(ontology_ceiling_area, heuristic_ceiling_area),
        "shell_ceiling_area_ratio": _ratio(
            ontology_shell_ceiling_area, heuristic_ceiling_area
        ),
        "ceiling_area_delta_m2": round(
            ontology_ceiling_area - heuristic_ceiling_area, 6
        ),
        "shell_ceiling_area_delta_m2": round(
            ontology_shell_ceiling_area - heuristic_ceiling_area, 6
        ),
        "roof_overlap_projected_area_m2": roof_overlap["overlap_projected_area_m2"],
        "roof_ontology_only_projected_area_m2": roof_overlap[
            "ontology_only_projected_area_m2"
        ],
        "roof_heuristic_only_projected_area_m2": roof_overlap[
            "heuristic_only_projected_area_m2"
        ],
        "roof_overlap_ratio_vs_heuristic": roof_overlap["overlap_ratio_vs_heuristic"],
        "semantic_ceiling_overlap_projected_area_m2": semantic_ceiling_overlap[
            "overlap_projected_area_m2"
        ],
        "semantic_ceiling_heuristic_only_projected_area_m2": semantic_ceiling_overlap[
            "heuristic_only_projected_area_m2"
        ],
        "semantic_ceiling_overlap_ratio_vs_heuristic": semantic_ceiling_overlap[
            "overlap_ratio_vs_heuristic"
        ],
        "shell_ceiling_overlap_projected_area_m2": shell_ceiling_overlap[
            "overlap_projected_area_m2"
        ],
        "shell_ceiling_heuristic_only_projected_area_m2": shell_ceiling_overlap[
            "heuristic_only_projected_area_m2"
        ],
        "shell_ceiling_overlap_ratio_vs_heuristic": shell_ceiling_overlap[
            "overlap_ratio_vs_heuristic"
        ],
    }


def _surface_polygon_xz(surface: dict) -> Polygon | None:
    corners = _surface_corners(surface)
    if len(corners) < 3:
        return None
    points: list[tuple[float, float]] = []
    for corner in corners:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            return None
        points.append((float(corner[0]), float(corner[2])))
    try:
        poly = Polygon(points)
    except Exception:
        return None
    if poly.is_empty or poly.area <= 1e-6:
        return None
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
        except Exception:
            return None
    if isinstance(poly, Polygon):
        return poly if poly.area > 1e-6 else None
    if getattr(poly, "geom_type", "") == "MultiPolygon":
        try:
            return max(
                (
                    part
                    for part in poly.geoms
                    if isinstance(part, Polygon) and part.area > 1e-6
                ),
                key=lambda part: part.area,
            )
        except ValueError:
            return None
    return None


def _union_projected_area_and_overlap(
    ontology_surfaces: list[dict], heuristic_surfaces: list[dict]
) -> dict[str, float]:
    ontology_polys = [
        poly
        for surface in ontology_surfaces
        if isinstance(surface, dict)
        for poly in [_surface_polygon_xz(surface)]
        if poly is not None
    ]
    heuristic_polys = [
        poly
        for surface in heuristic_surfaces
        if isinstance(surface, dict)
        for poly in [_surface_polygon_xz(surface)]
        if poly is not None
    ]
    ontology_union = unary_union(ontology_polys) if ontology_polys else None
    heuristic_union = unary_union(heuristic_polys) if heuristic_polys else None
    ontology_area = float(getattr(ontology_union, "area", 0.0) or 0.0)
    heuristic_area = float(getattr(heuristic_union, "area", 0.0) or 0.0)
    overlap_area = 0.0
    if ontology_union is not None and heuristic_union is not None:
        try:
            overlap_area = float(ontology_union.intersection(heuristic_union).area)
        except Exception:
            overlap_area = 0.0
    return {
        "ontology_projected_area_m2": round(ontology_area, 6),
        "heuristic_projected_area_m2": round(heuristic_area, 6),
        "overlap_projected_area_m2": round(overlap_area, 6),
        "ontology_only_projected_area_m2": round(
            max(0.0, ontology_area - overlap_area), 6
        ),
        "heuristic_only_projected_area_m2": round(
            max(0.0, heuristic_area - overlap_area), 6
        ),
        "overlap_ratio_vs_heuristic": round(overlap_area / heuristic_area, 6)
        if heuristic_area > 1e-6
        else None,
        "overlap_ratio_vs_ontology": round(overlap_area / ontology_area, 6)
        if ontology_area > 1e-6
        else None,
    }


def _build_payload(uuid: str) -> tuple[dict, dict, dict]:
    merged_path = PIPELINE_ROOT / uuid / "merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"No merged.json for {uuid}")
    graph = build_topology_graph(
        merged_path=merged_path,
        scan_dir=SCAN_CACHE_ROOT,
        uuid=uuid,
    )
    building = extract_building(
        uuid=uuid,
        pipeline_dir=PIPELINE_ROOT,
        scan_cache_root=SCAN_CACHE_ROOT,
        load_topology_graph=False,
    )
    if not building:
        raise RuntimeError(f"extract_building returned no building for {uuid}")
    roof = run_roof_algorithms(building, graph=graph)
    topology_cell_complex = (graph.geometry_index or {}).get("cell_complex") or {}
    summary, part_graph_room_ids = _build_ontology_summary(
        uuid=uuid,
        roof=roof,
        topology_cell_complex=topology_cell_complex,
        building=building,
    )
    parts = _build_ontology_part_payloads(
        uuid=uuid,
        summary=summary,
        part_graph_room_ids=part_graph_room_ids,
        topology_cell_complex=topology_cell_complex,
        roof_cell_complex=roof.get("roof_cell_complex") or {},
        occupied_room_cell_complex=roof.get("occupied_room_cell_complex") or {},
        building=building,
        roof=roof,
    )
    return summary, parts, roof


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_none_"
    header = (
        "| uuid | total | base_exterior | base_interior | floor | ceiling | roof | "
        "knee | unresolved | missing_roof_class | seconds |"
    )
    divider = (
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
    )
    lines = [header, divider]
    for row in rows:
        counts = row["category_counts"]
        lines.append(
            (
                "| {uuid} | {total} | {bew} | {biw} | {floor} | "
                "{ceiling} | {roof} | {knee} | {unresolved} | "
                "{missing_roof_class} | {seconds:.3f} |"
            ).format(
                uuid=row["uuid"],
                total=row["renderable_surface_total"],
                bew=counts.get("base_exterior_wall", 0),
                biw=counts.get("base_interior_wall", 0),
                floor=counts.get("base_room_floor", 0),
                ceiling=counts.get("base_room_ceiling", 0),
                roof=counts.get("exterior_roof", 0),
                knee=counts.get("knee_wall", 0),
                unresolved=counts.get("unresolved_region", 0),
                missing_roof_class=row.get("missing_roof_class", ""),
                seconds=row["timings_s"]["total"],
            )
        )
    return "\n".join(lines)


def _write_markdown(path: Path, report: dict) -> None:
    aggregate = report["aggregate"]
    top_missing_roof = report["top_missing_roof_examples"]
    top_unresolved = report["top_unresolved_examples"]
    top_heavy = report["top_heaviest_payloads"]
    bucket_counts = aggregate["bucket_counts"]
    category_building_counts = aggregate["category_building_counts"]
    lines = [
        "# Full-Building Ontology Payload Audit",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- attempted: `{aggregate['attempted']}`",
        f"- ok: `{aggregate['ok']}`",
        f"- failed: `{aggregate['failed']}`",
        f"- avg_runtime_s: `{aggregate['avg_runtime_s']:.3f}`",
        f"- avg_renderable_surface_total: "
        f"`{aggregate['avg_renderable_surface_total']:.1f}`",
        "",
        "## Coverage Buckets",
        "",
        f"- has_base_shell: `{bucket_counts.get('has_base_shell', 0)}`",
        f"- missing_base_shell: `{bucket_counts.get('missing_base_shell', 0)}`",
        f"- has_roof_surface: `{bucket_counts.get('has_roof_surface', 0)}`",
        f"- missing_roof_surface: `{bucket_counts.get('missing_roof_surface', 0)}`",
        f"- has_knee_wall: `{bucket_counts.get('has_knee_wall', 0)}`",
        f"- has_unresolved_region: `{bucket_counts.get('has_unresolved_region', 0)}`",
        f"- has_fenestration: `{bucket_counts.get('has_fenestration', 0)}`",
        f"- has_roof_cells: `{bucket_counts.get('has_roof_cells', 0)}`",
        f"- missing_roof_cells: `{bucket_counts.get('missing_roof_cells', 0)}`",
        f"- summary_is_ceiling_only: "
        f"`{bucket_counts.get('summary_is_ceiling_only', 0)}`",
        "",
        "## Missing Roof Classes",
        "",
    ]
    for name, count in sorted(report.get("missing_roof_class_counts", {}).items()):
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Roof Modes", ""])
    for name, count in sorted(report.get("roof_mode_counts", {}).items()):
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Area Metrics", ""])
    for key, value in (aggregate.get("area_metrics") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Category Presence", ""])
    for category in TARGET_CATEGORIES:
        lines.append(
            f"- `{category}` present in `{category_building_counts.get(category, 0)}` "
            f"buildings, "
            f"total surfaces `{aggregate['category_surface_totals'].get(category, 0)}`"
        )
    lines.extend(
        [
            "",
            "## Missing Roof Examples",
            "",
            _markdown_table(top_missing_roof),
            "",
            "## Unresolved Examples",
            "",
            _markdown_table(top_unresolved),
            "",
            "## Heaviest Payloads",
            "",
            _markdown_table(top_heavy),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    uuids = _select_uuids(PIPELINE_ROOT, args.uuid, args.limit)

    aggregate = {
        "attempted": 0,
        "ok": 0,
        "failed": 0,
        "avg_runtime_s": 0.0,
        "avg_renderable_surface_total": 0.0,
        "bucket_counts": {},
        "category_building_counts": {},
        "category_surface_totals": {},
    }
    bucket_counts: Counter[str] = Counter()
    category_building_counts: Counter[str] = Counter()
    category_surface_totals: Counter[str] = Counter()
    results: list[dict] = []
    errors: list[dict] = []

    for index, uuid in enumerate(uuids, start=1):
        aggregate["attempted"] = index
        started = perf_counter()
        try:
            summary, parts, roof = _build_payload(uuid)
            payload = parts.get(FULL_BUILDING_PART_ID)
            if payload is None:
                raise KeyError(f"Missing {FULL_BUILDING_PART_ID} payload")
            counts = Counter(
                str(surface.get("category") or "")
                for surface in (payload.get("renderable_surfaces") or [])
                if isinstance(surface, dict)
            )
            categories = _classify_building(
                counts=counts,
                part_payload=payload,
                summary=summary,
            )
            elapsed = perf_counter() - started
            record = {
                "uuid": uuid,
                "status": "ok",
                "timings_s": {
                    "total": round(elapsed, 6),
                },
                "renderable_surface_total": len(
                    payload.get("renderable_surfaces") or []
                ),
                "category_counts": {
                    category: int(counts.get(category, 0))
                    for category in TARGET_CATEGORIES
                },
                "payload_metadata": payload.get("metadata") or {},
                "summary_metadata": summary.get("metadata") or {},
                "roof_pipeline_counts": {
                    "flat_surface_count": len(
                        ((roof.get("roof_surfaces") or {}).get("flat")) or []
                    ),
                    "oblique_surface_count": len(
                        ((roof.get("roof_surfaces") or {}).get("oblique")) or []
                    ),
                    "roof_evidence_room_count": _roof_evidence_room_count(roof),
                },
                "building_part_count": len(summary.get("building_parts") or []),
                "area_metrics": _area_metrics(summary, payload, roof),
                "buckets": categories,
                "missing_roof_class": _classify_missing_roof_payload(
                    counts=counts,
                    roof=roof,
                    summary=summary,
                    part_payload=payload,
                ),
                "roof_mode": _roof_mode(payload.get("metadata") or {}),
            }
            results.append(record)
            aggregate["ok"] += 1
            bucket_counts.update(categories)
            for category in TARGET_CATEGORIES:
                category_surface_totals[category] += counts.get(category, 0)
                if counts.get(category, 0) > 0:
                    category_building_counts[category] += 1
        except Exception as exc:  # pragma: no cover - batch diagnostics
            elapsed = perf_counter() - started
            error = {
                "uuid": uuid,
                "status": "error",
                "timings_s": {"total": round(elapsed, 6)},
                "error": repr(exc),
            }
            errors.append(error)
            results.append(error)
            aggregate["failed"] += 1

        report = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "aggregate": {
                **aggregate,
                "bucket_counts": dict(sorted(bucket_counts.items())),
                "category_building_counts": dict(
                    sorted(category_building_counts.items())
                ),
                "category_surface_totals": dict(
                    sorted(category_surface_totals.items())
                ),
            },
            "results": results,
            "errors": errors,
        }
        _write_report(output_json, report)

    ok_results = [row for row in results if row.get("status") == "ok"]
    total_runtime = sum(row["timings_s"]["total"] for row in ok_results)
    total_surfaces = sum(row["renderable_surface_total"] for row in ok_results)
    if ok_results:
        aggregate["avg_runtime_s"] = total_runtime / len(ok_results)
        aggregate["avg_renderable_surface_total"] = total_surfaces / len(ok_results)
    aggregate["bucket_counts"] = dict(sorted(bucket_counts.items()))
    aggregate["category_building_counts"] = dict(
        sorted(category_building_counts.items())
    )
    aggregate["category_surface_totals"] = dict(sorted(category_surface_totals.items()))
    area_metrics_rows = [row.get("area_metrics") or {} for row in ok_results]
    roof_ratio_values = [
        float(row["roof_area_ratio"])
        for row in area_metrics_rows
        if row.get("roof_area_ratio") is not None
    ]
    ceiling_ratio_values = [
        float(row["ceiling_area_ratio"])
        for row in area_metrics_rows
        if row.get("ceiling_area_ratio") is not None
    ]
    shell_ceiling_ratio_values = [
        float(row["shell_ceiling_area_ratio"])
        for row in area_metrics_rows
        if row.get("shell_ceiling_area_ratio") is not None
    ]
    roof_overlap_values = [
        float(row["roof_overlap_ratio_vs_heuristic"])
        for row in area_metrics_rows
        if row.get("roof_overlap_ratio_vs_heuristic") is not None
    ]
    semantic_ceiling_overlap_values = [
        float(row["semantic_ceiling_overlap_ratio_vs_heuristic"])
        for row in area_metrics_rows
        if row.get("semantic_ceiling_overlap_ratio_vs_heuristic") is not None
    ]
    shell_ceiling_overlap_values = [
        float(row["shell_ceiling_overlap_ratio_vs_heuristic"])
        for row in area_metrics_rows
        if row.get("shell_ceiling_overlap_ratio_vs_heuristic") is not None
    ]
    aggregate["area_metrics"] = {
        "ontology_roof_area_m2": round(
            sum(
                float(row.get("ontology_roof_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "heuristic_roof_area_m2": round(
            sum(
                float(row.get("heuristic_roof_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "roof_area_ratio_avg": round(sum(roof_ratio_values) / len(roof_ratio_values), 6)
        if roof_ratio_values
        else None,
        "roof_overlap_ratio_vs_heuristic_avg": round(
            sum(roof_overlap_values) / len(roof_overlap_values), 6
        )
        if roof_overlap_values
        else None,
        "roof_overlap_projected_area_m2": round(
            sum(
                float(row.get("roof_overlap_projected_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "roof_heuristic_only_projected_area_m2": round(
            sum(
                float(row.get("roof_heuristic_only_projected_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "unresolved_area_m2": round(
            sum(
                float(row.get("unresolved_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "ontology_ceiling_area_m2": round(
            sum(
                float(row.get("ontology_ceiling_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "ontology_shell_ceiling_area_m2": round(
            sum(
                float(row.get("ontology_shell_ceiling_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "heuristic_ceiling_area_m2": round(
            sum(
                float(row.get("heuristic_ceiling_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "ceiling_area_ratio_avg": round(
            sum(ceiling_ratio_values) / len(ceiling_ratio_values), 6
        )
        if ceiling_ratio_values
        else None,
        "shell_ceiling_area_ratio_avg": round(
            sum(shell_ceiling_ratio_values) / len(shell_ceiling_ratio_values), 6
        )
        if shell_ceiling_ratio_values
        else None,
        "semantic_ceiling_overlap_ratio_vs_heuristic_avg": round(
            sum(semantic_ceiling_overlap_values) / len(semantic_ceiling_overlap_values),
            6,
        )
        if semantic_ceiling_overlap_values
        else None,
        "semantic_ceiling_overlap_projected_area_m2": round(
            sum(
                float(row.get("semantic_ceiling_overlap_projected_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "semantic_ceiling_heuristic_only_projected_area_m2": round(
            sum(
                float(
                    row.get("semantic_ceiling_heuristic_only_projected_area_m2", 0.0)
                    or 0.0
                )
                for row in area_metrics_rows
            ),
            6,
        ),
        "shell_ceiling_overlap_ratio_vs_heuristic_avg": round(
            sum(shell_ceiling_overlap_values) / len(shell_ceiling_overlap_values), 6
        )
        if shell_ceiling_overlap_values
        else None,
        "shell_ceiling_overlap_projected_area_m2": round(
            sum(
                float(row.get("shell_ceiling_overlap_projected_area_m2", 0.0) or 0.0)
                for row in area_metrics_rows
            ),
            6,
        ),
        "shell_ceiling_heuristic_only_projected_area_m2": round(
            sum(
                float(
                    row.get("shell_ceiling_heuristic_only_projected_area_m2", 0.0)
                    or 0.0
                )
                for row in area_metrics_rows
            ),
            6,
        ),
    }

    def _sorted_rows(predicate) -> list[dict]:
        return sorted(
            (row for row in ok_results if predicate(row)),
            key=lambda row: (
                -row["category_counts"].get("unresolved_region", 0),
                row["category_counts"].get("exterior_roof", 0),
                -row["renderable_surface_total"],
                row["uuid"],
            ),
        )

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "aggregate": aggregate,
        "missing_roof_class_counts": dict(
            sorted(
                Counter(
                    row["missing_roof_class"]
                    for row in ok_results
                    if row.get("missing_roof_class")
                ).items()
            )
        ),
        "roof_mode_counts": dict(
            sorted(
                Counter(
                    row["roof_mode"] for row in ok_results if row.get("roof_mode")
                ).items()
            )
        ),
        "top_missing_roof_examples": sorted(
            (
                row
                for row in ok_results
                if row["category_counts"].get("exterior_roof", 0) == 0
            ),
            key=lambda row: (-row["renderable_surface_total"], row["uuid"]),
        )[:15],
        "top_unresolved_examples": _sorted_rows(
            lambda row: row["category_counts"].get("unresolved_region", 0) > 0
        )[:15],
        "top_heaviest_payloads": sorted(
            ok_results,
            key=lambda row: (-row["renderable_surface_total"], row["uuid"]),
        )[:15],
        "results": results,
        "errors": errors,
    }
    _write_report(output_json, report)
    _write_markdown(output_md, report)
    print(output_json)
    print(output_md)
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "top_missing_roof_examples": [
                    row["uuid"] for row in report["top_missing_roof_examples"][:10]
                ],
                "top_unresolved_examples": [
                    row["uuid"] for row in report["top_unresolved_examples"][:10]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
