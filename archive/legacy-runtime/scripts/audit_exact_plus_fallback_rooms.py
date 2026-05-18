from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

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


def _room_bucket(room_summary: dict) -> str:
    if bool(room_summary.get("has_oblique_atom")):
        return "oblique_atom_room"
    if bool(room_summary.get("has_candidate_upper_void_relation")):
        return "upper_void_candidate_room"
    if bool(room_summary.get("has_candidate_attic_relation")):
        return "attic_candidate_room"
    if bool(room_summary.get("strong_knee_wall_signal")):
        return "knee_wall_signal_room"
    if bool(room_summary.get("strong_perimeter_sloped")):
        return "strong_perimeter_sloped_room"
    if int(room_summary.get("roof_evidence_score", 0) or 0) >= 4:
        return "high_roof_evidence_room"
    return "other_fallback_room"


def _build_payload(uuid: str) -> tuple[dict, dict]:
    merged_path = PIPELINE_ROOT / uuid / "merged.json"
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
    payloads = _build_ontology_part_payloads(
        uuid=uuid,
        summary=summary,
        part_graph_room_ids=part_graph_room_ids,
        topology_cell_complex=topology_cell_complex,
        roof_cell_complex=roof.get("roof_cell_complex") or {},
        occupied_room_cell_complex=roof.get("occupied_room_cell_complex") or {},
        building=building,
        roof=roof,
    )
    return summary, payloads


def main() -> None:
    audit_path = Path(".context/full_model_full_building_audit.json")
    json_path = Path(".context/exact_plus_fallback_room_audit.json")
    md_path = Path(".context/exact_plus_fallback_room_audit.md")
    audit = json.loads(audit_path.read_text())
    target_uuids = [
        row["uuid"]
        for row in (audit.get("results") or [])
        if row.get("status") == "ok"
        and int(
            ((row.get("payload_metadata") or {}).get("roof_fallback_surface_count", 0))
            or 0
        )
        > 0
    ]

    aggregate = {
        "building_count": len(target_uuids),
        "fallback_room_count": 0,
        "roomless_fallback_surface_count": 0,
        "avg_runtime_s": 0.0,
        "room_bucket_counts": {},
    }
    bucket_counts: Counter[str] = Counter()
    results: list[dict] = []

    def _write_report() -> None:
        aggregate["room_bucket_counts"] = dict(sorted(bucket_counts.items()))
        report = {
            "generated_from": str(audit_path),
            "aggregate": aggregate,
            "results": results,
            "example_buildings": [
                {
                    "uuid": row["uuid"],
                    "roof_fallback_surface_count": row["roof_fallback_surface_count"],
                    "fallback_rooms": row["fallback_rooms"][:5],
                    "roomless_fallback_surfaces": row["roomless_fallback_surfaces"][:5],
                }
                for row in sorted(
                    results,
                    key=lambda row: (-row["roof_fallback_surface_count"], row["uuid"]),
                )[:15]
            ],
        }
        json_path.write_text(json.dumps(report, indent=2))
        lines = [
            "# Exact Plus Fallback Room Audit",
            "",
            f"- source_audit: `{audit_path}`",
            f"- building_count: `{aggregate['building_count']}`",
            f"- fallback_room_count: `{aggregate['fallback_room_count']}`",
            f"- roomless_fallback_surface_count: "
            f"`{aggregate['roomless_fallback_surface_count']}`",
            f"- avg_runtime_s: `{aggregate['avg_runtime_s']:.3f}`",
            "",
            "## Room Buckets",
            "",
        ]
        for bucket, count in sorted(aggregate["room_bucket_counts"].items()):
            lines.append(f"- `{bucket}`: `{count}`")
        lines.extend(["", "## Example Buildings", ""])
        for row in report["example_buildings"]:
            lines.append(
                f"- `{row['uuid']}` "
                f"fallback_surfaces=`{row['roof_fallback_surface_count']}`"
            )
            for room in row["fallback_rooms"]:
                lines.append(
                    f"  room `{room['room_index']}` bucket=`{room['bucket']}` "
                    f"count=`{room['fallback_surface_count']}` "
                    f"evidence=`{room['room_summary']['roof_evidence_score']}`"
                )
            for surface in row["roomless_fallback_surfaces"]:
                lines.append(
                    f"  roomless hypothesis=`{surface['roof_hypothesis_id']}` "
                    f"story=`{surface['story']}` "
                    f"has_patch_for_hypothesis=`{surface['has_patch_for_hypothesis']}`"
                )
        md_path.write_text("\n".join(lines))
        return report

    for uuid in target_uuids:
        started = perf_counter()
        summary, payloads = _build_payload(uuid)
        payload = payloads[FULL_BUILDING_PART_ID]
        room_summaries = summary.get("room_summaries") or {}
        patch_hypotheses = {
            str(patch.get("roof_hypothesis_id") or "")
            for patch in (summary.get("oblique_coverage_patches") or [])
            if isinstance(patch, dict) and patch.get("roof_hypothesis_id")
        }
        fallback_by_room: dict[int, list[dict]] = {}
        roomless_fallback_surfaces: list[dict] = []
        for surface in payload.get("renderable_surfaces") or []:
            if str(surface.get("category") or "") != "exterior_roof":
                continue
            if str(surface.get("source_kind") or "") != "roof_surface_fallback":
                continue
            room_index = surface.get("room_index")
            if not isinstance(room_index, int):
                roomless_fallback_surfaces.append(
                    {
                        "roof_hypothesis_id": surface.get("roof_hypothesis_id"),
                        "story": surface.get("story"),
                        "source_id": surface.get("source_id"),
                        "has_patch_for_hypothesis": str(
                            surface.get("roof_hypothesis_id") or ""
                        )
                        in patch_hypotheses,
                    }
                )
                continue
            fallback_by_room.setdefault(room_index, []).append(surface)
        room_rows: list[dict] = []
        for room_index, surfaces in sorted(fallback_by_room.items()):
            room_id = f"room:{room_index}"
            room_summary = room_summaries.get(room_id) or {}
            bucket = _room_bucket(room_summary)
            bucket_counts[bucket] += 1
            room_rows.append(
                {
                    "room_index": room_index,
                    "room_id": room_id,
                    "fallback_surface_count": len(surfaces),
                    "bucket": bucket,
                    "room_summary": {
                        "roles": room_summary.get("roles") or [],
                        "roof_evidence_score": room_summary.get("roof_evidence_score"),
                        "has_oblique_atom": bool(room_summary.get("has_oblique_atom")),
                        "partially_covered_by_sloped_roof": bool(
                            room_summary.get("partially_covered_by_sloped_roof")
                        ),
                        "strong_perimeter_sloped": bool(
                            room_summary.get("strong_perimeter_sloped")
                        ),
                        "strong_knee_wall_signal": bool(
                            room_summary.get("strong_knee_wall_signal")
                        ),
                        "has_candidate_attic_relation": bool(
                            room_summary.get("has_candidate_attic_relation")
                        ),
                        "has_candidate_upper_void_relation": bool(
                            room_summary.get("has_candidate_upper_void_relation")
                        ),
                    },
                }
            )
        elapsed = perf_counter() - started
        aggregate["fallback_room_count"] += len(room_rows)
        aggregate["roomless_fallback_surface_count"] += len(roomless_fallback_surfaces)
        results.append(
            {
                "uuid": uuid,
                "timings_s": {"total": round(elapsed, 6)},
                "roof_cell_count": int(
                    (payload.get("metadata") or {}).get("roof_cell_count", 0) or 0
                ),
                "roof_exact_flat_surface_count": int(
                    (payload.get("metadata") or {}).get(
                        "roof_exact_flat_surface_count", 0
                    )
                    or 0
                ),
                "roof_coverage_patch_surface_count": int(
                    (payload.get("metadata") or {}).get(
                        "roof_coverage_patch_surface_count", 0
                    )
                    or 0
                ),
                "roof_fallback_surface_count": int(
                    (payload.get("metadata") or {}).get(
                        "roof_fallback_surface_count", 0
                    )
                    or 0
                ),
                "fallback_rooms": room_rows,
                "roomless_fallback_surfaces": roomless_fallback_surfaces,
            }
        )
        aggregate["avg_runtime_s"] = sum(
            row["timings_s"]["total"] for row in results
        ) / len(results)
        _write_report()

    if results:
        aggregate["avg_runtime_s"] = sum(
            row["timings_s"]["total"] for row in results
        ) / len(results)
    report = _write_report()
    print(json.dumps(report["aggregate"], indent=2))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
