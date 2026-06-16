"""Corpus runner for kinetic ceiling reconstruction step traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reconcile_tiers.ceiling_reconstruction.trace_export import (
    SELECTION,
    build_ksr_room_trace,
)
from reconcile_tiers.room_postprocessing.segment_tier_room_payload import (
    build_segment_tier_room_payload,
)


def _error_key(exc: BaseException) -> str:
    return type(exc).__name__


def export_kinetic_ceiling_steps_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    max_buildings: int = 10,
    corner_tol: float = 0.02,
    room_source: str = "segment-tier",
    segment_corner_tol: float = 0.05,
    segment_adjacency_tol: float = 0.5,
    write_segment_payload: bool = True,
    graph_cut_lambda: float = 0.75,
    bbox_margin: float = 1.0,
    k_intersections: int = 2,
) -> dict[str, Any]:
    """Export per-room KSR pipeline frames for the traces viewer."""
    use_segment_tier = room_source == "segment-tier"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    payload_paths = sorted(pipeline_dir.glob("*/tier_payload.json"))[:max_buildings]

    for payload_path in payload_paths:
        uuid = payload_path.parent.name
        try:
            payload = json.loads(payload_path.read_text())
        except Exception as exc:
            failure_counts[_error_key(exc)] += 1
            continue

        trace_payload = payload
        if use_segment_tier:
            try:
                trace_payload = build_segment_tier_room_payload(
                    payload,
                    corner_tol=segment_corner_tol,
                    adjacency_tol=segment_adjacency_tol,
                )
            except Exception as exc:
                failure_counts[_error_key(exc)] += 1
                continue
            if write_segment_payload:
                segment_path = (
                    payload_path.parent / "tier_payload_segment_tier_rooms.json"
                )
                to_write = {
                    k: v
                    for k, v in trace_payload.items()
                    if k != "segment_room_graph"
                }
                segment_path.write_text(
                    json.dumps(to_write, indent=2, sort_keys=True)
                )

        rooms = trace_payload.get("rooms") or []
        for room_index, room in enumerate(rooms):
            try:
                viewer_trace = build_ksr_room_trace(
                    trace_payload,
                    room,
                    corner_tol=corner_tol,
                    graph_cut_lambda=graph_cut_lambda,
                    bbox_margin=bbox_margin,
                    k_intersections=k_intersections,
                )
            except Exception as exc:
                failure_counts[_error_key(exc)] += 1
                continue
            stop_reason = str(viewer_trace["stop"]["reason"])
            status_counts[stop_reason] += 1
            trace_path = trace_dir / f"{uuid}__room_{room_index}.json"
            trace_path.write_text(
                json.dumps(viewer_trace, indent=2, sort_keys=True)
            )
            ksr_summary = viewer_trace.get("repair_summary") or {}
            record: dict[str, Any] = {
                "uuid": uuid,
                "room_index": room_index,
                "part_index": room_index,
                "locator_id": f"kinetic-ceiling-steps:{room_index}",
                "trace": trace_path.relative_to(output_dir).as_posix(),
                "stop_reason": stop_reason,
                "step_count": len(viewer_trace["frames"]),
                "frame_count": len(viewer_trace["frames"]),
                "initial_counts": viewer_trace["frames"][0]["counts"],
                "final_counts": viewer_trace["frames"][-1]["counts"],
                "ceiling_face_count": ksr_summary.get("ceiling_face_count"),
                "cell_count": ksr_summary.get("cell_count"),
                "graph_cut_status": ksr_summary.get("graph_cut_status"),
                "story": viewer_trace.get("story"),
                "room_source": room_source,
            }
            if use_segment_tier:
                record["segment_room_locator_id"] = room.get("locator_id")
                seg_graph = trace_payload.get("segment_room_graph") or {}
                seg_node = next(
                    (
                        n
                        for n in seg_graph.get("nodes") or []
                        if n.get("id") == room.get("locator_id")
                    ),
                    None,
                )
                if seg_node:
                    record["segment_room_area_m2"] = seg_node.get("area_m2")
                    record["floor_area_m2"] = seg_node.get("floor_area_m2")
            records.append(record)

    index = {
        "schema_version": 1,
        "domain": SELECTION,
        "selection": SELECTION,
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "max_buildings": max_buildings,
            "corner_tol": corner_tol,
            "room_source": room_source,
            "segment_corner_tol": segment_corner_tol,
            "segment_adjacency_tol": segment_adjacency_tol,
            "graph_cut_lambda": graph_cut_lambda,
            "bbox_margin": bbox_margin,
            "k_intersections": k_intersections,
        },
        "summary": {
            "buildings": len(payload_paths),
            "records": len(records),
            "status_counts": dict(status_counts.most_common()),
            "failure_counts": dict(failure_counts.most_common()),
            "ceiling_extracted_rooms": sum(
                1 for r in records if r["stop_reason"] == "ceiling_extracted"
            ),
            "partition_failed_rooms": sum(
                1 for r in records if r["stop_reason"] == "partition_failed"
            ),
            "graph_cut_failed_rooms": sum(
                1 for r in records if r["stop_reason"] == "label_failed"
            ),
            "no_ceiling_rooms": sum(
                1 for r in records if r["stop_reason"] == "no_ceiling_faces"
            ),
            "insufficient_tiles_rooms": sum(
                1 for r in records if r["stop_reason"] == "insufficient_tiles"
            ),
        },
        "records": records,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    index_url = f"/{index_path.as_posix()}"
    print(
        "Kinetic ceiling steps viewer:\n"
        f"  http://127.0.0.1:8080/reconcile_tiers/web/viewer-polyhedron-traces.html"
        f"?index={index_url}",
        flush=True,
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export kinetic ceiling reconstruction step traces.",
    )
    parser.add_argument("--pipeline-dir", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--domain",
        choices=("kinetic-ceiling-steps",),
        default="kinetic-ceiling-steps",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".context/kinetic-ceiling-traces"),
    )
    parser.add_argument("--max-buildings", type=int, default=10)
    parser.add_argument("--corner-tol", type=float, default=0.02)
    parser.add_argument(
        "--room-source",
        choices=("tier", "segment-tier"),
        default="segment-tier",
    )
    parser.add_argument("--segment-corner-tol", type=float, default=0.05)
    parser.add_argument("--segment-adjacency-tol", type=float, default=0.5)
    parser.add_argument("--graph-cut-lambda", type=float, default=0.75)
    parser.add_argument("--bbox-margin", type=float, default=1.0)
    parser.add_argument("--k-intersections", type=int, default=2)
    parser.add_argument(
        "--no-write-segment-payload",
        action="store_true",
        help="Skip writing tier_payload_segment_tier_rooms.json",
    )
    args = parser.parse_args(argv)

    export_kinetic_ceiling_steps_traces(
        pipeline_dir=args.pipeline_dir,
        output_dir=args.out_dir,
        max_buildings=args.max_buildings,
        corner_tol=args.corner_tol,
        room_source=args.room_source,
        segment_corner_tol=args.segment_corner_tol,
        segment_adjacency_tol=args.segment_adjacency_tol,
        write_segment_payload=not args.no_write_segment_payload,
        graph_cut_lambda=args.graph_cut_lambda,
        bbox_margin=args.bbox_margin,
        k_intersections=args.k_intersections,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
