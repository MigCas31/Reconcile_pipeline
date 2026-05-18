from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

from reconcile.extract3d.builder import extract_building
from reconcile.roof_algorithms_py import run_roof_algorithms
from reconcile_v2.graph_builder import build_topology_graph

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "pipeline-outputs"
SCAN_CACHE_ROOT = ROOT / ".scan-cache"
DEFAULT_OUTPUT = ROOT / ".context" / "building-runtime-audit.json"


def _row_failure_modes(row: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    status = str(row.get("status") or "")
    if status == "topology_error":
        return ["topology_error"]
    if status == "extract_error":
        return ["extract_error"]
    if status == "roof_error":
        return ["roof_error"]

    room_count = int(row.get("room_count") or 0)
    covered_room_count = int(row.get("occupied_room_count") or 0)
    exterior_wall_faces = int(row.get("occupied_exterior_wall_faces") or 0)
    fallback_cell_count = int(row.get("occupied_fallback_cell_count") or 0)
    occupied_cell_count = int(row.get("occupied_cell_count") or 0)
    topology_s = float(row.get("timings_s", {}).get("topology", 0.0) or 0.0)
    total_s = float(row.get("timings_s", {}).get("total", 0.0) or 0.0)

    if room_count > 0 and covered_room_count < room_count:
        modes.append("occupied_rooms_uncovered")
    if occupied_cell_count > 0 and exterior_wall_faces == 0:
        modes.append("occupied_shell_missing_exterior_walls")
    if (
        occupied_cell_count > 0
        and fallback_cell_count / max(occupied_cell_count, 1) >= 0.4
    ):
        modes.append("occupied_shell_high_fallback_ratio")
    if topology_s >= 5.0:
        modes.append("slow_topology")
    if total_s >= 8.0:
        modes.append("slow_total_pipeline")
    if int(row.get("warning_count") or 0) > 0:
        modes.append("warnings_emitted")
    return modes or ["ok"]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    partial_rows = [row for row in rows if row.get("status") == "partial"]
    error_rows = [
        row for row in rows if str(row.get("status") or "").endswith("_error")
    ]
    return {
        "building_count": total,
        "ok_count": len(ok_rows),
        "partial_count": len(partial_rows),
        "error_count": len(error_rows),
        "avg_total_runtime_s": round(
            sum(
                float(row.get("timings_s", {}).get("total", 0.0) or 0.0) for row in rows
            )
            / max(total, 1),
            3,
        ),
        "avg_topology_runtime_s": round(
            sum(
                float(row.get("timings_s", {}).get("topology", 0.0) or 0.0)
                for row in rows
            )
            / max(total, 1),
            3,
        ),
        "max_total_runtime_s": round(
            max(
                (
                    float(row.get("timings_s", {}).get("total", 0.0) or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
            3,
        ),
        "max_topology_runtime_s": round(
            max(
                (
                    float(row.get("timings_s", {}).get("topology", 0.0) or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
            3,
        ),
        "failure_mode_counts": {
            mode: sum(1 for row in rows if mode in (row.get("failure_modes") or []))
            for mode in sorted(
                {mode for row in rows for mode in (row.get("failure_modes") or [])}
            )
        },
    }


def _write_report(
    output_path: Path, rows: list[dict[str, Any]], started_at: float
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows, key=lambda row: str(row.get("uuid") or ""))
    report = {
        "generated_at_epoch_s": round(time.time(), 3),
        "elapsed_s": round(time.time() - started_at, 3),
        "summary": _summarize(ordered_rows),
        "rows": ordered_rows,
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2))
    tmp_path.replace(output_path)


def _candidate_uuids(limit: int | None, uuids: list[str] | None) -> list[str]:
    if uuids:
        return uuids[: limit or None]
    found = sorted(path.parent.name for path in PIPELINE_ROOT.glob("*/merged.json"))
    return found[: limit or None]


def _load_existing_rows(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        payload = json.loads(output_path.read_text())
    except Exception:
        return []
    rows = payload.get("rows") or []
    return [row for row in rows if isinstance(row, dict) and row.get("uuid")]


def _run_one(uuid: str) -> dict[str, Any]:
    merged_path = PIPELINE_ROOT / uuid / "merged.json"
    if not merged_path.exists():
        return {
            "uuid": uuid,
            "status": "topology_error",
            "timings_s": {"topology": 0.0, "extract": 0.0, "roof": 0.0, "total": 0.0},
            "error": "missing merged.json",
            "failure_modes": ["topology_error"],
        }

    t0 = time.time()
    warning_messages: list[str] = []
    graph = None
    building = None
    roof = None
    topology_s = extract_s = roof_s = 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            graph = build_topology_graph(
                merged_path=merged_path, scan_dir=SCAN_CACHE_ROOT, uuid=uuid
            )
        except Exception as exc:
            topology_s = time.time() - t0
            row = {
                "uuid": uuid,
                "status": "topology_error",
                "timings_s": {
                    "topology": round(topology_s, 3),
                    "extract": 0.0,
                    "roof": 0.0,
                    "total": round(time.time() - t0, 3),
                },
                "error": str(exc),
            }
            row["failure_modes"] = _row_failure_modes(row)
            return row
        topology_s = time.time() - t0

        t1 = time.time()
        try:
            building = extract_building(
                uuid=uuid,
                pipeline_dir=PIPELINE_ROOT,
                scan_cache_root=SCAN_CACHE_ROOT,
                load_topology_graph=False,
            )
        except Exception as exc:
            extract_s = time.time() - t1
            row = {
                "uuid": uuid,
                "status": "extract_error",
                "timings_s": {
                    "topology": round(topology_s, 3),
                    "extract": round(extract_s, 3),
                    "roof": 0.0,
                    "total": round(time.time() - t0, 3),
                },
                "error": str(exc),
            }
            row["failure_modes"] = _row_failure_modes(row)
            return row
        extract_s = time.time() - t1

        t2 = time.time()
        try:
            roof = run_roof_algorithms(building or {}, graph=graph)
        except Exception as exc:
            roof_s = time.time() - t2
            row = {
                "uuid": uuid,
                "status": "roof_error",
                "timings_s": {
                    "topology": round(topology_s, 3),
                    "extract": round(extract_s, 3),
                    "roof": round(roof_s, 3),
                    "total": round(time.time() - t0, 3),
                },
                "error": str(exc),
            }
            row["failure_modes"] = _row_failure_modes(row)
            return row
        roof_s = time.time() - t2

        warning_messages = [str(w.message) for w in caught]

    rooms = (building or {}).get("rooms") or []
    occupied = (roof or {}).get("occupied_room_cell_complex") or {}
    occupied_meta = occupied.get("metadata") or {}
    roof_cells_meta = ((roof or {}).get("roof_cell_complex") or {}).get(
        "metadata"
    ) or {}

    row = {
        "uuid": uuid,
        "status": "ok",
        "room_count": len(rooms),
        "graph_node_count": len(getattr(graph, "nodes", []) or []),
        "graph_edge_count": len(getattr(graph, "edges", []) or []),
        "timings_s": {
            "topology": round(topology_s, 3),
            "extract": round(extract_s, 3),
            "roof": round(roof_s, 3),
            "total": round(time.time() - t0, 3),
        },
        "occupied_cell_count": int(occupied_meta.get("cell_count") or 0),
        "occupied_room_count": int(occupied_meta.get("room_count") or 0),
        "occupied_fallback_cell_count": int(
            occupied_meta.get("fallback_cell_count") or 0
        ),
        "occupied_atom_bound_cell_count": int(
            occupied_meta.get("atom_bound_cell_count") or 0
        ),
        "occupied_exterior_wall_faces": int(
            (occupied_meta.get("face_class_counts") or {}).get("exterior_wall") or 0
        ),
        "occupied_interior_wall_faces": int(
            (occupied_meta.get("face_class_counts") or {}).get("interior_wall") or 0
        ),
        "roof_cell_count": int(roof_cells_meta.get("cell_count") or 0),
        "roof_attic_cell_count": int(roof_cells_meta.get("attic_cell_count") or 0),
        "roof_upper_void_cell_count": int(
            roof_cells_meta.get("upper_void_cell_count") or 0
        ),
        "warning_count": len(warning_messages),
        "warnings": warning_messages[:10],
    }
    if int(row["occupied_room_count"]) < int(row["room_count"]):
        row["status"] = "partial"
    row["failure_modes"] = _row_failure_modes(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checkpointed building runtime and ontology audit."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--uuid", dest="uuids", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    existing_rows = _load_existing_rows(args.output) if args.resume else []
    rows_by_uuid = {str(row["uuid"]): row for row in existing_rows}
    uuids = _candidate_uuids(args.limit, args.uuids)

    try:
        for index, uuid in enumerate(uuids, start=1):
            if args.resume and uuid in rows_by_uuid:
                continue
            row = _run_one(uuid)
            rows_by_uuid[uuid] = row
            _write_report(args.output, list(rows_by_uuid.values()), started_at)
            print(
                f"[{index}/{len(uuids)}] {uuid} "
                f"status={row['status']} total={row['timings_s']['total']:.3f}s "
                f"topology={row['timings_s']['topology']:.3f}s "
                f"modes={','.join(row.get('failure_modes') or [])}"
            )
    finally:
        _write_report(args.output, list(rows_by_uuid.values()), started_at)

    report = json.loads(args.output.read_text())
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
