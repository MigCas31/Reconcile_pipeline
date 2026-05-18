"""Corpus runner for room-shell topology trace exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers.polyhedron.cell_selector import (
    audit_payload_rooms,
    select_payload_cells_v2,
)
from reconcile_tiers.polyhedron.half_edge import build_from_planar_polygons
from reconcile_tiers.polyhedron.payload_adapter import (
    _payload_footprint_polygon,
    build_room_shell_from_tier_payload,
    payload_envelope_candidates_from_tier_payload,
    payload_faces_from_plane_evidence,
    payload_faces_from_tier_payload,
)
from reconcile_tiers.polyhedron.trace_export import export_topology_resolution

WriteMode = Literal["interesting", "all", "none"]


def export_corpus_room_shell_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    selection: str = "unique",
    max_steps: int = 32,
    write_mode: WriteMode = "interesting",
    valid_sample_limit: int = 20,
    coord_tol: float = 1e-3,
    corner_tol: float = 0.02,
    ceiling_limit: int = 1,
    min_ceiling_overlap_area_m2: float = 0.01,
) -> dict[str, Any]:
    """Try every room shell under ``pipeline_dir`` and write trace exports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    if write_mode != "none":
        trace_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    build_failures: list[dict[str, Any]] = []
    build_failure_counts: Counter[str] = Counter()
    stop_reason_counts: Counter[str] = Counter()
    written_count = 0
    valid_sample_count = 0
    attempted_rooms = 0
    built_rooms = 0

    payload_paths = sorted(pipeline_dir.glob("*/tier_payload.json"))
    for payload_path in payload_paths:
        uuid = payload_path.parent.name
        try:
            payload = json.loads(payload_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            build_failure_counts[_error_key(exc)] += 1
            build_failures.append(
                {
                    "uuid": uuid,
                    "room_index": None,
                    "error": str(exc),
                }
            )
            continue

        rooms = payload.get("rooms", []) or []
        for room_index in range(len(rooms)):
            attempted_rooms += 1
            try:
                polyhedron = build_room_shell_from_tier_payload(
                    payload,
                    room_index,
                    coord_tol=coord_tol,
                    corner_tol=corner_tol,
                    ceiling_limit=ceiling_limit,
                    min_ceiling_overlap_area_m2=min_ceiling_overlap_area_m2,
                )
            except (KeyError, TypeError, ValueError) as exc:
                build_failure_counts[_error_key(exc)] += 1
                if len(build_failures) < 100:
                    build_failures.append(
                        {
                            "uuid": uuid,
                            "room_index": room_index,
                            "error": str(exc),
                        }
                    )
                continue

            built_rooms += 1
            export = export_topology_resolution(
                polyhedron,
                max_steps=max_steps,
                selection=selection,
                edge_tol_m=coord_tol,
            )
            stop_reason = str(export["stop"]["reason"])
            stop_reason_counts[stop_reason] += 1
            interesting = _is_interesting_export(export)

            should_write = write_mode == "all" or (
                write_mode == "interesting" and interesting
            )
            if (
                write_mode == "interesting"
                and not interesting
                and valid_sample_count < valid_sample_limit
            ):
                should_write = True
                valid_sample_count += 1

            trace_relpath = None
            if should_write:
                trace_path = trace_dir / f"{uuid}__room_{room_index}.json"
                trace_path.write_text(json.dumps(export, indent=2, sort_keys=True))
                trace_relpath = trace_path.relative_to(output_dir).as_posix()
                written_count += 1

            records.append(
                {
                    "uuid": uuid,
                    "room_index": room_index,
                    "trace": trace_relpath,
                    "stop_reason": stop_reason,
                    "step_count": len(export["steps"]),
                    "frame_count": len(export["frames"]),
                    "initial_counts": export["frames"][0]["counts"],
                    "final_counts": export["frames"][-1]["counts"],
                    "interesting": interesting,
                }
            )

    index = {
        "schema_version": 1,
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "selection": selection,
            "max_steps": max_steps,
            "write_mode": write_mode,
            "valid_sample_limit": valid_sample_limit,
            "coord_tol": coord_tol,
            "corner_tol": corner_tol,
            "ceiling_limit": ceiling_limit,
            "min_ceiling_overlap_area_m2": min_ceiling_overlap_area_m2,
        },
        "summary": {
            "buildings": len(payload_paths),
            "attempted_rooms": attempted_rooms,
            "built_rooms": built_rooms,
            "failed_rooms": attempted_rooms - built_rooms,
            "written_traces": written_count,
            "interesting_traces": sum(1 for record in records if record["interesting"]),
            "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
            "build_failure_counts": dict(build_failure_counts.most_common()),
        },
        "records": records,
        "build_failure_samples": build_failures,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def export_corpus_envelope_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    selection: str = "unique",
    max_steps: int = 32,
    write_mode: WriteMode = "interesting",
    valid_sample_limit: int = 20,
    coord_tol: float = 1e-3,
    corner_tol: float = 0.02,
    min_top_overlap_ratio: float = 0.60,
    wing_level: bool = True,
    force_cell_selector: bool = False,
) -> dict[str, Any]:
    """Try building/wing-level envelope polyhedra under ``pipeline_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    audit_dir = output_dir / "audit"
    if write_mode != "none":
        trace_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    build_failures: list[dict[str, Any]] = []
    build_failure_counts: Counter[str] = Counter()
    stop_reason_counts: Counter[str] = Counter()
    attempted_parts = 0
    built_parts = 0
    written_count = 0
    valid_sample_count = 0

    payload_paths = sorted(pipeline_dir.glob("*/tier_payload.json"))
    for payload_path in payload_paths:
        uuid = payload_path.parent.name
        try:
            payload = json.loads(payload_path.read_text())
            faces = payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
            evidence_path = payload_path.parent / "plane_evidence.json"
            evidence = (
                json.loads(evidence_path.read_text())
                if evidence_path.exists()
                else None
            )
            faces.extend(
                payload_faces_from_plane_evidence(
                    evidence,
                    corner_tol=corner_tol,
                    include_filtered_candidates=True,
                )
            )
            ceiling_faces = [face for face in faces if face.kind == "ceiling"]
            audit = audit_payload_rooms(
                payload,
                ceiling_faces=ceiling_faces,
                corner_tol=corner_tol,
            )
            audit_path = audit_dir / f"{uuid}.json"
            audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
            candidates = payload_envelope_candidates_from_tier_payload(
                payload,
                ceiling_faces=ceiling_faces,
                force_cell_selector=force_cell_selector,
                wing_level=wing_level,
                min_top_overlap_ratio=min_top_overlap_ratio,
                corner_tol=corner_tol,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            build_failure_counts[_error_key(exc)] += 1
            if len(build_failures) < 100:
                build_failures.append(
                    {"uuid": uuid, "part_index": None, "error": str(exc)}
                )
            continue

        if not candidates:
            build_failure_counts["no envelope candidate"] += 1
            continue

        for part_index, candidate in enumerate(candidates):
            attempted_parts += 1
            try:
                polyhedron = build_from_planar_polygons(
                    [(face.corners, face.plane) for face in candidate.faces],
                    coord_tol=coord_tol,
                )
            except ValueError as exc:
                build_failure_counts[_error_key(exc)] += 1
                if len(build_failures) < 100:
                    build_failures.append(
                        {
                            "uuid": uuid,
                            "part_index": part_index,
                            "locator_id": candidate.locator_id,
                            "error": str(exc),
                        }
                    )
                continue

            built_parts += 1
            export = export_topology_resolution(
                polyhedron,
                max_steps=max_steps,
                selection=selection,
                edge_tol_m=coord_tol,
            )
            stop_reason = str(export["stop"]["reason"])
            stop_reason_counts[stop_reason] += 1
            interesting = _is_interesting_export(export)
            should_write = write_mode == "all" or (
                write_mode == "interesting" and interesting
            )
            if (
                write_mode == "interesting"
                and not interesting
                and valid_sample_count < valid_sample_limit
            ):
                should_write = True
                valid_sample_count += 1

            trace_relpath = None
            if should_write:
                safe_locator = candidate.locator_id.replace(":", "_")
                trace_path = trace_dir / f"{uuid}__{safe_locator}.json"
                trace_path.write_text(json.dumps(export, indent=2, sort_keys=True))
                trace_relpath = trace_path.relative_to(output_dir).as_posix()
                written_count += 1

            records.append(
                {
                    "uuid": uuid,
                    "part_index": part_index,
                    "locator_id": candidate.locator_id,
                    "trace": trace_relpath,
                    "stop_reason": stop_reason,
                    "step_count": len(export["steps"]),
                    "frame_count": len(export["frames"]),
                    "initial_counts": export["frames"][0]["counts"],
                    "final_counts": export["frames"][-1]["counts"],
                    "footprint_area_m2": candidate.footprint_area_m2,
                    "top_source": candidate.top_source,
                    "top_overlap_ratio": candidate.top_overlap_ratio,
                    "selector": candidate.selector,
                    "room_coverage": candidate.room_coverage,
                    "part_plane_groups": candidate.part_plane_groups,
                    "top_label_summary": candidate.top_label_summary,
                    "audit_path": audit_path.relative_to(output_dir).as_posix(),
                    "interesting": interesting,
                }
            )

    index = {
        "schema_version": 1,
        "domain": "envelope-cell-selector"
        if force_cell_selector
        else "envelope",
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "selection": selection,
            "max_steps": max_steps,
            "write_mode": write_mode,
            "valid_sample_limit": valid_sample_limit,
            "coord_tol": coord_tol,
            "corner_tol": corner_tol,
            "min_top_overlap_ratio": min_top_overlap_ratio,
            "wing_level": wing_level,
            "force_cell_selector": force_cell_selector,
        },
        "summary": {
            "buildings": len(payload_paths),
            "attempted_parts": attempted_parts,
            "built_parts": built_parts,
            "written_traces": written_count,
            "interesting_traces": sum(1 for record in records if record["interesting"]),
            "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
            "build_failure_counts": dict(build_failure_counts.most_common()),
        },
        "records": records,
        "build_failure_samples": build_failures,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def export_corpus_room_audit(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    corner_tol: float = 0.02,
) -> dict[str, Any]:
    """Write per-building missing-room/building-part evidence audits."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    build_attempt_counts: Counter[str] = Counter()
    payload_paths = sorted(pipeline_dir.glob("*/tier_payload.json"))

    for payload_path in payload_paths:
        uuid = payload_path.parent.name
        try:
            payload = json.loads(payload_path.read_text())
            evidence_path = payload_path.parent / "plane_evidence.json"
            evidence = (
                json.loads(evidence_path.read_text())
                if evidence_path.exists()
                else None
            )
            faces = payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
            faces.extend(
                payload_faces_from_plane_evidence(
                    evidence,
                    corner_tol=corner_tol,
                    include_filtered_candidates=True,
                )
            )
            audit = audit_payload_rooms(
                payload,
                ceiling_faces=[face for face in faces if face.kind == "ceiling"],
                corner_tol=corner_tol,
            )
        except Exception as exc:
            failure_counts[_error_key(exc)] += 1
            continue
        audit_path = audit_dir / f"{uuid}.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
        summary = audit.get("summary", {})
        reason_counts.update(summary.get("reasons", {}))
        attempts = audit.get("build_attempts", []) or []
        build_attempt_counts.update(
            str(attempt.get("reason") or attempt.get("result") or "unknown")
            for attempt in attempts
        )
        records.append(
            {
                "uuid": uuid,
                "audit_path": audit_path.relative_to(output_dir).as_posix(),
                "rooms_total": summary.get("rooms_total", 0),
                "rooms_ge80": summary.get("rooms_ge80", 0),
                "rooms_ge50": summary.get("rooms_ge50", 0),
                "dropped_rooms": summary.get("dropped_rooms", 0),
                "reasons": summary.get("reasons", {}),
                "build_attempt_reasons": dict(
                    Counter(
                        str(attempt.get("reason") or attempt.get("result") or "unknown")
                        for attempt in attempts
                    ).most_common()
                ),
            }
        )

    summary = {
        "buildings": len(payload_paths),
        "audited_buildings": len(records),
        "rooms_total": sum(int(record["rooms_total"]) for record in records),
        "rooms_ge80": sum(int(record["rooms_ge80"]) for record in records),
        "rooms_ge50": sum(int(record["rooms_ge50"]) for record in records),
        "dropped_rooms": sum(int(record["dropped_rooms"]) for record in records),
        "reason_counts": dict(reason_counts.most_common()),
        "build_attempt_counts": dict(build_attempt_counts.most_common()),
        "failure_counts": dict(failure_counts.most_common()),
    }
    index = {
        "schema_version": 1,
        "domain": "room-audit",
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {"corner_tol": corner_tol},
        "summary": summary,
        "records": records,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def export_manifold_repair_v3_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    max_buildings: int = 10,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
) -> dict[str, Any]:
    """Run the per-room manifold-repair pipeline (§12 IMPLEMENTATION_PLAN.md)
    over the corpus and emit viewer-compatible traces.

    For each room:
      1. ``repair_room`` collects tiles, builds the half-edge polyhedron,
         detects orphan-edge holes, picks fillers via PolyFit ILP, and
         applies them with Geniet ``FACE_CREATION_FROM_EDGE``.
      2. The watertight (or nearly-watertight) polyhedron is emitted as
         one ``EnvelopeCandidate`` per room.
      3. A viewer trace mirrors the ``selector-v2`` format so the
         existing polyhedron-buildings viewer can render the result.
    """
    from reconcile_tiers.polyhedron.manifold_repair import (
        envelope_candidate_from_repair,
        repair_room,
    )

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
        rooms = payload.get("rooms") or []
        for room_index, room in enumerate(rooms):
            try:
                repair = repair_room(
                    payload, room, corner_tol=corner_tol, coord_tol=coord_tol
                )
                envelope = envelope_candidate_from_repair(repair)
            except Exception as exc:
                failure_counts[_error_key(exc)] += 1
                continue
            status_counts[repair.status] += 1
            face_count = (
                len(envelope.faces) if envelope is not None else 0
            )
            tile_face_count = sum(
                1
                for face in (envelope.faces if envelope else [])
                if face.source != "polyhedron_v3_filler"
            )
            filler_face_count = face_count - tile_face_count
            viewer_trace = _manifold_repair_viewer_trace(repair, envelope)
            trace_path = (
                trace_dir / f"{uuid}__room_{room_index}.json"
            )
            trace_path.write_text(json.dumps(viewer_trace, indent=2, sort_keys=True))
            domain_polygon = _floor_polygon_xz(envelope) if envelope else []
            records.append(
                {
                    "uuid": uuid,
                    "part_index": room_index,
                    "locator_id": f"manifold-repair-room:{room_index}",
                    "trace": trace_path.relative_to(output_dir).as_posix(),
                    "stop_reason": repair.status,
                    "step_count": repair.fillers_applied,
                    "frame_count": len(viewer_trace["frames"]),
                    "initial_counts": viewer_trace["frames"][0]["counts"],
                    "final_counts": viewer_trace["frames"][-1]["counts"],
                    "candidate_count": face_count,
                    "selected_count": face_count,
                    "coverage_ratio": 1.0
                    if repair.status == "watertight"
                    else 0.0,
                    "domain_area": (
                        envelope.footprint_area_m2 if envelope else 0.0
                    ),
                    "domain_polygon": domain_polygon,
                    "tile_face_count": tile_face_count,
                    "filler_face_count": filler_face_count,
                    "skipped_tiles": len(repair.build.skipped_tiles),
                    "orphans_remaining": len(repair.build.orphan_half_edges),
                    "closed_holes": len(repair.extraction.closed_chains),
                    "open_chains": len(repair.extraction.open_chains),
                    "emitted_candidate": (
                        envelope.locator_id
                        if envelope is not None
                        and repair.status == "watertight"
                        else None
                    ),
                    "assembly_eligible_candidate": (
                        envelope.locator_id
                        if envelope is not None
                        and repair.status == "watertight"
                        else None
                    ),
                    "story": repair.story,
                }
            )

    _annotate_selector_v2_assembly(records)
    index = {
        "schema_version": 1,
        "domain": "manifold-repair-v3",
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "max_buildings": max_buildings,
            "corner_tol": corner_tol,
            "coord_tol": coord_tol,
        },
        "summary": {
            "buildings": len(payload_paths),
            "records": len(records),
            "status_counts": dict(status_counts.most_common()),
            "failure_counts": dict(failure_counts.most_common()),
            "watertight_rooms": sum(
                1 for r in records if r["stop_reason"] == "watertight"
            ),
            "holes_remaining_rooms": sum(
                1 for r in records if r["stop_reason"] == "holes_remaining"
            ),
            "skipped_rooms": sum(
                1 for r in records if r["stop_reason"] == "skipped"
            ),
            "tile_faces": sum(r["tile_face_count"] for r in records),
            "filler_faces": sum(r["filler_face_count"] for r in records),
            "building_summary": _selector_v2_building_summary(records),
        },
        "records": records,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def export_manifold_repair_steps_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    max_buildings: int = 10,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
    snap_tol: float = 0.05,
) -> dict[str, Any]:
    """Export per-room manifold-repair pipeline frames for the traces viewer.

    Each trace has one frame per stage (collect → coherence → roof XZ clip →
    merge → half-edge → holes → fillers). Load with
    ``viewer-polyhedron-traces.html?index=<out-dir>/index.json``.
    """
    from reconcile_tiers.polyhedron.manifold_repair_trace import (
        SELECTION,
        build_manifold_repair_room_trace,
    )

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
        rooms = payload.get("rooms") or []
        for room_index, room in enumerate(rooms):
            try:
                viewer_trace = build_manifold_repair_room_trace(
                    payload,
                    room,
                    corner_tol=corner_tol,
                    coord_tol=coord_tol,
                    snap_tol=snap_tol,
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
            repair_summary = viewer_trace.get("repair_summary") or {}
            records.append(
                {
                    "uuid": uuid,
                    "room_index": room_index,
                    "part_index": room_index,
                    "locator_id": f"manifold-repair-steps:{room_index}",
                    "trace": trace_path.relative_to(output_dir).as_posix(),
                    "stop_reason": stop_reason,
                    "step_count": len(viewer_trace["frames"]),
                    "frame_count": len(viewer_trace["frames"]),
                    "initial_counts": viewer_trace["frames"][0]["counts"],
                    "final_counts": viewer_trace["frames"][-1]["counts"],
                    "coherence_ok": repair_summary.get("coherence_ok"),
                    "roof_clip_clipped": repair_summary.get("roof_clip_clipped"),
                    "fillers_applied": repair_summary.get("fillers_applied"),
                    "story": viewer_trace.get("story"),
                }
            )

    index = {
        "schema_version": 1,
        "domain": SELECTION,
        "selection": SELECTION,
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "max_buildings": max_buildings,
            "corner_tol": corner_tol,
            "coord_tol": coord_tol,
            "snap_tol": snap_tol,
        },
        "summary": {
            "buildings": len(payload_paths),
            "records": len(records),
            "status_counts": dict(status_counts.most_common()),
            "failure_counts": dict(failure_counts.most_common()),
            "watertight_rooms": sum(
                1 for r in records if r["stop_reason"] == "watertight"
            ),
            "holes_remaining_rooms": sum(
                1 for r in records if r["stop_reason"] == "holes_remaining"
            ),
            "skipped_rooms": sum(
                1 for r in records if r["stop_reason"] == "skipped"
            ),
        },
        "records": records,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    index_url = f"/{index_path.as_posix()}"
    print(
        "Polyhedron steps viewer:\n"
        f"  http://127.0.0.1:8080/reconcile_tiers/web/viewer-polyhedron-traces.html"
        f"?index={index_url}",
        flush=True,
    )
    return index


def export_manifold_repair_building_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    max_buildings: int = 10,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
) -> dict[str, Any]:
    """One trace per BUILDING (not per room). Drops interior shared walls,
    storey boundaries, and duplicate-emission faces so the viewer renders
    only the building envelope (plan §3 Increment 7).
    """
    from reconcile_tiers.polyhedron.building_merge import (
        envelope_candidate_from_building,
        repair_building,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    payload_paths = sorted(pipeline_dir.glob("*/tier_payload.json"))[:max_buildings]

    total_faces = 0
    interior_faces = 0
    duplicate_faces = 0
    for payload_path in payload_paths:
        uuid = payload_path.parent.name
        try:
            payload = json.loads(payload_path.read_text())
        except Exception as exc:
            failure_counts[_error_key(exc)] += 1
            continue
        try:
            building = repair_building(
                payload, corner_tol=corner_tol, coord_tol=coord_tol
            )
            envelope = envelope_candidate_from_building(building)
        except Exception as exc:
            failure_counts[_error_key(exc)] += 1
            continue
        ext_count = sum(1 for f in building.faces if f.kind == "exterior")
        shrwall = sum(
            1 for f in building.faces if f.kind == "interior_shared_wall"
        )
        storey = sum(
            1 for f in building.faces if f.kind == "interior_storey_boundary"
        )
        dup = sum(1 for f in building.faces if f.kind == "duplicate")
        total_faces += len(building.faces)
        interior_faces += shrwall + storey
        duplicate_faces += dup

        viewer_trace = _manifold_repair_building_viewer_trace(
            uuid, building, envelope
        )
        trace_path = trace_dir / f"{uuid}.json"
        trace_path.write_text(
            json.dumps(viewer_trace, indent=2, sort_keys=True)
        )
        records.append(
            {
                "uuid": uuid,
                "part_index": 0,
                "locator_id": f"manifold-repair-building:{uuid}",
                "trace": trace_path.relative_to(output_dir).as_posix(),
                "stop_reason": "watertight",
                "step_count": 0,
                "frame_count": len(viewer_trace["frames"]),
                "initial_counts": viewer_trace["frames"][0]["counts"],
                "final_counts": viewer_trace["frames"][-1]["counts"],
                "candidate_count": len(building.faces),
                "selected_count": ext_count,
                "coverage_ratio": 1.0,
                "domain_area": 0.0,
                "domain_polygon": [],
                "tile_face_count": ext_count,
                "filler_face_count": 0,
                "interior_shared_wall_count": shrwall,
                "interior_storey_boundary_count": storey,
                "duplicate_face_count": dup,
                "room_count": len(building.rooms),
                "emitted_candidate": f"building-envelope-{uuid}",
                "assembly_eligible_candidate": f"building-envelope-{uuid}",
                "story": None,
            }
        )

    _annotate_selector_v2_assembly(records)
    index = {
        "schema_version": 1,
        "domain": "manifold-repair-building",
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "max_buildings": max_buildings,
            "corner_tol": corner_tol,
            "coord_tol": coord_tol,
        },
        "summary": {
            "buildings": len(payload_paths),
            "records": len(records),
            "total_faces": total_faces,
            "interior_faces": interior_faces,
            "duplicate_faces": duplicate_faces,
            "failure_counts": dict(failure_counts),
        },
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )
    print(json.dumps(index["summary"], indent=2, sort_keys=True))
    print(f"Wrote index: {output_dir / 'index.json'}")
    return index


def _manifold_repair_building_viewer_trace(
    uuid: str,
    building: Any,
    envelope: Any,
) -> dict[str, Any]:
    faces_payload: list[dict[str, Any]] = []
    if envelope is not None:
        for i, face in enumerate(envelope.faces):
            faces_payload.append(
                {
                    "id": i,
                    "plane_id": i,
                    "selected": True,
                    "label": face.source or face.kind,
                    "corners": face.corners,
                    "plane": {
                        "a": face.plane.a,
                        "b": face.plane.b,
                        "c": face.plane.c,
                        "d": face.plane.d,
                    },
                }
            )
    frames = [
        _selector_v2_frame("all_candidates", faces_payload),
        _selector_v2_frame("selected_faces", faces_payload),
    ]
    return {
        "schema_version": 1,
        "selection": "manifold-repair-building",
        "tolerances": {},
        "frames": frames,
        "steps": [],
        "stop": {
            "reason": "watertight",
            "message": f"{len(building.rooms)} rooms merged",
            "remaining_issues": [],
            "remaining_events": [],
        },
    }


def _manifold_repair_viewer_trace(
    repair: Any,
    envelope: Any,
) -> dict[str, Any]:
    """Format the repair result as a polyhedron-buildings viewer trace."""

    faces_payload: list[dict[str, Any]] = []
    if envelope is not None:
        for i, face in enumerate(envelope.faces):
            faces_payload.append(
                {
                    "id": i,
                    "plane_id": i,
                    "selected": True,
                    "label": face.source or face.kind,
                    "corners": face.corners,
                    "plane": {
                        "a": face.plane.a,
                        "b": face.plane.b,
                        "c": face.plane.c,
                        "d": face.plane.d,
                    },
                }
            )
    frames = [
        _selector_v2_frame("all_candidates", faces_payload),
        _selector_v2_frame("selected_faces", faces_payload),
    ]
    return {
        "schema_version": 1,
        "selection": "manifold-repair-v3",
        "tolerances": {},
        "frames": frames,
        "steps": [],
        "stop": {
            "reason": repair.status,
            "message": repair.skip_reason or repair.status,
            "remaining_issues": [],
            "remaining_events": [],
        },
    }


def _floor_polygon_xz(envelope: Any) -> list[list[float]]:
    if envelope is None:
        return []
    for face in envelope.faces:
        if face.kind == "floor":
            return [[float(c[0]), float(c[2])] for c in face.corners]
    return []


def export_selector_v2_traces(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    max_buildings: int = 10,
    corner_tol: float = 0.02,
    time_budget_seconds: float = 0.5,
    max_intersections: int = 10_000,
    max_candidates: int = 500,
) -> dict[str, Any]:
    """Write diagnostic Stage A selector traces in viewer-compatible form."""

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
            evidence_path = payload_path.parent / "plane_evidence.json"
            evidence = (
                json.loads(evidence_path.read_text())
                if evidence_path.exists()
                else None
            )
            faces = payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
            faces.extend(
                payload_faces_from_plane_evidence(
                    evidence,
                    corner_tol=corner_tol,
                    include_filtered_candidates=True,
                )
            )
            footprint = _payload_footprint_polygon(
                payload,
                room_buffer_m=0.3,
                footprint_shrink_m=0.3,
                corner_tol=corner_tol,
            )
            if footprint is None:
                raise ValueError("could not derive payload footprint")
            result = select_payload_cells_v2(
                payload,
                footprint=footprint,
                ceiling_faces=[face for face in faces if face.kind == "ceiling"],
                corner_tol=corner_tol,
                time_budget_seconds=time_budget_seconds,
                max_intersections=max_intersections,
                max_candidates=max_candidates,
            )
        except Exception as exc:
            failure_counts[_error_key(exc)] += 1
            continue

        for domain_index, trace in enumerate(result.domain_traces):
            viewer_trace = _selector_v2_viewer_trace(trace)
            trace_path = trace_dir / f"{uuid}__domain_{domain_index}.json"
            trace_path.write_text(json.dumps(viewer_trace, indent=2, sort_keys=True))
            status = str(trace.get("status") or "unknown")
            status_counts[status] += 1
            candidate_count = int(trace.get("candidate_count") or 0)
            selected_count = (
                int(trace.get("selection", {}).get("selected_count", 0))
                if isinstance(trace.get("selection"), dict)
                else 0
            )
            coverage_ratio = (
                trace.get("selection", {})
                .get("energy_breakdown", {})
                .get("coverage_ratio")
                if isinstance(trace.get("selection"), dict)
                else None
            )
            records.append(
                {
                    "uuid": uuid,
                    "part_index": domain_index,
                    "locator_id": f"selector-v2-domain:{domain_index}",
                    "trace": trace_path.relative_to(output_dir).as_posix(),
                    "stop_reason": status,
                    "step_count": 0,
                    "frame_count": len(viewer_trace["frames"]),
                    "initial_counts": viewer_trace["frames"][0]["counts"],
                    "final_counts": viewer_trace["frames"][-1]["counts"],
                    "candidate_count": candidate_count,
                    "selected_count": selected_count,
                    "coverage_ratio": coverage_ratio,
                    "domain_area": trace.get("domain_area"),
                    "domain_polygon": trace.get("domain_polygon"),
                    "selected_coverage_polygons": (
                        _selected_coverage_polygons_json(trace)
                    ),
                    "candidate_cap_exceeded": candidate_count > max_candidates,
                    "low_coverage": (
                        coverage_ratio is not None and float(coverage_ratio) < 0.95
                    ),
                    "emitted_candidate": trace.get("emitted_candidate"),
                    "assembly_eligible_candidate": trace.get(
                        "assembly_eligible_candidate"
                    ),
                }
            )

    _annotate_selector_v2_assembly(records)
    index = {
        "schema_version": 1,
        "domain": "selector-v2",
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "settings": {
            "max_buildings": max_buildings,
            "corner_tol": corner_tol,
            "time_budget_seconds": time_budget_seconds,
            "max_intersections": max_intersections,
            "max_candidates": max_candidates,
        },
        "summary": {
            "buildings": len(payload_paths),
            "records": len(records),
            "status_counts": dict(status_counts.most_common()),
            "failure_counts": dict(failure_counts.most_common()),
            "max_candidates": max(
                (int(record.get("candidate_count") or 0) for record in records),
                default=0,
            ),
            "low_coverage_domains": sum(
                1 for record in records if record.get("low_coverage")
            ),
            "candidate_cap_exceeded_domains": sum(
                1 for record in records if record.get("candidate_cap_exceeded")
            ),
            "emitted_candidates": sum(
                1 for record in records if record.get("emitted_candidate")
            ),
            "building_summary": _selector_v2_building_summary(records),
        },
        "records": records,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def _is_interesting_export(export: dict[str, Any]) -> bool:
    if export["steps"]:
        return True
    stop = export["stop"]
    return bool(
        stop["reason"] != "valid"
        or stop["remaining_issues"]
        or stop["remaining_events"]
    )


def _error_key(exc: Exception) -> str:
    message = str(exc).splitlines()[0]
    if message.startswith("duplicate directed edge"):
        return f"{type(exc).__name__}: duplicate directed edge"
    if message.startswith("non-watertight:"):
        return f"{type(exc).__name__}: non-watertight"
    if len(message) > 120:
        message = message[:117] + "..."
    return f"{type(exc).__name__}: {message}"


def _selector_v2_viewer_trace(trace: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        _selector_v2_viewer_face(candidate)
        for candidate in trace.get("candidates", []) or []
        if isinstance(candidate, dict)
    ]
    selected = [face for face in candidates if face.get("selected")]
    frames = [
        _selector_v2_frame("all_candidates", candidates),
        _selector_v2_frame("selected_faces", selected),
    ]
    status = str(trace.get("status") or "unknown")
    message = trace.get("error") or trace.get("reason") or status
    return {
        "schema_version": 1,
        "selection": "selector-v2",
        "tolerances": {},
        "frames": frames,
        "steps": [],
        "stop": {
            "reason": status,
            "message": str(message),
            "remaining_issues": [],
            "remaining_events": [],
        },
    }


def _selector_v2_frame(label: str, faces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "index": 0 if label == "all_candidates" else 1,
        "label": label,
        "counts": {"faces": len(faces), "vertices": 0, "half_edges": 0},
        "issues": [],
        "events": [],
        "event_error": None,
        "faces": faces,
    }


def _selector_v2_viewer_face(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("face_id"),
        "plane_id": candidate.get("plane_id"),
        "selected": bool(candidate.get("selected")),
        "label": candidate.get("confidence_label"),
        "corners": candidate.get("corners") or [],
        "plane": candidate.get("plane") or {},
    }


def _selector_v2_building_summary(
    records: list[dict[str, Any]],
    *,
    min_coverage_ratio: float = 0.95,
) -> dict[str, Any]:
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_uuid.setdefault(str(record.get("uuid") or ""), []).append(record)
    buckets: Counter[str] = Counter()
    complete: list[str] = []
    for uuid, items in by_uuid.items():
        if not items:
            continue
        emitted = [item for item in items if item.get("emitted_candidate")]
        errors = [item for item in items if item.get("stop_reason") == "error"]
        low = [
            item
            for item in emitted
            if item.get("coverage_ratio") is not None
            and float(item["coverage_ratio"]) < min_coverage_ratio
        ]
        assembly_coverage = max(
            (
                float(item.get("assembly_coverage_ratio") or 0.0)
                for item in items
            ),
            default=0.0,
        )
        if not emitted:
            bucket = "no_emitted"
        elif assembly_coverage >= min_coverage_ratio:
            bucket = "complete"
            complete.append(uuid)
        elif errors:
            bucket = "has_solver_errors"
        elif low:
            bucket = "has_low_coverage_emitted"
        else:
            bucket = "partial"
        buckets[bucket] += 1
    return {
        "building_count": len(by_uuid),
        "bucket_counts": dict(buckets.most_common()),
        "complete_buildings": sorted(complete),
        "coverage_p50": _selector_v2_building_coverage_percentile(records, 0.50),
        "coverage_p95": _selector_v2_building_coverage_percentile(records, 0.95),
    }


def _annotate_selector_v2_assembly(records: list[dict[str, Any]]) -> None:
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        record["assembly_candidate"] = False
        record["assembly_coverage_ratio"] = 0.0
        record["assembly_uncovered_area_m2"] = None
        by_uuid.setdefault(str(record.get("uuid") or ""), []).append(record)

    for items in by_uuid.values():
        target = _union_record_domains(items)
        if target is None or target.area <= 1e-9:
            continue
        selected: list[dict[str, Any]] = []
        covered = Polygon()
        eligible = [
            item
            for item in items
            if item.get("emitted_candidate")
            or item.get("assembly_eligible_candidate")
        ]
        while eligible:
            best: tuple[float, float, int, dict[str, Any], Polygon] | None = None
            for item in eligible:
                polygon = _record_assembly_polygon(item)
                if polygon is None or polygon.is_empty:
                    continue
                try:
                    clipped = polygon.intersection(target)
                    gain = float(clipped.difference(covered).area)
                except Exception:
                    continue
                if gain <= 0.05:
                    continue
                tie_break = -int(item.get("part_index") or 0)
                score = (gain, -float(clipped.area), tie_break)
                if best is None or score > best[:3]:
                    best = (gain, -float(clipped.area), tie_break, item, clipped)
            if best is None:
                break
            _gain, _neg_area, _tie_break, item, clipped = best
            selected.append(item)
            eligible.remove(item)
            covered = unary_union([covered, clipped])

        coverage = min(1.0, float(covered.area) / float(target.area))
        uncovered = max(0.0, float(target.area) - float(covered.area))
        selected_ids = {id(item) for item in selected}
        for item in items:
            item["assembly_coverage_ratio"] = coverage
            item["assembly_uncovered_area_m2"] = uncovered
            item["assembly_candidate"] = id(item) in selected_ids


def _union_record_domains(records: list[dict[str, Any]]) -> Polygon | None:
    polygons = [
        polygon
        for polygon in (_record_domain_polygon(record) for record in records)
        if polygon is not None and not polygon.is_empty and polygon.area > 1e-9
    ]
    if not polygons:
        return None
    try:
        return unary_union(polygons)
    except Exception:
        return None


def _record_domain_polygon(record: dict[str, Any]) -> Polygon | None:
    coords = record.get("domain_polygon")
    if not isinstance(coords, list) or len(coords) < 3:
        return None
    try:
        polygon = Polygon([(float(point[0]), float(point[1])) for point in coords])
    except Exception:
        return None
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 1e-9:
        return None
    return polygon


def _record_assembly_polygon(record: dict[str, Any]) -> Polygon | None:
    if record.get("emitted_candidate"):
        return _record_domain_polygon(record)
    polygons = []
    for coords in record.get("selected_coverage_polygons") or []:
        if not isinstance(coords, list) or len(coords) < 3:
            continue
        try:
            polygon = Polygon(
                [(float(point[0]), float(point[1])) for point in coords]
            )
        except Exception:
            continue
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 1e-9:
            continue
        polygons.append(polygon)
    if not polygons:
        return None
    try:
        return unary_union(polygons)
    except Exception:
        return None


def _selected_coverage_polygons_json(trace: dict[str, Any]) -> list[list[list[float]]]:
    polygons = []
    for candidate in trace.get("candidates", []) or []:
        if not isinstance(candidate, dict) or not candidate.get("selected"):
            continue
        corners = candidate.get("corners")
        if not isinstance(corners, list) or len(corners) < 3:
            continue
        try:
            polygon = Polygon(
                [
                    (float(corner[0]), float(corner[2]))
                    for corner in corners
                ]
            )
        except Exception:
            continue
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 1e-9:
            continue
        polygons.append(polygon)
    if not polygons:
        return []
    try:
        union = unary_union(polygons)
    except Exception:
        return []
    if union.geom_type == "Polygon":
        parts = [union]
    else:
        parts = [
            geom for geom in getattr(union, "geoms", [])
            if geom.geom_type == "Polygon"
        ]
    return [
        [[float(x), float(z)] for x, z in part.exterior.coords[:-1]]
        for part in parts
        if part.area > 1e-9
    ]


def _selector_v2_building_coverage_percentile(
    records: list[dict[str, Any]],
    fraction: float,
) -> float | None:
    by_uuid: dict[str, float] = {}
    for record in records:
        uuid = str(record.get("uuid") or "")
        by_uuid[uuid] = max(
            by_uuid.get(uuid, 0.0),
            float(record.get("assembly_coverage_ratio") or 0.0),
        )
    values = sorted(by_uuid.values())
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return float(values[index])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export topology trace JSON for corpus room shells.",
    )
    parser.add_argument("--pipeline-dir", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--domain",
        choices=(
            "room-shell",
            "envelope",
            "room-audit",
            "envelope-cell-selector",
            "selector-v2",
            "manifold-repair-v3",
            "manifold-repair-steps",
            "manifold-repair-building",
        ),
        default="room-shell",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".context/polyhedron-corpus-traces"),
    )
    parser.add_argument("--selection", choices=("unique", "first"), default="unique")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument(
        "--write-mode",
        choices=("interesting", "all", "none"),
        default="interesting",
    )
    parser.add_argument("--valid-sample-limit", type=int, default=20)
    parser.add_argument("--coord-tol", type=float, default=1e-3)
    parser.add_argument("--corner-tol", type=float, default=0.02)
    parser.add_argument("--ceiling-limit", type=int, default=1)
    parser.add_argument("--min-ceiling-overlap-area-m2", type=float, default=0.01)
    parser.add_argument("--min-top-overlap-ratio", type=float, default=0.60)
    parser.add_argument("--building-level", action="store_true")
    parser.add_argument("--max-buildings", type=int, default=10)
    parser.add_argument("--time-budget-seconds", type=float, default=0.5)
    parser.add_argument("--max-intersections", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=500)
    args = parser.parse_args(argv)

    if args.domain in ("envelope", "envelope-cell-selector"):
        index = export_corpus_envelope_traces(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            selection=args.selection,
            max_steps=args.max_steps,
            write_mode=args.write_mode,
            valid_sample_limit=args.valid_sample_limit,
            coord_tol=args.coord_tol,
            corner_tol=args.corner_tol,
            min_top_overlap_ratio=args.min_top_overlap_ratio,
            wing_level=not args.building_level,
            force_cell_selector=args.domain == "envelope-cell-selector",
        )
    elif args.domain == "room-audit":
        index = export_corpus_room_audit(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            corner_tol=args.corner_tol,
        )
    elif args.domain == "manifold-repair-v3":
        index = export_manifold_repair_v3_traces(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            max_buildings=args.max_buildings,
            corner_tol=args.corner_tol,
            coord_tol=args.coord_tol,
        )
    elif args.domain == "manifold-repair-steps":
        index = export_manifold_repair_steps_traces(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            max_buildings=args.max_buildings,
            corner_tol=args.corner_tol,
            coord_tol=args.coord_tol,
        )
    elif args.domain == "manifold-repair-building":
        index = export_manifold_repair_building_traces(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            max_buildings=args.max_buildings,
            corner_tol=args.corner_tol,
            coord_tol=args.coord_tol,
        )
    elif args.domain == "selector-v2":
        index = export_selector_v2_traces(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            max_buildings=args.max_buildings,
            corner_tol=args.corner_tol,
            time_budget_seconds=args.time_budget_seconds,
            max_intersections=args.max_intersections,
            max_candidates=args.max_candidates,
        )
    else:
        index = export_corpus_room_shell_traces(
            pipeline_dir=args.pipeline_dir,
            output_dir=args.out_dir,
            selection=args.selection,
            max_steps=args.max_steps,
            write_mode=args.write_mode,
            valid_sample_limit=args.valid_sample_limit,
            coord_tol=args.coord_tol,
            corner_tol=args.corner_tol,
            ceiling_limit=args.ceiling_limit,
            min_ceiling_overlap_area_m2=args.min_ceiling_overlap_area_m2,
        )
    print(json.dumps(index["summary"], indent=2, sort_keys=True))
    print(f"Wrote index: {args.out_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
