"""Viewer trace export for the kinetic ceiling reconstruction pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.ceiling_reconstruction.pipeline import KsrPipelineResult, run_ksr_pipeline
from reconcile_tiers.polyhedron.face_selection import CandidateFace
from reconcile_tiers.polyhedron.kinetic_partition import (
    BoundingPrism,
    ConvexCell,
    cells_to_candidate_faces,
)
from reconcile_tiers.polyhedron.manifold_repair import TileFace

SCHEMA_VERSION = 2
SELECTION = "kinetic-ceiling-steps"

PIPELINE_STEP_LABELS: dict[str, str] = {
    "segment_tier_input": "0. Segment-tier input",
    "oriented_samples": "1. Oriented samples",
    "planes_detected": "2. Planes detected",
    "planes_regularized": "3. Planes regularized",
    "kinetic_partition": "4. Kinetic partition",
    "graph_cut_labels": "5. Graph-cut labels",
    "ceiling_extracted": "6. Ceiling extracted",
}


def _plane_dict(plane: Plane) -> dict[str, float]:
    return {
        "a": float(plane.a),
        "b": float(plane.b),
        "c": float(plane.c),
        "d": float(plane.d),
    }


def tiles_to_viewer_faces(
    tiles: Sequence[TileFace],
    *,
    role: str = "tile",
    selected: bool = True,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for tile in tiles:
        corners = [[float(c[0]), float(c[1]), float(c[2])] for c in tile.corners]
        if len(corners) < 3:
            continue
        faces.append(
            {
                "id": tile.face_id,
                "plane_id": tile.face_id,
                "selected": selected,
                "label": tile.source,
                "role": role,
                "corners": corners,
                "plane": _plane_dict(tile.plane),
            }
        )
    return faces


def candidates_to_viewer_faces(
    candidates: Sequence[CandidateFace],
    *,
    role: str,
    label: str = "",
    selected: bool = True,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for cand in candidates:
        corners = [[float(c[0]), float(c[1]), float(c[2])] for c in cand.corners]
        if len(corners) < 3:
            continue
        faces.append(
            {
                "id": cand.face_id,
                "plane_id": cand.plane_id,
                "selected": selected,
                "label": label or cand.confidence_label,
                "role": role,
                "corners": corners,
                "plane": _plane_dict(cand.plane),
            }
        )
    return faces


def _plane_groups_to_faces(
    groups: Sequence[Any],
    *,
    role: str,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for idx, group in enumerate(groups):
        for tile in group.tiles:
            corners = [[float(c[0]), float(c[1]), float(c[2])] for c in tile.corners]
            if len(corners) < 3:
                continue
            faces.append(
                {
                    "id": tile.face_id,
                    "plane_id": idx,
                    "selected": True,
                    "label": group.kind,
                    "role": role,
                    "corners": corners,
                    "plane": _plane_dict(group.plane),
                }
            )
    return faces


def _plane_frame(plane: Plane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("degenerate plane")
    normal /= norm
    d = float(plane.d) / norm
    origin = normal * d
    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return origin, u, v


def _bbox_corners(prism: BoundingPrism) -> list[tuple[float, float, float]]:
    return [
        (prism.x_min, prism.y_min, prism.z_min),
        (prism.x_max, prism.y_min, prism.z_min),
        (prism.x_max, prism.y_max, prism.z_min),
        (prism.x_min, prism.y_max, prism.z_min),
        (prism.x_min, prism.y_min, prism.z_max),
        (prism.x_max, prism.y_min, prism.z_max),
        (prism.x_max, prism.y_max, prism.z_max),
        (prism.x_min, prism.y_max, prism.z_max),
    ]


def _project_point_to_plane(
    point: tuple[float, float, float],
    plane: Plane,
) -> tuple[float, float, float]:
    x, y, z = point
    n = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return point
    n /= norm
    d = float(plane.d) / norm
    p = np.array([x, y, z], dtype=float)
    dist = float(n @ p - d)
    proj = p - dist * n
    return (float(proj[0]), float(proj[1]), float(proj[2]))


def _clipped_plane_faces(
    planes: Sequence[Plane],
    prism: BoundingPrism | None,
    *,
    role: str = "partition_plane",
) -> list[dict[str, Any]]:
    if prism is None or not planes:
        return []
    faces: list[dict[str, Any]] = []
    corners = _bbox_corners(prism)
    for idx, plane in enumerate(planes):
        projected = [_project_point_to_plane(c, plane) for c in corners]
        try:
            origin, u, v = _plane_frame(plane)
        except ValueError:
            continue
        uv = np.asarray(
            [[float((np.asarray(p) - origin) @ u), float((np.asarray(p) - origin) @ v)] for p in projected],
            dtype=float,
        )
        if len(uv) < 3:
            continue
        lo_u, hi_u = float(uv[:, 0].min()), float(uv[:, 0].max())
        lo_v, hi_v = float(uv[:, 1].min()), float(uv[:, 1].max())
        if hi_u - lo_u <= 1e-6 or hi_v - lo_v <= 1e-6:
            continue
        quad_uv = [
            (lo_u, lo_v),
            (hi_u, lo_v),
            (hi_u, hi_v),
            (lo_u, hi_v),
        ]
        quad_xyz = [
            origin + u * uu + v * vv for uu, vv in quad_uv
        ]
        face_corners = [
            [float(p[0]), float(p[1]), float(p[2])] for p in quad_xyz
        ]
        faces.append(
            {
                "id": idx,
                "plane_id": idx,
                "selected": True,
                "label": "partition_plane",
                "role": role,
                "corners": face_corners,
                "plane": _plane_dict(plane),
            }
        )
    return faces


def _partition_faces(
    cells: Sequence[ConvexCell],
    planes: Sequence[Plane],
    prism: BoundingPrism | None,
) -> list[dict[str, Any]]:
    """Clipped structure planes plus chessboard cell interfaces for visualization."""
    faces = _clipped_plane_faces(planes, prism, role="partition_plane")
    if cells:
        viz_labels = {
            cell.cell_id: ("inside" if idx % 2 else "outside")
            for idx, cell in enumerate(cells)
        }
        chessboard = cells_to_candidate_faces(cells, viz_labels)
        faces.extend(
            candidates_to_viewer_faces(
                chessboard,
                role="partition_cell",
                label="cell",
            )
        )
    return faces


def _labeled_cell_faces(
    cells: Sequence[ConvexCell],
    labels: Mapping[int, str],
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for cell in cells:
        label = labels.get(cell.cell_id, "outside")
        role = "cell_solid" if label == "inside" else "cell_empty"
        xs = [v[0] for v in cell.vertices]
        ys = [v[1] for v in cell.vertices]
        zs = [v[2] for v in cell.vertices]
        if not xs:
            continue
        # Render cell as a bounding box quad strip (centroid marker + AABB faces)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        z0, z1 = min(zs), max(zs)
        box_corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y0, z1),
            (x0, y0, z1),
        ]
        plane = Plane.fit(box_corners)
        if isinstance(plane, FitFailure):
            continue
        faces.append(
            {
                "id": cell.cell_id,
                "plane_id": cell.cell_id,
                "selected": True,
                "label": label,
                "role": role,
                "corners": [[float(c[0]), float(c[1]), float(c[2])] for c in box_corners],
                "plane": _plane_dict(plane),
            }
        )
    return faces


def _pipeline_frame(
    index: int,
    pipeline_step: str,
    faces: list[dict[str, Any]],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = PIPELINE_STEP_LABELS.get(pipeline_step, pipeline_step)
    return {
        "index": index,
        "label": label,
        "pipeline_step": pipeline_step,
        "counts": {
            "faces": len(faces),
            "orphan_edges": 0,
            "coherence_edges": 0,
            "footprint_edges": 0,
            "vertices": 0,
            "half_edges": 0,
        },
        "issues": [],
        "events": [],
        "event_error": None,
        "faces": faces,
        "orphan_edges": [],
        "coherence_edges": [],
        "footprint_edges": [],
        "meta": meta or {},
    }


def _trace_document(
    frames: list[dict[str, Any]],
    *,
    status: str,
    message: str,
    room_index: int | None,
    story: Any,
    ksr_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": SELECTION,
        "room_index": room_index,
        "story": story,
        "tolerances": {},
        "frames": frames,
        "steps": [],
        "stop": {
            "reason": status,
            "message": message,
            "remaining_issues": [],
            "remaining_events": [],
        },
        "repair_summary": ksr_summary,
    }


def _room_index(room: Mapping[str, Any]) -> int | None:
    raw = room.get("room_index")
    if raw is not None:
        return int(raw)
    loc = str(room.get("locator_id") or "")
    if loc.startswith("room:"):
        parts = loc.split(":")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    return None


def build_ksr_room_trace(
    payload: Mapping[str, Any],
    room: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    graph_cut_lambda: float = 0.75,
    bbox_margin: float = 1.0,
    k_intersections: int = 2,
) -> dict[str, Any]:
    """Run KSR pipeline and capture one viewer frame per stage."""
    story = room.get("story")
    room_idx = _room_index(room)
    result = run_ksr_pipeline(
        payload,
        room,
        corner_tol=corner_tol,
        graph_cut_lambda=graph_cut_lambda,
        bbox_margin=bbox_margin,
        k_intersections=k_intersections,
    )

    frames: list[dict[str, Any]] = []
    structure_faces = tiles_to_viewer_faces(result.structure_tiles, role="tier_payload")

    if result.stop_reason == "insufficient_tiles":
        frames.append(
            _pipeline_frame(
                0,
                "segment_tier_input",
                structure_faces,
                meta={"evidence_tile_count": len(result.evidence_tiles)},
            )
        )
        frames.append(
            _pipeline_frame(
                1,
                "oriented_samples",
                structure_faces,
                meta={"status": "skipped", "reason": "insufficient_tiles"},
            )
        )
        return _trace_document(
            frames,
            status="insufficient_tiles",
            message="insufficient_tiles",
            room_index=room_idx,
            story=story,
        )

    frames.append(
        _pipeline_frame(
            0,
            "segment_tier_input",
            structure_faces,
            meta={"evidence_tile_count": len(result.evidence_tiles)},
        )
    )

    sample_meta = {
        "sample_count": int(result.oriented_samples.shape[0]),
        "structure_sample_count": int(result.oriented_samples_structure.shape[0]),
        "evidence_sample_count": int(result.oriented_samples_evidence.shape[0]),
        "has_normals": result.oriented_samples.shape[1] >= 6
        if result.oriented_samples.size
        else False,
    }
    frames.append(
        _pipeline_frame(
            1,
            "oriented_samples",
            structure_faces,
            meta=sample_meta,
        )
    )

    detected_faces = _plane_groups_to_faces(
        result.plane_groups_detected, role="plane_detected"
    )
    frames.append(
        _pipeline_frame(
            2,
            "planes_detected",
            detected_faces or structure_faces,
            meta={"plane_group_count": len(result.plane_groups_detected)},
        )
    )

    regularized_faces = _plane_groups_to_faces(
        result.plane_groups_regularized, role="plane_regularized"
    )
    frames.append(
        _pipeline_frame(
            3,
            "planes_regularized",
            regularized_faces or detected_faces or structure_faces,
            meta={"plane_group_count": len(result.plane_groups_regularized)},
        )
    )

    partition_faces = (
        _partition_faces(result.cells, result.partition_planes, result.bounding_prism)
        if result.cells
        else regularized_faces
    )
    frames.append(
        _pipeline_frame(
            4,
            "kinetic_partition",
            partition_faces,
            meta={
                "cell_count": len(result.cells),
                "plane_count": len(result.partition_planes),
                "partition_plane_count": len(result.partition_planes),
            },
        )
    )

    labeled_faces = (
        candidates_to_viewer_faces(
            result.boundary_faces,
            role="graph_cut_boundary",
            label="boundary",
        )
        + _labeled_cell_faces(result.cells, result.labels)
        if result.cells
        else partition_faces
    )
    frames.append(
        _pipeline_frame(
            5,
            "graph_cut_labels",
            labeled_faces,
            meta={
                "graph_cut_status": result.graph_cut_status,
                "solid_cells": sum(
                    1 for v in result.labels.values() if v == "inside"
                ),
                "empty_cells": sum(
                    1 for v in result.labels.values() if v == "outside"
                ),
                **result.meta,
            },
        )
    )

    ceiling_faces = candidates_to_viewer_faces(
        result.ceiling_faces,
        role="ceiling_reconstructed",
        label="kinetic_ceiling",
    )
    frames.append(
        _pipeline_frame(
            6,
            "ceiling_extracted",
            ceiling_faces or labeled_faces,
            meta={
                "ceiling_face_count": len(result.ceiling_faces),
                "boundary_face_count": len(result.boundary_faces),
                **result.meta,
            },
        )
    )

    return _trace_document(
        frames,
        status=result.stop_reason,
        message=result.stop_message,
        room_index=room_idx,
        story=story,
        ksr_summary={
            "status": result.stop_reason,
            "tile_count": len(result.structure_tiles),
            "evidence_tile_count": len(result.evidence_tiles),
            "sample_count": int(result.oriented_samples.shape[0]),
            "cell_count": len(result.cells),
            "ceiling_face_count": len(result.ceiling_faces),
            "graph_cut_status": result.graph_cut_status,
            **result.meta,
        },
    )
