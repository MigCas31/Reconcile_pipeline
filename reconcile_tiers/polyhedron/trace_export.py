"""JSON-friendly exports for topology-resolution traces.

The structures here are intentionally plain dictionaries so a future viewer
module can load them without knowing the Python dataclasses. A frame is a full
polyhedron snapshot; a step links one before frame to one after frame.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from itertools import combinations
from typing import Any

import numpy as np

from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdge,
    HalfEdgePolyhedron,
    Vertex,
    three_plane_intersection,
)
from reconcile_tiers.polyhedron.topology_events import (
    TopologyResolutionTrace,
    resolve_supported_topology_events,
)
from reconcile_tiers.polyhedron.validity import (
    detect_topological_events,
    validate_polyhedron,
)

SCHEMA_VERSION = 1


def export_topology_resolution(
    polyhedron: HalfEdgePolyhedron,
    *,
    max_steps: int = 32,
    selection: str = "unique",
    edge_tol_m: float = 1e-6,
    face_area_tol_m2: float = 1e-8,
) -> dict[str, Any]:
    """Run supported topology updates and capture viewer-ready frames.

    The input polyhedron is mutated, matching ``resolve_supported_topology_events``.
    Use a caller-side copy if the original topology must be retained.
    """

    frames = [
        polyhedron_snapshot(
            polyhedron,
            frame_index=0,
            label="initial",
            edge_tol_m=edge_tol_m,
            face_area_tol_m2=face_area_tol_m2,
        )
    ]
    steps: list[dict[str, Any]] = []
    stop_trace: TopologyResolutionTrace | None = None

    for _ in range(max_steps):
        trace = resolve_supported_topology_events(
            polyhedron,
            max_steps=1,
            selection=selection,
            edge_tol_m=edge_tol_m,
            face_area_tol_m2=face_area_tol_m2,
        )
        if not trace.steps:
            stop_trace = trace
            break

        step = trace.steps[0]
        after_frame_index = len(frames)
        frames.append(
            polyhedron_snapshot(
                polyhedron,
                frame_index=after_frame_index,
                label=f"after_step_{step.index}",
                edge_tol_m=edge_tol_m,
                face_area_tol_m2=face_area_tol_m2,
            )
        )
        steps.append(
            {
                "index": len(steps),
                "action": step.action,
                "trigger_ids": list(step.trigger_ids),
                "before_frame": after_frame_index - 1,
                "after_frame": after_frame_index,
                "before_counts": _to_jsonable(step.before),
                "after_counts": _to_jsonable(step.after),
                "result": _to_jsonable(step.result),
            }
        )
    else:
        stop_trace = resolve_supported_topology_events(
            polyhedron,
            max_steps=0,
            selection=selection,
            edge_tol_m=edge_tol_m,
            face_area_tol_m2=face_area_tol_m2,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "selection": selection,
        "tolerances": {
            "edge_tol_m": edge_tol_m,
            "face_area_tol_m2": face_area_tol_m2,
        },
        "frames": frames,
        "steps": steps,
        "stop": {
            "reason": stop_trace.stop_reason,
            "message": stop_trace.stop_message,
            "remaining_issues": _to_jsonable(stop_trace.remaining_issues),
            "remaining_events": _to_jsonable(stop_trace.remaining_events),
        },
    }


def polyhedron_snapshot(
    polyhedron: HalfEdgePolyhedron,
    *,
    frame_index: int = 0,
    label: str = "snapshot",
    edge_tol_m: float = 1e-6,
    face_area_tol_m2: float = 1e-8,
) -> dict[str, Any]:
    """Return a JSON-friendly full polyhedron snapshot."""

    issues = tuple(validate_polyhedron(polyhedron))
    events: tuple[Any, ...]
    event_error: str | None = None
    try:
        events = tuple(
            detect_topological_events(
                polyhedron,
                edge_tol_m=edge_tol_m,
                face_area_tol_m2=face_area_tol_m2,
            )
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        events = ()
        event_error = str(exc)

    return {
        "index": frame_index,
        "label": label,
        "counts": {
            "faces": len(polyhedron.faces),
            "vertices": len(polyhedron.vertices),
            "half_edges": len(polyhedron.half_edges),
        },
        "issues": _to_jsonable(issues),
        "events": _to_jsonable(events),
        "event_error": event_error,
        "faces": [_face_snapshot(polyhedron, face) for face in polyhedron.faces],
    }


def _face_snapshot(
    polyhedron: HalfEdgePolyhedron,
    face: Face,
) -> dict[str, Any]:
    ring = _face_ring(face)
    corners: list[list[float]] = []
    vertex_ids: list[int] = []
    half_edge_ids: list[int] = []
    errors: list[str] = []

    for half_edge in ring:
        vertex_ids.append(half_edge.origin.id)
        half_edge_ids.append(half_edge.id)
        try:
            position = _safe_vertex_position(polyhedron, half_edge.origin)
            corners.append(_point_to_json(position))
        except (ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"vertex {half_edge.origin.id}: {exc}")

    return {
        "id": face.id,
        "plane": {
            "a": face.plane.a,
            "b": face.plane.b,
            "c": face.plane.c,
            "d": face.plane.d,
        },
        "vertex_ids": vertex_ids,
        "half_edge_ids": half_edge_ids,
        "corners": corners,
        "errors": errors,
    }


def _safe_vertex_position(
    polyhedron: HalfEdgePolyhedron,
    vertex: Vertex,
    *,
    residual_tol_m: float = 1e-5,
) -> np.ndarray:
    faces = polyhedron.incident_faces(vertex)
    if len(faces) < 3:
        raise ValueError(
            f"Vertex {vertex.id} has only {len(faces)} incident faces; need 3"
        )

    best_position: np.ndarray | None = None
    best_residual = float("inf")
    for triplet in combinations(faces, 3):
        try:
            position = three_plane_intersection(
                triplet[0].plane,
                triplet[1].plane,
                triplet[2].plane,
            )
        except np.linalg.LinAlgError:
            continue
        residual = _max_plane_residual(position, faces)
        if residual < best_residual:
            best_position = position
            best_residual = residual
        if residual <= residual_tol_m:
            return position

    if best_position is not None:
        return best_position

    matrix = np.array(
        [[face.plane.a, face.plane.b, face.plane.c] for face in faces],
        dtype=float,
    )
    offsets = np.array([face.plane.d for face in faces], dtype=float)
    position, *_ = np.linalg.lstsq(matrix, offsets, rcond=None)
    return position


def _max_plane_residual(position: np.ndarray, faces: list[Face]) -> float:
    return max(
        abs(
            face.plane.a * position[0]
            + face.plane.b * position[1]
            + face.plane.c * position[2]
            - face.plane.d
        )
        for face in faces
    )


def _face_ring(face: Face) -> list[HalfEdge]:
    if face.half_edge is None:
        return []
    ring: list[HalfEdge] = []
    half_edge = face.half_edge
    for _ in range(512):
        ring.append(half_edge)
        if half_edge.next is None:
            return ring
        half_edge = half_edge.next
        if half_edge is face.half_edge:
            return ring
    return ring


def _point_to_json(point: np.ndarray) -> list[float]:
    return [float(point[0]), float(point[1]), float(point[2])]


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
