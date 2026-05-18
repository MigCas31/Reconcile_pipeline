"""Validity checks and topological-event diagnostics for half-edge polyhedra.

These checks mirror the paper's Section 3.1 validity rules before we mutate
topology. They are deliberately diagnostic-only: event *resolution* belongs in
separate operators once a failing topology shape is understood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdge,
    HalfEdgePolyhedron,
    Vertex,
)

IssueKind = Literal[
    "missing_opposite",
    "broken_opposite",
    "face_orbit_open",
    "face_orbit_mismatch",
    "vertex_orbit_open",
    "vertex_orbit_mismatch",
    "vertex_underconstrained",
    "vertex_planes_do_not_cointersect",
    "adjacent_coplanar_faces",
]

EventKind = Literal["edge_collapse", "triangle_face_collapse"]


@dataclass(frozen=True, slots=True)
class ValidityIssue:
    kind: IssueKind
    message: str
    ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TopologicalEvent:
    kind: EventKind
    message: str
    ids: tuple[int, ...]
    measure: float


def validate_polyhedron(
    polyhedron: HalfEdgePolyhedron,
    *,
    cointersection_tol_m: float = 1e-6,
    coplanar_normal_tol: float = 1e-6,
    coplanar_offset_tol_m: float = 1e-6,
) -> list[ValidityIssue]:
    """Return all currently detectable validity issues.

    The expected result for a valid imported object is an empty list.
    """

    issues: list[ValidityIssue] = []
    issues.extend(_validate_opposites(polyhedron))
    issues.extend(_validate_face_orbits(polyhedron))
    issues.extend(_validate_vertex_orbits(polyhedron))
    issues.extend(
        _validate_vertex_plane_intersections(
            polyhedron,
            cointersection_tol_m=cointersection_tol_m,
        )
    )
    issues.extend(
        _validate_adjacent_coplanar_faces(
            polyhedron,
            normal_tol=coplanar_normal_tol,
            offset_tol_m=coplanar_offset_tol_m,
        )
    )
    return issues


def detect_topological_events(
    polyhedron: HalfEdgePolyhedron,
    *,
    edge_tol_m: float = 1e-6,
    face_area_tol_m2: float = 1e-8,
) -> list[TopologicalEvent]:
    """Detect zero-measure events created by face shifts.

    Section 3.3 resolves these events by changing topology. This function only
    identifies them, so callers can choose an appropriate operator.
    """

    events: list[TopologicalEvent] = []
    events.extend(_detect_edge_collapses(polyhedron, edge_tol_m=edge_tol_m))
    events.extend(
        _detect_triangle_face_collapses(
            polyhedron,
            face_area_tol_m2=face_area_tol_m2,
        )
    )
    return events


def _validate_opposites(polyhedron: HalfEdgePolyhedron) -> list[ValidityIssue]:
    issues: list[ValidityIssue] = []
    for half_edge in polyhedron.half_edges:
        if half_edge.opposite is None:
            issues.append(
                ValidityIssue(
                    kind="missing_opposite",
                    message=f"half-edge {half_edge.id} has no opposite",
                    ids=(half_edge.id,),
                )
            )
        elif half_edge.opposite.opposite is not half_edge:
            issues.append(
                ValidityIssue(
                    kind="broken_opposite",
                    message=f"half-edge {half_edge.id} has non-reciprocal opposite",
                    ids=(half_edge.id, half_edge.opposite.id),
                )
            )
    return issues


def _validate_face_orbits(polyhedron: HalfEdgePolyhedron) -> list[ValidityIssue]:
    issues: list[ValidityIssue] = []
    assigned_by_face: dict[int, set[int]] = {}
    for half_edge in polyhedron.half_edges:
        if half_edge.face is not None:
            assigned_by_face.setdefault(half_edge.face.id, set()).add(half_edge.id)

    for face in polyhedron.faces:
        orbit = _face_orbit(face)
        assigned = assigned_by_face.get(face.id, set())
        if orbit is None:
            issues.append(
                ValidityIssue(
                    kind="face_orbit_open",
                    message=f"face {face.id} exterior orbit does not close",
                    ids=(face.id,),
                )
            )
            continue
        orbit_ids = {half_edge.id for half_edge in orbit}
        if orbit_ids != assigned:
            issues.append(
                ValidityIssue(
                    kind="face_orbit_mismatch",
                    message=(
                        f"face {face.id} orbit has {len(orbit_ids)} half-edges, "
                        f"but {len(assigned)} half-edges point to the face"
                    ),
                    ids=(face.id,),
                )
            )
    return issues


def _validate_vertex_orbits(polyhedron: HalfEdgePolyhedron) -> list[ValidityIssue]:
    issues: list[ValidityIssue] = []
    assigned_by_origin: dict[int, set[int]] = {}
    for half_edge in polyhedron.half_edges:
        assigned_by_origin.setdefault(half_edge.origin.id, set()).add(half_edge.id)

    for vertex in polyhedron.vertices:
        orbit = _vertex_orbit(vertex)
        assigned = assigned_by_origin.get(vertex.id, set())
        if orbit is None:
            issues.append(
                ValidityIssue(
                    kind="vertex_orbit_open",
                    message=f"vertex {vertex.id} orbit does not close",
                    ids=(vertex.id,),
                )
            )
            continue
        orbit_ids = {half_edge.id for half_edge in orbit}
        if orbit_ids != assigned:
            issues.append(
                ValidityIssue(
                    kind="vertex_orbit_mismatch",
                    message=(
                        f"vertex {vertex.id} orbit reaches {len(orbit_ids)} "
                        f"half-edges, but {len(assigned)} originate there"
                    ),
                    ids=(vertex.id,),
                )
            )
    return issues


def _validate_vertex_plane_intersections(
    polyhedron: HalfEdgePolyhedron,
    *,
    cointersection_tol_m: float,
) -> list[ValidityIssue]:
    issues: list[ValidityIssue] = []
    for vertex in polyhedron.vertices:
        faces = polyhedron.incident_faces(vertex)
        if len(faces) < 3:
            issues.append(
                ValidityIssue(
                    kind="vertex_underconstrained",
                    message=(
                        f"vertex {vertex.id} has {len(faces)} incident faces; "
                        "need at least 3"
                    ),
                    ids=(vertex.id,),
                )
            )
            continue
        residual = _plane_system_residual(faces)
        if residual > cointersection_tol_m:
            issues.append(
                ValidityIssue(
                    kind="vertex_planes_do_not_cointersect",
                    message=(
                        f"vertex {vertex.id} incident planes miss common "
                        f"intersection by {residual:.6g} m"
                    ),
                    ids=(vertex.id,),
                )
            )
    return issues


def _validate_adjacent_coplanar_faces(
    polyhedron: HalfEdgePolyhedron,
    *,
    normal_tol: float,
    offset_tol_m: float,
) -> list[ValidityIssue]:
    issues: list[ValidityIssue] = []
    seen_edges: set[tuple[int, int]] = set()
    for half_edge in polyhedron.half_edges:
        if half_edge.next is None or half_edge.opposite is None:
            continue
        edge_key = tuple(sorted((half_edge.id, half_edge.opposite.id)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        face_a = half_edge.face
        face_b = half_edge.opposite.face
        if face_a is None or face_b is None or face_a is face_b:
            continue
        if _planes_same_support(
            face_a,
            face_b,
            normal_tol=normal_tol,
            offset_tol_m=offset_tol_m,
        ):
            issues.append(
                ValidityIssue(
                    kind="adjacent_coplanar_faces",
                    message=(
                        f"faces {face_a.id} and {face_b.id} are adjacent and "
                        "share the same supporting plane"
                    ),
                    ids=(face_a.id, face_b.id),
                )
            )
    return issues


def _detect_edge_collapses(
    polyhedron: HalfEdgePolyhedron,
    *,
    edge_tol_m: float,
) -> list[TopologicalEvent]:
    events: list[TopologicalEvent] = []
    seen_edges: set[tuple[int, int]] = set()
    for half_edge in polyhedron.half_edges:
        if half_edge.next is None or half_edge.opposite is None:
            continue
        edge_key = tuple(sorted((half_edge.id, half_edge.opposite.id)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        length = _edge_length(polyhedron, half_edge)
        if length <= edge_tol_m:
            events.append(
                TopologicalEvent(
                    kind="edge_collapse",
                    message=(
                        f"edge {half_edge.id}/{half_edge.opposite.id} length "
                        f"{length:.6g} m"
                    ),
                    ids=(half_edge.id, half_edge.opposite.id),
                    measure=length,
                )
            )
    return events


def _detect_triangle_face_collapses(
    polyhedron: HalfEdgePolyhedron,
    *,
    face_area_tol_m2: float,
) -> list[TopologicalEvent]:
    events: list[TopologicalEvent] = []
    for face in polyhedron.faces:
        polygon = polyhedron.face_polygon(face)
        if len(polygon) != 3:
            continue
        area = _polygon_area_3d(polygon)
        if area <= face_area_tol_m2:
            events.append(
                TopologicalEvent(
                    kind="triangle_face_collapse",
                    message=f"triangle face {face.id} area {area:.6g} m2",
                    ids=(face.id,),
                    measure=area,
                )
            )
    return events


def _face_orbit(face: Face) -> list[HalfEdge] | None:
    if face.half_edge is None:
        return None
    orbit: list[HalfEdge] = []
    seen: set[int] = set()
    half_edge = face.half_edge
    for _ in range(512):
        if half_edge.id in seen:
            return orbit if half_edge is face.half_edge else None
        seen.add(half_edge.id)
        if half_edge.face is not face:
            return None
        orbit.append(half_edge)
        if half_edge.next is None:
            return None
        half_edge = half_edge.next
    return None


def _vertex_orbit(vertex: Vertex) -> list[HalfEdge] | None:
    if vertex.outgoing is None:
        return []
    orbit: list[HalfEdge] = []
    seen: set[int] = set()
    half_edge = vertex.outgoing
    for _ in range(512):
        if half_edge.id in seen:
            return orbit if half_edge is vertex.outgoing else None
        seen.add(half_edge.id)
        if half_edge.origin is not vertex:
            return None
        orbit.append(half_edge)
        if half_edge.opposite is None or half_edge.opposite.next is None:
            return None
        half_edge = half_edge.opposite.next
    return None


def _plane_system_residual(faces: list[Face]) -> float:
    matrix = np.asarray([[f.plane.a, f.plane.b, f.plane.c] for f in faces])
    rhs = np.asarray([f.plane.d for f in faces])
    try:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    residuals = matrix @ solution - rhs
    return float(np.max(np.abs(residuals)))


def _planes_same_support(
    face_a: Face,
    face_b: Face,
    *,
    normal_tol: float,
    offset_tol_m: float,
) -> bool:
    pa = face_a.plane
    pb = face_b.plane
    na = np.asarray([pa.a, pa.b, pa.c], dtype=float)
    nb = np.asarray([pb.a, pb.b, pb.c], dtype=float)
    na_norm = float(np.linalg.norm(na))
    nb_norm = float(np.linalg.norm(nb))
    if na_norm <= 1e-12 or nb_norm <= 1e-12:
        return False
    na /= na_norm
    nb /= nb_norm
    da = float(pa.d) / na_norm
    db = float(pb.d) / nb_norm
    if float(np.dot(na, nb)) < 0.0:
        nb = -nb
        db = -db
    return (
        float(np.linalg.norm(na - nb)) <= normal_tol
        and abs(da - db) <= offset_tol_m
    )


def _edge_length(polyhedron: HalfEdgePolyhedron, half_edge: HalfEdge) -> float:
    if half_edge.next is None:
        return float("inf")
    start = polyhedron.vertex_position(half_edge.origin)
    end = polyhedron.vertex_position(half_edge.next.origin)
    return float(np.linalg.norm(end - start))


def _polygon_area_3d(points: list[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.0
    origin = points[0]
    area_vector = np.zeros(3, dtype=float)
    for idx in range(1, len(points) - 1):
        area_vector += np.cross(points[idx] - origin, points[idx + 1] - origin)
    return 0.5 * float(np.linalg.norm(area_vector))
