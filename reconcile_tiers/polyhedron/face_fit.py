"""Small face-shift fitting loop for half-edge polyhedra.

This is intentionally fixed-topology: it only changes ``face.plane.d`` via
``HalfEdgePolyhedron.face_shift``. It is enough to test whether a watertight
room shell can recover small plane-offset errors before implementing the
paper's topology operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reconcile_tiers.polyhedron.half_edge import HalfEdgePolyhedron


@dataclass(frozen=True, slots=True)
class FaceEvidence:
    """Observed points that support one face's current normal."""

    face_id: int
    points: tuple[tuple[float, float, float], ...]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ScanPointEvidence:
    """Raw scan points gathered near one face plane."""

    face_id: int
    points: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class CoordinateDescentResult:
    initial_cost: float
    final_cost: float
    iterations: int
    accepted_shifts: int


def evidence_from_face_corners(
    face_corners: list[list[list[float]]],
    *,
    weight: float = 1.0,
) -> list[FaceEvidence]:
    """Create one evidence set per face from its source polygon corners."""

    out: list[FaceEvidence] = []
    for face_id, corners in enumerate(face_corners):
        points = tuple((float(c[0]), float(c[1]), float(c[2])) for c in corners)
        out.append(FaceEvidence(face_id=face_id, points=points, weight=weight))
    return out


def evidence_from_scan_points(
    polyhedron: HalfEdgePolyhedron,
    scan_points: np.ndarray,
    epsilon: float,
    *,
    distance_multiplier: float = 2.0,
    weight: float = 1.0,
) -> list[ScanPointEvidence]:
    """Gather raw scan points within ``distance_multiplier * epsilon`` of faces."""

    points = np.asarray(scan_points, dtype=float)
    if points.size == 0:
        points = np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("scan_points must have shape (N, 3)")
    max_distance = max(float(epsilon), 0.0) * max(float(distance_multiplier), 0.0)

    out: list[ScanPointEvidence] = []
    for face in polyhedron.faces:
        plane = face.plane
        residuals = np.abs(
            plane.a * points[:, 0]
            + plane.b * points[:, 1]
            + plane.c * points[:, 2]
            - plane.d
        )
        out.append(
            ScanPointEvidence(
                face_id=face.id,
                points=points[residuals <= max_distance].copy(),
                weight=weight,
            )
        )
    return out


def signed_distance_cost(
    polyhedron: HalfEdgePolyhedron,
    evidence: list[FaceEvidence | ScanPointEvidence],
) -> float:
    """Weighted mean squared signed distance from evidence points to planes."""

    faces_by_id = {face.id: face for face in polyhedron.faces}
    total = 0.0
    total_weight = 0.0
    for item in evidence:
        face = faces_by_id.get(item.face_id)
        if face is None:
            continue
        if len(item.points) == 0 or item.weight <= 0.0:
            continue
        plane = face.plane
        pts = np.asarray(item.points, dtype=float)
        residuals = (
            plane.a * pts[:, 0]
            + plane.b * pts[:, 1]
            + plane.c * pts[:, 2]
            - plane.d
        )
        total += float(item.weight) * float(np.mean(residuals * residuals))
        total_weight += float(item.weight)
    if total_weight <= 0.0:
        return 0.0
    return total / total_weight


def coordinate_descent_face_shifts(
    polyhedron: HalfEdgePolyhedron,
    evidence: list[FaceEvidence | ScanPointEvidence],
    *,
    initial_step_m: float = 0.10,
    min_step_m: float = 0.001,
    max_iterations: int = 40,
) -> CoordinateDescentResult:
    """Minimize signed-distance cost by trying +/- plane-offset shifts.

    The search is deliberately conservative: every accepted move must reduce
    global cost, and the step size halves after a full pass with no accepted
    moves.
    """

    step = float(initial_step_m)
    current = signed_distance_cost(polyhedron, evidence)
    initial = current
    accepted = 0
    iterations = 0

    while step >= min_step_m and iterations < max_iterations:
        improved_this_pass = False
        iterations += 1

        for face in polyhedron.faces:
            best_delta = 0.0
            best_cost = current
            for delta in (-step, step):
                polyhedron.face_shift(face, delta)
                candidate = signed_distance_cost(polyhedron, evidence)
                polyhedron.face_shift(face, -delta)
                if candidate + 1e-12 < best_cost:
                    best_cost = candidate
                    best_delta = delta

            if best_delta != 0.0:
                polyhedron.face_shift(face, best_delta)
                current = best_cost
                accepted += 1
                improved_this_pass = True

        if not improved_this_pass:
            step *= 0.5

    return CoordinateDescentResult(
        initial_cost=initial,
        final_cost=current,
        iterations=iterations,
        accepted_shifts=accepted,
    )
