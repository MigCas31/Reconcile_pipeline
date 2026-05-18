"""Plane reconciliation: FACE_SHIFT plane offsets so derived vertex
positions match explicit ones (Increment 6d).

After Increment 6c (wall-derived boundary), every vertex SHOULD be a
3-plane intersection. But due to numerical drift in tier_payload's
plane equations (planes fit upstream from noisy scans), the 3-plane
intersection of a vertex's incident planes may land 1-5 mm away from
the explicit corner stored at build time.

This module implements Geniet 2024 §3.2.1's `FACE_SHIFT` operator —
update a face's plane offset (`d`) so the plane translates without
rotating, propagating the shift to every vertex on that face's
boundary atomically.

The reconciliation iterates Gauss-Seidel: for each face Π, set
``d_Π = mean(n_Π · v_explicit)`` over Π's incident explicit vertices.
This makes Π pass through the centroid of those positions, reducing
average residual. Iterate until the max per-vertex drift is below
tolerance or a max iteration count is hit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdgePolyhedron,
)

__all__ = [
    "ReconciliationResult",
    "reconcile_planes",
]


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Outcome of plane reconciliation.

    ``plane_shifts_applied`` is the count of FACE_SHIFTs that produced a
    non-trivial offset change (>1 µm). ``final_max_drift_m`` is the worst
    explicit-vs-derived vertex drift after all iterations. ``converged``
    is True iff that drift fell below the requested tolerance.
    """

    plane_shifts_applied: int
    final_max_drift_m: float
    iterations: int
    converged: bool


def reconcile_planes(
    poly: HalfEdgePolyhedron,
    vertex_coords: Mapping[int, tuple[float, float, float]],
    *,
    tolerance_m: float = 0.005,
    max_iterations: int = 20,
) -> ReconciliationResult:
    """Iterate FACE_SHIFTs until vertex drift falls below ``tolerance_m``.

    ``vertex_coords`` is the explicit per-vertex position stored at
    build time (``RoomPolyhedronBuild.vertex_coords``). For each face,
    we compute the centroid of its incident explicit vertex positions
    along the face's normal and shift the face's offset to match.

    Mutates ``poly.faces[*].plane`` in place. Returns a summary; callers
    can flag rooms whose drift didn't converge for fallback.
    """
    n_shifts = 0
    final_drift = float("inf")
    iters = 0

    face_vertices = _face_vertex_index(poly)

    for iteration in range(max_iterations):
        iters = iteration + 1
        max_change = 0.0

        for face in poly.faces:
            vids = face_vertices.get(face.id, ())
            if len(vids) < 3:
                continue
            normal = np.array(
                [face.plane.a, face.plane.b, face.plane.c], dtype=float
            )
            n_norm = float(np.linalg.norm(normal))
            if n_norm <= 1e-12:
                continue
            normal_unit = normal / n_norm
            d_unit = float(face.plane.d) / n_norm
            distances: list[float] = []
            for vid in vids:
                if vid not in vertex_coords:
                    continue
                p = np.asarray(vertex_coords[vid], dtype=float)
                distances.append(float(normal_unit @ p))
            if not distances:
                continue
            target_d_unit = sum(distances) / len(distances)
            delta_unit = target_d_unit - d_unit
            if abs(delta_unit) < 1e-9:
                continue
            n_shifts += 1
            max_change = max(max_change, abs(delta_unit))
            new_d = (d_unit + delta_unit) * n_norm
            face.plane = Plane(
                a=face.plane.a,
                b=face.plane.b,
                c=face.plane.c,
                d=new_d,
            )

        final_drift = _max_vertex_drift(poly, vertex_coords, face_vertices)
        if final_drift <= tolerance_m and max_change < tolerance_m:
            return ReconciliationResult(
                plane_shifts_applied=n_shifts,
                final_max_drift_m=final_drift,
                iterations=iters,
                converged=True,
            )

    return ReconciliationResult(
        plane_shifts_applied=n_shifts,
        final_max_drift_m=final_drift,
        iterations=iters,
        converged=final_drift <= tolerance_m,
    )


def _face_vertex_index(
    poly: HalfEdgePolyhedron,
) -> dict[int, tuple[int, ...]]:
    """For each face, the unique vertex IDs around its boundary."""
    out: dict[int, tuple[int, ...]] = {}
    for face in poly.faces:
        if face.half_edge is None:
            continue
        seen: list[int] = []
        seen_set: set[int] = set()
        h = face.half_edge
        for _ in range(512):
            vid = h.origin.id
            if vid not in seen_set:
                seen.append(vid)
                seen_set.add(vid)
            nxt = h.next
            if nxt is None or nxt is face.half_edge:
                break
            h = nxt
        out[face.id] = tuple(seen)
    return out


def _max_vertex_drift(
    poly: HalfEdgePolyhedron,
    vertex_coords: Mapping[int, tuple[float, float, float]],
    face_vertices: Mapping[int, tuple[int, ...]],
) -> float:
    """Worst |p_explicit - p_derived| over vertices with ≥3 incident faces."""
    incident: dict[int, list[Face]] = {}
    for face in poly.faces:
        for vid in face_vertices.get(face.id, ()):
            incident.setdefault(vid, []).append(face)

    worst = 0.0
    for vid, faces in incident.items():
        if len(faces) < 3 or vid not in vertex_coords:
            continue
        explicit = np.asarray(vertex_coords[vid], dtype=float)
        # Solve for derived position via least-squares across all incident
        # planes (more stable than picking 3).
        A = np.array(
            [[f.plane.a, f.plane.b, f.plane.c] for f in faces], dtype=float
        )
        b = np.array([f.plane.d for f in faces], dtype=float)
        try:
            derived, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(derived)):
            continue
        worst = max(worst, float(np.linalg.norm(derived - explicit)))
    return worst
