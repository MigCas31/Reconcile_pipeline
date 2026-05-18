from __future__ import annotations

import math

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import (
    make_cube,
    refine_polyhedron,
    signed_distance_cost,
    validate_polyhedron,
)
from reconcile_tiers.polyhedron.face_fit import evidence_from_scan_points


def test_refine_polyhedron_restores_shifted_face_from_scan_points():
    cube = make_cube(size=2.0)
    scan_points = _face_center_points(cube)
    evidence = evidence_from_scan_points(cube, scan_points, epsilon=0.05)
    plus_x = next(face for face in cube.faces if face.plane.a > 0.5)
    original_d = plus_x.plane.d

    cube.face_shift(plus_x, 0.04)
    shifted_cost = signed_distance_cost(cube, evidence)

    result = refine_polyhedron(
        cube,
        scan_points,
        max_steps=4,
        max_shift_iterations_per_step=16,
        initial_step_m=0.01,
        min_step_m=0.001,
    )

    assert result.stop_reason in {"converged", "max_steps"}
    assert result.final_cost < shifted_cost * 0.05
    assert result.accepted_shifts > 0
    assert math.isclose(plus_x.plane.d, original_d, abs_tol=0.003)
    assert validate_polyhedron(cube) == []


def test_refine_polyhedron_reports_topology_blocked_in_unique_mode():
    cube = make_cube(size=2.0)
    plus_z = next(face for face in cube.faces if face.plane.c > 0.5)
    plus_z.plane = Plane(a=0.0, b=0.0, c=1.0, d=-1.0)

    result = refine_polyhedron(
        cube,
        np.empty((0, 3), dtype=float),
        max_steps=1,
        topology_selection="unique",
    )

    assert result.stop_reason == "topology_blocked"
    assert "multiple topological events" in result.stop_message
    assert result.topology_traces[0].stop_reason == "ambiguous_events"


def _face_center_points(polyhedron) -> np.ndarray:
    points = []
    for face in polyhedron.faces:
        corners = np.asarray(polyhedron.face_polygon(face), dtype=float)
        points.append(corners.mean(axis=0))
    return np.asarray(points, dtype=float)
