from __future__ import annotations

import math

import numpy as np

from reconcile_tiers.polyhedron import (
    FaceEvidence,
    coordinate_descent_face_shifts,
    evidence_from_face_corners,
    evidence_from_scan_points,
    make_cube,
    signed_distance_cost,
)


def test_signed_distance_cost_zero_for_source_face_corners():
    cube = make_cube(size=2.0)
    evidence = evidence_from_face_corners(
        [[list(c) for c in cube.face_polygon(face)] for face in cube.faces]
    )
    assert math.isclose(signed_distance_cost(cube, evidence), 0.0, abs_tol=1e-12)


def test_signed_distance_cost_uses_face_ids_not_list_offsets():
    cube = make_cube(size=2.0)
    face = cube.faces[0]
    face.id = 100
    evidence = [
        FaceEvidence(
            face_id=100,
            points=tuple(
                tuple(float(v) for v in point) for point in cube.face_polygon(face)
            ),
        )
    ]

    assert math.isclose(signed_distance_cost(cube, evidence), 0.0, abs_tol=1e-12)


def test_signed_distance_cost_ignores_stale_removed_face_evidence():
    cube = make_cube(size=2.0)
    evidence = [FaceEvidence(face_id=999, points=((1.0, 2.0, 3.0),))]

    assert math.isclose(signed_distance_cost(cube, evidence), 0.0, abs_tol=1e-12)


def test_coordinate_descent_restores_shifted_face_offset():
    cube = make_cube(size=2.0)
    evidence = evidence_from_face_corners(
        [[list(c) for c in cube.face_polygon(face)] for face in cube.faces]
    )
    plus_x = next(face for face in cube.faces if face.plane.a > 0.5)
    original_d = plus_x.plane.d

    cube.face_shift(plus_x, 0.25)
    shifted_cost = signed_distance_cost(cube, evidence)
    assert shifted_cost > 0.0

    result = coordinate_descent_face_shifts(
        cube,
        evidence,
        initial_step_m=0.10,
        min_step_m=0.0005,
        max_iterations=80,
    )

    assert result.final_cost < shifted_cost * 0.01
    assert result.accepted_shifts > 0
    assert math.isclose(plus_x.plane.d, original_d, abs_tol=0.002)
    assert cube.is_watertight()
    assert cube.faces_close()


def test_evidence_from_scan_points_gathers_near_plane_points():
    cube = make_cube(size=2.0)
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    evidence = evidence_from_scan_points(cube, points, epsilon=0.01)

    plus_x = next(face for face in cube.faces if face.plane.a > 0.5)
    minus_x = next(face for face in cube.faces if face.plane.a < -0.5)
    by_face = {item.face_id: item.points for item in evidence}
    assert len(by_face[plus_x.id]) == 2
    assert len(by_face[minus_x.id]) == 1
    assert signed_distance_cost(cube, evidence) > 0.0
