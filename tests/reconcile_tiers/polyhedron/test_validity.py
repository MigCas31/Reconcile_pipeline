from __future__ import annotations

import pytest

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import (
    build_from_planar_polygons,
    detect_topological_events,
    make_cube,
    validate_polyhedron,
)


def test_validate_cube_has_no_issues_or_events():
    cube = make_cube(size=2.0)
    assert validate_polyhedron(cube) == []
    assert detect_topological_events(cube) == []


def test_validate_reports_missing_opposite_and_open_vertex_orbit():
    cube = make_cube(size=2.0)
    half_edge = cube.half_edges[0]
    twin = half_edge.opposite
    assert twin is not None

    half_edge.opposite = None
    issues = validate_polyhedron(cube)

    kinds = {issue.kind for issue in issues}
    assert "missing_opposite" in kinds
    assert "vertex_orbit_open" in kinds


def test_validate_reports_face_orbit_mismatch():
    cube = make_cube(size=2.0)
    cube.faces[0].half_edge = cube.faces[1].half_edge

    issues = validate_polyhedron(cube)

    assert any(issue.kind == "face_orbit_open" for issue in issues)


def test_validate_reports_adjacent_coplanar_faces():
    cube = make_cube(size=2.0)
    plus_x = next(face for face in cube.faces if face.plane.a > 0.5)
    plus_y = next(face for face in cube.faces if face.plane.b > 0.5)
    plus_y.plane = plus_x.plane

    issues = validate_polyhedron(cube)

    assert any(issue.kind == "adjacent_coplanar_faces" for issue in issues)


def test_detect_edge_collapse_after_face_shift():
    cube = make_cube(size=2.0)
    plus_x = next(face for face in cube.faces if face.plane.a > 0.5)
    minus_x = next(face for face in cube.faces if face.plane.a < -0.5)
    # Move +X from x=1 to x=-1. Four X-direction edges become zero-length.
    cube.face_shift(plus_x, delta=-2.0)

    events = detect_topological_events(cube, edge_tol_m=1e-9)

    edge_events = [event for event in events if event.kind == "edge_collapse"]
    assert len(edge_events) == 4
    assert all(event.measure <= 1e-9 for event in edge_events)
    # Topology is unchanged by detection; the opposing face is still present.
    assert minus_x in cube.faces


def test_detect_triangle_face_collapse():
    poly = build_from_planar_polygons(_triangular_prism_polys())
    collapse = next(
        face
        for face in poly.faces
        if face.plane.b > 0.5 and face.plane.c > 0.5
    )
    poly.face_shift(collapse, delta=-(1.0 / (2.0**0.5)))

    events = detect_topological_events(poly, edge_tol_m=1e-9, face_area_tol_m2=1e-9)

    assert any(event.kind == "edge_collapse" for event in events)
    assert any(event.kind == "triangle_face_collapse" for event in events)


def test_validate_reports_overconstrained_vertex_planes():
    poly = build_from_planar_polygons(_overconstrained_vertex_polys())

    issues = validate_polyhedron(poly, cointersection_tol_m=1e-8)

    assert any(
        issue.kind == "vertex_planes_do_not_cointersect" for issue in issues
    )


def _triangular_prism_polys() -> list[tuple[list[list[float]], Plane]]:
    # Prism with triangular caps at x=-1 and x=1.
    a0 = [-1.0, 0.0, 0.0]
    b0 = [-1.0, 1.0, 0.0]
    c0 = [-1.0, 0.0, 1.0]
    a1 = [1.0, 0.0, 0.0]
    b1 = [1.0, 1.0, 0.0]
    c1 = [1.0, 0.0, 1.0]
    sqrt2 = 2.0**0.5
    return [
        ([a0, c0, b0], Plane(a=-1.0, b=0.0, c=0.0, d=1.0)),
        ([a1, b1, c1], Plane(a=1.0, b=0.0, c=0.0, d=1.0)),
        ([a0, a1, c1, c0], Plane(a=0.0, b=-1.0, c=0.0, d=0.0)),
        ([a0, b0, b1, a1], Plane(a=0.0, b=0.0, c=-1.0, d=0.0)),
        (
            [b0, c0, c1, b1],
            Plane(a=0.0, b=1.0 / sqrt2, c=1.0 / sqrt2, d=1.0 / sqrt2),
        ),
    ]


def _overconstrained_vertex_polys() -> list[tuple[list[list[float]], Plane]]:
    # Square pyramid: the apex is adjacent to four side faces. Offset one side
    # plane so those four side planes no longer share one common point.
    v00 = [-1.0, 0.0, -1.0]
    v10 = [1.0, 0.0, -1.0]
    v11 = [1.0, 0.0, 1.0]
    v01 = [-1.0, 0.0, 1.0]
    apex = [0.0, 1.0, 0.0]
    s2 = 1.0 / (2.0**0.5)
    return [
        ([v00, v10, v11, v01], Plane(a=0.0, b=-1.0, c=0.0, d=0.0)),
        ([v00, apex, v10], Plane(a=0.0, b=s2, c=-s2, d=s2 + 0.05)),
        ([v10, apex, v11], Plane(a=s2, b=s2, c=0.0, d=s2)),
        ([v11, apex, v01], Plane(a=0.0, b=s2, c=s2, d=s2)),
        ([v01, apex, v00], Plane(a=-s2, b=s2, c=0.0, d=s2)),
    ]


@pytest.mark.parametrize("polys", [_triangular_prism_polys()])
def test_fixture_prism_is_valid_before_shift(polys):
    poly = build_from_planar_polygons(polys)
    assert validate_polyhedron(poly) == []
