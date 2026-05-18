"""Tests for plane reconciliation (Increment 6d)."""

from __future__ import annotations

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.half_edge import make_cube
from reconcile_tiers.polyhedron.plane_reconciliation import reconcile_planes


def test_reconcile_planes_cube_with_one_offset_plane_converges():
    """Synthetic cube with one plane offset by 1 cm → reconciliation
    converges in <= 5 iterations and final drift is sub-mm."""
    cube = make_cube(size=2.0)  # half-size 1.0, vertex coords at ±1
    # Explicit vertex coords from the original cube planes (un-shifted).
    vertex_coords: dict[int, tuple[float, float, float]] = {}
    for v in cube.vertices:
        p = cube.vertex_position(v)
        vertex_coords[v.id] = (float(p[0]), float(p[1]), float(p[2]))

    # Shift the +X face by +1 cm — derived vertex positions on that
    # face now drift by 1 cm.
    px_face = cube.faces[0]
    px_face.plane = Plane(
        a=px_face.plane.a,
        b=px_face.plane.b,
        c=px_face.plane.c,
        d=px_face.plane.d + 0.01,
    )

    result = reconcile_planes(
        cube,
        vertex_coords,
        tolerance_m=0.001,
        max_iterations=10,
    )
    assert result.converged, (
        f"reconciliation failed: drift={result.final_max_drift_m:.6f} m"
    )
    assert result.iterations <= 5
    assert result.final_max_drift_m < 0.001


def test_reconcile_planes_clean_cube_no_shifts_needed():
    """Already-aligned cube → reconciliation applies zero shifts and
    converges trivially."""
    cube = make_cube(size=2.0)
    vertex_coords = {
        v.id: tuple(float(c) for c in cube.vertex_position(v))
        for v in cube.vertices
    }
    result = reconcile_planes(
        cube,
        vertex_coords,
        tolerance_m=0.001,
        max_iterations=5,
    )
    assert result.converged
    assert result.plane_shifts_applied == 0
    assert result.final_max_drift_m < 1e-9


def test_reconcile_planes_respects_max_iterations():
    """When inputs are inconsistent (planes can't possibly land all
    vertices simultaneously within tolerance), reconciliation stops
    after max_iterations rather than spinning forever."""
    cube = make_cube(size=2.0)
    # Create explicit coords that don't match the cube's planes — pull
    # one vertex 10 cm away. Reconciliation can't fully fix this because
    # only the 3 incident planes can move.
    vertex_coords = {
        v.id: tuple(float(c) for c in cube.vertex_position(v))
        for v in cube.vertices
    }
    target_v = cube.vertices[0]
    p = np.asarray(vertex_coords[target_v.id], dtype=float)
    vertex_coords[target_v.id] = (
        float(p[0] + 0.1),
        float(p[1]),
        float(p[2]),
    )
    result = reconcile_planes(
        cube,
        vertex_coords,
        tolerance_m=1e-6,
        max_iterations=3,
    )
    assert result.iterations == 3
