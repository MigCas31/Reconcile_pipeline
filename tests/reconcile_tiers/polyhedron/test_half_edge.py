"""Phase 1 tests: half-edge structure, vertex derivation, face_shift, cube."""

from __future__ import annotations

import math

import numpy as np
import pytest

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import (
    build_from_planar_polygons,
    make_cube,
    make_gable_house,
    three_plane_intersection,
)

# ---- three_plane_intersection -----------------------------------------------


def test_three_plane_intersection_origin():
    p_x = Plane(a=1.0, b=0.0, c=0.0, d=0.0)
    p_y = Plane(a=0.0, b=1.0, c=0.0, d=0.0)
    p_z = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    pt = three_plane_intersection(p_x, p_y, p_z)
    np.testing.assert_allclose(pt, [0.0, 0.0, 0.0], atol=1e-12)


def test_three_plane_intersection_offset():
    p_x = Plane(a=1.0, b=0.0, c=0.0, d=2.0)
    p_y = Plane(a=0.0, b=1.0, c=0.0, d=3.0)
    p_z = Plane(a=0.0, b=0.0, c=1.0, d=5.0)
    pt = three_plane_intersection(p_x, p_y, p_z)
    np.testing.assert_allclose(pt, [2.0, 3.0, 5.0], atol=1e-12)


def test_three_plane_intersection_degenerate_raises():
    p1 = Plane(a=1.0, b=0.0, c=0.0, d=0.0)
    p2 = Plane(a=1.0, b=0.0, c=0.0, d=1.0)  # parallel to p1
    p3 = Plane(a=0.0, b=1.0, c=0.0, d=0.0)
    with pytest.raises(np.linalg.LinAlgError):
        three_plane_intersection(p1, p2, p3)


# ---- cube fixture -----------------------------------------------------------


def test_cube_topology_counts():
    cube = make_cube(size=2.0)
    assert len(cube.faces) == 6
    assert len(cube.vertices) == 8
    assert len(cube.half_edges) == 24  # 6 faces x 4 edges, each edge has 2 sides


def test_cube_is_watertight():
    cube = make_cube(size=2.0)
    assert cube.is_watertight()


def test_cube_face_loops_close():
    cube = make_cube(size=2.0)
    assert cube.faces_close()


def test_cube_every_vertex_has_three_incident_faces():
    cube = make_cube(size=2.0)
    for v in cube.vertices:
        faces = cube.incident_faces(v)
        assert len(faces) == 3, f"vertex {v.id} has {len(faces)} incident faces"


def test_cube_every_face_has_four_boundary_half_edges():
    cube = make_cube(size=2.0)
    for face in cube.faces:
        h0 = face.half_edge
        assert h0 is not None
        count = 0
        h = h0
        for _ in range(8):
            count += 1
            h = h.next
            if h is h0:
                break
        assert count == 4, f"face {face.id} has {count}-edge boundary"


def test_cube_vertex_positions_at_corners():
    cube = make_cube(size=2.0)
    expected = set()
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                expected.add((sx, sy, sz))

    actual: set[tuple[float, float, float]] = set()
    for v in cube.vertices:
        pos = cube.vertex_position(v)
        actual.add((round(pos[0], 9), round(pos[1], 9), round(pos[2], 9)))

    assert actual == expected, (
        f"missing corners: {expected - actual}; extras: {actual - expected}"
    )


def test_cube_size_5():
    cube = make_cube(size=5.0)
    coords = np.array([cube.vertex_position(v) for v in cube.vertices])
    assert math.isclose(coords.min(), -2.5, abs_tol=1e-9)
    assert math.isclose(coords.max(), 2.5, abs_tol=1e-9)


# ---- face_shift -------------------------------------------------------------


def test_face_shift_moves_only_affected_vertices():
    """Shifting +X face by +1 moves the four +X vertices to x=2 while the four
    -X vertices stay at x=-1. This is the core demonstration of the paper's
    representation: change one plane, four vertices auto-update.
    """
    cube = make_cube(size=2.0)  # vertices at +/-1 on each axis
    plus_x_face = next(f for f in cube.faces if f.plane.a > 0.5)
    cube.face_shift(plus_x_face, delta=1.0)

    plus_x_xs = []
    minus_x_xs = []
    for v in cube.vertices:
        pos = cube.vertex_position(v)
        # Vertex id bit 2 is the +X side (per make_cube vid encoding).
        if v.id & 0b100:
            plus_x_xs.append(pos[0])
        else:
            minus_x_xs.append(pos[0])

    assert all(math.isclose(x, 2.0, abs_tol=1e-9) for x in plus_x_xs), plus_x_xs
    assert all(math.isclose(x, -1.0, abs_tol=1e-9) for x in minus_x_xs), minus_x_xs


def test_face_shift_preserves_y_and_z_of_affected_vertices():
    """Shifting +X face moves vertices in X only; Y and Z stay put."""
    cube = make_cube(size=2.0)
    plus_x_face = next(f for f in cube.faces if f.plane.a > 0.5)
    before = {v.id: cube.vertex_position(v).copy() for v in cube.vertices}
    cube.face_shift(plus_x_face, delta=0.5)
    after = {v.id: cube.vertex_position(v) for v in cube.vertices}
    for vid, pos_before in before.items():
        pos_after = after[vid]
        assert math.isclose(pos_after[1], pos_before[1], abs_tol=1e-9)
        assert math.isclose(pos_after[2], pos_before[2], abs_tol=1e-9)


def test_face_shift_keeps_invariants():
    """face_shift is a pure plane edit; topology is unchanged."""
    cube = make_cube(size=2.0)
    plus_y_face = next(f for f in cube.faces if f.plane.b > 0.5)
    cube.face_shift(plus_y_face, delta=10.0)
    assert cube.is_watertight()
    assert cube.faces_close()
    for v in cube.vertices:
        assert len(cube.incident_faces(v)) == 3


def test_face_shift_negative_delta_pulls_inward():
    cube = make_cube(size=4.0)  # vertices at +/-2
    plus_z_face = next(f for f in cube.faces if f.plane.c > 0.5)
    cube.face_shift(plus_z_face, delta=-1.0)
    plus_z_zs = [
        cube.vertex_position(v)[2] for v in cube.vertices if v.id & 0b001
    ]
    assert all(math.isclose(z, 1.0, abs_tol=1e-9) for z in plus_z_zs)


# ---- face polygon recovery --------------------------------------------------


def test_face_polygon_returns_four_corners():
    cube = make_cube(size=2.0)
    for face in cube.faces:
        poly = cube.face_polygon(face)
        assert len(poly) == 4


def test_face_polygon_corners_lie_on_plane():
    cube = make_cube(size=2.0)
    for face in cube.faces:
        plane = face.plane
        for corner in cube.face_polygon(face):
            x, y, z = corner
            residual = plane.a * x + plane.b * y + plane.c * z - plane.d
            assert abs(residual) < 1e-9, f"corner {corner} off plane {plane}"


# ---- regression: the corner-clipping bug cannot exist here ------------------


def test_oblique_face_does_not_extrapolate_below_floor():
    """The paper's representation rules out the big_yspan bug: a sloped face
    cannot have corners below the floor face because corners are *defined* as
    the intersection of the slope plane with adjacent faces (incl. the floor).

    We demonstrate by tilting one cube face into a steep oblique and
    confirming all vertex Y values stay within the (still flat) -Y..+Y
    range. The cube becomes a non-cuboid solid but no vertex escapes the
    bounding faces.
    """
    cube = make_cube(size=2.0)
    # Replace +Z face with a steep oblique tilted ~46 deg from horizontal.
    plus_z = next(f for f in cube.faces if f.plane.c > 0.5)
    plus_z.plane = Plane(
        a=-0.636, b=0.690, c=-0.346, d=2.0
    )  # similar tilt to the 1f03f6e0 worst case

    # All vertex Y values must stay within the +Y/-Y face Y range = [-1, +1],
    # because the +Y/-Y faces are still part of the polyhedron and any
    # vertex on the new oblique is also on either +Y or -Y (or +X/-X).
    for v in cube.vertices:
        pos = cube.vertex_position(v)
        assert -1.0 - 1e-9 <= pos[1] <= 1.0 + 1e-9, (
            f"vertex {v.id} escaped Y bounds: {pos}"
        )


# ---- build_from_planar_polygons --------------------------------------------


def _tetrahedron_polys() -> list[tuple[list[list[float]], Plane]]:
    """A regular-ish tetrahedron with vertices at standard positions.

    Vertices: V0=(1,1,1), V1=(1,-1,-1), V2=(-1,1,-1), V3=(-1,-1,1).
    Faces: each excludes one vertex; outward CCW order seen from outside.

    Plane equations precomputed by hand (centroid at origin, faces face out).
    """
    v0 = [1.0, 1.0, 1.0]
    v1 = [1.0, -1.0, -1.0]
    v2 = [-1.0, 1.0, -1.0]
    v3 = [-1.0, -1.0, 1.0]
    # Face F0 excludes V0: vertices V1, V2, V3 — outward normal points away
    # from V0 = away from (+1,+1,+1) = (-1,-1,-1)/sqrt(3).
    # Plane: -x - y - z = 1 (any of v1, v2, v3 satisfies it).
    # Normalize: a*x + b*y + c*z = d with (a,b,c) unit normal.
    s3 = 1.0 / math.sqrt(3.0)
    p0 = Plane(a=-s3, b=-s3, c=-s3, d=s3)  # passes through v1: -s3-(-s3)-(-s3)=s3 ✓
    p1 = Plane(a=s3, b=s3, c=-s3, d=s3)  # excludes v1, normal away from (1,-1,-1)
    p2 = Plane(a=s3, b=-s3, c=s3, d=s3)  # excludes v2
    p3 = Plane(a=-s3, b=s3, c=s3, d=s3)  # excludes v3
    return [
        # F0 = (V1, V2, V3) excludes V0.
        # CCW seen from outside (looking from V0 toward face): V1 -> V3 -> V2.
        ([v1, v3, v2], p0),
        # F1 = (V0, V3, V2) excludes V1. Looking from V1 toward face: V0 -> V2 -> V3.
        ([v0, v2, v3], p1),
        # F2 = (V0, V1, V3) excludes V2. Looking from V2: V0 -> V3 -> V1.
        ([v0, v3, v1], p2),
        # F3 = (V0, V1, V2) excludes V3. Looking from V3: V0 -> V1 -> V2.
        ([v0, v1, v2], p3),
    ]


def test_build_tetrahedron_topology():
    poly = build_from_planar_polygons(_tetrahedron_polys())
    assert len(poly.faces) == 4
    assert len(poly.vertices) == 4
    assert len(poly.half_edges) == 12  # 4 faces x 3 edges, each shared
    assert poly.is_watertight()
    assert poly.faces_close()


def test_build_tetrahedron_vertex_positions_match_input():
    polys = _tetrahedron_polys()
    poly = build_from_planar_polygons(polys)
    # Every input corner should appear as a derived vertex position.
    expected = set()
    for corners, _plane in polys:
        for c in corners:
            expected.add((round(c[0], 6), round(c[1], 6), round(c[2], 6)))
    actual = set()
    for v in poly.vertices:
        pos = poly.vertex_position(v)
        actual.add((round(pos[0], 6), round(pos[1], 6), round(pos[2], 6)))
    assert actual == expected, (
        f"missing: {expected - actual}; extra: {actual - expected}"
    )


def test_build_rejects_non_watertight():
    # Three triangles sharing a vertex (an open surface) — not closed.
    p_top = Plane(a=0.0, b=1.0, c=0.0, d=1.0)
    open_polys = [
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], p_top),
    ]
    with pytest.raises(ValueError, match="non-watertight"):
        build_from_planar_polygons(open_polys)


def test_build_rejects_duplicate_directed_edge():
    # Two faces with the same orientation along a shared edge — not a manifold.
    plane = Plane(a=0.0, b=1.0, c=0.0, d=0.0)
    polys = [
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 1.0]], plane),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, -1.0]], plane),
    ]
    with pytest.raises(ValueError, match=r"duplicate directed edge|non-watertight"):
        build_from_planar_polygons(polys)


# ---- make_gable_house -------------------------------------------------------


def test_gable_house_topology():
    house = make_gable_house()
    assert len(house.faces) == 7  # floor, 2 sides, 2 gable ends, 2 roof slopes
    assert len(house.vertices) == 10  # 4 floor + 4 eave + 2 ridge
    # Edges: 4 (floor) + 4 (eave) + 1 (ridge) + 4 (vertical at corners) +
    # 2 (gable diagonal -X side) + 2 (gable diagonal +X side ... wait, let me
    # just count from half-edges and divide).
    # Each face contributes its corner-count of half-edges. 4+4+4+5+5+4+4 = 30
    # half-edges → 15 edges (each shared by 2 faces).
    assert len(house.half_edges) == 30
    assert house.is_watertight()
    assert house.faces_close()


def test_gable_house_vertex_positions_match_input():
    house = make_gable_house(width=6.0, depth=8.0, eave_height=2.5, ridge_rise=1.5)
    expected_corners = {
        (0.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (6.0, 0.0, 8.0),
        (0.0, 0.0, 8.0),
        (0.0, 2.5, 0.0),
        (6.0, 2.5, 0.0),
        (6.0, 2.5, 8.0),
        (0.0, 2.5, 8.0),
        (3.0, 4.0, 0.0),
        (3.0, 4.0, 8.0),
    }
    actual = {
        tuple(round(c, 6) for c in house.vertex_position(v)) for v in house.vertices
    }
    assert actual == expected_corners


def test_gable_house_face_shift_lifts_ridge():
    """Shifting both roof-slope faces outward by the same amount raises the
    ridge while keeping the eaves where they are. This is the core demo of
    the paper for a real building shape: edit one plane, watch coupled
    geometry update without touching any vertex.
    """
    house = make_gable_house(width=6.0, depth=8.0, eave_height=2.5, ridge_rise=1.5)

    def find_ridge_y(poly):
        ys = [poly.vertex_position(v)[1] for v in poly.vertices]
        return max(ys)

    initial_ridge = find_ridge_y(house)
    assert math.isclose(initial_ridge, 4.0, abs_tol=1e-6)

    # The two roof slopes: the only faces with non-zero plane.b but plane.b < 1.
    roof_slopes = [
        f for f in house.faces if 0.01 < abs(f.plane.b) < 0.99 and abs(f.plane.c) < 1e-6
    ]
    assert len(roof_slopes) == 2

    # Shift each slope outward by 0.3 along its normal. The ridge should rise.
    for f in roof_slopes:
        house.face_shift(f, delta=0.3)

    new_ridge = find_ridge_y(house)
    assert new_ridge > initial_ridge
    # Geometric check: shifting two symmetric roof slopes outward by δ each,
    # the ridge rises by δ / sin(half-angle from horizontal). With our slope
    # normal (a, b) where a = ridge_rise/hyp and b = (w/2)/hyp, the rise per
    # unit shift is 1/b. So expected new ridge = 4.0 + 0.3/b.
    b = roof_slopes[0].plane.b
    expected = 4.0 + 0.3 / b
    assert math.isclose(new_ridge, expected, abs_tol=1e-6)
    assert house.is_watertight()


def test_gable_house_eaves_move_when_roof_shifts():
    """Eave vertices are at (side_wall ∩ gable_end ∩ roof_slope) — three
    planes, of which the roof plane changed. So eaves correctly slide up
    the side+gable vertical line as the roof shifts outward. We verify the
    expected Y from the roof plane equation: y_eave = h + δ / n_y, where
    n_y is the roof normal's Y component.
    """
    h, w, r, delta = 2.5, 6.0, 1.5, 0.2
    house = make_gable_house(width=w, depth=8.0, eave_height=h, ridge_rise=r)
    roof_slopes = [
        f for f in house.faces if 0.01 < abs(f.plane.b) < 0.99 and abs(f.plane.c) < 1e-6
    ]
    n_y = roof_slopes[0].plane.b
    for f in roof_slopes:
        house.face_shift(f, delta=delta)
    expected_eave_y = h + delta / n_y
    eave_ys = [
        y
        for v in house.vertices
        for y in [house.vertex_position(v)[1]]
        if expected_eave_y - 0.1 < y < expected_eave_y + 0.1
    ]
    assert len(eave_ys) == 4
    for y in eave_ys:
        assert math.isclose(y, expected_eave_y, abs_tol=1e-6)


def test_gable_house_eaves_unaffected_by_floor_shift():
    """Eaves do NOT depend on the floor plane — shifting the floor up should
    leave eave Y unchanged.
    """
    house = make_gable_house()
    floor = next(f for f in house.faces if f.plane.b < -0.5)
    eave_ys_before = sorted(
        house.vertex_position(v)[1]
        for v in house.vertices
        if 2.0 < house.vertex_position(v)[1] < 3.0
    )
    house.face_shift(floor, delta=0.4)  # floor.d goes from 0 to 0.4 → floor rises
    eave_ys_after = sorted(
        house.vertex_position(v)[1]
        for v in house.vertices
        if 2.0 < house.vertex_position(v)[1] < 3.0
    )
    assert len(eave_ys_before) == len(eave_ys_after) == 4
    for a, b in zip(eave_ys_before, eave_ys_after, strict=True):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_cube_via_constructor_matches_make_cube_topology():
    """Reconstructing a cube via build_from_planar_polygons gives the same
    counts as the hand-coded make_cube.
    """
    cube = make_cube(size=2.0)
    polys = []
    for face in cube.faces:
        corners = [list(c) for c in cube.face_polygon(face)]
        polys.append((corners, face.plane))
    rebuilt = build_from_planar_polygons(polys)
    assert len(rebuilt.faces) == len(cube.faces)
    assert len(rebuilt.vertices) == len(cube.vertices)
    assert len(rebuilt.half_edges) == len(cube.half_edges)
    assert rebuilt.is_watertight()
