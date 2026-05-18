from __future__ import annotations

import pytest

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import (
    TopologyResolutionError,
    build_from_planar_polygons,
    detect_topological_events,
    make_cube,
    resolve_single_adjacent_coplanar_face_merge,
    resolve_single_edge_collapse,
    resolve_single_edge_flip,
    resolve_single_face_creation,
    resolve_single_triangle_face_collapse,
    resolve_supported_topology_events,
    validate_polyhedron,
)
from tests.reconcile_tiers.polyhedron.test_validity import _triangular_prism_polys


def test_resolve_single_edge_collapse_rewires_selected_zero_edge():
    cube = _cube_with_four_zero_z_edges()
    event = _edge_events(cube)[0]

    result = resolve_single_edge_collapse(cube, event, edge_tol_m=1e-9)

    assert result.removed_half_edge_ids == event.ids
    assert len(cube.vertices) == 7
    assert len(cube.half_edges) == 22
    assert all(
        half_edge.id not in result.removed_half_edge_ids
        for half_edge in cube.half_edges
    )
    assert validate_polyhedron(cube) == []
    # This resolver intentionally handles only one edge; the other simultaneous
    # collapses remain queued for future event-resolution passes.
    assert _edge_events(cube)


def test_resolve_single_edge_collapse_requires_explicit_event_when_multiple_exist():
    cube = _cube_with_four_zero_z_edges()

    with pytest.raises(TopologyResolutionError, match="expected one"):
        resolve_single_edge_collapse(cube, edge_tol_m=1e-9)


def test_resolve_single_edge_collapse_rejects_missing_event():
    cube = make_cube(size=2.0)

    with pytest.raises(TopologyResolutionError, match="no edge-collapse"):
        resolve_single_edge_collapse(cube, edge_tol_m=1e-9)


def test_resolve_single_edge_collapse_rejects_triangle_adjacent_case():
    prism = build_from_planar_polygons(_triangular_prism_polys())
    collapse = next(
        face
        for face in prism.faces
        if face.plane.b > 0.5 and face.plane.c > 0.5
    )
    collapse_delta = -(1.0 / (2.0**0.5))
    prism.face_shift(collapse, delta=collapse_delta)
    event = _edge_events(prism)[0]

    with pytest.raises(TopologyResolutionError, match="face-collapse"):
        resolve_single_edge_collapse(prism, event, edge_tol_m=1e-9)


def test_resolve_single_triangle_face_collapse_removes_collapsed_triangle():
    cube = _cube_with_four_zero_z_edges()
    edge_event = _edge_events(cube)[0]
    resolve_single_edge_collapse(cube, edge_event, edge_tol_m=1e-9)
    triangle_event = next(
        event
        for event in detect_topological_events(
            cube,
            edge_tol_m=1e-9,
            face_area_tol_m2=1e-9,
        )
        if event.kind == "triangle_face_collapse"
    )

    result = resolve_single_triangle_face_collapse(
        cube,
        triangle_event,
        edge_tol_m=1e-9,
        face_area_tol_m2=1e-9,
    )

    assert result.removed_face_id == triangle_event.ids[0]
    assert len(cube.faces) == 5
    assert len(cube.vertices) == 6
    assert len(cube.half_edges) == 18
    assert all(
        half_edge.id not in result.removed_half_edge_ids
        for half_edge in cube.half_edges
    )
    issues = validate_polyhedron(cube)
    assert {issue.kind for issue in issues} == {"adjacent_coplanar_faces"}


def test_resolve_single_triangle_face_collapse_requires_explicit_event_when_multiple():
    cube = _cube_with_four_zero_z_edges()
    resolve_single_edge_collapse(cube, _edge_events(cube)[0], edge_tol_m=1e-9)

    with pytest.raises(TopologyResolutionError, match="expected one"):
        resolve_single_triangle_face_collapse(
            cube,
            edge_tol_m=1e-9,
            face_area_tol_m2=1e-9,
        )


def test_resolve_single_triangle_face_collapse_rejects_point_collapse():
    prism = build_from_planar_polygons(_triangular_prism_polys())
    collapse = next(
        face
        for face in prism.faces
        if face.plane.b > 0.5 and face.plane.c > 0.5
    )
    prism.face_shift(collapse, delta=-(1.0 / (2.0**0.5)))
    triangle_event = next(
        event
        for event in detect_topological_events(
            prism,
            edge_tol_m=1e-9,
            face_area_tol_m2=1e-9,
        )
        if event.kind == "triangle_face_collapse"
    )

    with pytest.raises(TopologyResolutionError, match="exactly one zero edge"):
        resolve_single_triangle_face_collapse(
            prism,
            triangle_event,
            edge_tol_m=1e-9,
            face_area_tol_m2=1e-9,
        )


def test_resolve_single_adjacent_coplanar_face_merge_removes_internal_split():
    box = build_from_planar_polygons(_box_with_split_top_polys())
    issues = validate_polyhedron(box)
    assert {issue.kind for issue in issues} == {"adjacent_coplanar_faces"}

    result = resolve_single_adjacent_coplanar_face_merge(box)

    assert result.kept_face_id != result.removed_face_id
    assert len(box.faces) == 6
    assert len(box.vertices) == 8
    assert len(box.half_edges) == 24
    assert all(
        half_edge.id not in result.removed_half_edge_ids for half_edge in box.half_edges
    )
    assert validate_polyhedron(box) == []


def test_resolve_single_adjacent_coplanar_face_merge_rejects_missing_pair():
    cube = make_cube(size=2.0)

    with pytest.raises(TopologyResolutionError, match="no adjacent coplanar"):
        resolve_single_adjacent_coplanar_face_merge(cube)


def test_resolve_single_face_creation_splits_face_through_existing_vertices():
    cube = make_cube(size=2.0)
    top = next(face for face in cube.faces if face.plane.b > 0.5)

    result = resolve_single_face_creation(
        cube,
        Plane(a=1.0, b=0.0, c=-1.0, d=0.0),
        top.id,
    )

    assert result.applied
    assert result.target_face_id == top.id
    assert len(result.created_face_ids) == 2
    assert len(cube.faces) == 7
    assert len(cube.vertices) == 8
    assert len(cube.half_edges) == 26
    assert cube.is_watertight()
    assert cube.faces_close()
    assert {issue.kind for issue in validate_polyhedron(cube)} == {
        "adjacent_coplanar_faces"
    }


def test_resolve_single_face_creation_rejects_boundary_vertex_insertion():
    cube = make_cube(size=2.0)
    top = next(face for face in cube.faces if face.plane.b > 0.5)

    with pytest.raises(TopologyResolutionError, match="existing vertices"):
        resolve_single_face_creation(
            cube,
            Plane(a=1.0, b=0.0, c=0.0, d=0.0),
            top.id,
        )


def test_resolve_single_edge_flip_replaces_triangular_diagonal():
    cube = make_cube(size=2.0)
    top = next(face for face in cube.faces if face.plane.b > 0.5)
    resolve_single_face_creation(
        cube,
        Plane(a=1.0, b=0.0, c=-1.0, d=0.0),
        top.id,
    )
    before_diagonal = _shared_top_diagonal(cube)

    result = resolve_single_edge_flip(cube, before_diagonal.id)

    after_diagonal = _shared_top_diagonal(cube)
    assert result.applied
    assert result.old_half_edge_ids == (
        before_diagonal.id,
        before_diagonal.opposite.id,
    )
    assert len(cube.faces) == 7
    assert len(cube.vertices) == 8
    assert len(cube.half_edges) == 26
    assert cube.is_watertight()
    assert cube.faces_close()
    assert _edge_endpoints(cube, after_diagonal) == {
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
    }


def test_resolve_single_edge_flip_requires_two_triangles():
    cube = make_cube(size=2.0)
    side_edge = next(
        half_edge
        for half_edge in cube.half_edges
        if half_edge.face is not None
        and half_edge.opposite is not None
        and half_edge.face.plane.b > 0.5
    )

    with pytest.raises(TopologyResolutionError, match="two triangles"):
        resolve_single_edge_flip(cube, side_edge.id)


def test_resolve_supported_topology_events_merges_unique_coplanar_issue():
    box = build_from_planar_polygons(_box_with_split_top_polys())

    trace = resolve_supported_topology_events(box)

    assert trace.stop_reason == "valid"
    assert trace.remaining_issues == ()
    assert trace.remaining_events == ()
    assert [step.action for step in trace.steps] == ["adjacent_coplanar_face_merge"]
    step = trace.steps[0]
    assert step.trigger_ids == (5, 6)
    assert step.before.faces == 7
    assert step.before.vertices == 10
    assert step.before.half_edges == 30
    assert step.after.faces == 6
    assert step.after.vertices == 8
    assert step.after.half_edges == 24
    assert validate_polyhedron(box) == []


def test_resolve_supported_topology_events_stops_on_ambiguous_events_by_default():
    cube = _cube_with_four_zero_z_edges()

    trace = resolve_supported_topology_events(
        cube,
        edge_tol_m=1e-9,
        face_area_tol_m2=1e-9,
    )

    assert trace.stop_reason == "ambiguous_events"
    assert trace.steps == ()
    assert len(trace.remaining_events) == 4
    assert len(cube.faces) == 6
    assert len(cube.vertices) == 8
    assert len(cube.half_edges) == 24


def test_resolve_supported_topology_events_first_mode_records_steps():
    cube = _cube_with_four_zero_z_edges()

    trace = resolve_supported_topology_events(
        cube,
        selection="first",
        max_steps=3,
        edge_tol_m=1e-9,
        face_area_tol_m2=1e-9,
    )

    assert trace.stop_reason == "max_steps"
    assert [step.action for step in trace.steps] == [
        "edge_collapse",
        "triangle_face_collapse",
        "adjacent_coplanar_face_merge",
    ]
    assert trace.steps[0].before.faces == 6
    assert trace.steps[0].after.vertices == 7
    assert trace.steps[1].after.faces == 5
    assert trace.steps[2].after.faces == 4
    assert trace.remaining_issues == ()
    assert {event.kind for event in trace.remaining_events} == {"edge_collapse"}


def _cube_with_four_zero_z_edges():
    cube = make_cube(size=2.0)
    plus_z = next(face for face in cube.faces if face.plane.c > 0.5)
    plus_z.plane = Plane(a=0.0, b=0.0, c=1.0, d=-1.0)
    return cube


def _edge_events(polyhedron):
    return [
        event
        for event in detect_topological_events(polyhedron, edge_tol_m=1e-9)
        if event.kind == "edge_collapse"
    ]


def _shared_top_diagonal(polyhedron):
    top_faces = [face for face in polyhedron.faces if face.plane.b > 0.5]
    assert len(top_faces) == 2
    for half_edge in polyhedron.half_edges:
        if (
            half_edge.face in top_faces
            and half_edge.opposite is not None
            and half_edge.opposite.face in top_faces
        ):
            return half_edge
    raise AssertionError("missing shared top diagonal")


def _edge_endpoints(polyhedron, half_edge):
    assert half_edge.next is not None
    return {
        tuple(round(float(value), 6) for value in polyhedron.vertex_position(vertex))
        for vertex in (half_edge.origin, half_edge.next.origin)
    }


def _box_with_split_top_polys() -> list[tuple[list[list[float]], Plane]]:
    a = [0.0, 0.0, 0.0]
    c = [2.0, 0.0, 0.0]
    d = [2.0, 0.0, 1.0]
    f = [0.0, 0.0, 1.0]
    a2 = [0.0, 1.0, 0.0]
    b2 = [1.0, 1.0, 0.0]
    c2 = [2.0, 1.0, 0.0]
    d2 = [2.0, 1.0, 1.0]
    e2 = [1.0, 1.0, 1.0]
    f2 = [0.0, 1.0, 1.0]
    return [
        ([a, c, d, f], Plane(a=0.0, b=-1.0, c=0.0, d=0.0)),
        ([a, a2, b2, c2, c], Plane(a=0.0, b=0.0, c=-1.0, d=0.0)),
        ([c, c2, d2, d], Plane(a=1.0, b=0.0, c=0.0, d=2.0)),
        ([f, d, d2, e2, f2], Plane(a=0.0, b=0.0, c=1.0, d=1.0)),
        ([a, f, f2, a2], Plane(a=-1.0, b=0.0, c=0.0, d=0.0)),
        ([a2, f2, e2, b2], Plane(a=0.0, b=1.0, c=0.0, d=1.0)),
        ([b2, e2, d2, c2], Plane(a=0.0, b=1.0, c=0.0, d=1.0)),
    ]
