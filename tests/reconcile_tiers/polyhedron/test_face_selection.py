from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import (
    CandidateFace,
    SelectionWeights,
    assemble_polyhedron,
    build_edge_incidence,
    face_selection_trace,
    generate_candidates,
    make_cube,
    make_gable_house,
    solve_face_selection_ilp,
    validate_polyhedron,
)


def test_cube_recovery_with_stray_plane():
    cube = make_cube(size=2.0)
    planes = [face.plane for face in cube.faces]
    planes.append(Plane(a=1.0, b=0.0, c=0.0, d=10.0))

    candidates = generate_candidates(
        planes,
        domain_polygon=Polygon([(-1.1, -1.1), (1.1, -1.1), (1.1, 1.1), (-1.1, 1.1)]),
        bounding_prism=(-1.1, 1.1),
        scan_points=_face_center_points(cube),
        epsilon=0.05,
    )

    assert len(candidates) == 6
    selection = solve_face_selection_ilp(
        candidates,
        build_edge_incidence(candidates),
        SelectionWeights(),
    )
    polyhedron = assemble_polyhedron(selection)

    assert selection.solver_status.startswith("milp_")
    assert {candidate.plane_id for candidate in selection.selected} == set(range(6))
    assert validate_polyhedron(polyhedron) == []


def test_gable_recovery():
    gable = make_gable_house(width=6.0, depth=8.0, eave_height=2.5, ridge_rise=1.5)
    planes = [face.plane for face in gable.faces]

    candidates = generate_candidates(
        planes,
        domain_polygon=Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 8.0), (0.0, 8.0)]),
        bounding_prism=(-0.1, 4.2),
        scan_points=_face_center_points(gable),
        epsilon=0.05,
    )

    selection = solve_face_selection_ilp(
        candidates,
        build_edge_incidence(candidates),
        SelectionWeights(),
    )
    polyhedron = assemble_polyhedron(selection)

    assert len(candidates) == 7
    assert len(selection.selected) == 7
    assert validate_polyhedron(polyhedron) == []


def test_l_shape_recovery():
    footprint = Polygon(
        [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]
    )
    planes = [
        Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
        Plane(a=0.0, b=1.0, c=0.0, d=3.0),
        *_vertical_boundary_planes(footprint),
        Plane(a=1.0, b=0.0, c=0.0, d=8.0),  # stray unsupported plane
    ]

    candidates = generate_candidates(
        planes,
        domain_polygon=footprint,
        bounding_prism=(0.0, 3.0),
        scan_points=np.empty((0, 3), dtype=float),
        epsilon=0.05,
    )

    selection = solve_face_selection_ilp(
        candidates,
        build_edge_incidence(candidates),
        SelectionWeights(),
    )
    polyhedron = assemble_polyhedron(selection)

    assert len(candidates) == 8
    assert len(selection.selected) == 8
    assert {candidate.plane_id for candidate in selection.selected} == set(range(8))
    assert selection.energy_breakdown["coverage_ratio"] == 1.0
    assert validate_polyhedron(polyhedron) == []


def test_l_shape_prism_support_filter_is_all_or_nothing():
    footprint = Polygon(
        [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]
    )
    planes = [
        Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
        Plane(a=0.0, b=1.0, c=0.0, d=3.0),
        *_vertical_boundary_planes(footprint),
    ]

    candidates = generate_candidates(
        planes,
        domain_polygon=footprint,
        bounding_prism=(0.0, 3.0),
        scan_points=np.empty((0, 3), dtype=float),
        epsilon=0.05,
        min_support_points=1,
    )

    assert candidates == []


def test_manifold_constraint_forces_edge_pair_completion():
    cheap = _candidate(0, edge_keys=((0, 1),), support_score=10.0, area=1.0)
    required_partner = _candidate(1, edge_keys=((0, 1),), support_score=0.0, area=0.1)

    selection = solve_face_selection_ilp(
        [cheap, required_partner],
        build_edge_incidence([cheap, required_partner]),
        SelectionWeights(data_fit=1.0, complexity=0.0, coverage=0.0),
    )

    assert {candidate.face_id for candidate in selection.selected} == {0, 1}


def test_complexity_weight_penalizes_selected_sharp_edges():
    first = _candidate(
        0,
        edge_keys=((0, 1),),
        support_score=0.0,
        area=1.0,
        plane=Plane(a=1.0, b=0.0, c=0.0, d=0.0),
    )
    second = _candidate(
        1,
        edge_keys=((0, 1),),
        support_score=0.0,
        area=1.0,
        plane=Plane(a=0.0, b=1.0, c=0.0, d=0.0),
    )

    selection = solve_face_selection_ilp(
        [first, second],
        build_edge_incidence([first, second]),
        SelectionWeights(data_fit=0.0, complexity=1.0, coverage=0.0),
    )

    assert {candidate.face_id for candidate in selection.selected} == {0, 1}
    assert selection.energy_breakdown["complexity"] == 1.0
    assert selection.objective > 0.9


def test_coverage_breakdown_uses_projected_union_not_face_area_sum():
    footprint = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    first = _candidate(
        0,
        edge_keys=((0, 1),),
        support_score=0.0,
        area=1.0,
        coverage_polygon=footprint,
        domain_area=1.0,
    )
    duplicate = _candidate(
        1,
        edge_keys=((0, 1),),
        support_score=0.0,
        area=1.0,
        coverage_polygon=footprint,
        domain_area=1.0,
    )

    selection = solve_face_selection_ilp(
        [first, duplicate],
        build_edge_incidence([first, duplicate]),
        SelectionWeights(data_fit=0.0, complexity=0.0, coverage=1.0),
    )

    assert {candidate.face_id for candidate in selection.selected} == {0, 1}
    assert selection.energy_breakdown["coverage_ratio"] == 1.0
    assert selection.energy_breakdown["coverage"] == 0.0


def test_face_selection_trace_is_json_friendly_summary():
    first = _candidate(0, edge_keys=((0, 1),), support_score=1.0, area=1.0)
    second = _candidate(1, edge_keys=((0, 1),), support_score=1.0, area=1.0)
    candidates = [first, second]
    incidence = build_edge_incidence(candidates)
    selection = solve_face_selection_ilp(candidates, incidence)

    trace = face_selection_trace(candidates, incidence, selection)

    assert trace["candidate_count"] == 2
    assert trace["edge_incidence"] == {
        "min": 2,
        "max": 2,
        "open_edges": 0,
        "manifold_edges": 1,
        "overfull_edges": 0,
    }
    assert trace["selection"]["selected_count"] == 2
    assert trace["candidates"][0]["selected"] is True


def test_lp_relax_fallback_path_still_repairs_cube():
    cube = make_cube(size=2.0)
    candidates = generate_candidates(
        [face.plane for face in cube.faces],
        domain_polygon=Polygon([(-1.1, -1.1), (1.1, -1.1), (1.1, 1.1), (-1.1, 1.1)]),
        bounding_prism=(-1.1, 1.1),
        scan_points=_face_center_points(cube),
        epsilon=0.05,
    )

    selection = solve_face_selection_ilp(
        candidates,
        build_edge_incidence(candidates),
        SelectionWeights(),
        time_budget_seconds=0.0,
    )
    polyhedron = assemble_polyhedron(selection)

    assert selection.solver_status == "lp_relaxation_time_budget"
    assert len(selection.selected) == 6
    assert validate_polyhedron(polyhedron) == []


def test_lp_relax_fallback_handles_non_contiguous_face_ids():
    first = _candidate(10, edge_keys=((0, 1),), support_score=10.0, area=1.0)
    second = _candidate(20, edge_keys=((0, 1),), support_score=0.0, area=0.1)

    selection = solve_face_selection_ilp(
        [first, second],
        build_edge_incidence([first, second]),
        SelectionWeights(data_fit=1.0, complexity=0.0, coverage=0.0),
        time_budget_seconds=0.0,
    )

    assert {candidate.face_id for candidate in selection.selected} == {10, 20}
    assert selection.objective < 0.0


def test_generate_candidates_refuses_unbounded_cubic_intersection_work():
    planes = [
        Plane(a=1.0, b=0.0, c=0.0, d=float(idx + 1))
        for idx in range(12)
    ]

    with pytest.raises(ValueError, match="three-plane intersections"):
        generate_candidates(
            planes,
            domain_polygon=Polygon(
                [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            ),
            bounding_prism=(0.0, 1.0),
            scan_points=np.empty((0, 3), dtype=float),
            max_intersections=10,
        )


def test_solve_face_selection_refuses_oversized_milp():
    candidates = [
        _candidate(idx, edge_keys=((idx, idx + 1),), support_score=1.0, area=1.0)
        for idx in range(4)
    ]

    with pytest.raises(ValueError, match="candidate cap"):
        solve_face_selection_ilp(
            candidates,
            build_edge_incidence(candidates),
            max_candidates=3,
        )


def _face_center_points(polyhedron) -> np.ndarray:
    points = []
    for face in polyhedron.faces:
        corners = np.asarray(polyhedron.face_polygon(face), dtype=float)
        points.append(corners.mean(axis=0))
    return np.asarray(points, dtype=float)


def _candidate(
    face_id: int,
    *,
    edge_keys: tuple[tuple[int, int], ...],
    support_score: float,
    area: float,
    plane: Plane | None = None,
    coverage_polygon: Polygon | None = None,
    domain_area: float = 0.0,
) -> CandidateFace:
    plane = plane or Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    return CandidateFace(
        face_id=face_id,
        plane_id=face_id,
        polygon=Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),
        edge_keys=edge_keys,
        supporting_points=np.empty((0, 3), dtype=float),
        support_density=0.0,
        confidence_label="test",
        corners=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        plane=plane,
        area=area,
        support_score=support_score,
        coverage_polygon=coverage_polygon or Polygon(),
        domain_area=domain_area,
    )


def _vertical_boundary_planes(footprint: Polygon) -> list[Plane]:
    coords = [(float(x), float(z)) for x, z in footprint.exterior.coords[:-1]]
    planes: list[Plane] = []
    for idx, start in enumerate(coords):
        end = coords[(idx + 1) % len(coords)]
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = float((dx * dx + dz * dz) ** 0.5)
        nx = dz / length
        nz = -dx / length
        planes.append(
            Plane(
                a=nx,
                b=0.0,
                c=nz,
                d=nx * start[0] + nz * start[1],
            )
        )
    return planes
