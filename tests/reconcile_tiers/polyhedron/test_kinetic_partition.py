from __future__ import annotations

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import (
    BoundingPrism,
    FaceSelectionResult,
    assemble_polyhedron,
    cells_to_candidate_faces,
    kinetic_partition,
    label_cells_inside_outside,
    make_cube,
    make_gable_house,
    validate_polyhedron,
)


def test_kinetic_two_planes_one_cell():
    cells = kinetic_partition(
        [
            Plane(a=1.0, b=0.0, c=0.0, d=-0.5),
            Plane(a=1.0, b=0.0, c=0.0, d=0.5),
        ],
        scan_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        bounding_prism=BoundingPrism(-2.0, 2.0, -1.0, 1.0, -1.0, 1.0),
    )
    labels = label_cells_inside_outside(
        cells,
        np.asarray([[0.0, 0.0, 0.0]], dtype=float),
    )

    assert len(cells) == 3
    assert list(labels.values()).count("inside") == 1
    inside_cell = next(cell for cell in cells if labels[cell.cell_id] == "inside")
    assert inside_cell.signs == (1, -1)


def test_kinetic_cube():
    cube = make_cube(size=2.0)
    cells = kinetic_partition(
        [face.plane for face in cube.faces],
        scan_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        bounding_prism=BoundingPrism(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
    )
    labels = label_cells_inside_outside(
        cells,
        np.asarray([[0.0, 0.0, 0.0]], dtype=float),
    )
    candidates = cells_to_candidate_faces(cells, labels)
    polyhedron = assemble_polyhedron(
        FaceSelectionResult(
            selected=tuple(candidates),
            objective=0.0,
            energy_breakdown={},
            solver_status="kinetic",
            elapsed_seconds=0.0,
        )
    )

    assert len(cells) == 27
    assert list(labels.values()).count("inside") == 1
    assert len(candidates) == 6
    assert validate_polyhedron(polyhedron) == []


def test_kinetic_gable_house():
    gable = make_gable_house(width=6.0, depth=8.0, eave_height=2.5, ridge_rise=1.5)
    cells = kinetic_partition(
        [face.plane for face in gable.faces],
        scan_points=np.asarray([[3.0, 1.0, 4.0]], dtype=float),
        bounding_prism=BoundingPrism(-1.0, 7.0, -1.0, 5.0, -1.0, 9.0),
    )
    labels = label_cells_inside_outside(
        cells,
        np.asarray([[3.0, 1.0, 4.0]], dtype=float),
    )
    candidates = cells_to_candidate_faces(cells, labels)
    polyhedron = assemble_polyhedron(
        FaceSelectionResult(
            selected=tuple(candidates),
            objective=0.0,
            energy_breakdown={},
            solver_status="kinetic",
            elapsed_seconds=0.0,
        )
    )

    assert len(cells) >= 8
    assert list(labels.values()).count("inside") == 1
    assert len(candidates) == 7
    assert validate_polyhedron(polyhedron) == []


def test_kinetic_dedupes_duplicate_planes_for_large_batches():
    planes = [Plane(a=1.0, b=0.0, c=0.0, d=0.0) for _ in range(1000)]

    cells = kinetic_partition(
        planes,
        scan_points=np.asarray([[0.5, 0.0, 0.0]], dtype=float),
        bounding_prism=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )

    assert len(cells) == 2
