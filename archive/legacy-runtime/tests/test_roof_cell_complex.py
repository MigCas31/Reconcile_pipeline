from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from reconcile.roof_algorithms_py.occupied_room_cell_complex import (
    _annotate_boundary_classes,
    _build_best_candidate_cell,
    _candidate_footprints,
    _merged_partition_regions,
    build_occupied_room_cell_complex,
)
from reconcile.roof_algorithms_py.roof_arrangement_kernel import (
    build_arranged_polyhedral_cell,
)
from reconcile.roof_algorithms_py.roof_cell_complex import (
    _poly_xz_from_3d,
    build_roof_cell_complex,
    filter_knee_walls_by_occupied_shell,
)
from reconcile.roof_algorithms_py.roof_coverage_graph import build_roof_coverage_graph


def _rect(x0: float, z0: float, x1: float, z1: float, y: float) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def test_poly_xz_from_3d_recovers_largest_polygon_from_make_valid_geometry_collection():
    corners = [
        [-3.1533, 0.0, 0.7598],
        [-3.1043, 0.0, 0.8281],
        [-3.0851, 0.0, 0.8547],
        [-3.0974, 0.0, 0.8636],
        [-0.7176, 0.0, 4.1630],
        [2.6241, 0.0, 1.7527],
        [0.2443, 0.0, -1.5467],
        [2.6069, 0.0, 1.7288],
        [2.6715, 0.0, 1.6823],
        [0.7014, 0.0, -1.0492],
        [3.1794, 0.0, -2.8365],
        [5.0732, 0.0, -0.2110],
        [3.1176, 0.0, -2.9223],
        [3.1211, 0.0, -2.9248],
        [3.1211, 0.0, -2.9248],
        [0.9251, 0.0, -1.3409],
        [0.7144, 0.0, -1.6331],
        [0.4328, 0.0, -1.4300],
        [-1.4658, 0.0, -4.0621],
        [-2.6269, 0.0, -3.2246],
        [-1.5467, 0.0, -4.0037],
        [0.1619, 0.0, -1.6349],
        [-1.5707, 0.0, -0.3851],
        [-3.2793, 0.0, -2.7540],
        [-3.3625, 0.0, -2.6940],
        [-3.3648, 0.0, -2.6973],
        [-4.8641, 0.0, -1.6191],
        [-3.3648, 0.0, -2.6973],
        [-2.6974, 0.0, -1.7691],
        [-1.6541, 0.0, -0.3183],
    ]

    poly = _poly_xz_from_3d(corners)

    assert poly is not None
    assert poly.is_valid
    assert poly.area > 18.0


def test_occupied_cell_recovers_from_wall_bottom() -> None:
    bldg = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [],
                "walls_computed": [
                    {
                        "corners": [
                            [0.0, -1.0, 0.0],
                            [4.0, -1.0, 0.0],
                            [4.0, 1.4, 0.0],
                            [0.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, -1.0, 0.0],
                            [4.0, -1.0, 3.0],
                            [4.0, 1.4, 3.0],
                            [4.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, -1.0, 3.0],
                            [0.0, -1.0, 3.0],
                            [0.0, 1.4, 3.0],
                            [4.0, 1.4, 3.0],
                        ]
                    },
                    {
                        "corners": [
                            [0.0, -1.0, 3.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 1.4, 0.0],
                            [0.0, 1.4, 3.0],
                        ]
                    },
                ],
                "walls_merged": [],
            }
        ]
    }

    result = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=[],
        building_part_graph={},
    )

    assert result["metadata"]["room_count"] == 1
    assert result["metadata"]["cell_count"] >= 1
    cell = result["cells"][0]
    assert cell["room_index"] == 0
    assert cell["exact_source_kind"] == "synthetic_top_boundary_atom"
    assert result["metadata"]["synthetic_atom_cell_count"] >= 1
    assert cell["volume_m3"] > 0.0


def test_occupied_cell_prefers_richer_merged_walls_over_sparse_computed_walls() -> None:
    bldg = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [],
                "walls_computed": [
                    {
                        "corners": [
                            [0.0, -1.0, 0.0],
                            [4.0, -1.0, 0.0],
                            [4.0, 1.4, 0.0],
                            [0.0, 1.4, 0.0],
                        ]
                    },
                ],
                "walls_merged": [
                    {
                        "corners": [
                            [0.0, -1.0, 0.0],
                            [4.0, -1.0, 0.0],
                            [4.0, 1.4, 0.0],
                            [0.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, -1.0, 0.0],
                            [4.0, -1.0, 3.0],
                            [4.0, 1.4, 3.0],
                            [4.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, -1.0, 3.0],
                            [0.0, -1.0, 3.0],
                            [0.0, 1.4, 3.0],
                            [4.0, 1.4, 3.0],
                        ]
                    },
                    {
                        "corners": [
                            [0.0, -1.0, 3.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 1.4, 0.0],
                            [0.0, 1.4, 3.0],
                        ]
                    },
                ],
            }
        ]
    }

    result = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=[],
        building_part_graph={},
    )

    assert result["metadata"]["room_count"] == 1
    assert result["metadata"]["cell_count"] >= 1
    assert result["metadata"]["synthetic_atom_cell_count"] >= 1


def test_occupied_cell_falls_back_when_partition_surface_is_coplanar_with_floor() -> (
    None
):
    bldg = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": _rect(0.0, 0.0, 4.0, 3.0, -1.0),
                "walls_computed": [
                    {
                        "corners": [
                            [0.0, -1.0, 0.0],
                            [4.0, -1.0, 0.0],
                            [4.0, 1.4, 0.0],
                            [0.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, -1.0, 0.0],
                            [4.0, -1.0, 3.0],
                            [4.0, 1.4, 3.0],
                            [4.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, -1.0, 3.0],
                            [0.0, -1.0, 3.0],
                            [0.0, 1.4, 3.0],
                            [4.0, 1.4, 3.0],
                        ]
                    },
                    {
                        "corners": [
                            [0.0, -1.0, 3.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 1.4, 0.0],
                            [0.0, 1.4, 3.0],
                        ]
                    },
                ],
                "walls_merged": [],
            }
        ]
    }
    room_partitions = [
        {
            "room_index": 0,
            "story": 0,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 0,
                    "kind": "flat",
                    "poly": _rect(0.0, 0.0, 4.0, 3.0, -1.0),
                }
            ],
        }
    ]

    result = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=room_partitions,
        building_part_graph={},
    )

    assert result["metadata"]["room_count"] == 1
    assert result["metadata"]["cell_count"] >= 1
    assert result["metadata"]["fallback_cell_count"] == 0
    assert all(cell["volume_m3"] > 0.0 for cell in result["cells"])


def test_occupied_cell_prefers_partition_room_outline_over_room_floor_polygon() -> None:
    bldg = {
        "rooms": [
            {
                "story": 0,
                # Deliberately larger than the partitioned outline.
                "floor_polygon": _rect(0.0, 0.0, 6.0, 4.0, -1.0),
                "walls_computed": [
                    {
                        "corners": [
                            [0.0, -1.0, 0.0],
                            [6.0, -1.0, 0.0],
                            [6.0, 1.4, 0.0],
                            [0.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [6.0, -1.0, 0.0],
                            [6.0, -1.0, 4.0],
                            [6.0, 1.4, 4.0],
                            [6.0, 1.4, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [6.0, -1.0, 4.0],
                            [0.0, -1.0, 4.0],
                            [0.0, 1.4, 4.0],
                            [6.0, 1.4, 4.0],
                        ]
                    },
                    {
                        "corners": [
                            [0.0, -1.0, 4.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 1.4, 0.0],
                            [0.0, 1.4, 4.0],
                        ]
                    },
                ],
                "walls_merged": [],
            }
        ]
    }
    room_partitions = [
        {
            "room_index": 0,
            "story": 0,
            "room_outline": _rect(0.0, 0.0, 4.0, 3.0, -1.0),
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 0,
                    "kind": "flat",
                    "poly": _rect(0.0, 0.0, 4.0, 3.0, 2.0),
                }
            ],
        }
    ]

    result = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=room_partitions,
        building_part_graph={},
    )

    assert result["metadata"]["synthetic_atom_cell_count"] == 0
    assert result["metadata"]["cell_count"] >= 1


def test_occupied_cell_collinear_no_remainder() -> None:
    footprint = [
        [6.978, -3.871, -6.866],
        [6.978, -3.871, -5.335],
        [6.978, -3.871, -4.051],
        [6.978, -3.871, -2.676],
        [6.978, -3.871, -0.31],
        [6.978, -3.871, 0.383],
        [7.02, -3.871, 0.354],
        [9.255, -3.871, -1.185],
        [10.789, -3.871, -2.242],
        [10.462, -3.871, -2.717],
        [9.28, -3.871, -4.434],
        [9.255, -3.871, -4.47],
        [9.035, -3.871, -4.79],
        [8.716, -3.871, -5.253],
        [8.227, -3.871, -5.963],
        [8.117, -3.871, -6.123],
        [7.605, -3.871, -6.866],
        [7.33, -3.871, -7.265],
        [6.978, -3.871, -7.022],
    ]
    top_poly = [[point[0], -1.608, point[2]] for point in footprint]
    bldg = {
        "rooms": [
            {
                "story": 1,
                "floor_polygon": footprint,
                "walls_computed": [
                    {
                        "corners": [
                            [6.978, -3.871, -6.866],
                            [10.789, -3.871, -2.242],
                            [10.789, -1.608, -2.242],
                            [6.978, -1.608, -6.866],
                        ]
                    }
                ],
                "walls_merged": [],
            }
        ]
    }
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "room_outline": footprint,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:collinear",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "poly": top_poly,
                }
            ],
        }
    ]

    result = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=room_partitions,
        building_part_graph={},
    )

    assert result["metadata"]["synthetic_atom_cell_count"] == 0
    assert result["metadata"]["cell_count"] >= 1


def test_candidate_footprints_adds_stable_convex_hull_retry_for_jagged_convex_ring():
    raw = Polygon(
        [
            (0.954, -3.04),
            (0.222, 0.807),
            (0.22221, 0.80704),
            (0.207, 0.887),
            (0.558, 0.954),
            (2.866, 1.392945),
            (2.866, -2.676),
            (1.306, -2.973),
        ]
    )

    candidates = _candidate_footprints(raw)

    assert len(candidates) >= 2
    assert len(candidates[0]) >= 3
    assert len(candidates[1]) < len(candidates[0])
    assert Polygon(candidates[1]).is_valid


def test_occupied_cell_retries_with_convex_hull() -> None:
    footprint = [
        [0.954, 3.604, -3.04],
        [0.222, 3.604, 0.807],
        [0.22221, 3.604, 0.80704],
        [0.207, 3.604, 0.887],
        [0.558, 3.604, 0.954],
        [2.866, 3.604, 1.392945],
        [2.866, 3.604, -2.676],
        [1.306, 3.604, -2.973],
    ]
    top_poly = [[point[0], 5.891, point[2]] for point in footprint]
    bldg = {
        "rooms": [
            {
                "story": 2,
                "floor_polygon": footprint,
                "walls_computed": [
                    {
                        "corners": [
                            [0.207, 3.604, -3.04],
                            [2.866, 3.604, -2.676],
                            [2.866, 5.891, -2.676],
                            [0.207, 5.891, -3.04],
                        ]
                    }
                ],
                "walls_merged": [],
            }
        ]
    }
    room_partitions = [
        {
            "room_index": 0,
            "story": 2,
            "room_outline": footprint,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:jagged",
                    "room_index": 0,
                    "story": 2,
                    "kind": "flat",
                    "roof_hypothesis_id": "roof-hypothesis:flat:test",
                    "poly": top_poly,
                    "top_y_m": 5.891,
                }
            ],
        }
    ]

    result = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=room_partitions,
        building_part_graph={},
    )

    assert result["metadata"]["synthetic_atom_cell_count"] == 0
    assert result["metadata"]["cell_count"] >= 1


def test_build_best_candidate_cell_prefers_oblique_candidate_with_top_face() -> None:
    candidate_footprints = [
        [
            (-7.892, 6.525),
            (-0.214, 5.399),
            (-0.495, 3.483),
            (-0.613, 2.68),
            (-0.616, 2.656),
            (-0.619, 2.636),
            (-8.297, 3.761),
        ],
        [
            (-0.619, 2.636),
            (-8.297, 3.761),
            (-7.892, 6.525),
            (-0.214, 5.399),
            (-0.616, 2.656),
        ],
    ]
    base_y = -1.061

    def top_y_at(x: float, z: float) -> float:
        return round(
            0.28265706156142384 * x - 0.03964282824712852 * z + 3.6947670734681176, 6
        )

    best = _build_best_candidate_cell(
        candidate_footprints=candidate_footprints,
        build_cell=lambda footprint: build_arranged_polyhedral_cell(
            cell_id=f"cell:{len(footprint)}",
            room_id="room:0",
            room_index=0,
            part_id="part:0",
            story=1,
            base_atom_id="atom:0",
            cell_kind="occupied_room",
            region_footprint=footprint,
            base_y=base_y,
            top_y_at=top_y_at,
            top_surface_kind="oblique",
            roof_hypothesis_id="roof-hypothesis:oblique:test",
            perimeter_side_face_indices=set(),
        ),
    )

    assert best is not None
    top_faces = [
        face for face in (best.get("faces") or []) if face.get("kind") == "top"
    ]
    assert len(top_faces) == 1
    assert top_faces[0]["role"] == "roof"
    assert float(top_faces[0]["area_m2"] or 0.0) > 1.0


def test_build_arranged_polyhedral_cell_clips_oblique_footprint_to_positive_clearance():
    base_y = -1.171
    footprint = [
        (0.087, -2.551),
        (-2.61, -0.08),
        (0.399, -0.419),
    ]

    def top_y_at(x: float, z: float) -> float:
        clearance = 0.16065712 * x + 1.09562616 * z + 2.67196517
        return base_y + clearance

    cell = build_arranged_polyhedral_cell(
        cell_id="cell:mixed-clearance",
        room_id="room:5",
        room_index=5,
        part_id=None,
        story=0,
        base_atom_id="atom:oblique",
        cell_kind="occupied_room",
        region_footprint=footprint,
        base_y=base_y,
        top_y_at=top_y_at,
        top_surface_kind="oblique",
        roof_hypothesis_id="roof-hypothesis:oblique:test",
        perimeter_side_face_indices={0, 1, 2},
    )

    assert cell is not None
    top_faces = [
        face for face in (cell.get("faces") or []) if face.get("kind") == "top"
    ]
    assert len(top_faces) == 1
    assert float(top_faces[0]["area_m2"] or 0.0) > 0.01
    assert all(
        float(corner[1]) > base_y for corner in (top_faces[0].get("corners") or [])
    )


def test_arranged_cell_preserves_positive_top() -> None:
    base_y = -1.061
    footprint = [
        (-1.029, -0.161),
        (-8.707, 0.965),
        (-8.297, 3.761),
        (-0.619, 2.636),
    ]

    def top_y_at(x: float, z: float) -> float:
        return 0.16053637344803862 * x + 1.095557970989379 * z + 1.5005767616073216

    cell = build_arranged_polyhedral_cell(
        cell_id="cell:oblique-quad",
        room_id="room:6",
        room_index=6,
        part_id=None,
        story=0,
        base_atom_id="atom:oblique:quad",
        cell_kind="occupied_room",
        region_footprint=footprint,
        base_y=base_y,
        top_y_at=top_y_at,
        top_surface_kind="oblique",
        roof_hypothesis_id="roof-hypothesis:oblique:test",
        perimeter_side_face_indices={0, 1, 2, 3},
    )

    assert cell is not None
    top_faces = [
        face for face in (cell.get("faces") or []) if face.get("kind") == "top"
    ]
    assert len(top_faces) == 1
    assert len(top_faces[0].get("corners") or []) == 4


def test_merged_partition_falls_back_to_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room_partition = {
        "room_index": 0,
        "story": 0,
        "partitions": [
            {
                "id": "atom:a",
                "room_index": 0,
                "story": 0,
                "kind": "flat",
                "roof_hypothesis_id": "roof-hypothesis:flat:test",
                "flat_role": "roof_flat",
                "top_y_m": 2.0,
                "poly": _rect(0.0, 0.0, 2.0, 1.0, 2.0),
            },
            {
                "id": "atom:b",
                "room_index": 0,
                "story": 0,
                "kind": "flat",
                "roof_hypothesis_id": "roof-hypothesis:flat:test",
                "flat_role": "roof_flat",
                "top_y_m": 2.0,
                "poly": _rect(2.0, 0.0, 4.0, 1.0, 2.0),
            },
        ],
    }

    # Simulate a robustness failure where the group union only returns one side.
    monkeypatch.setattr(
        "reconcile.roof_algorithms_py.occupied_room_cell_complex.unary_union",
        lambda polys: polys[0],
    )

    merged = _merged_partition_regions(room_partition)

    assert len(merged) == 2
    assert sorted(region["atom_ids"] for region in merged) == [["atom:a"], ["atom:b"]]


def test_annot_marks_story_boundary_exterior() -> None:
    cell = {
        "faces": [
            {
                "kind": "side",
                "role": "splitter",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 2.4, 0.0],
                    [0.0, 2.4, 0.0],
                ],
            }
        ]
    }
    room_poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)])
    story_union = room_poly

    _annotate_boundary_classes(cell, room_poly, story_union)

    assert cell["faces"][0]["metadata"]["boundary_class"] == "exterior_wall"


def test_annot_demotes_perimeter_to_splitter() -> None:
    cell = {
        "faces": [
            {
                "kind": "side",
                "role": "wall",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [0.0, 2.4, 0.0],
                    [1.0, 2.4, 1.0],
                    [1.0, 0.0, 1.0],
                ],
                "metadata": {"perimeter_facing": True},
            }
        ]
    }
    room_poly = Polygon([(3.0, 0.0), (5.0, 0.0), (5.0, 2.0), (3.0, 2.0)])
    story_union = room_poly

    _annotate_boundary_classes(cell, room_poly, story_union)

    assert cell["faces"][0]["metadata"]["boundary_class"] == "splitter"


def test_roof_cell_complex_builds_exact_attic_cell_from_flat_atom_and_oblique_roof():
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
                }
            ],
        }
    ]
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "roof_family_guess": "gable_or_multi_slope",
            }
        ],
        "room_membership": {"room:0": ["part:0"]},
        "hypothesis_membership": {"roof-hypothesis:oblique:0": ["part:0"]},
    }
    roof_coverage_graph = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "corners": [
                    [0.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 4.0, 4.0],
                    [0.0, 4.0, 4.0],
                ],
            }
        ],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    result = build_roof_cell_complex(
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "kind": "oblique",
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "corners": [
                    [0.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 4.0, 4.0],
                    [0.0, 4.0, 4.0],
                ],
                "cluster": {"avgAzimuth": 0.0, "avgIncl": 20.0},
                "center": {"x": 2.0, "y": 3.5, "z": 2.0},
            }
        ],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
    )

    assert result["metadata"]["exact_on_lattice"] is True
    assert result["metadata"]["attic_cell_count"] == 1
    assert result["metadata"]["knee_wall_count"] >= 1
    assert (
        result["metadata"]["backend"] == "exact_lattice_roof_wall_slab_arrangement_v2"
    )
    cell = result["cells"][0]
    assert cell["cell_kind"] == "attic"
    assert cell["volume_m3"] > 0.0
    assert cell["arrangement"]["plane_count"] >= 4
    face_roles = {face["role"] for face in cell["faces"]}
    assert "roof" in face_roles
    assert "slab" in face_roles
    assert "wall" in face_roles
    assert all("corners_lattice" in face for face in cell["faces"])


def test_roof_cell_complex_builds_upper_void_for_flat_transition_in_mixed_room() -> (
    None
):
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": True,
            "partitions": [
                {
                    "id": "atom:oblique:0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "oblique",
                    "poly": [
                        [0.0, 2.8, 0.0],
                        [2.0, 2.8, 0.0],
                        [2.0, 3.4, 4.0],
                        [0.0, 3.4, 4.0],
                    ],
                },
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ceiling_cap",
                    "poly": _rect(2.0, 0.0, 4.0, 4.0, 2.6),
                },
            ],
        }
    ]
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "roof_family_guess": "gable_or_multi_slope",
            }
        ],
        "room_membership": {"room:0": ["part:0"]},
        "hypothesis_membership": {"roof-hypothesis:oblique:0": ["part:0"]},
    }
    roof_coverage_graph = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "corners": [
                    [0.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 4.2, 4.0],
                    [0.0, 4.2, 4.0],
                ],
            }
        ],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    result = build_roof_cell_complex(
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "kind": "oblique",
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "corners": [
                    [0.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 4.2, 4.0],
                    [0.0, 4.2, 4.0],
                ],
                "cluster": {"avgAzimuth": 0.0, "avgIncl": 20.0},
                "center": {"x": 2.0, "y": 3.6, "z": 2.0},
            }
        ],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
    )

    assert result["metadata"]["upper_void_cell_count"] == 1
    assert result["cells"][0]["cell_kind"] == "upper_void"
    assert any(face["role"] == "roof" for face in result["cells"][0]["faces"])


def test_roof_cell_complex_keeps_only_perimeter_facing_knee_walls() -> None:
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
                }
            ],
        }
    ]
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "roof_family_guess": "gable_or_multi_slope",
            }
        ],
        "room_membership": {"room:0": ["part:0"]},
        "hypothesis_membership": {"roof-hypothesis:oblique:0": ["part:0"]},
    }
    selected_oblique_surfaces = [
        {
            "kind": "oblique",
            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
            "corners": [
                [0.0, 4.0, 0.0],
                [4.0, 4.0, 0.0],
                [0.0, 5.2, 4.0],
            ],
            "cluster": {"avgAzimuth": 0.0, "avgIncl": 20.0},
            "center": {"x": 1.333333, "y": 4.4, "z": 1.333333},
        }
    ]
    roof_coverage_graph = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    result = build_roof_cell_complex(
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
    )

    assert result["metadata"]["cell_count"] == 1
    assert result["metadata"]["knee_wall_count"] == 2


def test_roof_cell_complex_filters_knee_walls_without_occupied_shell_support() -> None:
    roof_cell_complex = {
        "cells": [],
        "knee_walls": [
            {
                "id": "knee:keep",
                "room_index": 0,
                "base_edge_m": 2.0,
                "corners": [
                    [0.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                ],
            },
            {
                "id": "knee:drop",
                "room_index": 0,
                "base_edge_m": 2.0,
                "corners": [
                    [0.0, 1.0, 3.0],
                    [2.0, 1.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [0.0, 2.0, 3.0],
                ],
            },
        ],
        "metadata": {"knee_wall_count": 2},
    }
    occupied_room_cell_complex = {
        "cells": [
            {
                "id": "occ:0",
                "room_index": 0,
                "faces": [
                    {
                        "id": "face:wall:0",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        "metadata": {"boundary_class": "exterior_wall"},
                    }
                ],
            }
        ]
    }

    filtered = filter_knee_walls_by_occupied_shell(
        roof_cell_complex=roof_cell_complex,
        occupied_room_cell_complex=occupied_room_cell_complex,
    )

    assert [wall["id"] for wall in filtered["knee_walls"]] == ["knee:keep"]
    assert filtered["metadata"]["knee_wall_count"] == 1
    assert filtered["metadata"]["knee_wall_dropped_by_occupied_shell"] == 1
    assert filtered["knee_walls"][0]["occupied_shell_support"][
        "supported_length_m"
    ] == pytest.approx(2.0, abs=1e-6)


def test_roof_cell_complex_splits_non_convex_region_into_multiple_polyhedral_cells():
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": [
                        [0.0, 2.4, 0.0],
                        [4.0, 2.4, 0.0],
                        [4.0, 2.4, 1.0],
                        [1.0, 2.4, 1.0],
                        [1.0, 2.4, 4.0],
                        [0.0, 2.4, 4.0],
                    ],
                }
            ],
        }
    ]
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "roof_family_guess": "gable_or_multi_slope",
            }
        ],
        "room_membership": {"room:0": ["part:0"]},
        "hypothesis_membership": {"roof-hypothesis:oblique:0": ["part:0"]},
    }
    selected_oblique_surfaces = [
        {
            "kind": "oblique",
            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
            "corners": [
                [0.0, 3.0, 0.0],
                [4.0, 3.0, 0.0],
                [4.0, 4.0, 4.0],
                [0.0, 4.0, 4.0],
            ],
            "cluster": {"avgAzimuth": 0.0, "avgIncl": 20.0},
            "center": {"x": 2.0, "y": 3.5, "z": 2.0},
        }
    ]
    roof_coverage_graph = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    result = build_roof_cell_complex(
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
    )

    assert result["metadata"]["cell_count"] >= 2
    assert all(cell["volume_m3"] > 0.0 for cell in result["cells"])
    assert all(cell["arrangement"]["vertex_count"] >= 4 for cell in result["cells"])


def test_occupied_cell_splits_room_shell_by_exact_top_boundary_atoms() -> None:
    building = {
        "rooms": [
            {
                "story": 1,
                "floor_polygon": _rect(0.0, 0.0, 4.0, 4.0, 0.0),
                "walls_computed": [
                    {
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [4.0, 3.0, 0.0],
                            [0.0, 3.0, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, 0.0, 0.0],
                            [4.0, 0.0, 4.0],
                            [4.0, 3.0, 4.0],
                            [4.0, 3.0, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [4.0, 0.0, 4.0],
                            [0.0, 0.0, 4.0],
                            [0.0, 3.0, 4.0],
                            [4.0, 3.0, 4.0],
                        ]
                    },
                    {
                        "corners": [
                            [0.0, 0.0, 4.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ]
                    },
                ],
            }
        ]
    }
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": True,
            "partitions": [
                {
                    "id": "atom:left",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "poly": _rect(0.0, 0.0, 2.0, 4.0, 2.2),
                },
                {
                    "id": "atom:right",
                    "room_index": 0,
                    "story": 1,
                    "kind": "oblique",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                    "poly": [
                        [2.0, 2.2, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 4.0],
                        [2.0, 2.2, 4.0],
                    ],
                },
            ],
        }
    ]
    building_part_graph = {
        "room_membership": {"room:0": ["part:0"]},
    }

    result = build_occupied_room_cell_complex(
        bldg=building,
        room_partitions=room_partitions,
        building_part_graph=building_part_graph,
    )

    assert result["metadata"]["cell_count"] == 2
    assert result["metadata"]["fallback_cell_count"] == 0
    assert result["metadata"]["atom_bound_cell_count"] == 2
    assert result["metadata"]["face_class_counts"]["exterior_wall"] >= 4
    assert all(
        cell["exact_source_kind"] == "top_boundary_atom" for cell in result["cells"]
    )
    assert {cell["top_boundary_atom_id"] for cell in result["cells"]} == {
        "atom:left",
        "atom:right",
    }
    assert any(
        cell["roof_hypothesis_id"] == "roof-hypothesis:oblique:0"
        for cell in result["cells"]
    )
    assert all(
        any(
            (face.get("metadata") or {}).get("boundary_class") == "ceiling"
            for face in (cell.get("faces") or [])
        )
        for cell in result["cells"]
    )
