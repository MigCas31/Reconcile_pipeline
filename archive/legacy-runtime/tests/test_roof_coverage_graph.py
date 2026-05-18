from __future__ import annotations

from reconcile.roof_algorithms_py.roof_coverage_graph import build_roof_coverage_graph


def _rect(x0: float, z0: float, x1: float, z1: float, y: float) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def test_coverage_marks_flat_confirmed_sloped() -> None:
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
    graph = build_roof_coverage_graph(
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

    coverage = graph["atom_coverage"]["atom:flat:0"]
    assert coverage["sloped_state"] == "confirmed"
    assert coverage["sloped_overlap_ratio"] > 0.9
    assert coverage["sloped_vertical_clearance_m"] > 0.12


def test_coverage_marks_overlap_partial() -> None:
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
    graph = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "corners": [
                    [0.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.1, 4.0],
                    [0.0, 2.1, 4.0],
                ],
            }
        ],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    coverage = graph["atom_coverage"]["atom:flat:0"]
    assert coverage["sloped_state"] == "partial"
    assert coverage["sloped_overlap_ratio"] > 0.9
    assert coverage["sloped_vertical_clearance_m"] < 0.0


def test_coverage_seeds_subpart_sibling() -> None:
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
        "hypothesis_membership": {
            "roof-hypothesis:oblique:0": ["part:0"],
            "roof-hypothesis:oblique:1": ["part:0"],
        },
    }
    graph = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "corners": [
                    [0.0, 2.6, 0.0],
                    [4.0, 2.6, 0.0],
                    [4.0, 3.0, 4.0],
                    [0.0, 3.0, 4.0],
                ],
                "cluster": {"segs": [{"room_idx": 0}]},
            },
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                "corners": [
                    [0.0, 3.2, 0.0],
                    [4.0, 3.2, 0.0],
                    [4.0, 3.8, 4.0],
                    [0.0, 3.8, 4.0],
                ],
                "cluster": {"segs": [{"room_idx": 0}]},
            },
        ],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    subpart_hypotheses = {
        subpart["roof_hypothesis_id"] for subpart in graph["subparts"]
    }
    assert "roof-hypothesis:oblique:1" in subpart_hypotheses
    assert "roof-hypothesis:oblique:0" in subpart_hypotheses
    assert graph["metadata"]["seeded_hypothesis_count"] == 1
