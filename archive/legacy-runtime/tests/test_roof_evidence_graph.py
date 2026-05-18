from __future__ import annotations

from reconcile.roof_algorithms_py.roof_building_parts import refine_building_part_graph
from reconcile.roof_algorithms_py.roof_evidence_graph import (
    annotate_roof_coverage_graph,
    build_roof_evidence_graph,
)


def _rect(
    x0: float, z0: float, x1: float, z1: float, y: float = 0.0
) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def test_roof_evidence_graph_promotes_part_family_from_room_structure() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
        }
    ]
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "partitions": [
                {
                    "id": "atom:flat:0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "poly": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ],
        }
    ]
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "room_ids": ["room:0"],
                "roof_family_guess": "flat_or_capped",
            }
        ],
        "edges": [],
        "room_membership": {"room:0": ["part:0"]},
        "hypothesis_membership": {"roof-hypothesis:oblique:0": ["part:0"]},
        "metadata": {},
    }
    roof_coverage_graph = {
        "atom_coverage": {
            "atom:flat:0": {
                "sloped_state": "partial",
                "sloped_hypothesis_id": "roof-hypothesis:oblique:0",
            }
        },
        "atom_subpart_membership": {"atom:flat:0": ["subpart:0"]},
        "room_subpart_membership": {"room:0": ["subpart:0"]},
        "subparts": [
            {
                "id": "subpart:0",
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "room_indices": [0],
                "part_ids": ["part:0"],
                "semantic_kind": "gable_run",
            }
        ],
    }
    roof_cell_complex = {"cells": [], "knee_walls": []}

    evidence = build_roof_evidence_graph(
        exposed_rooms=exposed_rooms,
        room_partitions=room_partitions,
        selected_oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "cluster": {"avgAzimuth": 0.0},
            },
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                "cluster": {"avgAzimuth": 180.0},
            },
        ],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
        roof_cell_complex=roof_cell_complex,
    )
    refined = refine_building_part_graph(
        building_part_graph=building_part_graph,
        roof_evidence_graph=evidence,
    )
    annotated_coverage = annotate_roof_coverage_graph(
        roof_coverage_graph=roof_coverage_graph,
        roof_evidence_graph=evidence,
    )

    room_evidence = evidence["room_evidence"]["room:0"]
    refined_part = refined["nodes"][0]
    annotated_subpart = annotated_coverage["subparts"][0]

    assert room_evidence["strong_gable_context"] is True
    assert room_evidence["strong_attic_context"] is True
    assert refined_part["roof_family_guess"] == "gable_or_multi_slope"
    assert refined_part["roof_family_guess_initial"] == "flat_or_capped"
    assert annotated_subpart["support_evidence_score"] >= 3
