from __future__ import annotations

from reconcile.roof_algorithms_py.roof_coverage_graph import build_roof_coverage_graph
from reconcile.roof_algorithms_py.roof_envelope_continuation import (
    continue_roof_envelopes,
)
from reconcile.roof_algorithms_py.roof_partitioning import (
    derive_room_ceiling_partitions,
)
from reconcile.roof_algorithms_py.top_boundary_graph import build_top_boundary_graph


def _rect(x0: float, z0: float, x1: float, z1: float, y: float) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def test_envcont_promotes_flat_to_void() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 1,
            "fp": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
            "wallTopY": 3.8,
            "wallTopMin": 2.4,
        },
        {
            "room_index": 1,
            "story": 1,
            "fp": _rect(4.05, 0.0, 8.05, 4.0, 2.4),
            "wallTopY": 3.8,
            "wallTopMin": 2.4,
        },
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "support_score": 0.72,
                "continuation_component_size": 2,
                "selected": True,
            }
        ],
        "edges": [
            {
                "id": "edge:covers:0",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:0",
                "to": "room:0",
                "selected": True,
                "evidence": {
                    "coverage_ratio": 1.0,
                    "coverage_area_m2": 16.0,
                    "edge_score": 0.85,
                },
            },
            {
                "id": "edge:continues:0",
                "type": "CONTINUES_AS",
                "from": "roof-hypothesis:oblique:0",
                "to": "roof-hypothesis:oblique:peer",
                "evidence": {
                    "exact_face_incidence": True,
                    "partition_atom_pairs": [["atom:room0", "atom:peer"]],
                },
            },
        ],
        "selected_room_assignments": {
            "room:0": ["roof-hypothesis:oblique:0"],
            "room:1": [],
        },
        "metadata": {},
    }
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "roof_family_guess": "gable_or_multi_slope",
            }
        ],
        "room_membership": {
            "room:0": ["part:0"],
            "room:1": ["part:0"],
        },
        "hypothesis_membership": {
            "roof-hypothesis:oblique:0": ["part:0"],
        },
    }
    selected_oblique_surfaces = [
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
            "story": 1,
            "dominant_story": 1,
        }
    ]
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:room0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
                }
            ],
        },
        {
            "room_index": 1,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:room1",
                    "room_index": 1,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": _rect(4.05, 0.0, 8.05, 4.0, 2.4),
                }
            ],
        },
    ]
    initial_coverage = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )

    result = continue_roof_envelopes(
        exposed_rooms=exposed_rooms,
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        hypothesis_graph=hypothesis_graph,
        building_part_graph=building_part_graph,
        roof_coverage_graph=initial_coverage,
    )

    assert result["metadata"]["continued_room_count"] == 1
    assert result["metadata"]["continuation_region_count"] == 1
    assert result["continued_room_indices"] == [1]
    assert result["hypothesis_graph"]["selected_room_assignments"]["room:1"] == [
        "roof-hypothesis:oblique:0"
    ]
    assert result["continuation_regions"][0]["continuation_mode"] == "arrangement_face"
    assert result["continuation_regions"][0]["exact_incidence_pair_count"] == 1
    assert result["continued_surfaces"][0]["continuation_source"] == "arrangement_face"

    final_partitions = derive_room_ceiling_partitions(
        room_records=exposed_rooms,
        oblique_roof_surfaces=result["selected_oblique_surfaces"],
        flat_roof_surfaces=[],
        hypothesis_graph=result["hypothesis_graph"],
    )
    room1 = next(
        room for room in final_partitions["room_partitions"] if room["room_index"] == 1
    )
    assert any(partition["kind"] == "oblique" for partition in room1["partitions"])

    final_coverage = build_roof_coverage_graph(
        room_partitions=final_partitions["room_partitions"],
        selected_oblique_surfaces=result["selected_oblique_surfaces"],
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )
    top_boundary_graph = build_top_boundary_graph(
        room_records=exposed_rooms,
        room_partitions=final_partitions["room_partitions"],
        building_part_graph=building_part_graph,
        roof_coverage_graph=final_coverage,
    )
    summary = top_boundary_graph["room_summaries"]["room:1"]
    assert summary["has_oblique_atom"] is True
    assert summary["has_candidate_attic_relation"] is False


def test_envcont_uses_subpart_semantics() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 1,
            "fp": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
            "wallTopY": 3.8,
            "wallTopMin": 2.4,
        },
        {
            "room_index": 1,
            "story": 1,
            "fp": _rect(4.05, 0.0, 8.05, 4.0, 2.4),
            "wallTopY": 3.8,
            "wallTopMin": 2.4,
        },
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "support_score": 0.72,
                "continuation_component_size": 2,
                "selected": True,
            }
        ],
        "edges": [
            {
                "id": "edge:covers:0",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:0",
                "to": "room:0",
                "selected": True,
                "evidence": {
                    "coverage_ratio": 1.0,
                    "coverage_area_m2": 16.0,
                    "edge_score": 0.85,
                },
            }
        ],
        "selected_room_assignments": {
            "room:0": ["roof-hypothesis:oblique:0"],
            "room:1": [],
        },
        "metadata": {},
    }
    building_part_graph = {
        "nodes": [
            {
                "id": "part:0",
                "type": "BuildingPart",
                "roof_family_guess": "mixed_or_partial",
            }
        ],
        "room_membership": {
            "room:0": ["part:0"],
            "room:1": ["part:0"],
        },
        "hypothesis_membership": {
            "roof-hypothesis:oblique:0": ["part:0"],
        },
    }
    selected_oblique_surfaces = [
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
            "story": 1,
            "dominant_story": 1,
        }
    ]
    room_partitions = [
        {
            "room_index": 0,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:room0",
                    "room_index": 0,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": _rect(0.0, 0.0, 4.0, 4.0, 2.4),
                }
            ],
        },
        {
            "room_index": 1,
            "story": 1,
            "mixed": False,
            "partitions": [
                {
                    "id": "atom:room1",
                    "room_index": 1,
                    "story": 1,
                    "kind": "flat",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                    "poly": _rect(4.05, 0.0, 8.05, 4.0, 2.4),
                }
            ],
        },
    ]
    initial_coverage = build_roof_coverage_graph(
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        selected_flat_surfaces=[],
        building_part_graph=building_part_graph,
    )
    initial_coverage["subparts"][0]["semantic_kind"] = "gable_run"

    result = continue_roof_envelopes(
        exposed_rooms=exposed_rooms,
        room_partitions=room_partitions,
        selected_oblique_surfaces=selected_oblique_surfaces,
        hypothesis_graph=hypothesis_graph,
        building_part_graph=building_part_graph,
        roof_coverage_graph=initial_coverage,
    )

    assert result["metadata"]["continued_room_count"] == 1
    assert result["continued_room_indices"] == [1]
