from __future__ import annotations

from reconcile.roof_algorithms_py.roof_flat_oblique_clipping import (
    clip_flat_surfaces_against_obliques,
)
from reconcile.roof_algorithms_py.roof_hypothesis_graph import (
    build_roof_hypothesis_graph,
    select_roof_surfaces_from_hypotheses,
)
from reconcile.roof_algorithms_py.roof_partitioning import (
    derive_room_ceiling_partitions,
    inject_simple_slant_partitions,
)
from reconcile_v2.models import GraphEdge, GraphNode, TopologyGraph


def _rect(
    x0: float, z0: float, x1: float, z1: float, y: float = 0.0
) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def _graph_for_rooms() -> TopologyGraph:
    return TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(
                id="room:r0",
                type="Room",
                story=0,
                bbox_xz=[0.0, 0.0, 4.0, 4.0],
                properties={"is_top_story": True},
            ),
            GraphNode(
                id="room:r1",
                type="Room",
                story=0,
                bbox_xz=[4.0, 0.0, 8.0, 4.0],
                properties={"is_top_story": True},
            ),
            GraphNode(
                id="cell:outside", type="Cell", properties={"cell_kind": "outside"}
            ),
        ],
        edges=[
            GraphEdge(
                id="exposes:r0",
                type="EXPOSES_TO",
                from_id="room:r0",
                to_id="cell:outside",
            ),
            GraphEdge(
                id="exposes:r1",
                type="EXPOSES_TO",
                from_id="room:r1",
                to_id="cell:outside",
            ),
        ],
    )


def test_hypothesis_graph_selects_meaningful_partial_flat_and_oblique_candidates() -> (
    None
):
    bldg = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": _rect(0.0, 0.0, 4.0, 4.0),
            }
        ]
    }
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.75,
        }
    ]
    oblique_surfaces = [
        {
            "dominant_story": 0,
            "corners": [
                [0.0, 2.5, 0.0],
                [4.0, 3.5, 0.0],
                [4.0, 3.5, 4.0],
                [0.0, 2.5, 4.0],
            ],
            "center": {"x": 2.0, "y": 3.0, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 15.0,
                "room_indices": [0],
                "segs": [{}, {}, {}, {}],
            },
            "space_boundary_ids": ["boundary:roof:o0"],
        }
    ]
    flat_surfaces = [
        {
            "kind": "intermediate",
            "story": 0,
            "room_index": 0,
            "y": 3.0,
            "corners": _rect(2.0, 0.0, 4.0, 4.0, y=3.0),
            "space_boundary_ids": ["boundary:roof:f0"],
        }
    ]

    hypothesis_graph = build_roof_hypothesis_graph(
        bldg=bldg,
        exposed_rooms=exposed_rooms,
        oblique_surfaces=oblique_surfaces,
        flat_surfaces=flat_surfaces,
        roof_graph={"nodes": [], "edges": [], "metadata": {}},
        graph=_graph_for_rooms(),
    )

    assert hypothesis_graph["selected_room_assignments"]["room:0"] == [
        "roof-hypothesis:oblique:0",
        "roof-hypothesis:flat:0",
    ]


def test_partitions_split_by_selected_hypotheses() -> None:
    bldg = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": _rect(0.0, 0.0, 4.0, 4.0),
            }
        ]
    }
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.75,
        }
    ]
    oblique_surfaces = [
        {
            "dominant_story": 0,
            "corners": [
                [0.0, 2.5, 0.0],
                [4.0, 3.5, 0.0],
                [4.0, 3.5, 4.0],
                [0.0, 2.5, 4.0],
            ],
            "center": {"x": 2.0, "y": 3.0, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 15.0,
                "room_indices": [0],
                "segs": [{}, {}, {}, {}],
            },
            "space_boundary_ids": ["boundary:roof:o0"],
        }
    ]
    flat_surfaces = [
        {
            "kind": "intermediate",
            "story": 0,
            "room_index": 0,
            "y": 3.0,
            "corners": _rect(2.0, 0.0, 4.0, 4.0, y=3.0),
            "space_boundary_ids": ["boundary:roof:f0"],
        }
    ]

    hypothesis_graph = build_roof_hypothesis_graph(
        bldg=bldg,
        exposed_rooms=exposed_rooms,
        oblique_surfaces=oblique_surfaces,
        flat_surfaces=flat_surfaces,
        roof_graph={"nodes": [], "edges": [], "metadata": {}},
        graph=_graph_for_rooms(),
    )
    selected_oblique, selected_flat = select_roof_surfaces_from_hypotheses(
        oblique_surfaces=oblique_surfaces,
        flat_surfaces=flat_surfaces,
        hypothesis_graph=hypothesis_graph,
    )
    partitions = derive_room_ceiling_partitions(
        room_records=exposed_rooms,
        oblique_roof_surfaces=selected_oblique,
        flat_roof_surfaces=selected_flat,
        hypothesis_graph=hypothesis_graph,
    )

    assert partitions["metadata"]["mixed_room_count"] == 0
    assert partitions["metadata"]["flat_partition_count"] == 0
    assert partitions["metadata"]["oblique_partition_count"] == 2
    assert partitions["metadata"]["split_line_count"] == 1
    total_area = sum(
        part["area_m2"] for part in partitions["room_partitions"][0]["partitions"]
    )
    assert round(total_area, 3) == 16.0


def test_partitions_split_by_globally_selected_competing_surface() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.5,
            "wallTopMin": 2.5,
        }
    ]
    oblique_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
            "dominant_story": 0,
            "corners": [
                [0.0, 2.5, 0.0],
                [4.0, 3.5, 0.0],
                [4.0, 3.5, 4.0],
                [0.0, 2.5, 4.0],
            ],
            "center": {"x": 2.0, "y": 3.0, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 14.0,
                "room_indices": [0],
                "segs": [{}, {}, {}, {}],
            },
        }
    ]
    flat_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:flat:0",
            "kind": "intermediate",
            "story": 0,
            "room_index": 0,
            "y": 3.0,
            "corners": _rect(0.0, 0.0, 4.0, 4.0, y=3.0),
        }
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "story": 1,
                "selected": True,
            },
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "story": 0,
                "selected": True,
            },
        ],
        "edges": [
            {
                "id": "edge:covers:flat",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:0",
                "to": "room:0",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            }
        ],
        "selected_hypothesis_ids": [
            "roof-hypothesis:oblique:0",
            "roof-hypothesis:flat:0",
        ],
        "selected_room_assignments": {"room:0": ["roof-hypothesis:flat:0"]},
    }

    partitions = derive_room_ceiling_partitions(
        room_records=exposed_rooms,
        oblique_roof_surfaces=oblique_surfaces,
        flat_roof_surfaces=flat_surfaces,
        hypothesis_graph=hypothesis_graph,
    )

    room = partitions["room_partitions"][0]
    total_area = sum(part["area_m2"] for part in room["partitions"])

    assert room["mixed"] is True
    assert any(part["kind"] == "oblique" for part in room["partitions"])
    assert any(part["kind"] == "flat" for part in room["partitions"])
    assert partitions["metadata"]["split_line_count"] >= 1
    assert round(total_area, 3) == 16.0


def test_partitions_ignore_unselected_flat_with_obliques() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.5,
            "wallTopMin": 2.5,
        }
    ]
    oblique_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
            "dominant_story": 0,
            "corners": [
                [0.0, 2.5, 0.0],
                [4.0, 3.5, 0.0],
                [4.0, 3.5, 4.0],
                [0.0, 2.5, 4.0],
            ],
            "center": {"x": 2.0, "y": 3.0, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 14.0,
                "room_indices": [0],
                "segs": [{}, {}, {}, {}],
            },
        }
    ]
    flat_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:flat:0",
            "kind": "intermediate",
            "story": 0,
            "room_index": 0,
            "y": 3.0,
            "corners": _rect(0.0, 0.0, 4.0, 4.0, y=3.0),
        }
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "story": 1,
                "selected": True,
            },
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "story": 0,
                "selected": True,
            },
        ],
        "edges": [
            {
                "id": "edge:covers:oblique",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:0",
                "to": "room:0",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            },
            {
                "id": "edge:covers:flat",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:0",
                "to": "room:0",
                "selected": False,
                "evidence": {"edge_score": 0.3},
            },
        ],
        "selected_hypothesis_ids": [
            "roof-hypothesis:oblique:0",
            "roof-hypothesis:flat:0",
        ],
        "selected_room_assignments": {"room:0": ["roof-hypothesis:oblique:0"]},
    }

    partitions = derive_room_ceiling_partitions(
        room_records=exposed_rooms,
        oblique_roof_surfaces=oblique_surfaces,
        flat_roof_surfaces=flat_surfaces,
        hypothesis_graph=hypothesis_graph,
    )

    room = partitions["room_partitions"][0]

    assert room["mixed"] is False
    assert all(part["kind"] == "oblique" for part in room["partitions"])
    assert partitions["metadata"]["flat_partition_count"] == 0


def test_partitions_fallback_to_graph_footprint_when_raw_polygon_is_invalid() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            # Degenerate raw polygon in XZ, should fail raw polygon path.
            "fp": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            "graph_fp_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            "floorY": 0.0,
            "wallTopY": 3.0,
            "wallTopMin": 2.5,
            "wallTopYs": [3.0, 3.0, 3.0, 3.0],
        }
    ]
    hypothesis_graph = {
        "nodes": [],
        "edges": [],
        "selected_hypothesis_ids": [],
        "selected_room_assignments": {},
    }

    partitions = derive_room_ceiling_partitions(
        room_records=exposed_rooms,
        oblique_roof_surfaces=[],
        flat_roof_surfaces=[],
        hypothesis_graph=hypothesis_graph,
    )

    assert [room["room_index"] for room in partitions["room_partitions"]] == [0]
    room = partitions["room_partitions"][0]
    assert room["partition_count"] == 1
    assert room["partitions"][0]["kind"] == "flat"
    assert round(room["partitions"][0]["area_m2"], 3) == 16.0
    assert room["partitions"][0]["flat_role"] == "implicit_room_shell_cap"
    assert room["partitions"][0]["top_y_m"] == 3.0


def test_partitions_reject_floor_level_flat_candidates() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "floorY": 0.0,
            "wallTopY": 3.5,
            "wallTopMin": 2.5,
            "wallTopYs": [2.5, 2.5, 3.5, 3.5],
        }
    ]
    oblique_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
            "dominant_story": 0,
            "corners": [
                [0.0, 2.5, 0.0],
                [4.0, 3.5, 0.0],
                [4.0, 3.5, 4.0],
                [0.0, 2.5, 4.0],
            ],
            "center": {"x": 2.0, "y": 3.0, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 14.0,
                "room_indices": [0],
                "segs": [{}, {}, {}, {}],
            },
        }
    ]
    flat_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:flat:0",
            "kind": "intermediate",
            "story": 0,
            "room_index": 0,
            "y": 0.0,
            "corners": _rect(0.0, 0.0, 4.0, 4.0, y=0.0),
        }
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "story": 0,
                "selected": True,
            },
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "story": 0,
                "selected": True,
            },
        ],
        "edges": [
            {
                "id": "edge:covers:flat",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:0",
                "to": "room:0",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            }
        ],
        "selected_hypothesis_ids": [
            "roof-hypothesis:oblique:0",
            "roof-hypothesis:flat:0",
        ],
        "selected_room_assignments": {
            "room:0": ["roof-hypothesis:flat:0", "roof-hypothesis:oblique:0"]
        },
    }

    partitions = derive_room_ceiling_partitions(
        room_records=exposed_rooms,
        oblique_roof_surfaces=oblique_surfaces,
        flat_roof_surfaces=flat_surfaces,
        hypothesis_graph=hypothesis_graph,
    )

    room = partitions["room_partitions"][0]
    assert all(partition["kind"] == "oblique" for partition in room["partitions"])
    assert partitions["metadata"]["rejected_flat_candidate_count"] == 1
    assert partitions["metadata"]["flat_partition_count"] == 0


def test_top_flat_is_clipped_at_oblique_intersection() -> None:
    flat = {
        "roof_hypothesis_id": "roof-hypothesis:flat:0",
        "kind": "top",
        "story": 0,
        "y": 3.0,
        "corners": _rect(0.0, 0.0, 4.0, 4.0, y=3.0),
    }
    oblique = {
        "roof_hypothesis_id": "roof-hypothesis:oblique:0",
        "dominant_story": 0,
        "corners": [
            [0.0, 2.5, 0.0],
            [4.0, 3.5, 0.0],
            [4.0, 3.5, 4.0],
            [0.0, 2.5, 4.0],
        ],
    }

    clipped = clip_flat_surfaces_against_obliques(
        flat_surfaces=[flat],
        oblique_surfaces=[oblique],
        hypothesis_graph={
            "nodes": [
                {
                    "id": "roof-hypothesis:flat:0",
                    "type": "RoofHypothesis",
                    "selected": True,
                }
            ],
            "edges": [],
            "selected_hypothesis_ids": ["roof-hypothesis:flat:0"],
            "selected_room_assignments": {},
        },
    )

    assert len(clipped) == 1
    assert clipped[0]["clipped_by_oblique_hypothesis_ids"] == [
        "roof-hypothesis:oblique:0"
    ]
    assert clipped[0]["pre_clip_area_m2"] == 16.0
    assert clipped[0]["post_clip_area_m2"] == 8.48
    assert max(corner[0] for corner in clipped[0]["corners"]) == 2.12


def test_ceiling_flat_support_keeps_lower_envelope_residual() -> None:
    clipped = clip_flat_surfaces_against_obliques(
        flat_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "kind": "intermediate",
                "story": 0,
                "room_index": 0,
                "y": 3.0,
                "corners": _rect(0.0, 0.0, 4.0, 4.0, y=3.0),
            }
        ],
        oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "dominant_story": 0,
                "corners": [
                    [0.0, 2.5, 0.0],
                    [4.0, 3.5, 0.0],
                    [4.0, 3.5, 4.0],
                    [0.0, 2.5, 4.0],
                ],
            }
        ],
        keep_side="below",
        suppress_consumed_top_flats=False,
    )

    assert len(clipped) == 1
    assert clipped[0]["post_clip_area_m2"] == 8.48
    assert min(corner[0] for corner in clipped[0]["corners"]) == 1.88


def test_top_flat_consumed_by_oblique_is_suppressed() -> None:
    graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "source_kind": "top",
                "selected": True,
            }
        ],
        "edges": [
            {
                "id": "edge:covers:flat",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:0",
                "to": "room:0",
                "selected": True,
            }
        ],
        "metadata": {"selected_hypothesis_count": 1, "selected_room_count": 1},
        "selected_hypothesis_ids": ["roof-hypothesis:flat:0"],
        "selected_room_assignments": {"room:0": ["roof-hypothesis:flat:0"]},
    }

    clipped = clip_flat_surfaces_against_obliques(
        flat_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "kind": "top",
                "story": 0,
                "y": 2.0,
                "corners": _rect(0.0, 0.0, 4.0, 4.0, y=2.0),
            }
        ],
        oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "dominant_story": 0,
                "corners": [
                    [0.0, 2.5, 0.0],
                    [4.0, 3.5, 0.0],
                    [4.0, 3.5, 4.0],
                    [0.0, 2.5, 4.0],
                ],
            }
        ],
        hypothesis_graph=graph,
    )

    assert clipped == []
    node = graph["nodes"][0]
    assert node["selected"] is False
    assert node["suppressed"] is True
    assert node["suppressed_reason"] == "top_flat_consumed_by_oblique_intersections"
    assert graph["selected_hypothesis_ids"] == []
    assert graph["selected_room_assignments"] == {}
    assert graph["edges"][0]["selected"] is False


def test_roomless_top_flat_is_suppressed_by_next_story_oblique() -> None:
    graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "source_kind": "top",
                "selected": True,
            }
        ],
        "edges": [],
        "selected_hypothesis_ids": ["roof-hypothesis:flat:0"],
        "selected_room_assignments": {},
    }

    clipped = clip_flat_surfaces_against_obliques(
        flat_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "kind": "top",
                "story": 0,
                "y": 1.0,
                "corners": _rect(0.0, 0.0, 4.0, 4.0, y=1.0),
            }
        ],
        oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "dominant_story": 1,
                "corners": [
                    [0.0, 2.5, 0.0],
                    [4.0, 3.5, 0.0],
                    [4.0, 3.5, 4.0],
                    [0.0, 2.5, 4.0],
                ],
            }
        ],
        hypothesis_graph=graph,
    )

    assert clipped == []
    assert graph["nodes"][0]["suppressed"] is True
    assert graph["nodes"][0]["suppressed_reason"] == (
        "top_flat_consumed_by_oblique_intersections"
    )


def test_room_specific_intermediate_flat_keeps_meaningful_residual() -> None:
    clipped = clip_flat_surfaces_against_obliques(
        flat_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "kind": "intermediate",
                "story": 0,
                "room_index": 0,
                "y": 3.0,
                "corners": _rect(0.0, 0.0, 4.0, 4.0, y=3.0),
            }
        ],
        oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "dominant_story": 0,
                "corners": [
                    [0.0, 2.5, 0.0],
                    [4.0, 3.5, 0.0],
                    [4.0, 3.5, 4.0],
                    [0.0, 2.5, 4.0],
                ],
            }
        ],
    )

    assert len(clipped) == 1
    assert clipped[0]["kind"] == "intermediate"
    assert clipped[0]["room_index"] == 0
    assert clipped[0]["post_clip_area_m2"] == 8.48


def test_non_overlapping_top_flat_remains_available() -> None:
    clipped = clip_flat_surfaces_against_obliques(
        flat_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "kind": "top",
                "story": 0,
                "y": 3.0,
                "corners": _rect(10.0, 10.0, 14.0, 14.0, y=3.0),
            }
        ],
        oblique_surfaces=[
            {
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "dominant_story": 0,
                "corners": [
                    [0.0, 2.5, 0.0],
                    [4.0, 3.5, 0.0],
                    [4.0, 3.5, 4.0],
                    [0.0, 2.5, 4.0],
                ],
            }
        ],
    )

    assert len(clipped) == 1
    assert "clipped_by_oblique_hypothesis_ids" not in clipped[0]
    assert clipped[0]["post_clip_area_m2"] == 16.0


def test_partitions_create_exact_flat_atom_for_non_exposed_room() -> None:
    room_records = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "floorY": 0.0,
            "wallTopY": 2.8,
            "wallTopMin": 2.8,
            "wallTopYs": [2.8, 2.8, 2.8, 2.8],
            "is_roof_candidate": False,
            "top_boundary_mode": "ceiling_below_occupied_volume",
            "top_boundary_reason": "occupied_cell_above",
        }
    ]
    partitions = derive_room_ceiling_partitions(
        room_records=room_records,
        oblique_roof_surfaces=[],
        flat_roof_surfaces=[],
        hypothesis_graph={
            "nodes": [],
            "edges": [],
            "selected_hypothesis_ids": [],
            "selected_room_assignments": {},
        },
    )

    room = partitions["room_partitions"][0]
    atom = room["partitions"][0]
    assert room["partition_count"] == 1
    assert atom["kind"] == "flat"
    assert atom["flat_role"] == "implicit_room_shell_cap"
    assert atom["flat_role_reason"] == "explicit_upper_occupancy_shell_cap"
    assert atom["top_boundary_mode"] == "ceiling_below_occupied_volume"
    assert atom["top_y_m"] == 2.8


def test_partitions_recover_invalid_room_polygon_from_geometry_collection_make_valid():
    room_records = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": [
                [4.789, 0.0, 0.453],
                [-5.341, 0.0, 7.447],
                [-5.344, 0.0, 7.450],
                [3.912, 0.0, -0.128],
                [-2.215, 0.0, -4.004],
                [4.789, 0.0, 0.453],
            ],
            "floorY": 0.0,
            "wallTopY": 2.8,
            "wallTopMin": 2.4,
            "wallTopYs": [2.4, 2.8, 2.8],
            "top_boundary_mode": "roof_candidate",
        }
    ]
    partitions = derive_room_ceiling_partitions(
        room_records=room_records,
        oblique_roof_surfaces=[],
        flat_roof_surfaces=[],
        hypothesis_graph={
            "nodes": [],
            "edges": [],
            "selected_hypothesis_ids": [],
            "selected_room_assignments": {},
        },
    )

    assert [room["room_index"] for room in partitions["room_partitions"]] == [0]
    assert partitions["room_partitions"][0]["partition_count"] == 1


def test_inject_slant_for_excluded_room() -> None:
    partitions = {
        "room_partitions": [],
        "flat": [],
        "oblique": [],
        "metadata": {
            "room_partition_count": 0,
            "flat_partition_count": 0,
            "oblique_partition_count": 0,
            "mixed_room_count": 0,
        },
    }
    room_records = [
        {
            "room_index": 6,
            "story": 1,
            "graph_room_id": "room:r6",
            "top_boundary_mode": "roof_candidate",
            "top_boundary_reason": "simple_slant",
        }
    ]
    updated = inject_simple_slant_partitions(
        partitions=partitions,
        simple_slant_ceilings=[
            {
                "kind": "simple_slant",
                "story": 1,
                "room_index": 6,
                "poly": [
                    [0.0, 2.4, 0.0],
                    [4.0, 2.4, 0.0],
                    [4.0, 3.1, 4.0],
                    [0.0, 3.1, 4.0],
                ],
            }
        ],
        room_records=room_records,
    )

    room_partition = updated["room_partitions"][0]
    assert room_partition["room_index"] == 6
    assert room_partition["partition_count"] == 1
    assert room_partition["partitions"][0]["kind"] == "oblique"
    assert updated["metadata"]["simple_slant_partition_count"] == 1


def test_hypothesis_graph_uses_continuation_edges_to_link_candidates() -> None:
    bldg = {
        "rooms": [
            {"story": 0, "floor_polygon": _rect(0.0, 0.0, 4.0, 4.0)},
            {"story": 0, "floor_polygon": _rect(4.0, 0.0, 8.0, 4.0)},
        ]
    }
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.8,
        },
        {
            "room_index": 1,
            "story": 0,
            "graph_room_id": "room:r1",
            "fp": _rect(4.0, 0.0, 8.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.8,
        },
    ]
    oblique_surfaces = [
        {
            "dominant_story": 0,
            "corners": [
                [0.0, 2.8, 0.0],
                [4.0, 3.0, 0.0],
                [4.0, 3.0, 4.0],
                [0.0, 2.8, 4.0],
            ],
            "center": {"x": 2.0, "y": 2.9, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 10.0,
                "room_indices": [0],
                "segs": [{}, {}, {}, {}],
            },
            "space_boundary_ids": ["boundary:roof:o0"],
        },
        {
            "dominant_story": 0,
            "corners": [
                [4.0, 3.0, 0.0],
                [8.0, 3.2, 0.0],
                [8.0, 3.2, 4.0],
                [4.0, 3.0, 4.0],
            ],
            "center": {"x": 6.0, "y": 3.1, "z": 2.0},
            "cluster": {
                "avgAzimuth": 90.0,
                "avgIncl": 10.0,
                "room_indices": [1],
                "segs": [{}, {}, {}, {}],
            },
            "space_boundary_ids": ["boundary:roof:o1"],
        },
    ]

    hypothesis_graph = build_roof_hypothesis_graph(
        bldg=bldg,
        exposed_rooms=exposed_rooms,
        oblique_surfaces=oblique_surfaces,
        flat_surfaces=[],
        roof_graph={
            "nodes": [],
            "edges": [
                {
                    "type": "CONTINUES_AS",
                    "from": "boundary:roof:o0",
                    "to": "boundary:roof:o1",
                    "evidence": {
                        "shared_edge_length_m": 4.0,
                        "relation_state": "confirmed",
                        "exact_face_incidence": True,
                        "partition_atom_pairs": [["a0", "a1"]],
                    },
                },
                {
                    "type": "CONTINUES_AS",
                    "from": "boundary:roof:o1",
                    "to": "boundary:roof:o0",
                    "evidence": {
                        "shared_edge_length_m": 4.0,
                        "relation_state": "confirmed",
                        "exact_face_incidence": True,
                        "partition_atom_pairs": [["a1", "a0"]],
                    },
                },
            ],
            "metadata": {},
        },
        graph=_graph_for_rooms(),
    )

    hypothesis_nodes = {
        node["id"]: node
        for node in hypothesis_graph["nodes"]
        if node.get("type") == "RoofHypothesis"
    }

    assert hypothesis_graph["metadata"]["continuation_edge_count"] == 2
    assert (
        hypothesis_nodes["roof-hypothesis:oblique:0"]["continuation_component_size"]
        == 2
    )
    assert (
        hypothesis_nodes["roof-hypothesis:oblique:1"]["continuation_component_size"]
        == 2
    )
