from __future__ import annotations

from reconcile.roof_algorithms_py.boundary_model import (
    build_boundary_face_model,
    derive_roof_surfaces_from_boundary_model,
)
from reconcile.roof_algorithms_py.raw_ceiling_sources import (
    collect_raw_oblique_rectangle_clusters,
)
from reconcile.roof_algorithms_py.roof_graph import build_roof_boundary_graph
from reconcile.roof_algorithms_py.roof_hypothesis_graph import (
    build_roof_hypothesis_graph,
)
from reconcile.roof_algorithms_py.segment_collection import collect_oblique_segments
from reconcile.roof_algorithms_py.simple_slant import identify_simple_slant_rooms
from reconcile_v2.models import GraphEdge, GraphNode, TopologyGraph


def _make_room() -> dict:
    return {
        "story": 0,
        "floor_polygon": [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ],
        "ceiling_type": "sloped",
        "ceiling_polygon": [
            [0.0, 2.2, 0.0],
            [4.0, 2.8, 0.0],
            [4.0, 2.8, 4.0],
            [0.0, 2.2, 4.0],
        ],
        "ceiling_ridge_height": 2.8,
        "ceiling_eave_height": 2.2,
        "walls_computed": [
            {
                "id": "wall-1",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 2.8, 0.0],
                    [0.0, 2.2, 0.0],
                ],
            }
        ],
    }


def _make_graph(
    exposed: bool, above: bool = False, adjacent: bool = False
) -> TopologyGraph:
    nodes = [
        GraphNode(
            id="room:r0",
            type="Room",
            story=0,
            bbox_xz=[0.0, 0.0, 4.0, 4.0],
            properties={"is_top_story": exposed},
        ),
        GraphNode(
            id="surface:floor-1",
            type="Surface",
            story=0,
            source_ids=["floor-1"],
            properties={"surface_kind": "floor"},
        ),
        GraphNode(
            id="surface:wall-1",
            type="Surface",
            story=0,
            source_ids=["wall-1"],
            properties={"surface_kind": "wall"},
        ),
        GraphNode(
            id="boundary:wall-1",
            type="Boundary",
            story=0,
        ),
        GraphNode(
            id="cell:outside",
            type="Cell",
            properties={"cell_kind": "outside"},
        ),
    ]
    if above:
        nodes.append(
            GraphNode(
                id="room:r-above",
                type="Room",
                story=1,
                bbox_xz=[0.0, 0.0, 4.0, 4.0],
                properties={"is_top_story": True},
            )
        )
    if adjacent:
        nodes.append(
            GraphNode(
                id="room:r1",
                type="Room",
                story=0,
                bbox_xz=[4.0, 0.0, 8.0, 4.0],
                properties={"is_top_story": True},
            )
        )
    edges = [
        GraphEdge(
            id="contains:room-floor",
            type="CONTAINS",
            from_id="room:r0",
            to_id="surface:floor-1",
        ),
        GraphEdge(
            id="contains:room-wall",
            type="CONTAINS",
            from_id="room:r0",
            to_id="surface:wall-1",
        ),
        GraphEdge(
            id="contains:room-boundary",
            type="CONTAINS",
            from_id="room:r0",
            to_id="boundary:wall-1",
        ),
        GraphEdge(
            id="bounds:boundary-room",
            type="BOUNDS",
            from_id="boundary:wall-1",
            to_id="room:r0",
        ),
        GraphEdge(
            id="bounds:boundary-surface",
            type="BOUNDS",
            from_id="boundary:wall-1",
            to_id="surface:wall-1",
        ),
    ]
    if exposed:
        edges.append(
            GraphEdge(
                id="exposes:room-outside",
                type="EXPOSES_TO",
                from_id="room:r0",
                to_id="cell:outside",
            )
        )
    if above:
        edges.extend(
            [
                GraphEdge(
                    id="above:r-above-r0",
                    type="ABOVE",
                    from_id="room:r-above",
                    to_id="room:r0",
                ),
                GraphEdge(
                    id="below:r0-r-above",
                    type="BELOW",
                    from_id="room:r0",
                    to_id="room:r-above",
                ),
            ]
        )
    if adjacent:
        edges.extend(
            [
                GraphEdge(
                    id="adj:r0-r1",
                    type="ADJACENT_TO",
                    from_id="room:r0",
                    to_id="room:r1",
                ),
                GraphEdge(
                    id="adj:r1-r0",
                    type="ADJACENT_TO",
                    from_id="room:r1",
                    to_id="room:r0",
                ),
            ]
        )
    return TopologyGraph(version="test", metadata={}, nodes=nodes, edges=edges)


def test_collect_oblique_segments_blocks_explicit_room_above() -> None:
    bldg = {"rooms": [_make_room()]}
    graph = _make_graph(exposed=False, above=True)

    segments = collect_oblique_segments(
        bldg=bldg,
        has_floor_above=lambda _x, _z, _s: False,
        graph=graph,
    )

    assert segments == []


def test_collect_oblique_segments_accepts_partial_graph_relations() -> None:
    bldg = {"rooms": [_make_room()]}
    graph = _make_graph(exposed=False, above=False)

    segments = collect_oblique_segments(
        bldg=bldg,
        has_floor_above=lambda _x, _z, _s: False,
        graph=graph,
    )

    assert segments


def test_identify_simple_slant_rooms_blocks_explicit_room_above() -> None:
    bldg = {"rooms": [_make_room()]}
    graph = _make_graph(exposed=False, above=True)

    result = identify_simple_slant_rooms(
        bldg=bldg,
        has_floor_above=lambda _x, _z, _s: False,
        all_stories=[0],
        graph=graph,
    )

    assert result["simple_slant_rooms"] == set()
    assert result["simple_slant_ceilings"] == []


def test_build_ceiling_planes_can_borrow_adjacent_roof_room_support() -> None:
    room0 = _make_room()
    room1 = {
        **_make_room(),
        "floor_polygon": [
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 4.0],
            [4.0, 0.0, 4.0],
        ],
        "walls_computed": [],
    }
    from reconcile.roof_algorithms_py.ceiling_plane_generation import (
        build_ceiling_planes,
    )

    cluster = {
        "avgIncl": 25.0,
        "avgAzimuth": 90.0,
        "segs": [
            {
                "a": [0.0, 2.0, 0.0],
                "b": [4.0, 3.0, 0.0],
                "story": 0,
                "room_idx": 0,
                "graph_room_id": "room:r0",
            },
            {
                "a": [0.0, 2.0, 4.0],
                "b": [4.0, 3.0, 4.0],
                "story": 0,
                "room_idx": 0,
                "graph_room_id": "room:r0",
            },
        ],
    }
    graph = _make_graph(exposed=True, adjacent=True)

    planes = build_ceiling_planes(
        [cluster], bldg={"rooms": [room0, room1]}, graph=graph
    )

    assert planes
    assert planes[0]["room_indices"] == [0, 1]


def test_build_ceiling_planes_uses_seed_room_extent_for_short_ridge_runs() -> None:
    from reconcile.roof_algorithms_py.ceiling_plane_generation import (
        build_ceiling_planes,
    )

    cluster = {
        "avgIncl": 42.0,
        "avgAzimuth": 90.0,
        "segs": [
            {
                "a": [0.2, 3.2, 2.00],
                "b": [2.2, 2.2, 2.04],
                "story": 0,
                "room_idx": 0,
            },
            {
                "a": [0.4, 3.1, 2.03],
                "b": [2.4, 2.1, 2.06],
                "story": 0,
                "room_idx": 0,
            },
        ],
    }

    assert build_ceiling_planes([cluster]) == []

    planes = build_ceiling_planes([cluster], bldg={"rooms": [_make_room()]})

    assert len(planes) == 1
    assert planes[0]["maxRidge"] - planes[0]["minRidge"] > 3.9
    assert planes[0]["seed_room_indices"] == [0]


def test_roof_hypothesis_graph_recovers_self_touching_oblique_polygon() -> None:
    bldg = {"rooms": [_make_room()]}
    exposed_rooms = [
        {
            "room_index": 0,
            "story": 0,
            "fp": _make_room()["floor_polygon"],
            "wallTopY": 3.0,
            "wallTopMin": 2.2,
        }
    ]
    oblique_surfaces = [
        {
            "dominant_story": 0,
            "corners": [
                [0.0, 2.2, 0.0],
                [4.0, 2.8, 0.0],
                [4.0, 2.8, 4.0],
                [2.0, 2.5, 2.0],
                [2.1, 2.5, 2.1],
                [2.0, 2.5, 2.0],
                [0.0, 2.2, 4.0],
            ],
            "cluster": {
                "room_indices": [0],
                "segs": [{"room_idx": 0}, {"room_idx": 0}],
            },
        }
    ]

    graph = build_roof_hypothesis_graph(
        bldg=bldg,
        exposed_rooms=exposed_rooms,
        oblique_surfaces=oblique_surfaces,
        flat_surfaces=[],
        roof_graph={},
    )

    assert graph["selected_hypothesis_ids"] == ["roof-hypothesis:oblique:0"]


def test_collect_raw_oblique_rectangle_clusters_uses_clean_large_rectangles() -> None:
    room = {
        **_make_room(),
        "raw_ceiling_planes": [
            {
                "corners": [
                    [0.0, 3.0, 0.0],
                    [0.0, 2.0, -3.0],
                    [5.0, 2.0, -3.0],
                    [5.0, 3.0, 0.0],
                ]
            },
            {
                "corners": [
                    [0.0, 3.0, 0.0],
                    [0.0, 2.0, -3.0],
                    [5.0, 2.0, -3.0],
                ]
            },
        ],
        "ceiling_type": None,
    }

    clusters = collect_raw_oblique_rectangle_clusters(
        bldg={"uuid": "b-test", "rooms": [room]},
        existing_clusters=[],
        has_floor_above=lambda _x, _z, _story: False,
    )

    assert len(clusters) == 1
    assert clusters[0]["source"] == "raw_ceiling_rectangle"
    assert clusters[0]["room_indices"] == [0]
    assert clusters[0]["raw_plane_ids"] == ["b-test::ceiling-raw::0:0:0"]
    assert len(clusters[0]["segs"]) == 2


def test_boundary_face_model_emits_shared_space_boundary_records() -> None:
    bldg = {"rooms": [_make_room()]}
    graph = _make_graph(exposed=True)
    roof_result = {
        "roof_surfaces": {
            "oblique": [
                {
                    "dominant_story": 0,
                    "corners": [
                        [0.0, 2.2, 0.0],
                        [4.0, 2.8, 0.0],
                        [4.0, 2.8, 4.0],
                        [0.0, 2.2, 4.0],
                    ],
                    "cluster": {"room_indices": [0]},
                }
            ],
            "flat": [],
        }
    }

    boundary_model = build_boundary_face_model(
        bldg=bldg, roof_result=roof_result, graph=graph
    )
    derived = derive_roof_surfaces_from_boundary_model(boundary_model)

    wall_boundary = next(
        boundary
        for boundary in boundary_model["boundaries"]
        if boundary["role"] == "wall"
    )
    slab_boundary = next(
        boundary
        for boundary in boundary_model["boundaries"]
        if boundary["role"] == "slab"
    )
    roof_surface = derived["oblique"][0]

    assert wall_boundary["graph_surface_id"] == "surface:wall-1"
    assert slab_boundary["graph_surface_id"] == "surface:floor-1"
    assert roof_surface["boundary_face_id"].startswith("face:")
    assert roof_surface["space_boundary_ids"]
    assert roof_surface["space_boundary_ids"][0].startswith("boundary:roof:")


def test_roof_boundary_graph_emits_continuation_edges() -> None:
    room0 = _make_room()
    room1 = {
        **_make_room(),
        "floor_polygon": [
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 4.0],
            [4.0, 0.0, 4.0],
        ],
        "walls_computed": [],
    }
    graph = _make_graph(exposed=True, adjacent=True)
    roof_result = {
        "roof_surfaces": {
            "oblique": [
                {
                    "dominant_story": 0,
                    "corners": [
                        [0.0, 2.2, 0.0],
                        [4.0, 2.8, 0.0],
                        [4.0, 2.8, 4.0],
                        [0.0, 2.2, 4.0],
                    ],
                    "cluster": {"room_indices": [0]},
                },
                {
                    "dominant_story": 0,
                    "corners": [
                        [4.0, 2.2, 0.0],
                        [8.0, 2.8, 0.0],
                        [8.0, 2.8, 4.0],
                        [4.0, 2.2, 4.0],
                    ],
                    "cluster": {"room_indices": [1]},
                },
            ],
            "flat": [],
        }
    }
    boundary_model = build_boundary_face_model(
        bldg={"rooms": [room0, room1]}, roof_result=roof_result, graph=graph
    )
    roof_graph = build_roof_boundary_graph(boundary_model, graph=graph)

    continuation_edges = [
        edge for edge in roof_graph["edges"] if edge["type"] == "CONTINUES_AS"
    ]
    assert continuation_edges
    assert all(
        edge["evidence"]["relation_state"] in {"confirmed", "partial"}
        for edge in continuation_edges
    )
    assert all(
        edge["evidence"]["exact_face_incidence"] is True for edge in continuation_edges
    )
    assert all(
        edge["evidence"]["shared_edge_length_m"] > 0.0 for edge in continuation_edges
    )
    assert roof_graph["metadata"]["partition_face_count"] >= 2


def test_roof_boundary_graph_does_not_continue_across_different_oblique_planes() -> (
    None
):
    room0 = _make_room()
    room1 = {
        **_make_room(),
        "floor_polygon": [
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 4.0],
            [4.0, 0.0, 4.0],
        ],
        "walls_computed": [],
    }
    graph = _make_graph(exposed=True, adjacent=True)
    roof_result = {
        "roof_surfaces": {
            "oblique": [
                {
                    "dominant_story": 0,
                    "corners": [
                        [0.0, 2.2, 0.0],
                        [4.0, 2.8, 0.0],
                        [4.0, 2.8, 4.0],
                        [0.0, 2.2, 4.0],
                    ],
                    "cluster": {"room_indices": [0]},
                },
                {
                    "dominant_story": 0,
                    "corners": [
                        [4.0, 2.8, 0.0],
                        [8.0, 2.2, 0.0],
                        [8.0, 2.2, 4.0],
                        [4.0, 2.8, 4.0],
                    ],
                    "cluster": {"room_indices": [1]},
                },
            ],
            "flat": [],
        }
    }
    boundary_model = build_boundary_face_model(
        bldg={"rooms": [room0, room1]}, roof_result=roof_result, graph=graph
    )
    roof_graph = build_roof_boundary_graph(boundary_model, graph=graph)

    continuation_edges = [
        edge for edge in roof_graph["edges"] if edge["type"] == "CONTINUES_AS"
    ]
    assert continuation_edges == []
