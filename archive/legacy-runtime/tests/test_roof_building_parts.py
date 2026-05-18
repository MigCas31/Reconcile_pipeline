from __future__ import annotations

from reconcile.roof_algorithms_py.graph_support import match_graph_room
from reconcile.roof_algorithms_py.roof_building_parts import (
    build_building_part_graph,
    classify_selected_flat_hypothesis_roles,
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


def _graph() -> TopologyGraph:
    return TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(id="room:r0", type="Room", story=0, bbox_xz=[0.0, 0.0, 4.0, 4.0]),
            GraphNode(id="room:r1", type="Room", story=0, bbox_xz=[4.0, 0.0, 8.0, 4.0]),
            GraphNode(
                id="room:r2", type="Room", story=0, bbox_xz=[8.0, 0.0, 12.0, 4.0]
            ),
        ],
        edges=[
            GraphEdge(
                id="adj:r0-r1",
                type="ADJACENT_TO",
                from_id="room:r0",
                to_id="room:r1",
                evidence={"relation_state": "confirmed"},
            ),
            GraphEdge(
                id="adj:r1-r2",
                type="ADJACENT_TO",
                from_id="room:r1",
                to_id="room:r2",
                evidence={"relation_state": "confirmed"},
            ),
        ],
    )


def _graph_with_lower_room() -> TopologyGraph:
    return TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(id="room:r0", type="Room", story=1, bbox_xz=[0.0, 0.0, 4.0, 4.0]),
            GraphNode(id="room:r1", type="Room", story=1, bbox_xz=[4.0, 0.0, 8.0, 4.0]),
            GraphNode(id="room:r2", type="Room", story=0, bbox_xz=[0.0, 0.0, 4.0, 4.0]),
        ],
        edges=[
            GraphEdge(
                id="adj:r0-r1",
                type="ADJACENT_TO",
                from_id="room:r0",
                to_id="room:r1",
                evidence={"relation_state": "confirmed"},
            ),
            GraphEdge(
                id="below:r2-r0",
                type="BELOW",
                from_id="room:r2",
                to_id="room:r0",
                evidence={"relation_state": "confirmed"},
            ),
        ],
    )


def _graph_with_isolated_room() -> TopologyGraph:
    return TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(id="room:r0", type="Room", story=0, bbox_xz=[0.0, 0.0, 4.0, 4.0]),
            GraphNode(id="room:r1", type="Room", story=0, bbox_xz=[4.0, 0.0, 8.0, 4.0]),
            GraphNode(
                id="room:r2", type="Room", story=0, bbox_xz=[20.0, 0.0, 24.0, 4.0]
            ),
        ],
        edges=[
            GraphEdge(
                id="adj:r0-r1",
                type="ADJACENT_TO",
                from_id="room:r0",
                to_id="room:r1",
                evidence={"relation_state": "confirmed"},
            ),
        ],
    )


def test_match_graph_room_prefers_exact_cell_footprint_overlap_over_bbox_centroid() -> (
    None
):
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(
                id="room:r0", type="Room", story=0, bbox_xz=[0.0, 0.0, 10.0, 10.0]
            ),
            GraphNode(
                id="room:r1", type="Room", story=0, bbox_xz=[0.0, 0.0, 10.0, 10.0]
            ),
        ],
        edges=[],
        geometry_index={
            "cell_complex": {
                "cells": [
                    {
                        "id": "cell:r0",
                        "kind": "room",
                        "source_id": "room:r0",
                        "properties": {
                            "xz_footprint": [
                                [0.0, 0.0],
                                [4.0, 0.0],
                                [4.0, 4.0],
                                [0.0, 4.0],
                            ]
                        },
                    },
                    {
                        "id": "cell:r1",
                        "kind": "room",
                        "source_id": "room:r1",
                        "properties": {
                            "xz_footprint": [
                                [4.0, 0.0],
                                [8.0, 0.0],
                                [8.0, 4.0],
                                [4.0, 4.0],
                            ]
                        },
                    },
                ]
            }
        },
    )

    matched = match_graph_room(graph, 0, _rect(4.1, 0.0, 7.9, 3.9))

    assert matched is not None
    assert matched.id == "room:r1"


def test_match_graph_room_uses_convex_hull_when_raw_floor_polygon_is_invalid() -> None:
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(id="room:r0", type="Room", story=0, bbox_xz=[0.0, 0.0, 4.0, 4.0]),
            GraphNode(id="room:r1", type="Room", story=0, bbox_xz=[4.0, 0.0, 8.0, 4.0]),
        ],
        edges=[],
        geometry_index={
            "cell_complex": {
                "cells": [
                    {
                        "id": "cell:r0",
                        "kind": "room",
                        "source_id": "room:r0",
                        "properties": {
                            "xz_footprint": [
                                [0.0, 0.0],
                                [4.0, 0.0],
                                [4.0, 4.0],
                                [0.0, 4.0],
                            ]
                        },
                    },
                    {
                        "id": "cell:r1",
                        "kind": "room",
                        "source_id": "room:r1",
                        "properties": {
                            "xz_footprint": [
                                [4.0, 0.0],
                                [8.0, 0.0],
                                [8.0, 4.0],
                                [4.0, 4.0],
                            ]
                        },
                    },
                ]
            }
        },
    )

    matched = match_graph_room(
        graph,
        0,
        [
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 4.0],
            [8.0, 0.0, 0.0],
            [6.0, 0.0, 2.0],
        ],
    )

    assert matched is not None
    assert matched.id == "room:r1"


def test_match_graph_room_uses_largest_polygon_from_make_valid_geometry_collection():
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(
                id="room:r0", type="Room", story=0, bbox_xz=[-6.0, -5.0, -2.0, 2.0]
            ),
            GraphNode(
                id="room:r1", type="Room", story=0, bbox_xz=[-4.0, -5.0, 6.0, 5.0]
            ),
        ],
        edges=[],
        geometry_index={
            "cell_complex": {
                "cells": [
                    {
                        "id": "cell:r0",
                        "kind": "room",
                        "source_id": "room:r0",
                        "properties": {
                            "xz_footprint": [
                                [-6.0, -5.0],
                                [-2.0, -5.0],
                                [-2.0, 2.0],
                                [-6.0, 2.0],
                            ]
                        },
                    },
                    {
                        "id": "cell:r1",
                        "kind": "room",
                        "source_id": "room:r1",
                        "properties": {
                            "xz_footprint": [
                                [-4.0, -5.0],
                                [6.0, -5.0],
                                [6.0, 5.0],
                                [-4.0, 5.0],
                            ]
                        },
                    },
                ]
            }
        },
    )

    matched = match_graph_room(
        graph,
        0,
        [
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
        ],
    )

    assert matched is not None
    assert matched.id == "room:r1"


def _hypothesis_graph() -> dict:
    return {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "selected": True,
            },
            {
                "id": "roof-hypothesis:oblique:1",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "selected": True,
            },
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "selected": True,
                "support_room_indices": [0],
            },
            {
                "id": "roof-hypothesis:flat:1",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "selected": True,
                "support_room_indices": [1],
            },
            {
                "id": "roof-hypothesis:flat:2",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "selected": True,
                "support_room_indices": [2],
            },
            {
                "id": "roof-hypothesis:flat:global",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "selected": True,
                "support_room_indices": [],
            },
        ],
        "edges": [
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:0",
                "to": "room:0",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            },
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:1",
                "to": "room:2",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            },
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:0",
                "to": "room:0",
                "selected": True,
                "evidence": {"edge_score": 0.7},
            },
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:1",
                "to": "room:1",
                "selected": True,
                "evidence": {"edge_score": 0.7},
            },
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:2",
                "to": "room:2",
                "selected": True,
                "evidence": {"edge_score": 0.7},
            },
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:flat:global",
                "to": "room:1",
                "selected": True,
                "evidence": {"edge_score": 0.5},
            },
            {
                "type": "CONTINUES_AS",
                "from": "roof-hypothesis:oblique:0",
                "to": "roof-hypothesis:oblique:1",
                "evidence": {"shared_edge_length_m": 4.0},
            },
        ],
    }


def test_part_graph_detects_sloped_from_room_oblique() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.1,
            "wallTopMin": 2.0,
        },
        {
            "room_index": 1,
            "graph_room_id": "room:r1",
            "fp": _rect(4.0, 0.0, 8.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.1,
        },
        {
            "room_index": 2,
            "graph_room_id": "room:r2",
            "fp": _rect(8.0, 0.0, 12.0, 4.0),
            "wallTopY": 3.1,
            "wallTopMin": 2.2,
        },
    ]

    part_graph = build_building_part_graph(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=_hypothesis_graph(),
        graph=_graph(),
    )

    assert part_graph["metadata"]["part_count"] >= 1
    assert any(
        node["roof_family_guess"] == "gable_or_multi_slope"
        for node in part_graph["nodes"]
    )


def test_building_part_graph_keeps_isolated_supported_room_as_its_own_part() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.1,
            "wallTopMin": 2.0,
        },
        {
            "room_index": 1,
            "graph_room_id": "room:r1",
            "fp": _rect(4.0, 0.0, 8.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.1,
        },
        {
            "room_index": 2,
            "graph_room_id": "room:r2",
            "fp": _rect(20.0, 0.0, 24.0, 4.0),
            "wallTopY": 3.2,
            "wallTopMin": 2.0,
        },
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "selected": True,
            },
            {
                "id": "roof-hypothesis:oblique:1",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "selected": True,
            },
        ],
        "edges": [
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:0",
                "to": "room:0",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            },
            {
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:1",
                "to": "room:2",
                "selected": True,
                "evidence": {"edge_score": 0.8},
            },
        ],
    }

    part_graph = build_building_part_graph(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=hypothesis_graph,
        graph=_graph_with_isolated_room(),
    )

    assert "room:2" in part_graph["room_membership"]
    assert part_graph["room_membership"]["room:2"]
    assert any(node["room_ids"] == ["room:2"] for node in part_graph["nodes"])


def test_room_specific_flat_hypotheses_become_caps_inside_sloped_part() -> None:
    exposed_rooms = [
        {
            "room_index": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.1,
            "wallTopMin": 2.0,
        },
        {
            "room_index": 1,
            "graph_room_id": "room:r1",
            "fp": _rect(4.0, 0.0, 8.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.1,
        },
        {
            "room_index": 2,
            "graph_room_id": "room:r2",
            "fp": _rect(8.0, 0.0, 12.0, 4.0),
            "wallTopY": 3.1,
            "wallTopMin": 2.2,
        },
    ]
    hypothesis_graph = _hypothesis_graph()
    part_graph = build_building_part_graph(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=hypothesis_graph,
        graph=_graph(),
    )

    roles = classify_selected_flat_hypothesis_roles(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=hypothesis_graph,
        building_part_graph=part_graph,
    )

    assert roles["roof-hypothesis:flat:0"]["role"] == "ceiling_cap"
    assert roles["roof-hypothesis:flat:1"]["role"] == "ceiling_cap"
    assert roles["roof-hypothesis:flat:2"]["role"] == "ceiling_cap"
    assert roles["roof-hypothesis:flat:global"]["role"] != "ceiling_cap"


def test_building_part_graph_uses_hypothesis_support_and_expands_to_lower_rooms() -> (
    None
):
    exposed_rooms = [
        {
            "room_index": 0,
            "graph_room_id": "room:r0",
            "fp": _rect(0.0, 0.0, 4.0, 4.0),
            "wallTopY": 3.1,
            "wallTopMin": 2.0,
            "story": 1,
        },
        {
            "room_index": 1,
            "graph_room_id": "room:r1",
            "fp": _rect(4.0, 0.0, 8.0, 4.0),
            "wallTopY": 3.0,
            "wallTopMin": 2.1,
            "story": 1,
        },
    ]
    bldg = {
        "rooms": [
            {"story": 1, "floor_polygon": _rect(0.0, 0.0, 4.0, 4.0)},
            {"story": 1, "floor_polygon": _rect(4.0, 0.0, 8.0, 4.0)},
            {"story": 0, "floor_polygon": _rect(0.0, 0.0, 4.0, 4.0)},
        ]
    }
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:flat:0",
                "type": "RoofHypothesis",
                "surface_kind": "flat",
                "selected": True,
                "support_room_indices": [0],
            }
        ],
        "edges": [],
    }

    part_graph = build_building_part_graph(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=hypothesis_graph,
        bldg=bldg,
        graph=_graph_with_lower_room(),
    )

    assert part_graph["room_membership"]["room:0"]
    assert (
        part_graph["room_membership"]["room:2"]
        == part_graph["room_membership"]["room:0"]
    )
    part_id = part_graph["room_membership"]["room:0"][0]
    part = next(node for node in part_graph["nodes"] if node["id"] == part_id)
    assert "room:2" in part["room_ids"]
