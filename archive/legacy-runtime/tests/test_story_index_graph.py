from __future__ import annotations

from reconcile.roof_algorithms_py.story_index import build_story_index
from reconcile_v2.models import GraphEdge, GraphNode, TopologyGraph


def _rect_floor(
    x0: float, z0: float, x1: float, z1: float, *, y: float = 0.0
) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def test_has_floor_above_uses_graph_partial_relation_when_geometric_probe_misses() -> (
    None
):
    bldg = {
        "rooms": [
            {"story": 0, "floor_polygon": _rect_floor(0.0, 0.0, 2.0, 2.0)},
            {
                "story": 1,
                "floor_polygon": _rect_floor(100.0, 100.0, 102.0, 102.0, y=3.0),
            },
        ]
    }
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(
                id="room:r0",
                type="Room",
                story=0,
                bbox_xz=[0.0, 0.0, 2.0, 2.0],
            ),
            GraphNode(
                id="room:r1",
                type="Room",
                story=1,
                bbox_xz=[0.0, 0.0, 2.0, 2.0],
            ),
        ],
        edges=[
            GraphEdge(
                id="e:above",
                type="ABOVE",
                from_id="room:r1",
                to_id="room:r0",
                evidence={"relation_state": "partial", "support_ratio": 0.12},
            ),
            GraphEdge(
                id="e:below",
                type="BELOW",
                from_id="room:r0",
                to_id="room:r1",
                evidence={"relation_state": "partial", "support_ratio": 0.12},
            ),
        ],
    )

    index = build_story_index(bldg, graph=graph)
    assert index["has_floor_above"](1.0, 1.0, 0) is True


def test_has_floor_above_ignores_weak_graph_relation_and_falls_back_to_geometry() -> (
    None
):
    bldg = {
        "rooms": [
            {"story": 0, "floor_polygon": _rect_floor(0.0, 0.0, 2.0, 2.0)},
            {
                "story": 1,
                "floor_polygon": _rect_floor(100.0, 100.0, 102.0, 102.0, y=3.0),
            },
        ]
    }
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(id="room:r0", type="Room", story=0, bbox_xz=[0.0, 0.0, 2.0, 2.0]),
            GraphNode(id="room:r1", type="Room", story=1, bbox_xz=[0.0, 0.0, 2.0, 2.0]),
        ],
        edges=[
            GraphEdge(
                id="e:above",
                type="ABOVE",
                from_id="room:r1",
                to_id="room:r0",
                evidence={"relation_state": "weak", "support_ratio": 0.01},
            ),
        ],
    )

    index = build_story_index(bldg, graph=graph)
    assert index["has_floor_above"](1.0, 1.0, 0) is False
