"""Tests for room_postprocessing corner-sharing graph."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.export import build_corner_graph
from reconcile_tiers.room_postprocessing.flatten_payload import flatten_tier_payload


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _shared_corner_payload() -> dict[str, Any]:
    """Two walls sharing one corner cluster; edges along shared corner not isolated."""

    shared = _pt(1.0, 0.0, 0.0)
    return {
        "uuid": "test-building",
        "rooms": [
            {
                "story": 0,
                "locator_id": "b::tier-room::0",
                "floor": {"corners": [_pt(0, 0, 0), _pt(2, 0, 0), _pt(2, 0, 2), _pt(0, 0, 2)]},
                "walls": [
                    {
                        "locator_id": "b::tier-wall::a",
                        "corners": [
                            _pt(0, 0, 0),
                            shared,
                            _pt(1.0, 2.0, 0.0),
                            _pt(0, 2.0, 0.0),
                        ],
                    },
                    {
                        "locator_id": "b::tier-wall::b",
                        "corners": [
                            shared,
                            _pt(2.0, 0.0, 0.0),
                            _pt(2.0, 2.0, 0.0),
                            _pt(1.0, 2.0, 0.0),
                        ],
                    },
                ],
            }
        ],
        "ceiling": [
            {
                "locator_id": "b::tier-ceiling::0",
                "corners": [_pt(0, 2, 0), _pt(2, 2, 0), _pt(2, 2, 2), _pt(0, 2, 2)],
            }
        ],
        "visual_shells": [],
        "gable_closures": [],
    }


def _floating_wall_payload() -> dict[str, Any]:
    """Single wall with unique corners — degree 0, all edges isolated."""

    return {
        "uuid": "float-building",
        "rooms": [
            {
                "story": 0,
                "floor": {"corners": [_pt(0, 0, 0), _pt(4, 0, 0), _pt(4, 0, 4), _pt(0, 0, 4)]},
                "walls": [
                    {
                        "locator_id": "float::tier-wall::solo",
                        "corners": [
                            _pt(10, 0, 0),
                            _pt(11, 0, 0),
                            _pt(11, 2, 0),
                            _pt(10, 2, 0),
                        ],
                    }
                ],
            }
        ],
        "ceiling": [],
    }


def test_flatten_includes_floor_wall_excludes_ceiling() -> None:
    payload = _shared_corner_payload()
    elements = flatten_tier_payload(payload)
    kinds = {e.kind for e in elements}
    assert "floor" in kinds
    assert "wall" in kinds
    assert "ceiling" not in kinds
    assert len([e for e in elements if e.kind == "wall"]) == 2


def test_shared_corner_walls_adjacent_not_isolated_on_shared_edge() -> None:
    graph = build_corner_graph(_shared_corner_payload(), corner_tol=0.05)
    wall_ids = {n["id"] for n in graph["nodes"] if n["kind"] == "wall"}
    assert len(wall_ids) == 2

    wall_a = "b::tier-wall::a"
    wall_b = "b::tier-wall::b"
    wall_wall_edges = [
        e
        for e in graph["edges"]
        if e["source"] in (wall_a, wall_b) and e["target"] in (wall_a, wall_b)
    ]
    assert len(wall_wall_edges) == 1

    segments = graph["wall_edge_segments"]
    shared_edge_segments = [
        s
        for s in segments
        if s["element_id"] in (wall_a, wall_b)
        and (
            (s["start"]["x"], s["start"]["z"]) == (1.0, 0.0)
            or (s["end"]["x"], s["end"]["z"]) == (1.0, 0.0)
        )
    ]
    assert any(not s["isolated"] for s in shared_edge_segments)


def test_floating_wall_degree_zero_all_edges_isolated() -> None:
    graph = build_corner_graph(_floating_wall_payload(), corner_tol=0.05)
    wall_nodes = [n for n in graph["nodes"] if n["kind"] == "wall"]
    assert len(wall_nodes) == 1
    assert wall_nodes[0]["degree"] == 0
    assert graph["edges"] == []

    wall_segments = [
        s for s in graph["wall_edge_segments"] if s["element_id"] == "float::tier-wall::solo"
    ]
    assert len(wall_segments) == 4
    assert all(s["isolated"] for s in wall_segments)
