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


def test_wall_graph_only_walls_and_wall_adjacency() -> None:
    graph = build_corner_graph(_shared_corner_payload(), corner_tol=0.05)
    wg = graph["wall_graph"]
    assert all(n["kind"] == "wall" for n in wg["nodes"])
    assert len(wg["nodes"]) == 2
    assert len(wg["edges"]) == 1
    pair = {wg["edges"][0]["source"], wg["edges"][0]["target"]}
    assert pair == {"b::tier-wall::a", "b::tier-wall::b"}


def _near_miss_three_wall_payload() -> dict[str, Any]:
    """Green/blue corners cluster; orange offset ~0.28 m — needs adjacency_tol."""

    junction = _pt(2.0, 0.0, 2.0)
    return {
        "uuid": "near-miss",
        "rooms": [
            {
                "story": 0,
                "floor": {
                    "corners": [
                        _pt(0, 0, 0),
                        _pt(3, 0, 0),
                        _pt(3, 0, 3),
                        _pt(0, 0, 3),
                    ]
                },
                "walls": [
                    {
                        "locator_id": "w-orange",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(1.72, 0.0, 2.0),
                            _pt(1.72, 2.0, 2.0),
                            _pt(0, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-green",
                        "corners": [
                            junction,
                            _pt(2.06, 0.0, 2.0),
                            _pt(2.06, 2.0, 2.0),
                            _pt(2.0, 2.0, 2.0),
                        ],
                    },
                    {
                        "locator_id": "w-blue",
                        "corners": [
                            _pt(2.06, 0.0, 2.0),
                            _pt(3.0, 0.0, 2.0),
                            _pt(3.0, 2.0, 2.0),
                            _pt(2.0, 2.0, 2.0),
                        ],
                    },
                ],
            }
        ],
    }


def test_near_miss_three_walls_connect_with_adjacency_tol() -> None:
    payload = _near_miss_three_wall_payload()
    graph = build_corner_graph(payload, corner_tol=0.05, adjacency_tol=0.5)
    wg = graph["wall_graph"]
    ids = {n["id"] for n in wg["nodes"]}
    assert ids == {"w-orange", "w-green", "w-blue"}

    graph_tight = build_corner_graph(payload, corner_tol=0.05, adjacency_tol=0.05)
    tight_orange_wall = [
        e
        for e in graph_tight["wall_graph"]["edges"]
        if "w-orange" in (e["source"], e["target"])
    ]
    assert len(tight_orange_wall) == 0

    assert len(wg["edges"]) == 3
    for a, b in (
        ("w-orange", "w-green"),
        ("w-orange", "w-blue"),
        ("w-green", "w-blue"),
    ):
        assert any(
            {e["source"], e["target"]} == {a, b} for e in wg["edges"]
        )


def test_strict_shared_corner_still_connects() -> None:
    graph = build_corner_graph(_shared_corner_payload(), corner_tol=0.05)
    wall_edges = [
        e
        for e in graph["edges"]
        if e["source"].startswith("b::tier-wall")
        and e["target"].startswith("b::tier-wall")
    ]
    assert len(wall_edges) == 1


def test_floating_wall_degree_zero_all_edges_isolated() -> None:
    graph = build_corner_graph(
        _floating_wall_payload(),
        corner_tol=0.05,
        adjacency_tol=0.5,
    )
    wall_nodes = [n for n in graph["nodes"] if n["kind"] == "wall"]
    assert len(wall_nodes) == 1
    assert wall_nodes[0]["degree"] == 0
    assert graph["edges"] == []

    wall_segments = [
        s for s in graph["wall_edge_segments"] if s["element_id"] == "float::tier-wall::solo"
    ]
    assert len(wall_segments) == 4
    assert all(s["isolated"] for s in wall_segments)
