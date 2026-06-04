"""Tests for segment-room cycle detection."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.export import build_corner_graph


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _four_wall_room_payload() -> dict[str, Any]:
    return {
        "uuid": "four-wall-room",
        "rooms": [
            {
                "story": 0,
                "walls": [
                    {
                        "locator_id": "w-south",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(4, 0, 0),
                            _pt(4, 2, 0),
                            _pt(0, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-east",
                        "corners": [
                            _pt(4, 0, 0),
                            _pt(4, 0, 3),
                            _pt(4, 2, 3),
                            _pt(4, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-north",
                        "corners": [
                            _pt(4, 0, 3),
                            _pt(0, 0, 3),
                            _pt(0, 2, 3),
                            _pt(4, 2, 3),
                        ],
                    },
                    {
                        "locator_id": "w-west",
                        "corners": [
                            _pt(0, 0, 3),
                            _pt(0, 0, 0),
                            _pt(0, 2, 0),
                            _pt(0, 2, 3),
                        ],
                    },
                ],
            }
        ],
    }


def _two_adjacent_rooms_payload() -> dict[str, Any]:
    """Two 2×2 m rooms sharing the wall at x=2."""

    return {
        "uuid": "two-rooms",
        "rooms": [
            {
                "story": 0,
                "walls": [
                    {
                        "locator_id": "w-left-south",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(2, 0, 0),
                            _pt(2, 2, 0),
                            _pt(0, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-shared",
                        "corners": [
                            _pt(2, 0, 0),
                            _pt(2, 0, 2),
                            _pt(2, 2, 2),
                            _pt(2, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-left-north",
                        "corners": [
                            _pt(2, 0, 2),
                            _pt(0, 0, 2),
                            _pt(0, 2, 2),
                            _pt(2, 2, 2),
                        ],
                    },
                    {
                        "locator_id": "w-left-west",
                        "corners": [
                            _pt(0, 0, 2),
                            _pt(0, 0, 0),
                            _pt(0, 2, 0),
                            _pt(0, 2, 2),
                        ],
                    },
                    {
                        "locator_id": "w-right-south",
                        "corners": [
                            _pt(2, 0, 0),
                            _pt(4, 0, 0),
                            _pt(4, 2, 0),
                            _pt(2, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-right-east",
                        "corners": [
                            _pt(4, 0, 0),
                            _pt(4, 0, 2),
                            _pt(4, 2, 2),
                            _pt(4, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-right-north",
                        "corners": [
                            _pt(4, 0, 2),
                            _pt(2, 0, 2),
                            _pt(2, 2, 2),
                            _pt(4, 2, 2),
                        ],
                    },
                ],
            }
        ],
    }


def _single_wall_quad_payload() -> dict[str, Any]:
    return {
        "uuid": "one-wall",
        "rooms": [
            {
                "walls": [
                    {
                        "locator_id": "w-single",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(1, 0, 0),
                            _pt(1, 2, 0),
                            _pt(0, 2, 0),
                        ],
                    }
                ],
            }
        ],
    }


def test_four_wall_room_yields_one_cycle() -> None:
    graph = build_corner_graph(_four_wall_room_payload(), corner_tol=0.05)
    rg = graph["segment_room_graph"]
    rooms = [n for n in rg["nodes"] if n["kind"] == "segment_room"]
    assert len(rooms) >= 1
    room = rooms[0]
    assert len(room["group_ids"]) >= 3
    assert len(room["wall_ids"]) >= 3
    assert room["area_m2"] >= 1.0


def test_single_wall_has_no_room_cycles() -> None:
    graph = build_corner_graph(_single_wall_quad_payload(), corner_tol=0.05)
    rg = graph["segment_room_graph"]
    assert rg["nodes"] == []
    sg = graph["wall_segment_graph"]
    assert all(n.get("orphan") for n in sg["nodes"])


def test_four_wall_room_groups_not_orphan() -> None:
    graph = build_corner_graph(_four_wall_room_payload(), corner_tol=0.05)
    sg = graph["wall_segment_graph"]
    room_groups = set()
    for room in graph["segment_room_graph"]["nodes"]:
        room_groups.update(room["group_ids"])
    for node in sg["nodes"]:
        if node["id"] in room_groups:
            assert node.get("in_room_cycle") is True
            assert node.get("orphan") is False


def _room_with_dead_end_stub_payload() -> dict[str, Any]:
    """Closed room plus a short internal wall whose free end is a graph leaf."""

    payload = _four_wall_room_payload()
    payload["rooms"][0]["walls"] = list(payload["rooms"][0]["walls"]) + [
        {
            "locator_id": "w-stub",
            "corners": [
                _pt(2.0, 0.0, 0.0),
                _pt(2.0, 0.0, 1.5),
                _pt(2.0, 2.0, 1.5),
                _pt(2.0, 2.0, 0.0),
            ],
        },
    ]
    return payload


def test_dead_end_stub_group_is_orphan() -> None:
    graph = build_corner_graph(_room_with_dead_end_stub_payload(), corner_tol=0.05)
    sg = graph["wall_segment_graph"]
    leaves = [n for n in sg["nodes"] if n.get("junction_degree", n.get("degree", 0)) <= 1]
    assert leaves
    assert all(n.get("orphan") for n in leaves)
    room_groups: set[str] = set()
    for room in graph["segment_room_graph"]["nodes"]:
        room_groups.update(room["group_ids"])
    for leaf in leaves:
        assert leaf["id"] not in room_groups


def test_two_adjacent_rooms_share_wall_edge() -> None:
    graph = build_corner_graph(_two_adjacent_rooms_payload(), corner_tol=0.05)
    rg = graph["segment_room_graph"]
    rooms = [n for n in rg["nodes"] if n["kind"] == "segment_room"]
    assert len(rooms) >= 2
    adj = rg["edges"]
    assert adj
    shared = adj[0].get("shared_wall_ids") or []
    assert "w-shared" in shared
