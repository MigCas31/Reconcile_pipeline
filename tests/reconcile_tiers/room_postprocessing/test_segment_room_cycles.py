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


def test_two_adjacent_rooms_share_wall_edge() -> None:
    graph = build_corner_graph(_two_adjacent_rooms_payload(), corner_tol=0.05)
    rg = graph["segment_room_graph"]
    rooms = [n for n in rg["nodes"] if n["kind"] == "segment_room"]
    assert len(rooms) >= 2
    adj = rg["edges"]
    assert adj
    shared = adj[0].get("shared_wall_ids") or []
    assert "w-shared" in shared
