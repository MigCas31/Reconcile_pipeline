"""Tests for wall vertical-segment graph (approx groups as nodes)."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.export import build_corner_graph


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _near_miss_three_wall_payload() -> dict[str, Any]:
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


def _group_for_segment(sg: dict[str, Any], segment_id: str) -> dict[str, Any] | None:
    for node in sg["nodes"]:
        if segment_id in node["segment_ids"]:
            return node
    return None


def _component_size(graph: dict[str, Any], start_id: str) -> int:
    edges = graph["edges"]
    adj: dict[str, list[str]] = {}
    for edge in edges:
        adj.setdefault(edge["source"], []).append(edge["target"])
        adj.setdefault(edge["target"], []).append(edge["source"])
    seen = {start_id}
    queue = [start_id]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def test_no_floor_nodes_in_wall_segment_graph() -> None:
    graph = build_corner_graph(_near_miss_three_wall_payload())
    sg = graph["wall_segment_graph"]
    assert all(n["kind"] == "approx_segment_group" for n in sg["nodes"])
    assert sg["segments"]
    assert not any(n["id"].startswith("room:") for n in sg["nodes"])


def test_single_wall_has_two_segments_one_group_edge() -> None:
    graph = build_corner_graph(_single_wall_quad_payload(), corner_tol=0.05)
    sg = graph["wall_segment_graph"]
    assert len(sg["segments"]) == 2
    assert len(sg["nodes"]) == 2
    assert len(sg["edges"]) == 1
    assert sg["edges"][0]["kind"] == "intra_wall"
    assert {sg["edges"][0]["source"], sg["edges"][0]["target"]} == {
        "approx_grp::0",
        "approx_grp::1",
    }


def _passing_wall_junction_payload() -> dict[str, Any]:
    """Three walls at a near-miss junction plus one long wall crossing through it."""

    base = _near_miss_three_wall_payload()
    room = base["rooms"][0]
    room["walls"] = list(room["walls"]) + [
        {
            "locator_id": "w-long",
            "corners": [
                _pt(0.5, 0.0, 2.0),
                _pt(2.8, 0.0, 2.0),
                _pt(2.8, 2.0, 2.0),
                _pt(0.5, 2.0, 2.0),
            ],
        },
    ]
    return base


def test_passing_wall_splits_and_shares_junction_approx_group() -> None:
    graph = build_corner_graph(_passing_wall_junction_payload(), adjacency_tol=0.5)
    wall_ids = {n["id"] for n in graph["nodes"] if n["kind"] == "wall"}
    assert "w-long::split::0" in wall_ids
    assert "w-long::split::1" in wall_ids

    sg = graph["wall_segment_graph"]
    split_segments = [s for s in sg["segments"] if s["wall_id"].startswith("w-long::split::")]
    assert len(split_segments) >= 4
    junction_seg = next(
        s
        for s in split_segments
        if abs(s["start"]["x"] - 2.0) < 0.2 or abs(s["end"]["x"] - 2.0) < 0.2
    )
    grp = _group_for_segment(sg, junction_seg["id"])
    assert grp is not None
    assert grp["segment_count"] >= 2
    assert len(grp["wall_ids"]) >= 2


def test_approx_groups_do_not_merge_across_stories() -> None:
    """Vertically aligned junction segments on different storeys stay separate."""

    payload = {
        "uuid": "stacked-stories",
        "rooms": [
            {
                "story": 0,
                "walls": [
                    {
                        "locator_id": "w-l0",
                        "corners": [
                            _pt(1.0, 0.0, 1.0),
                            _pt(1.1, 0.0, 1.1),
                            _pt(1.1, 2.0, 1.1),
                            _pt(1.0, 2.0, 1.0),
                        ],
                    },
                ],
            },
            {
                "story": 1,
                "walls": [
                    {
                        "locator_id": "w-l1",
                        "corners": [
                            _pt(1.0, 2.05, 1.0),
                            _pt(1.1, 2.05, 1.1),
                            _pt(1.1, 4.1, 1.1),
                            _pt(1.0, 4.1, 1.0),
                        ],
                    },
                ],
            },
        ],
    }
    graph = build_corner_graph(payload, corner_tol=0.05, adjacency_tol=0.5)
    sg = graph["wall_segment_graph"]
    seg_l0 = next(s for s in sg["segments"] if s["wall_id"] == "w-l0")
    seg_l1 = next(s for s in sg["segments"] if s["wall_id"] == "w-l1")
    grp_l0 = _group_for_segment(sg, seg_l0["id"])
    grp_l1 = _group_for_segment(sg, seg_l1["id"])
    assert grp_l0 is not None
    assert grp_l1 is not None
    assert grp_l0["id"] != grp_l1["id"]


def test_leaf_bridge_connects_degree_one_group_when_gap_within_bridge_tol() -> None:
    """Tight approx groups stay separate; leaf bridge links a dead-end to a neighbor."""

    from reconcile_tiers.room_postprocessing.corner_graph import (
        cluster_element_corners,
    )
    from reconcile_tiers.room_postprocessing.flatten_payload import flatten_tier_payload
    from reconcile_tiers.room_postprocessing.wall_segment_graph import (
        build_wall_segment_graph,
    )

    payload = _near_miss_three_wall_payload()
    elements = flatten_tier_payload(payload)
    corner_vids = cluster_element_corners(elements, 0.05)
    sg = build_wall_segment_graph(
        elements,
        corner_vids,
        0.05,
        0.05,
        leaf_bridge_gap=0.5,
    )
    bridge_edges = [e for e in sg["edges"] if e.get("kind") == "leaf_bridge"]
    assert bridge_edges

    orange_tip_grp = _group_for_segment(sg, "w-orange::vseg::1")
    assert orange_tip_grp is not None
    orange_bridges = [
        e
        for e in bridge_edges
        if orange_tip_grp["id"] in {e["source"], e["target"]}
    ]
    assert orange_bridges
    assert orange_tip_grp["degree"] >= 2


def _square_with_corner_gap_payload() -> dict[str, Any]:
    """2×2 m square with a near-miss gap at the NW corner closed by leaf bridge."""

    return {
        "uuid": "gap-square",
        "rooms": [
            {
                "story": 0,
                "walls": [
                    {
                        "locator_id": "w-s",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(2, 0, 0),
                            _pt(2, 2, 0),
                            _pt(0, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-e",
                        "corners": [
                            _pt(2, 0, 0),
                            _pt(2, 0, 2),
                            _pt(2, 2, 2),
                            _pt(2, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-n",
                        "corners": [
                            _pt(2, 0, 2),
                            _pt(0.2, 0, 2),
                            _pt(0.2, 2, 2),
                            _pt(2, 2, 2),
                        ],
                    },
                    {
                        "locator_id": "w-w",
                        "corners": [
                            _pt(0, 0, 1.8),
                            _pt(0, 0, 0),
                            _pt(0, 2, 0),
                            _pt(0, 2, 1.8),
                        ],
                    },
                ],
            }
        ],
    }


def test_near_miss_leaf_bridge_closes_room_cycle() -> None:
    """Tight approx groups + leaf bridge should close a segment room cycle."""

    payload = _square_with_corner_gap_payload()
    graph = build_corner_graph(
        payload,
        corner_tol=0.05,
        adjacency_tol=0.05,
        leaf_bridge_gap=0.5,
    )
    bridge_edges = [
        e
        for e in graph["wall_segment_graph"]["edges"]
        if e.get("kind") == "leaf_bridge"
    ]
    assert bridge_edges

    rooms = [
        n
        for n in graph["segment_room_graph"]["nodes"]
        if n["kind"] == "segment_room"
    ]
    assert len(rooms) == 1
    assert rooms[0]["area_m2"] >= 3.0

    bridged_groups = {
        gid
        for e in bridge_edges
        for gid in (e["source"], e["target"])
    }
    assert bridged_groups.issubset(set(rooms[0]["group_ids"]))
    for gid in bridged_groups:
        node = next(n for n in graph["wall_segment_graph"]["nodes"] if n["id"] == gid)
        assert node.get("in_room_cycle") is True
        assert node.get("orphan") is False


def test_near_miss_junction_segments_merge_into_one_approx_group() -> None:
    payload = _near_miss_three_wall_payload()
    graph = build_corner_graph(payload, corner_tol=0.05, adjacency_tol=0.5)
    sg = graph["wall_segment_graph"]
    orange_junction = "w-orange::vseg::1"
    green_junction = "w-green::vseg::1"
    segment_ids = {s["id"] for s in sg["segments"]}
    assert orange_junction in segment_ids
    assert green_junction in segment_ids

    orange_grp = _group_for_segment(sg, orange_junction)
    green_grp = _group_for_segment(sg, green_junction)
    assert orange_grp is not None
    assert green_grp is not None
    assert orange_grp["id"] == green_grp["id"]
    assert orange_grp["segment_count"] >= 2
    assert len(orange_grp["wall_ids"]) >= 2

    graph_tight = build_corner_graph(payload, corner_tol=0.05, adjacency_tol=0.05)
    sg_tight = graph_tight["wall_segment_graph"]
    tight_orange = _group_for_segment(sg_tight, orange_junction)
    tight_green = _group_for_segment(sg_tight, green_junction)
    assert tight_orange is not None
    assert tight_green is not None
    assert tight_orange["segment_count"] < orange_grp["segment_count"]
