"""Abstract graph-driven decision logic for migration away from local heuristics.

These functions intentionally describe *why* a decision should be taken in
terms of graph semantics rather than local geometry thresholds. Geometry still
matters, but it becomes evidence attached to graph relations instead of the
primary rule engine.
"""

from __future__ import annotations

from typing import Any

from .models import GraphNode, TopologyGraph


def _relation_state_from_edges(edges: list) -> str:
    if not edges:
        return "unknown"
    states = [
        str((edge.evidence or {}).get("relation_state", "confirmed")) for edge in edges
    ]
    if "confirmed" in states:
        return "confirmed"
    if "partial" in states:
        return "partial"
    return states[0]


def classify_room_relation_state(
    graph: TopologyGraph, room: GraphNode
) -> dict[str, Any]:
    above_edges = graph.edges_to(room.id, "ABOVE") + graph.edges_from(room.id, "BELOW")
    exposure_edges = graph.edges_from(room.id, "EXPOSES_TO")
    adjacency_edges = [
        edge
        for edge in graph.edges_from(room.id, "ADJACENT_TO")
        if edge.to_id.startswith("room:")
    ]
    return {
        "room_id": room.id,
        "above_state": _relation_state_from_edges(above_edges),
        "exposure_state": _relation_state_from_edges(exposure_edges),
        "adjacency_state": _relation_state_from_edges(adjacency_edges),
        "above_edges": above_edges,
        "exposure_edges": exposure_edges,
        "adjacency_edges": adjacency_edges,
    }


def _room_exposure_class(graph: TopologyGraph, room_id: str) -> str:
    room = graph.get_node(room_id)
    if room is None:
        return "unknown"
    relation = classify_room_relation_state(graph, room)
    if relation["exposure_state"] == "confirmed":
        return "external"
    if relation["above_state"] == "confirmed":
        return "stacked_internal"
    if relation["exposure_state"] == "partial" or relation["above_state"] == "partial":
        return "partial"
    return "internal"


def classify_gap_decision(graph: TopologyGraph, gap: GraphNode) -> dict[str, Any]:
    if bool((gap.properties or {}).get("derived_enclosed_void")):
        return {
            "action": "close",
            "policy": "internal_enclosed_void_repair",
            "reason": "derived_enclosed_void_from_cell_decomposition",
            "room_ids": [
                node.id
                for node in graph.neighbors(gap.id, "BOUNDED_BY")
                if node.type == "Room"
            ],
            "exposure_classes": ["internal"],
        }

    rooms = [
        node for node in graph.neighbors(gap.id, "BOUNDED_BY") if node.type == "Room"
    ]
    exposure_classes = sorted({_room_exposure_class(graph, room.id) for room in rooms})
    relation_profiles = [classify_room_relation_state(graph, room) for room in rooms]
    gap_kind = (gap.properties or {}).get("gap_kind")

    if gap_kind == "cross_story":
        if any(
            profile["above_state"] in {"partial", "unknown"}
            for profile in relation_profiles
        ):
            return {
                "action": "inspect",
                "policy": "partial_vertical_relation",
                "reason": "cross_story_gap_with_incomplete_vertical_support_relation",
                "room_ids": [room.id for room in rooms],
                "exposure_classes": exposure_classes,
            }
        return {
            "action": "close",
            "policy": "vertical_volume_repair",
            "reason": "cross_story_gap_between_occupied_volumes",
            "room_ids": [room.id for room in rooms],
            "exposure_classes": exposure_classes,
        }
    if not rooms:
        return {
            "action": "inspect",
            "policy": "unresolved_gap",
            "reason": "gap_not_bounded_by_rooms",
            "room_ids": [],
            "exposure_classes": exposure_classes,
        }
    if "external" in exposure_classes:
        return {
            "action": "inspect",
            "policy": "possible_intentional_opening",
            "reason": "gap_touches_exterior_exposed_room",
            "room_ids": [room.id for room in rooms],
            "exposure_classes": exposure_classes,
        }
    if any(
        profile["above_state"] in {"partial", "unknown"}
        or profile["exposure_state"] in {"partial", "unknown"}
        for profile in relation_profiles
    ):
        return {
            "action": "inspect",
            "policy": "partial_spatial_relation",
            "reason": "gap_bounded_by_rooms_with_partial_or_unknown_relations",
            "room_ids": [room.id for room in rooms],
            "exposure_classes": exposure_classes,
        }
    return {
        "action": "close",
        "policy": "internal_partition_repair",
        "reason": "gap_between_interior_rooms",
        "room_ids": [room.id for room in rooms],
        "exposure_classes": exposure_classes,
    }


def classify_stitch_decision(
    graph: TopologyGraph, left: GraphNode, right: GraphNode, evidence: dict[str, Any]
) -> dict[str, Any]:
    left_role = (left.properties or {}).get("surface_role", "unknown")
    right_role = (right.properties or {}).get("surface_role", "unknown")
    junction_type = evidence.get("junction_type", "BUTT")

    if left_role == "exterior_wall" or right_role == "exterior_wall":
        action = "stitch"
        policy = "weather_barrier_continuity"
    elif junction_type in {"L", "T", "CROSS"}:
        action = "stitch"
        policy = "junction_continuity"
    else:
        action = "inspect"
        policy = "ambiguous_internal_connection"

    return {
        "action": action,
        "policy": policy,
        "reason": f"{left_role}_to_{right_role}_{junction_type.lower()}",
        "from_surface": left.id,
        "to_surface": right.id,
        "junction_type": junction_type,
    }


def classify_roof_room(graph: TopologyGraph, room: GraphNode) -> dict[str, Any]:
    is_top_story = bool((room.properties or {}).get("is_top_story"))
    relation = classify_room_relation_state(graph, room)
    above_state = relation["above_state"]
    exposure_state = relation["exposure_state"]

    if above_state == "confirmed":
        mode = "ceiling_below_occupied_volume"
        reason = "another_room_above"
    elif exposure_state == "confirmed":
        mode = "roof_candidate"
        reason = "occupied_space_exposed_upwards"
    elif is_top_story:
        mode = "roof_candidate"
        reason = "top_story_without_above_relation"
    elif above_state == "partial" or exposure_state == "partial":
        mode = "underdetermined"
        reason = "partial_relations_require_geometry_fallback"
    else:
        mode = "underdetermined"
        reason = "partial_relations_require_geometry_fallback"

    return {
        "room_id": room.id,
        "mode": mode,
        "reason": reason,
        "is_top_story": is_top_story,
        "has_above": above_state == "confirmed",
        "exposed": exposure_state == "confirmed",
        "above_state": above_state,
        "exposure_state": exposure_state,
    }


def classify_overlap_resolution(
    graph: TopologyGraph, room: GraphNode
) -> dict[str, Any]:
    parent = graph.parent(room.id)
    area = float((room.properties or {}).get("area_m2", 0.0))
    adjacent_count = int((room.properties or {}).get("adjacent_room_count", 0))
    exposure = _room_exposure_class(graph, room.id)

    return {
        "room_id": room.id,
        "story_id": parent.id if parent else None,
        "ownership_priority": (
            "preserve" if exposure == "external" or adjacent_count > 1 else "secondary"
        ),
        "reason": f"{exposure}_adjacency_{adjacent_count}_area_{round(area, 3)}",
    }


def build_decision_snapshot(graph: TopologyGraph) -> dict[str, Any]:
    return {
        "gaps": {
            gap.id: classify_gap_decision(graph, gap)
            for gap in graph.nodes_by_type("Gap")
        },
        "roof_rooms": [
            classify_roof_room(graph, room) for room in graph.nodes_by_type("Room")
        ],
        "relation_states": [
            classify_room_relation_state(graph, room)
            for room in graph.nodes_by_type("Room")
        ],
        "overlap_resolution": [
            classify_overlap_resolution(graph, room)
            for room in graph.nodes_by_type("Room")
        ],
    }
