"""Build JSON-serializable corner graph export from tier_payload."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reconcile_tiers.room_postprocessing.corner_graph import (
    DEFAULT_ADJACENCY_TOL_M,
    approx_element_adjacency_pairs,
    cluster_element_corners,
    element_adjacency_pairs,
    isolated_wall_edges,
    merge_adjacency_pairs,
)
from reconcile_tiers.room_postprocessing.flatten_payload import flatten_tier_payload
from reconcile_tiers.room_postprocessing.models import CornerGraphExport
from reconcile_tiers.room_postprocessing.wall_junction_split import (
    split_walls_at_approx_junctions,
)
from reconcile_tiers.room_postprocessing.segment_room_cycles import (
    build_segment_room_graph,
)
from reconcile_tiers.room_postprocessing.wall_segment_graph import build_wall_segment_graph


def _node_dict(el_index: int, el: Any, degree: int) -> dict[str, Any]:
    return {
        "id": el.id,
        "index": el_index,
        "kind": el.kind,
        "locator_id": el.locator_id,
        "room_index": el.room_index,
        "story": el.story,
        "corner_count": len(el.corners),
        "corners": [
            {"x": c[0], "y": c[1], "z": c[2]} for c in el.corners
        ],
        "degree": degree,
    }


def build_corner_graph(
    payload: Mapping[str, Any],
    *,
    corner_tol: float = 0.05,
    adjacency_tol: float = DEFAULT_ADJACENCY_TOL_M,
) -> dict[str, Any]:
    """Flatten tier_payload, cluster corners, and export element graph + wall edges."""

    building_uuid = str(payload.get("uuid") or "")
    elements = flatten_tier_payload(payload)
    elements = split_walls_at_approx_junctions(
        elements,
        corner_tol,
        adjacency_tol,
    )
    corner_vids = cluster_element_corners(elements, corner_tol)
    strict_pairs = element_adjacency_pairs(elements, corner_vids)
    approx_pairs = approx_element_adjacency_pairs(elements, adjacency_tol)
    pairs = merge_adjacency_pairs(strict_pairs, approx_pairs)
    wall_segments = isolated_wall_edges(elements, corner_vids)

    degree: dict[int, int] = {i: 0 for i in range(len(elements))}
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1

    nodes = [
        _node_dict(i, el, degree.get(i, 0))
        for i, el in enumerate(elements)
    ]
    edges = [
        {
            "source": elements[a].id,
            "target": elements[b].id,
            "source_index": a,
            "target_index": b,
        }
        for a, b in pairs
    ]

    wall_indices = {i for i, el in enumerate(elements) if el.kind == "wall"}
    wall_pairs = [(a, b) for a, b in pairs if a in wall_indices and b in wall_indices]
    wall_degree: dict[int, int] = {i: 0 for i in wall_indices}
    for a, b in wall_pairs:
        wall_degree[a] += 1
        wall_degree[b] += 1
    wall_nodes = [
        _node_dict(i, elements[i], wall_degree.get(i, 0))
        for i in sorted(wall_indices)
    ]
    wall_edges = [
        {
            "source": elements[a].id,
            "target": elements[b].id,
            "source_index": a,
            "target_index": b,
        }
        for a, b in wall_pairs
    ]

    wall_segment_graph = build_wall_segment_graph(
        elements,
        corner_vids,
        corner_tol,
        adjacency_tol,
    )
    segment_room_graph = build_segment_room_graph(
        wall_segment_graph,
        corner_tol=corner_tol,
    )

    export = CornerGraphExport(
        building_uuid=building_uuid,
        corner_tol=corner_tol,
        nodes=nodes,
        edges=edges,
        wall_edge_segments=wall_segments,
    )
    return {
        "building_uuid": export.building_uuid,
        "corner_tol": export.corner_tol,
        "adjacency_tol": adjacency_tol,
        "element_count": len(elements),
        "nodes": export.nodes,
        "edges": export.edges,
        "wall_graph": {
            "nodes": wall_nodes,
            "edges": wall_edges,
        },
        "wall_segment_graph": wall_segment_graph,
        "segment_room_graph": segment_room_graph,
        "wall_edge_segments": export.wall_edge_segments,
    }
