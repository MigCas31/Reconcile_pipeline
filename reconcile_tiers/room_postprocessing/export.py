"""Build JSON-serializable corner graph export from tier_payload."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reconcile_tiers.room_postprocessing.corner_graph import (
    cluster_element_corners,
    element_adjacency_pairs,
    isolated_wall_edges,
)
from reconcile_tiers.room_postprocessing.flatten_payload import flatten_tier_payload
from reconcile_tiers.room_postprocessing.models import CornerGraphExport


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
) -> dict[str, Any]:
    """Flatten tier_payload, cluster corners, and export element graph + wall edges."""

    building_uuid = str(payload.get("uuid") or "")
    elements = flatten_tier_payload(payload)
    corner_vids = cluster_element_corners(elements, corner_tol)
    pairs = element_adjacency_pairs(elements, corner_vids)
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
        "element_count": len(elements),
        "nodes": export.nodes,
        "edges": export.edges,
        "wall_edge_segments": export.wall_edge_segments,
    }
