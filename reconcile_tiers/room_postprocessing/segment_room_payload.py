"""Build tier_payload-shaped JSON from segment-room cycles for polyhedron pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reconcile_tiers.room_postprocessing.corner_graph import (
    DEFAULT_ADJACENCY_TOL_M,
    cluster_element_corners,
)
from reconcile_tiers.room_postprocessing.flatten_payload import flatten_tier_payload
from reconcile_tiers.room_postprocessing.models import BuildingElement
from reconcile_tiers.room_postprocessing.room_floor_clip import attach_room_floor_polygons
from reconcile_tiers.room_postprocessing.room_shell_closure import (
    build_half_closed_room_shell,
)
from reconcile_tiers.room_postprocessing.segment_room_cycles import (
    build_segment_room_graph,
)
from reconcile_tiers.room_postprocessing.wall_junction_split import (
    split_walls_at_approx_junctions,
)
from reconcile_tiers.room_postprocessing.wall_near_segment_split import (
    split_walls_at_near_segments,
)
from reconcile_tiers.room_postprocessing.wall_segment_graph import build_wall_segment_graph

HALF_CLOSED_SHELL = "half_closed_floor_walls"


def _postprocess_elements(
    payload: Mapping[str, Any],
    *,
    corner_tol: float,
    adjacency_tol: float,
) -> tuple[list[BuildingElement], dict[str, Any], dict[str, Any]]:
    elements = flatten_tier_payload(payload)
    elements = split_walls_at_approx_junctions(
        elements,
        corner_tol,
        adjacency_tol,
    )
    elements = split_walls_at_near_segments(
        elements,
        corner_tol,
        adjacency_tol,
    )
    corner_vids = cluster_element_corners(elements, corner_tol)
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
    attach_room_floor_polygons(segment_room_graph, elements)
    return elements, segment_room_graph, wall_segment_graph


def _wall_element_for_id(
    wall_id: str,
    wall_by_id: Mapping[str, BuildingElement],
) -> BuildingElement | None:
    el = wall_by_id.get(wall_id)
    if el is not None:
        return el
    prefix = f"{wall_id}::split::"
    for key, candidate in wall_by_id.items():
        if key.startswith(prefix):
            return candidate
    return None


def _floor_y_for_room(
    wall_ids: Sequence[str],
    wall_by_id: Mapping[str, BuildingElement],
    story: int | None,
    elements: Sequence[BuildingElement],
) -> float | None:
    ys: list[float] = []
    for wid in wall_ids:
        el = _wall_element_for_id(wid, wall_by_id)
        if el is None:
            continue
        ys.extend(c[1] for c in el.corners)
    if ys:
        return min(ys)
    for el in elements:
        if el.kind != "floor" or el.story != story:
            continue
        ys.extend(c[1] for c in el.corners)
    return min(ys) if ys else None


def _segment_room_to_tier_room(
    seg_room: Mapping[str, Any],
    *,
    room_index: int,
    wall_by_id: Mapping[str, BuildingElement],
    elements: Sequence[BuildingElement],
    segments_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any] | None:
    wall_ids = list(dict.fromkeys(seg_room.get("wall_ids") or []))
    if len(wall_ids) < 3:
        return None

    poly_xz = seg_room.get("polygon_xz")
    if not poly_xz or len(poly_xz) < 3:
        return None

    story_raw = seg_room.get("story")
    story = int(story_raw) if story_raw is not None else 0
    floor_y = _floor_y_for_room(wall_ids, wall_by_id, story_raw, elements)
    if floor_y is None:
        return None

    shell = build_half_closed_room_shell(
        seg_room,
        segments_by_id,
        floor_y=floor_y,
    )
    if shell is None:
        return None

    room_id = str(seg_room.get("id") or f"room_cycle::{story}::{room_index}")
    return {
        "story": story,
        "locator_id": room_id,
        "room_index": room_index,
        "floor": [
            {
                "locator_id": f"{room_id}::floor",
                "corners": shell["floor_corners"],
            }
        ],
        "walls": shell["walls"],
        "doors": [],
        "windows": [],
    }


def build_segment_room_tier_payload(
    payload: Mapping[str, Any],
    *,
    corner_tol: float = 0.05,
    adjacency_tol: float = DEFAULT_ADJACENCY_TOL_M,
) -> dict[str, Any]:
    """Return a tier_payload copy whose ``rooms`` are segment-room cycles.

    Each room uses a half-closed floor+wall shell (shared junction corners).
    Building-level ``ceiling``, ``visual_shells``, and ``gable_closures`` are
    preserved from the input for polyhedron tile collection.
    """

    elements, segment_room_graph, wall_segment_graph = _postprocess_elements(
        payload,
        corner_tol=corner_tol,
        adjacency_tol=adjacency_tol,
    )
    wall_by_id = {el.id: el for el in elements if el.kind == "wall"}
    segments_by_id = {
        s["id"]: s for s in (wall_segment_graph.get("segments") or [])
    }

    tier_rooms: list[dict[str, Any]] = []
    for node in segment_room_graph.get("nodes") or []:
        if node.get("kind") != "segment_room":
            continue
        tier_room = _segment_room_to_tier_room(
            node,
            room_index=len(tier_rooms),
            wall_by_id=wall_by_id,
            elements=elements,
            segments_by_id=segments_by_id,
        )
        if tier_room is not None:
            tier_rooms.append(tier_room)

    out: dict[str, Any] = dict(payload)
    out["rooms"] = tier_rooms
    out["room_postprocessing_source"] = {
        "corner_tol": corner_tol,
        "adjacency_tol": adjacency_tol,
        "segment_room_count": len(tier_rooms),
        "shell": HALF_CLOSED_SHELL,
    }
    out["segment_room_graph"] = segment_room_graph
    return out
