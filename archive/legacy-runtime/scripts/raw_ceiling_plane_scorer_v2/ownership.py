from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.ops import unary_union

from .geometry import xz_polygon
from .models import RelationGraph


def _room_membership_map(roof_result: dict[str, Any]) -> dict[str, set[str]]:
    graph = roof_result.get("building_part_graph") or {}
    room_membership = graph.get("room_membership") or {}
    return {
        str(room_id): {str(part_id) for part_id in (part_ids or []) if part_id}
        for room_id, part_ids in room_membership.items()
    }


def _room_polys(building: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for room_index, room in enumerate(building.get("rooms") or []):
        poly = xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        result[f"room:{room_index}"] = poly
    return result


def _room_polys_by_story(building: dict[str, Any]) -> dict[int, dict[str, Any]]:
    by_story: dict[int, dict[str, Any]] = defaultdict(dict)
    for room_index, room in enumerate(building.get("rooms") or []):
        poly = xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        story = int(room.get("story", 0))
        by_story[story][f"room:{room_index}"] = poly
    return by_story


def _parts_for_poly(
    poly: Any, room_polys: dict[str, Any], room_membership: dict[str, set[str]]
) -> set[str]:
    parts: set[str] = set()
    if poly is None or poly.is_empty:
        return parts
    for room_id, room_poly in room_polys.items():
        try:
            overlap = poly.intersection(room_poly)
        except Exception:
            continue
        if overlap.is_empty or float(overlap.area) <= 1e-9:
            continue
        parts.update(room_membership.get(room_id, set()))
    return parts


def _source_part_overlap_fraction(
    poly: Any,
    source_part_ids: set[str],
    room_polys: dict[str, Any],
    room_membership: dict[str, set[str]],
) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    if not source_part_ids:
        return 0.0
    source_room_polys = [
        room_poly
        for room_id, room_poly in room_polys.items()
        if room_membership.get(room_id, set()) & source_part_ids
    ]
    if not source_room_polys:
        return 0.0
    try:
        source_union = unary_union(source_room_polys)
        overlap_area = float(poly.intersection(source_union).area)
    except Exception:
        return 0.0
    return overlap_area / max(float(poly.area), 1e-9)


def annotate_split_rows_with_ownership(
    split_piece_rows: list[dict[str, Any]],
    graph: RelationGraph,
    building: dict[str, Any],
    roof_result: dict[str, Any],
) -> list[dict[str, Any]]:
    relations_by_target = defaultdict(list)
    for relation in graph.relations:
        relations_by_target[relation.a_target_id].append(relation)
        relations_by_target[relation.b_target_id].append(relation)

    room_membership = _room_membership_map(roof_result)
    room_polys_all = _room_polys(building)
    room_polys_by_story = _room_polys_by_story(building)

    out: list[dict[str, Any]] = []
    for row in split_piece_rows:
        annotated = dict(row)
        target_id = str(row.get("target_element_id") or "")
        context = graph.contexts_by_target.get(target_id)

        covering_target_ids: list[str] = []
        covering_part_ids: set[str] = set()
        local_competitor_target_ids: list[str] = []
        mirror_partner_target_ids: list[str] = []

        for relation in relations_by_target.get(target_id, []):
            other_target_id = (
                relation.b_target_id
                if relation.a_target_id == target_id
                else relation.a_target_id
            )
            if (
                relation.relation_kind == "covering"
                and relation.covered_target_id == target_id
            ):
                covering_target_ids.append(other_target_id)
                other_context = graph.contexts_by_target.get(other_target_id)
                if other_context is not None:
                    covering_part_ids.update(other_context.building_part_ids)
            elif relation.relation_kind == "local_competitor":
                local_competitor_target_ids.append(other_target_id)
            elif relation.relation_kind == "mirror_pair":
                mirror_partner_target_ids.append(other_target_id)

        piece_poly = xz_polygon(row.get("corners") or [])
        piece_story = int(row.get("story") or 0)
        piece_part_ids_story = _parts_for_poly(
            piece_poly,
            room_polys_by_story.get(piece_story, {}),
            room_membership,
        )
        piece_part_ids_all = _parts_for_poly(
            piece_poly, room_polys_all, room_membership
        )
        piece_part_ids = piece_part_ids_all or piece_part_ids_story
        source_part_ids = set(context.source_part_ids) if context is not None else set()
        source_part_overlap_fraction = _source_part_overlap_fraction(
            piece_poly,
            source_part_ids,
            room_polys_by_story.get(piece_story, {}),
            room_membership,
        )
        if source_part_overlap_fraction <= 0.0:
            source_part_overlap_fraction = _source_part_overlap_fraction(
                piece_poly,
                source_part_ids,
                room_polys_all,
                room_membership,
            )

        annotated["piece_part_ids"] = sorted(piece_part_ids)
        annotated["piece_part_ids_story"] = sorted(piece_part_ids_story)
        annotated["covering_target_ids"] = sorted(set(covering_target_ids))
        annotated["covering_part_ids"] = sorted(covering_part_ids)
        annotated["local_competitor_target_ids"] = sorted(
            set(local_competitor_target_ids)
        )
        annotated["mirror_partner_target_ids"] = sorted(set(mirror_partner_target_ids))
        annotated["ownership_redundant"] = bool(covering_target_ids)
        annotated["source_part_overlap_fraction"] = round(
            source_part_overlap_fraction, 6
        )
        out.append(annotated)

    return out
