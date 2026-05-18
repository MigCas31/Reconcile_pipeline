from __future__ import annotations

from collections import defaultdict
from typing import Any

from scripts import prototype_raw_ceiling_plane_scorer as legacy

from .geometry import xz_polygon
from .models import PlaneContext


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) * 0.5)


def _room_membership_map(roof_result: dict[str, Any]) -> dict[str, set[str]]:
    graph = roof_result.get("building_part_graph") or {}
    room_membership = graph.get("room_membership") or {}
    return {
        str(room_id): {str(part_id) for part_id in (part_ids or []) if part_id}
        for room_id, part_ids in room_membership.items()
    }


def _main_part_ids(room_membership: dict[str, set[str]]) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for parts in room_membership.values():
        for part_id in parts:
            counts[part_id] += 1
    if not counts:
        return set()
    max_count = max(counts.values())
    candidates = sorted(
        part_id for part_id, count in counts.items() if count == max_count
    )
    return {candidates[0]} if candidates else set()


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


def _target_base_part_ids(
    target: legacy.TargetPlaneRecord,
    room_polys_story: dict[str, Any],
    room_polys_all: dict[str, Any],
    room_membership: dict[str, set[str]],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
    committed_hypothesis_meta: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    building_part_ids: set[str] = set()
    source_part_ids: set[str] = set()

    if target.target_kind == "committed_oblique":
        meta = committed_hypothesis_meta.get(target.element_id) or {}
        for part_id in meta.get("hypothesis_part_ids") or []:
            building_part_ids.add(str(part_id))
        source_part_ids.update(building_part_ids)
    elif target.target_kind == "ridge_eave_plane_group":
        diag = ridge_eave_target_diagnostics.get(target.element_id) or {}
        source_rooms = [
            str(room_id)
            for room_id in (diag.get("creator_source_room_ids") or [])
            if str(room_id)
        ]
        for room_id in source_rooms:
            source_part_ids.update(room_membership.get(room_id, set()))
        building_part_ids.update(source_part_ids)
    else:
        building_part_ids.update(
            _parts_for_poly(target.poly_xz, room_polys_story, room_membership)
        )
        source_part_ids.update(building_part_ids)

    if not building_part_ids:
        building_part_ids.update(
            _parts_for_poly(target.poly_xz, room_polys_all, room_membership)
        )
    if not source_part_ids:
        source_part_ids.update(building_part_ids)

    return building_part_ids, source_part_ids


def build_plane_contexts(
    targets: list[legacy.TargetPlaneRecord],
    plane_chain_supports: list[legacy.PlaneEaveChainSupportRecord],
    split_piece_rows: list[dict[str, Any]],
    building: dict[str, Any],
    roof_result: dict[str, Any],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, PlaneContext]:
    room_membership = _room_membership_map(roof_result)
    main_part_ids = _main_part_ids(room_membership)
    room_polys_all = _room_polys(building)
    room_polys_by_story = _room_polys_by_story(building)

    committed_targets = [
        target for target in targets if target.target_kind == "committed_oblique"
    ]
    committed_hypothesis_meta = legacy._committed_target_hypothesis_metadata(
        committed_targets,
        building,
        roof_result,
    )

    supports_by_target: dict[str, list[legacy.PlaneEaveChainSupportRecord]] = (
        defaultdict(list)
    )
    for support in plane_chain_supports:
        if not support.supported:
            continue
        supports_by_target[support.target_element_id].append(support)

    piece_rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_piece_rows:
        target_id = str(row.get("target_element_id") or "")
        if target_id:
            piece_rows_by_target[target_id].append(row)

    contexts: dict[str, PlaneContext] = {}
    for target in targets:
        supports = supports_by_target.get(target.element_id, [])
        source_edge_ids = tuple(
            sorted({str(support.chain_id) for support in supports if support.chain_id})
        )
        source_edge_overlap_m = float(
            sum(float(support.chain_length_m) for support in supports)
        )
        source_edge_alignment_deg = _median(
            [float(support.angle_delta_deg) for support in supports]
        )
        source_edge_height_residual_m = _median(
            [
                abs(float(support.height_residual_m))
                for support in supports
                if support.height_residual_m is not None
            ]
        )

        building_part_ids, source_part_ids = _target_base_part_ids(
            target,
            room_polys_by_story.get(int(target.story), {}),
            room_polys_all,
            room_membership,
            ridge_eave_target_diagnostics,
            committed_hypothesis_meta,
        )

        crossed_part_ids: set[str] = set()
        for row in piece_rows_by_target.get(target.element_id, []):
            piece_poly = xz_polygon(row.get("corners") or [])
            if piece_poly is None or piece_poly.is_empty:
                continue
            piece_story = int(
                row.get("story") if row.get("story") is not None else target.story
            )
            crossed_part_ids.update(
                _parts_for_poly(
                    piece_poly,
                    room_polys_by_story.get(piece_story, {}),
                    room_membership,
                )
            )
        if not crossed_part_ids:
            crossed_part_ids.update(
                _parts_for_poly(
                    target.poly_xz,
                    room_polys_by_story.get(int(target.story), {}),
                    room_membership,
                )
            )
        if not crossed_part_ids:
            crossed_part_ids.update(
                _parts_for_poly(target.poly_xz, room_polys_all, room_membership)
            )

        crossed_part_ids_for_flags = set(crossed_part_ids)
        crossed_part_ids_for_flags.update(
            _parts_for_poly(target.poly_xz, room_polys_all, room_membership)
        )

        in_main_body = bool(
            (building_part_ids or crossed_part_ids_for_flags) & main_part_ids
        )
        in_extension = bool(
            {*building_part_ids, *crossed_part_ids_for_flags} - main_part_ids
        )
        crosses_boundary = (in_main_body and in_extension) or (
            bool(crossed_part_ids_for_flags & main_part_ids)
            and bool(crossed_part_ids_for_flags - main_part_ids)
        )

        contexts[target.element_id] = PlaneContext(
            target_id=target.element_id,
            story=int(target.story),
            source_edge_ids=source_edge_ids,
            source_edge_overlap_m=source_edge_overlap_m,
            source_edge_alignment_deg=source_edge_alignment_deg,
            source_edge_height_residual_m=source_edge_height_residual_m,
            building_part_ids=tuple(sorted(building_part_ids)),
            source_part_ids=tuple(sorted(source_part_ids)),
            crossed_part_ids=tuple(sorted(crossed_part_ids)),
            in_main_body=in_main_body,
            in_extension=in_extension,
            crosses_main_extension_boundary=crosses_boundary,
        )

    return contexts
