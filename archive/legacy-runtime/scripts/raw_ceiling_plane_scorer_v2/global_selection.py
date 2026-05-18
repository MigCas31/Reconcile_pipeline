from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

from .config import GlobalSelectionConfig
from .geometry import iter_polygons, stored_row_plane_coeffs

_ENV_SMALL_HOLE_MAX_AREA_M2 = 1.0

# Ridge_eave pieces with these reasons are pre-suppressed by the XY conflict step
# in layer_policy but should still enter the ILP — resolving their conflict with
# oblique pieces is exactly what the ILP is designed to do.
_RIDGE_ILP_ELIGIBLE_REASONS: frozenset[str] = frozenset(
    {
        "ridge_eave_relation_owner",
        "ridge_eave_disjoint_extension_owner",
        "ridge_eave_through_building_owner",
        "ridge_eave_post_clip_rescued_for_coverage",
    }
)


@dataclass(frozen=True)
class _PieceEntry:
    row_index: int
    story: int
    piece_id: str
    piece_index: int
    poly: Polygon
    support_score: float
    prior_final: float
    is_dormer_exception: bool
    outside_soft_fraction: float
    hard_envelope: Any | None
    soft_envelope: Any | None
    plane_group_id: str
    part_ids: tuple[str, ...]


@dataclass(frozen=True)
class _CellEntry:
    poly: Polygon
    area: float
    center_x: float
    center_z: float
    covering_piece_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _GroupEntry:
    group_id: str
    poly: Any
    support_score: float
    prior_final: float
    is_dormer_exception: bool
    hard_envelope: Any | None
    soft_envelope: Any | None
    representative_piece_index: int
    member_piece_indexes: tuple[int, ...]


def _hard_invalid_pre_ilp_reason(
    row: dict[str, Any],
    cfg: GlobalSelectionConfig,
) -> str | None:
    pre_reason = str(row.get("final_layer_reason") or "")
    if pre_reason in cfg.hard_invalid_pre_reasons:
        return pre_reason
    if str(row.get("target_kind") or "") == "ridge_eave_plane_group":
        if pre_reason not in cfg.ridge_allowed_pre_reasons:
            return f"ridge_pre_reason:{pre_reason or 'missing'}"
    return None


def _piece_index_value(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return 0


def _row_xz_polygon(row: dict[str, Any]) -> Polygon | None:
    corners = row.get("corners") or []
    shell = [(float(point[0]), float(point[2])) for point in corners if len(point) >= 3]
    if len(shell) < 3:
        return None
    hole_rings: list[list[tuple[float, float]]] = []
    for hole in row.get("holes") or []:
        ring = [(float(point[0]), float(point[2])) for point in hole if len(point) >= 3]
        if len(ring) >= 3:
            hole_rings.append(ring)

    poly = Polygon(shell, holes=hole_rings or None)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _safe_union(polys: list[Any]) -> Any | None:
    valid = [
        poly
        for poly in polys
        if poly is not None and not getattr(poly, "is_empty", True)
    ]
    if not valid:
        return None
    try:
        geom = unary_union(valid)
    except Exception:
        return None
    if getattr(geom, "is_empty", True):
        return None
    return geom


def _fill_small_holes(
    geom: Any, max_hole_area_m2: float = _ENV_SMALL_HOLE_MAX_AREA_M2
) -> Any:
    if geom is None or getattr(geom, "is_empty", True) or max_hole_area_m2 <= 0.0:
        return geom

    cleaned_polys: list[Polygon] = []
    for poly in iter_polygons(geom):
        hole_rings: list[list[tuple[float, float]]] = []
        for ring in poly.interiors:
            ring_coords = list(ring.coords)
            if len(ring_coords) < 4:
                continue
            try:
                hole_poly = Polygon(ring_coords)
            except Exception:
                continue
            if hole_poly.is_empty:
                continue
            if float(hole_poly.area) > max_hole_area_m2:
                hole_rings.append(ring_coords)
        try:
            cleaned = Polygon(list(poly.exterior.coords), holes=hole_rings or None)
        except Exception:
            cleaned = poly
        if not cleaned.is_valid:
            try:
                cleaned = cleaned.buffer(0)
            except Exception:
                cleaned = poly
        if cleaned.is_empty:
            cleaned = poly
        if isinstance(cleaned, Polygon):
            cleaned_polys.append(cleaned)
    if not cleaned_polys:
        return geom
    merged = _safe_union(cleaned_polys)
    return merged if merged is not None else geom


def _room_membership_map(roof_result: dict[str, Any] | None) -> dict[str, set[str]]:
    graph = (roof_result or {}).get("building_part_graph") or {}
    room_membership = graph.get("room_membership") or {}
    return {
        str(room_id): {str(part_id) for part_id in (part_ids or []) if part_id}
        for room_id, part_ids in room_membership.items()
    }


def _exposed_room_keys(
    roof_result: dict[str, Any] | None,
) -> set[tuple[int, int]] | None:
    exposed = ((roof_result or {}).get("ceiling") or {}).get("exposed_rooms")
    if not exposed:
        return None
    keys: set[tuple[int, int]] = set()
    for entry in exposed:
        room_index = entry.get("room_index")
        story = entry.get("story")
        if room_index is None or story is None:
            continue
        keys.add((int(story), int(room_index)))
    return keys


def _rain_story_envelopes(
    building: dict[str, Any],
    roof_result: dict[str, Any] | None,
    story_gap_polygons: dict[int, list[Any]] | None,
) -> tuple[dict[int, Any], dict[tuple[int, str], Any]]:
    exposed_keys = _exposed_room_keys(roof_result)
    room_membership = _room_membership_map(roof_result)

    by_story: dict[int, list[Any]] = defaultdict(list)
    by_story_part: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for room_index, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        room_key = (story, room_index)
        if exposed_keys is not None and room_key not in exposed_keys:
            continue
        floor_polygon = room.get("floor_polygon") or []
        shell = [
            (float(point[0]), float(point[2]))
            for point in floor_polygon
            if isinstance(point, (list, tuple)) and len(point) >= 3
        ]
        if len(shell) < 3:
            continue
        poly = Polygon(shell)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid:
            continue
        by_story[story].append(poly)
        room_id = str(room.get("id") or f"room:{room_index}")
        part_ids = (
            room_membership.get(room_id)
            or room_membership.get(f"room:{room_index}")
            or set()
        )
        for part_id in part_ids:
            by_story_part[(story, str(part_id))].append(poly)

    story_env = {}
    for story, polys in by_story.items():
        merged = _safe_union(polys)
        if merged is None:
            continue
        story_env[story] = _fill_small_holes(merged)

    story_part_env = {}
    for key, polys in by_story_part.items():
        merged = _safe_union(polys)
        if merged is None:
            continue
        story_part_env[key] = _fill_small_holes(merged)

    gap_by_story = story_gap_polygons or {}
    for story, gaps in gap_by_story.items():
        rain_geom = story_env.get(int(story))
        if rain_geom is None:
            continue
        relevant_gap_polys: list[Any] = []
        for gap in gaps or []:
            if gap is None or getattr(gap, "is_empty", True):
                continue
            try:
                if not gap.intersection(rain_geom.buffer(0.5)).is_empty:
                    relevant_gap_polys.append(gap)
            except Exception:
                continue
        if not relevant_gap_polys:
            continue
        story_env[int(story)] = _fill_small_holes(
            _safe_union([rain_geom, *relevant_gap_polys]) or rain_geom
        )
        for key, part_geom in list(story_part_env.items()):
            key_story, _part_id = key
            if int(key_story) != int(story):
                continue
            near_part: list[Any] = []
            for gap in relevant_gap_polys:
                try:
                    if not gap.intersection(part_geom.buffer(0.5)).is_empty:
                        near_part.append(gap)
                except Exception:
                    continue
            if near_part:
                story_part_env[key] = _fill_small_holes(
                    _safe_union([part_geom, *near_part]) or part_geom
                )
    return story_env, story_part_env


def _piece_story_part_ids(row: dict[str, Any]) -> tuple[str, ...]:
    values = row.get("piece_part_ids") or row.get("source_part_ids") or []
    part_ids = sorted({str(value) for value in values if str(value)})
    return tuple(part_ids)


def _piece_envelopes(
    story: int,
    part_ids: tuple[str, ...],
    story_env: dict[int, Any],
    story_part_env: dict[tuple[int, str], Any],
    cfg: GlobalSelectionConfig,
) -> tuple[Any | None, Any | None]:
    part_envelopes = [
        story_part_env.get((int(story), part_id))
        for part_id in part_ids
        if story_part_env.get((int(story), part_id)) is not None
    ]
    base_envelope = (
        _safe_union(part_envelopes) if part_envelopes else story_env.get(int(story))
    )
    if base_envelope is None:
        return None, None
    try:
        hard_envelope = base_envelope.buffer(cfg.envelope_hard_buffer_m)
        soft_envelope = base_envelope.buffer(cfg.envelope_soft_buffer_m)
    except Exception:
        return None, None
    return hard_envelope, soft_envelope


def _target_family_group_ids(
    rows: list[dict[str, Any]],
    relation_graph: Any | None,
    include_target_kinds: tuple[str, ...],
) -> dict[str, str]:
    target_kind_by_id: dict[str, str] = {}
    for row in rows:
        target_id = str(row.get("target_element_id") or "")
        if not target_id:
            continue
        target_kind = str(row.get("target_kind") or "")
        if target_kind not in include_target_kinds:
            continue
        target_kind_by_id[target_id] = target_kind
    if not target_kind_by_id:
        return {}

    parent: dict[str, str] = {target_id: target_id for target_id in target_kind_by_id}

    def _find(target_id: str) -> str:
        root = parent[target_id]
        while root != parent[root]:
            parent[root] = parent[parent[root]]
            root = parent[root]
        node = target_id
        while node != root:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def _union(a: str, b: str) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for relation in getattr(relation_graph, "relations", []) or []:
        relation_kind = str(getattr(relation, "relation_kind", ""))
        if relation_kind not in {"same_face", "continuous_run_neighbor"}:
            continue
        a = str(getattr(relation, "a_target_id", ""))
        b = str(getattr(relation, "b_target_id", ""))
        if a not in target_kind_by_id or b not in target_kind_by_id:
            continue
        if target_kind_by_id[a] != target_kind_by_id[b]:
            continue
        _union(a, b)

    roots = {target_id: _find(target_id) for target_id in target_kind_by_id}
    group_targets: dict[str, list[str]] = defaultdict(list)
    for target_id, root in roots.items():
        group_targets[root].append(target_id)
    group_id_by_target: dict[str, str] = {}
    for members in group_targets.values():
        member_ids = sorted(members)
        group_id = f"target-family::{member_ids[0]}"
        for target_id in member_ids:
            group_id_by_target[target_id] = group_id
    return group_id_by_target


def _best_distinct_part_assignment(
    member_piece_indexes: tuple[int, ...],
    piece_entries: list[_PieceEntry],
    story_part_polys: dict[str, Any],
) -> dict[int, str]:
    members = list(member_piece_indexes)
    if len(members) < 2 or not story_part_polys:
        return {}

    score_by_member_part: dict[tuple[int, str], float] = {}
    candidate_parts_by_member: dict[int, list[str]] = {}
    for member_index in members:
        piece = piece_entries[member_index]
        candidate_parts = [
            part_id for part_id in piece.part_ids if part_id in story_part_polys
        ]
        if not candidate_parts:
            continue
        candidate_parts_by_member[member_index] = candidate_parts
        for part_id in candidate_parts:
            part_poly = story_part_polys.get(part_id)
            if part_poly is None:
                continue
            try:
                score = float(piece.poly.intersection(part_poly).area)
            except Exception:
                score = 0.0
            score_by_member_part[(member_index, part_id)] = score

    if len(candidate_parts_by_member) < 2:
        return {}

    eligible_members = sorted(
        candidate_parts_by_member.keys(),
        key=lambda member_index: len(candidate_parts_by_member.get(member_index, [])),
    )
    available_parts = sorted(
        {
            part_id
            for member_index in eligible_members
            for part_id in candidate_parts_by_member[member_index]
        }
    )
    if len(available_parts) < 2:
        return {}

    best_score = -1.0
    best_mapping: dict[int, str] = {}
    member_count = len(eligible_members)
    for part_selection in permutations(available_parts, member_count):
        mapping = {
            eligible_members[idx]: part_selection[idx] for idx in range(member_count)
        }
        valid = True
        score_total = 0.0
        for member_index, part_id in mapping.items():
            if part_id not in candidate_parts_by_member.get(member_index, []):
                valid = False
                break
            score_total += float(score_by_member_part.get((member_index, part_id), 0.0))
        if not valid:
            continue
        if score_total > best_score:
            best_score = score_total
            best_mapping = mapping
    if not best_mapping:
        return {}
    if best_score <= 1e-6:
        return {}
    return best_mapping


def _select_leftover_owner(
    geom: Any,
    member_piece_indexes: tuple[int, ...],
    piece_entries: list[_PieceEntry],
    current_geoms: dict[int, Any] | None = None,
) -> tuple[int | None, float]:
    best_member = None
    best_overlap = -1.0
    best_current_area = -1.0
    best_support = -1.0
    best_prior_final = -1.0
    overlap_tolerance = 1e-3
    for member_index in member_piece_indexes:
        piece_poly = piece_entries[member_index].poly
        try:
            overlap_area = float(geom.intersection(piece_poly).area)
        except Exception:
            overlap_area = 0.0
        current_area = float(
            getattr((current_geoms or {}).get(member_index), "area", 0.0) or 0.0
        )
        support = float(piece_entries[member_index].support_score)
        prior_final = float(piece_entries[member_index].prior_final)

        if best_member is None:
            best_member = member_index
            best_overlap = overlap_area
            best_current_area = current_area
            best_support = support
            best_prior_final = prior_final
            continue

        if overlap_area > best_overlap + overlap_tolerance:
            best_member = member_index
            best_overlap = overlap_area
            best_current_area = current_area
            best_support = support
            best_prior_final = prior_final
            continue

        if abs(overlap_area - best_overlap) <= overlap_tolerance:
            rank = (current_area, support, prior_final)
            best_rank = (best_current_area, best_support, best_prior_final)
            if rank > best_rank:
                best_member = member_index
                best_overlap = overlap_area
                best_current_area = current_area
                best_support = support
                best_prior_final = prior_final
    return best_member, best_overlap


def _partition_group_geometry_by_members(
    selected_group_geom: Any,
    group: _GroupEntry,
    piece_entries: list[_PieceEntry],
    story_part_polys: dict[str, Any],
) -> dict[int, Any]:
    member_geoms: dict[int, Any] = {}
    for member_index in group.member_piece_indexes:
        piece_poly = piece_entries[member_index].poly
        try:
            member_geom = selected_group_geom.intersection(piece_poly)
        except Exception:
            continue
        if getattr(member_geom, "is_empty", True):
            continue
        member_geoms[member_index] = member_geom
    if not member_geoms:
        return {}

    part_assignment = _best_distinct_part_assignment(
        group.member_piece_indexes,
        piece_entries,
        story_part_polys,
    )
    if part_assignment:
        adjusted: dict[int, Any] = {}
        for member_index, member_geom in member_geoms.items():
            part_id = part_assignment.get(member_index)
            part_poly = story_part_polys.get(part_id) if part_id else None
            if part_poly is None:
                adjusted[member_index] = member_geom
                continue
            try:
                clipped = member_geom.intersection(part_poly)
            except Exception:
                clipped = member_geom
            if getattr(clipped, "is_empty", True):
                continue
            adjusted[member_index] = clipped
        if adjusted:
            member_geoms = adjusted

    try:
        covered = _safe_union(list(member_geoms.values()))
        leftover = (
            selected_group_geom.difference(covered)
            if covered is not None
            else selected_group_geom
        )
    except Exception:
        leftover = None
    if leftover is not None and not getattr(leftover, "is_empty", True):
        best_member, best_score = _select_leftover_owner(
            leftover,
            group.member_piece_indexes,
            piece_entries,
            member_geoms,
        )
        if best_member is not None and best_score > 1e-9:
            current = member_geoms.get(best_member)
            merged = (
                _safe_union([current, leftover]) if current is not None else leftover
            )
            if merged is not None and not getattr(merged, "is_empty", True):
                member_geoms[best_member] = merged

    ordered_members = sorted(
        member_geoms.keys(),
        key=lambda member_index: (
            float(getattr(member_geoms[member_index], "area", 0.0) or 0.0),
            piece_entries[member_index].piece_id,
        ),
        reverse=True,
    )
    disjoint: dict[int, Any] = {}
    occupied = None
    for member_index in ordered_members:
        geom = member_geoms[member_index]
        if occupied is not None:
            try:
                geom = geom.difference(occupied)
            except Exception:
                geom = None
        if geom is None or getattr(geom, "is_empty", True):
            continue
        disjoint[member_index] = geom
        occupied = _safe_union([occupied, geom]) if occupied is not None else geom

    # Numerical edge cases in part clipping/disjoint subtraction can leave tiny
    # uncovered slivers inside the selected group footprint. Keep coverage
    # complete by assigning uncovered leftovers to a single best member.
    try:
        covered_disjoint = _safe_union(list(disjoint.values()))
        uncovered = (
            selected_group_geom.difference(covered_disjoint)
            if covered_disjoint is not None
            else selected_group_geom
        )
    except Exception:
        uncovered = None
    if uncovered is not None and not getattr(uncovered, "is_empty", True):
        uncovered_area = float(getattr(uncovered, "area", 0.0) or 0.0)
        if uncovered_area > 1e-9:
            best_member, best_overlap = _select_leftover_owner(
                uncovered,
                group.member_piece_indexes,
                piece_entries,
                disjoint,
            )
            if best_member is not None and best_overlap > 1e-9:
                current = disjoint.get(best_member)
                merged = (
                    _safe_union([current, uncovered])
                    if current is not None
                    else uncovered
                )
                if merged is not None and not getattr(merged, "is_empty", True):
                    disjoint[best_member] = merged
    return disjoint


def _row_plane_coeffs(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    corners = row.get("corners") or []
    points = [
        (float(point[0]), float(point[1]), float(point[2]))
        for point in corners
        if len(point) >= 3
    ]
    if len(points) < 3:
        return None

    p0 = points[0]
    for idx in range(1, len(points) - 1):
        p1 = points[idx]
        for jdx in range(idx + 1, len(points)):
            p2 = points[jdx]
            v1 = (
                p1[0] - p0[0],
                p1[1] - p0[1],
                p1[2] - p0[2],
            )
            v2 = (
                p2[0] - p0[0],
                p2[1] - p0[1],
                p2[2] - p0[2],
            )
            nx = v1[1] * v2[2] - v1[2] * v2[1]
            ny = v1[2] * v2[0] - v1[0] * v2[2]
            nz = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(nx) <= 1e-9 and abs(ny) <= 1e-9 and abs(nz) <= 1e-9:
                continue
            if abs(ny) <= 1e-9:
                continue
            d = -(nx * p0[0] + ny * p0[1] + nz * p0[2])
            return nx, ny, nz, d
    return None


def _row_output_plane_coeffs(
    row: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    stored = stored_row_plane_coeffs(row)
    if stored is not None:
        return stored
    return _row_plane_coeffs(row)


def _plane_y_at(
    plane_coeffs: tuple[float, float, float, float] | None, x: float, z: float
) -> float:
    if plane_coeffs is None:
        return 0.0
    nx, ny, nz, d = plane_coeffs
    if abs(ny) <= 1e-9:
        return 0.0
    return -(nx * x + nz * z + d) / ny


def _ring_to_3d_loop(
    coords: list[tuple[float, float]],
    plane_coeffs: tuple[float, float, float, float] | None,
    fallback_y: float,
) -> list[list[float]]:
    out: list[list[float]] = []
    for x, z in coords:
        y = (
            _plane_y_at(plane_coeffs, float(x), float(z))
            if plane_coeffs is not None
            else fallback_y
        )
        out.append([round(float(x), 6), round(float(y), 6), round(float(z), 6)])
    return out


def _is_dormer_exception(
    row: dict[str, Any],
    poly: Polygon,
    soft_envelope: Polygon | None,
    cfg: GlobalSelectionConfig,
) -> tuple[bool, float]:
    if soft_envelope is None:
        return False, 0.0
    total_area = max(float(poly.area), 1e-9)
    try:
        outside_soft = poly.difference(soft_envelope)
    except Exception:
        return False, 0.0
    outside_soft_area = float(getattr(outside_soft, "area", 0.0) or 0.0)
    outside_soft_fraction = outside_soft_area / total_area
    area = float(poly.area)
    inclination = float(row.get("target_inclination_deg") or 0.0)
    is_dormer = (
        area <= cfg.dormer_max_area_m2
        and inclination >= cfg.dormer_min_inclination_deg
        and cfg.dormer_min_outside_fraction
        <= outside_soft_fraction
        <= cfg.dormer_max_outside_fraction
    )
    return bool(is_dormer), outside_soft_fraction


def _build_cells(
    piece_entries: list[Any],
    cfg: GlobalSelectionConfig,
    *,
    extra_boundaries: list[Any] | None = None,
) -> list[_CellEntry]:
    if not piece_entries:
        return []
    try:
        boundaries = [piece.poly.boundary for piece in piece_entries]
        boundaries.extend(
            boundary
            for boundary in (extra_boundaries or [])
            if boundary is not None and not getattr(boundary, "is_empty", True)
        )
        linework = unary_union(boundaries)
    except Exception:
        return []
    cells: list[_CellEntry] = []
    for raw_cell in polygonize(linework):
        if raw_cell.is_empty:
            continue
        cell_area = float(raw_cell.area)
        if cell_area < cfg.min_cell_area_m2:
            continue
        rep = raw_cell.representative_point()
        center_x = float(rep.x)
        center_z = float(rep.y)
        covering_indexes = [
            piece_index
            for piece_index, piece in enumerate(piece_entries)
            if piece.poly.covers(rep)
        ]
        if not covering_indexes:
            continue
        cells.append(
            _CellEntry(
                poly=raw_cell,
                area=cell_area,
                center_x=center_x,
                center_z=center_z,
                covering_piece_indexes=tuple(sorted(covering_indexes)),
            )
        )
        if len(cells) >= cfg.max_cells:
            break
    return cells


def _apply_story_batch(
    rows_out: list[dict[str, Any]],
    piece_entries: list[_PieceEntry],
    cfg: GlobalSelectionConfig,
    story_part_polys: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(piece_entries) < 2:
        return []

    group_member_indexes: dict[str, list[int]] = defaultdict(list)
    for piece_index, piece in enumerate(piece_entries):
        group_member_indexes[piece.plane_group_id].append(piece_index)

    group_entries: list[_GroupEntry] = []
    for group_id, member_indexes in group_member_indexes.items():
        group_poly = _safe_union(
            [piece_entries[index].poly for index in member_indexes]
        )
        if group_poly is None:
            continue
        representative_piece_index = max(
            member_indexes,
            key=lambda index: (
                piece_entries[index].support_score,
                piece_entries[index].prior_final,
                piece_entries[index].piece_id,
            ),
        )
        hard_envelope = _safe_union(
            [
                piece_entries[index].hard_envelope
                for index in member_indexes
                if piece_entries[index].hard_envelope is not None
            ]
        )
        soft_envelope = _safe_union(
            [
                piece_entries[index].soft_envelope
                for index in member_indexes
                if piece_entries[index].soft_envelope is not None
            ]
        )
        group_entries.append(
            _GroupEntry(
                group_id=group_id,
                poly=group_poly,
                support_score=max(
                    piece_entries[index].support_score for index in member_indexes
                ),
                prior_final=max(
                    piece_entries[index].prior_final for index in member_indexes
                ),
                is_dormer_exception=any(
                    piece_entries[index].is_dormer_exception for index in member_indexes
                ),
                hard_envelope=hard_envelope,
                soft_envelope=soft_envelope,
                representative_piece_index=representative_piece_index,
                member_piece_indexes=tuple(sorted(member_indexes)),
            )
        )

    if len(group_entries) < 2:
        return []

    story_envelope_boundaries = []
    for group in group_entries:
        hard = group.hard_envelope
        if hard is None:
            continue
        story_envelope_boundaries.append(getattr(hard, "boundary", None))
    cells = _build_cells(
        group_entries,
        cfg,
        extra_boundaries=story_envelope_boundaries,
    )
    if not cells:
        return []

    plane_coeffs_by_group = {
        group_index: _row_plane_coeffs(
            rows_out[piece_entries[group.representative_piece_index].row_index]
        )
        for group_index, group in enumerate(group_entries)
    }
    topness_by_cell_group: dict[tuple[int, int], float] = {}
    for cell_index, cell in enumerate(cells):
        heights: list[tuple[int, float]] = []
        for group_index in cell.covering_piece_indexes:
            coeffs = plane_coeffs_by_group.get(group_index)
            y = _plane_y_at(coeffs, cell.center_x, cell.center_z)
            heights.append((group_index, y))
        heights.sort(key=lambda item: item[1], reverse=True)
        if not heights:
            continue
        denom = max(len(heights) - 1, 1)
        for rank, (group_index, _height) in enumerate(heights):
            topness_by_cell_group[(cell_index, group_index)] = 1.0 - (
                float(rank) / float(denom)
            )

    x_var_map: dict[tuple[int, int], int] = {}
    outside_soft_coeff_by_var: dict[int, float] = {}
    row_cell_eq: list[int] = []
    col_cell_eq: list[int] = []
    data_cell_eq: list[float] = []
    c_values: list[float] = []

    optimizable_cells: list[int] = []
    for cell_index, cell in enumerate(cells):
        feasible_group_indexes: list[int] = []
        for group_index in cell.covering_piece_indexes:
            group = group_entries[group_index]
            outside_hard_fraction = 0.0
            outside_soft_area = 0.0
            hard_envelope = group.hard_envelope
            soft_envelope = group.soft_envelope
            if hard_envelope is not None:
                try:
                    outside_hard = cell.poly.difference(hard_envelope)
                    outside_hard_area = float(getattr(outside_hard, "area", 0.0) or 0.0)
                except Exception:
                    outside_hard_area = 0.0
                outside_hard_fraction = outside_hard_area / max(cell.area, 1e-9)
            if (
                outside_hard_fraction > cfg.max_cell_hard_outside_fraction
                and not group.is_dormer_exception
            ):
                continue

            if soft_envelope is not None:
                try:
                    outside_soft = cell.poly.difference(soft_envelope)
                    outside_soft_area = float(getattr(outside_soft, "area", 0.0) or 0.0)
                except Exception:
                    outside_soft_area = 0.0

            var_index = len(c_values)
            x_var_map[(cell_index, group_index)] = var_index
            feasible_group_indexes.append(group_index)

            topness = topness_by_cell_group.get((cell_index, group_index), 0.0)
            outside_soft_weight = cfg.objective_weight_soft_outside * (
                cfg.dormer_soft_outside_penalty_scale
                if group.is_dormer_exception
                else 1.0
            )
            utility = (
                cell.area
                * (
                    cfg.objective_weight_support * group.support_score
                    + cfg.objective_weight_topness * topness
                    + cfg.objective_weight_prior_final * group.prior_final
                )
                - outside_soft_weight * outside_soft_area
            )
            c_values.append(-utility)
            outside_soft_coeff_by_var[var_index] = outside_soft_area

        if not feasible_group_indexes:
            continue
        eq_row_index = len(optimizable_cells)
        optimizable_cells.append(cell_index)
        for group_index in feasible_group_indexes:
            var_index = x_var_map[(cell_index, group_index)]
            row_cell_eq.append(eq_row_index)
            col_cell_eq.append(var_index)
            data_cell_eq.append(1.0)

    if not c_values:
        return []
    if len(c_values) > cfg.max_variables:
        return []

    n_x = len(c_values)
    n_groups = len(group_entries)
    y_var_offset = n_x
    n_vars = n_x + n_groups

    c = np.zeros(n_vars, dtype=float)
    if c_values:
        c[:n_x] = np.asarray(c_values, dtype=float)
    c[n_x:] = cfg.objective_weight_piece_activation

    constraints: list[LinearConstraint] = []
    if optimizable_cells:
        eq_matrix = coo_array(
            (
                np.asarray(data_cell_eq, dtype=float),
                (np.asarray(row_cell_eq), np.asarray(col_cell_eq)),
            ),
            shape=(len(optimizable_cells), n_vars),
        )
        eq_rhs = np.ones(len(optimizable_cells), dtype=float)
        constraints.append(LinearConstraint(eq_matrix, eq_rhs, eq_rhs))

    link_rows: list[int] = []
    link_cols: list[int] = []
    link_data: list[float] = []
    link_upper: list[float] = []
    for eq_index, ((_cell_index, group_index), x_var_index) in enumerate(
        x_var_map.items()
    ):
        y_var_index = y_var_offset + group_index
        link_rows.extend([eq_index, eq_index])
        link_cols.extend([x_var_index, y_var_index])
        link_data.extend([1.0, -1.0])
        link_upper.append(0.0)
    if link_upper:
        link_matrix = coo_array(
            (
                np.asarray(link_data, dtype=float),
                (np.asarray(link_rows), np.asarray(link_cols)),
            ),
            shape=(len(link_upper), n_vars),
        )
        lower = np.full(len(link_upper), -np.inf, dtype=float)
        upper = np.asarray(link_upper, dtype=float)
        constraints.append(LinearConstraint(link_matrix, lower, upper))

    soft_budget_base = _safe_union(
        [
            piece.soft_envelope
            for piece in group_entries
            if piece.soft_envelope is not None and not piece.is_dormer_exception
        ]
    )
    if soft_budget_base is not None:
        budget = (
            float(soft_budget_base.area) * cfg.envelope_soft_outside_budget_fraction
        )
        soft_rows: list[int] = []
        soft_cols: list[int] = []
        soft_data: list[float] = []
        for (_cell_index, group_index), x_var_index in x_var_map.items():
            group = group_entries[group_index]
            if group.is_dormer_exception:
                continue
            coeff = float(outside_soft_coeff_by_var.get(x_var_index, 0.0))
            if coeff <= 1e-9:
                continue
            soft_rows.append(0)
            soft_cols.append(x_var_index)
            soft_data.append(coeff)
        if soft_data:
            soft_matrix = coo_array(
                (
                    np.asarray(soft_data, dtype=float),
                    (np.asarray(soft_rows), np.asarray(soft_cols)),
                ),
                shape=(1, n_vars),
            )
            constraints.append(
                LinearConstraint(
                    soft_matrix,
                    np.asarray([-np.inf], dtype=float),
                    np.asarray([budget], dtype=float),
                )
            )

    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=np.ones(n_vars, dtype=int),
            bounds=Bounds(
                lb=np.zeros(n_vars, dtype=float),
                ub=np.ones(n_vars, dtype=float),
            ),
            options={
                "time_limit": cfg.solver_time_limit_s,
                "presolve": True,
            },
        )
    except Exception:
        return []

    if (
        not bool(getattr(result, "success", False))
        or getattr(result, "x", None) is None
    ):
        return []

    solution = np.rint(np.asarray(result.x, dtype=float)).astype(int)
    selected_cells_by_group: dict[int, list[Polygon]] = {
        group_index: [] for group_index in range(n_groups)
    }
    for (cell_index, group_index), x_var_index in x_var_map.items():
        if solution[x_var_index] <= 0:
            continue
        selected_cells_by_group[group_index].append(cells[cell_index].poly)

    selected_geom_by_group: dict[int, Any] = {}
    for group_index, group in enumerate(group_entries):
        selected_cells = selected_cells_by_group.get(group_index, [])
        if not selected_cells:
            continue
        try:
            selected_geom = unary_union(selected_cells).intersection(group.poly)
        except Exception:
            continue
        if getattr(selected_geom, "is_empty", True):
            continue
        selected_geom_by_group[group_index] = selected_geom

    selected_geom_by_piece_index: dict[int, Any] = {}
    for group_index, group in enumerate(group_entries):
        selected_group_geom = selected_geom_by_group.get(group_index)
        if selected_group_geom is None:
            continue
        partitioned = _partition_group_geometry_by_members(
            selected_group_geom,
            group,
            piece_entries,
            story_part_polys,
        )
        for member_index, geom in partitioned.items():
            if geom is None or getattr(geom, "is_empty", True):
                continue
            selected_geom_by_piece_index[member_index] = geom

    extra_rows: list[dict[str, Any]] = []
    group_index_by_group_id = {
        group.group_id: index for index, group in enumerate(group_entries)
    }
    for piece_index, piece in enumerate(piece_entries):
        row = rows_out[piece.row_index]
        group_index = group_index_by_group_id.get(piece.plane_group_id)
        selected_piece_geom = (
            selected_geom_by_piece_index.get(piece_index)
            if group_index is not None
            else None
        )
        group_member_count = (
            len(group_entries[group_index].member_piece_indexes)
            if group_index is not None
            else 1
        )
        pre_ilp_final = bool(row.get("final_layer"))
        pre_ilp_reason = str(row.get("final_layer_reason") or "")
        row["xy_global_selector_applied"] = True
        row["xy_global_selector_strategy"] = "ilp_mip_v1"
        row["xy_global_selector_dormer_exception"] = bool(piece.is_dormer_exception)
        row["xy_global_selector_soft_outside_fraction"] = round(
            piece.outside_soft_fraction,
            6,
        )
        row["xy_global_selector_plane_group_id"] = piece.plane_group_id
        row["xy_global_selector_plane_group_member_count"] = group_member_count
        row["final_layer_reason_pre_ilp"] = pre_ilp_reason

        if selected_piece_geom is None:
            row["xy_global_selector_unselected"] = True
            row["overlay_suppressed"] = True
            row["final_layer"] = False
            if pre_ilp_final:
                row["final_layer_reason"] = "xy_global_selector_unselected"
            continue

        try:
            selected_geom = selected_piece_geom
        except Exception:
            row["xy_global_selector_unselected"] = True
            row["overlay_suppressed"] = True
            row["final_layer"] = False
            if pre_ilp_final:
                row["final_layer_reason"] = "xy_global_selector_unselected"
            continue

        selected_polys = [
            poly
            for poly in iter_polygons(selected_geom)
            if float(getattr(poly, "area", 0.0) or 0.0) >= cfg.min_piece_area_m2
        ]
        if not selected_polys:
            row["xy_global_selector_unselected"] = True
            row["overlay_suppressed"] = True
            row["final_layer"] = False
            if pre_ilp_final:
                row["final_layer_reason"] = "xy_global_selector_unselected"
            continue
        selected_polys.sort(key=lambda poly: float(poly.area), reverse=True)

        plane_coeffs = _row_output_plane_coeffs(row)
        first_corner = next(
            (corner for corner in (row.get("corners") or []) if len(corner) >= 3),
            None,
        )
        fallback_y = float(first_corner[1]) if first_corner is not None else 0.0
        source_piece_id = str(row.get("piece_id") or "")
        source_piece_index = _piece_index_value(row.get("piece_index"))

        row["xy_global_selector_unselected"] = False
        row["final_layer"] = True
        if not pre_ilp_final:
            row["final_layer_reason"] = "xy_global_selector_selected_candidate"
        row["overlay_suppressed"] = False
        row["xy_global_selector_selected_area_m2"] = round(
            float(sum(float(poly.area) for poly in selected_polys)),
            6,
        )
        row["xy_global_selector_original_area_m2"] = round(float(piece.poly.area), 6)

        for poly_index, poly in enumerate(selected_polys):
            shell = list(poly.exterior.coords)[:-1]
            if len(shell) < 3:
                continue
            holes: list[list[list[float]]] = []
            for ring in poly.interiors:
                ring_coords = list(ring.coords)[:-1]
                if len(ring_coords) < 3:
                    continue
                try:
                    hole_area = float(abs(Polygon(ring_coords).area))
                except Exception:
                    hole_area = 0.0
                if hole_area <= 1e-5:
                    continue
                holes.append(_ring_to_3d_loop(ring_coords, plane_coeffs, fallback_y))
            corners = _ring_to_3d_loop(shell, plane_coeffs, fallback_y)

            if poly_index == 0:
                target_row = row
                target_row["piece_id"] = source_piece_id
                target_row["piece_index"] = source_piece_index
            else:
                target_row = dict(row)
                target_row["piece_id"] = f"{source_piece_id}:xy-ilp:{poly_index}"
                target_row["piece_index"] = source_piece_index + poly_index
                extra_rows.append(target_row)

            target_row["corners"] = corners
            target_row["holes"] = holes
            target_row["area_xz_m2"] = round(float(poly.area), 6)
            target_row["xy_global_selector_piece_index"] = poly_index
    return extra_rows


def apply_global_xy_selection(
    rows: list[dict[str, Any]],
    building: dict[str, Any],
    cfg: GlobalSelectionConfig,
    *,
    roof_result: dict[str, Any] | None = None,
    story_gap_polygons: dict[int, list[Any]] | None = None,
    relation_graph: Any | None = None,
) -> list[dict[str, Any]]:
    if not cfg.enabled:
        return rows

    rows_out = [dict(row) for row in rows]
    story_env, story_part_env = _rain_story_envelopes(
        building,
        roof_result,
        story_gap_polygons,
    )
    plane_group_id_by_target = _target_family_group_ids(
        rows_out,
        relation_graph,
        cfg.include_target_kinds,
    )

    # Collect already-final non-eligible pieces as external obstacles by story.
    # Pieces finalized by layer_policy before ILP (e.g.
    # ridge_eave_mirror_rescued_residual)
    # are excluded from eligible_entries; without pre-clipping, ILP can select pieces
    # that
    # overlap them and produce a final-layer conflict.
    external_obstacle_by_story: dict[int, Any] = {}
    for row in rows_out:
        if not bool(row.get("final_layer")):
            continue
        if bool(row.get("overlay_suppressed")):
            continue
        target_kind_obs = str(row.get("target_kind") or "")
        if target_kind_obs not in cfg.include_target_kinds:
            continue
        if (
            str(row.get("piece_role") or "") == "supported"
            and _hard_invalid_pre_ilp_reason(row, cfg) is None
        ):
            continue
        obstacle_poly = _row_xz_polygon(row)
        if obstacle_poly is None or getattr(obstacle_poly, "is_empty", True):
            continue
        obs_story = int(row.get("story") or 0)
        existing = external_obstacle_by_story.get(obs_story)
        try:
            external_obstacle_by_story[obs_story] = (
                existing.union(obstacle_poly) if existing is not None else obstacle_poly
            )
        except Exception:
            pass

    eligible_entries: list[_PieceEntry] = []
    for row_index, row in enumerate(rows_out):
        target_kind = str(row.get("target_kind") or "")
        if target_kind not in cfg.include_target_kinds:
            continue
        if str(row.get("piece_role") or "") != "supported":
            continue
        if bool(row.get("overlay_suppressed")):
            # ILP-eligible ridge_eave pieces may arrive pre-suppressed from the
            # XY conflict step; include them so the ILP resolves the conflict.
            if not (
                target_kind == "ridge_eave_plane_group"
                and str(row.get("final_layer_reason") or "")
                in _RIDGE_ILP_ELIGIBLE_REASONS
            ):
                continue
        hard_invalid_reason = _hard_invalid_pre_ilp_reason(row, cfg)
        if hard_invalid_reason is not None:
            row["xy_global_selector_hard_valid_pre"] = False
            row["xy_global_selector_hard_filter_reason"] = hard_invalid_reason
            continue
        row["xy_global_selector_hard_valid_pre"] = True
        poly = _row_xz_polygon(row)
        if poly is None:
            continue
        story = int(row.get("story") or 0)
        obstacle = external_obstacle_by_story.get(story)
        if obstacle is not None and not getattr(obstacle, "is_empty", True):
            try:
                poly = poly.difference(obstacle)
            except Exception:
                pass
            if poly is None or getattr(poly, "is_empty", True):
                continue
        area = float(poly.area)
        if area < cfg.min_piece_area_m2:
            continue
        part_ids = _piece_story_part_ids(row)
        hard_envelope, soft_envelope = _piece_envelopes(
            story,
            part_ids,
            story_env,
            story_part_env,
            cfg,
        )
        is_dormer, outside_soft_fraction = _is_dormer_exception(
            row,
            poly,
            soft_envelope,
            cfg,
        )
        eligible_entries.append(
            _PieceEntry(
                row_index=row_index,
                story=story,
                piece_id=str(row.get("piece_id") or ""),
                piece_index=_piece_index_value(row.get("piece_index")),
                poly=poly,
                support_score=float(row.get("support_score") or 0.0),
                prior_final=1.0 if bool(row.get("final_layer")) else 0.0,
                is_dormer_exception=is_dormer,
                outside_soft_fraction=outside_soft_fraction,
                hard_envelope=hard_envelope,
                soft_envelope=soft_envelope,
                plane_group_id=plane_group_id_by_target.get(
                    str(row.get("target_element_id") or ""),
                    str(row.get("target_element_id") or str(row.get("piece_id") or "")),
                ),
                part_ids=part_ids,
            )
        )

    if len(eligible_entries) < 2:
        return rows_out

    entries_by_story: dict[int, list[_PieceEntry]] = defaultdict(list)
    for entry in eligible_entries:
        entries_by_story[int(entry.story)].append(entry)

    extra_rows: list[dict[str, Any]] = []
    for story, story_entries in entries_by_story.items():
        story_part_polys = {
            part_id: geom
            for (part_story, part_id), geom in story_part_env.items()
            if int(part_story) == int(story)
        }
        extra_rows.extend(
            _apply_story_batch(
                rows_out,
                story_entries,
                cfg,
                story_part_polys,
            )
        )
    if extra_rows:
        rows_out.extend(extra_rows)
    return rows_out
