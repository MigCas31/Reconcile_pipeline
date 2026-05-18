"""Hybrid 3D-first authoritative zone derivation for reconstruction.

The reconstruction ILP still reasons in XZ, but zone ownership should come
from compact physical support, not from broad semantic subparts. This module
therefore:

1. builds compact 3D support components from occupied room cells,
2. projects those components into XZ and relabels overlapping support
   footprints into disjoint ownership regions,
3. treats roof coverage subparts as priors that refine geometry *inside*
   owned support, never as ownership truth.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from reconcile.extract3d.overlaps import decompose_polys, floor_polygon_to_shapely
from reconcile.roof_algorithms_py.graph_utils import stable_hash as _stable_hash

from ..audit import note_geom_skip

# Threshold under which a wing is too small or too marginal a slice of its
# parent component to deserve a separate zone — avoids splintering a building
# along jog-induced slivers (porches, bay windows) that look like rectangles
# to the grid decomposition but aren't structurally distinct wings.
_WING_SPLIT_MIN_AREA_M2 = 8.0
_WING_SPLIT_MIN_RATIO = 0.18

Point2D = tuple[float, float]

_AREA_EPS = 0.01
_OWNERSHIP_CELL_MIN_AREA_M2 = 0.05
_SUPPORT_TOUCH_TOL_M = 0.15
_VERTICAL_OVERLAP_TOL_M = 0.10
_MIN_SUPPORT_VOLUME_M3 = 0.5
_ROOM_CLAIM_INTERSECTION_M2 = 0.25
_FALLBACK_MIN_AREA_M2 = 2.0
_FALLBACK_MIN_VOLUME_M3 = 3.0
_ZONE_MIN_PIECE_AREA_M2 = 0.25
_ZONE_CLEANUP_BUFFER_M = 0.10
_OWNERSHIP_SMOOTH_CELL_AREA_M2 = 0.5
_OWNERSHIP_SMOOTH_SHARED_RATIO = 0.70
_OWNERSHIP_SMOOTH_SWITCH_TOL = 1.5
_SHADOW_OVERLAP_RATIO = 0.80
_SEED_CLIP_MIN_RATIO = 0.40
_SCORE_W_VOLUME_DENSITY = 1.5
_ADJACENCY_MIN_SHARED_LEN_M = 0.05
_COMPONENT_BUFFER_M = 0.05
_SEED_REASSIGN_OVERLAP_MIN = 0.25
_AZ_AGREE_OVERLAP_MIN = 0.35
_CONF_W_SEED = 0.45
_CONF_W_VOLUME = 0.30
_CONF_W_ROOMS = 0.25
_CONF_FALLBACK_PENALTY = 0.7
_COVERAGE_MIN_RATIO = 0.6
_RESIDUAL_MIN_RATIO_OF_PIECE = 0.15
_ROOF_PIECE_MIN_OVERLAP_M2 = 0.05


@dataclass
class _SupportComponent:
    id: str
    occupied_cell_ids: list[str]
    roof_hypothesis_ids: list[str]
    room_ids: list[str]
    room_indices: list[int]
    top_boundary_atom_ids: list[str]
    footprint_poly: Polygon
    volume_m3: float
    bbox_xyz: list[float] | None
    part_hint_ids: list[str]


@dataclass
class _SeedSubpart:
    id: str
    roof_hypothesis_id: str | None
    room_indices: list[int]
    part_hint_ids: list[str]
    semantic_kind: str
    support_evidence_score: int
    avg_azimuth_deg: float | None
    polygon: Polygon
    adjacent_subpart_ids: list[str]
    atom_ids: list[str]


def _poly_from_xz_ring(points: list) -> Polygon | None:
    ring = [
        (float(pt[0]), float(pt[1]))
        for pt in points
        if isinstance(pt, (list, tuple)) and len(pt) >= 2
    ]
    if len(ring) < 3:
        return None
    try:
        poly = Polygon(ring)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    largest = _largest_polygon(poly)
    if largest is None or largest.area <= _AREA_EPS:
        return None
    return largest


def _poly_from_xyz_ring(points: list) -> Polygon | None:
    return _poly_from_xz_ring(
        [
            [float(pt[0]), float(pt[2])]
            for pt in points
            if isinstance(pt, (list, tuple)) and len(pt) >= 3
        ]
    )


def _normalize_geom(geom):
    if geom is None or getattr(geom, "is_empty", True):
        return None
    try:
        cleaned = geom.buffer(0)
        if not getattr(cleaned, "is_empty", True):
            geom = cleaned
    except Exception as exc:
        note_geom_skip(exc, "zones.normalize_buffer0")
    return None if getattr(geom, "is_empty", True) else geom


def _largest_polygon(geom) -> Polygon | None:
    polygons = _expand_polygons(_normalize_geom(geom))
    if not polygons:
        return None
    return max(polygons, key=lambda part: float(part.area))


def _expand_polygons(geom) -> list[Polygon]:
    geom = _normalize_geom(geom)
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [
            part
            for part in geom.geoms
            if isinstance(part, Polygon) and not part.is_empty
        ]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for item in geom.geoms:
            out.extend(_expand_polygons(item))
        return out
    return [
        part
        for part in getattr(geom, "geoms", [])
        if isinstance(part, Polygon) and not part.is_empty
    ]


def _polygons_above_area(geom, min_area: float) -> list[Polygon]:
    return [poly for poly in _expand_polygons(geom) if poly.area > min_area]


def _union_polys(polys: list[Polygon]):
    if not polys:
        return None
    return _normalize_geom(unary_union(polys))


def _polygon_xz_ring(poly: Polygon) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[round(float(x), 6), round(float(z), 6)] for x, z in coords]


def _bbox_union(bboxes: list[list[float]]) -> list[float] | None:
    if not bboxes:
        return None
    return [
        round(min(b[0] for b in bboxes), 6),
        round(min(b[1] for b in bboxes), 6),
        round(min(b[2] for b in bboxes), 6),
        round(max(b[3] for b in bboxes), 6),
        round(max(b[4] for b in bboxes), 6),
        round(max(b[5] for b in bboxes), 6),
    ]


def _bbox_vertical_overlap(
    left: list[float] | None, right: list[float] | None
) -> float:
    if not left or not right or len(left) < 6 or len(right) < 6:
        return 0.0
    return max(
        0.0,
        min(float(left[4]), float(right[4])) - max(float(left[1]), float(right[1])),
    )


def _intersection_area(left, right) -> float:
    if left is None or right is None:
        return 0.0
    try:
        return float(left.intersection(right).area)
    except Exception:
        return 0.0


def _cell_footprint(cell: dict[str, Any]) -> Polygon | None:
    for face_kind in ("top", "bottom"):
        for face in cell.get("faces") or []:
            if str(face.get("kind") or "") != face_kind:
                continue
            poly = _poly_from_xyz_ring(face.get("corners") or [])
            if poly is not None:
                return poly
    for face in cell.get("faces") or []:
        poly = _poly_from_xyz_ring(face.get("corners") or [])
        if poly is not None:
            return poly
    return None


def _room_partitions_by_room(
    roof_building: dict[str, Any],
    bldg: dict[str, Any] | None,
) -> dict[int, list[Polygon]]:
    by_room: dict[int, list[Polygon]] = defaultdict(list)
    room_partitions = (roof_building.get("ceiling_partitions") or {}).get(
        "room_partitions"
    ) or []
    for room_partition in room_partitions:
        room_index = room_partition.get("room_index")
        if not isinstance(room_index, int):
            continue
        for partition in room_partition.get("partitions") or []:
            poly = floor_polygon_to_shapely(partition.get("poly") or [])
            if poly is not None and not poly.is_empty and poly.area > _AREA_EPS:
                by_room[room_index].extend(_polygons_above_area(poly, _AREA_EPS))
        outline_poly = floor_polygon_to_shapely(
            room_partition.get("room_outline") or []
        )
        if (
            outline_poly is not None
            and not outline_poly.is_empty
            and outline_poly.area > _AREA_EPS
        ):
            by_room[room_index].extend(_polygons_above_area(outline_poly, _AREA_EPS))
    if bldg:
        for room_index, room in enumerate(bldg.get("rooms") or []):
            if by_room.get(room_index):
                continue
            floor_poly = floor_polygon_to_shapely(room.get("floor_polygon") or [])
            if (
                floor_poly is not None
                and not floor_poly.is_empty
                and floor_poly.area > _AREA_EPS
            ):
                by_room[room_index].extend(_polygons_above_area(floor_poly, _AREA_EPS))
    return by_room


def _room_ids_for_component(cells: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    room_ids = sorted(
        {str(cell.get("room_id")) for cell in cells if cell.get("room_id")}
    )
    room_indices = sorted(
        {
            int(cell.get("room_index"))
            for cell in cells
            if isinstance(cell.get("room_index"), int)
        }
    )
    return room_ids, room_indices


def _component_part_hints(
    room_ids: list[str],
    room_membership: dict[str, list[str]],
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for room_id in room_ids:
        for part_id in room_membership.get(room_id, []):
            counts[str(part_id)] += 1
    return sorted(counts, key=lambda part_id: (-counts[part_id], part_id))


def _split_cells_by_wings(
    *,
    cells: list[dict[str, Any]],
    polys: list[Polygon],
    footprint_poly: Polygon,
    wings_fn,
) -> list[dict[str, Any]]:
    """Split a connected support component along rectangular-wing boundaries.

    Returns a list of {cells, polys, bboxes} dicts. When the footprint
    decomposes into a single wing (or no substantial wings), returns the
    component as a single entry — preserving today's behaviour for
    rectangular and non-rectilinear buildings. For L/T/U buildings each
    distinct rectangle becomes a sub-component, with cells assigned to
    whichever wing's polygon contains the cell's footprint centroid.
    """
    cell_bboxes = [
        cell.get("bbox_xyz")
        if isinstance(cell.get("bbox_xyz"), list)
        and len(cell.get("bbox_xyz") or []) >= 6
        else None
        for cell in cells
    ]
    fallback_bboxes = [bbox for bbox in cell_bboxes if bbox is not None]
    fallback = [{"cells": list(cells), "polys": list(polys), "bboxes": fallback_bboxes}]
    if footprint_poly is None or footprint_poly.is_empty:
        return fallback
    wings = wings_fn(footprint_poly)
    if len(wings) <= 1:
        return fallback
    parent_area = float(footprint_poly.area)
    qualified_wings = [
        wing
        for wing in wings
        if wing.area_m2 >= _WING_SPLIT_MIN_AREA_M2
        and (parent_area <= 0 or wing.area_m2 / parent_area >= _WING_SPLIT_MIN_RATIO)
    ]
    if len(qualified_wings) <= 1:
        return fallback
    buckets: list[dict[str, Any]] = [
        {"cells": [], "polys": [], "bboxes": []} for _ in qualified_wings
    ]
    for idx, (cell, poly) in enumerate(zip(cells, polys, strict=False)):
        centroid = poly.centroid
        chosen: int | None = None
        for wi, wing in enumerate(qualified_wings):
            if wing.polygon.contains(centroid):
                chosen = wi
                break
        if chosen is None:
            best_dist = float("inf")
            for wi, wing in enumerate(qualified_wings):
                d = wing.polygon.distance(centroid)
                if d < best_dist:
                    best_dist = d
                    chosen = wi
        if chosen is None:
            continue
        buckets[chosen]["cells"].append(cell)
        buckets[chosen]["polys"].append(poly)
        bbox = cell_bboxes[idx]
        if bbox is not None:
            buckets[chosen]["bboxes"].append(bbox)
    populated = [bucket for bucket in buckets if bucket["cells"]]
    if len(populated) <= 1:
        return fallback
    return populated


def _build_support_components(
    *,
    building_uuid: str,
    occupied_cells: list[dict[str, Any]],
    room_membership: dict[str, list[str]],
) -> list[_SupportComponent]:
    prepared: list[dict[str, Any]] = []
    for cell in occupied_cells:
        poly = _cell_footprint(cell)
        if poly is None or poly.area <= _AREA_EPS:
            continue
        prepared.append(
            {
                "cell": cell,
                "poly": poly,
                "bbox": cell.get("bbox_xyz"),
                "room_id": str(cell.get("room_id") or ""),
                "room_index": cell.get("room_index"),
                "part_id": str(cell.get("part_id") or ""),
                "top_atom_ids": {
                    str(atom_id)
                    for atom_id in (cell.get("top_boundary_atom_ids") or [])
                    if atom_id
                },
                "roof_hypothesis_id": str(cell.get("roof_hypothesis_id") or ""),
            }
        )

    adjacency: list[set[int]] = [set() for _ in range(len(prepared))]
    for i, left in enumerate(prepared):
        for j in range(i + 1, len(prepared)):
            right = prepared[j]
            connected = False
            if left["room_id"] and left["room_id"] == right["room_id"]:
                connected = True
            elif left["top_atom_ids"] and left["top_atom_ids"].intersection(
                right["top_atom_ids"]
            ):
                connected = True
            elif (
                left["roof_hypothesis_id"]
                and left["roof_hypothesis_id"] == right["roof_hypothesis_id"]
                and (
                    not left["part_id"]
                    or not right["part_id"]
                    or left["part_id"] == right["part_id"]
                )
                and left["poly"]
                .buffer(_SUPPORT_TOUCH_TOL_M)
                .intersects(right["poly"].buffer(_SUPPORT_TOUCH_TOL_M))
            ):
                connected = True
            else:
                vertical_overlap = _bbox_vertical_overlap(left["bbox"], right["bbox"])
                if vertical_overlap >= _VERTICAL_OVERLAP_TOL_M:
                    same_part_hint = (
                        not left["part_id"]
                        or not right["part_id"]
                        or left["part_id"] == right["part_id"]
                    )
                    if same_part_hint:
                        try:
                            connected = (
                                left["poly"]
                                .buffer(_SUPPORT_TOUCH_TOL_M)
                                .intersects(right["poly"].buffer(_SUPPORT_TOUCH_TOL_M))
                            )
                        except Exception:
                            connected = False
            if connected:
                adjacency[i].add(j)
                adjacency[j].add(i)

    raw_components: list[list[int]] = []
    seen: set[int] = set()
    for idx in range(len(prepared)):
        if idx in seen:
            continue
        stack = [idx]
        current_component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            current_component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        if current_component:
            raw_components.append(sorted(current_component))

    from reconcile.wing_decomposition import decompose_to_wings as _decompose_to_wings

    room_component_counts: dict[str, int] = defaultdict(int)
    support_components: list[_SupportComponent] = []
    for component_index, component in enumerate(raw_components):
        cells = [prepared[idx]["cell"] for idx in component]
        polys = [prepared[idx]["poly"] for idx in component]
        room_ids, _room_indices = _room_ids_for_component(cells)
        for room_id in room_ids:
            room_component_counts[room_id] += 1
        union_poly = _largest_polygon(unary_union(polys))
        if union_poly is None or union_poly.area <= _AREA_EPS:
            continue
        sub_components = _split_cells_by_wings(
            cells=cells,
            polys=polys,
            footprint_poly=union_poly,
            wings_fn=_decompose_to_wings,
        )
        for sub_index, sub in enumerate(sub_components):
            sub_cells = sub["cells"]
            sub_polys = sub["polys"]
            sub_room_ids, sub_room_indices = _room_ids_for_component(sub_cells)
            sub_union = _largest_polygon(unary_union(sub_polys))
            if sub_union is None or sub_union.area <= _AREA_EPS:
                continue
            sub_hash = _stable_hash(
                [
                    str(component_index),
                    str(sub_index),
                    str(sorted(c["id"] for c in sub_cells)),
                ],
                20,
            )
            support_components.append(
                _SupportComponent(
                    id=f"{building_uuid}::hybrid-zone-support::{sub_hash}",
                    occupied_cell_ids=sorted(
                        str(cell["id"]) for cell in sub_cells if cell.get("id")
                    ),
                    roof_hypothesis_ids=sorted(
                        {
                            str(cell.get("roof_hypothesis_id"))
                            for cell in sub_cells
                            if cell.get("roof_hypothesis_id")
                        }
                    ),
                    room_ids=sub_room_ids,
                    room_indices=sub_room_indices,
                    top_boundary_atom_ids=sorted(
                        {
                            str(atom_id)
                            for cell in sub_cells
                            for atom_id in (cell.get("top_boundary_atom_ids") or [])
                            if atom_id
                        }
                    ),
                    footprint_poly=sub_union,
                    volume_m3=round(
                        sum(float(cell.get("volume_m3") or 0.0) for cell in sub_cells),
                        6,
                    ),
                    bbox_xyz=_bbox_union(sub["bboxes"]),
                    part_hint_ids=_component_part_hints(sub_room_ids, room_membership),
                )
            )

    kept: list[_SupportComponent] = []
    for component in support_components:
        if component.volume_m3 >= _MIN_SUPPORT_VOLUME_M3:
            kept.append(component)
            continue
        if any(
            room_component_counts.get(room_id, 0) <= 1 for room_id in component.room_ids
        ):
            kept.append(component)
    return kept


def _build_seed_subparts(
    *,
    roof_building: dict[str, Any],
    support_components: list[_SupportComponent],
) -> list[_SeedSubpart]:
    coverage_graph = roof_building.get("roof_coverage_graph") or {}
    room_to_subparts = coverage_graph.get("room_subpart_membership") or {}
    atom_to_subparts = coverage_graph.get("atom_subpart_membership") or {}
    subpart_to_atoms: dict[str, set[str]] = defaultdict(set)
    for atom_id, subpart_ids in atom_to_subparts.items():
        for subpart_id in subpart_ids or []:
            subpart_to_atoms[str(subpart_id)].add(str(atom_id))

    support_rooms = {
        room_index
        for component in support_components
        for room_index in component.room_indices
    }
    seeds: list[_SeedSubpart] = []
    for subpart in coverage_graph.get("subparts") or []:
        subpart_id = str(subpart.get("id") or "")
        if not subpart_id:
            continue
        room_indices = [
            int(room_index)
            for room_index in (subpart.get("room_indices") or [])
            if isinstance(room_index, int)
        ]
        polygon = _poly_from_xz_ring(subpart.get("polygon_xz") or [])
        if polygon is None or polygon.area <= _AREA_EPS:
            continue
        support_score = int(subpart.get("support_evidence_score", 0) or 0)
        semantic_kind = str(subpart.get("semantic_kind", "slope_run") or "slope_run")
        eligible = support_score >= 3 or semantic_kind in {"gable_run", "l_t_branch"}
        if not eligible:
            for room_index in room_indices:
                room_id = f"room:{room_index}"
                if room_index not in support_rooms:
                    continue
                memberships = [
                    str(value)
                    for value in (room_to_subparts.get(room_id) or [])
                    if value
                ]
                if memberships == [subpart_id] or len(memberships) == 1:
                    eligible = True
                    break
        if not eligible:
            continue
        avg_azimuth = subpart.get("avg_azimuth")
        avg_azimuth = (
            float(avg_azimuth)
            if avg_azimuth is not None and isfinite(float(avg_azimuth))
            else None
        )
        seeds.append(
            _SeedSubpart(
                id=subpart_id,
                roof_hypothesis_id=(
                    str(subpart.get("roof_hypothesis_id"))
                    if subpart.get("roof_hypothesis_id")
                    else None
                ),
                room_indices=sorted(set(room_indices)),
                part_hint_ids=sorted(
                    str(value) for value in (subpart.get("part_ids") or []) if value
                ),
                semantic_kind=semantic_kind,
                support_evidence_score=support_score,
                avg_azimuth_deg=avg_azimuth,
                polygon=polygon,
                adjacent_subpart_ids=sorted(
                    str(value)
                    for value in (subpart.get("adjacent_subpart_ids") or [])
                    if value
                ),
                atom_ids=sorted(subpart_to_atoms.get(subpart_id, set())),
            )
        )
    return seeds


def _azimuth_diff_deg(a: float, b: float) -> float:
    raw = abs(float(a) - float(b)) % 360.0
    return min(raw, 360.0 - raw)


def _component_room_union(
    component: _SupportComponent,
    room_partition_polys: dict[int, list[Polygon]],
):
    polys = [
        poly
        for room_index in component.room_indices
        for poly in room_partition_polys.get(room_index, [])
        if poly is not None and not poly.is_empty
    ]
    return _union_polys(polys)


def _component_seed_union(seeds: list[_SeedSubpart]):
    return _union_polys([seed.polygon for seed in seeds if seed.polygon is not None])


def _seed_component_score(
    component: _SupportComponent,
    seed: _SeedSubpart,
    *,
    footprint_override=None,
) -> tuple[float, float, float, int, int]:
    support_geom = (
        footprint_override
        if footprint_override is not None
        else component.footprint_poly
    )
    if support_geom is None or getattr(support_geom, "is_empty", True):
        return 0.0, 0.0, 0.0, 0, 0
    shared_atoms = len(set(component.top_boundary_atom_ids).intersection(seed.atom_ids))
    shared_rooms = len(set(component.room_indices).intersection(seed.room_indices))
    same_hypothesis = int(
        bool(
            seed.roof_hypothesis_id
            and seed.roof_hypothesis_id in component.roof_hypothesis_ids
        )
    )
    overlap_area = _intersection_area(support_geom, seed.polygon)
    min_area = max(
        min(float(getattr(support_geom, "area", 0.0) or 0.0), float(seed.polygon.area)),
        _AREA_EPS,
    )
    overlap_ratio = overlap_area / min_area
    part_hint_match = int(
        bool(set(component.part_hint_ids).intersection(seed.part_hint_ids))
    )
    score = (
        6.0 * shared_atoms
        + 4.0 * shared_rooms
        + 3.0 * same_hypothesis
        + 2.0 * overlap_ratio
        + 1.0 * part_hint_match
    )
    if footprint_override is not None:
        score += 3.0 * overlap_ratio
    return score, overlap_area, overlap_ratio, shared_atoms, shared_rooms


def _initial_seed_component_map(
    support_components: list[_SupportComponent],
    seeds: list[_SeedSubpart],
) -> dict[str, list[_SeedSubpart]]:
    component_seed_map: dict[str, list[_SeedSubpart]] = defaultdict(list)
    for seed in seeds:
        for component in support_components:
            score, overlap_area, _overlap_ratio, shared_atoms, shared_rooms = (
                _seed_component_score(component, seed)
            )
            if shared_rooms <= 0 and shared_atoms <= 0 and overlap_area < 0.10:
                continue
            if score <= 0.0:
                continue
            component_seed_map[component.id].append(seed)
    for component_id, attached in component_seed_map.items():
        component_seed_map[component_id] = sorted(
            {seed.id: seed for seed in attached}.values(),
            key=lambda seed: seed.id,
        )
    return component_seed_map


def _seed_spill_penalty(
    component: _SupportComponent, seeds: list[_SeedSubpart]
) -> float:
    attached_rooms = {room_index for seed in seeds for room_index in seed.room_indices}
    if not attached_rooms:
        return 0.0
    spill = attached_rooms - set(component.room_indices)
    return len(spill) / max(len(attached_rooms), 1)


def _build_atomic_support_cells(
    support_components: list[_SupportComponent],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for component in support_components:
        poly = component.footprint_poly
        if poly is None or poly.area <= _AREA_EPS:
            continue
        if not cells:
            for piece in _polygons_above_area(poly, _OWNERSHIP_CELL_MIN_AREA_M2):
                cells.append({"poly": piece, "covering_component_ids": {component.id}})
            continue
        next_cells: list[dict[str, Any]] = []
        residual_parts: list[Polygon] = [poly]
        for cell in cells:
            cell_poly = cell["poly"]
            intersection = None
            try:
                intersection = cell_poly.intersection(poly)
            except Exception:
                intersection = None
            for piece in _polygons_above_area(
                intersection, _OWNERSHIP_CELL_MIN_AREA_M2
            ):
                next_cells.append(
                    {
                        "poly": piece,
                        "covering_component_ids": set(
                            cell["covering_component_ids"]
                        ).union({component.id}),
                    }
                )
            try:
                diff_existing = cell_poly.difference(poly)
            except Exception:
                diff_existing = cell_poly
            for piece in _polygons_above_area(
                diff_existing, _OWNERSHIP_CELL_MIN_AREA_M2
            ):
                next_cells.append(
                    {
                        "poly": piece,
                        "covering_component_ids": set(cell["covering_component_ids"]),
                    }
                )
            updated_residuals: list[Polygon] = []
            for residual in residual_parts:
                try:
                    diff_new = residual.difference(cell_poly)
                except Exception:
                    diff_new = residual
                updated_residuals.extend(
                    _polygons_above_area(diff_new, _OWNERSHIP_CELL_MIN_AREA_M2)
                )
            residual_parts = updated_residuals
        for residual in residual_parts:
            next_cells.append(
                {
                    "poly": residual,
                    "covering_component_ids": {component.id},
                }
            )
        cells = next_cells
    return cells


def _assign_support_projection_ownership(
    support_components: list[_SupportComponent],
    room_partition_polys: dict[int, list[Polygon]],
    component_seed_map: dict[str, list[_SeedSubpart]],
) -> dict[str, Any]:
    component_by_id = {component.id: component for component in support_components}
    component_room_union = {
        component.id: _component_room_union(component, room_partition_polys)
        for component in support_components
    }
    component_seed_union = {
        component.id: _component_seed_union(component_seed_map.get(component.id) or [])
        for component in support_components
    }
    raw_cells = _build_atomic_support_cells(support_components)
    if not raw_cells:
        return {}

    for cell in raw_cells:
        cover_ids = sorted(cell["covering_component_ids"])
        scores: dict[str, float] = {}
        densities: dict[str, float] = {}
        for component_id in cover_ids:
            component = component_by_id[component_id]
            footprint_area = max(float(component.footprint_poly.area), _AREA_EPS)
            densities[component_id] = component.volume_m3 / footprint_area
        max_density = max(densities.values()) if densities else 1.0
        for component_id in cover_ids:
            component = component_by_id[component_id]
            cell_poly = cell["poly"]
            room_cover_ratio = _intersection_area(
                cell_poly, component_room_union.get(component_id)
            ) / max(float(cell_poly.area), _AREA_EPS)
            seed_cover_ratio = _intersection_area(
                cell_poly, component_seed_union.get(component_id)
            ) / max(float(cell_poly.area), _AREA_EPS)
            part_hint_specificity = (
                1.0
                if len(component.part_hint_ids) == 1
                else 0.5
                if len(component.part_hint_ids) > 1
                else 0.0
            )
            volume_density_norm = densities[component_id] / max(max_density, _AREA_EPS)
            attached_seeds = component_seed_map.get(component_id) or []
            roof_hypothesis_match = float(
                any(
                    seed.roof_hypothesis_id
                    and seed.roof_hypothesis_id in component.roof_hypothesis_ids
                    for seed in attached_seeds
                )
            )
            seed_spill_penalty = _seed_spill_penalty(component, attached_seeds)
            no_local_room_penalty = 1.0 if room_cover_ratio < 0.10 else 0.0
            scores[component_id] = (
                5.0 * room_cover_ratio
                + 3.0 * seed_cover_ratio
                + 2.0 * part_hint_specificity
                + _SCORE_W_VOLUME_DENSITY * volume_density_norm
                + 1.0 * roof_hypothesis_match
                - 4.0 * seed_spill_penalty
                - 3.0 * no_local_room_penalty
            )
        cell["scores"] = scores
        owner = max(
            cover_ids,
            key=lambda component_id: (scores[component_id], component_id),
        )
        cell["owner_component_id"] = owner

    adjacency: dict[int, dict[int, float]] = defaultdict(dict)
    for left_idx in range(len(raw_cells)):
        for right_idx in range(left_idx + 1, len(raw_cells)):
            left_poly = raw_cells[left_idx]["poly"]
            right_poly = raw_cells[right_idx]["poly"]
            try:
                shared_len = float(
                    left_poly.boundary.intersection(right_poly.boundary).length
                )
            except Exception:
                shared_len = 0.0
            if shared_len < _ADJACENCY_MIN_SHARED_LEN_M:
                continue
            adjacency[left_idx][right_idx] = shared_len
            adjacency[right_idx][left_idx] = shared_len

    for _ in range(2):
        updated_owners: dict[int, str] = {}
        for idx, cell in enumerate(raw_cells):
            if float(cell["poly"].area) >= _OWNERSHIP_SMOOTH_CELL_AREA_M2:
                continue
            owner_lengths: dict[str, float] = defaultdict(float)
            total_shared = 0.0
            for neighbor_idx, shared_len in adjacency.get(idx, {}).items():
                neighbor_owner = raw_cells[neighbor_idx]["owner_component_id"]
                owner_lengths[neighbor_owner] += shared_len
                total_shared += shared_len
            if total_shared <= 0.0:
                continue
            target_owner, target_len = max(
                owner_lengths.items(), key=lambda item: (item[1], item[0])
            )
            if target_owner == cell["owner_component_id"]:
                continue
            if target_owner not in cell["covering_component_ids"]:
                continue
            if target_len / total_shared < _OWNERSHIP_SMOOTH_SHARED_RATIO:
                continue
            current_score = float(cell["scores"].get(cell["owner_component_id"], -1e9))
            target_score = float(cell["scores"].get(target_owner, -1e9))
            if current_score - target_score > _OWNERSHIP_SMOOTH_SWITCH_TOL:
                continue
            updated_owners[idx] = target_owner
        for idx, owner in updated_owners.items():
            raw_cells[idx]["owner_component_id"] = owner

    owned_polys: dict[str, list[Polygon]] = defaultdict(list)
    for cell in raw_cells:
        owned_polys[cell["owner_component_id"]].append(cell["poly"])

    owned_by_component: dict[str, Any] = {}
    for component_id, polys in owned_polys.items():
        merged = _union_polys(polys)
        if merged is None or getattr(merged, "is_empty", True):
            continue
        try:
            cleaned = merged.buffer(_COMPONENT_BUFFER_M).buffer(-_COMPONENT_BUFFER_M)
            if not getattr(cleaned, "is_empty", True):
                merged = cleaned
        except Exception as exc:
            note_geom_skip(exc, "zones.component_buffer")
        merged = _normalize_geom(merged)
        if merged is not None and float(merged.area) > _AREA_EPS:
            owned_by_component[component_id] = merged
    return owned_by_component


def _final_seed_component_map(
    support_components: list[_SupportComponent],
    seeds: list[_SeedSubpart],
    owned_support_by_component: dict[str, Any],
) -> dict[str, list[_SeedSubpart]]:
    component_seed_map: dict[str, list[_SeedSubpart]] = defaultdict(list)
    for seed in seeds:
        scored_candidates: list[tuple[float, float, float, str]] = []
        for component in support_components:
            owned_support = owned_support_by_component.get(component.id)
            score, overlap_area, overlap_ratio, shared_atoms, shared_rooms = (
                _seed_component_score(
                    component,
                    seed,
                    footprint_override=owned_support,
                )
            )
            if shared_rooms <= 0 and shared_atoms <= 0 and overlap_area < 0.10:
                continue
            scored_candidates.append((score, overlap_area, overlap_ratio, component.id))
        if not scored_candidates:
            continue
        best_score = max(row[0] for row in scored_candidates)
        for score, overlap_area, overlap_ratio, component_id in scored_candidates:
            if overlap_area < 0.10:
                continue
            if score < best_score - 6.0 and overlap_ratio < _SEED_REASSIGN_OVERLAP_MIN:
                continue
            component_seed_map[component_id].append(seed)
    for component_id, attached in component_seed_map.items():
        component_seed_map[component_id] = sorted(
            {seed.id: seed for seed in attached}.values(),
            key=lambda seed: (
                -seed.support_evidence_score,
                -float(seed.polygon.area),
                seed.id,
            ),
        )
    return component_seed_map


def _seed_pair_compatible(left: _SeedSubpart, right: _SeedSubpart) -> bool:
    if left.id == right.id:
        return True
    adjacency_hint = (
        right.id in left.adjacent_subpart_ids or left.id in right.adjacent_subpart_ids
    )
    overlap_area = _intersection_area(left.polygon, right.polygon)
    min_area = max(min(float(left.polygon.area), float(right.polygon.area)), _AREA_EPS)
    overlap_frac = overlap_area / min_area
    az_diff = None
    if left.avg_azimuth_deg is not None and right.avg_azimuth_deg is not None:
        az_diff = _azimuth_diff_deg(left.avg_azimuth_deg, right.avg_azimuth_deg)
    if az_diff is not None and az_diff < 35.0 and overlap_frac >= _AZ_AGREE_OVERLAP_MIN:
        return False
    if adjacency_hint:
        if left.semantic_kind == "gable_run" or right.semantic_kind == "gable_run":
            return az_diff is not None and 140.0 <= az_diff <= 220.0
        if left.semantic_kind == "l_t_branch" or right.semantic_kind == "l_t_branch":
            return az_diff is not None and 60.0 <= az_diff <= 120.0
        if az_diff is None:
            return True
        return az_diff >= 35.0 or overlap_frac < _AZ_AGREE_OVERLAP_MIN
    if left.roof_hypothesis_id and left.roof_hypothesis_id == right.roof_hypothesis_id:
        return overlap_frac < _AZ_AGREE_OVERLAP_MIN
    return overlap_frac < 0.10


def _seed_groups_for_component(seeds: list[_SeedSubpart]) -> list[list[_SeedSubpart]]:
    groups: list[list[_SeedSubpart]] = []
    ordered = sorted(
        seeds,
        key=lambda seed: (
            -seed.support_evidence_score,
            -float(seed.polygon.area),
            seed.id,
        ),
    )
    for seed in ordered:
        placed = False
        for group in groups:
            if all(_seed_pair_compatible(seed, other) for other in group):
                group.append(seed)
                placed = True
                break
        if not placed:
            groups.append([seed])
    return groups


def _split_zone_pieces(
    geom, seed_centroids: list[tuple[float, float]]
) -> list[Polygon]:
    geom = _normalize_geom(geom)
    if geom is None:
        return []
    try:
        cleaned = geom.buffer(_ZONE_CLEANUP_BUFFER_M).buffer(-_ZONE_CLEANUP_BUFFER_M)
        if getattr(cleaned, "is_empty", True):
            cleaned = geom
    except Exception:
        cleaned = geom
    pieces: list[Polygon] = []
    for piece in decompose_polys(cleaned):
        if piece.is_empty or piece.area <= _AREA_EPS:
            continue
        keep = piece.area >= _ZONE_MIN_PIECE_AREA_M2
        if not keep:
            keep = any(
                piece.buffer(1e-6).contains(Point(x, z)) for x, z in seed_centroids
            )
        if keep:
            pieces.append(piece)
    if pieces:
        return pieces
    largest = _largest_polygon(geom)
    return [largest] if largest is not None else []


def _zone_part_id(
    room_ids: list[str],
    part_nodes: dict[str, dict[str, Any]],
) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for part_id, node in part_nodes.items():
        node_room_ids = {str(room_id) for room_id in (node.get("room_ids") or [])}
        overlap = len(node_room_ids.intersection(room_ids))
        if overlap > 0:
            counts[part_id] = overlap
    if not counts:
        return None
    best_part_id, best_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    if best_count * 2 <= len(room_ids):
        return None
    return best_part_id


def _zone_confidence(
    *,
    seeds: list[_SeedSubpart],
    room_ids: list[str],
    volume_m3: float,
    fallback_kind: str | None,
) -> float:
    if seeds:
        seed_strength = min(
            mean(min(seed.support_evidence_score / 5.0, 1.0) for seed in seeds),
            1.0,
        )
    else:
        seed_strength = 0.15 if volume_m3 >= _FALLBACK_MIN_VOLUME_M3 else 0.0
    volume_strength = min(volume_m3 / 20.0, 1.0)
    room_strength = min(len(room_ids) / 3.0, 1.0)
    confidence = (
        _CONF_W_SEED * seed_strength
        + _CONF_W_VOLUME * volume_strength
        + _CONF_W_ROOMS * room_strength
    )
    if fallback_kind:
        confidence *= _CONF_FALLBACK_PENALTY
    return round(max(0.0, min(confidence, 1.0)), 6)


def _owned_room_indices(
    owned_support,
    component: _SupportComponent,
    room_partition_polys: dict[int, list[Polygon]],
    attached_seeds: list[_SeedSubpart],
) -> list[int]:
    local_indices = [
        room_index
        for room_index in component.room_indices
        if sum(
            _intersection_area(poly, owned_support)
            for poly in room_partition_polys.get(room_index, [])
        )
        >= _ROOM_CLAIM_INTERSECTION_M2
    ]
    owned_indices = set(local_indices or component.room_indices)
    spill_candidates = {
        room_index
        for seed in attached_seeds
        for room_index in seed.room_indices
        if room_index not in owned_indices
    }
    for room_index in spill_candidates:
        overlap_area = sum(
            _intersection_area(poly, owned_support)
            for poly in room_partition_polys.get(room_index, [])
        )
        if overlap_area >= _ROOM_CLAIM_INTERSECTION_M2:
            owned_indices.add(room_index)
    return sorted(owned_indices)


def _room_union_inside_owned_support(
    room_indices: list[int],
    owned_support,
    room_partition_polys: dict[int, list[Polygon]],
):
    polys: list[Polygon] = []
    for room_index in room_indices:
        for poly in room_partition_polys.get(room_index, []):
            if _intersection_area(poly, owned_support) >= _ROOM_CLAIM_INTERSECTION_M2:
                polys.append(poly)
    return _union_polys(polys)


def _split_piece_by_part_overlap(
    *,
    piece: Polygon,
    component: _SupportComponent,
    piece_room_ids: list[str],
    room_partition_polys: dict[int, list[Polygon]],
    room_membership: dict[str, list[str]],
    part_nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    distinct_part_ids = {
        part_id
        for room_id in piece_room_ids
        for part_id in room_membership.get(room_id, [])
    }
    if len(distinct_part_ids) <= 1:
        return [
            {
                "poly": piece,
                "room_ids": piece_room_ids,
                "part_id": _zone_part_id(piece_room_ids, part_nodes),
            }
        ]

    local_room_ids = [
        room_id for room_id in piece_room_ids if room_id in set(component.room_ids)
    ]
    group_room_ids: dict[str, list[str]] = {}
    if local_room_ids:
        group_room_ids["__local__"] = sorted(local_room_ids)
    for room_id in piece_room_ids:
        if room_id in local_room_ids:
            continue
        memberships = [
            str(part_id) for part_id in room_membership.get(room_id, []) if part_id
        ]
        group_key = memberships[0] if memberships else "__unassigned__"
        group_room_ids.setdefault(group_key, []).append(room_id)
    if len(group_room_ids) <= 1:
        return [
            {
                "poly": piece,
                "room_ids": piece_room_ids,
                "part_id": _zone_part_id(piece_room_ids, part_nodes),
            }
        ]

    group_polys: dict[str, Any] = {}
    group_scores: dict[str, tuple[float, float, int]] = {}
    for group_key, room_ids in group_room_ids.items():
        polys = []
        overlap_sum = 0.0
        for room_id in room_ids:
            room_index = int(room_id.split(":")[-1])
            for poly in room_partition_polys.get(room_index, []):
                overlap = _intersection_area(poly, piece)
                if overlap <= _AREA_EPS:
                    continue
                overlap_sum += overlap
                try:
                    clipped = poly.intersection(piece)
                except Exception:
                    clipped = None
                polys.extend(_polygons_above_area(clipped, _AREA_EPS))
        merged = _union_polys(polys)
        if merged is None or getattr(merged, "is_empty", True):
            continue
        group_polys[group_key] = merged
        group_scores[group_key] = (
            overlap_sum,
            float(len(room_ids)),
            float(group_key == "__local__"),
        )
    if len(group_polys) <= 1:
        return [
            {
                "poly": piece,
                "room_ids": piece_room_ids,
                "part_id": _zone_part_id(piece_room_ids, part_nodes),
            }
        ]

    cells: list[dict[str, Any]] = []
    for group_key, geom in group_polys.items():
        for poly in _expand_polygons(geom):
            if poly.area <= _AREA_EPS:
                continue
            if not cells:
                cells.append({"poly": poly, "covering_groups": {group_key}})
                continue
            next_cells: list[dict[str, Any]] = []
            residual_parts: list[Polygon] = [poly]
            for cell in cells:
                cell_poly = cell["poly"]
                try:
                    intersection = cell_poly.intersection(poly)
                except Exception:
                    intersection = None
                for piece_part in _polygons_above_area(intersection, _AREA_EPS):
                    next_cells.append(
                        {
                            "poly": piece_part,
                            "covering_groups": set(cell["covering_groups"]).union(
                                {group_key}
                            ),
                        }
                    )
                try:
                    diff_existing = cell_poly.difference(poly)
                except Exception:
                    diff_existing = cell_poly
                for piece_part in _polygons_above_area(diff_existing, _AREA_EPS):
                    next_cells.append(
                        {
                            "poly": piece_part,
                            "covering_groups": set(cell["covering_groups"]),
                        }
                    )
                updated_residuals: list[Polygon] = []
                for residual in residual_parts:
                    try:
                        diff_new = residual.difference(cell_poly)
                    except Exception:
                        diff_new = residual
                    updated_residuals.extend(_polygons_above_area(diff_new, _AREA_EPS))
                residual_parts = updated_residuals
            for residual in residual_parts:
                next_cells.append({"poly": residual, "covering_groups": {group_key}})
            cells = next_cells

    assigned: dict[str, list[Polygon]] = defaultdict(list)
    for cell in cells:
        cover_keys = sorted(cell["covering_groups"])
        if not cover_keys:
            continue
        owner = max(
            cover_keys,
            key=lambda key: (
                group_scores.get(key, (0.0, 0.0, 0.0))[2],
                group_scores.get(key, (0.0, 0.0, 0.0))[0],
                group_scores.get(key, (0.0, 0.0, 0.0))[1],
                key,
            ),
        )
        assigned[owner].append(cell["poly"])

    variants: list[dict[str, Any]] = []
    assigned_union_parts: list[Polygon] = []
    for group_key, polys in assigned.items():
        geom = _union_polys(polys)
        if geom is None or getattr(geom, "is_empty", True):
            continue
        for subpiece in _expand_polygons(geom):
            if subpiece.area <= _AREA_EPS:
                continue
            sub_room_ids = [
                room_id
                for room_id in group_room_ids.get(group_key, [])
                if any(
                    _intersection_area(room_poly, subpiece) >= _AREA_EPS
                    for room_poly in room_partition_polys.get(
                        int(room_id.split(":")[-1]), []
                    )
                )
            ]
            if not sub_room_ids:
                sub_room_ids = list(group_room_ids.get(group_key, []))
            part_id = None
            if group_key == "__local__" and component.part_hint_ids:
                part_id = component.part_hint_ids[0]
            elif group_key != "__unassigned__":
                part_id = group_key
            if part_id is None:
                part_id = _zone_part_id(sub_room_ids, part_nodes)
            variants.append(
                {
                    "poly": subpiece,
                    "room_ids": sorted(sub_room_ids),
                    "part_id": part_id,
                }
            )
            assigned_union_parts.append(subpiece)

    assigned_union = _union_polys(assigned_union_parts)
    covered_area = float(getattr(assigned_union, "area", 0.0) or 0.0)
    piece_area = max(float(piece.area), _AREA_EPS)
    if covered_area < _COVERAGE_MIN_RATIO * piece_area:
        return [
            {
                "poly": piece,
                "room_ids": piece_room_ids,
                "part_id": _zone_part_id(piece_room_ids, part_nodes),
            }
        ]

    try:
        residual = (
            piece.difference(assigned_union) if assigned_union is not None else piece
        )
    except Exception:
        residual = None
    residual_geom = _normalize_geom(residual)
    if residual_geom is not None and float(residual_geom.area) >= max(
        0.5, _RESIDUAL_MIN_RATIO_OF_PIECE * piece_area
    ):
        for residual_piece in _expand_polygons(residual_geom):
            if residual_piece.area <= _AREA_EPS:
                continue
            residual_room_ids = [
                room_id
                for room_id in piece_room_ids
                if any(
                    _intersection_area(room_poly, residual_piece) >= _AREA_EPS
                    for room_poly in room_partition_polys.get(
                        int(room_id.split(":")[-1]), []
                    )
                )
            ]
            if residual_room_ids:
                variants.append(
                    {
                        "poly": residual_piece,
                        "room_ids": sorted(residual_room_ids),
                        "part_id": _zone_part_id(residual_room_ids, part_nodes),
                    }
                )
    return variants or [
        {
            "poly": piece,
            "room_ids": piece_room_ids,
            "part_id": _zone_part_id(piece_room_ids, part_nodes),
        }
    ]


def _zone_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_poly = left.get("_poly")
    right_poly = right.get("_poly")
    if left_poly is None or right_poly is None:
        return 0.0
    overlap_area = _intersection_area(left_poly, right_poly)
    min_area = min(float(left_poly.area), float(right_poly.area))
    if min_area <= _AREA_EPS:
        return 0.0
    return overlap_area / min_area


def _suppress_duplicate_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_support_ids: dict[str, set[str]] = defaultdict(set)
    for zone in zones:
        support_component_id = str(zone.get("support_component_id") or "")
        if not support_component_id:
            continue
        for seed_id in zone.get("seed_subpart_ids") or []:
            if seed_id:
                seed_support_ids[str(seed_id)].add(support_component_id)

    def _zone_priority(zone: dict[str, Any]) -> tuple[float, ...]:
        seed_reuse_count = sum(
            max(len(seed_support_ids.get(str(seed_id), set())) - 1, 0)
            for seed_id in (zone.get("seed_subpart_ids") or [])
            if seed_id
        )
        return (
            float(int(bool(zone.get("part_id")))),
            -float(seed_reuse_count),
            -float(len(zone.get("seed_spill_room_indices") or [])),
            float(zone.get("confidence") or 0.0),
            float(zone.get("owned_support_area_m2") or 0.0),
        )

    drop_ids: set[str] = set()
    for i, left in enumerate(zones):
        if left["id"] in drop_ids:
            continue
        for j in range(i + 1, len(zones)):
            right = zones[j]
            if right["id"] in drop_ids:
                continue
            overlap_ratio = _zone_overlap_ratio(left, right)
            if overlap_ratio < _SHADOW_OVERLAP_RATIO:
                continue
            left_part = left.get("part_id")
            right_part = right.get("part_id")
            if (
                left.get("support_component_id")
                and left.get("support_component_id")
                == right.get("support_component_id")
                and list(left.get("room_ids") or [])
                == list(right.get("room_ids") or [])
            ):
                keep = max(
                    [left, right], key=lambda zone: (_zone_priority(zone), zone["id"])
                )
                drop_ids.add(right["id"] if keep is left else left["id"])
                continue
            if bool(left_part) ^ bool(right_part):
                drop_ids.add(left["id"] if not left_part else right["id"])
                continue
            if (
                set(left.get("seed_hypothesis_ids") or [])
                and set(left.get("seed_hypothesis_ids") or [])
                == set(right.get("seed_hypothesis_ids") or [])
                and left.get("support_component_id")
                != right.get("support_component_id")
            ):
                keep = max(
                    [left, right],
                    key=lambda zone: (_zone_priority(zone), zone["id"]),
                )
                drop_ids.add(right["id"] if keep is left else left["id"])
    kept: list[dict[str, Any]] = []
    for zone in zones:
        if zone["id"] in drop_ids:
            continue
        zone.pop("_poly", None)
        kept.append(zone)
    return kept


def derive_authoritative_zones(
    *,
    building_uuid: str,
    roof_building: dict,
    bldg: dict | None,
) -> list[dict]:
    """Derive compact authoritative reconstruction zones from roof results."""
    if not roof_building:
        return []

    building_part_graph = roof_building.get("building_part_graph") or {}
    part_nodes = {
        str(node["id"]): node
        for node in (building_part_graph.get("nodes") or [])
        if isinstance(node, dict) and node.get("id")
    }
    room_membership = {
        str(room_id): [str(part_id) for part_id in (part_ids or []) if part_id]
        for room_id, part_ids in (
            building_part_graph.get("room_membership") or {}
        ).items()
    }
    occupied_cells = (roof_building.get("occupied_room_cell_complex") or {}).get(
        "cells"
    ) or []
    roof_cells = (roof_building.get("roof_cell_complex") or {}).get("cells") or []
    room_partition_polys = _room_partitions_by_room(roof_building, bldg)

    support_components = _build_support_components(
        building_uuid=building_uuid,
        occupied_cells=occupied_cells,
        room_membership=room_membership,
    )
    if not support_components:
        return []

    seeds = _build_seed_subparts(
        roof_building=roof_building,
        support_components=support_components,
    )
    initial_component_seed_map = _initial_seed_component_map(support_components, seeds)
    owned_support_by_component = _assign_support_projection_ownership(
        support_components,
        room_partition_polys,
        initial_component_seed_map,
    )
    if not owned_support_by_component:
        return []
    component_seed_map = _final_seed_component_map(
        support_components,
        seeds,
        owned_support_by_component,
    )

    roof_cell_polys: list[tuple[dict[str, Any], Polygon]] = []
    for roof_cell in roof_cells:
        poly = _cell_footprint(roof_cell)
        if poly is not None and poly.area > _AREA_EPS:
            roof_cell_polys.append((roof_cell, poly))

    zones: list[dict[str, Any]] = []
    claimed_room_ids: set[str] = set()

    for component in support_components:
        owned_support = owned_support_by_component.get(component.id)
        if owned_support is None or getattr(owned_support, "is_empty", True):
            continue
        attached_seeds = component_seed_map.get(component.id) or []
        owned_room_indices = _owned_room_indices(
            owned_support,
            component,
            room_partition_polys,
            attached_seeds,
        )
        owned_room_ids = sorted(
            f"room:{room_index}" for room_index in owned_room_indices
        )
        attached_seed_ids = [seed.id for seed in attached_seeds]
        spill_room_indices = sorted(
            {
                room_index
                for seed in attached_seeds
                for room_index in seed.room_indices
                if room_index not in component.room_indices
            }
        )
        seed_groups = (
            _seed_groups_for_component(attached_seeds) if attached_seeds else []
        )
        if not seed_groups:
            if (
                float(owned_support.area) >= _FALLBACK_MIN_AREA_M2
                and component.volume_m3 >= _FALLBACK_MIN_VOLUME_M3
                and any(room_id not in claimed_room_ids for room_id in owned_room_ids)
            ):
                seed_groups = [[]]
            else:
                continue

        for _group_index, seed_group in enumerate(seed_groups):
            group_seed_ids = [seed.id for seed in seed_group]
            dropped_seed_ids = sorted(set(attached_seed_ids) - set(group_seed_ids))
            fallback_kind = None
            zone_geom = owned_support
            if seed_group:
                seed_union = _union_polys([seed.polygon for seed in seed_group])
                clipped = None
                try:
                    clipped = (
                        owned_support.intersection(seed_union)
                        if seed_union is not None
                        else None
                    )
                except Exception:
                    clipped = None
                if (
                    clipped is None
                    or getattr(clipped, "is_empty", True)
                    or float(clipped.area)
                    < _SEED_CLIP_MIN_RATIO * float(owned_support.area)
                ):
                    fallback_kind = "owned_support_dominant"
                    zone_geom = owned_support
                else:
                    zone_geom = clipped
            else:
                fallback_kind = "support_component_without_subpart"

            room_union = _room_union_inside_owned_support(
                owned_room_indices,
                owned_support,
                room_partition_polys,
            )
            if room_union is not None:
                try:
                    room_clip = room_union.intersection(owned_support)
                    zone_geom = _normalize_geom(zone_geom.union(room_clip))
                except Exception:
                    zone_geom = _normalize_geom(zone_geom)
            if zone_geom is None or getattr(zone_geom, "is_empty", True):
                continue

            seed_centroids = [
                (float(seed.polygon.centroid.x), float(seed.polygon.centroid.y))
                for seed in seed_group
            ]
            pieces = _split_zone_pieces(zone_geom, seed_centroids)
            for piece_index, piece in enumerate(pieces):
                if piece is None or piece.is_empty or piece.area <= _AREA_EPS:
                    continue
                piece_room_ids = [
                    room_id
                    for room_id in owned_room_ids
                    if any(
                        _intersection_area(room_poly, piece) >= _AREA_EPS
                        for room_poly in room_partition_polys.get(
                            int(room_id.split(":")[-1]), []
                        )
                    )
                ]
                if not piece_room_ids:
                    piece_room_ids = list(owned_room_ids)
                variants = _split_piece_by_part_overlap(
                    piece=piece,
                    component=component,
                    piece_room_ids=piece_room_ids,
                    room_partition_polys=room_partition_polys,
                    room_membership=room_membership,
                    part_nodes=part_nodes,
                )
                seed_hypothesis_ids = sorted(
                    {
                        seed.roof_hypothesis_id
                        for seed in seed_group
                        if seed.roof_hypothesis_id
                    }
                )
                semantic_kinds = sorted(
                    {seed.semantic_kind for seed in seed_group if seed.semantic_kind}
                )
                avg_azimuths = sorted(
                    round(float(seed.avg_azimuth_deg), 6)
                    for seed in seed_group
                    if seed.avg_azimuth_deg is not None
                )
                for variant_index, variant in enumerate(variants):
                    variant_poly = variant["poly"]
                    variant_room_ids = sorted(variant["room_ids"])
                    piece_room_indices = sorted(
                        int(room_id.split(":")[-1]) for room_id in variant_room_ids
                    )
                    part_id = variant.get("part_id")
                    if part_id is None:
                        part_id = _zone_part_id(variant_room_ids, part_nodes)
                    matching_roof_cell_ids = []
                    for roof_cell, roof_poly in roof_cell_polys:
                        overlap_area = _intersection_area(variant_poly, roof_poly)
                        if overlap_area < _ROOF_PIECE_MIN_OVERLAP_M2:
                            continue
                        if (
                            seed_hypothesis_ids
                            and str(roof_cell.get("roof_hypothesis_id") or "")
                            not in seed_hypothesis_ids
                            and int(roof_cell.get("room_index", -1))
                            not in piece_room_indices
                        ):
                            continue
                        if roof_cell.get("id"):
                            matching_roof_cell_ids.append(str(roof_cell["id"]))

                    zone_hash = _stable_hash(
                        [
                            component.id,
                            str(group_seed_ids),
                            str(piece_index),
                            str(variant_index),
                            str(part_id),
                        ],
                        20,
                    )
                    zone_id = f"{building_uuid}::hybrid-zone::{zone_hash}"
                    confidence = _zone_confidence(
                        seeds=seed_group,
                        room_ids=variant_room_ids,
                        volume_m3=component.volume_m3,
                        fallback_kind=fallback_kind,
                    )
                    zones.append(
                        {
                            "id": zone_id,
                            "part_id": part_id,
                            "source": "roof_algorithms_py.hybrid_compact_zone_v2",
                            "kind": "hybrid_compact_zone",
                            "support_component_id": component.id,
                            "ownership_source": "support_projection_assignment_v1",
                            "owned_support_area_m2": round(
                                float(owned_support.area), 6
                            ),
                            "owner_part_hint_ids": list(component.part_hint_ids),
                            "attached_seed_ids": list(attached_seed_ids),
                            "dropped_seed_ids": list(dropped_seed_ids),
                            "seed_spill_room_indices": list(spill_room_indices),
                            "room_ids": variant_room_ids,
                            "room_indices": piece_room_indices,
                            "occupied_cell_ids": list(component.occupied_cell_ids),
                            "roof_cell_ids": sorted(set(matching_roof_cell_ids)),
                            "seed_subpart_ids": group_seed_ids,
                            "seed_hypothesis_ids": seed_hypothesis_ids,
                            "semantic_kinds": semantic_kinds,
                            "avg_azimuths_deg": avg_azimuths,
                            "footprint_xz": _polygon_xz_ring(variant_poly),
                            "bbox_xyz": component.bbox_xyz,
                            "volume_m3": round(component.volume_m3, 6),
                            "footprint_area_m2": round(float(variant_poly.area), 6),
                            "confidence": confidence,
                            "fallback_kind": fallback_kind,
                            "_poly": variant_poly,
                        }
                    )
                    claimed_room_ids.update(variant_room_ids)

    return _suppress_duplicate_zones(zones)
