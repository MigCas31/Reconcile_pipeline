from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .graph_utils import room_key as _room_key
from .graph_utils import stable_hash as _stable_hash
from .roof_arrangement_kernel import build_arranged_polyhedral_cell
from .roof_cell_complex import (
    AREA_EPS,
    EPS,
    _avg_y,
    _convex_subregions,
    _perimeter_side_face_indices,
    _poly_xz_from_3d,
    _surface_y_at,
)


def _largest_polygon(geom: Any) -> Polygon | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    polys = [
        poly
        for poly in getattr(geom, "geoms", [])
        if isinstance(poly, Polygon) and not poly.is_empty
    ]
    if not polys:
        return None
    return max(polys, key=lambda poly: float(poly.area))


def _decompose_polygons(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    polys = [
        poly
        for poly in getattr(geom, "geoms", [])
        if isinstance(poly, Polygon) and not poly.is_empty
    ]
    return polys


def _story_unions(bldg: dict[str, Any]) -> dict[int, Polygon]:
    polys_by_story: dict[int, list[Polygon]] = defaultdict(list)
    for room in bldg.get("rooms") or []:
        story = int(room.get("story", 0) or 0)
        poly = _room_footprint_polygon(room)
        if poly is not None:
            polys_by_story[story].append(poly)
    out: dict[int, Polygon] = {}
    for story, polys in polys_by_story.items():
        if not polys:
            continue
        try:
            union_poly = _largest_polygon(unary_union(polys))
        except Exception:
            union_poly = None
        if union_poly is not None:
            out[story] = union_poly
    return out


def _footprint_from_polygon(poly: Polygon) -> list[tuple[float, float]]:
    def same_point(
        left: tuple[float, float], right: tuple[float, float], tol: float = 1e-5
    ) -> bool:
        return abs(left[0] - right[0]) <= tol and abs(left[1] - right[1]) <= tol

    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    footprint: list[tuple[float, float]] = []
    for x, z, *_ in coords:
        point = (round(float(x), 6), round(float(z), 6))
        if footprint and same_point(point, footprint[-1]):
            continue
        footprint.append(point)
    if len(footprint) >= 2 and same_point(footprint[0], footprint[-1]):
        footprint.pop()
    if len(footprint) < 3:
        return footprint

    simplified: list[tuple[float, float]] = []
    count = len(footprint)
    for index, current in enumerate(footprint):
        prev = footprint[index - 1]
        nxt = footprint[(index + 1) % count]
        vx1 = current[0] - prev[0]
        vz1 = current[1] - prev[1]
        vx2 = nxt[0] - current[0]
        vz2 = nxt[1] - current[1]
        cross = vx1 * vz2 - vz1 * vx2
        if abs(cross) <= 1e-5:
            continue
        simplified.append(current)
    if len(simplified) >= 3:
        footprint = simplified
    return footprint


def _candidate_footprints(poly: Polygon) -> list[list[tuple[float, float]]]:
    candidates: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()

    def add(candidate_poly: Polygon | None) -> None:
        if (
            candidate_poly is None
            or candidate_poly.is_empty
            or candidate_poly.area <= AREA_EPS
        ):
            return
        footprint = _footprint_from_polygon(candidate_poly)
        if len(footprint) < 3:
            return
        key = tuple(footprint)
        if key in seen:
            return
        seen.add(key)
        candidates.append(footprint)

    add(poly)
    try:
        hull = poly.convex_hull
    except Exception:
        hull = None
    if isinstance(hull, Polygon) and not hull.is_empty:
        area_delta = abs(float(hull.area) - float(poly.area))
        if area_delta <= max(AREA_EPS, float(poly.area) * 0.02):
            add(hull)
    return candidates


def _top_face_area(cell: dict[str, Any]) -> float:
    best = 0.0
    for face in cell.get("faces") or []:
        if not isinstance(face, dict):
            continue
        if str(face.get("kind") or "") != "top":
            continue
        best = max(best, float(face.get("area_m2") or 0.0))
    return best


def _build_best_candidate_cell(
    *,
    candidate_footprints: list[list[tuple[float, float]]],
    build_cell,
) -> dict[str, Any] | None:
    best_cell: dict[str, Any] | None = None
    best_score: tuple[int, float, float] | None = None
    for footprint in candidate_footprints:
        cell = build_cell(footprint)
        if cell is None or float(cell.get("volume_m3", 0.0) or 0.0) <= EPS:
            continue
        top_area = _top_face_area(cell)
        score = (
            1 if top_area > AREA_EPS else 0,
            round(top_area, 6),
            round(float(cell.get("volume_m3", 0.0) or 0.0), 6),
        )
        if best_score is None or score > best_score:
            best_cell = cell
            best_score = score
    return best_cell


def _room_fallback_top_y(room: dict[str, Any], base_y: float) -> float:
    top_ys: list[float] = []
    for wall in _room_walls(room):
        corners = wall.get("corners") or []
        ys = sorted(
            float(corner[1])
            for corner in corners
            if isinstance(corner, (list, tuple)) and len(corner) >= 3
        )
        if len(ys) >= 2:
            top_ys.extend(ys[-2:])
    if not top_ys:
        return round(float(base_y) + 2.4, 6)
    return round(float(statistics.median(top_ys)), 6)


def _room_walls(room: dict[str, Any]) -> list[dict[str, Any]]:
    computed = [
        wall for wall in (room.get("walls_computed") or []) if isinstance(wall, dict)
    ]
    merged = [
        wall for wall in (room.get("walls_merged") or []) if isinstance(wall, dict)
    ]
    if len(merged) > len(computed):
        return merged
    return computed or merged


def _room_wall_bottom_points(room: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for wall in _room_walls(room):
        corners = [
            corner
            for corner in (wall.get("corners") or [])
            if isinstance(corner, (list, tuple)) and len(corner) >= 3
        ]
        if len(corners) < 2:
            continue
        min_y = min(float(corner[1]) for corner in corners)
        for corner in corners:
            if float(corner[1]) > min_y + 0.02:
                continue
            point = (round(float(corner[0]), 6), round(float(corner[2]), 6))
            if point in seen:
                continue
            seen.add(point)
            points.append(point)
    return points


def _room_footprint_polygon(room: dict[str, Any]) -> Polygon | None:
    floor_poly = _poly_xz_from_3d(room.get("floor_polygon") or [])
    if floor_poly is not None and floor_poly.area > AREA_EPS:
        return floor_poly
    bottom_points = _room_wall_bottom_points(room)
    if len(bottom_points) < 3:
        return None
    try:
        hull = MultiPoint(bottom_points).convex_hull
    except Exception:
        return None
    if hull is None or hull.is_empty:
        return None
    if not isinstance(hull, Polygon):
        return None
    if not hull.is_valid:
        try:
            hull = make_valid(hull)
        except Exception:
            return None
        hull = _largest_polygon(hull)
    if hull is None or hull.is_empty or hull.area <= AREA_EPS:
        return None
    return hull


def _partition_room_outline_polygon(
    room_partition: dict[str, Any] | None,
) -> Polygon | None:
    if not isinstance(room_partition, dict):
        return None
    outline = room_partition.get("room_outline") or []
    poly = _poly_xz_from_3d(outline)
    if poly is None or poly.area <= AREA_EPS:
        return None
    return poly


def _room_base_y(room: dict[str, Any]) -> float:
    floor_polygon = room.get("floor_polygon") or []
    if floor_polygon:
        return _avg_y(floor_polygon)
    bottom_ys: list[float] = []
    for wall in _room_walls(room):
        corners = [
            corner
            for corner in (wall.get("corners") or [])
            if isinstance(corner, (list, tuple)) and len(corner) >= 3
        ]
        if len(corners) < 2:
            continue
        min_y = min(float(corner[1]) for corner in corners)
        bottom_ys.extend(
            float(corner[1]) for corner in corners if float(corner[1]) <= min_y + 0.02
        )
    if bottom_ys:
        return round(float(statistics.median(bottom_ys)), 6)
    return 0.0


def _partition_group_key(atom: dict[str, Any]) -> tuple[str, str, str, float | None]:
    kind = str(atom.get("kind") or "flat")
    hypothesis_id = str(atom.get("roof_hypothesis_id") or "")
    flat_role = str(atom.get("flat_role") or "")
    top_y = atom.get("top_y_m")
    rounded_top_y = (
        round(float(top_y), 3)
        if isinstance(top_y, (int, float)) and kind == "flat"
        else None
    )
    return kind, hypothesis_id, flat_role, rounded_top_y


def _merged_partition_regions(room_partition: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float | None], dict[str, Any]] = {}
    for atom in room_partition.get("partitions") or []:
        poly = _poly_xz_from_3d(atom.get("poly") or [])
        if poly is None or poly.area <= AREA_EPS:
            continue
        key = _partition_group_key(atom)
        group = groups.setdefault(
            key,
            {
                "surface": {
                    "kind": str(atom.get("kind") or "flat"),
                    "corners": atom.get("poly") or [],
                    "roof_hypothesis_id": atom.get("roof_hypothesis_id"),
                },
                "atom_ids": [],
                "polys": [],
                "atoms": [],
            },
        )
        atom_id = str(atom.get("id") or "")
        if atom_id:
            group["atom_ids"].append(atom_id)
        group["polys"].append(poly)
        group["atoms"].append(
            {
                "atom_id": atom_id,
                "polygon": poly,
            }
        )

    merged_regions: list[dict[str, Any]] = []
    for group in groups.values():
        try:
            union_poly = unary_union(group["polys"])
        except Exception:
            union_poly = None
        union_suspect = union_poly is None or getattr(union_poly, "is_empty", True)
        if not union_suspect:
            for atom in group["atoms"]:
                try:
                    missing_area = float(atom["polygon"].difference(union_poly).area)
                except Exception:
                    missing_area = float(atom["polygon"].area)
                if missing_area > AREA_EPS:
                    union_suspect = True
                    break
        if union_suspect:
            for atom in group["atoms"]:
                piece = atom["polygon"]
                if piece is None or piece.is_empty or piece.area <= AREA_EPS:
                    continue
                merged_regions.append(
                    {
                        "surface": dict(group["surface"]),
                        "atom_ids": [atom["atom_id"]] if atom["atom_id"] else [],
                        "polygon": piece,
                    }
                )
            continue
        for piece in _decompose_polygons(union_poly):
            if not piece.is_valid:
                try:
                    piece = make_valid(piece)
                except Exception:
                    continue
                if not isinstance(piece, Polygon):
                    piece = _largest_polygon(piece)
            if piece is None or piece.is_empty or piece.area <= AREA_EPS:
                continue
            merged_regions.append(
                {
                    "surface": dict(group["surface"]),
                    "atom_ids": sorted(set(group["atom_ids"])),
                    "polygon": piece,
                }
            )
    return merged_regions


def _wall_face_xz_edge(face: dict[str, Any]) -> LineString | None:
    corners = face.get("corners") or []
    unique: list[tuple[float, float]] = []
    for corner in corners:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            continue
        xz = (round(float(corner[0]), 6), round(float(corner[2]), 6))
        if xz not in unique:
            unique.append(xz)
    if len(unique) < 2:
        return None
    line = LineString([unique[0], unique[1]])
    return line if line.length > EPS else None


def _boundary_overlap_length(
    face: dict[str, Any], boundary_geom: Polygon | LineString | None
) -> float:
    if boundary_geom is None:
        return 0.0
    wall_edge = _wall_face_xz_edge(face)
    if wall_edge is None:
        return 0.0
    try:
        boundary = (
            boundary_geom.boundary
            if isinstance(boundary_geom, Polygon)
            else boundary_geom
        )
        overlap = wall_edge.intersection(boundary)
    except Exception:
        return 0.0
    return float(getattr(overlap, "length", 0.0) or 0.0)


def _annotate_boundary_classes(
    cell: dict[str, Any], room_poly: Polygon, story_union: Polygon | None
) -> None:
    for face in cell.get("faces") or []:
        metadata = dict(face.get("metadata") or {})
        kind = str(face.get("kind") or "")
        role = str(face.get("role") or "")
        boundary_class: str
        if kind == "bottom":
            boundary_class = "floor"
        elif kind == "top":
            boundary_class = "ceiling"
        elif kind == "side":
            room_overlap = _boundary_overlap_length(face, room_poly)
            story_overlap = _boundary_overlap_length(face, story_union)
            if story_overlap > EPS:
                boundary_class = "exterior_wall"
            elif room_overlap > EPS:
                boundary_class = "interior_wall"
            elif role == "wall" and bool(metadata.get("perimeter_facing")):
                boundary_class = "splitter"
            elif role == "wall":
                boundary_class = "interior_wall"
            else:
                boundary_class = "splitter"
        else:
            boundary_class = "splitter"
        metadata["boundary_class"] = boundary_class
        face["metadata"] = metadata


def _synthetic_surface_from_polygon(poly: Polygon, *, top_y: float) -> dict[str, Any]:
    corners = [
        [float(x), float(top_y), float(z)]
        for x, z, *_ in list(poly.exterior.coords)[:-1]
    ]
    return {
        "kind": "flat",
        "corners": corners,
        "roof_hypothesis_id": None,
    }


def build_occupied_room_cell_complex(
    *,
    bldg: dict[str, Any],
    room_partitions: list[dict[str, Any]],
    building_part_graph: dict[str, Any],
) -> dict[str, Any]:
    room_partitions_by_index = {
        int(room_partition["room_index"]): room_partition
        for room_partition in room_partitions
        if isinstance(room_partition, dict)
        and isinstance(room_partition.get("room_index"), int)
    }
    room_membership = building_part_graph.get("room_membership") or {}
    story_unions = _story_unions(bldg)

    cells: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    covered_room_ids: set[str] = set()
    atom_bound_cell_count = 0
    fallback_cell_count = 0
    synthetic_atom_cell_count = 0

    for room_index, room in enumerate(bldg.get("rooms") or []):
        room_id = _room_key(room_index)
        story = int(room.get("story", 0) or 0)
        room_partition = room_partitions_by_index.get(room_index)
        room_poly = _partition_room_outline_polygon(
            room_partition
        ) or _room_footprint_polygon(room)
        if room_poly is None or room_poly.area <= AREA_EPS:
            continue
        base_y = _room_base_y(room)
        part_ids = [
            str(part_id) for part_id in (room_membership.get(room_id) or []) if part_id
        ]
        part_id = part_ids[0] if part_ids else None
        story_union = story_unions.get(story)
        room_cell_count_before = len(cells)
        merged_regions = _merged_partition_regions(room_partition or {})
        covered_polys: list[Polygon] = []
        for merged_region in merged_regions:
            surface = dict(merged_region["surface"])
            atom_ids = list(merged_region["atom_ids"])
            merged_poly = merged_region["polygon"]
            region_built = False
            for convex_index, convex_region in enumerate(
                _convex_subregions(merged_poly)
            ):
                if convex_region.is_empty or convex_region.area <= AREA_EPS:
                    continue
                perimeter_side_face_indices = _perimeter_side_face_indices(
                    convex_region, room_poly
                )
                candidate_footprints = _candidate_footprints(convex_region)

                def _build(
                    footprint,
                    atom_ids=atom_ids,
                    convex_index=convex_index,
                    room_id=room_id,
                    room_index=room_index,
                    part_id=part_id,
                    story=story,
                    base_y=base_y,
                    surface=surface,
                    perimeter_side_face_indices=perimeter_side_face_indices,
                ):
                    cell_id_hash = _stable_hash(
                        [
                            room_id,
                            ":".join(atom_ids),
                            str(convex_index),
                            str(footprint),
                        ],
                        20,
                    )
                    return build_arranged_polyhedral_cell(
                        cell_id=f"occupied-cell:{cell_id_hash}",
                        room_id=room_id,
                        room_index=room_index,
                        part_id=part_id,
                        story=story,
                        base_atom_id=(
                            atom_ids[0] if atom_ids else f"fallback:{room_id}"
                        ),
                        cell_kind="occupied_room",
                        region_footprint=footprint,
                        base_y=base_y,
                        top_y_at=lambda x, z, top_surface=surface: _surface_y_at(
                            top_surface, x, z
                        ),
                        top_surface_kind=surface["kind"],
                        roof_hypothesis_id=surface["roof_hypothesis_id"],
                        perimeter_side_face_indices=perimeter_side_face_indices,
                    )

                cell = _build_best_candidate_cell(
                    candidate_footprints=candidate_footprints,
                    build_cell=_build,
                )
                if cell is not None:
                    region_built = True
                    atom_bound_cell_count += 1
                    covered_room_ids.add(room_id)
                    cell["top_boundary_atom_id"] = atom_ids[0] if atom_ids else None
                    cell["top_boundary_atom_ids"] = atom_ids
                    cell["ceiling_partition_kind"] = surface["kind"]
                    cell["exact_source_kind"] = "top_boundary_atom"
                    _annotate_boundary_classes(cell, room_poly, story_union)
                    cells.append(cell)
                    for atom_id in atom_ids:
                        edges.append(
                            {
                                "id": f"edge:occupied-atom:{
                                    _stable_hash(
                                        [atom_id, cell['id']],
                                        20,
                                    )
                                }",
                                "type": "BOUNDS_OCCUPIED_CELL",
                                "from": atom_id,
                                "to": cell["id"],
                            }
                        )
            if region_built:
                covered_polys.append(merged_poly)
        remainder = room_poly
        if covered_polys:
            try:
                remainder = room_poly.difference(unary_union(covered_polys))
            except Exception:
                remainder = room_poly

        top_y = _room_fallback_top_y(room, base_y)
        if top_y <= base_y + 0.05:
            continue
        for remainder_index, remainder_piece in enumerate(
            _decompose_polygons(remainder)
        ):
            if remainder_piece.is_empty or remainder_piece.area <= AREA_EPS:
                continue
            for convex_index, convex_region in enumerate(
                _convex_subregions(remainder_piece)
            ):
                if convex_region.is_empty or convex_region.area <= AREA_EPS:
                    continue
                synthetic_atom_id = f"implicit-flat-atom:{
                    _stable_hash(
                        [room_id, str(remainder_index), str(convex_index), 'remainder'],
                        20,
                    )
                }"
                synthetic_surface = _synthetic_surface_from_polygon(
                    convex_region, top_y=top_y
                )
                perimeter_side_face_indices = _perimeter_side_face_indices(
                    convex_region, room_poly
                )
                candidate_footprints = _candidate_footprints(convex_region)

                def _build_remainder(
                    footprint,
                    remainder_index=remainder_index,
                    convex_index=convex_index,
                    room_id=room_id,
                    room_index=room_index,
                    part_id=part_id,
                    story=story,
                    synthetic_atom_id=synthetic_atom_id,
                    base_y=base_y,
                    top_y=top_y,
                    perimeter_side_face_indices=perimeter_side_face_indices,
                ):
                    cell_id_hash = _stable_hash(
                        [
                            room_id,
                            "implicit-flat",
                            str(remainder_index),
                            str(convex_index),
                            str(footprint),
                        ],
                        20,
                    )
                    return build_arranged_polyhedral_cell(
                        cell_id=f"occupied-cell:{cell_id_hash}",
                        room_id=room_id,
                        room_index=room_index,
                        part_id=part_id,
                        story=story,
                        base_atom_id=synthetic_atom_id,
                        cell_kind="occupied_room",
                        region_footprint=footprint,
                        base_y=base_y,
                        top_y_at=lambda _x, _z, y=top_y: y,
                        top_surface_kind="flat",
                        roof_hypothesis_id=None,
                        perimeter_side_face_indices=perimeter_side_face_indices,
                    )

                cell = _build_best_candidate_cell(
                    candidate_footprints=candidate_footprints,
                    build_cell=_build_remainder,
                )
                if cell is not None:
                    atom_bound_cell_count += 1
                    synthetic_atom_cell_count += 1
                    covered_room_ids.add(room_id)
                    cell["top_boundary_atom_id"] = synthetic_atom_id
                    cell["top_boundary_atom_ids"] = [synthetic_atom_id]
                    cell["ceiling_partition_kind"] = "flat"
                    cell["exact_source_kind"] = "synthetic_top_boundary_atom"
                    cell["synthetic_top_boundary_surface"] = synthetic_surface
                    _annotate_boundary_classes(cell, room_poly, story_union)
                    cells.append(cell)
                    edges.append(
                        {
                            "id": f"edge:occupied-atom:{
                                _stable_hash(
                                    [synthetic_atom_id, cell['id']],
                                    20,
                                )
                            }",
                            "type": "BOUNDS_OCCUPIED_CELL",
                            "from": synthetic_atom_id,
                            "to": cell["id"],
                        }
                    )

        if len(cells) > room_cell_count_before:
            continue

        for convex_index, convex_region in enumerate(_convex_subregions(room_poly)):
            if convex_region.is_empty or convex_region.area <= AREA_EPS:
                continue
            synthetic_atom_id = f"implicit-flat-atom:{
                _stable_hash(
                    [room_id, 'whole-room', str(convex_index)],
                    20,
                )
            }"
            synthetic_surface = _synthetic_surface_from_polygon(
                convex_region, top_y=top_y
            )
            perimeter_side_face_indices = _perimeter_side_face_indices(
                convex_region, room_poly
            )
            candidate_footprints = _candidate_footprints(convex_region)

            def _build_whole(
                footprint,
                convex_index=convex_index,
                room_id=room_id,
                room_index=room_index,
                part_id=part_id,
                story=story,
                synthetic_atom_id=synthetic_atom_id,
                base_y=base_y,
                top_y=top_y,
                perimeter_side_face_indices=perimeter_side_face_indices,
            ):
                cell_id_hash = _stable_hash(
                    [
                        room_id,
                        "implicit-flat-whole-room",
                        str(convex_index),
                        str(footprint),
                    ],
                    20,
                )
                return build_arranged_polyhedral_cell(
                    cell_id=f"occupied-cell:{cell_id_hash}",
                    room_id=room_id,
                    room_index=room_index,
                    part_id=part_id,
                    story=story,
                    base_atom_id=synthetic_atom_id,
                    cell_kind="occupied_room",
                    region_footprint=footprint,
                    base_y=base_y,
                    top_y_at=lambda _x, _z, y=top_y: y,
                    top_surface_kind="flat",
                    roof_hypothesis_id=None,
                    perimeter_side_face_indices=perimeter_side_face_indices,
                )

            cell = _build_best_candidate_cell(
                candidate_footprints=candidate_footprints,
                build_cell=_build_whole,
            )
            if cell is not None:
                atom_bound_cell_count += 1
                synthetic_atom_cell_count += 1
                covered_room_ids.add(room_id)
                cell["top_boundary_atom_id"] = synthetic_atom_id
                cell["top_boundary_atom_ids"] = [synthetic_atom_id]
                cell["ceiling_partition_kind"] = "flat"
                cell["exact_source_kind"] = "synthetic_top_boundary_atom"
                cell["synthetic_top_boundary_surface"] = synthetic_surface
                _annotate_boundary_classes(cell, room_poly, story_union)
                cells.append(cell)
                edges.append(
                    {
                        "id": f"edge:occupied-atom:{
                            _stable_hash(
                                [synthetic_atom_id, cell['id']],
                                20,
                            )
                        }",
                        "type": "BOUNDS_OCCUPIED_CELL",
                        "from": synthetic_atom_id,
                        "to": cell["id"],
                    }
                )

    face_class_counts: dict[str, int] = defaultdict(int)
    for cell in cells:
        for face in cell.get("faces") or []:
            boundary_class = str(
                (face.get("metadata") or {}).get("boundary_class") or ""
            )
            if boundary_class:
                face_class_counts[boundary_class] += 1

    return {
        "cells": cells,
        "edges": edges,
        "metadata": {
            "backend": "exact_lattice_room_shell_arrangement_v1",
            "numeric_model": "fixed_point_mm",
            "exact_on_lattice": True,
            "polyhedral_kernel": "halfspace_triple_intersection",
            "cell_count": len(cells),
            "room_count": len(covered_room_ids),
            "atom_bound_cell_count": atom_bound_cell_count,
            "fallback_cell_count": fallback_cell_count,
            "synthetic_atom_cell_count": synthetic_atom_cell_count,
            "face_class_counts": dict(sorted(face_class_counts.items())),
        },
    }
