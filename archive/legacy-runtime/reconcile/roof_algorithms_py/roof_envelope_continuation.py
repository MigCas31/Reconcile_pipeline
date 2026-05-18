from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .graph_utils import room_key as _room_key
from .roof_cell_complex import _avg_y, _poly_xz_from_3d, _surface_y_at

EPS = 1e-6
AREA_EPS = 0.01
MIN_CONTINUATION_CLEARANCE_M = 0.12
MAX_CONTINUATION_DISTANCE_M = 4.0


def _poly_xz(corners: list) -> Polygon | None:
    poly = _poly_xz_from_3d(corners)
    if poly is None or poly.is_empty or poly.area <= AREA_EPS:
        return None
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda geom: geom.area)
        except Exception:
            return None
    return poly if isinstance(poly, Polygon) else None


def _decompose_polys(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [
            item
            for item in geom.geoms
            if isinstance(item, Polygon) and not item.is_empty
        ]
    return [
        item
        for item in getattr(geom, "geoms", [])
        if isinstance(item, Polygon) and not item.is_empty
    ]


def _build_room_adjacency(exposed_rooms: list[dict[str, Any]]) -> dict[int, set[int]]:
    polys: dict[int, Polygon] = {}
    for room in exposed_rooms:
        poly = _poly_xz(room.get("fp") or [])
        if poly is not None:
            polys[int(room["room_index"])] = poly

    adjacency: dict[int, set[int]] = {room_index: set() for room_index in polys}
    room_indices = sorted(polys)
    for idx, left_room in enumerate(room_indices):
        left_poly = polys[left_room]
        for right_room in room_indices[idx + 1 :]:
            right_poly = polys[right_room]
            try:
                is_adjacent = (
                    left_poly.distance(right_poly) <= 0.25
                    or left_poly.boundary.intersection(right_poly.boundary).length
                    > 0.05
                )
            except Exception:
                is_adjacent = False
            if is_adjacent:
                adjacency[left_room].add(right_room)
                adjacency[right_room].add(left_room)
    return adjacency


def _continuation_stats(hypothesis_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for node in hypothesis_graph.get("nodes") or []:
        if node.get("type") != "RoofHypothesis":
            continue
        hypothesis_id = str(node["id"])
        stats[hypothesis_id] = {
            "support_score": float(node.get("support_score", 0.0) or 0.0),
            "continuation_component_size": int(
                node.get("continuation_component_size", 1) or 1
            ),
            "exact_incidence_edges": 0,
        }
    for edge in hypothesis_graph.get("edges") or []:
        if edge.get("type") != "CONTINUES_AS":
            continue
        if not bool((edge.get("evidence") or {}).get("exact_face_incidence")):
            continue
        left = str(edge.get("from"))
        right = str(edge.get("to"))
        if left in stats:
            stats[left]["exact_incidence_edges"] += 1
        if right in stats:
            stats[right]["exact_incidence_edges"] += 1
    return stats


def _exact_incidence_pairs_by_hypothesis(
    hypothesis_graph: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    pairs_by_hypothesis: dict[str, dict[str, Any]] = {}
    for edge in hypothesis_graph.get("edges") or []:
        if edge.get("type") != "CONTINUES_AS":
            continue
        evidence = edge.get("evidence") or {}
        if not bool(evidence.get("exact_face_incidence")):
            continue
        hypothesis_id = str(edge.get("from") or "")
        if not hypothesis_id:
            continue
        record = pairs_by_hypothesis.setdefault(
            hypothesis_id,
            {
                "atom_ids": set(),
                "atom_pairs": set(),
            },
        )
        for pair in evidence.get("partition_atom_pairs") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            left = str(pair[0] or "")
            right = str(pair[1] or "")
            if not left or not right:
                continue
            record["atom_ids"].add(left)
            record["atom_ids"].add(right)
            record["atom_pairs"].add(tuple(sorted((left, right))))
    out: dict[str, dict[str, Any]] = {}
    for hypothesis_id, record in pairs_by_hypothesis.items():
        out[hypothesis_id] = {
            "atom_ids": sorted(record["atom_ids"]),
            "atom_pairs": [list(pair) for pair in sorted(record["atom_pairs"])],
        }
    return out


def _lift_polygon_to_surface(
    poly: Polygon, surface: dict[str, Any]
) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    corners: list[list[float]] = []
    for x, z, *_ in coords:
        fx = round(float(x), 6)
        fz = round(float(z), 6)
        corners.append([fx, round(_surface_y_at(surface, fx, fz), 6), fz])
    return corners


def continue_roof_envelopes(
    *,
    exposed_rooms: list[dict[str, Any]],
    room_partitions: list[dict[str, Any]],
    selected_oblique_surfaces: list[dict[str, Any]],
    hypothesis_graph: dict[str, Any],
    building_part_graph: dict[str, Any],
    roof_coverage_graph: dict[str, Any],
) -> dict[str, Any]:
    room_by_index = {int(room["room_index"]): room for room in exposed_rooms}
    room_partitions_by_index = {
        int(room["room_index"]): room for room in room_partitions
    }
    room_adjacency = _build_room_adjacency(exposed_rooms)
    room_membership = building_part_graph.get("room_membership") or {}
    part_nodes = {
        str(node["id"]): node
        for node in (building_part_graph.get("nodes") or [])
        if isinstance(node, dict)
    }
    covered_rooms_by_hypothesis: dict[str, set[int]] = defaultdict(set)
    subparts_by_id = {
        str(subpart["id"]): subpart
        for subpart in (roof_coverage_graph.get("subparts") or [])
        if isinstance(subpart, dict)
    }
    room_subpart_membership = roof_coverage_graph.get("room_subpart_membership") or {}
    atom_subpart_membership = roof_coverage_graph.get("atom_subpart_membership") or {}
    coverage_by_atom = roof_coverage_graph.get("atom_coverage") or {}
    for subpart in roof_coverage_graph.get("subparts") or []:
        hypothesis_id = str(subpart.get("roof_hypothesis_id") or "")
        if not hypothesis_id:
            continue
        for room_index in subpart.get("room_indices") or []:
            if isinstance(room_index, int):
                covered_rooms_by_hypothesis[hypothesis_id].add(room_index)

    continuation_stats = _continuation_stats(hypothesis_graph)
    exact_incidence_by_hypothesis = _exact_incidence_pairs_by_hypothesis(
        hypothesis_graph
    )
    surfaces_by_hypothesis: dict[str, list[tuple[int, dict[str, Any], Polygon]]] = (
        defaultdict(list)
    )
    for index, surface in enumerate(selected_oblique_surfaces):
        hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
        poly = _poly_xz(surface.get("corners") or [])
        if not hypothesis_id or poly is None:
            continue
        surfaces_by_hypothesis[hypothesis_id].append((index, surface, poly))

    covered_atoms_by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for room_partition in room_partitions:
        room_index = int(room_partition["room_index"])
        for atom in room_partition.get("partitions") or []:
            atom_id = str(atom.get("id"))
            atom_poly = _poly_xz(atom.get("poly") or [])
            if atom_poly is None:
                continue
            coverage = coverage_by_atom.get(atom_id) or {}
            sloped_state = str(coverage.get("sloped_state", "none"))
            sloped_hypothesis_id = str(coverage.get("sloped_hypothesis_id") or "")
            if sloped_state not in {"confirmed", "partial"} or not sloped_hypothesis_id:
                continue
            covered_atoms_by_hypothesis[sloped_hypothesis_id].append(
                {
                    "atom_id": atom_id,
                    "room_index": room_index,
                    "poly": atom_poly,
                    "kind": str(atom.get("kind", "flat")),
                    "subpart_ids": [
                        str(subpart_id)
                        for subpart_id in atom_subpart_membership.get(atom_id, [])
                    ],
                }
            )

    augmented_graph = deepcopy(hypothesis_graph)
    selected_room_assignments = {
        room_id: list(assignments)
        for room_id, assignments in (
            augmented_graph.get("selected_room_assignments") or {}
        ).items()
    }
    augmented_graph["selected_room_assignments"] = selected_room_assignments
    graph_edges = list(augmented_graph.get("edges") or [])

    continuation_proposals: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    continued_surfaces: list[dict[str, Any]] = []
    continuation_regions: list[dict[str, Any]] = []
    continued_room_indices: set[int] = set()

    for room_index, room_partition in room_partitions_by_index.items():
        room_id = _room_key(room_index)
        room = room_by_index.get(room_index) or {}
        room_slant = float(room.get("wallTopY", 0.0) - room.get("wallTopMin", 0.0))
        if room_slant < 0.6:
            continue
        if any(
            partition.get("kind") == "oblique"
            for partition in (room_partition.get("partitions") or [])
        ):
            continue
        part_ids = [str(part_id) for part_id in room_membership.get(room_id, [])]
        if not part_ids:
            continue
        part_id = part_ids[0]
        part = part_nodes.get(part_id) or {}
        part_subparts = [
            subpart
            for subpart in (roof_coverage_graph.get("subparts") or [])
            if part_id in [str(value) for value in (subpart.get("part_ids") or [])]
        ]
        part_has_structural_slope_semantics = any(
            str(subpart.get("semantic_kind", "slope_run"))
            in {"gable_run", "l_t_branch"}
            for subpart in part_subparts
        )
        if (
            str(part.get("roof_family_guess", "unknown")) != "gable_or_multi_slope"
            and not part_has_structural_slope_semantics
        ):
            continue

        for atom in room_partition.get("partitions") or []:
            if atom.get("kind") != "flat":
                continue
            atom_poly = _poly_xz(atom.get("poly") or [])
            if atom_poly is None:
                continue
            atom_area = float(atom_poly.area)
            room_poly = _poly_xz(room.get("fp") or [])
            if room_poly is None or room_poly.area <= AREA_EPS:
                continue
            rep = atom_poly.representative_point()
            atom_base_y = _avg_y(atom.get("poly") or [])

            best: dict[str, Any] | None = None
            for hypothesis_id, surfaces in surfaces_by_hypothesis.items():
                if part_id not in {
                    str(v)
                    for v in (
                        building_part_graph.get("hypothesis_membership") or {}
                    ).get(hypothesis_id, [])
                }:
                    continue
                stats = continuation_stats.get(hypothesis_id) or {}
                exact_incidence = exact_incidence_by_hypothesis.get(hypothesis_id) or {}
                exact_atom_ids = {
                    str(atom_id) for atom_id in (exact_incidence.get("atom_ids") or [])
                }
                adjacent_covered = sorted(
                    room_index_value
                    for room_index_value in room_adjacency.get(room_index, set())
                    if room_index_value
                    in covered_rooms_by_hypothesis.get(hypothesis_id, set())
                )
                if not adjacent_covered:
                    continue
                semantic_kinds = {
                    str(subparts_by_id[subpart_id].get("semantic_kind", "slope_run"))
                    for room_key in (
                        _room_key(room_idx) for room_idx in adjacent_covered
                    )
                    for subpart_id in room_subpart_membership.get(room_key, [])
                    if subpart_id in subparts_by_id
                    and subparts_by_id[subpart_id].get("roof_hypothesis_id")
                    == hypothesis_id
                }
                source_atoms = []
                for source_atom in covered_atoms_by_hypothesis.get(hypothesis_id, []):
                    if source_atom["room_index"] not in adjacent_covered:
                        continue
                    if (
                        exact_atom_ids
                        and str(source_atom["atom_id"]) not in exact_atom_ids
                    ):
                        continue
                    gap_distance = float(atom_poly.distance(source_atom["poly"]))
                    try:
                        shared_boundary = float(
                            atom_poly.boundary.intersection(
                                source_atom["poly"].boundary
                            ).length
                        )
                    except Exception:
                        shared_boundary = 0.0
                    if shared_boundary <= 0.05 and gap_distance > 0.25:
                        continue
                    source_atoms.append(
                        {
                            **source_atom,
                            "gap_distance": gap_distance,
                            "shared_boundary_m": shared_boundary,
                        }
                    )
                    for subpart_id in source_atom.get("subpart_ids") or []:
                        if subpart_id in subparts_by_id:
                            semantic_kinds.add(
                                str(
                                    subparts_by_id[subpart_id].get(
                                        "semantic_kind", "slope_run"
                                    )
                                )
                            )

                nearest = min(
                    surfaces,
                    key=lambda item: atom_poly.distance(item[2]),
                )
                source_index, source_surface, source_poly = nearest
                source_story = int(
                    source_surface.get(
                        "story",
                        source_surface.get(
                            "dominant_story", room_partition.get("story", 0)
                        ),
                    )
                    or 0
                )
                if source_story != int(room_partition.get("story", 0)):
                    continue

                distance = float(atom_poly.distance(source_poly))
                if distance > MAX_CONTINUATION_DISTANCE_M:
                    continue
                clearance = float(
                    _surface_y_at(source_surface, float(rep.x), float(rep.y))
                    - atom_base_y
                )
                if clearance <= MIN_CONTINUATION_CLEARANCE_M:
                    continue

                support_score = float(stats.get("support_score", 0.0))
                if support_score < 0.45:
                    continue
                exact_incidence_edges = int(stats.get("exact_incidence_edges", 0))
                component_size = int(stats.get("continuation_component_size", 1))
                exact_atom_pairs = list(exact_incidence.get("atom_pairs") or [])
                exact_pair_count = len(exact_atom_pairs)
                source_incidence_score = 0.0
                if source_atoms:
                    source_incidence_score = max(
                        min(0.6, 0.2 * atom_data["shared_boundary_m"])
                        + max(0.0, 0.2 - atom_data["gap_distance"])
                        for atom_data in source_atoms
                    )
                score = (
                    1.0
                    + min(0.45, 0.12 * len(adjacent_covered))
                    + min(0.45, 0.15 * component_size)
                    + min(0.35, 0.08 * exact_incidence_edges)
                    + min(0.4, 0.3 * clearance)
                    + 0.25 * support_score
                    + source_incidence_score
                    + (0.18 if "gable_run" in semantic_kinds else 0.0)
                    + (0.08 if "l_t_branch" in semantic_kinds else 0.0)
                    - min(1.0, 0.25 * distance)
                )
                if best is None or score > float(best["score"]) + EPS:
                    best = {
                        "hypothesis_id": hypothesis_id,
                        "source_index": source_index,
                        "source_surface": source_surface,
                        "adjacent_covered": adjacent_covered,
                        "clearance": clearance,
                        "distance": distance,
                        "support_score": support_score,
                        "score": score,
                        "source_atoms": source_atoms,
                        "source_incidence_score": source_incidence_score,
                        "semantic_kinds": sorted(semantic_kinds),
                        "exact_incidence_atom_pairs": exact_atom_pairs,
                        "exact_incidence_pair_count": exact_pair_count,
                        "exact_source_atom_ids": sorted(
                            {str(atom_data["atom_id"]) for atom_data in source_atoms}
                        ),
                    }

            if best is None:
                continue

            continuation_proposals[(room_index, str(best["hypothesis_id"]))].append(
                {
                    "room_id": room_id,
                    "room_index": room_index,
                    "room_area": float(room_poly.area),
                    "room_slant": room_slant,
                    "atom_id": str(atom["id"]),
                    "atom_area": atom_area,
                    "atom_poly": atom_poly,
                    "best": best,
                }
            )

    for (room_index, hypothesis_id), proposals in continuation_proposals.items():
        if not proposals:
            continue
        room_id = proposals[0]["room_id"]
        assignments = selected_room_assignments.setdefault(room_id, [])
        if hypothesis_id not in assignments:
            assignments.append(hypothesis_id)
        union_poly = unary_union([proposal["atom_poly"] for proposal in proposals])
        merged_regions = [
            region
            for region in _decompose_polys(union_poly)
            if not region.is_empty and region.area > AREA_EPS
        ]
        best_proposal = max(
            proposals, key=lambda proposal: float(proposal["best"]["score"])
        )
        continued_room_indices.add(room_index)
        for region_index, region in enumerate(merged_regions):
            polygon_xz = [
                [round(float(x), 6), round(float(z), 6)]
                for x, z, *_ in list(region.exterior.coords)[:-1]
            ]
            lifted_polygon = _lift_polygon_to_surface(
                region, best_proposal["best"]["source_surface"]
            )
            exact_pair_count = int(
                best_proposal["best"].get("exact_incidence_pair_count", 0) or 0
            )
            continuation_regions.append(
                {
                    "id": (
                        f"continuation-region:{hypothesis_id}:{room_id}:{region_index}"
                    ),
                    "roof_hypothesis_id": hypothesis_id,
                    "room_id": room_id,
                    "room_index": room_index,
                    "continuation_mode": "arrangement_face"
                    if exact_pair_count > 0
                    else "arrangement_region",
                    "polygon_xz": polygon_xz,
                    "polygon": lifted_polygon,
                    "source_surface_index": best_proposal["best"]["source_index"],
                    "source_surface_kind": str(
                        best_proposal["best"]["source_surface"].get("kind") or "oblique"
                    ),
                    "source_atom_ids": list(
                        best_proposal["best"].get("exact_source_atom_ids") or []
                    ),
                    "target_atom_ids": [proposal["atom_id"] for proposal in proposals],
                    "exact_incidence_atom_pairs": list(
                        best_proposal["best"].get("exact_incidence_atom_pairs") or []
                    ),
                    "exact_incidence_pair_count": exact_pair_count,
                    "adjacent_room_indices": list(
                        best_proposal["best"]["adjacent_covered"]
                    ),
                    "distance_m": round(float(best_proposal["best"]["distance"]), 6),
                    "clearance_m": round(float(best_proposal["best"]["clearance"]), 6),
                    "continuation_score": round(
                        float(best_proposal["best"]["score"]), 6
                    ),
                    "hypothesis_support_score": round(
                        float(best_proposal["best"]["support_score"]), 6
                    ),
                    "semantic_kinds": list(
                        best_proposal["best"].get("semantic_kinds") or []
                    ),
                }
            )
            continued_surface = dict(best_proposal["best"]["source_surface"])
            continued_surface["corners"] = lifted_polygon
            continued_surface["continued_from_surface_index"] = best_proposal["best"][
                "source_index"
            ]
            continued_surface["continuation_source"] = "arrangement_face"
            continued_surface["continuation_room_index"] = room_index
            continued_surface["continuation_region_index"] = region_index
            continued_surface["continuation_atom_id"] = proposals[0]["atom_id"]
            continued_surface["continuation_atom_ids"] = [
                proposal["atom_id"] for proposal in proposals
            ]
            continued_surface["continuation_score"] = round(
                float(best_proposal["best"]["score"]), 6
            )
            continued_surface["continuation_adjacent_room_indices"] = list(
                best_proposal["best"]["adjacent_covered"]
            )
            continued_surface["continuation_distance_m"] = round(
                float(best_proposal["best"]["distance"]), 6
            )
            continued_surface["continuation_clearance_m"] = round(
                float(best_proposal["best"]["clearance"]), 6
            )
            continued_surface["continuation_incidence_score"] = round(
                float(best_proposal["best"].get("source_incidence_score", 0.0)), 6
            )
            continued_surface["continuation_semantic_kinds"] = list(
                best_proposal["best"].get("semantic_kinds") or []
            )
            continued_surface["continuation_exact_incidence_pair_count"] = int(
                best_proposal["best"].get("exact_incidence_pair_count", 0) or 0
            )
            continued_surfaces.append(continued_surface)

        synthetic_edge_score = min(
            1.0,
            max(
                0.68,
                float(best_proposal["best"]["support_score"])
                + 0.18
                + min(0.15, 0.05 * len(best_proposal["best"]["adjacent_covered"]))
                + min(0.12, 0.08 * float(best_proposal["best"]["clearance"])),
            ),
        )
        total_area = sum(float(proposal["atom_area"]) for proposal in proposals)
        graph_edges.append(
            {
                "id": f"edge:continues-room:{hypothesis_id}:{room_id}",
                "type": "COVERS_ROOM",
                "from": hypothesis_id,
                "to": room_id,
                "selected": True,
                "synthetic": True,
                "evidence": {
                    "coverage_ratio": round(
                        total_area / max(float(proposals[0]["room_area"]), AREA_EPS), 6
                    ),
                    "coverage_area_m2": round(total_area, 6),
                    "room_slant_delta_m": round(float(proposals[0]["room_slant"]), 6),
                    "hypothesis_support_score": round(
                        float(best_proposal["best"]["support_score"]), 6
                    ),
                    "edge_score": round(synthetic_edge_score, 6),
                    "relation_state": "continued",
                    "continuation_source": "arrangement_face",
                    "continuation_atom_id": proposals[0]["atom_id"],
                    "continuation_atom_ids": [
                        proposal["atom_id"] for proposal in proposals
                    ],
                    "continuation_semantic_kinds": list(
                        best_proposal["best"].get("semantic_kinds") or []
                    ),
                    "continuation_exact_incidence_pair_count": int(
                        best_proposal["best"].get("exact_incidence_pair_count", 0) or 0
                    ),
                },
            }
        )

    augmented_graph["edges"] = graph_edges
    metadata = dict(augmented_graph.get("metadata") or {})
    metadata["continued_room_count"] = len(continued_room_indices)
    metadata["continued_surface_count"] = len(continued_surfaces)
    metadata["continuation_region_count"] = len(continuation_regions)
    augmented_graph["metadata"] = metadata

    return {
        "selected_oblique_surfaces": list(selected_oblique_surfaces)
        + continued_surfaces,
        "hypothesis_graph": augmented_graph,
        "continued_surfaces": continued_surfaces,
        "continuation_regions": continuation_regions,
        "continued_room_indices": sorted(continued_room_indices),
        "metadata": {
            "continued_room_count": len(continued_room_indices),
            "continued_surface_count": len(continued_surfaces),
            "continuation_region_count": len(continuation_regions),
        },
    }
