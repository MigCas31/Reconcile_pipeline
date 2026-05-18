from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.geometry import GeometryCollection, Polygon
from shapely.validation import make_valid

from .graph_utils import room_key as _room_key
from .graph_utils import stable_hash as _stable_hash
from .roof_cell_complex import _avg_y, _poly_xz_from_3d, _surface_y_at

EPS = 1e-6
AREA_EPS = 0.01
SEED_ROOM_BUFFER_M = 0.75


def _surface_poly(surface: dict[str, Any]) -> Polygon | None:
    poly = _poly_xz_from_3d(surface.get("corners") or [])
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


def _largest_polygon(geom: Any) -> Polygon | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    if getattr(geom, "geom_type", "") == "MultiPolygon":
        try:
            return max(
                (
                    part
                    for part in geom.geoms
                    if isinstance(part, Polygon) and part.area > AREA_EPS
                ),
                key=lambda part: part.area,
            )
        except ValueError:
            return None
    if isinstance(geom, GeometryCollection):
        best = None
        best_area = 0.0
        for child in geom.geoms:
            poly = _largest_polygon(child)
            if poly is None:
                continue
            area = float(poly.area)
            if area > best_area:
                best = poly
                best_area = area
        return best
    return None


def _angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def build_roof_coverage_graph(
    *,
    room_partitions: list[dict[str, Any]],
    selected_oblique_surfaces: list[dict[str, Any]],
    selected_flat_surfaces: list[dict[str, Any]],
    building_part_graph: dict[str, Any],
) -> dict[str, Any]:
    part_nodes = {
        str(node["id"]): node
        for node in (building_part_graph.get("nodes") or [])
        if isinstance(node, dict)
    }
    room_membership = building_part_graph.get("room_membership") or {}
    hypothesis_membership = building_part_graph.get("hypothesis_membership") or {}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    atom_coverage: dict[str, dict[str, Any]] = {}
    atom_regions: dict[str, dict[str, Any]] = {}
    room_summaries: dict[str, dict[str, Any]] = {}
    room_polys_by_hypothesis: dict[str, dict[int, Polygon]] = defaultdict(dict)
    atom_polys: dict[str, Polygon] = {}
    room_polys: dict[int, Polygon] = {}
    oblique_surface_meta: dict[str, dict[str, Any]] = {}

    envelopes: list[dict[str, Any]] = []
    for envelope_kind, surfaces in (
        ("oblique", selected_oblique_surfaces),
        ("flat", selected_flat_surfaces),
    ):
        for index, surface in enumerate(surfaces):
            polygon = _surface_poly(surface)
            if polygon is None:
                continue
            hypothesis_id = str(
                surface.get("roof_hypothesis_id") or f"{envelope_kind}:{index}"
            )
            envelope_id = f"roof-envelope:{envelope_kind}:{
                _stable_hash(
                    [hypothesis_id, str(index)],
                    20,
                )
            }"
            part_ids = [
                str(part_id) for part_id in hypothesis_membership.get(hypothesis_id, [])
            ]
            envelope = {
                "id": envelope_id,
                "type": "RoofEnvelope",
                "envelope_kind": envelope_kind,
                "roof_hypothesis_id": hypothesis_id,
                "part_ids": part_ids,
                "surface": surface,
                "polygon": polygon,
            }
            envelopes.append(envelope)
            if envelope_kind == "oblique":
                cluster = surface.get("cluster") or {}
                oblique_surface_meta[hypothesis_id] = {
                    "avg_azimuth": float(cluster.get("avgAzimuth", 0.0) or 0.0),
                    "part_ids": part_ids,
                }
            nodes.append(
                {
                    "id": envelope_id,
                    "type": "RoofEnvelope",
                    "envelope_kind": envelope_kind,
                    "roof_hypothesis_id": hypothesis_id,
                    "part_ids": part_ids,
                    "story": surface.get("story", surface.get("dominant_story")),
                }
            )

    for room_partition in room_partitions:
        room_id = _room_key(int(room_partition["room_index"]))
        room_part_ids = [str(part_id) for part_id in room_membership.get(room_id, [])]
        part_id = room_part_ids[0] if room_part_ids else None
        part = part_nodes.get(part_id or "") or {}
        part_family = str(part.get("roof_family_guess", "unknown"))

        room_has_confirmed_sloped = False
        room_has_partial_sloped = False
        room_has_flat_shell = False

        for partition in room_partition.get("partitions") or []:
            atom_id = str(partition["id"])
            atom_poly = _poly_xz_from_3d(partition.get("poly") or [])
            if atom_poly is None:
                continue
            atom_polys[atom_id] = atom_poly
            room_polys[int(partition["room_index"])] = (
                atom_poly
                if int(partition["room_index"]) not in room_polys
                else room_polys[int(partition["room_index"])].union(atom_poly)
            )
            atom_area = max(float(atom_poly.area), AREA_EPS)
            atom_base_y = _avg_y(partition.get("poly") or [])
            rep = atom_poly.representative_point()
            best = {
                "sloped_state": "none",
                "sloped_overlap_ratio": 0.0,
                "sloped_vertical_clearance_m": None,
                "sloped_envelope_id": None,
                "sloped_hypothesis_id": None,
                "sloped_part_match": False,
                "flat_shell_overlap_ratio": 0.0,
                "flat_shell_hypothesis_id": None,
            }

            for envelope in envelopes:
                try:
                    overlap = atom_poly.intersection(envelope["polygon"])
                except Exception:
                    continue
                overlap_area = float(getattr(overlap, "area", 0.0))
                if overlap_area <= AREA_EPS:
                    continue
                overlap_ratio = min(1.0, overlap_area / atom_area)
                same_part = bool(
                    part_id and part_id in (envelope.get("part_ids") or [])
                )

                if envelope["envelope_kind"] == "oblique":
                    vertical_clearance = (
                        _surface_y_at(envelope["surface"], float(rep.x), float(rep.y))
                        - atom_base_y
                    )
                    if vertical_clearance > 0.12:
                        state = "confirmed"
                    elif overlap_ratio >= 0.2 or (
                        same_part and part_family == "gable_or_multi_slope"
                    ):
                        state = "partial"
                    else:
                        state = "weak"
                    if {"none": 0, "weak": 1, "partial": 2, "confirmed": 3}[state] > {
                        "none": 0,
                        "weak": 1,
                        "partial": 2,
                        "confirmed": 3,
                    }[best["sloped_state"]] or (
                        state == best["sloped_state"]
                        and overlap_ratio > best["sloped_overlap_ratio"] + EPS
                    ):
                        best.update(
                            {
                                "sloped_state": state,
                                "sloped_overlap_ratio": round(overlap_ratio, 6),
                                "sloped_vertical_clearance_m": round(
                                    vertical_clearance, 6
                                ),
                                "sloped_envelope_id": envelope["id"],
                                "sloped_hypothesis_id": envelope["roof_hypothesis_id"],
                                "sloped_part_match": same_part,
                            }
                        )
                    edges.append(
                        {
                            "id": f"edge:covers-atom:{
                                _stable_hash(
                                    [envelope['id'], atom_id],
                                    20,
                                )
                            }",
                            "type": "COVERS_ATOM",
                            "from": envelope["id"],
                            "to": atom_id,
                            "evidence": {
                                "coverage_kind": "sloped_envelope",
                                "relation_state": state,
                                "overlap_area_m2": round(overlap_area, 6),
                                "overlap_ratio": round(overlap_ratio, 6),
                                "vertical_clearance_m": round(vertical_clearance, 6),
                                "same_part": same_part,
                            },
                        }
                    )
                else:
                    if overlap_ratio > best["flat_shell_overlap_ratio"] + EPS:
                        best.update(
                            {
                                "flat_shell_overlap_ratio": round(overlap_ratio, 6),
                                "flat_shell_hypothesis_id": envelope[
                                    "roof_hypothesis_id"
                                ],
                            }
                        )

            atom_coverage[atom_id] = best
            atom_regions[atom_id] = {
                "room_index": partition.get("room_index"),
                "story": partition.get("story"),
                "kind": partition.get("kind"),
                "polygon_xz": [
                    [round(float(x), 6), round(float(y), 6)]
                    for x, y in list(atom_poly.exterior.coords)[:-1]
                ],
            }
            if best["sloped_state"] == "confirmed":
                room_has_confirmed_sloped = True
            elif best["sloped_state"] == "partial":
                room_has_partial_sloped = True
            if best["flat_shell_overlap_ratio"] > 0.1:
                room_has_flat_shell = True
            hypothesis_id = best.get("sloped_hypothesis_id")
            if hypothesis_id and best["sloped_state"] in {"confirmed", "partial"}:
                room_index = int(partition["room_index"])
                current = room_polys_by_hypothesis[hypothesis_id].get(room_index)
                room_polys_by_hypothesis[hypothesis_id][room_index] = (
                    atom_poly if current is None else current.union(atom_poly)
                )

        room_summaries[room_id] = {
            "room_index": room_partition["room_index"],
            "story": room_partition["story"],
            "part_ids": room_part_ids,
            "covered_by_sloped_roof": room_has_confirmed_sloped,
            "partially_covered_by_sloped_roof": room_has_partial_sloped,
            "covered_by_flat_shell": room_has_flat_shell,
        }

    coverage_subparts: list[dict[str, Any]] = []
    room_subpart_membership: dict[str, list[str]] = defaultdict(list)
    subpart_records: dict[str, dict[str, Any]] = {}
    seeded_hypothesis_count = 0
    for envelope in envelopes:
        if envelope.get("envelope_kind") != "oblique":
            continue
        hypothesis_id = str(envelope.get("roof_hypothesis_id") or "")
        if not hypothesis_id or room_polys_by_hypothesis.get(hypothesis_id):
            continue
        surface = envelope.get("surface") or {}
        cluster = surface.get("cluster") or {}
        seed_room_indices = sorted(
            {
                int(seg.get("room_idx"))
                for seg in (cluster.get("segs") or [])
                if isinstance(seg, dict) and isinstance(seg.get("room_idx"), int)
            }
        )
        if not seed_room_indices:
            seed_room_indices = sorted(room_polys)
        seeded_any = False
        envelope_parts = {
            str(part_id) for part_id in (envelope.get("part_ids") or []) if part_id
        }
        for room_index in seed_room_indices:
            room_poly = room_polys.get(room_index)
            if room_poly is None:
                continue
            room_parts = {
                str(part_id)
                for part_id in (room_membership.get(_room_key(room_index)) or [])
                if part_id
            }
            if (
                envelope_parts
                and room_parts
                and not envelope_parts.intersection(room_parts)
            ):
                continue
            try:
                clipped = room_poly.intersection(envelope["polygon"])
            except Exception:
                continue
            clipped_poly = _largest_polygon(clipped)
            if clipped_poly is None or clipped_poly.area <= AREA_EPS:
                try:
                    clipped_poly = _largest_polygon(
                        room_poly.buffer(SEED_ROOM_BUFFER_M).intersection(
                            envelope["polygon"]
                        )
                    )
                except Exception:
                    clipped_poly = None
            if clipped_poly is None or clipped_poly.area <= AREA_EPS:
                continue
            room_polys_by_hypothesis[hypothesis_id][room_index] = clipped_poly
            seeded_any = True
        if seeded_any:
            seeded_hypothesis_count += 1
    for hypothesis_id, room_polys in room_polys_by_hypothesis.items():
        room_ids = sorted(room_polys)
        adjacency: dict[int, set[int]] = {room_index: set() for room_index in room_ids}
        for idx, left_room in enumerate(room_ids):
            left_poly = room_polys[left_room]
            for right_room in room_ids[idx + 1 :]:
                right_poly = room_polys[right_room]
                try:
                    touches = left_poly.buffer(0.15).intersects(right_poly.buffer(0.15))
                except Exception:
                    touches = False
                if touches:
                    adjacency[left_room].add(right_room)
                    adjacency[right_room].add(left_room)
        seen: set[int] = set()
        for room_index in room_ids:
            if room_index in seen:
                continue
            stack = [room_index]
            component: list[int] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component.append(current)
                stack.extend(sorted(adjacency.get(current, set()) - seen))
            subpart_id = f"coverage-subpart:{
                _stable_hash(
                    [hypothesis_id, str(sorted(component))],
                    20,
                )
            }"
            part_ids = [
                str(part_id) for part_id in hypothesis_membership.get(hypothesis_id, [])
            ]
            union_poly = None
            for member in component:
                member_poly = room_polys.get(member)
                if member_poly is None:
                    continue
                union_poly = (
                    member_poly if union_poly is None else union_poly.union(member_poly)
                )
            record = {
                "id": subpart_id,
                "roof_hypothesis_id": hypothesis_id,
                "room_indices": sorted(component),
                "part_ids": part_ids,
                "avg_azimuth": (oblique_surface_meta.get(hypothesis_id) or {}).get(
                    "avg_azimuth"
                ),
                "semantic_kind": "slope_run",
                "adjacent_subpart_ids": [],
                "polygon": union_poly,
            }
            coverage_subparts.append(record)
            subpart_records[subpart_id] = record
            for member in component:
                room_subpart_membership[_room_key(member)].append(subpart_id)

    atom_subpart_membership: dict[str, list[str]] = defaultdict(list)
    for atom_id, coverage in atom_coverage.items():
        hypothesis_id = str(coverage.get("sloped_hypothesis_id") or "")
        if not hypothesis_id or str(coverage.get("sloped_state", "none")) not in {
            "confirmed",
            "partial",
        }:
            continue
        room_index = atom_regions.get(atom_id, {}).get("room_index")
        atom_poly = atom_polys.get(atom_id)
        if not isinstance(room_index, int) or atom_poly is None:
            continue
        for subpart in coverage_subparts:
            if subpart.get("roof_hypothesis_id") != hypothesis_id:
                continue
            if room_index not in (subpart.get("room_indices") or []):
                continue
            subpart_poly = subpart.get("polygon")
            intersects = True
            if subpart_poly is not None:
                try:
                    intersects = atom_poly.intersection(subpart_poly).area > AREA_EPS
                except Exception:
                    intersects = True
            if intersects:
                atom_subpart_membership[atom_id].append(str(subpart["id"]))

    subpart_ids = sorted(subpart_records)
    for idx, left_id in enumerate(subpart_ids):
        left = subpart_records[left_id]
        left_poly = left.get("polygon")
        if left_poly is None:
            continue
        left_az = left.get("avg_azimuth")
        left_parts = set(left.get("part_ids") or [])
        semantics: set[str] = set()
        adjacent_ids: list[str] = []
        for right_id in subpart_ids[idx + 1 :]:
            right = subpart_records[right_id]
            right_poly = right.get("polygon")
            if right_poly is None:
                continue
            if not left_parts.intersection(right.get("part_ids") or []):
                continue
            try:
                touches = left_poly.buffer(0.15).intersects(right_poly.buffer(0.15))
            except Exception:
                touches = False
            if not touches:
                continue
            adjacent_ids.append(right_id)
            subpart_records[right_id]["adjacent_subpart_ids"].append(left_id)
            right_az = right.get("avg_azimuth")
            if left_az is None or right_az is None:
                continue
            diff = _angle_diff(float(left_az), float(right_az))
            if 140.0 <= diff <= 220.0:
                semantics.add("gable_run")
                right_semantic = set(
                    [
                        subpart_records[right_id].get("semantic_kind", "slope_run"),
                        "gable_run",
                    ]
                )
                subpart_records[right_id]["semantic_kind"] = (
                    "gable_run" if "gable_run" in right_semantic else "slope_run"
                )
            elif 60.0 <= diff <= 120.0:
                semantics.add("l_t_branch")
                if subpart_records[right_id].get("semantic_kind") != "gable_run":
                    subpart_records[right_id]["semantic_kind"] = "l_t_branch"
        left["adjacent_subpart_ids"].extend(adjacent_ids)
        if "gable_run" in semantics:
            left["semantic_kind"] = "gable_run"
        elif "l_t_branch" in semantics and left.get("semantic_kind") != "gable_run":
            left["semantic_kind"] = "l_t_branch"

    serializable_subparts: list[dict[str, Any]] = []
    for subpart in coverage_subparts:
        poly = subpart.get("polygon")
        poly_for_serialization = _largest_polygon(poly)
        serializable_subparts.append(
            {
                "id": subpart["id"],
                "roof_hypothesis_id": subpart["roof_hypothesis_id"],
                "room_indices": subpart["room_indices"],
                "part_ids": subpart.get("part_ids") or [],
                "semantic_kind": subpart.get("semantic_kind", "slope_run"),
                "adjacent_subpart_ids": sorted(
                    set(subpart.get("adjacent_subpart_ids") or [])
                ),
                "polygon_xz": (
                    [
                        [round(float(x), 6), round(float(y), 6)]
                        for x, y in list(poly_for_serialization.exterior.coords)[:-1]
                    ]
                    if poly_for_serialization is not None
                    else []
                ),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "atom_coverage": atom_coverage,
        "atom_regions": atom_regions,
        "room_summaries": room_summaries,
        "subparts": serializable_subparts,
        "room_subpart_membership": dict(room_subpart_membership),
        "atom_subpart_membership": {
            atom_id: sorted(set(subparts))
            for atom_id, subparts in atom_subpart_membership.items()
        },
        "metadata": {
            "envelope_count": len(nodes),
            "coverage_edge_count": len(edges),
            "confirmed_sloped_atom_count": sum(
                1
                for cov in atom_coverage.values()
                if cov.get("sloped_state") == "confirmed"
            ),
            "partial_sloped_atom_count": sum(
                1
                for cov in atom_coverage.values()
                if cov.get("sloped_state") == "partial"
            ),
            "subpart_count": len(serializable_subparts),
            "seeded_hypothesis_count": seeded_hypothesis_count,
            "gable_subpart_count": sum(
                1
                for subpart in serializable_subparts
                if subpart.get("semantic_kind") == "gable_run"
            ),
            "l_t_subpart_count": sum(
                1
                for subpart in serializable_subparts
                if subpart.get("semantic_kind") == "l_t_branch"
            ),
        },
    }
