from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid

from .graph_support import classify_graph_roof_room
from .graph_utils import room_key as _room_key
from .graph_utils import stable_hash as _stable_hash

EPS = 1e-6
AREA_EPS = 0.01
LATTICE_SCALE_MM = 1000


def _snap(value: float) -> float:
    return round(float(value) * LATTICE_SCALE_MM) / LATTICE_SCALE_MM


def _poly_xz(corners: list) -> Polygon | None:
    points: list[tuple[float, float]] = []
    for corner in corners or []:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            continue
        points.append((_snap(float(corner[0])), _snap(float(corner[2]))))
    if len(points) < 3:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda geom: geom.area)
            elif isinstance(poly, GeometryCollection):
                polys = [
                    geom
                    for geom in poly.geoms
                    if isinstance(geom, Polygon)
                    and not geom.is_empty
                    and geom.area > AREA_EPS
                ]
                poly = max(polys, key=lambda geom: geom.area) if polys else None
        except Exception:
            return None
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= AREA_EPS:
        return None
    return poly


def _candidate_id(kind: str, index: int) -> str:
    return f"roof-hypothesis:{kind}:{index}"


def _candidate_boundary_ids(surface: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(boundary_id)
            for boundary_id in (surface.get("space_boundary_ids") or [])
            if boundary_id
        }
    )


def _room_graph_support(
    graph, story: int, floor_polygon: list
) -> tuple[str | None, float, str]:
    graph_room, decision = classify_graph_roof_room(graph, story, floor_polygon)
    if decision is None:
        return None, 0.5, "unknown"
    if decision["mode"] == "roof_candidate":
        state = decision.get("exposure_state", "unknown")
        return (
            graph_room.id if graph_room is not None else None,
            (1.0 if state == "confirmed" else 0.8),
            state,
        )
    if decision["mode"] == "underdetermined":
        return (
            graph_room.id if graph_room is not None else None,
            0.55,
            decision.get("exposure_state", "partial"),
        )
    return (
        graph_room.id if graph_room is not None else None,
        0.1,
        decision.get("above_state", "confirmed"),
    )


def _candidate_base_score(
    kind: str, surface: dict[str, Any], graph_support: float
) -> float:
    if kind == "oblique":
        cluster = surface.get("cluster") or {}
        seg_count = len(cluster.get("segs") or [])
        room_count = len(set(cluster.get("room_indices") or []))
        score = 0.25
        score += min(0.35, 0.06 * seg_count)
        score += min(0.15, 0.05 * room_count)
        score += 0.25 * graph_support
        return min(1.0, score)

    seg_count = len(surface.get("segments") or [])
    surface_kind = str(surface.get("kind") or "flat")
    score = 0.2 if surface_kind == "top" else 0.3
    score += min(0.2, 0.05 * seg_count)
    score += 0.35 * graph_support
    return min(1.0, score)


def _coverage_score(
    *,
    candidate_score: float,
    coverage_ratio: float,
    room_slant_delta: float,
    kind: str,
) -> float:
    score = candidate_score * (0.45 + 0.55 * max(0.0, min(1.0, coverage_ratio)))
    if kind == "oblique":
        score += min(0.18, room_slant_delta * 0.25)
    else:
        score += max(0.0, 0.12 - room_slant_delta * 0.2)
        score -= min(0.4, max(0.0, room_slant_delta - 0.4) * 0.35)
    return min(1.0, score)


def build_roof_hypothesis_graph(
    *,
    bldg: dict,
    exposed_rooms: list[dict[str, Any]],
    oblique_surfaces: list[dict[str, Any]],
    flat_surfaces: list[dict[str, Any]],
    roof_graph: dict[str, Any] | None,
    graph=None,
) -> dict[str, Any]:
    room_nodes: list[dict[str, Any]] = []
    for room_data in exposed_rooms:
        room_index = int(room_data["room_index"])
        room_key = _room_key(room_index)
        story = int(room_data["story"])
        graph_room_id, graph_support, graph_state = _room_graph_support(
            graph,
            story,
            room_data.get("fp") or [],
        )
        room_node = {
            "id": room_key,
            "type": "RoomTopFace",
            "story": story,
            "room_index": room_index,
            "graph_room_id": graph_room_id,
            "area_m2": round(_poly_xz(room_data.get("fp") or []).area, 6)
            if _poly_xz(room_data.get("fp") or [])
            else 0.0,
            "wall_top_y": round(float(room_data["wallTopY"]), 6),
            "wall_top_min_y": round(float(room_data["wallTopMin"]), 6),
            "slant_delta_m": round(
                float(room_data["wallTopY"] - room_data["wallTopMin"]), 6
            ),
            "graph_relation_state": graph_state,
            "graph_support_score": round(graph_support, 6),
        }
        room_nodes.append(room_node)

    nodes: list[dict[str, Any]] = list(room_nodes)
    edges: list[dict[str, Any]] = []

    candidate_nodes: list[dict[str, Any]] = []
    candidates_by_id: dict[str, dict[str, Any]] = {}
    boundary_to_candidate: dict[str, set[str]] = defaultdict(set)

    for kind, surfaces in (("oblique", oblique_surfaces), ("flat", flat_surfaces)):
        for index, surface in enumerate(surfaces):
            story = int(surface.get("dominant_story", surface.get("story", 0)) or 0)
            polygon = _poly_xz(surface.get("corners") or [])
            if polygon is None:
                continue

            support_room_indices: list[int]
            if kind == "oblique":
                support_room_indices = sorted(
                    set((surface.get("cluster") or {}).get("room_indices") or [])
                )
            else:
                room_index = surface.get("room_index")
                support_room_indices = (
                    [int(room_index)] if isinstance(room_index, int) else []
                )

            support_scores = []
            support_states = []
            graph_room_ids: list[str] = []
            for room_index in support_room_indices:
                rooms = bldg.get("rooms", [])
                if room_index < 0 or room_index >= len(rooms):
                    continue
                room = rooms[room_index] or {}
                graph_room_id, support_score, graph_state = _room_graph_support(
                    graph,
                    int(room.get("story", story)),
                    room.get("floor_polygon") or [],
                )
                support_scores.append(support_score)
                support_states.append(graph_state)
                if graph_room_id:
                    graph_room_ids.append(graph_room_id)

            graph_support = (
                sum(support_scores) / len(support_scores) if support_scores else 0.5
            )
            node_id = _candidate_id(kind, index)
            node = {
                "id": node_id,
                "type": "RoofHypothesis",
                "surface_kind": kind,
                "source_kind": str(surface.get("kind") or kind),
                "story": story,
                "source_surface_index": index,
                "source_boundary_ids": _candidate_boundary_ids(surface),
                "support_room_indices": support_room_indices,
                "support_graph_room_ids": sorted(set(graph_room_ids)),
                "graph_relation_states": sorted(set(support_states)) or ["unknown"],
                "footprint_area_m2": round(float(polygon.area), 6),
                "support_score": round(
                    _candidate_base_score(kind, surface, graph_support), 6
                ),
                "selected": False,
            }
            nodes.append(node)
            candidate_nodes.append(node)
            candidates_by_id[node_id] = {
                "node": node,
                "surface": surface,
                "polygon": polygon,
            }
            for boundary_id in node["source_boundary_ids"]:
                boundary_to_candidate[boundary_id].add(node_id)

    continuation_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in (roof_graph or {}).get("edges") or []:
        if edge.get("type") != "CONTINUES_AS":
            continue
        left_ids = boundary_to_candidate.get(str(edge.get("from")), set())
        right_ids = boundary_to_candidate.get(str(edge.get("to")), set())
        for left_id in left_ids:
            for right_id in right_ids:
                if left_id == right_id:
                    continue
                pair = tuple(sorted((left_id, right_id)))
                current = continuation_pairs.get(pair)
                edge_evidence = edge.get("evidence") or {}
                shared_length = float(edge_evidence.get("shared_edge_length_m", 0.0))
                if current is None or shared_length > current["shared_edge_length_m"]:
                    continuation_pairs[pair] = {
                        "shared_edge_length_m": shared_length,
                        "relation_state": edge_evidence.get(
                            "relation_state", "partial"
                        ),
                        "exact_face_incidence": bool(
                            edge_evidence.get("exact_face_incidence")
                        ),
                        "partition_atom_pairs": edge_evidence.get(
                            "partition_atom_pairs"
                        )
                        or [],
                    }

    neighbors: dict[str, set[str]] = defaultdict(set)
    for (left_id, right_id), evidence in continuation_pairs.items():
        neighbors[left_id].add(right_id)
        neighbors[right_id].add(left_id)
        edges.append(
            {
                "id": f"edge:continues:{_stable_hash([left_id, right_id], 20)}",
                "type": "CONTINUES_AS",
                "from": left_id,
                "to": right_id,
                "evidence": evidence,
            }
        )
        edges.append(
            {
                "id": f"edge:continues:{_stable_hash([right_id, left_id], 20)}",
                "type": "CONTINUES_AS",
                "from": right_id,
                "to": left_id,
                "evidence": evidence,
            }
        )

    seen: set[str] = set()
    component_size: dict[str, int] = {}
    for node in candidate_nodes:
        node_id = node["id"]
        if node_id in seen:
            continue
        stack = [node_id]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(neighbors.get(current, set()) - seen)
        for member in component:
            component_size[member] = len(component)

    for node in candidate_nodes:
        boost = 0.05 * max(0, component_size.get(node["id"], 1) - 1)
        node["continuation_component_size"] = component_size.get(node["id"], 1)
        node["support_score"] = round(min(1.0, float(node["support_score"]) + boost), 6)

    selected_by_room: dict[str, list[str]] = {}
    for room_data in exposed_rooms:
        room_index = int(room_data["room_index"])
        room_key = _room_key(room_index)
        room_polygon = _poly_xz(room_data.get("fp") or [])
        if room_polygon is None:
            continue
        room_area = max(room_polygon.area, AREA_EPS)
        slant_delta = float(room_data["wallTopY"] - room_data["wallTopMin"])

        room_edges: list[dict[str, Any]] = []
        for candidate_id, candidate in candidates_by_id.items():
            node = candidate["node"]
            if int(node["story"]) != int(room_data["story"]):
                continue
            try:
                intersection = room_polygon.intersection(candidate["polygon"])
            except Exception:
                continue
            if intersection.is_empty or intersection.area <= AREA_EPS:
                continue
            coverage_ratio = float(intersection.area) / room_area
            edge_score = _coverage_score(
                candidate_score=float(node["support_score"]),
                coverage_ratio=coverage_ratio,
                room_slant_delta=slant_delta,
                kind=str(node["surface_kind"]),
            )
            evidence = {
                "coverage_ratio": round(coverage_ratio, 6),
                "coverage_area_m2": round(float(intersection.area), 6),
                "room_slant_delta_m": round(slant_delta, 6),
                "hypothesis_support_score": round(float(node["support_score"]), 6),
                "edge_score": round(edge_score, 6),
            }
            edge = {
                "id": f"edge:covers:{_stable_hash([candidate_id, room_key], 20)}",
                "type": "COVERS_ROOM",
                "from": candidate_id,
                "to": room_key,
                "selected": False,
                "evidence": evidence,
            }
            room_edges.append(edge)

        room_edges.sort(key=lambda edge: edge["evidence"]["edge_score"], reverse=True)
        if room_edges:
            max_score = float(room_edges[0]["evidence"]["edge_score"])
            selected_ids: list[str] = []
            for edge in room_edges:
                edge_score = float(edge["evidence"]["edge_score"])
                coverage_ratio = float(edge["evidence"]["coverage_ratio"])
                if coverage_ratio < 0.05:
                    continue
                if edge_score >= max(0.33, max_score - 0.12) or (
                    coverage_ratio >= 0.2 and edge_score >= 0.45
                ):
                    edge["selected"] = True
                    selected_ids.append(str(edge["from"]))
            if not selected_ids:
                room_edges[0]["selected"] = True
                selected_ids = [str(room_edges[0]["from"])]
            selected_by_room[room_key] = selected_ids
            for hypothesis_id in selected_ids:
                candidates_by_id[hypothesis_id]["node"]["selected"] = True

        edges.extend(room_edges)

    for node in candidate_nodes:
        if node["selected"]:
            continue
        if str(node.get("surface_kind")) != "oblique":
            continue
        continuation_selected = (
            int(node.get("continuation_component_size", 1)) >= 2
            and float(node.get("support_score", 0.0)) >= 0.45
        )
        structurally_selected = float(node.get("support_score", 0.0)) >= 0.49
        if continuation_selected or structurally_selected:
            node["selected"] = True

    selected_ids = sorted(node["id"] for node in candidate_nodes if node["selected"])

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "room_count": len(room_nodes),
            "hypothesis_count": len(candidate_nodes),
            "selected_hypothesis_count": len(selected_ids),
            "selected_room_count": len(selected_by_room),
            "continuation_edge_count": sum(
                1 for edge in edges if edge["type"] == "CONTINUES_AS"
            ),
        },
        "selected_hypothesis_ids": selected_ids,
        "selected_room_assignments": selected_by_room,
    }


def select_roof_surfaces_from_hypotheses(
    *,
    oblique_surfaces: list[dict[str, Any]],
    flat_surfaces: list[dict[str, Any]],
    hypothesis_graph: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_ids = set(hypothesis_graph.get("selected_hypothesis_ids") or [])
    nodes_by_id = {
        node["id"]: node
        for node in hypothesis_graph.get("nodes") or []
        if node.get("type") == "RoofHypothesis"
    }
    selected_oblique: list[dict[str, Any]] = []
    selected_flat: list[dict[str, Any]] = []

    for kind, surfaces, out in (
        ("oblique", oblique_surfaces, selected_oblique),
        ("flat", flat_surfaces, selected_flat),
    ):
        for index, surface in enumerate(surfaces):
            node_id = _candidate_id(kind, index)
            if node_id not in selected_ids:
                continue
            node = nodes_by_id.get(node_id) or {}
            enriched = dict(surface)
            enriched["roof_hypothesis_id"] = node_id
            enriched["roof_hypothesis_support_score"] = node.get("support_score")
            out.append(enriched)

    return selected_oblique, selected_flat
