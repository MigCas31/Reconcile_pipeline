from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.geometry import Polygon
from shapely.validation import make_valid

from .graph_support import match_graph_room
from .graph_utils import room_key as _room_key
from .graph_utils import stable_hash as _stable_hash


def _selected_cover_edges(hypothesis_graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        edge
        for edge in (hypothesis_graph.get("edges") or [])
        if edge.get("type") == "COVERS_ROOM" and edge.get("selected")
    ]


def _selected_hypothesis_nodes(
    hypothesis_graph: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        node["id"]: node
        for node in (hypothesis_graph.get("nodes") or [])
        if node.get("type") == "RoofHypothesis" and node.get("selected")
    }


def _room_nodes(exposed_rooms: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_room_key(int(room["room_index"])): room for room in exposed_rooms}


def _all_room_nodes(
    *,
    exposed_rooms: list[dict[str, Any]],
    bldg: dict | None,
    graph,
) -> dict[str, dict[str, Any]]:
    nodes = {
        room_id: dict(room) for room_id, room in _room_nodes(exposed_rooms).items()
    }
    if bldg is None:
        return nodes
    for room_index, room in enumerate(bldg.get("rooms", [])):
        room_id = _room_key(room_index)
        if room_id in nodes:
            continue
        floor_polygon = room.get("floor_polygon") or []
        story = int(room.get("story", 0))
        graph_room = match_graph_room(graph, story, floor_polygon)
        nodes[room_id] = {
            "room": room,
            "room_index": room_index,
            "graph_room_id": graph_room.id if graph_room is not None else None,
            "fp": floor_polygon,
            "story": story,
        }
    return nodes


def _room_adjacency(
    room_nodes: dict[str, dict[str, Any]], graph
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {room_id: set() for room_id in room_nodes}
    if graph is None:
        polys: dict[str, Polygon] = {}
        for room_id, room in room_nodes.items():
            points = [
                (float(p[0]), float(p[2]))
                for p in (room.get("fp") or [])
                if len(p) >= 3
            ]
            if len(points) < 3:
                continue
            poly = Polygon(points)
            if not poly.is_valid:
                try:
                    poly = make_valid(poly)
                    if poly.geom_type == "MultiPolygon":
                        poly = max(poly.geoms, key=lambda geom: geom.area)
                except Exception:
                    continue
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            polys[room_id] = poly
        room_ids = sorted(polys)
        for idx, left_id in enumerate(room_ids):
            left = polys[left_id]
            for right_id in room_ids[idx + 1 :]:
                right = polys[right_id]
                if (
                    left.distance(right) <= 0.25
                    or left.boundary.intersection(right.boundary).length > 0.05
                ):
                    adjacency[left_id].add(right_id)
                    adjacency[right_id].add(left_id)
        return adjacency

    graph_room_to_local = {
        str(room.get("graph_room_id")): room_id
        for room_id, room in room_nodes.items()
        if room.get("graph_room_id")
    }
    for graph_room_id, local_room_id in graph_room_to_local.items():
        for edge in graph.edges_from(graph_room_id, "ADJACENT_TO"):
            state = str((edge.evidence or {}).get("relation_state", "unknown"))
            if state not in {"confirmed", "partial"}:
                continue
            other = graph_room_to_local.get(edge.to_id)
            if other is None:
                continue
            adjacency[local_room_id].add(other)
            adjacency[other].add(local_room_id)
        for edge in graph.edges_to(graph_room_id, "ADJACENT_TO"):
            state = str((edge.evidence or {}).get("relation_state", "unknown"))
            if state not in {"confirmed", "partial"}:
                continue
            other = graph_room_to_local.get(edge.from_id)
            if other is None:
                continue
            adjacency[local_room_id].add(other)
            adjacency[other].add(local_room_id)
    return adjacency


def _full_room_adjacency(
    room_nodes: dict[str, dict[str, Any]], graph
) -> dict[str, set[str]]:
    adjacency = _room_adjacency(room_nodes, graph)
    if graph is None:
        return adjacency
    graph_room_to_local = {
        str(room.get("graph_room_id")): room_id
        for room_id, room in room_nodes.items()
        if room.get("graph_room_id")
    }
    for graph_room_id, local_room_id in graph_room_to_local.items():
        for edge in graph.edges_from(graph_room_id, "BELOW"):
            state = str((edge.evidence or {}).get("relation_state", "unknown"))
            if state not in {"confirmed", "partial"}:
                continue
            other = graph_room_to_local.get(edge.to_id)
            if other is None:
                continue
            adjacency.setdefault(local_room_id, set()).add(other)
            adjacency.setdefault(other, set()).add(local_room_id)
        for edge in graph.edges_to(graph_room_id, "BELOW"):
            state = str((edge.evidence or {}).get("relation_state", "unknown"))
            if state not in {"confirmed", "partial"}:
                continue
            other = graph_room_to_local.get(edge.from_id)
            if other is None:
                continue
            adjacency.setdefault(local_room_id, set()).add(other)
            adjacency.setdefault(other, set()).add(local_room_id)
        for edge in graph.edges_from(graph_room_id, "ABOVE"):
            state = str((edge.evidence or {}).get("relation_state", "unknown"))
            if state not in {"confirmed", "partial"}:
                continue
            other = graph_room_to_local.get(edge.to_id)
            if other is None:
                continue
            adjacency.setdefault(local_room_id, set()).add(other)
            adjacency.setdefault(other, set()).add(local_room_id)
        for edge in graph.edges_to(graph_room_id, "ABOVE"):
            state = str((edge.evidence or {}).get("relation_state", "unknown"))
            if state not in {"confirmed", "partial"}:
                continue
            other = graph_room_to_local.get(edge.from_id)
            if other is None:
                continue
            adjacency.setdefault(local_room_id, set()).add(other)
            adjacency.setdefault(other, set()).add(local_room_id)
    return adjacency


def _nearest_part_id(
    *,
    start_room_id: str,
    adjacency: dict[str, set[str]],
    room_part_membership: dict[str, list[str]],
) -> str | None:
    seen = {start_room_id}
    frontier = [start_room_id]
    while frontier:
        next_frontier: list[str] = []
        candidate_counts: dict[str, int] = defaultdict(int)
        for room_id in frontier:
            for neighbor in sorted(adjacency.get(room_id, set())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_frontier.append(neighbor)
                for part_id in room_part_membership.get(neighbor, []):
                    candidate_counts[str(part_id)] += 1
        if candidate_counts:
            best_score = max(candidate_counts.values())
            return sorted(
                part_id
                for part_id, score in candidate_counts.items()
                if score == best_score
            )[0]
        frontier = next_frontier
    return None


def _biconnected_room_components(
    adjacency: dict[str, set[str]],
) -> tuple[list[set[str]], set[str]]:
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    edge_stack: list[tuple[str, str]] = []
    articulation: set[str] = set()
    components: list[set[str]] = []
    time = 0

    def pop_component(stop_edge: tuple[str, str]) -> None:
        component: set[str] = set()
        while edge_stack:
            edge = edge_stack.pop()
            component.add(edge[0])
            component.add(edge[1])
            if edge == stop_edge or edge == (stop_edge[1], stop_edge[0]):
                break
        if component:
            components.append(component)

    def dfs(u: str) -> None:
        nonlocal time
        time += 1
        discovery[u] = time
        low[u] = time
        child_count = 0
        for v in sorted(adjacency.get(u, set())):
            if v not in discovery:
                parent[v] = u
                child_count += 1
                edge_stack.append((u, v))
                dfs(v)
                low[u] = min(low[u], low[v])
                if (parent.get(u) is None and child_count > 1) or (
                    parent.get(u) is not None and low[v] >= discovery[u]
                ):
                    articulation.add(u)
                    pop_component((u, v))
            elif v != parent.get(u) and discovery[v] < discovery[u]:
                low[u] = min(low[u], discovery[v])
                edge_stack.append((u, v))

    for node in sorted(adjacency):
        if node in discovery:
            continue
        parent[node] = None
        dfs(node)
        if edge_stack:
            component: set[str] = set()
            while edge_stack:
                edge = edge_stack.pop()
                component.add(edge[0])
                component.add(edge[1])
            if component:
                components.append(component)

    if not components:
        for room_id in adjacency:
            components.append({room_id})
    else:
        covered_rooms = {room_id for component in components for room_id in component}
        for room_id in sorted(adjacency):
            if room_id not in covered_rooms:
                components.append({room_id})

    return components, articulation


def build_building_part_graph(
    *,
    exposed_rooms: list[dict[str, Any]],
    hypothesis_graph: dict[str, Any],
    bldg: dict | None = None,
    graph=None,
) -> dict[str, Any]:
    room_nodes = _room_nodes(exposed_rooms)
    all_room_nodes = _all_room_nodes(
        exposed_rooms=exposed_rooms, bldg=bldg, graph=graph
    )
    selected_hypotheses = _selected_hypothesis_nodes(hypothesis_graph)
    selected_cover_edges = _selected_cover_edges(hypothesis_graph)
    room_to_hypotheses: dict[str, set[str]] = defaultdict(set)
    for edge in selected_cover_edges:
        room_to_hypotheses[str(edge["to"])].add(str(edge["from"]))
    graph_room_to_local = {
        str(room.get("graph_room_id")): room_id
        for room_id, room in room_nodes.items()
        if room.get("graph_room_id")
    }
    for hypothesis_id, hypothesis in selected_hypotheses.items():
        for room_index in hypothesis.get("support_room_indices") or []:
            if isinstance(room_index, int):
                room_to_hypotheses[_room_key(room_index)].add(hypothesis_id)
        for graph_room_id in hypothesis.get("support_graph_room_ids") or []:
            local_room_id = graph_room_to_local.get(str(graph_room_id))
            if local_room_id is not None:
                room_to_hypotheses[local_room_id].add(hypothesis_id)

    room_adjacency = _room_adjacency(room_nodes, graph)
    components, articulation_rooms = _biconnected_room_components(room_adjacency)

    continuation = defaultdict(set)
    for edge in hypothesis_graph.get("edges") or []:
        if edge.get("type") != "CONTINUES_AS":
            continue
        left = str(edge.get("from"))
        right = str(edge.get("to"))
        if left in selected_hypotheses and right in selected_hypotheses:
            continuation[left].add(right)
            continuation[right].add(left)

    part_nodes: list[dict[str, Any]] = []
    part_edges: list[dict[str, Any]] = []
    room_part_membership: dict[str, list[str]] = defaultdict(list)
    hypothesis_part_membership: dict[str, list[str]] = defaultdict(list)

    for index, component in enumerate(components):
        room_ids = sorted(component)
        hypothesis_ids: set[str] = set()
        for room_id in room_ids:
            hypothesis_ids.update(room_to_hypotheses.get(room_id, set()))
        for hypothesis_id in list(hypothesis_ids):
            hypothesis_ids.update(continuation.get(hypothesis_id, set()))

        room_slants = [
            float(room_nodes[room_id]["wallTopY"] - room_nodes[room_id]["wallTopMin"])
            for room_id in room_ids
            if room_id in room_nodes
        ]
        oblique_ids = sorted(
            hypothesis_id
            for hypothesis_id in hypothesis_ids
            if (selected_hypotheses.get(hypothesis_id) or {}).get("surface_kind")
            == "oblique"
        )
        flat_ids = sorted(
            hypothesis_id
            for hypothesis_id in hypothesis_ids
            if (selected_hypotheses.get(hypothesis_id) or {}).get("surface_kind")
            == "flat"
        )
        slant_ratio = (
            sum(1 for slant in room_slants if slant >= 0.6) / len(room_slants)
            if room_slants
            else 0.0
        )
        if len(oblique_ids) >= 2 and slant_ratio >= 0.5:
            roof_family_guess = "gable_or_multi_slope"
        elif len(oblique_ids) == 0:
            roof_family_guess = "flat_or_capped"
        else:
            roof_family_guess = "mixed_or_partial"

        part_id = f"building-part:{
            _stable_hash(
                [str(index), str(room_ids), str(oblique_ids), str(flat_ids)],
                20,
            )
        }"
        part_node = {
            "id": part_id,
            "type": "BuildingPart",
            "room_ids": room_ids,
            "hypothesis_ids": sorted(hypothesis_ids),
            "oblique_hypothesis_ids": oblique_ids,
            "flat_hypothesis_ids": flat_ids,
            "articulation_room_ids": sorted(
                room_id for room_id in room_ids if room_id in articulation_rooms
            ),
            "slant_ratio": round(slant_ratio, 6),
            "max_room_slant_delta_m": round(max(room_slants), 6)
            if room_slants
            else 0.0,
            "roof_family_guess": roof_family_guess,
        }
        part_nodes.append(part_node)
        for room_id in room_ids:
            room_part_membership[room_id].append(part_id)
            part_edges.append(
                {
                    "id": f"edge:part-room:{_stable_hash([part_id, room_id], 20)}",
                    "type": "CONTAINS_ROOM",
                    "from": part_id,
                    "to": room_id,
                }
            )
        for hypothesis_id in sorted(hypothesis_ids):
            hypothesis_part_membership[hypothesis_id].append(part_id)
            part_edges.append(
                {
                    "id": f"edge:part-hypothesis:{
                        _stable_hash(
                            [part_id, hypothesis_id],
                            20,
                        )
                    }",
                    "type": "CONTAINS_HYPOTHESIS",
                    "from": part_id,
                    "to": hypothesis_id,
                }
            )

    part_nodes_by_id = {str(node["id"]): node for node in part_nodes}
    full_adjacency = _full_room_adjacency(all_room_nodes, graph)

    if not part_nodes and all_room_nodes:
        all_components, _ = _biconnected_room_components(full_adjacency)
        for index, component in enumerate(all_components):
            room_ids = sorted(component)
            part_id = f"building-part:{
                _stable_hash(
                    ['full-building', str(index), str(room_ids)],
                    20,
                )
            }"
            node = {
                "id": part_id,
                "type": "BuildingPart",
                "room_ids": room_ids,
                "hypothesis_ids": [],
                "oblique_hypothesis_ids": [],
                "flat_hypothesis_ids": [],
                "articulation_room_ids": [],
                "slant_ratio": 0.0,
                "max_room_slant_delta_m": 0.0,
                "roof_family_guess": "unclassified",
            }
            part_nodes.append(node)
            part_nodes_by_id[part_id] = node
            for room_id in room_ids:
                room_part_membership[room_id].append(part_id)
                part_edges.append(
                    {
                        "id": f"edge:part-room:{_stable_hash([part_id, room_id], 20)}",
                        "type": "CONTAINS_ROOM",
                        "from": part_id,
                        "to": room_id,
                    }
                )

    for room_id in sorted(all_room_nodes):
        if room_part_membership.get(room_id):
            continue
        nearest_part_id = _nearest_part_id(
            start_room_id=room_id,
            adjacency=full_adjacency,
            room_part_membership=room_part_membership,
        )
        if nearest_part_id is None:
            continue
        room_part_membership[room_id].append(nearest_part_id)
        part_nodes_by_id.setdefault(nearest_part_id, {}).setdefault(
            "room_ids", []
        ).append(room_id)
        part_edges.append(
            {
                "id": f"edge:part-room:{_stable_hash([nearest_part_id, room_id], 20)}",
                "type": "CONTAINS_ROOM",
                "from": nearest_part_id,
                "to": room_id,
            }
        )

    for node in part_nodes:
        room_ids = sorted(set(node.get("room_ids") or []))
        node["room_ids"] = room_ids

    return {
        "nodes": part_nodes,
        "edges": part_edges,
        "room_membership": {
            room_id: sorted(part_ids)
            for room_id, part_ids in room_part_membership.items()
        },
        "hypothesis_membership": {
            hypothesis_id: sorted(part_ids)
            for hypothesis_id, part_ids in hypothesis_part_membership.items()
        },
        "metadata": {
            "part_count": len(part_nodes),
            "articulation_room_count": len(articulation_rooms),
        },
    }


def refine_building_part_graph(
    *,
    building_part_graph: dict[str, Any],
    roof_evidence_graph: dict[str, Any],
) -> dict[str, Any]:
    part_evidence = {
        str(part_id): evidence
        for part_id, evidence in (
            roof_evidence_graph.get("part_evidence") or {}
        ).items()
        if isinstance(evidence, dict)
    }
    refined_nodes: list[dict[str, Any]] = []
    for node in building_part_graph.get("nodes") or []:
        refined = dict(node)
        evidence = part_evidence.get(str(node.get("id"))) or {}
        refined["roof_family_guess_initial"] = node.get("roof_family_guess")
        if evidence.get("refined_roof_family_guess"):
            refined["roof_family_guess"] = str(evidence["refined_roof_family_guess"])
        refined["coverage_subpart_ids"] = list(evidence.get("subpart_ids") or [])
        refined["gable_subpart_count"] = int(
            evidence.get("gable_subpart_count", 0) or 0
        )
        refined["branch_subpart_count"] = int(
            evidence.get("branch_subpart_count", 0) or 0
        )
        refined["strong_sloped_room_count"] = int(
            evidence.get("strong_sloped_room_count", 0) or 0
        )
        refined["knee_wall_room_count"] = int(
            evidence.get("knee_wall_room_count", 0) or 0
        )
        refined["flat_cap_room_count"] = int(
            evidence.get("flat_cap_room_count", 0) or 0
        )
        refined["opposed_slope_pair_count"] = int(
            evidence.get("opposed_slope_pair_count", 0) or 0
        )
        refined_nodes.append(refined)

    refined = {
        "nodes": refined_nodes,
        "edges": [dict(edge) for edge in (building_part_graph.get("edges") or [])],
        "room_membership": {
            room_id: list(part_ids)
            for room_id, part_ids in (
                building_part_graph.get("room_membership") or {}
            ).items()
        },
        "hypothesis_membership": {
            hypothesis_id: list(part_ids)
            for hypothesis_id, part_ids in (
                building_part_graph.get("hypothesis_membership") or {}
            ).items()
        },
        "metadata": dict(building_part_graph.get("metadata") or {}),
    }
    refined["metadata"].update(
        {
            "evidence_refined_part_count": sum(
                1
                for node in refined_nodes
                if node.get("roof_family_guess")
                != node.get("roof_family_guess_initial")
            ),
            "evidence_gable_part_count": sum(
                1
                for node in refined_nodes
                if node.get("roof_family_guess") == "gable_or_multi_slope"
            ),
        }
    )
    return refined


def classify_selected_flat_hypothesis_roles(
    *,
    exposed_rooms: list[dict[str, Any]],
    hypothesis_graph: dict[str, Any],
    building_part_graph: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    room_nodes = _room_nodes(exposed_rooms)
    selected_hypotheses = _selected_hypothesis_nodes(hypothesis_graph)
    part_nodes = {node["id"]: node for node in (building_part_graph.get("nodes") or [])}
    room_membership = building_part_graph.get("room_membership") or {}
    hypothesis_membership = building_part_graph.get("hypothesis_membership") or {}

    roles: dict[str, dict[str, Any]] = {}
    for hypothesis_id, node in selected_hypotheses.items():
        if node.get("surface_kind") != "flat":
            continue
        support_rooms = [
            int(v)
            for v in (node.get("support_room_indices") or [])
            if isinstance(v, int)
        ]
        support_room_id = (
            _room_key(support_rooms[0]) if len(support_rooms) == 1 else None
        )
        role = "roof_flat"
        reason = "selected_flat_hypothesis"
        if support_room_id is not None:
            parts = [
                part_nodes[part_id]
                for part_id in room_membership.get(support_room_id, [])
                if part_id in part_nodes
            ]
            room = room_nodes.get(support_room_id)
            slant_delta = (
                float(room["wallTopY"] - room["wallTopMin"])
                if room is not None
                else 0.0
            )
            if (
                any(
                    part.get("roof_family_guess") == "gable_or_multi_slope"
                    and float(part.get("max_room_slant_delta_m", 0.0)) >= 0.6
                    and len(part.get("oblique_hypothesis_ids") or []) >= 2
                    for part in parts
                )
                and slant_delta >= 0.6
            ):
                role = "ceiling_cap"
                reason = "room_specific_flat_inside_sloped_building_part"
        elif any(
            part_nodes[part_id].get("roof_family_guess") == "gable_or_multi_slope"
            for part_id in hypothesis_membership.get(hypothesis_id, [])
            if part_id in part_nodes
        ):
            role = "ambiguous_flat_over_sloped_part"
            reason = "flat_hypothesis_spans_sloped_building_part"

        roles[hypothesis_id] = {
            "role": role,
            "reason": reason,
        }

    return roles
