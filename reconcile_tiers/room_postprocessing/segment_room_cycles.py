"""Detect rooms as bounded cycles in the wall-segment junction graph (per story)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from reconcile_tiers.room_postprocessing.minimum_cycle_basis import (
    minimum_cycle_basis,
)
from reconcile_tiers.room_postprocessing.segment_group_representative import (
    base_wall_id,
    perimeter_sides_for_cycle,
    perimeter_wall_quads_for_sides,
    representative_segments_for_cycle,
)


def _segment_bottom_xz(seg: dict[str, Any]) -> tuple[float, float]:
    s = seg["start"]
    e = seg["end"]
    if s["y"] <= e["y"]:
        return s["x"], s["z"]
    return e["x"], e["z"]


def _segment_ids_by_group(
    group_nodes: list[dict[str, Any]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in group_nodes:
        out[node["id"]] = list(node.get("segment_ids") or [])
    return out


def _junction_positions(
    group_nodes: list[dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """Centroid XZ per approx group from segment endpoints."""

    positions: dict[str, tuple[float, float]] = {}
    for node in group_nodes:
        xs: list[float] = []
        zs: list[float] = []
        for seg_id in node.get("segment_ids") or []:
            seg = segments_by_id.get(seg_id)
            if not seg:
                continue
            for key in ("start", "end"):
                pt = seg[key]
                xs.append(pt["x"])
                zs.append(pt["z"])
        if xs:
            positions[node["id"]] = (sum(xs) / len(xs), sum(zs) / len(zs))
    return positions


def _group_story(
    group_nodes: list[dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
) -> dict[str, int | None]:
    story_by_group: dict[str, int | None] = {}
    for node in group_nodes:
        story = node.get("story")
        if story is not None:
            story_by_group[node["id"]] = story
            continue
        for seg_id in node.get("segment_ids") or []:
            seg = segments_by_id.get(seg_id)
            if seg and seg.get("story") is not None:
                story_by_group[node["id"]] = seg["story"]
                break
        else:
            story_by_group[node["id"]] = None
    return story_by_group


def _wall_span_edges(
    segments: list[dict[str, Any]],
    seg_to_group: dict[str, str],
    corner_tol: float = 0.05,
) -> list[dict[str, Any]]:
    """Undirected junction-to-junction edges along each wall's bottom rim."""

    by_wall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        by_wall[seg["wall_id"]].append(seg)

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for wall_id, wall_segs in by_wall.items():
        group_bottoms: dict[str, tuple[float, float]] = {}
        for seg in wall_segs:
            gid = seg_to_group.get(seg["id"])
            if not gid:
                continue
            bx, bz = _segment_bottom_xz(seg)
            if gid in group_bottoms:
                ox, oz = group_bottoms[gid]
                group_bottoms[gid] = ((ox + bx) / 2, (oz + bz) / 2)
            else:
                group_bottoms[gid] = (bx, bz)

        if len(group_bottoms) < 2:
            continue

        ordered = _order_groups_on_wall(wall_segs, group_bottoms, corner_tol)
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            key = (min(a, b), max(a, b), wall_id)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "wall_id": wall_id,
                }
            )
    return edges


def _order_groups_on_wall(
    wall_segs: list[dict[str, Any]],
    group_bottoms: dict[str, tuple[float, float]],
    corner_tol: float,
) -> list[str]:
    """Sort junction groups along the wall bottom edge in XZ."""

    points = list(group_bottoms.items())
    if len(points) <= 2:
        return [g for g, _ in sorted(points, key=lambda item: (item[1][0], item[1][1]))]

    corners: list[tuple[float, float, float]] = []
    for seg in wall_segs:
        for key in ("start", "end"):
            p = seg[key]
            corners.append((p["x"], p["y"], p["z"]))

    y_min = min(c[1] for c in corners)
    bottom = [(c[0], c[2]) for c in corners if abs(c[1] - y_min) <= corner_tol]
    if len(bottom) >= 2:
        best = 0.0
        axis_a, axis_b = bottom[0], bottom[1]
        for i in range(len(bottom)):
            for j in range(i + 1, len(bottom)):
                dx = bottom[j][0] - bottom[i][0]
                dz = bottom[j][1] - bottom[i][1]
                d2 = dx * dx + dz * dz
                if d2 > best:
                    best = d2
                    axis_a, axis_b = bottom[i], bottom[j]
        ax, az = axis_b[0] - axis_a[0], axis_b[1] - axis_a[1]
        len2 = ax * ax + az * az
        if len2 > 1e-12:

            def proj(p: tuple[float, float]) -> float:
                px, pz = p[0] - axis_a[0], p[1] - axis_a[1]
                return (px * ax + pz * az) / len2

            return [g for g, _ in sorted(points, key=lambda item: proj(item[1]))]

    return [g for g, _ in sorted(points, key=lambda item: (item[1][0], item[1][1]))]


def _room_graph_edges(
    span_edges: list[dict[str, Any]],
    wall_segment_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    """Wall-span edges plus leaf-bridge group edges for room MCB."""

    pair_seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []

    for edge in span_edges:
        a, b = edge["source"], edge["target"]
        key = (min(a, b), max(a, b))
        if key in pair_seen:
            continue
        pair_seen.add(key)
        out.append(dict(edge))

    for edge in wall_segment_graph.get("edges") or []:
        if edge.get("kind") != "leaf_bridge":
            continue
        a, b = edge["source"], edge["target"]
        key = (min(a, b), max(a, b))
        if key in pair_seen:
            continue
        pair_seen.add(key)
        out.append(
            {
                "source": a,
                "target": b,
                "wall_id": "leaf_bridge",
                "kind": "leaf_bridge",
            }
        )
    return out


def _edge_weight(
    a: str,
    b: str,
    positions: dict[str, tuple[float, float]],
) -> float:
    pa = positions.get(a)
    pb = positions.get(b)
    if pa is None or pb is None:
        return 1.0
    return math.hypot(pb[0] - pa[0], pb[1] - pa[1])


def _order_cycle_nodes(
    cycle_nodes: list[str],
    room_edges: list[dict[str, Any]],
) -> list[str]:
    """Order MCB cycle nodes into a closed walk along room-graph edges."""

    node_set = set(cycle_nodes)
    if len(node_set) < 3:
        return list(cycle_nodes)

    adj: dict[str, list[str]] = defaultdict(list)
    for edge in room_edges:
        a, b = edge["source"], edge["target"]
        if a in node_set and b in node_set:
            adj[a].append(b)
            adj[b].append(a)

    start = cycle_nodes[0]
    if start not in adj or len(adj[start]) < 2:
        for n in cycle_nodes:
            if n in adj and len(adj[n]) >= 2:
                start = n
                break

    ordered = [start]
    prev: str | None = None
    cur = start
    for _ in range(len(node_set) + 1):
        neighbors = [n for n in adj.get(cur, []) if n != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt == start and len(ordered) >= 3:
            break
        ordered.append(nxt)
        prev, cur = cur, nxt

    return ordered if len(ordered) >= 3 else list(cycle_nodes)


def _polygon_area_xz(vertices: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x0, z0 = vertices[i]
        x1, z1 = vertices[(i + 1) % n]
        area += x0 * z1 - x1 * z0
    return abs(area) * 0.5


def _connected_components_from_edges(
    group_ids: list[str],
    room_edges: list[dict[str, Any]],
) -> list[set[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in room_edges:
        a, b = edge["source"], edge["target"]
        adj[a].append(b)
        adj[b].append(a)

    seen: set[str] = set()
    components: list[set[str]] = []
    for gid in group_ids:
        if gid in seen:
            continue
        comp: set[str] = set()
        queue = [gid]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for nxt in adj.get(cur, []):
                if nxt not in seen:
                    queue.append(nxt)
        if comp:
            components.append(comp)
    return components


def _filter_mcb_cycles(
    cycles: list[list[str]],
    room_edges: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    min_room_area_m2: float,
) -> list[list[str]]:
    """Order cycles, apply min-area filter, drop largest per connected component."""

    components = _connected_components_from_edges(list(positions), room_edges)
    comp_for_node: dict[str, int] = {}
    for idx, comp in enumerate(components):
        for n in comp:
            comp_for_node[n] = idx

    by_component: dict[int, list[tuple[float, list[str]]]] = defaultdict(list)
    for cycle_nodes in cycles:
        ordered = _order_cycle_nodes(cycle_nodes, room_edges)
        if len(set(ordered)) < 3:
            continue
        poly = [positions[g] for g in ordered if g in positions]
        if len(poly) < 3:
            continue
        area = _polygon_area_xz(poly)
        if area < min_room_area_m2:
            continue
        comp_idx = comp_for_node.get(ordered[0], 0)
        by_component[comp_idx].append((area, ordered))

    accepted: list[list[str]] = []
    for scored in by_component.values():
        accepted.extend(cycle for _, cycle in scored)
    return accepted


def _room_adjacency_edges(
    room_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wall_to_rooms: dict[str, list[str]] = defaultdict(list)
    for room in room_nodes:
        for wall_id in room.get("wall_ids") or []:
            wall_to_rooms[wall_id].append(room["id"])

    pair_walls: dict[tuple[str, str], list[str]] = defaultdict(list)
    for wall_id, room_ids in wall_to_rooms.items():
        unique = sorted(set(room_ids))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair_walls[(unique[i], unique[j])].append(wall_id)

    edges: list[dict[str, Any]] = []
    for (a, b), walls in sorted(pair_walls.items()):
        edges.append(
            {
                "source": a,
                "target": b,
                "kind": "adjacent",
                "shared_wall_ids": sorted(set(walls)),
            }
        )
    return edges


def _filter_edges_to_groups(
    edges: list[dict[str, Any]],
    allowed: set[str],
) -> list[dict[str, Any]]:
    return [
        edge
        for edge in edges
        if edge["source"] in allowed and edge["target"] in allowed
    ]


def _physical_wall_ids(perimeter_sides: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            base_wall_id(str(s["wall_id"]))
            for s in perimeter_sides
            if s.get("wall_id") and s["wall_id"] != "leaf_bridge"
        }
    )


def _junction_degrees(wall_segment_graph: dict[str, Any]) -> dict[str, int]:
    """Undirected degree of each approx group in the segment junction graph."""

    degree: dict[str, int] = defaultdict(int)
    for edge in wall_segment_graph.get("edges") or []:
        src = edge.get("source")
        tgt = edge.get("target")
        if src:
            degree[src] += 1
        if tgt:
            degree[tgt] += 1
    return dict(degree)


def annotate_orphan_segment_groups(
    wall_segment_graph: dict[str, Any],
    groups_in_room_cycle: set[str],
) -> None:
    """Mark approx groups not used in any room cycle (orphans) on segment graph nodes."""

    junction_degree = _junction_degrees(wall_segment_graph)
    for node in wall_segment_graph.get("nodes") or []:
        gid = node["id"]
        in_cycle = gid in groups_in_room_cycle
        node["junction_degree"] = junction_degree.get(gid, 0)
        node["in_room_cycle"] = in_cycle
        node["orphan"] = not in_cycle


def build_segment_room_graph(
    wall_segment_graph: dict[str, Any],
    *,
    corner_tol: float = 0.05,
    min_room_area_m2: float = 1.0,
    leaf_bridge_gap: float | None = None,
) -> dict[str, Any]:
    """Rooms as minimum cycle basis loops on wall-span + leaf-bridge edges."""

    _ = leaf_bridge_gap  # bridges read from wall_segment_graph edges
    group_nodes = wall_segment_graph.get("nodes") or []
    segments = wall_segment_graph.get("segments") or []
    if not group_nodes or not segments:
        return {"nodes": [], "edges": [], "groups_in_room_cycle": []}

    segments_by_id = {s["id"]: s for s in segments}
    seg_to_group: dict[str, str] = {}
    for node in group_nodes:
        for seg_id in node.get("segment_ids") or []:
            seg_to_group[seg_id] = node["id"]

    story_by_group = _group_story(group_nodes, segments_by_id)
    groups_by_story: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for node in group_nodes:
        groups_by_story[story_by_group.get(node["id"])].append(node)

    segment_ids_by_group = _segment_ids_by_group(group_nodes)
    groups_in_room_cycle: set[str] = set()
    room_nodes: list[dict[str, Any]] = []
    room_index = 0

    for story, story_groups in sorted(
        groups_by_story.items(),
        key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0),
    ):
        group_ids = [n["id"] for n in story_groups]
        positions = _junction_positions(story_groups, segments_by_id)
        story_segments = [
            s
            for s in segments
            if story_by_group.get(seg_to_group.get(s["id"], "")) == story
        ]
        span_edges = _wall_span_edges(story_segments, seg_to_group, corner_tol)
        room_edges = _room_graph_edges(span_edges, wall_segment_graph)

        edge_pairs = [(e["source"], e["target"]) for e in room_edges]
        mcb_cycles = minimum_cycle_basis(
            group_ids,
            edge_pairs,
            weight_fn=lambda a, b: _edge_weight(a, b, positions),
        )
        faces = _filter_mcb_cycles(
            mcb_cycles,
            room_edges,
            positions,
            min_room_area_m2,
        )

        for cycle in faces:
            groups_in_room_cycle.update(cycle)
            part_positions = {g: positions[g] for g in cycle if g in positions}
            part_edges = _filter_edges_to_groups(room_edges, set(cycle))

            perimeter_sides = perimeter_sides_for_cycle(
                cycle,
                part_edges,
                segment_ids_by_group,
                segments_by_id,
                part_positions,
            )
            rep_segment_ids, _, by_group = representative_segments_for_cycle(
                cycle,
                part_edges,
                segment_ids_by_group,
                segments_by_id,
                part_positions,
                perimeter_sides=perimeter_sides,
            )
            perimeter_wall_quads = perimeter_wall_quads_for_sides(
                perimeter_sides,
                segments_by_id,
                part_positions,
            )
            perimeter_wall_ids = _physical_wall_ids(perimeter_sides)
            if len(perimeter_wall_ids) < 3:
                continue

            poly = [part_positions[g] for g in cycle if g in part_positions]
            area = _polygon_area_xz(poly) if len(poly) >= 3 else 0.0
            story_key = story if story is not None else "none"
            room_id = f"room_cycle::{story_key}::{room_index}"
            room_index += 1
            room_nodes.append(
                {
                    "id": room_id,
                    "kind": "segment_room",
                    "story": story,
                    "group_ids": list(cycle),
                    "wall_ids": perimeter_wall_ids,
                    "perimeter_sides": perimeter_sides,
                    "perimeter_wall_quads": perimeter_wall_quads,
                    "segment_ids": rep_segment_ids,
                    "representative_by_group": by_group,
                    "polygon_xz": [{"x": x, "z": z} for x, z in poly],
                    "area_m2": area,
                }
            )

    adj_edges = _room_adjacency_edges(room_nodes)
    degree: dict[str, int] = {n["id"]: 0 for n in room_nodes}
    for edge in adj_edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    for node in room_nodes:
        node["degree"] = degree.get(node["id"], 0)

    return {
        "nodes": room_nodes,
        "edges": adj_edges,
        "groups_in_room_cycle": sorted(groups_in_room_cycle),
    }
