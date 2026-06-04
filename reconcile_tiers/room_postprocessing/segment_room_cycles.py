"""Detect rooms as bounded cycles in the wall-segment junction graph (per story)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

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

        # Order groups along wall bottom using horizontal edge direction when possible
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

    # Infer bottom rim direction from horizontal polygon edges on this wall
    rim_dirs: list[tuple[float, float]] = []
    corners: list[tuple[float, float, float]] = []
    for seg in wall_segs:
        for key in ("start", "end"):
            p = seg[key]
            corners.append((p["x"], p["y"], p["z"]))

    y_min = min(c[1] for c in corners)
    bottom = [(c[0], c[2]) for c in corners if abs(c[1] - y_min) <= corner_tol]
    if len(bottom) >= 2:
        # Use farthest pair on bottom as axis
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


def _angle(x: float, z: float) -> float:
    return math.atan2(z, x)


def _next_ccw_neighbor(
    node: str,
    prev: str,
    neighbors: list[str],
    positions: dict[str, tuple[float, float]],
) -> str:
    """Next neighbor clockwise around node when entering from prev."""

    cx, cz = positions[node]
    px, pz = positions[prev]
    in_angle = _angle(px - cx, pz - cz)

    best: str | None = None
    best_delta = math.tau
    for nxt in neighbors:
        if nxt == prev:
            continue
        nx, nz = positions[nxt]
        out_angle = _angle(nx - cx, nz - cz)
        delta = (in_angle - out_angle) % math.tau
        if 1e-9 < delta < best_delta:
            best_delta = delta
            best = nxt
    if best is None:
        raise ValueError(f"no ccw neighbor at {node} from {prev}")
    return best


def _walk_face(
    start: str,
    nxt: str,
    adj: dict[str, list[str]],
    positions: dict[str, tuple[float, float]],
) -> list[str] | None:
    """Traverse one face from directed edge start→nxt; return vertex cycle or None."""

    path = [start, nxt]
    prev, cur = start, nxt
    for _ in range(len(adj) * 4):
        neighbors = adj.get(cur, [])
        if len(neighbors) < 2:
            return None
        nxt_node = _next_ccw_neighbor(cur, prev, neighbors, positions)
        if nxt_node == start and len(path) >= 3:
            return path
        path.append(nxt_node)
        prev, cur = cur, nxt_node
    return None


def _polygon_area_xz(vertices: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x0, z0 = vertices[i]
        x1, z1 = vertices[(i + 1) % n]
        area += x0 * z1 - x1 * z0
    return abs(area) * 0.5


def _find_faces(
    group_ids: list[str],
    span_edges: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> list[list[str]]:
    """Enumerate bounded face cycles via CCW planar walk."""

    adj: dict[str, list[str]] = defaultdict(list)
    for edge in span_edges:
        a, b = edge["source"], edge["target"]
        if a not in positions or b not in positions:
            continue
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)

    for node in adj:
        adj[node].sort(
            key=lambda n: _angle(
                positions[n][0] - positions[node][0],
                positions[n][1] - positions[node][1],
            ),
        )

    seen_directed: set[tuple[str, str]] = set()
    faces: list[list[str]] = []

    for edge in span_edges:
        a, b = edge["source"], edge["target"]
        if (a, b) in seen_directed:
            continue
        cycle = _walk_face(a, b, adj, positions)
        if not cycle:
            continue
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            seen_directed.add((u, v))
        faces.append(cycle)

    return faces


def _filter_faces(
    faces: list[list[str]],
    positions: dict[str, tuple[float, float]],
    min_room_area_m2: float,
) -> list[list[str]]:
    if not faces:
        return []
    scored: list[tuple[float, list[str]]] = []
    for cycle in faces:
        if len(cycle) < 3:
            continue
        poly = [positions[g] for g in cycle if g in positions]
        if len(poly) < 3:
            continue
        area = _polygon_area_xz(poly)
        if area < min_room_area_m2:
            continue
        scored.append((area, cycle))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    # Drop largest face (exterior)
    if len(scored) > 1:
        scored = scored[1:]
    return [cycle for _, cycle in scored]


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


def build_segment_room_graph(
    wall_segment_graph: dict[str, Any],
    *,
    corner_tol: float = 0.05,
    min_room_area_m2: float = 1.0,
) -> dict[str, Any]:
    """Rooms as bounded cycles of junction groups connected by wall-span edges."""

    group_nodes = wall_segment_graph.get("nodes") or []
    segments = wall_segment_graph.get("segments") or []
    if not group_nodes or not segments:
        return {"nodes": [], "edges": []}

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
        faces = _find_faces(group_ids, span_edges, positions)
        faces = _filter_faces(faces, positions, min_room_area_m2)

        for cycle in faces:
            wall_ids: set[str] = set()
            segment_ids: list[str] = []
            for i in range(len(cycle)):
                a = cycle[i]
                b = cycle[(i + 1) % len(cycle)]
                for edge in span_edges:
                    src, tgt = edge["source"], edge["target"]
                    if (src == a and tgt == b) or (src == b and tgt == a):
                        wall_ids.add(edge["wall_id"])
            for gid in cycle:
                segment_ids.extend(segment_ids_by_group.get(gid, []))

            poly = [positions[g] for g in cycle if g in positions]
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
                    "wall_ids": sorted(wall_ids),
                    "segment_ids": sorted(set(segment_ids)),
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

    return {"nodes": room_nodes, "edges": adj_edges}
