"""Wall vertical-segment graph: approx clusters as nodes, cross-group links as edges."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reconcile_tiers.room_postprocessing.corner_graph import (
    DEFAULT_LEAF_BRIDGE_GAP_M,
    merge_adjacency_pairs,
)
from reconcile_tiers.room_postprocessing.models import BuildingElement
from reconcile_tiers.room_postprocessing.segment_group_representative import (
    base_wall_id,
)


@dataclass(frozen=True, slots=True)
class WallVerticalSegment:
    id: str
    wall_id: str
    element_index: int
    edge_index: int
    room_index: int | None
    story: int | None
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    cluster_start: int
    cluster_end: int


def _is_vertical_edge(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    corner_tol: float,
) -> bool:
    dx = p1[0] - p0[0]
    dz = p1[2] - p0[2]
    dy = abs(p1[1] - p0[1])
    return math.hypot(dx, dz) <= corner_tol and dy > corner_tol


def extract_wall_vertical_segments(
    elements: Sequence[BuildingElement],
    corner_vids: list[list[int]],
    corner_tol: float,
) -> list[WallVerticalSegment]:
    segments: list[WallVerticalSegment] = []
    for elem_index, el in enumerate(elements):
        if el.kind != "wall":
            continue
        if getattr(el, "synthetic", False):
            continue
        vids = corner_vids[elem_index]
        n = len(el.corners)
        if n < 2:
            continue
        for edge_index in range(n):
            p0 = el.corners[edge_index]
            p1 = el.corners[(edge_index + 1) % n]
            if not _is_vertical_edge(p0, p1, corner_tol):
                continue
            seg_id = f"{el.id}::vseg::{edge_index}"
            segments.append(
                WallVerticalSegment(
                    id=seg_id,
                    wall_id=el.id,
                    element_index=elem_index,
                    edge_index=edge_index,
                    room_index=el.room_index,
                    story=el.story,
                    start=p0,
                    end=p1,
                    cluster_start=vids[edge_index],
                    cluster_end=vids[(edge_index + 1) % n],
                )
            )
    return segments


def _strict_segment_pairs(
    segments: Sequence[WallVerticalSegment],
) -> set[tuple[int, int]]:
    cluster_to_seg: dict[int, set[int]] = defaultdict(set)
    for seg_index, seg in enumerate(segments):
        cluster_to_seg[seg.cluster_start].add(seg_index)
        cluster_to_seg[seg.cluster_end].add(seg_index)

    pairs: set[tuple[int, int]] = set()
    for seg_indices in cluster_to_seg.values():
        sorted_indices = sorted(seg_indices)
        for i in range(len(sorted_indices)):
            for j in range(i + 1, len(sorted_indices)):
                pairs.add((sorted_indices[i], sorted_indices[j]))
    return pairs


def _same_story(a: int | None, b: int | None) -> bool:
    """Approx links only within one storey (aligned segments on different floors stay separate)."""

    return a == b


def _approx_segment_pairs(
    segments: Sequence[WallVerticalSegment],
    adjacency_tol: float,
) -> set[tuple[int, int]]:
    n = len(segments)
    if n < 2:
        return set()
    tol_sq = adjacency_tol * adjacency_tol
    pairs: set[tuple[int, int]] = set()
    endpoints = [
        (seg_index, seg.start, seg.end, seg.story)
        for seg_index, seg in enumerate(segments)
    ]
    for i in range(n):
        _, a0, a1, story_a = endpoints[i]
        arr_a = np.array([a0, a1], dtype=float)
        for j in range(i + 1, n):
            _, b0, b1, story_b = endpoints[j]
            if not _same_story(story_a, story_b):
                continue
            arr_b = np.array([b0, b1], dtype=float)
            for pa in arr_a:
                diffs = arr_b - pa
                d2 = np.einsum("ij,ij->i", diffs, diffs)
                if np.any(d2 <= tol_sq):
                    pairs.add((i, j))
                    break
    return pairs


def _intra_wall_segment_pairs(
    segments: Sequence[WallVerticalSegment],
) -> set[tuple[int, int]]:
    """Connect vertical segments on the same wall (shared rim of the quad)."""

    by_wall: dict[str, list[int]] = defaultdict(list)
    for seg_index, seg in enumerate(segments):
        by_wall[seg.wall_id].append(seg_index)

    pairs: set[tuple[int, int]] = set()
    for indices in by_wall.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                pairs.add((min(a, b), max(a, b)))
    return pairs


def _segment_dict(seg: WallVerticalSegment) -> dict[str, Any]:
    return {
        "id": seg.id,
        "wall_id": seg.wall_id,
        "initial_wall_id": base_wall_id(seg.wall_id),
        "kind": "wall_segment",
        "element_index": seg.element_index,
        "edge_index": seg.edge_index,
        "room_index": seg.room_index,
        "story": seg.story,
        "start": {"x": seg.start[0], "y": seg.start[1], "z": seg.start[2]},
        "end": {"x": seg.end[0], "y": seg.end[1], "z": seg.end[2]},
        "cluster_start": seg.cluster_start,
        "cluster_end": seg.cluster_end,
    }


def _approx_segment_groups(
    segment_count: int,
    approx: set[tuple[int, int]],
) -> list[list[int]]:
    """Connected components of segments linked only by approx adjacency."""

    parent = list(range(segment_count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[i]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in approx:
        union(a, b)

    by_root: dict[int, list[int]] = defaultdict(list)
    for i in range(segment_count):
        by_root[find(i)].append(i)

    return [sorted(indices) for indices in by_root.values()]


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _endpoint_distance_sq(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _leaf_bridge_segment_pairs(
    segments: Sequence[WallVerticalSegment],
    seg_to_group: dict[int, int],
    group_degree: dict[int, int],
    *,
    bridge_gap: float,
) -> set[tuple[int, int]]:
    """Cross-group links for degree-1 junction groups to nearby foreign segments.

    When a stub or near-miss free end has no wall between it and another segment
    but endpoints are within ``bridge_gap``, add a graph edge without merging
    the approx groups.
    """

    leaf_groups = {g for g, deg in group_degree.items() if deg == 1}
    if not leaf_groups or bridge_gap <= 0.0:
        return set()

    tol_sq = bridge_gap * bridge_gap
    pairs: set[tuple[int, int]] = set()
    for seg_index, seg in enumerate(segments):
        group_a = seg_to_group.get(seg_index)
        if group_a is None or group_a not in leaf_groups:
            continue
        endpoints_a = (seg.start, seg.end)
        for other_index, other in enumerate(segments):
            if other_index == seg_index:
                continue
            group_b = seg_to_group.get(other_index)
            if group_b is None or group_b == group_a:
                continue
            if not _same_story(seg.story, other.story):
                continue
            for pa in endpoints_a:
                for pb in (other.start, other.end):
                    if _endpoint_distance_sq(pa, pb) <= tol_sq:
                        pairs.add(_pair_key(seg_index, other_index))
                        break
                else:
                    continue
                break
    return pairs


def _apply_leaf_bridges_to_group_graph(
    segments: Sequence[WallVerticalSegment],
    seg_to_group: dict[int, int],
    group_pair_kinds: dict[tuple[int, int], str],
    group_degree: dict[int, int],
    *,
    bridge_gap: float,
) -> set[tuple[int, int]]:
    """Add ``leaf_bridge`` group edges; return new segment pairs."""

    bridge_pairs = _leaf_bridge_segment_pairs(
        segments,
        seg_to_group,
        group_degree,
        bridge_gap=bridge_gap,
    )
    for a, b in bridge_pairs:
        ga = seg_to_group[a]
        gb = seg_to_group[b]
        if ga == gb:
            continue
        gkey = _pair_key(ga, gb)
        if gkey in group_pair_kinds:
            continue
        group_pair_kinds[gkey] = "leaf_bridge"
        group_degree[ga] = group_degree.get(ga, 0) + 1
        group_degree[gb] = group_degree.get(gb, 0) + 1
    return bridge_pairs


def build_wall_segment_graph(
    elements: Sequence[BuildingElement],
    corner_vids: list[list[int]],
    corner_tol: float,
    adjacency_tol: float,
    *,
    leaf_bridge_gap: float | None = None,
) -> dict[str, Any]:
    """Approx-connected segment clusters as graph nodes; segment geometry in ``segments``."""

    segments = extract_wall_vertical_segments(elements, corner_vids, corner_tol)
    strict = _strict_segment_pairs(segments)
    approx = _approx_segment_pairs(segments, adjacency_tol)
    intra = _intra_wall_segment_pairs(segments)
    all_pairs = merge_adjacency_pairs(
        sorted(strict),
        merge_adjacency_pairs(sorted(approx), sorted(intra)),
    )

    pair_kinds: dict[tuple[int, int], str] = {}
    for pair in strict:
        pair_kinds[_pair_key(*pair)] = "junction"
    for pair in approx:
        key = _pair_key(*pair)
        if key not in pair_kinds:
            pair_kinds[key] = "approx"
    for pair in intra:
        pair_kinds[_pair_key(*pair)] = "intra_wall"

    raw_groups = _approx_segment_groups(len(segments), approx)
    raw_groups.sort(key=lambda g: segments[g[0]].id if g else "")
    seg_to_group: dict[int, int] = {}
    group_nodes: list[dict[str, Any]] = []
    for group_index, member_indices in enumerate(raw_groups):
        for seg_index in member_indices:
            seg_to_group[seg_index] = group_index
        member_segments = [segments[i] for i in member_indices]
        wall_ids = sorted({s.wall_id for s in member_segments})
        segment_dicts = [_segment_dict(s) for s in member_segments]
        group_id = f"approx_grp::{group_index}"
        group_nodes.append(
            {
                "id": group_id,
                "kind": "approx_segment_group",
                "segment_ids": [s.id for s in member_segments],
                "wall_ids": wall_ids,
                "initial_wall_by_segment": {
                    sd["id"]: sd["initial_wall_id"] for sd in segment_dicts
                },
                "segment_count": len(member_segments),
                "room_index": member_segments[0].room_index,
                "story": member_segments[0].story,
            }
        )

    group_pair_kinds: dict[tuple[int, int], str] = {}
    for a, b in all_pairs:
        ga = seg_to_group[a]
        gb = seg_to_group[b]
        if ga == gb:
            continue
        gkey = _pair_key(ga, gb)
        kind = pair_kinds.get(_pair_key(a, b), "junction")
        if gkey not in group_pair_kinds:
            group_pair_kinds[gkey] = kind

    group_degree: dict[int, int] = {i: 0 for i in range(len(group_nodes))}
    for ga, gb in group_pair_kinds:
        group_degree[ga] += 1
        group_degree[gb] += 1

    bridge_gap = (
        DEFAULT_LEAF_BRIDGE_GAP_M if leaf_bridge_gap is None else leaf_bridge_gap
    )
    leaf_bridge_pairs = _apply_leaf_bridges_to_group_graph(
        segments,
        seg_to_group,
        group_pair_kinds,
        group_degree,
        bridge_gap=bridge_gap,
    )
    for a, b in leaf_bridge_pairs:
        key = _pair_key(a, b)
        if key not in pair_kinds:
            pair_kinds[key] = "leaf_bridge"

    for group_index, node in enumerate(group_nodes):
        node["degree"] = group_degree.get(group_index, 0)

    group_edges = [
        {
            "source": group_nodes[ga]["id"],
            "target": group_nodes[gb]["id"],
            "source_index": ga,
            "target_index": gb,
            "kind": group_pair_kinds[(ga, gb)],
        }
        for ga, gb in sorted(group_pair_kinds)
    ]

    return {
        "segments": [_segment_dict(s) for s in segments],
        "nodes": group_nodes,
        "edges": group_edges,
    }
