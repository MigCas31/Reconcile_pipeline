"""Corner clustering, element adjacency, and isolated wall edge detection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from reconcile_tiers.room_postprocessing.models import BuildingElement


def cluster_element_corners(
    elements: Sequence[BuildingElement],
    tol: float,
) -> list[list[int]]:
    """Union-find cluster of element corners; returns per-element vertex cluster ids."""

    if not elements:
        return []

    pts: list[tuple[float, float, float]] = []
    for el in elements:
        pts.extend(el.corners)
    n = len(pts)
    if n == 0:
        return [[] for _ in elements]

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    arr = np.array(pts, dtype=float)
    tol_sq = tol * tol
    for i in range(n):
        diffs = arr - arr[i]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        for j in np.flatnonzero(d2 <= tol_sq):
            j_int = int(j)
            if j_int != i:
                union(i, j_int)

    per_element_vids: list[list[int]] = []
    cursor = 0
    for el in elements:
        vids = [find(cursor + k) for k in range(len(el.corners))]
        cursor += len(el.corners)
        per_element_vids.append(vids)

    return per_element_vids


def element_adjacency_pairs(
    elements: Sequence[BuildingElement],
    corner_vids: list[list[int]],
) -> list[tuple[int, int]]:
    """Undirected element pairs that share at least one corner cluster."""

    cluster_to_elements: dict[int, set[int]] = defaultdict(set)
    for elem_index, vids in enumerate(corner_vids):
        for vid in vids:
            cluster_to_elements[vid].add(elem_index)

    pairs: set[tuple[int, int]] = set()
    for elem_indices in cluster_to_elements.values():
        sorted_indices = sorted(elem_indices)
        for i in range(len(sorted_indices)):
            for j in range(i + 1, len(sorted_indices)):
                a, b = sorted_indices[i], sorted_indices[j]
                pairs.add((a, b))
    return sorted(pairs)


def isolated_wall_edges(
    elements: Sequence[BuildingElement],
    corner_vids: list[list[int]],
) -> list[dict[str, object]]:
    """Wall polygon edges whose endpoint clusters appear on exactly one element."""

    cluster_element_count: dict[int, int] = defaultdict(int)
    for vids in corner_vids:
        for vid in set(vids):
            cluster_element_count[vid] += 1

    segments: list[dict[str, object]] = []
    for elem_index, el in enumerate(elements):
        if el.kind != "wall":
            continue
        vids = corner_vids[elem_index]
        n = len(vids)
        if n < 2:
            continue
        for edge_index in range(n):
            c0 = vids[edge_index]
            c1 = vids[(edge_index + 1) % n]
            isolated = (
                cluster_element_count[c0] == 1 and cluster_element_count[c1] == 1
            )
            p0 = el.corners[edge_index]
            p1 = el.corners[(edge_index + 1) % n]
            segments.append(
                {
                    "element_id": el.id,
                    "element_index": elem_index,
                    "edge_index": edge_index,
                    "isolated": isolated,
                    "start": {"x": p0[0], "y": p0[1], "z": p0[2]},
                    "end": {"x": p1[0], "y": p1[1], "z": p1[2]},
                    "cluster_start": c0,
                    "cluster_end": c1,
                }
            )
    return segments
