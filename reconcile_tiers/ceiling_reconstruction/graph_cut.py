"""Graph-cut cell labeling with interior-scan boundary conditions."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from reconcile_tiers.polyhedron.kinetic_partition import (
    BoundingPrism,
    ConvexCell,
    _points_in_cell,
)

Label = Literal["inside", "outside"]
# ``inside`` = solid structure; ``outside`` = empty room air (matches kinetic_partition).

_INF = 10**9
_BBOX_FACE_NAMES = ("x_max", "x_min", "y_max", "y_min", "z_max", "z_min")


@dataclass(frozen=True, slots=True)
class GraphCutResult:
    labels: dict[int, Label]
    lambda_param: float
    data_energy: float
    smoothness_energy: float
    solver_status: str


def _cell_adjacency(
    cells: Sequence[ConvexCell],
) -> list[tuple[int, int, float]]:
    """Return ``(cell_a, cell_b, shared_weight)`` for sign-flip neighbors."""
    by_signs = {cell.signs: cell for cell in cells}
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        for plane_id, sign in enumerate(cell.signs):
            if sign == 0:
                continue
            neighbor_signs = list(cell.signs)
            neighbor_signs[plane_id] = -sign
            neighbor = by_signs.get(tuple(neighbor_signs))
            if neighbor is None:
                continue
            key = tuple(sorted((cell.cell_id, neighbor.cell_id)))
            if key in seen:
                continue
            seen.add(key)
            weight = max(0.01, min(cell.volume, neighbor.volume))
            pairs.append((cell.cell_id, neighbor.cell_id, weight))
    return pairs


def _data_votes(
    cells: Sequence[ConvexCell],
    scan_points: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Per-cell ``(empty_vote, solid_vote)`` from oriented samples."""
    votes: dict[int, tuple[float, float]] = {
        cell.cell_id: (0.0, 0.0) for cell in cells
    }
    if scan_points.size == 0:
        return votes
    allow_normals = scan_points.shape[1] >= 6
    xyz = scan_points[:, :3]
    normals = scan_points[:, 3:6] if allow_normals else None
    for cell in cells:
        contained = _points_in_cell(xyz, cell)
        empty_v = float(np.count_nonzero(contained))
        solid_v = 0.0
        if normals is not None:
            for point, normal in zip(xyz, normals, strict=True):
                if not _points_in_cell(point.reshape(1, 3), cell)[0]:
                    continue
                centroid = np.asarray(cell.centroid, dtype=float)
                to_center = centroid - point
                if float(normal @ to_center) > 0:
                    empty_v += 1.0
                else:
                    solid_v += 1.0
        votes[cell.cell_id] = (empty_v, solid_v)
    return votes


def _cells_touching_bbox_faces(
    cells: Sequence[ConvexCell],
) -> dict[str, list[int]]:
    """Map bbox face name to cells bounded by that prism face."""
    faces: dict[str, list[int]] = {name: [] for name in _BBOX_FACE_NAMES}
    for cell in cells:
        seen_ids: set[int] = set()
        for boundary in cell.boundary_planes:
            if boundary.source_kind != "bbox":
                continue
            face_id = int(boundary.source_id)
            if face_id < 0 or face_id >= len(_BBOX_FACE_NAMES):
                continue
            if face_id in seen_ids:
                continue
            seen_ids.add(face_id)
            faces[_BBOX_FACE_NAMES[face_id]].append(cell.cell_id)
    return faces


class _FlowGraph:
    def __init__(self, n: int) -> None:
        self.n = n
        self.adj: list[list[list[int | float]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: float) -> None:
        cap = float(max(0.0, cap))
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0.0, len(self.adj[u]) - 1])

    def max_flow(self, source: int, sink: int) -> float:
        total = 0.0
        while True:
            parent: list[tuple[int, int] | None] = [None] * self.n
            queue: deque[int] = deque([source])
            parent[source] = (-1, -1)
            while queue:
                u = queue.popleft()
                for i, edge in enumerate(self.adj[u]):
                    v = int(edge[0])
                    cap = float(edge[1])
                    if cap > 1e-12 and parent[v] is None and v != source:
                        parent[v] = (u, i)
                        if v == sink:
                            queue.clear()
                            break
                        queue.append(v)
            if parent[sink] is None:
                break
            path_cap = _INF
            v = sink
            while v != source:
                assert parent[v] is not None
                u, idx = parent[v]
                path_cap = min(path_cap, float(self.adj[u][idx][1]))
                v = u
            v = sink
            while v != source:
                assert parent[v] is not None
                u, idx = parent[v]
                rev_idx = int(self.adj[u][idx][2])
                self.adj[u][idx][1] = float(self.adj[u][idx][1]) - path_cap
                self.adj[v][rev_idx][1] = float(self.adj[v][rev_idx][1]) + path_cap
                v = u
            total += path_cap
        return total


def graph_cut_label_cells(
    cells: Sequence[ConvexCell],
    scan_points: np.ndarray,
    bounding_prism: BoundingPrism,
    *,
    lambda_param: float = 0.75,
) -> GraphCutResult:
    """Label cells as ``inside`` (solid) or ``outside`` (empty room air).

    Interior room scan boundary (Y-up): ``y_max`` → solid; floor and lateral
    bbox faces → empty (CGAL doc ZMAX/ZMIN mapped to Y axis).
    """
    _ = bounding_prism
    if not cells:
        return GraphCutResult(
            labels={},
            lambda_param=lambda_param,
            data_energy=0.0,
            smoothness_energy=0.0,
            solver_status="no_cells",
        )

    cell_ids = [cell.cell_id for cell in cells]
    n_cells = len(cell_ids)
    id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
    ext_base = n_cells
    ext_index = {name: ext_base + idx for idx, name in enumerate(_BBOX_FACE_NAMES)}
    source = ext_base + len(_BBOX_FACE_NAMES)
    sink = source + 1
    graph = _FlowGraph(sink + 1)

    votes = _data_votes(cells, scan_points)
    total_inliers = max(
        1.0,
        float(scan_points.shape[0]) if scan_points.size else 1.0,
    )
    data_energy = 0.0
    for cell in cells:
        idx = id_to_idx[cell.cell_id]
        empty_v, solid_v = votes[cell.cell_id]
        w_empty = empty_v / total_inliers
        w_solid = solid_v / total_inliers
        if w_empty > 0:
            graph.add_edge(source, idx, w_empty)
            data_energy += w_empty
        if w_solid > 0:
            graph.add_edge(idx, sink, w_solid)
            data_energy += w_solid

    smoothness_energy = 0.0
    for a_id, b_id, weight in _cell_adjacency(cells):
        cap = lambda_param * weight
        graph.add_edge(id_to_idx[a_id], id_to_idx[b_id], cap)
        smoothness_energy += cap

    bbox_touch = _cells_touching_bbox_faces(cells)
    for cell in cells:
        idx = id_to_idx[cell.cell_id]
        for boundary in cell.boundary_planes:
            if boundary.source_kind != "bbox":
                continue
            face_name = _BBOX_FACE_NAMES[int(boundary.source_id)]
            ext = ext_index[face_name]
            cap = max(0.01, lambda_param * max(cell.volume, 0.01))
            graph.add_edge(idx, ext, cap)
            smoothness_energy += cap

    graph.add_edge(ext_index["y_max"], sink, _INF)
    for face in ("y_min", "x_min", "x_max", "z_min", "z_max"):
        graph.add_edge(source, ext_index[face], _INF)

    graph.max_flow(source, sink)

    reachable = _reachable_from_source(graph, source)
    labels: dict[int, Label] = {}
    for cell in cells:
        idx = id_to_idx[cell.cell_id]
        labels[cell.cell_id] = "outside" if idx in reachable else "inside"

    return GraphCutResult(
        labels=labels,
        lambda_param=lambda_param,
        data_energy=data_energy,
        smoothness_energy=smoothness_energy,
        solver_status="optimal",
    )


def _reachable_from_source(graph: _FlowGraph, source: int) -> set[int]:
    seen: set[int] = {source}
    queue: deque[int] = deque([source])
    while queue:
        u = queue.popleft()
        for edge in graph.adj[u]:
            v = int(edge[0])
            cap = float(edge[1])
            if cap > 1e-12 and v not in seen:
                seen.add(v)
                queue.append(v)
    return seen
