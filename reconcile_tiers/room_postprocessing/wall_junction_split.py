"""Split walls that pass through approx junctions without a corner there."""

from __future__ import annotations

import math
from collections.abc import Sequence

from reconcile_tiers.room_postprocessing.corner_graph import cluster_element_corners
from reconcile_tiers.room_postprocessing.models import BuildingElement
from reconcile_tiers.room_postprocessing.wall_segment_graph import (
    _approx_segment_groups,
    _approx_segment_pairs,
    extract_wall_vertical_segments,
)


def _is_horizontal_edge(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    corner_tol: float,
) -> bool:
    dy = abs(p1[1] - p0[1])
    return dy <= corner_tol and math.hypot(p0[0] - p1[0], p0[2] - p1[2]) > corner_tol


def _lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def _xz_point_to_segment(
    px: float,
    pz: float,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    ax, az = a[0], a[2]
    bx, bz = b[0], b[2]
    dx, dz = bx - ax, bz - az
    len2 = dx * dx + dz * dz
    if len2 < 1e-12:
        return math.hypot(px - ax, pz - az), 0.0
    t = ((px - ax) * dx + (pz - az) * dz) / len2
    cx = ax + t * dx
    cz = az + t * dz
    return math.hypot(px - cx, pz - cz), t


def _wall_has_corner_near(
    wall: BuildingElement,
    anchor: tuple[float, float, float],
    tol: float,
) -> bool:
    tol_sq = tol * tol
    for c in wall.corners:
        d2 = sum((c[k] - anchor[k]) ** 2 for k in range(3))
        if d2 <= tol_sq:
            return True
    return False


def _multi_wall_junction_anchors(
    segments: Sequence,
    approx: set[tuple[int, int]],
    adjacency_tol: float,
) -> list[tuple[float, float, float]]:
    """XZ junction sites from approx groups spanning multiple walls."""

    anchors: list[tuple[float, float, float]] = []
    for member_indices in _approx_segment_groups(len(segments), approx):
        wall_ids = {segments[i].wall_id for i in member_indices}
        if len(wall_ids) < 2:
            continue
        pts: list[tuple[float, float, float]] = []
        for idx in member_indices:
            seg = segments[idx]
            pts.extend([seg.start, seg.end])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        anchors.append((sum(xs) / len(xs), min(ys), sum(zs) / len(zs)))
    # Deduplicate nearby anchors
    unique: list[tuple[float, float, float]] = []
    tol_sq = adjacency_tol * adjacency_tol
    for anchor in anchors:
        if all(
            sum((anchor[k] - u[k]) ** 2 for k in range(3)) > tol_sq
            for u in unique
        ):
            unique.append(anchor)
    return unique


def _split_quad_at_bottom_param(
    corners: tuple[tuple[float, float, float], ...],
    edge_i0: int,
    t: float,
    jx: float,
    jz: float,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    """Split a 4-corner wall quad across horizontal edge edge_i0 at parameter t."""

    c = list(corners)
    n = len(c)
    if n != 4:
        raise ValueError("wall junction split expects a 4-corner quad")
    i1 = (edge_i0 + 1) % n
    i3 = (edge_i0 + n - 1) % n
    i2 = (i1 + 1) % n
    j_bot = (jx, _lerp3(c[edge_i0], c[i1], t)[1], jz)
    j_top = (jx, _lerp3(c[i3], c[i2], t)[1], jz)
    wall_a = (c[edge_i0], j_bot, j_top, c[i3])
    wall_b = (j_bot, c[i1], c[i2], j_top)
    return wall_a, wall_b


def _find_horizontal_splits(
    wall: BuildingElement,
    anchors: Sequence[tuple[float, float, float]],
    corner_tol: float,
    adjacency_tol: float,
) -> list[tuple[int, float, float, float]]:
    """(edge_i0, t, jx, jz) splits along bottom/top horizontal edges."""

    splits: list[tuple[int, float, float, float]] = []
    n = len(wall.corners)
    if n < 4:
        return splits
    for anchor in anchors:
        if _wall_has_corner_near(wall, anchor, corner_tol):
            continue
        ax, _, az = anchor
        for edge_i0 in range(n):
            p0 = wall.corners[edge_i0]
            p1 = wall.corners[(edge_i0 + 1) % n]
            if not _is_horizontal_edge(p0, p1, corner_tol):
                continue
            dist, t = _xz_point_to_segment(ax, az, p0, p1)
            if dist > adjacency_tol or t <= 0.03 or t >= 0.97:
                continue
            splits.append((edge_i0, t, ax, az))
    return splits


def _apply_wall_splits(
    wall: BuildingElement,
    splits: list[tuple[int, float, float, float]],
) -> list[BuildingElement]:
    """Apply ordered splits; returns one or more wall elements."""

    if not splits:
        return [wall]
    by_edge: dict[int, list[tuple[float, float, float]]] = {}
    for edge_i0, t, jx, jz in splits:
        by_edge.setdefault(edge_i0, []).append((t, jx, jz))

    pieces: list[tuple[tuple[float, float, float], ...]] = [wall.corners]
    for edge_i0 in sorted(by_edge):
        next_pieces: list[tuple[tuple[float, float, float], ...]] = []
        for corners in pieces:
            if len(corners) != 4:
                next_pieces.append(corners)
                continue
            current = corners
            for t, jx, jz in sorted(by_edge[edge_i0]):
                dist, t_check = _xz_point_to_segment(
                    jx,
                    jz,
                    current[edge_i0],
                    current[(edge_i0 + 1) % 4],
                )
                if dist > 0.05 or t_check <= 0.03 or t_check >= 0.97:
                    continue
                left, right = _split_quad_at_bottom_param(current, edge_i0, t, jx, jz)
                next_pieces.append(left)
                current = right
            next_pieces.append(current)
        pieces = next_pieces

    out: list[BuildingElement] = []
    for idx, corners in enumerate(pieces):
        if len(corners) < 3:
            continue
        suffix = f"::split::{idx}" if len(pieces) > 1 else ""
        out.append(
            BuildingElement(
                id=f"{wall.id}{suffix}",
                kind=wall.kind,
                locator_id=wall.locator_id,
                corners=corners,
                room_index=wall.room_index,
                story=wall.story,
            )
        )
    return out or [wall]


def split_walls_at_approx_junctions(
    elements: Sequence[BuildingElement],
    corner_tol: float,
    adjacency_tol: float,
) -> list[BuildingElement]:
    """Break walls that cross multi-wall approx junctions so corners share segments."""

    corner_vids = cluster_element_corners(elements, corner_tol)
    segments = extract_wall_vertical_segments(elements, corner_vids, corner_tol)
    approx = _approx_segment_pairs(segments, adjacency_tol)
    anchors = _multi_wall_junction_anchors(segments, approx, adjacency_tol)
    if not anchors:
        return list(elements)

    result: list[BuildingElement] = []
    for el in elements:
        if el.kind != "wall" or len(el.corners) != 4:
            result.append(el)
            continue
        splits = _find_horizontal_splits(el, anchors, corner_tol, adjacency_tol)
        result.extend(_apply_wall_splits(el, splits))
    return result
