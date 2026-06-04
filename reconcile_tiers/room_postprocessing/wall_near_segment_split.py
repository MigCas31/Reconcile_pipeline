"""Split walls when a nearby wall's vertical segment projects onto their rim."""

from __future__ import annotations

from collections.abc import Sequence

from reconcile_tiers.room_postprocessing.corner_graph import cluster_element_corners
from reconcile_tiers.room_postprocessing.models import BuildingElement
from reconcile_tiers.room_postprocessing.wall_junction_split import (
    _apply_wall_splits,
    _find_horizontal_splits,
    _is_horizontal_edge,
    _lerp3,
    _wall_has_corner_near_xz,
    _xz_point_to_segment,
)
from reconcile_tiers.room_postprocessing.wall_segment_graph import (
    WallVerticalSegment,
    extract_wall_vertical_segments,
)


def _same_story(a: int | None, b: int | None) -> bool:
    return a == b


def _wall_base_id(wall_id: str) -> str:
    return wall_id.split("::split::", 1)[0]


def _segment_on_wall_family(seg: WallVerticalSegment, wall: BuildingElement) -> bool:
    base = _wall_base_id(wall.id)
    seg_base = _wall_base_id(seg.wall_id)
    return seg_base == base


def _projected_rim_anchor(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """Point on the wall horizontal rim (segment endpoint projected in XZ)."""

    return _lerp3(p0, p1, t)


def _is_floor_rim_edge(
    wall: BuildingElement,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    corner_tol: float,
) -> bool:
    """Bottom horizontal edge only — one junction point per wall in plan."""

    if not _is_horizontal_edge(p0, p1, corner_tol):
        return False
    floor_y = min(c[1] for c in wall.corners)
    return abs(p0[1] - floor_y) <= corner_tol and abs(p1[1] - floor_y) <= corner_tol


def _wall_has_vertical_segment_near_xz(
    wall: BuildingElement,
    anchor: tuple[float, float, float],
    segments: Sequence[WallVerticalSegment],
    adjacency_tol: float,
) -> bool:
    """True when this wall (or a split piece) already has a vertical edge at the site."""

    base = _wall_base_id(wall.id)
    tol_sq = adjacency_tol * adjacency_tol
    ax, az = anchor[0], anchor[2]
    for seg in segments:
        if _wall_base_id(seg.wall_id) != base:
            continue
        for pt in (seg.start, seg.end):
            d2 = (pt[0] - ax) ** 2 + (pt[2] - az) ** 2
            if d2 <= tol_sq:
                return True
    return False


def _anchors_from_segments_near_walls(
    segments: Sequence[WallVerticalSegment],
    walls: Sequence[BuildingElement],
    corner_tol: float,
    adjacency_tol: float,
) -> list[tuple[float, float, float]]:
    """Junction sites where a segment endpoint is near another wall's horizontal rim."""

    anchors: list[tuple[float, float, float]] = []
    tol_sq = adjacency_tol * adjacency_tol

    for seg in segments:
        for pt in (seg.start, seg.end):
            px, _, pz = pt
            for wall in walls:
                if wall.kind != "wall" or len(wall.corners) != 4:
                    continue
                if _segment_on_wall_family(seg, wall):
                    continue
                if not _same_story(seg.story, wall.story):
                    continue
                n = len(wall.corners)
                if "::split::" in wall.id:
                    continue
                for edge_i0 in range(n):
                    p0 = wall.corners[edge_i0]
                    p1 = wall.corners[(edge_i0 + 1) % n]
                    if not _is_floor_rim_edge(wall, p0, p1, corner_tol):
                        continue
                    dist, t = _xz_point_to_segment(px, pz, p0, p1)
                    if dist > adjacency_tol or t <= 0.03 or t >= 0.97:
                        continue
                    anchor = _projected_rim_anchor(p0, p1, t)
                    if _wall_has_corner_near_xz(wall, anchor, corner_tol):
                        continue
                    if _wall_has_vertical_segment_near_xz(
                        wall, anchor, segments, corner_tol
                    ):
                        continue
                    if all(
                        (anchor[0] - u[0]) ** 2 + (anchor[2] - u[2]) ** 2 > tol_sq
                        for u in anchors
                    ):
                        anchors.append(anchor)
    return anchors


def split_walls_at_near_segments(
    elements: Sequence[BuildingElement],
    corner_tol: float,
    adjacency_tol: float,
    *,
    max_passes: int = 2,
) -> list[BuildingElement]:
    """Insert corners on walls approached by another wall's vertical segment (XZ proj)."""

    elements_list = list(elements)
    for _ in range(max_passes):
        walls = [el for el in elements_list if el.kind == "wall" and len(el.corners) == 4]
        if not walls:
            break
        corner_vids = cluster_element_corners(elements_list, corner_tol)
        segments = extract_wall_vertical_segments(elements_list, corner_vids, corner_tol)
        anchors = _anchors_from_segments_near_walls(
            segments,
            walls,
            corner_tol,
            adjacency_tol,
        )
        if not anchors:
            break

        changed = False
        next_elements: list[BuildingElement] = []
        for el in elements_list:
            if (
                el.kind != "wall"
                or len(el.corners) != 4
                or "::split::" in el.id
            ):
                next_elements.append(el)
                continue
            splits = _find_horizontal_splits(el, anchors, corner_tol, adjacency_tol)
            pieces = _apply_wall_splits(el, splits, corner_tol=corner_tol)
            if len(pieces) > 1:
                changed = True
            next_elements.extend(pieces)
        elements_list = next_elements
        if not changed:
            break
    return elements_list
