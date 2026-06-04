"""Pick one representative vertical segment per wall at each approx group on a room cycle."""

from __future__ import annotations

import math
from typing import Any


def _segment_bottom_xz(seg: dict[str, Any]) -> tuple[float, float]:
    s = seg["start"]
    e = seg["end"]
    if s["y"] <= e["y"]:
        return s["x"], s["z"]
    return e["x"], e["z"]


def _segment_vertical_extent(seg: dict[str, Any]) -> float:
    return abs(float(seg["end"]["y"]) - float(seg["start"]["y"]))


def _xz_dist_sq(ax: float, az: float, bx: float, bz: float) -> float:
    dx, dz = ax - bx, az - bz
    return dx * dx + dz * dz


def _incident_wall_ids_on_cycle(
    group_id: str,
    cycle: list[str],
    part_edges: list[dict[str, Any]],
) -> list[str]:
    """Wall ids from span edges connecting this group to its cycle neighbors."""

    if group_id not in cycle:
        return []
    n = len(cycle)
    idx = cycle.index(group_id)
    prev_id = cycle[(idx - 1) % n]
    next_id = cycle[(idx + 1) % n]
    walls: list[str] = []
    seen: set[str] = set()
    for edge in part_edges:
        src, tgt = edge["source"], edge["target"]
        wall_id = edge.get("wall_id")
        if not wall_id:
            continue
        pairs = ((group_id, prev_id), (group_id, next_id), (prev_id, group_id), (next_id, group_id))
        for a, b in pairs:
            if {src, tgt} == {a, b} and wall_id not in seen:
                walls.append(str(wall_id))
                seen.add(wall_id)
    return walls


def _pick_segment_on_wall(
    segment_ids: list[str],
    wall_id: str,
    segments_by_id: dict[str, dict[str, Any]],
    junction_xz: tuple[float, float],
) -> str | None:
    """Segment on ``wall_id`` whose bottom is closest to the group junction in XZ."""

    jx, jz = junction_xz
    best_id: str | None = None
    best_dist = math.inf
    best_height = -1.0
    for seg_id in segment_ids:
        seg = segments_by_id.get(seg_id)
        if not seg or seg.get("wall_id") != wall_id:
            continue
        bx, bz = _segment_bottom_xz(seg)
        dist = _xz_dist_sq(jx, jz, bx, bz)
        height = _segment_vertical_extent(seg)
        if dist < best_dist - 1e-12 or (
            abs(dist - best_dist) <= 1e-12 and height > best_height
        ):
            best_dist = dist
            best_height = height
            best_id = seg_id
    return best_id


def representative_segments_for_cycle(
    cycle: list[str],
    part_edges: list[dict[str, Any]],
    segment_ids_by_group: dict[str, list[str]],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """One segment per incident wall at each group on the room cycle.

    Returns:
        ``segment_ids``: ordered representative segment ids (deduped, stable order)
        ``wall_ids``: sorted unique wall ids for those segments
        ``by_group``: group id → representative segment ids at that junction
    """

    rep_segments: list[str] = []
    rep_walls: list[str] = []
    by_group: dict[str, list[str]] = {}
    seen_seg: set[str] = set()

    for group_id in cycle:
        member_ids = segment_ids_by_group.get(group_id) or []
        junction = positions.get(group_id)
        if not member_ids or junction is None:
            continue
        wall_ids = _incident_wall_ids_on_cycle(group_id, cycle, part_edges)
        if not wall_ids:
            continue
        group_reps: list[str] = []
        for wall_id in wall_ids:
            seg_id = _pick_segment_on_wall(
                member_ids, wall_id, segments_by_id, junction
            )
            if seg_id is None or seg_id in seen_seg:
                continue
            seen_seg.add(seg_id)
            group_reps.append(seg_id)
            rep_segments.append(seg_id)
            rep_walls.append(wall_id)
        if group_reps:
            by_group[group_id] = group_reps

    unique_walls = sorted(set(rep_walls))
    return rep_segments, unique_walls, by_group
