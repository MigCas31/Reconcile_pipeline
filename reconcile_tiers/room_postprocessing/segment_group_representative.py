"""Perimeter wall selection and representative segments for segment-room cycles."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def base_wall_id(wall_id: str) -> str:
    """Physical wall id without junction-split suffix."""

    if "::split::" in wall_id:
        return wall_id.split("::split::", 1)[0]
    return wall_id


def _wall_matches_base(wall_id: str, base: str) -> bool:
    return wall_id == base or wall_id.startswith(f"{base}::split::")


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


def _wall_id_between_groups(
    group_a: str,
    group_b: str,
    part_edges: list[dict[str, Any]],
) -> str | None:
    for edge in part_edges:
        src, tgt = edge["source"], edge["target"]
        if {src, tgt} == {group_a, group_b}:
            wall_id = edge.get("wall_id")
            if wall_id:
                return str(wall_id)
    return None


def _pick_segment_on_wall_base(
    segment_ids: list[str],
    base: str,
    segments_by_id: dict[str, dict[str, Any]],
    junction_xz: tuple[float, float],
) -> str | None:
    """Segment on any split piece of ``base`` closest to the junction in XZ."""

    jx, jz = junction_xz
    best_id: str | None = None
    best_dist = math.inf
    best_height = -1.0
    for seg_id in segment_ids:
        seg = segments_by_id.get(seg_id)
        if not seg or not _wall_matches_base(str(seg.get("wall_id", "")), base):
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


def _side_rim_len(
    side: dict[str, Any],
    positions: dict[str, tuple[float, float]],
) -> float:
    pa = positions.get(side["source_group"])
    pb = positions.get(side["target_group"])
    if pa is None or pb is None:
        return 0.0
    return math.hypot(pb[0] - pa[0], pb[1] - pa[1])


def _merge_sides_for_base(
    group: list[dict[str, Any]],
    base: str,
    positions: dict[str, tuple[float, float]],
    segments_by_id: dict[str, dict[str, Any]],
    segment_ids_by_group: dict[str, list[str]],
) -> dict[str, Any]:
    """One perimeter side spanning the farthest junction pair for this physical wall."""

    groups: set[str] = set()
    for side in group:
        groups.add(side["source_group"])
        groups.add(side["target_group"])
    gs = list(groups)
    best_pair: tuple[str, str] | None = None
    best_len = -1.0
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            ga, gb = gs[i], gs[j]
            pa = positions.get(ga)
            pb = positions.get(gb)
            if pa is None or pb is None:
                continue
            length = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if length > best_len:
                best_len = length
                best_pair = (ga, gb)
    if best_pair is None:
        return group[0]
    ga, gb = best_pair
    wall_id = max(group, key=lambda s: _side_rim_len(s, positions))["wall_id"]
    ja, jb = positions[ga], positions[gb]
    seg_a = _pick_segment_on_wall_base(
        segment_ids_by_group.get(ga) or [], base, segments_by_id, ja
    )
    seg_b = _pick_segment_on_wall_base(
        segment_ids_by_group.get(gb) or [], base, segments_by_id, jb
    )
    return {
        "source_group": ga,
        "target_group": gb,
        "wall_id": wall_id,
        "segment_id_a": seg_a or group[0]["segment_id_a"],
        "segment_id_b": seg_b or group[0]["segment_id_b"],
    }


def dedupe_perimeter_sides_by_base(
    sides: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    segments_by_id: dict[str, dict[str, Any]],
    segment_ids_by_group: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """At most one perimeter side per physical wall (merge split pieces)."""

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for side in sides:
        by_base[base_wall_id(side["wall_id"])].append(side)
    out: list[dict[str, Any]] = []
    for base, group in sorted(by_base.items()):
        if len(group) == 1:
            out.append(group[0])
        else:
            out.append(
                _merge_sides_for_base(
                    group,
                    base,
                    positions,
                    segments_by_id,
                    segment_ids_by_group,
                )
            )
    return out


def perimeter_sides_for_cycle(
    cycle: list[str],
    part_edges: list[dict[str, Any]],
    segment_ids_by_group: dict[str, list[str]],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """One side per cycle edge (junction pair), deduped by physical wall id."""

    sides: list[dict[str, Any]] = []
    n = len(cycle)
    for i in range(n):
        group_a = cycle[i]
        group_b = cycle[(i + 1) % n]
        wall_id = _wall_id_between_groups(group_a, group_b, part_edges)
        if not wall_id:
            continue
        ja = positions.get(group_a)
        jb = positions.get(group_b)
        if ja is None or jb is None:
            continue
        seg_a = _pick_segment_on_wall(
            segment_ids_by_group.get(group_a) or [],
            wall_id,
            segments_by_id,
            ja,
        )
        seg_b = _pick_segment_on_wall(
            segment_ids_by_group.get(group_b) or [],
            wall_id,
            segments_by_id,
            jb,
        )
        if seg_a is None or seg_b is None:
            continue
        sides.append(
            {
                "source_group": group_a,
                "target_group": group_b,
                "wall_id": wall_id,
                "segment_id_a": seg_a,
                "segment_id_b": seg_b,
            }
        )
    return dedupe_perimeter_sides_by_base(
        sides,
        positions,
        segments_by_id,
        segment_ids_by_group,
    )


def _bottom_top_at_junction(
    seg: dict[str, Any],
    jx: float,
    jz: float,
) -> tuple[dict[str, float], dict[str, float]]:
    s = seg["start"]
    e = seg["end"]
    ds = _xz_dist_sq(s["x"], s["z"], jx, jz)
    de = _xz_dist_sq(e["x"], e["z"], jx, jz)
    near, far = (s, e) if ds <= de else (e, s)
    if float(near["y"]) <= float(far["y"]):
        bot, top = near, far
    else:
        bot, top = far, near
    return (
        {"x": float(bot["x"]), "y": float(bot["y"]), "z": float(bot["z"])},
        {"x": float(top["x"]), "y": float(top["y"]), "z": float(top["z"])},
    )


def wall_dict_from_perimeter_side(
    side: dict[str, Any],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any] | None:
    """Wall quad between two junctions using segment endpoints, not full scan mesh."""

    seg_a = segments_by_id.get(side["segment_id_a"])
    seg_b = segments_by_id.get(side["segment_id_b"])
    pa = positions.get(side["source_group"])
    pb = positions.get(side["target_group"])
    if seg_a is None or seg_b is None or pa is None or pb is None:
        return None
    bot_a, top_a = _bottom_top_at_junction(seg_a, pa[0], pa[1])
    bot_b, top_b = _bottom_top_at_junction(seg_b, pb[0], pb[1])
    return {
        "locator_id": base_wall_id(side["wall_id"]),
        "corners": [bot_a, bot_b, top_b, top_a],
    }
