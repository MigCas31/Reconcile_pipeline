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


def _pick_segment_closest_to_junction(
    segment_ids: list[str],
    segments_by_id: dict[str, dict[str, Any]],
    junction_xz: tuple[float, float],
) -> str | None:
    """Any segment whose bottom is closest to the group junction in XZ."""

    jx, jz = junction_xz
    best_id: str | None = None
    best_dist = math.inf
    best_height = -1.0
    for seg_id in segment_ids:
        seg = segments_by_id.get(seg_id)
        if not seg:
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


def _candidate_segments(
    group: str,
    wall_id: str,
    segment_ids_by_group: dict[str, list[str]],
    segments_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Member segments on ``wall_id`` (or split pieces of its base)."""

    if wall_id == "leaf_bridge":
        return sorted(segment_ids_by_group.get(group) or [])

    base = base_wall_id(wall_id)
    out: list[str] = []
    for seg_id in sorted(segment_ids_by_group.get(group) or []):
        seg = segments_by_id.get(seg_id)
        if not seg:
            continue
        wid = str(seg.get("wall_id", ""))
        if wid == wall_id or _wall_matches_base(wid, base):
            out.append(seg_id)
    return out


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


def _side_quad_area(
    side: dict[str, Any],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> float:
    """XZ rim length × vertical extent of the perimeter wall quad."""

    wall = wall_dict_from_perimeter_side(side, segments_by_id, positions)
    if wall is None:
        return math.inf
    corners = wall["corners"]
    if len(corners) < 4:
        return math.inf
    rim = math.hypot(
        corners[1]["x"] - corners[0]["x"],
        corners[1]["z"] - corners[0]["z"],
    )
    ys = [float(c["y"]) for c in corners]
    height = max(ys) - min(ys)
    return rim * max(height, 1e-9)


def _pick_segment_pair_min_area(
    group_a: str,
    group_b: str,
    wall_id: str,
    segment_ids_by_group: dict[str, list[str]],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> tuple[str | None, str | None]:
    """Segment pair minimizing wall quad area for this cycle edge."""

    ja = positions.get(group_a)
    jb = positions.get(group_b)
    if ja is None or jb is None:
        return None, None

    cands_a = _candidate_segments(group_a, wall_id, segment_ids_by_group, segments_by_id)
    cands_b = _candidate_segments(group_b, wall_id, segment_ids_by_group, segments_by_id)

    if not cands_a:
        cands_a = (
            [_pick_segment_closest_to_junction(
                segment_ids_by_group.get(group_a) or [], segments_by_id, ja
            )]
            if segment_ids_by_group.get(group_a)
            else []
        )
    if not cands_b:
        cands_b = (
            [_pick_segment_closest_to_junction(
                segment_ids_by_group.get(group_b) or [], segments_by_id, jb
            )]
            if segment_ids_by_group.get(group_b)
            else []
        )

    best_a: str | None = None
    best_b: str | None = None
    best_area = math.inf
    for seg_a in cands_a:
        if not seg_a:
            continue
        for seg_b in cands_b:
            if not seg_b:
                continue
            side = {
                "source_group": group_a,
                "target_group": group_b,
                "wall_id": wall_id,
                "segment_id_a": seg_a,
                "segment_id_b": seg_b,
            }
            area = _side_quad_area(side, segments_by_id, positions)
            if area < best_area - 1e-12 or (
                abs(area - best_area) <= 1e-12
                and (seg_a, seg_b) < (best_a or "", best_b or "")
            ):
                best_area = area
                best_a, best_b = seg_a, seg_b
    return best_a, best_b


def representatives_from_perimeter_sides(
    sides: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Derive representative segment ids and wall ids from perimeter sides."""

    rep_segments: list[str] = []
    rep_walls: list[str] = []
    by_group: dict[str, list[str]] = defaultdict(list)
    seen_seg: set[str] = set()

    for side in sides:
        wall_base = base_wall_id(side["wall_id"])
        if wall_base != "leaf_bridge":
            rep_walls.append(wall_base)
        for group_id, key in (
            (side["source_group"], "segment_id_a"),
            (side["target_group"], "segment_id_b"),
        ):
            seg_id = side.get(key)
            if not seg_id:
                continue
            if seg_id not in seen_seg:
                seen_seg.add(seg_id)
                rep_segments.append(seg_id)
            if seg_id not in by_group[group_id]:
                by_group[group_id].append(seg_id)

    return rep_segments, sorted(set(rep_walls)), dict(by_group)


def representative_segments_for_cycle(
    cycle: list[str],
    part_edges: list[dict[str, Any]],
    segment_ids_by_group: dict[str, list[str]],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    *,
    perimeter_sides: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Representative segments derived from min-area perimeter sides."""

    sides = perimeter_sides
    if sides is None:
        sides = perimeter_sides_for_cycle(
            cycle,
            part_edges,
            segment_ids_by_group,
            segments_by_id,
            positions,
        )
    return representatives_from_perimeter_sides(sides)


def perimeter_wall_quads_for_sides(
    sides: list[dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Junction-to-junction wall quads for each perimeter side."""

    out: list[dict[str, Any]] = []
    for side in sides:
        wall = wall_dict_from_perimeter_side(side, segments_by_id, positions)
        if wall is not None:
            out.append(wall)
    return out


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


def _merge_sides_for_base(
    group: list[dict[str, Any]],
    segments_by_id: dict[str, dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Keep the perimeter side with smallest wall quad area for this physical wall."""

    return min(
        group,
        key=lambda side: (
            _side_quad_area(side, segments_by_id, positions),
            side.get("segment_id_a") or "",
            side.get("segment_id_b") or "",
        ),
    )


def dedupe_perimeter_sides_by_base(
    sides: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    segments_by_id: dict[str, dict[str, Any]],
    segment_ids_by_group: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """At most one perimeter side per physical wall (smallest area wins)."""

    _ = segment_ids_by_group
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for side in sides:
        by_base[base_wall_id(side["wall_id"])].append(side)
    out: list[dict[str, Any]] = []
    for base, group in sorted(by_base.items()):
        if len(group) == 1:
            out.append(group[0])
        else:
            out.append(_merge_sides_for_base(group, segments_by_id, positions))
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
        seg_a, seg_b = _pick_segment_pair_min_area(
            group_a,
            group_b,
            wall_id,
            segment_ids_by_group,
            segments_by_id,
            positions,
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
