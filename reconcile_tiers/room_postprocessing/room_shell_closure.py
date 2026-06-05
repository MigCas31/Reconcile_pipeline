"""Build a half-closed room shell (floor + walls, open top) with shared junction corners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reconcile_tiers.room_postprocessing.segment_group_representative import (
    base_wall_id,
)


def _junction_index(
    group_id: str,
    group_ids: Sequence[str],
) -> int | None:
    try:
        return list(group_ids).index(group_id)
    except ValueError:
        return None


def junction_corner_3d(
    group_id: str,
    *,
    group_ids: Sequence[str],
    polygon_xz: Sequence[Mapping[str, float]],
    floor_y: float,
) -> dict[str, float] | None:
    """Bottom corner at cycle junction XZ and shared floor Y."""

    idx = _junction_index(group_id, group_ids)
    if idx is None or idx >= len(polygon_xz):
        return None
    p = polygon_xz[idx]
    return {
        "x": float(p["x"]),
        "y": float(floor_y),
        "z": float(p["z"]),
    }


def junction_top_y(
    group_id: str,
    *,
    representative_by_group: Mapping[str, Sequence[str]],
    segments_by_id: Mapping[str, Mapping[str, Any]],
) -> float | None:
    """Max segment top Y among representative segments at this junction."""

    ys: list[float] = []
    for seg_id in representative_by_group.get(group_id) or []:
        seg = segments_by_id.get(str(seg_id))
        if not seg:
            continue
        for key in ("start", "end"):
            pt = seg.get(key)
            if pt is not None:
                ys.append(float(pt["y"]))
    return max(ys) if ys else None


def _junction_top_corner(
    group_id: str,
    *,
    group_ids: Sequence[str],
    polygon_xz: Sequence[Mapping[str, float]],
    floor_y: float,
    representative_by_group: Mapping[str, Sequence[str]],
    segments_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, float] | None:
    bot = junction_corner_3d(
        group_id,
        group_ids=group_ids,
        polygon_xz=polygon_xz,
        floor_y=floor_y,
    )
    if bot is None:
        return None
    top_y = junction_top_y(
        group_id,
        representative_by_group=representative_by_group,
        segments_by_id=segments_by_id,
    )
    if top_y is None or top_y < floor_y:
        top_y = floor_y
    return {"x": bot["x"], "y": float(top_y), "z": bot["z"]}


def wall_dict_from_perimeter_side_closed(
    side: Mapping[str, Any],
    *,
    group_ids: Sequence[str],
    polygon_xz: Sequence[Mapping[str, float]],
    floor_y: float,
    representative_by_group: Mapping[str, Sequence[str]],
    segments_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Vertical wall quad with shared junction corners (open top, per-junction height)."""

    src = str(side["source_group"])
    tgt = str(side["target_group"])
    bot_a = junction_corner_3d(
        src,
        group_ids=group_ids,
        polygon_xz=polygon_xz,
        floor_y=floor_y,
    )
    bot_b = junction_corner_3d(
        tgt,
        group_ids=group_ids,
        polygon_xz=polygon_xz,
        floor_y=floor_y,
    )
    top_a = _junction_top_corner(
        src,
        group_ids=group_ids,
        polygon_xz=polygon_xz,
        floor_y=floor_y,
        representative_by_group=representative_by_group,
        segments_by_id=segments_by_id,
    )
    top_b = _junction_top_corner(
        tgt,
        group_ids=group_ids,
        polygon_xz=polygon_xz,
        floor_y=floor_y,
        representative_by_group=representative_by_group,
        segments_by_id=segments_by_id,
    )
    if bot_a is None or bot_b is None or top_a is None or top_b is None:
        return None
    return {
        "locator_id": base_wall_id(str(side["wall_id"])),
        "corners": [bot_a, bot_b, top_b, top_a],
    }


def build_half_closed_room_shell(
    seg_room: Mapping[str, Any],
    segments_by_id: Mapping[str, Mapping[str, Any]],
    *,
    floor_y: float,
) -> dict[str, Any] | None:
    """Floor ring + wall quads sharing exact junction coordinates."""

    group_ids = list(seg_room.get("group_ids") or [])
    polygon_xz = seg_room.get("polygon_xz") or []
    if len(group_ids) < 3 or len(polygon_xz) < 3:
        return None

    rep_by_group = seg_room.get("representative_by_group") or {}
    perimeter_sides = seg_room.get("perimeter_sides") or []
    if len(perimeter_sides) < 3:
        return None

    floor_corners = [
        junction_corner_3d(
            gid,
            group_ids=group_ids,
            polygon_xz=polygon_xz,
            floor_y=floor_y,
        )
        for gid in group_ids
    ]
    if any(c is None for c in floor_corners):
        return None

    walls: list[dict[str, Any]] = []
    for side in perimeter_sides:
        wall = wall_dict_from_perimeter_side_closed(
            side,
            group_ids=group_ids,
            polygon_xz=polygon_xz,
            floor_y=floor_y,
            representative_by_group=rep_by_group,
            segments_by_id=segments_by_id,
        )
        if wall is not None:
            walls.append(wall)

    if len(walls) < 3:
        return None

    return {
        "floor_corners": floor_corners,
        "walls": walls,
    }
