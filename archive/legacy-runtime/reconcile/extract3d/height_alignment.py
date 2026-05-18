"""Per-story coplanar snap for rooms whose floors and ceilings are nearly flush.

Scan noise leaves neighbouring rooms on the same story with floor Ys that differ
by a few centimetres even when the real slab is continuous. The existing
dominant-cohort machinery (see ``ceilings.compute_story_wall_top_cohort``)
only fires on outliers 10-40 cm off the cohort, so 2-5 cm residuals survive and
read in the viewer as a visible step between slabs that should be flush.

``align_room_heights`` groups rooms on the same story whose floor slab Ys differ
by at most ``FLOOR_ALIGN_TOL_M`` *and* whose representative wall height differs
by at most ``WALL_HEIGHT_ALIGN_TOL_M``, then snaps each room's floor polygon,
wall bottom corners, and wall top corners to the group median. Per-wall heights
are preserved: knee walls and other intentional outliers stay put because the
corner-level snap only fires when a corner is within ``CORNER_SNAP_TOL_M`` of
the room's own min/max wall Y.
"""

from __future__ import annotations

from statistics import median

from .lineage import STEP_ALIGN_ROOM_HEIGHTS, record

FLOOR_ALIGN_TOL_M = 0.06
WALL_HEIGHT_ALIGN_TOL_M = 0.05
CORNER_SNAP_TOL_M = 0.05

_MIN_WALL_SPAN_M = 0.05
_MIN_WALL_HEIGHT_M = 0.10


def _room_floor_y(room: dict) -> float | None:
    fp = room.get("floor_polygon") or []
    if len(fp) < 3:
        return None
    return float(median(c[1] for c in fp))


def _wall_span_xz(corners: list) -> float:
    xs = [c[0] for c in corners]
    zs = [c[2] for c in corners]
    return max((max(xs) - min(xs)) ** 2 + (max(zs) - min(zs)) ** 2, 0.0) ** 0.5


def _room_wall_height(room: dict) -> float | None:
    heights = []
    for wall in room.get("walls_computed") or []:
        corners = wall.get("corners") or []
        if len(corners) < 3:
            continue
        if _wall_span_xz(corners) < _MIN_WALL_SPAN_M:
            continue
        ys = [c[1] for c in corners]
        h = max(ys) - min(ys)
        if h < _MIN_WALL_HEIGHT_M:
            continue
        heights.append(h)
    if not heights:
        return None
    return float(median(heights))


def _group_rooms(
    entries: list[tuple[float, float, dict]],
) -> list[list[tuple[float, float, dict]]]:
    """Greedy 1D sweep on floor_y with a hard 6 cm spread cap.

    ``entries`` are ``(floor_y, wall_height, room)``. A room joins the current
    group iff the resulting span stays within ``FLOOR_ALIGN_TOL_M`` *and* its
    wall height is within ``WALL_HEIGHT_ALIGN_TOL_M`` of the group's running
    wall-height median. Otherwise we start a new group. Groups of size 1 are
    filtered out upstream — they never need alignment.
    """
    entries = sorted(entries, key=lambda e: e[0])
    groups: list[list[tuple[float, float, dict]]] = []
    current: list[tuple[float, float, dict]] = []
    for entry in entries:
        fy, wh, _room = entry
        if not current:
            current.append(entry)
            continue
        group_floor_ys = [e[0] for e in current]
        group_spread = max(fy - min(group_floor_ys), max(group_floor_ys) - fy)
        spread_ok = (
            fy - min(group_floor_ys) <= FLOOR_ALIGN_TOL_M
            and group_spread <= FLOOR_ALIGN_TOL_M
        )
        wh_ok = abs(wh - median(e[1] for e in current)) <= WALL_HEIGHT_ALIGN_TOL_M
        if spread_ok and wh_ok:
            current.append(entry)
        else:
            groups.append(current)
            current = [entry]
    if current:
        groups.append(current)
    return groups


def _room_wall_ceiling_ref(room: dict) -> float | None:
    """Median wall-top Y across the room's eligible walls — the "ceiling" line.

    Used to tell a ceiling-touching wall corner apart from a knee wall: only
    corners within ``CORNER_SNAP_TOL_M`` of this reference get lifted.
    """
    tops = []
    for wall in room.get("walls_computed") or []:
        corners = wall.get("corners") or []
        if len(corners) < 3:
            continue
        if _wall_span_xz(corners) < _MIN_WALL_SPAN_M:
            continue
        ys = [c[1] for c in corners]
        if max(ys) - min(ys) < _MIN_WALL_HEIGHT_M:
            continue
        tops.append(max(ys))
    if not tops:
        return None
    return float(median(tops))


def _apply_group(
    group: list[tuple[float, float, dict]],
    metrics: dict,
) -> None:
    floor_ys = [fy for fy, _wh, _r in group]
    ceiling_ys = [fy + wh for fy, wh, _r in group]
    target_floor_y = float(median(floor_ys))
    target_ceiling_y = float(median(ceiling_ys))
    metrics["aligned_groups"] += 1

    for floor_y, _wall_h, room in group:
        room_ceiling_ref = _room_wall_ceiling_ref(room)

        # Floor polygon: unify every corner Y to the target slab Y.
        fp = room.get("floor_polygon") or []
        if len(fp) >= 3:
            before = floor_y
            for corner in fp:
                corner[1] = target_floor_y
            shift = abs(target_floor_y - before)
            metrics["max_floor_shift"] = max(metrics["max_floor_shift"], shift)
            if shift > 1e-6:
                record(
                    room,
                    STEP_ALIGN_ROOM_HEIGHTS,
                    "modified",
                    f"floor_y={before:.3f}->{target_floor_y:.3f}",
                )
        metrics["aligned_rooms"] += 1

        for wall in room.get("walls_computed") or []:
            corners = wall.get("corners") or []
            if len(corners) < 3:
                continue
            ys = [c[1] for c in corners]
            wall_min_y = min(ys)
            wall_max_y = max(ys)
            if wall_max_y - wall_min_y < _MIN_WALL_HEIGHT_M:
                continue
            # Only ceiling-touching walls (max close to the room's ceiling line)
            # are candidates for top-snap. Knee walls and partial walls fall
            # below the ``CORNER_SNAP_TOL_M`` band and stay where they are.
            top_snap_ok = (
                room_ceiling_ref is not None
                and abs(wall_max_y - room_ceiling_ref) <= CORNER_SNAP_TOL_M
            )
            # A wall only touches the floor slab if its minimum Y is within the
            # snap band of the room's original floor; partial walls starting
            # mid-air fall outside and stay put.
            bot_snap_ok = abs(wall_min_y - floor_y) <= CORNER_SNAP_TOL_M

            bot_changed = False
            top_changed = False
            bot_before = wall_min_y
            top_before = wall_max_y
            for corner in corners:
                y = corner[1]
                if bot_snap_ok and abs(y - wall_min_y) <= CORNER_SNAP_TOL_M:
                    new_y = target_floor_y
                    if abs(new_y - y) > 1e-6:
                        corner[1] = new_y
                        bot_changed = True
                elif top_snap_ok and abs(y - wall_max_y) <= CORNER_SNAP_TOL_M:
                    new_y = target_ceiling_y
                    if abs(new_y - y) > 1e-6:
                        corner[1] = new_y
                        top_changed = True
            if bot_changed or top_changed:
                metrics["aligned_walls"] += 1
                parts = []
                if bot_changed:
                    parts.append(f"bot={bot_before:.3f}->{target_floor_y:.3f}")
                    metrics["max_floor_shift"] = max(
                        metrics["max_floor_shift"], abs(target_floor_y - bot_before)
                    )
                if top_changed:
                    parts.append(f"top={top_before:.3f}->{target_ceiling_y:.3f}")
                    metrics["max_top_shift"] = max(
                        metrics["max_top_shift"], abs(target_ceiling_y - top_before)
                    )
                record(wall, STEP_ALIGN_ROOM_HEIGHTS, "modified", ", ".join(parts))


def align_room_heights(rooms_out: list[dict]) -> dict:
    """Snap coplanar room groups on each story to a common floor/ceiling Y.

    Modifies ``rooms_out`` in place: ``floor_polygon`` corner Ys and any
    ``walls_computed`` corner Y within ``CORNER_SNAP_TOL_M`` of the wall's own
    min/max. Returns a metrics dict with ``aligned_groups``, ``aligned_rooms``,
    ``aligned_walls``, ``max_floor_shift``, ``max_top_shift``.
    """
    metrics = {
        "aligned_groups": 0,
        "aligned_rooms": 0,
        "aligned_walls": 0,
        "max_floor_shift": 0.0,
        "max_top_shift": 0.0,
    }

    by_story: dict[int, list[tuple[float, float, dict]]] = {}
    for room in rooms_out:
        floor_y = _room_floor_y(room)
        wall_h = _room_wall_height(room)
        if floor_y is None or wall_h is None:
            continue
        story = room.get("story", 0)
        by_story.setdefault(story, []).append((floor_y, wall_h, room))

    for entries in by_story.values():
        if len(entries) < 2:
            continue
        for group in _group_rooms(entries):
            if len(group) < 2:
                continue
            _apply_group(group, metrics)

    return metrics
