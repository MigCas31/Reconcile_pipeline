from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median

from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)

FLOOR_ALIGN_TOL_M = 0.06
WALL_HEIGHT_ALIGN_TOL_M = 0.05

# Two related but distinct tolerances. CORNER_SNAP_TOL_M selects which corners
# of a wall count as "floor corners" vs "ceiling corners" vs middle (and
# whether a wall is even compatible with the room's claimed floor / ceiling).
# MEMBER_SHIFT_CAP_M caps how far a flat-ceiling room can be shifted by the
# LWM merge -- bigger shifts mean more aggressive harmonization but more
# downstream blast radius. They are tuned independently.
CORNER_SNAP_TOL_M = 0.05
MEMBER_SHIFT_CAP_M = 0.10

MIN_WALL_SPAN_M = 0.05
MIN_WALL_HEIGHT_M = 0.10

# A room is "flat-ceiling" if every wall in the room tops out within this
# vertical band. Sloped/vaulted rooms exceed it and stay on the conservative
# alignment path.
FLAT_CEILING_TOL_M = 0.05


@dataclass(frozen=True, slots=True)
class _RoomEntry:
    room_idx: int
    floor_y: float
    height: float
    total_wall_length: float
    total_length_height: float  # sum over walls of wall_length * wall_height
    flat_ceiling: bool


def _wall_span_xz(corners: list[list[float]]) -> float:
    xs = [corner[0] for corner in corners]
    zs = [corner[2] for corner in corners]
    return max((max(xs) - min(xs)) ** 2 + (max(zs) - min(zs)) ** 2, 0.0) ** 0.5


def _valid_wall_metrics(
    wall: ExtractedWall,
) -> tuple[float, float, float, float] | None:
    """Return (min_y, max_y, height, length) for walls that pass the size gates."""
    corners = wall.corners
    if len(corners) < 3:
        return None
    length = _wall_span_xz(corners)
    if length < MIN_WALL_SPAN_M:
        return None
    ys = [corner[1] for corner in corners]
    height = max(ys) - min(ys)
    if height < MIN_WALL_HEIGHT_M:
        return None
    return min(ys), max(ys), height, length


def _room_floor_y(room: ExtractedRoom) -> float | None:
    if len(room.floor_polygon) < 3:
        return None
    return float(median(corner[1] for corner in room.floor_polygon))


def _room_wall_height(room: ExtractedRoom) -> float | None:
    heights = [m[2] for wall in room.walls_computed if (m := _valid_wall_metrics(wall))]
    return float(median(heights)) if heights else None


def _room_wall_ceiling_ref(room: ExtractedRoom) -> float | None:
    tops = [m[1] for wall in room.walls_computed if (m := _valid_wall_metrics(wall))]
    return float(median(tops)) if tops else None


def _room_is_flat_ceiling(room: ExtractedRoom) -> bool:
    """True if every wall in the room tops out within FLAT_CEILING_TOL_M.

    Single-wall rooms default to flat (nothing to disagree with). Rooms with
    mixed wall-top elevations are oblique: vaulted ceilings, dormers,
    half-floors. Those keep the conservative alignment path."""
    tops = [m[1] for wall in room.walls_computed if (m := _valid_wall_metrics(wall))]
    if len(tops) < 2:
        return True
    return (max(tops) - min(tops)) <= FLAT_CEILING_TOL_M


def _room_length_sums(room: ExtractedRoom) -> tuple[float, float]:
    """(Σ wall_length, Σ wall_length x wall_height) over walls that pass gates.

    These are the moments needed for length-weighted-mean snap targets that
    preserve total wall area exactly."""
    total_l = 0.0
    total_lh = 0.0
    for wall in room.walls_computed:
        m = _valid_wall_metrics(wall)
        if m is None:
            continue
        _, _, height, length = m
        total_l += length
        total_lh += length * height
    return total_l, total_lh


def _build_entry(idx: int, room: ExtractedRoom) -> _RoomEntry | None:
    floor_y = _room_floor_y(room)
    height = _room_wall_height(room)
    if floor_y is None or height is None:
        return None
    total_l, total_lh = _room_length_sums(room)
    if total_l < MIN_WALL_SPAN_M:
        return None
    return _RoomEntry(
        room_idx=idx,
        floor_y=floor_y,
        height=height,
        total_wall_length=total_l,
        total_length_height=total_lh,
        flat_ceiling=_room_is_flat_ceiling(room),
    )


def _group_oblique(entries: list[_RoomEntry]) -> list[list[_RoomEntry]]:
    """Cumulative-spread + median-height grouping (the legacy rule).

    Used for rooms whose ceiling is sloped -- we don't loosen the tolerance
    there because ceiling slope is a real architectural signal."""
    sorted_entries = sorted(entries, key=lambda entry: entry.floor_y)
    groups: list[list[_RoomEntry]] = []
    current: list[_RoomEntry] = []
    for entry in sorted_entries:
        if not current:
            current.append(entry)
            continue
        floor_ys = [item.floor_y for item in current]
        spread_ok = (
            entry.floor_y - min(floor_ys) <= FLOOR_ALIGN_TOL_M
            and max(entry.floor_y - min(floor_ys), max(floor_ys) - entry.floor_y)
            <= FLOOR_ALIGN_TOL_M
        )
        median_h = median(item.height for item in current)
        height_ok = abs(entry.height - median_h) <= WALL_HEIGHT_ALIGN_TOL_M
        if spread_ok and height_ok:
            current.append(entry)
        else:
            groups.append(current)
            current = [entry]
    if current:
        groups.append(current)
    return groups


def _group_flat(entries: list[_RoomEntry]) -> list[list[_RoomEntry]]:
    """Cap-aware greedy clustering for flat-ceiling rooms.

    Sort by floor_y. Walk the rooms in order, accumulating into a group as long
    as the resulting LWM snap target keeps every member's floor AND ceiling
    shift within MEMBER_SHIFT_CAP_M. When adding the next room would breach
    the cap, close the current group and start a new one.

    Mode breaks emerge naturally -- a gap larger than the cap forces a shift
    that exceeds the cap somewhere in the candidate group, so the greedy cuts
    there."""
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda e: e.floor_y)
    groups: list[list[_RoomEntry]] = []
    current: list[_RoomEntry] = []
    for entry in sorted_entries:
        if not current:
            current = [entry]
            continue
        candidate = [*current, entry]
        target_floor_y, target_ceiling_y = _length_weighted_targets(candidate)
        if (
            _max_member_shift(candidate, target_floor_y, target_ceiling_y)
            > MEMBER_SHIFT_CAP_M
        ):
            groups.append(current)
            current = [entry]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _length_weighted_targets(group: list[_RoomEntry]) -> tuple[float, float]:
    """Return (target_floor_y, target_ceiling_y) snapping targets that preserve
    the group's total wall area exactly.

    Math: total area = Σ (wall_length x wall_height). After snap, every wall in
    the group has the same height H = target_ceiling_y - target_floor_y, so the
    new total is H x Σ wall_length. Equating these gives
        H = Σ (wall_length x wall_height) / Σ wall_length
    i.e. the length-weighted mean of wall heights. Floor_y is free (area is
    invariant to it) -- we pick the length-weighted mean of room floor_y for
    visual continuity."""
    total_length = sum(e.total_wall_length for e in group)
    total_length_height = sum(e.total_length_height for e in group)
    target_height = total_length_height / total_length
    target_floor_y = sum(e.total_wall_length * e.floor_y for e in group) / total_length
    return target_floor_y, target_floor_y + target_height


def _median_targets(group: list[_RoomEntry]) -> tuple[float, float]:
    """Legacy median targets used for oblique-ceiling groups."""
    floor_ys = [e.floor_y for e in group]
    ceiling_ys = [e.floor_y + e.height for e in group]
    target_floor_y = float(median(floor_ys))
    target_ceiling_y = float(median(ceiling_ys))
    return target_floor_y, target_ceiling_y


def _max_member_shift(
    group: list[_RoomEntry],
    target_floor_y: float,
    target_ceiling_y: float,
) -> float:
    """Largest shift any member room would absorb under the snap target."""
    worst = 0.0
    for entry in group:
        worst = max(worst, abs(target_floor_y - entry.floor_y))
        room_ceiling_y = entry.floor_y + entry.height
        worst = max(worst, abs(target_ceiling_y - room_ceiling_y))
    return worst


def _per_story_y_bounds(rooms: list[ExtractedRoom]) -> dict[int, tuple[float, float]]:
    """Per-story (min room floor_y, max room ceiling_y) using the room's own
    floor / ceiling references -- NOT raw wall extremes.

    Scan walls sometimes extend past the room's actual ceiling (they bridge to
    the next story for clip_walls_to_story_bounds to chop later). Using
    `max(wall_max_y)` here over-constrains the inter-story clamp because those
    extended walls aren't real ceilings. Median of valid wall tops per room is
    a much better proxy for the "natural" ceiling -- the same signal
    `_room_wall_ceiling_ref` already uses for snap selection."""
    bounds: dict[int, tuple[float, float]] = {}
    for room in rooms:
        floor_y = _room_floor_y(room)
        ceiling_ref = _room_wall_ceiling_ref(room)
        if floor_y is None or ceiling_ref is None:
            continue
        story_min, story_max = bounds.get(room.story, (float("inf"), float("-inf")))
        bounds[room.story] = (
            min(story_min, floor_y),
            max(story_max, ceiling_ref),
        )
    return bounds


def _clamp_targets_by_neighbour_stories(
    story: int,
    target_floor_y: float,
    target_ceiling_y: float,
    story_bounds: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """Clamp the LWM snap target so the group's floor cannot dip below the
    story below's max ceiling, and its ceiling cannot rise above the story
    above's min floor. Conservative (no XZ check) -- over-clamps in rare
    non-overlapping cases, but never introduces inter-story overlap."""
    floor_floor = story_bounds.get(story - 1, (float("-inf"), float("-inf")))[1]
    ceiling_ceiling = story_bounds.get(story + 1, (float("inf"), float("inf")))[0]
    target_floor_y = max(target_floor_y, floor_floor)
    target_ceiling_y = min(target_ceiling_y, ceiling_ceiling)
    return target_floor_y, target_ceiling_y


@dataclass(frozen=True, slots=True)
class _AnchorPoint:
    """A snapped (anchor) room contributing a Y delta to its same-story
    neighbours during propagation."""

    story: int
    xz_centroid: tuple[float, float]
    delta: float


def _room_xz_centroid(room: ExtractedRoom) -> tuple[float, float]:
    if not room.floor_polygon:
        return (0.0, 0.0)
    xs = [corner[0] for corner in room.floor_polygon]
    zs = [corner[2] for corner in room.floor_polygon]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


def _translate_element_y(element: ExtractedElement, shift: float) -> ExtractedElement:
    return replace(
        element,
        corners=[
            [corner[0], corner[1] + shift, corner[2]] for corner in element.corners
        ],
    )


def _translate_elements_y(
    elements: list[ExtractedElement], shift: float
) -> list[ExtractedElement]:
    if abs(shift) < 1e-12:
        return elements
    return [_translate_element_y(element, shift) for element in elements]


def _remap_y(
    y: float,
    floor_y: float,
    room_ceiling_ref: float | None,
    target_floor_y: float,
    target_ceiling_y: float,
) -> float:
    if room_ceiling_ref is None or abs(room_ceiling_ref - floor_y) <= 1e-9:
        return y + (target_floor_y - floor_y)
    t = (y - floor_y) / (room_ceiling_ref - floor_y)
    return target_floor_y + t * (target_ceiling_y - target_floor_y)


def _remap_element_y(
    element: ExtractedElement,
    floor_y: float,
    room_ceiling_ref: float | None,
    target_floor_y: float,
    target_ceiling_y: float,
) -> ExtractedElement:
    return replace(
        element,
        corners=[
            [
                corner[0],
                _remap_y(
                    corner[1],
                    floor_y,
                    room_ceiling_ref,
                    target_floor_y,
                    target_ceiling_y,
                ),
                corner[2],
            ]
            for corner in element.corners
        ],
    )


def _remap_elements_y(
    elements: list[ExtractedElement],
    floor_y: float,
    room_ceiling_ref: float | None,
    target_floor_y: float,
    target_ceiling_y: float,
) -> list[ExtractedElement]:
    if abs(target_floor_y - floor_y) < 1e-12 and (
        room_ceiling_ref is None or abs(target_ceiling_y - room_ceiling_ref) < 1e-12
    ):
        return elements
    return [
        _remap_element_y(
            element, floor_y, room_ceiling_ref, target_floor_y, target_ceiling_y
        )
        for element in elements
    ]


def _translate_room_y(room: ExtractedRoom, shift: float) -> ExtractedRoom:
    """Bulk-shift every room-local geometric corner by `shift` in Y.

    Used for non-anchor rooms during propagation so they track the same-story
    anchors without the snap target being applied per-corner -- preserves the
    room's wall height exactly."""
    new_floor = [
        [corner[0], corner[1] + shift, corner[2]] for corner in room.floor_polygon
    ]
    new_walls = [
        replace(
            wall,
            corners=[
                [corner[0], corner[1] + shift, corner[2]] for corner in wall.corners
            ],
        )
        for wall in room.walls_computed
    ]
    return replace(
        room,
        floor_polygon=new_floor,
        walls_computed=new_walls,
        doors=_translate_elements_y(room.doors, shift),
        windows=_translate_elements_y(room.windows, shift),
        openings=_translate_elements_y(room.openings, shift),
    )


def _propagate_to_non_anchors(
    rooms: list[ExtractedRoom],
    anchor_indices: set[int],
    anchors: list[_AnchorPoint],
) -> tuple[list[ExtractedRoom], int]:
    """Translate non-anchor rooms by a 1/distance-weighted blend of same-story
    anchor deltas. Same-story only -- adjacent stories rely on the inter-story
    clamp for consistency. Skip-floors / mezzanines share `room.story` so they
    flow through the same path."""
    out = list(rooms)
    translated = 0
    by_story: dict[int, list[_AnchorPoint]] = {}
    for anchor in anchors:
        by_story.setdefault(anchor.story, []).append(anchor)
    for idx, room in enumerate(rooms):
        if idx in anchor_indices:
            continue
        story_anchors = by_story.get(room.story)
        if not story_anchors:
            continue
        cx, cz = _room_xz_centroid(room)
        weighted_delta = 0.0
        weight_sum = 0.0
        for anchor in story_anchors:
            ax, az = anchor.xz_centroid
            dist = ((ax - cx) ** 2 + (az - cz) ** 2) ** 0.5
            weight = 1.0 / max(dist, 1e-3)
            weighted_delta += weight * anchor.delta
            weight_sum += weight
        if weight_sum <= 0:
            continue
        shift = weighted_delta / weight_sum
        if abs(shift) < 1e-6:
            continue
        out[idx] = _translate_room_y(room, shift)
        translated += 1
    return out, translated


def _align_wall(
    wall: ExtractedWall,
    floor_y: float,
    room_ceiling_ref: float | None,
    target_floor_y: float,
    target_ceiling_y: float,
) -> tuple[ExtractedWall, bool, float, float]:
    corners = [list(corner) for corner in wall.corners]
    if len(corners) < 3:
        return wall, False, 0.0, 0.0
    ys = [corner[1] for corner in corners]
    wall_min_y = min(ys)
    wall_max_y = max(ys)
    if wall_max_y - wall_min_y < MIN_WALL_HEIGHT_M:
        return wall, False, 0.0, 0.0

    top_snap_ok = (
        room_ceiling_ref is not None
        and abs(wall_max_y - room_ceiling_ref) <= CORNER_SNAP_TOL_M
    )
    bottom_snap_ok = abs(wall_min_y - floor_y) <= CORNER_SNAP_TOL_M
    bottom_shift = 0.0
    top_shift = 0.0
    changed = False
    for corner in corners:
        y = corner[1]
        if bottom_snap_ok and abs(y - wall_min_y) <= CORNER_SNAP_TOL_M:
            if abs(target_floor_y - y) > 1e-6:
                corner[1] = target_floor_y
                bottom_shift = max(bottom_shift, abs(target_floor_y - y))
                changed = True
        elif top_snap_ok and abs(y - wall_max_y) <= CORNER_SNAP_TOL_M:
            if abs(target_ceiling_y - y) > 1e-6:
                corner[1] = target_ceiling_y
                top_shift = max(top_shift, abs(target_ceiling_y - y))
                changed = True
    if not changed:
        return wall, False, 0.0, 0.0
    return replace(wall, corners=corners), True, bottom_shift, top_shift


def align_room_heights(
    rooms: list[ExtractedRoom],
) -> tuple[list[ExtractedRoom], dict[str, float | int]]:
    metrics: dict[str, float | int] = {
        "aligned_groups": 0,
        "aligned_rooms": 0,
        "aligned_walls": 0,
        "max_floor_shift": 0.0,
        "max_top_shift": 0.0,
        "flat_ceiling_rooms": 0,
        "oblique_ceiling_rooms": 0,
        "translated_rooms": 0,
    }
    anchor_indices: set[int] = set()
    anchors: list[_AnchorPoint] = []

    by_story: dict[int, list[_RoomEntry]] = {}
    for idx, room in enumerate(rooms):
        entry = _build_entry(idx, room)
        if entry is None:
            continue
        if entry.flat_ceiling:
            metrics["flat_ceiling_rooms"] += 1
        else:
            metrics["oblique_ceiling_rooms"] += 1
        by_story.setdefault(room.story, []).append(entry)

    story_bounds = _per_story_y_bounds(rooms)

    out = list(rooms)
    for story_idx, entries in by_story.items():
        if len(entries) < 2:
            continue

        flat = [e for e in entries if e.flat_ceiling]
        oblique = [e for e in entries if not e.flat_ceiling]
        flat_groups = [(group, True) for group in _group_flat(flat)]
        oblique_groups = [(group, False) for group in _group_oblique(oblique)]

        for group, is_flat in flat_groups + oblique_groups:
            if len(group) < 2:
                continue
            if is_flat:
                target_floor_y, target_ceiling_y = _length_weighted_targets(group)
            else:
                target_floor_y, target_ceiling_y = _median_targets(group)
            target_floor_y, target_ceiling_y = _clamp_targets_by_neighbour_stories(
                story_idx, target_floor_y, target_ceiling_y, story_bounds
            )
            # After clamping, re-verify the per-room shift cap.
            if (
                _max_member_shift(group, target_floor_y, target_ceiling_y)
                > MEMBER_SHIFT_CAP_M
            ):
                continue
            metrics["aligned_groups"] += 1

            for entry in group:
                room = out[entry.room_idx]
                room_ceiling_ref = _room_wall_ceiling_ref(room)
                floor_polygon = [list(corner) for corner in room.floor_polygon]
                if len(floor_polygon) >= 3:
                    for corner in floor_polygon:
                        corner[1] = target_floor_y
                    metrics["max_floor_shift"] = max(
                        float(metrics["max_floor_shift"]),
                        abs(target_floor_y - entry.floor_y),
                    )

                aligned_walls = []
                for wall in room.walls_computed:
                    aligned_wall, changed, bottom_shift, top_shift = _align_wall(
                        wall,
                        entry.floor_y,
                        room_ceiling_ref,
                        target_floor_y,
                        target_ceiling_y,
                    )
                    aligned_walls.append(aligned_wall)
                    if changed:
                        metrics["aligned_walls"] += 1
                        metrics["max_floor_shift"] = max(
                            float(metrics["max_floor_shift"]), bottom_shift
                        )
                        metrics["max_top_shift"] = max(
                            float(metrics["max_top_shift"]), top_shift
                        )

                out[entry.room_idx] = replace(
                    room,
                    floor_polygon=floor_polygon,
                    walls_computed=aligned_walls,
                    doors=_remap_elements_y(
                        room.doors,
                        entry.floor_y,
                        room_ceiling_ref,
                        target_floor_y,
                        target_ceiling_y,
                    ),
                    windows=_remap_elements_y(
                        room.windows,
                        entry.floor_y,
                        room_ceiling_ref,
                        target_floor_y,
                        target_ceiling_y,
                    ),
                    openings=_remap_elements_y(
                        room.openings,
                        entry.floor_y,
                        room_ceiling_ref,
                        target_floor_y,
                        target_ceiling_y,
                    ),
                )
                metrics["aligned_rooms"] += 1
                anchor_indices.add(entry.room_idx)
                anchors.append(
                    _AnchorPoint(
                        story=story_idx,
                        xz_centroid=_room_xz_centroid(rooms[entry.room_idx]),
                        delta=target_floor_y - entry.floor_y,
                    )
                )

    if anchors:
        out, translated = _propagate_to_non_anchors(out, anchor_indices, anchors)
        metrics["translated_rooms"] = translated

    return out, metrics
