from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)

MIN_OVERLAP_M2 = 0.01
MAX_HALF_FLOOR_M = 1.50
OVERLAP_BUFFER_M = 0.15
CONTAINED_ROOM_COVER_RATIO = 0.98
NESTED_ROOM_CUT_WIDTH_M = 0.02
NESTED_ROOM_CUT_BUFFER_M = 1e-6


def _floor_polygon_to_shapely(floor_polygon: list[list[float]]) -> Polygon | None:
    if len(floor_polygon) < 3:
        return None
    coords = [(corner[0], corner[2]) for corner in floor_polygon]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = make_valid(Polygon(coords))
    if poly.is_empty or poly.area < MIN_OVERLAP_M2:
        return None
    if isinstance(poly, Polygon):
        return poly
    parts = _decompose_polys(poly)
    return max(parts, key=lambda part: part.area) if parts else None


def _decompose_polys(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [item for item in getattr(geom, "geoms", []) if isinstance(item, Polygon)]


def _element_xz_midpoint(corners: list[list[float]]) -> tuple[float, float] | None:
    if len(corners) < 2:
        return None
    return (
        float(np.mean([corner[0] for corner in corners])),
        float(np.mean([corner[2] for corner in corners])),
    )


def _max_opening_top_y(room: ExtractedRoom) -> float | None:
    tops = [
        max(corner[1] for corner in element.corners)
        for element in [*room.doors, *room.windows, *room.openings]
        if len(element.corners) >= 3
    ]
    return max(tops) if tops else None


def _elements_in_overlap(
    elements: list[ExtractedElement],
    overlap_poly: Polygon,
    buffer: float = OVERLAP_BUFFER_M,
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    buffered = overlap_poly.buffer(buffer)
    inside: list[ExtractedElement] = []
    outside: list[ExtractedElement] = []
    for element in elements:
        midpoint = _element_xz_midpoint(element.corners)
        if midpoint is not None and buffered.contains(Point(midpoint)):
            inside.append(element)
        else:
            outside.append(element)
    return inside, outside


def _wall_base_segment_xz(corners: list[list[float]]) -> LineString | None:
    if len(corners) < 2:
        return None
    start = (float(corners[0][0]), float(corners[0][2]))
    end = (float(corners[1][0]), float(corners[1][2]))
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
        return None
    return LineString([start, end])


def _wall_xz_normal(corners: list[list[float]]) -> tuple[float, float] | None:
    if len(corners) < 2:
        return None
    dx = corners[1][0] - corners[0][0]
    dz = corners[1][2] - corners[0][2]
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return None
    return (-dz / length, dx / length)


def _wall_segment_overlap_with_region(
    wall: ExtractedWall,
    region_poly: Polygon,
    buffer: float = OVERLAP_BUFFER_M,
) -> tuple[float, object | None]:
    if region_poly.is_empty:
        return 0.0, None
    segment = _wall_base_segment_xz(wall.corners)
    if segment is None:
        return 0.0, None
    clipped = segment.intersection(region_poly.buffer(buffer))
    if clipped.is_empty:
        return 0.0, None
    return float(clipped.length), clipped


def _project_line_interval(
    line, origin: np.ndarray, direction: np.ndarray
) -> tuple[float, float] | None:
    coords = []
    if getattr(line, "geom_type", "") == "LineString":
        coords = list(line.coords)
    else:
        for geom in getattr(line, "geoms", []):
            if getattr(geom, "geom_type", "") == "LineString":
                coords.extend(list(geom.coords))
    if not coords:
        return None
    ts = [
        (coord[0] - origin[0]) * direction[0] + (coord[1] - origin[1]) * direction[1]
        for coord in coords
    ]
    return min(ts), max(ts)


def _winner_wall_covers_overlap_segment(
    wall: ExtractedWall,
    winner_walls: list[ExtractedWall],
    overlap_poly: Polygon,
    *,
    buffer: float = OVERLAP_BUFFER_M,
    max_offset_m: float = 0.2,
    max_angle_deg: float = 12.0,
    min_overlap_fraction: float = 0.35,
) -> bool:
    wall_segment = _wall_base_segment_xz(wall.corners)
    if wall_segment is None:
        return True
    overlap_len, overlap_segment = _wall_segment_overlap_with_region(
        wall, overlap_poly, buffer=buffer
    )
    if overlap_segment is None or overlap_len <= 1e-6:
        return False

    direction = np.array(wall_segment.coords[1]) - np.array(wall_segment.coords[0])
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= 1e-6:
        return True
    unit_dir = direction / segment_length
    origin = np.array(wall_segment.coords[0], dtype=float)
    interval = _project_line_interval(overlap_segment, origin, unit_dir)
    if interval is None:
        return False
    target_start, target_end = interval
    target_length = max(target_end - target_start, 0.0)
    if target_length <= 1e-6:
        return False

    loser_normal = _wall_xz_normal(wall.corners)
    for winner_wall in winner_walls:
        winner_segment = _wall_base_segment_xz(winner_wall.corners)
        winner_normal = _wall_xz_normal(winner_wall.corners)
        if winner_segment is None or loser_normal is None or winner_normal is None:
            continue
        dot = abs(
            loser_normal[0] * winner_normal[0] + loser_normal[1] * winner_normal[1]
        )
        angle_deg = math.degrees(math.acos(min(dot, 1.0)))
        if angle_deg > max_angle_deg:
            continue
        if float(winner_segment.distance(overlap_segment)) > max_offset_m:
            continue
        winner_interval = _project_line_interval(winner_segment, origin, unit_dir)
        if winner_interval is None:
            continue
        winner_start, winner_end = winner_interval
        projected_overlap = max(
            0.0, min(target_end, winner_end) - max(target_start, winner_start)
        )
        if projected_overlap / target_length >= min_overlap_fraction:
            return True
    return False


def _floor_from_polygon(poly: Polygon, floor_y: float) -> list[list[float]]:
    return [[coord[0], floor_y, coord[1]] for coord in list(poly.exterior.coords)[:-1]]


def _lines_from_geometry(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    return [
        item
        for item in getattr(geom, "geoms", [])
        if isinstance(item, LineString) and item.length > 1e-6
    ]


def _clip_wall_to_region(
    wall: ExtractedWall,
    region_poly: Polygon,
    *,
    buffer: float = OVERLAP_BUFFER_M,
    min_length_m: float = 0.10,
) -> ExtractedWall | None:
    wall_segment = _wall_base_segment_xz(wall.corners)
    if wall_segment is None or region_poly.is_empty:
        return None
    clipped = wall_segment.intersection(region_poly.buffer(buffer))
    parts = [
        part for part in _lines_from_geometry(clipped) if part.length >= min_length_m
    ]
    if not parts:
        return None
    part = max(parts, key=lambda item: item.length)
    coords = list(part.coords)
    if len(coords) < 2:
        return None

    original_coords = list(wall_segment.coords)
    direction = np.array(original_coords[1]) - np.array(original_coords[0])
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return None
    unit_dir = direction / length
    origin = np.array(original_coords[0], dtype=float)
    endpoints = [
        np.array(coords[0], dtype=float),
        np.array(coords[-1], dtype=float),
    ]
    endpoints.sort(key=lambda coord: float((coord - origin) @ unit_dir))

    def y_pair_at(original_point: tuple[float, float]) -> tuple[float, float]:
        candidates = [
            float(corner[1])
            for corner in wall.corners
            if math.hypot(
                corner[0] - original_point[0],
                corner[2] - original_point[1],
            )
            < 1e-4
        ]
        if not candidates:
            candidates = [float(corner[1]) for corner in wall.corners]
        return min(candidates), max(candidates)

    original_start_bottom_y, original_start_top_y = y_pair_at(original_coords[0])
    original_end_bottom_y, original_end_top_y = y_pair_at(original_coords[1])

    def y_at(endpoint: np.ndarray, y0: float, y1: float) -> float:
        t = float((endpoint - origin) @ unit_dir) / length
        t = max(0.0, min(1.0, t))
        return y0 + t * (y1 - y0)

    start, end = endpoints
    start_top_y = y_at(start, original_start_top_y, original_end_top_y)
    end_top_y = y_at(end, original_start_top_y, original_end_top_y)
    start_bottom_y = y_at(start, original_start_bottom_y, original_end_bottom_y)
    end_bottom_y = y_at(end, original_start_bottom_y, original_end_bottom_y)
    return replace(
        wall,
        corners=[
            [float(start[0]), start_top_y, float(start[1])],
            [float(end[0]), end_top_y, float(end[1])],
            [float(end[0]), end_bottom_y, float(end[1])],
            [float(start[0]), start_bottom_y, float(start[1])],
        ],
    )


def _element_near_wall(
    element: ExtractedElement,
    wall: ExtractedWall,
    *,
    max_distance_m: float = 0.20,
) -> bool:
    midpoint = _element_xz_midpoint(element.corners)
    wall_segment = _wall_base_segment_xz(wall.corners)
    if midpoint is None or wall_segment is None:
        return False
    return float(wall_segment.distance(Point(midpoint))) <= max_distance_m


def _element_belongs_to_room_walls(
    element: ExtractedElement,
    room: ExtractedRoom,
) -> bool:
    walls = room.walls_computed
    if element.parent_wall_id:
        return any(wall.id == element.parent_wall_id for wall in walls)
    return any(_element_near_wall(element, wall) for wall in walls)


def _element_belongs_to_walls(
    element: ExtractedElement,
    walls: list[ExtractedWall],
) -> bool:
    if element.parent_wall_id:
        return any(wall.id == element.parent_wall_id for wall in walls)
    return any(_element_near_wall(element, wall) for wall in walls)


def _wall_tracks_region_boundary(
    wall: ExtractedWall,
    region_poly: Polygon,
    *,
    tolerance_m: float = 0.12,
    min_fraction: float = 0.50,
) -> bool:
    wall_segment = _wall_base_segment_xz(wall.corners)
    if wall_segment is None or region_poly.is_empty:
        return False
    boundary_band = region_poly.boundary.buffer(tolerance_m, cap_style="flat")
    overlap = wall_segment.intersection(boundary_band)
    return float(overlap.length) / float(wall_segment.length) >= min_fraction


def _room_has_slanted_wall_top(
    room: ExtractedRoom,
    min_span_m: float = 0.15,
) -> bool:
    for wall in room.walls_computed or room.walls_merged:
        ys = [corner[1] for corner in wall.corners]
        if not ys:
            continue
        mid_y = (max(ys) + min(ys)) / 2.0
        top_ys = [corner[1] for corner in wall.corners if corner[1] > mid_y - 0.01]
        if top_ys and max(top_ys) - min(top_ys) > min_span_m:
            return True
    return False


def _has_nested_room_evidence(room: ExtractedRoom) -> bool:
    walls = room.walls_computed or room.walls_merged
    has_boundary = len(walls) >= 3
    has_room_signal = bool(room.doors or room.windows or room.openings or room.storages)
    return has_boundary and has_room_signal


def _should_preserve_contained_room(
    room: ExtractedRoom,
    poly: Polygon,
    overlap: object,
    clipped_poly: Polygon,
) -> bool:
    if clipped_poly.area >= MIN_OVERLAP_M2 or not _has_nested_room_evidence(room):
        return False
    if poly.area <= MIN_OVERLAP_M2:
        return False
    overlap_area = float(getattr(overlap, "area", 0.0))
    return overlap_area / float(poly.area) >= CONTAINED_ROOM_COVER_RATIO


def _cut_nested_room_from_parent(
    parent_poly: Polygon, nested_poly: Polygon
) -> Polygon | None:
    if parent_poly.is_empty or nested_poly.is_empty:
        return None
    cut_start = nested_poly.representative_point()
    cut_end = parent_poly.exterior.interpolate(parent_poly.exterior.project(cut_start))
    slot = LineString([cut_start, cut_end]).buffer(
        NESTED_ROOM_CUT_WIDTH_M,
        cap_style="flat",
        join_style="mitre",
    )
    cutter = make_valid(
        unary_union([nested_poly.buffer(NESTED_ROOM_CUT_BUFFER_M, join_style=2), slot])
    )
    try:
        clipped = make_valid(parent_poly.difference(cutter))
    except Exception:
        return None
    parts = [
        part
        for part in _decompose_polys(clipped)
        if part.is_valid and not part.is_empty and part.area > MIN_OVERLAP_M2
    ]
    if not parts:
        return None
    return max(parts, key=lambda part: part.area)


def _subtract_nested_room_from_claimed(
    room_state: list[ExtractedRoom],
    story_claim_order: list[tuple[int, Polygon]],
    nested_poly: Polygon,
) -> list[tuple[int, Polygon]]:
    updated_claims: list[tuple[int, Polygon]] = []
    for winner_room_index, winner_poly in story_claim_order:
        if winner_poly.is_empty or nested_poly.is_empty:
            updated_claims.append((winner_room_index, winner_poly))
            continue
        overlap_area = float(winner_poly.intersection(nested_poly).area)
        if overlap_area / float(nested_poly.area) < CONTAINED_ROOM_COVER_RATIO:
            updated_claims.append((winner_room_index, winner_poly))
            continue
        clipped_winner = _cut_nested_room_from_parent(winner_poly, nested_poly)
        if clipped_winner is None:
            updated_claims.append((winner_room_index, winner_poly))
            continue
        winner_room = room_state[winner_room_index]
        floor_y = (
            float(np.mean([corner[1] for corner in winner_room.floor_polygon]))
            if winner_room.floor_polygon
            else 0.0
        )
        room_state[winner_room_index] = replace(
            winner_room,
            floor_polygon=_floor_from_polygon(clipped_winner, floor_y),
        )
        updated_claims.append((winner_room_index, clipped_winner))
    return updated_claims


def clip_floor_overlaps(rooms: list[ExtractedRoom]) -> list[ExtractedRoom]:
    room_state = list(rooms)
    story_entries_raw: dict[int, list[tuple[int, Polygon, float, float]]] = defaultdict(
        list
    )
    for room_index, room in enumerate(room_state):
        poly = _floor_polygon_to_shapely(room.floor_polygon)
        if poly is None:
            continue
        floor_y = float(np.mean([corner[1] for corner in room.floor_polygon]))
        story_entries_raw[room.story].append(
            (room_index, poly, float(poly.area), floor_y)
        )

    story_entries: dict[int, list[tuple[int, Polygon, float, float]]] = defaultdict(
        list
    )
    for story, entries in story_entries_raw.items():
        if len(entries) < 2:
            continue
        median_y = float(np.median([floor_y for _, _, _, floor_y in entries]))
        for room_index, poly, area, floor_y in entries:
            if abs(floor_y - median_y) <= MAX_HALF_FLOOR_M:
                story_entries[story].append((room_index, poly, area, floor_y))

    story_claim_order: dict[int, list[tuple[int, Polygon]]] = defaultdict(list)
    for story, entries in story_entries.items():
        entries.sort(key=lambda item: -item[2])
        claimed = None
        for room_index, poly, _, floor_y in entries:
            if claimed is None:
                claimed = poly
                story_claim_order[story].append((room_index, poly))
                continue

            overlap = poly.intersection(claimed)
            if overlap.area < MIN_OVERLAP_M2:
                claimed = make_valid(unary_union([claimed, poly]))
                story_claim_order[story].append((room_index, poly))
                continue

            clipped = make_valid(poly.difference(claimed))
            parts = _decompose_polys(clipped)
            if len(parts) > 1:
                clipped_poly = max(parts, key=lambda part: part.area)
            elif len(parts) == 1:
                clipped_poly = parts[0]
            else:
                clipped_poly = Polygon()

            room = room_state[room_index]
            if _should_preserve_contained_room(room, poly, overlap, clipped_poly):
                story_claim_order[story] = _subtract_nested_room_from_claimed(
                    room_state,
                    story_claim_order[story],
                    poly,
                )
                claimed_parts = [
                    claim_poly for _, claim_poly in story_claim_order[story]
                ]
                claimed = make_valid(unary_union([*claimed_parts, poly]))
                story_claim_order[story].append((room_index, poly))
                continue

            removed_region = make_valid(poly.difference(clipped_poly))
            overlap_for_test = make_valid(unary_union(_decompose_polys(overlap)))
            candidate_removed_walls: list[ExtractedWall] = []
            kept_walls: list[ExtractedWall] = []
            for wall in room.walls_computed:
                overlap_len, _ = _wall_segment_overlap_with_region(wall, removed_region)
                if overlap_len > 1e-6:
                    candidate_removed_walls.append(wall)
                else:
                    kept_walls.append(wall)

            removed_doors, kept_doors = _elements_in_overlap(
                room.doors, overlap_for_test
            )
            removed_windows, kept_windows = _elements_in_overlap(
                room.windows, overlap_for_test
            )

            winner_walls: list[ExtractedWall] = []
            for winner_room_index, winner_poly in story_claim_order[story]:
                if winner_poly.intersects(removed_region.buffer(OVERLAP_BUFFER_M)):
                    winner_walls.extend(room_state[winner_room_index].walls_computed)

            for wall in candidate_removed_walls:
                if _winner_wall_covers_overlap_segment(
                    wall, winner_walls, removed_region
                ):
                    if not _room_has_slanted_wall_top(room):
                        clipped_wall = _clip_wall_to_region(wall, clipped_poly)
                        if clipped_wall is not None and (
                            _wall_tracks_region_boundary(clipped_wall, clipped_poly)
                            or any(
                                _element_near_wall(element, clipped_wall)
                                for element in [
                                    *kept_doors,
                                    *kept_windows,
                                    *room.openings,
                                ]
                            )
                        ):
                            kept_walls.append(clipped_wall)
                    continue
                kept_walls.append(wall)

            transfer_doors: list[ExtractedElement] = []
            for door in removed_doors:
                if _element_belongs_to_walls(door, kept_walls):
                    kept_doors.append(door)
                else:
                    transfer_doors.append(door)
            transfer_windows: list[ExtractedElement] = []
            for window in removed_windows:
                if _element_belongs_to_walls(window, kept_walls):
                    kept_windows.append(window)
                else:
                    transfer_windows.append(window)

            room_state[room_index] = replace(
                room,
                walls_computed=kept_walls,
                doors=kept_doors,
                windows=kept_windows,
                floor_polygon=_floor_from_polygon(clipped_poly, floor_y)
                if clipped_poly.area > MIN_OVERLAP_M2
                else [],
            )

            if transfer_doors or transfer_windows:
                moved_door_ids: set[str] = set()
                moved_window_ids: set[str] = set()
                for winner_room_index, winner_poly in story_claim_order[story]:
                    if not winner_poly.intersects(overlap_for_test):
                        continue
                    winner_room = room_state[winner_room_index]
                    winner_doors = list(winner_room.doors)
                    winner_windows = list(winner_room.windows)
                    for door in transfer_doors:
                        midpoint = _element_xz_midpoint(door.corners)
                        existing = {item.id for item in winner_doors}
                        if (
                            midpoint is not None
                            and winner_poly.buffer(OVERLAP_BUFFER_M).contains(
                                Point(midpoint)
                            )
                            and door.id not in existing
                            and _element_belongs_to_room_walls(door, winner_room)
                        ):
                            winner_doors.append(door)
                            moved_door_ids.add(door.id)
                    for window in transfer_windows:
                        midpoint = _element_xz_midpoint(window.corners)
                        existing = {item.id for item in winner_windows}
                        if (
                            midpoint is not None
                            and winner_poly.buffer(OVERLAP_BUFFER_M).contains(
                                Point(midpoint)
                            )
                            and window.id not in existing
                            and _element_belongs_to_room_walls(window, winner_room)
                        ):
                            winner_windows.append(window)
                            moved_window_ids.add(window.id)
                    room_state[winner_room_index] = replace(
                        winner_room,
                        doors=winner_doors,
                        windows=winner_windows,
                    )
                unmoved_doors = [
                    door for door in transfer_doors if door.id not in moved_door_ids
                ]
                unmoved_windows = [
                    window
                    for window in transfer_windows
                    if window.id not in moved_window_ids
                ]
                if unmoved_doors or unmoved_windows:
                    current_room = room_state[room_index]
                    room_state[room_index] = replace(
                        current_room,
                        doors=[*current_room.doors, *unmoved_doors],
                        windows=[*current_room.windows, *unmoved_windows],
                    )

            if clipped_poly.area > MIN_OVERLAP_M2:
                claimed = make_valid(unary_union([claimed, clipped_poly]))
            story_claim_order[story].append(
                (
                    room_index,
                    clipped_poly if clipped_poly.area > MIN_OVERLAP_M2 else Polygon(),
                )
            )

    return room_state


def clip_walls_to_story_bounds(
    rooms: list[ExtractedRoom],
    story_y_map: dict[int, float],
) -> tuple[list[ExtractedRoom], dict[str, int]]:
    top_epsilon = 0.05
    bottom_tolerance = 0.30
    max_half_floor = 1.50
    min_story_ratio = 0.75

    sorted_stories = sorted(story_y_map)
    story_wall_heights: dict[int, list[float]] = defaultdict(list)
    for room in rooms:
        for wall in room.walls_computed:
            if len(wall.corners) < 3:
                continue
            ys = [corner[1] for corner in wall.corners]
            height = max(ys) - min(ys)
            if height > 0.1:
                story_wall_heights[room.story].append(height)

    story_median_height = {
        story: float(np.median(heights))
        for story, heights in story_wall_heights.items()
        if heights
    }

    story_bounds = {}
    for idx, story in enumerate(sorted_stories):
        floor_y = story_y_map[story]
        ceiling_y = (
            story_y_map[sorted_stories[idx + 1]]
            if idx + 1 < len(sorted_stories)
            else None
        )
        if ceiling_y is not None and story in story_median_height:
            gap = ceiling_y - floor_y
            median_height = story_median_height[story]
            if gap < min_story_ratio * median_height:
                ceiling_y = None
                for next_idx in range(idx + 2, len(sorted_stories)):
                    candidate = story_y_map[sorted_stories[next_idx]]
                    if candidate - floor_y >= min_story_ratio * median_height:
                        ceiling_y = candidate
                        break
        story_bounds[story] = (floor_y, ceiling_y)

    walls_clipped = 0
    walls_checked = 0
    out: list[ExtractedRoom] = []
    for room in rooms:
        bounds = story_bounds.get(room.story)
        if bounds is None:
            out.append(room)
            continue

        room_floor_y = None
        if len(room.floor_polygon) >= 3:
            room_floor_y = float(np.mean([corner[1] for corner in room.floor_polygon]))
            if abs(room_floor_y - story_y_map[room.story]) > max_half_floor:
                out.append(room)
                continue

        floor_y, ceiling_y = bounds
        effective_floor_y = floor_y
        if room_floor_y is not None:
            wall_bottoms = [
                min(corner[1] for corner in wall.corners)
                for wall in room.walls_computed
                if len(wall.corners) >= 3
            ]
            if wall_bottoms and abs(room_floor_y - min(wall_bottoms)) <= 0.10:
                effective_floor_y = room_floor_y

        opening_top_y = _max_opening_top_y(room)
        if (
            ceiling_y is not None
            and opening_top_y is not None
            and ceiling_y < opening_top_y + top_epsilon
        ):
            ceiling_y = None
            for candidate_story in sorted_stories:
                candidate_y = story_y_map[candidate_story]
                if (
                    candidate_y > opening_top_y + top_epsilon
                    and candidate_y > effective_floor_y
                ):
                    ceiling_y = candidate_y
                    break

        clipped_walls: list[ExtractedWall] = []
        for wall in room.walls_computed:
            walls_checked += 1
            corners = wall.corners
            if len(corners) < 3:
                clipped_walls.append(wall)
                continue
            ys = [corner[1] for corner in corners]
            wall_min_y = min(ys)
            wall_max_y = max(ys)
            need_clip = (
                ceiling_y is not None and wall_max_y > ceiling_y + top_epsilon
            ) or wall_min_y < effective_floor_y - bottom_tolerance
            if not need_clip:
                clipped_walls.append(wall)
                continue

            new_corners = [list(corner) for corner in corners]
            if ceiling_y is not None:
                for corner in new_corners:
                    if corner[1] > ceiling_y:
                        corner[1] = ceiling_y
            if wall_min_y < effective_floor_y - bottom_tolerance:
                for corner in new_corners:
                    if corner[1] < effective_floor_y:
                        corner[1] = effective_floor_y
            clipped_walls.append(replace(wall, corners=new_corners))
            walls_clipped += 1

        out.append(replace(room, walls_computed=clipped_walls))

    return out, {"walls_clipped": walls_clipped, "walls_checked": walls_checked}
