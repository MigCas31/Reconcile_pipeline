"""Floor-overlap and story-bound wall clipping helpers."""

import math
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile_v2.decision_logic import classify_overlap_resolution

from .lineage import STEP_CLIP_FLOOR_OVERLAPS, STEP_CLIP_WALLS_STORY, record


def floor_polygon_to_shapely(floor_polygon_3d):
    if not floor_polygon_3d or len(floor_polygon_3d) < 3:
        return None
    coords = [(corner[0], corner[2]) for corner in floor_polygon_3d]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.area < 0.01:
            return None
        return poly
    except Exception:
        return None


def decompose_polys(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def element_xz_midpoint(corners):
    if not corners or len(corners) < 2:
        return None
    x_val = np.mean([corner[0] for corner in corners])
    z_val = np.mean([corner[2] for corner in corners])
    return (x_val, z_val)


def _elements_in_overlap(elements, overlap_poly, buffer=0.15):
    buffered = overlap_poly.buffer(buffer)
    inside = []
    outside = []
    for element in elements:
        midpoint = element_xz_midpoint(element.get("corners", []))
        if midpoint and buffered.contains(Point(midpoint)):
            inside.append(element)
        else:
            outside.append(element)
    return inside, outside


def _wall_xz_normal(corners):
    if len(corners) < 2:
        return None
    dx = corners[1][0] - corners[0][0]
    dz = corners[1][2] - corners[0][2]
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return None
    return (-dz / length, dx / length)


def _wall_base_segment_xz(corners):
    if len(corners) < 2:
        return None
    start = (float(corners[0][0]), float(corners[0][2]))
    end = (float(corners[1][0]), float(corners[1][2]))
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
        return None
    return LineString([start, end])


def _wall_segment_overlap_with_region(wall, region_poly, buffer=0.15):
    if region_poly is None or region_poly.is_empty:
        return 0.0, None
    segment = _wall_base_segment_xz(wall.get("corners", []))
    if segment is None:
        return 0.0, None
    clipped = segment.intersection(region_poly.buffer(buffer))
    if clipped.is_empty:
        return 0.0, None
    return float(clipped.length), clipped


def _project_line_interval(line, origin, direction):
    coords = []
    if getattr(line, "geom_type", "") == "LineString":
        coords = list(line.coords)
    else:
        for geom in getattr(line, "geoms", []):
            if getattr(geom, "geom_type", "") == "LineString":
                coords.extend(list(geom.coords))
    if not coords:
        raise ValueError("line geometry has no coordinate sequence")
    ts = [
        (coord[0] - origin[0]) * direction[0] + (coord[1] - origin[1]) * direction[1]
        for coord in coords
    ]
    return min(ts), max(ts)


def _winner_wall_covers_overlap_segment(
    wall,
    winner_walls,
    overlap_poly,
    *,
    buffer=0.15,
    max_offset_m=0.2,
    max_angle_deg=12.0,
    min_overlap_fraction=0.35,
):
    wall_segment = _wall_base_segment_xz(wall.get("corners", []))
    if wall_segment is None:
        return True, {"reason": "invalid_loser_segment"}

    overlap_len, overlap_segment = _wall_segment_overlap_with_region(
        wall, overlap_poly, buffer=buffer
    )
    if overlap_segment is None or overlap_len <= 1e-6:
        return False, {
            "reason": "no_overlap_segment",
            "overlap_length_m": round(overlap_len, 6),
        }

    direction = np.array(wall_segment.coords[1]) - np.array(wall_segment.coords[0])
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= 1e-6:
        return True, {"reason": "degenerate_loser_segment"}
    unit_dir = direction / segment_length
    origin = np.array(wall_segment.coords[0], dtype=float)
    target_start, target_end = _project_line_interval(overlap_segment, origin, unit_dir)
    target_length = max(target_end - target_start, 0.0)
    if target_length <= 1e-6:
        return False, {
            "reason": "zero_target_interval",
            "overlap_length_m": round(overlap_len, 6),
        }

    best = {
        "reason": "no_winner_match",
        "overlap_length_m": round(overlap_len, 6),
        "best_fraction": 0.0,
        "best_offset_m": None,
        "best_angle_deg": None,
    }
    loser_normal = _wall_xz_normal(wall.get("corners", []))

    for winner_wall in winner_walls:
        winner_segment = _wall_base_segment_xz(winner_wall.get("corners", []))
        if winner_segment is None:
            continue
        winner_normal = _wall_xz_normal(winner_wall.get("corners", []))
        if loser_normal is None or winner_normal is None:
            continue
        dot = abs(
            loser_normal[0] * winner_normal[0] + loser_normal[1] * winner_normal[1]
        )
        angle_deg = math.degrees(math.acos(min(dot, 1.0)))
        if angle_deg > max_angle_deg:
            continue
        offset_m = float(winner_segment.distance(overlap_segment))
        if offset_m > max_offset_m:
            continue

        winner_start, winner_end = _project_line_interval(
            winner_segment, origin, unit_dir
        )
        projected_overlap = max(
            0.0, min(target_end, winner_end) - max(target_start, winner_start)
        )
        overlap_fraction = (
            projected_overlap / target_length if target_length > 1e-6 else 0.0
        )
        if overlap_fraction > best["best_fraction"]:
            best = {
                "reason": "winner_match",
                "overlap_length_m": round(overlap_len, 6),
                "best_fraction": round(overlap_fraction, 6),
                "best_offset_m": round(offset_m, 6),
                "best_angle_deg": round(angle_deg, 6),
            }
        if overlap_fraction >= min_overlap_fraction:
            return True, best

    return False, best


def _graph_overlap_priorities(rooms_out, graph) -> dict[int, tuple[int, float]]:
    priorities: dict[int, tuple[int, float]] = {}
    for idx, room in enumerate(rooms_out):
        poly = floor_polygon_to_shapely(room.get("floor_polygon", []))
        if poly is None:
            continue
        best_room = None
        best_area = 0.0
        for node in graph.nodes_by_type("Room"):
            bbox = node.bbox_xz
            if (
                node.story != room.get("story")
                or not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                continue
            box_poly = Polygon(
                [
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3]),
                ]
            )
            area = poly.intersection(box_poly).area
            if area > best_area:
                best_area = area
                best_room = node
        if best_room is None:
            continue
        decision = classify_overlap_resolution(graph, best_room)
        priority = 0 if decision["ownership_priority"] == "preserve" else 1
        priorities[idx] = (priority, -best_area)
    return priorities


def clip_floor_overlaps(rooms_out, graph=None):
    """Clip overlapping floors per story; transfer openings to winner rooms."""
    min_overlap = 0.01
    max_half_floor = 1.50

    story_entries_raw = defaultdict(list)
    for room_index, room in enumerate(rooms_out):
        poly = floor_polygon_to_shapely(room["floor_polygon"])
        if poly and poly.area > 0.01:
            floor_y = float(np.mean([corner[1] for corner in room["floor_polygon"]]))
            story_entries_raw[room["story"]].append(
                (
                    room_index,
                    poly,
                    poly.area,
                    floor_y,
                )
            )

    story_entries = defaultdict(list)
    for story, entries in story_entries_raw.items():
        if len(entries) < 2:
            continue
        median_y = float(np.median([floor_y for _, _, _, floor_y in entries]))
        for room_index, poly, area, floor_y in entries:
            if abs(floor_y - median_y) <= max_half_floor:
                story_entries[story].append((room_index, poly, area, floor_y))

    metrics = []
    story_claim_order = defaultdict(list)

    for story, entries in story_entries.items():
        priorities = (
            _graph_overlap_priorities(rooms_out, graph) if graph is not None else {}
        )
        entries.sort(
            key=lambda item: (
                priorities.get(item[0], (1, 0.0))[0],
                priorities.get(item[0], (1, 0.0))[1],
                -item[2],
            )
        )
        claimed = None

        for room_index, poly, orig_area, floor_y in entries:
            if claimed is None:
                claimed = poly
                story_claim_order[story].append((room_index, poly))
                continue

            overlap = poly.intersection(claimed)
            if overlap.area < min_overlap:
                claimed = make_valid(unary_union([claimed, poly]))
                story_claim_order[story].append((room_index, poly))
                continue

            clipped = make_valid(poly.difference(claimed))
            parts = decompose_polys(clipped)
            if len(parts) > 1:
                clipped = max(parts, key=lambda p: p.area)
            elif len(parts) == 1:
                clipped = parts[0]
            else:
                clipped = Polygon()

            overlap_for_test = make_valid(unary_union(decompose_polys(overlap)))
            removed_region = make_valid(poly.difference(clipped))
            candidate_removed_walls = []
            kept_walls = []
            wall_decisions = []
            for wall in rooms_out[room_index]["walls_computed"]:
                overlap_len, _ = _wall_segment_overlap_with_region(
                    wall, removed_region, buffer=0.15
                )
                if overlap_len > 1e-6:
                    candidate_removed_walls.append(wall)
                    wall_decisions.append(
                        {
                            "wall_id": wall.get("id"),
                            "candidate_overlap_length_m": round(overlap_len, 6),
                            "decision": "candidate",
                        }
                    )
                else:
                    kept_walls.append(wall)
            removed_doors, kept_doors = _elements_in_overlap(
                rooms_out[room_index].get("doors", []), overlap_for_test
            )
            removed_windows, kept_windows = _elements_in_overlap(
                rooms_out[room_index].get("windows", []), overlap_for_test
            )

            winner_walls = []
            for winner_room_index, winner_poly in story_claim_order[story]:
                if winner_poly.intersects(removed_region.buffer(0.15)):
                    winner_walls.extend(rooms_out[winner_room_index]["walls_computed"])

            removed_walls = []
            for wall in candidate_removed_walls:
                covered, detail = _winner_wall_covers_overlap_segment(
                    wall, winner_walls, removed_region
                )
                if covered:
                    removed_walls.append(wall)
                    record(
                        wall,
                        STEP_CLIP_FLOOR_OVERLAPS,
                        "removed",
                        (
                            "removed_region_overlap="
                            f"{detail.get('overlap_length_m', 0.0):.3f}m "
                            f"winner_overlap_fraction={
                                detail.get(
                                    'best_fraction',
                                    0.0,
                                ):.3f}"
                        ),
                    )
                    wall_decisions.append(
                        {
                            "wall_id": wall.get("id"),
                            "decision": "removed",
                            **detail,
                        }
                    )
                else:
                    kept_walls.append(wall)
                    wall_decisions.append(
                        {
                            "wall_id": wall.get("id"),
                            "decision": "kept",
                            **detail,
                        }
                    )
            rooms_out[room_index]["walls_computed"] = kept_walls
            rooms_out[room_index]["walls_removed_overlap"] = removed_walls
            rooms_out[room_index]["doors"] = kept_doors
            rooms_out[room_index]["windows"] = kept_windows

            if removed_doors or removed_windows:
                for winner_room_index, winner_poly in story_claim_order[story]:
                    if not winner_poly.intersects(overlap_for_test):
                        continue
                    winner_room = rooms_out[winner_room_index]
                    for door in removed_doors:
                        midpoint = element_xz_midpoint(door.get("corners", []))
                        if midpoint and winner_poly.buffer(0.15).contains(
                            Point(midpoint)
                        ):
                            existing = {
                                item["id"] for item in winner_room.get("doors", [])
                            }
                            if door["id"] not in existing:
                                record(
                                    door,
                                    STEP_CLIP_FLOOR_OVERLAPS,
                                    "transferred",
                                    f"to room {winner_room_index}",
                                )
                                winner_room.setdefault("doors", []).append(door)
                    for window in removed_windows:
                        midpoint = element_xz_midpoint(window.get("corners", []))
                        if midpoint and winner_poly.buffer(0.15).contains(
                            Point(midpoint)
                        ):
                            existing = {
                                item["id"] for item in winner_room.get("windows", [])
                            }
                            if window["id"] not in existing:
                                record(
                                    window,
                                    STEP_CLIP_FLOOR_OVERLAPS,
                                    "transferred",
                                    f"to room {winner_room_index}",
                                )
                                winner_room.setdefault("windows", []).append(window)

            metrics.append(
                {
                    "room_index": room_index,
                    "story": story,
                    "original_area_m2": round(orig_area, 3),
                    "clipped_area_m2": round(clipped.area, 3),
                    "overlap_area_m2": round(overlap.area, 3),
                    "walls_removed": len(removed_walls),
                    "wall_decisions": wall_decisions,
                    "doors_transferred": len(removed_doors),
                    "windows_transferred": len(removed_windows),
                }
            )

            rooms_out[room_index]["floor_polygon_original"] = list(
                rooms_out[room_index]["floor_polygon"]
            )
            if clipped.area > min_overlap:
                coords_2d = list(clipped.exterior.coords)[:-1]
                rooms_out[room_index]["floor_polygon"] = [
                    [coord[0], floor_y, coord[1]] for coord in coords_2d
                ]
                rooms_out[room_index]["floor_clipped"] = True

                overlap_parts = decompose_polys(overlap)
                if overlap_parts:
                    biggest = max(overlap_parts, key=lambda p: p.area)
                    coords_2d = list(biggest.exterior.coords)[:-1]
                    rooms_out[room_index]["floor_overlap_region"] = [
                        [coord[0], floor_y, coord[1]] for coord in coords_2d
                    ]
            else:
                rooms_out[room_index]["floor_polygon"] = []
                rooms_out[room_index]["floor_clipped"] = True

            if clipped.area > min_overlap:
                claimed = make_valid(unary_union([claimed, clipped]))
            story_claim_order[story].append(
                (
                    room_index,
                    clipped if clipped.area > min_overlap else Polygon(),
                )
            )

    return metrics


def clip_walls_to_story_bounds(rooms_out, story_y_map):
    top_epsilon = 0.05
    bottom_tol = 0.30
    max_half_floor = 1.50
    min_story_ratio = 0.75

    sorted_stories = sorted(story_y_map.keys())
    story_wall_heights = defaultdict(list)
    for room in rooms_out:
        for wall in room.get("walls_computed", []):
            corners = wall.get("corners", [])
            if len(corners) >= 3:
                ys = [corner[1] for corner in corners]
                height = max(ys) - min(ys)
                if height > 0.1:
                    story_wall_heights[room["story"]].append(height)

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
            median_h = story_median_height[story]
            if gap < min_story_ratio * median_h:
                ceiling_y = None
                for next_idx in range(idx + 2, len(sorted_stories)):
                    candidate = story_y_map[sorted_stories[next_idx]]
                    if (candidate - floor_y) >= min_story_ratio * median_h:
                        ceiling_y = candidate
                        break

        story_bounds[story] = (floor_y, ceiling_y)

    walls_clipped = 0
    walls_checked = 0

    for room in rooms_out:
        story = room["story"]
        if story not in story_bounds:
            continue

        floor_polygon = room["floor_polygon"]
        room_floor_y = None
        if floor_polygon and len(floor_polygon) >= 3:
            room_floor_y = float(np.mean([corner[1] for corner in floor_polygon]))
            if abs(room_floor_y - story_y_map[story]) > max_half_floor:
                continue

        floor_y, ceiling_y = story_bounds[story]

        # Split-level opt-out: if the room's own floor polygon agrees with its
        # walls' minimum y, this room is an extension at its own elevation (e.g.
        # sunroom or garage step-down). Trust the room over the story aggregate
        # for the bottom-clip baseline. Ceiling baseline stays on the story.
        effective_floor_y = floor_y
        if room_floor_y is not None:
            wall_bottoms = [
                min(c[1] for c in w["corners"])
                for w in room["walls_computed"]
                if len(w.get("corners") or []) >= 3
            ]
            if wall_bottoms and abs(room_floor_y - min(wall_bottoms)) <= 0.10:
                effective_floor_y = room_floor_y

        for wall in room["walls_computed"]:
            walls_checked += 1
            corners = wall["corners"]
            if len(corners) < 3:
                continue

            ys = [corner[1] for corner in corners]
            wall_min_y = min(ys)
            wall_max_y = max(ys)
            need_clip = (
                ceiling_y is not None and wall_max_y > ceiling_y + top_epsilon
            ) or (wall_min_y < effective_floor_y - bottom_tol)
            if not need_clip:
                continue

            new_corners = [list(corner) for corner in corners]
            if ceiling_y is not None:
                for corner in new_corners:
                    if corner[1] > ceiling_y:
                        corner[1] = ceiling_y

            if wall_min_y < effective_floor_y - bottom_tol:
                for corner in new_corners:
                    if corner[1] < effective_floor_y:
                        corner[1] = effective_floor_y

            wall["corners_original"] = corners
            wall["corners"] = new_corners
            wall["wall_clipped"] = True
            record(
                wall,
                STEP_CLIP_WALLS_STORY,
                "modified",
                f"y: [{wall_min_y:.2f},{wall_max_y:.2f}]"
                f" -> [{effective_floor_y:.2f},{ceiling_y}]",
            )
            walls_clipped += 1

    return {"walls_clipped": walls_clipped, "walls_checked": walls_checked}
