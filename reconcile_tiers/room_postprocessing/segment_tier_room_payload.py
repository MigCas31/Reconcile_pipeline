"""Classify tier_payload elements into segment-room cycles."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from shapely.geometry import LineString, Polygon

from reconcile_tiers.room_postprocessing.corner_graph import (
    DEFAULT_ADJACENCY_TOL_M,
)
from reconcile_tiers.room_postprocessing.export import build_corner_graph
from reconcile_tiers.room_postprocessing.segment_group_representative import (
    base_wall_id,
)

SEGMENT_TIER_SHELL = "segment_tier_classification"
SEGMENT_TIER_GEOMETRY_SOURCE = "perimeter_walls"

_DEFAULT_BOUNDARY_TOL_M = 0.12
_DEFAULT_FLOOR_OVERLAP_MIN = 0.20
_DEFAULT_FLOOR_OVERLAP_MIN_AREA_M2 = 0.05


def _corners_3d(
    raw: Sequence[Mapping[str, float] | Sequence[float]],
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for c in raw:
        try:
            if isinstance(c, Mapping):
                out.append((float(c["x"]), float(c["y"]), float(c["z"])))
            else:
                out.append((float(c[0]), float(c[1]), float(c[2])))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def _cycle_polygon(seg_room: Mapping[str, Any]) -> Polygon | None:
    poly_xz = seg_room.get("polygon_xz") or []
    if len(poly_xz) < 3:
        return None
    try:
        poly = Polygon([(float(p["x"]), float(p["z"])) for p in poly_xz])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not isinstance(poly, Polygon):
            return None
        return poly
    except Exception:
        return None


def _wall_bottom_rim_xz(
    wall: Mapping[str, Any],
    *,
    y_tol: float = 0.02,
) -> LineString | None:
    corners = _corners_3d(wall.get("corners") or [])
    if len(corners) < 3:
        return None
    y_min = min(c[1] for c in corners)
    bottom = [(c[0], c[2]) for c in corners if c[1] <= y_min + y_tol]
    if len(bottom) < 2:
        ordered = sorted(corners, key=lambda c: c[1])
        bottom = [(ordered[0][0], ordered[0][2]), (ordered[1][0], ordered[1][2])]
    # Deduplicate nearly identical XZ points
    deduped: list[tuple[float, float]] = []
    for pt in bottom:
        if not deduped or (pt[0] - deduped[-1][0]) ** 2 + (pt[1] - deduped[-1][1]) ** 2 > 1e-8:
            deduped.append(pt)
    if len(deduped) < 2:
        return None
    return LineString(deduped)


def _locator_matches_wall_hints(locator_id: str, wall_hints: set[str]) -> bool:
    if not wall_hints:
        return False
    base = base_wall_id(locator_id)
    if base in wall_hints or locator_id in wall_hints:
        return True
    for hint in wall_hints:
        if _wall_matches_hint(locator_id, hint):
            return True
    return False


def _wall_matches_hint(locator_id: str, hint: str) -> bool:
    hint_base = base_wall_id(hint)
    loc_base = base_wall_id(locator_id)
    return loc_base == hint_base


def wall_on_cycle_boundary(
    wall: Mapping[str, Any],
    cycle_poly: Polygon,
    *,
    boundary_tol: float = _DEFAULT_BOUNDARY_TOL_M,
    min_rim_length_m: float = 0.10,
) -> bool:
    """True when the wall bottom rim lies along the cycle perimeter in XZ."""

    rim = _wall_bottom_rim_xz(wall)
    if rim is None or rim.length < min_rim_length_m:
        return False
    boundary = cycle_poly.boundary
    if rim.distance(boundary) > boundary_tol:
        return False
    return True


def floor_overlaps_cycle(
    floor: Mapping[str, Any],
    cycle_poly: Polygon,
    *,
    min_overlap_ratio: float = _DEFAULT_FLOOR_OVERLAP_MIN,
    min_overlap_area_m2: float = _DEFAULT_FLOOR_OVERLAP_MIN_AREA_M2,
) -> bool:
    corners = _corners_3d(floor.get("corners") or [])
    if len(corners) < 3:
        return False
    try:
        floor_poly = Polygon([(c[0], c[2]) for c in corners])
        if not floor_poly.is_valid:
            floor_poly = floor_poly.buffer(0)
        if floor_poly.is_empty:
            return False
        inter = cycle_poly.intersection(floor_poly)
        if inter.is_empty:
            return False
        inter_area = float(inter.area)
        if inter_area < min_overlap_area_m2:
            return False
        floor_area = float(floor_poly.area)
        cycle_area = float(cycle_poly.area)
        if floor_area > 1e-9 and inter_area / floor_area >= min_overlap_ratio:
            return True
        if cycle_area > 1e-9 and inter_area / cycle_area >= min_overlap_ratio:
            return True
        return False
    except Exception:
        return False


def assign_tier_elements_to_cycle(
    seg_room: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    boundary_tol: float = _DEFAULT_BOUNDARY_TOL_M,
    floor_overlap_min: float = _DEFAULT_FLOOR_OVERLAP_MIN,
) -> dict[str, Any] | None:
    """Pick perimeter wall quads + verbatim tier floors/doors/windows for one cycle."""

    cycle_poly = _cycle_polygon(seg_room)
    if cycle_poly is None:
        return None

    story_raw = seg_room.get("story")
    story = int(story_raw) if story_raw is not None else 0

    walls: list[dict[str, Any]] = []
    for quad in seg_room.get("perimeter_wall_quads") or []:
        if not isinstance(quad, Mapping):
            continue
        corners = quad.get("corners")
        if not isinstance(corners, list) or len(corners) < 4:
            continue
        walls.append(copy.deepcopy(dict(quad)))

    floors: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    seen_floors: set[str] = set()
    contributing_tier_rooms: list[int] = []

    for room_index, tier_room in enumerate(payload.get("rooms") or []):
        if not isinstance(tier_room, Mapping):
            continue
        tier_story = tier_room.get("story")
        if tier_story is not None and int(tier_story) != story:
            continue

        room_contributed = False

        floor_raw = tier_room.get("floor")
        floor_pieces: list[Mapping[str, Any]] = []
        if isinstance(floor_raw, Mapping):
            floor_pieces = [floor_raw]
        elif isinstance(floor_raw, list):
            floor_pieces = [f for f in floor_raw if isinstance(f, Mapping)]

        for floor in floor_pieces:
            locator = str(floor.get("locator_id") or "")
            if locator and locator in seen_floors:
                continue
            if not floor_overlaps_cycle(
                floor,
                cycle_poly,
                min_overlap_ratio=floor_overlap_min,
            ):
                continue
            floors.append(copy.deepcopy(dict(floor)))
            if locator:
                seen_floors.add(locator)
            room_contributed = True

        if room_contributed:
            contributing_tier_rooms.append(room_index)
            doors.extend(
                copy.deepcopy(dict(d))
                for d in (tier_room.get("doors") or [])
                if isinstance(d, Mapping)
            )
            windows.extend(
                copy.deepcopy(dict(w))
                for w in (tier_room.get("windows") or [])
                if isinstance(w, Mapping)
            )

    if len(walls) < 3 or len(floors) < 1:
        return None

    room_id = str(seg_room.get("id") or f"room_cycle::{story}::unknown")
    return {
        "story": story,
        "locator_id": room_id,
        "segment_room_id": room_id,
        "walls": walls,
        "floor": floors,
        "doors": doors,
        "windows": windows,
        "tier_room_indices": contributing_tier_rooms,
    }


def build_segment_tier_room_payload(
    payload: Mapping[str, Any],
    *,
    corner_tol: float = 0.05,
    adjacency_tol: float = DEFAULT_ADJACENCY_TOL_M,
    boundary_tol: float = _DEFAULT_BOUNDARY_TOL_M,
    floor_overlap_min: float = _DEFAULT_FLOOR_OVERLAP_MIN,
) -> dict[str, Any]:
    """Segment cycles for room list; perimeter walls + tier floors/doors/windows."""

    graph = build_corner_graph(
        payload,
        corner_tol=corner_tol,
        adjacency_tol=adjacency_tol,
    )
    segment_room_graph = graph.get("segment_room_graph") or {}

    tier_rooms: list[dict[str, Any]] = []
    for node in segment_room_graph.get("nodes") or []:
        if node.get("kind") != "segment_room":
            continue
        assigned = assign_tier_elements_to_cycle(
            node,
            payload,
            boundary_tol=boundary_tol,
            floor_overlap_min=floor_overlap_min,
        )
        if assigned is None:
            continue
        assigned["room_index"] = len(tier_rooms)
        tier_rooms.append(assigned)

    out: dict[str, Any] = dict(payload)
    out["rooms"] = tier_rooms
    out["room_postprocessing_source"] = {
        "corner_tol": corner_tol,
        "adjacency_tol": adjacency_tol,
        "segment_room_count": len(tier_rooms),
        "shell": SEGMENT_TIER_SHELL,
        "geometry_source": SEGMENT_TIER_GEOMETRY_SOURCE,
    }
    out["segment_room_graph"] = segment_room_graph
    return out
