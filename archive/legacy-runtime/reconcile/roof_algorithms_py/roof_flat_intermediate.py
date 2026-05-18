from __future__ import annotations

from .graph_support import (
    classify_graph_roof_room,
    graph_allows_roof_candidate,
    graph_requires_geometry_fallback,
)
from .roof_flat_geometry import bbox


def _room_polygon_at_y(fp: list, y: float) -> list:
    """Lift a room's floor polygon to y without AABB-ifying it.

    Earlier we used bbox_surface(fp, y) which replaced the room shape with
    its axis-aligned bounding box + 0.3m pad — for L-shaped / concave rooms
    that leaks far outside the building footprint.
    """
    return [(p[0], y, p[2]) for p in fp]


def _room_ceiling_y(room: dict) -> float | None:
    # Prefer ceiling_polygon max y (matches the actual ceiling plane).
    cp = room.get("ceiling_polygon") or []
    ys = [c[1] for c in cp if isinstance(c, (list, tuple)) and len(c) >= 2]
    if ys:
        return max(ys)
    # Fall back to max wall-corner y (wall-top ≈ ceiling interface).
    wall_ys: list[float] = []
    for key in ("walls_merged", "walls_computed"):
        for wall in room.get(key) or []:
            for c in wall.get("corners") or []:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    wall_ys.append(c[1])
    return max(wall_ys) if wall_ys else None


def collect_intermediate_flat_surfaces(
    *, bldg: dict, has_floor_above, bldg_min_y: float, graph=None
) -> tuple[list, list]:
    flat_surfaces = []
    roof_legend_flat = []

    for room_idx, room in enumerate(bldg.get("rooms", [])):
        fp = room.get("floor_polygon")
        if not fp or len(fp) < 3:
            continue

        fcx = sum(p[0] for p in fp) / len(fp)
        fcz = sum(p[2] for p in fp) / len(fp)
        story = int(room.get("story", 0))
        _graph_room, roof_decision = classify_graph_roof_room(graph, story, fp)
        if roof_decision is not None and not graph_allows_roof_candidate(roof_decision):
            continue
        if graph_requires_geometry_fallback(roof_decision) and has_floor_above(
            fcx, fcz, story
        ):
            continue

        min_x, max_x, min_z, max_z = bbox(fp)
        if max(max_x - min_x, max_z - min_z) < 2.0:
            continue

        # Intermediate flat represents the top of the room's stack (roof or
        # floor-of-story-above interface). Place it at the ceiling plane, not
        # the floor plane — using floor y buries the surface below room height.
        ceiling_y = _room_ceiling_y(room)
        y = ceiling_y if ceiling_y is not None else fp[0][1]
        corners = _room_polygon_at_y(fp, y)
        flat_surfaces.append(
            {
                "kind": "intermediate",
                "story": story,
                "y": y,
                "corners": corners,
                "room_index": room_idx,
                "graph_room_id": _graph_room.id if _graph_room is not None else None,
            }
        )
        roof_legend_flat.append(
            {
                "count": 1,
                "avgIncl": 0.0,
                "avgAzimuth": None,
                "flat": True,
                "height": (y - bldg_min_y) if bldg_min_y != float("inf") else y,
            }
        )

    return flat_surfaces, roof_legend_flat
