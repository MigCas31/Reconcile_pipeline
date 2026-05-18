from __future__ import annotations

from .math_utils import convex_hull_2d

# Buffer distance (metres) applied to each room polygon before union.
# Bridges wall-thickness gaps between adjacent rooms so the union forms
# a single connected outline rather than disjoint islands.
_ROOM_BUFFER_M = 0.3


def _union_room_footprint(floor_polys_2d: list[list[tuple[float, float]]]):
    """Return the exterior ring of the buffered union of 2-D room polygons.

    Uses Shapely ``unary_union`` so the result preserves concavities
    (L-shapes, T-shapes, extensions).  Falls back to ``None`` on failure
    so the caller can retry with the convex-hull path.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return None

    shapely_polys = []
    for pts in floor_polys_2d:
        if len(pts) < 3:
            continue
        try:
            p = Polygon(pts).buffer(_ROOM_BUFFER_M, join_style="mitre")
            if p.is_valid and not p.is_empty:
                shapely_polys.append(p)
        except Exception:
            continue

    if not shapely_polys:
        return None

    merged = unary_union(shapely_polys)
    if merged.is_empty:
        return None

    # For MultiPolygon results pick the largest component.
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)

    # Negative buffer to undo the expansion, keeping the merged shape.
    shrunk = merged.buffer(-_ROOM_BUFFER_M, join_style="mitre")
    if shrunk.is_empty or shrunk.geom_type not in ("Polygon", "MultiPolygon"):
        # If shrinking collapses the shape, keep the buffered version.
        shrunk = merged

    if shrunk.geom_type == "MultiPolygon":
        shrunk = max(shrunk.geoms, key=lambda g: g.area)

    coords = list(shrunk.exterior.coords)
    if len(coords) >= 3:
        # Shapely returns a closed ring; drop the duplicate closing point.
        if coords[-1] == coords[0]:
            coords = coords[:-1]
        return [(x, z) for x, z in coords]
    return None


def build_building_footprint(exposed_rooms: list, _story_floor_polys: dict) -> dict:
    if not exposed_rooms:
        return {"building_footprint": None, "top_story": float("-inf")}

    top_story = max(r["story"] for r in exposed_rooms)
    floor_polys_2d: list[list[tuple[float, float]]] = [
        [(p[0], p[2]) for p in er["fp"]]
        for er in exposed_rooms
        if er.get("fp") and len(er["fp"]) >= 3
    ]

    # --- primary path: concave union via Shapely ---
    footprint = _union_room_footprint(floor_polys_2d)
    if footprint and len(footprint) >= 3:
        return {"building_footprint": footprint, "top_story": top_story}

    # --- fallback: hull of roof-candidate room points only ---
    fp_pts = [
        {"x": p[0], "z": p[2]} for er in exposed_rooms for p in (er.get("fp") or [])
    ]
    hull = convex_hull_2d(fp_pts if len(fp_pts) >= 3 else [])
    footprint = [(p["x"], p["z"]) for p in hull] if len(hull) >= 3 else None

    return {"building_footprint": footprint, "top_story": top_story}
