from __future__ import annotations

from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import Footprint

ROOM_BUFFER_M = 0.3
FOOTPRINT_SHRINK_M = 0.3


def _largest_polygon(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda g: g.area)
    return None


def build_building_footprint(
    model: BuildingModel, room_indices: set[int] | None = None
) -> Footprint | None:
    polygons = []
    points: list[tuple[float, float]] = []
    top_story = max((room.story for room in model.rooms), default=0)
    for room in model.rooms:
        if room_indices is not None and room.index not in room_indices:
            continue
        if len(room.floor_polygon) < 3:
            continue
        xz = [(float(p[0]), float(p[2])) for p in room.floor_polygon]
        points.extend(xz)
        poly = Polygon(xz)
        if not poly.is_valid:
            poly = make_valid_polygon(poly)
        if poly is None:
            continue
        if poly.is_empty or poly.area <= 0.0:
            continue
        polygons.append(poly.buffer(ROOM_BUFFER_M, join_style="mitre"))
    if not polygons:
        return None

    merged = _largest_polygon(unary_union(polygons))
    if merged is None:
        return None
    shrunk = _largest_polygon(merged.buffer(-FOOTPRINT_SHRINK_M, join_style="mitre"))
    if shrunk is None or shrunk.area <= 0.0:
        hull = MultiPoint(points).convex_hull
        shrunk = _largest_polygon(hull)
    if shrunk is None or shrunk.area <= 0.0:
        return None
    coords = list(shrunk.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return Footprint(
        polygon_xz=[(float(x), float(z)) for x, z in coords],
        top_story=top_story,
        area=float(shrunk.area),
    )
