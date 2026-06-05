"""Clip scan floor polygons to segment-room wall delineation (plan view)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from reconcile_tiers.room_postprocessing.models import BuildingElement


def _floor_element_polygon_xz(el: BuildingElement) -> Polygon | None:
    if len(el.corners) < 3:
        return None
    ring = [(float(c[0]), float(c[2])) for c in el.corners]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = make_valid(poly)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _exterior_ring_xz(geom: Polygon | MultiPolygon) -> list[tuple[float, float]] | None:
    if isinstance(geom, MultiPolygon):
        if geom.is_empty:
            return None
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon) or geom.is_empty:
        return None
    coords = list(geom.exterior.coords)
    if len(coords) < 4:
        return None
    return [(float(x), float(z)) for x, z in coords[:-1]]


def attach_room_floor_polygons(
    segment_room_graph: dict[str, Any],
    elements: Sequence[BuildingElement],
) -> None:
    """Set ``floor_polygon_xz`` on each room as scan floor ∩ wall-delineated room polygon."""

    floors_by_story: dict[int | None, list[Polygon]] = defaultdict(list)
    for el in elements:
        if el.kind != "floor":
            continue
        poly = _floor_element_polygon_xz(el)
        if poly is not None:
            floors_by_story[el.story].append(poly)

    for room in segment_room_graph.get("nodes") or []:
        poly_xz = room.get("polygon_xz")
        if not poly_xz or len(poly_xz) < 3:
            continue
        room_poly = Polygon([(float(p["x"]), float(p["z"])) for p in poly_xz])
        if not room_poly.is_valid:
            room_poly = make_valid(room_poly)
        if not isinstance(room_poly, Polygon) or room_poly.is_empty:
            continue

        story = room.get("story")
        floor_polys = floors_by_story.get(story, [])
        if floor_polys:
            from shapely.ops import unary_union

            union_floor = unary_union(floor_polys)
            clipped = room_poly.intersection(union_floor)
        else:
            clipped = room_poly

        if clipped.is_empty:
            continue
        ring = _exterior_ring_xz(clipped)
        if not ring or len(ring) < 3:
            continue
        room["floor_polygon_xz"] = [{"x": x, "z": z} for x, z in ring]
        room["floor_area_m2"] = round(float(clipped.area), 3)
