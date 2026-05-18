from __future__ import annotations

from shapely.geometry import Polygon

from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import ObliqueSurface, SplitObliqueSurface

NEAR_TOUCH_BUFFER_M = 0.35
RAW_OWNERSHIP_MIN_RATIO = 0.30
MIN_SPLIT_AREA_M2 = 0.05


def _surface_index(surface: ObliqueSurface, fallback_idx: int) -> int:
    """Use the surface's stable source_index when set; fall back to position
    otherwise (legacy callers that don't set source_index)."""
    return surface.source_index if surface.source_index >= 0 else fallback_idx


def split_oblique_surfaces(
    obliques: list[ObliqueSurface], model: BuildingModel | None = None
) -> list[SplitObliqueSurface]:
    if model is not None:
        split = _split_by_room_cells(obliques, model)
        if split:
            return split
    return [
        SplitObliqueSurface(
            corners=list(surface.corners),
            plane=surface.plane,
            arrangement_cell_id=f"cell:{_surface_index(surface, idx)}",
            source_oblique_index=_surface_index(surface, idx),
            dominant_story=surface.dominant_story,
        )
        for idx, surface in enumerate(obliques)
        if len(surface.corners) >= 3
    ]


def _split_by_room_cells(
    obliques: list[ObliqueSurface], model: BuildingModel
) -> list[SplitObliqueSurface]:
    room_cells = []
    for room in model.rooms:
        if len(room.floor_polygon) < 3:
            continue
        poly = Polygon([(p[0], p[2]) for p in room.floor_polygon])
        if not poly.is_valid:
            poly = make_valid_polygon(poly)
        if poly is None:
            continue
        if not poly.is_empty and poly.area > MIN_SPLIT_AREA_M2:
            room_cells.append((room.index, poly))

    out: list[SplitObliqueSurface] = []
    for fallback_idx, surface in enumerate(obliques):
        oblique_idx = _surface_index(surface, fallback_idx)
        roof_poly = Polygon([(p[0], p[2]) for p in surface.corners])
        if not roof_poly.is_valid:
            roof_poly = make_valid_polygon(roof_poly)
        if roof_poly is None:
            continue
        if roof_poly.is_empty:
            continue
        for room_idx, room_poly in room_cells:
            intersection = roof_poly.intersection(room_poly)
            for part_idx, part in enumerate(_polygon_parts(intersection)):
                ring = _lift_ring(part, surface)
                if len(ring) < 3:
                    continue
                out.append(
                    SplitObliqueSurface(
                        corners=ring,
                        plane=surface.plane,
                        arrangement_cell_id=f"cell:{oblique_idx}:room:{room_idx}:{part_idx}",
                        source_oblique_index=oblique_idx,
                        dominant_story=surface.dominant_story,
                    )
                )
    return out


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area >= MIN_SPLIT_AREA_M2 else []
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if part.area >= MIN_SPLIT_AREA_M2]
    if geometry.geom_type == "GeometryCollection":
        return [
            part
            for part in geometry.geoms
            if part.geom_type == "Polygon" and part.area >= MIN_SPLIT_AREA_M2
        ]
    return []


def _lift_ring(polygon: Polygon, surface: ObliqueSurface) -> list[list[float]]:
    coords = list(polygon.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    ring: list[list[float]] = []
    for x, z in coords:
        y = surface.plane.y_at(float(x), float(z))
        if y is None:
            return []
        ring.append([float(x), float(y), float(z)])
    return ring
