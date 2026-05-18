from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract.building import BuildingModel, ExtractedRoom
from reconcile_tiers.payload.schema import KneeWallKind
from reconcile_tiers.roof.roof import ObliqueSurface, ThermalSurface

BARRIER_REACH_M = 0.30
SURFACE_SUPPORT_TOLERANCE_M = 0.30
ROOM_MAX_Y_TOLERANCE_M = 0.05
LID_PERCENTILE = 90
LID_RAW_PLANE_DELTA_M = 0.10
LID_APEX_MARGIN_M = 0.30
THERMAL_KINDS = frozenset({KneeWallKind.KNEE})


def attic_lid_y_by_room(
    model: BuildingModel,
    thermal: list[ThermalSurface],
) -> dict[int, float]:
    """Attic-lid height per room — branches on raw flat ceiling planes.

    A "flat raw plane" is a `raw_ceiling_plane` whose Y extent is within
    LID_RAW_PLANE_DELTA_M (= horizontal, with noise tolerance). The room
    branches:

    - If at least one flat raw plane sits more than LID_APEX_MARGIN_M
      below the room's apex (`ceiling_ridge_height`), the room is a real
      attic conversion and the lid is the p90 of those mid-pitch flat
      planes. p90 (rather than p10) so a stray low patch doesn't drag the
      lid down — same noise-tolerance reasoning as the wall version.
    - Otherwise the room is cathedral / flat-ceilinged: the slope (if any)
      runs to the apex, and the lid is the apex itself.

    Rooms with no apex (`ceiling_ridge_height is None`) and no qualifying
    flat plane stay unset.

    `thermal` is currently unused but kept in the signature so callers
    don't need to change if a future refinement needs to consult it.
    """
    _ = thermal

    out: dict[int, float] = {}
    for room in model.rooms:
        apex = room.ceiling_ridge_height
        flat_below_apex: list[float] = []
        for plane in room.raw_ceiling_planes:
            ys = [float(c[1]) for c in plane.corners if len(c) >= 2]
            if len(ys) < 3:
                continue
            if max(ys) - min(ys) > LID_RAW_PLANE_DELTA_M:
                continue
            mean_y = sum(ys) / len(ys)
            if apex is not None and mean_y >= apex - LID_APEX_MARGIN_M:
                continue
            flat_below_apex.append(mean_y)

        if flat_below_apex:
            out[room.index] = float(np.percentile(flat_below_apex, LID_PERCENTILE))
        elif apex is not None:
            out[room.index] = float(apex)
    return out


def _surface_support_geometry(surface: ObliqueSurface):
    return _xz_polygon_geometry(surface.corners)


def _xz_polygon_geometry(corners: list[list[float]]):
    if len(corners) < 3:
        return None
    geom = make_valid(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    if geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        polygons = [geom] if geom.area > 1e-9 else []
    else:
        polygons = [
            part
            for part in getattr(geom, "geoms", [])
            if isinstance(part, Polygon) and part.area > 1e-9
        ]
    if not polygons:
        return None
    return unary_union(polygons)


def _wall_top_supported_by_surface(
    top: list[list[float]], surface: ObliqueSurface
) -> bool:
    support = _surface_support_geometry(surface)
    if support is None:
        return False
    line = LineString([(float(p[0]), float(p[2])) for p in top])
    if line.length <= 1e-9:
        return False
    return bool(support.buffer(SURFACE_SUPPORT_TOLERANCE_M).covers(line))


def _room_max_y(room: ExtractedRoom) -> float | None:
    ys: list[float] = []

    def collect(corners: list[list[float]]) -> None:
        for corner in corners:
            if len(corner) >= 2:
                ys.append(float(corner[1]))

    collect(room.floor_polygon)
    collect(room.ceiling_polygon)
    for raw in room.raw_ceiling_planes:
        collect(raw.corners)
    for item in [
        *room.walls_merged,
        *room.walls_computed,
        *room.doors,
        *room.windows,
        *room.openings,
        *room.storages,
        *room.synthetic_walls,
    ]:
        collect(item.corners)

    return max(ys) if ys else None


def _knee_walls(
    model: BuildingModel, obliques: list[ObliqueSurface]
) -> list[ThermalSurface]:
    out: list[ThermalSurface] = []
    for room in model.rooms:
        room_max_y = _room_max_y(room)
        for wall in room.walls_computed:
            if len(wall.corners) < 3:
                continue
            top = sorted(wall.corners, key=lambda p: p[1], reverse=True)[:2]
            if len(top) != 2:
                continue
            for surface in obliques:
                if surface.dominant_story != room.story:
                    continue
                if not _wall_top_supported_by_surface(top, surface):
                    continue
                lifted = []
                for p in top:
                    y = surface.plane.y_at(p[0], p[2])
                    if y is None:
                        lifted = []
                        break
                    lifted.append([p[0], y, p[2]])
                if len(lifted) != 2:
                    continue
                if (
                    room_max_y is not None
                    and max(p[1] for p in lifted) > room_max_y + ROOM_MAX_Y_TOLERANCE_M
                ):
                    continue
                gaps = [lifted[idx][1] - top[idx][1] for idx in range(2)]
                if min(gaps) <= BARRIER_REACH_M:
                    continue
                out.append(
                    ThermalSurface(
                        corners=[top[0], top[1], lifted[1], lifted[0]],
                        kind=KneeWallKind.KNEE,
                        room_index=room.index,
                        source="wall_top_to_oblique",
                        source_wall_id=wall.id,
                    )
                )
                break
    return out


def build_thermal_surfaces(
    model: BuildingModel, obliques: list[ObliqueSurface]
) -> list[ThermalSurface]:
    return _knee_walls(model, obliques)
