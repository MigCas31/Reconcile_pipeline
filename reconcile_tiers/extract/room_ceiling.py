from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract.building import ExtractedRoom

CEILING_FLAT_RANGE_M = 0.10


@dataclass(frozen=True, slots=True)
class RoomCeiling:
    plane: Plane | None
    flat_y: float | None

    def y_at(self, x: float, z: float) -> float | None:
        if self.plane is not None:
            return self.plane.y_at(x, z)
        return self.flat_y


def room_ceiling(room: ExtractedRoom) -> RoomCeiling | None:
    polygon = room.ceiling_polygon
    if not polygon or len(polygon) < 3:
        return None
    ys = [float(corner[1]) for corner in polygon]
    if room.ceiling_type == "sloped" and max(ys) - min(ys) > CEILING_FLAT_RANGE_M:
        plane = Plane.fit(polygon)
        if not isinstance(plane, FitFailure):
            return RoomCeiling(plane=plane, flat_y=None)
    return RoomCeiling(plane=None, flat_y=float(np.median(ys)))


def _ceiling_xz_polygon(room: ExtractedRoom) -> Polygon | None:
    """XZ projection of the room's ceiling polygon — the actual extent where
    the ceiling exists. Outside this footprint, the ceiling plane has no
    physical support and any plane evaluation is extrapolation."""
    polygon = room.ceiling_polygon
    if not polygon or len(polygon) < 3:
        return None
    coords = [(float(c[0]), float(c[2])) for c in polygon]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = make_valid(Polygon(coords))
    except Exception:
        return None
    if not isinstance(poly, Polygon) or poly.is_empty or not poly.is_valid:
        return None
    return poly


def build_story_ceiling_lookup(
    rooms: list[ExtractedRoom],
) -> dict[int, list[tuple[Polygon, RoomCeiling]]]:
    by_story: dict[int, list[tuple[Polygon, RoomCeiling]]] = defaultdict(list)
    for room in rooms:
        ceiling = room_ceiling(room)
        if ceiling is None:
            continue
        poly = _ceiling_xz_polygon(room)
        if poly is None:
            continue
        by_story[room.story].append((poly, ceiling))
    return by_story


def ceiling_y_at_xz(
    lookup: dict[int, list[tuple[Polygon, RoomCeiling]]],
    story: int,
    x: float,
    z: float,
) -> float | None:
    entries = lookup.get(story)
    if not entries:
        return None
    pt = Point(float(x), float(z))
    for poly, ceiling in entries:
        try:
            inside = poly.covers(pt)
        except Exception:
            inside = False
        if inside:
            y = ceiling.y_at(float(x), float(z))
            if y is not None:
                return float(y)
    return None
