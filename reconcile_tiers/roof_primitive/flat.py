"""Flat wing primitive classification."""

from __future__ import annotations

from statistics import median

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof_primitive.types import FlatParams

MIN_WING_COVERAGE_RATIO = 0.70
MAX_FLAT_Y_SPAN_M = 0.15


def classify_flat(
    wing,
    wing_index: int,
    model: BuildingModel,
    *,
    exclude_room_indices: set[int] | None = None,
) -> FlatParams | None:
    """Classify a wing as flat when flat ceiling/floor evidence covers it."""
    excluded = exclude_room_indices or set()
    wing_poly = wing.polygon
    wing_area = max(float(wing_poly.area), 1e-9)
    polys = []
    ys = []
    story_votes: dict[int, int] = {}

    for room in model.rooms:
        if room.index in excluded or len(room.floor_polygon) < 3:
            continue
        room_poly = _xz_polygon(room.floor_polygon)
        if room_poly is None or not wing_poly.intersects(room_poly):
            continue
        if wing_poly.intersection(room_poly).area / max(room_poly.area, 1e-9) < 0.5:
            continue
        corners = room.ceiling_polygon if len(room.ceiling_polygon) >= 3 else []
        if room.ceiling_type != "flat" or not corners:
            continue
        y_values = [float(p[1]) for p in corners]
        if max(y_values) - min(y_values) > MAX_FLAT_Y_SPAN_M:
            continue
        poly = _xz_polygon(corners)
        if poly is None:
            continue
        polys.append(poly)
        ys.extend(y_values)
        story_votes[room.story] = story_votes.get(room.story, 0) + 1

    if not polys or not ys:
        return None
    try:
        coverage = unary_union(polys).intersection(wing_poly).area
    except Exception:
        return None
    if coverage / wing_area < MIN_WING_COVERAGE_RATIO:
        return None
    dominant_story = (
        max(story_votes, key=lambda s: (story_votes[s], s)) if story_votes else 0
    )
    coords = list(wing_poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return FlatParams(
        wing_index=wing_index,
        y=float(median(ys)),
        polygon_xz=tuple((float(x), float(z)) for x, z in coords),
        dominant_story=dominant_story,
    )


def _xz_polygon(corners: list[list[float]]) -> Polygon | None:
    try:
        return make_valid_polygon(
            Polygon([(float(p[0]), float(p[2])) for p in corners])
        )
    except Exception:
        return None
