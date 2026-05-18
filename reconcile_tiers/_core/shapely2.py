from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from numbers import Integral

from shapely import coverage_union_all
from shapely import make_valid as _make_valid
from shapely.strtree import STRtree


def make_valid(geometry):
    return _make_valid(geometry)


def make_valid_polygon(geometry):
    repaired = make_valid(geometry)
    if repaired.is_empty:
        return None
    if repaired.geom_type == "Polygon":
        return repaired
    polygon_parts = [
        part
        for part in getattr(repaired, "geoms", [])
        if part.geom_type == "Polygon" and part.area > 0.0
    ]
    if not polygon_parts:
        return None
    return max(polygon_parts, key=lambda part: part.area)


def coverage_union(geometries):
    return coverage_union_all(list(geometries))


def query_intersecting(geometries: Sequence, query_geometry) -> list:
    tree = STRtree(geometries)
    hits = tree.query(query_geometry)
    if len(hits) == 0:
        return []
    first = hits[0]
    if isinstance(first, Integral):
        return [
            geometries[int(idx)]
            for idx in hits
            if geometries[int(idx)].intersects(query_geometry)
        ]
    return [geom for geom in hits if geom.intersects(query_geometry)]


def _bounds_side_lengths(geometry) -> list[float]:
    try:
        minx, miny, maxx, maxy = geometry.bounds
    except Exception:
        return []
    sides = [abs(float(maxx) - float(minx)), abs(float(maxy) - float(miny))]
    return [side for side in sides if math.isfinite(side)]


def oriented_rectangle_side_lengths(geometry) -> list[float]:
    """Return side lengths for Shapely's oriented envelope without noisy warnings."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*oriented_envelope",
            )
            rect = geometry.minimum_rotated_rectangle
    except Exception:
        return _bounds_side_lengths(geometry)

    exterior = getattr(rect, "exterior", None)
    if exterior is None:
        return _bounds_side_lengths(geometry)
    coords = list(exterior.coords)
    if len(coords) < 5:
        return _bounds_side_lengths(geometry)

    sides = []
    for idx in range(4):
        x0, y0 = coords[idx]
        x1, y1 = coords[idx + 1]
        side = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
        if not math.isfinite(side):
            return _bounds_side_lengths(geometry)
        sides.append(side)
    return sides


def split_polygon_at_y(
    poly: list[list[float]], bound_y: float
) -> tuple[list[list[float]], list[list[float]]]:
    below: list[list[float]] = []
    above: list[list[float]] = []
    n = len(poly)
    for idx in range(n):
        cur = poly[idx]
        nxt = poly[(idx + 1) % n]
        cur_below = cur[1] <= bound_y
        nxt_below = nxt[1] <= bound_y
        if cur_below:
            below.append(list(cur))
        else:
            above.append(list(cur))
        if cur_below != nxt_below:
            denom = nxt[1] - cur[1]
            if abs(denom) > 1e-9:
                t = (bound_y - cur[1]) / denom
                cross = [
                    cur[0] + t * (nxt[0] - cur[0]),
                    bound_y,
                    cur[2] + t * (nxt[2] - cur[2]),
                ]
                below.append(cross)
                above.append(list(cross))
    return below, above
