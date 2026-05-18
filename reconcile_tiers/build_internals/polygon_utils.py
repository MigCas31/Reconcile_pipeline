"""Pure polygon / corner geometry helpers used across `reconcile_tiers.build`.

Extracted from `build.py` so the orchestrator doesn't carry low-level
Shapely glue. All functions are stateless and operate on plain corner lists
or Shapely geometries.

Re-exported from `reconcile_tiers.build` for backwards-compatible imports.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import Plane


def _dedupe_points(corners: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for corner in corners:
        if (
            not out
            or sum((corner[idx] - out[-1][idx]) ** 2 for idx in range(3)) > 1e-10
        ):
            out.append(corner)
    if (
        len(out) >= 2
        and sum((out[0][idx] - out[-1][idx]) ** 2 for idx in range(3)) <= 1e-10
    ):
        out.pop()
    return out


def _polygon_parts_2d(geometry) -> list:
    if geometry is None or geometry.is_empty:
        return []
    from shapely.geometry import Polygon

    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 1e-6 else []
    return [
        part
        for part in getattr(geometry, "geoms", [])
        if isinstance(part, Polygon) and part.area > 1e-6
    ]


def _corners_xz_polygon(corners: list[list[float]]):
    if len(corners) < 3:
        return None
    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon

    poly = make_valid_polygon(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    if poly is None or poly.is_empty or poly.area <= 0.0:
        return None
    return poly


def _room_floor_xz_polygon(room):
    return _corners_xz_polygon(room.floor_polygon)


def _xz_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    from shapely.geometry import Polygon

    poly = Polygon([(float(p[0]), float(p[2])) for p in corners])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return float(poly.area) if not poly.is_empty else 0.0


def _polygon_xz_from_corners(corners: list[list[float]]):
    if len(corners) < 3:
        return None
    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon

    try:
        poly = make_valid_polygon(
            Polygon([(float(p[0]), float(p[2])) for p in corners])
        )
    except Exception:
        return None
    if poly is None or poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _vec3_on_plane_from_polygon(poly, plane: Plane) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    out: list[list[float]] = []
    for x, z in coords:
        y = plane.y_at(float(x), float(z))
        if y is None:
            return []
        out.append([float(x), float(y), float(z)])
    return out
