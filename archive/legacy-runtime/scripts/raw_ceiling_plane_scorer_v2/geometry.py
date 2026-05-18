from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon


def xz_polygon(corners: list[list[float]]) -> Polygon | None:
    points = [
        (float(point[0]), float(point[2])) for point in corners if len(point) >= 3
    ]
    if len(points) < 3:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def iter_polygons(geom: Any) -> list[Polygon]:
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    polygons: list[Polygon] = []
    geoms = getattr(geom, "geoms", None)
    if geoms is None:
        return polygons
    for part in geoms:
        if isinstance(part, Polygon) and not part.is_empty:
            polygons.append(part)
    return polygons


def safe_intersection_area(a: Polygon, b: Polygon) -> float:
    try:
        overlap = a.intersection(b)
    except Exception:
        return 0.0
    if overlap.is_empty:
        return 0.0
    return float(overlap.area)


def safe_overlap_fraction(subject: Polygon, cover: Polygon) -> float:
    area = float(subject.area)
    if area <= 1e-9:
        return 0.0
    return safe_intersection_area(subject, cover) / area


def stored_row_plane_coeffs(
    row: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    raw_coeffs = row.get("target_plane_coeffs")
    if isinstance(raw_coeffs, (list, tuple)) and len(raw_coeffs) >= 4:
        try:
            a, b, c, d = (float(raw_coeffs[idx]) for idx in range(4))
        except (TypeError, ValueError):
            pass
        else:
            if abs(b) > 1e-9:
                return a, b, c, d

    raw_point = row.get("target_plane_point")
    raw_normal = row.get("target_normal")
    if (
        not isinstance(raw_point, (list, tuple))
        or len(raw_point) < 3
        or not isinstance(raw_normal, (list, tuple))
        or len(raw_normal) < 3
    ):
        return None
    try:
        px, py, pz = (float(raw_point[idx]) for idx in range(3))
        a, b, c = (float(raw_normal[idx]) for idx in range(3))
    except (TypeError, ValueError):
        return None
    if abs(b) <= 1e-9:
        return None
    d = -(a * px + b * py + c * pz)
    return a, b, c, d
