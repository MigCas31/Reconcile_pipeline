from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import atan2, cos, degrees, hypot, radians, sin, sqrt

import numpy as np


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def bidirectional_angle_diff_deg(a: float, b: float) -> float:
    return min(angle_diff_deg(a, b), angle_diff_deg(a, b + 180.0))


def bidirectional_mean_deg(angles: Iterable[float]) -> float:
    vals = list(angles)
    if not vals:
        return 0.0
    sin_sum = sum(sin(radians(a * 2.0)) for a in vals)
    cos_sum = sum(cos(radians(a * 2.0)) for a in vals)
    return (degrees(atan2(sin_sum, cos_sum)) / 2.0) % 180.0


def circular_mean_deg(angles: Iterable[float]) -> float:
    vals = list(angles)
    if not vals:
        return 0.0
    sin_sum = sum(sin(radians(a)) for a in vals)
    cos_sum = sum(cos(radians(a)) for a in vals)
    return degrees(atan2(sin_sum, cos_sum)) % 360.0


def plane_normal_from_azimuth_inclination(
    azimuth_deg: float, incl_deg: float
) -> tuple[float, float, float]:
    azimuth = radians(azimuth_deg)
    incl = radians(incl_deg)
    sx, sy, sz = sin(azimuth) * cos(incl), -sin(incl), cos(azimuth) * cos(incl)
    rx, rz = -cos(azimuth), sin(azimuth)
    return sy * rz, sz * rx - sx * rz, -sy * rx


def segment_midpoint(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        (float(a[0]) + float(b[0])) / 2.0,
        (float(a[1]) + float(b[1])) / 2.0,
        (float(a[2]) + float(b[2])) / 2.0,
    ]


def xz_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def edge_inclination_deg(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    dz = float(b[2]) - float(a[2])
    return degrees(atan2(abs(dy), hypot(dx, dz)))


def edge_azimuth_deg(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(b[0]) - float(a[0])
    dz = float(b[2]) - float(a[2])
    return degrees(atan2(dx, dz)) % 360.0


def polygon_xz_area(corners: Sequence[Sequence[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    total = 0.0
    for idx, a in enumerate(corners):
        b = corners[(idx + 1) % len(corners)]
        total += float(a[0]) * float(b[2]) - float(b[0]) * float(a[2])
    return abs(total) / 2.0


def xz_polygon_to_vec3(
    poly: Sequence[tuple[float, float]], y: float
) -> list[list[float]]:
    return [[float(x), float(y), float(z)] for x, z in poly]


def wall_quads(wall) -> list[list[list[float]]]:
    corners = getattr(wall, "corners", None)
    if corners is None and isinstance(wall, dict):
        corners = wall.get("corners")
    quads: list[list[list[float]]] = []
    if corners and len(corners) >= 3:
        quads.append([[float(c[0]), float(c[1]), float(c[2])] for c in corners])

    for attr in ("descent_strip", "uplift_strip"):
        strip = getattr(wall, attr, None)
        if strip is None and isinstance(wall, dict):
            strip = wall.get(attr)
        if not strip:
            continue
        is_nested = bool(
            strip
            and isinstance(strip[0], (list, tuple))
            and strip[0]
            and isinstance(strip[0][0], (list, tuple))
        )
        strips = strip if is_nested else [strip]
        for quad in strips:
            if quad and len(quad) >= 3:
                quads.append([[float(c[0]), float(c[1]), float(c[2])] for c in quad])
    return quads


def wall_normal_xz(corners: Sequence[Sequence[float]]) -> tuple[float, float]:
    pts = sorted(corners, key=lambda c: float(c[1]))
    if len(pts) < 2:
        return 0.0, 0.0
    b0, b1 = pts[0], pts[1]
    dx = float(b1[0]) - float(b0[0])
    dz = float(b1[2]) - float(b0[2])
    length = hypot(dx, dz)
    if length < 1e-9:
        return 0.0, 0.0
    return -dz / length, dx / length


def as_np_point(point: Sequence[float]) -> np.ndarray:
    return np.asarray([float(point[0]), float(point[1]), float(point[2])], dtype=float)
