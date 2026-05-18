from __future__ import annotations

import math
from collections.abc import Sequence


def _points(corners: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    return [tuple(float(v) for v in point[:3]) for point in corners]


def newell_normal(corners: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    pts = _points(corners)
    if len(pts) < 3:
        return (0.0, 0.0, 0.0)
    nx = ny = nz = 0.0
    for idx, (x0, y0, z0) in enumerate(pts):
        x1, y1, z1 = pts[(idx + 1) % len(pts)]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return (nx, ny, nz)


def polygon_area_3d(corners: Sequence[Sequence[float]]) -> float:
    nx, ny, nz = newell_normal(corners)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def is_planar(corners: Sequence[Sequence[float]], tol: float) -> bool:
    pts = _points(corners)
    if len(pts) < 3:
        return False
    nx, ny, nz = newell_normal(pts)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-12:
        return False
    px, py, pz = pts[0]
    for x, y, z in pts[1:]:
        dist = abs(nx * (x - px) + ny * (y - py) + nz * (z - pz)) / norm
        if dist > tol:
            return False
    return True


def polygon_xz_signed_area(corners: Sequence[Sequence[float]]) -> float:
    pts = _points(corners)
    if len(pts) < 3:
        return 0.0
    twice_area = 0.0
    for idx, (x0, _, z0) in enumerate(pts):
        x1, _, z1 = pts[(idx + 1) % len(pts)]
        twice_area += x0 * z1 - x1 * z0
    return 0.5 * twice_area
