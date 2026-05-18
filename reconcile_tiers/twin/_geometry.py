"""Geometric predicates used by twin invariants.

Float-precision only. No building parameters, no scan-precision tolerances.
The single tolerance is `FLOAT_EPS = 1e-6`, used as the margin for
"mathematically equal" on real-valued geometric quantities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from reconcile_tiers.payload.schema import Plane, Vec3

FLOAT_EPS: float = 1e-6


def polygon_area_xz(corners: Sequence[Vec3]) -> float:
    """Signed area magnitude of the polygon's XZ projection (shoelace)."""
    n = len(corners)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        total += a.x * b.z - b.x * a.z
    return abs(total) * 0.5


def y_span(corners: Iterable[Vec3]) -> float:
    """Difference between max and min Y of a corner set."""
    ys = [c.y for c in corners]
    if not ys:
        return 0.0
    return max(ys) - min(ys)


def plane_is_horizontal(plane: Plane) -> bool:
    """The plane's normal is parallel to the Y axis (horizontal surface)."""
    return abs(abs(plane.b) - 1.0) < FLOAT_EPS


def plane_is_vertical(plane: Plane) -> bool:
    """The plane's normal is perpendicular to the Y axis (vertical surface)."""
    return abs(plane.b) < FLOAT_EPS


def plane_is_oblique(plane: Plane) -> bool:
    """Strictly oblique: not horizontal and not vertical."""
    return not (plane_is_horizontal(plane) or plane_is_vertical(plane))


def plane_normal_y(plane: Plane) -> float:
    """The Y component of the plane's normalised normal."""
    return float(plane.b)


def points_equal(a: Vec3, b: Vec3) -> bool:
    """True iff two points coincide within float precision."""
    return (
        abs(a.x - b.x) < FLOAT_EPS
        and abs(a.y - b.y) < FLOAT_EPS
        and abs(a.z - b.z) < FLOAT_EPS
    )


def segment_length(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def segment_endpoints_horizontal(a: Vec3, b: Vec3) -> bool:
    """A line segment is horizontal when its endpoints share Y."""
    return abs(a.y - b.y) < FLOAT_EPS


def midpoint_y(a: Vec3, b: Vec3) -> float:
    return 0.5 * (a.y + b.y)


def polygon_is_planar(corners: Sequence[Vec3]) -> bool:
    """All corners lie on one plane within float precision.

    Uses the plane fit through the first three corners; checks the rest.
    Degenerate polygons (collinear first three) return False.
    """
    n = len(corners)
    if n < 3:
        return False
    p0, p1, p2 = corners[0], corners[1], corners[2]
    ux, uy, uz = p1.x - p0.x, p1.y - p0.y, p1.z - p0.z
    vx, vy, vz = p2.x - p0.x, p2.y - p0.y, p2.z - p0.z
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nlen < FLOAT_EPS:
        return False
    nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
    d = -(nx * p0.x + ny * p0.y + nz * p0.z)
    for i in range(3, n):
        c = corners[i]
        residual = abs(nx * c.x + ny * c.y + nz * c.z + d)
        if residual > FLOAT_EPS:
            return False
    return True


def polygon_xz_contains_point(corners: Sequence[Vec3], px: float, pz: float) -> bool:
    """Even-odd point-in-polygon on the XZ projection. Boundary may classify
    either way (acceptable for invariant checks; we only use this where
    boundary is excluded by float-precision construction)."""
    n = len(corners)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        ai, aj = corners[i], corners[j]
        if (ai.z > pz) != (aj.z > pz):
            x_at = ai.x + (pz - ai.z) * (aj.x - ai.x) / (aj.z - ai.z + FLOAT_EPS)
            if px < x_at:
                inside = not inside
        j = i
    return inside


def edge_endpoints(corners: Sequence[Vec3]) -> list[tuple[Vec3, Vec3]]:
    """All adjacent (corner[i], corner[i+1]) pairs, wrapping at the end."""
    n = len(corners)
    return [(corners[i], corners[(i + 1) % n]) for i in range(n)]


def fit_plane(points: Sequence[Vec3]) -> Plane | None:
    """Best-fit plane through ≥3 points, normal facing upward.

    Uses SVD on the centered point matrix; the smallest singular vector
    is the plane normal. Returns `None` if the input is degenerate
    (collinear points → plane is undefined).

    The normal is unit-length (so `plane.b` is the signed Y component
    of the unit normal, as the rest of `_geometry` assumes).
    """
    import numpy as np

    if len(points) < 3:
        return None
    arr = np.array([(p.x, p.y, p.z) for p in points], dtype=float)
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular[1] < FLOAT_EPS:
        # Two singular values vanish → input is collinear, no plane.
        return None
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    d = -(nx * centroid[0] + ny * centroid[1] + nz * centroid[2])
    return Plane(a=nx, b=ny, c=nz, d=float(d))


def plane_plane_intersection(
    p1: Plane, p2: Plane
) -> tuple[Vec3, tuple[float, float, float]] | None:
    """Intersect two planes; return `(point_on_line, unit_direction)` or
    `None` if the planes are parallel (within FLOAT_EPS on the cross
    product magnitude)."""
    import numpy as np

    n1 = np.array([p1.a, p1.b, p1.c], dtype=float)
    n2 = np.array([p2.a, p2.b, p2.c], dtype=float)
    direction = np.cross(n1, n2)
    mag = float(np.linalg.norm(direction))
    if mag < FLOAT_EPS:
        return None
    direction /= mag
    a = np.array([n1, n2, direction], dtype=float)
    rhs = np.array([-p1.d, -p2.d, 0.0], dtype=float)
    try:
        point = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return None
    return (
        Vec3(x=float(point[0]), y=float(point[1]), z=float(point[2])),
        (float(direction[0]), float(direction[1]), float(direction[2])),
    )


def lift_xz_to_plane(corners_xz: Sequence[Vec3], plane: Plane) -> tuple[Vec3, ...]:
    """Replace each corner's Y with the plane's Y at that (x, z).

    Requires the plane to be non-vertical (`plane.b != 0`). Used to lift
    a 2D floor-shape polygon onto a fitted ceiling plane for the
    canonical Ceiling polygon.
    """
    if abs(plane.b) < FLOAT_EPS:
        raise ValueError("cannot lift onto a vertical plane")
    out: list[Vec3] = []
    for c in corners_xz:
        y = -(plane.a * c.x + plane.c * c.z + plane.d) / plane.b
        out.append(Vec3(x=c.x, y=y, z=c.z))
    return tuple(out)
