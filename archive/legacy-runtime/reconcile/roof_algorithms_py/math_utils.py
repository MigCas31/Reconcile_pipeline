from __future__ import annotations

from collections.abc import Callable, Sequence
from math import cos, sin, sqrt

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def ridge_line_intersection_2d(pi: dict, pj: dict) -> dict | None:
    """Compute the 2D (x, z) intersection of two planes' ridge lines.

    Returns ``{"x": ..., "z": ...}`` or ``None`` if the ridge lines are
    parallel (or near-parallel).
    """
    dx = pj["ref"]["x"] - pi["ref"]["x"]
    dz = pj["ref"]["z"] - pi["ref"]["z"]
    det = pi["ridgeX"] * pj["ridgeZ"] - pi["ridgeZ"] * pj["ridgeX"]
    if abs(det) < 1e-9:
        return None
    t = (dx * pj["ridgeZ"] - dz * pj["ridgeX"]) / det
    return {
        "x": pi["ref"]["x"] + t * pi["ridgeX"],
        "z": pi["ref"]["z"] + t * pi["ridgeZ"],
    }


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def point_in_poly_xz(px: float, pz: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = poly[i][0], poly[i][2]
        xj, zj = poly[j][0], poly[j][2]
        if (zi > pz) != (zj > pz) and px < (xj - xi) * (pz - zi) / (zj - zi) + xi:
            inside = not inside
        j = i
    return inside


def point_in_poly_2d(px: float, pz: float, poly: Sequence[Point2]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if (zi > pz) != (zj > pz) and px < (xj - xi) * (pz - zi) / (zj - zi) + xi:
            inside = not inside
        j = i
    return inside


def plane_normal(azimuth_deg: float, incl_deg: float) -> dict:
    a = azimuth_deg * 3.141592653589793 / 180.0
    i = incl_deg * 3.141592653589793 / 180.0
    sx, sy, sz = sin(a) * cos(i), -sin(i), cos(a) * cos(i)
    rx, rz = -cos(a), sin(a)
    return {"x": sy * rz, "y": sz * rx - sx * rz, "z": -sy * rx}


def seg_midpoint(seg: dict) -> dict:
    return {
        "x": (seg["a"][0] + seg["b"][0]) / 2.0,
        "y": (seg["a"][1] + seg["b"][1]) / 2.0,
        "z": (seg["a"][2] + seg["b"][2]) / 2.0,
    }


def plane_y_at(plane: dict, x: float, z: float) -> float:
    n, ref = plane["n"], plane["ref"]
    return ref["y"] - (n["x"] * (x - ref["x"]) + n["z"] * (z - ref["z"])) / n["y"]


def clip_by_max_y(poly: Sequence[Point3], max_y: float) -> list[Point3]:
    if len(poly) < 3:
        return list(poly)
    out: list[Point3] = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i + 1) % n]
        c_in, n_in = curr[1] <= max_y, nxt[1] <= max_y
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = (max_y - curr[1]) / (nxt[1] - curr[1])
            out.append(
                (
                    curr[0] + t * (nxt[0] - curr[0]),
                    max_y,
                    curr[2] + t * (nxt[2] - curr[2]),
                )
            )
    return out


def clip_by_half_plane_xz(
    poly: Sequence[Point3], ox: float, oz: float, nx: float, nz: float
) -> list[Point3]:
    if len(poly) < 3:
        return list(poly)
    out: list[Point3] = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i + 1) % n]
        cv = nx * (curr[0] - ox) + nz * (curr[2] - oz)
        nv = nx * (nxt[0] - ox) + nz * (nxt[2] - oz)
        c_in, n_in = cv >= 0.0, nv >= 0.0
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = cv / (cv - nv)
            out.append(
                (
                    curr[0] + t * (nxt[0] - curr[0]),
                    curr[1] + t * (nxt[1] - curr[1]),
                    curr[2] + t * (nxt[2] - curr[2]),
                )
            )
    return out


def clip_poly_by_ridge(
    poly: Sequence[Point2],
    r_dir_x: float,
    r_dir_z: float,
    ref_x: float,
    ref_z: float,
    bound: float,
    keep_above: bool,
) -> list[Point2]:
    if len(poly) < 3:
        return list(poly)
    out: list[Point2] = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i + 1) % n]
        cv = (curr[0] - ref_x) * r_dir_x + (curr[1] - ref_z) * r_dir_z - bound
        nv = (nxt[0] - ref_x) * r_dir_x + (nxt[1] - ref_z) * r_dir_z - bound
        c_in = cv >= 0.0 if keep_above else cv <= 0.0
        n_in = nv >= 0.0 if keep_above else nv <= 0.0
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = cv / (cv - nv)
            out.append(
                (
                    curr[0] + t * (nxt[0] - curr[0]),
                    curr[1] + t * (nxt[1] - curr[1]),
                )
            )
    return out


def clip_poly_by_half_plane_2d(
    poly: Sequence[Point2], dx: float, dz: float, offset: float
) -> list[Point2]:
    if len(poly) < 3:
        return list(poly)
    out: list[Point2] = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i + 1) % n]
        cv = dx * curr[0] + dz * curr[1] + offset
        nv = dx * nxt[0] + dz * nxt[1] + offset
        c_in, n_in = cv <= 0.0, nv <= 0.0
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = cv / (cv - nv)
            out.append(
                (
                    curr[0] + t * (nxt[0] - curr[0]),
                    curr[1] + t * (nxt[1] - curr[1]),
                )
            )
    return out


def convex_hull_2d(points: Sequence[dict]) -> list[dict]:
    if len(points) < 3:
        return list(points)
    pts = sorted(points, key=lambda p: (p["x"], p["z"]))

    def cross(o: dict, a: dict, b: dict) -> float:
        return (a["x"] - o["x"]) * (b["z"] - o["z"]) - (a["z"] - o["z"]) * (
            b["x"] - o["x"]
        )

    lower: list[dict] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[dict] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    lower.pop()
    upper.pop()
    return lower + upper


def point_near_footprint(
    px: float,
    pz: float,
    footprint: Sequence[Point2],
    margin: float,
    point_in_poly_fn: Callable[
        [float, float, Sequence[Point2]], bool
    ] = point_in_poly_2d,
) -> bool:
    if point_in_poly_fn(px, pz, footprint):
        return True
    if margin <= 0:
        return False
    n = len(footprint)
    for i in range(n):
        j = (i + 1) % n
        ax, az = footprint[i]
        bx, bz = footprint[j]
        edx, edz = bx - ax, bz - az
        len2 = edx * edx + edz * edz
        if len2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * edx + (pz - az) * edz) / len2))
        cx, cz = ax + t * edx, az + t * edz
        if sqrt((px - cx) ** 2 + (pz - cz) ** 2) <= margin:
            return True
    return False
