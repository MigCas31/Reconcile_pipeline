"""Triangulation and ceiling-fallback helpers used by `compute_gap_walls`.

Extracted from `extract/gaps.py` so the gap-wall builder is split from the
geometry primitives it consumes. Public re-exports live in `gaps.py`.
"""

from __future__ import annotations

from shapely.geometry import Point

from reconcile_tiers.extract.room_ceiling import ceiling_y_at_xz

MIN_TRI_QUALITY = 0.02


def _ytop_at_xz(xz_pt, snapped, eps: float = 1e-3) -> float:
    eps2 = eps * eps
    for item in snapped:
        dx = float(xz_pt[0]) - float(item["xz"][0])
        dz = float(xz_pt[1]) - float(item["xz"][1])
        if dx * dx + dz * dz < eps2:
            return float(item["ytop"])
    best = None
    for idx in range(len(snapped)):
        nxt = (idx + 1) % len(snapped)
        p0 = snapped[idx]["xz"]
        p1 = snapped[nxt]["xz"]
        ex = float(p1[0]) - float(p0[0])
        ez = float(p1[1]) - float(p0[1])
        length2 = ex * ex + ez * ez
        if length2 < 1e-12:
            continue
        t = (
            (float(xz_pt[0]) - float(p0[0])) * ex
            + (float(xz_pt[1]) - float(p0[1])) * ez
        ) / length2
        t_clamped = max(0.0, min(1.0, t))
        proj_x = float(p0[0]) + t_clamped * ex
        proj_z = float(p0[1]) + t_clamped * ez
        dist = (float(xz_pt[0]) - proj_x) ** 2 + (float(xz_pt[1]) - proj_z) ** 2
        if best is None or dist < best[0]:
            best = (
                dist,
                t_clamped,
                float(snapped[idx]["ytop"]),
                float(snapped[nxt]["ytop"]),
            )
    if best is None:
        return float(snapped[0]["ytop"])
    _dist, t, y0, y1 = best
    return y0 + t * (y1 - y0)


def _apply_room_ceiling_fallback(
    piece_snapped: list[dict],
    story_ceiling_lookup: dict,
    story: int,
) -> None:
    """For vertices without scan-derived ceiling support, look up the host
    story's ceiling at the vertex XZ. When found, anchor the vertex at that
    ceiling height and mark it preserved. When no room covers the XZ
    (gap polygon extends past room footprints — common in half-height /
    split-level layouts), leave the existing ``ytop`` from wall snapping in
    place; the sliver filter and inclination guard will catch genuinely
    degenerate caps without dropping legitimate closures."""
    for item in piece_snapped:
        if item.get("preserve_ceiling_profile"):
            continue
        x, z = float(item["xz"][0]), float(item["xz"][1])
        ceiling_y = ceiling_y_at_xz(story_ceiling_lookup, story, x, z)
        if ceiling_y is not None:
            item["ytop"] = float(ceiling_y)
            item["preserve_ceiling_profile"] = True


def _triangle_quality(a: list[float], b: list[float], c: list[float]) -> float:
    """Shewchuk-style quality measure: 2*area / longest_edge^2.
    Triangles below ~0.05 are slivers (extreme aspect ratio)."""
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    e_lens = [
        ((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5,
        ((cx - bx) ** 2 + (cy - by) ** 2 + (cz - bz) ** 2) ** 0.5,
        ((ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2) ** 0.5,
    ]
    longest = max(e_lens)
    if longest <= 1e-9:
        return 0.0
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    twice_area = (nx * nx + ny * ny + nz * nz) ** 0.5
    return float(twice_area / (longest * longest))


def _edge_on_room_boundary(p0, p1, room_boundary, eps: float = 0.02) -> bool:
    if room_boundary is None:
        return False
    midpoint = Point(
        (float(p0[0]) + float(p1[0])) / 2.0, (float(p0[1]) + float(p1[1])) / 2.0
    )
    try:
        return midpoint.distance(room_boundary) < eps
    except Exception:
        return False


def earclip_2d(coords, eps: float = 1e-3):
    if len(coords) < 3:
        return []
    cleaned = []
    src_idx = []
    for idx, coord in enumerate(coords):
        if cleaned:
            dx = coord[0] - cleaned[-1][0]
            dy = coord[1] - cleaned[-1][1]
            if dx * dx + dy * dy < eps * eps:
                continue
        cleaned.append(coord)
        src_idx.append(idx)
    if len(cleaned) >= 2:
        dx = cleaned[0][0] - cleaned[-1][0]
        dy = cleaned[0][1] - cleaned[-1][1]
        if dx * dx + dy * dy < eps * eps:
            cleaned.pop()
            src_idx.pop()
    changed = True
    while changed and len(cleaned) > 3:
        changed = False
        next_coords = []
        next_idx = []
        for idx in range(len(cleaned)):
            prev = cleaned[(idx - 1) % len(cleaned)]
            curr = cleaned[idx]
            nxt = cleaned[(idx + 1) % len(cleaned)]
            cross_v = (curr[0] - prev[0]) * (nxt[1] - prev[1]) - (curr[1] - prev[1]) * (
                nxt[0] - prev[0]
            )
            if abs(cross_v) > eps:
                next_coords.append(curr)
                next_idx.append(src_idx[idx])
            else:
                changed = True
        if len(next_coords) < 3:
            break
        cleaned, src_idx = next_coords, next_idx
    n = len(cleaned)
    if n < 3:
        return []
    if n == 3:
        return [(src_idx[0], src_idx[1], src_idx[2])]
    area2 = sum(
        cleaned[i][0] * cleaned[(i + 1) % n][1]
        - cleaned[(i + 1) % n][0] * cleaned[i][1]
        for i in range(n)
    )
    if area2 < 0:
        cleaned = list(reversed(cleaned))
        src_idx = list(reversed(src_idx))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_triangle(p, a, b, c):
        d1 = cross(p, a, b)
        d2 = cross(p, b, c)
        d3 = cross(p, c, a)
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (has_neg and has_pos)

    work = list(range(n))
    triangles = []
    while len(work) > 3:
        ear_found = False
        for idx in range(len(work)):
            ip, ic, inext = (
                work[(idx - 1) % len(work)],
                work[idx],
                work[(idx + 1) % len(work)],
            )
            a, b, c = cleaned[ip], cleaned[ic], cleaned[inext]
            if cross(a, b, c) <= 0:
                continue
            inside = False
            for j in range(len(work)):
                if j in ((idx - 1) % len(work), idx, (idx + 1) % len(work)):
                    continue
                if point_in_triangle(cleaned[work[j]], a, b, c):
                    inside = True
                    break
            if inside:
                continue
            triangles.append((src_idx[ip], src_idx[ic], src_idx[inext]))
            work.pop(idx)
            ear_found = True
            break
        if not ear_found:
            for idx in range(1, len(work) - 1):
                triangles.append(
                    (src_idx[work[0]], src_idx[work[idx]], src_idx[work[idx + 1]])
                )
            return triangles
    triangles.append((src_idx[work[0]], src_idx[work[1]], src_idx[work[2]]))
    return triangles
