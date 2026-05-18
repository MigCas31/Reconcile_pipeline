"""Coplanar raw-ceiling merge.

RoomPlan over-segments large planar surfaces (one attic slope captured as
2-3 patches sharing an edge but emitted as separate planes). This module
detects coplanar+touching raw-ceiling polygons in the same room and
unions them into one polygon. Thresholds chosen from a corpus histogram
of pairwise normal angle vs max point-to-plane distance across 223
buildings: 3°/5 cm catches 326 (building, room) cases without sweeping
in unrelated near-parallel patches.

Extracted from `extract/ceilings.py`; the public entry point
`merge_coplanar_raw_ceilings` is re-exported there.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract.building import ExtractedRoom, RawCeilingPlane

# Constants are duplicated from `ceilings.py` to keep this module standalone
# at import time; they describe the merge contract (which planes are coplanar
# enough to unify, which can't be lifted back to 3D).
COPLANAR_NORMAL_ANGLE_DEG = 3.0
COPLANAR_OFFSET_M = 0.05
COPLANAR_MIN_NY = 0.10  # skip near-vertical planes -- can't lift back to 3D


def _plane_geometry(corners: list[list[float]]) -> tuple | None:
    if len(corners) < 3:
        return None
    n = len(corners)
    nx = ny = nz = 0.0
    for i in range(n):
        a, b = corners[i], corners[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-9:
        return None
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    cx = sum(c[0] for c in corners) / n
    cy = sum(c[1] for c in corners) / n
    cz = sum(c[2] for c in corners) / n
    d = nx * cx + ny * cy + nz * cz
    poly = make_valid(Polygon([(c[0], c[2]) for c in corners]))
    if not poly.is_valid or poly.is_empty or not isinstance(poly, Polygon):
        return None
    if poly.area < 1e-6:  # degenerate (line / point in XZ -- can't merge meaningfully)
        return None
    return (nx, ny, nz), d, poly


def _coplanar_within_tolerance(g_a: tuple, g_b: tuple) -> bool:
    na, da, _ = g_a
    nb, db, poly_b = g_b
    dot = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]
    ang = math.degrees(math.acos(min(1.0, abs(dot))))
    if ang > COPLANAR_NORMAL_ANGLE_DEG:
        return False
    sign = 1.0 if dot >= 0 else -1.0
    db_signed = sign * db
    nb_signed = (sign * nb[0], sign * nb[1], sign * nb[2])
    poly_a = g_a[2]
    # Sample plane-A vertices' distance to plane-B and vice versa via centroids
    # of each polygon -- cheap proxy for max distance between the two patches.
    for poly, n_other, d_other in (
        (poly_a, nb_signed, db_signed),
        (poly_b, na, da),
    ):
        cx = poly.centroid.x
        cz = poly.centroid.y
        # We need y of the patch at (cx, cz). Use the patch's own plane to get y.
        n_self, d_self, _ = g_a if poly is poly_a else g_b
        if abs(n_self[1]) < 1e-6:
            return False
        y_self = (d_self - n_self[0] * cx - n_self[2] * cz) / n_self[1]
        dist = abs(n_other[0] * cx + n_other[1] * y_self + n_other[2] * cz - d_other)
        if dist > COPLANAR_OFFSET_M:
            return False
    return True


def _xz_polygons_touch(poly_a: Polygon, poly_b: Polygon) -> bool:
    # Buffered touch test: Shapely's `intersects` is True for shared points;
    # we want shared-edge or overlap. Buffer by 1 cm and require non-zero
    # intersection area.
    inter = poly_a.buffer(0.01).intersection(poly_b.buffer(0.01))
    if inter.is_empty:
        return False
    return float(inter.area) > 1e-6


def _project_xz_polygon_to_plane(
    poly: Polygon, normal: tuple[float, float, float], d: float
) -> list[list[float]]:
    nx, ny, nz = normal
    coords = list(poly.exterior.coords)[:-1]  # drop closing duplicate
    out: list[list[float]] = []
    for x, z in coords:
        y = (d - nx * x - nz * z) / ny
        pt = [round(float(x), 4), round(float(y), 4), round(float(z), 4)]
        if out and out[-1] == pt:
            continue
        out.append(pt)
    while len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    return out


def _merge_plane_group(
    planes: list[RawCeilingPlane], geoms: list[tuple]
) -> list[RawCeilingPlane]:
    # Area-weighted consensus plane (anchor first plane's normal direction).
    anchor_n = geoms[0][0]
    weighted_n = np.zeros(3, dtype=float)
    total_w = 0.0
    for n_vec, _d_val, poly in geoms:
        w = float(poly.area)
        sign = (
            1.0
            if (
                n_vec[0] * anchor_n[0] + n_vec[1] * anchor_n[1] + n_vec[2] * anchor_n[2]
            )
            >= 0
            else -1.0
        )
        weighted_n[0] += sign * n_vec[0] * w
        weighted_n[1] += sign * n_vec[1] * w
        weighted_n[2] += sign * n_vec[2] * w
        total_w += w
    if total_w <= 0:
        return list(planes)
    norm_len = float(np.linalg.norm(weighted_n))
    if norm_len < 1e-9:
        return list(planes)
    n_consensus = (
        float(weighted_n[0] / norm_len),
        float(weighted_n[1] / norm_len),
        float(weighted_n[2] / norm_len),
    )
    # Area-weighted d, evaluated at each patch's centroid using the patch's
    # own plane to reconstruct y, then projected onto the consensus normal.
    d_consensus = 0.0
    for n_vec, d_val, poly in geoms:
        cx = poly.centroid.x
        cz = poly.centroid.y
        if abs(n_vec[1]) < 1e-6:
            continue
        y_p = (d_val - n_vec[0] * cx - n_vec[2] * cz) / n_vec[1]
        d_consensus += float(poly.area) * (
            n_consensus[0] * cx + n_consensus[1] * y_p + n_consensus[2] * cz
        )
    d_consensus /= total_w
    if abs(n_consensus[1]) < COPLANAR_MIN_NY:
        return list(planes)

    components = _bridged_union_components([g[2] for g in geoms])
    out: list[RawCeilingPlane] = []
    for comp in components:
        comp_clean = comp.simplify(0.01, preserve_topology=True)
        if not isinstance(comp_clean, Polygon) or comp_clean.is_empty:
            continue
        corners = _project_xz_polygon_to_plane(comp_clean, n_consensus, d_consensus)
        if len(corners) >= 3:
            out.append(RawCeilingPlane(corners=corners))
    return out if out else list(planes)


# Shared-corner bridging: when two coplanar patches touch only at a single XZ
# point, plain unary_union returns a MultiPolygon. Dilating each input by a
# small epsilon before union creates a thin "neck" at the shared corner so the
# union becomes a single Polygon; we then erode back by the same epsilon and
# fall back to the un-bridged union if the erode dissolves the neck (i.e. the
# patches really were disjoint, not just point-touching).
COPLANAR_BRIDGE_EPS_M = 0.02


def _bridged_union_components(polys: list[Polygon]) -> list[Polygon]:
    raw = unary_union(polys)
    raw_components = _polygon_components(raw)
    if len(raw_components) <= 1:
        return raw_components
    bridged = unary_union([p.buffer(COPLANAR_BRIDGE_EPS_M) for p in polys])
    if not isinstance(bridged, Polygon):
        return raw_components
    eroded = bridged.buffer(-COPLANAR_BRIDGE_EPS_M)
    eroded_components = _polygon_components(eroded)
    if len(eroded_components) == 1:
        return eroded_components
    return raw_components


def _polygon_components(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    return []


def merge_coplanar_raw_ceilings(rooms: list[ExtractedRoom]) -> list[ExtractedRoom]:
    out: list[ExtractedRoom] = []
    for room in rooms:
        planes = list(room.raw_ceiling_planes)
        if len(planes) < 2:
            out.append(room)
            continue
        geoms: list[tuple | None] = [_plane_geometry(p.corners) for p in planes]

        n = len(planes)
        parent = list(range(n))

        def find(x: int, parent: list[int] = parent) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            gi = geoms[i]
            if gi is None:
                continue
            for j in range(i + 1, n):
                gj = geoms[j]
                if gj is None:
                    continue
                if not _coplanar_within_tolerance(gi, gj):
                    continue
                if not _xz_polygons_touch(gi[2], gj[2]):
                    continue
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged_planes: list[RawCeilingPlane] = []
        for members in groups.values():
            if len(members) == 1:
                merged_planes.append(planes[members[0]])
                continue
            valid = [(planes[i], geoms[i]) for i in members if geoms[i] is not None]
            if len(valid) < 2:
                merged_planes.extend(planes[i] for i in members)
                continue
            ms = _merge_plane_group([v[0] for v in valid], [v[1] for v in valid])
            merged_planes.extend(ms)
        out.append(replace(room, raw_ceiling_planes=merged_planes))
    return out
