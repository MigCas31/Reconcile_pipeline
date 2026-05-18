"""Post-split continuity-merge for `SplitObliqueSurface` cells.

This module previously hosted Phase 1's post-edit yaw-snap and
continuity-merge for `ObliqueSurface`. Both were retired (2026-04-30) once
priors-at-source replaced them: cluster-level azimuth snap (LAYER 2 in
`roof/clustering.py`) and segment-level filter (LAYER 1 in
`roof/segments.py`) produce axis-aligned planes by construction, so there's
nothing left to post-edit on `roof.oblique`. Wall-axis helpers moved to
`reconcile_tiers/_core/wall_axis.py`.

What stays here is `clean_split_oblique_surfaces`: a continuity-merge for
**post-split** cells. `roof/arrangement.py:split_oblique_surfaces` fans each
source oblique into one cell per intersecting room. Co-planar cells from the
same source (and occasionally from different sources) get emitted as separate
pieces, inflating the audit's `roof_fragmented` count. This pass collapses
connected co-planar cells back into one emission. Disjoint co-planar cells
are left as-is — they're separate physical surfaces that just happen to
share a plane equation.

Cohort effect (2026-04-29 measurement, with the convention fix in place):
~21% reduction in `roof_cross_extension` flags. The most measurable Phase 1
contribution; kept for that reason.

Trade-off (documented for downstream adjacency callers): merging across
rooms loses room-level adjacency tagging granularity. The merged cell
inherits `arrangement_cell_id` and `source_oblique_index` from the largest
input cell. Downstream adjacency analysis (heat-loss calc) operates on
emitted pieces; merging means one fewer piece with a single adjacency tag.
For cohort measurement this is acceptable; production heat-loss may need
tuning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import Plane, planes_equivalent
from reconcile_tiers.roof.roof import SplitObliqueSurface


def _planes_match(p1: Plane, p2: Plane) -> bool:
    return planes_equivalent(p1, p2)


def _parse_room_idx_from_cell(cell_id: str) -> int | None:
    """Extract the room index from `cell:<source>:room:<room>:<part>` cell IDs.

    Returns None for non-matching shapes — those cells get their own merge
    bucket so we never accidentally cross-merge per-room slices.
    """
    parts = cell_id.split(":")
    if len(parts) < 5 or parts[2] != "room":
        return None
    try:
        return int(parts[3])
    except ValueError:
        return None


def clean_split_oblique_surfaces(
    surfaces: list[SplitObliqueSurface],
    *,
    enabled: bool = True,
) -> list[SplitObliqueSurface]:
    """Merge co-planar split cells that share `source_oblique_index` AND
    `room_index` AND whose XZ polygons union into a single connected
    `Polygon`.

    Scoped to within-source + within-room: `arrangement.split_oblique_surfaces`
    intentionally fans one source oblique into per-room cells, and Layer 2
    cluster snap makes their planes identical — so merging by plane alone
    collapses the per-room arrangement back to per-oblique blobs and loses
    all room-level adjacency tagging granularity. The audit's
    `roof_fragmented` signal targets within-room/within-source fragmentation,
    which is what this pass cleans up.
    """
    if not enabled or len(surfaces) < 2:
        return surfaces
    n = len(surfaces)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    rooms = [_parse_room_idx_from_cell(s.arrangement_cell_id) for s in surfaces]
    for i in range(n):
        for j in range(i + 1, n):
            if not _planes_match(surfaces[i].plane, surfaces[j].plane):
                continue
            same_source = (
                surfaces[i].source_oblique_index == surfaces[j].source_oblique_index
            )
            same_room = rooms[i] is not None and rooms[i] == rooms[j]
            # Same-source per-room cells are intentional fragments from
            # `arrangement.split_oblique_surfaces`; merging across rooms would
            # collapse the per-room arrangement back to a single blob and
            # lose room-level adjacency tagging. Only merge same-source cells
            # if they're in the same room (genuine within-room fragmentation).
            # Cross-source coplanar cells get merged unconditionally — that's
            # the original `roof_cross_extension` fix.
            if same_source and not same_room:
                continue
            union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[SplitObliqueSurface] = []
    for indices in groups.values():
        group = [surfaces[i] for i in indices]
        if len(group) == 1:
            out.append(group[0])
            continue
        merged = _merge_coplanar_split_group(group)
        if merged is None:
            out.extend(group)
        else:
            out.append(merged)
    return out or surfaces


def _merge_coplanar_split_group(
    group: list[SplitObliqueSurface],
) -> SplitObliqueSurface | None:
    """Union XZ polygons; if connected, emit one merged SplitObliqueSurface."""
    from reconcile_tiers._core.shapely2 import make_valid_polygon

    polys: list[Polygon] = []
    for s in group:
        try:
            poly = Polygon([(float(c[0]), float(c[2])) for c in s.corners])
        except Exception:
            return None
        if not poly.is_valid:
            poly = make_valid_polygon(poly)
        if poly is None or poly.is_empty or poly.area < 1e-9:
            continue
        polys.append(poly)
    if len(polys) < 2:
        return None
    merged_xz = unary_union(polys)
    if not isinstance(merged_xz, Polygon) or merged_xz.is_empty:
        return None  # disjoint -> coplanar coincidence, not a fragment

    merged_plane = _split_group_area_weighted_plane(group)
    coords = list(merged_xz.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    new_corners: list[list[float]] = []
    for x, z in coords:
        y = merged_plane.y_at(float(x), float(z))
        if y is None:
            return None
        new_corners.append([float(x), float(y), float(z)])
    if len(new_corners) < 3:
        return None

    largest = max(group, key=lambda s: _xz_polygon_area_3d(s.corners))
    return SplitObliqueSurface(
        corners=new_corners,
        plane=merged_plane,
        arrangement_cell_id=largest.arrangement_cell_id,
        source_oblique_index=largest.source_oblique_index,
        dominant_story=largest.dominant_story,
    )


def _xz_polygon_area_3d(corners: Sequence[Sequence[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    pts = [(float(c[0]), float(c[2])) for c in corners]
    area2 = 0.0
    n = len(pts)
    for i in range(n):
        x1, z1 = pts[i]
        x2, z2 = pts[(i + 1) % n]
        area2 += x1 * z2 - x2 * z1
    return abs(area2) * 0.5


def _split_group_area_weighted_plane(group: list[SplitObliqueSurface]) -> Plane:
    weights = [_xz_polygon_area_3d(s.corners) for s in group]
    total = sum(weights)
    if total < 1e-9:
        return group[0].plane
    a = sum(s.plane.a * w for s, w in zip(group, weights, strict=False)) / total
    b = sum(s.plane.b * w for s, w in zip(group, weights, strict=False)) / total
    c = sum(s.plane.c * w for s, w in zip(group, weights, strict=False)) / total
    d = sum(s.plane.d * w for s, w in zip(group, weights, strict=False)) / total
    norm = math.sqrt(a * a + b * b + c * c)
    if norm < 1e-9:
        return group[0].plane
    return Plane(a=a / norm, b=b / norm, c=c / norm, d=d / norm)
