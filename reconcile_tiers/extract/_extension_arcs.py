"""Synthetic perimeter-arc helpers used by `compute_descent_strips` and
`compute_uplift_strips`.

Detects boundary segments on the upper-story footprint that no scanned
upper wall covers, and assigns each missing arc to the most-likely owning
room. Wing membership is also computed here since it's only consumed by
the extension pipeline.

Extracted from `extension.py`; the public entry points stay there.
"""

from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from reconcile_tiers.extract.building import ExtractedRoom

MIN_SYNTHETIC_ARC_LENGTH_M = 0.10
PERIMETER_COVERAGE_BUFFER_M = 0.05


def _missing_perimeter_arcs(
    boundary,
    perimeter_segments: list[LineString],
) -> list[LineString]:
    if not perimeter_segments:
        if isinstance(boundary, LineString):
            single = boundary
        else:
            parts = [
                g for g in getattr(boundary, "geoms", []) if isinstance(g, LineString)
            ]
            single = max(parts, key=lambda g: g.length) if parts else None
        return (
            [single]
            if single is not None and single.length >= MIN_SYNTHETIC_ARC_LENGTH_M
            else []
        )
    covered = unary_union(
        [
            seg.buffer(PERIMETER_COVERAGE_BUFFER_M, cap_style="flat")
            for seg in perimeter_segments
        ]
    )
    remaining = boundary.difference(covered)
    arcs: list[LineString] = []
    if remaining.is_empty:
        return arcs
    if isinstance(remaining, LineString):
        candidates = [remaining]
    else:
        candidates = [
            g for g in getattr(remaining, "geoms", []) if isinstance(g, LineString)
        ]
    for arc in candidates:
        if arc.length >= MIN_SYNTHETIC_ARC_LENGTH_M:
            arcs.append(arc)
    return arcs


def _arc_room_owner(
    arc: LineString,
    rooms_with_polys: list[tuple[int, Polygon]],
    *,
    wings: list | None = None,
    rooms: list[ExtractedRoom] | None = None,
) -> int | None:
    """Pick the room that owns an uncovered perimeter arc.

    Default rule (no `wings`): smallest midpoint-to-boundary distance.

    Wing-aware rule: filter candidate rooms to those whose floor polygon
    overlaps the same wing as the arc midpoint, then apply the distance
    tiebreaker. Falls back to the unrestricted set if no candidate room
    sits in the arc's wing — never silently drops an arc.
    """
    midpoint = arc.interpolate(0.5, normalized=True)
    candidates = rooms_with_polys
    if wings and rooms is not None:
        from reconcile_tiers._core.wing_decomposition import wing_polygon_for_room

        arc_wing_poly = None
        for wing in wings:
            if (
                wing.polygon.contains(midpoint)
                or wing.polygon.boundary.distance(midpoint) <= 1e-6
            ):
                arc_wing_poly = wing.polygon
                break
        if arc_wing_poly is not None:
            filtered = []
            for idx, poly in rooms_with_polys:
                if 0 <= idx < len(rooms):
                    rw = wing_polygon_for_room(rooms[idx], wings)
                    if rw is arc_wing_poly:
                        filtered.append((idx, poly))
            if filtered:
                candidates = filtered
    best_idx = None
    best_dist = float("inf")
    for idx, poly in candidates:
        d = poly.boundary.distance(midpoint)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _stable_synth_id(room_index: int, arc: LineString, suffix: int) -> str:
    coords = list(arc.coords)
    a = coords[0]
    b = coords[-1]
    return (
        f"synth-descent:room{room_index}:"
        f"{a[0]:.3f},{a[1]:.3f}-{b[0]:.3f},{b[1]:.3f}#{suffix}"
    )


def _build_wing_membership(
    rooms: list[ExtractedRoom],
    wings: list | None,
) -> dict[int, int | None]:
    """Map room_index -> Wing.index (or None when no wings or no overlap)."""
    if not wings:
        return {idx: None for idx in range(len(rooms))}
    from reconcile_tiers._core.wing_decomposition import wing_polygon_for_room

    out: dict[int, int | None] = {}
    poly_to_index = {id(w.polygon): w.index for w in wings}
    for idx, room in enumerate(rooms):
        wp = wing_polygon_for_room(room, wings)
        out[idx] = poly_to_index.get(id(wp)) if wp is not None else None
    return out
