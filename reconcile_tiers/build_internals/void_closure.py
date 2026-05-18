"""Residual ceiling-void and within-story free-edge closure helpers.

Re-checks the final rendered roof/ceiling coverage and adds gap caps for
real residual voids; appends V1-style side walls for within-story gap
edges the slab-pair extractor missed. Re-exported from
`reconcile_tiers.build`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from reconcile_tiers.assemble.gaps_to_pieces import assemble_gap_pieces
from reconcile_tiers.build_internals.constants import (
    _DUPLICATE_WALL_DIRECTION_DOT_MIN,
    _DUPLICATE_WALL_ENDPOINT_TOL_M,
    _DUPLICATE_WALL_OVERLAP_RATIO_MIN,
    _WITHIN_STORY_FREE_EDGE_BOUNDARY_NEAR_M,
    _WITHIN_STORY_FREE_EDGE_MAX_SOURCE_DEFORMATION_M,
    _WITHIN_STORY_FREE_EDGE_MAX_SOURCE_DEFORMATION_RATIO,
    _WITHIN_STORY_FREE_EDGE_MIN_SUPPORT_RATIO,
    _WITHIN_STORY_FREE_EDGE_SUPPORT_BUFFER_M,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingPiece, GapKind

_RESIDUAL_VOID_COVERAGE_GAP_KINDS = {
    GapKind.GAP_CEILING,
    GapKind.STITCH_CEIL,
    GapKind.EXTERIOR_CEIL,
}


def _piece_corners_as_lists(piece) -> list[list[float]]:
    return [
        [float(corner.x), float(corner.y), float(corner.z)] for corner in piece.corners
    ]


def _close_residual_room_ceiling_voids(
    model: BuildingModel,
    ceilings: list[CeilingPiece],
) -> BuildingModel:
    """Add gap caps for room footprint left open after final ceiling synthesis.

    The early room-ceiling-void pass runs before raw filtering, painting,
    dormer reconstruction, and late synthesis. Those later steps can remove or
    split ownership, so re-check the final rendered roof/ceiling coverage and
    only add closures for real residual voids.
    """
    from reconcile_tiers.extract.gaps import (
        compute_gap_walls,
        compute_room_ceiling_voids,
        story_y_map_from_rooms,
    )

    coverage_polygons = [_piece_corners_as_lists(piece) for piece in ceilings]
    for gap in assemble_gap_pieces(model):
        if gap.kind in _RESIDUAL_VOID_COVERAGE_GAP_KINDS:
            coverage_polygons.append(_piece_corners_as_lists(gap))

    extra_gaps = compute_room_ceiling_voids(model.rooms, coverage_polygons)
    if not extra_gaps:
        return model

    extra_walls, draped_extras = compute_gap_walls(
        extra_gaps,
        model.rooms,
        story_y_map_from_rooms(model.rooms),
    )
    if not extra_walls and not draped_extras:
        return model
    return replace(
        model,
        gap_walls=list(model.gap_walls) + extra_walls,
        cross_floor_gaps=list(model.cross_floor_gaps) + draped_extras,
    )


def _wall_edge_xz(wall) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(wall.corners) < 2:
        return None
    a, b = wall.corners[0], wall.corners[1]
    return (float(a[0]), float(a[2])), (float(b[0]), float(b[2]))


def _point_to_segment_distance(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, pz = point
    ax, az = a
    bx, bz = b
    vx, vz = bx - ax, bz - az
    length2 = vx * vx + vz * vz
    if length2 < 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * vx + (pz - az) * vz) / length2))
    qx, qz = ax + t * vx, az + t * vz
    return math.hypot(px - qx, pz - qz)


def _projected_overlap_m(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    vx, vz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(vx, vz)
    if length < 1e-9:
        return 0.0
    ux, uz = vx / length, vz / length
    c_proj = (c[0] - a[0]) * ux + (c[1] - a[1]) * uz
    d_proj = (d[0] - a[0]) * ux + (d[1] - a[1]) * uz
    return max(0.0, min(length, max(c_proj, d_proj)) - max(0.0, min(c_proj, d_proj)))


def _is_duplicate_side_wall(candidate, existing_walls: list) -> bool:
    cand_edge = _wall_edge_xz(candidate)
    if cand_edge is None:
        return True
    ca, cb = cand_edge
    cand_len = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
    if cand_len < 1e-6:
        return True

    for existing in existing_walls:
        if getattr(existing, "type", None) != "within_story":
            continue
        existing_edge = _wall_edge_xz(existing)
        if existing_edge is None:
            continue
        ea, eb = existing_edge
        existing_len = math.hypot(eb[0] - ea[0], eb[1] - ea[1])
        if existing_len < 1e-6:
            continue
        dot = (cb[0] - ca[0]) * (eb[0] - ea[0]) + (cb[1] - ca[1]) * (eb[1] - ea[1])
        if abs(dot / (cand_len * existing_len)) < _DUPLICATE_WALL_DIRECTION_DOT_MIN:
            continue
        endpoint_distance = max(
            _point_to_segment_distance(ca, ea, eb),
            _point_to_segment_distance(cb, ea, eb),
        )
        reverse_endpoint_distance = max(
            _point_to_segment_distance(ea, ca, cb),
            _point_to_segment_distance(eb, ca, cb),
        )
        endpoint_min = min(endpoint_distance, reverse_endpoint_distance)
        if endpoint_min > _DUPLICATE_WALL_ENDPOINT_TOL_M:
            continue
        overlap = max(
            _projected_overlap_m(ca, cb, ea, eb),
            _projected_overlap_m(ea, eb, ca, cb),
        )
        overlap_ratio = overlap / max(min(cand_len, existing_len), 1e-9)
        if overlap_ratio >= _DUPLICATE_WALL_OVERLAP_RATIO_MIN:
            return True
    return False


def _story_floor_unions_for_rooms(rooms: list) -> dict[int, Any]:
    from shapely.ops import unary_union

    from reconcile_tiers.extract.gaps import floor_polygon_to_shapely

    by_story: dict[int, list] = {}
    for room in rooms:
        poly = floor_polygon_to_shapely(room.floor_polygon)
        if poly is None or poly.is_empty:
            continue
        by_story.setdefault(room.story, []).append(poly)
    unions: dict[int, Any] = {}
    for story, polys in by_story.items():
        try:
            unions[story] = unary_union(polys)
        except Exception:
            continue
    return unions


def _source_gap_edge_for_wall(candidate, gaps_by_anchor: dict[str, Any]):
    wall_id = str(getattr(candidate, "id", ""))
    parts = wall_id.split(":")
    if len(parts) < 8 or parts[0] != "gw" or parts[-2] != "edge":
        return None
    try:
        edge_idx = int(parts[-1])
    except ValueError:
        return None
    anchor = ":".join(parts[1:5])
    gap = gaps_by_anchor.get(anchor)
    if gap is None:
        return None
    corners = getattr(gap, "corners", None) or []
    edge_vertices = (
        corners[:-1] if len(corners) >= 4 and corners[0] == corners[-1] else corners
    )
    if edge_idx < 0 or edge_idx >= len(edge_vertices):
        return None
    c0 = edge_vertices[edge_idx]
    c1 = edge_vertices[(edge_idx + 1) % len(edge_vertices)]
    try:
        from shapely.geometry import LineString

        return LineString([(float(c0[0]), float(c0[2])), (float(c1[0]), float(c1[2]))])
    except Exception:
        return None


def _is_deformed_from_source_gap_edge(
    candidate, gaps_by_anchor: dict[str, Any]
) -> bool:
    """True when snap-to-wall moved a fallback edge into a new chord.

    Late within-story free-edge recovery is allowed to align scan-noisy vertices
    to nearby wall runs, but it should still represent the same gap edge. If the
    snapped candidate no longer matches its source edge, rendering it creates a
    diagonal wall across the void rather than a wall-thickness closure.
    """
    source_edge = _source_gap_edge_for_wall(candidate, gaps_by_anchor)
    cand_edge = _wall_edge_xz(candidate)
    if source_edge is None or cand_edge is None:
        return False
    try:
        from shapely.geometry import LineString

        candidate_edge = LineString([cand_edge[0], cand_edge[1]])
        deformation = candidate_edge.hausdorff_distance(source_edge)
        scale = max(candidate_edge.length, source_edge.length, 1e-9)
    except Exception:
        return False
    return (
        deformation > _WITHIN_STORY_FREE_EDGE_MAX_SOURCE_DEFORMATION_M
        and deformation / scale > _WITHIN_STORY_FREE_EDGE_MAX_SOURCE_DEFORMATION_RATIO
    )


def _is_unsupported_within_story_free_edge(
    candidate, story_floor_unions: dict[int, Any]
) -> bool:
    """True when the late V1 fallback is about to emit an arbitrary void chord.

    `_close_within_story_free_edges` exists to recover vertical closures at
    missed wall-thickness ends. Edges that sit away from the story floor
    boundary are not that physical case; they are usually chords from a clipped
    void polygon and should not render as walls.
    """
    if getattr(candidate, "type", None) != "within_story":
        return False
    edge = _wall_edge_xz(candidate)
    if edge is None:
        return True
    ca, cb = edge
    edge_len = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
    if edge_len < 1e-6:
        return False
    floor_union = story_floor_unions.get(getattr(candidate, "story", None))
    if floor_union is None or getattr(floor_union, "is_empty", True):
        return False

    try:
        from shapely.geometry import LineString

        line = LineString([ca, cb])
        boundary = floor_union.boundary
        midpoint_distance = line.interpolate(0.5, normalized=True).distance(boundary)
        if midpoint_distance <= _WITHIN_STORY_FREE_EDGE_BOUNDARY_NEAR_M:
            return False
        supported_len = line.intersection(
            boundary.buffer(_WITHIN_STORY_FREE_EDGE_SUPPORT_BUFFER_M)
        ).length
    except Exception:
        return False
    support_ratio = supported_len / max(edge_len, 1e-9)
    return support_ratio < _WITHIN_STORY_FREE_EDGE_MIN_SUPPORT_RATIO


def _close_within_story_free_edges(model: BuildingModel) -> BuildingModel:
    """Append V1-style side walls for within-story gap edges not already closed.

    The slab-pair extractor emits robust floor/ceiling caps, but can miss short
    vertical free-edge closures where the older V1 gap-wall pass correctly
    treated the opening as wall thickness rather than a real void.
    """
    from reconcile_tiers.extract.gaps import (
        _stable_gap_anchor_id,
        compute_gap_walls,
        story_y_map_from_rooms,
    )

    within_story_gaps = [
        gap for gap in model.cross_floor_gaps if gap.type == "within_story"
    ]
    if not within_story_gaps:
        return model
    legacy_walls, _updated_gaps = compute_gap_walls(
        within_story_gaps,
        model.rooms,
        story_y_map_from_rooms(model.rooms),
    )
    existing = list(model.gap_walls)
    story_floor_unions = _story_floor_unions_for_rooms(model.rooms)
    gaps_by_anchor = {_stable_gap_anchor_id(gap): gap for gap in within_story_gaps}
    extras = []
    for wall in legacy_walls:
        if wall.type != "within_story":
            continue
        if _is_deformed_from_source_gap_edge(wall, gaps_by_anchor):
            continue
        if _is_unsupported_within_story_free_edge(wall, story_floor_unions):
            continue
        if _is_duplicate_side_wall(wall, existing + extras):
            continue
        extras.append(wall)
    if not extras:
        return model
    return replace(model, gap_walls=existing + extras)


def _inject_residual_void_walls(
    model: BuildingModel,
    roof,
    candidates: list,
) -> BuildingModel:
    """Detect rooms with no overhead ceiling coverage and append closure
    walls / cross-floor caps to `model`.

    Parity with the V1 ceiling-coverage signal: a room with no candidate
    over its footprint gets synthesised gap walls + draped caps so the
    existing gap rendering path closes the void. The gable-oblique XZ
    union is passed in so void detection ignores footprints already
    covered by a roof oblique above.
    """
    from reconcile_tiers.build_internals.gable_selection import _oblique_xz_polygon
    from reconcile_tiers.extract.gaps import (
        compute_gap_walls,
        compute_room_ceiling_voids,
        story_y_map_from_rooms,
    )

    gable_oblique_xz = None
    if roof.oblique:
        from shapely.ops import unary_union as _oblique_union

        oblique_polys = [
            p
            for ob in roof.oblique
            if (p := _oblique_xz_polygon(ob)) is not None and not p.is_empty
        ]
        if oblique_polys:
            gable_oblique_xz = (
                _oblique_union(oblique_polys)
                if len(oblique_polys) > 1
                else oblique_polys[0]
            )

    extra_gaps = compute_room_ceiling_voids(
        model.rooms,
        [candidate.corners for candidate in candidates],
        gable_oblique_xz=gable_oblique_xz,
    )
    if not extra_gaps:
        return model
    extra_walls, draped_extras = compute_gap_walls(
        extra_gaps,
        model.rooms,
        story_y_map_from_rooms(model.rooms),
    )
    return replace(
        model,
        gap_walls=list(model.gap_walls) + extra_walls,
        cross_floor_gaps=list(model.cross_floor_gaps) + draped_extras,
    )
