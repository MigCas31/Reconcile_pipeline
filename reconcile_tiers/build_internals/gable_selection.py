"""Gable-pair selection used by `reconcile_tiers.build`.

Two-stage gate (2D footprint + ridge fallback) that decides which opposing
oblique pair covers a given room, plus the polygon-clipping helpers it
depends on. Re-exported from `reconcile_tiers.build`.
"""

from __future__ import annotations

from reconcile_tiers.build_internals.constants import (
    GABLE_PAIR_MIN_ROOM_OVERLAP_RATIO,
    GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG,
    GABLE_PARTNER_INCL_TOLERANCE_DEG,
    GABLE_RIDGE_Y_EPSILON_M,
    ROOF_WALL_CLIP_EPS_M,
)
from reconcile_tiers.build_internals.polygon_utils import _dedupe_points
from reconcile_tiers.build_internals.wings import (
    _filter_candidates_by_wing,
    _filter_obliques_by_wing,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import RoofModel


def _select_gable_obliques_for_room(
    room,
    model: BuildingModel,
    roof: RoofModel,
    *,
    wing_poly=None,
    min_overlap_ratio: float | None = None,
):
    """Two-stage gate that resolves the opposing gable pair covering a room.

    First tries the 2D footprint overlap test (synthetic-shell anchored, fast). If
    that fails, falls back to the ridge-based 3D test (scan-anchored, catches the
    case where synthesised gable shells are narrower than the room footprint but
    the ridge does pass over it). Used by the FLAT_EMIT skip and wall-clip paths.

    When ``wing_poly`` is given, restricts candidate obliques (and the ridge
    fallback's plane candidates) to those whose XZ extent overlaps that wing,
    so a perpendicular wing's gable cannot be selected for a room sitting in
    another wing of an L/T/U-shaped building.

    Returns ``(selected_obliques, oblique_union_xz)`` matching the existing
    `_select_gable_oblique_pair` shape so callers can swap interchangeably.
    """
    if len(room.floor_polygon) < 3:
        return None
    try:
        from shapely.geometry import Polygon

        from reconcile_tiers._core.shapely2 import make_valid_polygon

        room_poly = make_valid_polygon(
            Polygon([(float(p[0]), float(p[2])) for p in room.floor_polygon])
        )
    except Exception:
        return None
    if room_poly is None or room_poly.is_empty:
        return None
    obliques = (
        _filter_obliques_by_wing(roof.oblique, wing_poly)
        if wing_poly is not None
        else roof.oblique
    )
    if not obliques:
        return None
    selected = _select_gable_oblique_pair(
        obliques, room_poly, min_overlap_ratio=min_overlap_ratio
    )
    if selected is not None:
        return selected
    return _select_gable_obliques_via_ridge(room, model, roof, wing_poly=wing_poly)


def _room_ceiling_y(room, roof: RoofModel) -> float | None:
    y = roof.kinks.attic_lid_y(room.index)
    if y is not None:
        return float(y)
    if len(room.ceiling_polygon) >= 3:
        return sum(float(p[1]) for p in room.ceiling_polygon) / len(
            room.ceiling_polygon
        )
    return None


def _room_eave_y(room, roof: RoofModel) -> float | None:
    y = roof.kinks.eave_y(room.index)
    if y is not None:
        return float(y)
    ys = [float(corner[1]) for wall in room.walls_computed for corner in wall.corners]
    return max(ys) if ys else None


def _select_gable_obliques_via_ridge(
    room,
    model: BuildingModel,
    roof: RoofModel,
    *,
    wing_poly=None,
):
    """Ridge-based fallback for `_select_gable_obliques_for_room`.

    Accepts the room when (a) `roof.kinks.ridge_y` exists, (b) the room's ceiling Y
    sits between its eave Y and the building ridge Y, and (c) the room's XZ
    footprint overlaps the primary gable pair's plan-view extent -- using the same
    `(ridge, slope)` frame and bounds the clipper itself uses
    (`reconcile_tiers.roof.clipping`). All inputs are scan-derived; no plane
    extrapolation is performed.

    When ``wing_poly`` is given, the candidate plane set is filtered to those
    whose XZ extent overlaps that wing -- so the primary gable pair is picked
    from within the room's wing rather than from the largest pair anywhere on
    the building.
    """
    if roof.kinks.ridge_y is None:
        return None
    from reconcile_tiers.roof.clipping import (
        SLOPE_MARGIN_M,
        axes,
        compute_primary_gable_pair,
        project,
    )

    candidates = (
        _filter_candidates_by_wing(roof.planes, wing_poly)
        if wing_poly is not None
        else roof.planes
    )
    pair = compute_primary_gable_pair(candidates)
    if pair is None:
        return None
    ridge_y, cand_a, cand_b = pair

    ceiling_y = _room_ceiling_y(room, roof)
    eave_y = _room_eave_y(room, roof)
    if ceiling_y is None or eave_y is None:
        return None
    if not (
        eave_y - GABLE_RIDGE_Y_EPSILON_M
        <= ceiling_y
        < ridge_y - GABLE_RIDGE_Y_EPSILON_M
    ):
        return None

    bounds = _candidate_pair_bounds(cand_a, cand_b, model)
    if bounds is None:
        return None
    min_r, max_r, min_s, max_s = bounds
    ridge_x, ridge_z, slope_x, slope_z = axes(cand_a)
    ref_x = float(cand_a.cluster.ref_pt[0])
    ref_z = float(cand_a.cluster.ref_pt[2])

    xz_pts = [(float(p[0]), float(p[2])) for p in room.floor_polygon]
    cx = sum(p[0] for p in xz_pts) / len(xz_pts)
    cz = sum(p[1] for p in xz_pts) / len(xz_pts)
    xz_pts.append((cx, cz))

    inside = False
    for px, pz in xz_pts:
        r = project((px, pz), ref_x, ref_z, ridge_x, ridge_z)
        s = project((px, pz), ref_x, ref_z, slope_x, slope_z)
        if (
            min_r <= r <= max_r
            and min_s - SLOPE_MARGIN_M <= s <= max_s + SLOPE_MARGIN_M
        ):
            inside = True
            break
    if not inside:
        return None

    a_obs = [ob for ob in roof.oblique if ob.cluster is cand_a.cluster]
    b_obs = [ob for ob in roof.oblique if ob.cluster is cand_b.cluster]
    if not a_obs or not b_obs:
        return None
    selected_obliques = [a_obs[0], b_obs[0]]

    from shapely.ops import unary_union

    pieces = [
        poly
        for ob in selected_obliques
        if (poly := _oblique_xz_polygon(ob)) is not None
    ]
    union = unary_union(pieces) if pieces else None
    return selected_obliques, union


def _candidate_pair_bounds(
    cand_a, cand_b, model: BuildingModel
) -> tuple[float, float, float, float] | None:
    """Bounds for a primary gable pair in cand_a's ridge/slope frame.

    A top-story room can sit between scan evidence from the two opposing roof
    faces. Using only one candidate's source span rejects those rooms even when
    the paired gable clearly brackets them.
    """
    from reconcile_tiers.roof.clipping import axes, project

    ridge_x, ridge_z, slope_x, slope_z = axes(cand_a)
    ref_x = float(cand_a.cluster.ref_pt[0])
    ref_z = float(cand_a.cluster.ref_pt[2])
    points: list[tuple[float, float]] = []
    for candidate in (cand_a, cand_b):
        for segment in candidate.cluster.segments:
            points.extend(
                [
                    (float(segment.a[0]), float(segment.a[2])),
                    (float(segment.b[0]), float(segment.b[2])),
                ]
            )
            room_idx = segment.room_index
            if room_idx is None or room_idx < 0 or room_idx >= len(model.rooms):
                continue
            points.extend(
                (float(point[0]), float(point[2]))
                for point in model.rooms[room_idx].floor_polygon
            )
    if not points:
        # Original code referenced an unbound name `candidate_bounds`, which
        # would raise NameError if reached. Preserve the (defective) behavior
        # by returning None - tests baselined against the prior code never
        # exercised this path successfully.
        return None
    r_vals = [project(point, ref_x, ref_z, ridge_x, ridge_z) for point in points]
    s_vals = [project(point, ref_x, ref_z, slope_x, slope_z) for point in points]
    return min(r_vals), max(r_vals), min(s_vals), max(s_vals)


def _clip_polygon_below_roof_plane(
    corners: list[list[float]], roof_plane
) -> list[list[float]]:
    """Clip a 3D polygon to the solid below one roof plane."""
    if len(corners) < 3:
        return []

    a, b, c, d = (
        float(roof_plane.a),
        float(roof_plane.b),
        float(roof_plane.c),
        float(roof_plane.d),
    )

    def signed(point: list[float]) -> float:
        return a * point[0] + b * point[1] + c * point[2] - d

    def intersect(
        p0: list[float], p1: list[float], s0: float, s1: float
    ) -> list[float]:
        denom = s0 - s1
        if abs(denom) <= 1e-12:
            return list(p0)
        t = max(0.0, min(1.0, s0 / denom))
        return [p0[idx] + t * (p1[idx] - p0[idx]) for idx in range(3)]

    out: list[list[float]] = []
    for idx, cur in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        s_cur = signed(cur)
        s_next = signed(nxt)
        cur_inside = s_cur <= ROOF_WALL_CLIP_EPS_M
        next_inside = s_next <= ROOF_WALL_CLIP_EPS_M
        if cur_inside and next_inside:
            out.append(list(nxt))
        elif cur_inside and not next_inside:
            out.append(intersect(cur, nxt, s_cur, s_next))
        elif not cur_inside and next_inside:
            out.append(intersect(cur, nxt, s_cur, s_next))
            out.append(list(nxt))

    return _dedupe_points(out)


def _is_gable_partner_pair(surface, other) -> bool:
    from reconcile_tiers.roof.geometry import angle_diff_deg

    diff = angle_diff_deg(surface.cluster.avg_azimuth, other.cluster.avg_azimuth)
    if (
        not (180.0 - GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG)
        <= diff
        <= (180.0 + GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG)
    ):
        return False
    if (
        abs(surface.cluster.avg_incl - other.cluster.avg_incl)
        > GABLE_PARTNER_INCL_TOLERANCE_DEG
    ):
        return False
    return True


def _has_gable_partner(surface, all_obliques) -> bool:
    """True if the surface has an opposing oblique with similar pitch (a gable pair)."""
    for other in all_obliques:
        if other is surface:
            continue
        if _is_gable_partner_pair(surface, other):
            return True
    return False


def _oblique_xz_polygon(surface):
    if len(surface.corners) < 3:
        return None
    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon

    poly = make_valid_polygon(
        Polygon([(float(p[0]), float(p[2])) for p in surface.corners])
    )
    if poly is None or poly.is_empty:
        return None
    return poly


def _select_gable_oblique_pair(
    obliques, room_poly, *, min_overlap_ratio: float | None = None
):
    from shapely.ops import unary_union

    if room_poly is None or room_poly.is_empty or room_poly.area <= 0.0:
        return None
    threshold = (
        GABLE_PAIR_MIN_ROOM_OVERLAP_RATIO
        if min_overlap_ratio is None
        else min_overlap_ratio
    )

    indexed = [
        (oblique, poly)
        for oblique in obliques
        if (poly := _oblique_xz_polygon(oblique)) is not None
    ]
    best: tuple[float, float, object, object, object] | None = None
    for a_pos, (a_oblique, a_poly) in enumerate(indexed):
        for b_oblique, b_poly in indexed[a_pos + 1 :]:
            if not _is_gable_partner_pair(a_oblique, b_oblique):
                continue
            pair_union = unary_union([a_poly, b_poly])
            overlap = pair_union.intersection(room_poly).area
            overlap_ratio = overlap / room_poly.area
            if overlap_ratio < threshold:
                continue
            score = (overlap, pair_union.area)
            if best is None or score > (best[0], best[1]):
                best = (overlap, pair_union.area, a_oblique, b_oblique, pair_union)
    if best is None:
        return None
    return [best[2], best[3]], best[4]
