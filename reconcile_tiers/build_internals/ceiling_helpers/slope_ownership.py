"""Slope-ownership and gable-claim gates.

Decides which entity (gable, computed-oblique arrangement, or per-room raw
plane) owns a given kink slope, and detects unsupported / artifact slopes
that should be removed before late synthesis.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.build_internals.ceiling_helpers._misc import (
    _flat_lid_y,
    _hybrid_room_lid_y,
    _lower_plane_half,
    _parse_room_idx,
)
from reconcile_tiers.build_internals.constants import (
    _CAVITY_CLOSURE_OVERLAP_RATIO_MIN,
    _DIP_TEST_EPS_Y,
    _DIP_TEST_GRID_M,
    GABLE_OWNED_PARTIAL_SLOPE_MAX_ROOM_COVERAGE,
    GABLE_OWNED_SLOPE_MIN_SHELL_COVERAGE,
    RAW_CEILING_REDUNDANT_COVERAGE_RATIO,
    ROOM_OBLIQUE_FLAT_LID_TOL_M,
    UNSUPPORTED_KINK_FLAT_LID_MIN_COVERAGE_RATIO,
    UNSUPPORTED_KINK_FLAT_NEIGHBOR_LID_TOL_M,
    UNSUPPORTED_KINK_LOCAL_NEIGHBOR_MIN_OVERLAP_M2,
    UNSUPPORTED_KINK_LOCAL_NEIGHBOR_TOL_M,
    UNSUPPORTED_KINK_SHALLOW_SLOPE_MAX_INCL_DEG,
    UNSUPPORTED_KINK_SLOPE_MIN_DROP_M,
)
from reconcile_tiers.build_internals.polygon_utils import (
    _corners_xz_polygon,
    _polygon_parts_2d,
    _room_floor_xz_polygon,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import RoofModel


def _computed_oblique_arrangement_owns_kink_slope(
    corners: list[list[float]],
    room,
    roof: RoofModel,
    *,
    min_coverage_ratio: float = RAW_CEILING_REDUNDANT_COVERAGE_RATIO,
    min_covering_surfaces: int = 1,
) -> bool:
    """Priors-on duplicate guard for observed kink slopes.

    Default builds keep scan-observed kink slopes directly. Under architectural
    priors, suppress the observed slope only when same-room computed oblique
    cells cover it and those cells can survive the late eave/lid clip.
    """
    eave_y = roof.kinks.eave_y(room.index)
    lid_y = roof.kinks.attic_lid_y(room.index)
    if (
        eave_y is not None
        and lid_y is not None
        and (lid_y - eave_y) < ROOM_OBLIQUE_FLAT_LID_TOL_M
    ):
        return False
    if len(corners) < 3:
        return False

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from reconcile_tiers._core.shapely2 import make_valid

    try:
        target = make_valid(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    except Exception:
        return False
    if target is None or target.is_empty or not target.is_valid or target.area <= 1e-9:
        return False

    computed_polys = []
    for surface in roof.oblique_split or roof.oblique:
        if (
            _parse_room_idx(surface.arrangement_cell_id) != room.index
            or len(surface.corners) < 3
        ):
            continue
        try:
            poly = make_valid(
                Polygon([(float(p[0]), float(p[2])) for p in surface.corners])
            )
        except Exception:
            continue
        if (
            poly is not None
            and not poly.is_empty
            and poly.is_valid
            and poly.area > 1e-9
        ):
            computed_polys.append(poly)
    if not computed_polys:
        return False

    overlaps = [
        float(poly.intersection(target).area)
        for poly in computed_polys
        if float(poly.intersection(target).area) / float(target.area) >= 0.10
    ]
    if len(overlaps) < min_covering_surfaces:
        return False
    computed_union = (
        unary_union(computed_polys) if len(computed_polys) > 1 else computed_polys[0]
    )
    return (
        float(computed_union.intersection(target).area) / float(target.area)
        >= min_coverage_ratio
    )


def _room_has_roof_detail_support(room, roof: RoofModel) -> bool:
    if room.index in roof.simple_slant_room_indices:
        return True
    if any(candidate.room_index == room.index for candidate in roof.dormer_candidates):
        return True
    for surface in roof.thermal:
        if surface.room_index != room.index:
            continue
        kind = getattr(surface.kind, "value", str(surface.kind))
        if kind in {"knee", "dormer_front", "dormer_cheek", "dormer_header"}:
            return True
    return False


def _local_rooms_near_slope(model: BuildingModel, room, slope_poly) -> list:
    nearby = []
    slope_neighborhood = slope_poly.buffer(UNSUPPORTED_KINK_LOCAL_NEIGHBOR_TOL_M)
    for other in model.rooms:
        if other.index == room.index or other.story != room.story:
            continue
        other_poly = _room_floor_xz_polygon(other)
        if other_poly is None or other_poly.is_empty:
            continue
        if slope_poly.distance(other_poly) > UNSUPPORTED_KINK_LOCAL_NEIGHBOR_TOL_M:
            continue
        if (
            slope_neighborhood.intersection(other_poly).area
            < UNSUPPORTED_KINK_LOCAL_NEIGHBOR_MIN_OVERLAP_M2
        ):
            continue
        nearby.append(other)
    return nearby


def _roof_oblique_supports_kink_slope(room, slope_poly, roof: RoofModel) -> bool:
    for surface in roof.oblique_split:
        if _parse_room_idx(surface.arrangement_cell_id) != room.index:
            continue
        surface_poly = _corners_xz_polygon(surface.corners)
        if surface_poly is None:
            continue
        overlap_ratio = float(surface_poly.intersection(slope_poly).area) / max(
            float(slope_poly.area), 1e-9
        )
        if overlap_ratio >= _CAVITY_CLOSURE_OVERLAP_RATIO_MIN:
            return True
    return False


def _gable_oblique_shell_coverage(slope_poly, gable_obliques) -> float:
    if slope_poly is None or slope_poly.is_empty or slope_poly.area <= 1e-9:
        return 0.0
    if not gable_obliques:
        return 0.0
    from shapely.ops import unary_union

    polys = []
    for oblique in gable_obliques:
        poly = _corners_xz_polygon(oblique.corners)
        if poly is not None and not poly.is_empty and poly.area > 1e-9:
            polys.append(poly)
    if not polys:
        return 0.0
    try:
        union = unary_union(polys) if len(polys) > 1 else polys[0]
        return float(union.intersection(slope_poly).area) / float(slope_poly.area)
    except Exception:
        return 0.0


def _gable_owns_partial_kink_slope(
    room,
    slope_polygon: list[list[float]],
    roof: RoofModel,
    *,
    is_top_gable_room: bool,
    gable_owns_room: bool,
    gable_obliques,
    total_slope_area: float | None = None,
) -> bool:
    if not is_top_gable_room or not gable_owns_room:
        return False
    room_poly = _room_floor_xz_polygon(room)
    slope_poly = _corners_xz_polygon(slope_polygon)
    if (
        room_poly is None
        or slope_poly is None
        or room_poly.area <= 1e-9
        or slope_poly.area <= 1e-9
    ):
        return False
    shell_coverage = _gable_oblique_shell_coverage(slope_poly, gable_obliques)
    if shell_coverage < GABLE_OWNED_SLOPE_MIN_SHELL_COVERAGE:
        return False
    slope_area = (
        total_slope_area if total_slope_area is not None else float(slope_poly.area)
    )
    room_coverage = float(slope_area) / float(room_poly.area)
    return room_coverage < GABLE_OWNED_PARTIAL_SLOPE_MAX_ROOM_COVERAGE


def _unsupported_low_flat_room_kink_slope(
    model: BuildingModel,
    room,
    slope_polygon: list[list[float]],
    roof: RoofModel,
    *,
    is_top_gable_room: bool,
    gable_owns_room: bool,
) -> bool:
    if room.ceiling_type not in {"flat", None} or is_top_gable_room or gable_owns_room:
        return False
    if _room_has_roof_detail_support(room, roof):
        return False
    room_poly = _room_floor_xz_polygon(room)
    flat_poly = _corners_xz_polygon(room.ceiling_flat_polygon)
    slope_poly = _corners_xz_polygon(slope_polygon)
    if (
        room_poly is None
        or flat_poly is None
        or slope_poly is None
        or room_poly.area <= 1e-9
        or slope_poly.area <= 1e-9
    ):
        return False
    if _roof_oblique_supports_kink_slope(room, slope_poly, roof):
        return False
    flat_coverage = float(flat_poly.intersection(room_poly).area) / float(
        room_poly.area
    )
    if flat_coverage < UNSUPPORTED_KINK_FLAT_LID_MIN_COVERAGE_RATIO:
        return False
    room_lid_y = _flat_lid_y(room)
    if room_lid_y is None:
        return False

    for neighbor in _local_rooms_near_slope(model, room, slope_poly):
        neighbor_lid_y = _flat_lid_y(neighbor)
        if neighbor_lid_y is None:
            return False
        if abs(neighbor_lid_y - room_lid_y) > UNSUPPORTED_KINK_FLAT_NEIGHBOR_LID_TOL_M:
            return False

    from math import atan2, degrees, hypot

    plane = Plane.fit(slope_polygon)
    if isinstance(plane, FitFailure):
        return False
    incl = degrees(atan2(hypot(float(plane.a), float(plane.c)), abs(float(plane.b))))
    slope_min_y = min(float(point[1]) for point in slope_polygon)
    return (
        room.ceiling_type is None
        or (room_lid_y - slope_min_y) >= UNSUPPORTED_KINK_SLOPE_MIN_DROP_M
        or incl <= UNSUPPORTED_KINK_SHALLOW_SLOPE_MAX_INCL_DEG
    )


def _flat_lower_room_below_closed_gable(
    room,
    roof: RoofModel,
    *,
    is_gable: bool,
    is_top_gable_room: bool,
    gable_owns_room: bool,
) -> bool:
    """True when a global gable sits above a lower flat room, not in it."""
    if (
        not is_gable
        or is_top_gable_room
        or not gable_owns_room
        or room.ceiling_type != "flat"
    ):
        return False
    eave_y = roof.kinks.eave_y(room.index)
    lid_y = roof.kinks.attic_lid_y(room.index)
    if eave_y is None or lid_y is None:
        return False
    return (lid_y - eave_y) < ROOM_OBLIQUE_FLAT_LID_TOL_M


def _gable_planes_dip_into_room(
    room,
    selected_obliques,
    *,
    eps_y: float = _DIP_TEST_EPS_Y,
) -> bool:
    """True iff a gable plane's Y dips below the room's eave Y inside the room.

    For each oblique in the pair, partition the room footprint via
    `_lower_plane_half` (the same partition the hybrid emission uses) and
    sample the plane Y on a regular grid inside that side's polygon. If any
    sample sits more than `eps_y` below the room's eave Y, the slope physically
    enters the room's interior airspace and the room is hybrid (partly flat,
    partly oblique) -- even when `ceiling_type` classifies it as flat or when
    the oblique's *detected* xz footprint is small.
    """
    if len(selected_obliques) != 2:
        return False
    room_poly = _room_floor_xz_polygon(room)
    if room_poly is None or room_poly.is_empty:
        return False
    eave_y = _hybrid_room_lid_y(room)
    if eave_y is None:
        return False

    from shapely.geometry import Point

    threshold = eave_y - eps_y
    for oblique, other in (
        (selected_obliques[0], selected_obliques[1]),
        (selected_obliques[1], selected_obliques[0]),
    ):
        plane = oblique.plane
        if abs(float(plane.b)) < 1e-6:
            continue
        side = _lower_plane_half(room_poly, plane, other.plane)
        if side is None or side.is_empty:
            continue
        for part in _polygon_parts_2d(side):
            if part.area < 0.05:
                continue
            min_x, min_z, max_x, max_z = part.bounds
            nx = max(2, int((max_x - min_x) / _DIP_TEST_GRID_M) + 1)
            nz = max(2, int((max_z - min_z) / _DIP_TEST_GRID_M) + 1)
            for i in range(nx):
                x = min_x + (max_x - min_x) * (i + 0.5) / nx
                for j in range(nz):
                    z = min_z + (max_z - min_z) * (j + 0.5) / nz
                    if not part.contains(Point(x, z)):
                        continue
                    y = plane.y_at(x, z)
                    if y is not None and y < threshold:
                        return True
    return False
