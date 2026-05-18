"""Kink-flat artifact detection.

Distinguishes a real horizontal patch in a kinked room from a high-ridge
plateau between two oblique faces (which would be wrongly emitted as a
flat ceiling). Also computes the per-room kink-Y (`attic_lid_y`) used to
split candidates above vs. below the eave/lid.
"""

from __future__ import annotations

from reconcile_tiers.build_internals.ceiling_helpers._misc import (
    _raw_plane_xz_area,
    _room_oblique_raw_coverage,
)
from reconcile_tiers.build_internals.constants import (
    KINK_FLAT_RIDGE_ARTIFACT_MAX_FLAT_FLOOR_RATIO,
    KINK_FLAT_RIDGE_ARTIFACT_MAX_SELECTED_SLOPE_RATIO,
    KINK_FLAT_RIDGE_ARTIFACT_MIN_RAW_OBLIQUE_RATIO,
    KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_M2,
    KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_RATIO,
    KINK_FLAT_RIDGE_ARTIFACT_ROOF_Y_CLEARANCE_M,
    KINK_FLAT_RIDGE_ARTIFACT_WALLTOP_YSPAN_M,
    KINK_FLAT_RIDGE_ARTIFACT_Y_TOL_M,
    ROOM_RAW_OWNER_MIN_AREA_M2,
)
from reconcile_tiers.build_internals.polygon_utils import (
    _corners_xz_polygon,
    _polygon_parts_2d,
    _xz_area,
)
from reconcile_tiers.config import architectural_priors_enabled
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import RoofModel


def _gable_building_kink_y(model: BuildingModel, roof: RoofModel) -> float | None:
    upper_story = max((r.story for r in model.rooms), default=0)
    upper_room_indices = {r.index for r in model.rooms if r.story == upper_story}
    candidates = [
        y
        for idx, y in roof.kinks.attic_lid_y_by_room.items()
        if idx in upper_room_indices
    ]
    return min(candidates) if candidates else None


def _room_kink_y(room, roof: RoofModel) -> float | None:
    """Y where the room's vertical envelope ends and the attic begins.

    For most rooms this is `attic_lid_y_by_room` (a p10-of-wall-tops
    heuristic). For vaulted rooms -- where the ceiling literally reaches
    the ridge -- that heuristic underestimates the lid because it picks
    up small intermediate wall fragments. Under
    architectural priors, detect vaulted rooms by checking max
    wall-top y against the heuristic lid: when the gap exceeds ~10 cm,
    the room is open above the heuristic and the actual lid sits at
    the wall-top max. Without this, primitive-synthesised gable slopes
    (which span eave->ridge by construction) get split at the heuristic
    lid in `_ceiling_candidates`, the above-lid portion is dropped as
    attic, and the rendered slope is truncated mid-gable.

    `ARCHITECTURAL_PRIORS=0` retains the prior behavior (heuristic only) -- the
    cohort goldens are baselined against it.
    """
    lid = roof.kinks.attic_lid_y(room.index)
    if lid is None:
        return None
    if not architectural_priors_enabled():
        return lid
    walls = room.walls_computed or room.walls_merged
    wall_top_max = max(
        (max(float(c[1]) for c in w.corners) for w in walls if len(w.corners) >= 3),
        default=None,
    )
    if wall_top_max is not None and wall_top_max > lid + 0.10:
        return float(wall_top_max)
    return lid


def _kink_flat_is_high_ridge_artifact(room) -> bool:
    """True when a kinked room's horizontal raw patch is the high ridge
    plateau between oblique faces, not a physically flat roof/ceiling segment.
    """
    if not room.ceiling_is_kinked or len(room.ceiling_flat_polygon) < 3:
        return False
    valid_slope_polygons = [
        slope_polygon
        for slope_polygon in (getattr(room, "ceiling_slope_polygons", None) or [])
        if len(slope_polygon) >= 3
    ]
    slope_area = sum(_xz_area(slope_polygon) for slope_polygon in valid_slope_polygons)
    if slope_area < ROOM_RAW_OWNER_MIN_AREA_M2:
        return False
    from reconcile_tiers.extract.ceilings import build_ceiling_from_wall_tops

    wall_top = build_ceiling_from_wall_tops(room)
    if wall_top is None or len(wall_top) < 3:
        return False
    wall_ys = [float(point[1]) for point in wall_top]
    if max(wall_ys) - min(wall_ys) < KINK_FLAT_RIDGE_ARTIFACT_WALLTOP_YSPAN_M:
        return False
    flat_ys = [float(point[1]) for point in room.ceiling_flat_polygon]
    flat_y = sum(flat_ys) / len(flat_ys)
    if flat_y < max(wall_ys) - KINK_FLAT_RIDGE_ARTIFACT_Y_TOL_M:
        return False
    if room.ceiling_type is None:
        _oblique_area, oblique_ratio = _room_oblique_raw_coverage(room)
        return (
            len(valid_slope_polygons) >= 2
            and oblique_ratio >= KINK_FLAT_RIDGE_ARTIFACT_MIN_RAW_OBLIQUE_RATIO
        )
    if room.ceiling_type != "sloped":
        return False
    floor_area = _raw_plane_xz_area(room.floor_polygon)
    if floor_area <= 1e-6:
        return False
    _oblique_area, oblique_ratio = _room_oblique_raw_coverage(room)
    flat_ratio = _xz_area(room.ceiling_flat_polygon) / floor_area
    selected_slope_ratio = slope_area / floor_area
    return (
        flat_ratio <= KINK_FLAT_RIDGE_ARTIFACT_MAX_FLAT_FLOOR_RATIO
        and selected_slope_ratio <= KINK_FLAT_RIDGE_ARTIFACT_MAX_SELECTED_SLOPE_RATIO
        and oblique_ratio >= KINK_FLAT_RIDGE_ARTIFACT_MIN_RAW_OBLIQUE_RATIO
    )


def _kink_flat_conflicts_with_oblique_roof(room, roof: RoofModel) -> bool:
    """True when a kink flat patch sits above the oblique roof covering it.

    A real flat ceiling under a gable should be below the roof plane above the
    same XZ footprint. If the reconstructed oblique roof is materially lower
    than the horizontal kink patch, the flat patch is a RoomPlan ridge plateau
    artifact and should not become a tier ceiling.
    """
    if not room.ceiling_is_kinked or len(room.ceiling_flat_polygon) < 3:
        return False
    flat_poly = _corners_xz_polygon(room.ceiling_flat_polygon)
    if flat_poly is None:
        return False
    flat_area = float(flat_poly.area)
    if flat_area <= 1e-6:
        return False
    floor_area = _raw_plane_xz_area(room.floor_polygon)
    if floor_area <= 1e-6:
        return False
    flat_ratio = flat_area / floor_area
    if flat_ratio > KINK_FLAT_RIDGE_ARTIFACT_MAX_FLAT_FLOOR_RATIO:
        return False
    _oblique_area, oblique_ratio = _room_oblique_raw_coverage(room)
    if oblique_ratio < KINK_FLAT_RIDGE_ARTIFACT_MIN_RAW_OBLIQUE_RATIO:
        return False
    flat_y = sum(float(point[1]) for point in room.ceiling_flat_polygon) / len(
        room.ceiling_flat_polygon
    )

    conflicting_area = 0.0
    for surface in roof.oblique:
        if surface.dominant_story != room.story:
            continue
        roof_poly = _corners_xz_polygon(surface.corners)
        if roof_poly is None:
            continue
        overlap = flat_poly.intersection(roof_poly)
        if overlap.is_empty:
            continue
        for part in _polygon_parts_2d(overlap):
            if part.area < KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_M2:
                continue
            sample = part.representative_point()
            roof_y = surface.plane.y_at(float(sample.x), float(sample.y))
            if roof_y is None:
                continue
            if flat_y - float(roof_y) >= KINK_FLAT_RIDGE_ARTIFACT_ROOF_Y_CLEARANCE_M:
                conflicting_area += float(part.area)

    return (
        conflicting_area >= KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_M2
        and conflicting_area / flat_area
        >= KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_RATIO
    )
