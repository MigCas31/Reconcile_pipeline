"""Raw-plane owner promotion: turn raw scan ceiling planes into room-owned
candidates so the painter renders a single owned plane instead of fragmented
RAW_FALLBACK pieces.

Raw scan fragments should not be the visible roof owner when they can be
interpreted as a flat or oblique plane over a room/building part. This emits
owned candidates from the raw plane evidence first; the later raw fallback
then only survives where no assigned flat/oblique owner explains the same
XZ + height.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.assemble.ceiling_painter import CeilingCandidate
from reconcile_tiers.build_internals.ceiling_helpers.kink_detection import (
    _kink_flat_conflicts_with_oblique_roof,
    _kink_flat_is_high_ridge_artifact,
)
from reconcile_tiers.build_internals.ceiling_helpers.slope_ownership import (
    _gable_owns_partial_kink_slope,
    _unsupported_low_flat_room_kink_slope,
)
from reconcile_tiers.build_internals.constants import (
    ROOM_RAW_OWNER_COMPATIBLE_COVERAGE,
    ROOM_RAW_OWNER_FLAT_MAX_INCL_DEG,
    ROOM_RAW_OWNER_MAX_INCL_DEG,
    ROOM_RAW_OWNER_MIN_AREA_M2,
    ROOM_RAW_OWNER_MIN_INCL_DEG,
)
from reconcile_tiers.build_internals.polygon_utils import (
    _polygon_parts_2d,
    _polygon_xz_from_corners,
    _room_floor_xz_polygon,
    _vec3_on_plane_from_polygon,
)
from reconcile_tiers.build_internals.raw_ceiling_filter import _planes_match_over_xz
from reconcile_tiers.build_internals.raw_snapping import (
    _fit_candidate,
    _is_horizontal,
    _raw_ceiling_crosses_upper_floor_slab,
    _snap_raw_ceiling_corners,
    _snap_raw_to_oblique,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingSource
from reconcile_tiers.roof.roof import RoofModel


def _candidate_xz_polygon(candidate: CeilingCandidate):
    return _polygon_xz_from_corners(candidate.corners)


def _compatible_owner_coverage(
    target_poly,
    target_plane: Plane,
    owners: list[CeilingCandidate],
) -> float:
    if target_poly is None or target_poly.is_empty or target_poly.area <= 0.0:
        return 0.0
    from shapely.ops import unary_union

    overlaps = []
    for owner in owners:
        owner_poly = _candidate_xz_polygon(owner)
        if owner_poly is None:
            continue
        try:
            overlap = owner_poly.intersection(target_poly)
        except Exception:
            continue
        if overlap.is_empty or overlap.area <= 1e-9:
            continue
        if _planes_match_over_xz(target_plane, owner.plane, overlap):
            overlaps.append(overlap)
    if not overlaps:
        return 0.0
    try:
        covered = unary_union(overlaps).intersection(target_poly).area
    except Exception:
        covered = 0.0
    return float(covered) / float(target_poly.area)


def _raw_plane_owner_candidates_for_room(
    model: BuildingModel,
    room,
    roof: RoofModel,
    existing_owners: list[CeilingCandidate],
    *,
    wall_axis_math: float | None,
    is_top_gable_room: bool,
    gable_owns_room: bool,
    gable_obliques=None,
) -> list[CeilingCandidate]:
    """Promote plausible raw plane evidence into room-owned computed planes."""
    if room.ceiling_type == "flat" and not room.ceiling_is_kinked:
        return []
    room_poly = _room_floor_xz_polygon(room)
    owners = list(existing_owners)
    flat_ridge_artifact = _kink_flat_is_high_ridge_artifact(
        room
    ) or _kink_flat_conflicts_with_oblique_roof(room, roof)
    from shapely.ops import unary_union

    from reconcile_tiers._core.plane import plane_key

    groups: dict[
        tuple[CeilingSource, tuple[int, int, int, int]],
        tuple[Plane, int, list],
    ] = {}
    for plane_idx, raw in enumerate(room.raw_ceiling_planes):
        corners = raw.corners
        if _raw_ceiling_crosses_upper_floor_slab(room, corners, model):
            continue
        if _gable_owns_partial_kink_slope(
            room,
            corners,
            roof,
            is_top_gable_room=is_top_gable_room,
            gable_owns_room=gable_owns_room,
            gable_obliques=gable_obliques,
        ):
            continue
        is_attic_lid = is_top_gable_room and _is_horizontal(corners)
        if is_attic_lid and gable_owns_room:
            continue
        if is_attic_lid:
            continue
        if is_top_gable_room:
            snapped = _snap_raw_to_oblique(corners, roof.oblique)
            if snapped != corners:
                corners = snapped
            elif wall_axis_math is not None:
                corners = _snap_raw_ceiling_corners(corners, wall_axis_math)
        elif wall_axis_math is not None:
            corners = _snap_raw_ceiling_corners(corners, wall_axis_math)

        plane = Plane.fit(corners)
        if isinstance(plane, FitFailure):
            continue
        from math import atan2, degrees, hypot

        incl = degrees(
            atan2(hypot(float(plane.a), float(plane.c)), abs(float(plane.b)))
        )
        if incl > ROOM_RAW_OWNER_MAX_INCL_DEG:
            continue
        raw_poly = _polygon_xz_from_corners(corners)
        if raw_poly is None:
            continue
        domain = raw_poly
        if room_poly is not None:
            try:
                domain = raw_poly.intersection(room_poly)
            except Exception:
                continue
        parts = [
            part
            for part in _polygon_parts_2d(domain)
            if part.area >= ROOM_RAW_OWNER_MIN_AREA_M2
        ]
        for _part_idx, part in enumerate(parts):
            is_flat_owner = incl < ROOM_RAW_OWNER_FLAT_MAX_INCL_DEG
            if is_flat_owner and flat_ridge_artifact:
                continue
            if not is_flat_owner and incl < ROOM_RAW_OWNER_MIN_INCL_DEG:
                continue
            if (
                _compatible_owner_coverage(part, plane, owners)
                >= ROOM_RAW_OWNER_COMPATIBLE_COVERAGE
            ):
                continue
            owner_corners = _vec3_on_plane_from_polygon(part, plane)
            if len(owner_corners) < 3:
                continue
            if not is_flat_owner and _unsupported_low_flat_room_kink_slope(
                model,
                room,
                owner_corners,
                roof,
                is_top_gable_room=is_top_gable_room,
                gable_owns_room=gable_owns_room,
            ):
                continue
            source = (
                CeilingSource.FLAT_EMIT
                if is_flat_owner
                else CeilingSource.ROOF_ARRANGEMENT
            )
            key = (source, plane_key(plane))
            if key not in groups:
                groups[key] = (plane, plane_idx, [])
            groups[key][2].append(part)

    out: list[CeilingCandidate] = []
    for (source, _key), (plane, first_plane_idx, group_parts) in groups.items():
        try:
            domain = unary_union(group_parts)
        except Exception:
            domain = group_parts[0] if group_parts else None
        for part_idx, part in enumerate(_polygon_parts_2d(domain)):
            if part.area < ROOM_RAW_OWNER_MIN_AREA_M2:
                continue
            if (
                _compatible_owner_coverage(part, plane, owners)
                >= ROOM_RAW_OWNER_COMPATIBLE_COVERAGE
            ):
                continue
            owner_corners = _vec3_on_plane_from_polygon(part, plane)
            if len(owner_corners) < 3:
                continue
            is_flat_owner = source == CeilingSource.FLAT_EMIT
            uuid_part = model.uuid
            ridx = room.index
            fpi = first_plane_idx
            pi = part_idx
            if is_flat_owner and pi == 0:
                locator = f"{uuid_part}::tier-ceiling-flat::{ridx}:raw{fpi}"
            elif is_flat_owner:
                locator = f"{uuid_part}::tier-ceiling-flat::{ridx}:raw{fpi}_{pi}"
            elif pi == 0:
                locator = (
                    f"{uuid_part}::tier-ceiling-roof-arrangement-room::{ridx}:raw{fpi}"
                )
            else:
                locator = (
                    f"{uuid_part}::tier-ceiling-roof-arrangement-room::"
                    f"{ridx}:raw{fpi}_{pi}"
                )
            cand = _fit_candidate(
                owner_corners,
                source,
                locator,
                story=room.story,
            )
            if cand is not None:
                out.append(cand)
                owners.append(cand)
    return out


def _raw_oblique_owner_candidates_for_room(*args, **kwargs) -> list[CeilingCandidate]:
    return _raw_plane_owner_candidates_for_room(*args, **kwargs)
