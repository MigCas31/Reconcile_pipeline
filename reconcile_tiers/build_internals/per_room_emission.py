"""Per-room ceiling-candidate emission pipeline.

Decomposes the per-room loop in `reconcile_tiers.build._ceiling_candidates`
into seven sequential phase functions sharing two read-only context
dataclasses (`_BuildContext`, `_RoomContext`) and one mutable
in-progress accumulator (`_EmissionState`).

The phases run in this order; downstream phases read guard flags
(`gable_room_emitted`, `kinked_emitted`, `skip_remaining_phases`) set
by upstream phases. The order is load-bearing — see the module-level
plan in `.claude/plans/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reconcile_tiers._core.shapely2 import split_polygon_at_y as _split_polygon_at_y
from reconcile_tiers.assemble.ceiling_painter import CeilingCandidate
from reconcile_tiers.build_internals.ceiling_helpers import (
    _kink_flat_conflicts_with_oblique_roof,
    _kink_flat_is_high_ridge_artifact,
    _lower_plane_half,
    _parse_room_idx,
    _room_kink_y,
    _synthesised_flat_candidate_for_room,
    _vec3_at_y_from_polygon,
)
from reconcile_tiers.build_internals.constants import (
    _HYBRID_DOMAIN_MIN_AREA_M2,
    ATTIC_SHELL_STORY,
    KINK_MIN_SPLIT_HEIGHT_M,
    SYNTHETIC_GABLE_OBSERVED_Y_TOL_M,
)
from reconcile_tiers.build_internals.gable_selection import (
    _has_gable_partner,
    _select_gable_obliques_for_room,
)
from reconcile_tiers.build_internals.polygon_utils import (
    _polygon_parts_2d,
    _room_floor_xz_polygon,
    _xz_area,
)
from reconcile_tiers.build_internals.raw_snapping import _fit_candidate
from reconcile_tiers.build_internals.wings import (
    _compute_wings,
    _wing_polygon_for_room,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingSource
from reconcile_tiers.roof.roof import RoofModel


@dataclass(frozen=True)
class _BuildContext:
    """Per-build inputs, computed once before the room loop."""

    model: BuildingModel
    roof: RoofModel
    is_gable: bool
    gable_kink_y: float | None
    top_story: int
    wings: list
    wall_axis_math: float | None
    synthesis_flat_xz: object | None  # shapely geometry or None


@dataclass(frozen=True)
class _RoomContext:
    """Read-only per-room derivations, built once at the top of each
    iteration of the room loop. Captures every piece of state that the
    seven phase functions read.
    """

    build: _BuildContext
    room: object
    wing_poly: object | None
    is_top_gable_room: bool
    gable_selection: tuple | None
    gable_owns_room: bool
    gable_obliques: list
    hybrid_selection: tuple | None
    active_selection: tuple | None
    flat_below_closed_gable: bool
    kink_slope_polygons: list
    valid_kink_slope_count: int
    valid_kink_slope_area: float
    flat_ridge_artifact: bool


@dataclass
class _EmissionState:
    """In-progress emission for one room. Phase functions append to
    `candidates` (final output) and `room_owners` (read by Phase 6 to
    skip raws explained by an existing owner). Guard flags are set by
    early phases and read by later phases."""

    candidates: list[CeilingCandidate] = field(default_factory=list)
    room_owners: list[CeilingCandidate] = field(default_factory=list)
    gable_room_emitted: bool = False
    kinked_emitted: bool = False
    skip_remaining_phases: bool = False


def _build_room_context(build: _BuildContext, room) -> _RoomContext:
    """Compute every per-room field the phase functions consume.

    Mirrors the original derivations at the top of the room loop in
    `_ceiling_candidates`. Pure: no side effects, no candidate
    emission.
    """
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _flat_lower_room_below_closed_gable,
    )

    wing_poly = _wing_polygon_for_room(room, build.wings) if build.wings else None
    is_top_gable_room = build.is_gable and room.story == build.top_story
    gable_selection = (
        _select_gable_obliques_for_room(
            room, build.model, build.roof, wing_poly=wing_poly
        )
        if build.is_gable
        else None
    )
    gable_owns_room = gable_selection is not None
    gable_obliques = gable_selection[0] if gable_selection is not None else []
    flat_below_closed_gable = _flat_lower_room_below_closed_gable(
        room,
        build.roof,
        is_gable=build.is_gable,
        is_top_gable_room=is_top_gable_room,
        gable_owns_room=gable_owns_room,
    )
    hybrid_selection = (
        _select_gable_obliques_for_room(
            room,
            build.model,
            build.roof,
            wing_poly=wing_poly,
            min_overlap_ratio=0.10,
        )
        if (
            is_top_gable_room
            and build.is_gable
            and gable_selection is None
            and room.ceiling_type != "sloped"
        )
        else None
    )
    active_selection = (
        gable_selection if gable_selection is not None else hybrid_selection
    )
    kink_slope_polygons = (
        room.ceiling_slope_polygons
        if getattr(room, "ceiling_slope_polygons", None)
        else [room.ceiling_slope_polygon]
    )
    valid_kink_slope_count = sum(
        1 for slope_polygon in kink_slope_polygons if len(slope_polygon) >= 3
    )
    valid_kink_slope_area = sum(
        _xz_area(slope_polygon)
        for slope_polygon in kink_slope_polygons
        if len(slope_polygon) >= 3
    )
    flat_ridge_artifact = _kink_flat_is_high_ridge_artifact(
        room
    ) or _kink_flat_conflicts_with_oblique_roof(room, build.roof)

    return _RoomContext(
        build=build,
        room=room,
        wing_poly=wing_poly,
        is_top_gable_room=is_top_gable_room,
        gable_selection=gable_selection,
        gable_owns_room=gable_owns_room,
        gable_obliques=gable_obliques,
        hybrid_selection=hybrid_selection,
        active_selection=active_selection,
        flat_below_closed_gable=flat_below_closed_gable,
        kink_slope_polygons=kink_slope_polygons,
        valid_kink_slope_count=valid_kink_slope_count,
        valid_kink_slope_area=valid_kink_slope_area,
        flat_ridge_artifact=flat_ridge_artifact,
    )


def _compute_hybrid_domain(
    ctx: _RoomContext, selected_obliques: list, hybrid_lid_y: float | None
):
    """For a kinked hybrid room, return the XZ subdomain that the gable
    arrangement should cover (room footprint minus scan-surfaced slopes
    that the kinked branch will actually emit).

    Returns ``(domain, fire_gable)``. ``fire_gable=False`` means scan
    coverage is already complete and the gable phase should not fire.
    Pure (non-kinked) hybrid rooms get ``(None, True)`` — `_room_gable_candidates`
    will use the full room polygon.
    """
    if hybrid_lid_y is None:
        return None, True
    room = ctx.room
    if not getattr(room, "ceiling_is_kinked", False):
        return None, True

    candidate_scan_polys = [
        sp
        for sp in (
            room.ceiling_slope_polygons
            if getattr(room, "ceiling_slope_polygons", None)
            else [getattr(room, "ceiling_slope_polygon", [])]
        )
        if len(sp) >= 3
    ]
    surfaced_polys = []
    for sp in candidate_scan_polys:
        if _gable_owns_partial_kink_slope_for(
            ctx, sp, selected_obliques, candidate_scan_polys
        ):
            continue
        if _unsupported_low_flat_for(ctx, sp):
            continue
        poly = _corners_xz_polygon_for_kink(sp)
        if poly is not None:
            surfaced_polys.append(poly)
    if not surfaced_polys:
        return None, True

    from shapely.ops import unary_union

    room_poly_xz = _room_floor_xz_polygon_for_ctx(ctx)
    if room_poly_xz is None:
        return None, True
    scan_union = unary_union(surfaced_polys)
    hybrid_domain = room_poly_xz.difference(scan_union)
    if hybrid_domain.is_empty or hybrid_domain.area < _HYBRID_DOMAIN_MIN_AREA_M2:
        return None, False
    return hybrid_domain, True


def _gable_owns_partial_kink_slope_for(
    ctx: _RoomContext, sp, selected_obliques: list, candidate_scan_polys: list
) -> bool:
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _gable_owns_partial_kink_slope,
    )
    from reconcile_tiers.build_internals.polygon_utils import _xz_area

    return _gable_owns_partial_kink_slope(
        ctx.room,
        sp,
        ctx.build.roof,
        is_top_gable_room=ctx.is_top_gable_room,
        gable_owns_room=ctx.gable_owns_room,
        gable_obliques=selected_obliques,
        total_slope_area=sum(_xz_area(s) for s in candidate_scan_polys),
    )


def _unsupported_low_flat_for(ctx: _RoomContext, sp) -> bool:
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _unsupported_low_flat_room_kink_slope,
    )

    return _unsupported_low_flat_room_kink_slope(
        ctx.build.model,
        ctx.room,
        sp,
        ctx.build.roof,
        is_top_gable_room=ctx.is_top_gable_room,
        gable_owns_room=ctx.gable_owns_room,
    )


def _corners_xz_polygon_for_kink(sp):
    from reconcile_tiers.build_internals.polygon_utils import _corners_xz_polygon

    return _corners_xz_polygon(sp)


def _room_floor_xz_polygon_for_ctx(ctx: _RoomContext):
    from reconcile_tiers.build_internals.polygon_utils import _room_floor_xz_polygon

    return _room_floor_xz_polygon(ctx.room)


def _emit_top_story_gable_or_hybrid(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 2: top-story gable / hybrid emission.

    Three sub-cases on top-story rooms with an active gable selection:
    - Pure-sloped: full-room oblique split.
    - Hybrid (flat/None classified, gable plane dips into volume):
      emit oblique under the lid + flat lid above.
    - Slope sits above the room: do nothing here, let Phase 4 handle.

    Sets ``state.gable_room_emitted`` when candidates were appended.
    """
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _gable_planes_dip_into_room,
        _hybrid_room_lid_y,
    )

    if not (ctx.is_top_gable_room and ctx.active_selection is not None):
        return
    selected_obliques, _oblique_union = ctx.active_selection
    room = ctx.room
    if room.ceiling_type == "sloped":
        hybrid_lid_y: float | None = None
        fire_gable = True
    elif _gable_planes_dip_into_room(room, selected_obliques):
        hybrid_lid_y = _hybrid_room_lid_y(room)
        fire_gable = hybrid_lid_y is not None
    else:
        hybrid_lid_y = None
        fire_gable = False
    if not fire_gable:
        return
    hybrid_domain, fire_gable = _compute_hybrid_domain(
        ctx, selected_obliques, hybrid_lid_y
    )
    if not fire_gable:
        return
    gable_candidates = _room_gable_candidates(
        room,
        ctx.build.model,
        selected_obliques,
        domain=hybrid_domain,
        wing_poly=ctx.wing_poly,
        room_ceiling_locators=True,
        flat_lid_y=hybrid_lid_y,
    )
    if gable_candidates:
        state.candidates.extend(gable_candidates)
        state.room_owners.extend(gable_candidates)
        state.gable_room_emitted = True


def _emit_kinked_flat_lid_and_synth(ctx: _RoomContext, state: _EmissionState) -> None:
    """Sub-phase 3a: emit the kinked-room flat lid + synth-mirror obliques.

    Skipped when the room's flat patch is a high-ridge artifact
    (`ctx.flat_ridge_artifact`). Otherwise: clip the flat XZ against
    unmirrored gable ridges via `_split_flat_for_unmirrored_obliques`,
    emit synthesised opposing-slope mirrors (unless a real gable
    selection exists for this room — in which case the gable shell
    will cover that slope), then emit the (possibly clipped) flat lid
    pieces. Skips the lid entirely when Phase 2's hybrid gable already
    emitted a `:hybrid-lid` covering the same XZ.
    """
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _split_flat_for_unmirrored_obliques,
        _vec3_at_y_from_polygon,
    )
    from reconcile_tiers.build_internals.polygon_utils import (
        _corners_xz_polygon,
        _polygon_parts_2d,
    )
    from reconcile_tiers.build_internals.raw_snapping import _fit_candidate
    from reconcile_tiers.payload.schema import CeilingSource

    if ctx.flat_ridge_artifact:
        return

    model = ctx.build.model
    roof = ctx.build.roof
    room = ctx.room

    flat_locator = f"{model.uuid}::tier-ceiling-flat::{room.index}"
    synth_prefix = (
        f"{model.uuid}::tier-ceiling-roof-arrangement-room::{room.index}:synth-mirror"
    )
    kink_flat_xz = _corners_xz_polygon(room.ceiling_flat_polygon)
    flat_y_kink = sum(float(p[1]) for p in room.ceiling_flat_polygon) / len(
        room.ceiling_flat_polygon
    )
    # Subtract scan-derived slope polygons from synth area: where the scan
    # already saw a slope, no need to mirror -- the kink path emits those
    # directly below.
    slope_polys = [
        _corners_xz_polygon(sp) for sp in ctx.kink_slope_polygons if len(sp) >= 3
    ]
    kink_domain = kink_flat_xz
    synth_pieces: list = []
    # Skip mirror synth when a gable selection exists for this room: the
    # gable arrangement (`_room_gable_candidates` or the visual-shell
    # layer) is responsible for the slopes; a room-level synth would
    # double-emit (regression test `test_gable_kink_keeps_flat_lid`).
    gable_will_cover = ctx.gable_selection is not None
    if kink_flat_xz is not None:
        kink_domain, synth_pieces = _split_flat_for_unmirrored_obliques(
            kink_flat_xz,
            flat_y_kink,
            roof,
            room.story,
            extra_subtract_polys=[p for p in slope_polys if p is not None],
        )
    if not gable_will_cover:
        for synth_idx, (synth_corners, _plane) in enumerate(synth_pieces):
            synth_cand = _fit_candidate(
                synth_corners,
                CeilingSource.ROOF_ARRANGEMENT,
                f"{synth_prefix}:{synth_idx}",
                story=room.story,
            )
            if synth_cand is not None:
                state.candidates.append(synth_cand)
                state.room_owners.append(synth_cand)
    emitted_flat = False
    # When the hybrid gable path already emitted a flat lid that subtracts
    # the oblique xz, skip the kinked-branch lid -- its un-subtracted
    # kink_flat_xz at FLAT_EMIT priority would shadow the new oblique
    # pieces.
    skip_kink_flat = state.gable_room_emitted
    if (
        not skip_kink_flat
        and kink_domain is not None
        and not getattr(kink_domain, "is_empty", False)
    ):
        for part_idx, part in enumerate(_polygon_parts_2d(kink_domain)):
            corners = _vec3_at_y_from_polygon(part, flat_y_kink)
            flat_cand = _fit_candidate(
                corners,
                CeilingSource.FLAT_EMIT,
                flat_locator if part_idx == 0 else f"{flat_locator}:covered:{part_idx}",
                story=room.story,
            )
            if flat_cand is not None:
                state.candidates.append(flat_cand)
                state.room_owners.append(flat_cand)
                emitted_flat = True
    if not skip_kink_flat and not emitted_flat and not synth_pieces:
        flat_cand = _fit_candidate(
            room.ceiling_flat_polygon,
            CeilingSource.FLAT_EMIT,
            flat_locator,
            story=room.story,
        )
        if flat_cand is not None:
            state.candidates.append(flat_cand)
            state.room_owners.append(flat_cand)


def _emit_kinked_scan_slopes(ctx: _RoomContext, state: _EmissionState) -> None:
    """Sub-phase 3b: emit each scan-observed kink slope as ROOF_ARRANGEMENT.

    Per slope, applies four drop conditions in order:
    1. Gable owns this partial slope (`_gable_owns_partial_kink_slope`).
    2. Hybrid gable already emitted AND this scan slope's plane normal
       doesn't match any selected gable oblique (= rogue noise plane).
    3. Unsupported low-flat room kink slope.
    4. Under priors, sloped-classified room with multiple slopes where
       computed obliques already own this one.
    Surviving slopes are fit and appended.
    """
    from math import atan2, degrees, hypot

    from reconcile_tiers._core.plane import FitFailure, Plane
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _computed_oblique_arrangement_owns_kink_slope,
        _gable_owns_partial_kink_slope,
        _unsupported_low_flat_room_kink_slope,
    )
    from reconcile_tiers.build_internals.constants import (
        GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG,
        GABLE_PARTNER_INCL_TOLERANCE_DEG,
    )
    from reconcile_tiers.build_internals.raw_snapping import _fit_candidate
    from reconcile_tiers.config import architectural_priors_enabled
    from reconcile_tiers.payload.schema import CeilingSource
    from reconcile_tiers.roof.geometry import angle_diff_deg

    model = ctx.build.model
    roof = ctx.build.roof
    room = ctx.room
    base_locator = f"{model.uuid}::tier-ceiling-roof-arrangement-room::{room.index}"
    for slope_idx, slope_polygon in enumerate(ctx.kink_slope_polygons):
        if len(slope_polygon) < 3:
            continue
        if _gable_owns_partial_kink_slope(
            room,
            slope_polygon,
            roof,
            is_top_gable_room=ctx.is_top_gable_room,
            gable_owns_room=ctx.gable_owns_room,
            gable_obliques=ctx.gable_obliques,
            total_slope_area=ctx.valid_kink_slope_area,
        ):
            continue
        # Suppress rogue scan slopes when the hybrid gable path already
        # emitted a gable-aligned slope for this room AND the scan slope's
        # own plane normal does NOT match any of the selected gable
        # obliques. Real opposing scan slopes (matching gable partners)
        # keep firing.
        gable_set = (
            ctx.gable_obliques
            if ctx.gable_obliques
            else (ctx.active_selection[0] if ctx.active_selection is not None else [])
        )
        if state.gable_room_emitted and gable_set:
            sp_plane = Plane.fit(slope_polygon)
            if not isinstance(sp_plane, FitFailure):
                sp_az = degrees(atan2(-sp_plane.a, -sp_plane.c)) % 360.0
                sp_incl = degrees(atan2(hypot(sp_plane.a, sp_plane.c), abs(sp_plane.b)))
                any_match = any(
                    angle_diff_deg(sp_az, ob.cluster.avg_azimuth)
                    <= GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG
                    and abs(sp_incl - ob.cluster.avg_incl)
                    <= GABLE_PARTNER_INCL_TOLERANCE_DEG
                    for ob in gable_set
                )
                if not any_match:
                    continue
        if _unsupported_low_flat_room_kink_slope(
            model,
            room,
            slope_polygon,
            roof,
            is_top_gable_room=ctx.is_top_gable_room,
            gable_owns_room=ctx.gable_owns_room,
        ):
            continue
        if (
            architectural_priors_enabled()
            and room.ceiling_type == "sloped"
            and ctx.valid_kink_slope_count > 1
            and _computed_oblique_arrangement_owns_kink_slope(
                slope_polygon,
                room,
                roof,
                min_covering_surfaces=1,
            )
        ):
            continue
        slope_cand = _fit_candidate(
            slope_polygon,
            CeilingSource.ROOF_ARRANGEMENT,
            base_locator if slope_idx == 0 else f"{base_locator}:{slope_idx}",
            story=room.story,
        )
        if slope_cand is not None:
            state.candidates.append(slope_cand)
            state.room_owners.append(slope_cand)


def _emit_kinked_room(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 3: kinked room (knee-wall) emission.

    Fires when the scan saw both a flat ceiling patch and a slanted
    patch in the same room. Emits two-part candidates so the painter
    keeps both, instead of letting the wall-derived single plane,
    flat lid, or gable arrangement blanket the kink. The scan
    evidence is more specific than the top-story gable ownership rule.
    Sets `state.kinked_emitted` so Phases 5 and downstream act on it.
    """
    from reconcile_tiers.build_internals.constants import (
        TOP_GABLE_KINK_MIN_SLOPE_AREA_M2,
    )

    room = ctx.room
    if not (
        room.ceiling_is_kinked
        and not ctx.flat_below_closed_gable
        and len(room.ceiling_flat_polygon) >= 3
        and ctx.valid_kink_slope_count > 0
        and (
            not ctx.is_top_gable_room
            or ctx.valid_kink_slope_area >= TOP_GABLE_KINK_MIN_SLOPE_AREA_M2
        )
    ):
        return
    _emit_kinked_flat_lid_and_synth(ctx, state)
    _emit_kinked_scan_slopes(ctx, state)
    state.kinked_emitted = True


def _emit_pure_flat_room(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 4: emit the full-room flat ceiling lid for flat rooms.

    Skipped if Phase 2 (gable) or Phase 3 (kinked) already emitted for
    this room, or if the room isn't flat-classified, or if the room
    has no usable ceiling polygon. Otherwise emits via
    `_flat_room_ceiling_candidates` (which internally splits the
    floor footprint against unmirrored oblique ridges) and signals
    `state.skip_remaining_phases` so Phases 5-7 are skipped — the
    legacy code's `continue`.

    Note: the `flat_emit_skip` branch (gated `False`) is unreachable
    today; preserved verbatim for archaeology.
    """
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _flat_ceiling_candidates_for_domain,
        _flat_room_ceiling_candidates,
        _upper_floor_coverage_xz,
    )
    from reconcile_tiers.build_internals.constants import (
        SPLIT_LEVEL_EXPOSED_MIN_AREA_M2,
    )
    from reconcile_tiers.build_internals.polygon_utils import _polygon_parts_2d
    from reconcile_tiers.build_internals.raw_snapping import _fit_candidate
    from reconcile_tiers.payload.schema import CeilingSource

    model = ctx.build.model
    roof = ctx.build.roof
    room = ctx.room

    flat_emit_skip = False
    if (
        state.kinked_emitted
        or state.gable_room_emitted
        or room.ceiling_type != "flat"
        or len(room.ceiling_polygon) < 3
    ):
        return
    if flat_emit_skip and ctx.gable_selection is not None:
        # Dead code (flat_emit_skip is hardcoded False); preserved verbatim
        # for archaeology. Will not execute today.
        from reconcile_tiers.build_internals.ceiling_helpers import (
            _room_gable_candidates,
        )

        selected_obliques, _oblique_union = ctx.gable_selection
        room_poly, upper_coverage = _upper_floor_coverage_xz(room, model)
        if room_poly is None:
            state.skip_remaining_phases = True
            return
        exposed = (
            room_poly
            if upper_coverage is None
            else room_poly.difference(upper_coverage)
        )
        exposed_area = sum(part.area for part in _polygon_parts_2d(exposed))
        if exposed_area <= SPLIT_LEVEL_EXPOSED_MIN_AREA_M2:
            candidate = _fit_candidate(
                room.ceiling_polygon,
                CeilingSource.FLAT_EMIT,
                f"{model.uuid}::tier-ceiling-flat::{room.index}",
                story=room.story,
            )
            if candidate is not None:
                state.candidates.append(candidate)
            state.skip_remaining_phases = True
            return
        if upper_coverage is not None:
            state.candidates.extend(
                _flat_ceiling_candidates_for_domain(
                    room,
                    upper_coverage,
                    f"{model.uuid}::tier-ceiling-flat::{room.index}",
                )
            )
        state.candidates.extend(
            _room_gable_candidates(
                room, model, selected_obliques, exposed, wing_poly=ctx.wing_poly
            )
        )
        state.skip_remaining_phases = True
        return
    for candidate in _flat_room_ceiling_candidates(
        room,
        f"{model.uuid}::tier-ceiling-flat::{room.index}",
        roof=roof,
        synth_locator_prefix=(
            f"{model.uuid}::tier-ceiling-roof-arrangement-room::"
            f"{room.index}:synth-mirror"
        ),
    ):
        state.candidates.append(candidate)
        state.room_owners.append(candidate)
    state.skip_remaining_phases = True


def _emit_raw_plane_owners(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 6: promote plausible raw planes to ROOF_ARRANGEMENT/FLAT_EMIT
    owners.

    Skipped when the top-story gable already emitted for this room
    (`state.gable_room_emitted`). Otherwise delegates to
    `_raw_plane_owner_candidates_for_room`, which considers existing
    `state.room_owners` to avoid promoting raws already explained.
    """
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _raw_plane_owner_candidates_for_room,
    )

    if state.gable_room_emitted:
        return
    raw_owner_candidates = _raw_plane_owner_candidates_for_room(
        ctx.build.model,
        ctx.room,
        ctx.build.roof,
        state.room_owners,
        wall_axis_math=ctx.build.wall_axis_math,
        is_top_gable_room=ctx.is_top_gable_room,
        gable_owns_room=ctx.gable_owns_room,
        gable_obliques=ctx.gable_obliques,
    )
    if raw_owner_candidates:
        state.candidates.extend(raw_owner_candidates)
        state.room_owners.extend(raw_owner_candidates)


def _emit_raw_fallback(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 7: per-raw-plane RAW_FALLBACK / ATTIC_FLAT_LID emission.

    For each scan-derived raw ceiling plane on the room: drop if it
    crosses an upper-floor slab; classify as ATTIC_FLAT_LID if the room
    is top-gable AND the plane is horizontal (else RAW_FALLBACK); apply
    snapping (gable-oblique snap for top-gable rooms; axis-prior snap
    when priors are on); fit and emit. Drops a horizontal raw if it's
    a kink-flat ridge artifact (`flat_ridge_artifact` and not
    attic-lid). Suppresses attic-lids when the gable owns the room.
    """
    from reconcile_tiers.assemble.synthesis import _extend_flat_to_obliques
    from reconcile_tiers.build_internals.raw_snapping import (
        _fit_candidate,
        _is_horizontal,
        _raw_ceiling_crosses_upper_floor_slab,
        _snap_raw_ceiling_corners,
        _snap_raw_to_oblique,
    )
    from reconcile_tiers.payload.schema import CeilingSource

    model = ctx.build.model
    roof = ctx.build.roof
    room = ctx.room
    for plane_idx, raw in enumerate(room.raw_ceiling_planes):
        corners = raw.corners
        if _raw_ceiling_crosses_upper_floor_slab(room, corners, model):
            continue
        is_attic_lid = ctx.is_top_gable_room and _is_horizontal(corners)
        if not is_attic_lid and _is_horizontal(corners) and ctx.flat_ridge_artifact:
            continue
        if is_attic_lid and ctx.gable_owns_room:
            continue
        if is_attic_lid:
            corners = _extend_flat_to_obliques(
                corners, roof.oblique, room.floor_polygon
            )
        elif ctx.is_top_gable_room:
            # Keep the scan-derived raw plane at RAW_FALLBACK priority (40).
            # Synthesised gable shells paint on top via ROOF_ARRANGEMENT (80);
            # the raw plane only shows through where the shells don't extend
            # (dormer flanks, building extensions, etc.). Suppressing the raw
            # plane here would lose dormer / extension geometry whenever the
            # synthetic shell footprint disagrees with the scan.
            corners = _snap_raw_to_oblique(corners, roof.oblique)
        if ctx.build.wall_axis_math is not None and not is_attic_lid:
            corners = _snap_raw_ceiling_corners(corners, ctx.build.wall_axis_math)
        candidate = _fit_candidate(
            corners,
            CeilingSource.ATTIC_FLAT_LID
            if is_attic_lid
            else CeilingSource.RAW_FALLBACK,
            f"{model.uuid}::tier-ceiling-raw::{room.index}:{plane_idx}",
            story=room.story,
        )
        if candidate is not None:
            state.candidates.append(candidate)


def _emit_sloped_non_top_story(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 5: emit the wall-derived sloped ceiling polygon for sloped
    non-top-story rooms.

    Without this, only RAW_FALLBACK runs for these rooms, fragmenting
    into noisy sub-pieces. Under priors, emit the wall-derived polygon
    as ROOF_ARRANGEMENT (priority 80, V2-mapped to COMPUTED_OBLIQUE);
    the painter clips RAW_FALLBACK underneath. Top-story gable rooms
    keep their existing path (gable arrangement owns those).
    """
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _ceiling_polygon_is_planar,
    )
    from reconcile_tiers.build_internals.raw_snapping import _fit_candidate
    from reconcile_tiers.payload.schema import CeilingSource

    room = ctx.room
    if state.kinked_emitted:
        return
    if ctx.build.wall_axis_math is None:
        return
    if room.ceiling_type != "sloped":
        return
    if len(room.ceiling_polygon) < 3:
        return
    if ctx.is_top_gable_room:
        return
    if not _ceiling_polygon_is_planar(room.ceiling_polygon):
        return
    candidate = _fit_candidate(
        room.ceiling_polygon,
        CeilingSource.ROOF_ARRANGEMENT,
        f"{ctx.build.model.uuid}::tier-ceiling-roof-arrangement-room::{room.index}",
        story=room.story,
    )
    if candidate is not None:
        state.candidates.append(candidate)
        state.room_owners.append(candidate)


def _emit_synthesised_flat(ctx: _RoomContext, state: _EmissionState) -> None:
    """Phase 1: synthesised FLAT_EMIT for ceiling-less rooms.

    Only fires under priors when the room has no clean ceiling source
    today (`ceiling_type is None`) and the building's roof.flat union
    covers the room footprint. For sloped rooms a wall-derived sloped
    polygon is computed elsewhere; for flat rooms the existing FLAT_EMIT
    path runs in Phase 4.
    """
    if ctx.build.synthesis_flat_xz is None or ctx.room.ceiling_type is not None:
        return
    for cand in _synthesised_flat_candidate_for_room(
        ctx.build.model,
        ctx.room,
        ctx.build.synthesis_flat_xz,
        roof=ctx.build.roof,
    ):
        state.candidates.append(cand)
        state.room_owners.append(cand)


# ---------------------------------------------------------------------------
# Helpers called by the phase functions but large enough to live separately.
# ---------------------------------------------------------------------------


def _room_gable_candidates(
    room,
    model: BuildingModel,
    selected_obliques,
    domain=None,
    wing_poly=None,
    *,
    room_ceiling_locators: bool = False,
    flat_lid_y: float | None = None,
) -> list[CeilingCandidate]:
    """Emit gable oblique pieces for a room.

    When `flat_lid_y` is set, the room is treated as hybrid: each oblique side
    is clipped at `flat_lid_y` and only the slope portion below the lid is
    emitted as oblique. The complementary (x,z) area inside `room_poly` -- the
    region where every oblique plane sits above the lid -- is emitted as a
    single FLAT_EMIT lid at `flat_lid_y`. This is what lets a kneewall room
    classified `ceiling_type="flat"` still receive its slope side.
    """
    if len(selected_obliques) != 2 or len(room.floor_polygon) < 3:
        return []
    room_poly = domain if domain is not None else _room_floor_xz_polygon(room)
    if room_poly is None or room_poly.is_empty:
        return []

    observed_y = [
        float(point[1])
        for model_room in model.rooms
        for seq in [model_room.floor_polygon, model_room.ceiling_polygon]
        for point in seq
    ]
    observed_y.extend(
        float(point[1])
        for model_room in model.rooms
        for wall in model_room.walls_computed
        for point in wall.corners
    )
    observed_y.extend(
        float(point[1])
        for model_room in model.rooms
        for raw in model_room.raw_ceiling_planes
        for point in raw.corners
    )
    # cap_y bounds the synthetic slope tip. Legacy bound is
    # `max(observed_y) + 0.6m` (global scan vertex max), which clips slopes
    # below the detected ridge in 58% of the corpus when scan didn't capture
    # the upper walls. For *hybrid* rooms (flat_lid_y is set) we lift the cap
    # to the detected ridge corner Y + 10 cm so the synthesised slope can
    # actually reach the lid where it should meet the flat ceiling. Pure
    # sloped rooms keep the legacy cap to avoid downstream regressions in
    # `_clip_piece_to_eave` and residual-void closure.
    observed_cap_y = (
        max(observed_y) + SYNTHETIC_GABLE_OBSERVED_Y_TOL_M - 1e-6
        if observed_y
        else None
    )
    if flat_lid_y is not None:
        ridge_corner_max_y = max(
            (float(c[1]) for ob in selected_obliques for c in ob.corners),
            default=None,
        )
        if ridge_corner_max_y is not None:
            ridge_cap_y = ridge_corner_max_y + 0.10
            cap_y = (
                ridge_cap_y
                if observed_cap_y is None
                else max(observed_cap_y, ridge_cap_y)
            )
        else:
            cap_y = observed_cap_y
    else:
        cap_y = observed_cap_y

    from shapely.geometry import Polygon as _ShPolygon
    from shapely.ops import unary_union as _unary_union

    candidates: list[CeilingCandidate] = []
    oblique_below_lid_xz: list = []  # for residual flat-lid computation

    # Floor of the room — used as the lower bound for emitted oblique corners
    # so the slope cannot extrapolate downward through the floor when the
    # room polygon is wider than the actual physical roof segment over it.
    # Without this clip, steep obliques over wide rooms produce 5–8 m
    # ceiling_yspan_excessive defects (127/446 buildings before the fix).
    floor_y_max = max((float(p[1]) for p in room.floor_polygon), default=None)

    for side_idx, (oblique, other) in enumerate(
        (
            (selected_obliques[0], selected_obliques[1]),
            (selected_obliques[1], selected_obliques[0]),
        )
    ):
        geom = _lower_plane_half(room_poly, oblique.plane, other.plane, wing_poly)
        for part_idx, part in enumerate(_polygon_parts_2d(geom)):
            corners: list[list[float]] = []
            for x, z in list(part.exterior.coords)[:-1]:
                y = oblique.plane.y_at(float(x), float(z))
                if y is None:
                    corners = []
                    break
                corners.append([float(x), float(y), float(z)])
            if len(corners) < 3:
                continue
            # Lower clip at floor_y. Keep the `above`-floor part; discard
            # corners that fall below. A 0.05 m slack matches the
            # ceiling_below_floor audit rule's tolerance so legitimate
            # near-floor obliques aren't over-clipped.
            if floor_y_max is not None:
                _below, corners = _split_polygon_at_y(corners, floor_y_max - 0.05)
                if len(corners) < 3:
                    continue
            if flat_lid_y is not None:
                corners, _above = _split_polygon_at_y(corners, flat_lid_y)
                if len(corners) < 3:
                    continue
            if cap_y is not None:
                corners, _above = _split_polygon_at_y(corners, cap_y)
                if len(corners) < 3:
                    continue
            if flat_lid_y is not None:
                xz_poly = _ShPolygon([(float(p[0]), float(p[2])) for p in corners])
                if xz_poly.is_valid and xz_poly.area > 1e-6:
                    oblique_below_lid_xz.append(xz_poly)
            if room_ceiling_locators:
                tag = (
                    f"hybrid{side_idx}_{part_idx}"
                    if flat_lid_y is not None
                    else f"gable{side_idx}_{part_idx}"
                )
                locator_id = (
                    f"{model.uuid}::tier-ceiling-roof-arrangement-room::{room.index}:"
                    f"{tag}"
                )
            else:
                locator_id = (
                    f"{model.uuid}::tier-ceiling-roof-arrangement-room::{room.index}:"
                    f"{side_idx}:{part_idx}"
                )
            candidate = _fit_candidate(
                corners,
                CeilingSource.ROOF_ARRANGEMENT,
                locator_id,
                story=room.story,
            )
            if candidate is not None:
                candidates.append(candidate)

    if flat_lid_y is not None:
        if oblique_below_lid_xz:
            below_union = _unary_union(oblique_below_lid_xz)
            residual = room_poly.difference(below_union)
        else:
            residual = room_poly
        flat_locator = f"{model.uuid}::tier-ceiling-flat::{room.index}:hybrid-lid"
        for part_idx, part in enumerate(_polygon_parts_2d(residual)):
            lid_corners = _vec3_at_y_from_polygon(part, flat_lid_y)
            lid_locator = (
                flat_locator if part_idx == 0 else f"{flat_locator}:{part_idx}"
            )
            lid_cand = _fit_candidate(
                lid_corners,
                CeilingSource.FLAT_EMIT,
                lid_locator,
                story=room.story,
            )
            if lid_cand is not None:
                candidates.append(lid_cand)
    return candidates


def _emit_arrangement_and_attic_candidates(
    model: BuildingModel,
    roof: RoofModel,
    is_gable: bool,
    top_story: int,
    rooms_by_index: dict,
    gable_kink_y: float | None,
) -> list[CeilingCandidate]:
    """Emit roof-arrangement and attic-shell ceiling candidates.

    Three sub-passes share the wing-clipping helpers:
    1. Per-cell `roof.oblique_split` surfaces, kink-Y-split into below
       (ROOF_ARRANGEMENT) and above (ROOF_ARRANGEMENT_ATTIC, only when
       not gable) parts. Each part is wing-clipped before emission.
    2. Fallback: if `oblique_split` is empty, emit `roof.oblique`
       directly as ROOF_ARRANGEMENT.
    3. Gable attic shells: for each gable-paired oblique surface, emit
       its full footprint as ROOF_ARRANGEMENT_ATTIC under the special
       `ATTIC_SHELL_STORY` bucket so it isn't subtracted by interior
       solids or lids.
    """
    from shapely.geometry import Polygon as _ShPoly

    from reconcile_tiers._core.shapely2 import make_valid_polygon as _make_valid
    from reconcile_tiers._core.wing_decomposition import (
        clip_3d_polygon_to_wing,
        wing_polygon_for_xz_polygon,
    )

    out: list[CeilingCandidate] = []
    building_wings = _compute_wings(model)

    def _wing_for_corners(corners) -> object | None:
        if not building_wings or len(building_wings) == 1 or len(corners) < 3:
            return building_wings[0].polygon if building_wings else None
        xz_poly = _make_valid(_ShPoly([(float(p[0]), float(p[2])) for p in corners]))
        if xz_poly is None or xz_poly.is_empty:
            return None
        return wing_polygon_for_xz_polygon(xz_poly, building_wings)

    def _emit_wing_clipped(
        corners, plane, source, locator_fn, story, arrangement_cell_id=None
    ):
        wing_poly = _wing_for_corners(corners)
        clipped_parts = clip_3d_polygon_to_wing(corners, plane, wing_poly)
        if not clipped_parts:
            return
        for part_idx, part in enumerate(clipped_parts):
            cand = _fit_candidate(
                part,
                source,
                locator_fn(part_idx, len(clipped_parts)),
                arrangement_cell_id=arrangement_cell_id,
                story=story,
            )
            if cand is not None:
                out.append(cand)

    for idx, surface in enumerate(roof.oblique_split):
        room_idx = _parse_room_idx(surface.arrangement_cell_id)
        room = rooms_by_index.get(room_idx)
        if room is not None and room.story != surface.dominant_story:
            continue
        if is_gable and room is not None and room.story != top_story:
            continue
        kink_y = _room_kink_y(room, roof) if room is not None else None
        if kink_y is None and is_gable:
            kink_y = gable_kink_y
        ys = [p[1] for p in surface.corners]
        if kink_y is None:
            below_corners, above_corners = surface.corners, []
        elif min(ys) >= kink_y - KINK_MIN_SPLIT_HEIGHT_M:
            below_corners, above_corners = [], list(surface.corners)
        elif max(ys) <= kink_y + KINK_MIN_SPLIT_HEIGHT_M:
            below_corners, above_corners = list(surface.corners), []
        else:
            below_corners, above_corners = _split_polygon_at_y(surface.corners, kink_y)
        if len(below_corners) >= 3:
            _emit_wing_clipped(
                below_corners,
                surface.plane,
                CeilingSource.ROOF_ARRANGEMENT,
                lambda part_idx, n_parts, idx=idx, uuid=model.uuid: (
                    f"{uuid}::tier-ceiling-roof-arrangement::{idx}"
                    if n_parts == 1
                    else f"{uuid}::tier-ceiling-roof-arrangement::{idx}:{part_idx}"
                ),
                surface.dominant_story,
                arrangement_cell_id=surface.arrangement_cell_id,
            )
        if len(above_corners) >= 3 and not is_gable:
            _emit_wing_clipped(
                above_corners,
                surface.plane,
                CeilingSource.ROOF_ARRANGEMENT_ATTIC,
                lambda part_idx, n_parts, idx=idx, uuid=model.uuid: (
                    f"{uuid}::tier-ceiling-roof-arrangement-attic::{idx}"
                    if n_parts == 1
                    else (
                        f"{uuid}::tier-ceiling-roof-arrangement-attic::{idx}:{part_idx}"
                    )
                ),
                surface.dominant_story,
                arrangement_cell_id=surface.arrangement_cell_id,
            )
    if not roof.oblique_split:
        for fallback_idx, surface in enumerate(roof.oblique):
            idx = surface.source_index if surface.source_index >= 0 else fallback_idx
            _emit_wing_clipped(
                surface.corners,
                surface.plane,
                CeilingSource.ROOF_ARRANGEMENT,
                lambda part_idx, n_parts, idx=idx, uuid=model.uuid: (
                    f"{uuid}::tier-ceiling-roof-arrangement::{idx}"
                    if n_parts == 1
                    else f"{uuid}::tier-ceiling-roof-arrangement::{idx}:{part_idx}"
                ),
                surface.dominant_story,
            )

    if is_gable:
        for fallback_idx, surface in enumerate(roof.oblique):
            if len(surface.corners) < 3:
                continue
            if not _has_gable_partner(surface, roof.oblique):
                continue
            obl_idx = (
                surface.source_index if surface.source_index >= 0 else fallback_idx
            )
            _emit_wing_clipped(
                surface.corners,
                surface.plane,
                CeilingSource.ROOF_ARRANGEMENT_ATTIC,
                lambda part_idx, n_parts, obl_idx=obl_idx, uuid=model.uuid: (
                    f"{uuid}::tier-ceiling-roof-arrangement-attic-full::{obl_idx}"
                    if n_parts == 1
                    else (
                        f"{uuid}::tier-ceiling-roof-arrangement-attic-full::"
                        f"{obl_idx}:{part_idx}"
                    )
                ),
                ATTIC_SHELL_STORY,
            )

    return out


def _wall_axis_for_priors(model: BuildingModel) -> float | None:
    """Return the building principal axis (math deg) if architectural priors
    are enabled and wall coverage is high enough to trust it. Otherwise
    `None` -- callers leave their geometry alone.
    """
    from reconcile_tiers._core.wall_axis import axis_from_building_model
    from reconcile_tiers.build_internals.constants import _PRIORS_COVERAGE_MIN
    from reconcile_tiers.config import architectural_priors_enabled

    if not architectural_priors_enabled():
        return None
    info = axis_from_building_model(model)
    if info is None or info[1] < _PRIORS_COVERAGE_MIN:
        return None
    return info[0]


def _ceiling_candidates(
    model: BuildingModel, roof: RoofModel
) -> list[CeilingCandidate]:
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _gable_building_kink_y,
        _synthesis_flat_xz_union,
    )

    candidates: list[CeilingCandidate] = []
    rooms_by_index = {room.index: room for room in model.rooms}
    is_gable = roof.kinks.ridge_y is not None
    gable_kink_y = _gable_building_kink_y(model, roof) if is_gable else None
    top_story = max((r.story for r in model.rooms), default=0)
    wings = _compute_wings(model) if is_gable else []
    wall_axis_math = _wall_axis_for_priors(model)
    # When priors are on, build a synthesis flat-XZ union once. We use it to
    # emit per-room FLAT_EMIT candidates for rooms whose floor is covered by
    # roof.flat (Pattern A from .context/plans/synthesis-owns-primitive.md).
    # Without this, rooms with ceiling_type=None and no synthesised oblique
    # fall through with only RAW_FALLBACK pieces -- producing the user-visible
    # noise on flat-roof buildings like 494a97c4 and d39db17d.
    synthesis_flat_xz = (
        _synthesis_flat_xz_union(roof) if wall_axis_math is not None else None
    )
    build_ctx = _BuildContext(
        model=model,
        roof=roof,
        is_gable=is_gable,
        gable_kink_y=gable_kink_y,
        top_story=top_story,
        wings=wings,
        wall_axis_math=wall_axis_math,
        synthesis_flat_xz=synthesis_flat_xz,
    )
    for room in model.rooms:
        room_ctx = _build_room_context(build_ctx, room)
        state = _EmissionState()
        _emit_synthesised_flat(room_ctx, state)
        _emit_top_story_gable_or_hybrid(room_ctx, state)
        _emit_kinked_room(room_ctx, state)
        _emit_pure_flat_room(room_ctx, state)
        if not state.skip_remaining_phases:
            _emit_sloped_non_top_story(room_ctx, state)
            _emit_raw_plane_owners(room_ctx, state)
            _emit_raw_fallback(room_ctx, state)
        candidates.extend(state.candidates)

    candidates.extend(
        _emit_arrangement_and_attic_candidates(
            model, roof, is_gable, top_story, rooms_by_index, gable_kink_y
        )
    )

    return candidates
