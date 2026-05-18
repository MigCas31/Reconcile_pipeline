"""Tests for the ridge-based gable-coverage gate in `reconcile_tiers.build`.

The gate decides whether a top-story room sits under the building's primary gable.
A first 2D footprint test (synthesised gable shells must overlap the room ≥
GABLE_PAIR_MIN_ROOM_OVERLAP_RATIO of its floor) is followed by a ridge-based 3D
fallback that catches the case where the synthesised shells are narrower than the
room footprint.
"""

from __future__ import annotations

from dataclasses import replace
from math import atan, degrees

import pytest
from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.build_internals.gable_selection import (
    _select_gable_obliques_for_room,
    _select_gable_obliques_via_ridge,
)
from reconcile_tiers.build_internals.per_room_emission import _ceiling_candidates
from reconcile_tiers.build_internals.wings import (
    _filter_candidates_by_wing,
    _filter_obliques_by_wing,
)
from reconcile_tiers.extract.building import BuildingModel, ExtractedRoom
from reconcile_tiers.roof.clipping import (
    compute_gable_ridge_y,
    compute_primary_gable_pair,
)
from reconcile_tiers.roof.roof import (
    ObliqueSurface,
    RoofCluster,
    RoofKinks,
    RoofModel,
    RoofPlaneCandidate,
    RoofSegment,
)


def _gable_planes(ridge_y: float, eave_y: float, slope_half_extent_z: float):
    """Two opposing gable planes meeting along the line z=0 at y=ridge_y.

    Plane A: positive-z slope, faces +z. Plane B: negative-z slope, faces -z.
    The slope drops from ridge_y at z=0 to eave_y at z=±slope_half_extent_z.
    """
    slope = (ridge_y - eave_y) / slope_half_extent_z
    plane_a = Plane(a=0.0, b=1.0, c=slope, d=ridge_y)
    plane_b = Plane(a=0.0, b=1.0, c=-slope, d=ridge_y)
    return plane_a, plane_b, slope


def _make_gable_roof(
    *,
    ridge_y: float = 4.0,
    eave_y: float = 2.5,
    ridge_extent_x: tuple[float, float] = (-5.0, 5.0),
    slope_half_extent_z: float = 3.0,
    rooms: list[ExtractedRoom] | None = None,
    eave_y_by_room: dict[int, float] | None = None,
    attic_lid_y_by_room: dict[int, float] | None = None,
) -> tuple[BuildingModel, RoofModel]:
    plane_a, plane_b, slope = _gable_planes(ridge_y, eave_y, slope_half_extent_z)
    incl = degrees(atan(slope))
    seg_a = RoofSegment(
        a=[ridge_extent_x[0], eave_y, slope_half_extent_z],
        b=[ridge_extent_x[1], eave_y, slope_half_extent_z],
        incl=incl,
        azimuth=0.0,
        length=ridge_extent_x[1] - ridge_extent_x[0],
        story=0,
        room_index=-1,
    )
    seg_b = RoofSegment(
        a=[ridge_extent_x[0], eave_y, -slope_half_extent_z],
        b=[ridge_extent_x[1], eave_y, -slope_half_extent_z],
        incl=incl,
        azimuth=180.0,
        length=ridge_extent_x[1] - ridge_extent_x[0],
        story=0,
        room_index=-1,
    )
    cluster_a = RoofCluster(
        segments=[seg_a],
        avg_incl=incl,
        avg_azimuth=0.0,
        ref_pt=[0.0, eave_y, slope_half_extent_z],
    )
    cluster_b = RoofCluster(
        segments=[seg_b],
        avg_incl=incl,
        avg_azimuth=180.0,
        ref_pt=[0.0, eave_y, -slope_half_extent_z],
    )
    ridge_span = ridge_extent_x[1] - ridge_extent_x[0]
    poly_a = [
        (ridge_extent_x[0], 0.0),
        (ridge_extent_x[1], 0.0),
        (ridge_extent_x[1], slope_half_extent_z),
        (ridge_extent_x[0], slope_half_extent_z),
    ]
    poly_b = [
        (ridge_extent_x[0], -slope_half_extent_z),
        (ridge_extent_x[1], -slope_half_extent_z),
        (ridge_extent_x[1], 0.0),
        (ridge_extent_x[0], 0.0),
    ]
    cand_a = RoofPlaneCandidate(
        cluster=cluster_a,
        plane=plane_a,
        polygon_xz=poly_a,
        ridge_span=ridge_span,
        dominant_story=0,
    )
    cand_b = RoofPlaneCandidate(
        cluster=cluster_b,
        plane=plane_b,
        polygon_xz=poly_b,
        ridge_span=ridge_span,
        dominant_story=0,
    )
    ob_a = ObliqueSurface(
        corners=[
            [x, eave_y if z == slope_half_extent_z else ridge_y, z] for x, z in poly_a
        ],
        plane=plane_a,
        cluster=cluster_a,
        dominant_story=0,
        ridge={"x": 1.0, "z": 0.0, "min": ridge_extent_x[0], "max": ridge_extent_x[1]},
    )
    ob_b = ObliqueSurface(
        corners=[
            [x, eave_y if z == -slope_half_extent_z else ridge_y, z] for x, z in poly_b
        ],
        plane=plane_b,
        cluster=cluster_b,
        dominant_story=0,
        ridge={"x": 1.0, "z": 0.0, "min": ridge_extent_x[0], "max": ridge_extent_x[1]},
    )
    rooms = rooms or []
    stories_found = max((room.story for room in rooms), default=0) + 1
    model = BuildingModel(
        uuid="synthetic-gable-coverage",
        address=None,
        stories_found=stories_found,
        split_level=False,
        rooms=rooms,
        scan_rooms_found=len(rooms),
        scan_rooms_transformed=len(rooms),
    )
    kinks = RoofKinks(
        eave_y_by_room=dict(eave_y_by_room or {}),
        attic_lid_y_by_room=dict(attic_lid_y_by_room or {}),
        ridge_y=ridge_y,
    )
    roof = RoofModel(
        simple_slant_room_indices=set(),
        segments=[seg_a, seg_b],
        clusters=[cluster_a, cluster_b],
        footprint=None,
        planes=[cand_a, cand_b],
        clipped_planes=[],
        oblique=[ob_a, ob_b],
        flat=[],
        oblique_split=[],
        dormer_candidates=[],
        thermal=[],
        kinks=kinks,
    )
    return model, roof


def _flat_ceiling_room(
    *,
    index: int,
    footprint_xz: list[tuple[float, float]],
    ceiling_y: float,
    eave_y: float | None = None,
    story: int = 0,
) -> ExtractedRoom:
    eave = ceiling_y if eave_y is None else eave_y
    floor_y = float(story) * 3.0
    floor = [[x, floor_y, z] for x, z in footprint_xz]
    ceiling = [[x, ceiling_y, z] for x, z in footprint_xz]
    return ExtractedRoom(
        index=index,
        story=story,
        floor_polygon=floor,
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=ceiling,
        ceiling_type="flat",
        ceiling_eave_height=eave,
        ceiling_ridge_height=None,
    )


def _candidates_for_locator(candidates, marker: str):
    return [candidate for candidate in candidates if marker in candidate.locator_id]


def test_compute_primary_gable_pair_returns_same_ridge_y_as_legacy():
    _, roof = _make_gable_roof()
    pair = compute_primary_gable_pair(roof.planes)
    assert pair is not None
    ridge_y, cand_a, cand_b = pair
    assert ridge_y == compute_gable_ridge_y(roof.planes)
    assert cand_a.ridge_span >= cand_b.ridge_span


def test_ridge_gate_accepts_flat_ceiling_room_below_gable():
    """The bug fix: a flat-ceiling room under the gable, with synthetic shells too
    narrow to cover its footprint in 2D, is recognised via the ridge fallback."""
    room = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-2.0, -2.5), (2.0, -2.5), (2.0, 2.5), (-2.0, 2.5)],
        ceiling_y=3.0,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        rooms=[room],
        eave_y_by_room={0: 3.0},
        attic_lid_y_by_room={0: 3.0},
    )
    selected = _select_gable_obliques_via_ridge(room, model, roof)
    assert selected is not None
    obs, _union = selected
    assert {ob.cluster.avg_azimuth for ob in obs} == {0.0, 180.0}


def test_ridge_gate_uses_combined_pair_span_for_rooms_between_slopes():
    """Rooms can sit under the gable between evidence from the two opposing faces.

    The one-sided ridge bound rejects this layout because candidate A only spans
    the left side, while candidate B carries the right-side continuation.
    """
    target = _flat_ceiling_room(
        index=2,
        footprint_xz=[(2.0, 1.0), (4.0, 1.0), (4.0, 2.0), (2.0, 2.0)],
        ceiling_y=3.0,
    )
    support_a = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-5.0, 0.0), (-1.0, 0.0), (-1.0, 3.0), (-5.0, 3.0)],
        ceiling_y=3.0,
    )
    support_b = _flat_ceiling_room(
        index=1,
        footprint_xz=[(-1.0, -3.0), (5.0, -3.0), (5.0, 0.0), (-1.0, 0.0)],
        ceiling_y=3.0,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        ridge_extent_x=(-5.0, 5.0),
        rooms=[support_a, support_b, target],
        eave_y_by_room={0: 3.0, 1: 3.0, 2: 3.0},
        attic_lid_y_by_room={0: 3.0, 1: 3.0, 2: 3.0},
    )

    cand_a, cand_b = roof.planes
    ob_a, ob_b = roof.oblique
    short_a = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        ridge_extent_x=(-5.0, -1.0),
        rooms=[],
    )[1]
    short_b = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        ridge_extent_x=(-1.0, 5.0),
        rooms=[],
    )[1]
    cand_a = replace(
        cand_a,
        cluster=short_a.planes[0].cluster,
        polygon_xz=short_a.planes[0].polygon_xz,
    )
    cand_b = replace(
        cand_b,
        cluster=short_b.planes[1].cluster,
        polygon_xz=short_b.planes[1].polygon_xz,
    )
    roof = replace(
        roof,
        planes=[cand_a, cand_b],
        oblique=[
            replace(ob_a, corners=short_a.oblique[0].corners, cluster=cand_a.cluster),
            replace(ob_b, corners=short_b.oblique[1].corners, cluster=cand_b.cluster),
        ],
    )

    assert _select_gable_obliques_for_room(target, model, roof) is not None


def test_ridge_gate_rejects_room_outside_ridge_extent():
    """A room placed past the gable's ridge_span along the ridge axis should fail
    horizontal containment even though its ceiling sits below the ridge Y."""
    room = _flat_ceiling_room(
        index=0,
        footprint_xz=[(20.0, -1.0), (22.0, -1.0), (22.0, 1.0), (20.0, 1.0)],
        ceiling_y=3.0,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        ridge_extent_x=(-5.0, 5.0),
        rooms=[room],
        eave_y_by_room={0: 3.0},
        attic_lid_y_by_room={0: 3.0},
    )
    assert _select_gable_obliques_via_ridge(room, model, roof) is None


def test_ridge_gate_rejects_when_ceiling_above_ridge():
    """Vertical containment guard: ceiling_y >= ridge_y - ε ⇒ reject."""
    room = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        ceiling_y=4.5,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        rooms=[room],
        eave_y_by_room={0: 4.5},
        attic_lid_y_by_room={0: 4.5},
    )
    assert _select_gable_obliques_via_ridge(room, model, roof) is None


def test_ridge_gate_short_circuits_when_ridge_y_missing():
    """No primary gable ⇒ no ridge fallback, regardless of room geometry."""
    room = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        ceiling_y=3.0,
    )
    model, roof = _make_gable_roof(
        rooms=[room],
        eave_y_by_room={0: 3.0},
        attic_lid_y_by_room={0: 3.0},
    )
    roof_no_ridge = RoofModel(
        simple_slant_room_indices=roof.simple_slant_room_indices,
        segments=roof.segments,
        clusters=roof.clusters,
        footprint=roof.footprint,
        planes=roof.planes,
        clipped_planes=roof.clipped_planes,
        oblique=roof.oblique,
        flat=roof.flat,
        oblique_split=roof.oblique_split,
        dormer_candidates=roof.dormer_candidates,
        thermal=roof.thermal,
        kinks=RoofKinks(
            eave_y_by_room=dict(roof.kinks.eave_y_by_room),
            attic_lid_y_by_room=dict(roof.kinks.attic_lid_y_by_room),
            ridge_y=None,
        ),
    )
    assert _select_gable_obliques_via_ridge(room, model, roof_no_ridge) is None


def test_select_gable_obliques_for_room_combines_2d_and_ridge():
    """For a room covered by both 2D and ridge tests, the fast 2D path returns
    first (and the result is non-None either way)."""
    # Room sits centrally under the gable: 2D overlap is large.
    room = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
        ceiling_y=3.0,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        ridge_extent_x=(-3.0, 3.0),
        slope_half_extent_z=3.0,
        rooms=[room],
        eave_y_by_room={0: 3.0},
        attic_lid_y_by_room={0: 3.0},
    )
    assert _select_gable_obliques_for_room(room, model, roof) is not None


def test_exposed_split_level_room_under_gable_emits_full_flat():
    """Under the wall-derived-lid design, a flat-classified room under a
    gable always emits its full horizontal flat ceiling — no per-room gable
    pieces stand in for the exposed portion. The gable above lives in
    visual_shells."""
    lower = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-2.0, -2.5), (2.0, -2.5), (2.0, 2.5), (-2.0, 2.5)],
        ceiling_y=3.0,
        story=0,
    )
    upper = _flat_ceiling_room(
        index=1,
        footprint_xz=[(20.0, -1.0), (22.0, -1.0), (22.0, 1.0), (20.0, 1.0)],
        ceiling_y=6.0,
        story=1,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        rooms=[lower, upper],
        eave_y_by_room={0: 3.0, 1: 6.0},
        attic_lid_y_by_room={0: 3.0, 1: 6.0},
    )

    candidates = _ceiling_candidates(model, roof)

    flat = _candidates_for_locator(candidates, "::tier-ceiling-flat::0")
    assert len(flat) == 1
    flat_area = Polygon([(point[0], point[2]) for point in flat[0].corners]).area
    assert flat_area == pytest.approx(20.0)
    assert not _candidates_for_locator(
        candidates, "::tier-ceiling-roof-arrangement-room::0:"
    )


def test_partially_covered_split_level_room_emits_full_flat():
    """The covered/exposed split is no longer reflected at ceiling-emit
    time — the room emits a single flat ceiling across its whole
    footprint. (Adjacency tagging downstream still distinguishes
    internalToHeated vs externalAir per piece.)"""
    lower = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-2.0, -2.5), (2.0, -2.5), (2.0, 2.5), (-2.0, 2.5)],
        ceiling_y=3.0,
        story=0,
    )
    upper = _flat_ceiling_room(
        index=1,
        footprint_xz=[(-2.0, -2.5), (0.0, -2.5), (0.0, 2.5), (-2.0, 2.5)],
        ceiling_y=6.0,
        story=1,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        rooms=[lower, upper],
        eave_y_by_room={0: 3.0, 1: 6.0},
        attic_lid_y_by_room={0: 3.0, 1: 6.0},
    )

    candidates = _ceiling_candidates(model, roof)
    flat = _candidates_for_locator(candidates, "::tier-ceiling-flat::0")

    assert len(flat) == 1
    flat_area = Polygon([(point[0], point[2]) for point in flat[0].corners]).area
    assert flat_area == pytest.approx(20.0)
    assert not _candidates_for_locator(
        candidates, "::tier-ceiling-roof-arrangement-room::0:"
    )


def test_fully_covered_lower_room_under_gable_keeps_flat_ceiling():
    lower = _flat_ceiling_room(
        index=0,
        footprint_xz=[(-2.0, -2.5), (2.0, -2.5), (2.0, 2.5), (-2.0, 2.5)],
        ceiling_y=3.0,
        story=0,
    )
    upper = _flat_ceiling_room(
        index=1,
        footprint_xz=[(-2.0, -2.5), (2.0, -2.5), (2.0, 2.5), (-2.0, 2.5)],
        ceiling_y=6.0,
        story=1,
    )
    model, roof = _make_gable_roof(
        ridge_y=4.0,
        eave_y=2.5,
        rooms=[lower, upper],
        eave_y_by_room={0: 3.0, 1: 6.0},
        attic_lid_y_by_room={0: 3.0, 1: 6.0},
    )

    candidates = _ceiling_candidates(model, roof)

    assert len(_candidates_for_locator(candidates, "::tier-ceiling-flat::0")) == 1
    assert not _candidates_for_locator(
        candidates, "::tier-ceiling-roof-arrangement-room::0:"
    )


def test_wing_filter_drops_obliques_outside_wing():
    """`_filter_obliques_by_wing` keeps only obliques whose XZ extent overlaps the wing.

    An L-shaped building has two wings; the main wing's gable obliques must not
    appear when filtering against the perpendicular extension wing.
    """
    _, roof = _make_gable_roof(
        ridge_extent_x=(-5.0, 5.0),
        slope_half_extent_z=3.0,
    )
    main_wing = Polygon([(-5.0, -3.0), (5.0, -3.0), (5.0, 3.0), (-5.0, 3.0)])
    extension_wing = Polygon([(6.0, -8.0), (10.0, -8.0), (10.0, 0.0), (6.0, 0.0)])

    in_main = _filter_obliques_by_wing(roof.oblique, main_wing)
    in_extension = _filter_obliques_by_wing(roof.oblique, extension_wing)

    assert len(in_main) == 2
    assert in_extension == []

    in_main_cands = _filter_candidates_by_wing(roof.planes, main_wing)
    in_ext_cands = _filter_candidates_by_wing(roof.planes, extension_wing)
    assert len(in_main_cands) == 2
    assert in_ext_cands == []


def test_select_gable_obliques_returns_none_for_room_in_unrelated_wing():
    """A room in a perpendicular extension wing must not pick the main wing's gable.

    Build the gable wide enough that, without wing filtering, the 2D footprint
    overlap test would accept the extension room. The wing filter must drop the
    main wing's obliques so no gable applies to the extension room.
    """
    extension_room = _flat_ceiling_room(
        index=0,
        footprint_xz=[(2.0, -2.0), (4.0, -2.0), (4.0, 2.0), (2.0, 2.0)],
        ceiling_y=3.0,
    )
    model, roof = _make_gable_roof(
        ridge_extent_x=(-5.0, 5.0),
        slope_half_extent_z=3.0,
        rooms=[extension_room],
        eave_y_by_room={0: 3.0},
        attic_lid_y_by_room={0: 3.0},
    )
    extension_wing = Polygon([(6.0, -8.0), (10.0, -8.0), (10.0, 0.0), (6.0, 0.0)])

    selected_no_wing = _select_gable_obliques_for_room(extension_room, model, roof)
    selected_with_wing = _select_gable_obliques_for_room(
        extension_room, model, roof, wing_poly=extension_wing
    )

    assert selected_no_wing is not None  # main-wing gable would have been picked
    assert selected_with_wing is None  # wing-filtered: no obliques in this wing
