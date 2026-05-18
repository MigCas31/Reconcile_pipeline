"""Tests for plane-group union widening in the ridge/eave scorer.

Covers two motivating cases the raw candidate-footprint union can't express:

1. ``within_story`` gap strips (target 38f71f1d-...): candidates on either
   side of a thin gap polygon, connected by the 0.3 m buffer, leave a hole
   in the union where the physical roof plane actually continues.
2. Diagonal cuts where a scanned interior wall split the candidate footprint
   but the roof plane extends over the wall (target
   e0155eef-...::wall-computed::E2E5D435-...). No gap polygon is emitted here;
   the hole is just incomplete scan coverage.

The fix is morphological closing (buffer +C, -C) intersected with the
building's room-floor-polygon union so closing can't invent area outside
the physical building. Closing must run AFTER the scan clip because
``scan_poly`` derives from candidate footprints too and doesn't include
the holes -- clipping first would just trim what closing fills.
"""

from __future__ import annotations

from shapely.geometry import Point, Polygon

from scripts.score_candidates_ridge_eave import (
    PLANE_GROUP_CLOSING_M,
    PLANE_GROUP_FLOOD_REACH_M,
    _build_plane_groups,
    _building_envelope_fp,
    _building_fp_union,
    _candidate_member_polygon,
    _creator_eave_proximity_score,
    _pair_beats,
)


def _rect_xz(x0: float, z0: float, x1: float, z1: float):
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]


_PLANE = [0.0, 1.0, 0.0, -3.0]


def _candidate(cand_id: str, fp_xz, *, area: float, parent: str = "seg") -> dict:
    return {
        "id": cand_id,
        "plane": _PLANE,
        "footprint_xz": fp_xz,
        "area_m2": area,
        "support_m2": area,
        "azimuth_deg": 0.0,
        "inclination_deg": 20.0,
        "parent_segment_id": parent,
    }


def _plane_group_with_members(*members: dict) -> dict:
    return {"members": list(members)}


def test_closing_fills_thin_gap_between_room_candidates():
    # Two candidates leave a 0.5 m notch in X. The 0.3 m connect buffer
    # merges them into a single plane-group but the strip remains a hole
    # in the MultiPolygon union. Morphological closing at
    # PLANE_GROUP_CLOSING_M (~0.75 m) bridges the notch; intersecting the
    # closed polygon with the building_fp keeps closing's footprint physical.
    left = _candidate("L", _rect_xz(0, 0, 4, 4), area=16.0, parent="segL")
    right = _candidate("R", _rect_xz(4.5, 0, 8.5, 4), area=16.0, parent="segR")

    groups_no_fp = _build_plane_groups("b-test", [left, right])
    assert len(groups_no_fp) == 1
    union_no_fp = groups_no_fp[0]["union"]
    assert union_no_fp.geom_type == "MultiPolygon", (
        "without widening, the 0.5 m strip is a hole between the two rectangles"
    )
    assert not union_no_fp.contains(Point(4.25, 2.0))
    assert abs(union_no_fp.area - 32.0) < 0.01

    building_fp = Polygon(_rect_xz(0, 0, 8.5, 4))
    groups = _build_plane_groups("b-test", [left, right], building_fp=building_fp)
    assert len(groups) == 1
    merged = groups[0]["union"]
    assert merged.geom_type == "Polygon", (
        "closing fills the strip, collapsing the notched multipolygon"
    )
    assert merged.contains(Point(4.25, 2.0))
    # Closing rounds the outer corners slightly; area is ~34 m^2 (32 from
    # candidates + 2 from the filled notch) minus <0.1 m^2 corner smoothing.
    assert 33.8 < merged.area < 34.1


def test_closing_clipped_to_building_fp():
    # Closing a non-convex candidate union can invent area outside the
    # building; intersecting with building_fp keeps the widening inside.
    cand = _candidate("C", _rect_xz(0, 0, 4, 4), area=16.0)
    # Tight fp equal to the candidate footprint -- closing can't extend.
    tight_fp = Polygon(_rect_xz(0, 0, 4, 4))
    groups = _build_plane_groups("b-test", [cand], building_fp=tight_fp)
    assert len(groups) == 1
    assert abs(groups[0]["union"].area - 16.0) < 0.01


def test_closing_happens_after_scan_clip():
    # ``scan_poly`` derives from candidate footprints too and so does not
    # include the between-room notch. If closing happened BEFORE the scan
    # clip, ``intersection(scan_poly)`` would immediately re-carve the hole
    # -- which is what target 38f71f1d-... hit in production. The fix orders
    # scan clip first, then closing, so closing survives.
    left = _candidate("L", _rect_xz(0, 0, 4, 4), area=16.0, parent="segL")
    right = _candidate("R", _rect_xz(4.5, 0, 8.5, 4), area=16.0, parent="segR")
    scan_poly = Polygon(_rect_xz(0, 0, 4, 4)).union(Polygon(_rect_xz(4.5, 0, 8.5, 4)))
    building_fp = Polygon(_rect_xz(0, 0, 8.5, 4))

    groups = _build_plane_groups(
        "b-test", [left, right], scan_poly=scan_poly, building_fp=building_fp
    )
    assert len(groups) == 1
    merged = groups[0]["union"]
    assert merged.contains(Point(4.25, 2.0)), (
        "closing must survive even when scan_poly is tight to candidates"
    )
    assert 33.8 < merged.area < 34.1


def test_closing_scale_is_bounded():
    # Two candidates separated by a gap larger than 2*PLANE_GROUP_CLOSING_M
    # must NOT be bridged by closing -- closing bridges sub-2C notches only,
    # and those candidates wouldn't share a plane-group in the first place
    # (connect-buffer is 0.3 m), but this guards against runaway widening
    # if the closing constant is ever raised without thinking.
    gap_too_wide = 2.0 * PLANE_GROUP_CLOSING_M + 1.0
    left = _candidate("L", _rect_xz(0, 0, 4, 4), area=16.0, parent="segL")
    right = _candidate(
        "R",
        _rect_xz(4.0 + gap_too_wide, 0, 8.0 + gap_too_wide, 4),
        area=16.0,
        parent="segR",
    )
    building_fp = Polygon(_rect_xz(0, 0, 8.0 + gap_too_wide, 4))
    groups = _build_plane_groups("b-test", [left, right], building_fp=building_fp)
    # 0.3 m buffer can't connect them, so they stay two plane-groups.
    assert len(groups) == 2
    # Neither group should balloon past its candidate area.
    for g in groups:
        assert g["union"].area < 16.5


def test_building_fp_union_from_room_floor_polygons():
    # _building_fp_union reads rooms[].floor_polygon (3D, X/Y/Z) and unions
    # the XZ projection. We ignore cross_floor_gaps and gap_walls entirely --
    # room floors are the physical building envelope.
    bldg = {
        "rooms": [
            {
                "floor_polygon": [
                    [0, 0, 0],
                    [4, 0, 0],
                    [4, 0, 3],
                    [0, 0, 3],
                ],
            },
            {
                "floor_polygon": [
                    [4, 0, 0],
                    [7, 0, 0],
                    [7, 0, 3],
                    [4, 0, 3],
                ],
            },
        ],
        # These fields must NOT contribute to the fp union:
        "cross_floor_gaps": [{"type": "within_story", "corners": [[100, 0, 100]] * 4}],
        "gap_walls": [{"corners": [[0, 0, 0], [1, 0, 0], [1, 2.5, 0], [0, 2.5, 0]]}],
    }
    fp = _building_fp_union(bldg)
    assert fp is not None
    # Two rooms of 12 m^2 sharing an edge -> 21 m^2 total.
    assert abs(fp.area - 21.0) < 0.01
    # Far-away points must not be inside.
    assert not fp.contains(Point(100.0, 100.0))


def test_building_fp_union_empty_for_no_rooms():
    assert _building_fp_union({}) is None
    assert _building_fp_union({"rooms": []}) is None
    assert _building_fp_union(None) is None


def test_building_envelope_fp_includes_cross_floor_gaps():
    # _building_envelope_fp unions rooms AND every cross_floor_gap (both
    # within_story and cross_story). This is the flood-fill target.
    bldg = {
        "rooms": [
            {"floor_polygon": [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]]},
        ],
        "cross_floor_gaps": [
            # within_story gap that shares an edge with the room
            {
                "type": "within_story",
                "corners": [[4, 0, 0], [6, 0, 0], [6, 0, 3], [4, 0, 3]],
            },
            # cross_story gap sharing an edge too
            {
                "type": "cross_story",
                "corners": [[0, 0, 3], [4, 0, 3], [4, 0, 5], [0, 0, 5]],
            },
        ],
    }
    env = _building_envelope_fp(bldg)
    assert env is not None
    # Room 12 + within 6 + cross 8 = 26 m^2
    assert abs(env.area - 26.0) < 0.01


def test_flood_fill_covers_cross_story_extension():
    # A plane-group candidate sits in a 4x4 room; an adjacent cross_story
    # gap extends further in Z. The candidate's outer Z edge is at z=4, the
    # gap extends to z=6 -- a 2 m extension. Flood-fill at reach=2.0 m
    # should pull the gap's full extent into the plane-group union.
    cand = _candidate("C", _rect_xz(0, 0, 4, 4), area=16.0)
    building_fp = Polygon(_rect_xz(0, 0, 4, 4))
    envelope = Polygon(_rect_xz(0, 0, 4, 6))  # room + cross_story extension
    groups = _build_plane_groups(
        "b-test", [cand], building_fp=building_fp, envelope_fp=envelope
    )
    assert len(groups) == 1
    merged = groups[0]["union"]
    assert merged.contains(Point(2.0, 5.0)), (
        "flood-fill must pull the cross_story extension into the plane-group"
    )
    # Full envelope is 24 m^2; we should cover it all (candidate + extension).
    assert 23.5 < merged.area < 24.1


def test_flood_fill_skips_detached_components():
    # A plane-group on a main building; a detached outbuilding's floor
    # polygon is a separate envelope component far away. Flood-fill must
    # NOT absorb the detached component -- it connects to other plane-groups
    # in the building, not this one.
    cand = _candidate("C", _rect_xz(0, 0, 4, 4), area=16.0)
    building_fp = Polygon(_rect_xz(0, 0, 4, 4))
    main = Polygon(_rect_xz(0, 0, 4, 4))
    detached = Polygon(_rect_xz(20, 20, 24, 24))  # far outbuilding
    envelope = main.union(detached)
    groups = _build_plane_groups(
        "b-test", [cand], building_fp=building_fp, envelope_fp=envelope
    )
    assert len(groups) == 1
    merged = groups[0]["union"]
    assert not merged.contains(Point(22.0, 22.0)), (
        "detached components must stay separate"
    )
    # Roughly candidate area; no absorption of the 16 m^2 outbuilding.
    assert merged.area < 17.0


def test_flood_fill_distance_cap():
    # Envelope has a long thin tail extending from z=4 to z=4 + 2*REACH.
    # Flood-fill at REACH=2.0 m can only reach half way down the tail --
    # the far end is more than REACH m from any candidate point.
    reach = PLANE_GROUP_FLOOD_REACH_M
    cand = _candidate("C", _rect_xz(0, 0, 4, 4), area=16.0)
    building_fp = Polygon(_rect_xz(0, 0, 4, 4))
    envelope = Polygon(_rect_xz(0, 0, 4, 4 + 2.5 * reach))
    groups = _build_plane_groups(
        "b-test", [cand], building_fp=building_fp, envelope_fp=envelope
    )
    assert len(groups) == 1
    merged = groups[0]["union"]
    # Near the candidate edge: included.
    assert merged.contains(Point(2.0, 4.0 + 0.5 * reach))
    # Beyond REACH from the candidate: excluded.
    assert not merged.contains(Point(2.0, 4.0 + 1.5 * reach))


def test_creator_eave_proximity_prefers_members_near_eave():
    eave = _candidate_member_polygon(_candidate("e", _rect_xz(0, 0, 4, 0.1), area=0.4))
    assert eave is not None
    near_group = _plane_group_with_members(
        _candidate("near-1", _rect_xz(0, 0, 2, 1), area=2.0),
        _candidate("near-2", _rect_xz(2, 0, 4, 1), area=2.0),
    )
    far_group = _plane_group_with_members(
        _candidate("far-1", _rect_xz(0, 4, 2, 5), area=2.0),
        _candidate("far-2", _rect_xz(2, 4, 4, 5), area=2.0),
    )

    near_score = _creator_eave_proximity_score(near_group, eave.exterior)
    far_score = _creator_eave_proximity_score(far_group, eave.exterior)

    assert near_score > far_score
    assert near_score > 0.5
    assert far_score < 0.05


def test_pair_beats_uses_creator_eave_proximity_to_break_score_tie():
    prev = {
        "best_score": 0.8793,
        "creator_eave_proximity": 0.25,
    }
    better_tie = {
        "score": 0.8793,
        "creator_eave_proximity": 0.75,
    }
    worse_tie = {
        "score": 0.8793,
        "creator_eave_proximity": 0.10,
    }
    lower_score = {
        "score": 0.80,
        "creator_eave_proximity": 1.0,
    }

    assert _pair_beats(prev, better_tie) is True
    assert _pair_beats(prev, worse_tie) is False
    assert _pair_beats(prev, lower_score) is False
