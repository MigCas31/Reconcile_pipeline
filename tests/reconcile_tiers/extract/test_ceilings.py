import os
from collections import Counter
from pathlib import Path

import pytest

from reconcile_tiers.extract.building import (
    ExtractedRoom,
    RawCeilingPlane,
    extract_building_model,
)
from reconcile_tiers.extract.ceilings import SLOPE_THRESH_M, _detect_kinked_ceiling


def _kink_room(raw_planes: list[list[list[float]]]) -> ExtractedRoom:
    return ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ],
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[RawCeilingPlane(corners=p) for p in raw_planes],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )


def test_detect_kinked_ceiling_flat_plus_slope_returns_both_polygons():
    flat_lid = [
        [0.0, 2.4, 0.0],
        [2.0, 2.4, 0.0],
        [2.0, 2.4, 2.0],
        [0.0, 2.4, 2.0],
    ]  # 4 m^2
    slope = [
        [2.0, 1.0, 0.0],
        [4.0, 2.4, 0.0],
        [4.0, 2.4, 4.0],
        [2.0, 1.0, 4.0],
    ]  # ~10 m^2, ~35° pitch
    room = _kink_room([flat_lid, slope])

    kink = _detect_kinked_ceiling(room)
    assert kink is not None
    flat_corners, slope_polygons = kink
    slope_corners = slope_polygons[0]
    assert {round(p[1], 2) for p in flat_corners} == {2.4}
    slope_ys = [p[1] for p in slope_corners]
    assert max(slope_ys) - min(slope_ys) > SLOPE_THRESH_M


def test_detect_kinked_ceiling_keeps_second_distinct_slope():
    flat_lid = [[2.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 2.4, 4.0], [2.0, 2.4, 4.0]]
    left_slope = [[0.0, 1.0, 0.0], [2.0, 2.4, 0.0], [2.0, 2.4, 4.0], [0.0, 1.0, 4.0]]
    right_slope = [[4.0, 2.4, 0.0], [6.0, 1.0, 0.0], [6.0, 1.0, 4.0], [4.0, 2.4, 4.0]]

    kink = _detect_kinked_ceiling(_kink_room([flat_lid, left_slope, right_slope]))

    assert kink is not None
    _flat_corners, slope_polygons = kink
    assert len(slope_polygons) == 2


def test_detect_kinked_ceiling_keeps_more_than_two_distinct_slopes():
    flat_lid = [[0.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 2.4, 1.0], [0.0, 2.4, 1.0]]
    slope_x = [[0.0, 1.0, 1.0], [2.0, 2.4, 1.0], [2.0, 2.4, 3.0], [0.0, 1.0, 3.0]]
    slope_neg_x = [[2.0, 2.4, 1.0], [4.0, 1.0, 1.0], [4.0, 1.0, 3.0], [2.0, 2.4, 3.0]]
    slope_z = [[0.0, 1.0, 3.0], [2.0, 1.0, 3.0], [2.0, 2.4, 5.0], [0.0, 2.4, 5.0]]

    kink = _detect_kinked_ceiling(_kink_room([flat_lid, slope_x, slope_neg_x, slope_z]))

    assert kink is not None
    _flat_corners, slope_polygons = kink
    assert len(slope_polygons) == 3


def test_detect_kinked_ceiling_only_flat_returns_none():
    flat_lid = [
        [0.0, 2.4, 0.0],
        [4.0, 2.4, 0.0],
        [4.0, 2.4, 4.0],
        [0.0, 2.4, 4.0],
    ]  # 16 m^2
    assert _detect_kinked_ceiling(_kink_room([flat_lid])) is None


def test_detect_kinked_ceiling_only_slope_returns_none():
    slope = [[0.0, 1.0, 0.0], [4.0, 2.4, 0.0], [4.0, 2.4, 4.0], [0.0, 1.0, 4.0]]
    assert _detect_kinked_ceiling(_kink_room([slope])) is None


def test_detect_kinked_ceiling_below_area_threshold_returns_none():
    # Two real-but-tiny planes (each well under INCLINED_CEILING_MIN_AREA_M2):
    # noise, not a real knee-wall.
    tiny_flat = [[0.0, 2.4, 0.0], [0.3, 2.4, 0.0], [0.3, 2.4, 0.3], [0.0, 2.4, 0.3]]
    tiny_slope = [[1.0, 1.0, 0.0], [1.3, 2.4, 0.0], [1.3, 2.4, 0.3], [1.0, 1.0, 0.3]]
    assert _detect_kinked_ceiling(_kink_room([tiny_flat, tiny_slope])) is None


def test_detect_kinked_ceiling_sums_fragmented_flat_patches():
    # RoomPlan often fragments a single flat lid into ~3 small planes; the
    # cumulative gate must accept their sum even when each individual patch
    # is below INCLINED_CEILING_MIN_AREA_M2.
    frag_a = [
        [0.0, 2.4, 0.0],
        [0.6, 2.4, 0.0],
        [0.6, 2.4, 0.6],
        [0.0, 2.4, 0.6],
    ]  # 0.36 m^2
    frag_b = [
        [0.6, 2.4, 0.0],
        [1.2, 2.4, 0.0],
        [1.2, 2.4, 0.6],
        [0.6, 2.4, 0.6],
    ]  # 0.36 m^2
    slope = [[2.0, 1.0, 0.0], [4.0, 2.4, 0.0], [4.0, 2.4, 4.0], [2.0, 1.0, 4.0]]
    room = _kink_room([frag_a, frag_b, slope])

    assert _detect_kinked_ceiling(room) is not None


def test_kinked_corpus_room_emits_flat_and_oblique_pair():
    # 05cecad4 room 6 has a flat scan patch (0.99 m^2) above a slanted patch
    # (3.87 m^2 dropping 1.1 m). Before the fix, only the wall-derived slope
    # was emitted as one tier-ceiling-computed-oblique-room; the flat lid
    # was lost in the painter as redundant.
    from reconcile_tiers.build import build_tier_payload

    uuid = "05cecad4-119e-4dd2-beaf-d4af36973644"
    payload = build_tier_payload(uuid, Path("pipeline-outputs"), Path(".scan-cache"))
    pieces_for_room_6 = [
        p
        for p in payload.ceiling
        if p.locator_id.endswith(f"{uuid}::tier-ceiling-flat::6")
        or p.locator_id.endswith(f"{uuid}::tier-ceiling-computed-oblique-room::6")
    ]
    locators = {p.locator_id for p in pieces_for_room_6}
    assert f"{uuid}::tier-ceiling-flat::6" in locators
    assert f"{uuid}::tier-ceiling-computed-oblique-room::6" in locators


# Raw-plane counts are baselined to the priors-OFF default. Skip when priors
# are on — `merge_coplanar_raw_ceilings` runs in that path and intentionally
# reduces fragmented coplanar planes per room.
_PRIORS_ON_SKIP = pytest.mark.skipif(
    os.environ.get("ARCHITECTURAL_PRIORS") == "1",
    reason="priors-ON merges coplanar raw ceilings; raw_plane counts shift",
)


@pytest.fixture(autouse=True)
def _force_legacy_priors_off(monkeypatch):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "0")


@_PRIORS_ON_SKIP
@pytest.mark.parametrize(
    ("uuid", "expected_types", "expected_polygons", "expected_raw_planes"),
    [
        ("c72ad855-9e52-46f1-886d-a9f37911521f", {"flat": 5, None: 5}, 5, 21),
        ("f40dcc9f-b97b-4bef-8b40-ba011aabf0bd", {"flat": 9}, 9, 9),
        ("2ea3b759-e047-424c-8034-f8ee5b811fb4", {"flat": 11}, 11, 12),
        ("107e8496-9bff-42bb-b776-720f44b70e55", {"flat": 7, "sloped": 2}, 9, 9),
    ],
)
def test_infer_ceilings_matches_legacy_cohort_types_and_raw_plane_counts(
    uuid,
    expected_types,
    expected_polygons,
    expected_raw_planes,
):
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    assert Counter(room.ceiling_type for room in model.rooms) == expected_types
    assert sum(1 for room in model.rooms if room.ceiling_polygon) == expected_polygons
    assert (
        sum(len(room.raw_ceiling_planes) for room in model.rooms) == expected_raw_planes
    )


def test_neighbour_consensus_does_not_force_room_with_scan_slope_evidence_flat():
    # Morelvej 68 / room 4 has slanted wall tops (top-Y span 0.60 m and 0.73 m)
    # and 1.4 m^2 of inclined raw ceiling planes (incl 6°-23°). All other
    # rooms on the story are flat, so the neighbour-consensus median is within
    # 1.3 cm of room 4's wall-top p50 — without a veto, the classifier collapses
    # the room to FLAT_EMIT and the scan slope is lost.
    model = extract_building_model(
        "feccbd0c-0420-4775-b5b7-49b99559947e",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    room4 = next(r for r in model.rooms if r.index == 4)
    assert room4.ceiling_type == "sloped"
    assert room4.ceiling_ridge_height is not None
    assert room4.ceiling_eave_height is not None
    assert room4.ceiling_ridge_height - room4.ceiling_eave_height > SLOPE_THRESH_M


def test_flat_ceiling_polygons_are_horizontal_and_above_floors():
    model = extract_building_model(
        "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    for room in model.rooms:
        assert room.ceiling_type == "flat"
        assert room.ceiling_polygon
        ceiling_ys = {round(corner[1], 4) for corner in room.ceiling_polygon}
        floor_y = sum(corner[1] for corner in room.floor_polygon) / len(
            room.floor_polygon
        )
        assert len(ceiling_ys) == 1
        assert next(iter(ceiling_ys)) > floor_y
        assert (
            room.ceiling_eave_height
            == room.ceiling_ridge_height
            == next(iter(ceiling_ys))
        )
