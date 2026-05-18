from collections import Counter
from math import isfinite
from pathlib import Path

import pytest

from reconcile_tiers.extract.building import (
    ExteriorGapIndicator,
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
    extract_building_model,
)
from reconcile_tiers.extract.exterior import (
    compute_gap_closures,
    detect_exterior_gap_indicators,
)


@pytest.mark.parametrize(
    ("uuid", "expected_indicators", "expected_closures"),
    [
        (
            "c72ad855-9e52-46f1-886d-a9f37911521f",
            {"door": 1, "storage": 2},
            {"side": 6, "floor": 3, "ceiling": 3},
        ),
        (
            "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
            {"door": 2, "opening": 1, "storage": 1},
            {"side": 8, "floor": 4, "ceiling": 4},
        ),
        (
            "2ea3b759-e047-424c-8034-f8ee5b811fb4",
            {"door": 2},
            {"side": 4, "floor": 2, "ceiling": 2},
        ),
    ],
)
def test_exterior_gap_indicators_and_closures_match_legacy_cohort(
    uuid,
    expected_indicators,
    expected_closures,
):
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    assert (
        Counter(indicator.element_type for indicator in model.exterior_gap_indicators)
        == expected_indicators
    )
    assert Counter(closure.type for closure in model.gap_closures) == expected_closures
    for indicator in model.exterior_gap_indicators:
        assert indicator.element_id
        assert indicator.wall_id
        assert indicator.wall_distance_m > 0.0
        assert indicator.element_width_m >= 0.5
    for closure in model.gap_closures:
        assert len(closure.corners) == 4
        assert all(isfinite(coord) for corner in closure.corners for coord in corner)


def test_storage_gap_closure_uses_parent_wall_profile():
    indicator = ExteriorGapIndicator(
        story=0,
        element_type="storage",
        element_id="storage-a",
        element_corners=[
            [0.2, 2.0, 0.0],
            [0.8, 2.0, 0.0],
            [0.8, 2.5, 0.0],
            [0.2, 2.5, 0.0],
        ],
        element_width_m=0.6,
        wall_id="wall-b",
        wall_corners=[
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 3.0, 0.5],
            [0.0, 3.0, 0.5],
        ],
        wall_distance_m=0.5,
        angle_deg=0.0,
        confidence="medium",
        parent_wall_id="wall-a",
        parent_wall_corners=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
    )

    closures = compute_gap_closures([indicator])

    assert Counter(closure.type for closure in closures) == {
        "side": 2,
        "floor": 1,
        "ceiling": 1,
    }
    ys = [
        coord
        for closure in closures
        for corner in closure.corners
        for coord in [corner[1]]
    ]
    assert min(ys) == 0.0
    assert max(ys) == 3.0


def test_wide_door_with_parallel_wall_at_1_1_meters_emits_indicator():
    # Mirrors building d98923f0 stitch:0:36: a wide door in one room sits
    # ~1.10m from a parallel wall in another room, with the wall slightly
    # narrower than the door but well within the 0.9x ratio. The door's
    # XZ extent projects onto the wall axis with full coverage.
    door_room_wall = ExtractedWall(
        id="door-parent",
        source="test",
        corners=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
    )
    door = ExtractedElement(
        id="door-wide",
        source="test",
        parent_wall_id="door-parent",
        corners=[
            [0.0, 0.0, 0.0],
            [2.95, 0.0, 0.0],
            [2.95, 2.4, 0.0],
            [0.0, 2.4, 0.0],
        ],
    )
    door_room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, -1.0],
            [3.0, 0.0, -1.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        walls_merged=[],
        walls_computed=[door_room_wall],
        doors=[door],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    parallel_wall = ExtractedWall(
        id="wall-across",
        source="test",
        corners=[
            [0.05, 0.0, 1.10],
            [2.95, 0.0, 1.10],
            [2.95, 3.0, 1.10],
            [0.05, 3.0, 1.10],
        ],
    )
    other_room = ExtractedRoom(
        index=1,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 1.10],
            [3.0, 0.0, 1.10],
            [3.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        walls_merged=[],
        walls_computed=[parallel_wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )

    indicators = detect_exterior_gap_indicators([door_room, other_room])
    closures = compute_gap_closures(indicators)

    matched = [i for i in indicators if i.element_id == "door-wide"]
    assert len(matched) == 1
    assert matched[0].wall_id == "wall-across"
    assert 1.05 < matched[0].wall_distance_m < 1.20
    assert any(closure.type == "floor" for closure in closures)
    assert any(closure.type == "ceiling" for closure in closures)


def test_door_far_from_wall_with_offset_does_not_match_via_projection_gate():
    # Same 1.1m distance, but the wall is shifted along its axis so the door
    # projects only ~30% onto the wall — must be rejected by the new
    # projection-coverage gate.
    parent = ExtractedWall(
        id="parent",
        source="test",
        corners=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
    )
    door = ExtractedElement(
        id="door",
        source="test",
        parent_wall_id="parent",
        corners=[
            [0.0, 0.0, 0.0],
            [2.95, 0.0, 0.0],
            [2.95, 2.4, 0.0],
            [0.0, 2.4, 0.0],
        ],
    )
    offset_wall = ExtractedWall(
        id="offset",
        source="test",
        corners=[
            [2.0, 0.0, 1.10],
            [4.95, 0.0, 1.10],
            [4.95, 3.0, 1.10],
            [2.0, 3.0, 1.10],
        ],
    )
    rooms = [
        ExtractedRoom(
            index=0,
            story=0,
            floor_polygon=[
                [0.0, 0.0, -1.0],
                [3.0, 0.0, -1.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            walls_merged=[],
            walls_computed=[parent],
            doors=[door],
            windows=[],
            openings=[],
            storages=[],
            raw_ceiling_planes=[],
            raw_ceiling_source=None,
            ceiling_polygon=[],
            ceiling_type=None,
            ceiling_eave_height=None,
            ceiling_ridge_height=None,
        ),
        ExtractedRoom(
            index=1,
            story=0,
            floor_polygon=[
                [2.0, 0.0, 1.10],
                [5.0, 0.0, 1.10],
                [5.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
            ],
            walls_merged=[],
            walls_computed=[offset_wall],
            doors=[],
            windows=[],
            openings=[],
            storages=[],
            raw_ceiling_planes=[],
            raw_ceiling_source=None,
            ceiling_polygon=[],
            ceiling_type=None,
            ceiling_eave_height=None,
            ceiling_ridge_height=None,
        ),
    ]
    indicators = detect_exterior_gap_indicators(rooms)
    assert all(i.element_id != "door" for i in indicators)


def test_door_gap_closure_uses_parent_wall_profile_not_door_head():
    parent_wall = ExtractedWall(
        id="wall-parent",
        source="test",
        corners=[
            [0.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    opposite_wall = ExtractedWall(
        id="wall-opposite",
        source="test",
        corners=[
            [0.0, 2.6, 0.5],
            [3.0, 2.6, 0.5],
            [3.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
        ],
    )
    door = ExtractedElement(
        id="door-a",
        source="test",
        parent_wall_id="wall-parent",
        corners=[
            [0.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [2.5, 2.0, 0.0],
            [0.5, 2.0, 0.0],
        ],
    )
    indicator = ExteriorGapIndicator(
        story=0,
        element_type="door",
        element_id=door.id,
        element_corners=door.corners,
        element_width_m=2.0,
        wall_id=opposite_wall.id,
        wall_corners=opposite_wall.corners,
        wall_distance_m=0.5,
        angle_deg=0.0,
        confidence="medium",
        parent_wall_id=parent_wall.id,
        parent_wall_corners=parent_wall.corners,
    )

    closures = compute_gap_closures([indicator])

    ceiling = next(closure for closure in closures if closure.type == "ceiling")
    ceiling_ys = [corner[1] for corner in ceiling.corners]
    assert min(ceiling_ys) == 2.6
    assert max(ceiling_ys) == 3.0
