from collections import Counter
from math import isfinite
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.extract.building import (
    ExtractedGap,
    ExtractedRoom,
    ExtractedWall,
    extract_building_model,
)
from reconcile_tiers.extract.gaps import (
    MAX_CEILING_CAP_INCLINATION_DEG,
    _emit_gaps,
    _surface_inclination_deg,
    compute_gap_walls,
)


def _projected_xz_area(corners):
    area2 = 0.0
    for idx, corner in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        area2 += corner[0] * nxt[2] - nxt[0] * corner[2]
    return abs(area2) * 0.5


@pytest.mark.parametrize(
    (
        "uuid",
        "expected_gap_types",
        "expected_floor_vertices",
        "expected_gap_wall_types",
    ),
    [
        (
            "c72ad855-9e52-46f1-886d-a9f37911521f",
            {"within_story": 14, "cross_story": 9},
            [18, 8, 11, 4, 4, 21, 4, 15, 8, 7],
            {"within_story": 26, "gap_floor": 16, "gap_ceiling": 16},
        ),
        (
            "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
            {"within_story": 13},
            [4, 15, 14, 16, 14, 14, 14, 11, 4],
            {"within_story": 30, "gap_floor": 17, "gap_ceiling": 17},
        ),
        (
            "2ea3b759-e047-424c-8034-f8ee5b811fb4",
            {"within_story": 20},
            [14, 13, 21, 47, 4, 4, 18, 4, 10, 9, 4],
            {"within_story": 37, "gap_floor": 21, "gap_ceiling": 21},
        ),
    ],
)
def test_cross_floor_gap_detection_and_absorption_matches_legacy_cohort(
    uuid,
    expected_gap_types,
    expected_floor_vertices,
    expected_gap_wall_types,
):
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    assert Counter(gap.type for gap in model.cross_floor_gaps) == expected_gap_types
    assert [len(room.floor_polygon) for room in model.rooms] == expected_floor_vertices

    within_story = [gap for gap in model.cross_floor_gaps if gap.type == "within_story"]
    assert within_story
    assert all(gap.room_index is not None for gap in within_story)
    assert all(gap.ceiling_corners for gap in within_story)
    assert Counter(wall.type for wall in model.gap_walls) == expected_gap_wall_types

    for wall in model.gap_walls:
        assert len(wall.corners) >= 3
        assert all(isfinite(coord) for corner in wall.corners for coord in corner)
        if wall.type == "within_story":
            assert len(wall.corners) == 4
            ys = [corner[1] for corner in wall.corners]
            assert max(ys) - min(ys) > 0.5
        elif wall.type == "gap_floor":
            ys = [corner[1] for corner in wall.corners]
            assert max(ys) - min(ys) < 1e-6
            assert _projected_xz_area(wall.corners) > 1e-6
        elif wall.type == "gap_ceiling":
            assert _projected_xz_area(wall.corners) > 1e-6


def test_within_story_gap_neighborhoods_do_not_emit_duplicate_gap_wall_surfaces():
    model = extract_building_model(
        "0a5032e9-85a0-4970-9143-c430bbdaa0f5",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    gap_wall_ids = [wall.id for wall in model.gap_walls]

    assert len(gap_wall_ids) == len(set(gap_wall_ids))


def test_gap_caps_do_not_overlap_each_other_on_motivating_building():
    payload = build_tier_payload(
        "0fe789ce-e653-4828-863c-0f6ce7fee21d",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )
    cap_kinds = {"gap_floor", "gap_ceiling"}
    caps = [gap for gap in payload.gaps if gap.kind.value in cap_kinds]
    offenders = []
    for idx, gap in enumerate(caps):
        if len(gap.corners) < 3:
            continue
        poly = Polygon([(corner.x, corner.z) for corner in gap.corners])
        if not poly.is_valid or poly.is_empty:
            continue
        y = round(sum(corner.y for corner in gap.corners) / len(gap.corners), 3)
        for other in caps[idx + 1 :]:
            if other.kind != gap.kind or len(other.corners) < 3:
                continue
            other_y = round(
                sum(corner.y for corner in other.corners) / len(other.corners), 3
            )
            if other_y != y:
                continue
            other_poly = Polygon([(corner.x, corner.z) for corner in other.corners])
            if not other_poly.is_valid or other_poly.is_empty:
                continue
            overlap = poly.intersection(other_poly).area
            if overlap > 0.01:
                offenders.append((gap.locator_id, other.locator_id, round(overlap, 3)))

    assert offenders == []


def test_within_story_gap_cap_uses_assigned_sloped_room_plane():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[
            [0.0, 1.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        ceiling_type="sloped",
        ceiling_eave_height=1.0,
        ceiling_ridge_height=3.0,
    )
    gap = ExtractedGap(
        story=0,
        type="within_story",
        corners=[
            [2.0, 0.0, 0.5],
            [4.0, 0.0, 0.5],
            [4.0, 0.0, 1.5],
            [2.0, 0.0, 1.5],
            [2.0, 0.0, 0.5],
        ],
        area_m2=2.0,
        compactness=0.7,
        confidence="high",
        centroid=[3.0, 0.0, 1.0],
        ceiling_corners=[],
        room_index=0,
    )

    walls, _updated_gaps = compute_gap_walls(
        [gap],
        [room],
        {0: 0.0},
        pre_absorption_floor_polygons=[
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
        ],
    )

    ceiling_caps = [wall for wall in walls if wall.type == "gap_ceiling"]
    assert ceiling_caps
    assert any(
        max(corner[1] for corner in cap.corners)
        - min(corner[1] for corner in cap.corners)
        > 0.5
        for cap in ceiling_caps
    )


def test_within_story_gap_cap_uses_assigned_flat_room_ceiling_not_wall_profile():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        walls_merged=[],
        walls_computed=[
            ExtractedWall(
                id="low-support",
                source="test",
                corners=[
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 0.0],
                ],
            ),
            ExtractedWall(
                id="high-support",
                source="test",
                corners=[
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 0.0],
                ],
            ),
        ],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[
            [0.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
        ],
        ceiling_type="flat",
        ceiling_eave_height=3.0,
        ceiling_ridge_height=3.0,
    )
    gap = ExtractedGap(
        story=0,
        type="within_story",
        corners=[
            [2.0, 0.0, 0.2],
            [4.0, 0.0, 0.2],
            [4.0, 0.0, 1.8],
            [2.0, 0.0, 1.8],
            [2.0, 0.0, 0.2],
        ],
        area_m2=3.2,
        compactness=0.7,
        confidence="high",
        centroid=[3.0, 0.0, 1.0],
        ceiling_corners=[],
        room_index=0,
    )

    walls, _updated_gaps = compute_gap_walls(
        [gap],
        [room],
        {0: 0.0},
        pre_absorption_floor_polygons=[
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
        ],
    )

    ceiling_caps = [wall for wall in walls if wall.type == "gap_ceiling"]
    assert ceiling_caps
    for cap in ceiling_caps:
        ys = [corner[1] for corner in cap.corners]
        assert max(ys) - min(ys) < 1e-6
        assert ys[0] == pytest.approx(3.0)


def test_within_story_gap_ceiling_does_not_emit_wall_like_cap():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 7.0, 1.0],
            [0.0, 7.0, 1.0],
        ],
        ceiling_type="sloped",
        ceiling_eave_height=1.0,
        ceiling_ridge_height=7.0,
    )
    gap = ExtractedGap(
        story=0,
        type="within_story",
        corners=[
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.1, 0.0, 0.0],
        ],
        area_m2=0.8,
        compactness=0.7,
        confidence="high",
        centroid=[0.5, 0.0, 0.5],
        ceiling_corners=[],
        room_index=0,
    )

    walls, _updated_gaps = compute_gap_walls(
        [gap],
        [room],
        {0: 0.0},
        pre_absorption_floor_polygons=[
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ],
    )

    ceiling_caps = [wall for wall in walls if wall.type == "gap_ceiling"]
    assert all(
        (_surface_inclination_deg(cap.corners) or 0.0)
        <= MAX_CEILING_CAP_INCLINATION_DEG
        for cap in ceiling_caps
    )


def test_gap_wall_snap_uses_finite_wall_segment_not_infinite_line():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        walls_merged=[],
        walls_computed=[
            ExtractedWall(
                id="support",
                source="test",
                corners=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                ],
            )
        ],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[
            [0.0, 2.0, -1.0],
            [1.0, 2.0, -1.0],
            [1.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        ceiling_type="flat",
        ceiling_eave_height=2.0,
        ceiling_ridge_height=2.0,
    )
    gap = ExtractedGap(
        story=0,
        type="within_story",
        corners=[
            [0.2, 0.0, 0.1],
            [1.2, 0.0, 0.1],
            [1.2, 0.0, 1.3],
            [0.2, 0.0, 1.3],
            [0.2, 0.0, 0.1],
        ],
        area_m2=1.2,
        compactness=0.7,
        confidence="high",
        centroid=[0.7, 0.0, 0.7],
        ceiling_corners=[],
        room_index=0,
    )

    walls, _updated_gaps = compute_gap_walls([gap], [room], {0: 0.0})

    snapped_xz = [
        (round(corner[0], 6), round(corner[2], 6))
        for wall in walls
        for corner in wall.corners
        if wall.type in {"within_story", "gap_floor", "gap_ceiling"}
    ]
    assert (1.2, 0.0) not in snapped_xz
    assert (1.0, 0.0) in snapped_xz


def test_gap_side_wall_does_not_extend_past_ceiling_polygon_extent():
    # When a gap polygon lies outside the host room's CEILING polygon (only
    # partially or wholly), the legacy code extrapolated the sloped ceiling
    # plane into the gap region, sending side-wall tops far above the actual
    # ridge. The fix refuses extrapolation: ceiling y is only sourced from
    # the plane where the XZ is covered by the ceiling polygon. Outside that
    # extent the wall snaps to the wall-top profile (the legacy "no scan
    # ceiling here" fallback), which is still scan-derived.
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        walls_merged=[],
        walls_computed=[
            ExtractedWall(
                id="ridge-wall",
                source="test",
                corners=[
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 0.0],
                ],
            ),
        ],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        # Sloped ceiling, eave at y=1, ridge at y=3. Polygon covers xin[0,2].
        ceiling_polygon=[
            [0.0, 1.0, 0.0],
            [2.0, 3.0, 0.0],
            [2.0, 3.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        ceiling_type="sloped",
        ceiling_eave_height=1.0,
        ceiling_ridge_height=3.0,
    )
    # Gap entirely outside the ceiling polygon, at xin[3,5]. If we extrapolate
    # the plane there, ytop ≈ 5-7 m (well above the actual ridge at y=3).
    gap = ExtractedGap(
        story=0,
        type="within_story",
        corners=[
            [3.0, 0.0, 0.5],
            [5.0, 0.0, 0.5],
            [5.0, 0.0, 1.5],
            [3.0, 0.0, 1.5],
            [3.0, 0.0, 0.5],
        ],
        area_m2=2.0,
        compactness=0.7,
        confidence="high",
        centroid=[4.0, 0.0, 1.0],
        ceiling_corners=[],
        room_index=0,
    )
    walls, _ = compute_gap_walls(
        [gap],
        [room],
        {0: 0.0},
        pre_absorption_floor_polygons=[room.floor_polygon],
    )
    side_walls = [w for w in walls if w.type == "within_story"]
    for w in side_walls:
        ys = [corner[1] for corner in w.corners]
        # The wall must NOT extend past the ridge (y=3) of the host room's
        # ceiling polygon. The legacy extrapolation produced ytops ≈ 5-7.
        assert max(ys) <= 3.0 + 0.05, (
            f"wall extends past ceiling ridge: y_max={max(ys):.3f}"
        )


def test_gap_ceiling_flat_sliver_triangle_is_filtered():
    # Synthetic earclip sliver: a long-thin triangle that is geometrically flat
    # (no y range). Earclipping a near-degenerate gap polygon used to emit
    # these as "ceiling" caps; the fix drops them via the combined
    # quality-and-flatness filter (matches the user-reported 2.6m x 0.03m
    # sliver pathology).
    room = ExtractedRoom(
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
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[
            [0.0, 2.5, 0.0],
            [4.0, 2.5, 0.0],
            [4.0, 2.5, 4.0],
            [0.0, 2.5, 4.0],
        ],
        ceiling_type="flat",
        ceiling_eave_height=2.5,
        ceiling_ridge_height=2.5,
    )
    # Long-thin gap polygon: 3.0m long, 0.02m wide -> earclip produces slivers.
    gap = ExtractedGap(
        story=0,
        type="within_story",
        corners=[
            [0.5, 0.0, 1.0],
            [3.5, 0.0, 1.0],
            [3.5, 0.0, 1.02],
            [0.5, 0.0, 1.02],
            [0.5, 0.0, 1.0],
        ],
        area_m2=0.06,
        compactness=0.05,
        confidence="high",
        centroid=[2.0, 0.0, 1.01],
        ceiling_corners=[],
        room_index=0,
    )
    walls, _ = compute_gap_walls(
        [gap],
        [room],
        {0: 0.0},
        pre_absorption_floor_polygons=[
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 4.0], [0.0, 0.0, 4.0]]
        ],
    )
    ceiling_caps = [w for w in walls if w.type == "gap_ceiling"]
    # Filter caught the slivers -- no flat sliver caps emitted, even though
    # the host room has a perfectly good ceiling polygon at y=2.5.
    for cap in ceiling_caps:
        ys = [corner[1] for corner in cap.corners]
        # Surviving caps must either have y-spread (sloped) or non-sliver shape.
        if max(ys) - min(ys) < 0.10:
            from reconcile_tiers.extract.gaps import _triangle_quality

            assert (
                _triangle_quality(cap.corners[0], cap.corners[1], cap.corners[2])
                >= 0.05
            )


def test_uncovered_wall_top_is_closed_by_gap_ceiling_cap():
    """Building 'a085ad69-...' has a listed wall whose top sits beneath an
    uncovered area; the assembled payload must still contain a gap_ceiling
    piece just above the wall at the same XZ.
    """
    payload = build_tier_payload(
        "a085ad69-b4d4-4296-bc5e-731a0cceec5e",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    target_uuid = "C0A8F9A2-AC40-4011-B3E7-AA5120B6C26F"
    target_wall = next(
        (
            wall
            for room in payload.rooms
            for wall in room.walls
            if target_uuid.lower() in wall.locator_id.lower()
        ),
        None,
    )
    assert target_wall is not None

    wall_xs = [corner.x for corner in target_wall.corners]
    wall_zs = [corner.z for corner in target_wall.corners]
    wall_top_y = max(corner.y for corner in target_wall.corners)

    nearby_caps = []
    for cap in payload.gaps:
        if cap.kind.value != "gap_ceiling":
            continue
        cxs = [corner.x for corner in cap.corners]
        czs = [corner.z for corner in cap.corners]
        cap_y = sum(corner.y for corner in cap.corners) / len(cap.corners)
        bbox_overlap = max(min(wall_xs) - 0.5, min(cxs)) <= min(
            max(wall_xs) + 0.5, max(cxs)
        ) and max(min(wall_zs) - 0.5, min(czs)) <= min(max(wall_zs) + 0.5, max(czs))
        if bbox_overlap and cap_y >= wall_top_y - 0.1:
            nearby_caps.append(cap)

    assert nearby_caps, "no gap_ceiling cap above the listed wall after detection"


def test_holed_gap_region_emits_single_ring_with_matching_area():
    region = Polygon(
        [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        holes=[[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]],
    )

    emitted = []
    _emit_gaps(emitted, [region], story=0, floor_y=0.0, gap_type="within_story")

    assert len(emitted) == 1
    emitted_poly = Polygon([(corner[0], corner[2]) for corner in emitted[0].corners])
    assert emitted_poly.is_valid
    assert not emitted_poly.interiors
    assert emitted[0].area_m2 == pytest.approx(emitted_poly.area, abs=0.001)
    assert emitted_poly.area == pytest.approx(region.area, abs=0.001)
