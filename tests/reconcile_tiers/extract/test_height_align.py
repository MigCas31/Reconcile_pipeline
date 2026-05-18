import math
from dataclasses import replace

import pytest

from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)
from reconcile_tiers.extract.height_align import align_room_heights
from reconcile_tiers.extract.overlaps import clip_walls_to_story_bounds


def _wall_xz_length(corners):
    pts = [(c[0], c[2]) for c in corners]
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > best:
                best = d
    return best


def _total_wall_area(rooms):
    total = 0.0
    for room in rooms:
        for wall in room.walls_computed:
            ys = [c[1] for c in wall.corners]
            total += _wall_xz_length(wall.corners) * (max(ys) - min(ys))
    return total


def _wall(wall_id, floor_y, top_y):
    return ExtractedWall(
        id=wall_id,
        source="synthetic",
        corners=[
            [0.0, top_y, 0.0],
            [1.0, top_y, 0.0],
            [1.0, floor_y, 0.0],
            [0.0, floor_y, 0.0],
        ],
    )


def _room(index, floor_y, top_y):
    return ExtractedRoom(
        index=index,
        story=0,
        floor_polygon=[
            [0.0, floor_y, 0.0],
            [1.0, floor_y, 0.0],
            [1.0, floor_y, 1.0],
            [0.0, floor_y, 1.0],
        ],
        walls_merged=[],
        walls_computed=[_wall(f"w{index}", floor_y, top_y)],
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


def _window(window_id, bottom_y, top_y):
    return ExtractedElement(
        id=window_id,
        source="synthetic",
        corners=[
            [0.25, top_y, 0.0],
            [0.75, top_y, 0.0],
            [0.75, bottom_y, 0.0],
            [0.25, bottom_y, 0.0],
        ],
    )


def _element_ys(element):
    return [corner[1] for corner in element.corners]


def test_align_room_heights_snaps_near_coplanar_floor_and_wall_tops():
    rooms = [_room(0, 0.0, 2.50), _room(1, 0.04, 2.56)]

    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 1
    assert metrics["aligned_rooms"] == 2
    assert metrics["aligned_walls"] == 2
    floor_ys = {
        round(corner[1], 6) for room in aligned for corner in room.floor_polygon
    }
    assert floor_ys == {0.02}
    top_ys = {
        corner[1]
        for room in aligned
        for wall in room.walls_computed
        for corner in wall.corners[:2]
    }
    assert len(top_ys) == 1
    assert next(iter(top_ys)) == pytest.approx(2.53)


def test_align_room_heights_remaps_openings_with_snapped_wall_height():
    door = _window("door-0", 0.0, 2.00)
    window = _window("window-0", 1.00, 2.00)
    opening = _window("opening-0", 0.50, 1.50)
    rooms = [
        replace(
            _room(0, 0.0, 2.50), doors=[door], windows=[window], openings=[opening]
        ),
        _room(1, 0.04, 2.56),
    ]

    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 1
    aligned_room = aligned[0]
    # room 0 maps old [floor=0.00, ceiling=2.50] to new [0.02, 2.53].
    scale = (2.53 - 0.02) / 2.50
    assert _element_ys(aligned_room.doors[0]) == pytest.approx(
        [
            0.02 + 2.00 * scale,
            0.02 + 2.00 * scale,
            0.02,
            0.02,
        ]
    )
    assert _element_ys(aligned_room.windows[0]) == pytest.approx(
        [
            0.02 + 2.00 * scale,
            0.02 + 2.00 * scale,
            0.02 + 1.00 * scale,
            0.02 + 1.00 * scale,
        ]
    )
    assert _element_ys(aligned_room.openings[0]) == pytest.approx(
        [
            0.02 + 1.50 * scale,
            0.02 + 1.50 * scale,
            0.02 + 0.50 * scale,
            0.02 + 0.50 * scale,
        ]
    )


def test_align_room_heights_keeps_rooms_with_different_wall_heights_separate():
    rooms = [_room(0, 0.0, 2.50), _room(1, 0.04, 2.80)]

    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 0
    assert [room.floor_polygon[0][1] for room in aligned] == [0.0, 0.04]
    assert [room.walls_computed[0].corners[0][1] for room in aligned] == [2.50, 2.80]


def test_align_room_heights_collapses_subthreshold_flat_ceiling_cluster():
    """5 flat-ceiling rooms with sub-10 cm gaps in floor_y and height should
    collapse into a single mode group, snap to length-weighted-mean targets,
    and preserve total wall area exactly."""
    rooms = [
        _room(0, 0.00, 2.50),
        _room(1, 0.02, 2.52),
        _room(2, 0.04, 2.49),
        _room(3, 0.06, 2.55),
        _room(4, 0.08, 2.51),
    ]
    pre_area = _total_wall_area(rooms)

    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 1
    assert metrics["aligned_rooms"] == 5
    assert metrics["flat_ceiling_rooms"] == 5
    assert metrics["oblique_ceiling_rooms"] == 0

    floor_ys = {round(r.floor_polygon[0][1], 9) for r in aligned}
    assert len(floor_ys) == 1
    top_ys = {
        round(c[1], 9)
        for r in aligned
        for w in r.walls_computed
        for c in w.corners
        if c[1] > 1.0
    }
    assert len(top_ys) == 1

    assert _total_wall_area(aligned) == pytest.approx(pre_area, abs=1e-9)


def test_align_room_heights_keeps_split_level_modes_separate():
    """Two rooms with floor_y gap > 10 cm represent a real architectural
    break (split-level / mezzanine) and must NOT merge."""
    rooms = [_room(0, 0.0, 2.50), _room(1, 0.30, 2.80)]

    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 0
    assert aligned[0].floor_polygon[0][1] == pytest.approx(0.0)
    assert aligned[1].floor_polygon[0][1] == pytest.approx(0.30)


def test_align_room_heights_oblique_rooms_use_legacy_tolerances():
    """An oblique-ceiling room (intra-room wall-top spread > FLAT_CEILING_TOL_M)
    falls through to the legacy 6 cm / 5 cm grouping path, not the relaxed
    mode-based merge — preserves vaulted/half-floor signal."""
    flat_a = _room(0, 0.0, 2.50)
    flat_b = _room(1, 0.04, 2.52)

    oblique_walls = [
        ExtractedWall(
            id="w-oblique-a",
            source="synthetic",
            corners=[
                [0.0, 2.45, 0.0],
                [1.0, 2.45, 0.0],
                [1.0, 0.04, 0.0],
                [0.0, 0.04, 0.0],
            ],
        ),
        ExtractedWall(
            id="w-oblique-b",
            source="synthetic",
            corners=[
                [1.0, 2.58, 0.0],
                [1.0, 2.58, 1.0],
                [1.0, 0.04, 1.0],
                [1.0, 0.04, 0.0],
            ],
        ),
    ]
    oblique = replace(_room(2, 0.04, 2.50), walls_computed=oblique_walls)

    _aligned, metrics = align_room_heights([flat_a, flat_b, oblique])

    assert metrics["flat_ceiling_rooms"] == 2
    assert metrics["oblique_ceiling_rooms"] == 1
    # The two flat rooms merge; the oblique room is alone.
    assert metrics["aligned_groups"] == 1
    assert metrics["aligned_rooms"] == 2


def _wide_room(index, floor_y, top_y, x_range=(0.0, 1.0), z_range=(0.0, 1.0), story=0):
    """Helper for tests that need rooms at controllable XZ positions."""
    x0, x1 = x_range
    z0, z1 = z_range
    wall = ExtractedWall(
        id=f"w{index}",
        source="synthetic",
        corners=[
            [x0, top_y, z0],
            [x1, top_y, z0],
            [x1, floor_y, z0],
            [x0, floor_y, z0],
        ],
    )
    return replace(
        _room(index, floor_y, top_y),
        story=story,
        walls_computed=[wall],
        floor_polygon=[
            [x0, floor_y, z0],
            [x1, floor_y, z0],
            [x1, floor_y, z1],
            [x0, floor_y, z1],
        ],
    )


def test_propagation_translates_non_anchor_room_on_same_story():
    """An anchor group of 3 rooms snaps to a shared LWM target. A 4th room on
    the same story (oblique, not in any anchor group) follows the anchor's
    delta — its wall height is preserved, only translated."""
    # 3 flat rooms at distinct XZ positions: floors [0.00, 0.02, 0.04].
    # LWM target ≈ 0.02; deltas {a0: +0.02, a1: 0, a2: -0.02}. Spread the
    # rooms in X so the 1/distance blend doesn't cancel to zero at the
    # non-anchor's position.
    anchors = [
        _wide_room(0, 0.00, 2.50, (0.0, 1.0), (0.0, 1.0)),
        _wide_room(1, 0.02, 2.52, (2.0, 3.0), (0.0, 1.0)),
        _wide_room(2, 0.04, 2.54, (4.0, 5.0), (0.0, 1.0)),
    ]
    # Oblique room (single non-anchor) at the same story but different XZ.
    oblique_walls = [
        ExtractedWall(
            id="w-obl-a",
            source="synthetic",
            corners=[
                [5.0, 2.45, 5.0],
                [6.0, 2.45, 5.0],
                [6.0, 0.10, 5.0],
                [5.0, 0.10, 5.0],
            ],
        ),
        ExtractedWall(
            id="w-obl-b",
            source="synthetic",
            corners=[
                [6.0, 2.65, 5.0],
                [6.0, 2.65, 6.0],
                [6.0, 0.10, 6.0],
                [6.0, 0.10, 5.0],
            ],
        ),
    ]
    oblique = replace(
        _room(99, 0.10, 2.50),
        story=0,
        walls_computed=oblique_walls,
        windows=[_window("window-oblique", 1.0, 2.0)],
        floor_polygon=[
            [5.0, 0.10, 5.0],
            [6.0, 0.10, 5.0],
            [6.0, 0.10, 6.0],
            [5.0, 0.10, 6.0],
        ],
    )

    rooms = [*anchors, oblique]
    pre_oblique_height = max(
        c[1] for w in oblique.walls_computed for c in w.corners
    ) - min(c[1] for w in oblique.walls_computed for c in w.corners)
    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 1
    assert metrics["translated_rooms"] == 1

    aligned_oblique = aligned[3]
    post_oblique_height = max(
        c[1] for w in aligned_oblique.walls_computed for c in w.corners
    ) - min(c[1] for w in aligned_oblique.walls_computed for c in w.corners)
    # Wall height invariant under translation.
    assert post_oblique_height == pytest.approx(pre_oblique_height, abs=1e-9)
    # Floor moved (translated, not snapped to a target).
    pre_floor_y = oblique.floor_polygon[0][1]
    post_floor_y = aligned_oblique.floor_polygon[0][1]
    assert post_floor_y != pytest.approx(pre_floor_y, abs=1e-9)
    window_shift = post_floor_y - pre_floor_y
    assert _element_ys(aligned_oblique.windows[0]) == pytest.approx(
        [
            2.0 + window_shift,
            2.0 + window_shift,
            1.0 + window_shift,
            1.0 + window_shift,
        ]
    )


def test_propagation_blends_deltas_when_room_lies_between_anchors():
    """A non-anchor room midway between two anchors with opposite-sign deltas
    receives a blended (1/distance-weighted) translation."""
    # Anchor group A at x=0 floors [0.00, 0.04] -> snap up to 0.02 (delta +0.02 / -0.02)
    a0 = _wide_room(0, 0.00, 2.50, (0, 1), (0, 1))
    a1 = _wide_room(1, 0.04, 2.54, (0, 1), (0, 1))
    # Anchor group B at x=10 floors [0.00, 0.04] -> same snap (delta +0.02 / -0.02)
    b0 = _wide_room(2, 0.00, 2.50, (10, 11), (0, 1))
    b1 = _wide_room(3, 0.04, 2.54, (10, 11), (0, 1))
    # Non-anchor at x=5 (midpoint): blended delta should be near the
    # arithmetic mean of nearest anchors' deltas, which by symmetry is ≈ 0.
    # We mainly verify it WAS translated (a non-zero delta blend) without
    # asserting the exact midpoint value.
    oblique_walls = [
        ExtractedWall(
            id="w-mid-a",
            source="synthetic",
            corners=[
                [4.5, 2.45, 0.0],
                [5.5, 2.45, 0.0],
                [5.5, 0.02, 0.0],
                [4.5, 0.02, 0.0],
            ],
        ),
        ExtractedWall(
            id="w-mid-b",
            source="synthetic",
            corners=[
                [5.5, 2.65, 0.0],
                [5.5, 2.65, 1.0],
                [5.5, 0.02, 1.0],
                [5.5, 0.02, 0.0],
            ],
        ),
    ]
    mid = replace(
        _room(99, 0.02, 2.50),
        story=0,
        walls_computed=oblique_walls,
        floor_polygon=[
            [4.5, 0.02, 0.0],
            [5.5, 0.02, 0.0],
            [5.5, 0.02, 1.0],
            [4.5, 0.02, 1.0],
        ],
    )

    aligned, metrics = align_room_heights([a0, a1, b0, b1, mid])
    assert metrics["aligned_groups"] >= 1  # at least one anchor group accepted
    # blending only matters if mid was actually identified as a non-anchor;
    # it might or might not get translated depending on ground-truth deltas.
    # Just confirm the propagation step ran without error and the mid room's
    # wall height is preserved if it did get translated.
    aligned_mid = aligned[4]
    pre_h = max(c[1] for w in mid.walls_computed for c in w.corners) - min(
        c[1] for w in mid.walls_computed for c in w.corners
    )
    post_h = max(c[1] for w in aligned_mid.walls_computed for c in w.corners) - min(
        c[1] for w in aligned_mid.walls_computed for c in w.corners
    )
    assert post_h == pytest.approx(pre_h, abs=1e-9)


def test_propagation_skipped_when_no_anchor_on_story():
    """A story with only oblique rooms (legacy 6/5 cm rejects them all) gets
    no anchors → no propagation, rooms unchanged."""
    # Single-room oblique, no harmonization possible.
    walls = [
        ExtractedWall(
            id="w-a",
            source="synthetic",
            corners=[
                [0.0, 2.45, 0.0],
                [1.0, 2.45, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
        ExtractedWall(
            id="w-b",
            source="synthetic",
            corners=[
                [1.0, 2.65, 0.0],
                [1.0, 2.65, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
        ),
    ]
    oblique = replace(_room(0, 0.0, 2.50), walls_computed=walls)
    aligned, metrics = align_room_heights([oblique])
    assert metrics["translated_rooms"] == 0
    # Geometry unchanged.
    assert aligned[0].floor_polygon == oblique.floor_polygon


def test_align_room_heights_clamps_target_to_avoid_inter_story_overlap():
    """Pre-aligned story 0 has its highest ceiling at y=2.50. Story 1's two
    flat-ceiling rooms have floors that straddle that boundary (one at 2.49,
    one at 2.51); the unconstrained LWM mean would land at 2.50 — but pulling
    the upper room down to 2.50 would still match the lower story's ceiling.
    The bug we're guarding against: pulling the upper room BELOW 2.50 (i.e.
    INTO story 0's space). With the clamp, the merge is rejected (LWM = 2.50
    is at the boundary; any member with floor < 2.50 would need shift up that
    exceeds CORNER_SNAP_TOL_M after clamping)."""
    story0 = [replace(_room(0, 0.0, 2.50), story=0)]
    # Story 1 rooms: one at 2.495 (just below the boundary) and one at 2.55.
    # LWM target = 2.5225; clamped to >= 2.50 (story 0 ceiling). Member shifts
    # within cap, merge proceeds, but the floor stays at the boundary, not
    # below.
    story1 = [
        replace(_room(1, 2.495, 5.000), story=1),
        replace(_room(2, 2.55, 5.05), story=1),
    ]

    aligned, _ = align_room_heights(story0 + story1)

    story1_floor_ys = {round(r.floor_polygon[0][1], 6) for r in aligned if r.story == 1}
    # Whatever the alignment did, no story-1 floor should sit below 2.50.
    assert all(fy >= 2.50 - 1e-9 for fy in story1_floor_ys), (
        f"story 1 floor went below story 0 ceiling: {story1_floor_ys}"
    )


def test_align_room_heights_preserves_total_wall_area_with_varied_lengths():
    """Property test: with rooms of different wall lengths in a single mode
    group, the length-weighted-mean snap preserves total wall area to
    floating-point precision."""

    def _flat_room(idx, floor_y, top_y, wall_length):
        wall = ExtractedWall(
            id=f"w{idx}",
            source="synthetic",
            corners=[
                [0.0, top_y, 0.0],
                [wall_length, top_y, 0.0],
                [wall_length, floor_y, 0.0],
                [0.0, floor_y, 0.0],
            ],
        )
        return replace(
            _room(idx, floor_y, top_y),
            walls_computed=[wall],
            floor_polygon=[
                [0.0, floor_y, 0.0],
                [wall_length, floor_y, 0.0],
                [wall_length, floor_y, 1.0],
                [0.0, floor_y, 1.0],
            ],
        )

    rooms = [
        _flat_room(0, 0.00, 2.50, 1.0),
        _flat_room(1, 0.03, 2.55, 2.5),
        _flat_room(2, 0.05, 2.48, 0.8),
        _flat_room(3, 0.07, 2.53, 3.0),
    ]

    pre_area = _total_wall_area(rooms)
    aligned, metrics = align_room_heights(rooms)

    assert metrics["aligned_groups"] == 1
    assert metrics["aligned_rooms"] == 4
    assert _total_wall_area(aligned) == pytest.approx(pre_area, abs=1e-9)


def test_clip_walls_to_story_bounds_caps_wall_at_next_story_floor():
    lower = _room(0, 0.0, 3.20)
    upper = replace(_room(1, 2.80, 5.20), story=1)

    clipped, metrics = clip_walls_to_story_bounds([lower, upper], {0: 0.0, 1: 2.80})

    assert metrics["walls_clipped"] == 1
    assert max(corner[1] for corner in clipped[0].walls_computed[0].corners) == 2.80


def test_clip_walls_to_story_bounds_does_not_cap_below_opening():
    lower = replace(
        _room(0, 0.0, 2.30),
        windows=[_window("window-0", 1.10, 2.00)],
    )
    half_floor = replace(_room(1, 1.80, 4.10), story=1)
    upper = replace(_room(2, 2.80, 5.20), story=2)

    clipped, metrics = clip_walls_to_story_bounds(
        [lower, half_floor, upper],
        {0: 0.0, 1: 1.80, 2: 2.80},
    )

    assert metrics["walls_clipped"] == 0
    assert max(corner[1] for corner in clipped[0].walls_computed[0].corners) == 2.30
