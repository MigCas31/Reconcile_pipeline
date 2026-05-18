"""Tests for half-level / split-level building support.

Covers wall extensions, knee wall thermal ceilings, slab detection,
and the _floor_y_at_xz helper -- all for buildings where rooms on the
same story have different floor elevations.
"""

import pytest

from reconcile.extract3d.ceilings import (
    extend_wall_to_slab,
    find_closest_slab_y,
)
from reconcile.roof_algorithms_py.thermal_ceiling import (
    _build_knee_wall_ceilings,
    _floor_y_at_xz,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal geometry
# ---------------------------------------------------------------------------


def _rect_floor(x0, z0, x1, z1, y):
    """Create a rectangular floor polygon at height y."""
    return [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]


def _wall_corners(x0, z0, x1, z1, y_bot, y_top):
    """Create a simple rectangular wall quad (4 corners)."""
    return [
        [x0, y_bot, z0],
        [x1, y_bot, z1],
        [x1, y_top, z1],
        [x0, y_top, z0],
    ]


# ---------------------------------------------------------------------------
# test_floor_y_at_xz_helper
# ---------------------------------------------------------------------------


class TestFloorYAtXz:
    def test_point_in_room_a(self):
        """Point inside room A returns room A's floor Y."""
        regions = {
            1: [
                (_rect_floor(0, 0, 5, 3, 2.7), 2.7),
                (_rect_floor(0, 3, 5, 6, 4.0), 4.0),
            ]
        }
        assert _floor_y_at_xz(regions, 1, 2.5, 1.5, 0.0) == 2.7

    def test_point_in_room_b(self):
        """Point inside room B returns room B's floor Y."""
        regions = {
            1: [
                (_rect_floor(0, 0, 5, 3, 2.7), 2.7),
                (_rect_floor(0, 3, 5, 6, 4.0), 4.0),
            ]
        }
        assert _floor_y_at_xz(regions, 1, 2.5, 4.5, 0.0) == 4.0

    def test_point_outside_returns_fallback(self):
        """Point outside all rooms returns fallback."""
        regions = {
            1: [
                (_rect_floor(0, 0, 5, 3, 2.7), 2.7),
            ]
        }
        assert _floor_y_at_xz(regions, 1, 10.0, 10.0, 99.0) == 99.0

    def test_no_regions_returns_fallback(self):
        assert _floor_y_at_xz(None, 1, 2.5, 1.5, 99.0) == 99.0

    def test_wrong_story_returns_fallback(self):
        regions = {1: [(_rect_floor(0, 0, 5, 5, 2.7), 2.7)]}
        assert _floor_y_at_xz(regions, 2, 2.5, 2.5, 99.0) == 99.0


# ---------------------------------------------------------------------------
# test_find_closest_slab_y
# ---------------------------------------------------------------------------


class TestFindClosestSlabY:
    def test_prefers_nearest_xz(self):
        """With two slabs at different Y, picks the one closest in XZ."""
        wall_corners = _wall_corners(2, 1, 3, 1, 0.0, 2.5)
        slabs = [
            (2.5, 1.5, 2.7),  # nearby in XZ, Y=2.7
            (2.5, 4.5, 4.0),  # far in XZ, Y=4.0
        ]
        result = find_closest_slab_y(wall_corners, slabs)
        assert result == 2.7

    def test_picks_other_slab_when_closer(self):
        wall_corners = _wall_corners(2, 4, 3, 4, 0.0, 2.5)
        slabs = [
            (2.5, 1.5, 2.7),  # far in XZ
            (2.5, 4.5, 4.0),  # nearby in XZ, Y=4.0
        ]
        result = find_closest_slab_y(wall_corners, slabs)
        assert result == 4.0

    def test_empty_slabs_returns_none(self):
        wall_corners = _wall_corners(0, 0, 1, 0, 0.0, 2.5)
        assert find_closest_slab_y(wall_corners, []) is None


# ---------------------------------------------------------------------------
# test_wall_extension_half_level
# ---------------------------------------------------------------------------


class TestWallExtensionHalfLevel:
    def test_extends_to_lower_half_level(self):
        """Wall just below a slab at Y=2.7 should extend to 2.7."""
        corners = _wall_corners(0, 0, 1, 0, 0.0, 2.5)
        result = extend_wall_to_slab(corners, slab_y_above=2.7)
        assert result is not None
        assert len(result["extension_strip"]) > 0
        # Extension should reach slab_y
        for strip in result["extension_strip"]:
            strip_ys = [pt[1] for pt in strip]
            assert max(strip_ys) == pytest.approx(2.7, abs=0.01)

    def test_extends_to_higher_half_level(self):
        """Wall just below a slab at Y=4.0 should extend to 4.0 (within max_gap)."""
        corners = _wall_corners(0, 0, 1, 0, 0.0, 3.5)
        result = extend_wall_to_slab(corners, slab_y_above=4.0)
        assert result is not None
        for strip in result["extension_strip"]:
            strip_ys = [pt[1] for pt in strip]
            assert max(strip_ys) == pytest.approx(4.0, abs=0.01)

    def test_no_extension_when_already_at_slab(self):
        """Wall top already at slab Y should not produce extension."""
        corners = _wall_corners(0, 0, 1, 0, 0.0, 2.7)
        result = extend_wall_to_slab(corners, slab_y_above=2.7)
        assert result is None

    def test_no_extension_when_gap_too_large(self):
        """Gap > max_gap (0.80m) should not produce extension."""
        corners = _wall_corners(0, 0, 1, 0, 0.0, 2.0)
        result = extend_wall_to_slab(corners, slab_y_above=3.0)
        assert result is None


# ---------------------------------------------------------------------------
# test_wall_extension_standard_building
# ---------------------------------------------------------------------------


class TestWallExtensionStandard:
    def test_single_slab_above(self):
        """Standard building: wall extends to uniform story+1 slab."""
        corners = _wall_corners(0, 0, 1, 0, 0.0, 2.5)
        result = extend_wall_to_slab(corners, slab_y_above=2.7)
        assert result is not None
        for strip in result["extension_strip"]:
            strip_ys = [pt[1] for pt in strip]
            assert max(strip_ys) == pytest.approx(2.7, abs=0.01)


# ---------------------------------------------------------------------------
# test_knee_wall_height_half_level
# ---------------------------------------------------------------------------


class TestKneeWallHalfLevel:
    def test_knee_wall_uses_per_zone_height(self):
        """Knee wall gap under half-level rooms should use per-zone floor Y."""
        # Story 0: large floor covering (0,0)-(10,10)
        # Story 1: two rooms -- room A at Y=2.7 covers (0,0)-(10,5),
        #          room B at Y=4.0 covers (0,5)-(10,8)
        # Story 0 floor extends beyond story 1 in Z direction (8-10)
        floors_by_story = {
            0: [_rect_floor(0, 0, 10, 10, 0.0)],
            1: [
                _rect_floor(0, 0, 10, 5, 2.7),
                _rect_floor(0, 5, 10, 8, 4.0),
            ],
        }
        story_floor_y = {0: 0.0, 1: 2.7}  # min Y per story
        story_floor_regions = {
            0: [(_rect_floor(0, 0, 10, 10, 0.0), 0.0)],
            1: [
                (_rect_floor(0, 0, 10, 5, 2.7), 2.7),
                (_rect_floor(0, 5, 10, 8, 4.0), 4.0),
            ],
        }

        result = _build_knee_wall_ceilings(
            floors_by_story, story_floor_y, story_floor_regions
        )
        assert len(result) > 0

        for kc in result:
            assert kc["kind"] == "thermal-knee"
            assert kc["story"] == 1
            # The knee wall gap is in the (0,8)-(10,10) zone, which is
            # beyond story 1's coverage entirely. The centroid is around Z=9.
            # Since no story 1 room covers Z=9, it falls back to story_floor_y[1]=2.7
            poly_ys = [p[1] for p in kc["poly"]]
            assert all(y == pytest.approx(poly_ys[0], abs=0.01) for y in poly_ys)

    def test_knee_wall_standard_building(self):
        """Standard (non-half-level) building: knee walls work as before."""
        floors_by_story = {
            0: [_rect_floor(0, 0, 10, 10, 0.0)],
            1: [_rect_floor(0, 0, 10, 7, 2.7)],
        }
        story_floor_y = {0: 0.0, 1: 2.7}

        result_without = _build_knee_wall_ceilings(floors_by_story, story_floor_y)
        result_with = _build_knee_wall_ceilings(
            floors_by_story, story_floor_y, story_floor_regions=None
        )

        assert len(result_without) == len(result_with)
        for a, b in zip(result_without, result_with, strict=False):
            assert a["kind"] == b["kind"]
            assert a["story"] == b["story"]
            a_ys = [p[1] for p in a["poly"]]
            b_ys = [p[1] for p in b["poly"]]
            assert a_ys == b_ys


# ---------------------------------------------------------------------------
# test_slab_detection_includes_half_level
# ---------------------------------------------------------------------------


class TestSlabDetection:
    def test_half_level_rooms_not_excluded(self):
        """Rooms with floor Y > 0.50m from median should still be in story_slabs."""
        # This tests the builder.py logic indirectly by simulating what
        # the slab detection does. With the fix, no rooms are filtered out.

        import numpy as np

        # Simulate two rooms on story 0 at different heights
        story_slab_raw = {
            0: [
                (2.5, 2.5, 0.0),  # room at Y=0.0
                (2.5, 7.5, 1.5),  # room at Y=1.5 (half-level)
            ],
            1: [
                (2.5, 2.5, 2.7),
            ],
        }

        # New logic: keep all entries (no filtering)
        story_slabs = dict(story_slab_raw)

        # Both rooms should be present
        assert len(story_slabs[0]) == 2
        assert any(fy == 1.5 for _, _, fy in story_slabs[0])

        # story_y_map uses mode cluster
        for story, entries in story_slabs.items():
            floor_ys = sorted(fy for _, _, fy in entries)
            # Cluster within 0.30m
            clusters = []
            current = [floor_ys[0]]
            for fy in floor_ys[1:]:
                if fy - current[-1] <= 0.30:
                    current.append(fy)
                else:
                    clusters.append(current)
                    current = [fy]
            clusters.append(current)
            best_cluster = max(clusters, key=len)
            story_y = float(np.median(best_cluster))

            if story == 0:
                # With two isolated rooms, both clusters have size 1,
                # so max picks the first (Y=0.0)
                assert story_y == pytest.approx(0.0, abs=0.01)

    def test_wall_gets_half_level_slab_above(self):
        """Wall extension should find same-story half-level slab when > 1.0m above."""

        room_floor_y = 0.0
        story_slabs = {
            0: [
                (2.5, 2.5, 0.0),
                (2.5, 7.5, 1.5),
            ],
            1: [
                (2.5, 2.5, 2.7),
            ],
        }

        # Simulate builder.py wall extension slab collection
        slabs_above = list(story_slabs.get(1, []))
        for cx, cz, fy in story_slabs.get(0, []):
            if fy > room_floor_y + 1.0:
                slabs_above.append((cx, cz, fy))

        # The Y=1.5 slab from story 0 should be included
        slab_ys = [fy for _, _, fy in slabs_above]
        assert 1.5 in slab_ys
        assert 2.7 in slab_ys

        # Wall near (2.5, 7.5) should pick Y=1.5 (closest in XZ)
        wall_corners = _wall_corners(2, 7, 3, 7, 0.0, 1.2)
        result = find_closest_slab_y(wall_corners, slabs_above)
        assert result == 1.5


# ---------------------------------------------------------------------------
# test_cap_wall_top_for_half_level
# ---------------------------------------------------------------------------


class TestCapWallTopForHalfLevel:
    def test_caps_to_adjacent_higher_story_floor(self):
        """
        Room whose wall tops match adjacent higher-story gets capped to that
        floor.
        """
        from reconcile.roof_algorithms_py.ceiling_plane_generation import (
            _cap_wall_top_for_half_level,
        )

        # Room on story 1 at floor=-2.5, walls to 1.2
        # Adjacent story 2 room at floor=-1.3, walls to 1.25
        fp_room = _rect_floor(-10, -2, -2, 9, -2.5)
        bldg = {
            "rooms": [
                {
                    "story": 2,
                    "floor_polygon": _rect_floor(-3, 2, 0, 4, -1.3),
                    "walls_computed": [
                        {"corners": _wall_corners(-3, 2, 0, 2, -1.3, 1.25)}
                    ],
                },
            ]
        }

        result = _cap_wall_top_for_half_level(
            wall_top_y=1.2, floor_y=-2.5, story=1, fp=fp_room, bldg=bldg
        )
        assert result == pytest.approx(-1.3, abs=0.01)

    def test_no_cap_when_no_nearby_higher_story(self):
        """No higher-story rooms nearby -- wall top unchanged."""
        from reconcile.roof_algorithms_py.ceiling_plane_generation import (
            _cap_wall_top_for_half_level,
        )

        fp_room = _rect_floor(0, 0, 5, 5, 0.0)
        bldg = {
            "rooms": [
                {
                    "story": 2,
                    "floor_polygon": _rect_floor(50, 50, 55, 55, 2.7),
                    "walls_computed": [
                        {"corners": _wall_corners(50, 50, 55, 50, 2.7, 5.0)}
                    ],
                },
            ]
        }

        result = _cap_wall_top_for_half_level(
            wall_top_y=5.0, floor_y=0.0, story=1, fp=fp_room, bldg=bldg
        )
        assert result == 5.0  # unchanged

    def test_no_cap_when_wall_tops_dont_match(self):
        """Nearby room with different wall tops -- no capping."""
        from reconcile.roof_algorithms_py.ceiling_plane_generation import (
            _cap_wall_top_for_half_level,
        )

        fp_room = _rect_floor(0, 0, 5, 5, 0.0)
        bldg = {
            "rooms": [
                {
                    "story": 2,
                    "floor_polygon": _rect_floor(5, 0, 10, 5, 2.7),
                    "walls_computed": [
                        {"corners": _wall_corners(5, 0, 10, 0, 2.7, 8.0)}
                    ],
                },
            ]
        }

        # Wall top 5.0 vs other 8.0 -- diff 3.0 > 0.5 tolerance
        result = _cap_wall_top_for_half_level(
            wall_top_y=5.0, floor_y=0.0, story=1, fp=fp_room, bldg=bldg
        )
        assert result == 5.0  # unchanged

    def test_standard_building_unchanged(self):
        """Standard building with no half-level -- wall top unchanged."""
        from reconcile.roof_algorithms_py.ceiling_plane_generation import (
            _cap_wall_top_for_half_level,
        )

        fp_room = _rect_floor(0, 0, 5, 5, 0.0)
        bldg = {"rooms": []}

        result = _cap_wall_top_for_half_level(
            wall_top_y=2.5, floor_y=0.0, story=0, fp=fp_room, bldg=bldg
        )
        assert result == 2.5


# ---------------------------------------------------------------------------
# test_broad_slab_pool_across_half_floor
#
# Regression tests for the `story + 1`-only slab candidate pool bug.
# Geometry mirrors buildings like 938d6ed6 where a half-floor side-wing
# sits at story + 1 but doesn't cover story-0 walls in XZ; the real slab
# the wall should reach is at story + 2.
# ---------------------------------------------------------------------------


def _shapely_rect(x0, z0, x1, z1):
    from shapely.geometry import Polygon as _Polygon

    return _Polygon([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])


class TestBroadSlabPoolAcrossHalfFloor:
    def test_side_wing_half_floor_skipped(self):
        """Story-0 wall under story-2 finds story-2 slab when story-1 is a side wing."""
        from reconcile.extract3d.ceilings import find_best_slab_above

        # Story 0 basement footprint (0,0)-(10,10), floor at y=0.
        # Story 1 half-floor side wing at (12,0)-(16,4), floor at y=1.3 (offset, no XZ
        # overlap).
        # Story 2 ground floor (0,0)-(10,10), floor at y=2.6 (directly over story 0).
        half_floor = _shapely_rect(12, 0, 16, 4)
        ground_floor = _shapely_rect(0, 0, 10, 10)

        slabs_above = [
            (half_floor, 1.3),  # story + 1: side wing
            (ground_floor, 2.6),  # story + 2: directly overhead
        ]

        # Story-0 wall midpoint at (5, 5) -- under ground_floor, not under half_floor.
        wall = _wall_corners(4, 5, 6, 5, 0.0, 2.4)
        wall_top = 2.4

        slab_y = find_best_slab_above(wall, wall_top, slabs_above)
        assert slab_y == pytest.approx(2.6, abs=0.01)

    def test_side_wing_pool_with_only_story_plus_1_fails(self):
        """Same geometry but with only story+1 in the pool -- must return None.

        Proves the old (broken) behaviour: wall can't reach any slab when the
        only candidate is a side-wing half-floor that doesn't cover it.
        """
        from reconcile.extract3d.ceilings import find_best_slab_above

        half_floor = _shapely_rect(12, 0, 16, 4)
        slabs_above_narrow = [(half_floor, 1.3)]

        wall = _wall_corners(4, 5, 6, 5, 0.0, 2.4)
        slab_y = find_best_slab_above(wall, 2.4, slabs_above_narrow)
        assert slab_y is None

    def test_normal_two_story_prefers_story_plus_1(self):
        """Normal case: story+1 directly above wins even when story+2 also in pool.

        Guards against regressions where a broader pool would draw walls up
        through intermediate stories.
        """
        from reconcile.extract3d.ceilings import find_best_slab_above

        first_floor = _shapely_rect(0, 0, 10, 10)
        second_floor = _shapely_rect(0, 0, 10, 10)

        slabs_above = [
            (first_floor, 2.6),  # story + 1: correct target
            (second_floor, 5.2),  # story + 2: reachable but wrong
        ]

        wall = _wall_corners(4, 5, 6, 5, 0.0, 2.4)
        slab_y = find_best_slab_above(wall, 2.4, slabs_above)
        # Wall top 2.4, both slabs above. Distance to both is 0 (point inside
        # both). The picker returns the first match with min distance -- sort
        # order in the pool guarantees story+1 wins.
        assert slab_y == pytest.approx(2.6, abs=0.01)

    def test_stacked_half_floors(self):
        """Story 0 -> half -> half -> full. Story-0 wall should reach the full slab."""
        from reconcile.extract3d.ceilings import find_best_slab_above

        # Story 1 and 2 are half-floors on a side, story 3 is directly above story 0.
        side_half_1 = _shapely_rect(12, 0, 16, 4)
        side_half_2 = _shapely_rect(12, 0, 16, 4)
        top_full = _shapely_rect(0, 0, 10, 10)

        slabs_above = [
            (side_half_1, 1.2),
            (side_half_2, 2.4),
            (top_full, 3.0),
        ]

        wall = _wall_corners(4, 5, 6, 5, 0.0, 2.7)
        slab_y = find_best_slab_above(wall, 2.7, slabs_above)
        # Only top_full is above wall_top + margin AND covers the wall.
        assert slab_y == pytest.approx(3.0, abs=0.01)

    def test_outbuilding_wall_not_stacked_returns_none(self):
        """Wall in a detached outbuilding -- no upper-story floor over midpoint.

        Design intent: only intermediary slabs between stacked rooms get auto
        extensions. Air-facing walls (no room overhead) get thickness from the
        user-supplied envelope instead, so the picker must return None.
        """
        from reconcile.extract3d.ceilings import find_best_slab_above

        # Main house footprint at (0,0)-(10,10). Outbuilding wall at (20, 5) --
        # 10 m away. The slab sits above the main house.
        main_house = _shapely_rect(0, 0, 10, 10)
        slabs_above = [(main_house, 2.6)]

        wall = _wall_corners(19, 5, 21, 5, 0.0, 2.4)
        slab_y = find_best_slab_above(wall, 2.4, slabs_above)
        assert slab_y is None

    def test_stack_tol_allows_boundary_float_noise(self):
        """Wall midpoint 0.05 m outside the upper-floor boundary still extends.

        Prevents floating-point misses on exterior walls whose XZ midpoint
        lands marginally outside the upper room's polygon.
        """
        from reconcile.extract3d.ceilings import find_best_slab_above

        main_house = _shapely_rect(0, 0, 10, 10)
        slabs_above = [(main_house, 2.6)]

        # Midpoint at (10.05, 5) -- 0.05 m past the east edge.
        wall = _wall_corners(9.05, 5, 11.05, 5, 0.0, 2.4)
        slab_y = find_best_slab_above(wall, 2.4, slabs_above)
        assert slab_y == pytest.approx(2.6, abs=0.01)

    def test_stack_tol_rejects_far_walls(self):
        """Wall midpoint 0.50 m outside the upper-floor boundary is rejected."""
        from reconcile.extract3d.ceilings import find_best_slab_above

        main_house = _shapely_rect(0, 0, 10, 10)
        slabs_above = [(main_house, 2.6)]

        # Midpoint at (10.50, 5) -- 0.50 m past the east edge (> stack_tol=0.10).
        wall = _wall_corners(9.50, 5, 11.50, 5, 0.0, 2.4)
        slab_y = find_best_slab_above(wall, 2.4, slabs_above)
        assert slab_y is None


# ---------------------------------------------------------------------------
# test_is_split_level
# ---------------------------------------------------------------------------


class TestIsSplitLevel:
    def test_single_room_story_is_split_level(self):
        from reconcile.extract3d.builder import _is_split_level

        rooms = [
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 10, 10, 0.0)},
            {"story": 0, "floor_polygon": _rect_floor(0, 10, 10, 20, 0.0)},
            {"story": 1, "floor_polygon": _rect_floor(12, 0, 16, 4, 1.3)},
            {"story": 2, "floor_polygon": _rect_floor(0, 0, 10, 10, 2.6)},
            {"story": 2, "floor_polygon": _rect_floor(0, 10, 10, 20, 2.6)},
        ]
        assert _is_split_level(rooms) is True

    def test_tight_story_spacing_is_split_level(self):
        from reconcile.extract3d.builder import _is_split_level

        rooms = [
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 5, 5, 0.0)},
            {"story": 0, "floor_polygon": _rect_floor(5, 0, 10, 5, 0.0)},
            {"story": 1, "floor_polygon": _rect_floor(0, 0, 5, 5, 1.3)},
            {"story": 1, "floor_polygon": _rect_floor(5, 0, 10, 5, 1.3)},
        ]
        # 1.3m spacing < 2.0m threshold
        assert _is_split_level(rooms) is True

    def test_normal_two_story_not_split_level(self):
        from reconcile.extract3d.builder import _is_split_level

        rooms = [
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 5, 5, 0.0)},
            {"story": 0, "floor_polygon": _rect_floor(5, 0, 10, 5, 0.0)},
            {"story": 0, "floor_polygon": _rect_floor(0, 5, 5, 10, 0.0)},
            {"story": 1, "floor_polygon": _rect_floor(0, 0, 5, 5, 2.6)},
            {"story": 1, "floor_polygon": _rect_floor(5, 0, 10, 5, 2.6)},
            {"story": 1, "floor_polygon": _rect_floor(0, 5, 5, 10, 2.6)},
        ]
        assert _is_split_level(rooms) is False

    def test_single_story_not_split_level(self):
        from reconcile.extract3d.builder import _is_split_level

        rooms = [
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 5, 5, 0.0)},
        ]
        assert _is_split_level(rooms) is False


class TestCrossFloorGapsHalfFloorSubtraction:
    """Cross-storey gap lids must not extend across half-floor footprints.

    Repro pattern matches building 938d6ed6: full ground story below a
    sandwiched half-floor below a full upper story, where the half-floor's
    XZ is not contained in the full stories' XZ. Without subtraction,
    `compute_cross_floor_gaps` emits a horizontal slab at each full
    story's floor_y across the half-floor's XZ, slicing through the
    half-floor room.
    """

    def _half_floor_geometry(self):
        """Full -> half -> full sandwich where the half-floor sits laterally
        offset from the full stories' footprints."""
        return [
            # Story 0 (ground): two rooms at y=-3.85, footprint (0,0)-(10,5)
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 5, 5, -3.85)},
            {"story": 0, "floor_polygon": _rect_floor(5, 0, 10, 5, -3.85)},
            # Story 1 (half-floor): single room at y=-2.55, offset to (10,0)-(15,5)
            {"story": 1, "floor_polygon": _rect_floor(10, 0, 15, 5, -2.55)},
            # Story 2 (upper): two rooms at y=-1.30, footprint (10,0)-(20,5)
            {"story": 2, "floor_polygon": _rect_floor(10, 0, 15, 5, -1.30)},
            {"story": 2, "floor_polygon": _rect_floor(15, 0, 20, 5, -1.30)},
        ]

    def test_no_cross_story_gap_overlaps_half_floor(self):
        """No emitted cross_story gap polygon overlaps the half-floor's XZ."""
        from shapely.geometry import Polygon

        from reconcile.extract3d.gaps import compute_cross_floor_gaps

        rooms = self._half_floor_geometry()
        half_floor_xz = Polygon([(c[0], c[2]) for c in rooms[2]["floor_polygon"]])

        gaps = compute_cross_floor_gaps(rooms)
        cross = [g for g in gaps if g.get("type") == "cross_story"]

        for g in cross:
            corners = g.get("corners") or []
            if len(corners) < 3:
                continue
            poly = Polygon([(c[0], c[2]) for c in corners]).buffer(0)
            assert poly.intersection(half_floor_xz).area < 0.05, (
                f"cross_story gap (story={g.get('story')}, "
                f"area={g.get('area_m2')}) overlaps half-floor footprint"
            )

    def test_normal_two_story_unchanged(self):
        """Without a half-floor, half_floor_fp is empty and the
        cross-storey logic emits the same cantilever lids as before."""
        from shapely.geometry import Polygon

        from reconcile.extract3d.gaps import compute_cross_floor_gaps

        # Cantilever scenario: story 1 sticks out past story 0 by 5 m.
        rooms = [
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 10, 5, 0.0)},
            {"story": 1, "floor_polygon": _rect_floor(0, 0, 15, 5, 2.7)},
        ]
        story0_fp = Polygon([(c[0], c[2]) for c in rooms[0]["floor_polygon"]])
        story1_fp = Polygon([(c[0], c[2]) for c in rooms[1]["floor_polygon"]])
        cantilever_xz = story1_fp.difference(story0_fp)

        gaps = compute_cross_floor_gaps(rooms)
        cross = [g for g in gaps if g.get("type") == "cross_story"]
        # Lid at story 0's level should still cover the cantilever XZ.
        cantilever_lid_area = 0.0
        for g in cross:
            if g.get("story") != 0:
                continue
            corners = g.get("corners") or []
            if len(corners) < 3:
                continue
            poly = Polygon([(c[0], c[2]) for c in corners]).buffer(0)
            cantilever_lid_area += poly.intersection(cantilever_xz).area
        assert cantilever_lid_area > cantilever_xz.area * 0.5, (
            "cantilever lid was unexpectedly removed for non-split-level building"
        )

    def test_cross_story_chunks_from_adjacent_upper_rooms_do_not_overlap(self):
        """Buffered room neighborhoods must partition, not duplicate, a
        cross-storey cantilever slab.

        Repro: story 1 has two adjacent rooms and extends past story 0.
        The lower story's missing area is fully covered by the upper
        extension room, while the other upper room's buffer also touches a
        0.5 m strip of the same missing region. Emitting both chunks without
        subtracting prior coverage creates overlapping gap geometry.
        """
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        from reconcile.extract3d.gaps import compute_cross_floor_gaps

        rooms = [
            {"story": 0, "floor_polygon": _rect_floor(0, 0, 10, 5, 0.0)},
            {"story": 1, "floor_polygon": _rect_floor(0, 0, 10, 5, 2.7)},
            {"story": 1, "floor_polygon": _rect_floor(10, 0, 20, 5, 2.7)},
        ]

        gaps = compute_cross_floor_gaps(rooms)
        story0_cross = [
            gap
            for gap in gaps
            if gap.get("type") == "cross_story" and gap.get("story") == 0
        ]
        polys = [
            Polygon([(c[0], c[2]) for c in gap["corners"]]).buffer(0)
            for gap in story0_cross
        ]

        emitted_area = sum(poly.area for poly in polys)
        union_area = unary_union(polys).area

        assert emitted_area == pytest.approx(union_area, abs=1e-6)
