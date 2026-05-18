"""Tests for reconcile.complexity_tiers."""

from __future__ import annotations

from reconcile.complexity_tiers import (
    TIER_LABELS,
    _angle_diff_deg,
    _polygon_area_3d,
    classify_building,
    detect_gable,
)


def _building(
    stories: int, *, split_level: bool = False, rooms: list | None = None
) -> dict:
    return {
        "uuid": "test-uuid",
        "stories_found": stories,
        "split_level": split_level,
        "rooms": rooms or [{"story": s, "floor_polygon": []} for s in range(stories)],
    }


def _roof(oblique: list | None = None, flat: list | None = None) -> dict:
    return {"roof_surfaces": {"oblique": oblique or [], "flat": flat or []}}


def _square_corners(y: float = 3.0, size: float = 5.0) -> list[list[float]]:
    s = size / 2.0
    return [[-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s]]


def _oblique(az: float, incl: float, area: float = 25.0) -> dict:
    return {
        "cluster": {"avgAzimuth": az, "avgIncl": incl},
        "corners": _square_corners(size=area**0.5),
    }


class TestAngleDiff:
    def test_same(self):
        assert _angle_diff_deg(10, 10) == 0

    def test_wraparound(self):
        assert _angle_diff_deg(350, 10) == 20
        assert _angle_diff_deg(10, 350) == 20

    def test_opposite(self):
        assert _angle_diff_deg(0, 180) == 180
        assert _angle_diff_deg(90, 270) == 180


class TestPolygonArea:
    def test_degenerate(self):
        assert _polygon_area_3d([]) == 0.0
        assert _polygon_area_3d([[0, 0, 0], [1, 0, 0]]) == 0.0

    def test_unit_square(self):
        corners = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]
        assert abs(_polygon_area_3d(corners) - 1.0) < 1e-9

    def test_tilted_square(self):
        # 2x2 square tilted in the XZ plane — area still 4.
        corners = [[0, 0, 0], [2, 0, 0], [2, 2, 2], [0, 2, 2]]
        assert abs(_polygon_area_3d(corners) - (2 * (2 * 2**0.5))) < 1e-6


class TestDetectGable:
    def test_empty(self):
        assert detect_gable([]) is False

    def test_single_plane(self):
        assert detect_gable([_oblique(0, 30)]) is False

    def test_classic_gable(self):
        assert detect_gable([_oblique(0, 30), _oblique(180, 30)]) is True

    def test_opposite_but_different_pitch(self):
        assert detect_gable([_oblique(0, 30), _oblique(180, 50)]) is False

    def test_close_to_opposite(self):
        # 13° and 193° → azimuth delta 180°; within tolerance.
        assert detect_gable([_oblique(13, 22), _oblique(193, 24)]) is True

    def test_non_opposite_pair(self):
        # 90° apart, not a gable.
        assert detect_gable([_oblique(0, 30), _oblique(90, 30)]) is False

    def test_dormers_do_not_disqualify(self):
        # Two large opposite planes (area 100 each) plus a small dormer pair
        # totaling ~20 area — dominant pair still covers 200/220 = 91 %.
        big_a = _oblique(0, 30, area=100.0)
        big_b = _oblique(180, 30, area=100.0)
        dormer_a = _oblique(90, 45, area=10.0)
        dormer_b = _oblique(270, 45, area=10.0)
        assert detect_gable([big_a, big_b, dormer_a, dormer_b]) is True

    def test_small_gable_with_big_other(self):
        # Dominant pair is the big perpendicular one (fails opposite) + the
        # small gable is only 20/140 ≈ 14 % of area → reject.
        big_a = _oblique(0, 30, area=60.0)
        big_b = _oblique(90, 30, area=60.0)
        small_a = _oblique(180, 30, area=10.0)
        small_b = _oblique(0, 30, area=10.0)
        # Note: the _big_a/small_b_ pair are opposite (0 vs 180 via small_a
        # and 0), but area fraction is below 70%.
        assert detect_gable([big_a, big_b, small_a, small_b]) is False


class TestClassifyBuilding:
    def test_tier1_single_storey_flat(self):
        r = classify_building(_building(1), _roof(flat=[{"y": 3}]))
        assert r["tier"] == 1
        assert r["tier_label"] == TIER_LABELS[1]

    def test_tier1_without_any_roof_data(self):
        r = classify_building(_building(1), None)
        assert r["tier"] == 1

    def test_tier2_two_storeys_flat(self):
        r = classify_building(_building(2), _roof(flat=[{"y": 3}, {"y": 6}]))
        assert r["tier"] == 2

    def test_tier3_three_storeys_flat(self):
        r = classify_building(_building(3), _roof(flat=[{"y": 3}]))
        assert r["tier"] == 3

    def test_tier4_half_height(self):
        r = classify_building(_building(2, split_level=True), _roof())
        assert r["tier"] == 4

    def test_tier5_single_storey_slanted_no_gable(self):
        r = classify_building(_building(1), _roof(oblique=[_oblique(0, 30)]))
        assert r["tier"] == 5

    def test_tier6_gable_no_flat(self):
        roof = _roof(oblique=[_oblique(0, 30), _oblique(180, 30)])
        r = classify_building(_building(1), roof)
        # gable dominates storey count → tier 6, not tier 5
        assert r["tier"] == 6

    def test_tier6_multi_storey_gable(self):
        roof = _roof(oblique=[_oblique(0, 30), _oblique(180, 30)])
        r = classify_building(_building(2), roof)
        assert r["tier"] == 6

    def test_tier7_mixed_gable_plus_flat(self):
        roof = _roof(
            oblique=[_oblique(0, 30), _oblique(180, 30)],
            flat=[{"y": 3}],
        )
        r = classify_building(_building(2), roof)
        assert r["tier"] == 7

    def test_tier8_fallback_many_stories(self):
        # 5 storeys, flat — beyond the 1/2/3 buckets, no half-height.
        r = classify_building(_building(5), _roof(flat=[{"y": 3}]))
        assert r["tier"] == 8

    def test_tier8_fallback_oblique_without_gable_multistorey(self):
        # 2 storeys + a non-gable oblique: doesn't fit any tier cleanly.
        roof = _roof(oblique=[_oblique(0, 30)])
        r = classify_building(_building(2), roof)
        assert r["tier"] == 8

    def test_signals_present(self):
        roof = _roof(oblique=[_oblique(0, 30), _oblique(180, 30)], flat=[{"y": 3}])
        r = classify_building(_building(2), roof)
        sig = r["signals"]
        assert sig["n_stories"] == 2
        assert sig["n_oblique"] == 2
        assert sig["n_flat"] == 1
        assert sig["has_half_height"] is False
        assert sig["has_gable"] is True

    def test_falls_back_to_rooms_story_count(self):
        # No stories_found field, but rooms have story indices.
        building = {
            "uuid": "x",
            "split_level": False,
            "rooms": [{"story": 0}, {"story": 1}, {"story": 1}],
        }
        r = classify_building(building, _roof())
        assert r["signals"]["n_stories"] == 2
        assert r["tier"] == 2
