import json
from collections import Counter
from math import cos, radians, sin, sqrt, tan
from pathlib import Path

import pytest

from reconcile.complexity_tiers import classify_building as legacy_classify_building
from reconcile_tiers.classify.tiers import TIER_LABELS, classify_building, detect_gable


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


def _oblique(az: float | None, incl: float | None, area: float = 25.0) -> dict:
    if az is None or incl is None:
        size = area**0.5
        corners = [[0, 0, 0], [size, 0, 0], [size, 0, size], [0, 0, size]]
    else:
        theta = radians(az)
        slope_x, slope_z = sin(theta), cos(theta)
        ridge_x, ridge_z = cos(theta), -sin(theta)
        pitch = tan(radians(incl))
        side = sqrt(area * max(cos(radians(incl)), 1e-6))
        half = side / 2.0
        corners = []
        for u, v in [(-half, 0.0), (half, 0.0), (half, side), (-half, side)]:
            x = u * ridge_x + v * slope_x
            z = u * ridge_z + v * slope_z
            y = 1.0 - pitch * v
            corners.append([x, y, z])
    return {
        "cluster": {"avgAzimuth": az, "avgIncl": incl},
        "corners": corners,
    }


def test_flat_storey_tiers_match_existing_labels():
    assert classify_building(_building(1), _roof())["tier_label"] == TIER_LABELS[1]
    assert classify_building(_building(2), _roof())["tier"] == 2
    assert classify_building(_building(3), _roof())["tier"] == 3


def test_half_height_and_slanted_room_tiers():
    assert classify_building(_building(2, split_level=True), _roof())["tier"] == 4
    assert (
        classify_building(_building(1), _roof(oblique=[_oblique(0, 30)]))["tier"] == 5
    )


def test_gable_and_mixed_gable_tiers():
    gable = [_oblique(0, 30), _oblique(180, 30)]

    assert classify_building(_building(2), _roof(oblique=gable))["tier"] == 6
    assert (
        classify_building(_building(2), _roof(oblique=gable, flat=[{"story": 2}]))[
            "tier"
        ]
        == 7
    )


def test_invalid_oblique_meta_does_not_dilute_gable_area_fraction():
    valid_a = _oblique(0, 30, area=10.0)
    valid_b = _oblique(180, 30, area=10.0)
    invalid_large = _oblique(None, 30, area=1000.0)

    assert detect_gable([valid_a, valid_b, invalid_large]) is True


def test_low_pitch_opposing_pair_with_plausible_ridge_is_gable():
    assert detect_gable([_oblique(133.5, 8.77), _oblique(313.5, 10.1)]) is True


def _oblique_at_y(az: float, incl: float, y_offset: float, area: float = 25.0) -> dict:
    base = _oblique(az, incl, area=area)
    base["corners"] = [[c[0], c[1] + y_offset, c[2]] for c in base["corners"]]
    return base


def test_detect_gable_strips_near_flat_artefact_below_pitched_pair():
    # Real ~44° gable pair with ~9° artefact surfaces below. Without
    # stripping, the artefacts dominate the area-fraction denominator and
    # has_gable would be False.
    pitched_a = _oblique_at_y(0, 44.0, 0.5, area=25.0)
    pitched_b = _oblique_at_y(180, 44.0, 0.5, area=25.0)
    artefact_a = _oblique_at_y(0, 9.0, -1.0, area=100.0)
    artefact_b = _oblique_at_y(180, 9.0, -1.0, area=100.0)

    assert detect_gable([pitched_a, pitched_b, artefact_a, artefact_b]) is True


def test_detect_gable_keeps_legitimate_shallow_lower_tier():
    # Two-tier gambrel: 17° lower pair + 51° upper pair. Lower band is
    # not "near-flat" (>=15°) so stripping must NOT fire. Without the
    # artefact path, total_area includes both tiers — area fraction is
    # ~50% per tier, below the 70% gable threshold.
    lower_a = _oblique_at_y(0, 17.0, -2.0, area=25.0)
    lower_b = _oblique_at_y(180, 17.0, -2.0, area=25.0)
    upper_a = _oblique_at_y(0, 51.0, 0.5, area=25.0)
    upper_b = _oblique_at_y(180, 51.0, 0.5, area=25.0)

    assert detect_gable([lower_a, lower_b, upper_a, upper_b]) is False


def test_strip_helper_does_not_fire_when_low_band_sits_above_high_band():
    from reconcile_tiers.classify.tiers import _oblique_meta, _strip_artefact_lower_band

    # Cap at 10° (eligible by pitch) but geometrically ABOVE the 60°
    # slopes. The mean-y guard must keep all 4 surfaces in `valid`.
    surfaces = [
        _oblique_at_y(0, 60.0, -1.0, area=10.0),
        _oblique_at_y(180, 60.0, -1.0, area=10.0),
        _oblique_at_y(0, 10.0, 1.5, area=10.0),
        _oblique_at_y(180, 10.0, 1.5, area=10.0),
    ]
    valid = [(s, *_oblique_meta(s)) for s in surfaces]

    assert len(_strip_artefact_lower_band(valid)) == 4


def test_story_count_falls_back_to_room_indices():
    building = {
        "uuid": "x",
        "split_level": False,
        "rooms": [{"story": 0}, {"story": 1}],
    }

    result = classify_building(building, _roof())

    assert result["signals"]["n_stories"] == 2
    assert result["tier"] == 2


@pytest.mark.parametrize(
    "building,roof,expected_tier",
    [
        (_building(1), _roof(), 1),
        (_building(2), _roof(), 2),
        (_building(3), _roof(), 3),
        (_building(4), _roof(), 8),
        (_building(2, split_level=True), _roof(), 4),
        (_building(1), _roof(oblique=[_oblique(45, 20)]), 5),
        (_building(2), _roof(oblique=[_oblique(0, 30), _oblique(180, 30)]), 6),
        (
            _building(2),
            _roof(oblique=[_oblique(0, 30), _oblique(180, 30)], flat=[{"story": 1}]),
            7,
        ),
        (_building(2), _roof(oblique=[_oblique(0, 30), _oblique(90, 30)]), 8),
        (_building(1), _roof(oblique=[_oblique(0, 5), _oblique(180, 5)]), 5),
        (_building(2), _roof(oblique=[_oblique(0, 30), _oblique(150, 30)]), 6),
        (_building(2), _roof(oblique=[_oblique(0, 30), _oblique(140, 30)]), 8),
        (_building(2), _roof(oblique=[_oblique(0, 30), _oblique(180, 45)]), 8),
        (
            _building(2),
            _roof(
                oblique=[
                    _oblique(0, 30, area=10),
                    _oblique(180, 30, area=10),
                    _oblique(90, 30, area=20),
                ]
            ),
            8,
        ),
        (
            _building(2),
            _roof(
                oblique=[
                    _oblique(0, 30, area=10),
                    _oblique(180, 30, area=10),
                    _oblique(90, 30, area=5),
                ]
            ),
            6,
        ),
        (_building(2), _roof(flat=[{"story": 0}, {"story": 1}]), 2),
        (
            _building(2),
            _roof(oblique=[_oblique(0, 30), _oblique(180, 30)], flat=[{"story": 0}]),
            7,
        ),
        (
            _building(2),
            _roof(oblique=[_oblique(0, 30), _oblique(180, 30)], flat=[{"story": 2}]),
            7,
        ),
        ({"uuid": "x", "split_level": False, "rooms": []}, _roof(), 8),
        (
            {"uuid": "x", "split_level": True, "rooms": [{"story": 0}, {"story": 1}]},
            _roof(),
            4,
        ),
        (_building(1), _roof(oblique=[_oblique(None, 30)]), 5),
        (_building(2), _roof(oblique=[_oblique(None, 30), _oblique(180, 30)]), 8),
        (_building(2), _roof(oblique=[_oblique(0, None), _oblique(180, 30)]), 8),
        (_building(2), _roof(oblique=[_oblique(350, 30), _oblique(170, 30)]), 6),
        (
            _building(2),
            _roof(oblique=[_oblique(10, 30), _oblique(190, 30), _oblique(90, 20)]),
            8,
        ),
    ],
)
def test_ported_tier_classifier_cases(building, roof, expected_tier):
    assert classify_building(building, roof)["tier"] == expected_tier


def test_ridge_aware_gable_detection_documents_committed_corpus_drift():
    buildings = json.loads(Path("reconcile/buildings_3d.json").read_text())
    roofs = json.loads(Path("reconcile/roof_algorithms_py_results.json").read_text())

    legacy_counts = Counter()
    current_counts = Counter()
    drift = []
    for building in buildings:
        uuid = building["uuid"]
        legacy = legacy_classify_building(building, roofs.get(uuid))
        current = classify_building(building, roofs.get(uuid))
        legacy_counts[legacy["tier"]] += 1
        current_counts[current["tier"]] += 1
        if legacy["tier"] != current["tier"]:
            drift.append((uuid, legacy["tier"], current["tier"]))

    assert drift == [
        ("1900be91-8684-4316-98e2-c4fef6e6296f", 6, 8),
        ("287808db-3826-4351-b9a1-6f9831bdc870", 7, 8),
        ("5c557e06-393e-466e-a957-f7391b76b8ff", 5, 7),
        ("8809dcb7-86ec-480b-8b5b-3e6584918c43", 8, 7),
        ("bc2779a4-d0a2-4ba8-abbf-10129d3f82de", 6, 8),
        ("c7ee24cf-908b-4f72-bf73-78ce74a4cb50", 5, 7),
    ]
    assert legacy_counts == {1: 173, 2: 19, 3: 1, 4: 9, 5: 35, 6: 54, 7: 109, 8: 46}
    assert current_counts == {1: 173, 2: 19, 3: 1, 4: 9, 5: 33, 6: 52, 7: 111, 8: 48}


@pytest.mark.parametrize(
    "uuid",
    [
        "c72ad855-9e52-46f1-886d-a9f37911521f",
        "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
        "2ea3b759-e047-424c-8034-f8ee5b811fb4",
    ],
)
def test_cohort_tiers_match_current_legacy_classifier(uuid):
    buildings = {
        building["uuid"]: building
        for building in json.loads(Path("reconcile/buildings_3d.json").read_text())
    }
    roofs = json.loads(Path("reconcile/roof_algorithms_py_results.json").read_text())

    legacy = legacy_classify_building(buildings[uuid], roofs.get(uuid))
    current = classify_building(buildings[uuid], roofs.get(uuid))

    assert current == legacy
