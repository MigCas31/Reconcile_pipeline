from math import cos, radians, sin, sqrt, tan

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.classify.roof_type import classify_oblique_roof
from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.payload.schema import RoofType
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster, build_roof_model


def _surface(azimuth, incl=35.0, area_scale=1.0, corners=None):
    if corners is None:
        theta = radians(azimuth)
        slope_x, slope_z = sin(theta), cos(theta)
        ridge_x, ridge_z = cos(theta), -sin(theta)
        pitch = tan(radians(incl))
        side = sqrt(area_scale * max(cos(radians(incl)), 1e-6))
        half = side / 2.0
        corners = []
        for u, v in [(-half, 0.0), (half, 0.0), (half, side), (-half, side)]:
            x = u * ridge_x + v * slope_x
            z = u * ridge_z + v * slope_z
            y = 1.0 - pitch * v
            corners.append([x, y, z])
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        plane = Plane(a=0.0, b=1.0, c=0.0, d=0.0)
    return ObliqueSurface(
        corners=corners,
        plane=plane,
        cluster=RoofCluster(
            segments=[], avg_incl=incl, avg_azimuth=azimuth, ref_pt=[0, 0, 0]
        ),
        dominant_story=0,
        ridge={},
    )


def test_roof_type_none_and_shed():
    assert classify_oblique_roof([]) == RoofType.NONE
    assert classify_oblique_roof([_surface(90)]) == RoofType.SHED


def test_roof_type_gable_from_dominant_opposing_pair():
    assert classify_oblique_roof([_surface(90), _surface(270)]) == RoofType.GABLE


def test_roof_type_cross_gable_from_perpendicular_opposing_pairs():
    roof = [_surface(0), _surface(180), _surface(90), _surface(270)]

    assert classify_oblique_roof(roof) == RoofType.CROSS_GABLE


def test_roof_type_hip_from_two_non_perpendicular_opposing_pairs():
    roof = [_surface(20), _surface(200), _surface(65), _surface(245)]

    assert classify_oblique_roof(roof) == RoofType.HIP


def test_roof_type_mansard_from_two_pitch_bands():
    roof = [
        _surface(0, 30),
        _surface(180, 31),
        _surface(90, 30),
        _surface(270, 31),
        _surface(45, 70),
        _surface(225, 70),
    ]

    assert classify_oblique_roof(roof) == RoofType.MANSARD


def test_roof_type_pyramid_from_common_apex():
    apex = [0.0, 3.0, 0.0]
    roof = [
        _surface(20, corners=[[-1, 0, -1], [1, 0, -1], apex]),
        _surface(120, corners=[[1, 0, -1], [1, 0, 1], apex]),
        _surface(220, corners=[[1, 0, 1], [-1, 0, 1], apex]),
        _surface(320, corners=[[-1, 0, 1], [-1, 0, -1], apex]),
    ]

    assert classify_oblique_roof(roof) == RoofType.PYRAMID


def test_roof_type_complex_fallback():
    assert (
        classify_oblique_roof([_surface(0), _surface(70), _surface(150)])
        == RoofType.COMPLEX
    )


def _surface_at_y(azimuth, incl, y_offset):
    surface = _surface(azimuth, incl=incl)
    shifted = [[c[0], c[1] + y_offset, c[2]] for c in surface.corners]
    plane = Plane.fit(shifted)
    if isinstance(plane, FitFailure):
        plane = surface.plane
    return ObliqueSurface(
        corners=shifted,
        plane=plane,
        cluster=surface.cluster,
        dominant_story=surface.dominant_story,
        ridge=surface.ridge,
    )


def test_near_flat_artefact_below_pitched_pair_classifies_as_gable():
    # The reported e9f0631f case: a real ~44° gable pair + two near-flat
    # surfaces (~9° pitch) that sit *below* the gable peak (attic-floor
    # fragments, dormer scraps). Old rule 7 returned MANSARD; new rule
    # strips the artefact band and recovers GABLE.
    roof = [
        _surface_at_y(0, 44.0, 0.5),  # pitched gable face, mid-y
        _surface_at_y(180, 44.0, 0.5),  # opposing gable face, mid-y
        _surface_at_y(0, 9.0, -1.0),  # near-flat artefact, below
        _surface_at_y(180, 9.0, -1.0),  # near-flat artefact, below
    ]

    assert classify_oblique_roof(roof) == RoofType.GABLE


def test_legitimate_shallow_lower_tier_is_not_stripped():
    # Two-tier (gambrel-style) roof with a ~17° lower pair and a ~50°
    # upper pair. Both bands are real shedding slopes. The artefact
    # stripper requires lower band pitch < 15°, so it should NOT fire,
    # and the building should fall through to the existing rules and
    # land at MANSARD or COMPLEX (not GABLE).
    roof = [
        _surface_at_y(0, 17.0, -2.0),
        _surface_at_y(180, 17.0, -2.0),
        _surface_at_y(0, 51.0, 0.5),
        _surface_at_y(180, 51.0, 0.5),
    ]

    assert classify_oblique_roof(roof) != RoofType.GABLE


def test_mean_y_guard_prevents_artefact_strip_when_low_band_sits_above_high_band():
    # Inverted-mansard: low-pitch surfaces sit *above* the steep slopes.
    # The mean-y guard must NOT strip them (they are a real cap, not
    # artefact), so the original 4-surface set reaches the existing HIP
    # rule and classifies as HIP — not GABLE.
    roof = [
        _surface_at_y(0, 60.0, -1.0),
        _surface_at_y(180, 60.0, -1.0),
        _surface_at_y(0, 10.0, 1.5),
        _surface_at_y(180, 10.0, 1.5),
    ]

    # Stripping the low-pitch band would leave the 60° pair and
    # classify as GABLE. The guard must prevent that.
    assert classify_oblique_roof(roof) != RoofType.GABLE


def test_roof_type_matches_documented_hip_and_mansard_cohort_picks():
    hip_model = extract_building_model(
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
        "pipeline-outputs",
        ".scan-cache",
    )
    mansard_model = extract_building_model(
        "52f91e67-3891-4729-8bf3-be2c0a6a0d04",
        "pipeline-outputs",
        ".scan-cache",
    )

    assert classify_oblique_roof(build_roof_model(hip_model).oblique) == RoofType.HIP
    # `52f91e67` was previously documented as mansard because two low-pitch
    # noise surfaces (same azimuth, ~14° pitch) gave the appearance of a
    # 2-tier pitch stratification on top of the genuine ~48° gable pair.
    # `roof.priors.filter_obliques_by_roof_type` drops those stranded noise
    # surfaces, leaving the gable pair the scan actually supports. If the
    # cohort doc reflected the noisy reading, the corrected classification
    # is gable.
    assert (
        classify_oblique_roof(build_roof_model(mansard_model).oblique) == RoofType.GABLE
    )
