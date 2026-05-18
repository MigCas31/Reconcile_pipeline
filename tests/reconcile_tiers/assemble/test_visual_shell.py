from reconcile_tiers._core.plane import Plane
from reconcile_tiers.assemble.visual_shell import (
    GABLE_END_MIN_PITCH_DEG,
    GABLE_END_PITCH_BAND_GAP_DEG,
    _has_steeper_band_member,
)
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster


def _surface(incl: float) -> ObliqueSurface:
    return ObliqueSurface(
        corners=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        plane=Plane(a=0.0, b=1.0, c=0.0, d=0.0),
        cluster=RoofCluster(
            segments=[], avg_incl=incl, avg_azimuth=0.0, ref_pt=[0.0, 0.0, 0.0]
        ),
        dominant_story=0,
        ridge={},
    )


def test_steeper_band_detected_when_gap_exceeds_threshold():
    near_flat = _surface(9.0)
    pitched = _surface(44.0)

    assert _has_steeper_band_member(near_flat, [near_flat, pitched]) is True


def test_no_steeper_band_when_pitches_are_close():
    a = _surface(14.0)
    b = _surface(16.0)

    assert _has_steeper_band_member(a, [a, b]) is False


def test_gap_at_threshold_qualifies_as_steeper_band():
    near_flat = _surface(10.0)
    pitched = _surface(10.0 + GABLE_END_PITCH_BAND_GAP_DEG)

    assert _has_steeper_band_member(near_flat, [near_flat, pitched]) is True


def test_low_pitch_gable_has_no_steeper_band():
    a = _surface(GABLE_END_MIN_PITCH_DEG - 1.0)
    b = _surface(GABLE_END_MIN_PITCH_DEG + 1.0)

    assert _has_steeper_band_member(a, [a, b]) is False
    assert _has_steeper_band_member(b, [a, b]) is False
