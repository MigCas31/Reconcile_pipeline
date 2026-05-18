"""Unit tests for ``reconcile_v3.analysis.derived_features``.

Covers the family-guess lookup table, azimuth folding, and the Tier A
flag combinators. Inputs are plain dicts — the module only reads keys,
never raw geometry.
"""

from __future__ import annotations

from reconcile_v3.analysis.derived_features import (
    _az_diff_180,
    _family_guess,
    derived_features,
)


def test_az_diff_180_folds_above_90():
    assert _az_diff_180(10.0, 170.0) == 20.0
    assert _az_diff_180(0.0, 90.0) == 90.0
    assert _az_diff_180(None, 0.0) is None


def test_family_guess_from_status():
    assert _family_guess({"part_gable_status": "gable_complete"}) == "gable"
    assert _family_guess({"part_gable_status": "gable_ambiguous"}) == "complex"


def test_family_guess_from_slanted_count():
    def mk(n):
        return {"part_gable_status": None, "part_gable_metric_n_slanted_roofs": n}

    assert _family_guess(mk(0)) == "flat"
    assert _family_guess(mk(1)) == "shed"
    assert _family_guess(mk(2)) == "gable"
    assert _family_guess(mk(3)) == "complex"
    assert _family_guess(mk(5)) == "hip"
    assert _family_guess(mk(None)) == "unknown"


def test_gable_match_when_plane_near_leg():
    row = {
        "part_gable_status": "gable_complete",
        "part_gable_metric_n_slanted_roofs": 2,
        "plane_azimuth_deg": 92.0,
        "part_gable_metric_az0": 90.0,
        "part_gable_metric_az1": 270.0,
        "part_gable_metric_major_az": 0.0,
        "part_gable_metric_ridge_az": 0.0,
    }
    out = derived_features(row)
    assert out["derived_part_roof_family_guess"] == "gable"
    assert out["derived_plane_matches_gable_family"] is True
    assert out["derived_plane_min_az_to_gable_leg_deg"] == 2.0
    # az0=90, az1=270 -> axial difference 0.
    assert out["derived_gable_leg_az_delta_deg"] == 0.0


def test_cross_story_flag_and_orphan_oblique():
    row = {
        "part_gable_status": None,
        "part_gable_metric_n_slanted_roofs": 1,
        "plane_pitch_is_architectural": True,
        "opposing_count": 0,
        "member_story_delta_max": 1.0,
    }
    out = derived_features(row)
    assert out["derived_cluster_any_cross_story"] is True
    assert out["derived_is_orphan_oblique"] is True
    assert out["derived_part_n_slanted_roofs_eq_1"] is True
    assert out["derived_part_n_slanted_roofs_ge_4"] is False


def test_missing_inputs_return_none():
    out = derived_features({})
    assert out["derived_part_roof_family_guess"] == "unknown"
    assert out["derived_plane_matches_gable_family"] is None
    assert out["derived_ridge_vs_major_axis_deg"] is None
    assert out["derived_plane_az_vs_bld_major_deg"] is None
    assert out["derived_cluster_any_cross_story"] is None
    assert out["derived_part_n_slanted_roofs_eq_2"] is None
