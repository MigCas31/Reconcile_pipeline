"""Phase B.3 tests for reconcile_v3/reconstruction/realism.py.

Tight unit tests on score_building — we fabricate selections + reference
envelopes with hand-computed IoU / coverage / over-coverage numbers.
"""

from __future__ import annotations

import math

from reconcile_v3.reconstruction.realism import (
    aggregate_scores,
    score_building,
)

_BLDG = "test-building"


def _face(fp_xz: list[tuple[float, float]], az_deg: float = 0.0) -> dict:
    return {
        "id": f"{_BLDG}::candidate::test",
        "footprint_xz": [[x, z] for (x, z) in fp_xz],
        "azimuth_deg": az_deg,
    }


def _roof(corners_xz: list[tuple[float, float]], az_deg: float = 0.0) -> dict:
    # V3SlantedRoof.corners is 3D; wrap each XZ pair as [x, y, z].
    return {
        "corners": [[x, 0.0, z] for (x, z) in corners_xz],
        "azimuth_deg": az_deg,
    }


def _ceiling(fp_xz: list[tuple[float, float]]) -> dict:
    return {"footprint_xz": [[x, 0.0, z] for (x, z) in fp_xz]}


def test_identical_selection_and_reference_iou_one() -> None:
    sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    sel = [_face(sq)]
    ref = {"slanted_roofs": [_roof(sq)], "flat_ceilings": []}
    s = score_building(_BLDG, sel, ref)
    assert math.isclose(s.iou_xz, 1.0, abs_tol=1e-6)
    assert math.isclose(s.coverage_ref, 1.0, abs_tol=1e-6)
    assert math.isclose(s.over_coverage, 0.0, abs_tol=1e-6)
    assert s.note == ""


def test_selection_half_of_reference_gives_half_coverage() -> None:
    ref_sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]  # area 16
    sel_half = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]  # area 8
    s = score_building(
        _BLDG,
        [_face(sel_half)],
        {"slanted_roofs": [_roof(ref_sq)], "flat_ceilings": []},
    )
    assert math.isclose(s.coverage_ref, 0.5, abs_tol=1e-6)
    assert math.isclose(s.over_coverage, 0.0, abs_tol=1e-6)
    # IoU = 8 / 16
    assert math.isclose(s.iou_xz, 0.5, abs_tol=1e-6)


def test_selection_overshoots_reference_gives_over_coverage() -> None:
    # Reference is 10 m². Selection equals reference plus a 2 m² bump —
    # coverage_ref = 1.0, over_coverage = 0.2, IoU = 10/12.
    ref_sq = [(0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)]
    sel_sq = [(0.0, 0.0), (5.0, 0.0), (5.0, 2.4), (0.0, 2.4)]
    s = score_building(
        _BLDG,
        [_face(sel_sq)],
        {"slanted_roofs": [_roof(ref_sq)], "flat_ceilings": []},
    )
    assert math.isclose(s.coverage_ref, 1.0, abs_tol=1e-6)
    assert math.isclose(s.over_coverage, 0.2, abs_tol=1e-6)
    assert math.isclose(s.iou_xz, 10.0 / 12.0, abs_tol=1e-6)


def test_disjoint_selection_gives_zero() -> None:
    # Selection on opposite half of the ridge from the reference — the
    # Phase A bug regression case.
    ref_sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
    sel_sq = [(0.0, 2.0), (4.0, 2.0), (4.0, 4.0), (0.0, 4.0)]
    s = score_building(
        _BLDG,
        [_face(sel_sq)],
        {"slanted_roofs": [_roof(ref_sq)], "flat_ceilings": []},
    )
    assert math.isclose(s.iou_xz, 0.0, abs_tol=1e-6)
    assert math.isclose(s.coverage_ref, 0.0, abs_tol=1e-6)


def test_no_reference_returns_none_iou_and_note() -> None:
    sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    s = score_building(_BLDG, [_face(sq)], {"slanted_roofs": [], "flat_ceilings": []})
    assert s.iou_xz is None
    assert "reference" in s.note


def test_empty_selection_with_present_reference() -> None:
    sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    s = score_building(_BLDG, [], {"slanted_roofs": [_roof(sq)], "flat_ceilings": []})
    assert s.iou_xz == 0.0
    assert s.coverage_ref == 0.0
    assert "selection empty" in s.note


def test_azimuth_mismatch_fraction() -> None:
    # Two selected faces, both near the same reference centroid. One
    # has a close azimuth (0° vs 10°), the other is 180° off.
    ref_sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    ref = {
        "slanted_roofs": [_roof(ref_sq, az_deg=0.0)],
        "flat_ceilings": [],
    }
    sel = [
        _face([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)], az_deg=10.0),
        _face([(0.0, 2.0), (4.0, 2.0), (4.0, 4.0), (0.0, 4.0)], az_deg=180.0),
    ]
    s = score_building(_BLDG, sel, ref)
    # 1/2 mismatches; accept small float noise.
    assert math.isclose(s.azimuth_mismatch_frac, 0.5, abs_tol=1e-6)


def test_flat_ceiling_reference_counted() -> None:
    # Reference has only a flat ceiling (no slanted roofs) — the selection
    # that overlaps it should still score positive coverage.
    ceil = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    sel_half = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
    s = score_building(
        _BLDG,
        [_face(sel_half)],
        {"slanted_roofs": [], "flat_ceilings": [_ceiling(ceil)]},
    )
    assert math.isclose(s.coverage_ref, 0.5, abs_tol=1e-6)


def test_aggregate_scores_ignores_none_iou() -> None:
    ref_sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    full = {"slanted_roofs": [_roof(ref_sq)], "flat_ceilings": []}
    scores = [
        score_building("a", [_face(ref_sq)], full),  # iou=1
        score_building(
            "b", [_face(ref_sq)], {"slanted_roofs": [], "flat_ceilings": []}
        ),  # no ref → None
    ]
    agg = aggregate_scores(scores)
    assert agg["n_buildings"] == 2
    assert agg["n_with_reference"] == 1
    assert math.isclose(agg["mean_iou"], 1.0, abs_tol=1e-6)
    assert math.isclose(agg["frac_iou_ge_0_9"], 1.0, abs_tol=1e-6)
