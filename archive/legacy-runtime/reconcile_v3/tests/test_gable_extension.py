"""Unit tests for classify_gable_extension — five status branches.

Synthetic fixtures are minimal V3Part + V3SlantedRoof tuples; the classifier
is a pure function over them. One integration test pins the subject building
(``d32d5562``) as ``gable_along_extend``.
"""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import pytest

from reconcile_v3.audit import HypothesisTrace
from reconcile_v3.io.load import load_building
from reconcile_v3.models import V3Part, V3SlantedRoof
from reconcile_v3.pipeline import run_pipeline
from reconcile_v3.stages.gable_extension import _classify_single_part

BUILDINGS_JSON = (
    Path(__file__).resolve().parent.parent.parent / "reconcile" / "buildings_3d.json"
)
SUBJECT_UUID = "d32d5562-5763-4c71-a816-6732c638fa6a"


def _rect_xz(width_x: float, length_z: float, y: float = 0.0):
    hx, hz = width_x / 2.0, length_z / 2.0
    return [(-hx, y, -hz), (hx, y, -hz), (hx, y, hz), (-hx, y, hz)]


def _slope_plane(azimuth_deg: float, inclination_deg: float, ref_point):
    """Build plane (a,b,c,d) with downslope azimuth + inclination through ref."""
    az = radians(azimuth_deg)
    i = radians(inclination_deg)
    # Upward normal: tilt from +Y by ``inclination`` toward the horizontal
    # direction opposite to the downslope.
    nx = -sin(az) * sin(i)
    ny = cos(i)
    nz = -cos(az) * sin(i)
    rx, ry, rz = ref_point
    d = -(nx * rx + ny * ry + nz * rz)
    return (float(nx), float(ny), float(nz), float(d))


def _slope_corners(
    plane, x_span: float, z_lo: float, z_hi: float, y_fallback: float = 0.0
):
    """Build a roof-plane patch across ``x_span`` between ``z_lo`` and ``z_hi``."""
    a, b, c, d = plane
    xs = [-x_span / 2.0, x_span / 2.0, x_span / 2.0, -x_span / 2.0]
    zs = [z_lo, z_lo, z_hi, z_hi]
    out = []
    for x, z in zip(xs, zs, strict=False):
        if abs(b) > 1e-6:
            y = -(a * x + c * z + d) / b
        else:
            y = y_fallback
        out.append((float(x), float(y), float(z)))
    return out


def _make_part(footprint_xz, stories=(0,), part_id="uuid::v3-part::part-0"):
    return V3Part(
        id=part_id,
        room_ids=("room:0",),
        footprint_xz=footprint_xz,
        stories=tuple(stories),
        trace=HypothesisTrace(stage="test", rule="test"),
    )


def _make_roof(plane, corners, roof_id="uuid::v3-slanted-roof::cluster-0"):
    return V3SlantedRoof(
        id=roof_id,
        plane=plane,
        corners=corners,
        source_segment_ids=(),
        trace=HypothesisTrace(stage="test", rule="test"),
    )


def test_not_gable_zero_roofs():
    part = _make_part(_rect_xz(6.0, 12.0))
    result = _classify_single_part(part, roofs=[], n_dormers=0, n_arch_flats=0)
    assert result.status == "not_gable"
    assert any(r.startswith("n_slanted_roofs=0") for r in result.tier1_reasons)


def test_not_gable_mismatched_inclinations():
    ridge_y = 3.0
    plane_a = _slope_plane(90.0, 45.0, (0.0, ridge_y, 0.0))
    plane_b = _slope_plane(270.0, 15.0, (0.0, ridge_y, 0.0))  # saltbox
    corners_a = _slope_corners(plane_a, 12.0, -3.0, 0.0)
    corners_b = _slope_corners(plane_b, 12.0, 0.0, 3.0)
    part = _make_part(_rect_xz(6.0, 12.0))
    result = _classify_single_part(
        part,
        roofs=[
            _make_roof(plane_a, corners_a, "uuid::v3-slanted-roof::a"),
            _make_roof(plane_b, corners_b, "uuid::v3-slanted-roof::b"),
        ],
        n_dormers=0,
        n_arch_flats=0,
    )
    assert result.status == "not_gable"
    assert any(r.startswith("dincl=") for r in result.tier1_reasons)


def test_gable_along_extend_synthetic():
    ridge_y = 3.0
    # Ridge runs along +Z (the long axis of a 12m x 6m part).
    # Slopes fall east (+X) and west (-X).
    plane_east = _slope_plane(90.0, 40.0, (0.0, ridge_y, 0.0))
    plane_west = _slope_plane(270.0, 40.0, (0.0, ridge_y, 0.0))
    # Give slopes partial coverage — only over +Z half of the footprint.
    corners_e = _slope_corners(plane_east, 6.0, -2.0, 4.0)
    corners_w = _slope_corners(plane_west, 6.0, -2.0, 4.0)
    part = _make_part(_rect_xz(6.0, 12.0))
    result = _classify_single_part(
        part,
        roofs=[
            _make_roof(plane_east, corners_e, "uuid::v3-slanted-roof::e"),
            _make_roof(plane_west, corners_w, "uuid::v3-slanted-roof::w"),
        ],
        n_dormers=0,
        n_arch_flats=0,
    )
    assert result.status == "gable_along_extend", result.decision_reason
    assert result.ridge_line is not None
    assert result.uncovered_region_xz is not None


def test_gable_complete_when_fully_covered():
    ridge_y = 3.0
    plane_east = _slope_plane(90.0, 40.0, (0.0, ridge_y, 0.0))
    plane_west = _slope_plane(270.0, 40.0, (0.0, ridge_y, 0.0))
    # Full coverage — slopes span the entire footprint.
    corners_e = _slope_corners(plane_east, 6.0, -6.0, 6.0)
    corners_w = _slope_corners(plane_west, 6.0, -6.0, 6.0)
    part = _make_part(_rect_xz(6.0, 12.0))
    result = _classify_single_part(
        part,
        roofs=[
            _make_roof(plane_east, corners_e, "uuid::v3-slanted-roof::e"),
            _make_roof(plane_west, corners_w, "uuid::v3-slanted-roof::w"),
        ],
        n_dormers=0,
        n_arch_flats=0,
    )
    assert result.status == "gable_complete", result.decision_reason


def test_gable_cross_review_ridge_across_short_axis():
    ridge_y = 3.0
    # Part: long axis is +Z (12 m). Force ridge to run along +X (the short
    # axis) by flipping slope azimuths to 0°/180° (slopes fall along ±Z).
    plane_n = _slope_plane(0.0, 40.0, (0.0, ridge_y, 0.0))
    plane_s = _slope_plane(180.0, 40.0, (0.0, ridge_y, 0.0))
    corners_n = _slope_corners(plane_n, 6.0, -2.0, 0.0)
    corners_s = _slope_corners(plane_s, 6.0, 0.0, 2.0)
    part = _make_part(_rect_xz(6.0, 12.0))
    result = _classify_single_part(
        part,
        roofs=[
            _make_roof(plane_n, corners_n, "uuid::v3-slanted-roof::n"),
            _make_roof(plane_s, corners_s, "uuid::v3-slanted-roof::s"),
        ],
        n_dormers=0,
        n_arch_flats=0,
    )
    assert result.status == "gable_cross_review", result.decision_reason


def test_not_gable_when_dormers_present():
    ridge_y = 3.0
    plane_east = _slope_plane(90.0, 40.0, (0.0, ridge_y, 0.0))
    plane_west = _slope_plane(270.0, 40.0, (0.0, ridge_y, 0.0))
    corners_e = _slope_corners(plane_east, 6.0, -2.0, 4.0)
    corners_w = _slope_corners(plane_west, 6.0, -2.0, 4.0)
    part = _make_part(_rect_xz(6.0, 12.0))
    result = _classify_single_part(
        part,
        roofs=[
            _make_roof(plane_east, corners_e, "uuid::v3-slanted-roof::e"),
            _make_roof(plane_west, corners_w, "uuid::v3-slanted-roof::w"),
        ],
        n_dormers=1,
        n_arch_flats=0,
    )
    assert result.status == "not_gable"
    assert any(r.startswith("dormers=") for r in result.tier2_reasons)


def test_ambiguous_on_invalid_footprint():
    ridge_y = 3.0
    plane_east = _slope_plane(90.0, 40.0, (0.0, ridge_y, 0.0))
    plane_west = _slope_plane(270.0, 40.0, (0.0, ridge_y, 0.0))
    corners_e = _slope_corners(plane_east, 6.0, -2.0, 4.0)
    corners_w = _slope_corners(plane_west, 6.0, -2.0, 4.0)
    # Empty footprint
    part = _make_part([])
    result = _classify_single_part(
        part,
        roofs=[
            _make_roof(plane_east, corners_e, "uuid::v3-slanted-roof::e"),
            _make_roof(plane_west, corners_w, "uuid::v3-slanted-roof::w"),
        ],
        n_dormers=0,
        n_arch_flats=0,
    )
    assert result.status == "gable_ambiguous"


@pytest.mark.skipif(not BUILDINGS_JSON.exists(), reason="buildings_3d.json missing")
def test_subject_d32d5562_is_extension_grade():
    data = load_building(BUILDINGS_JSON, SUBJECT_UUID)
    building = run_pipeline(data)
    assert building.parts, "Subject building should have at least one part"
    part = building.parts[0]
    ge = part.gable_extension
    assert ge is not None
    assert ge.status == "gable_along_extend", ge.decision_reason
    m = ge.metrics
    assert m["n_slanted_roofs"] == 2
    assert m["daz180"] < 5.0
    assert m["dincl"] < 5.0
    assert m["ridge_vs_major"] < 5.0
    assert m["elong"] > 1.2
    assert m["coverage"] < 0.9
    assert m["n_dormers"] == 0
