"""Wing decomposition for L/T/U-shaped building footprints."""

from __future__ import annotations

from shapely.geometry import Polygon

from reconcile_tiers._core.wing_decomposition import (
    Wing,
    assign_xz_points_to_wings,
    decompose_to_wings,
)


def test_simple_rectangle_yields_single_main_wing():
    rect = Polygon([(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)])
    wings = decompose_to_wings(rect)
    assert len(wings) == 1
    assert wings[0].role == "main"
    assert abs(wings[0].area_m2 - 32.0) < 1e-9
    # Long axis runs along +X (the 8-unit side); math angle == 0 deg.
    assert abs(wings[0].long_axis_math) < 1.0


def test_long_axis_math_runs_along_longest_edge_2_to_1():
    # Tall rectangle: 4 wide x 8 tall. Long axis along +Z (math 90 deg).
    tall = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 8.0), (0.0, 8.0)])
    wings = decompose_to_wings(tall)
    assert len(wings) == 1
    assert abs(wings[0].long_axis_math - 90.0) < 1.0


def test_long_axis_math_per_wing_for_l_shape():
    # L-shape with one wing 8x4 (long axis +X = 0 deg) and one wing 4x4 (square,
    # arbitrary but stable). Confirm each wing gets a well-defined long axis.
    l_shape = Polygon(
        [
            (0.0, 0.0),
            (8.0, 0.0),
            (8.0, 4.0),
            (4.0, 4.0),
            (4.0, 8.0),
            (0.0, 8.0),
        ]
    )
    wings = decompose_to_wings(l_shape)
    assert len(wings) >= 2
    for w in wings:
        # long_axis_math is in [0, 180); finite and well-defined per wing.
        assert 0.0 <= w.long_axis_math < 180.0


def test_l_shape_decomposes_into_two_wings():
    l_shape = Polygon(
        [
            (0.0, 0.0),
            (8.0, 0.0),
            (8.0, 4.0),
            (4.0, 4.0),
            (4.0, 8.0),
            (0.0, 8.0),
        ]
    )
    wings = decompose_to_wings(l_shape)
    assert len(wings) >= 2
    # The two wings together cover the full L-shaped footprint (within rounding).
    total_area = sum(w.area_m2 for w in wings)
    assert abs(total_area - l_shape.area) < 0.5
    # The largest wing is labelled "main"; the rest are "extension".
    assert wings[0].role == "main"
    for extra in wings[1:]:
        assert extra.role == "extension"


def test_assign_xz_points_to_wings_routes_each_point_to_its_containing_wing():
    main = Wing(
        index=0,
        polygon=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]),
        area_m2=16.0,
        role="main",
    )
    ext = Wing(
        index=1,
        polygon=Polygon([(5.0, 0.0), (8.0, 0.0), (8.0, 4.0), (5.0, 4.0)]),
        area_m2=12.0,
        role="extension",
    )
    points = [(2.0, 2.0), (6.5, 2.0), (4.5, 2.0), (100.0, 100.0)]
    result = assign_xz_points_to_wings([main, ext], points)
    # Inside main, inside extension, between (snaps to nearest within 0.5m), far
    # outside.
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] in (0, 1)
    assert result[3] is None


def test_non_rectilinear_footprint_falls_back_to_single_wing():
    """Triangles and other non-axis-aligned shapes fall back to a single main wing."""
    triangle = Polygon([(0.0, 0.0), (10.0, 0.0), (5.0, 8.66)])
    wings = decompose_to_wings(triangle)
    assert len(wings) == 1
    assert wings[0].role == "main"


def test_empty_polygon_returns_no_wings():
    assert decompose_to_wings(Polygon()) == []
