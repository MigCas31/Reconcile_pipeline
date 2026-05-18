import pytest

from reconcile_tiers._core.newell import (
    is_planar,
    newell_normal,
    polygon_area_3d,
    polygon_xz_signed_area,
)


def test_newell_normal_tracks_horizontal_winding():
    minus_y_winding = [
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ]

    assert newell_normal(minus_y_winding)[1] < 0
    assert newell_normal(list(reversed(minus_y_winding)))[1] > 0


def test_polygon_area_3d_matches_legacy_newell_area_for_sloped_quad():
    corners = [
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 4.0],
        [0.0, 1.0, 4.0],
    ]

    assert polygon_area_3d(corners) == pytest.approx(3.0 * (17.0**0.5))


def test_planarity_uses_distance_to_newell_plane():
    planar = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
    ]
    warped = [*planar[:3], [0.0, 0.65, 1.0]]

    assert is_planar(planar, tol=1e-6)
    assert not is_planar(warped, tol=0.01)


def test_polygon_xz_signed_area_ignores_y():
    ccw_xz = [
        [0.0, 7.0, 0.0],
        [2.0, -3.0, 0.0],
        [2.0, 4.0, 1.0],
        [0.0, 9.0, 1.0],
    ]

    assert polygon_xz_signed_area(ccw_xz) == pytest.approx(2.0)
    assert polygon_xz_signed_area(list(reversed(ccw_xz))) == pytest.approx(-2.0)
