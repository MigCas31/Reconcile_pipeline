import math

import pytest

from reconcile_tiers._core.plane import FitFailure, Plane


def test_fit_horizontal_plane():
    corners = [
        [-1.0, 2.5, -1.0],
        [1.0, 2.5, -1.0],
        [1.0, 2.5, 1.0],
        [-1.0, 2.5, 1.0],
    ]

    plane = Plane.fit(corners)

    assert isinstance(plane, Plane)
    assert plane.a == pytest.approx(0.0, abs=1e-9)
    assert abs(plane.b) == pytest.approx(1.0, abs=1e-9)
    assert plane.c == pytest.approx(0.0, abs=1e-9)
    assert abs(plane.d) == pytest.approx(2.5, abs=1e-9)
    assert plane.y_at(4.0, -3.0) == pytest.approx(2.5)


def test_fit_sloped_plane_round_trips_y():
    corners = [
        [-1.0, 1.0, -1.0],
        [1.0, 2.0, -1.0],
        [1.0, 2.5, 1.0],
        [-1.0, 1.5, 1.0],
    ]

    plane = Plane.fit(corners)

    assert isinstance(plane, Plane)
    for x, y, z in corners:
        assert plane.y_at(x, z) == pytest.approx(y, abs=1e-9)


def test_fit_rejects_near_vertical_with_reason():
    # This wall-like plane has normal roughly along X, so solving y=f(x,z)
    # would be physically wrong for ceiling/roof use.
    corners = [
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 2.0],
        [1.0, 0.0, 2.0],
    ]

    assert Plane.fit(corners) == FitFailure.NEAR_VERTICAL


def test_fit_rejects_collinear_points():
    corners = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]

    assert Plane.fit(corners) == FitFailure.DEGENERATE


def test_fit_rejects_too_few_points():
    assert Plane.fit([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]) == FitFailure.TOO_FEW_POINTS


@pytest.mark.parametrize(
    "corners",
    [
        "not numeric",
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    ],
)
def test_fit_rejects_invalid_input(corners):
    assert Plane.fit(corners) == FitFailure.DEGENERATE


def test_y_at_returns_none_for_non_ceiling_grade_plane():
    plane = Plane(a=1.0, b=0.01, c=0.0, d=1.0)

    assert plane.y_at(0.0, 0.0) is None


@pytest.mark.parametrize(
    "tilt_deg",
    [5.0, 10.0, 25.0, 45.0, 75.0],
)
def test_plane_fit_round_trip_for_ceiling_grade_tilts(tilt_deg):
    slope = math.tan(math.radians(tilt_deg))
    corners = [
        [-2.0, 1.2 - 2.0 * slope, -1.0],
        [2.0, 1.2 + 2.0 * slope, -1.0],
        [2.0, 1.2 + 2.0 * slope, 1.0],
        [-2.0, 1.2 - 2.0 * slope, 1.0],
    ]

    plane = Plane.fit(corners)

    assert isinstance(plane, Plane)
    assert abs(plane.b) >= Plane.MIN_NY
    for x, y, z in corners:
        assert plane.y_at(x, z) == pytest.approx(y, abs=1e-8)
