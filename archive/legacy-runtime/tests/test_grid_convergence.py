"""Tests for grid convergence computation.

Verified against web-main north-reference.ts implementation.
"""

import math

from reconcile.grid_convergence import (
    EPSG_2154,
    EPSG_25832,
    compute_grid_convergence_deg,
    grid_convergence_for_country,
)


def test_copenhagen_utm32n():
    """Copenhagen (55.68N, 12.57E) with EPSG:25832 (central meridian 9)."""
    conv = compute_grid_convergence_deg(55.68, 12.57, EPSG_25832)
    # Expected: atan(tan((12.57-9)*pi/180) * sin(55.68*pi/180)) in degrees
    expected = math.degrees(
        math.atan(math.tan(math.radians(12.57 - 9)) * math.sin(math.radians(55.68)))
    )
    assert abs(conv - expected) < 1e-10
    # Should be roughly 2 degrees
    assert 1.5 < conv < 3.0


def test_paris_lambert93():
    """Paris (48.86N, 2.35E) with EPSG:2154 (Lambert-93)."""
    conv = compute_grid_convergence_deg(48.86, 2.35, EPSG_2154)
    # Expected: (2.35-3)*pi/180 * 0.725607765053267 in degrees
    expected = math.degrees(math.radians(2.35 - 3) * 0.725607765053267)
    assert abs(conv - expected) < 1e-10
    # Should be roughly -0.47 degrees (west of central meridian)
    assert -1.0 < conv < 0.0


def test_central_meridian_zero_convergence():
    """On the central meridian, grid convergence should be zero."""
    conv = compute_grid_convergence_deg(55.0, 9.0, EPSG_25832)
    assert abs(conv) < 1e-10


def test_equator_zero_convergence_utm():
    """At the equator, UTM grid convergence should be zero (sin(0) = 0)."""
    conv = compute_grid_convergence_deg(0.0, 12.0, EPSG_25832)
    assert abs(conv) < 1e-10


def test_grid_convergence_for_country_dk():
    """Denmark uses EPSG:25832."""
    conv = grid_convergence_for_country("DK", 55.68, 12.57)
    expected = compute_grid_convergence_deg(55.68, 12.57, EPSG_25832)
    assert conv == expected


def test_grid_convergence_for_country_fr():
    """France uses EPSG:2154."""
    conv = grid_convergence_for_country("FR", 48.86, 2.35)
    expected = compute_grid_convergence_deg(48.86, 2.35, EPSG_2154)
    assert conv == expected


def test_grid_convergence_for_unknown_country():
    """Unknown countries return 0."""
    assert grid_convergence_for_country("XX", 55.0, 12.0) == 0.0


def test_symmetry():
    """East and west of central meridian should have opposite sign."""
    east = compute_grid_convergence_deg(55.0, 12.0, EPSG_25832)
    west = compute_grid_convergence_deg(55.0, 6.0, EPSG_25832)
    assert east > 0
    assert west < 0
    # For small angles and same latitude, magnitudes should be roughly equal
    assert abs(abs(east) - abs(west)) < 0.1
