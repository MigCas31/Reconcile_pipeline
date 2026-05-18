import importlib
import pkgutil

import numpy as np
import pytest

import reconcile_tiers
import reconcile_tiers._core
from reconcile.complexity_tiers import _polygon_area_3d as legacy_polygon_area_3d
from reconcile.extract_3d import compute_svd as legacy_compute_svd
from reconcile.viewer_server import _fit_plane_coeffs, _ring_to_3d_on_plane
from reconcile_tiers._core.newell import polygon_area_3d
from reconcile_tiers._core.plane import Plane
from reconcile_tiers._core.svd import SvdFailure, compute_svd


def test_plane_fit_y_at_matches_viewer_server_plane_lift():
    corners = [
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 3.0, 2.0],
        [0.0, 2.0, 2.0],
    ]
    ring = [(0.2, 0.4), (1.7, 0.3), (1.8, 1.5), (0.3, 1.6), (0.2, 0.4)]

    legacy_plane = _fit_plane_coeffs(corners)
    legacy_lifted = _ring_to_3d_on_plane(ring, legacy_plane)
    plane = Plane.fit(corners)

    assert isinstance(plane, Plane)
    for legacy_point in legacy_lifted:
        x, y, z = legacy_point
        assert plane.y_at(x, z) == pytest.approx(y, abs=1e-9)


def test_newell_area_matches_complexity_tiers_legacy_helper():
    polygons = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 4.0], [0.0, 1.0, 4.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    ]

    for polygon in polygons:
        assert polygon_area_3d(polygon) == pytest.approx(
            legacy_polygon_area_3d(polygon)
        )


def test_compute_svd_matches_legacy_extract_3d_residuals():
    src = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.25],
            [1.0, 1.0, 1.0],
        ]
    )
    rot = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    trans = np.array([5.0, -2.0, 0.75])
    dst = (rot @ src.T).T + trans
    dst_noisy = dst.copy()
    dst_noisy[-1] += np.array([0.005, -0.003, 0.002])

    result = compute_svd(src, dst_noisy)
    legacy_result = legacy_compute_svd(src, dst_noisy)

    assert not isinstance(result, SvdFailure)
    _, _, residual_cm = result
    _, _, legacy_residual_cm = legacy_result
    assert residual_cm == pytest.approx(legacy_residual_cm)


def test_core_package_does_not_import_legacy_reconcile_modules():
    forbidden = ("reconcile.", "reconcile_v2", "reconcile_v3")
    for module_info in pkgutil.walk_packages(
        reconcile_tiers._core.__path__, "reconcile_tiers._core."
    ):
        module = importlib.import_module(module_info.name)
        for value in module.__dict__.values():
            if getattr(value, "__module__", "").startswith(forbidden):
                pytest.fail(f"{module_info.name} exposes legacy import {value!r}")
