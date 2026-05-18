import math

import numpy as np
import pytest

from reconcile_tiers._core.svd import SvdFailure, compute_svd


def test_compute_svd_recovers_rotation_translation_with_zero_residual():
    src = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    theta = math.radians(90.0)
    rot_expected = np.array(
        [
            [math.cos(theta), 0.0, -math.sin(theta)],
            [0.0, 1.0, 0.0],
            [math.sin(theta), 0.0, math.cos(theta)],
        ]
    )
    trans_expected = np.array([2.0, -1.0, 4.0])
    dst = (rot_expected @ src.T).T + trans_expected

    result = compute_svd(src, dst)

    assert not isinstance(result, SvdFailure)
    rot, trans, residual_cm = result
    assert rot == pytest.approx(rot_expected)
    assert trans == pytest.approx(trans_expected)
    assert residual_cm == pytest.approx(0.0, abs=1e-10)


def test_compute_svd_reports_shape_mismatch():
    src = np.zeros((3, 3))
    dst = np.zeros((4, 3))

    assert compute_svd(src, dst) == SvdFailure.SHAPE_MISMATCH


def test_compute_svd_reports_too_few_points():
    src = np.zeros((2, 3))

    assert compute_svd(src, src) == SvdFailure.TOO_FEW_POINTS


def test_compute_svd_reports_degenerate_collinear_points():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    dst = src + np.array([1.0, 0.0, 0.0])

    assert compute_svd(src, dst) == SvdFailure.DEGENERATE
