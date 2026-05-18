from __future__ import annotations

from enum import StrEnum

import numpy as np


class SvdFailure(StrEnum):
    SHAPE_MISMATCH = "shape_mismatch"
    TOO_FEW_POINTS = "too_few_points"
    DEGENERATE = "degenerate"


def compute_svd(src, dst):
    src_arr = np.asarray(src, dtype=float)
    dst_arr = np.asarray(dst, dtype=float)
    if src_arr.shape != dst_arr.shape or src_arr.ndim != 2 or src_arr.shape[1] != 3:
        return SvdFailure.SHAPE_MISMATCH
    if src_arr.shape[0] < 3:
        return SvdFailure.TOO_FEW_POINTS
    if np.linalg.matrix_rank(src_arr - src_arr.mean(axis=0)) < 2:
        return SvdFailure.DEGENERATE

    src_center = src_arr.mean(axis=0)
    dst_center = dst_arr.mean(axis=0)
    h_mat = (src_arr - src_center).T @ (dst_arr - dst_center)
    u_mat, _, v_t = np.linalg.svd(h_mat)
    rot = v_t.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        v_t[-1] *= -1
        rot = v_t.T @ u_mat.T
    trans = dst_center - rot @ src_center
    residual_cm = float(
        np.max(np.linalg.norm(dst_arr - (rot @ src_arr.T).T - trans, axis=1)) * 100.0
    )
    return rot, trans, residual_cm
