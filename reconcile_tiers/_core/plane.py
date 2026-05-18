from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np

PLANE_KEY_NORMAL_BUCKET = 2e-3
PLANE_KEY_OFFSET_BUCKET_M = 0.02

PlaneKey = tuple[int, int, int, int]

# Two pieces are considered the same plane when normalized normals match
# within SAME_PLANE_NORMAL_TOL and offsets within SAME_PLANE_OFFSET_TOL_M.
# Used by:
#   - reconcile_tiers/audit/roof_defects.py (detect_fragmentation)
#   - reconcile_tiers/roof/architectural_priors.py (per-room oblique merge)
#   - reconcile_tiers/assemble/coplanar_merge.py (pre-emit dedupe)
SAME_PLANE_NORMAL_TOL = 0.05  # ~3 deg normal angle
SAME_PLANE_OFFSET_TOL_M = 0.5


def planes_equivalent(p1: Any, p2: Any) -> bool:
    """Whether two planes describe the same surface within tolerance.

    Accepts anything with .a/.b/.c/.d attrs (Plane dataclass) OR mapping types
    (raw JSON dicts from tier_payload). Symmetric.
    """
    return (
        abs(_pa(p1) - _pa(p2)) < SAME_PLANE_NORMAL_TOL
        and abs(_pb(p1) - _pb(p2)) < SAME_PLANE_NORMAL_TOL
        and abs(_pc(p1) - _pc(p2)) < SAME_PLANE_NORMAL_TOL
        and abs(_pd(p1) - _pd(p2)) < SAME_PLANE_OFFSET_TOL_M
    )


def _pa(p: Any) -> float:
    return float(getattr(p, "a", None) if hasattr(p, "a") else p.get("a", 0.0))


def _pb(p: Any) -> float:
    return float(getattr(p, "b", None) if hasattr(p, "b") else p.get("b", 0.0))


def _pc(p: Any) -> float:
    return float(getattr(p, "c", None) if hasattr(p, "c") else p.get("c", 0.0))


def _pd(p: Any) -> float:
    return float(getattr(p, "d", None) if hasattr(p, "d") else p.get("d", 0.0))


class FitFailure(StrEnum):
    TOO_FEW_POINTS = "too_few_points"
    NEAR_VERTICAL = "near_vertical"
    DEGENERATE = "degenerate"


@dataclass(frozen=True, slots=True)
class Plane:
    a: float
    b: float
    c: float
    d: float

    MIN_NY: ClassVar[float] = 0.087

    @classmethod
    def fit(cls, corners: Sequence[Sequence[float]]) -> Plane | FitFailure:
        try:
            pts = np.asarray(corners, dtype=float)
        except (TypeError, ValueError):
            return FitFailure.DEGENERATE
        if pts.ndim != 2 or pts.shape[1] != 3:
            return FitFailure.DEGENERATE
        if pts.shape[0] < 3:
            return FitFailure.TOO_FEW_POINTS

        centroid = pts.mean(axis=0)
        centered = pts - centroid
        if np.linalg.matrix_rank(centered) < 2:
            return FitFailure.DEGENERATE

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1].astype(float)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            return FitFailure.DEGENERATE
        normal /= norm
        if normal[1] < 0:
            normal *= -1.0

        a, b, c = (float(v) for v in normal)
        if abs(b) < cls.MIN_NY:
            return FitFailure.NEAR_VERTICAL
        d = float(normal @ centroid)
        return cls(a=a, b=b, c=c, d=d)

    def y_at(self, x: float, z: float) -> float | None:
        if abs(self.b) < self.MIN_NY:
            return None
        return (self.d - self.a * x - self.c * z) / self.b


def fit_plane_any(
    corners: Sequence[Sequence[float]],
) -> tuple[float, float, float, float] | None:
    try:
        pts = np.asarray(corners, dtype=float)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    if np.linalg.matrix_rank(centered, tol=1e-9) < 2:
        return None
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal /= norm
    a, b, c = (float(v) for v in normal)
    d = float(normal @ centroid)
    return (a, b, c, d)


def plane_key(plane: Any) -> PlaneKey:
    a = float(plane.a if hasattr(plane, "a") else plane[0])
    b = float(plane.b if hasattr(plane, "b") else plane[1])
    c = float(plane.c if hasattr(plane, "c") else plane[2])
    d = float(plane.d if hasattr(plane, "d") else plane[3])
    n = math.sqrt(a * a + b * b + c * c) or 1.0
    a, b, c, d = a / n, b / n, c / n, d / n
    components = (a, b, c)
    dom = max(range(3), key=lambda i: abs(components[i]))
    if components[dom] < 0.0:
        a, b, c, d = -a, -b, -c, -d
    return (
        round(a / PLANE_KEY_NORMAL_BUCKET),
        round(b / PLANE_KEY_NORMAL_BUCKET),
        round(c / PLANE_KEY_NORMAL_BUCKET),
        round(d / PLANE_KEY_OFFSET_BUCKET_M),
    )
