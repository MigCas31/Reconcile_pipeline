"""Sample oriented points from tier_payload tiles for KSR data terms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reconcile_tiers.polyhedron.manifold_repair import TileFace

_SAMPLES_PER_EDGE = 3


def _room_centroid(tiles: Sequence[TileFace]) -> np.ndarray:
    points: list[tuple[float, float, float]] = []
    for tile in tiles:
        points.extend(tile.corners)
    if not points:
        return np.zeros(3, dtype=float)
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _inward_normal(tile: TileFace, room_center: np.ndarray) -> np.ndarray:
    """Outward tile normal flipped to point into the room volume."""
    normal = np.array([tile.plane.a, tile.plane.b, tile.plane.c], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    normal /= norm
    centroid = np.mean(np.asarray(tile.corners, dtype=float), axis=0)
    if float(normal @ (room_center - centroid)) < 0.0:
        normal *= -1.0
    return normal


def _sample_polygon(
    corners: Sequence[tuple[float, float, float]],
    *,
    samples_per_edge: int,
) -> list[tuple[float, float, float]]:
    if len(corners) < 3:
        return []
    pts: list[tuple[float, float, float]] = [tuple(float(v) for v in c) for c in corners]
    n = len(corners)
    for i in range(n):
        a = np.asarray(corners[i], dtype=float)
        b = np.asarray(corners[(i + 1) % n], dtype=float)
        for t in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            p = a + t * (b - a)
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts


def sample_oriented_points(
    tiles: Sequence[TileFace],
    *,
    samples_per_edge: int = _SAMPLES_PER_EDGE,
) -> np.ndarray:
    """Return ``(N, 6)`` array ``x,y,z,nx,ny,nz`` with normals into the room."""
    if not tiles:
        return np.empty((0, 6), dtype=float)
    room_center = _room_centroid(tiles)
    rows: list[list[float]] = []
    for tile in tiles:
        normal = _inward_normal(tile, room_center)
        for point in _sample_polygon(tile.corners, samples_per_edge=samples_per_edge):
            rows.append(
                [
                    point[0],
                    point[1],
                    point[2],
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                ]
            )
    if not rows:
        return np.empty((0, 6), dtype=float)
    return np.asarray(rows, dtype=float)
