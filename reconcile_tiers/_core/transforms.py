from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def parse_transform(flat: Sequence[float]) -> np.ndarray:
    values = np.asarray(flat, dtype=float)
    if values.size != 16:
        raise ValueError(f"expected 16 transform values, got {values.size}")
    return values.reshape((4, 4))


def corners_to_world(
    corners: Sequence[Sequence[float]], transform: np.ndarray
) -> list[list[float]]:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {matrix.shape}")
    out: list[list[float]] = []
    for corner in corners:
        if len(corner) < 3:
            raise ValueError("corner must have at least 3 coordinates")
        world = matrix @ np.array(
            [float(corner[0]), float(corner[1]), float(corner[2]), 1.0]
        )
        out.append([float(world[0]), float(world[1]), float(world[2])])
    return out


def hybrid_wall_corners(
    merged_wall: dict, raw_wall: dict, floor_y: float | None = None
) -> list[list[float]]:
    transform = parse_transform(merged_wall["transform"])
    merged_pc = merged_wall.get("polygonCorners") or []
    raw_pc = raw_wall.get("polygonCorners") or []
    if len(merged_pc) >= 3:
        local = merged_pc
    elif len(raw_pc) >= 3:
        local = raw_pc
    else:
        width = float(raw_wall["dimensions"][0]) / 2.0
        height = float(raw_wall["dimensions"][1]) / 2.0
        local = [
            [-width, -height, 0.0],
            [width, -height, 0.0],
            [width, height, 0.0],
            [-width, height, 0.0],
        ]
    corners = corners_to_world(local, transform)
    if floor_y is None:
        return corners
    current_bottom = min(corner[1] for corner in corners)
    dy = float(floor_y) - current_bottom
    return [[corner[0], corner[1] + dy, corner[2]] for corner in corners]
