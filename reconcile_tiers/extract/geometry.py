from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def parse_roomplan_transform(flat: Sequence[float]) -> np.ndarray:
    values = np.asarray(flat, dtype=float)
    if values.size != 16:
        raise ValueError(f"expected 16 transform values, got {values.size}")
    return values.reshape((4, 4), order="F")


def corners_to_world(
    corners: Sequence[Sequence[float]], transform_flat: Sequence[float]
) -> list[list[float]]:
    transform = parse_roomplan_transform(transform_flat)
    out: list[list[float]] = []
    for corner in corners:
        world = transform @ np.array(
            [float(corner[0]), float(corner[1]), float(corner[2]), 1.0]
        )
        out.append([float(world[0]), float(world[1]), float(world[2])])
    return out


def wall_world_corners(wall: dict) -> list[list[float]]:
    polygon = wall.get("polygonCorners") or []
    if len(polygon) >= 3:
        return corners_to_world(polygon, wall["transform"])
    width = float(wall["dimensions"][0]) / 2.0
    height = float(wall["dimensions"][1]) / 2.0
    local = [
        [-width, -height, 0.0],
        [width, -height, 0.0],
        [width, height, 0.0],
        [-width, height, 0.0],
    ]
    return corners_to_world(local, wall["transform"])


def hybrid_wall_corners(
    merged_wall: dict, raw_wall: dict, floor_y: float | None = None
) -> list[list[float]]:
    merged_polygon = merged_wall.get("polygonCorners") or []
    raw_polygon = raw_wall.get("polygonCorners") or []
    if len(merged_polygon) >= 3:
        local = merged_polygon
    elif len(raw_polygon) >= 3:
        local = raw_polygon
    else:
        width = float(raw_wall["dimensions"][0]) / 2.0
        height = float(raw_wall["dimensions"][1]) / 2.0
        local = [
            [-width, -height, 0.0],
            [width, -height, 0.0],
            [width, height, 0.0],
            [-width, height, 0.0],
        ]
    corners = corners_to_world(local, merged_wall["transform"])
    if floor_y is None:
        return corners
    bottom = min(corner[1] for corner in corners)
    dy = floor_y - bottom
    return [[corner[0], corner[1] + dy, corner[2]] for corner in corners]
