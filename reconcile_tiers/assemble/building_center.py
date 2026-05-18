from __future__ import annotations

from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import Vec3


def compute_building_center(model: BuildingModel) -> Vec3:
    points = [
        corner
        for room in model.rooms
        for wall in room.walls_computed
        for corner in wall.corners
        if len(corner) >= 3
    ]
    if not points:
        return Vec3(x=0.0, y=0.0, z=0.0)
    inv = 1.0 / len(points)
    return Vec3(
        x=sum(float(point[0]) for point in points) * inv,
        y=sum(float(point[1]) for point in points) * inv,
        z=sum(float(point[2]) for point in points) * inv,
    )
