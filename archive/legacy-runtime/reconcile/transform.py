"""Coordinate transform utilities for RoomPlan geometry."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

from .models import Floor, Surface, Transform, Vec3


def corners_to_world(corners: list[Vec3], transform: Transform) -> list[Vec3]:
    """Apply 4x4 transform to polygon corners."""
    return transform.apply_corners(corners)


def corners_to_shapely_xz(corners: list[Vec3]) -> Polygon | None:
    """Convert 3D corners to 2D Shapely Polygon on XZ plane (Y is up in RoomPlan)."""
    if len(corners) < 3:
        return None
    coords = [(c.x, c.z) for c in corners]
    # Close the polygon if not already closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def floor_polygon_world(floor: Floor) -> Polygon | None:
    """Get a floor's polygon in world XZ coordinates."""
    if not floor.polygon_corners:
        return None
    world_corners = corners_to_world(floor.polygon_corners, floor.transform)
    return corners_to_shapely_xz(world_corners)


def room_floor_polygon(room) -> Polygon | None:
    """Get the first floor polygon of a room in world coords."""
    if not room.floors:
        return None
    return floor_polygon_world(room.floors[0])


def wall_center_world(wall: Surface) -> Vec3:
    """Get the world-space center of a wall (from transform translation)."""
    return wall.transform.translation


def wall_normal_2d(wall: Surface) -> np.ndarray:
    """Extract wall's normal direction in XZ plane from transform.
    Column 2 of rotation matrix = local Z axis = wall thickness direction.
    """
    rot = wall.transform.matrix[:3, :3]
    normal_3d = rot[:, 2]
    return np.array([normal_3d[0], normal_3d[2]])  # XZ components


def polygon_edges(poly: Polygon) -> list[LineString]:
    """Extract edges of a polygon as LineStrings."""
    coords = list(poly.exterior.coords)
    return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]


def edge_direction(edge: LineString) -> np.ndarray:
    """Unit direction vector of a line segment."""
    coords = list(edge.coords)
    d = np.array(coords[1]) - np.array(coords[0])
    length = np.linalg.norm(d)
    if length < 1e-10:
        return np.array([1.0, 0.0])
    return d / length


def edge_angle_diff(edge_a: LineString, edge_b: LineString) -> float:
    """Angle difference between two edges in degrees (0-90 range)."""
    d_a = edge_direction(edge_a)
    d_b = edge_direction(edge_b)
    cos_angle = abs(np.dot(d_a, d_b))
    cos_angle = min(cos_angle, 1.0)
    return np.degrees(np.arccos(cos_angle))


def edge_perpendicular_distance(edge_a: LineString, edge_b: LineString) -> float:
    """Perpendicular distance between two parallel edges.
    Projects midpoint of edge_b onto the line of edge_a.
    """
    coords_a = list(edge_a.coords)
    a_start = np.array(coords_a[0])
    a_dir = edge_direction(edge_a)
    a_normal = np.array([-a_dir[1], a_dir[0]])

    # Midpoint of edge_b
    coords_b = list(edge_b.coords)
    b_mid = (np.array(coords_b[0]) + np.array(coords_b[1])) / 2

    return abs(np.dot(b_mid - a_start, a_normal))


def edge_overlap_fraction(edge_a: LineString, edge_b: LineString) -> float:
    """
    Fraction of edge_a that overlaps with edge_b when projected onto edge_a's
    direction.
    """
    d = edge_direction(edge_a)
    coords_a = list(edge_a.coords)
    coords_b = list(edge_b.coords)

    a_start = np.array(coords_a[0])
    a_len = edge_a.length

    if a_len < 1e-10:
        return 0.0

    # Project edge_b endpoints onto edge_a direction
    b_projs = [np.dot(np.array(c) - a_start, d) for c in coords_b]
    b_min, b_max = min(b_projs), max(b_projs)

    # Overlap range
    overlap_start = max(0, b_min)
    overlap_end = min(a_len, b_max)
    overlap = max(0, overlap_end - overlap_start)

    return overlap / a_len
