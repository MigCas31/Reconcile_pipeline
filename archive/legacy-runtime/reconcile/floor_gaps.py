"""Per-wall thickness computation from floor polygon gaps between adjacent rooms."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from .models import Building, Room
from .transform import (
    edge_angle_diff,
    edge_overlap_fraction,
    edge_perpendicular_distance,
    polygon_edges,
    room_floor_polygon,
)


@dataclass
class WallThickness:
    room_a_id: str | None
    room_b_id: str | None
    edge_a: LineString  # floor polygon edge from room A
    edge_b: LineString  # floor polygon edge from room B
    thickness_cm: float  # perpendicular distance
    centerline: LineString  # midline between the two edges
    confidence: str  # "high" / "medium" / "low"


def find_adjacent_rooms(
    rooms: list[Room],
    max_gap_m: float = 0.4,
) -> list[tuple[Room, Room, Polygon, Polygon, float]]:
    """Find pairs of rooms whose floor polygons are close but not overlapping."""
    room_polys = []
    for r in rooms:
        poly = room_floor_polygon(r)
        if poly is not None and poly.is_valid and poly.area > 0.01:
            room_polys.append((r, poly))

    pairs = []
    for i, (room_a, poly_a) in enumerate(room_polys):
        for j, (room_b, poly_b) in enumerate(room_polys):
            if j <= i:
                continue
            if room_a.story != room_b.story:
                continue

            dist = poly_a.distance(poly_b)
            if 0 < dist < max_gap_m:
                pairs.append((room_a, room_b, poly_a, poly_b, dist))

    return pairs


def find_parallel_edges(
    poly_a: Polygon,
    poly_b: Polygon,
    angle_tolerance_deg: float = 15.0,
    max_distance_m: float = 0.4,
    min_overlap: float = 0.3,
) -> list[tuple[LineString, LineString, float]]:
    """Find pairs of parallel, close edges between two polygons."""
    edges_a = polygon_edges(poly_a)
    edges_b = polygon_edges(poly_b)

    pairs = []
    for ea in edges_a:
        if ea.length < 0.1:  # skip tiny edges
            continue
        for eb in edges_b:
            if eb.length < 0.1:
                continue

            angle = edge_angle_diff(ea, eb)
            if angle > angle_tolerance_deg:
                continue

            dist = edge_perpendicular_distance(ea, eb)
            if dist > max_distance_m or dist < 0.005:  # skip overlapping or far edges
                continue

            overlap = edge_overlap_fraction(ea, eb)
            if overlap < min_overlap:
                continue

            pairs.append((ea, eb, dist))

    return pairs


def compute_centerline(edge_a: LineString, edge_b: LineString) -> LineString:
    """Compute the midline between two parallel edges."""
    n_points = 10
    points_a = [
        edge_a.interpolate(i / n_points, normalized=True) for i in range(n_points + 1)
    ]
    midpoints = []
    for pa in points_a:
        # Find nearest point on edge_b
        pb = edge_b.interpolate(edge_b.project(Point(pa.x, pa.y)))
        midpoints.append(((pa.x + pb.x) / 2, (pa.y + pb.y) / 2))

    return LineString(midpoints)


def compute_wall_thicknesses(
    building: Building,
) -> list[WallThickness]:
    """Compute per-wall thickness from floor polygon gaps.

    Uses the merged building's rooms (which have correct placement via transforms).
    """
    adjacent = find_adjacent_rooms(building.rooms)
    thicknesses = []

    for room_a, room_b, poly_a, poly_b, _approx_dist in adjacent:
        parallel = find_parallel_edges(poly_a, poly_b)

        for edge_a, edge_b, dist in parallel:
            centerline = compute_centerline(edge_a, edge_b)
            thickness_cm = dist * 100

            # Classify confidence
            if thickness_cm > 8:
                confidence = "high"
            elif thickness_cm > 3:
                confidence = "medium"
            else:
                confidence = "low"

            thicknesses.append(
                WallThickness(
                    room_a_id=room_a.identifier,
                    room_b_id=room_b.identifier,
                    edge_a=edge_a,
                    edge_b=edge_b,
                    thickness_cm=thickness_cm,
                    centerline=centerline,
                    confidence=confidence,
                )
            )

    return thicknesses
