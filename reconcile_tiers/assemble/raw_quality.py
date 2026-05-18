"""v2-only: cleanliness score for raw scanned ceiling planes.

The score is the weighted product of three signals available without scan
point clouds:

1. fit_residual_score: how well the four scanned corners lie on a fitted plane.
   Captures internal noise — a plane that bows or warps in the middle has high
   residual.
2. boundary_support_score: fraction of the plane perimeter that runs within
   ``SUPPORT_TOLERANCE_M`` of a wall-top from the same room. Captures whether
   the plane is anchored to room geometry or floats.
3. oblique_agreement_score: 1.0 if no coplanar computed oblique competes;
   bonus if a coplanar oblique exists with matching orientation; penalty if a
   nearby oblique disagrees in normal direction. Captures whether the raw
   plane is corroborated by the cluster-derived obliques.

A point-density component (points per m² from the source point cloud) is
omitted because the extraction pipeline does not currently retain raw scan
points. It can be added later as a fourth multiplicand without changing
callers.
"""

from __future__ import annotations

from shapely.geometry import LineString, Polygon

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract.building import ExtractedRoom, RawCeilingPlane
from reconcile_tiers.roof.roof import ObliqueSurface

SUPPORT_TOLERANCE_M = 0.30
RESIDUAL_REF_M = 0.05
COPLANAR_NORMAL_TOL = 0.087  # ~5 deg between unit normals
DISAGREEMENT_NORMAL_TOL = 0.26  # ~15 deg
AGREEMENT_BONUS = 1.2
DISAGREEMENT_PENALTY = 0.5


def score_raw_plane(
    plane: RawCeilingPlane,
    room: ExtractedRoom,
    obliques: list[ObliqueSurface],
) -> float:
    if len(plane.corners) < 3:
        return 0.0
    fit_score = _fit_residual_score(plane.corners)
    support_score = _boundary_support_score(plane.corners, room)
    agreement_score = _oblique_agreement_score(plane.corners, obliques)
    return max(0.0, min(1.0, fit_score * support_score * agreement_score))


def _fit_residual_score(corners: list[list[float]]) -> float:
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return 0.0
    residuals = [
        abs(plane.a * c[0] + plane.b * c[1] + plane.c * c[2] - plane.d) for c in corners
    ]
    if not residuals:
        return 0.0
    rms = (sum(r * r for r in residuals) / len(residuals)) ** 0.5
    return max(0.0, 1.0 - min(rms / RESIDUAL_REF_M, 1.0))


def _boundary_support_score(corners: list[list[float]], room: ExtractedRoom) -> float:
    poly = _xz_polygon(corners)
    if poly is None:
        return 0.0
    perimeter = poly.length
    if perimeter <= 1e-9:
        return 0.0
    wall_tops: list[LineString] = []
    for wall in room.walls_merged + room.walls_computed:
        if len(wall.corners) < 2:
            continue
        top_pts = sorted(wall.corners, key=lambda p: p[1], reverse=True)[:2]
        if len(top_pts) != 2:
            continue
        line = LineString(
            [(top_pts[0][0], top_pts[0][2]), (top_pts[1][0], top_pts[1][2])]
        )
        if line.length > 1e-9:
            wall_tops.append(line)
    if not wall_tops:
        return 0.5  # no wall data — neutral, don't fully penalize
    boundary = poly.exterior
    supported = 0.0
    sample_step = max(perimeter / 64.0, 0.05)
    distance = 0.0
    while distance < perimeter:
        pt = boundary.interpolate(distance)
        if any(line.distance(pt) <= SUPPORT_TOLERANCE_M for line in wall_tops):
            supported += sample_step
        distance += sample_step
    return min(1.0, supported / perimeter)


def _oblique_agreement_score(
    corners: list[list[float]], obliques: list[ObliqueSurface]
) -> float:
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return 1.0
    raw_normal = (plane.a, plane.b, plane.c)
    poly = _xz_polygon(corners)
    if poly is None or poly.area < 1e-6:
        return 1.0
    best = 1.0
    for surface in obliques:
        surface_poly = _xz_polygon(surface.corners)
        if surface_poly is None or surface_poly.area < 1e-6:
            continue
        try:
            inter = poly.intersection(surface_poly).area
        except Exception:
            inter = 0.0
        if inter < 0.5 * min(poly.area, surface_poly.area):
            continue
        oblique_normal = (surface.plane.a, surface.plane.b, surface.plane.c)
        delta = _normal_delta(raw_normal, oblique_normal)
        if delta <= COPLANAR_NORMAL_TOL:
            best = max(best, AGREEMENT_BONUS)
        elif delta >= DISAGREEMENT_NORMAL_TOL:
            best = min(best, DISAGREEMENT_PENALTY)
    return min(1.0, best)


def _normal_delta(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    ax, ay, az = a
    bx, by, bz = b
    dot = ax * bx + ay * by + az * bz
    return abs(1.0 - abs(dot))


def _xz_polygon(corners: list[list[float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = make_valid(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    except Exception:
        return None
    if poly.is_empty:
        return None
    if isinstance(poly, Polygon):
        return poly if poly.area > 1e-9 else None
    parts = [
        part
        for part in getattr(poly, "geoms", [])
        if isinstance(part, Polygon) and part.area > 1e-9
    ]
    if not parts:
        return None
    return max(parts, key=lambda part: part.area)
