from __future__ import annotations

from math import atan2, degrees, hypot

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster

MIN_Y_RANGE_M = 0.15
MAX_Y_RANGE_M = 2.0
MAX_SLOPED_ROOMS_PER_STORY = 2
MAX_AZIMUTH_SPAN_DEG = 90.0


def _plane_azimuth(plane: Plane) -> float:
    return degrees(atan2(-plane.a, -plane.c)) % 360.0


def _plane_inclination(plane: Plane) -> float:
    return degrees(atan2(hypot(plane.a, plane.c), abs(plane.b)))


def detect_simple_slant_rooms(model: BuildingModel) -> set[int]:
    by_story: dict[int, list[tuple[int, float]]] = {}
    for room in model.rooms:
        planes = room.raw_ceiling_planes
        if not planes:
            continue
        corners = [corner for raw in planes for corner in raw.corners]
        if len(corners) < 3:
            continue
        y_values = [float(c[1]) for c in corners]
        y_range = max(y_values) - min(y_values)
        if y_range < MIN_Y_RANGE_M or y_range > MAX_Y_RANGE_M:
            continue
        fit = Plane.fit(corners)
        if isinstance(fit, FitFailure):
            continue
        incl = _plane_inclination(fit)
        if incl <= 5.0 or incl >= 80.0:
            continue
        by_story.setdefault(room.story, []).append((room.index, _plane_azimuth(fit)))

    simple: set[int] = set()
    for entries in by_story.values():
        if len(entries) > MAX_SLOPED_ROOMS_PER_STORY:
            continue
        azimuths = [az for _, az in entries]
        if azimuths and max(azimuths) - min(azimuths) >= MAX_AZIMUTH_SPAN_DEG:
            continue
        simple.update(idx for idx, _ in entries)
    return simple


def build_simple_slant_surfaces(
    model: BuildingModel,
    room_indices: set[int],
    *,
    wall_axis_math: float | None = None,
) -> list[ObliqueSurface]:
    surfaces: list[ObliqueSurface] = []
    if not room_indices:
        return surfaces
    for room in model.rooms:
        if room.index not in room_indices:
            continue
        corners = room.ceiling_polygon
        if len(corners) < 3 and room.raw_ceiling_planes:
            corners = room.raw_ceiling_planes[0].corners
        if len(corners) < 3:
            continue
        fit = Plane.fit(corners)
        if isinstance(fit, FitFailure):
            continue
        if wall_axis_math is not None:
            from reconcile_tiers._core.wall_axis import snap_corners_and_plane_to_axis

            snapped = snap_corners_and_plane_to_axis(
                corners, fit, wall_axis_math=wall_axis_math
            )
            if snapped is not None:
                corners, fit = snapped
        incl = _plane_inclination(fit)
        if incl <= 5.0 or incl >= 80.0:
            continue
        avg_azimuth = _plane_azimuth(fit)
        ref_pt = [
            sum(float(point[0]) for point in corners) / len(corners),
            sum(float(point[1]) for point in corners) / len(corners),
            sum(float(point[2]) for point in corners) / len(corners),
        ]
        surfaces.append(
            ObliqueSurface(
                corners=[
                    [float(point[0]), float(point[1]), float(point[2])]
                    for point in corners
                ],
                plane=fit,
                cluster=RoofCluster(
                    segments=[], avg_incl=incl, avg_azimuth=avg_azimuth, ref_pt=ref_pt
                ),
                dominant_story=room.story,
                ridge=_ridge_for(corners, avg_azimuth),
            )
        )
    return surfaces


def _ridge_for(corners: list[list[float]], avg_azimuth: float) -> dict[str, float]:
    from math import cos, radians, sin

    angle = radians(avg_azimuth + 90.0)
    rx = sin(angle)
    rz = cos(angle)
    projections = [float(point[0]) * rx + float(point[2]) * rz for point in corners]
    return {"x": rx, "z": rz, "min": min(projections), "max": max(projections)}
