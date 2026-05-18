from __future__ import annotations

from math import atan2, cos, degrees, hypot, radians, sin, sqrt

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.geometry import (
    angle_diff_deg,
    plane_normal_from_azimuth_inclination,
    polygon_xz_area,
)
from reconcile_tiers.roof.roof import ClippedRoofPlane, ObliqueSurface, RoofCluster

EAVE_OVERHANG_M = 0.5
RAW_OBLIQUE_MIN_INCL_DEG = 10.0
RAW_OBLIQUE_MAX_INCL_DEG = 75.0
RAW_OBLIQUE_MIN_AREA_M2 = 5.0
RAW_OBLIQUE_MAX_RESIDUAL_M = 0.08
RAW_OBLIQUE_MIN_RIDGE_EDGE_LEN_M = 2.0
RAW_OBLIQUE_MAX_RIDGE_EDGE_ANGLE_DELTA_DEG = 15.0
RAW_OBLIQUE_MAX_DUPLICATE_PLANE_DISTANCE_M = 0.5


def story_floor_y(model: BuildingModel) -> dict[int, float]:
    out: dict[int, float] = {}
    for room in model.rooms:
        if not room.floor_polygon:
            continue
        y = min(float(p[1]) for p in room.floor_polygon)
        out[room.story] = min(out.get(room.story, y), y)
    return out


def _ridge_for(clipped: ClippedRoofPlane) -> dict[str, float]:
    angle = radians(clipped.candidate.cluster.avg_azimuth + 90.0)
    rx = sin(angle)
    rz = cos(angle)
    projections = [x * rx + z * rz for x, z in clipped.polygon_xz]
    return {"x": rx, "z": rz, "min": min(projections), "max": max(projections)}


def build_oblique_surfaces(
    clipped_planes: list[ClippedRoofPlane], floor_y_by_story: dict[int, float]
) -> list[ObliqueSurface]:
    surfaces: list[ObliqueSurface] = []
    for clipped in clipped_planes:
        plane = clipped.candidate.plane
        corners: list[list[float]] = []
        for x, z in clipped.polygon_xz:
            y = plane.y_at(x, z)
            if y is None:
                corners = []
                break
            corners.append([float(x), float(y), float(z)])
        if len(corners) < 3:
            continue
        floor_y = floor_y_by_story.get(clipped.candidate.dominant_story)
        if floor_y is not None:
            corners = _clip_min_y(corners, floor_y)
            if len(corners) < 3:
                continue
        if clipped.max_y is not None:
            corners = _clip_max_y(corners, clipped.max_y)
            if len(corners) < 3:
                continue
        surfaces.append(
            ObliqueSurface(
                corners=corners,
                plane=plane,
                cluster=clipped.candidate.cluster,
                dominant_story=clipped.candidate.dominant_story,
                ridge=_ridge_for(clipped),
            )
        )
    return surfaces


def _clip_min_y(poly: list[list[float]], min_y: float) -> list[list[float]]:
    return _clip_y(poly, min_y, keep_greater=True)


def _clip_max_y(poly: list[list[float]], max_y: float) -> list[list[float]]:
    return _clip_y(poly, max_y, keep_greater=False)


def _clip_y(
    poly: list[list[float]], bound: float, keep_greater: bool
) -> list[list[float]]:
    out: list[list[float]] = []
    n = len(poly)
    for idx, current in enumerate(poly):
        nxt = poly[(idx + 1) % n]
        current_inside = current[1] >= bound if keep_greater else current[1] <= bound
        next_inside = nxt[1] >= bound if keep_greater else nxt[1] <= bound
        if current_inside:
            out.append(current)
        if current_inside != next_inside:
            denom = nxt[1] - current[1]
            if abs(denom) > 1e-9:
                t = (bound - current[1]) / denom
                out.append(
                    [
                        current[0] + t * (nxt[0] - current[0]),
                        bound,
                        current[2] + t * (nxt[2] - current[2]),
                    ]
                )
    return out


def build_raw_oblique_surfaces(
    model: BuildingModel,
    existing: list[ObliqueSurface],
    *,
    wall_axis_math: float | None = None,
) -> list[ObliqueSurface]:
    top_story = max((room.story for room in model.rooms), default=0)
    out: list[ObliqueSurface] = []
    for room in model.rooms:
        if room.story >= top_story:
            continue
        for raw in room.raw_ceiling_planes:
            if len(raw.corners) != 4:
                continue
            fit = Plane.fit(raw.corners)
            if isinstance(fit, FitFailure):
                continue
            corners = list(raw.corners)
            if wall_axis_math is not None:
                from reconcile_tiers._core.wall_axis import (
                    snap_corners_and_plane_to_axis,
                )

                snapped = snap_corners_and_plane_to_axis(
                    corners, fit, wall_axis_math=wall_axis_math
                )
                if snapped is not None:
                    corners, fit = snapped
            if _plane_residual(corners, fit) > RAW_OBLIQUE_MAX_RESIDUAL_M:
                continue
            incl = degrees(atan2(hypot(fit.a, fit.c), abs(fit.b)))
            if incl <= RAW_OBLIQUE_MIN_INCL_DEG or incl >= RAW_OBLIQUE_MAX_INCL_DEG:
                continue
            if not _valid_raw_rectangle(corners):
                continue
            avg_azimuth = degrees(atan2(-fit.a, -fit.c)) % 360.0
            if len(_ridge_like_edges(corners, avg_azimuth)) < 2:
                continue
            ref_pt = [
                sum(p[0] for p in corners) / len(corners),
                sum(p[1] for p in corners) / len(corners),
                sum(p[2] for p in corners) / len(corners),
            ]
            if _duplicates_existing(ref_pt, incl, avg_azimuth, existing):
                continue
            out.append(
                ObliqueSurface(
                    corners=[[float(p[0]), float(p[1]), float(p[2])] for p in corners],
                    plane=fit,
                    cluster=RoofCluster(
                        segments=[],
                        avg_incl=incl,
                        avg_azimuth=avg_azimuth,
                        ref_pt=ref_pt,
                    ),
                    dominant_story=room.story,
                    ridge=_ridge_for_raw(corners, avg_azimuth),
                )
            )
    return out


def _valid_raw_rectangle(corners: list[list[float]]) -> bool:
    xz = {(round(float(point[0]), 6), round(float(point[2]), 6)) for point in corners}
    return len(xz) == 4 and polygon_xz_area(corners) >= RAW_OBLIQUE_MIN_AREA_M2


def _plane_residual(corners: list[list[float]], plane: Plane) -> float:
    return max(
        abs(plane.a * p[0] + plane.b * p[1] + plane.c * p[2] - plane.d) for p in corners
    )


def _ridge_like_edges(
    corners: list[list[float]], avg_azimuth: float
) -> list[tuple[list[float], list[float], float]]:
    ridge_azimuth = (avg_azimuth + 90.0) % 360.0
    edges = []
    for idx, start in enumerate(corners):
        end = corners[(idx + 1) % len(corners)]
        dx = float(end[0]) - float(start[0])
        dz = float(end[2]) - float(start[2])
        length_xz = sqrt(dx * dx + dz * dz)
        if length_xz < RAW_OBLIQUE_MIN_RIDGE_EDGE_LEN_M:
            continue
        edge_az = degrees(atan2(dx, dz)) % 180.0
        delta = min(
            angle_diff_deg(edge_az, ridge_azimuth % 180.0),
            angle_diff_deg((edge_az + 180.0) % 360.0, ridge_azimuth),
        )
        if delta <= RAW_OBLIQUE_MAX_RIDGE_EDGE_ANGLE_DELTA_DEG:
            edges.append((start, end, length_xz))
    return edges


def _duplicates_existing(
    ref_pt: list[float], incl: float, avg_azimuth: float, existing: list[ObliqueSurface]
) -> bool:
    for surface in existing:
        cluster = surface.cluster
        if angle_diff_deg(avg_azimuth, cluster.avg_azimuth) > 10.0:
            continue
        if abs(incl - cluster.avg_incl) > 10.0:
            continue
        n = plane_normal_from_azimuth_inclination(cluster.avg_azimuth, cluster.avg_incl)
        n_len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if n_len <= 1e-9:
            continue
        cluster_ref = cluster.ref_pt
        dist = (
            abs(
                (ref_pt[0] - cluster_ref[0]) * n[0]
                + (ref_pt[1] - cluster_ref[1]) * n[1]
                + (ref_pt[2] - cluster_ref[2]) * n[2]
            )
            / n_len
        )
        if dist <= RAW_OBLIQUE_MAX_DUPLICATE_PLANE_DISTANCE_M:
            return True
    return False


def _ridge_for_raw(corners: list[list[float]], avg_azimuth: float) -> dict[str, float]:
    angle = radians(avg_azimuth + 90.0)
    rx = sin(angle)
    rz = cos(angle)
    projections = [p[0] * rx + p[2] * rz for p in corners]
    return {"x": rx, "z": rz, "min": min(projections), "max": max(projections)}
