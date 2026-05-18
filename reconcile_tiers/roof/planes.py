from __future__ import annotations

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.roof.geometry import (
    plane_normal_from_azimuth_inclination,
    polygon_xz_area,
)
from reconcile_tiers.roof.roof import Footprint, RoofCluster, RoofPlaneCandidate

RIDGE_SPAN_MIN_M = 2.0


def _cluster_points(cluster: RoofCluster) -> list[list[float]]:
    return [point for segment in cluster.segments for point in (segment.a, segment.b)]


def _plane_from_cluster(cluster: RoofCluster) -> Plane | None:
    nx, ny, nz = plane_normal_from_azimuth_inclination(
        cluster.avg_azimuth, cluster.avg_incl
    )
    if abs(ny) < Plane.MIN_NY:
        return None
    if ny < 0.0:
        nx, ny, nz = -nx, -ny, -nz
    ref = cluster.ref_pt
    d = nx * ref[0] + ny * ref[1] + nz * ref[2]
    return Plane(a=nx, b=ny, c=nz, d=d)


def _ridge_span(points: list[list[float]], footprint: Footprint) -> float:
    xs = [p[0] for p in points]
    zs = [p[2] for p in points]
    point_span = max(max(xs) - min(xs), max(zs) - min(zs))
    fp_x = [x for x, _ in footprint.polygon_xz]
    fp_z = [z for _, z in footprint.polygon_xz]
    footprint_span = max(max(fp_x) - min(fp_x), max(fp_z) - min(fp_z))
    return max(point_span, footprint_span)


def build_roof_planes(
    clusters: list[RoofCluster], footprint: Footprint
) -> list[RoofPlaneCandidate]:
    candidates: list[RoofPlaneCandidate] = []
    if (
        not footprint.polygon_xz
        or polygon_xz_area([[x, 0.0, z] for x, z in footprint.polygon_xz]) <= 0.0
    ):
        return candidates
    for cluster in clusters:
        points = _cluster_points(cluster)
        plane = _plane_from_cluster(cluster)
        if plane is None:
            continue
        span = _ridge_span(points, footprint)
        if span < RIDGE_SPAN_MIN_M:
            continue
        candidates.append(
            RoofPlaneCandidate(
                cluster=cluster,
                plane=plane,
                polygon_xz=list(footprint.polygon_xz),
                ridge_span=span,
                dominant_story=cluster.dominant_story,
            )
        )
    return candidates
