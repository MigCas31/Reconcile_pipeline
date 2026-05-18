from __future__ import annotations

from dataclasses import replace

from reconcile_tiers.roof.geometry import (
    angle_diff_deg,
    circular_mean_deg,
    plane_normal_from_azimuth_inclination,
)
from reconcile_tiers.roof.roof import RoofCluster, RoofSegment

AZIMUTH_TOL_DEG = 30.0
INCL_TOL_DEG = 15.0
COPLANAR_TOL_M = 0.5
MIN_CLUSTER_SIZE = 2

# Architectural prior (LAYER 2): clusters whose `avg_azimuth` is within this
# many degrees of an axis-aligned target snap to that target. Above this,
# the deviation is a real architectural feature; leave alone.
CLUSTER_AXIS_SNAP_TOL_DEG = 20.0


def _midpoint_plane_residual(cluster: RoofCluster, segment: RoofSegment) -> float:
    nx, ny, nz = plane_normal_from_azimuth_inclination(
        cluster.avg_azimuth, cluster.avg_incl
    )
    denom = (nx * nx + ny * ny + nz * nz) ** 0.5
    if denom <= 1e-12:
        return float("inf")
    mid = segment.midpoint
    ref = cluster.ref_pt
    return (
        abs((mid[0] - ref[0]) * nx + (mid[1] - ref[1]) * ny + (mid[2] - ref[2]) * nz)
        / denom
    )


def _cluster_with(cl: RoofCluster, segment: RoofSegment) -> RoofCluster | None:
    if angle_diff_deg(segment.azimuth, cl.avg_azimuth) >= AZIMUTH_TOL_DEG:
        return None
    if abs(segment.incl - cl.avg_incl) >= INCL_TOL_DEG:
        return None
    new_segments = [*cl.segments, segment]
    if _midpoint_plane_residual(cl, segment) > COPLANAR_TOL_M:
        return None
    count = len(new_segments)
    return RoofCluster(
        segments=new_segments,
        avg_incl=sum(s.incl for s in new_segments) / count,
        avg_azimuth=circular_mean_deg(s.azimuth for s in new_segments),
        ref_pt=cl.ref_pt,
    )


def _snap_cluster_to_wall_axis(
    cluster: RoofCluster,
    wall_axis_math: float,
    tol_deg: float,
) -> RoofCluster:
    """LAYER 2: snap cluster's `avg_azimuth` to the nearest wall-axis-aligned
    compass direction if within `tol_deg`.

    Compass / math convention reflection means the axis-aligned slope_az
    candidates are `{-axis_math + k*90}` (mod 360); see `_core/wall_axis.py`.
    """
    from reconcile_tiers._core.wall_axis import nearest_axis_aligned_compass

    target = nearest_axis_aligned_compass(cluster.avg_azimuth, wall_axis_math)
    diff = abs((cluster.avg_azimuth - target) % 360.0)
    delta = min(diff, 360.0 - diff)
    if delta > tol_deg:
        return cluster
    return replace(cluster, avg_azimuth=target)


def cluster_oblique_segments(
    segments: list[RoofSegment],
    *,
    wall_axis_math: float | None = None,
) -> list[RoofCluster]:
    """Cluster oblique segments by azimuth + inclination + coplanarity.

    When `wall_axis_math` is supplied (LAYER 2 of architectural priors),
    clusters whose averaged direction is close to a wall-axis-aligned target
    snap to that target. The plane built from each cluster (in
    `roof/planes.py:_plane_from_cluster`) is parametric in `avg_azimuth`, so
    snapping the cluster direction is equivalent to a constrained plane fit.
    """
    clusters: list[RoofCluster] = []
    for segment in segments:
        assigned = False
        for idx, cluster in enumerate(clusters):
            updated = _cluster_with(cluster, segment)
            if updated is None:
                continue
            clusters[idx] = updated
            assigned = True
            break
        if not assigned:
            clusters.append(
                RoofCluster(
                    segments=[segment],
                    avg_incl=segment.incl,
                    avg_azimuth=segment.azimuth % 360.0,
                    ref_pt=segment.midpoint,
                )
            )
    out = [
        replace(cluster, avg_azimuth=cluster.avg_azimuth % 360.0)
        for cluster in clusters
        if len(cluster.segments) >= MIN_CLUSTER_SIZE
    ]
    if wall_axis_math is not None:
        out = [
            _snap_cluster_to_wall_axis(c, wall_axis_math, CLUSTER_AXIS_SNAP_TOL_DEG)
            for c in out
        ]
    return out
