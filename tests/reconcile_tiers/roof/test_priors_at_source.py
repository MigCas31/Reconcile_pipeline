"""Unit tests for priors-at-source LAYER 1 (segment filter), LAYER 2
(cluster azimuth snap), and LAYER 3 (emission-time plane snap), plus the
shared wall-axis kernel in `_core/wall_axis.py`.
"""

from __future__ import annotations

import math

from reconcile_tiers._core.plane import Plane
from reconcile_tiers._core.wall_axis import (
    axis_misalignment_deg,
    nearest_axis_aligned_compass,
    principal_axis_and_coverage,
)
from reconcile_tiers.roof.clustering import (
    CLUSTER_AXIS_SNAP_TOL_DEG,
    _snap_cluster_to_wall_axis,
    cluster_oblique_segments,
)
from reconcile_tiers.roof.roof import RoofCluster, RoofSegment

# ---- wall-axis kernel -------------------------------------------------------


def _wall_quad(
    x1: float, z1: float, x2: float, z2: float, height: float = 2.5
) -> list[list[float]]:
    return [
        [x1, 0.0, z1],
        [x2, 0.0, z2],
        [x2, height, z2],
        [x1, height, z1],
    ]


def test_principal_axis_axis_aligned_walls_give_zero() -> None:
    walls = [
        _wall_quad(0, 0, 8, 0),
        _wall_quad(8, 0, 8, 8),
        _wall_quad(8, 8, 0, 8),
        _wall_quad(0, 8, 0, 0),
    ]
    info = principal_axis_and_coverage(walls)
    assert info is not None
    axis, coverage = info
    assert (
        axis == 0.0
    )  # walls along +/-X (math 0 deg) and +/-Z (math 90 deg), folded mod 90 -> 0 deg
    assert coverage == 1.0


def test_principal_axis_rotated_walls_recover_angle() -> None:
    """Walls rotated 60 deg in XZ -- the kernel should recover axis ~ 60 deg."""
    rot = math.radians(60.0)
    cos_r, sin_r = math.cos(rot), math.sin(rot)

    def rot_pt(x: float, z: float) -> tuple[float, float]:
        return (x * cos_r - z * sin_r, x * sin_r + z * cos_r)

    base = [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)]
    rotated = [rot_pt(x, z) for x, z in base]
    walls = [_wall_quad(*rotated[i], *rotated[i + 1]) for i in range(4)]
    info = principal_axis_and_coverage(walls)
    assert info is not None
    axis, coverage = info
    assert abs(axis - 60.0) < 2.0
    assert coverage > 0.95


def test_axis_misalignment_compass_math_convention() -> None:
    """Regression for the compass-vs-math convention bug.

    Walls at math 60 deg (direction vector (cos 60 deg, sin 60 deg) in XZ). A slope
    perpendicular to walls has down-direction (-sin 60 deg, cos 60 deg) -> compass
    `atan2(-sin 60 deg, cos 60 deg) ≡ 300 deg`. A slope parallel to walls has compass
    `atan2(cos 60 deg, sin 60 deg) ≡ 30 deg`. Both axis-aligned: sum + axis ≡ 0 mod 90.
    """
    assert axis_misalignment_deg(300.0, 60.0) < 0.01  # perpendicular
    assert axis_misalignment_deg(30.0, 60.0) < 0.01  # parallel
    # Off-axis case from 7cabc39b: compass 119.6 deg on axis 60 deg aligns within 0.4
    # deg
    assert axis_misalignment_deg(119.6, 60.0) < 1.0


def test_nearest_axis_aligned_compass_picks_nearest() -> None:
    # Wall axis math 60 deg -> axis-aligned compass candidates {-60, 30, 120, 210, 300}
    # mod 360.
    # A slope at compass 119.6 deg should snap to 120 deg.
    target = nearest_axis_aligned_compass(119.6, 60.0)
    assert abs(target - 120.0) < 0.01


# ---- LAYER 2: cluster snap --------------------------------------------------


def _segment(azimuth_compass: float, incl_deg: float = 30.0) -> RoofSegment:
    return RoofSegment(
        a=[0.0, 0.0, 0.0],
        b=[1.0, 1.0, 1.0],
        incl=incl_deg,
        azimuth=azimuth_compass,
        length=1.0,
        story=0,
        room_index=0,
        wall_id="w",
    )


def _cluster_with_az(azimuth_compass: float) -> RoofCluster:
    seg = _segment(azimuth_compass)
    return RoofCluster(
        segments=[seg, seg],
        avg_incl=30.0,
        avg_azimuth=azimuth_compass,
        ref_pt=[0.0, 0.0, 0.0],
    )


def test_cluster_snap_pulls_close_azimuth_to_axis() -> None:
    # Wall axis math 60 deg, cluster avg 119.6 deg -> should snap to 120 deg.
    cluster = _cluster_with_az(119.6)
    snapped = _snap_cluster_to_wall_axis(cluster, wall_axis_math=60.0, tol_deg=20.0)
    assert abs(snapped.avg_azimuth - 120.0) < 0.01


def test_cluster_snap_leaves_far_azimuth_alone() -> None:
    # Wall axis math 60 deg, cluster avg 95 deg (delta to nearest target 120 deg = 25
    # deg, > 20 deg tol).
    cluster = _cluster_with_az(95.0)
    snapped = _snap_cluster_to_wall_axis(cluster, wall_axis_math=60.0, tol_deg=20.0)
    assert snapped.avg_azimuth == 95.0


def test_cluster_oblique_segments_no_axis_is_identity() -> None:
    """Without `wall_axis_math`, behavior matches the pre-priors version."""
    segs = [_segment(119.6), _segment(119.5), _segment(119.7)]
    out = cluster_oblique_segments(segs)
    assert len(out) == 1
    assert abs(out[0].avg_azimuth - 119.6) < 0.5


def test_cluster_oblique_segments_with_axis_snaps_avg() -> None:
    segs = [_segment(119.6), _segment(119.5), _segment(119.7)]
    out = cluster_oblique_segments(segs, wall_axis_math=60.0)
    assert len(out) == 1
    assert abs(out[0].avg_azimuth - 120.0) < 0.01


def test_cluster_snap_tol_constant_is_reasonable() -> None:
    # Sanity: 20 deg matches the audit's segment/eave tolerances.
    assert CLUSTER_AXIS_SNAP_TOL_DEG == 20.0


# ---- LAYER 3: emission-time plane snap -------------------------------------


def _slope_az_deg(plane: Plane) -> float:
    return math.degrees(math.atan2(-plane.a, -plane.c)) % 360.0


def _incl_deg(plane: Plane) -> float:
    return math.degrees(math.atan2(math.hypot(plane.a, plane.c), abs(plane.b)))


def test_snap_aligns_near_axis_plane_to_target() -> None:
    # 4-corner roof patch in XZ, slope_az ~119.6 deg (math axis 60 deg wall: target 120
    # deg).
    from reconcile_tiers._core.wall_axis import snap_corners_and_plane_to_axis

    az_rad = math.radians(119.6)
    incl_rad = math.radians(30.0)
    a = -math.sin(az_rad) * math.sin(incl_rad)
    c = -math.cos(az_rad) * math.sin(incl_rad)
    b = math.cos(incl_rad)
    plane = Plane(a=a, b=b, c=c, d=b * 5.0)  # plane at y≈5
    corners = [
        [0.0, plane.y_at(0.0, 0.0) or 0.0, 0.0],
        [4.0, plane.y_at(4.0, 0.0) or 0.0, 0.0],
        [4.0, plane.y_at(4.0, 4.0) or 0.0, 4.0],
        [0.0, plane.y_at(0.0, 4.0) or 0.0, 4.0],
    ]
    snapped = snap_corners_and_plane_to_axis(corners, plane, wall_axis_math=60.0)
    assert snapped is not None
    new_corners, new_plane = snapped
    assert abs(_slope_az_deg(new_plane) - 120.0) < 0.05
    # Inclination preserved.
    assert abs(_incl_deg(new_plane) - 30.0) < 0.05
    # XZ unchanged; only Y recomputed.
    for old, new in zip(corners, new_corners, strict=False):
        assert abs(new[0] - old[0]) < 1e-9
        assert abs(new[2] - old[2]) < 1e-9


def test_snap_returns_none_for_far_off_axis() -> None:
    from reconcile_tiers._core.wall_axis import snap_corners_and_plane_to_axis

    # slope_az 95 deg on math axis 60 deg -> axis_misalignment ≈ 25 deg, > 20 deg tol.
    az_rad = math.radians(95.0)
    incl_rad = math.radians(30.0)
    a = -math.sin(az_rad) * math.sin(incl_rad)
    c = -math.cos(az_rad) * math.sin(incl_rad)
    b = math.cos(incl_rad)
    plane = Plane(a=a, b=b, c=c, d=b * 5.0)
    corners = [[0.0, 5.0, 0.0], [3.0, 5.0, 0.0], [3.0, 5.0, 3.0], [0.0, 5.0, 3.0]]
    assert snap_corners_and_plane_to_axis(corners, plane, wall_axis_math=60.0) is None


def test_snap_returns_none_for_near_flat_plane() -> None:
    from reconcile_tiers._core.wall_axis import snap_corners_and_plane_to_axis

    # Pure horizontal: a=c=0, no slope direction.
    plane = Plane(a=0.0, b=1.0, c=0.0, d=2.5)
    corners = [[0.0, 2.5, 0.0], [4.0, 2.5, 0.0], [4.0, 2.5, 4.0], [0.0, 2.5, 4.0]]
    assert snap_corners_and_plane_to_axis(corners, plane, wall_axis_math=60.0) is None


def test_snap_preserves_centroid_height() -> None:
    """Snap is volume-conserving at the corner-cloud centroid: average Y is
    unchanged (the rotation pivots around the centroid)."""
    from reconcile_tiers._core.wall_axis import snap_corners_and_plane_to_axis

    az_rad = math.radians(119.6)
    incl_rad = math.radians(30.0)
    a = -math.sin(az_rad) * math.sin(incl_rad)
    c = -math.cos(az_rad) * math.sin(incl_rad)
    b = math.cos(incl_rad)
    plane = Plane(a=a, b=b, c=c, d=b * 5.0)
    corners = [
        [0.0, plane.y_at(0.0, 0.0) or 0.0, 0.0],
        [4.0, plane.y_at(4.0, 0.0) or 0.0, 0.0],
        [4.0, plane.y_at(4.0, 4.0) or 0.0, 4.0],
        [0.0, plane.y_at(0.0, 4.0) or 0.0, 4.0],
    ]
    old_avg_y = sum(p[1] for p in corners) / 4
    snapped = snap_corners_and_plane_to_axis(corners, plane, wall_axis_math=60.0)
    assert snapped is not None
    new_corners, _ = snapped
    new_avg_y = sum(p[1] for p in new_corners) / 4
    assert abs(new_avg_y - old_avg_y) < 1e-6


def test_layer3_tol_constant_matches_layer2() -> None:
    from reconcile_tiers._core.wall_axis import PLANE_AXIS_SNAP_TOL_DEG

    assert PLANE_AXIS_SNAP_TOL_DEG == CLUSTER_AXIS_SNAP_TOL_DEG
