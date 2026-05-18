"""Yaw-clustering of ARKit referenceOriginTransform for session detection."""

from __future__ import annotations

import math

import numpy as np

from reconcile_tiers._core.session_clusters import (
    cluster_rooms_by_session,
    yaw_from_transform,
)


def _yaw_transform_16(yaw_deg: float) -> list[float]:
    """Build a 4x4 column-major rotation-about-Y matrix as a flat 16-list."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return list(mat.T.reshape(16))


def test_yaw_from_transform_recovers_input_yaw():
    for yaw in (-170.0, -45.0, 0.0, 30.0, 90.0, 179.0):
        recovered = yaw_from_transform(_yaw_transform_16(yaw))
        assert abs(recovered - yaw) < 1e-6


def test_single_session_building_yields_one_cluster():
    yaws = [10.0, 10.5, 9.8, 10.2, 11.0]
    clusters = cluster_rooms_by_session(yaws, yaw_eps_deg=2.0)
    assert set(clusters.values()) == {0}
    assert clusters == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}


def test_two_distinct_sessions_split_into_two_clusters():
    # First three rooms at yaw ~10°, next two at ~80°.
    yaws = [10.0, 10.5, 9.8, 80.0, 80.5]
    clusters = cluster_rooms_by_session(yaws, yaw_eps_deg=2.0)
    # Session id assigned by first appearance in input order.
    assert clusters[0] == 0
    assert clusters[1] == 0
    assert clusters[2] == 0
    assert clusters[3] == 1
    assert clusters[4] == 1


def test_wraparound_at_180_degrees_is_one_cluster():
    # +179.5° and -179.5° are 1° apart on the circle, should cluster together.
    yaws = [179.5, -179.5, 179.0, -179.0]
    clusters = cluster_rooms_by_session(yaws, yaw_eps_deg=2.0)
    assert len(set(clusters.values())) == 1


def test_yaws_just_outside_eps_split():
    # 2.5° apart with eps=2.0° => two clusters.
    yaws = [0.0, 2.5]
    clusters = cluster_rooms_by_session(yaws, yaw_eps_deg=2.0)
    assert clusters[0] != clusters[1]


def test_none_yaws_become_singleton_sessions():
    # First two rooms share a session; third has no transform; fourth shares with first
    # two.
    yaws = [10.0, 10.5, None, 10.2]
    clusters = cluster_rooms_by_session(yaws, yaw_eps_deg=2.0)
    assert clusters[0] == clusters[1] == clusters[3]
    assert clusters[2] != clusters[0]
    # Each None gets a unique id.
    none_yaws = [None, None]
    none_clusters = cluster_rooms_by_session(none_yaws)
    assert none_clusters[0] != none_clusters[1]


def test_empty_rooms_return_empty_partition():
    assert cluster_rooms_by_session([]) == {}
