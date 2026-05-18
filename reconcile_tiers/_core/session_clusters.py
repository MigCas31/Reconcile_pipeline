"""Cluster rooms by ARKit session via referenceOriginTransform yaw.

Rooms scanned within the same ARKit session share the same world-tracking
origin, so their `referenceOriginTransform` rotation (yaw about Y) is fixed
per session even as device translation drifts. A different yaw means a
different session -- surveyors restarted scanning, ARKit lost tracking, etc.

The algorithm itself is a DBSCAN-style 1D scan on a circle (yaws wrap at
+/-180°). Indices in the returned dict align with positions in the input
sequence; callers map those to whatever room identity they prefer.

Algorithm and the 2.0° yaw eps were validated in `detect_session_breaks.py`:
on a 8-problem / 5-control building cohort, 7/8 problem buildings had >=3
yaw clusters at this eps, vs 0/5 control buildings with >2.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from reconcile_tiers.ingest.merged import MergedRoom


def yaw_from_transform(transform_16: Sequence[float]) -> float:
    """Yaw in degrees from an Apple 4x4 column-major affine transform.

    `transform_16` is the 16-element flat list as serialised by RoomPlan.
    Returns the angle of the rotation submatrix's first column projected
    into the XZ plane (rotation about Y, the up axis).
    """
    if len(transform_16) != 16:
        raise ValueError(f"transform must have 16 values, got {len(transform_16)}")
    mat = np.asarray(transform_16, dtype=np.float64).reshape(4, 4).T
    R = mat[:3, :3]
    return math.degrees(math.atan2(R[0, 2], R[0, 0]))


def yaws_from_merged_rooms(merged_rooms: Sequence[MergedRoom]) -> list[float | None]:
    """Per-room yaw aligned with `merged_rooms` order. None when no transform."""
    out: list[float | None] = []
    for room in merged_rooms:
        transform = room.reference_origin_transform
        if not transform or len(transform) != 16:
            out.append(None)
            continue
        out.append(yaw_from_transform(transform))
    return out


def _cluster_yaws_indexed(
    indexed_yaws: list[tuple[int, float]],
    *,
    eps_deg: float,
) -> list[list[int]]:
    """Sort by yaw, single-pass adjacency clustering, then merge wrap-around."""
    if not indexed_yaws:
        return []
    sorted_iy = sorted(indexed_yaws, key=lambda iy: iy[1])
    clusters: list[list[tuple[int, float]]] = [[sorted_iy[0]]]
    for iy in sorted_iy[1:]:
        prev_yaw = clusters[-1][-1][1]
        diff = abs(iy[1] - prev_yaw)
        diff = min(diff, 360.0 - diff)
        if diff <= eps_deg:
            clusters[-1].append(iy)
        else:
            clusters.append([iy])
    if len(clusters) > 1:
        first_yaw = clusters[0][0][1]
        last_yaw = clusters[-1][-1][1]
        wrap = min(
            abs(last_yaw - first_yaw),
            abs(last_yaw - first_yaw + 360.0),
            abs(last_yaw - first_yaw - 360.0),
        )
        if wrap <= eps_deg:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
    return [[idx for idx, _ in cluster] for cluster in clusters]


def cluster_rooms_by_session(
    yaws: Sequence[float | None],
    *,
    yaw_eps_deg: float = 2.0,
) -> dict[int, int]:
    """Map input position -> session_id.

    `yaws[i]` is the yaw (degrees) for room at position `i`, or None when
    no `referenceOriginTransform` is available. Rooms with None yaws each
    receive their own singleton session_id, since their session identity
    cannot be inferred and conflating them would be unsound.

    Session IDs are assigned in order of first appearance in the input
    sequence so the partition is deterministic across runs.
    """
    indexed_known: list[tuple[int, float]] = []
    none_indices: list[int] = []
    for idx, yaw in enumerate(yaws):
        if yaw is None:
            none_indices.append(idx)
        else:
            indexed_known.append((idx, float(yaw)))

    clusters = _cluster_yaws_indexed(indexed_known, eps_deg=yaw_eps_deg)

    # Assign session ids by first appearance in the input order.
    cluster_of_idx: dict[int, frozenset[int]] = {}
    for cluster in clusters:
        members = frozenset(cluster)
        for idx in cluster:
            cluster_of_idx[idx] = members

    session_of_cluster: dict[frozenset[int], int] = {}
    out: dict[int, int] = {}
    next_id = 0
    for idx in range(len(yaws)):
        if idx in cluster_of_idx:
            members = cluster_of_idx[idx]
            sid = session_of_cluster.get(members)
            if sid is None:
                sid = next_id
                next_id += 1
                session_of_cluster[members] = sid
            out[idx] = sid
    for idx in none_indices:
        out[idx] = next_id
        next_id += 1
    return out


def cluster_merged_rooms_by_session(
    merged_rooms: Sequence[MergedRoom],
    *,
    yaw_eps_deg: float = 2.0,
) -> dict[int, int]:
    """Convenience: extract yaws from `MergedRoom`s and cluster.

    Returned dict maps `MergedRoom.index` -> session_id. When the input
    sequence is the canonical merged-rooms order (which already aligns
    `MergedRoom.index` with sequence position), this matches
    `cluster_rooms_by_session` exactly; the explicit re-keying makes the
    contract robust to callers that pass a filtered subset.
    """
    yaws = yaws_from_merged_rooms(merged_rooms)
    by_position = cluster_rooms_by_session(yaws, yaw_eps_deg=yaw_eps_deg)
    return {merged_rooms[pos].index: sid for pos, sid in by_position.items()}
