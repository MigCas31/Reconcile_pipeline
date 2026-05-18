"""Compute per-room rigid transforms from raw room space to merged building space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Building, Room, Vec3
from .transform import corners_to_world


@dataclass
class RoomTransform:
    room_id: str | None
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3
    residual_cm: float

    def apply(self, point: Vec3) -> Vec3:
        p = np.array([point.x, point.y, point.z])
        q = self.rotation @ p + self.translation
        return Vec3(float(q[0]), float(q[1]), float(q[2]))

    def apply_list(self, points: list[Vec3]) -> list[Vec3]:
        return [self.apply(p) for p in points]

    def apply_4x4(self, matrix: np.ndarray) -> np.ndarray:
        """Apply this room transform to a 4x4 surface transform matrix."""
        result = matrix.copy()
        # Rotate the rotation part
        result[:3, :3] = self.rotation @ matrix[:3, :3]
        # Rotate + translate the translation part
        result[:3, 3] = self.rotation @ matrix[:3, 3] + self.translation
        return result


def _compute_rigid_transform(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute rigid transform (R, t) mapping src -> dst using SVD.
    Returns (R, t, max_residual_cm).
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)

    H = (src - src_c).T @ (dst - dst_c)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_c - R @ src_c

    transformed = (R @ src.T).T + t
    residual = np.max(np.linalg.norm(dst - transformed, axis=1)) * 100

    return R, t, residual


def compute_room_transforms(
    merged_building: Building,
    raw_rooms: list[Room],
) -> dict[str, RoomTransform]:
    """Compute rigid transforms from raw room space to merged building space.

    For each raw room, finds the merged room that shares wall UUIDs and uses
    their floor polygon corners as anchor points. If a raw room's walls were
    all dropped by the merge, falls back to matching by floor polygon shape.

    Returns:
        Dict mapping raw room identifier -> RoomTransform
    """
    # Build wall UUID -> raw room index
    raw_wall_to_room_id = {}
    for rr in raw_rooms:
        for w in rr.walls + rr.doors + rr.windows:
            raw_wall_to_room_id[w.identifier] = rr.identifier

    # For each raw room, find ALL merged rooms that share wall UUIDs
    raw_to_merged: dict[str, list] = {rr.identifier: [] for rr in raw_rooms}
    for mr in merged_building.rooms:
        for w in mr.walls + mr.doors + mr.windows:
            raw_id = raw_wall_to_room_id.get(w.identifier)
            if raw_id is not None and mr not in raw_to_merged[raw_id]:
                raw_to_merged[raw_id].append(mr)

    # Pre-compute merged floor polygons for fallback matching
    merged_floors = []
    for mr in merged_building.rooms:
        if mr.floors and mr.floors[0].polygon_corners:
            mc = corners_to_world(mr.floors[0].polygon_corners, mr.floors[0].transform)
            merged_floors.append((mr, np.array([[c.x, c.y, c.z] for c in mc])))

    transforms = {}

    for rr in raw_rooms:
        if not rr.floors or not rr.floors[0].polygon_corners:
            continue

        rc = corners_to_world(rr.floors[0].polygon_corners, rr.floors[0].transform)
        rc_arr = np.array([[c.x, c.y, c.z] for c in rc])

        # Try matched merged rooms first
        best_R, best_t, best_res = None, None, float("inf")

        for mr in raw_to_merged[rr.identifier]:
            if not mr.floors or not mr.floors[0].polygon_corners:
                continue
            mc = corners_to_world(mr.floors[0].polygon_corners, mr.floors[0].transform)
            mc_arr = np.array([[c.x, c.y, c.z] for c in mc])
            if len(mc_arr) != len(rc_arr):
                continue
            R, t, res = _compute_rigid_transform(rc_arr, mc_arr)
            if res < best_res:
                best_R, best_t, best_res = R, t, res

        # Fallback: find merged floor polygon with same corner count and closest match
        if best_R is None:
            for _mr, mc_arr in merged_floors:
                if len(mc_arr) != len(rc_arr):
                    continue
                R, t, res = _compute_rigid_transform(rc_arr, mc_arr)
                if res < best_res:
                    best_R, best_t, best_res = R, t, res

        if best_R is not None and best_res < 50.0:  # max 50cm residual
            transforms[rr.identifier] = RoomTransform(
                room_id=rr.identifier,
                rotation=best_R,
                translation=best_t,
                residual_cm=best_res,
            )

    return transforms
