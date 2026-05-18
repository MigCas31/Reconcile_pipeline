from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from reconcile_tiers._core.svd import SvdFailure, compute_svd
from reconcile_tiers.ingest.merged import MergedDoc

TransformMethod = Literal["floor-svd", "hybrid", "wall-center-svd"]
METHOD_RANK: dict[TransformMethod, int] = {
    "floor-svd": 2,
    "hybrid": 1,
    "wall-center-svd": 0,
}


@dataclass(frozen=True, slots=True)
class RoomTransform:
    rotation: np.ndarray
    translation: np.ndarray
    residual_cm: float
    method: TransformMethod

    def apply(self, corners: Sequence[Sequence[float]]) -> list[list[float]]:
        points = np.asarray(corners, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("corners must be an Nx3 coordinate array")
        transformed = (self.rotation @ points.T).T + self.translation
        return transformed.tolist()


def _merged_data(merged: MergedDoc | dict) -> dict:
    return merged.data if isinstance(merged, MergedDoc) else merged


def _parse_roomplan_transform(flat: Sequence[float]) -> np.ndarray:
    values = np.asarray(flat, dtype=float)
    if values.size != 16:
        raise ValueError(f"expected 16 transform values, got {values.size}")
    return values.reshape((4, 4), order="F")


def _corners_to_roomplan_world(
    corners: Sequence[Sequence[float]],
    transform_flat: Sequence[float],
) -> list[list[float]]:
    transform = _parse_roomplan_transform(transform_flat)
    out: list[list[float]] = []
    for corner in corners:
        if len(corner) < 3:
            raise ValueError("corner must have at least 3 coordinates")
        world = transform @ np.array(
            [float(corner[0]), float(corner[1]), float(corner[2]), 1.0]
        )
        out.append([float(world[0]), float(world[1]), float(world[2])])
    return out


def compute_room_transforms(
    raw_rooms: Sequence[tuple[str, dict]],
    merged: MergedDoc | dict,
) -> dict[str, RoomTransform]:
    data = _merged_data(merged)
    transforms: dict[str, RoomTransform] = {}
    merged_rooms = data.get("rooms") or []

    merged_wall_map: dict[str, dict] = {}
    for merged_room in merged_rooms:
        for wall in merged_room.get("walls", []):
            identifier = wall.get("identifier")
            if identifier:
                merged_wall_map[identifier] = wall

    for room_name, room_data in raw_rooms:
        raw_wall_ids = {
            wall.get("identifier")
            for wall in room_data.get("walls", [])
            if wall.get("identifier")
        }
        if not raw_wall_ids:
            continue

        best_idx = -1
        best_overlap = 0
        for index, merged_room in enumerate(merged_rooms):
            merged_ids = {
                wall.get("identifier")
                for wall in merged_room.get("walls", [])
                if wall.get("identifier")
            }
            overlap = len(raw_wall_ids & merged_ids)
            if overlap > best_overlap:
                best_idx = index
                best_overlap = overlap
        if best_idx < 0:
            continue

        merged_room = merged_rooms[best_idx]
        raw_floors = room_data.get("floors")
        merged_floors = merged_room.get("floors")
        if (
            raw_floors
            and raw_floors[0].get("polygonCorners")
            and merged_floors
            and merged_floors[0].get("polygonCorners")
        ):
            raw_floor = np.asarray(
                _corners_to_roomplan_world(
                    raw_floors[0]["polygonCorners"],
                    raw_floors[0]["transform"],
                )
            )
            merged_floor = np.asarray(
                _corners_to_roomplan_world(
                    merged_floors[0]["polygonCorners"],
                    merged_floors[0]["transform"],
                )
            )
            if len(raw_floor) == len(merged_floor):
                result = compute_svd(raw_floor, merged_floor)
                if not isinstance(result, SvdFailure):
                    rotation, translation, residual_cm = result
                    if residual_cm < 50.0:
                        transforms[room_name] = RoomTransform(
                            rotation=rotation,
                            translation=translation,
                            residual_cm=residual_cm,
                            method="floor-svd",
                        )
                        continue

        src_points: list[np.ndarray] = []
        dst_points: list[np.ndarray] = []
        for raw_wall in room_data.get("walls", []):
            identifier = raw_wall.get("identifier")
            merged_wall = merged_wall_map.get(identifier)
            if merged_wall is None:
                continue
            src_points.append(_parse_roomplan_transform(raw_wall["transform"])[:3, 3])
            dst_points.append(
                _parse_roomplan_transform(merged_wall["transform"])[:3, 3]
            )

        if len(src_points) >= 3:
            result = compute_svd(np.asarray(src_points), np.asarray(dst_points))
            if not isinstance(result, SvdFailure):
                rotation, translation, residual_cm = result
                if residual_cm < 200.0:
                    transforms[room_name] = RoomTransform(
                        rotation=rotation,
                        translation=translation,
                        residual_cm=residual_cm,
                        method="wall-center-svd",
                    )

    return transforms
