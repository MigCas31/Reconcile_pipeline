"""Load merged.json and scan-cache room JSONs into data models."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .models import Building, Floor, Room, Surface, Transform, Vec3

# Files in scan-cache that are NOT individual room scans
_SKIP_FILES = {"data.json", "arworldmap.json"}
_SKIP_PREFIXES = ("ceiling_", "merged_")


def _parse_surface(data: dict) -> Surface:
    dims = data["dimensions"]
    return Surface(
        identifier=data["identifier"],
        category=data["category"],
        confidence=data["confidence"],
        dimensions=Vec3(dims[0], dims[1], dims[2]),
        transform=Transform.from_flat(data["transform"]),
        polygon_corners=[Vec3.from_list(c) for c in data.get("polygonCorners", [])],
        story=data.get("story", 0),
        completed_edges=data.get("completedEdges", []),
        parent_identifier=data.get("parentIdentifier"),
        curve=data.get("curve"),
    )


def _parse_floor(data: dict) -> Floor:
    dims = data["dimensions"]
    return Floor(
        identifier=data["identifier"],
        category=data["category"],
        confidence=data["confidence"],
        dimensions=Vec3(dims[0], dims[1], dims[2]),
        transform=Transform.from_flat(data["transform"]),
        polygon_corners=[Vec3.from_list(c) for c in data.get("polygonCorners", [])],
        story=data.get("story", 0),
        completed_edges=data.get("completedEdges", []),
        parent_identifier=data.get("parentIdentifier"),
        curve=data.get("curve"),
    )


def _parse_room(data: dict, room_id: str | None = None) -> Room:
    return Room(
        identifier=room_id,
        walls=[_parse_surface(w) for w in data.get("walls", [])],
        doors=[_parse_surface(d) for d in data.get("doors", [])],
        windows=[_parse_surface(w) for w in data.get("windows", [])],
        floors=[_parse_floor(f) for f in data.get("floors", [])],
        story=data.get("story", 0),
        reference_origin_transform=(
            Transform.from_flat(data["referenceOriginTransform"])
            if data.get("referenceOriginTransform")
            else None
        ),
    )


def load_merged(path: Path) -> Building:
    """Load a merged.json into a Building model."""
    with open(path) as f:
        data = json.load(f)

    # Rooms in merged.json don't have identifiers — derive from wall UUIDs
    rooms = []
    for i, room_data in enumerate(data.get("rooms", [])):
        # Use first wall's identifier as room ID, or index
        [w["identifier"] for w in room_data.get("walls", [])]
        room_id = f"merged_room_{i}"
        rooms.append(_parse_room(room_data, room_id=room_id))

    return Building(
        rooms=rooms,
        top_level_walls=[_parse_surface(w) for w in data.get("walls", [])],
        top_level_doors=[_parse_surface(d) for d in data.get("doors", [])],
        top_level_windows=[_parse_surface(w) for w in data.get("windows", [])],
        top_level_floors=[_parse_floor(f) for f in data.get("floors", [])],
        version=data.get("version", 2),
    )


def load_room_scan(path: Path) -> Room:
    """Load a single room JSON from scan-cache."""
    with open(path) as f:
        data = json.load(f)
    room_id = path.stem  # filename without extension
    return _parse_room(data, room_id=room_id)


def load_scan_cache(scan_dir: Path) -> list[Room]:
    """Load all room JSONs from a scan-cache directory."""
    rooms = []
    for fname in sorted(os.listdir(scan_dir)):
        if not fname.endswith(".json"):
            continue
        if fname in _SKIP_FILES:
            continue
        if any(fname.startswith(p) for p in _SKIP_PREFIXES):
            continue
        rooms.append(load_room_scan(scan_dir / fname))
    return rooms


def find_scan_cache_for_building(
    building_uuid: str,
    scan_cache_root: Path,
) -> Path | None:
    """Match a pipeline-outputs UUID to its scan-cache directory."""
    for entry in os.listdir(scan_cache_root):
        if building_uuid in entry:
            return scan_cache_root / entry
    return None


def _rotation_angle_from_transform(t: Transform) -> float:
    """Extract the Y-axis rotation angle (yaw) from a 4x4 transform matrix.

    Returns angle in degrees [0, 360).
    """
    r = t.matrix[:3, :3]
    angle = math.degrees(math.atan2(r[0, 2], r[0, 0]))
    return angle % 360


def count_scan_sessions(rooms: list[Room], angle_tolerance: float = 3.0) -> int:
    """Count distinct scan sessions by clustering referenceOriginTransform.

    Uses rotation angle as primary discriminator, then sub-clusters by
    translation position within each rotation group. Different sessions
    produce different coordinate origins (both rotation and translation).
    """
    transforms = []
    for room in rooms:
        if room.reference_origin_transform is not None:
            angle = _rotation_angle_from_transform(room.reference_origin_transform)
            trans = room.reference_origin_transform.translation
            transforms.append((angle, trans))

    if not transforms:
        return 1

    # First cluster by rotation angle
    sorted_by_angle = sorted(transforms, key=lambda t: t[0])
    angle_groups: list[list[tuple]] = [[sorted_by_angle[0]]]
    for item in sorted_by_angle[1:]:
        diff = min(
            abs(item[0] - angle_groups[-1][0][0]),
            360 - abs(item[0] - angle_groups[-1][0][0]),
        )
        if diff <= angle_tolerance:
            angle_groups[-1].append(item)
        else:
            angle_groups.append([item])

    # Check wraparound
    if len(angle_groups) > 1:
        diff = min(
            abs(angle_groups[0][0][0] - angle_groups[-1][0][0]),
            360 - abs(angle_groups[0][0][0] - angle_groups[-1][0][0]),
        )
        if diff <= angle_tolerance:
            angle_groups[0].extend(angle_groups.pop())

    # Within each rotation group, sub-cluster by translation Y coordinate
    # (floor level). Different sessions on different floors have distinct Y offsets.
    total_clusters = 0
    y_tolerance = 1.0  # 1 meter
    for group in angle_groups:
        y_values = sorted(item[1].y for item in group)
        sub_clusters = 1
        for i in range(1, len(y_values)):
            if abs(y_values[i] - y_values[i - 1]) > y_tolerance:
                sub_clusters += 1
        total_clusters += sub_clusters

    return total_clusters
