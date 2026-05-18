"""Scan-cache IO and raw/merged correspondence helpers."""

import json
import os
import re

import numpy as np

from .common import compute_svd, corners_to_world, parse_transform

METHOD_RANK = {"floor-svd": 2, "hybrid": 1, "wall-center-svd": 0}


def find_merged_path(uuid, pipeline_dir):
    for entry in os.listdir(pipeline_dir):
        if entry.startswith(uuid) and os.path.isdir(pipeline_dir / entry):
            merged_path = pipeline_dir / entry / "merged.json"
            if merged_path.exists():
                return merged_path
    return None


def find_scan_cache_dir(uuid, scan_cache_root):
    for entry in os.listdir(scan_cache_root):
        if uuid in entry and os.path.isdir(scan_cache_root / entry):
            return scan_cache_root / entry
    return None


def parse_address_from_scan_dir(scan_dir):
    name = (
        scan_dir.name if hasattr(scan_dir, "name") else os.path.basename(str(scan_dir))
    )
    match = re.search(
        r"scans_[^_]+_(.+?)_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_",
        name,
    )
    if not match:
        return None
    return match.group(1).replace("__", ", ").replace("_", " ")


def load_raw_rooms(scan_dir):
    rooms = []
    for filename in sorted(os.listdir(scan_dir)):
        if not filename.endswith(".json"):
            continue
        if filename in ("data.json", "arworldmap.json"):
            continue
        if filename.startswith("ceiling_") or filename.startswith("merged_"):
            continue
        with open(scan_dir / filename) as handle:
            rooms.append((filename, json.load(handle)))
    return rooms


def load_raw_ceilings(scan_dir):
    """Load per-room raw ceiling polygons captured by Apple RoomPlan.

    Reads `ceiling_<room-id>.json` (per-session, in raw scan world space — not
    the pre-merged variant) plus the sibling `ceiling_metadata_<room-id>.json`.
    Each ceiling file can contain multiple "wall" entries, one per planar
    surface (flat + sloped segments for vaulted/pitched ceilings). All planes
    are returned; the caller is expected to apply the same per-room SVD
    `(rot, trans)` used for walls to remap into merged-building space.

    Keyed by the raw-room filename `<room-id>.json` so it lines up with the
    entries returned by `load_raw_rooms`.
    """
    ceilings = {}
    if scan_dir is None or not os.path.isdir(scan_dir):
        return ceilings
    prefix = "ceiling_"
    skip = ("ceiling_merged_", "ceiling_metadata_")
    for filename in sorted(os.listdir(scan_dir)):
        if not filename.startswith(prefix) or not filename.endswith(".json"):
            continue
        if any(filename.startswith(s) for s in skip):
            continue
        room_id = filename[len(prefix) : -len(".json")]
        room_key = f"{room_id}.json"
        with open(scan_dir / filename) as handle:
            data = json.load(handle)
        source = "scan"
        meta_path = scan_dir / f"ceiling_metadata_{room_id}.json"
        if meta_path.exists():
            try:
                with open(meta_path) as meta_handle:
                    meta = json.load(meta_handle)
                src_obj = meta.get("ceilingSource") or {}
                if isinstance(src_obj, dict) and src_obj:
                    source = next(iter(src_obj.keys()))
            except (json.JSONDecodeError, OSError):
                pass
        planes = []
        for wall in data.get("walls") or []:
            corners_local = wall.get("polygonCorners") or []
            transform = wall.get("transform")
            if len(corners_local) < 3 or transform is None:
                continue
            planes.append({"corners_local": corners_local, "transform": transform})
        ceilings[room_key] = {"planes": planes, "source": source}
    return ceilings


def build_raw_to_merged_index(raw_rooms, merged_data):
    """Map each raw room filename to its best-matching merged room index.

    Uses wall-identifier overlap — the same criterion as
    `compute_room_transforms` — so the result is consistent with the transform
    method selected for that room.
    """
    mapping = {}
    merged_rooms = merged_data.get("rooms", [])
    for room_name, room_data in raw_rooms:
        raw_wall_ids = {wall["identifier"] for wall in room_data.get("walls", [])}
        if not raw_wall_ids:
            continue
        best_idx = -1
        best_overlap = 0
        for idx, merged_room in enumerate(merged_rooms):
            merged_ids = {wall["identifier"] for wall in merged_room.get("walls", [])}
            overlap = len(raw_wall_ids & merged_ids)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_idx >= 0:
            mapping[room_name] = best_idx
    return mapping


def compute_room_transforms(raw_rooms, merged_data):
    """Compute raw-room -> merged-building transforms using SVD."""
    transforms = {}
    merged_wall_map = {}
    for merged_room in merged_data.get("rooms", []):
        for wall in merged_room.get("walls", []):
            merged_wall_map[wall["identifier"]] = wall

    for room_name, room_data in raw_rooms:
        raw_wall_ids = {wall["identifier"] for wall in room_data.get("walls", [])}
        if not raw_wall_ids:
            continue

        best_idx = -1
        best_overlap = 0
        for idx, merged_room in enumerate(merged_data["rooms"]):
            merged_ids = {wall["identifier"] for wall in merged_room.get("walls", [])}
            overlap = len(raw_wall_ids & merged_ids)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap

        if best_idx < 0:
            continue

        merged_room = merged_data["rooms"][best_idx]
        raw_floor = room_data.get("floors")
        merged_floor = merged_room.get("floors")

        if (
            raw_floor
            and raw_floor[0].get("polygonCorners")
            and merged_floor
            and merged_floor[0].get("polygonCorners")
        ):
            raw_fc = np.array(
                corners_to_world(
                    raw_floor[0]["polygonCorners"], raw_floor[0]["transform"]
                )
            )
            merged_fc = np.array(
                corners_to_world(
                    merged_floor[0]["polygonCorners"], merged_floor[0]["transform"]
                )
            )
            if len(raw_fc) == len(merged_fc):
                rot, trans, residual = compute_svd(raw_fc, merged_fc)
                if residual < 50.0:
                    transforms[room_name] = (rot, trans, residual, "floor-svd")
                    continue

        src_points = []
        dst_points = []
        for raw_wall in room_data.get("walls", []):
            merged_wall = merged_wall_map.get(raw_wall["identifier"])
            if merged_wall is None:
                continue
            src_points.append(parse_transform(raw_wall["transform"])[:3, 3])
            dst_points.append(parse_transform(merged_wall["transform"])[:3, 3])

        if len(src_points) >= 3:
            src_arr = np.array(src_points)
            dst_arr = np.array(dst_points)
            rot, trans, residual = compute_svd(src_arr, dst_arr)
            if residual < 200.0:
                transforms[room_name] = (rot, trans, residual, "wall-center-svd")

    return transforms


def build_raw_indices(raw_rooms, raw_transforms):
    """Index raw geometry by identifier, preferring better transform methods."""
    raw_wall_data = {}
    raw_door_data = {}
    raw_window_data = {}
    raw_opening_data = {}
    raw_storage_data = {}

    for room_name, room_data in raw_rooms:
        if room_name not in raw_transforms:
            continue
        rot, trans, _residual, method = raw_transforms[room_name]
        rank = METHOD_RANK.get(method, 0)

        def should_replace(existing, _rank=rank):
            return existing is None or _rank > METHOD_RANK.get(existing[3], 0)

        for wall in room_data.get("walls", []):
            current = raw_wall_data.get(wall["identifier"])
            if should_replace(current):
                raw_wall_data[wall["identifier"]] = (wall, rot, trans, method)

        for door in room_data.get("doors", []):
            current = raw_door_data.get(door["identifier"])
            if should_replace(current):
                raw_door_data[door["identifier"]] = (door, rot, trans, method)

        for window in room_data.get("windows", []):
            current = raw_window_data.get(window["identifier"])
            if should_replace(current):
                raw_window_data[window["identifier"]] = (window, rot, trans, method)

        for opening in room_data.get("openings", []):
            current = raw_opening_data.get(opening["identifier"])
            if should_replace(current):
                raw_opening_data[opening["identifier"]] = (opening, rot, trans, method)

        for obj in room_data.get("objects", []):
            category = obj.get("category", {})
            if not (isinstance(category, dict) and "storage" in category):
                continue
            current = raw_storage_data.get(obj["identifier"])
            if should_replace(current):
                raw_storage_data[obj["identifier"]] = (obj, rot, trans, method)

    return {
        "walls": raw_wall_data,
        "doors": raw_door_data,
        "windows": raw_window_data,
        "openings": raw_opening_data,
        "storages": raw_storage_data,
    }


def build_merged_id_sets(merged):
    ids = {
        "doors": set(),
        "windows": set(),
        "openings": set(),
        "storages": set(),
    }
    for room in merged["rooms"]:
        ids["doors"].update(door["identifier"] for door in room.get("doors", []))
        ids["windows"].update(
            window["identifier"] for window in room.get("windows", [])
        )
        ids["openings"].update(
            opening["identifier"] for opening in room.get("openings", [])
        )
        for obj in room.get("objects", []):
            category = obj.get("category", {})
            if isinstance(category, dict) and "storage" in category:
                ids["storages"].add(obj["identifier"])
    return ids
