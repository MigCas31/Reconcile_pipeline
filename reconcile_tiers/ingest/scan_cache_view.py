"""Export raw scan-cache room geometry for the scan-cache viewer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from reconcile_tiers.ingest.merged import find_merged_path, load_merged
from reconcile_tiers.ingest.scan_cache import (
    find_scan_cache_dir,
    load_raw_ceilings,
    load_raw_rooms,
    parse_address_from_scan_dir,
)

_UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def parse_uuid_from_scan_dir(name: str) -> str | None:
    match = _UUID_RE.search(name)
    return match.group(1).lower() if match else None


def _parse_roomplan_transform(flat: list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(flat, dtype=float)
    if values.size != 16:
        raise ValueError(f"expected 16 transform values, got {values.size}")
    return values.reshape((4, 4), order="F")


def element_world_corners(element: dict[str, Any]) -> list[list[float]]:
    """RoomPlan surface corners in scan world space (Y-up)."""
    transform = element.get("transform")
    if transform is None:
        return []
    matrix = _parse_roomplan_transform(transform)
    polygon = element.get("polygonCorners") or []
    if len(polygon) >= 3:
        local = polygon
    else:
        dims = element.get("dimensions") or [0.0, 0.0, 0.0]
        half_w = float(dims[0]) / 2.0
        half_h = float(dims[1]) / 2.0
        local = [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ]
    out: list[list[float]] = []
    for corner in local:
        if len(corner) < 3:
            continue
        world = matrix @ np.array(
            [float(corner[0]), float(corner[1]), float(corner[2]), 1.0]
        )
        out.append([float(world[0]), float(world[1]), float(world[2])])
    return out


def _ceiling_plane_corners(plane: dict[str, Any]) -> list[list[float]]:
    transform = plane.get("transform")
    corners_local = plane.get("corners_local") or []
    if transform is None or len(corners_local) < 3:
        return []
    matrix = _parse_roomplan_transform(transform)
    out: list[list[float]] = []
    for corner in corners_local:
        if len(corner) < 3:
            continue
        world = matrix @ np.array(
            [float(corner[0]), float(corner[1]), float(corner[2]), 1.0]
        )
        out.append([float(world[0]), float(world[1]), float(world[2])])
    return out


def _surface_payload(
    element: dict[str, Any],
    *,
    kind: str,
    room_file: str,
    index: int,
) -> dict[str, Any] | None:
    corners = element_world_corners(element)
    if len(corners) < 3:
        return None
    return {
        "kind": kind,
        "room_file": room_file,
        "index": index,
        "identifier": str(element.get("identifier") or ""),
        "story": element.get("story"),
        "corners": corners,
    }


def list_scan_buildings(scan_root: Path | str) -> list[dict[str, Any]]:
    root = Path(scan_root)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        uuid = parse_uuid_from_scan_dir(entry.name)
        if uuid is None:
            continue
        rooms = load_raw_rooms(entry)
        rows.append(
            {
                "uuid": uuid,
                "address": parse_address_from_scan_dir(entry) or entry.name,
                "room_count": len(rooms),
                "scan_dir": entry.name,
            }
        )
    rows.sort(key=lambda row: (row.get("address") or "", row["uuid"]))
    return rows


def _roomplan_wall_overlay_from_rooms(
    rooms: list[dict[str, Any]],
    *,
    source: str,
    source_path: str,
    classification: str | None = None,
) -> dict[str, Any]:
    rooms_out: list[dict[str, Any]] = []
    for room_index, room_data in enumerate(rooms):
        walls: list[dict[str, Any]] = []
        for index, wall in enumerate(room_data.get("walls") or []):
            corners = element_world_corners(wall)
            if len(corners) < 3:
                continue
            walls.append(
                {
                    "index": index,
                    "identifier": str(wall.get("identifier") or ""),
                    "story": wall.get("story", room_data.get("story")),
                    "corners": corners,
                }
            )
        rooms_out.append(
            {
                "index": room_index,
                "story": room_data.get("story"),
                "walls": walls,
            }
        )
    wall_count = sum(len(room["walls"]) for room in rooms_out)
    overlay: dict[str, Any] = {
        "source": source,
        "room_count": len(rooms_out),
        "wall_count": wall_count,
        "source_path": source_path,
        "rooms": rooms_out,
    }
    if classification is not None:
        overlay["classification"] = classification
    return overlay


def export_merged_wall_overlay(
    uuid: str,
    pipeline_dir: Path | str,
) -> dict[str, Any] | None:
    """Apple merged.json room walls in merged building space."""
    merged_path = find_merged_path(uuid, pipeline_dir)
    if merged_path is None:
        return None
    merged = load_merged(uuid, pipeline_dir)
    rooms = [room.data for room in merged.rooms]
    return _roomplan_wall_overlay_from_rooms(
        rooms,
        source="merged.json",
        source_path=str(merged_path),
    )


def export_reconciled_wall_overlay(
    uuid: str,
    pipeline_dir: Path | str,
) -> dict[str, Any] | None:
    """Trust-table reconciled.json room walls."""
    path = Path(pipeline_dir) / uuid / "reconciled.json"
    if not path.is_file():
        return None
    with path.open() as handle:
        data = json.load(handle)
    rooms = data.get("rooms")
    if not isinstance(rooms, list):
        return None
    classification = (data.get("reconciliation") or {}).get("classification")
    return _roomplan_wall_overlay_from_rooms(
        rooms,
        source="reconciled.json",
        source_path=str(path),
        classification=str(classification) if classification else None,
    )


def _vec3_corners(raw_corners: list[Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for corner in raw_corners or []:
        if isinstance(corner, dict):
            out.append(
                [float(corner["x"]), float(corner["y"]), float(corner["z"])]
            )
        elif isinstance(corner, (list, tuple)) and len(corner) >= 3:
            out.append([float(corner[0]), float(corner[1]), float(corner[2])])
    return out


def export_tier_payload_wall_overlay(
    uuid: str,
    pipeline_dir: Path | str,
) -> dict[str, Any] | None:
    """Reconstructed tier_payload.json room walls."""
    path = Path(pipeline_dir) / uuid / "tier_payload.json"
    if not path.is_file():
        return None
    with path.open() as handle:
        data = json.load(handle)
    rooms = data.get("rooms")
    if not isinstance(rooms, list):
        return None
    rooms_out: list[dict[str, Any]] = []
    for room_index, room in enumerate(rooms):
        if not isinstance(room, dict):
            continue
        walls: list[dict[str, Any]] = []
        for index, wall in enumerate(room.get("walls") or []):
            if not isinstance(wall, dict):
                continue
            corners = _vec3_corners(wall.get("corners") or [])
            if len(corners) < 3:
                continue
            walls.append(
                {
                    "index": index,
                    "identifier": str(wall.get("locator_id") or ""),
                    "story": room.get("story"),
                    "corners": corners,
                }
            )
        rooms_out.append(
            {
                "index": room_index,
                "story": room.get("story"),
                "walls": walls,
            }
        )
    wall_count = sum(len(room["walls"]) for room in rooms_out)
    classification = (data.get("classification") or {}).get("tier_label")
    return {
        "source": "tier_payload.json",
        "room_count": len(rooms_out),
        "wall_count": wall_count,
        "source_path": str(path),
        "classification": str(classification) if classification else None,
        "tier": (data.get("classification") or {}).get("tier"),
        "rooms": rooms_out,
    }


def export_scan_building(
    uuid: str,
    scan_root: Path | str,
    pipeline_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    scan_dir = find_scan_cache_dir(uuid, scan_root)
    if scan_dir is None:
        return None

    raw_rooms = load_raw_rooms(scan_dir)
    raw_ceilings = load_raw_ceilings(scan_dir)
    rooms_out: list[dict[str, Any]] = []

    for room_file, room_data in raw_rooms:
        surfaces: list[dict[str, Any]] = []
        for index, wall in enumerate(room_data.get("walls") or []):
            payload = _surface_payload(
                wall, kind="wall", room_file=room_file, index=index
            )
            if payload:
                surfaces.append(payload)
        for index, floor in enumerate(room_data.get("floors") or []):
            payload = _surface_payload(
                floor, kind="floor", room_file=room_file, index=index
            )
            if payload:
                surfaces.append(payload)
        for index, door in enumerate(room_data.get("doors") or []):
            payload = _surface_payload(
                door, kind="door", room_file=room_file, index=index
            )
            if payload:
                surfaces.append(payload)
        for index, window in enumerate(room_data.get("windows") or []):
            payload = _surface_payload(
                window, kind="window", room_file=room_file, index=index
            )
            if payload:
                surfaces.append(payload)

        ceiling_entry = raw_ceilings.get(room_file) or {}
        for index, plane in enumerate(ceiling_entry.get("planes") or []):
            corners = _ceiling_plane_corners(plane)
            if len(corners) < 3:
                continue
            surfaces.append(
                {
                    "kind": "ceiling",
                    "room_file": room_file,
                    "index": index,
                    "identifier": "",
                    "story": room_data.get("story"),
                    "corners": corners,
                    "ceiling_source": ceiling_entry.get("source"),
                }
            )

        rooms_out.append(
            {
                "filename": room_file,
                "story": room_data.get("story"),
                "wall_count": len(room_data.get("walls") or []),
                "surfaces": surfaces,
            }
        )

    all_corners = [
        corner
        for room in rooms_out
        for surface in room["surfaces"]
        for corner in surface["corners"]
    ]
    center = {"x": 0.0, "y": 0.0, "z": 0.0}
    if all_corners:
        arr = np.asarray(all_corners, dtype=float)
        mean = arr.mean(axis=0)
        center = {"x": float(mean[0]), "y": float(mean[1]), "z": float(mean[2])}

    pipeline_overlays: dict[str, Any] = {}
    if pipeline_dir is not None:
        merged_overlay = export_merged_wall_overlay(uuid, pipeline_dir)
        if merged_overlay is not None:
            pipeline_overlays["merged"] = merged_overlay
        reconciled_overlay = export_reconciled_wall_overlay(uuid, pipeline_dir)
        if reconciled_overlay is not None:
            pipeline_overlays["reconciled"] = reconciled_overlay
        tier_overlay = export_tier_payload_wall_overlay(uuid, pipeline_dir)
        if tier_overlay is not None:
            pipeline_overlays["tier_payload"] = tier_overlay

    return {
        "schema_version": "1",
        "uuid": uuid,
        "address": parse_address_from_scan_dir(scan_dir) or scan_dir.name,
        "scan_dir": scan_dir.name,
        "room_count": len(rooms_out),
        "building_center": center,
        "rooms": rooms_out,
        "pipeline_overlays": pipeline_overlays,
        # Back-compat for older viewer code paths.
        "merged_overlay": pipeline_overlays.get("merged"),
    }
