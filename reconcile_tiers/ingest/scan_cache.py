from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScanMetadata:
    has_basement: bool
    room_heating: dict[str, str] = field(
        default_factory=dict
    )  # {room_key: roomHeatingType}


def load_scan_metadata(scan_dir: Path | str | None) -> ScanMetadata:
    if scan_dir is None:
        return ScanMetadata(has_basement=False)
    data_path = Path(scan_dir) / "data.json"
    if not data_path.exists():
        return ScanMetadata(has_basement=False)
    with data_path.open() as f:
        data = json.load(f)
    floors = data.get("homeMetadata", {}).get("floors", [])
    has_basement = any(
        fl.get("floorType") == "basement" or (fl.get("floorToGradeHeightInCM") or 0) > 0
        for fl in floors
    )
    room_heating: dict[str, str] = {}
    for fl in floors:
        for room in fl.get("rooms", []):
            rid = room.get("id")
            ht = room.get("roomHeatingType")
            if rid and ht:
                room_heating[f"{rid}.json"] = ht
    return ScanMetadata(has_basement=has_basement, room_heating=room_heating)


def find_scan_cache_dir(uuid: str, scan_root: Path | str) -> Path | None:
    root = Path(scan_root)
    if not root.exists():
        return None
    for entry in sorted(root.iterdir()):
        if uuid in entry.name and entry.is_dir():
            return entry
    return None


def parse_address_from_scan_dir(scan_dir: Path | str) -> str | None:
    name = Path(scan_dir).name
    match = re.search(
        r"scans_[^_]+_(.+?)_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_",
        name,
    )
    if not match:
        return None
    return match.group(1).replace("__", ", ").replace("_", " ")


def load_raw_rooms(scan_dir: Path | str | None) -> list[tuple[str, dict[str, Any]]]:
    if scan_dir is None:
        return []
    root = Path(scan_dir)
    rooms: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.iterdir()):
        filename = path.name
        if not filename.endswith(".json"):
            continue
        if filename in {"data.json", "arworldmap.json"}:
            continue
        if filename.startswith("ceiling_") or filename.startswith("merged_"):
            continue
        with path.open() as handle:
            rooms.append((filename, json.load(handle)))
    return rooms


def load_raw_ceilings(scan_dir: Path | str | None) -> dict[str, dict[str, Any]]:
    ceilings: dict[str, dict[str, Any]] = {}
    if scan_dir is None:
        return ceilings
    root = Path(scan_dir)
    if not root.exists():
        return ceilings

    for path in sorted(root.iterdir()):
        filename = path.name
        if not filename.startswith("ceiling_") or not filename.endswith(".json"):
            continue
        if filename.startswith("ceiling_merged_") or filename.startswith(
            "ceiling_metadata_"
        ):
            continue

        room_id = filename[len("ceiling_") : -len(".json")]
        room_key = f"{room_id}.json"
        with path.open() as handle:
            data = json.load(handle)

        source = "scan"
        metadata_path = root / f"ceiling_metadata_{room_id}.json"
        if metadata_path.exists():
            try:
                with metadata_path.open() as metadata_handle:
                    metadata = json.load(metadata_handle)
                ceiling_source = metadata.get("ceilingSource") or {}
                if isinstance(ceiling_source, dict) and ceiling_source:
                    source = str(next(iter(ceiling_source.keys())))
            except (json.JSONDecodeError, OSError):
                pass

        planes: list[dict[str, Any]] = []
        for wall in data.get("walls") or []:
            corners_local = wall.get("polygonCorners") or []
            transform = wall.get("transform")
            if len(corners_local) < 3 or transform is None:
                continue
            planes.append({"corners_local": corners_local, "transform": transform})
        ceilings[room_key] = {"planes": planes, "source": source}

    return ceilings
