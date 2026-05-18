from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MergedRoom:
    index: int
    story: int | None
    data: dict[str, Any]

    @property
    def reference_origin_transform(self) -> list[float]:
        transform = self.data.get("referenceOriginTransform") or []
        return [float(value) for value in transform]


@dataclass(frozen=True, slots=True)
class MergedDoc:
    uuid: str
    path: Path
    data: dict[str, Any]
    rooms: list[MergedRoom]


def find_merged_path(uuid: str, pipeline_dir: Path | str) -> Path | None:
    root = Path(pipeline_dir)
    direct = root / uuid / "merged.json"
    if direct.exists():
        return direct
    if root.name.startswith(uuid) and (root / "merged.json").exists():
        return root / "merged.json"
    if not root.exists():
        return None
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(uuid):
            candidate = entry / "merged.json"
            if candidate.exists():
                return candidate
    return None


def load_merged(uuid: str, pipeline_dir: Path | str) -> MergedDoc:
    path = find_merged_path(uuid, pipeline_dir)
    if path is None:
        raise FileNotFoundError(
            f"merged.json not found for {uuid!r} under {pipeline_dir}"
        )

    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    raw_rooms = data.get("rooms")
    if not isinstance(raw_rooms, list):
        raise ValueError(f"{path} must contain a top-level rooms list")

    rooms: list[MergedRoom] = []
    for index, room in enumerate(raw_rooms):
        if not isinstance(room, dict):
            raise ValueError(f"{path} rooms[{index}] must be a JSON object")
        transform = room.get("referenceOriginTransform")
        if transform is not None and len(transform) != 16:
            raise ValueError(
                f"{path} rooms[{index}].referenceOriginTransform must have 16 values"
            )
        story = room.get("story")
        rooms.append(
            MergedRoom(
                index=index,
                story=int(story) if story is not None else None,
                data=room,
            )
        )

    return MergedDoc(uuid=uuid, path=path, data=data, rooms=rooms)
