"""Per-building wall index keyed by ``wall_id`` (Band 2 source-wall features).

Each ``cluster_members[i].source_wall_id`` on a merged-roof proposal points
to a wall in ``buildings_3d.json``. This module builds a one-shot index

    {building_uuid: {wall_id: {room_id, kind, corners_xyz}}}

so the feature expander can resolve each member to its originating wall
without re-scanning the JSON 8k+ times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_wall_index(buildings_3d_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read ``buildings_3d.json`` once; return per-UUID wall lookups."""
    with buildings_3d_path.open() as f:
        buildings = json.load(f)
    if not isinstance(buildings, list):  # defensive
        raise ValueError(
            f"Expected a list of buildings in {buildings_3d_path}, got "
            f"{type(buildings).__name__}"
        )
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for b in buildings:
        uuid = b.get("uuid")
        if not uuid:
            continue
        idx: dict[str, dict[str, Any]] = {}
        for room in b.get("rooms") or []:
            room_id = room.get("id") or room.get("uuid")
            story = room.get("story")
            for kind in ("walls_merged", "walls_computed"):
                for w in room.get(kind) or []:
                    wid = w.get("id")
                    if not wid:
                        continue
                    idx[wid] = {
                        "room_id": room_id,
                        "story": story,
                        "kind": kind,
                        "corners": w.get("corners") or [],
                    }
        out[uuid] = idx
    return out
