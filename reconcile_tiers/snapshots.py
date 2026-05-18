"""Light-weight before/after diffing of two tier_payload.json snapshots.

Used by the Gemini ``compare_before_after`` tool to surface the visible
deltas after a parameter-tweak rebuild: piece counts per kind, total area,
and locator-set differences.

Snapshots are taken simply by copying ``pipeline-outputs/<uuid>/tier_payload.json``
into ``.context/snapshots/<uuid>/<timestamp>.json`` before a rebuild. The
``compare_before_after`` tool diffs two of these JSON files.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_ROOT = WORKSPACE_ROOT / ".context" / "snapshots"
PIPELINE_OUTPUTS = WORKSPACE_ROOT / "pipeline-outputs"


def _ts() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def take(uuid: str) -> Path:
    """Copy the current tier_payload.json into the snapshot store."""
    src = PIPELINE_OUTPUTS / uuid / "tier_payload.json"
    if not src.exists():
        raise FileNotFoundError(f"tier_payload.json missing for {uuid}")
    dst_dir = SNAPSHOT_ROOT / uuid
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{_ts()}.json"
    shutil.copy2(src, dst)
    return dst


def list_snapshots(uuid: str) -> list[Path]:
    d = SNAPSHOT_ROOT / uuid
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    def count(key: str) -> int:
        return len(payload.get(key) or [])

    return {
        "rooms": count("rooms"),
        "ceiling": count("ceiling"),
        "knee_walls": count("knee_walls"),
        "dormer_faces": count("dormer_faces"),
        "gable_closures": count("gable_closures"),
        "gaps": count("gaps"),
        "visual_shells": count("visual_shells"),
        "locators": _count_locators(payload),
    }


def _count_locators(payload: dict[str, Any]) -> int:
    n = 0
    for key in ("ceiling", "knee_walls", "dormer_faces", "gable_closures", "gaps"):
        n += sum(1 for e in payload.get(key) or [] if e.get("locator_id"))
    for room in payload.get("rooms") or []:
        if room.get("locator_id"):
            n += 1
        for wall in room.get("walls") or []:
            if wall.get("locator_id"):
                n += 1
    return n


def _gather_locators(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("ceiling", "knee_walls", "dormer_faces", "gable_closures", "gaps"):
        for entry in payload.get(key) or []:
            loc = entry.get("locator_id")
            if isinstance(loc, str):
                out.add(loc)
    for room in payload.get("rooms") or []:
        loc = room.get("locator_id")
        if isinstance(loc, str):
            out.add(loc)
        for wall in room.get("walls") or []:
            wloc = wall.get("locator_id")
            if isinstance(wloc, str):
                out.add(wloc)
    return out


def diff(before_path: Path | str, after_path: Path | str) -> dict[str, Any]:
    """Return a small JSON-friendly summary of differences between two payloads."""
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    bsum = _summarise(before)
    asum = _summarise(after)
    deltas = {key: asum[key] - bsum[key] for key in bsum}

    before_locs = _gather_locators(before)
    after_locs = _gather_locators(after)
    added = sorted(after_locs - before_locs)
    removed = sorted(before_locs - after_locs)

    return {
        "before": str(before_path),
        "after": str(after_path),
        "summary_before": bsum,
        "summary_after": asum,
        "deltas": deltas,
        "added_locator_count": len(added),
        "removed_locator_count": len(removed),
        "added_locators_sample": added[:20],
        "removed_locators_sample": removed[:20],
    }


__all__ = ["diff", "list_snapshots", "take"]
