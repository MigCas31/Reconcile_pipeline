"""rule_knee_wall_misplaced: knee wall not within ~1 m of top story Y_max."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import (
    KNEE_WALL_TOP_STORY_TOL_M,
    FlagItem,
    _make_item,
    _story_y_bands,
    _y_range,
)


def rule_knee_wall_misplaced(payload: dict[str, Any]) -> list[FlagItem]:
    knee_walls = payload.get("knee_walls") or []
    if not knee_walls:
        return []
    bands = _story_y_bands(payload.get("rooms") or [])
    if not bands:
        return []
    top_y = max(band[1] for band in bands.values())

    items: list[FlagItem] = []
    for wall in knee_walls:
        yr = _y_range(wall.get("corners") or [])
        if yr is None:
            continue
        mid_y = (yr[0] + yr[1]) * 0.5
        delta = abs(mid_y - top_y)
        if delta > KNEE_WALL_TOP_STORY_TOL_M:
            items.append(
                _make_item(
                    wall.get("locator_id"),
                    rule="knee_wall_misplaced",
                    severity="med",
                    evidence={
                        "y_range": [float(yr[0]), float(yr[1])],
                        "top_story_y_max": float(top_y),
                        "delta_m": float(delta),
                    },
                )
            )
    items.sort(key=lambda it: -it["evidence"]["delta_m"])
    return items
