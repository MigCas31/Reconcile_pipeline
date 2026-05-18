"""rule_silent_drop: polygons the renderer drops (corners<3 or area too small)."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import (
    MIN_POLYGON_AREA_M2,
    FlagItem,
    _make_item,
    _polygon_area_3d,
    _room_floor_pieces,
)


def rule_silent_drop(payload: dict[str, Any]) -> list[FlagItem]:
    items: list[FlagItem] = []
    sources: list[tuple[str, list[dict[str, Any]]]] = [
        ("ceiling", payload.get("ceiling") or []),
        ("knee_wall", payload.get("knee_walls") or []),
        ("gap", payload.get("gaps") or []),
        ("dormer_face", payload.get("dormer_faces") or []),
        ("gable_closure", payload.get("gable_closures") or []),
    ]
    for room in payload.get("rooms") or []:
        for wall in room.get("walls") or []:
            sources.append(("room.wall", [wall]))
        floor_pieces = [
            {
                "locator_id": room.get("locator_id"),
                "corners": piece.get("corners") or [],
            }
            for piece in _room_floor_pieces(room)
        ]
        sources.append(("room.floor", floor_pieces))

    for category, group in sources:
        for item in group:
            corners = item.get("corners") or []
            if len(corners) < 3:
                items.append(
                    _make_item(
                        item.get("locator_id"),
                        rule="silent_drop",
                        severity="low",
                        evidence={
                            "category": category,
                            "reason": f"corners.length={len(corners)}",
                        },
                    )
                )
                continue
            area = _polygon_area_3d(corners)
            if area < MIN_POLYGON_AREA_M2:
                items.append(
                    _make_item(
                        item.get("locator_id"),
                        rule="silent_drop",
                        severity="low",
                        evidence={
                            "category": category,
                            "area_m2": area,
                            "min_area_m2": MIN_POLYGON_AREA_M2,
                        },
                    )
                )
    return items
