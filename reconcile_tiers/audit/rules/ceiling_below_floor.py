"""rule_ceiling_below_floor: ceiling sits below the room's floor it covers.

Story-aware: for each ceiling piece, find the room whose floor xz polygon
contains the centroid AND whose story matches the ceiling's Y-range. The
older XZ-only heuristic was matching top-story ceilings against ground-floor
rooms (sharing footprint area) and flagging them as below-floor when they
were correctly placed at the top story. The diagnostic showed 99.9% of those
flags were that XZ-only mismatch.
"""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon

from reconcile_tiers.audit.rules._shared import (
    CEILING_BELOW_FLOOR_Y_SLACK_M,
    FlagItem,
    _corners_xz,
    _make_item,
    _room_floor_pieces,
    _safe_polygon,
    _story_y_bands,
    _y_range,
)


def rule_ceiling_below_floor(payload: dict[str, Any]) -> list[FlagItem]:
    items: list[FlagItem] = []
    rooms = payload.get("rooms") or []

    bands = _story_y_bands(rooms)

    floor_records: list[tuple[Polygon, float, dict[str, Any]]] = []
    for room in rooms:
        for piece in _room_floor_pieces(room):
            corners = piece.get("corners") or []
            poly = _safe_polygon(_corners_xz(corners))
            if poly is None:
                continue
            yr = _y_range(corners)
            if yr is None:
                continue
            floor_records.append((poly, yr[1], room))

    if not floor_records:
        return items

    slack = CEILING_BELOW_FLOOR_Y_SLACK_M
    for piece in payload.get("ceiling") or []:
        corners = piece.get("corners") or []
        cyr = _y_range(corners)
        if cyr is None:
            continue
        cpoly = _safe_polygon(_corners_xz(corners))
        if cpoly is None:
            continue
        rep = cpoly.representative_point()
        ceiling_mid_y = 0.5 * (cyr[0] + cyr[1])

        # Prefer the floor whose room Y-band actually contains the ceiling
        # mid-Y (story-aware). If no story matches, fall back to the first
        # XZ-containing room — that path keeps catching real inversions where
        # a piece is below every story's floor.
        story_match: tuple[float, float, dict[str, Any]] | None = None
        xz_fallback: tuple[float, float, dict[str, Any]] | None = None
        for floor_poly, floor_y_max, room in floor_records:
            if not floor_poly.contains(rep):
                continue
            if xz_fallback is None:
                xz_fallback = (floor_y_max, room.get("story") or 0, room)
            room_story = room.get("story")
            if room_story is None or room_story not in bands:
                continue
            band_lo, band_hi = bands[int(room_story)]
            if band_lo - 0.5 <= ceiling_mid_y <= band_hi + 0.5:
                story_match = (floor_y_max, int(room_story), room)
                break
        best = story_match or xz_fallback
        if best is None:
            continue
        floor_y_max, _story, room = best
        if cyr[1] < floor_y_max - slack:
            items.append(
                _make_item(
                    piece.get("locator_id"),
                    rule="ceiling_below_floor",
                    severity="high",
                    evidence={
                        "ceiling_y_range": [float(cyr[0]), float(cyr[1])],
                        "floor_y_max": float(floor_y_max),
                        "delta_below_m": float(floor_y_max - cyr[1]),
                        "room_locator_id": room.get("locator_id"),
                    },
                )
            )
    return items
