"""Flatten tier_payload into a building-wide element list."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reconcile_tiers.room_postprocessing.models import BuildingElement


def _corners_3d(
    raw: Sequence[Mapping[str, Any] | Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    out: list[tuple[float, float, float]] = []
    for c in raw:
        try:
            if isinstance(c, Mapping):
                out.append((float(c["x"]), float(c["y"]), float(c["z"])))
            else:
                out.append((float(c[0]), float(c[1]), float(c[2])))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return tuple(out)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _element_id(
    *,
    kind: str,
    locator_id: str | None,
    room_index: int | None,
    index: int,
) -> str:
    if locator_id:
        return locator_id
    if room_index is not None:
        return f"{kind}:{room_index}:{index}"
    return f"{kind}:{index}"


def _append_piece(
    elements: list[BuildingElement],
    *,
    kind: str,
    piece: Mapping[str, Any],
    room_index: int | None,
    story: int | None,
    index: int,
) -> None:
    corners = _corners_3d(piece.get("corners") or [])
    if len(corners) < 3:
        return
    locator_id = piece.get("locator_id")
    loc_str = str(locator_id) if locator_id else None
    elements.append(
        BuildingElement(
            id=_element_id(
                kind=kind,
                locator_id=loc_str,
                room_index=room_index,
                index=index,
            ),
            kind=kind,
            locator_id=loc_str,
            corners=corners,
            room_index=room_index,
            story=story,
        )
    )


def flatten_tier_payload(payload: Mapping[str, Any]) -> list[BuildingElement]:
    """Collect floors, walls, shells, and gables (ceilings excluded for corner graph)."""

    elements: list[BuildingElement] = []

    for room_index, room in enumerate(payload.get("rooms") or []):
        if not isinstance(room, Mapping):
            continue
        story = _int_or_none(room.get("story"))
        room_loc = str(room.get("locator_id") or f"room:{room_index}")

        floor_raw = room.get("floor")
        floor_pieces: list[Mapping[str, Any]] = []
        if isinstance(floor_raw, Mapping):
            floor_pieces = [floor_raw]
        elif isinstance(floor_raw, list):
            floor_pieces = [f for f in floor_raw if isinstance(f, Mapping)]

        for floor_index, floor in enumerate(floor_pieces):
            corners = _corners_3d(floor.get("corners") or [])
            if len(corners) < 3:
                continue
            loc = floor.get("locator_id") or f"{room_loc}::floor::{floor_index}"
            loc_str = str(loc)
            elements.append(
                BuildingElement(
                    id=loc_str,
                    kind="floor",
                    locator_id=loc_str,
                    corners=corners,
                    room_index=room_index,
                    story=story,
                )
            )

        for wall_index, wall in enumerate(room.get("walls") or []):
            if not isinstance(wall, Mapping):
                continue
            _append_piece(
                elements,
                kind="wall",
                piece=wall,
                room_index=room_index,
                story=story,
                index=wall_index,
            )

    for shell_index, shell in enumerate(payload.get("visual_shells") or []):
        if isinstance(shell, Mapping):
            _append_piece(
                elements,
                kind="visual_shell",
                piece=shell,
                room_index=None,
                story=None,
                index=shell_index,
            )

    for gable_index, gable in enumerate(payload.get("gable_closures") or []):
        if isinstance(gable, Mapping):
            _append_piece(
                elements,
                kind="gable_closure",
                piece=gable,
                room_index=None,
                story=None,
                index=gable_index,
            )

    for knee_index, knee in enumerate(payload.get("knee_walls") or []):
        if isinstance(knee, Mapping):
            _append_piece(
                elements,
                kind="knee_wall",
                piece=knee,
                room_index=None,
                story=None,
                index=knee_index,
            )

    return elements
