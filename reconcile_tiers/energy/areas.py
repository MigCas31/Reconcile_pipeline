"""Area helpers for tier-payload surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reconcile_tiers._core.newell import polygon_area_3d


def _coords(corners: Sequence[Any] | None) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for corner in corners or []:
        if isinstance(corner, dict):
            out.append(
                (
                    float(corner.get("x", 0.0)),
                    float(corner.get("y", 0.0)),
                    float(corner.get("z", 0.0)),
                )
            )
        elif hasattr(corner, "x") and hasattr(corner, "y") and hasattr(corner, "z"):
            out.append((float(corner.x), float(corner.y), float(corner.z)))
        else:
            out.append((float(corner[0]), float(corner[1]), float(corner[2])))
    return out


def surface_area(corners: Sequence[Any] | None) -> float:
    """Return 3D polygon area, or 0 for malformed/degenerate polygons."""

    try:
        coords = _coords(corners)
        if len(coords) < 3:
            return 0.0
        return max(0.0, float(polygon_area_3d(coords)))
    except Exception:
        return 0.0


def opening_area(opening: Any) -> float:
    return surface_area(
        getattr(opening, "corners", None)
        if not isinstance(opening, dict)
        else opening.get("corners")
    )


def wall_area_minus_cutouts(wall: Any) -> float:
    corners = wall.get("corners") if isinstance(wall, dict) else wall.corners
    cutouts = wall.get("cutouts", []) if isinstance(wall, dict) else wall.cutouts
    area = surface_area(corners)
    for cutout in cutouts or []:
        area -= opening_area(cutout)
    return max(0.0, area)


def ceiling_piece_area(piece: Any) -> float:
    corners = piece.get("corners") if isinstance(piece, dict) else piece.corners
    holes = piece.get("holes", []) if isinstance(piece, dict) else piece.holes
    area = surface_area(corners)
    for hole in holes or []:
        area -= surface_area(hole)
    return max(0.0, area)


def room_floor_area(room: Any) -> float:
    floor = room.get("floor") if isinstance(room, dict) else room.floor
    pieces = floor if isinstance(floor, list) else [floor]
    return sum(
        surface_area(piece.get("corners") if isinstance(piece, dict) else piece.corners)
        for piece in pieces
        if piece
    )
