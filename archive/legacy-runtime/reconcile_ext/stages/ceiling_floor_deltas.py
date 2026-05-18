"""R1.4 — flag neighbouring rooms with floor or ceiling Y steps.

A single-story rear extension typically sits on a slab-on-grade (floor a
few centimeters below the main-house floor, or sometimes higher if there's
a ground-level concrete step). Its ceiling is often lower than the main
house's first-floor ceiling. Both signals corroborate extension seams.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from ..constants import CEILING_DELTA_STRONG_M, FLOOR_DELTA_STRONG_M
from ..models import ExtHeightDelta, ExtSnapshot, SnapshotRoom


def _room_polygon(room: SnapshotRoom) -> Polygon | None:
    if len(room.floor_polygon) < 3:
        return None
    try:
        poly = Polygon([(c[0], c[2]) for c in room.floor_polygon])
    except Exception:
        return None
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _adjacency(
    snapshot: ExtSnapshot,
) -> list[tuple[SnapshotRoom, SnapshotRoom, tuple[float, float]]]:
    """Pairs of rooms whose floor polygons touch (or are within scan noise),
    plus a representative XZ anchor point of the shared boundary.
    """
    polys: list[tuple[SnapshotRoom, Polygon]] = []
    for room in snapshot.rooms:
        p = _room_polygon(room)
        if p is None:
            continue
        polys.append((room, p))

    out: list[tuple[SnapshotRoom, SnapshotRoom, tuple[float, float]]] = []
    for i in range(len(polys)):
        room_a, poly_a = polys[i]
        buf_a = poly_a.buffer(0.05)
        for j in range(i + 1, len(polys)):
            room_b, poly_b = polys[j]
            # Touching or near-touching counts as adjacent.
            if not buf_a.intersects(poly_b):
                continue
            inter = buf_a.intersection(poly_b)
            if inter.is_empty:
                continue
            # Representative point on the shared boundary.
            rep = inter.representative_point()
            out.append((room_a, room_b, (float(rep.x), float(rep.y))))
    return out


def detect_height_deltas(snapshot: ExtSnapshot) -> list[ExtHeightDelta]:
    results: list[ExtHeightDelta] = []

    for room_a, room_b, anchor in _adjacency(snapshot):
        # Floor delta.
        floor_delta = room_a.floor_y - room_b.floor_y
        if abs(floor_delta) >= FLOOR_DELTA_STRONG_M:
            results.append(
                ExtHeightDelta(
                    id=f"floor-delta::{room_a.id}::{room_b.id}",
                    kind="floor-delta",
                    room_a_id=room_a.id,
                    room_b_id=room_b.id,
                    delta_m=float(floor_delta),
                    y_a=float(room_a.floor_y),
                    y_b=float(room_b.floor_y),
                    anchor_xz=anchor,
                    rationale=(
                        f"Adjacent floors differ by {abs(floor_delta):.2f}m "
                        f"({room_a.floor_y:.2f} vs {room_b.floor_y:.2f})"
                    ),
                )
            )

        # Ceiling delta — only when both have a ceiling Y estimate and
        # they're on the same story (otherwise a 2nd-story / 1st-story
        # neighbour trivially differs).
        if (
            room_a.ceiling_y is not None
            and room_b.ceiling_y is not None
            and room_a.story == room_b.story
        ):
            ceiling_delta = room_a.ceiling_y - room_b.ceiling_y
            if abs(ceiling_delta) >= CEILING_DELTA_STRONG_M:
                results.append(
                    ExtHeightDelta(
                        id=f"ceiling-delta::{room_a.id}::{room_b.id}",
                        kind="ceiling-delta",
                        room_a_id=room_a.id,
                        room_b_id=room_b.id,
                        delta_m=float(ceiling_delta),
                        y_a=float(room_a.ceiling_y),
                        y_b=float(room_b.ceiling_y),
                        anchor_xz=anchor,
                        rationale=(
                            f"Adjacent ceilings (same story) differ by "
                            f"{abs(ceiling_delta):.2f}m "
                            f"({room_a.ceiling_y:.2f} vs {room_b.ceiling_y:.2f})"
                        ),
                    )
                )

    return results
