"""gaps — first-class gap detection & closure.

Uses ``reconcile.extract3d.gaps.compute_cross_floor_gaps`` as the detector.
For each gap:

- 0 or 1 neighbouring rooms → ``closed``: emit a ``V3Gap`` plus a child
  ``V3Slab`` at the gap floor Y and a child ``V3FlatCeiling`` at the
  ceiling Y derived from the neighbour's wall tops. Never mutate any
  ``V3Room.floor_polygon`` (that is the anti-pattern this design replaces).
- ≥ 2 neighbouring rooms → ``ambiguous``: emit ``V3UnresolvedRegion``.
"""

from __future__ import annotations

import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.validation import make_valid

from reconcile.extract3d.gaps import compute_cross_floor_gaps
from reconcile.extract3d.overlaps import floor_polygon_to_shapely

from ..audit import HypothesisTrace
from ..constants import SCAN_NOISE_M
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3FlatCeiling, V3Gap, V3Slab, V3UnresolvedRegion


def _wall_top_y(wall: dict) -> float | None:
    corners = wall.get("corners") or []
    if len(corners) < 3:
        return None
    return float(max(c[1] for c in corners))


def _room_ceiling_y(room: V3Room) -> float | None:
    tops = [y for y in (_wall_top_y(w) for w in room.walls) if y is not None]
    if not tops:
        return None
    return float(np.median(tops))


def _gap_polygon(gap: dict) -> Polygon | None:
    corners = gap.get("corners") or []
    if len(corners) < 4:
        return None
    coords_2d = [(c[0], c[2]) for c in corners]
    if coords_2d[0] != coords_2d[-1]:
        coords_2d.append(coords_2d[0])
    poly = Polygon(coords_2d)
    if not poly.is_valid:
        poly = make_valid(poly)
    return (
        poly
        if isinstance(poly, Polygon) and poly.area > SCAN_NOISE_M * SCAN_NOISE_M
        else None
    )


def _count_neighbours(
    gap_poly: Polygon, rooms_with_polys: list[tuple[V3Room, Polygon]], gap_story: int
) -> list[V3Room]:
    neighbours: list[V3Room] = []
    for room, room_poly in rooms_with_polys:
        if room.story != gap_story:
            continue
        shared = gap_poly.boundary.intersection(room_poly.boundary)
        if shared is None or shared.is_empty:
            continue
        if getattr(shared, "length", 0.0) >= SCAN_NOISE_M:
            neighbours.append(room)
    return neighbours


def close_obvious_gaps(
    building_uuid: str, rooms: list[V3Room]
) -> tuple[list[V3Gap], list[V3Slab], list[V3FlatCeiling], list[V3UnresolvedRegion]]:
    rooms_as_dict = [
        {
            "story": r.story,
            "floor_polygon": [list(c) for c in r.floor_polygon],
            "index": r.index,
        }
        for r in rooms
    ]
    gaps_out: list[V3Gap] = []
    slabs_out: list[V3Slab] = []
    ceilings_out: list[V3FlatCeiling] = []
    unresolved: list[V3UnresolvedRegion] = []

    try:
        detected = compute_cross_floor_gaps(rooms_as_dict)
    except GEOSException as exc:
        unresolved.append(
            V3UnresolvedRegion(
                id=make_element_id(building_uuid, "v3-unresolved", "gap-detect-failed"),
                room_id=None,
                footprint_xz=[],
                y_range=None,
                reason="gap-detection-failed",
                context={"error": str(exc)[:200]},
                trace=HypothesisTrace(
                    stage="gaps",
                    rule="gap-detection-geos-failure",
                    inputs={"room_count": len(rooms_as_dict)},
                    decision_reason=(
                        "compute_cross_floor_gaps raised a GEOS overlay assertion; "
                        "cannot detect gaps for this building."
                    ),
                ),
            )
        )
        return gaps_out, slabs_out, ceilings_out, unresolved

    rooms_with_polys: list[tuple[V3Room, Polygon]] = []
    for r in rooms:
        poly = floor_polygon_to_shapely(r.floor_polygon)
        if poly is not None:
            rooms_with_polys.append((r, poly))

    for idx, gap in enumerate(detected):
        gap_poly = _gap_polygon(gap)
        if gap_poly is None:
            continue
        floor_y = float(gap["corners"][0][1])
        story = int(gap["story"])
        neighbours = _count_neighbours(gap_poly, rooms_with_polys, story)

        inner = f"{gap.get('type', 'gap')}-{story}-{idx}"
        polygon_xz = [(float(c[0]), floor_y, float(c[2])) for c in gap["corners"]]

        if len(neighbours) >= 2:
            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(building_uuid, "v3-unresolved", f"gap-{inner}"),
                    room_id=None,
                    footprint_xz=polygon_xz,
                    y_range=(floor_y, floor_y),
                    reason="gap-between-rooms",
                    context={
                        "neighbor_room_ids": [n.identifier for n in neighbours],
                        "gap_type": gap.get("type"),
                        "story": story,
                        "area_m2": gap.get("area_m2"),
                    },
                    trace=HypothesisTrace(
                        stage="gaps",
                        rule="multi-neighbour-gap-ambiguous",
                        inputs={
                            "neighbor_count": len(neighbours),
                            "story": story,
                            "area_m2": gap.get("area_m2"),
                        },
                        decision_reason=(
                            "Gap borders 2+ rooms; assignment is ambiguous. "
                            "Flag for visual review rather than guess."
                        ),
                    ),
                )
            )
            continue

        ceiling_y = None
        if neighbours:
            ceiling_y = _room_ceiling_y(neighbours[0])
        if ceiling_y is None:
            fallback_ys = [
                y
                for y in (
                    _room_ceiling_y(r) for r, _ in rooms_with_polys if r.story == story
                )
                if y is not None
            ]
            if fallback_ys:
                ceiling_y = float(np.median(fallback_ys))

        if ceiling_y is None:
            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(building_uuid, "v3-unresolved", f"gap-{inner}"),
                    room_id=None,
                    footprint_xz=polygon_xz,
                    y_range=(floor_y, floor_y),
                    reason="gap-no-ceiling-evidence",
                    context={"neighbor_count": len(neighbours), "story": story},
                    trace=HypothesisTrace(
                        stage="gaps",
                        rule="gap-needs-ceiling-evidence",
                        inputs={"neighbor_count": len(neighbours), "story": story},
                        decision_reason=(
                            "Gap has 0-1 neighbours but no room on this story "
                            "had derivable wall-top Y — cannot cap the gap."
                        ),
                    ),
                )
            )
            continue

        neighbour_ids: tuple[str | None, str | None] = (
            (neighbours[0].identifier, None) if neighbours else (None, None)
        )
        gap_trace = HypothesisTrace(
            stage="gaps",
            rule="closed-gap-0-or-1-neighbour",
            inputs={
                "neighbor_count": len(neighbours),
                "story": story,
                "floor_y": floor_y,
                "ceiling_y": ceiling_y,
                "area_m2": gap.get("area_m2"),
            },
            decision_reason=(
                "Gap borders ≤1 room (interior hole or edge-of-building) — "
                "cap with floor + flat ceiling derived from neighbour wall tops."
            ),
        )
        gap_id = make_element_id(building_uuid, "v3-gap", inner)
        gaps_out.append(
            V3Gap(
                id=gap_id,
                footprint_xz=polygon_xz,
                floor_y=floor_y,
                ceiling_y=ceiling_y,
                between_rooms=neighbour_ids,
                status="closed",
                trace=gap_trace,
            )
        )
        slabs_out.append(
            V3Slab(
                id=make_element_id(building_uuid, "v3-slab", f"gap-{inner}"),
                room_id=None,
                polygon=polygon_xz,
                trace=HypothesisTrace(
                    stage="gaps",
                    rule="closed-gap-floor",
                    inputs={"gap_id": gap_id, "floor_y": floor_y},
                    decision_reason="Floor at gap Y (matches story floor).",
                ),
            )
        )
        ceilings_out.append(
            V3FlatCeiling(
                id=make_element_id(building_uuid, "v3-flat-ceiling", f"gap-{inner}"),
                room_id=None,
                footprint_xz=[
                    (float(c[0]), ceiling_y, float(c[2])) for c in gap["corners"]
                ],
                y=ceiling_y,
                over="gap",
                trace=HypothesisTrace(
                    stage="gaps",
                    rule="closed-gap-ceiling",
                    inputs={"gap_id": gap_id, "ceiling_y": ceiling_y},
                    decision_reason=(
                        "Ceiling at neighbour-room wall-top median (story "
                        "ceiling median if no direct neighbour)."
                    ),
                ),
            )
        )

    return gaps_out, slabs_out, ceilings_out, unresolved
