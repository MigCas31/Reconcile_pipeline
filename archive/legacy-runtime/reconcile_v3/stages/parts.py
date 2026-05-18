"""parts — split a building into connected footprint components.

A part is a set of rooms whose XZ footprints (optionally per-story)
are connected under SCAN_NOISE_M-level morphological closure. Rooms
that don't connect to any part produce V3UnresolvedRegion(orphan-room).
No ontology hypothesis graph — footprint adjacency only.
"""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile.extract3d.overlaps import decompose_polys, floor_polygon_to_shapely

from ..audit import HypothesisTrace
from ..constants import SCAN_NOISE_M
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Part, V3UnresolvedRegion


def _polygon_to_xz_corners(
    poly: Polygon, floor_y: float = 0.0
) -> list[tuple[float, float, float]]:
    coords = list(poly.exterior.coords)
    return [(float(x), floor_y, float(z)) for x, z in coords]


def identify_building_parts(
    building_uuid: str, rooms: list[V3Room]
) -> tuple[list[V3Part], list[V3UnresolvedRegion]]:
    room_polys: list[tuple[V3Room, Polygon]] = []
    for room in rooms:
        poly = floor_polygon_to_shapely(room.floor_polygon)
        if poly is not None:
            room_polys.append((room, poly))

    if not room_polys:
        return [], []

    # Morphological closure across scan-noise gaps.
    buffered = unary_union(
        [p.buffer(SCAN_NOISE_M, join_style=2) for _, p in room_polys]
    )
    components = decompose_polys(
        make_valid(buffered.buffer(-SCAN_NOISE_M, join_style=2))
    )

    parts: list[V3Part] = []
    unresolved: list[V3UnresolvedRegion] = []
    assigned_rooms: set[int] = set()

    for i, component in enumerate(components):
        if component.area < SCAN_NOISE_M * SCAN_NOISE_M:
            continue
        member_room_ids: list[str] = []
        member_stories: set[int] = set()
        for room, poly in room_polys:
            if component.intersection(poly).area >= SCAN_NOISE_M * SCAN_NOISE_M:
                member_room_ids.append(room.identifier)
                member_stories.add(room.story)
                assigned_rooms.add(room.index)
        if not member_room_ids:
            continue
        trace = HypothesisTrace(
            stage="parts",
            rule="footprint-union-morphological-closure",
            inputs={
                "component_index": i,
                "room_count": len(member_room_ids),
                "closure_m": SCAN_NOISE_M,
                "area_m2": round(float(component.area), 3),
            },
            decision_reason=(
                "Rooms whose floor polygons union under SCAN_NOISE_M morphological "
                "closure are the same building part."
            ),
        )
        parts.append(
            V3Part(
                id=make_element_id(building_uuid, "v3-part", f"part-{i}"),
                room_ids=tuple(member_room_ids),
                footprint_xz=_polygon_to_xz_corners(component),
                stories=tuple(sorted(member_stories)),
                trace=trace,
            )
        )

    for room, poly in room_polys:
        if room.index in assigned_rooms:
            continue
        unresolved.append(
            V3UnresolvedRegion(
                id=make_element_id(
                    building_uuid, "v3-unresolved", f"orphan-room-{room.index}"
                ),
                room_id=room.identifier,
                footprint_xz=_polygon_to_xz_corners(poly),
                y_range=None,
                reason="orphan-room",
                context={"room_index": room.index, "story": room.story},
                trace=HypothesisTrace(
                    stage="parts",
                    rule="orphan-room-not-in-any-component",
                    inputs={"room_id": room.identifier},
                    decision_reason=(
                        "Room footprint did not join any morphologically-closed "
                        "component above SCAN_NOISE_M² area."
                    ),
                ),
            )
        )

    return parts, unresolved
