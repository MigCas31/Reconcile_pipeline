"""Step 1 of the assembly: Floor per Room.

Anchor: the room's `floor_polygon` from extraction. RoomPlan emits floor
corners at slightly varying Y values; the canonical Floor is at the median
Y of those corners. The original polygon is preserved as ScanEvidence.

No scan-precision tolerance enters this step — `Floor` only requires
horizontality within FLOAT_EPS, and we make that true by construction
(the polygon we emit is at exactly one Y).
"""

from __future__ import annotations

import statistics

from reconcile_tiers.extract.building import ExtractedRoom
from reconcile_tiers.payload.schema import Vec3
from reconcile_tiers.twin.types import Evidence, Floor, Provenance


def floor_for_room(room: ExtractedRoom, *, building_uuid: str) -> Floor:
    """Construct a canonical Floor primitive for `room`.

    Snapping rule: every corner is moved to the median Y of the input
    corners. The original (unsnapped) corners are stored as ScanEvidence.
    """
    if not room.floor_polygon:
        raise ValueError(f"room {room.index}: empty floor_polygon")

    canonical_y = statistics.median(c[1] for c in room.floor_polygon)
    snapped = tuple(
        Vec3(x=float(c[0]), y=float(canonical_y), z=float(c[2]))
        for c in room.floor_polygon
    )
    raw = tuple(
        Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in room.floor_polygon
    )

    evidence = Evidence(
        provenance=Provenance(kind="scan", source="extracted_room.floor_polygon"),
        geometry=raw,
    )

    return Floor(
        id=f"{building_uuid}::floor::{room.story}:{room.index}",
        polygon=snapped,
        evidence=(evidence,),
    )
