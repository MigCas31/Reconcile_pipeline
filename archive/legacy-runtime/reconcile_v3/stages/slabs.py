"""slabs — one floor slab per room, at the floor polygon's Y."""

from __future__ import annotations

import numpy as np

from ..audit import HypothesisTrace
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Slab


def build_room_slabs(building_uuid: str, rooms: list[V3Room]) -> list[V3Slab]:
    slabs: list[V3Slab] = []
    for room in rooms:
        if len(room.floor_polygon) < 3:
            continue
        ys = [c[1] for c in room.floor_polygon]
        floor_y = float(np.median(ys))
        inner = f"room-{room.index}"
        eid = make_element_id(building_uuid, "v3-slab", inner)
        trace = HypothesisTrace(
            stage="slabs",
            rule="floor-polygon-as-slab",
            inputs={"room_id": room.identifier, "floor_y": floor_y},
            decision_reason=(
                "Floor polygon from step 1 scan data is trusted; slab Y is "
                "the median of polygon corner Ys."
            ),
        )
        slabs.append(
            V3Slab(
                id=eid,
                room_id=room.identifier,
                polygon=[
                    (float(c[0]), floor_y, float(c[2])) for c in room.floor_polygon
                ],
                trace=trace,
            )
        )
    return slabs
