"""Step 4 of the assembly: Room container.

Phase B-2 scope: rooms whose ceiling is flat. Sloped and composite
(kinked) ceilings land in later phases. Rooms outside scope return
`None`; their per-room evidence (Floor + Walls + Openings) is still
returned so the residual stream can carry partial primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconcile_tiers.extract.building import ExtractedRoom
from reconcile_tiers.twin.assemble.ceilings import ceiling_for_room
from reconcile_tiers.twin.assemble.floors import floor_for_room
from reconcile_tiers.twin.assemble.openings import openings_by_wall_id
from reconcile_tiers.twin.assemble.walls import (
    wall_planes_for_room,
    walls_for_room,
)
from reconcile_tiers.twin.types import (
    Ceiling,
    Evidence,
    Floor,
    Opening,
    Room,
    Wall,
)


@dataclass(frozen=True, slots=True)
class RoomBuild:
    """Output of `assemble_room`: the (optional) finished `Room`, plus the
    pieces always built (Floor, Walls, Openings) and any orphan evidence.

    `room` is `None` for rooms outside Phase B-2 scope (sloped/kinked
    ceilings, etc.); the partial primitives are still returned for use
    by the residual stream and by Phase B-3 once it lands.
    """

    floor: Floor
    walls: tuple[Wall, ...]
    openings_by_wall_id: dict[str, tuple[Opening, ...]]
    ceiling: Ceiling | None
    room: Room | None
    orphan_evidence: tuple[Evidence, ...]


def assemble_room(room: ExtractedRoom, *, building_uuid: str) -> RoomBuild:
    """Run the per-room assembly pipeline for `room`."""
    floor = floor_for_room(room, building_uuid=building_uuid)

    planes = wall_planes_for_room(room, floor=floor)
    openings_map, opening_orphans = openings_by_wall_id(
        room, wall_planes=planes, building_uuid=building_uuid
    )

    walls, wall_orphans = walls_for_room(
        room,
        floor=floor,
        building_uuid=building_uuid,
        openings_by_wall_id=openings_map,
    )

    ceiling, ceiling_scan_evidence = ceiling_for_room(
        room, floor=floor, walls=walls, building_uuid=building_uuid
    )
    del ceiling_scan_evidence  # already attached to ceiling.evidence

    finished: Room | None = None
    if ceiling is not None and len(walls) >= 3:
        finished = Room(
            id=f"{building_uuid}::room::{room.story}:{room.index}",
            story_index=room.story,
            floor=floor,
            walls=walls,
            ceiling=ceiling,
        )

    return RoomBuild(
        floor=floor,
        walls=walls,
        openings_by_wall_id=openings_map,
        ceiling=ceiling,
        room=finished,
        orphan_evidence=wall_orphans + opening_orphans,
    )
