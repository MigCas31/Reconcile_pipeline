"""Phase B+ assembly: BuildingModel + RoofModel → Twin.

Each step constructs primitives by anchoring them to already-constructed
neighbours, then assigning scan evidence to them. Steps run in topological
order; later steps depend on the output of earlier ones.

Phase B-1 scope: per-room Floor and Wall only. Ceilings, Openings,
Stories, Wings, Roofs, and Gaps land in later phases.
"""

from __future__ import annotations

from reconcile_tiers.twin.assemble.buildings import assemble_building
from reconcile_tiers.twin.assemble.ceilings import (
    ceiling_for_flat_room,
    ceiling_for_room,
)
from reconcile_tiers.twin.assemble.floors import floor_for_room
from reconcile_tiers.twin.assemble.openings import openings_by_wall_id
from reconcile_tiers.twin.assemble.rooms import RoomBuild, assemble_room
from reconcile_tiers.twin.assemble.stories import stories_from_rooms
from reconcile_tiers.twin.assemble.walls import (
    wall_planes_for_room,
    walls_for_room,
)
from reconcile_tiers.twin.assemble.wings import wings_for_rooms

__all__ = [
    "RoomBuild",
    "assemble_building",
    "assemble_room",
    "ceiling_for_flat_room",
    "ceiling_for_room",
    "floor_for_room",
    "openings_by_wall_id",
    "stories_from_rooms",
    "wall_planes_for_room",
    "walls_for_room",
    "wings_for_rooms",
]
