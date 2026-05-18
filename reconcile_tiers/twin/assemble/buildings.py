"""Step 8 (orchestrator): Building-level twin assembly.

Walks every `ExtractedRoom`, runs the per-room pipeline, then groups
into Stories, wraps in a single Wing, and constructs the Building.
Returns a `Twin` (Building + Residual). Rooms outside per-room scope
contribute Floors and Walls but no Room primitive; their Floors are
not yet attached to a Story (Phase D will attach orphan Floors as
their own residual entries).

Phase C-1+C-2 scope: structural primitives only — no Roof yet (that's
Phase C-3). The Building can construct because Wing accepts a Roof of
None and our Phase A invariants don't require one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.twin.assemble.gaps import gaps_from_holes
from reconcile_tiers.twin.assemble.roofs import roof_for_wing
from reconcile_tiers.twin.assemble.rooms import RoomBuild, assemble_room
from reconcile_tiers.twin.assemble.wings import wings_for_rooms
from reconcile_tiers.twin.types import (
    Building,
    Evidence,
    InvariantViolation,
    Residual,
    Twin,
    Wing,
)


def assemble_building(
    model: BuildingModel,
    *,
    pipeline_dir: Path | None = None,
) -> tuple[Twin | None, list[RoomBuild]]:
    """Run the full per-building twin assembly pipeline.

    Returns `(twin, room_builds)`. `twin` is None when the building
    yields no constructable Wing (e.g. all rooms outside scope, or zero
    floor polygons). `room_builds` always carries the per-room outputs
    so callers / diagnostics can introspect partial primitives.
    """
    builds: list[RoomBuild] = [
        assemble_room(r, building_uuid=model.uuid) for r in model.rooms
    ]

    finished_rooms = tuple(b.room for b in builds if b.room is not None)
    wings_no_roof, hole_polygons = wings_for_rooms(
        finished_rooms, building_uuid=model.uuid
    )

    classification_roof_type = _classification_roof_type(model.uuid, pipeline_dir)

    wings_with_roof: list[Wing] = []
    for wing in wings_no_roof:
        roof = roof_for_wing(
            wing,
            classification_roof_type=classification_roof_type,
            building_uuid=model.uuid,
        )
        wings_with_roof.append(replace(wing, roof=roof) if roof is not None else wing)
    wings = tuple(wings_with_roof)

    orphan_evidence: tuple[Evidence, ...] = tuple(
        ev for b in builds for ev in b.orphan_evidence
    )
    fully_inferred_ids: tuple[str, ...] = ()
    gaps = gaps_from_holes(hole_polygons, building_uuid=model.uuid, wings=wings)

    if not wings:
        return None, builds

    try:
        building = Building(
            id=f"{model.uuid}::building",
            uuid=model.uuid,
            wings=wings,
        )
    except InvariantViolation:
        return None, builds

    twin = Twin(
        building=building,
        residual=Residual(
            orphan_evidence=orphan_evidence,
            unclaimed_gaps=gaps,
            fully_inferred_primitive_ids=fully_inferred_ids,
        ),
    )
    return twin, builds


def _classification_roof_type(uuid: str, pipeline_dir: Path | None) -> str | None:
    if pipeline_dir is None:
        pipeline_dir = Path("pipeline-outputs")
    payload_path = pipeline_dir / uuid / "tier_payload.json"
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text())
    except Exception:
        return None
    classification = payload.get("classification") or {}
    rt = classification.get("roof_type")
    return str(rt) if rt is not None else None
