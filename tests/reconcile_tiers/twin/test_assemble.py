"""Phase B-1 tests: per-room Floor + Wall assembly against real corpus
buildings. The four walkthrough buildings are used as fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.twin.assemble import floor_for_room, walls_for_room

WALKTHROUGH_UUIDS = [
    "53c380e7-407e-48f7-8c72-b449e2334798",  # clean gable
    "d683f65d-ad08-4f69-ab23-dbd7ec1bcaed",  # kinked-room
    "9bc73438-328e-440e-a4be-25efa61d3b7c",  # split-level
    "7cabc39b-6328-4a6e-9491-822fa6b3c3fb",  # extension
]

PIPELINE = Path("pipeline-outputs")
SCAN_CACHE = Path(".scan-cache")


def _building_available(uuid: str) -> bool:
    return (PIPELINE / uuid / "tier_payload.json").exists()


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_floor_constructs_for_every_room(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    for room in model.rooms:
        floor = floor_for_room(room, building_uuid=uuid)
        assert floor.polygon, f"empty floor polygon for room {room.index}"
        # Floor.__post_init__ already enforces horizontality and area.
        assert floor.evidence, "floor has no evidence"


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_walls_construct_with_no_orphans(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    total_orphans = 0
    for room in model.rooms:
        floor = floor_for_room(room, building_uuid=uuid)
        walls, orphans = walls_for_room(room, floor=floor, building_uuid=uuid)
        total_orphans += len(orphans)
        # Every Wall passed __post_init__ — verticality, horizontal edges,
        # positive height, openings (none yet) coplanar with plane.
        for w in walls:
            assert w.evidence, f"wall {w.id} has no evidence"
    assert total_orphans == 0, (
        f"{uuid}: expected zero orphans across all rooms, got {total_orphans}"
    )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_walls_dedupe_merged_and_computed(uuid):
    """walls_merged and walls_computed are two evidence streams for the
    same physical wall; the twin must yield one Wall per wall id, with
    both pieces of evidence attached."""
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    for room in model.rooms:
        floor = floor_for_room(room, building_uuid=uuid)
        walls, _ = walls_for_room(room, floor=floor, building_uuid=uuid)
        n_distinct_ids = len({w.id for w in walls})
        assert n_distinct_ids == len(walls), (
            f"{uuid} room {room.index}: duplicate wall ids in twin output"
        )
        # Walls present in BOTH walls_merged and walls_computed should
        # carry 2 pieces of evidence.
        merged_ids = {w.id for w in room.walls_merged}
        computed_ids = {w.id for w in room.walls_computed}
        for w in walls:
            wall_uuid = w.id.rsplit("::", 1)[-1]
            n_evidence_expected = (wall_uuid in merged_ids) + (
                wall_uuid in computed_ids
            )
            assert len(w.evidence) == n_evidence_expected, (
                f"{w.id}: expected {n_evidence_expected} evidence, "
                f"got {len(w.evidence)}"
            )
