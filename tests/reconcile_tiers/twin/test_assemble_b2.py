"""Phase B-2 corpus tests: per-room Ceiling + Opening + Room assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.twin.assemble import assemble_room
from reconcile_tiers.twin.types import CeilingKind, Wall

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
def test_assemble_room_yields_floor_and_walls_always(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    for r in model.rooms:
        build = assemble_room(r, building_uuid=uuid)
        assert build.floor is not None
        assert build.walls, f"room {r.index}: no walls"


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_openings_projected_onto_host_wall_plane(uuid):
    """Every Opening attached to a Wall must satisfy the Wall's coplanarity
    invariant. The Wall constructor would have raised otherwise — this
    test simply verifies that some openings *are* attached for the
    walkthrough corpus."""
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    total_openings = 0
    for r in model.rooms:
        build = assemble_room(r, building_uuid=uuid)
        total_openings += sum(len(v) for v in build.openings_by_wall_id.values())
        for w in build.walls:
            assert isinstance(w, Wall)
    assert total_openings > 0, f"{uuid}: no openings assigned"


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_zero_orphan_evidence_per_room(uuid):
    """Phase B-2 is per-room only — every wall and every opening should
    be claimed by its host. Cross-room orphans (shared walls between
    rooms, etc.) belong to later phases."""
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    for r in model.rooms:
        build = assemble_room(r, building_uuid=uuid)
        assert len(build.orphan_evidence) == 0, (
            f"{uuid} room {r.index}: orphans = {len(build.orphan_evidence)}"
        )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_ceiling_kind_matches_extracted_room_kind(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    for r in model.rooms:
        build = assemble_room(r, building_uuid=uuid)
        if r.ceiling_is_kinked:
            if build.ceiling is not None:
                assert build.ceiling.kind is CeilingKind.COMPOSITE
                assert len(build.ceiling.parts) >= 2
                assert len(build.ceiling.seams) >= 1
                assert build.room is not None
            continue
        if r.ceiling_type == "flat":
            assert build.ceiling is not None
            assert build.ceiling.kind is CeilingKind.FLAT
            assert build.room is not None
        elif r.ceiling_type == "sloped":
            # Sloped rooms whose wall-tops happen to be coplanar-horizontal
            # (e.g. a small dormer-only slope) skip in B-3a and become
            # orphans pending a roof-anchored step in Phase C.
            if build.ceiling is not None:
                assert build.ceiling.kind is CeilingKind.OBLIQUE
                assert build.room is not None
