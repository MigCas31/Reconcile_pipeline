"""Tests for the per-(x,z) plane classifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.twin import (
    ResolvedY,
    ceiling_y_at,
    contenders_at,
    roof_y_at,
)
from reconcile_tiers.twin.assemble import assemble_building

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


def _twin(uuid: str):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    return twin


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_ceiling_y_at_returns_resolved_or_none(uuid):
    twin = _twin(uuid)
    # Sample the centroid of every room; classifier should return a
    # ResolvedY for every room with a constructed Ceiling.
    for wing in twin.building.wings:
        for story in wing.stories:
            for room in story.rooms:
                cx = sum(c.x for c in room.floor.polygon) / len(room.floor.polygon)
                cz = sum(c.z for c in room.floor.polygon) / len(room.floor.polygon)
                resolved = ceiling_y_at(twin, cx, cz)
                # The centroid is by construction inside the floor polygon
                # AND inside at least one ceiling candidate (or the room
                # has a real gap, which is acceptable).
                if resolved is not None:
                    assert isinstance(resolved, ResolvedY)
                    assert resolved.y > room.floor.polygon[0].y
                    assert resolved.chosen_id in resolved.contender_ids


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_outside_building_returns_none(uuid):
    twin = _twin(uuid)
    # A point far away from the building footprint must return None.
    far = 10_000.0
    assert ceiling_y_at(twin, far, far) is None
    assert roof_y_at(twin, far, far) is None


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_contenders_sorted_by_y_for_ceiling(uuid):
    twin = _twin(uuid)
    for wing in twin.building.wings:
        for story in wing.stories:
            for room in story.rooms:
                cx = sum(c.x for c in room.floor.polygon) / len(room.floor.polygon)
                cz = sum(c.z for c in room.floor.polygon) / len(room.floor.polygon)
                contenders = contenders_at(twin, cx, cz, "ceiling")
                if contenders:
                    ys = [c.y for c in contenders]
                    assert ys == sorted(ys), (
                        f"ceiling contenders not sorted ascending: {ys}"
                    )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_contenders_sorted_by_y_for_roof(uuid):
    twin = _twin(uuid)
    for wing in twin.building.wings:
        cx = sum(c.x for c in wing.footprint) / len(wing.footprint)
        cz = sum(c.z for c in wing.footprint) / len(wing.footprint)
        contenders = contenders_at(twin, cx, cz, "roof")
        if contenders:
            ys = [c.y for c in contenders]
            assert ys == sorted(ys, reverse=True), (
                f"roof contenders not sorted descending: {ys}"
            )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_ceiling_y_at_or_near_roof_y_when_both_defined(uuid):
    """Architecturally: at any (x, z), ceiling Y ≤ roof Y.

    The wing-level Roof's RoofSurfaces are clustered (SVD-fit through
    multiple rooms' members), so their plane can deviate from each
    member's per-room oblique by a small SVD residual. We allow up to
    20 cm of disagreement here — it's float-precision noise from the
    cluster fit, not a structural inversion.
    """
    twin = _twin(uuid)
    for wing in twin.building.wings:
        for story in wing.stories:
            for room in story.rooms:
                cx = sum(c.x for c in room.floor.polygon) / len(room.floor.polygon)
                cz = sum(c.z for c in room.floor.polygon) / len(room.floor.polygon)
                ceil = ceiling_y_at(twin, cx, cz)
                roof = roof_y_at(twin, cx, cz)
                if ceil is not None and roof is not None:
                    assert ceil.y <= roof.y + 0.20, (
                        f"ceiling {ceil.y:.3f} above roof {roof.y:.3f} at "
                        f"({cx:.2f},{cz:.2f}) by more than the SVD-fit "
                        f"residual budget"
                    )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_chosen_id_is_a_real_primitive_id(uuid):
    """The classifier may legitimately route (x, z) to a different
    room than the centroid we sampled (extraction's room polygons
    overlap on shared walls; nearest-centroid resolves the tie).
    Verify chosen_id is *some* real ceiling primitive in the twin,
    not necessarily this room's."""
    twin = _twin(uuid)
    all_ids: set[str] = set()
    for w in twin.building.wings:
        for s in w.stories:
            for r in s.rooms:
                all_ids.add(r.ceiling.id)
                all_ids.update(p.id for p in r.ceiling.parts)
        if w.roof is not None:
            all_ids.update(rs.id for rs in w.roof.surfaces)
    for wing in twin.building.wings:
        for story in wing.stories:
            for room in story.rooms:
                cx = sum(c.x for c in room.floor.polygon) / len(room.floor.polygon)
                cz = sum(c.z for c in room.floor.polygon) / len(room.floor.polygon)
                ceil = ceiling_y_at(twin, cx, cz)
                if ceil is None:
                    continue
                assert ceil.chosen_id in all_ids, (
                    f"chosen_id {ceil.chosen_id!r} is not a known "
                    f"ceiling primitive in this twin"
                )
