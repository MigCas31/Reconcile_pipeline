"""Phase C corpus tests: full Twin (Building + Residual) per walkthrough
building."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.twin import Twin
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


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_assemble_building_yields_twin(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, builds = assemble_building(model)
    assert isinstance(twin, Twin), f"{uuid}: no twin produced"
    assert twin.building.uuid == uuid
    assert len(twin.building.wings) >= 1
    assert builds, "no per-room builds returned"


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_split_level_has_two_stories(uuid):
    """The split-level walkthrough building must yield 2 stories
    (half-height upper + ground)."""
    if uuid != "9bc73438-328e-440e-a4be-25efa61d3b7c":
        pytest.skip("only the split-level uuid is asserted here")
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    n_stories = sum(len(w.stories) for w in twin.building.wings)
    assert n_stories == 2


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_wing_footprint_is_horizontal(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    for wing in twin.building.wings:
        ys = [c.y for c in wing.footprint]
        assert max(ys) - min(ys) <= 1e-6, (
            f"wing {wing.id} footprint Y-span exceeds float precision"
        )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_residual_present_even_for_full_buildings(uuid):
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    # The residual stream is always emitted (may be empty); Phase D
    # populates `unclaimed_gaps` and `fully_inferred_primitive_ids`.
    assert twin.residual is not None


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_at_least_one_wing_has_a_roof(uuid):
    """Each walkthrough building has at least one wing with oblique
    top-story sub-ceilings (i.e. a non-flat roof). The Roof primitive
    must be attached to that wing."""
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    n_wings_with_roof = sum(1 for w in twin.building.wings if w.roof is not None)
    assert n_wings_with_roof >= 1, f"{uuid}: no wing carries a Roof"


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_clean_gable_produces_two_distinct_roof_planes(uuid):
    """A gable has two architectural planes. Each plane's plan-view
    extent may be split into multiple RoofSurfaces when extraction's
    per-room polygons don't touch — that's a polygon-coverage detail,
    not an architectural fact. Verify exactly 2 distinct *planes*."""
    if uuid != "53c380e7-407e-48f7-8c72-b449e2334798":
        pytest.skip("only the clean gable uuid is asserted here")
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    plane_keys = set()
    for wing in twin.building.wings:
        if wing.roof is None:
            continue
        for surface in wing.roof.surfaces:
            # Round so SVD-fit residuals don't multiply the count.
            plane_keys.add(
                (
                    round(surface.plane.a, 3),
                    round(surface.plane.b, 3),
                    round(surface.plane.c, 3),
                    round(surface.plane.d, 3),
                )
            )
    assert len(plane_keys) == 2, (
        f"expected 2 distinct architectural planes, got {len(plane_keys)}: {plane_keys}"
    )


@pytest.mark.parametrize("uuid", WALKTHROUGH_UUIDS)
def test_every_roof_surface_is_oblique_with_upward_normal(uuid):
    """`RoofSurface.__post_init__` already enforces this; this test
    confirms the assembler hasn't accidentally promoted a flat ceiling
    to a roof surface."""
    if not _building_available(uuid):
        pytest.skip(f"corpus building {uuid} not available")
    model = extract_building_model(uuid, PIPELINE, SCAN_CACHE)
    twin, _ = assemble_building(model)
    assert twin is not None
    for wing in twin.building.wings:
        if wing.roof is None:
            continue
        for surface in wing.roof.surfaces:
            assert 0.0 < surface.plane.b < 1.0, (
                f"{surface.id}: plane.b={surface.plane.b} not strictly oblique-upward"
            )
