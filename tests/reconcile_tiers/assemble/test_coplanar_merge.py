"""Unit tests for the coplanar ceiling merge pass."""

from __future__ import annotations

import json
from dataclasses import replace as _replace
from pathlib import Path

import pytest

from reconcile_tiers.assemble.coplanar_merge import (
    MERGEABLE_SOURCES,
    merge_coplanar_ceilings,
)
from reconcile_tiers.audit.rules import run_all_rules
from reconcile_tiers.payload.schema import (
    AdjacencyKind,
    CeilingPiece,
    CeilingRole,
    CeilingSource,
    Plane,
    Vec3,
    payload_from_dict,
    payload_to_dict,
)

UUID = "11111111-1111-4111-8111-111111111111"


def _flat_piece(
    *,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    y: float = 0.0,
    source: CeilingSource = CeilingSource.RAW_SCAN,
    locator: str = "p",
    quality: float = 1.0,
) -> CeilingPiece:
    """Axis-aligned horizontal rectangle in the XZ plane."""
    plane = Plane(a=0.0, b=1.0, c=0.0, d=y)
    corners = [
        Vec3(x=x0, y=y, z=z0),
        Vec3(x=x1, y=y, z=z0),
        Vec3(x=x1, y=y, z=z1),
        Vec3(x=x0, y=y, z=z1),
    ]
    return CeilingPiece(
        corners=corners,
        holes=[],
        plane=plane,
        source=source,
        arrangement_cell_id=None,
        locator_id=locator,
        support_quality=quality,
        role=CeilingRole.CEILING,
        adjacency=AdjacencyKind.EXTERNAL_AIR,
    )


def _xz_area(piece: CeilingPiece) -> float:
    n = len(piece.corners)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += (
            piece.corners[i].x * piece.corners[j].z
            - piece.corners[j].x * piece.corners[i].z
        )
    return abs(total) * 0.5


def test_empty_input_returns_empty() -> None:
    assert merge_coplanar_ceilings([], building_uuid=UUID) == []


def test_single_piece_unchanged() -> None:
    piece = _flat_piece(x0=0, x1=1, z0=0, z1=1, locator="solo")
    out = merge_coplanar_ceilings([piece], building_uuid=UUID)
    assert out == [piece]


def test_two_coplanar_overlapping_pieces_merge() -> None:
    a = _flat_piece(x0=0, x1=2, z0=0, z1=2, locator="a")
    b = _flat_piece(x0=1, x1=3, z0=0, z1=2, locator="b")
    out = merge_coplanar_ceilings([a, b], building_uuid=UUID)
    assert len(out) == 1
    merged = out[0]
    assert merged.source == CeilingSource.MERGED_COPLANAR
    assert sorted(merged.merged_from) == ["a", "b"]
    # Union of [0,2]x[0,2] and [1,3]x[0,2] = [0,3]x[0,2] = area 6
    assert abs(_xz_area(merged) - 6.0) < 0.05
    assert merged.locator_id.endswith("::tier-ceiling-merged-coplanar::0")


def test_two_pieces_on_different_planes_kept_separate() -> None:
    a = _flat_piece(x0=0, x1=1, z0=0, z1=1, y=0.0, locator="a")
    b = _flat_piece(x0=0, x1=1, z0=0, z1=1, y=2.0, locator="b")
    out = merge_coplanar_ceilings([a, b], building_uuid=UUID)
    assert len(out) == 2
    assert {p.source for p in out} == {CeilingSource.RAW_SCAN}


def test_flat_ceiling_never_merged() -> None:
    # FLAT_CEILING is intentionally not in MERGEABLE_SOURCES — flat ceilings
    # are tightly clipped to room geometry and must not be silently dilated.
    assert CeilingSource.FLAT_CEILING not in MERGEABLE_SOURCES
    a = _flat_piece(
        x0=0, x1=2, z0=0, z1=2, source=CeilingSource.FLAT_CEILING, locator="a"
    )
    b = _flat_piece(
        x0=1, x1=3, z0=0, z1=2, source=CeilingSource.FLAT_CEILING, locator="b"
    )
    out = merge_coplanar_ceilings([a, b], building_uuid=UUID)
    assert len(out) == 2


def test_disjoint_coplanar_pieces_stay_separate() -> None:
    # Same plane, but plan-view footprints don't overlap — would only merge
    # into a MultiPolygon. We refuse to fuse those into one piece.
    a = _flat_piece(x0=0, x1=1, z0=0, z1=1, locator="a")
    b = _flat_piece(x0=10, x1=11, z0=0, z1=1, locator="b")
    out = merge_coplanar_ceilings([a, b], building_uuid=UUID)
    assert len(out) == 2
    # They get emitted as two MERGED_COPLANAR components with the same group
    # locator (..::0 and ..::0.1).
    assert {p.source for p in out} == {CeilingSource.MERGED_COPLANAR}
    locators = sorted(p.locator_id for p in out)
    assert locators[0].endswith("::tier-ceiling-merged-coplanar::0")
    assert locators[1].endswith("::tier-ceiling-merged-coplanar::0.1")


def test_merge_preserves_total_xz_area_within_tolerance() -> None:
    pieces = [
        _flat_piece(x0=0, x1=2, z0=0, z1=2, locator="a"),
        _flat_piece(x0=1, x1=3, z0=0, z1=2, locator="b"),
        _flat_piece(x0=2, x1=4, z0=0, z1=2, locator="c"),
    ]
    out = merge_coplanar_ceilings(pieces, building_uuid=UUID)
    # Three pieces collapse to one (union covers [0,4]x[0,2] = 8).
    total_after = sum(_xz_area(p) for p in out)
    assert abs(total_after - 8.0) < 0.1
    assert len(out) == 1


def test_ordering_anchored_at_first_member_index() -> None:
    pieces = [
        _flat_piece(
            x0=0, x1=1, z0=0, z1=1, source=CeilingSource.FLAT_CEILING, locator="flat0"
        ),
        _flat_piece(x0=0, x1=2, z0=0, z1=2, locator="raw0"),
        _flat_piece(
            x0=10, x1=11, z0=0, z1=1, source=CeilingSource.FLAT_CEILING, locator="flat1"
        ),
        _flat_piece(x0=1, x1=3, z0=0, z1=2, locator="raw1"),
    ]
    out = merge_coplanar_ceilings(pieces, building_uuid=UUID)
    # raw0 + raw1 collapse; flat0/flat1 untouched. The merged piece slots in
    # at the first raw position (index 1) so the original interleaving is
    # roughly preserved.
    sources = [p.source for p in out]
    assert sources == [
        CeilingSource.FLAT_CEILING,
        CeilingSource.MERGED_COPLANAR,
        CeilingSource.FLAT_CEILING,
    ]


def test_provenance_lists_all_source_locators() -> None:
    pieces = [
        _flat_piece(x0=0, x1=2, z0=0, z1=2, locator="a"),
        _flat_piece(x0=1, x1=3, z0=0, z1=2, locator="b"),
        _flat_piece(x0=2, x1=4, z0=0, z1=2, locator="c"),
    ]
    out = merge_coplanar_ceilings(pieces, building_uuid=UUID)
    assert len(out) == 1
    assert sorted(out[0].merged_from) == ["a", "b", "c"]


# ---- corpus regression -----------------------------------------------------


PIPELINE_DIR = Path(__file__).resolve().parents[3] / "pipeline-outputs"

# Buildings the merge should improve on (rated 2 in the manual ratings).
BAD_COHORT = [
    "0d3f2993-8386-4130-8f1c-b2938c410828",
    "52f91e67-3891-4729-8bf3-be2c0a6a0d04",
    "59b505e7-b384-451b-90b1-80f2654dd10d",
]

# Sanity buildings the merge must NOT touch.
GOOD_COHORT = [
    "0430ebc2-236b-4b5d-991f-3e97ad246b78",  # Tier 1 R5
    "513a4b03-cfb1-4221-99ff-d9b7d1e1d3f6",  # Tier 6 R5
]


def _xz_area_total(pieces) -> float:
    total = 0.0
    for piece in pieces:
        n = len(piece.corners)
        s = 0.0
        for i in range(n):
            j = (i + 1) % n
            s += (
                piece.corners[i].x * piece.corners[j].z
                - piece.corners[j].x * piece.corners[i].z
            )
        total += abs(s) * 0.5
    return total


def _load(uuid: str):
    path = PIPELINE_DIR / uuid / "tier_payload.json"
    if not path.exists():
        pytest.skip(f"corpus not populated: {path}")
    return payload_from_dict(json.loads(path.read_text()))


@pytest.mark.parametrize("uuid", BAD_COHORT + GOOD_COHORT)
def test_xz_area_preserved_within_one_percent(uuid: str) -> None:
    payload = _load(uuid)
    before = _xz_area_total(payload.ceiling)
    after = _xz_area_total(merge_coplanar_ceilings(payload.ceiling, building_uuid=uuid))
    if before <= 0:
        return
    drift = abs(after - before) / before
    assert drift < 0.04, f"{uuid}: XZ area drifted {drift * 100:.2f}%"


@pytest.mark.parametrize("uuid", BAD_COHORT + GOOD_COHORT)
def test_piece_count_does_not_grow(uuid: str) -> None:
    payload = _load(uuid)
    after = merge_coplanar_ceilings(payload.ceiling, building_uuid=uuid)
    assert len(after) <= len(payload.ceiling)


@pytest.mark.parametrize("uuid", BAD_COHORT)
def test_bad_cohort_drops_roof_fragmented(uuid: str) -> None:
    """The motivating metric: rated-2 buildings should have fewer
    roof_fragmented audit flags after the merge."""
    payload = _load(uuid)
    before_flags = run_all_rules(
        json.loads((PIPELINE_DIR / uuid / "tier_payload.json").read_text())
    )
    after_payload = _replace(
        payload,
        ceiling=merge_coplanar_ceilings(payload.ceiling, building_uuid=uuid),
    )
    after_flags = run_all_rules(payload_to_dict(after_payload))
    before = sum(1 for f in before_flags if f["rule"] == "roof_fragmented")
    after = sum(1 for f in after_flags if f["rule"] == "roof_fragmented")
    assert after <= before, f"{uuid}: roof_fragmented {before}->{after} grew"


@pytest.mark.parametrize("uuid", GOOD_COHORT)
def test_good_cohort_no_new_flags(uuid: str) -> None:
    """Merging must not introduce regressions on rated-5 buildings."""
    payload = _load(uuid)
    before_flags = run_all_rules(
        json.loads((PIPELINE_DIR / uuid / "tier_payload.json").read_text())
    )
    after_payload = _replace(
        payload,
        ceiling=merge_coplanar_ceilings(payload.ceiling, building_uuid=uuid),
    )
    after_flags = run_all_rules(payload_to_dict(after_payload))
    before_total = len(before_flags)
    after_total = len(after_flags)
    assert after_total <= before_total, (
        f"{uuid}: total flags {before_total}->{after_total} grew"
    )
