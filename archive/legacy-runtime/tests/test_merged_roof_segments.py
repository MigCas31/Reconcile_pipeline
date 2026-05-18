"""Smoke test for the merged_slanted_roof_proposals stage.

Runs the full v3 pipeline on one building known to have multiple merged
planes (Brydevej 14 = ``016980bc``) and verifies that:

  - ``merged_roof_segments`` is populated,
  - each segment has the provenance needed for reverse engineering
    (member ids, cluster params, building boundary, opposing planes,
    room/piece ref),
  - segments are emitted per rain-exposed piece (so a cluster spanning
    two rooms yields at least two segments).
"""

from __future__ import annotations

from pathlib import Path

from reconcile_v3.io.load import load_building
from reconcile_v3.pipeline import run_pipeline

BUILDINGS_JSON = Path(__file__).parent.parent / "reconcile" / "buildings_3d.json"
BRYDEVEJ_UUID = "016980bc-6762-4022-bfbf-17df4112e10c"


def test_merged_segments_emitted_with_full_provenance() -> None:
    data = load_building(BUILDINGS_JSON, BRYDEVEJ_UUID)
    building = run_pipeline(data)

    assert building.roof_proposals, "expected permissive proposals for Brydevej"
    assert building.merged_roof_segments, "expected merged segments from clustering"

    seg = building.merged_roof_segments[0]
    assert seg.cluster_canonical_id.startswith(f"{BRYDEVEJ_UUID}::v3-roof-proposal::")
    assert seg.member_proposal_ids, "cluster must have at least one member"
    assert seg.corners and len(seg.corners) >= 3
    assert seg.footprint_xz and len(seg.footprint_xz) >= 3
    assert len(seg.merged_plane) == 4
    assert seg.cluster_params["normal_dot_min"] == 0.94
    assert seg.cluster_params["d_abs_max"] == 0.50
    assert seg.building_boundary_xz is not None
    assert seg.room_boundary_refs and seg.room_boundary_refs[0]["kind"] in {
        "room",
        "gap",
    }
    assert seg.features["area_m2"] > 0.0
    # Each snapshot carries the full member dict (so labels are self-contained).
    m0 = seg.member_snapshots[0]
    for key in ("id", "plane", "corners", "features", "heuristic_label"):
        assert key in m0, f"member snapshot missing {key}"


def test_segments_split_by_room_and_opposing_planes() -> None:
    data = load_building(BUILDINGS_JSON, BRYDEVEJ_UUID)
    building = run_pipeline(data)

    # A cluster that spans multiple rain-exposed pieces must emit multiple
    # segments (one per piece); aggregate by canonical id.
    by_cluster: dict[str, list] = {}
    for s in building.merged_roof_segments:
        by_cluster.setdefault(s.cluster_canonical_id, []).append(s)

    # At least one cluster should have produced >1 segment (room/piece split).
    multi_piece_clusters = [c for c, segs in by_cluster.items() if len(segs) > 1]
    assert multi_piece_clusters, "expected at least one cluster split across pieces"

    # Opposing-plane clipping: at least one cluster is expected to have
    # opposing clusters and therefore non-empty opposing metadata.
    assert any(s.opposing_cluster_canonicals for s in building.merged_roof_segments), (
        "expected at least one cluster with opposing planes"
    )


def test_opposing_seams_split_into_both_sides() -> None:
    """When a cluster has an opposing plane, we must emit BOTH the
    rain-hitting and the covered half as independent segments — so that
    each side of a ridge is separately labelable."""

    data = load_building(BUILDINGS_JSON, BRYDEVEJ_UUID)
    building = run_pipeline(data)

    # Find a cluster that has at least one opposing plane and inspect the
    # side mix. The previous (clip-only) implementation would have emitted
    # only rain-hitting sides; the split implementation produces both.
    sides_seen_by_cluster: dict[str, set[str]] = {}
    for seg in building.merged_roof_segments:
        if not seg.opposing_cluster_canonicals:
            continue
        sides = seg.trace.inputs.get("opposing_sides") or []
        for _canonical, side in sides:
            sides_seen_by_cluster.setdefault(seg.cluster_canonical_id, set()).add(side)

    both_sides = [
        c for c, sides in sides_seen_by_cluster.items() if {"rain", "covered"} <= sides
    ]
    assert both_sides, (
        "expected at least one cluster to emit segments on BOTH sides of an "
        "opposing seam (split semantics, not clip-and-discard)"
    )
