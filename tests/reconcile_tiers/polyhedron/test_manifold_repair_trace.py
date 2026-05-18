"""Tests for manifold-repair pipeline step trace export."""

from __future__ import annotations

from reconcile_tiers.polyhedron.manifold_repair import (
    build_room_polyhedron,
    prepare_room_tiles,
)
from reconcile_tiers.polyhedron.manifold_repair_trace import (
    SELECTION,
    build_manifold_repair_building_trace,
    build_manifold_repair_room_trace,
    orphan_edges_for_frame,
)
from tests.reconcile_tiers.polyhedron.test_manifold_repair import _cube_tiles
from tests.reconcile_tiers.polyhedron.test_payload_adapter import _cube_payload


def test_prepare_room_tiles_merges_coplanar():
    tiles = _cube_tiles()
    merged = prepare_room_tiles(tiles, coord_tol=1e-3, snap_tol=0.0, merge_coplanar=True)
    assert len(merged) == len(tiles)


def test_orphan_edges_for_missing_wall():
    build = build_room_polyhedron(_cube_tiles()[:5])
    edges = orphan_edges_for_frame(build)
    assert len(edges) == 4
    assert all("a" in e and "b" in e for e in edges)


def test_build_manifold_repair_room_trace_nine_frames():
    payload = _cube_payload()
    trace = build_manifold_repair_room_trace(payload, payload["rooms"][0])
    assert trace["selection"] == SELECTION
    assert len(trace["frames"]) == 9
    steps = [f["pipeline_step"] for f in trace["frames"]]
    assert steps[0] == "tier_payload_input"
    assert steps[1] == "input_tiles"
    assert steps[2] == "tile_coherence"
    assert steps[-1] == "fillers_applied"
    assert trace["frames"][0]["faces"] == []
    coherence = trace["frames"][2]
    assert coherence["meta"]["ok"] is True
    assert trace["stop"]["reason"] == "watertight"


def test_holes_detected_frame_has_orphan_edges():
    payload = _cube_payload()
    payload["rooms"][0]["walls"] = payload["rooms"][0]["walls"][:3]
    trace = build_manifold_repair_room_trace(payload, payload["rooms"][0])
    holes = trace["frames"][5]
    assert holes["pipeline_step"] == "holes_detected"
    assert len(holes["orphan_edges"]) >= 4


def test_filler_candidates_frame_includes_candidates():
    payload = _cube_payload()
    payload["rooms"][0]["walls"] = payload["rooms"][0]["walls"][:3]
    trace = build_manifold_repair_room_trace(payload, payload["rooms"][0])
    candidates = trace["frames"][6]
    roles = {f["role"] for f in candidates["faces"]}
    assert "filler_candidate" in roles


def test_build_manifold_repair_building_trace_three_frames():
    trace = build_manifold_repair_building_trace(_cube_payload())
    assert len(trace["frames"]) == 3
    assert trace["frames"][0]["pipeline_step"] == "tier_payload_input"
    assert trace["frames"][0]["faces"] == []
    assert trace["frames"][1]["pipeline_step"] == "rooms_repaired"
    assert trace["frames"][2]["pipeline_step"] == "building_exterior"
    assert len(trace["frames"][2]["faces"]) >= 1
