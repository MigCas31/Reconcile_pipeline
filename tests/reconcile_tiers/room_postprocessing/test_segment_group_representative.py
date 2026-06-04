"""Tests for per-group representative segment selection on room cycles."""

from __future__ import annotations

from reconcile_tiers.room_postprocessing.export import build_corner_graph
from reconcile_tiers.room_postprocessing.segment_group_representative import (
    representative_segments_for_cycle,
)
from tests.reconcile_tiers.room_postprocessing.test_segment_room_cycles import (
    _four_wall_room_payload,
)


def test_representative_picks_one_segment_per_incident_wall_at_corner() -> None:
    """Multi-wall group: only walls on cycle span edges, not every wall in the cluster."""

    segment_ids_by_group = {
        "g0": ["s-orange", "s-green", "s-blue"],
    }
    segments_by_id = {
        "s-orange": {
            "id": "s-orange",
            "wall_id": "w-orange",
            "start": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end": {"x": 0.0, "y": 2.0, "z": 0.0},
        },
        "s-green": {
            "id": "s-green",
            "wall_id": "w-green",
            "start": {"x": 2.0, "y": 0.0, "z": 2.0},
            "end": {"x": 2.0, "y": 2.0, "z": 2.0},
        },
        "s-blue": {
            "id": "s-blue",
            "wall_id": "w-blue",
            "start": {"x": 3.0, "y": 0.0, "z": 2.0},
            "end": {"x": 3.0, "y": 2.0, "z": 2.0},
        },
    }
    positions = {"g0": (2.0, 1.0), "g1": (5.0, 1.0), "g2": (5.0, 5.0)}
    cycle = ["g0", "g1", "g2"]
    part_edges = [
        {"source": "g0", "target": "g1", "wall_id": "w-orange"},
        {"source": "g1", "target": "g2", "wall_id": "w-east"},
        {"source": "g2", "target": "g0", "wall_id": "w-north"},
    ]
    seg_ids, wall_ids, by_group = representative_segments_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
    )
    assert wall_ids == ["w-orange"]
    assert seg_ids == ["s-orange"]
    assert by_group["g0"] == ["s-orange"]
    assert "w-blue" not in wall_ids
    assert "w-green" not in wall_ids
    assert "s-blue" not in seg_ids
    assert "s-green" not in seg_ids


def test_corner_group_gets_two_reps_for_two_incident_walls() -> None:
    segment_ids_by_group = {
        "corner": ["s-n", "s-e"],
    }
    segments_by_id = {
        "s-n": {
            "id": "s-n",
            "wall_id": "w-north",
            "start": {"x": 2.0, "y": 0.0, "z": 0.0},
            "end": {"x": 2.0, "y": 2.0, "z": 0.0},
        },
        "s-e": {
            "id": "s-e",
            "wall_id": "w-east",
            "start": {"x": 4.0, "y": 0.0, "z": 2.0},
            "end": {"x": 4.0, "y": 2.0, "z": 2.0},
        },
    }
    positions = {
        "corner": (2.0, 0.0),
        "along_n": (0.0, 0.0),
        "along_e": (4.0, 4.0),
    }
    cycle = ["along_n", "corner", "along_e"]
    part_edges = [
        {"source": "along_n", "target": "corner", "wall_id": "w-north"},
        {"source": "corner", "target": "along_e", "wall_id": "w-east"},
        {"source": "along_e", "target": "along_n", "wall_id": "w-south"},
    ]
    seg_ids, wall_ids, by_group = representative_segments_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
    )
    assert set(by_group["corner"]) == {"s-n", "s-e"}
    assert set(wall_ids) >= {"w-east", "w-north"}


def test_four_wall_room_one_rep_per_incident_wall_per_corner() -> None:
    graph = build_corner_graph(_four_wall_room_payload(), corner_tol=0.05)
    room = graph["segment_room_graph"]["nodes"][0]
    assert len(room["wall_ids"]) == 4
    assert len(room["segment_ids"]) == 8
    for gid in room["group_ids"]:
        reps = room["representative_by_group"][gid]
        assert 1 <= len(reps) <= 2
        node = next(n for n in graph["wall_segment_graph"]["nodes"] if n["id"] == gid)
        for seg_id in reps:
            seg = next(
                s for s in graph["wall_segment_graph"]["segments"] if s["id"] == seg_id
            )
            assert seg["wall_id"] in node["wall_ids"]
