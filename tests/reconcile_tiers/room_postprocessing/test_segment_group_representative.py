"""Tests for per-group representative segment selection on room cycles."""

from __future__ import annotations

import math

from reconcile_tiers.room_postprocessing.export import build_corner_graph
from reconcile_tiers.room_postprocessing.segment_group_representative import (
    base_wall_id,
    perimeter_sides_for_cycle,
    representative_segments_for_cycle,
)
from tests.reconcile_tiers.room_postprocessing.test_segment_room_cycles import (
    _four_wall_room_payload,
)


def test_representative_picks_one_segment_per_incident_wall_at_corner() -> None:
    """Multi-wall group: only walls on cycle span edges, not every wall in the cluster."""

    segment_ids_by_group = {
        "g0": ["s-orange", "s-green", "s-blue", "s-north-a"],
        "g1": ["s-orange-b", "s-east-b"],
        "g2": ["s-north-b", "s-east-c"],
    }
    segments_by_id = {
        "s-orange": {
            "id": "s-orange",
            "wall_id": "w-orange",
            "start": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end": {"x": 0.0, "y": 2.0, "z": 0.0},
        },
        "s-orange-b": {
            "id": "s-orange-b",
            "wall_id": "w-orange",
            "start": {"x": 5.0, "y": 0.0, "z": 1.0},
            "end": {"x": 5.0, "y": 2.0, "z": 1.0},
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
        "s-north-a": {
            "id": "s-north-a",
            "wall_id": "w-north",
            "start": {"x": 2.0, "y": 0.0, "z": 1.0},
            "end": {"x": 2.0, "y": 2.0, "z": 1.0},
        },
        "s-north-b": {
            "id": "s-north-b",
            "wall_id": "w-north",
            "start": {"x": 5.0, "y": 0.0, "z": 5.0},
            "end": {"x": 5.0, "y": 2.0, "z": 5.0},
        },
        "s-east-b": {
            "id": "s-east-b",
            "wall_id": "w-east",
            "start": {"x": 5.0, "y": 0.0, "z": 1.0},
            "end": {"x": 5.0, "y": 2.0, "z": 1.0},
        },
        "s-east-c": {
            "id": "s-east-c",
            "wall_id": "w-east",
            "start": {"x": 5.0, "y": 0.0, "z": 5.0},
            "end": {"x": 5.0, "y": 2.0, "z": 5.0},
        },
    }
    positions = {"g0": (2.0, 1.0), "g1": (5.0, 1.0), "g2": (5.0, 5.0)}
    cycle = ["g0", "g1", "g2"]
    part_edges = [
        {"source": "g0", "target": "g1", "wall_id": "w-orange"},
        {"source": "g1", "target": "g2", "wall_id": "w-east"},
        {"source": "g2", "target": "g0", "wall_id": "w-north"},
    ]
    sides = perimeter_sides_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
    )
    seg_ids, wall_ids, by_group = representative_segments_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
        perimeter_sides=sides,
    )
    assert "w-orange" in wall_ids
    assert "s-orange" in by_group["g0"] or "s-orange-b" in by_group["g0"]
    assert "w-blue" not in wall_ids
    assert "w-green" not in wall_ids
    assert "s-blue" not in seg_ids
    assert "s-green" not in seg_ids


def test_corner_group_gets_two_reps_for_two_incident_walls() -> None:
    segment_ids_by_group = {
        "corner": ["s-n", "s-e"],
        "along_n": ["s-n-b"],
        "along_e": ["s-e-b"],
    }
    segments_by_id = {
        "s-n": {
            "id": "s-n",
            "wall_id": "w-north",
            "start": {"x": 2.0, "y": 0.0, "z": 0.0},
            "end": {"x": 2.0, "y": 2.0, "z": 0.0},
        },
        "s-n-b": {
            "id": "s-n-b",
            "wall_id": "w-north",
            "start": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end": {"x": 0.0, "y": 2.0, "z": 0.0},
        },
        "s-e": {
            "id": "s-e",
            "wall_id": "w-east",
            "start": {"x": 4.0, "y": 0.0, "z": 2.0},
            "end": {"x": 4.0, "y": 2.0, "z": 2.0},
        },
        "s-e-b": {
            "id": "s-e-b",
            "wall_id": "w-east",
            "start": {"x": 4.0, "y": 0.0, "z": 4.0},
            "end": {"x": 4.0, "y": 2.0, "z": 4.0},
        },
        "s-s-b": {
            "id": "s-s-b",
            "wall_id": "w-south",
            "start": {"x": 2.0, "y": 0.0, "z": 4.0},
            "end": {"x": 2.0, "y": 2.0, "z": 4.0},
        },
    }
    segment_ids_by_group["along_e"] = ["s-e-b", "s-s-b"]
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
    sides = perimeter_sides_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
    )
    seg_ids, wall_ids, by_group = representative_segments_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
        perimeter_sides=sides,
    )
    assert set(by_group["corner"]) == {"s-n", "s-e"}
    assert set(wall_ids) >= {"w-east", "w-north"}


def test_min_area_picks_shorter_split_side_over_long_span() -> None:
    """Dedupe keeps smallest quad area, not farthest junction pair."""

    cycle = ["g0", "g1", "g2", "g3"]
    part_edges = [
        {"source": "g0", "target": "g1", "wall_id": "w-long::split::0"},
        {"source": "g1", "target": "g2", "wall_id": "w-long::split::1"},
        {"source": "g2", "target": "g3", "wall_id": "w-east"},
        {"source": "g3", "target": "g0", "wall_id": "w-north"},
    ]
    segment_ids_by_group = {
        gid: [f"s-{gid}-long", f"s-{gid}-other"]
        for gid in cycle
    }
    segments_by_id = {}
    for gid in cycle:
        x = float(cycle.index(gid))
        segments_by_id[f"s-{gid}-long"] = {
            "id": f"s-{gid}-long",
            "wall_id": "w-long::split::0" if gid in ("g0", "g1") else "w-long::split::1",
            "start": {"x": x, "y": 0.0, "z": 0.0},
            "end": {"x": x, "y": 2.0, "z": 0.0},
        }
        segments_by_id[f"s-{gid}-other"] = {
            "id": f"s-{gid}-other",
            "wall_id": "w-east" if gid == "g2" else "w-north",
            "start": {"x": x, "y": 0.0, "z": 4.0},
            "end": {"x": x, "y": 2.0, "z": 4.0},
        }
    positions = {gid: (float(cycle.index(gid)), 0.0) for gid in cycle}
    sides = perimeter_sides_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
    )
    long_sides = [s for s in sides if base_wall_id(s["wall_id"]) == "w-long"]
    assert len(long_sides) == 1
    kept = long_sides[0]
    rim = math.hypot(
        positions[kept["target_group"]][0] - positions[kept["source_group"]][0],
        positions[kept["target_group"]][1] - positions[kept["source_group"]][1],
    )
    assert rim <= 2.0


def test_perimeter_sides_dedupe_split_pieces_to_one_physical_wall() -> None:
    cycle = ["g0", "g1", "g2", "g3"]
    part_edges = [
        {"source": "g0", "target": "g1", "wall_id": "w-long::split::0"},
        {"source": "g1", "target": "g2", "wall_id": "w-long::split::1"},
        {"source": "g2", "target": "g3", "wall_id": "w-east"},
        {"source": "g3", "target": "g0", "wall_id": "w-north"},
    ]
    segment_ids_by_group = {
        gid: [f"s-{gid}-long", f"s-{gid}-other"]
        for gid in cycle
    }
    segments_by_id = {}
    for gid in cycle:
        x = float(cycle.index(gid))
        segments_by_id[f"s-{gid}-long"] = {
            "id": f"s-{gid}-long",
            "wall_id": "w-long::split::0" if gid in ("g0", "g1") else "w-long::split::1",
            "start": {"x": x, "y": 0.0, "z": 0.0},
            "end": {"x": x, "y": 2.0, "z": 0.0},
        }
        segments_by_id[f"s-{gid}-other"] = {
            "id": f"s-{gid}-other",
            "wall_id": "w-east" if gid == "g2" else "w-north",
            "start": {"x": x, "y": 0.0, "z": 4.0},
            "end": {"x": x, "y": 2.0, "z": 4.0},
        }
    positions = {gid: (float(cycle.index(gid)), 0.0) for gid in cycle}
    sides = perimeter_sides_for_cycle(
        cycle,
        part_edges,
        segment_ids_by_group,
        segments_by_id,
        positions,
    )
    long_sides = [s for s in sides if base_wall_id(s["wall_id"]) == "w-long"]
    assert len(long_sides) == 1


def test_four_wall_room_one_rep_per_incident_wall_per_corner() -> None:
    graph = build_corner_graph(_four_wall_room_payload(), corner_tol=0.05)
    room = graph["segment_room_graph"]["nodes"][0]
    assert len(room["wall_ids"]) == 4
    assert len(room["perimeter_sides"]) == 4
    assert len(room["perimeter_wall_quads"]) == 4
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


def test_perimeter_wall_quad_rim_not_full_wall_extent() -> None:
    graph = build_corner_graph(_four_wall_room_payload(), corner_tol=0.05)
    room = graph["segment_room_graph"]["nodes"][0]
    cycle_len = sum(
        math.hypot(
            room["polygon_xz"][(i + 1) % len(room["polygon_xz"])]["x"]
            - room["polygon_xz"][i]["x"],
            room["polygon_xz"][(i + 1) % len(room["polygon_xz"])]["z"]
            - room["polygon_xz"][i]["z"],
        )
        for i in range(len(room["polygon_xz"]))
    )
    for quad in room["perimeter_wall_quads"]:
        corners = quad["corners"]
        rim = math.hypot(
            corners[1]["x"] - corners[0]["x"],
            corners[1]["z"] - corners[0]["z"],
        )
        assert rim <= cycle_len + 0.01
