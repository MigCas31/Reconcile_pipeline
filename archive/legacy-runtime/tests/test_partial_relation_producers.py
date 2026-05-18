from __future__ import annotations

import json
import tempfile
from pathlib import Path

from reconcile.loader import load_merged
from reconcile_v2.graph_builder import build_topology_graph
from reconcile_v2.topology import infer_gap_records, infer_intra_story_adjacency


def _identity_flat() -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _surface(identifier: str, story: int, tx: float, ty: float, tz: float) -> dict:
    tf = _identity_flat()
    tf[12] = tx
    tf[13] = ty
    tf[14] = tz
    return {
        "identifier": identifier,
        "category": {"wall": {"hasOpening": False}},
        "confidence": {"confidence": "high"},
        "dimensions": [2.0, 2.4, 0.1],
        "transform": tf,
        "polygonCorners": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
        ],
        "story": story,
        "completedEdges": [],
        "parentIdentifier": None,
        "curve": None,
    }


def _floor(
    identifier: str,
    story: int,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    y: float = 0.0,
) -> dict:
    tf = _identity_flat()
    tf[13] = y
    return {
        "identifier": identifier,
        "category": {"floor": {}},
        "confidence": {"confidence": "high"},
        "dimensions": [abs(x1 - x0), 0.0, abs(z1 - z0)],
        "transform": tf,
        "polygonCorners": [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]],
        "story": story,
        "completedEdges": [],
        "parentIdentifier": None,
        "curve": None,
    }


def _room(
    identifier: str,
    story: int,
    floor_bounds: tuple[float, float, float, float],
    *,
    floor_y: float = 0.0,
) -> dict:
    x0, z0, x1, z1 = floor_bounds
    return {
        "story": story,
        "walls": [
            _surface(f"wall_{identifier}", story, (x0 + x1) * 0.5, floor_y + 1.2, z0)
        ],
        "doors": [],
        "windows": [],
        "openings": [],
        "objects": [],
        "floors": [_floor(f"floor_{identifier}", story, x0, z0, x1, z1, y=floor_y)],
        "referenceOriginTransform": _identity_flat(),
    }


def _write_fixture(
    root: Path, rooms: list[dict], uuid: str = "11111111-2222-3333-4444-555555555555"
) -> tuple[Path, Path, str]:
    merged_path = root / "merged.json"
    scan_dir = root / "scan-cache"
    scan_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        "version": 2,
        "walls": [wall for room in rooms for wall in room["walls"]],
        "doors": [],
        "windows": [],
        "openings": [],
        "objects": [],
        "floors": [floor for room in rooms for floor in room["floors"]],
        "rooms": rooms,
        "sections": [],
    }
    merged_path.write_text(json.dumps(merged))
    for idx, room in enumerate(rooms):
        (scan_dir / f"room_{idx}.json").write_text(json.dumps(room))
    return merged_path, scan_dir, uuid


def test_half_floor_adjacency_is_marked_partial_upstream() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rooms = [
            _room("a", 0, (0.0, 0.0, 2.0, 2.0), floor_y=0.0),
            _room("b", 0, (2.1, 0.0, 4.1, 2.0), floor_y=0.4),
        ]
        merged_path, scan_dir, uuid = _write_fixture(Path(tmp), rooms)
        building = load_merged(merged_path)

        adj = infer_intra_story_adjacency(building)
        assert adj
        assert adj[0].relation_state == "partial"
        assert adj[0].floor_delta_m > 0.15
        assert adj[0].support_ratio < 0.35

        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )
        adjacency_edges = [edge for edge in graph.edges if edge.type == "ADJACENT_TO"]
        assert adjacency_edges
        assert all(
            (edge.evidence or {}).get("relation_state") == "partial"
            for edge in adjacency_edges
        )


def test_cross_story_gap_records_emit_partial_state_support() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rooms = [
            _room("a", 0, (0.0, 0.0, 2.0, 3.0), floor_y=0.0),
            _room("b", 0, (2.0, 1.0, 4.0, 3.0), floor_y=0.0),
            _room("c", 1, (0.0, 0.0, 4.0, 3.0), floor_y=3.0),
        ]
        merged_path, _scan_dir, _uuid = _write_fixture(Path(tmp), rooms)
        building = load_merged(merged_path)

        gaps, _footprints = infer_gap_records(building)
        cross_story = [gap for gap in gaps if gap.kind == "cross_story"]

        assert cross_story
        assert all(
            gap.relation_state in {"partial", "confirmed"} for gap in cross_story
        )
        assert all(gap.support_ratio >= 0.35 for gap in cross_story)
