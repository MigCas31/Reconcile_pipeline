from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest
from shapely.geometry import Point

from reconcile.extract3d.gaps import (
    assign_gaps_to_rooms,
    compute_gap_walls,
    recommend_gap_actions,
)
from reconcile.extract3d.overlaps import clip_floor_overlaps, floor_polygon_to_shapely
from reconcile.extract3d.stitch import recommend_stitch_actions
from reconcile.roof_algorithms_py.math_utils import plane_normal, plane_y_at
from reconcile.roof_algorithms_py.occupied_room_cell_complex import (
    build_occupied_room_cell_complex,
)
from reconcile.roof_algorithms_py.roof_cell_complex import (
    _surface_y_at as cell_complex_surface_y_at,
)
from reconcile.roof_algorithms_py.roof_partitioning import _height_at, _height_model
from reconcile.roof_algorithms_py.steps import build_roof_graph_context
from reconcile.viewer_server import (
    FULL_BUILDING_PART_ID,
    UNASSIGNED_PART_ID,
    _build_ontology_part_payloads,
    _build_ontology_summary,
)
from reconcile.viewer_server import (
    _surface_y_at as viewer_surface_y_at,
)
from reconcile_v2.decision_logic import (
    build_decision_snapshot,
    classify_gap_decision,
    classify_overlap_resolution,
    classify_roof_room,
    classify_room_relation_state,
    classify_stitch_decision,
)
from reconcile_v2.graph_builder import build_topology_graph
from reconcile_v2.graph_validation import validate_graph


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


def _surface(
    identifier: str, story: int, tx: float, ty: float, tz: float, kind: str = "wall"
) -> dict:
    tf = _identity_flat()
    tf[12] = tx
    tf[13] = ty
    tf[14] = tz
    return {
        "identifier": identifier,
        "category": {kind: {"hasOpening": False}},
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
    identifier: str, story: int, x0: float, z0: float, x1: float, z1: float
) -> dict:
    return {
        "identifier": identifier,
        "category": {"floor": {}},
        "confidence": {"confidence": "high"},
        "dimensions": [abs(x1 - x0), 0.0, abs(z1 - z0)],
        "transform": _identity_flat(),
        "polygonCorners": [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]],
        "story": story,
        "completedEdges": [],
        "parentIdentifier": None,
        "curve": None,
    }


def _room(
    room_idx: int,
    story: int,
    wall_id: str,
    floor_id: str,
    floor_bounds: tuple[float, float, float, float],
) -> dict:
    x0, z0, x1, z1 = floor_bounds
    return {
        "story": story,
        "walls": [
            _surface(wall_id, story=story, tx=(x0 + x1) / 2, ty=1.2, tz=z0, kind="wall")
        ],
        "doors": [],
        "windows": [],
        "openings": [],
        "objects": [],
        "floors": [_floor(floor_id, story=story, x0=x0, z0=z0, x1=x1, z1=z1)],
        "referenceOriginTransform": _identity_flat(),
    }


def _scan_room(
    story: int,
    wall_id: str,
    floor_id: str,
    floor_bounds: tuple[float, float, float, float],
) -> dict:
    return _room(0, story, wall_id, floor_id, floor_bounds)


def _make_fixture(root: Path) -> tuple[Path, Path, str]:
    uuid = "11111111-2222-3333-4444-555555555555"
    merged_path = root / "merged.json"
    scan_dir = root / "scan-cache"
    scan_dir.mkdir(parents=True, exist_ok=True)

    room_a = _room(0, 0, "wall_a", "floor_a", (0.0, 0.0, 2.0, 2.0))
    room_b = _room(1, 0, "wall_b", "floor_b", (2.1, 0.0, 4.1, 2.0))

    merged = {
        "version": 2,
        "walls": room_a["walls"] + room_b["walls"],
        "doors": [],
        "windows": [],
        "openings": [],
        "objects": [],
        "floors": room_a["floors"] + room_b["floors"],
        "rooms": [room_a, room_b],
        "sections": [],
    }
    merged_path.write_text(json.dumps(merged))

    (scan_dir / "room_a.json").write_text(
        json.dumps(_scan_room(0, "wall_a", "floor_a", (0.0, 0.0, 2.0, 2.0)))
    )
    (scan_dir / "room_b.json").write_text(
        json.dumps(_scan_room(0, "wall_b", "floor_b", (2.1, 0.0, 4.1, 2.0)))
    )
    return merged_path, scan_dir, uuid


def _make_overlapping_fixture(root: Path) -> tuple[Path, Path, str]:
    uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    merged_path = root / "merged.json"
    scan_dir = root / "scan-cache"
    scan_dir.mkdir(parents=True, exist_ok=True)

    room_a = _room(0, 0, "wall_a", "floor_a", (0.0, 0.0, 2.5, 2.0))
    room_b = _room(1, 0, "wall_b", "floor_b", (1.5, 0.0, 4.0, 2.0))
    merged = {
        "version": 2,
        "walls": room_a["walls"] + room_b["walls"],
        "doors": [],
        "windows": [],
        "openings": [],
        "objects": [],
        "floors": room_a["floors"] + room_b["floors"],
        "rooms": [room_a, room_b],
        "sections": [],
    }
    merged_path.write_text(json.dumps(merged))
    (scan_dir / "room_a.json").write_text(
        json.dumps(_scan_room(0, "wall_a", "floor_a", (0.0, 0.0, 2.5, 2.0)))
    )
    (scan_dir / "room_b.json").write_text(
        json.dumps(_scan_room(0, "wall_b", "floor_b", (1.5, 0.0, 4.0, 2.0)))
    )
    return merged_path, scan_dir, uuid


def _make_enclosed_void_fixture(root: Path) -> tuple[Path, Path, str]:
    uuid = "99999999-aaaa-bbbb-cccc-dddddddddddd"
    merged_path = root / "merged.json"
    scan_dir = root / "scan-cache"
    scan_dir.mkdir(parents=True, exist_ok=True)

    room_left = _room(0, 0, "wall_left", "floor_left", (0.0, 0.0, 1.0, 3.0))
    room_right = _room(1, 0, "wall_right", "floor_right", (2.0, 0.0, 3.0, 3.0))
    room_bottom = _room(2, 0, "wall_bottom", "floor_bottom", (1.0, 0.0, 2.0, 1.0))
    room_top = _room(3, 0, "wall_top", "floor_top", (1.0, 2.0, 2.0, 3.0))
    rooms = [room_left, room_right, room_bottom, room_top]
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

    (scan_dir / "room_left.json").write_text(
        json.dumps(_scan_room(0, "wall_left", "floor_left", (0.0, 0.0, 1.0, 3.0)))
    )
    (scan_dir / "room_right.json").write_text(
        json.dumps(_scan_room(0, "wall_right", "floor_right", (2.0, 0.0, 3.0, 3.0)))
    )
    (scan_dir / "room_bottom.json").write_text(
        json.dumps(_scan_room(0, "wall_bottom", "floor_bottom", (1.0, 0.0, 2.0, 1.0)))
    )
    (scan_dir / "room_top.json").write_text(
        json.dumps(_scan_room(0, "wall_top", "floor_top", (1.0, 2.0, 2.0, 3.0)))
    )
    return merged_path, scan_dir, uuid


def test_enriched_graph_has_cells_and_indexes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        cells = graph.nodes_by_type("Cell")
        assert cells
        assert graph.get_node("cell:outside") is not None
        room = graph.nodes_by_type("Room")[0]
        assert graph.parent(room.id) is not None
        assert graph.find_path(graph.parent(room.id).id, room.id) is not None
        assert "cell_complex" in graph.geometry_index
        assert graph.quality["counts"]["cells"] >= 1


def test_enriched_graph_sets_story_room_properties_and_roles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        room = graph.nodes_by_type("Room")[0]
        story = graph.parent(room.id)
        wall = next(
            node
            for node in graph.nodes_by_type("Surface")
            if (node.properties or {}).get("surface_kind") == "wall"
        )

        assert room.properties["floor_height_y"] <= room.properties["ceiling_height_y"]
        assert "height_m" in (story.properties or {})
        assert (wall.properties or {}).get("surface_role") in {
            "interior_wall",
            "exterior_wall",
        }


def test_graph_validation_and_hybrid_hooks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        diagnostics = validate_graph(graph)
        gap_actions = recommend_gap_actions(graph)
        stitch_actions = recommend_stitch_actions(graph)
        roof_context = build_roof_graph_context(graph)

        assert isinstance(diagnostics, list)
        assert isinstance(gap_actions, dict)
        assert isinstance(stitch_actions, list)
        assert isinstance(roof_context["top_story_rooms"], list)
        assert isinstance(roof_context["roof_candidate_rooms"], list)


def test_oblique_surface_uses_cluster_plane_without_explicit_kind() -> None:
    surface = {
        "roof_hypothesis_id": "roof-hypothesis:oblique:1",
        "cluster": {
            "avgAzimuth": 315.0,
            "avgIncl": 20.0,
            "refPt": {"x": 0.0, "y": 2.0, "z": 0.0},
        },
        "corners": [
            [0.0, 10.0, 0.0],
            [2.0, 10.0, 0.0],
            [2.0, 10.0, 2.0],
            [0.0, 10.0, 2.0],
        ],
    }
    plane = {
        "n": plane_normal(315.0, 20.0),
        "ref": {"x": 0.0, "y": 2.0, "z": 0.0},
    }
    expected_origin = round(plane_y_at(plane, 0.0, 0.0), 6)
    expected_far = round(plane_y_at(plane, 2.0, 0.0), 6)

    assert viewer_surface_y_at(surface, 0.0, 0.0) == expected_origin
    assert viewer_surface_y_at(surface, 2.0, 0.0) == expected_far
    assert viewer_surface_y_at(surface, 0.0, 0.0) != viewer_surface_y_at(
        surface, 2.0, 0.0
    )
    assert cell_complex_surface_y_at(surface, 0.0, 0.0) == pytest.approx(
        expected_origin, abs=1e-3
    )
    assert cell_complex_surface_y_at(surface, 2.0, 0.0) == pytest.approx(
        expected_far, abs=1e-3
    )


def test_partition_height_model_prefers_cluster_plane_over_corners() -> None:
    surface = {
        "kind": "oblique",
        "roof_hypothesis_id": "roof-hypothesis:oblique:1",
        "cluster": {
            "avgAzimuth": 300.0,
            "avgIncl": 30.0,
            "refPt": {"x": 1.0, "y": 3.0, "z": -1.0},
        },
        "corners": [
            [0.0, 12.0, 0.0],
            [2.0, 12.0, 0.0],
            [2.0, 12.0, 2.0],
            [0.0, 12.0, 2.0],
        ],
    }
    model = _height_model("oblique", surface, {"wallTopY": 12.0, "wallTopMin": 12.0})
    plane = {
        "n": plane_normal(300.0, 30.0),
        "ref": {"x": 1.0, "y": 3.0, "z": -1.0},
    }

    for x, z in [(0.0, 0.0), (2.0, 1.0), (-1.0, 3.0)]:
        assert _height_at(model, x, z) == pytest.approx(
            round(plane_y_at(plane, x, z), 3), abs=2e-3
        )


def test_abstract_decision_layer_matches_graph_semantics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        rooms = graph.nodes_by_type("Room")
        room = rooms[0]
        roof_decision = classify_roof_room(graph, room)
        relation_state = classify_room_relation_state(graph, room)
        overlap_decision = classify_overlap_resolution(graph, room)

        assert roof_decision["mode"] in {
            "roof_candidate",
            "ceiling_below_occupied_volume",
            "underdetermined",
        }
        assert relation_state["above_state"] in {
            "confirmed",
            "partial",
            "unknown",
            "weak",
        }
        assert relation_state["exposure_state"] in {
            "confirmed",
            "partial",
            "unknown",
            "weak",
        }
        assert overlap_decision["ownership_priority"] in {"preserve", "secondary"}

        connect_edge = next(
            (edge for edge in graph.edges if edge.type == "CONNECTS_TO"), None
        )
        assert connect_edge is not None
        left = graph.get_node(connect_edge.from_id)
        right = graph.get_node(connect_edge.to_id)
        stitch_decision = classify_stitch_decision(
            graph, left, right, connect_edge.evidence or {}
        )
        assert stitch_decision["action"] in {"stitch", "inspect"}

        gap = next((node for node in graph.nodes_by_type("Gap")), None)
        if gap is not None:
            gap_decision = classify_gap_decision(graph, gap)
            assert gap_decision["action"] in {"close", "inspect"}

        snapshot = build_decision_snapshot(graph)
        assert "gaps" in snapshot
        assert "roof_rooms" in snapshot
        assert "relation_states" in snapshot
        assert "overlap_resolution" in snapshot


def test_partial_relation_state_annotations_exist_on_graph_edges() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        relation_edges = [
            edge
            for edge in graph.edges
            if edge.type in {"ADJACENT_TO", "EXPOSES_TO", "ABOVE", "BELOW"}
        ]
        assert relation_edges
        assert all(
            (edge.evidence or {}).get("relation_state")
            in {"confirmed", "partial", "weak"}
            for edge in relation_edges
        )


def test_exact_lattice_kernel_and_interference_edges() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_overlapping_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        cell_complex = graph.geometry_index["cell_complex"]
        assert cell_complex["metadata"]["topology_exact_on_lattice"] is True
        assert cell_complex["metadata"]["numeric_model"] == "fixed_point_mm"
        assert (
            cell_complex["metadata"]["backend"]
            == "exact_lattice_polyhedral_arrangement_v3"
        )
        room_cell = next(
            cell for cell in cell_complex["cells"] if cell.get("kind") == "room"
        )
        assert any(
            face.get("vertices_lattice") for face in room_cell.get("faces") or []
        )
        assert any(face.get("role") == "wall" for face in room_cell.get("faces") or [])
        assert any(edge.type == "INTERFERES" for edge in graph.edges)


def test_enclosed_void_cells_are_promoted_to_gap_nodes_and_closure_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_enclosed_void_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        derived_gaps = [
            gap
            for gap in graph.nodes_by_type("Gap")
            if bool((gap.properties or {}).get("derived_enclosed_void"))
        ]
        assert derived_gaps

        gap = derived_gaps[0]
        bounded_rooms = [
            node
            for node in graph.neighbors(gap.id, "BOUNDED_BY")
            if node.type == "Room"
        ]
        assert bounded_rooms
        assert any(
            edge.type == "HAS_GAP" and edge.to_id == gap.id for edge in graph.edges
        )
        decision = classify_gap_decision(graph, gap)
        assert decision["action"] == "close"
        assert decision["policy"] == "internal_enclosed_void_repair"


def test_gap_walls_carry_ontology_gap_ids_for_enclosed_voids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_enclosed_void_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        rooms_out = [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 3.0],
                    [0.0, 0.0, 3.0],
                ],
                "walls_computed": [
                    {
                        "id": "w0",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 2.4, 0.0],
                            [0.0, 2.4, 0.0],
                        ],
                    }
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 3.0],
                    [2.0, 0.0, 3.0],
                ],
                "walls_computed": [
                    {
                        "id": "w1",
                        "corners": [
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0],
                            [3.0, 2.4, 0.0],
                            [2.0, 2.4, 0.0],
                        ],
                    }
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ],
                "walls_computed": [
                    {
                        "id": "w2",
                        "corners": [
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 2.4, 0.0],
                            [1.0, 2.4, 0.0],
                        ],
                    }
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [1.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 3.0],
                    [1.0, 0.0, 3.0],
                ],
                "walls_computed": [
                    {
                        "id": "w3",
                        "corners": [
                            [1.0, 0.0, 2.0],
                            [2.0, 0.0, 2.0],
                            [2.0, 2.4, 2.0],
                            [1.0, 2.4, 2.0],
                        ],
                    }
                ],
            },
        ]
        gaps = [
            {
                "story": 0,
                "type": "within_story",
                "corners": [
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                ],
                "confidence": "high",
                "centroid": [1.5, 0.0, 1.5],
            }
        ]
        walls = compute_gap_walls(gaps, rooms_out, story_y_map={0: 0.0}, graph=graph)

        attributed = [
            wall
            for wall in walls
            if isinstance(wall.get("ontology_gap_id"), str)
            and wall["ontology_gap_id"].startswith("gap:")
        ]
        assert attributed
        assert any(str(wall.get("id", "")).startswith("gw:gap:") for wall in attributed)


def test_assign_gaps_to_rooms_closes_ontology_approved_voids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_enclosed_void_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        rooms_out = [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 3.0],
                    [0.0, 0.0, 3.0],
                ],
                "walls_computed": [],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 3.0],
                    [2.0, 0.0, 3.0],
                ],
                "walls_computed": [],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ],
                "walls_computed": [],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [1.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 3.0],
                    [1.0, 0.0, 3.0],
                ],
                "walls_computed": [],
            },
        ]
        gaps = [
            {
                "story": 0,
                "type": "within_story",
                "corners": [
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                ],
                "confidence": "high",
                "centroid": [1.5, 0.0, 1.5],
            }
        ]

        assign_gaps_to_rooms(gaps, rooms_out, graph=graph)

        assert any(gap.get("room_index") is not None for gap in gaps)
        void_point = Point(1.5, 1.5)
        closed = False
        for room in rooms_out:
            poly = floor_polygon_to_shapely(room["floor_polygon"])
            if poly is None:
                continue
            if poly.buffer(1e-6).contains(void_point):
                closed = True
                break
        assert closed


def test_overlap_clipping_can_consume_graph_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path, scan_dir, uuid = _make_overlapping_fixture(Path(tmp))
        graph = build_topology_graph(
            merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
        )

        rooms_out = [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                    [2.5, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "walls_computed": [
                    {
                        "id": "wa",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [2.5, 0.0, 0.0],
                            [2.5, 2.4, 0.0],
                            [0.0, 2.4, 0.0],
                        ],
                    }
                ],
                "doors": [],
                "windows": [],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [1.5, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [1.5, 0.0, 2.0],
                ],
                "walls_computed": [
                    {
                        "id": "wb",
                        "corners": [
                            [1.5, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [4.0, 2.4, 0.0],
                            [1.5, 2.4, 0.0],
                        ],
                    }
                ],
                "doors": [],
                "windows": [],
            },
        ]
        metrics = clip_floor_overlaps(copy.deepcopy(rooms_out), graph=graph)
        assert metrics
        assert any(metric["overlap_area_m2"] > 0 for metric in metrics)


def test_viewer_ontology_summary_assigns_unassigned_atoms_and_subparts() -> None:
    roof = {
        "ceiling_partitions": {
            "room_partitions": [
                {
                    "room_index": 0,
                    "story": 0,
                    "graph_room_id": "room:merged_room_0",
                    "partitions": [
                        {
                            "id": "atom:0",
                            "poly": [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 2.0]],
                            "top_y_m": 2.0,
                            "supporting_roof_hypothesis_ids": [
                                "roof-hypothesis:oblique:0"
                            ],
                            "flat_role_reason": "exact_attic_cell_above_flat_atom",
                        }
                    ],
                },
                {
                    "room_index": 1,
                    "story": 0,
                    "graph_room_id": "room:merged_room_1",
                    "partitions": [
                        {
                            "id": "atom:1",
                            "poly": [[3.0, 2.5, 0.0], [5.0, 2.5, 0.0], [5.0, 2.5, 2.0]],
                            "top_y_m": 2.5,
                            "supporting_roof_hypothesis_ids": [
                                "roof-hypothesis:oblique:1"
                            ],
                            "flat_role_reason": (
                                "flat_atom_covered_by_sloped_roof_without_exact_upper_cell"
                            ),
                        }
                    ],
                },
            ]
        },
        "building_part_graph": {
            "nodes": [
                {
                    "id": "building-part:0",
                    "type": "BuildingPart",
                    "room_ids": ["room:0"],
                    "roof_family_guess": "gable_or_multi_slope",
                }
            ],
            "room_membership": {"room:0": ["building-part:0"]},
        },
        "roof_coverage_graph": {
            "subparts": [
                {
                    "id": "subpart:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                    "room_indices": [1],
                    "polygon_xz": [[3.0, 0.0], [5.0, 0.0], [5.0, 2.0]],
                    "semantic_kind": "slope_run",
                }
            ],
            "atom_subpart_membership": {"atom:1": ["subpart:0"]},
            "metadata": {},
        },
        "top_boundary_graph": {
            "nodes": [
                {
                    "id": "atom:0",
                    "type": "TopBoundaryAtom",
                    "room_id": "room:0",
                    "room_index": 0,
                    "part_id": "building-part:0",
                    "role": "attic_floor",
                    "kind": "flat",
                },
                {
                    "id": "atom:1",
                    "type": "TopBoundaryAtom",
                    "room_id": "room:1",
                    "room_index": 1,
                    "part_id": None,
                    "role": "flat_transition_cap_candidate",
                    "kind": "flat",
                },
            ],
            "room_summaries": {
                "room:0": {"has_resolved_roof_relation": True},
                "room:1": {
                    "room_index": 1,
                    "story": 0,
                    "part_ids": [],
                    "partially_covered_by_sloped_roof": True,
                    "has_candidate_attic_relation": True,
                    "has_candidate_upper_void_relation": False,
                    "roof_evidence_score": 6,
                    "has_resolved_roof_relation": False,
                },
            },
            "metadata": {},
        },
        "roof_surfaces": {
            "oblique": [
                {
                    "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                    "kind": "oblique",
                    "corners": [
                        [3.0, 3.0, 0.0],
                        [5.0, 3.5, 0.0],
                        [5.0, 3.5, 2.0],
                        [3.0, 3.0, 2.0],
                    ],
                }
            ]
        },
        "roof_evidence_graph": {"metadata": {}},
        "roof_continuation_diagnostics": {
            "continuation_regions": [
                {
                    "id": "continuation-region:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                    "room_id": "room:1",
                    "room_index": 1,
                    "continuation_mode": "arrangement_face",
                    "polygon": [
                        [3.0, 3.0, 0.0],
                        [5.0, 3.5, 0.0],
                        [5.0, 3.5, 2.0],
                        [3.0, 3.0, 2.0],
                    ],
                    "polygon_xz": [[3.0, 0.0], [5.0, 0.0], [5.0, 2.0], [3.0, 2.0]],
                    "exact_incidence_pair_count": 1,
                }
            ]
        },
        "roof_cell_complex": {
            "cells": [{"id": "roof-cell:0", "part_id": None, "room_id": "room:1"}],
            "knee_walls": [{"id": "knee:0", "part_id": None, "room_index": 1}],
        },
        "dormers": [],
    }
    topology_cell_complex = {
        "cells": [
            {"id": "cell:0", "kind": "room", "source_id": "room:merged_room_0"},
            {"id": "cell:1", "kind": "room", "source_id": "room:merged_room_1"},
        ]
    }

    summary, part_graph_room_ids = _build_ontology_summary(
        uuid="test-uuid",
        roof=roof,
        topology_cell_complex=topology_cell_complex,
    )

    assert any(part["id"] == "building-part:0" for part in summary["building_parts"])
    assert any(part["id"] == UNASSIGNED_PART_ID for part in summary["building_parts"])
    atom_by_id = {atom["id"]: atom for atom in summary["semantic_atoms"]}
    assert atom_by_id["atom:0"]["effective_part_id"] == "building-part:0"
    assert atom_by_id["atom:1"]["effective_part_id"] == UNASSIGNED_PART_ID
    assert summary["coverage_subparts"][0]["effective_part_ids"] == [UNASSIGNED_PART_ID]
    assert len(summary["oblique_coverage_patches"]) == 1
    assert summary["oblique_coverage_patches"][0]["effective_part_ids"] == [
        UNASSIGNED_PART_ID
    ]
    assert summary["oblique_coverage_patches"][0]["room_indices"] == [1]
    assert summary["oblique_coverage_patches"][0]["room_ids"] == ["room:1"]
    assert summary["oblique_coverage_patches"][0]["coverage_subpart_id"] == "subpart:0"
    assert "subpart:0" in summary["oblique_coverage_patches"][0]["id"]
    assert len(summary["roof_continuation_diagnostics"]) == 1
    assert summary["roof_continuation_diagnostics"][0]["effective_part_ids"] == [
        UNASSIGNED_PART_ID
    ]
    assert (
        summary["roof_continuation_diagnostics"][0]["continuation_mode"]
        == "arrangement_face"
    )
    assert len(summary["unresolved_regions"]) == 1
    assert summary["unresolved_regions"][0]["room_id"] == "room:1"
    assert [surface["category"] for surface in summary["renderable_surfaces"]] == [
        "attic_floor",
        "unresolved_region",
    ]
    assert summary["metadata"]["renderable_surface_counts"] == {
        "attic_floor": 1,
        "unresolved_region": 1,
    }
    assert summary["metadata"]["roof_continuation_region_count"] == 1
    assert part_graph_room_ids["building-part:0"] == {"room:merged_room_0"}
    assert part_graph_room_ids[UNASSIGNED_PART_ID] == {"room:merged_room_1"}


def test_viewer_ontology_part_payloads_are_sliced_per_part() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": "building-part:0",
                "room_ids": ["room:0", "room:2"],
                "room_indices": [0, 2],
            },
            {"id": UNASSIGNED_PART_ID, "room_ids": ["room:1"], "room_indices": [1]},
        ],
        "dormers": [
            {"id": "dormer:0", "room_index": 0},
            {"id": "dormer:1", "room_index": 1},
        ],
    }
    part_graph_room_ids = {
        "building-part:0": {"room:merged_room_0"},
        UNASSIGNED_PART_ID: {"room:merged_room_1"},
    }
    topology_cell_complex = {
        "cells": [
            {
                "id": "cell:0",
                "kind": "room",
                "source_id": "room:merged_room_0",
                "story": 0,
                "properties": {
                    "xz_footprint": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
                },
                "faces": [
                    {
                        "id": "face:bottom:0",
                        "role": "slab",
                        "boundary_kind": "bottom",
                        "vertices": [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 0.0, 2.0],
                            [0.0, 0.0, 2.0],
                        ],
                        "metadata": {"face_kind": "bottom"},
                    },
                    {
                        "id": "face:top:0",
                        "role": "slab",
                        "boundary_kind": "top",
                        "vertices": [
                            [0.0, 2.4, 0.0],
                            [2.0, 2.4, 0.0],
                            [2.0, 2.4, 2.0],
                            [0.0, 2.4, 2.0],
                        ],
                        "metadata": {"face_kind": "top"},
                    },
                    {
                        "id": "face:wall:0",
                        "role": "wall",
                        "boundary_kind": "side",
                        "vertices": [
                            [0.0, 0.0, 0.0],
                            [0.0, 2.4, 0.0],
                            [0.0, 2.4, 2.0],
                            [0.0, 0.0, 2.0],
                        ],
                        "metadata": {"face_kind": "side", "perimeter_facing": True},
                    },
                ],
            },
            {
                "id": "cell:1",
                "kind": "room",
                "source_id": "room:merged_room_1",
                "story": 0,
                "properties": {
                    "xz_footprint": [[2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0]]
                },
                "faces": [
                    {
                        "id": "face:bottom:1",
                        "role": "slab",
                        "boundary_kind": "bottom",
                        "vertices": [
                            [2.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [4.0, 0.0, 2.0],
                            [2.0, 0.0, 2.0],
                        ],
                        "metadata": {"face_kind": "bottom"},
                    },
                    {
                        "id": "face:top:1",
                        "role": "slab",
                        "boundary_kind": "top",
                        "vertices": [
                            [2.0, 2.4, 0.0],
                            [4.0, 2.4, 0.0],
                            [4.0, 2.4, 2.0],
                            [2.0, 2.4, 2.0],
                        ],
                        "metadata": {"face_kind": "top"},
                    },
                    {
                        "id": "face:wall:1",
                        "role": "wall",
                        "boundary_kind": "side",
                        "vertices": [
                            [4.0, 0.0, 0.0],
                            [4.0, 2.4, 0.0],
                            [4.0, 2.4, 2.0],
                            [4.0, 0.0, 2.0],
                        ],
                        "metadata": {"face_kind": "side", "perimeter_facing": True},
                    },
                ],
            },
            {
                "id": "cell:2",
                "kind": "room",
                "source_id": "room:merged_room_2",
                "story": 0,
                "properties": {
                    "xz_footprint": [[4.0, 0.0], [6.0, 0.0], [6.0, 2.0], [4.0, 2.0]]
                },
                "faces": [
                    {
                        "id": "face:bottom:2",
                        "role": "slab",
                        "boundary_kind": "bottom",
                        "vertices": [
                            [4.0, 0.0, 0.0],
                            [6.0, 0.0, 0.0],
                            [6.0, 0.0, 2.0],
                            [4.0, 0.0, 2.0],
                        ],
                        "metadata": {"face_kind": "bottom"},
                    },
                    {
                        "id": "face:top:2",
                        "role": "slab",
                        "boundary_kind": "top",
                        "vertices": [
                            [4.0, 2.4, 0.0],
                            [6.0, 2.4, 0.0],
                            [6.0, 2.4, 2.0],
                            [4.0, 2.4, 2.0],
                        ],
                        "metadata": {"face_kind": "top"},
                    },
                    {
                        "id": "face:wall:2",
                        "role": "wall",
                        "boundary_kind": "side",
                        "vertices": [
                            [6.0, 0.0, 0.0],
                            [6.0, 2.4, 0.0],
                            [6.0, 2.4, 2.0],
                            [6.0, 0.0, 2.0],
                        ],
                        "metadata": {"face_kind": "side", "perimeter_facing": True},
                    },
                ],
            },
            {"id": "cell:gap", "kind": "gap", "source_id": "gap:0"},
        ]
    }
    roof_cell_complex = {
        "cells": [
            {
                "id": "roof:0",
                "part_id": "building-part:0",
                "room_id": "room:0",
                "cell_kind": "attic",
                "faces": [
                    {
                        "id": "f:roof:0",
                        "role": "roof",
                        "corners": [[0.0, 1.0, 0.0], [1.0, 1.5, 0.0], [0.0, 1.5, 1.0]],
                    },
                    {
                        "id": "f:wall:0",
                        "role": "wall",
                        "corners": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.5, 1.0]],
                        "metadata": {"perimeter_facing": True},
                    },
                ],
            },
            {
                "id": "roof:1",
                "part_id": None,
                "room_id": "room:1",
                "cell_kind": "upper_void",
                "faces": [
                    {
                        "id": "f:roof:1",
                        "role": "roof",
                        "corners": [[2.0, 1.0, 0.0], [3.0, 1.2, 0.0], [2.0, 1.2, 1.0]],
                    },
                    {
                        "id": "f:int:1",
                        "role": "wall",
                        "corners": [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.2, 1.0]],
                        "metadata": {"perimeter_facing": False},
                    },
                ],
            },
        ],
        "knee_walls": [
            {
                "id": "knee:0",
                "part_id": "building-part:0",
                "room_index": 0,
                "corners": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            },
            {
                "id": "knee:1",
                "part_id": None,
                "room_index": 1,
                "corners": [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
            },
        ],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids=part_graph_room_ids,
        topology_cell_complex=topology_cell_complex,
        roof_cell_complex=roof_cell_complex,
    )

    assert [cell["id"] for cell in payloads["building-part:0"]["roof_cells"]] == [
        "roof:0"
    ]
    assert [
        face["id"] for face in payloads["building-part:0"]["roof_cells"][0]["faces"]
    ] == ["f:roof:0", "f:wall:0"]
    assert [wall["id"] for wall in payloads["building-part:0"]["knee_walls"]] == [
        "knee:0"
    ]
    assert [
        surface["category"]
        for surface in payloads["building-part:0"]["renderable_surfaces"]
    ] == [
        "exterior_roof",
        "knee_wall",
        "occupied_room_floor",
        "exterior_wall",
        "occupied_room_floor",
        "occupied_room_ceiling",
        "exterior_wall",
    ]
    assert payloads["building-part:0"]["metadata"]["renderable_surface_counts"] == {
        "exterior_roof": 1,
        "exterior_wall": 2,
        "knee_wall": 1,
        "occupied_room_ceiling": 1,
        "occupied_room_floor": 2,
    }
    assert [d["id"] for d in payloads["building-part:0"]["dormers"]] == ["dormer:0"]

    assert [cell["id"] for cell in payloads[UNASSIGNED_PART_ID]["roof_cells"]] == [
        "roof:1"
    ]
    assert [
        face["id"] for face in payloads[UNASSIGNED_PART_ID]["roof_cells"][0]["faces"]
    ] == ["f:roof:1"]
    assert [wall["id"] for wall in payloads[UNASSIGNED_PART_ID]["knee_walls"]] == [
        "knee:1"
    ]
    assert [
        surface["category"]
        for surface in payloads[UNASSIGNED_PART_ID]["renderable_surfaces"]
    ] == [
        "exterior_roof",
        "knee_wall",
        "occupied_room_floor",
        "exterior_wall",
    ]
    assert payloads[UNASSIGNED_PART_ID]["metadata"]["renderable_surface_counts"] == {
        "exterior_roof": 1,
        "exterior_wall": 1,
        "knee_wall": 1,
        "occupied_room_floor": 1,
    }
    assert [d["id"] for d in payloads[UNASSIGNED_PART_ID]["dormers"]] == ["dormer:1"]


def test_viewer_ontology_summary_falls_back_to_room_ids_for_part_room_indices() -> None:
    roof = {
        "building_part_graph": {
            "nodes": [
                {
                    "id": "building-part:0",
                    "type": "BuildingPart",
                    "room_ids": ["room:4"],
                    "hypothesis_ids": [],
                    "oblique_hypothesis_ids": [],
                    "flat_hypothesis_ids": [],
                }
            ],
            "room_membership": {"room:4": ["building-part:0"]},
        },
        "roof_coverage_graph": {
            "subparts": [],
            "atom_subpart_membership": {},
            "metadata": {},
        },
        "top_boundary_graph": {"room_summaries": {}, "metadata": {}},
        "roof_evidence_graph": {"metadata": {}},
        "roof_cell_complex": {"cells": [], "knee_walls": []},
        "roof_surfaces": {"oblique": [], "flat": []},
        "ceiling_partitions": {"room_partitions": []},
        "dormers": [],
    }

    summary, part_graph_room_ids = _build_ontology_summary(
        uuid="test-uuid",
        roof=roof,
        topology_cell_complex={"cells": []},
    )

    assert part_graph_room_ids["building-part:0"] == set()
    part = next(
        part for part in summary["building_parts"] if part["id"] == "building-part:0"
    )
    assert part["room_indices"] == [4]


def test_viewer_ontology_part_payloads_emit_clipped_base_room_shell() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {"id": "building-part:0", "room_ids": ["room:0"], "room_indices": [0]},
        ],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "windows": [
                    {
                        "id": "window:0",
                        "corners": [
                            [0.5, 1.0, 0.0],
                            [1.2, 1.0, 0.0],
                            [1.2, 1.8, 0.0],
                            [0.5, 1.8, 0.0],
                        ],
                    }
                ],
                "doors": [],
                "openings": [],
                "walls_computed": [
                    {
                        "id": "wall:0",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 3.0, 0.0],
                            [0.0, 3.0, 0.0],
                        ],
                        "extension_strip": None,
                    }
                ],
            }
        ]
    }
    roof = {
        "ceiling_partitions": {
            "room_partitions": [
                {
                    "room_index": 0,
                    "story": 0,
                    "partitions": [
                        {
                            "id": "atom:0",
                            "kind": "oblique",
                            "poly": [
                                [0.0, 2.0, 0.0],
                                [2.0, 3.0, 0.0],
                                [2.0, 3.0, 2.0],
                                [0.0, 2.0, 2.0],
                            ],
                            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                        }
                    ],
                }
            ]
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={"building-part:0": set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        building=building,
        roof=roof,
    )

    surfaces = payloads["building-part:0"]["renderable_surfaces"]
    categories = [surface["category"] for surface in surfaces]
    assert categories == ["base_room_floor", "base_window", "base_exterior_wall"]
    assert "occupied_room_floor" not in categories
    wall_surface = next(
        surface for surface in surfaces if surface["category"] == "base_exterior_wall"
    )
    top_ys = sorted(corner[1] for corner in wall_surface["corners"])[-2:]
    assert top_ys == [2.0, 3.0]
    assert len(wall_surface["holes"]) == 1
    assert payloads["building-part:0"]["metadata"]["renderable_surface_counts"] == {
        "base_room_floor": 1,
        "base_window": 1,
        "base_exterior_wall": 1,
    }


def test_viewer_ontology_part_payloads_segment_wall_by_split_atoms() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {"id": "building-part:0", "room_ids": ["room:0"], "room_indices": [0]},
        ],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "walls_computed": [
                    {
                        "id": "wall:0",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [4.0, 3.0, 0.0],
                            [0.0, 3.0, 0.0],
                        ],
                        "extension_strip": None,
                    }
                ],
            }
        ]
    }
    roof = {
        "ceiling_partitions": {
            "room_partitions": [
                {
                    "room_index": 0,
                    "story": 0,
                    "partitions": [
                        {
                            "id": "atom:left",
                            "kind": "flat",
                            "poly": [
                                [0.0, 2.0, 0.0],
                                [2.0, 2.0, 0.0],
                                [2.0, 2.0, 2.0],
                                [0.0, 2.0, 2.0],
                            ],
                            "roof_hypothesis_id": "roof-hypothesis:flat:0",
                        },
                        {
                            "id": "atom:right",
                            "kind": "oblique",
                            "poly": [
                                [2.0, 2.0, 0.0],
                                [4.0, 3.0, 0.0],
                                [4.0, 3.0, 2.0],
                                [2.0, 2.0, 2.0],
                            ],
                            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                        },
                    ],
                }
            ]
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={"building-part:0": set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        building=building,
        roof=roof,
    )

    wall_surfaces = [
        surface
        for surface in payloads["building-part:0"]["renderable_surfaces"]
        if surface["category"] == "base_exterior_wall"
    ]
    assert len(wall_surfaces) == 2
    top_profiles = sorted(
        sorted(corner[1] for corner in surface["corners"])[-2:]
        for surface in wall_surfaces
    )
    assert top_profiles[0] == pytest.approx([2.0, 2.0], abs=1e-5)
    assert top_profiles[1] == pytest.approx([2.0, 3.0], abs=1e-5)


def test_viewer_ontology_part_payloads_prefer_exact_occupied_room_cells() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
            {"id": "building-part:0", "room_ids": ["room:0"], "room_indices": [0]},
        ],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "windows": [
                    {
                        "id": "window:0",
                        "corners": [
                            [2.5, 1.0, 0.0],
                            [3.2, 1.0, 0.0],
                            [3.2, 1.8, 0.0],
                            [2.5, 1.8, 0.0],
                        ],
                    }
                ],
                "doors": [],
                "openings": [],
                "walls_computed": [
                    {
                        "id": "wall:0",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [4.0, 3.0, 0.0],
                            [0.0, 3.0, 0.0],
                        ],
                    }
                ],
            }
        ]
    }
    roof = {
        "ceiling_partitions": {
            "room_partitions": [
                {
                    "room_index": 0,
                    "story": 0,
                    "partitions": [
                        {
                            "id": "atom:left",
                            "kind": "flat",
                            "poly": [
                                [0.0, 2.0, 0.0],
                                [2.0, 2.0, 0.0],
                                [2.0, 2.0, 2.0],
                                [0.0, 2.0, 2.0],
                            ],
                        },
                        {
                            "id": "atom:right",
                            "kind": "oblique",
                            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                            "poly": [
                                [2.0, 2.0, 0.0],
                                [4.0, 3.0, 0.0],
                                [4.0, 3.0, 2.0],
                                [2.0, 2.0, 2.0],
                            ],
                        },
                    ],
                }
            ]
        }
    }
    roof["occupied_room_cell_complex"] = build_occupied_room_cell_complex(
        bldg=building,
        room_partitions=roof["ceiling_partitions"]["room_partitions"],
        building_part_graph={"room_membership": {"room:0": ["building-part:0"]}},
    )

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={"building-part:0": set(), FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex=roof["occupied_room_cell_complex"],
        building=building,
        roof=roof,
    )

    surfaces = payloads["building-part:0"]["renderable_surfaces"]
    wall_surfaces = [
        surface for surface in surfaces if surface["category"] == "base_exterior_wall"
    ]
    floor_surfaces = [
        surface for surface in surfaces if surface["category"] == "base_room_floor"
    ]
    ceiling_surfaces = [
        surface for surface in surfaces if surface["category"] == "base_room_ceiling"
    ]
    window_surfaces = [
        surface for surface in surfaces if surface["category"] == "base_window"
    ]
    front_wall_surfaces = [
        surface
        for surface in wall_surfaces
        if all(abs(corner[2]) <= 1e-6 for corner in surface["corners"])
    ]
    assert len(front_wall_surfaces) == 2
    assert len(floor_surfaces) == 2
    assert len(ceiling_surfaces) == 2
    assert len(window_surfaces) == 1
    assert all(
        surface["source_kind"] == "occupied_room_cell_face"
        for surface in wall_surfaces + floor_surfaces + ceiling_surfaces
    )
    top_profiles = sorted(
        sorted(corner[1] for corner in surface["corners"])[-2:]
        for surface in front_wall_surfaces
    )
    assert top_profiles[0] == pytest.approx([2.0, 2.0], abs=1e-5)
    assert top_profiles[1] == pytest.approx([2.0, 3.0], abs=1e-5)
    ceiling_profiles = sorted(
        sorted(corner[1] for corner in surface["corners"])
        for surface in ceiling_surfaces
    )
    assert ceiling_profiles[0] == pytest.approx([2.0, 2.0, 2.0, 2.0], abs=1e-5)
    assert ceiling_profiles[1] == pytest.approx([2.0, 2.0, 3.0, 3.0], abs=1e-5)
    assert sum(len(surface.get("holes") or []) for surface in wall_surfaces) == 1
    assert window_surfaces[0]["source_kind"] == "window"


def test_viewer_ontology_part_payloads_promote_fallbacks_unres():
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:0": {
                "room_index": 0,
                "story": 0,
                "partially_covered_by_sloped_roof": True,
                "roof_evidence_score": 5,
                "has_oblique_atom": True,
            }
        },
        "unresolved_regions": [],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "windows": [],
                "doors": [],
                "openings": [],
                "walls_computed": [],
            }
        ]
    }
    roof = {
        "roof_surfaces": {
            "flat": [],
            "oblique": [
                {
                    "story": 0,
                    "room_index": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.8, 0.0],
                        [4.0, 3.8, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "oblique",
                    "boundary_face_id": "face:roof-oblique:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building=building,
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    categories = [surface["category"] for surface in payload["renderable_surfaces"]]
    assert "exterior_roof" in categories
    assert "unresolved_region" in categories
    roof_surface = next(
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    )
    unresolved = next(
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "unresolved_region"
    )
    assert roof_surface["source_kind"] == "roof_surface_fallback"
    assert roof_surface["room_index"] == 0
    assert unresolved["room_index"] == 0
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 0
    assert payload["metadata"]["roof_fallback_surface_count"] == 1
    assert payload["metadata"]["unresolved_region_count"] == 1


def test_drop_roomless_flat_resolved() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:0": {
                "room_index": 0,
                "story": 0,
                "has_resolved_roof_relation": True,
                "partially_covered_by_sloped_roof": False,
                "roof_evidence_score": 0,
            }
        },
        "semantic_atoms": [],
        "unresolved_regions": [],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "windows": [],
                "doors": [],
                "openings": [],
                "walls_computed": [],
            }
        ]
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 0,
                    "room_index": None,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:roof-flat:0",
                    "roof_hypothesis_id": "roof-hypothesis:flat:0",
                    "flat_role": "roof_flat",
                }
            ],
            "oblique": [],
        },
        "roof_hypothesis_graph": {
            "selected_hypothesis_ids": ["roof-hypothesis:flat:0"],
            "edges": [
                {
                    "type": "COVERS_ROOM",
                    "from": "roof-hypothesis:flat:0",
                    "to": "room:0",
                    "selected": True,
                }
            ],
        },
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building=building,
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    categories = [surface["category"] for surface in payload["renderable_surfaces"]]
    assert "unresolved_region" not in categories
    assert "exterior_roof" not in categories
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_drop_roomless_flat_unresolved_room() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:0": {
                "room_index": 0,
                "story": 0,
                "has_resolved_roof_relation": False,
                "partially_covered_by_sloped_roof": False,
                "roof_evidence_score": 0,
            }
        },
        "semantic_atoms": [],
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 0,
                    "room_index": None,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:roof-flat:1",
                    "roof_hypothesis_id": "roof-hypothesis:flat:1",
                    "flat_role": "roof_flat",
                }
            ],
            "oblique": [],
        },
        "roof_hypothesis_graph": {
            "selected_hypothesis_ids": ["roof-hypothesis:flat:1"],
            "edges": [
                {
                    "type": "COVERS_ROOM",
                    "from": "roof-hypothesis:flat:1",
                    "to": "room:0",
                    "selected": True,
                }
            ],
        },
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [],
                    "walls_computed": [],
                    "windows": [],
                    "doors": [],
                    "openings": [],
                }
            ]
        },
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    categories = [surface["category"] for surface in payload["renderable_surfaces"]]
    assert "unresolved_region" not in categories
    assert "exterior_roof" not in categories
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_viewer_ontology_part_payloads_include_summary_unresolved_regions() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {"id": "building-part:0", "room_ids": ["room:0"], "room_indices": [0]},
        ],
        "unresolved_regions": [
            {
                "id": "unresolved-coverage:room:0",
                "room_id": "room:0",
                "room_index": 0,
                "story": 0,
                "effective_part_ids": ["building-part:0"],
                "polygon": [
                    [0.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [0.0, 2.0, 2.0],
                ],
                "polygon_xz": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                "roof_evidence_score": 5,
            }
        ],
        "dormers": [],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={"building-part:0": set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [],
                    "walls_computed": [],
                    "windows": [],
                    "doors": [],
                    "openings": [],
                }
            ]
        },
        roof={"roof_surfaces": {"flat": [], "oblique": []}},
    )

    payload = payloads["building-part:0"]
    unresolved_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "unresolved_region"
    ]
    assert len(unresolved_surfaces) == 1
    assert payload["unresolved_regions"][0]["room_id"] == "room:0"
    assert payload["metadata"]["unresolved_region_count"] == 1


def test_renderable_surface_from_atom_includes_flat_ceiling_role() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [
            {
                "id": "atom:flat",
                "type": "TopBoundaryAtom",
                "room_id": "room:0",
                "room_index": 0,
                "story": 0,
                "effective_part_id": FULL_BUILDING_PART_ID,
                "role": "flat_ceiling",
                "kind": "flat",
                "poly": [
                    [0.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 2.0],
                    [0.0, 2.0, 2.0],
                ],
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "top_y_m": 2.0,
            }
        ],
        "unresolved_regions": [],
        "dormers": [],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [],
                    "walls_computed": [],
                    "windows": [],
                    "doors": [],
                    "openings": [],
                }
            ]
        },
        roof={"roof_surfaces": {"flat": [], "oblique": []}},
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    semantic_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "room_ceiling_flat"
    ]
    assert len(semantic_surfaces) == 1
    assert semantic_surfaces[0]["source_kind"] == "semantic_atom"
    assert semantic_surfaces[0]["source_id"] == "atom:flat"


def test_demote_synth_to_unres() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "windows": [],
                "doors": [],
                "openings": [],
                "walls_computed": [
                    {
                        "id": "wall:0",
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [4.0, 2.4, 0.0],
                            [0.0, 2.4, 0.0],
                        ],
                    }
                ],
            }
        ]
    }
    roof = {
        "ceiling_partitions": {"room_partitions": []},
        "roof_surfaces": {"flat": [], "oblique": []},
    }
    occupied_room_cell_complex = build_occupied_room_cell_complex(
        bldg=building,
        room_partitions=[],
        building_part_graph={"room_membership": {"room:0": [FULL_BUILDING_PART_ID]}},
    )

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex=occupied_room_cell_complex,
        building=building,
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    categories = [surface["category"] for surface in payload["renderable_surfaces"]]
    assert "fallback_room_ceiling" not in categories
    assert categories.count("unresolved_region") == 1
    unresolved = next(
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "unresolved_region"
    )
    assert unresolved["room_id"] == "room:0"
    assert unresolved["source_kind"] == "unresolved_region"
    assert payload["metadata"]["unresolved_region_count"] == 1


def test_promote_exact_flat_shell() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:0": {
                "room_index": 0,
                "story": 0,
                "partially_covered_by_sloped_roof": False,
                "strong_perimeter_sloped": False,
                "strong_knee_wall_signal": False,
                "has_candidate_attic_relation": False,
                "has_candidate_upper_void_relation": False,
                "has_oblique_atom": False,
                "roof_evidence_score": 0,
            }
        },
        "unresolved_regions": [],
        "dormers": [],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [],
                    "walls_computed": [],
                    "windows": [],
                    "doors": [],
                    "openings": [],
                }
            ]
        },
        roof={
            "roof_surfaces": {
                "flat": [
                    {
                        "story": 0,
                        "room_index": 0,
                        "corners": [
                            [0.0, 3.0, 0.0],
                            [4.0, 3.0, 0.0],
                            [4.0, 3.0, 2.0],
                            [0.0, 3.0, 2.0],
                        ],
                        "surface_kind": "flat",
                        "boundary_face_id": "face:roof-flat:0",
                        "roof_hypothesis_id": "roof-hypothesis:flat:0",
                        "flat_role": "roof_flat",
                    }
                ],
                "oblique": [],
            }
        },
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surface = next(
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    )
    assert roof_surface["source_kind"] == "roof_surface_exact_flat"
    assert [surface["category"] for surface in payload["renderable_surfaces"]].count(
        "unresolved_region"
    ) == 0
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 1
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_promote_exact_flat_shell_void() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:0": {
                "room_index": 0,
                "story": 0,
                "partially_covered_by_sloped_roof": True,
                "strong_perimeter_sloped": True,
                "strong_knee_wall_signal": True,
                "has_attic_relation": False,
                "has_upper_void_relation": True,
                "has_resolved_roof_relation": True,
                "has_candidate_attic_relation": False,
                "has_candidate_upper_void_relation": False,
                "has_oblique_atom": False,
                "roof_evidence_score": 7,
            }
        },
        "unresolved_regions": [],
        "dormers": [],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [],
                    "walls_computed": [],
                    "windows": [],
                    "doors": [],
                    "openings": [],
                }
            ]
        },
        roof={
            "roof_surfaces": {
                "flat": [
                    {
                        "story": 0,
                        "room_index": 0,
                        "corners": [
                            [0.0, 3.0, 0.0],
                            [4.0, 3.0, 0.0],
                            [4.0, 3.0, 2.0],
                            [0.0, 3.0, 2.0],
                        ],
                        "surface_kind": "flat",
                        "boundary_face_id": "face:roof-flat:resolved",
                        "roof_hypothesis_id": "roof-hypothesis:flat:resolved",
                        "flat_role": "roof_flat",
                    }
                ],
                "oblique": [],
            }
        },
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surface = next(
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    )
    assert roof_surface["source_kind"] == "roof_surface_exact_flat"
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 1
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_no_promote_flat_shell_sloped() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:1"],
                "room_indices": [1],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:1": {
                "room_index": 1,
                "story": 0,
                "roles": ["sloped_ceiling"],
                "partially_covered_by_sloped_roof": True,
                "covered_by_sloped_roof": False,
                "strong_perimeter_sloped": True,
                "strong_knee_wall_signal": False,
                "has_attic_relation": False,
                "has_upper_void_relation": False,
                "has_resolved_roof_relation": True,
                "has_candidate_attic_relation": False,
                "has_candidate_upper_void_relation": False,
                "has_oblique_atom": True,
                "roof_evidence_score": 2,
            }
        },
        "unresolved_regions": [],
        "dormers": [],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [],
                    "walls_computed": [],
                    "windows": [],
                    "doors": [],
                    "openings": [],
                }
            ]
        },
        roof={
            "roof_surfaces": {
                "flat": [
                    {
                        "story": 0,
                        "room_index": 1,
                        "corners": [
                            [0.0, 3.0, 0.0],
                            [4.0, 3.0, 0.0],
                            [4.0, 3.0, 2.0],
                            [0.0, 3.0, 2.0],
                        ],
                        "surface_kind": "flat",
                        "boundary_face_id": "face:roof-flat:sloped-room",
                        "roof_hypothesis_id": "roof-hypothesis:flat:sloped-room",
                        "flat_role": "roof_flat",
                    }
                ],
                "oblique": [],
            }
        },
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert roof_surfaces == []
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 0
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_no_promote_patches_shell() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {"id": "building-part:0", "room_ids": ["room:0"], "room_indices": [0]},
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:0"],
                "room_indices": [0],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {
            "room:0": {
                "room_index": 0,
                "story": 0,
                "partially_covered_by_sloped_roof": True,
                "roof_evidence_score": 5,
                "has_oblique_atom": True,
            }
        },
        "oblique_coverage_patches": [
            {
                "id": (
                    "roof-coverage-patch:roof-hypothesis:oblique:0:coverage-subpart:0:0"
                ),
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "coverage_subpart_id": "coverage-subpart:0",
                "effective_part_ids": ["building-part:0"],
                "room_indices": [0],
                "room_ids": ["room:0"],
                "polygon": [
                    [0.0, 3.0, 0.0],
                    [2.0, 3.4, 0.0],
                    [2.0, 3.4, 2.0],
                    [0.0, 3.0, 2.0],
                ],
                "polygon_xz": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                "surface_kind": "oblique",
                "story": 0,
                "coverage_semantic_kind": "gable_run",
                "continuation_source": "coverage_subpart",
            }
        ],
        "unresolved_regions": [],
        "dormers": [],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "windows": [],
                "doors": [],
                "openings": [],
                "walls_computed": [],
            }
        ]
    }
    roof = {
        "roof_surfaces": {
            "flat": [],
            "oblique": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.8, 0.0],
                        [4.0, 3.8, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "oblique",
                    "boundary_face_id": "face:roof-oblique:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={"building-part:0": set(), FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building=building,
        roof=roof,
    )

    part_payload = payloads["building-part:0"]
    assert [
        surface["category"] for surface in part_payload["renderable_surfaces"]
    ].count("exterior_roof") == 0
    assert part_payload["metadata"]["roof_coverage_patch_surface_count"] == 0
    assert part_payload["metadata"]["roof_fallback_surface_count"] == 0
    assert part_payload["metadata"]["unresolved_region_count"] == 0

    full_payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in full_payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert len(roof_surfaces) == 0
    assert [
        surface["category"] for surface in full_payload["renderable_surfaces"]
    ].count("unresolved_region") == 1
    assert full_payload["metadata"]["roof_coverage_patch_surface_count"] == 0
    assert full_payload["metadata"]["roof_fallback_surface_count"] == 0
    assert full_payload["metadata"]["unresolved_region_count"] == 1


def test_replace_roomless_oblique_atoms() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:1"],
                "room_indices": [1],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [
            {
                "id": "atom:oblique:1",
                "type": "TopBoundaryAtom",
                "kind": "oblique",
                "role": "sloped_ceiling",
                "room_id": "room:1",
                "room_index": 1,
                "story": 0,
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "effective_part_id": FULL_BUILDING_PART_ID,
                "poly": [
                    [0.0, 3.0, 0.0],
                    [2.0, 3.4, 0.0],
                    [2.0, 3.4, 2.0],
                    [0.0, 3.0, 2.0],
                ],
            }
        ],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [],
            "oblique": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.8, 0.0],
                        [4.0, 3.8, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "oblique",
                    "boundary_face_id": "face:roof-oblique:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert len(roof_surfaces) == 1
    assert roof_surfaces[0]["source_kind"] == "roof_atom_patch"
    assert roof_surfaces[0]["room_index"] == 1
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_reject_roomless_oblique_no_atoms() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": [],
                "room_indices": [],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [],
            "oblique": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.8, 0.0],
                        [4.0, 3.8, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "oblique",
                    "boundary_face_id": "face:roof-oblique:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    assert [surface["category"] for surface in payload["renderable_surfaces"]].count(
        "exterior_roof"
    ) == 0
    assert [surface["category"] for surface in payload["renderable_surfaces"]].count(
        "unresolved_region"
    ) == 1
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 1


def test_replace_roomless_flat_atoms() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:1"],
                "room_indices": [1],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [
            {
                "id": "atom:flat:1",
                "type": "TopBoundaryAtom",
                "kind": "flat",
                "flat_role": "roof_flat",
                "role": "flat_ceiling",
                "room_id": "room:1",
                "room_index": 1,
                "story": 0,
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "effective_part_id": FULL_BUILDING_PART_ID,
                "poly": [
                    [0.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
            }
        ],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:roof-flat:0",
                    "roof_hypothesis_id": "roof-hypothesis:flat:0",
                    "flat_role": "roof_flat",
                }
            ],
            "oblique": [],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert len(roof_surfaces) == 1
    assert roof_surfaces[0]["source_kind"] == "roof_atom_patch"
    assert roof_surfaces[0]["surface_kind"] == "flat"
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 1
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_no_promote_transition_caps_flat_atoms() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:1"],
                "room_indices": [1],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [
            {
                "id": "atom:flat:transition",
                "type": "TopBoundaryAtom",
                "kind": "flat",
                "flat_role": "roof_flat",
                "role": "flat_transition_cap",
                "room_id": "room:1",
                "room_index": 1,
                "story": 0,
                "roof_hypothesis_id": "roof-hypothesis:flat:0",
                "effective_part_id": FULL_BUILDING_PART_ID,
                "poly": [
                    [0.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
            }
        ],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:roof-flat:0",
                    "roof_hypothesis_id": "roof-hypothesis:flat:0",
                    "flat_role": "roof_flat",
                }
            ],
            "oblique": [],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert roof_surfaces == []


def test_viewer_ontology_part_payloads_replace_room_flat_fallback_with_room_atoms() -> (
    None
):
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:5"],
                "room_indices": [5],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [
            {
                "id": "atom:flat:room:5",
                "type": "TopBoundaryAtom",
                "kind": "flat",
                "flat_role": "roof_flat",
                "role": "flat_ceiling",
                "room_id": "room:5",
                "room_index": 5,
                "story": 1,
                "roof_hypothesis_id": "roof-hypothesis:flat:new",
                "effective_part_id": FULL_BUILDING_PART_ID,
                "poly": [
                    [0.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
            }
        ],
        "room_summaries": {
            "room:5": {
                "has_resolved_roof_relation": False,
                "covered_by_sloped_roof": False,
                "partially_covered_by_sloped_roof": False,
                "strong_perimeter_sloped": False,
                "strong_knee_wall_signal": False,
                "has_candidate_attic_relation": False,
                "has_candidate_upper_void_relation": False,
                "has_oblique_atom": False,
                "roof_evidence_score": 0,
            }
        },
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 1,
                    "corners": [
                        [-1.0, 3.0, -1.0],
                        [3.0, 3.0, -1.0],
                        [3.0, 3.0, 3.0],
                        [-1.0, 3.0, 3.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:roof-flat:legacy-room-5",
                    "roof_hypothesis_id": "roof-hypothesis:flat:legacy",
                    "flat_role": "roof_flat",
                    "room_index": 5,
                }
            ],
            "oblique": [],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: {"room:5"}},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert len(roof_surfaces) == 1
    assert roof_surfaces[0]["source_kind"] == "roof_atom_patch"
    assert roof_surfaces[0]["source_id"] == "atom:flat:room:5"
    assert roof_surfaces[0]["roof_hypothesis_id"] == "roof-hypothesis:flat:new"
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 1
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_reject_roomless_flat_no_exact() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": [],
                "room_indices": [],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "semantic_atoms": [],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:roof-flat:0",
                    "roof_hypothesis_id": "roof-hypothesis:flat:0",
                    "flat_role": "roof_flat",
                }
            ],
            "oblique": [],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    assert [surface["category"] for surface in payload["renderable_surfaces"]].count(
        "exterior_roof"
    ) == 0
    assert [surface["category"] for surface in payload["renderable_surfaces"]].count(
        "unresolved_region"
    ) == 0
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 0
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_suppress_roomless_when_exact() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": ["room:1", "room:2"],
                "room_indices": [1, 2],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [],
            "oblique": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.8, 0.0],
                        [4.0, 3.8, 2.0],
                        [0.0, 3.0, 2.0],
                    ],
                    "surface_kind": "oblique",
                    "boundary_face_id": "face:roof-oblique:0",
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ],
        }
    }
    roof_cell_complex = {
        "cells": [
            {
                "id": "roof-cell:0",
                "room_id": "room:1",
                "room_index": 1,
                "part_id": FULL_BUILDING_PART_ID,
                "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                "cell_kind": "attic",
                "faces": [
                    {
                        "id": "f:roof:0",
                        "role": "roof",
                        "corners": [
                            [0.0, 3.0, 0.0],
                            [2.0, 3.4, 0.0],
                            [2.0, 3.4, 2.0],
                            [0.0, 3.0, 2.0],
                        ],
                    }
                ],
            }
        ],
        "knee_walls": [],
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex=roof_cell_complex,
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    roof_surfaces = [
        surface
        for surface in payload["renderable_surfaces"]
        if surface["category"] == "exterior_roof"
    ]
    assert len(roof_surfaces) == 1
    assert roof_surfaces[0]["source_kind"] == "roof_cell_face"
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0


def test_viewer_ontology_part_payloads_skip_roomless_ambiguous_flat_surfaces() -> None:
    summary = {
        "uuid": "test-uuid",
        "building_parts": [
            {
                "id": FULL_BUILDING_PART_ID,
                "room_ids": [],
                "room_indices": [],
                "synthetic": True,
                "synthetic_role": "full_building",
            },
        ],
        "room_summaries": {},
        "unresolved_regions": [],
        "dormers": [],
    }
    roof = {
        "roof_surfaces": {
            "flat": [
                {
                    "story": 0,
                    "corners": [
                        [0.0, 3.0, 0.0],
                        [4.0, 3.0, 0.0],
                        [4.0, 3.0, 4.0],
                        [0.0, 3.0, 4.0],
                    ],
                    "surface_kind": "flat",
                    "boundary_face_id": "face:flat:ambiguous",
                    "roof_hypothesis_id": "roof-hypothesis:flat:0",
                    "flat_role": "ambiguous_flat_over_sloped_part",
                }
            ],
            "oblique": [],
        }
    }

    payloads = _build_ontology_part_payloads(
        uuid="test-uuid",
        summary=summary,
        part_graph_room_ids={FULL_BUILDING_PART_ID: set()},
        topology_cell_complex={"cells": []},
        roof_cell_complex={"cells": [], "knee_walls": []},
        occupied_room_cell_complex={"cells": []},
        building={"rooms": []},
        roof=roof,
    )

    payload = payloads[FULL_BUILDING_PART_ID]
    assert [surface["category"] for surface in payload["renderable_surfaces"]].count(
        "exterior_roof"
    ) == 0
    assert payload["metadata"]["roof_exact_flat_surface_count"] == 0
    assert payload["metadata"]["roof_fallback_surface_count"] == 0
    assert payload["metadata"]["unresolved_region_count"] == 0
