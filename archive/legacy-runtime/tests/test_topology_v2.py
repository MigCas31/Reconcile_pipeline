from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reconcile.cli import reconcile_building
from reconcile.loader import load_merged
from reconcile_v2.builder import build_topology_graph
from reconcile_v2.ifc_mapping import run_ifc_conformance_checks
from reconcile_v2.output import validate_against_schema
from reconcile_v2.stitch_geometry import _build_edge_topology, build_stitched_geometry
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
        "openings": [
            {
                "identifier": f"opening_{room_idx}",
                "category": {"opening": {}},
                "confidence": {"confidence": "high"},
                "dimensions": [0.8, 2.0, 0.1],
                "transform": _identity_flat(),
                "polygonCorners": [
                    [0.2, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [0.2, 2.0, 0.0],
                ],
                "story": story,
                "completedEdges": [],
                "parentIdentifier": wall_id,
                "curve": None,
            }
        ],
        "objects": [
            {
                "identifier": f"storage_{room_idx}",
                "category": {"storage": {}},
                "confidence": {"confidence": "high"},
                "dimensions": [0.6, 1.0, 0.4],
                "transform": _identity_flat(),
                "polygonCorners": [
                    [0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0],
                    [0.6, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "story": story,
                "completedEdges": [],
                "parentIdentifier": None,
                "curve": None,
            }
        ],
        "floors": [_floor(floor_id, story=story, x0=x0, z0=z0, x1=x1, z1=z1)],
        "referenceOriginTransform": _identity_flat(),
    }


def _scan_room(
    room_id: str,
    story: int,
    wall_id: str,
    floor_id: str,
    floor_bounds: tuple[float, float, float, float],
) -> dict:
    # Same shape as room data; stored as separate JSON in scan-cache.
    return _room(0, story, wall_id, floor_id, floor_bounds)


class TestTopologyV2(unittest.TestCase):
    def _make_fixture(self, root: Path) -> tuple[Path, Path, str]:
        uuid = "11111111-2222-3333-4444-555555555555"
        merged_path = root / "merged.json"
        scan_dir = root / "scan-cache"
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Room A and B separated by 0.1m -> adjacency/gap detector should detect.
        room_a = _room(0, 0, "wall_a", "floor_a", (0.0, 0.0, 2.0, 2.0))
        room_b = _room(1, 0, "wall_b", "floor_b", (2.1, 0.0, 4.1, 2.0))

        merged = {
            "version": 2,
            "walls": room_a["walls"] + room_b["walls"],
            "doors": [],
            "windows": [],
            "openings": room_a["openings"] + room_b["openings"],
            "objects": room_a["objects"] + room_b["objects"],
            "floors": room_a["floors"] + room_b["floors"],
            "rooms": [room_a, room_b],
            "sections": [],
        }
        merged_path.write_text(json.dumps(merged))

        (scan_dir / "room_a.json").write_text(
            json.dumps(
                _scan_room("room_a", 0, "wall_a", "floor_a", (0.0, 0.0, 2.0, 2.0))
            )
        )
        (scan_dir / "room_b.json").write_text(
            json.dumps(
                _scan_room("room_b", 0, "wall_b", "floor_b", (2.1, 0.0, 4.1, 2.0))
            )
        )

        return merged_path, scan_dir, uuid

    def test_node_edge_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, scan_dir, uuid = self._make_fixture(Path(tmp))
            graph = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )
            node_types = {n.type for n in graph.nodes}
            edge_types = {e.type for e in graph.edges}
            surf_kinds = {
                (n.properties or {}).get("surface_kind")
                for n in graph.nodes
                if n.type == "Surface"
            }

            self.assertTrue(
                {"Building", "Story", "Room", "Surface", "Boundary", "Gap"}.issubset(
                    node_types
                )
            )
            self.assertTrue(
                {
                    "CONTAINS",
                    "BOUNDS",
                    "ADJACENT_TO",
                    "HAS_GAP",
                    "DERIVED_FROM",
                }.issubset(edge_types)
            )
            self.assertIn("opening", surf_kinds)
            self.assertIn("storage", surf_kinds)

    def test_story_assignment_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, scan_dir, uuid = self._make_fixture(Path(tmp))
            graph = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )
            building = load_merged(merged_path)
            report = run_ifc_conformance_checks(graph, building)
            self.assertTrue(report.passed)

    def test_ifc_conformance_failure_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, scan_dir, uuid = self._make_fixture(Path(tmp))
            graph = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )
            # Remove story->room CONTAINS edges to force room parent failure.
            graph.edges = [
                e
                for e in graph.edges
                if not (
                    e.type == "CONTAINS"
                    and e.from_id.startswith("story:")
                    and e.to_id.startswith("room:")
                )
            ]
            report = run_ifc_conformance_checks(graph, load_merged(merged_path))
            self.assertFalse(report.passed)
            self.assertTrue(any(i.code == "ROOM_PARENT_STORY" for i in report.issues))

    def test_schema_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, scan_dir, uuid = self._make_fixture(Path(tmp))
            graph = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )
            errs = validate_against_schema(graph.as_dict())
            self.assertEqual([], errs)

    def test_wall_thickness_metrics_and_story(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, _scan_dir, _uuid = self._make_fixture(Path(tmp))
            building = load_merged(merged_path)

            adjs = infer_intra_story_adjacency(building)
            self.assertTrue(adjs)
            adj = adjs[0]
            self.assertEqual(0, adj.story)
            self.assertGreater(adj.thickness_cm, 0.0)
            self.assertGreaterEqual(adj.thickness_p95_cm, adj.thickness_p05_cm)
            self.assertGreaterEqual(adj.overlap_ratio, 0.0)
            self.assertGreaterEqual(adj.confidence_score, 0.0)

            gaps, _fps = infer_gap_records(building)
            intra = [g for g in gaps if g.kind == "intra_story"]
            self.assertTrue(intra)
            self.assertTrue(all(g.story == 0 for g in intra))
            self.assertIn("p05_cm", intra[0].metrics)
            self.assertIn("p95_cm", intra[0].metrics)
            self.assertIn("std_cm", intra[0].metrics)

    def test_synthetic_gap_and_extension_surfaces_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, scan_dir, uuid = self._make_fixture(Path(tmp))
            graph = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )
            payload = graph.as_dict()

            synth = payload.get("quality", {}).get("synthetic_surfaces", {})
            self.assertIn("wall_extension_strip", synth)
            self.assertIn("pipeline_wall_extension_strip", synth)
            self.assertIn("intra_story_gap_thickness_wall", synth)
            self.assertIn("cross_story_gap_floor_closure", synth)
            self.assertIn("pipeline_gap_closure", synth)
            self.assertIn("pipeline_stitch_wall", synth)
            self.assertIn("pipeline_gap_wall", synth)
            self.assertGreaterEqual(synth["intra_story_gap_thickness_wall"], 1)

            kinds = {
                (n.get("properties") or {}).get("synthetic_kind")
                for n in payload["nodes"]
                if n["type"] == "Surface"
            }
            self.assertIn("intra_story_gap_thickness_wall", kinds)

    def test_golden_deterministic_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_path, scan_dir, uuid = self._make_fixture(Path(tmp))
            g1 = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )
            g2 = build_topology_graph(
                merged_path=merged_path, scan_dir=scan_dir, uuid=uuid
            )

            self.assertEqual([n.id for n in g1.nodes], [n.id for n in g2.nodes])
            self.assertEqual([e.id for e in g1.edges], [e.id for e in g2.edges])

    def test_regression_guard_existing_cli_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged_path, scan_dir, uuid = self._make_fixture(root)
            output_path = root / "reconciled.json"
            result = reconcile_building(
                merged_path=merged_path,
                scan_dir=scan_dir,
                output_path=output_path,
                uuid=uuid,
            )
            payload = json.loads(output_path.read_text())

            # Existing contract shape remains
            # (no topology-v2 keys in baseline artifact).
            self.assertIn("rooms", payload)
            self.assertIn("walls", payload)
            self.assertIn("reconciliation", payload)
            self.assertNotIn("nodes", payload)
            self.assertNotIn("edges", payload)

            for key in [
                "uuid",
                "classification",
                "match_rate",
                "wall_count_merged",
                "wall_count_scan",
                "flags",
            ]:
                self.assertIn(key, result)

    def test_stitched_geometry_shared_nodes(self):
        nodes = [
            {
                "id": "surface:a",
                "type": "Surface",
                "story": 0,
                "properties": {"surface_kind": "floor"},
                "transform_ref": "t:a",
            },
            {
                "id": "surface:b",
                "type": "Surface",
                "story": 0,
                "properties": {"surface_kind": "floor"},
                "transform_ref": "t:b",
            },
        ]
        geometry_index = {
            "transforms": {
                "t:a": _identity_flat(),
                "t:b": _identity_flat(),
            },
            "surface_polygons": {
                "surface:a": [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
                "surface:b": [[1, 0, 0], [2, 0, 0], [2, 0, 1], [1, 0, 1]],
            },
            "room_floor_polygons": {},
        }
        stitched = build_stitched_geometry(nodes, geometry_index)
        stats = stitched["stats"]
        self.assertEqual(2, stats["input_surfaces"])
        self.assertGreaterEqual(stats["stitched_surfaces"], 1)
        self.assertGreaterEqual(stats["shared_nodes"], 4)
        self.assertTrue(
            any(s["kind"] == "floor" for s in stitched["stitched_surfaces"])
        )
        self.assertIn("edge_topology", stitched)
        self.assertIn("planar_arrangement", stitched)
        self.assertIn("clusters", stitched["planar_arrangement"])
        self.assertIn("story_footprint_ledger", stitched)
        self.assertIn("render_surfaces", stitched)
        self.assertTrue(stitched["validation"]["manifold_pass"])
        self.assertGreaterEqual(stitched["stats"]["shared_edges"], 4)
        self.assertIn("split_insertions", stitched["stats"])

    def test_edge_topology_detects_t_junction(self):
        shared_nodes = {
            "a": [0.0, 0.0, 0.0],
            "b": [2.0, 0.0, 0.0],
            "c": [1.0, 0.0, 0.0],
            "d": [1.0, 0.0, 1.0],
            "e": [0.0, 0.0, 1.0],
            "f": [2.0, 0.0, 1.0],
        }
        stitched_surfaces = [
            {
                "id": "surface:base",
                "node_ids": ["a", "b", "f", "e"],
                "holes": [],
            },
            {
                "id": "surface:branch",
                "node_ids": ["c", "d", "f"],
                "holes": [],
            },
        ]

        edge_topology = _build_edge_topology(
            stitched_surfaces,
            shared_nodes,
            t_junction_tol_m=0.01,
        )

        self.assertGreaterEqual(edge_topology["validation"]["t_junction_count"], 1)
        self.assertTrue(
            any(
                issue["code"] == "T_JUNCTION"
                for issue in edge_topology["validation"]["issues"]
            )
        )


if __name__ == "__main__":
    unittest.main()
