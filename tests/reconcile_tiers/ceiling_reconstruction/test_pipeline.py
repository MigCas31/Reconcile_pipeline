"""Tests for kinetic ceiling reconstruction."""

from __future__ import annotations

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.ceiling_reconstruction.graph_cut import graph_cut_label_cells
from reconcile_tiers.ceiling_reconstruction.input_tiles import (
    collect_ksr_room_tiles,
    room_bounding_prism,
)
from reconcile_tiers.ceiling_reconstruction.oriented_samples import sample_oriented_points
from reconcile_tiers.ceiling_reconstruction.plane_regularization import (
    detect_plane_groups,
    regularize_planes,
)
from reconcile_tiers.ceiling_reconstruction.pipeline import run_ksr_pipeline
from reconcile_tiers.ceiling_reconstruction.trace_export import (
    _partition_faces,
    build_ksr_room_trace,
)
from reconcile_tiers.polyhedron.kinetic_partition import BoundingPrism, kinetic_partition
from reconcile_tiers.polyhedron.manifold_repair import TileFace


def _box_room(*, with_ceiling: bool = True) -> tuple[dict, dict]:
    """4m x 4m x 2.5m room centered at origin."""
    floor_y = 0.0
    ceil_y = 2.5
    payload = {
        "rooms": [],
        "ceiling": [],
        "visual_shells": [],
        "gable_closures": [],
    }
    walls = [
        ([(-2, floor_y, -2), (-2, ceil_y, -2), (-2, ceil_y, 2), (-2, floor_y, 2)], Plane(1, 0, 0, -2)),
        ([(2, floor_y, -2), (2, floor_y, 2), (2, ceil_y, 2), (2, ceil_y, -2)], Plane(-1, 0, 0, -2)),
        ([(-2, floor_y, -2), (2, floor_y, -2), (2, ceil_y, -2), (-2, ceil_y, -2)], Plane(0, 0, 1, -2)),
        ([(-2, floor_y, 2), (-2, ceil_y, 2), (2, ceil_y, 2), (2, floor_y, 2)], Plane(0, 0, -1, -2)),
    ]
    floor_corners = [(-2, floor_y, -2), (2, floor_y, -2), (2, floor_y, 2), (-2, floor_y, 2)]
    room = {
        "room_index": 0,
        "story": 0,
        "locator_id": "room:0",
        "floor": [{"corners": [{"x": c[0], "y": c[1], "z": c[2]} for c in floor_corners], "plane": {"a": 0, "b": 1, "c": 0, "d": floor_y}, "locator_id": "floor:0"}],
        "walls": [],
    }
    for i, (corners, plane) in enumerate(walls):
        room["walls"].append(
            {
                "corners": [{"x": c[0], "y": c[1], "z": c[2]} for c in corners],
                "plane": {"a": plane.a, "b": plane.b, "c": plane.c, "d": plane.d},
                "locator_id": f"wall:{i}",
            }
        )
    if with_ceiling:
        ceil_corners = [(-2, ceil_y, -2), (2, ceil_y, -2), (2, ceil_y, 2), (-2, ceil_y, 2)]
        payload["ceiling"] = [
            {
                "corners": [{"x": c[0], "y": c[1], "z": c[2]} for c in ceil_corners],
                "plane": {"a": 0, "b": 1, "c": 0, "d": ceil_y},
                "locator_id": "ceiling:0",
            }
        ]
    payload["rooms"] = [room]
    return payload, room


def _tile(
    face_id: int,
    corners: list[tuple[float, float, float]],
    plane: Plane,
    source: str,
) -> TileFace:
    return TileFace(
        face_id=face_id,
        corners=tuple(corners),
        plane=plane,
        source=source,
        locator_id=f"{source}:{face_id}",
        story=0,
        room_index=0,
    )


def test_collect_ksr_room_tiles_splits_structure_evidence() -> None:
    payload, room = _box_room(with_ceiling=True)
    ksr = collect_ksr_room_tiles(payload, room)
    assert len(ksr.structure) == 5  # 4 walls + floor
    assert len(ksr.evidence) == 1
    assert all(t.source in ("wall", "floor") for t in ksr.structure)
    assert ksr.evidence[0].source == "ceiling"


def test_room_bounding_prism_ignores_ceiling_height() -> None:
    payload, room = _box_room(with_ceiling=True)
    ksr = collect_ksr_room_tiles(payload, room)
    prism = room_bounding_prism(ksr.structure, margin=0.5)
    assert prism.y_max <= 2.5 + 0.5 + 0.1
    assert prism.y_min >= -0.5 - 0.1


def test_oriented_samples_normals_point_inward() -> None:
    payload, room = _box_room()
    ksr = collect_ksr_room_tiles(payload, room)
    samples = sample_oriented_points(ksr.structure)
    assert samples.shape[1] == 6
    assert samples.shape[0] > 0
    center = np.array([0.0, 1.25, 0.0])
    for row in samples:
        point = row[:3]
        normal = row[3:6]
        to_center = center - point
        assert float(normal @ to_center) > 0.0


def test_plane_regularization_merges_parallel_ceiling_fragments() -> None:
    plane_a = Plane(a=0.0, b=1.0, c=0.0, d=2.5)
    plane_b = Plane(a=0.0, b=1.0, c=0.0, d=2.52)
    tiles = [
        _tile(0, [(-1, 2.5, -1), (1, 2.5, -1), (1, 2.5, 1), (-1, 2.5, 1)], plane_a, "ceiling"),
        _tile(1, [(1, 2.52, -1), (2, 2.52, -1), (2, 2.52, 1), (1, 2.52, 1)], plane_b, "ceiling"),
        _tile(2, [(-2, 0, -2), (2, 0, -2), (2, 0, 2), (-2, 0, 2)], Plane(0, 1, 0, 0), "floor"),
        _tile(3, [(-2, 0, -2), (-2, 2.5, -2), (-2, 2.5, 2), (-2, 0, 2)], Plane(1, 0, 0, 2), "wall"),
    ]
    detected = detect_plane_groups(tiles)
    assert len(detected) >= 3
    groups, planes = regularize_planes(tiles)
    ceiling_groups = [g for g in groups if g.kind == "ceiling"]
    assert len(ceiling_groups) == 1
    assert len(planes) >= 3


def test_graph_cut_zmax_boundary_labels_upper_solid() -> None:
    floor = Plane(a=0.0, b=1.0, c=0.0, d=0.0)
    ceil = Plane(a=0.0, b=1.0, c=0.0, d=3.0)
    prism = BoundingPrism(x_min=-1, x_max=5, y_min=-1, y_max=5, z_min=-1, z_max=5)
    samples = np.array(
        [[1.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.5, 1.0, 0.0, -1.0, 0.0]],
        dtype=float,
    )
    cells = kinetic_partition([floor, ceil], samples[:, :3], prism)
    assert cells
    result = graph_cut_label_cells(cells, samples, prism, lambda_param=0.5)
    assert result.solver_status == "optimal"
    top_cells = [c for c in cells if float(c.centroid[1]) >= 2.5]
    assert top_cells
    assert any(result.labels[c.cell_id] == "inside" for c in top_cells)


def test_partition_faces_non_empty() -> None:
    floor = Plane(a=0.0, b=1.0, c=0.0, d=0.0)
    wall = Plane(a=1.0, b=0.0, c=0.0, d=0.0)
    prism = BoundingPrism(x_min=-2, x_max=2, y_min=0, y_max=3, z_min=-2, z_max=2)
    samples = np.array([[0.5, 1.0, 0.0]], dtype=float)
    cells = kinetic_partition([floor, wall], samples, prism)
    faces = _partition_faces(cells, [floor, wall], prism)
    assert len(faces) > 1


def test_pipeline_trace_seven_frames_with_ceiling() -> None:
    payload, room = _box_room(with_ceiling=True)
    trace = build_ksr_room_trace(payload, room)
    assert trace["schema_version"] == 2
    assert trace["selection"] == "kinetic-ceiling-steps"
    assert len(trace["frames"]) == 7
    steps = [f["pipeline_step"] for f in trace["frames"]]
    assert steps[-1] == "ceiling_extracted"
    frame0_labels = {f["label"] for f in trace["frames"][0]["faces"]}
    assert frame0_labels <= {"wall", "floor"}
    frame4 = trace["frames"][4]
    assert frame4["pipeline_step"] == "kinetic_partition"
    if frame4["meta"].get("cell_count", 0) > 0:
        assert frame4["counts"]["faces"] > 0
    result = run_ksr_pipeline(payload, room)
    assert result.stop_reason in ("ceiling_extracted", "no_ceiling_faces", "partition_failed")
    if result.ceiling_faces:
        for face in result.ceiling_faces:
            assert float(face.plane.b) >= 0.17
