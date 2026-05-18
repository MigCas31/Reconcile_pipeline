from __future__ import annotations

import json

from reconcile_tiers.polyhedron import (
    export_topology_resolution,
    make_cube,
    polyhedron_snapshot,
)
from reconcile_tiers.polyhedron.half_edge import build_from_planar_polygons
from tests.reconcile_tiers.polyhedron.test_topology_events import (
    _box_with_split_top_polys,
    _cube_with_four_zero_z_edges,
)


def test_polyhedron_snapshot_exports_cube_faces_as_jsonable_polygons():
    cube = make_cube(size=2.0)

    snapshot = polyhedron_snapshot(cube)

    assert snapshot["counts"] == {"faces": 6, "vertices": 8, "half_edges": 24}
    assert len(snapshot["faces"]) == 6
    assert all(len(face["corners"]) == 4 for face in snapshot["faces"])
    assert all(not face["errors"] for face in snapshot["faces"])
    json.dumps(snapshot)


def test_export_topology_resolution_records_coplanar_merge_frames():
    box = build_from_planar_polygons(_box_with_split_top_polys())

    export = export_topology_resolution(box)

    assert export["schema_version"] == 1
    assert export["stop"]["reason"] == "valid"
    assert [step["action"] for step in export["steps"]] == [
        "adjacent_coplanar_face_merge"
    ]
    assert len(export["frames"]) == 2
    assert export["frames"][0]["counts"]["faces"] == 7
    assert export["frames"][1]["counts"] == {
        "faces": 6,
        "vertices": 8,
        "half_edges": 24,
    }
    assert export["steps"][0]["before_frame"] == 0
    assert export["steps"][0]["after_frame"] == 1
    json.dumps(export)


def test_export_topology_resolution_first_mode_records_viewer_playback_steps():
    cube = _cube_with_four_zero_z_edges()

    export = export_topology_resolution(
        cube,
        selection="first",
        max_steps=3,
        edge_tol_m=1e-9,
        face_area_tol_m2=1e-9,
    )

    assert export["stop"]["reason"] == "max_steps"
    assert [step["action"] for step in export["steps"]] == [
        "edge_collapse",
        "triangle_face_collapse",
        "adjacent_coplanar_face_merge",
    ]
    assert len(export["frames"]) == 4
    assert [step["before_frame"] for step in export["steps"]] == [0, 1, 2]
    assert [step["after_frame"] for step in export["steps"]] == [1, 2, 3]
    assert export["frames"][2]["counts"]["faces"] == 5
    assert export["frames"][2]["issues"][0]["kind"] == "adjacent_coplanar_faces"
    json.dumps(export)
