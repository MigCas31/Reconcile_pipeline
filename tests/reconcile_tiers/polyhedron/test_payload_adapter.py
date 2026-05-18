from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

import reconcile_tiers.polyhedron.payload_adapter as payload_adapter
from reconcile_tiers._core.wing_decomposition import Wing
from reconcile_tiers.polyhedron import (
    build_envelope_polyhedra_from_tier_payload,
    build_polyhedron_from_tier_payload,
    build_room_shell_from_tier_payload,
    payload_envelope_candidates_from_tier_payload,
    payload_faces_for_room_shell,
    payload_faces_from_plane_evidence,
    payload_faces_from_tier_payload,
)


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _cube_payload(*, duplicate_floor_corner: bool = False) -> dict:
    floor_corners = [
        _pt(0.0, 0.0, 0.0),
        _pt(2.0, 0.0, 0.0),
        _pt(2.0, 0.0, 2.0),
        _pt(0.0, 0.0, 2.0),
    ]
    if duplicate_floor_corner:
        floor_corners.insert(3, _pt(2.0, 0.0, 2.0))

    return {
        "rooms": [
            {
                "story": 0,
                "locator_id": "building::tier-room::0",
                "floor": [
                    {
                        "corners": floor_corners,
                        "holes": [],
                        "adjacency": "internalToHeated",
                    }
                ],
                "walls": [
                    {
                        "locator_id": "building::tier-wall::px",
                        "corners": [
                            _pt(2.0, 0.0, 0.0),
                            _pt(2.0, 2.0, 0.0),
                            _pt(2.0, 2.0, 2.0),
                            _pt(2.0, 0.0, 2.0),
                        ],
                    },
                    {
                        "locator_id": "building::tier-wall::nx",
                        "corners": [
                            _pt(0.0, 0.0, 0.0),
                            _pt(0.0, 0.0, 2.0),
                            _pt(0.0, 2.0, 2.0),
                            _pt(0.0, 2.0, 0.0),
                        ],
                    },
                    {
                        "locator_id": "building::tier-wall::pz",
                        "corners": [
                            _pt(0.0, 0.0, 2.0),
                            _pt(2.0, 0.0, 2.0),
                            _pt(2.0, 2.0, 2.0),
                            _pt(0.0, 2.0, 2.0),
                        ],
                    },
                    {
                        "locator_id": "building::tier-wall::nz",
                        "corners": [
                            _pt(0.0, 0.0, 0.0),
                            _pt(0.0, 2.0, 0.0),
                            _pt(2.0, 2.0, 0.0),
                            _pt(2.0, 0.0, 0.0),
                        ],
                    },
                ],
                "doors": [],
                "windows": [],
            }
        ],
        "ceiling": [
            {
                "locator_id": "building::tier-ceiling-flat::0",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(2.0, 2.0, 2.0),
                    _pt(2.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "flat_ceiling",
            }
        ],
    }


def test_payload_adapter_extracts_oriented_faces():
    faces = payload_faces_from_tier_payload(_cube_payload())
    assert [f.kind for f in faces].count("floor") == 1
    assert [f.kind for f in faces].count("wall") == 4
    assert [f.kind for f in faces].count("ceiling") == 1

    floor = next(f for f in faces if f.kind == "floor")
    assert floor.plane.b < -0.5
    assert math.isclose(floor.plane.d, 0.0, abs_tol=1e-9)

    plus_x = next(f for f in faces if f.locator_id.endswith("::px"))
    assert plus_x.plane.a > 0.5
    assert math.isclose(plus_x.plane.d, 2.0, abs_tol=1e-9)


def test_payload_faces_from_plane_evidence_preserves_raw_candidates():
    evidence = {
        "raw_ceiling_planes": [
            {
                "locator_id": "building::raw-ceiling-plane::3:2",
                "room_index": 3,
                "story": 1,
                "source": "raw_observed_ceiling_plane",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(2.0, 2.2, 0.0),
                    _pt(2.0, 2.2, 2.0),
                    _pt(0.0, 2.0, 2.0),
                ],
                "plane": {"a": -0.1, "b": 1.0, "c": 0.0, "d": 2.0},
            }
        ],
        "ceiling_candidates": [
            {
                "locator_id": "building::tier-ceiling-raw::3:2",
                "room_index": 3,
                "story": 1,
                "source": "raw_fallback",
                "kept_after_raw_gate": False,
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(1.0, 2.1, 0.0),
                    _pt(1.0, 2.1, 1.0),
                    _pt(0.0, 2.0, 1.0),
                ],
                "plane": {"a": -0.1, "b": 1.0, "c": 0.0, "d": 2.0},
            }
        ],
    }

    faces = payload_faces_from_plane_evidence(evidence)

    assert len(faces) == 2
    assert {face.room_index for face in faces} == {3}
    assert {face.story for face in faces} == {1}
    assert {face.source for face in faces} == {
        "raw_observed_ceiling_plane",
        "raw_fallback",
    }


def test_build_polyhedron_from_cube_payload():
    poly = build_polyhedron_from_tier_payload(_cube_payload(), coord_tol=1e-6)
    assert len(poly.faces) == 6
    assert len(poly.vertices) == 8
    assert len(poly.half_edges) == 24
    assert poly.is_watertight()
    assert poly.faces_close()


def test_build_polyhedron_from_payload_removes_duplicate_ring_corners():
    poly = build_polyhedron_from_tier_payload(
        _cube_payload(duplicate_floor_corner=True),
        coord_tol=1e-6,
        corner_tol=1e-3,
    )
    assert len(poly.faces) == 6
    assert len(poly.half_edges) == 24


def test_payload_faces_for_room_shell_selects_overlapping_ceiling():
    payload = _cube_payload()
    # Flat ceiling locators often do not encode a room id, so selection must
    # use floor/ceiling XZ overlap.
    payload["ceiling"][0]["locator_id"] = "building::tier-ceiling-flat::unknown"

    faces = payload_faces_for_room_shell(payload, room_index=0)
    assert [f.kind for f in faces].count("floor") == 1
    assert [f.kind for f in faces].count("wall") == 4
    assert [f.kind for f in faces].count("ceiling") == 1


def test_build_room_shell_ignores_other_room_faces():
    payload = _cube_payload()
    second = _cube_payload()["rooms"][0]
    second["locator_id"] = "building::tier-room::1"
    second["floor"][0]["corners"] = [
        _pt(3.0, 0.0, 0.0),
        _pt(5.0, 0.0, 0.0),
        _pt(5.0, 0.0, 2.0),
        _pt(3.0, 0.0, 2.0),
    ]
    # Deliberately leave second-room walls at their original coordinates. The
    # whole payload is incoherent, but room 0 remains a valid shell.
    payload["rooms"].append(second)

    poly = build_room_shell_from_tier_payload(payload, room_index=0, coord_tol=1e-6)
    assert len(poly.faces) == 6
    assert len(poly.vertices) == 8
    assert poly.is_watertight()


def test_payload_envelope_candidate_builds_from_building_footprint():
    payload = _cube_payload()

    candidates = payload_envelope_candidates_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
    )
    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert len(candidates) == 1
    assert candidates[0].locator_id == "envelope-wing:0"
    assert candidates[0].top_overlap_ratio >= 0.99
    assert len(built) == 1
    _candidate, poly = built[0]
    assert len(poly.faces) == 6
    assert len(poly.vertices) == 8
    assert len(poly.half_edges) == 24
    assert poly.is_watertight()


def test_payload_envelope_candidate_aggregates_fragmented_coplanar_top():
    payload = _cube_payload()
    payload["ceiling"] = []
    for index, (x0, z0, x1, z1) in enumerate(
        [
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 2.0, 1.0),
            (1.0, 1.0, 2.0, 2.0),
            (0.0, 1.0, 1.0, 2.0),
        ]
    ):
        payload["ceiling"].append(
            {
                "locator_id": f"building::tier-ceiling-flat::{index}",
                "corners": [
                    _pt(x0, 2.0, z0),
                    _pt(x0, 2.0, z1),
                    _pt(x1, 2.0, z1),
                    _pt(x1, 2.0, z0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "flat_ceiling",
            }
        )

    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert len(built) == 1
    candidate, poly = built[0]
    assert "tier-ceiling-flat::0" in candidate.top_source
    assert "tier-ceiling-flat::3" in candidate.top_source
    assert candidate.top_overlap_ratio >= 0.99
    assert len(poly.faces) == 6
    assert len(poly.vertices) == 8
    assert poly.is_watertight()


def test_payload_envelope_candidate_handles_single_sloped_top_plane():
    payload = _cube_payload()
    payload["ceiling"][0]["corners"] = [
        _pt(0.0, 2.0, 0.0),
        _pt(0.0, 2.0, 2.0),
        _pt(2.0, 3.0, 2.0),
        _pt(2.0, 3.0, 0.0),
    ]
    payload["ceiling"][0]["plane"] = {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0}
    payload["ceiling"][0]["source"] = "computed_oblique"

    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert len(built) == 1
    candidate, poly = built[0]
    assert candidate.top_source == "building::tier-ceiling-flat::0"
    assert len(poly.faces) == 6
    assert len(poly.vertices) == 8
    assert poly.is_watertight()
    top = next(face for face in poly.faces if face.plane.b > 0.5)
    ys = [poly.vertex_position(vertex)[1] for vertex in poly.vertices]
    assert round(min(ys), 6) == 0.0
    assert round(max(ys), 6) == 3.0
    assert top.plane.a < -0.4


def test_payload_envelope_candidate_builds_two_plane_gable_partition():
    payload = _cube_payload()
    payload["ceiling"] = [
        {
            "locator_id": "building::tier-ceiling-computed-oblique::left",
            "corners": [
                _pt(0.0, 2.0, 0.0),
                _pt(0.0, 2.0, 2.0),
                _pt(1.0, 3.0, 2.0),
                _pt(1.0, 3.0, 0.0),
            ],
            "plane": {"a": -1.0, "b": 1.0, "c": 0.0, "d": 2.0},
            "source": "computed_oblique",
        },
        {
            "locator_id": "building::tier-ceiling-computed-oblique::right",
            "corners": [
                _pt(1.0, 3.0, 0.0),
                _pt(1.0, 3.0, 2.0),
                _pt(2.0, 2.0, 2.0),
                _pt(2.0, 2.0, 0.0),
            ],
            "plane": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 4.0},
            "source": "computed_oblique",
        },
    ]

    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert len(built) == 1
    candidate, poly = built[0]
    assert "left" in candidate.top_source
    assert "right" in candidate.top_source
    assert len(poly.faces) == 7
    assert len(poly.vertices) == 10
    assert len(poly.half_edges) == 30
    assert poly.is_watertight()
    assert poly.faces_close()
    ys = sorted(round(poly.vertex_position(vertex)[1], 6) for vertex in poly.vertices)
    assert ys.count(3.0) == 2


def test_payload_envelope_candidate_uses_ridge_aware_mixed_flat_sloped_top():
    payload = _cube_payload()
    payload["rooms"][0]["floor"][0]["corners"] = [
        _pt(0.0, 0.0, 0.0),
        _pt(4.0, 0.0, 0.0),
        _pt(4.0, 0.0, 2.0),
        _pt(0.0, 0.0, 2.0),
    ]
    payload["ceiling"] = [
        {
            "locator_id": "building::tier-ceiling-flat::extended-lid",
            "corners": [
                _pt(0.0, 2.0, 0.0),
                _pt(0.0, 2.0, 2.0),
                _pt(4.0, 2.0, 2.0),
                _pt(4.0, 2.0, 0.0),
            ],
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
            "source": "flat_ceiling",
        },
        {
            "locator_id": "building::tier-ceiling-computed-oblique::right-slope",
            "corners": [
                _pt(2.0, 2.0, 0.0),
                _pt(2.0, 2.0, 2.0),
                _pt(4.0, 3.0, 2.0),
                _pt(4.0, 3.0, 0.0),
            ],
            "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 1.0},
            "source": "computed_oblique",
        },
    ]

    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert len(built) == 1
    candidate, poly = built[0]
    assert "extended-lid" in candidate.top_source
    assert "right-slope" in candidate.top_source
    assert poly.is_watertight()
    top_planes = [face.plane for face in poly.faces if face.plane.b > 0.5]
    assert len(top_planes) == 2
    assert any(plane.a < -0.4 for plane in top_planes)
    ys = sorted(round(poly.vertex_position(vertex)[1], 6) for vertex in poly.vertices)
    assert max(ys) == 3.0


def test_ridge_alignment_normalizes_each_building_part_not_whole_building():
    target = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    story_0 = affinity.translate(
        affinity.rotate(target, 4.0, origin=(2.0, 1.0), use_radians=False),
        xoff=0.25,
    )
    story_1 = affinity.translate(
        affinity.rotate(target, -4.0, origin=(2.0, 1.0), use_radians=False),
        xoff=-0.25,
    )
    part = unary_union([story_0, story_1]).buffer(0)
    payload = {
        "rooms": [
            {
                "story": 0,
                "locator_id": "building::tier-room::0",
                "floor": [
                    {
                        "corners": [
                            _pt(x, 0.0, z)
                            for x, z in story_0.exterior.coords[:-1]
                        ]
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            },
            {
                "story": 1,
                "locator_id": "building::tier-room::1",
                "floor": [
                    {
                        "corners": [
                            _pt(x, 3.0, z)
                            for x, z in story_1.exterior.coords[:-1]
                        ]
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            },
        ],
        "ceiling": [
            {
                "locator_id": "building::tier-ceiling-computed-oblique::left",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(2.0, 3.0, 2.0),
                    _pt(2.0, 3.0, 0.0),
                ],
                "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "computed_oblique",
            },
            {
                "locator_id": "building::tier-ceiling-computed-oblique::right",
                "corners": [
                    _pt(2.0, 3.0, 0.0),
                    _pt(2.0, 3.0, 2.0),
                    _pt(4.0, 2.0, 2.0),
                    _pt(4.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.5, "b": 1.0, "c": 0.0, "d": 4.0},
                "source": "computed_oblique",
            },
        ],
    }
    ceiling_faces = payload_adapter.payload_faces_from_tier_payload(
        payload,
        include=("ceiling",),
    )

    aligned = payload_adapter._ridge_aligned_part_footprint(
        payload,
        part,
        ceiling_faces,
        corner_tol=1e-6,
    )

    assert aligned is not None
    assert (
        aligned.symmetric_difference(target).area
        < part.symmetric_difference(target).area * 0.5
    )


def test_ridge_alignment_shift_gate_scales_with_ridge_span():
    short = payload_adapter.RidgeFrame(
        axis_deg=0.0,
        line_a=1.0,
        line_b=0.0,
        line_c=0.0,
        support_length_m=2.0,
    )
    long = payload_adapter.RidgeFrame(
        axis_deg=0.0,
        line_a=1.0,
        line_b=0.0,
        line_c=0.0,
        support_length_m=8.0,
    )

    assert math.isclose(payload_adapter._max_ridge_perp_shift_for_frame(short), 0.7)
    assert math.isclose(payload_adapter._max_ridge_perp_shift_for_frame(long), 2.8)


def test_payload_envelope_candidate_uses_roof_labeled_room_cells_as_primary(
    monkeypatch,
):
    payload = {
        "rooms": [],
        "ceiling": [
            {
                "locator_id": "building::tier-ceiling-flat::low",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(4.0, 2.0, 2.0),
                    _pt(4.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "flat_ceiling",
            },
            {
                "locator_id": "building::tier-ceiling-flat::high",
                "corners": [
                    _pt(4.0, 3.0, 0.0),
                    _pt(4.0, 3.0, 2.0),
                    _pt(6.0, 3.0, 2.0),
                    _pt(6.0, 3.0, 0.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 3.0},
                "source": "flat_ceiling",
            },
        ],
    }
    for room_index, (x0, x1) in enumerate([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]):
        payload["rooms"].append(
            {
                "story": 0,
                "locator_id": f"building::tier-room::{room_index}",
                "floor": [
                    {
                        "corners": [
                            _pt(x0, 0.0, 0.0),
                            _pt(x1, 0.0, 0.0),
                            _pt(x1, 0.0, 2.0),
                            _pt(x0, 0.0, 2.0),
                        ],
                        "holes": [],
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            }
        )

    whole = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)])
    monkeypatch.setattr(
        payload_adapter,
        "decompose_to_wings",
        lambda _footprint: [Wing(0, whole, whole.area, "main")],
    )
    monkeypatch.setattr(
        payload_adapter,
        "decompose_to_wings_v2",
        lambda _footprint, **_kwargs: [Wing(0, whole, whole.area, "main")],
    )

    candidates = payload_envelope_candidates_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.9,
    )
    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert [candidate.locator_id for candidate in candidates] == [
        "envelope-cell-selector:0",
        "envelope-cell-selector:1",
    ]
    assert all(candidate.selector == "cell-selector" for candidate in candidates)
    assert "low" in candidates[0].top_source
    assert "high" in candidates[1].top_source
    assert len(built) == 2
    assert all(poly.is_watertight() for _candidate, poly in built)


def test_roof_labeled_room_cells_keep_flat_ceilings_under_one_gable():
    payload = {"rooms": [], "ceiling": []}
    for room_index, (x0, x1) in enumerate([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]):
        payload["rooms"].append(
            {
                "story": 0,
                "locator_id": f"building::tier-room::{room_index}",
                "floor": [
                    {
                        "corners": [
                            _pt(x0, 0.0, 0.0),
                            _pt(x1, 0.0, 0.0),
                            _pt(x1, 0.0, 2.0),
                            _pt(x0, 0.0, 2.0),
                        ],
                        "holes": [],
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            }
        )
        payload["ceiling"].append(
            {
                "locator_id": f"building::tier-ceiling-flat::{room_index}",
                "corners": [
                    _pt(x0, 2.0 + room_index * 0.1, 0.0),
                    _pt(x0, 2.0 + room_index * 0.1, 2.0),
                    _pt(x1, 2.0 + room_index * 0.1, 2.0),
                    _pt(x1, 2.0 + room_index * 0.1, 0.0),
                ],
                "plane": {
                    "a": 0.0,
                    "b": 1.0,
                    "c": 0.0,
                    "d": 2.0 + room_index * 0.1,
                },
                "source": "flat_ceiling",
            }
        )

    payload["ceiling"].extend(
        [
            {
                "locator_id": "building::tier-ceiling-computed-oblique::left",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(3.0, 3.5, 2.0),
                    _pt(3.0, 3.5, 0.0),
                ],
                "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "computed_oblique",
            },
            {
                "locator_id": "building::tier-ceiling-computed-oblique::right",
                "corners": [
                    _pt(3.0, 3.5, 0.0),
                    _pt(3.0, 3.5, 2.0),
                    _pt(6.0, 2.0, 2.0),
                    _pt(6.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.5, "b": 1.0, "c": 0.0, "d": 5.0},
                "source": "computed_oblique",
            },
        ]
    )

    faces = payload_faces_from_tier_payload(payload)
    ceiling_faces = [face for face in faces if face.kind == "ceiling"]
    footprint = payload_adapter._payload_footprint_polygon(
        payload,
        room_buffer_m=0.0,
        footprint_shrink_m=0.0,
        corner_tol=0.02,
    )

    parts = payload_adapter._payload_roof_labeled_room_part_polygons(
        payload,
        footprint=footprint,
        ceiling_faces=ceiling_faces,
        corner_tol=0.02,
    )

    assert len(parts) == 1
    assert math.isclose(parts[0].area, 12.0)


def test_payload_envelope_candidate_builds_multi_piece_stepped_top():
    payload = _cube_payload()
    payload["ceiling"] = [
        {
            "locator_id": "building::tier-ceiling-flat::low-a",
            "corners": [
                _pt(0.0, 2.0, 0.0),
                _pt(0.0, 2.0, 1.0),
                _pt(1.0, 2.0, 1.0),
                _pt(1.0, 2.0, 0.0),
            ],
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
            "source": "flat_ceiling",
        },
        {
            "locator_id": "building::tier-ceiling-flat::high",
            "corners": [
                _pt(1.0, 3.0, 0.0),
                _pt(1.0, 3.0, 2.0),
                _pt(2.0, 3.0, 2.0),
                _pt(2.0, 3.0, 0.0),
            ],
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 3.0},
            "source": "flat_ceiling",
        },
        {
            "locator_id": "building::tier-ceiling-flat::low-b",
            "corners": [
                _pt(0.0, 2.0, 1.0),
                _pt(0.0, 2.0, 2.0),
                _pt(1.0, 2.0, 2.0),
                _pt(1.0, 2.0, 1.0),
            ],
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
            "source": "flat_ceiling",
        },
    ]

    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=False,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert len(built) == 1
    candidate, poly = built[0]
    assert "high" in candidate.top_source
    assert len(poly.faces) == 8
    assert len(poly.vertices) == 12
    assert len(poly.half_edges) == 36
    assert poly.is_watertight()
    ys = sorted(round(poly.vertex_position(vertex)[1], 6) for vertex in poly.vertices)
    assert ys.count(2.0) == 4
    assert ys.count(3.0) == 4


def test_payload_envelope_candidate_uses_room_graph_parts_as_fallback():
    payload = {
        "rooms": [
            {
                "story": 0,
                "locator_id": "building::tier-room::0",
                "floor": [
                    {
                        "corners": [
                            _pt(0.0, 0.0, 0.0),
                            _pt(3.0, 0.0, 0.0),
                            _pt(3.0, 0.0, 2.0),
                            _pt(0.0, 0.0, 2.0),
                        ],
                        "holes": [],
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            },
            {
                "story": 0,
                "locator_id": "building::tier-room::1",
                "floor": [
                    {
                        "corners": [
                            _pt(3.0, 0.0, 0.0),
                            _pt(6.0, 0.0, 0.0),
                            _pt(6.0, 0.0, 2.0),
                            _pt(3.0, 0.0, 2.0),
                        ],
                        "holes": [],
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            },
        ],
        "ceiling": [
            {
                "locator_id": "building::tier-ceiling-flat::room-0",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(3.0, 2.0, 2.0),
                    _pt(3.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "flat_ceiling",
            }
        ],
    }

    candidates = payload_envelope_candidates_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.6,
    )
    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.6,
        coord_tol=1e-6,
    )

    assert len(candidates) == 1
    assert candidates[0].locator_id.startswith("envelope-cell-selector:")
    assert candidates[0].selector == "cell-selector"
    assert len(built) == 1
    _candidate, poly = built[0]
    assert poly.is_watertight()


def test_payload_envelope_candidate_uses_room_graph_fallback_for_uncovered_parts(
    monkeypatch,
):
    payload = {
        "rooms": [
            {
                "story": 0,
                "locator_id": "building::tier-room::0",
                "floor": [
                    {
                        "corners": [
                            _pt(0.0, 0.0, 0.0),
                            _pt(3.0, 0.0, 0.0),
                            _pt(3.0, 0.0, 2.0),
                            _pt(0.0, 0.0, 2.0),
                        ],
                        "holes": [],
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            },
            {
                "story": 0,
                "locator_id": "building::tier-room::1",
                "floor": [
                    {
                        "corners": [
                            _pt(3.0, 0.0, 0.0),
                            _pt(6.0, 0.0, 0.0),
                            _pt(6.0, 0.0, 2.0),
                            _pt(3.0, 0.0, 2.0),
                        ],
                        "holes": [],
                    }
                ],
                "walls": [],
                "doors": [],
                "windows": [],
            },
        ],
        "ceiling": [
            {
                "locator_id": "building::tier-ceiling-flat::left",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(3.0, 2.0, 2.0),
                    _pt(3.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "flat_ceiling",
            },
            {
                "locator_id": "building::tier-ceiling-flat::right",
                "corners": [
                    _pt(3.0, 2.5, 0.0),
                    _pt(3.0, 2.5, 2.0),
                    _pt(6.0, 2.5, 2.0),
                    _pt(6.0, 2.5, 0.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.5},
                "source": "flat_ceiling",
            },
        ],
    }
    whole = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)])
    left = Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)])
    right = Polygon([(3.0, 0.0), (6.0, 0.0), (6.0, 2.0), (3.0, 2.0)])
    monkeypatch.setattr(
        payload_adapter,
        "decompose_to_wings",
        lambda _footprint: [Wing(0, whole, whole.area, "main")],
    )
    monkeypatch.setattr(
        payload_adapter,
        "decompose_to_wings_v2",
        lambda _footprint, **_kwargs: [
            Wing(0, left, left.area, "main"),
            Wing(1, right, right.area, "extension"),
        ],
    )

    candidates = payload_envelope_candidates_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.9,
    )
    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.9,
        coord_tol=1e-6,
    )

    assert [candidate.locator_id for candidate in candidates] == [
        "envelope-cell-selector:0",
        "envelope-cell-selector:1",
    ]
    assert all(candidate.selector == "cell-selector" for candidate in candidates)
    assert "left" in candidates[0].top_source
    assert "right" in candidates[1].top_source
    assert len(built) == 2
    assert all(poly.is_watertight() for _candidate, poly in built)
