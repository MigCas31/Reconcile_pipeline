from __future__ import annotations

import pytest

from reconcile.element_locator import (
    build_trace,
    find_element,
    is_ontology_kind,
    is_tier_kind,
    parse_element_id,
)


def _sample_buildings() -> list[dict]:
    return [
        {
            "uuid": "11111111-2222-3333-4444-555555555555",
            "address": "Testvej 1",
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
                    "floor_overlap_region": [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0.5]],
                    "raw_ceiling_source": "scan",
                    "raw_ceiling_planes": [
                        {
                            "corners": [[0, 2, 0], [1, 2.1, 0], [1, 2.1, 1], [0, 2, 1]],
                        }
                    ],
                    "walls_merged": [
                        {
                            "id": "wm-1",
                            "source": "merged",
                            "corners": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                        }
                    ],
                    "walls_computed": [
                        {
                            "id": "wc-1",
                            "source": "scan-room",
                            "corners": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                        }
                    ],
                    "doors": [
                        {
                            "id": "door-1",
                            "source": "scan-room",
                            "corners": [[0, 0, 0], [0.5, 0, 0], [0.5, 1, 0]],
                        }
                    ],
                    "windows": [],
                }
            ],
            "cross_floor_gaps": [
                {
                    "id": "cg-1",
                    "type": "cross_story",
                    "confidence": "high",
                    "corners": [[0, 0, 0], [1, 0, 0], [1, 0.1, 0]],
                }
            ],
            "stitch_walls": [
                {
                    "id": "sw-1",
                    "story": 0,
                    "corners": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                }
            ],
            "gap_walls": [
                {
                    "id": "gw-1",
                    "type": "gap_wall",
                    "corners": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                }
            ],
            "exterior_gap_indicators": [
                {
                    "id": "eg-1",
                    "element_corners": [[0, 0, 0], [0.5, 0, 0], [0.5, 1, 0]],
                    "wall_corners": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                }
            ],
            "gap_closures": [
                {
                    "id": "gc-1",
                    "type": "side",
                    "corners": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                }
            ],
            "roof_surfaces": {
                "oblique": [
                    {
                        "corners": [[0, 2, 0], [2, 3, 0], [2, 3, 2], [0, 2, 2]],
                        "dominant_story": 1,
                        "cluster": {"avgAzimuth": 180, "avgIncl": 30},
                    }
                ],
                "flat": [
                    {
                        "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
                        "kind": "flat",
                        "story": 1,
                        "y": 3.0,
                    }
                ],
            },
            "ceiling": {
                "flat": [{"poly": [[0, 2.5, 0], [2, 2.5, 0], [2, 2.5, 2]], "story": 0}],
                "oblique": [
                    {
                        "poly": [[0, 2, 0], [1, 3, 0], [1, 3, 1]],
                        "kind": "clipped",
                        "story": 1,
                        "plane_index": 0,
                    }
                ],
                "simple_slant": [
                    {"poly": [[0, 2, 0], [1, 2.5, 0], [1, 2.5, 1]], "story": 0}
                ],
            },
        }
    ]


def test_parse_element_id_valid_expected():
    parsed = parse_element_id("11111111-2222-3333-4444-555555555555::door::door-1")
    assert parsed.building_uuid == "11111111-2222-3333-4444-555555555555"
    assert parsed.kind == "door"
    assert parsed.element_id == "door-1"


def test_parse_element_id_invalid_format_raises():
    with pytest.raises(ValueError):
        parse_element_id("bad-format")


def test_is_tier_kind_detects_static_tier_viewer_ids():
    assert is_tier_kind("tier-knee-wall")
    assert not is_tier_kind("ontology-knee-wall")


def test_find_element_tier_payload_knee_wall():
    payload = {
        "uuid": "11111111-2222-3333-4444-555555555555",
        "address": "Testvej 1",
        "rooms": [],
        "gaps": [],
        "ceiling": [],
        "knee_walls": [
            {
                "locator_id": "11111111-2222-3333-4444-555555555555::tier-knee-wall::0",
                "kind": "knee",
                "corners": [
                    {"x": 0, "y": 1, "z": 0},
                    {"x": 1, "y": 1, "z": 0},
                    {"x": 1, "y": 2, "z": 0},
                ],
            }
        ],
    }

    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::tier-knee-wall::0",
        tier_payloads={"11111111-2222-3333-4444-555555555555": payload},
    )

    assert result["json_path"] == "knee_walls[0]"
    assert result["building_address"] == "Testvej 1"
    assert result["source"] == "knee"
    assert result["corners_count"] == 3


def test_find_element_tier_payload_wall_extension():
    wall_locator = "11111111-2222-3333-4444-555555555555::tier-wall::0:wall-a"
    payload = {
        "uuid": "11111111-2222-3333-4444-555555555555",
        "address": None,
        "rooms": [
            {
                "story": 0,
                "locator_id": "11111111-2222-3333-4444-555555555555::tier-room::0",
                "floor": {"corners": []},
                "walls": [
                    {
                        "locator_id": wall_locator,
                        "corners": [],
                        "extension_strip": [
                            {"x": 0, "y": 1, "z": 0},
                            {"x": 1, "y": 1, "z": 0},
                            {"x": 1, "y": 2, "z": 0},
                        ],
                    }
                ],
            }
        ],
        "gaps": [],
        "ceiling": [],
        "knee_walls": [],
    }

    result = find_element(
        [],
        f"{wall_locator}:extension",
        tier_payloads={"11111111-2222-3333-4444-555555555555": payload},
    )

    assert result["json_path"] == "rooms[0].walls[0].extension_strip"
    assert result["room_index"] == 0
    assert result["story"] == 0
    assert result["corners_count"] == 3


def test_find_element_room_surface_expected_details():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::wall-computed::wc-1",
    )
    assert result["json_path"] == "rooms[0].walls_computed[0]"
    assert result["story"] == 0
    assert result["source"] == "scan-room"


def test_find_element_wall_computed_full_model_suffix():
    # Full-model viewer appends :<story>:<room_index> to wall IDs (viewer-main.js:1924).
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::wall-computed::wc-1:0:0",
    )
    assert result["json_path"] == "rooms[0].walls_computed[0]"
    assert result["story"] == 0


def test_find_element_wall_computed_suffix_story_mismatch_raises():
    # story hint 1 doesn't match the room's story 0 — should not resolve.
    with pytest.raises(LookupError):
        find_element(
            _sample_buildings(),
            "11111111-2222-3333-4444-555555555555::wall-computed::wc-1:1:0",
        )


def test_find_element_gap_collection_expected_details():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::gap-cross-story::cg-1",
    )
    assert result["json_path"] == "cross_floor_gaps[0]"
    assert result["kind"] == "gap-cross-story"


def test_find_element_gap_wall_viewer_fallback_counter_id():
    building = {
        "uuid": "11111111-2222-3333-4444-555555555555",
        "address": "Testvej 1",
        "rooms": [],
        "cross_floor_gaps": [
            {
                "type": "within_story",
                "confidence": "medium",
                "story": 0,
                "corners": [[0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1]],
            }
        ],
        "gap_walls": [
            {
                "type": "gap_floor",
                "story": 0,
                "corners": [[0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1]],
            },
            {
                "type": "gap_ceiling",
                "story": 0,
                "corners": [[0, 1, 0], [2, 1, 0], [2, 1, 1], [0, 1, 1]],
            },
        ],
    }
    result = find_element(
        [building],
        "11111111-2222-3333-4444-555555555555::gap-wall::gap_ceiling:4",
    )
    assert result["json_path"] == "gap_walls[1]"
    assert result["story"] == 0
    assert result["corners_count"] == 4


def test_find_element_gap_wall_stable_persisted_id():
    building = {
        "uuid": "11111111-2222-3333-4444-555555555555",
        "address": "Testvej 1",
        "rooms": [],
        "gap_walls": [
            {
                "id": "gw:gap:within_story:0:abcd1234ef567890:gap_ceiling:polygon",
                "type": "gap_ceiling",
                "story": 0,
                "corners": [[0, 1, 0], [2, 1, 0], [2, 1, 1], [0, 1, 1]],
            }
        ],
    }
    result = find_element(
        [building],
        "11111111-2222-3333-4444-555555555555::gap-wall::gw:gap:within_story:0:abcd1234ef567890:gap_ceiling:polygon",
    )
    assert result["json_path"] == "gap_walls[0]"
    assert result["story"] == 0
    assert result["corners_count"] == 4


def test_find_element_floor_locator_expected_details():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::floor::0:0",
    )
    assert result["json_path"] == "rooms[0].floor_polygon"
    assert result["corners_count"] == 3


def test_find_element_ceiling_raw():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::ceiling-raw::0:0:0",
    )
    assert result["json_path"] == "rooms[0].raw_ceiling_planes[0]"
    assert result["story"] == 0
    assert result["corners_count"] == 4


def test_find_element_thermal_ceiling_gap():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::thermal-ceiling::thermal:gap:0",
    )
    assert result["json_path"] == "cross_floor_gaps[0]"
    assert result["source"] == "cross_floor_gap"
    assert result["corners_count"] == 3


def test_find_element_roof_oblique():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::roof-oblique::oblique:0",
    )
    assert result["json_path"] == "roof_surfaces.oblique[0]"
    assert result["corners_count"] == 4
    assert result["story"] == 1


def test_find_element_roof_flat():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::roof-flat::flat:0",
    )
    assert result["json_path"] == "roof_surfaces.flat[0]"
    assert result["corners_count"] == 4


def test_find_element_ceiling_flat():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::ceiling-flat::ceiling-flat:0",
    )
    assert result["json_path"] == "ceiling.flat[0]"
    assert result["corners_count"] == 3


def test_find_element_ceiling_simple_slant():
    result = find_element(
        _sample_buildings(),
        "11111111-2222-3333-4444-555555555555::ceiling-simple-slant::ceiling-slant:0",
    )
    assert result["json_path"] == "ceiling.simple_slant[0]"
    assert result["corners_count"] == 3


def test_find_element_roof_invalid_index_raises():
    with pytest.raises(LookupError):
        find_element(
            _sample_buildings(),
            "11111111-2222-3333-4444-555555555555::roof-oblique::oblique:99",
        )


def test_find_element_roof_oblique_falls_back_to_roof_results():
    buildings = [
        {
            "uuid": "11111111-2222-3333-4444-555555555555",
            "address": "Testvej 1",
            "rooms": [],
        }
    ]
    result = find_element(
        buildings,
        "11111111-2222-3333-4444-555555555555::roof-oblique::oblique:0",
        roof_results=_sample_roof_surface_results(),
    )
    assert result["json_path"] == "roof_surfaces.oblique[0]"
    assert result["corners_count"] == 4
    assert result["story"] == 1


def test_find_element_ceiling_oblique_falls_back_to_roof_results():
    buildings = [
        {
            "uuid": "11111111-2222-3333-4444-555555555555",
            "address": "Testvej 1",
            "rooms": [],
        }
    ]
    result = find_element(
        buildings,
        "11111111-2222-3333-4444-555555555555::ceiling-oblique::ceiling-oblique:0",
        roof_results=_sample_roof_surface_results(),
    )
    assert result["json_path"] == "ceiling.oblique[0]"
    assert result["corners_count"] == 3
    assert result["story"] == 1


def test_find_element_missing_id_raises():
    with pytest.raises(LookupError):
        find_element(
            _sample_buildings(),
            "11111111-2222-3333-4444-555555555555::door::nope",
        )


def _sample_roof_results() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        uuid: {
            "ceiling_partitions": {
                "oblique": [
                    {
                        "id": "ceiling-partition:abc123abc123abc123ab",
                        "room_index": 2,
                        "story": 1,
                        "kind": "oblique",
                        "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                        "poly": [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
                        "area_m2": 0.5,
                    }
                ],
                "flat": [
                    {
                        "id": "ceiling-partition:ff1122ff1122ff1122ff",
                        "room_index": 3,
                        "story": 1,
                        "kind": "flat",
                        "flat_role": "roof_flat",
                        "roof_hypothesis_id": "roof-hypothesis:flat:0",
                        "poly": [[0, 3.3, 0], [2, 3.3, 0], [2, 3.3, 2], [0, 3.3, 2]],
                        "area_m2": 4.0,
                        "top_y_m": 3.3,
                    }
                ],
                "room_partitions": [
                    {
                        "room_index": 2,
                        "story": 1,
                        "partitions": [
                            {
                                "id": "ceiling-partition:abc123abc123abc123ab",
                                "kind": "oblique",
                            }
                        ],
                    }
                ],
            },
            "knee_walls": [
                {
                    "id": "knee-wall:deadbeefcafebabe1234",
                    "room_index": 2,
                    "height_m": 0.6,
                    "corners": [[0, 0, 0], [1, 0, 0], [1, 0.6, 0], [0, 0.6, 0]],
                }
            ],
            "roof_evidence_graph": {
                "atom_evidence": {
                    "ceiling-partition:abc123abc123abc123ab": {
                        "atom_id": "ceiling-partition:abc123abc123abc123ab",
                        "semantic_kinds": ["gable_run"],
                        "subpart_ids": ["coverage-subpart:xyz"],
                    }
                }
            },
            "top_boundary_graph": {
                "nodes": [
                    {"id": "ceiling-partition:abc123abc123abc123ab"},
                ]
            },
            "roof_cell_complex": {
                "cells": [
                    {
                        "id": "roof-cell:attic:c0b7aeb9a909c05ebd10",
                        "type": "Cell",
                        "cell_kind": "attic",
                        "story": 0,
                        "room_id": "room:10",
                        "room_index": 10,
                        "part_id": "building-part:abc",
                        "base_atom_id": "ceiling-partition:ff1122ff1122ff1122ff",
                        "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                        "roof_surface_kind": "oblique",
                        "volume_m3": 11.1,
                        "bbox_xyz": [0.5, 3.5, -3.8, 5.2, 6.7, 0.7],
                        "faces": [
                            {
                                "id": "arr-face:bottomslab0000000000",
                                "kind": "bottom",
                                "role": "slab",
                                "source_kind": "bottom_cap",
                                "corners": [
                                    [0, 3.5, 0],
                                    [5, 3.5, 0],
                                    [5, 3.5, 4],
                                    [0, 3.5, 4],
                                ],
                            },
                            {
                                "id": "arr-face:5b790a478078551d5568",
                                "kind": "top",
                                "role": "roof",
                                "source_kind": "oblique",
                                "corners": [
                                    [1.9, 6.7, 0.7],
                                    [0.5, 6.7, -2.2],
                                    [3.9, 4.0, -3.8],
                                    [5.2, 4.1, -0.9],
                                ],
                                "area_m2": 14.4,
                                "metadata": {
                                    "face_kind": "top",
                                    "roof_hypothesis_id": "roof-hypothesis:oblique:1",
                                },
                            },
                        ],
                    }
                ],
                "edges": [],
                "knee_walls": [],
                "metadata": {},
            },
        }
    }


def _sample_roof_surface_results() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        uuid: {
            "roof_surfaces": {
                "oblique": [
                    {
                        "corners": [[0, 2, 0], [2, 3, 0], [2, 3, 2], [0, 2, 2]],
                        "dominant_story": 1,
                        "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                    }
                ],
                "flat": [
                    {
                        "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
                        "story": 1,
                        "roof_hypothesis_id": "roof-hypothesis:flat:0",
                    }
                ],
            },
            "ceiling": {
                "flat": [
                    {
                        "poly": [[0, 2.5, 0], [2, 2.5, 0], [2, 2.5, 2], [0, 2.5, 2]],
                        "story": 0,
                    }
                ],
                "oblique": [
                    {
                        "poly": [[0, 2, 0], [1, 3, 0], [1, 3, 1]],
                        "story": 1,
                        "plane_index": 0,
                    }
                ],
                "simple_slant": [
                    {
                        "poly": [[0, 2, 0], [1, 2.5, 0], [1, 2.5, 1]],
                        "story": 0,
                    }
                ],
            },
        }
    }


def _sample_raw_ceiling_plane_splits() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        "available": True,
        "buildings": {
            uuid: [
                {
                    "uuid": uuid,
                    "story": 1,
                    "target_element_id": f"{uuid}::roof-oblique::oblique:0",
                    "target_kind": "committed_oblique",
                    "piece_id": f"{uuid}::roof-oblique::oblique:0#supported:0:0",
                    "piece_role": "supported",
                    "support_score": 0.92,
                    "chain_ids": [f"{uuid}::eave-chain::1:0"],
                    "corners": [[0, 2, 0], [2, 2, 0], [2, 3, 1], [0, 3, 1]],
                    "holes": [],
                }
            ]
        },
    }


def _sample_raw_ceiling_plane_splits_v2() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        "available": True,
        "buildings": {
            uuid: [
                {
                    "uuid": uuid,
                    "story": 1,
                    "target_element_id": (
                        f"{uuid}::ridge-eave-candidate::plane-group::v2abc"
                    ),
                    "target_kind": "ridge_eave_plane_group",
                    "piece_id": (
                        f"{uuid}::ridge-eave-candidate::plane-group::"
                        f"v2abc#supported:0:0"
                    ),
                    "piece_role": "supported",
                    "support_score": 0.97,
                    "chain_ids": [f"{uuid}::eave-chain::1:4"],
                    "source_edge_ids": [f"{uuid}::edge::42"],
                    "corners": [[0, 2, 0], [2, 2, 0], [2, 2.7, 1], [0, 2.7, 1]],
                    "holes": [],
                }
            ]
        },
    }


def _sample_candidate_faces() -> list[dict]:
    uuid = "11111111-2222-3333-4444-555555555555"
    return [
        {
            "building_uuid": uuid,
            "faces": [
                {
                    "id": f"{uuid}::candidate::segment-0:room-0:piece-0:seg-0",
                    "room_index": 0,
                    "story": 0,
                    "footprint_xz": [[0, 0], [1, 0], [1, 1], [0, 1]],
                }
            ],
        }
    ]


def _sample_reconstruction() -> list[dict]:
    uuid = "11111111-2222-3333-4444-555555555555"
    return [
        {
            "building_uuid": uuid,
            "status": "solved",
            "decision": "auto_accept",
            "selected_face_ids": [f"{uuid}::candidate::segment-0:room-0:piece-0:seg-0"],
        }
    ]


def _sample_ridge_eave_scores() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        "buildings": [
            {
                "building_uuid": uuid,
                "candidates": [
                    {
                        "id": f"{uuid}::candidate::segment-0:room-0:piece-0:seg-0",
                        "best_score": 0.8,
                    }
                ],
                "plane_groups": [
                    {
                        "id": f"{uuid}::plane-group::abc123",
                        "best_score": 0.9,
                        "union_xz": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
            }
        ]
    }


def _sample_overextend() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        "buildings": {
            uuid: [
                {
                    "surface_kind": "oblique",
                    "surface_index": 0,
                    "story": 1,
                    "corners": [[0, 3, 0], [1, 3, 0], [1, 3, 1]],
                }
            ]
        }
    }


def _sample_raw_disagreement() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        "buildings": {
            uuid: [
                {
                    "pair_index": 7,
                    "story": 0,
                    "corners": [[0, 2, 0], [1, 2, 0], [1, 2, 1]],
                }
            ]
        }
    }


def _sample_clean_ceiling() -> dict:
    uuid = "11111111-2222-3333-4444-555555555555"
    return {
        "buildings": {
            uuid: [
                {
                    "element_id": f"{uuid}::clean-ceiling::0:0",
                    "story": 0,
                    "room_index": 0,
                    "replacement_mode": "single-oblique",
                    "corners": [[0, 2.3, 0], [1, 2.3, 0], [1, 2.3, 1]],
                }
            ]
        }
    }


def test_find_candidate_face():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::candidate-face::segment-0:room-0:piece-0:seg-0",
        candidate_faces=_sample_candidate_faces(),
    )
    assert result["kind"] == "candidate-face"
    assert result["story"] == 0
    assert result["corners_count"] == 4


def test_find_reconstruction_face():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::reconstruction-face::segment-0:room-0:piece-0:seg-0",
        reconstruction=_sample_reconstruction(),
        candidate_faces=_sample_candidate_faces(),
    )
    assert result["kind"] == "reconstruction-face"
    assert result["source"] == "auto_accept"


def test_find_ridge_eave_candidate_candidate_row():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::ridge-eave-candidate::segment-0:room-0:piece-0:seg-0",
        ridge_eave_scores=_sample_ridge_eave_scores(),
    )
    assert result["kind"] == "ridge-eave-candidate"
    assert result["source"] == "scored-candidate"


def test_find_ridge_eave_candidate_plane_group_row():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::ridge-eave-candidate::plane-group::abc123::below-ridge",
        ridge_eave_scores=_sample_ridge_eave_scores(),
    )
    assert result["kind"] == "ridge-eave-candidate"
    assert result["source"] == "plane-group"


def test_find_roof_overextend():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::roof-overextend::oblique:0",
        computed_overextend=_sample_overextend(),
    )
    assert result["kind"] == "roof-overextend"
    assert result["story"] == 1


def test_find_raw_disagreement():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::raw-disagreement::7",
        raw_disagreement=_sample_raw_disagreement(),
    )
    assert result["kind"] == "raw-disagreement"
    assert result["story"] == 0


def test_find_clean_ceiling():
    result = find_element(
        [],
        "11111111-2222-3333-4444-555555555555::clean-ceiling::0:0",
        ceiling_replacement=_sample_clean_ceiling(),
    )
    assert result["kind"] == "clean-ceiling"
    assert result["source"] == "single-oblique"


def test_is_ontology_kind():
    assert is_ontology_kind("ontology-renderable-ceiling")
    assert is_ontology_kind("ontology-base-exterior-wall")
    assert is_ontology_kind("ontology-knee-wall")
    assert not is_ontology_kind("ceiling-flat")
    assert not is_ontology_kind("wall-merged")


def test_find_raw_eave_split_piece():
    uuid = "11111111-2222-3333-4444-555555555555"
    token = f"{uuid}::raw-eave-split::{uuid}::roof-oblique::oblique:0#supported:0:0"
    result = find_element(
        [],
        token,
        raw_ceiling_plane_splits=_sample_raw_ceiling_plane_splits(),
    )
    assert result["kind"] == "raw-eave-split"
    assert result["json_path"] == f"buildings[{uuid}][0]"
    assert result["story"] == 1
    assert result["source"] == "supported:committed_oblique"
    assert result["corners_count"] == 4


def test_find_raw_eave_split_v1_piece():
    uuid = "11111111-2222-3333-4444-555555555555"
    token = f"{uuid}::raw-eave-split-v1::{uuid}::roof-oblique::oblique:0#supported:0:0"
    result = find_element(
        [],
        token,
        raw_ceiling_plane_splits=_sample_raw_ceiling_plane_splits(),
    )
    assert result["kind"] == "raw-eave-split-v1"
    assert result["json_path"] == f"buildings[{uuid}][0]"
    assert result["story"] == 1


def test_find_raw_eave_split_v2_piece():
    uuid = "11111111-2222-3333-4444-555555555555"
    token = (
        f"{uuid}::raw-eave-split-v2::"
        f"{uuid}::ridge-eave-candidate::plane-group::v2abc#supported:0:0"
    )
    result = find_element(
        [],
        token,
        raw_ceiling_plane_splits_v2=_sample_raw_ceiling_plane_splits_v2(),
    )
    assert result["kind"] == "raw-eave-split-v2"
    assert result["json_path"] == f"buildings[{uuid}][0]"
    assert result["source"] == "supported:ridge_eave_plane_group"


def test_find_ontology_renderable_ceiling_atom():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-ceiling"
        "::renderable:room_ceiling_sloped:ceiling-partition:abc123abc123abc123ab"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    assert result["kind"] == "ontology-renderable-ceiling"
    assert result["category"] == "room_ceiling_sloped"
    assert result["source_id"] == "ceiling-partition:abc123abc123abc123ab"
    assert result["atom"]["room_index"] == 2
    assert result["atom"]["kind"] == "oblique"
    assert "ceiling_partitions.oblique[0]" in result["provenance_paths"]
    assert (
        "ceiling_partitions.room_partitions[0].partitions[0]"
        in result["provenance_paths"]
    )
    assert result["evidence"]["semantic_kinds"] == ["gable_run"]


def test_find_ontology_knee_wall_atom():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-knee-wall"
        "::renderable:knee_wall:knee-wall:deadbeefcafebabe1234"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    assert result["source_id"] == "knee-wall:deadbeefcafebabe1234"
    assert result["atom"]["height_m"] == 0.6
    assert "knee_walls[0]" in result["provenance_paths"]


def test_find_ontology_missing_source_raises():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-ceiling"
        "::renderable:room_ceiling_sloped:ceiling-partition:does_not_exist"
    )
    with pytest.raises(LookupError):
        find_element([], token, roof_results=_sample_roof_results())


def test_find_ontology_missing_roof_results_raises():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-ceiling"
        "::renderable:room_ceiling_sloped:ceiling-partition:abc"
    )
    with pytest.raises(LookupError):
        find_element([], token)


def test_find_ontology_bad_renderable_format_raises():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-ceiling"
        "::not_a_renderable_id"
    )
    with pytest.raises(LookupError):
        find_element([], token, roof_results=_sample_roof_results())


def test_find_ontology_roof_atom_patch_flat():
    """roof-atom-patch:flat: prefix is stripped; underlying atom resolves."""
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-roof"
        "::renderable:exterior_roof:roof-atom-patch:flat:ceiling-partition:ff1122ff1122ff1122ff"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    assert result["kind"] == "ontology-renderable-roof"
    assert result["category"] == "exterior_roof"
    assert (
        result["source_id"]
        == "roof-atom-patch:flat:ceiling-partition:ff1122ff1122ff1122ff"
    )
    assert result["atom"]["id"] == "ceiling-partition:ff1122ff1122ff1122ff"
    assert result["atom"]["flat_role"] == "roof_flat"
    assert "ceiling_partitions.flat[0]" in result["provenance_paths"]


def test_find_ontology_roof_atom_patch_trace_points_at_roof_partitioning():
    """build_trace resolves pipeline_step to roof_partitioning.py, not viewer_server."""
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-roof"
        "::renderable:exterior_roof:roof-atom-patch:flat:ceiling-partition:ff1122ff1122ff1122ff"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    traced = build_trace(result)
    assert traced["pipeline_step"]["file"].endswith("roof_partitioning.py")


def test_find_ontology_roof_cell_face_composite():
    """Resolves <cell_id>:arr-face:<face_hash> to the face dict and parent cell."""
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-roof"
        "::renderable:exterior_roof:roof-cell:attic:c0b7aeb9a909c05ebd10:arr-face:5b790a478078551d5568"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    assert result["source_id"] == (
        "roof-cell:attic:c0b7aeb9a909c05ebd10:arr-face:5b790a478078551d5568"
    )
    assert result["atom"]["id"] == "arr-face:5b790a478078551d5568"
    assert result["atom"]["role"] == "roof"
    assert result["atom"]["source_kind"] == "oblique"
    assert "roof_cell_complex.cells[0].faces[1]" in result["provenance_paths"]
    parent = result["parent_cell"]
    assert parent["id"] == "roof-cell:attic:c0b7aeb9a909c05ebd10"
    assert parent["room_index"] == 10
    assert parent["cell_kind"] == "attic"
    assert parent["roof_hypothesis_id"] == "roof-hypothesis:oblique:1"


def test_find_ontology_roof_cell_bare_resolves_to_cell():
    """Bare <cell_id> (no face suffix) resolves to the cell itself."""
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-roof"
        "::renderable:exterior_roof:roof-cell:attic:c0b7aeb9a909c05ebd10"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    assert result["atom"]["id"] == "roof-cell:attic:c0b7aeb9a909c05ebd10"
    assert result["atom"]["cell_kind"] == "attic"
    assert result["provenance_paths"] == ["roof_cell_complex.cells[0]"]
    assert result["parent_cell"]["room_index"] == 10


def test_find_ontology_roof_cell_face_missing_face_raises():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-roof"
        "::renderable:exterior_roof:roof-cell:attic:c0b7aeb9a909c05ebd10:arr-face:deadbeefdeadbeefdead"
    )
    with pytest.raises(LookupError):
        find_element([], token, roof_results=_sample_roof_results())


def test_find_ontology_roof_cell_trace_points_at_roof_cell_complex():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-roof"
        "::renderable:exterior_roof:roof-cell:attic:c0b7aeb9a909c05ebd10:arr-face:5b790a478078551d5568"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    traced = build_trace(result)
    assert traced["pipeline_step"]["file"].endswith("roof_cell_complex.py")


def test_build_trace_attaches_thresholds_and_step():
    token = (
        "11111111-2222-3333-4444-555555555555"
        "::ontology-renderable-ceiling"
        "::renderable:room_ceiling_sloped:ceiling-partition:abc123abc123abc123ab"
    )
    result = find_element([], token, roof_results=_sample_roof_results())
    traced = build_trace(result)
    assert traced["pipeline_step"]["file"].endswith("roof_partitioning.py")
    assert any("ROOM_TOP_MIN_CLEARANCE_M" in t["note"] for t in traced["thresholds"])
