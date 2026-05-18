from __future__ import annotations

import math

from reconcile_v3.analysis import advanced_features as advf
from reconcile_v3.analysis import context_features as ctxf
from reconcile_v3.analysis import exhaustive_features as exf
from reconcile_v3.analysis import feature_expansion as fe
from reconcile_v3.analysis.v3_context import _cache_is_compatible


def _record() -> dict:
    plane = [0.0, math.cos(math.radians(35.0)), math.sin(math.radians(35.0)), -1.0]
    return {
        "building_uuid": "b",
        "proposal_id": "b::v3-merged-roof-segment::seg-0",
        "cluster_canonical_id": "cluster-0",
        "label": "accepted",
        "heuristic_label": "accepted",
        "merged_plane": plane,
        "segment_corners_xyz": [
            [-2.0, 4.0, -1.0],
            [2.0, 4.0, -1.0],
            [2.0, 2.0, 3.0],
            [-2.0, 2.0, 3.0],
        ],
        "building_boundary_xz": [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]],
        "opposing_planes": [
            [0.0, math.cos(math.radians(35.0)), -math.sin(math.radians(35.0)), -1.0]
        ],
        "opposing_cluster_canonicals": ["cluster-1"],
        "side_pieces": [],
        "cluster_params": {"confidence": 0.92, "algorithm_version": "v3"},
        "room_boundary_refs": [
            {
                "kind": "room",
                "room_id": "room:1",
                "room_index": 1,
                "story": 1,
                "piece_index": 0,
            }
        ],
        "features_snapshot": {
            "area_m2": 16.0,
            "perimeter_m": 16.0,
            "member_count": 2,
            "opposing_cluster_count": 1,
            "piece_kind": "room",
            "rain_hitting_side_count": 1,
            "covered_side_count": 1,
            "clipped_by_building_boundary": False,
        },
        "cluster_members": [
            {
                "id": "m0",
                "source_room_id": "room:1",
                "source_wall_id": "w0",
                "slab_room_id": "room:1",
                "heuristic_label": "accepted",
                "plane": plane,
                "corners": [
                    [-2.0, 4.0, -1.0],
                    [2.0, 4.0, -1.0],
                    [2.0, 2.0, 3.0],
                    [-2.0, 2.0, 3.0],
                ],
                "features": {
                    "segment_story": 1,
                    "segment_azimuth_deg": 0.0,
                    "segment_incl_deg": 35.0,
                    "segment_length_m": 4.0,
                    "segment_mid_y_m": 3.0,
                    "plane_height_above_slab_m": 2.2,
                    "plane_y_at_piece_centroid_m": 3.0,
                    "piece_area_m2": 16.0,
                    "piece_perimeter_m": 16.0,
                    "piece_compactness": 0.7,
                    "piece_bbox_aspect": 1.0,
                    "piece_min_width_m": 4.0,
                    "piece_vertex_count": 4,
                    "rain_exposure_ratio": 1.0,
                    "slant_delta_over_piece_m": 2.0,
                    "seg_mid_to_piece_centroid_xz_m": 0.0,
                    "slab_area_m2": 16.0,
                    "slab_floor_y_m": 1.0,
                    "slab_vertex_count": 4,
                    "story_delta": 0.0,
                },
                "trace": {"rule": "seed-a"},
            },
            {
                "id": "m1",
                "source_room_id": "room:1",
                "source_wall_id": "w1",
                "slab_room_id": "room:1",
                "heuristic_label": "accepted",
                "plane": [
                    0.0,
                    math.cos(math.radians(36.0)),
                    math.sin(math.radians(36.0)),
                    -1.02,
                ],
                "corners": [
                    [-2.0, 4.0, -1.0],
                    [2.0, 4.0, -1.0],
                    [2.0, 2.0, 3.0],
                    [-2.0, 2.0, 3.0],
                ],
                "features": {
                    "segment_story": 1,
                    "segment_azimuth_deg": 2.0,
                    "segment_incl_deg": 36.0,
                    "segment_length_m": 4.0,
                    "segment_mid_y_m": 3.0,
                    "plane_height_above_slab_m": 2.1,
                    "plane_y_at_piece_centroid_m": 3.0,
                    "piece_area_m2": 16.0,
                    "piece_perimeter_m": 16.0,
                    "piece_compactness": 0.69,
                    "piece_bbox_aspect": 1.0,
                    "piece_min_width_m": 4.0,
                    "piece_vertex_count": 4,
                    "rain_exposure_ratio": 1.0,
                    "slant_delta_over_piece_m": 2.0,
                    "seg_mid_to_piece_centroid_xz_m": 0.1,
                    "slab_area_m2": 16.0,
                    "slab_floor_y_m": 1.0,
                    "slab_vertex_count": 4,
                    "story_delta": 0.0,
                },
                "trace": {"rule": "seed-a"},
            },
        ],
    }


def _building_context() -> dict:
    return {
        "context_schema_version": 2,
        "parts": [
            {
                "id": "part-0",
                "room_ids": ["room:1"],
                "stories": [1],
                "footprint_xz": [
                    [-4.0, 0.0, -4.0],
                    [4.0, 0.0, -4.0],
                    [4.0, 0.0, 4.0],
                    [-4.0, 0.0, 4.0],
                ],
                "gable_extension": {
                    "status": "gable_complete",
                    "metrics": {"n_slanted_roofs": 2, "major_az": 0.0},
                    "ridge_line": [[-1.5, 4.0, -1.0], [1.5, 4.0, -1.0]],
                    "uncovered_region_xz": [],
                },
            }
        ],
        "wall_extensions": [
            {
                "id": "ext-0",
                "wall_id": "w0",
                "room_id": "room:1",
                "strip_corners": [
                    [-2.0, 0.0, -1.0],
                    [2.0, 0.0, -1.0],
                    [2.0, 0.0, -0.8],
                    [-2.0, 0.0, -0.8],
                ],
                "behind_knee_wall": True,
            }
        ],
        "dormers": [
            {
                "id": "d-0",
                "front_wall_id": "w0",
                "roof_surface_id": "roof-0",
                "corners": [
                    [-0.5, 3.0, 0.0],
                    [0.5, 3.0, 0.0],
                    [0.5, 3.2, 0.6],
                    [-0.5, 3.2, 0.6],
                ],
            }
        ],
        "slanted_roofs": [
            {
                "id": "roof-0",
                "plane": [
                    0.0,
                    math.cos(math.radians(35.0)),
                    math.sin(math.radians(35.0)),
                    -1.0,
                ],
                "corners": [
                    [-2.0, 4.0, -1.0],
                    [2.0, 4.0, -1.0],
                    [2.0, 2.0, 3.0],
                    [-2.0, 2.0, 3.0],
                ],
            }
        ],
        "merged_roof_segments": [
            {
                "id": "b::v3-merged-roof-segment::seg-0",
                "cluster_canonical_id": "cluster-0",
                "merged_plane": [
                    0.0,
                    math.cos(math.radians(35.0)),
                    math.sin(math.radians(35.0)),
                    -1.0,
                ],
                "corners": [
                    [-2.0, 4.0, -1.0],
                    [2.0, 4.0, -1.0],
                    [2.0, 2.0, 3.0],
                    [-2.0, 2.0, 3.0],
                ],
                "footprint_xz": [
                    [-2.0, 0.0, -1.0],
                    [2.0, 0.0, -1.0],
                    [2.0, 0.0, 3.0],
                    [-2.0, 0.0, 3.0],
                ],
                "features": {"area_m2": 16.0},
                "trace": {
                    "stage": "merged",
                    "rule": "cluster-split",
                    "decision_reason": "synthetic ridge segment",
                },
            },
            {
                "id": "b::v3-merged-roof-segment::seg-1",
                "cluster_canonical_id": "cluster-1",
                "merged_plane": [
                    0.0,
                    math.cos(math.radians(35.0)),
                    -math.sin(math.radians(35.0)),
                    -1.0,
                ],
                "corners": [
                    [-2.0, 4.0, -1.0],
                    [2.0, 4.0, -1.0],
                    [2.0, 2.0, -5.0],
                    [-2.0, 2.0, -5.0],
                ],
                "footprint_xz": [
                    [-2.0, 0.0, -1.0],
                    [2.0, 0.0, -1.0],
                    [2.0, 0.0, -5.0],
                    [-2.0, 0.0, -5.0],
                ],
                "features": {"area_m2": 16.0},
            },
        ],
        "slabs": [
            {
                "id": "s-0",
                "room_id": "room:1",
                "polygon": [
                    [-2.5, 1.0, -1.5],
                    [2.5, 1.0, -1.5],
                    [2.5, 1.0, 3.5],
                    [-2.5, 1.0, 3.5],
                ],
            }
        ],
        "flat_ceilings": [
            {
                "id": "fc-0",
                "room_id": "room:1",
                "footprint_xz": [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0],
                    [-1.0, 0.0, 2.0],
                ],
                "y": 2.5,
                "over": "room",
            }
        ],
        "gaps": [
            {
                "id": "g-0",
                "footprint_xz": [
                    [3.0, 0.0, 3.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 4.0],
                    [3.0, 0.0, 4.0],
                ],
                "floor_y": 1.0,
                "ceiling_y": 2.0,
                "status": "closed",
            }
        ],
        "unresolved": [
            {
                "id": "u-0",
                "footprint_xz": [
                    [-4.0, 0.0, 3.0],
                    [-3.0, 0.0, 3.0],
                    [-3.0, 0.0, 4.0],
                    [-4.0, 0.0, 4.0],
                ],
                "reason": "test",
            }
        ],
        "roof_proposals_count": 2,
        "merged_roof_segments_count": 2,
        "slab_count": 1,
        "flat_ceiling_count": 1,
        "slanted_roof_count": 1,
        "dormer_count": 1,
        "unresolved_region_count": 1,
        "part_count": 1,
    }


def _building_features() -> dict:
    return {
        "bld_footprint_area_m2": 100.0,
        "bld_footprint_perimeter_m": 40.0,
        "bld_footprint_centroid_x": 0.0,
        "bld_footprint_centroid_z": 0.0,
        "bld_footprint_principal_axis_deg": 0.0,
        "bld_footprint_bbox_aspect": 1.0,
        "bld_footprint_solidity": 1.0,
        "bld_footprint_convexity_deficiency": 0.0,
        "bld_footprint_interior_ring_count": 0,
        "bld_footprint_is_L_shape": False,
        "bld_footprint_is_T_shape": False,
        "bld_footprint_is_U_shape": False,
        "bld_footprint_is_rectangle": True,
        "bld_height_m": 6.0,
        "bld_y_min_m": 0.0,
        "bld_y_max_m": 6.0,
        "bld_story_count": 2,
        "bld_typical_story_height_m": 3.0,
        "bld_has_basement": False,
        "bld_room_count": 1,
        "bld_wall_count": 2,
        "bld_door_count": 0,
        "bld_window_count": 0,
        "bld_cross_floor_gap_count": 0,
        "bld_scan_quality_score": 0.0,
        "bld_dominant_wall_azimuth_deg": 0.0,
        "bld_wall_azimuth_entropy": 0.0,
        "scan_quality_overlap_fraction": 0.0,
        "scan_quality_cross_floor_gap_density": 0.01,
    }


def _wall_index() -> dict:
    return {
        "w0": {
            "room_id": "room:1",
            "story": 1,
            "kind": "walls_merged",
            "corners": [
                [-2.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
                [2.0, 2.5, -1.0],
                [-2.0, 2.5, -1.0],
            ],
        },
        "w1": {
            "room_id": "room:1",
            "story": 1,
            "kind": "walls_merged",
            "corners": [
                [2.0, 0.0, -1.0],
                [2.0, 0.0, 3.0],
                [2.0, 2.5, 3.0],
                [2.0, 2.5, -1.0],
            ],
        },
    }


def _aux_context() -> dict:
    return {
        "aux_schema_version": 1,
        "topology": {
            "node_count": 4,
            "edge_count": 2,
            "quality": {},
            "surface_nodes": [
                {
                    "ifc_class": "IfcWall",
                    "properties": {"surface_role": "exterior_wall"},
                },
                {
                    "ifc_class": "IfcWall",
                    "properties": {"surface_role": "interior_wall"},
                },
            ],
            "adjacency_edges": [
                {
                    "type": "ADJACENT_TO",
                    "from_id": "room:merged_room_1",
                    "to_id": "room:merged_room_2",
                    "evidence": {"thickness_cm": 18.0, "thickness_std_cm": 2.0},
                }
            ],
            "cell_complex_cells": [],
        },
        "ontology": {
            "summary_metadata": {},
            "building_parts": [
                {
                    "id": "bp-1",
                    "roof_family_guess": "gable_or_multi_slope",
                    "hypothesis_ids": ["h1"],
                    "oblique_hypothesis_ids": ["h1"],
                    "flat_hypothesis_ids": [],
                    "articulation_room_ids": ["room:1"],
                    "room_indices": [1],
                    "polygon_xz": [[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]],
                }
            ],
            "coverage_subparts": [
                {
                    "id": "sub-1",
                    "semantic_kind": "gable_run",
                    "room_indices": [1],
                    "polygon_xz": [[-2.0, -1.0], [2.0, -1.0], [2.0, 3.0], [-2.0, 3.0]],
                    "roof_hypothesis_id": "h1",
                }
            ],
            "semantic_atoms": [
                {
                    "id": "atom-1",
                    "sloped_coverage_state": "confirmed",
                    "support_evidence_score": 2,
                    "area_m2": 8.0,
                    "poly": [
                        [-2.0, 4.0, -1.0],
                        [2.0, 4.0, -1.0],
                        [2.0, 2.0, 3.0],
                        [-2.0, 2.0, 3.0],
                    ],
                }
            ],
            "room_summaries": {
                "room:1": {
                    "covered_by_sloped_roof": True,
                    "mixed": False,
                    "roles": ["sloped_ceiling"],
                    "slant_delta_m": 1.2,
                }
            },
            "roof_coverage_metadata": {},
            "top_boundary_metadata": {},
            "roof_evidence_metadata": {},
            "full_model_metadata": {},
            "full_model_roof_cells": [
                {
                    "id": "cell-1",
                    "roof_surface_kind": "oblique",
                    "cell_kind": "attic",
                    "part_id": "bp-1",
                    "faces": [
                        {
                            "role": "roof",
                            "corners": [
                                [-2.0, 4.0, -1.0],
                                [2.0, 4.0, -1.0],
                                [2.0, 2.0, 3.0],
                                [-2.0, 2.0, 3.0],
                            ],
                        }
                    ],
                }
            ],
            "full_model_knee_walls": [{"part_id": "bp-1"}],
            "full_model_renderable_surfaces": [{"category": "room_ceiling_sloped"}],
            "roof_surfaces_oblique": [
                {
                    "roof_hypothesis_id": "h1",
                    "avg_azimuth_deg": 0.0,
                    "avg_incl_deg": 35.0,
                    "corners": [
                        [-2.0, 4.0, -1.0],
                        [2.0, 4.0, -1.0],
                        [2.0, 2.0, 3.0],
                        [-2.0, 2.0, 3.0],
                    ],
                }
            ],
            "roof_surfaces_flat": [],
        },
    }


def test_exhaustive_feature_band_emits_new_geometry_context_features():
    rec = _record()
    ctx = _building_context()
    row = fe.expand(rec)
    row.update(_building_features())
    row.update(ctxf.context_features(rec, ctx))
    row.update(
        advf.advanced_features(rec, wall_index=_wall_index(), building_context=ctx)
    )
    row.update(
        exf.exhaustive_features(
            rec, row, building_context=ctx, wall_index=_wall_index()
        )
    )

    assert row["poly_vertex_count"] == 4
    assert row["edge_is_ridge_count"] >= 1
    assert row["part_gable_has_ridge_line"] is True
    assert row["seg_has_behind_knee_wall_extension"] is True
    assert row["bld_slab_count"] == 1
    assert row["typ_gable_candidate"] is True
    assert row["seg_requires_dormer_second_pass"] is True
    assert row["plane_interior_crossing_depth_m"] == 2.0
    assert row["plane_downslope_exit_distance_to_footprint_m"] == 6.0
    assert row["plane_downslope_points_outside"] is False
    assert row["plane_eave_edge_to_exterior_shell_m"] == 2.0
    assert row["pipeline_v3_git_sha"] is None or len(row["pipeline_v3_git_sha"]) == 40


def test_exhaustive_feature_band_emits_auxiliary_ontology_v2_features():
    rec = _record()
    ctx = _building_context()
    row = fe.expand(rec)
    row.update(_building_features())
    row.update(ctxf.context_features(rec, ctx))
    row.update(
        advf.advanced_features(rec, wall_index=_wall_index(), building_context=ctx)
    )
    row.update(
        exf.exhaustive_features(
            rec,
            row,
            building_context=ctx,
            wall_index=_wall_index(),
            aux_context=_aux_context(),
        )
    )

    assert row["swall_thickness_mean_m"] == 0.18
    assert row["part_knee_wall_count"] == 1
    assert row["ont_part_family_guess"] == "gable_or_multi_slope"
    assert row["seg_hypothesis_match_selected"] is True
    assert row["xm_v1_oblique_match_exists"] is True
    assert row["ont_room_is_mixed"] is False
    assert row["seg_any_room_mixed"] is False


def test_exhaustive_feature_band_emits_failure_mode_features():
    rec = _record()
    ctx = _building_context()
    ctx["dormers"] = []
    ctx["dormer_count"] = 0
    ctx["wall_extensions"] = []
    ctx["slabs"] = [
        {
            "id": "s-0",
            "room_id": "room:1",
            "polygon": [
                [-5.0, 1.0, -4.0],
                [5.0, 1.0, -4.0],
                [5.0, 1.0, 6.0],
                [-5.0, 1.0, 6.0],
            ],
        }
    ]
    row = fe.expand(rec)
    row.update(_building_features() | {"bld_story_count": 4, "bld_room_count": 14})
    row.update(ctxf.context_features(rec, ctx))
    interior_walls = {
        key: {**value, "is_exterior": False, "kind": "walls_merged"}
        for key, value in _wall_index().items()
    }
    row.update(
        advf.advanced_features(rec, wall_index=interior_walls, building_context=ctx)
    )
    row.update(
        exf.exhaustive_features(
            rec, row, building_context=ctx, wall_index=interior_walls
        )
    )

    assert row["room_floor_area_total_m2"] == 100.0
    assert row["seg_room_coverage_fraction_mean"] == 0.16
    assert row["seg_is_small_partial_room_slant"] is True
    assert row["swall_supports_only_interior"] is True
    assert row["plane_exterior_edge_contact_fraction"] == 0.0
    assert row["plane_eave_exterior_contact_fraction"] == 0.0
    assert row["artefact_internal_staircase_candidate"] is True
    assert row["artefact_internal_staircase_score"] == 1.0
    assert row["seg_requires_dormer_second_pass"] is False


def test_v3_context_cache_version_check_requires_new_fields():
    assert _cache_is_compatible(
        {
            "b": {
                "context_schema_version": 2,
                "merged_roof_segments": [],
                "slabs": [],
                "flat_ceilings": [],
                "gaps": [],
                "unresolved": [],
                "part_count": 0,
            }
        }
    )
    assert not _cache_is_compatible({"b": {"parts": [], "wall_extensions": []}})
