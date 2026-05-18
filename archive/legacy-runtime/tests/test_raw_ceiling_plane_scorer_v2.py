from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from scripts import prototype_raw_ceiling_plane_scorer as legacy
from scripts.raw_ceiling_plane_scorer_v2.config import (
    GlobalSelectionConfig,
    LayerPolicyConfig,
    RelationConfig,
    ScorerV2Config,
)
from scripts.raw_ceiling_plane_scorer_v2.global_selection import (
    apply_global_xy_selection,
)
from scripts.raw_ceiling_plane_scorer_v2.intersection_seams import (
    compute_intersection_seam_pieces,
)
from scripts.raw_ceiling_plane_scorer_v2.layer_policy import (
    _final_piece_priority,
    classify_split_piece_rows,
)
from scripts.raw_ceiling_plane_scorer_v2.models import (
    PlaneContext,
    PlaneRelation,
    RelationGraph,
)
from scripts.raw_ceiling_plane_scorer_v2.plane_relations import (
    build_plane_relation_graph,
)
from scripts.raw_ceiling_plane_scorer_v2.relation_context import build_plane_contexts
from scripts.raw_ceiling_plane_scorer_v2.runner import (
    score_building_v2,
    score_buildings_v2,
    score_corpus_v2,
)
from scripts.raw_ceiling_plane_scorer_v2.splitter import (
    _clip_ridge_eave_supported_to_source_parts,
    split_piece_rows,
)


def _target(
    element_id: str,
    poly: Polygon,
    *,
    story: int = 0,
    azimuth_deg: float = 0.0,
    inclination_deg: float = 30.0,
    y: float = 3.0,
    kind: str = "ridge_eave_plane_group",
) -> legacy.TargetPlaneRecord:
    return legacy.TargetPlaneRecord(
        uuid="b-test",
        story=story,
        target_kind=kind,
        target_index=0,
        element_id=element_id,
        poly_xz=poly,
        normal=np.asarray([0.0, 1.0, 0.0], dtype=float),
        azimuth_deg=azimuth_deg,
        inclination_deg=inclination_deg,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(0.0, y, 0.0),
    )


def test_plane_context_anchoring_aggregates_source_edge_metrics() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]],
            }
        ]
    }
    roof_result = {
        "building_part_graph": {
            "room_membership": {"room:0": ["building-part:main"]},
            "hypothesis_membership": {},
        }
    }
    target = _target(
        "b-test::ridge-eave-candidate::plane-group::x",
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
    )
    split_piece_rows = [
        {
            "target_element_id": target.element_id,
            "corners": [[0, 3, 0], [4, 3, 0], [4, 3, 4], [0, 3, 4]],
        }
    ]
    supports = [
        legacy.PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id=target.element_id,
            target_kind="ridge_eave_plane_group",
            chain_id="chain-a",
            chain_azimuth_deg=0.0,
            ridge_azimuth_deg=0.0,
            angle_delta_deg=10.0,
            boundary_distance_m=0.1,
            overlap_fraction=1.0,
            height_residual_m=0.2,
            support_score=0.95,
            supported=True,
            chain_length_m=1.5,
        ),
        legacy.PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id=target.element_id,
            target_kind="ridge_eave_plane_group",
            chain_id="chain-b",
            chain_azimuth_deg=0.0,
            ridge_azimuth_deg=0.0,
            angle_delta_deg=20.0,
            boundary_distance_m=0.1,
            overlap_fraction=1.0,
            height_residual_m=0.4,
            support_score=0.9,
            supported=True,
            chain_length_m=1.5,
        ),
    ]

    contexts = build_plane_contexts(
        [target],
        supports,
        split_piece_rows,
        building,
        roof_result,
        ridge_eave_target_diagnostics={},
    )
    context = contexts[target.element_id]
    assert context.source_edge_ids == ("chain-a", "chain-b")
    assert context.source_edge_overlap_m == pytest.approx(3.0)
    assert context.source_edge_alignment_deg == pytest.approx(15.0)
    assert context.source_edge_height_residual_m == pytest.approx(0.3)


def test_plane_context_marks_main_extension_boundary_crossing() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
            },
            {
                "story": 0,
                "floor_polygon": [[2, 0, 0], [4, 0, 0], [4, 0, 2], [2, 0, 2]],
            },
        ]
    }
    roof_result = {
        "building_part_graph": {
            "room_membership": {
                "room:0": ["building-part:main"],
                "room:1": ["building-part:ext"],
            },
            "hypothesis_membership": {},
        }
    }
    target = _target(
        "b-test::ceiling-oblique::0",
        Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        kind="candidate_oblique",
    )

    contexts = build_plane_contexts(
        [target],
        [],
        [
            {
                "target_element_id": target.element_id,
                "corners": [[0, 3, 0], [4, 3, 0], [4, 3, 2], [0, 3, 2]],
            }
        ],
        building,
        roof_result,
        ridge_eave_target_diagnostics={},
    )
    context = contexts[target.element_id]
    assert context.in_main_body is True
    assert context.in_extension is True
    assert context.crosses_main_extension_boundary is True


def test_plane_relation_classification_same_face_mirror_covering_competitor() -> None:
    t_same_a = _target(
        "a",
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        azimuth_deg=10.0,
        inclination_deg=35.0,
    )
    t_same_b = _target(
        "b",
        Polygon([(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]),
        azimuth_deg=12.0,
        inclination_deg=34.0,
    )
    t_mirror = _target(
        "c",
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        azimuth_deg=190.0,
        inclination_deg=35.0,
    )
    t_cont_a = _target(
        "g",
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        azimuth_deg=45.0,
        inclination_deg=35.0,
    )
    t_cont_b = _target(
        "h",
        Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
        azimuth_deg=225.0,
        inclination_deg=35.5,
    )
    t_cover_small = _target(
        "d",
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
        azimuth_deg=60.0,
        inclination_deg=35.0,
    )
    t_cover_big = _target(
        "e",
        Polygon([(0.8, 0.8), (2.2, 0.8), (2.2, 2.2), (0.8, 2.2)]),
        azimuth_deg=80.0,
        inclination_deg=35.0,
    )
    t_comp = _target(
        "f",
        Polygon([(2, 2), (5, 2), (5, 5), (2, 5)]),
        azimuth_deg=70.0,
        inclination_deg=50.0,
    )

    contexts = {
        target.element_id: PlaneContext(
            target_id=target.element_id,
            story=0,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        )
        for target in [
            t_same_a,
            t_same_b,
            t_mirror,
            t_cont_a,
            t_cont_b,
            t_cover_small,
            t_cover_big,
            t_comp,
        ]
    }

    graph = build_plane_relation_graph(
        [
            t_same_a,
            t_same_b,
            t_mirror,
            t_cont_a,
            t_cont_b,
            t_cover_small,
            t_cover_big,
            t_comp,
        ],
        contexts,
        RelationConfig(min_relation_overlap_m2=0.05, covering_min_overlap_fraction=0.8),
    )
    kinds = {(r.a_target_id, r.b_target_id): r.relation_kind for r in graph.relations}
    assert any(r.relation_kind == "same_face" for r in graph.relations)
    assert any(r.relation_kind == "mirror_pair" for r in graph.relations)
    assert kinds[("g", "h")] == "continuous_run_neighbor"
    assert any(r.relation_kind == "covering" for r in graph.relations)
    assert any(r.relation_kind == "local_competitor" for r in graph.relations)


def test_global_xy_selection_groups_continuous_run_neighbors_into_one_family() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [6, 0, 0], [6, 0, 2], [0, 0, 2]],
            }
        ]
    }
    rows = [
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-a",
            "piece_id": "piece-a",
            "piece_index": 0,
            "target_inclination_deg": 35.0,
            "support_score": 0.8,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "corners": [[0, 3.0, 0], [2, 3.0, 0], [2, 3.0, 2], [0, 3.0, 2]],
            "holes": [],
            "area_xz_m2": 4.0,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-b",
            "piece_id": "piece-b",
            "piece_index": 1,
            "target_inclination_deg": 35.5,
            "support_score": 0.82,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "corners": [[2, 3.01, 0], [4, 3.01, 0], [4, 3.01, 2], [2, 3.01, 2]],
            "holes": [],
            "area_xz_m2": 4.0,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-c",
            "piece_id": "piece-c",
            "piece_index": 2,
            "target_inclination_deg": 20.0,
            "support_score": 0.5,
            "final_layer": False,
            "final_layer_reason": "candidate_oblique",
            "corners": [[4.5, 4.0, 0], [6.0, 4.0, 0], [6.0, 4.0, 2], [4.5, 4.0, 2]],
            "holes": [],
            "area_xz_m2": 3.0,
        },
    ]
    contexts = {
        "roof-a": PlaneContext(
            target_id="roof-a",
            story=0,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        "roof-b": PlaneContext(
            target_id="roof-b",
            story=0,
            source_edge_ids=("chain-2",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        "roof-c": PlaneContext(
            target_id="roof-c",
            story=0,
            source_edge_ids=("chain-3",),
            source_edge_overlap_m=1.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    relation_graph = RelationGraph(
        contexts_by_target=contexts,
        relations=[
            PlaneRelation(
                a_target_id="roof-a",
                b_target_id="roof-b",
                relation_kind="continuous_run_neighbor",
                overlap_m2=0.0,
                shared_boundary_len_m=2.0,
                azimuth_delta_deg=180.0,
                inclination_delta_deg=0.5,
                height_residual_m=0.01,
                same_story=True,
                same_part=True,
            )
        ],
    )
    cfg = GlobalSelectionConfig(
        enabled=True,
        objective_weight_support=1.0,
        objective_weight_topness=2.5,
        objective_weight_prior_final=0.0,
        envelope_hard_buffer_m=0.0,
        envelope_soft_buffer_m=0.0,
        envelope_soft_outside_budget_fraction=1.0,
    )

    out = apply_global_xy_selection(rows, building, cfg, relation_graph=relation_graph)
    by_id = {str(row.get("piece_id")): row for row in out}
    assert (
        by_id["piece-a"]["xy_global_selector_plane_group_id"]
        == by_id["piece-b"]["xy_global_selector_plane_group_id"]
    )
    assert by_id["piece-a"]["xy_global_selector_plane_group_member_count"] == 2
    assert by_id["piece-b"]["xy_global_selector_plane_group_member_count"] == 2


def test_splitter_clips_ridge_eave_piece_to_source_part_union() -> None:
    target = _target(
        "b-test::ridge-eave-candidate::plane-group::x",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        kind="ridge_eave_plane_group",
    )
    piece = legacy.TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=target.element_id,
        target_kind="ridge_eave_plane_group",
        piece_id=f"{target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=8.0,
        support_score=0.9,
        chain_ids=("chain-1",),
        corners=[[0.0, 3.0, 0.0], [4.0, 3.0, 0.0], [4.0, 3.0, 2.0], [0.0, 3.0, 2.0]],
        holes=[],
    )
    building = {
        "rooms": [
            {
                "id": "room:0",
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "id": "room:1",
                "story": 0,
                "floor_polygon": [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                ],
            },
        ]
    }
    roof_result = {
        "building_part_graph": {
            "room_membership": {
                "room:0": ["building-part:main"],
                "room:1": ["building-part:ext"],
            }
        }
    }
    diagnostics = {
        target.element_id: {
            "creator_source_room_ids": ["room:0"],
        }
    }

    out = _clip_ridge_eave_supported_to_source_parts(
        [piece],
        [target],
        building,
        roof_result,
        diagnostics,
    )
    assert len(out) == 1
    assert out[0].piece_id == piece.piece_id
    assert out[0].area_xz_m2 == pytest.approx(4.0)
    xs = [float(corner[0]) for corner in out[0].corners]
    assert max(xs) == pytest.approx(2.0, abs=1e-6)


def test_split_piece_rows_serializes_target_plane_metadata() -> None:
    normal = np.asarray([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0)
    target = legacy.TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="candidate_oblique",
        target_index=1,
        element_id="b-test::ceiling-oblique::1",
        poly_xz=Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]),
        normal=normal,
        azimuth_deg=90.0,
        inclination_deg=45.0,
        ridge_dir_xz=(0.0, 1.0),
        area_xz_m2=2.0,
        plane_point=(0.0, 4.0, 0.0),
    )
    piece = legacy.TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=target.element_id,
        target_kind=target.target_kind,
        piece_id=f"{target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=2.0,
        support_score=0.9,
        chain_ids=(),
        corners=[[0.0, 4.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 1.0], [0.0, 4.0, 1.0]],
        holes=[],
    )

    rows = split_piece_rows([piece], [target], ridge_eave_target_diagnostics={})

    assert len(rows) == 1
    row = rows[0]
    assert row["target_plane_point"] == pytest.approx([0.0, 4.0, 0.0], abs=1e-6)
    assert row["target_normal"] == pytest.approx(list(normal), abs=1e-9)
    assert row["target_plane_coeffs"] == pytest.approx(
        [
            normal[0],
            normal[1],
            normal[2],
            -np.dot(normal, np.asarray(target.plane_point)),
        ],
        abs=1e-9,
    )


def test_splitter_fallback_source_clip_excludes_non_source_story_parts() -> None:
    target = _target(
        "b-test::ridge-eave-candidate::plane-group::x",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        story=0,
        kind="ridge_eave_plane_group",
    )
    piece = legacy.TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=target.element_id,
        target_kind="ridge_eave_plane_group",
        piece_id=f"{target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=8.0,
        support_score=0.9,
        chain_ids=("chain-1",),
        corners=[[0.0, 3.0, 0.0], [4.0, 3.0, 0.0], [4.0, 3.0, 2.0], [0.0, 3.0, 2.0]],
        holes=[],
    )
    building = {
        "rooms": [
            {
                "id": "room:src-upper",
                "story": 1,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "id": "room:ext-story0",
                "story": 0,
                "floor_polygon": [
                    [2.5, 0.0, 0.0],
                    [6.5, 0.0, 0.0],
                    [6.5, 0.0, 2.0],
                    [2.5, 0.0, 2.0],
                ],
            },
        ]
    }
    roof_result = {
        "building_part_graph": {
            "room_membership": {
                "room:src-upper": ["building-part:main"],
                "room:ext-story0": ["building-part:ext"],
            }
        }
    }
    diagnostics = {
        target.element_id: {
            "creator_source_room_ids": ["room:src-upper"],
        }
    }

    out = _clip_ridge_eave_supported_to_source_parts(
        [piece],
        [target],
        building,
        roof_result,
        diagnostics,
    )
    assert len(out) == 1
    poly = Polygon([(float(c[0]), float(c[2])) for c in out[0].corners])
    if not poly.is_valid:
        poly = poly.buffer(0)
    ext_poly = Polygon([(2.5, 0.0), (6.5, 0.0), (6.5, 2.0), (2.5, 2.0)])
    assert float(poly.intersection(ext_poly).area) == pytest.approx(0.0, abs=1e-6)


def test_layer_policy_enforces_anchor_and_cross_part_ownership() -> None:
    target_id = "b-test::ridge-eave-candidate::plane-group::x"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=("chain-1",),
        source_edge_overlap_m=2.0,
        source_edge_alignment_deg=10.0,
        source_edge_height_residual_m=0.2,
        building_part_ids=("building-part:main", "building-part:ext"),
        source_part_ids=("building-part:main",),
        crossed_part_ids=("building-part:ext",),
        in_main_body=True,
        in_extension=True,
        crosses_main_extension_boundary=True,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])

    rows = [
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": target_id,
            "source_part_overlap_fraction": 0.05,
            "ownership_redundant": False,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": [],
        }
    ]
    classified = classify_split_piece_rows(
        rows, graph, LayerPolicyConfig(min_cross_part_source_overlap_fraction=0.2)
    )
    assert classified[0]["final_layer"] is False
    assert classified[0]["final_layer_reason"] == "ridge_eave_cross_part_unowned"


def test_layer_policy_allows_disjoint_source_crossed_bypass() -> None:
    target_id = "b-test::ridge-eave-candidate::plane-group::x"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=("chain-1",),
        source_edge_overlap_m=2.0,
        source_edge_alignment_deg=10.0,
        source_edge_height_residual_m=0.2,
        building_part_ids=("building-part:ext",),
        source_part_ids=("building-part:main",),
        crossed_part_ids=("building-part:ext",),
        in_main_body=False,
        in_extension=True,
        crosses_main_extension_boundary=False,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])
    rows = [
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": target_id,
            "source_part_overlap_fraction": 0.3,
            "ownership_redundant": True,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": [],
        }
    ]
    classified = classify_split_piece_rows(rows, graph, LayerPolicyConfig())
    assert classified[0]["final_layer"] is True
    assert classified[0]["final_layer_reason"] == "ridge_eave_disjoint_extension_owner"


def test_layer_policy_committed_keeps_single_strongest_supported_piece() -> None:
    target_id = "b-test::roof-oblique::oblique:0"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=(),
        source_edge_overlap_m=0.0,
        source_edge_alignment_deg=None,
        source_edge_height_residual_m=None,
        building_part_ids=("building-part:main",),
        source_part_ids=("building-part:main",),
        crossed_part_ids=("building-part:main",),
        in_main_body=True,
        in_extension=False,
        crosses_main_extension_boundary=False,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "p-small",
            "area_xz_m2": 2.0,
            "support_score": 0.8,
            "source_part_overlap_fraction": 1.0,
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "p-big",
            "area_xz_m2": 3.0,
            "support_score": 0.7,
            "source_part_overlap_fraction": 1.0,
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "residual",
            "target_element_id": target_id,
            "piece_id": "p-residual",
            "area_xz_m2": 1.0,
        },
    ]
    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["p-big"]["final_layer"] is True
    assert by_id["p-big"]["final_layer_reason"] == "committed_relation_owner"
    assert by_id["p-small"]["final_layer"] is False
    assert by_id["p-small"]["final_layer_reason"] == "committed_union_demoted"
    assert by_id["p-residual"]["final_layer"] is False
    assert by_id["p-residual"]["final_layer_reason"] == "committed_residual"


def test_layer_policy_marks_non_owner_with_zero_source_overlap_cross_part_unowned() -> (
    None
):
    target_id = "b-test::roof-oblique::oblique:0"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=(),
        source_edge_overlap_m=0.0,
        source_edge_alignment_deg=None,
        source_edge_height_residual_m=None,
        building_part_ids=("building-part:main", "building-part:ext"),
        source_part_ids=("building-part:main",),
        crossed_part_ids=("building-part:main", "building-part:ext"),
        in_main_body=True,
        in_extension=True,
        crosses_main_extension_boundary=True,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "p-owner",
            "area_xz_m2": 3.0,
            "support_score": 0.9,
            "source_part_overlap_fraction": 1.0,
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "p-wrong-unit",
            "area_xz_m2": 2.0,
            "support_score": 0.8,
            "source_part_overlap_fraction": 0.0,
        },
    ]
    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["p-owner"]["final_layer_reason"] == "committed_relation_owner"
    assert by_id["p-wrong-unit"]["final_layer"] is False
    assert by_id["p-wrong-unit"]["final_layer_reason"] == "committed_cross_part_unowned"


def test_layer_policy_promotes_committed_residual_within_source_parts() -> None:
    target_id = "b-test::roof-oblique::oblique:0"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=(),
        source_edge_overlap_m=0.0,
        source_edge_alignment_deg=None,
        source_edge_height_residual_m=None,
        building_part_ids=("building-part:main", "building-part:ext"),
        source_part_ids=("building-part:main", "building-part:ext"),
        crossed_part_ids=("building-part:main",),
        in_main_body=True,
        in_extension=True,
        crosses_main_extension_boundary=True,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])
    # Keep supported and residual polygons disjoint in XZ so the post-pass's
    # coplanar-overlap guard doesn't block promotion.
    plane_coeffs = [0.0, 1.0, 0.0, 0.0]
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "p-supported",
            "area_xz_m2": 4.0,
            "support_score": 0.9,
            "source_part_ids": ["building-part:main", "building-part:ext"],
            "piece_part_ids_story": ["building-part:main"],
            "target_plane_coeffs": plane_coeffs,
            "corners": [
                [0.0, 3.0, 0.0],
                [2.0, 3.0, 0.0],
                [2.0, 3.0, 2.0],
                [0.0, 3.0, 2.0],
            ],
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "residual",
            "target_element_id": target_id,
            "piece_id": "p-residual-contained",
            "area_xz_m2": 2.0,
            "source_part_ids": ["building-part:main", "building-part:ext"],
            "piece_part_ids_story": ["building-part:ext"],
            "target_plane_coeffs": plane_coeffs,
            "corners": [
                [3.0, 3.0, 0.0],
                [5.0, 3.0, 0.0],
                [5.0, 3.0, 1.0],
                [3.0, 3.0, 1.0],
            ],
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "residual",
            "target_element_id": target_id,
            "piece_id": "p-residual-outside",
            "area_xz_m2": 1.5,
            "source_part_ids": ["building-part:main", "building-part:ext"],
            "piece_part_ids_story": ["building-part:other"],
            "target_plane_coeffs": plane_coeffs,
            "corners": [
                [6.0, 3.0, 0.0],
                [8.0, 3.0, 0.0],
                [8.0, 3.0, 1.0],
                [6.0, 3.0, 1.0],
            ],
        },
    ]
    out = classify_split_piece_rows(rows, graph, LayerPolicyConfig())
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["p-residual-contained"]["final_layer"] is True
    assert (
        by_id["p-residual-contained"]["final_layer_reason"]
        == "committed_residual_part_contained"
    )
    assert by_id["p-residual-outside"]["final_layer"] is False
    assert by_id["p-residual-outside"]["final_layer_reason"] == "committed_residual"


def test_layer_policy_suppresses_candidate_same_face_shadowed_by_committed() -> None:
    candidate_target = "b-test::ceiling-oblique::ceiling-oblique:1"
    committed_target = "b-test::ceiling-oblique::ceiling-oblique:2"
    contexts = {
        candidate_target: PlaneContext(
            target_id=candidate_target,
            story=1,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=3.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        committed_target: PlaneContext(
            target_id=committed_target,
            story=1,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=3.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    relations = [
        PlaneRelation(
            a_target_id=candidate_target,
            b_target_id=committed_target,
            relation_kind="same_face",
            overlap_m2=10.0,
            azimuth_delta_deg=0.1,
            inclination_delta_deg=0.1,
            height_residual_m=0.01,
            same_story=True,
            same_part=True,
        )
    ]
    graph = RelationGraph(contexts_by_target=contexts, relations=relations)

    rows = [
        {
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": candidate_target,
            "piece_id": "candidate-piece",
            "area_xz_m2": 4.0,
            "support_score": 0.95,
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-piece",
            "area_xz_m2": 4.2,
            "support_score": 0.9,
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
        },
    ]
    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["committed-piece"]["final_layer"] is True
    assert by_id["candidate-piece"]["final_layer"] is False
    assert (
        by_id["candidate-piece"]["final_layer_reason"]
        == "same_face_shadowed_by_committed"
    )
    assert by_id["candidate-piece"]["overlay_suppressed"] is True


def test_layer_policy_clips_candidate_same_face_overlap_and_keeps_residual_visible():
    candidate_target = "b-test::ceiling-oblique::ceiling-oblique:1"
    committed_target = "b-test::ceiling-oblique::ceiling-oblique:2"
    contexts = {
        candidate_target: PlaneContext(
            target_id=candidate_target,
            story=1,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=3.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        committed_target: PlaneContext(
            target_id=committed_target,
            story=1,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=3.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(
        contexts_by_target=contexts,
        relations=[
            PlaneRelation(
                a_target_id=candidate_target,
                b_target_id=committed_target,
                relation_kind="same_face",
                overlap_m2=4.0,
                azimuth_delta_deg=0.1,
                inclination_delta_deg=0.1,
                height_residual_m=0.01,
                same_story=True,
                same_part=True,
            )
        ],
    )
    rows = [
        {
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": candidate_target,
            "piece_id": "candidate-piece",
            "piece_index": 0,
            "area_xz_m2": 4.2,
            "support_score": 0.95,
            "corners": [[0, 3, 0], [2.1, 3, 0], [2.1, 3, 2], [0, 3, 2]],
            "holes": [],
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-piece",
            "piece_index": 0,
            "area_xz_m2": 4.0,
            "support_score": 0.9,
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    candidate = by_id["candidate-piece"]
    assert candidate["overlay_suppressed"] is False
    assert candidate["final_layer_reason"] == "same_face_shadow_clipped"
    assert candidate["xy_conflict_clipped"] is True
    assert candidate["same_face_shadow_overlap_fraction"] >= 0.95
    assert candidate["area_xz_m2"] == pytest.approx(0.2, rel=0.0, abs=1e-6)
    xs = [float(corner[0]) for corner in candidate["corners"]]
    assert min(xs) == pytest.approx(2.0, abs=1e-6)
    assert max(xs) == pytest.approx(2.1, abs=1e-6)


def test_layer_policy_does_not_suppress_candidate_without_covering_owner() -> None:
    candidate_target = "b-test::ceiling-oblique::ceiling-oblique:1"
    ridge_target = "b-test::ridge-eave-candidate::plane-group::owner"
    contexts = {
        candidate_target: PlaneContext(
            target_id=candidate_target,
            story=1,
            source_edge_ids=("chain-1", "chain-2"),
            source_edge_overlap_m=3.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=1,
            source_edge_ids=("chain-1", "chain-2", "chain-3"),
            source_edge_overlap_m=4.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(
        contexts_by_target=contexts,
        relations=[
            PlaneRelation(
                a_target_id=candidate_target,
                b_target_id=ridge_target,
                relation_kind="same_face",
                overlap_m2=2.0,
                azimuth_delta_deg=0.1,
                inclination_delta_deg=0.1,
                height_residual_m=0.01,
                same_story=True,
                same_part=True,
            )
        ],
    )
    rows = [
        {
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": candidate_target,
            "piece_id": "candidate-piece",
            "piece_index": 0,
            "area_xz_m2": 4.0,
            "support_score": 0.95,
            "chain_ids": ["chain-1", "chain-2"],
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-piece",
            "piece_index": 0,
            "area_xz_m2": 8.0,
            "support_score": 0.9,
            "chain_ids": ["chain-1", "chain-2", "chain-3"],
            "corners": [[1, 3, 0], [5, 3, 0], [5, 3, 2], [1, 3, 2]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["ridge-piece"]["final_layer"] is True
    assert by_id["ridge-piece"]["final_layer_reason"] == "ridge_eave_relation_owner"
    assert by_id["candidate-piece"]["final_layer"] is False
    assert by_id["candidate-piece"]["final_layer_reason"] == "candidate_oblique"
    assert by_id["candidate-piece"].get("overlay_suppressed") is not True
    assert by_id["candidate-piece"].get("same_face_shadow_target_id") is None


def test_suppress_chains_subset_same_face_ridge_owner() -> None:
    candidate_target = "b-test::ceiling-oblique::ceiling-oblique:1"
    ridge_target = "b-test::ridge-eave-candidate::plane-group::owner"
    covering_target = "b-test::ridge-eave-candidate::plane-group::covering"
    contexts = {
        candidate_target: PlaneContext(
            target_id=candidate_target,
            story=1,
            source_edge_ids=("chain-1", "chain-2"),
            source_edge_overlap_m=3.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        covering_target: PlaneContext(
            target_id=covering_target,
            story=1,
            source_edge_ids=("chain-4",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=1,
            source_edge_ids=("chain-1", "chain-2", "chain-3"),
            source_edge_overlap_m=4.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(
        contexts_by_target=contexts,
        relations=[
            PlaneRelation(
                a_target_id=candidate_target,
                b_target_id=ridge_target,
                relation_kind="same_face",
                overlap_m2=2.0,
                azimuth_delta_deg=0.1,
                inclination_delta_deg=0.1,
                height_residual_m=0.01,
                same_story=True,
                same_part=True,
            ),
            PlaneRelation(
                a_target_id=candidate_target,
                b_target_id=covering_target,
                relation_kind="covering",
                overlap_m2=3.5,
                azimuth_delta_deg=90.0,
                inclination_delta_deg=12.0,
                height_residual_m=0.2,
                same_story=True,
                same_part=True,
                covering_target_id=covering_target,
                covered_target_id=candidate_target,
            ),
        ],
    )
    rows = [
        {
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": candidate_target,
            "piece_id": "candidate-piece",
            "piece_index": 0,
            "area_xz_m2": 4.0,
            "support_score": 0.95,
            "chain_ids": ["chain-1", "chain-2"],
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-piece",
            "piece_index": 0,
            "area_xz_m2": 8.0,
            "support_score": 0.9,
            "chain_ids": ["chain-1", "chain-2", "chain-3"],
            "corners": [[1, 3, 0], [5, 3, 0], [5, 3, 2], [1, 3, 2]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": covering_target,
            "piece_id": "covering-piece",
            "piece_index": 1,
            "area_xz_m2": 10.0,
            "support_score": 0.85,
            "chain_ids": ["chain-4"],
            "corners": [[0, 3, 0], [5, 3, 0], [5, 3, 3], [0, 3, 3]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["ridge-piece"]["final_layer"] is True
    assert by_id["ridge-piece"]["final_layer_reason"] == "ridge_eave_relation_owner"
    assert by_id["candidate-piece"]["final_layer"] is False
    assert (
        by_id["candidate-piece"]["final_layer_reason"]
        == "same_face_chain_subset_shadowed"
    )
    assert by_id["candidate-piece"]["overlay_suppressed"] is True
    assert by_id["candidate-piece"]["same_face_shadow_target_id"] == ridge_target


def test_layer_policy_suppresses_committed_demoted_piece_shadowed_by_owner() -> None:
    target_id = "b-test::roof-oblique::oblique:0"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=(),
        source_edge_overlap_m=0.0,
        source_edge_alignment_deg=None,
        source_edge_height_residual_m=None,
        building_part_ids=("building-part:main",),
        source_part_ids=("building-part:main",),
        crossed_part_ids=("building-part:main",),
        in_main_body=True,
        in_extension=False,
        crosses_main_extension_boundary=False,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "owner",
            "area_xz_m2": 5.0,
            "support_score": 0.9,
            "corners": [[0, 3, 0], [3, 3, 0], [3, 3, 2], [0, 3, 2]],
            "source_part_overlap_fraction": 1.0,
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "demoted",
            "area_xz_m2": 4.9,
            "support_score": 0.8,
            "corners": [[0, 3, 0], [3, 3, 0], [3, 3, 2], [0, 3, 2]],
            "source_part_overlap_fraction": 1.0,
        },
    ]
    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["owner"]["final_layer"] is True
    assert by_id["demoted"]["final_layer_reason"] == "committed_union_demoted"
    assert by_id["demoted"]["overlay_suppressed"] is True


def test_layer_policy_clips_committed_demoted_overlap_and_keeps_residual_visible() -> (
    None
):
    target_id = "b-test::roof-oblique::oblique:0"
    context = PlaneContext(
        target_id=target_id,
        story=0,
        source_edge_ids=(),
        source_edge_overlap_m=0.0,
        source_edge_alignment_deg=None,
        source_edge_height_residual_m=None,
        building_part_ids=("building-part:main",),
        source_part_ids=("building-part:main",),
        crossed_part_ids=("building-part:main",),
        in_main_body=True,
        in_extension=False,
        crosses_main_extension_boundary=False,
    )
    graph = RelationGraph(contexts_by_target={target_id: context}, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "owner",
            "piece_index": 0,
            "area_xz_m2": 6.0,
            "support_score": 0.9,
            "corners": [[0, 3, 0], [3, 3, 0], [3, 3, 2], [0, 3, 2]],
            "holes": [],
            "source_part_overlap_fraction": 1.0,
        },
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": target_id,
            "piece_id": "demoted",
            "piece_index": 1,
            "area_xz_m2": 6.0,
            "support_score": 0.8,
            "corners": [[-0.06, 3, 0], [2.94, 3, 0], [2.94, 3, 2], [-0.06, 3, 2]],
            "holes": [],
            "source_part_overlap_fraction": 1.0,
        },
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["owner"]["final_layer"] is True
    demoted = by_id["demoted"]
    assert demoted["overlay_suppressed"] is False
    assert demoted["final_layer_reason"] == "committed_union_demoted_clipped"
    assert demoted["xy_conflict_clipped"] is True
    assert demoted["area_xz_m2"] == pytest.approx(0.12, rel=0.0, abs=1e-6)
    xs = [float(corner[0]) for corner in demoted["corners"]]
    assert min(xs) == pytest.approx(-0.06, abs=1e-6)
    assert max(xs) == pytest.approx(0.0, abs=1e-6)


def test_layer_policy_keeps_nonfinal_ridge_eave_visible_after_clipping_strategy() -> (
    None
):
    committed_target = "b-test::roof-oblique::oblique:1"
    ridge_target = "b-test::ridge-eave-candidate::plane-group::x"
    contexts = {
        committed_target: PlaneContext(
            target_id=committed_target,
            story=1,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=0,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=2.0,
            source_edge_height_residual_m=2.0,
            building_part_ids=("building-part:ext",),
            source_part_ids=("building-part:ext",),
            crossed_part_ids=("building-part:main", "building-part:ext"),
            in_main_body=False,
            in_extension=True,
            crosses_main_extension_boundary=True,
        ),
    }
    graph = RelationGraph(contexts_by_target=contexts, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-owner",
            "area_xz_m2": 8.0,
            "support_score": 0.9,
            "corners": [[0, 3, 0], [4, 3, 0], [4, 3, 2], [0, 3, 2]],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-wide",
            "area_xz_m2": 20.0,
            "support_score": 0.8,
            "source_part_overlap_fraction": 0.8,
            "ownership_redundant": False,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": [],
            "corners": [[-1, 3, -1], [5, 3, -1], [5, 3, 3], [-1, 3, 3]],
        },
    ]
    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    assert by_id["committed-owner"]["final_layer"] is True
    assert by_id["ridge-wide"]["final_layer"] is False
    assert by_id["ridge-wide"]["final_layer_reason"] == "ridge_eave_unanchored"
    assert by_id["ridge-wide"].get("overlay_suppressed") is not True


def test_layer_policy_promotes_orthogonal_through_building_ridge_eave_piece() -> None:
    ridge_target = "b-test::ridge-eave-candidate::plane-group::x"
    cover_target = "b-test::roof-oblique::oblique:1"
    contexts = {
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=1,
            source_edge_ids=("chain-1", "chain-2"),
            source_edge_overlap_m=5.0,
            source_edge_alignment_deg=2.0,
            source_edge_height_residual_m=4.5,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:ext",),
            crossed_part_ids=("building-part:ext", "building-part:main"),
            in_main_body=True,
            in_extension=True,
            crosses_main_extension_boundary=True,
        ),
        cover_target: PlaneContext(
            target_id=cover_target,
            story=1,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(
        contexts_by_target=contexts,
        relations=[
            PlaneRelation(
                a_target_id=ridge_target,
                b_target_id=cover_target,
                relation_kind="covering",
                overlap_m2=20.0,
                azimuth_delta_deg=90.0,
                inclination_delta_deg=8.0,
                height_residual_m=3.0,
                same_story=True,
                same_part=True,
                covering_target_id=cover_target,
                covered_target_id=ridge_target,
            )
        ],
    )

    rows = [
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-through",
            "support_score": 0.9,
            "provenance_relevance_flag": "suspect_interior_slice",
            "source_part_overlap_fraction": 0.0,
            "ownership_redundant": True,
            "piece_part_ids": ["building-part:main", "building-part:ext"],
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": [],
            "corners": [[0, 3, 0], [4, 3, 0], [4, 3, 2], [0, 3, 2]],
        }
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    assert out[0]["final_layer"] is True
    assert out[0]["final_layer_reason"] == "ridge_eave_through_building_owner"


def test_layer_policy_keeps_parallel_covering_ridge_eave_piece_nonfinal() -> None:
    ridge_target = "b-test::ridge-eave-candidate::plane-group::x"
    cover_target = "b-test::roof-oblique::oblique:1"
    contexts = {
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=1,
            source_edge_ids=("chain-1", "chain-2"),
            source_edge_overlap_m=5.0,
            source_edge_alignment_deg=2.0,
            source_edge_height_residual_m=4.5,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:ext",),
            crossed_part_ids=("building-part:ext", "building-part:main"),
            in_main_body=True,
            in_extension=True,
            crosses_main_extension_boundary=True,
        ),
        cover_target: PlaneContext(
            target_id=cover_target,
            story=1,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(
        contexts_by_target=contexts,
        relations=[
            PlaneRelation(
                a_target_id=ridge_target,
                b_target_id=cover_target,
                relation_kind="covering",
                overlap_m2=20.0,
                azimuth_delta_deg=10.0,
                inclination_delta_deg=8.0,
                height_residual_m=3.0,
                same_story=True,
                same_part=True,
                covering_target_id=cover_target,
                covered_target_id=ridge_target,
            )
        ],
    )

    rows = [
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-through",
            "support_score": 0.9,
            "provenance_relevance_flag": "suspect_interior_slice",
            "source_part_overlap_fraction": 0.0,
            "ownership_redundant": True,
            "piece_part_ids": ["building-part:main", "building-part:ext"],
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": [],
            "corners": [[0, 3, 0], [4, 3, 0], [4, 3, 2], [0, 3, 2]],
        }
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    assert out[0]["final_layer"] is False
    assert out[0]["final_layer_reason"] == "ridge_eave_unanchored"


def test_global_xy_selection_ilp_partitions_overlap_by_top_plane() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [3, 0, 0], [3, 0, 2], [0, 0, 2]],
            }
        ]
    }
    rows = [
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-a",
            "piece_id": "piece-a",
            "piece_index": 0,
            "target_inclination_deg": 30.0,
            "support_score": 0.7,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "corners": [[0, 3.0, 0], [2, 3.0, 0], [2, 3.0, 2], [0, 3.0, 2]],
            "holes": [],
            "area_xz_m2": 4.0,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-b",
            "piece_id": "piece-b",
            "piece_index": 1,
            "target_inclination_deg": 35.0,
            "support_score": 0.8,
            "final_layer": False,
            "final_layer_reason": "candidate_oblique",
            "corners": [[1.5, 4.0, 0], [3.0, 4.0, 0], [3.0, 4.0, 2], [1.5, 4.0, 2]],
            "holes": [],
            "area_xz_m2": 3.0,
        },
    ]
    cfg = GlobalSelectionConfig(
        enabled=True,
        objective_weight_support=1.0,
        objective_weight_topness=2.5,
        objective_weight_prior_final=0.0,
        envelope_hard_buffer_m=0.0,
        envelope_soft_buffer_m=0.0,
        envelope_soft_outside_budget_fraction=1.0,
    )

    out = apply_global_xy_selection(rows, building, cfg)
    by_id = {str(row.get("piece_id")): row for row in out}
    piece_a = by_id["piece-a"]
    piece_b = by_id["piece-b"]

    assert piece_a["final_layer"] is True
    assert piece_b["final_layer"] is True
    assert piece_a["final_layer_reason"] == "committed_relation_owner"
    assert piece_b["final_layer_reason"] == "xy_global_selector_selected_candidate"
    assert piece_a["overlay_suppressed"] is False
    assert piece_b["overlay_suppressed"] is False
    assert piece_a["area_xz_m2"] == pytest.approx(3.0, abs=1e-6)
    assert piece_b["area_xz_m2"] == pytest.approx(3.0, abs=1e-6)
    xs_a = [float(corner[0]) for corner in piece_a["corners"]]
    assert min(xs_a) == pytest.approx(0.0, abs=1e-6)
    assert max(xs_a) == pytest.approx(1.5, abs=1e-6)


def test_global_xy_selection_keeps_dormer_exception_outside_envelope() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
            }
        ]
    }
    rows = [
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "main-roof",
            "piece_id": "main-roof-piece",
            "piece_index": 0,
            "target_inclination_deg": 30.0,
            "support_score": 0.7,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "corners": [[0, 3.0, 0], [2, 3.0, 0], [2, 3.0, 2], [0, 3.0, 2]],
            "holes": [],
            "area_xz_m2": 4.0,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": "dormer",
            "piece_id": "dormer-piece",
            "piece_index": 1,
            "target_inclination_deg": 40.0,
            "support_score": 0.9,
            "final_layer": False,
            "final_layer_reason": "candidate_oblique",
            "corners": [
                [1.6, 3.8, 0.8],
                [2.6, 3.8, 0.8],
                [2.6, 3.8, 1.4],
                [1.6, 3.8, 1.4],
            ],
            "holes": [],
            "area_xz_m2": 0.6,
        },
    ]
    cfg = GlobalSelectionConfig(
        enabled=True,
        envelope_hard_buffer_m=0.0,
        envelope_soft_buffer_m=0.0,
        envelope_soft_outside_budget_fraction=0.0,
        max_cell_hard_outside_fraction=0.05,
        dormer_max_area_m2=2.0,
        dormer_min_outside_fraction=0.05,
        dormer_max_outside_fraction=0.7,
    )

    out = apply_global_xy_selection(rows, building, cfg)
    by_id = {str(row.get("piece_id")): row for row in out}
    dormer = by_id["dormer-piece"]

    assert dormer["xy_global_selector_dormer_exception"] is True
    assert dormer["final_layer"] is True
    assert dormer["final_layer_reason"] == "xy_global_selector_selected_candidate"
    assert dormer["overlay_suppressed"] is False
    assert dormer["area_xz_m2"] == pytest.approx(0.6, abs=1e-6)
    xs = [float(corner[0]) for corner in dormer["corners"]]
    assert max(xs) == pytest.approx(2.6, abs=1e-6)


def test_global_xy_selection_uses_stored_target_plane_for_rebuilt_candidate() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [3, 0, 0], [3, 0, 2], [0, 0, 2]],
            }
        ]
    }
    rows = [
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-a",
            "piece_id": "piece-a",
            "piece_index": 0,
            "target_inclination_deg": 30.0,
            "support_score": 0.7,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "corners": [
                [0.0, 3.0, 0.0],
                [1.5, 3.0, 0.0],
                [1.5, 3.0, 2.0],
                [0.0, 3.0, 2.0],
            ],
            "holes": [],
            "area_xz_m2": 3.0,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-b",
            "piece_id": "piece-b",
            "piece_index": 1,
            "target_inclination_deg": 45.0,
            "support_score": 0.8,
            "final_layer": False,
            "final_layer_reason": "candidate_oblique",
            "target_plane_coeffs": [1.0, 1.0, 0.0, -4.0],
            "corners": [
                [1.5, 4.0, 0.0],
                [3.0, 4.0, 0.0],
                [3.0, 4.0, 2.0],
                [1.5, 2.5, 2.0],
            ],
            "holes": [],
            "area_xz_m2": 3.0,
        },
    ]
    cfg = GlobalSelectionConfig(
        enabled=True,
        objective_weight_support=1.0,
        objective_weight_topness=2.5,
        objective_weight_prior_final=0.0,
        envelope_hard_buffer_m=0.0,
        envelope_soft_buffer_m=0.0,
        envelope_soft_outside_budget_fraction=1.0,
    )

    out = apply_global_xy_selection(rows, building, cfg)
    by_id = {str(row.get("piece_id")): row for row in out}
    piece_b = by_id["piece-b"]

    ys = sorted({round(float(corner[1]), 6) for corner in piece_b["corners"]})
    assert piece_b["final_layer"] is True
    assert piece_b["final_layer_reason"] == "xy_global_selector_selected_candidate"
    assert ys == pytest.approx([1.0, 2.5], abs=1e-6)


def test_global_xy_selection_contract_blocks_hard_invalid_pre_rows() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [[0, 0, 0], [3, 0, 0], [3, 0, 2], [0, 0, 2]],
            }
        ]
    }
    rows = [
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-a",
            "piece_id": "piece-a",
            "piece_index": 0,
            "target_inclination_deg": 30.0,
            "support_score": 0.7,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "corners": [[0, 3.0, 0], [1.6, 3.0, 0], [1.6, 3.0, 2], [0, 3.0, 2]],
            "holes": [],
            "area_xz_m2": 3.2,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-b",
            "piece_id": "piece-b",
            "piece_index": 1,
            "target_inclination_deg": 35.0,
            "support_score": 0.8,
            "final_layer": False,
            "final_layer_reason": "candidate_oblique",
            "corners": [[1.4, 4.0, 0], [3.0, 4.0, 0], [3.0, 4.0, 2], [1.4, 4.0, 2]],
            "holes": [],
            "area_xz_m2": 3.2,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": "ridge-bad",
            "piece_id": "ridge-bad",
            "piece_index": 2,
            "target_inclination_deg": 40.0,
            "support_score": 0.99,
            "final_layer": False,
            "final_layer_reason": "ridge_eave_unanchored",
            "corners": [[1.4, 6.0, 0], [3.0, 6.0, 0], [3.0, 6.0, 2], [1.4, 6.0, 2]],
            "holes": [],
            "area_xz_m2": 3.2,
        },
    ]
    cfg = GlobalSelectionConfig(
        enabled=True,
        objective_weight_support=1.0,
        objective_weight_topness=2.5,
        objective_weight_prior_final=0.0,
        envelope_hard_buffer_m=0.0,
        envelope_soft_buffer_m=0.0,
        envelope_soft_outside_budget_fraction=1.0,
    )

    out = apply_global_xy_selection(rows, building, cfg)
    by_id = {str(row.get("piece_id")): row for row in out}
    ridge = by_id["ridge-bad"]
    piece_b = by_id["piece-b"]

    assert ridge.get("xy_global_selector_hard_valid_pre") is False
    assert ridge.get("xy_global_selector_hard_filter_reason") == "ridge_eave_unanchored"
    assert ridge.get("xy_global_selector_applied") is not True
    assert ridge.get("final_layer") is False
    assert piece_b.get("final_layer") is True
    assert piece_b.get("final_layer_reason") == "xy_global_selector_selected_candidate"


def test_global_xy_selection_uses_rain_exposed_floor_domain_with_gaps() -> None:
    building = {
        "rooms": [
            {
                "id": "room:0",
                "story": 0,
                "floor_polygon": [[0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1]],
            },
            {
                "id": "room:1",
                "story": 0,
                "floor_polygon": [[2, 0, 0], [4, 0, 0], [4, 0, 1], [2, 0, 1]],
            },
        ]
    }
    roof_result = {
        "ceiling": {
            "exposed_rooms": [
                {"story": 0, "room_index": 0},
            ]
        },
        "building_part_graph": {
            "room_membership": {
                "room:0": ["building-part:main"],
                "room:1": ["building-part:ext"],
            }
        },
    }
    story_gap_polygons = {
        0: [Polygon([(2.0, 0.0), (2.4, 0.0), (2.4, 1.0), (2.0, 1.0)])],
    }
    rows = [
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-a",
            "piece_id": "piece-a",
            "piece_index": 0,
            "target_inclination_deg": 30.0,
            "support_score": 0.8,
            "final_layer": True,
            "final_layer_reason": "committed_relation_owner",
            "piece_part_ids": ["building-part:main", "building-part:ext"],
            "corners": [[0, 3.0, 0], [4, 3.0, 0], [4, 3.0, 1], [0, 3.0, 1]],
            "holes": [],
            "area_xz_m2": 4.0,
        },
        {
            "uuid": "b-test",
            "story": 0,
            "target_kind": "candidate_oblique",
            "piece_role": "supported",
            "target_element_id": "roof-b",
            "piece_id": "piece-b",
            "piece_index": 1,
            "target_inclination_deg": 35.0,
            "support_score": 0.7,
            "final_layer": False,
            "final_layer_reason": "candidate_oblique",
            "piece_part_ids": ["building-part:main"],
            "corners": [[2.4, 4.0, 0], [4.0, 4.0, 0], [4.0, 4.0, 1], [2.4, 4.0, 1]],
            "holes": [],
            "area_xz_m2": 1.6,
        },
    ]
    cfg = GlobalSelectionConfig(
        enabled=True,
        envelope_hard_buffer_m=0.0,
        envelope_soft_buffer_m=0.0,
        max_cell_hard_outside_fraction=0.0,
        envelope_soft_outside_budget_fraction=1.0,
        objective_weight_prior_final=0.0,
        dormer_max_area_m2=1.0,
    )

    out = apply_global_xy_selection(
        rows,
        building,
        cfg,
        roof_result=roof_result,
        story_gap_polygons=story_gap_polygons,
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    piece_b = by_id["piece-b"]
    piece_a_parts = [
        row for row in out if str(row.get("piece_id") or "").startswith("piece-a")
    ]
    total_piece_a_area = float(
        sum(float(row.get("area_xz_m2") or 0.0) for row in piece_a_parts)
    )
    max_piece_a_x = max(
        float(corner[0])
        for row in piece_a_parts
        for corner in (row.get("corners") or [])
        if len(corner) >= 3
    )

    assert total_piece_a_area == pytest.approx(2.4, abs=1e-6)
    assert piece_b["overlay_suppressed"] is True
    assert max_piece_a_x == pytest.approx(2.4, abs=1e-6)


def test_layer_policy_enforces_single_final_owner_per_xz_across_target_kinds() -> None:
    committed_target = "b-test::roof-oblique::oblique:0"
    ridge_target = "b-test::ridge-eave-candidate::plane-group::x"
    contexts = {
        committed_target: PlaneContext(
            target_id=committed_target,
            story=0,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=0,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(contexts_by_target=contexts, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-owner",
            "support_score": 0.95,
            "area_xz_m2": 16.0,
            "corners": [[0, 3, 0], [4, 3, 0], [4, 3, 4], [0, 3, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-overlap",
            "support_score": 0.9,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "area_xz_m2": 16.0,
            "corners": [[0, 4, 0], [4, 4, 0], [4, 4, 4], [0, 4, 4]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(rows, graph, LayerPolicyConfig())
    by_id = {str(row.get("piece_id")): row for row in out}
    committed = by_id["committed-owner"]
    ridge = by_id["ridge-overlap"]

    assert committed["final_layer"] is True
    assert committed.get("overlay_suppressed") is not True
    assert ridge["final_layer"] is True
    assert ridge.get("overlay_suppressed") is True
    assert ridge.get("xy_conflict_clip_reason") == "final_layer_conflict"


def test_layer_policy_uses_stored_target_plane_for_conflict_clip() -> None:
    committed_target = "b-test::roof-oblique::oblique:0"
    ridge_target = "b-test::ridge-eave-candidate::plane-group::x"
    contexts = {
        committed_target: PlaneContext(
            target_id=committed_target,
            story=0,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_target: PlaneContext(
            target_id=ridge_target,
            story=0,
            source_edge_ids=("chain-1",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(contexts_by_target=contexts, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-owner",
            "support_score": 0.95,
            "area_xz_m2": 8.0,
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 4], [0, 3, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_target,
            "piece_id": "ridge-overlap",
            "support_score": 0.9,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[4.0, 0.0], [4.0, 4.0]]],
            "target_plane_coeffs": [1.0, 1.0, 0.0, -4.0],
            "area_xz_m2": 16.0,
            "corners": [
                [0.0, 4.0, 0.0],
                [4.0, 4.0, 0.0],
                [4.0, 4.0, 4.0],
                [0.0, 4.0, 4.0],
            ],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(rows, graph, LayerPolicyConfig())
    by_id = {str(row.get("piece_id")): row for row in out}
    ridge = by_id["ridge-overlap"]

    xs = sorted({round(float(corner[0]), 6) for corner in ridge["corners"]})
    ys = sorted({round(float(corner[1]), 6) for corner in ridge["corners"]})
    assert ridge["final_layer"] is True
    assert ridge.get("overlay_suppressed") is not True
    assert ridge.get("xy_conflict_clipped") is True
    assert xs == pytest.approx([2.0, 4.0], abs=1e-6)
    assert ys == pytest.approx([0.0, 2.0], abs=1e-6)


def test_layer_policy_priority_prefers_candidate_over_ridge() -> None:
    candidate_priority = _final_piece_priority(
        {
            "target_kind": "candidate_oblique",
            "support_score": 0.0,
            "area_xz_m2": 0.0,
        }
    )
    ridge_priority = _final_piece_priority(
        {
            "target_kind": "ridge_eave_plane_group",
            "support_score": 0.0,
            "area_xz_m2": 0.0,
        }
    )
    assert candidate_priority > ridge_priority


def test_layer_policy_suppresses_unanchored_ridge_before_final_conflict() -> None:
    committed_target = "b-test::roof-oblique::oblique:0"
    ridge_bad_target = "b-test::ridge-eave-candidate::plane-group::bad"
    ridge_good_target = "b-test::ridge-eave-candidate::plane-group::good"
    contexts = {
        committed_target: PlaneContext(
            target_id=committed_target,
            story=0,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_bad_target: PlaneContext(
            target_id=ridge_bad_target,
            story=0,
            source_edge_ids=("chain-bad",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_good_target: PlaneContext(
            target_id=ridge_good_target,
            story=0,
            source_edge_ids=("chain-good",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(contexts_by_target=contexts, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-owner",
            "support_score": 0.95,
            "area_xz_m2": 8.0,
            "corners": [[0, 3, 0], [2, 3, 0], [2, 3, 4], [0, 3, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_bad_target,
            "piece_id": "ridge-bad",
            "support_score": 0.95,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[0.0, 0.0], [0.0, 4.0]]],
            "area_xz_m2": 16.0,
            "corners": [[0, 4, 0], [4, 4, 0], [4, 4, 4], [0, 4, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_good_target,
            "piece_id": "ridge-good",
            "support_score": 0.90,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[4.0, 0.0], [4.0, 4.0]]],
            "area_xz_m2": 16.0,
            "corners": [[0, 4, 0], [4, 4, 0], [4, 4, 4], [0, 4, 4]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(rows, graph, LayerPolicyConfig())
    by_id = {str(row.get("piece_id")): row for row in out}
    ridge_bad = by_id["ridge-bad"]
    ridge_good = by_id["ridge-good"]

    assert ridge_bad["overlay_suppressed"] is True
    assert ridge_bad["final_layer"] is False
    assert ridge_bad["final_layer_reason"] == "ridge_eave_post_clip_unanchored"

    assert ridge_good["final_layer"] is True
    assert ridge_good.get("overlay_suppressed") is not True


def test_layer_policy_rescues_one_unanchored_ridge_when_it_would_leave_gap() -> None:
    committed_target = "b-test::roof-oblique::oblique:0"
    ridge_hi_target = "b-test::ridge-eave-candidate::plane-group::hi"
    ridge_lo_target = "b-test::ridge-eave-candidate::plane-group::lo"
    contexts = {
        committed_target: PlaneContext(
            target_id=committed_target,
            story=0,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_hi_target: PlaneContext(
            target_id=ridge_hi_target,
            story=0,
            source_edge_ids=("chain-hi",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_lo_target: PlaneContext(
            target_id=ridge_lo_target,
            story=0,
            source_edge_ids=("chain-lo",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(contexts_by_target=contexts, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-owner",
            "support_score": 0.95,
            "area_xz_m2": 10.0,
            "corners": [[0, 3, 0], [2.6, 3, 0], [2.6, 3, 4], [0, 3, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_hi_target,
            "piece_id": "ridge-hi",
            "support_score": 0.95,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[0.0, 0.0], [0.0, 4.0]]],
            "area_xz_m2": 16.0,
            "corners": [[0, 6, 0], [4, 6, 0], [4, 6, 4], [0, 6, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_lo_target,
            "piece_id": "ridge-lo",
            "support_score": 0.90,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[0.0, 0.0], [0.0, 4.0]]],
            "area_xz_m2": 16.0,
            "corners": [[0, 2, 0], [4, 2, 0], [4, 2, 4], [0, 2, 4]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    ridge_hi = by_id["ridge-hi"]
    ridge_lo = by_id["ridge-lo"]

    assert ridge_hi.get("overlay_suppressed") is True
    assert ridge_hi.get("final_layer_reason") == "ridge_eave_post_clip_unanchored"

    assert ridge_lo.get("overlay_suppressed") is not True
    assert ridge_lo.get("final_layer") is True
    assert (
        ridge_lo.get("final_layer_reason")
        == "ridge_eave_post_clip_rescued_for_coverage"
    )


def test_layer_policy_rescue_avoids_extreme_y_outlier() -> None:
    committed_target = "b-test::roof-oblique::oblique:0"
    ridge_normal_target = "b-test::ridge-eave-candidate::plane-group::normal"
    ridge_outlier_target = "b-test::ridge-eave-candidate::plane-group::outlier"
    contexts = {
        committed_target: PlaneContext(
            target_id=committed_target,
            story=0,
            source_edge_ids=(),
            source_edge_overlap_m=0.0,
            source_edge_alignment_deg=None,
            source_edge_height_residual_m=None,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_normal_target: PlaneContext(
            target_id=ridge_normal_target,
            story=0,
            source_edge_ids=("chain-normal",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
        ridge_outlier_target: PlaneContext(
            target_id=ridge_outlier_target,
            story=0,
            source_edge_ids=("chain-outlier",),
            source_edge_overlap_m=2.0,
            source_edge_alignment_deg=1.0,
            source_edge_height_residual_m=0.1,
            building_part_ids=("building-part:main",),
            source_part_ids=("building-part:main",),
            crossed_part_ids=("building-part:main",),
            in_main_body=True,
            in_extension=False,
            crosses_main_extension_boundary=False,
        ),
    }
    graph = RelationGraph(contexts_by_target=contexts, relations=[])
    rows = [
        {
            "target_kind": "committed_oblique",
            "piece_role": "supported",
            "target_element_id": committed_target,
            "piece_id": "committed-owner",
            "support_score": 0.95,
            "area_xz_m2": 10.0,
            "corners": [[0, 3, 0], [2.6, 3, 0], [2.6, 3, 4], [0, 3, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_normal_target,
            "piece_id": "ridge-normal",
            "support_score": 0.85,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[0.0, 0.0], [0.0, 4.0]]],
            "area_xz_m2": 16.0,
            "corners": [[0, 2, 0], [4, 2, 0], [4, 2, 4], [0, 2, 4]],
            "holes": [],
        },
        {
            "target_kind": "ridge_eave_plane_group",
            "creator_rain_area_fraction": 1.0,
            "piece_role": "supported",
            "target_element_id": ridge_outlier_target,
            "piece_id": "ridge-outlier",
            "support_score": 0.99,
            "source_part_overlap_fraction": 1.0,
            "local_competitor_target_ids": [],
            "mirror_partner_target_ids": ["mirror"],
            "source_edge_segments_xz": [[[0.0, 0.0], [0.0, 4.0]]],
            "area_xz_m2": 16.0,
            "corners": [[0, -20, 0], [4, -20, 0], [4, -20, 4], [0, -20, 4]],
            "holes": [],
        },
    ]

    out = classify_split_piece_rows(
        rows,
        graph,
        LayerPolicyConfig(ridge_post_clip_enable_rescue=True),
    )
    by_id = {str(row.get("piece_id")): row for row in out}
    normal = by_id["ridge-normal"]
    outlier = by_id["ridge-outlier"]

    assert normal.get("overlay_suppressed") is not True
    assert normal.get("final_layer") is True
    assert (
        normal.get("final_layer_reason") == "ridge_eave_post_clip_rescued_for_coverage"
    )

    assert outlier.get("overlay_suppressed") is True
    assert outlier.get("final_layer_reason") == "ridge_eave_post_clip_unanchored"


@pytest.fixture(scope="module")
def _real_payloads() -> tuple[list[dict], dict, dict[str, dict], dict[str, dict]]:
    repo = Path(__file__).resolve().parent.parent
    with (repo / "reconcile" / "buildings_3d.json").open() as handle:
        buildings = json.load(handle)
    with (repo / "reconcile" / "roof_algorithms_py_results.json").open() as handle:
        roof_results = json.load(handle)
    with (
        repo / "reports" / "ridge_eave_scores_20260420" / "scores.json"
    ).open() as handle:
        ridge_payload = json.load(handle)
    with (repo / "reconcile" / "reconcile_v3_results.json").open() as handle:
        v3_payload = json.load(handle)
    ridge_by_uuid = {
        str(entry.get("building_uuid")): entry
        for entry in (ridge_payload.get("buildings") or [])
        if entry.get("building_uuid")
    }
    v3_by_uuid = {
        str(entry.get("building_uuid")): entry
        for entry in v3_payload
        if entry.get("building_uuid")
    }
    return buildings, roof_results, ridge_by_uuid, v3_by_uuid


@pytest.mark.parametrize(
    "uuid",
    [
        "117d172e-00d6-436e-8df2-050f25977602",
        "c87c1e25-ff00-44ec-b823-b0966c81af70",
        "e0155eef-34a5-4642-bca6-39b83ee42af1",
    ],
)
def test_regression_uuid_emits_relation_explanations(
    uuid: str,
    _real_payloads: tuple[list[dict], dict, dict[str, dict], dict[str, dict]],
) -> None:
    buildings, roof_results, ridge_by_uuid, v3_by_uuid = _real_payloads
    selected = [building for building in buildings if str(building.get("uuid")) == uuid]
    if not selected:
        pytest.skip(f"missing building {uuid}")

    output = score_buildings_v2(
        selected,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_by_uuid,
        v3_results_by_uuid=v3_by_uuid,
    )

    sidecar_building = output.relation_sidecar["buildings"].get(uuid)
    assert sidecar_building is not None
    assert sidecar_building["contexts"]
    assert sidecar_building["relations"]

    ridge_contexts = [
        ctx
        for ctx in sidecar_building["contexts"]
        if "ridge-eave-candidate" in str(ctx.get("target_id") or "")
    ]
    assert ridge_contexts
    assert any(
        float(ctx.get("source_edge_overlap_m") or 0.0) > 0.0 for ctx in ridge_contexts
    )

    building_rows = [
        row for row in output.split_piece_rows if str(row.get("uuid")) == uuid
    ]
    assert building_rows
    ridge_rows = [
        row
        for row in building_rows
        if str(row.get("target_kind")) == "ridge_eave_plane_group"
    ]
    assert ridge_rows
    assert all("covering_target_ids" in row for row in ridge_rows)
    assert all(str(row.get("final_layer_reason") or "") for row in ridge_rows)


@pytest.mark.parametrize(
    ("uuid", "target_element_id"),
    [
        (
            "117d172e-00d6-436e-8df2-050f25977602",
            "117d172e-00d6-436e-8df2-050f25977602::ridge-eave-candidate::plane-group::31658ecf9141",
        ),
        (
            "c87c1e25-ff00-44ec-b823-b0966c81af70",
            "c87c1e25-ff00-44ec-b823-b0966c81af70::ridge-eave-candidate::plane-group::43aa23800ceb",
        ),
    ],
)
def test_regression_through_building_ridge_eave_piece_is_final(
    uuid: str,
    target_element_id: str,
    _real_payloads: tuple[list[dict], dict, dict[str, dict], dict[str, dict]],
) -> None:
    buildings, roof_results, ridge_by_uuid, v3_by_uuid = _real_payloads
    building = next(
        building for building in buildings if str(building.get("uuid")) == uuid
    )

    output = score_building_v2(
        building,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_by_uuid,
        v3_results_by_uuid=v3_by_uuid,
    )

    supported_rows = [
        row
        for row in output.split_piece_rows
        if str(row.get("target_element_id") or "") == target_element_id
        and str(row.get("piece_role") or "") == "supported"
    ]
    final_rows = [r for r in supported_rows if r.get("final_layer") is True]
    assert final_rows, (
        f"Expected at least one final supported piece for {target_element_id}"
    )
    assert any(
        r.get("final_layer_reason") == "ridge_eave_relation_owner" for r in final_rows
    ), (
        f"Expected ridge_eave_relation_owner reason; got "
        f"{[r.get('final_layer_reason') for r in final_rows]}"
    )


def test_regression_ilp_same_face_partition_keeps_full_face_without_sliver_hole(
    _real_payloads: tuple[list[dict], dict, dict[str, dict], dict[str, dict]],
) -> None:
    uuid = "5c557e06-393e-466e-a957-f7391b76b8ff"
    buildings, roof_results, ridge_by_uuid, v3_by_uuid = _real_payloads
    building = next(
        building for building in buildings if str(building.get("uuid")) == uuid
    )

    output = score_building_v2(
        building,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_by_uuid,
        v3_results_by_uuid=v3_by_uuid,
        config=ScorerV2Config(global_selection=GlobalSelectionConfig(enabled=True)),
    )

    row = next(
        row
        for row in output.split_piece_rows
        if str(row.get("piece_id") or "")
        == f"{uuid}::ceiling-oblique::ceiling-oblique:2#supported:0:0"
    )
    hole_areas = [
        float(Polygon([(point[0], point[2]) for point in hole]).area)
        for hole in (row.get("holes") or [])
        if len(hole) >= 3
    ]
    assert row["overlay_suppressed"] is False
    assert row["final_layer"] is True
    assert row["final_layer_reason"] == "xy_global_selector_selected_candidate"
    assert row["final_layer_reason_pre_ilp"] == "candidate_oblique"
    assert max(hole_areas, default=0.0) < 1e-3
    demoted = next(
        item
        for item in output.split_piece_rows
        if str(item.get("piece_id") or "")
        == f"{uuid}::roof-oblique::oblique:0#supported:0:0"
    )
    row_poly = Polygon(
        [(point[0], point[2]) for point in row["corners"]],
        holes=[
            [(point[0], point[2]) for point in hole]
            for hole in (row.get("holes") or [])
            if len(hole) >= 3
        ]
        or None,
    )
    demoted_poly = Polygon(
        [(point[0], point[2]) for point in demoted["corners"]],
        holes=[
            [(point[0], point[2]) for point in hole]
            for hole in (demoted.get("holes") or [])
            if len(hole) >= 3
        ]
        or None,
    )
    assert demoted["overlay_suppressed"] is False
    assert demoted["final_layer"] is True
    assert demoted["final_layer_reason"] == "committed_relation_owner"
    assert demoted["final_layer_reason_pre_ilp"] == "committed_relation_owner"
    assert float(row_poly.buffer(0).intersection(demoted_poly.buffer(0)).area) < 1e-5
    residual_ilp_rows = [
        other
        for other in output.split_piece_rows
        if str(other.get("piece_id") or "").startswith(
            f"{uuid}::ceiling-oblique::ceiling-oblique:1#supported:1:0:xy-ilp:"
        )
    ]
    assert (
        sum(float(other.get("area_xz_m2") or 0.0) for other in residual_ilp_rows) < 0.1
    )


def test_regression_ilp_includes_ridge_eave_in_local_xy_selection(
    _real_payloads: tuple[list[dict], dict, dict[str, dict], dict[str, dict]],
) -> None:
    uuid = "5c557e06-393e-466e-a957-f7391b76b8ff"
    buildings, roof_results, ridge_by_uuid, v3_by_uuid = _real_payloads
    building = next(
        building for building in buildings if str(building.get("uuid")) == uuid
    )

    output = score_building_v2(
        building,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_by_uuid,
        v3_results_by_uuid=v3_by_uuid,
        config=ScorerV2Config(global_selection=GlobalSelectionConfig(enabled=True)),
    )

    ridge_rows = [
        row
        for row in output.split_piece_rows
        if str(row.get("target_kind") or "") == "ridge_eave_plane_group"
        and str(row.get("piece_role") or "") == "supported"
    ]
    assert ridge_rows
    assert any(bool(row.get("xy_global_selector_applied")) for row in ridge_rows)

    def _piece_poly(row: dict) -> Polygon:
        shell = [
            (float(point[0]), float(point[2]))
            for point in (row.get("corners") or [])
            if len(point) >= 3
        ]
        holes = [
            [(float(point[0]), float(point[2])) for point in hole if len(point) >= 3]
            for hole in (row.get("holes") or [])
            if len(hole) >= 3
        ]
        poly = Polygon(shell, holes=holes or None)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    final_visible = [
        row
        for row in output.split_piece_rows
        if bool(row.get("final_layer")) and not bool(row.get("overlay_suppressed"))
    ]
    ridge_final = [
        row
        for row in final_visible
        if str(row.get("target_kind") or "") == "ridge_eave_plane_group"
    ]
    non_ridge_final = [
        row
        for row in final_visible
        if str(row.get("target_kind") or "") != "ridge_eave_plane_group"
    ]
    max_overlap = 0.0
    for ridge in ridge_final:
        ridge_poly = _piece_poly(ridge)
        if ridge_poly.is_empty:
            continue
        for other in non_ridge_final:
            other_poly = _piece_poly(other)
            if other_poly.is_empty:
                continue
            overlap = float(ridge_poly.intersection(other_poly).area)
            max_overlap = max(max_overlap, overlap)
    assert max_overlap < 1e-3


def test_regression_ilp_contract_does_not_resurrect_hard_invalid_rows(
    _real_payloads: tuple[list[dict], dict, dict[str, dict], dict[str, dict]],
) -> None:
    uuid = "d32d5562-5763-4c71-a816-6732c638fa6a"
    buildings, roof_results, ridge_by_uuid, v3_by_uuid = _real_payloads
    building = next(
        building for building in buildings if str(building.get("uuid")) == uuid
    )
    cfg = GlobalSelectionConfig(enabled=True)
    hard_invalid = set(cfg.hard_invalid_pre_reasons)
    ridge_allowed = set(cfg.ridge_allowed_pre_reasons)

    output = score_building_v2(
        building,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_by_uuid,
        v3_results_by_uuid=v3_by_uuid,
        config=ScorerV2Config(global_selection=cfg),
    )

    resurrected = [
        row
        for row in output.split_piece_rows
        if bool(row.get("final_layer"))
        and not bool(row.get("overlay_suppressed"))
        and str(row.get("final_layer_reason") or "")
        == "xy_global_selector_selected_candidate"
        and str(row.get("final_layer_reason_pre_ilp") or "") in hard_invalid
    ]
    assert not resurrected

    ridge_resurrected = [
        row
        for row in output.split_piece_rows
        if bool(row.get("final_layer"))
        and not bool(row.get("overlay_suppressed"))
        and str(row.get("target_kind") or "") == "ridge_eave_plane_group"
        and str(row.get("final_layer_reason") or "")
        == "xy_global_selector_selected_candidate"
        and str(row.get("final_layer_reason_pre_ilp") or "") not in ridge_allowed
    ]
    assert not ridge_resurrected


def test_runner_exposes_requested_public_api_names() -> None:
    building = {"uuid": "b-empty", "rooms": []}
    roof_results = {"b-empty": {"ceiling": {"planes": []}}}
    corpus = score_corpus_v2([building], roof_results)
    assert hasattr(corpus, "per_target_rows")
    alias = score_buildings_v2([building], roof_results)
    assert alias.per_target_rows == corpus.per_target_rows
    one = score_building_v2(building, roof_results)
    assert one.building_uuid == "b-empty"


def test_story_extent_envelope_fills_small_holes() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 1.7],
                    [0.0, 0.0, 1.7],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 2.3],
                    [4.0, 0.0, 2.3],
                    [4.0, 0.0, 4.0],
                    [0.0, 0.0, 4.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 1.7],
                    [1.7, 0.0, 1.7],
                    [1.7, 0.0, 2.3],
                    [0.0, 0.0, 2.3],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [2.3, 0.0, 1.7],
                    [4.0, 0.0, 1.7],
                    [4.0, 0.0, 2.3],
                    [2.3, 0.0, 2.3],
                ],
            },
        ]
    }

    envelope = legacy.build_story_extent_envelopes(building)[0]
    polys = legacy._iter_polygons(envelope)
    assert len(polys) == 1
    assert len(polys[0].interiors) == 0
    assert float(polys[0].area) == pytest.approx(16.0, abs=1e-6)


def test_story_extent_envelope_keeps_large_holes() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 3.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 4.0],
                    [0.0, 0.0, 4.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 3.0],
                    [0.0, 0.0, 3.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [3.0, 0.0, 1.0],
                    [4.0, 0.0, 1.0],
                    [4.0, 0.0, 3.0],
                    [3.0, 0.0, 3.0],
                ],
            },
        ]
    }

    envelope = legacy.build_story_extent_envelopes(building)[0]
    polys = legacy._iter_polygons(envelope)
    assert len(polys) == 1
    assert len(polys[0].interiors) == 1
    assert float(polys[0].area) == pytest.approx(12.0, abs=1e-6)


def _seam_supported_piece(
    target: legacy.TargetPlaneRecord,
    poly: Polygon,
    idx: int = 0,
) -> legacy.TargetSplitPieceRecord:
    coords = list(poly.exterior.coords)[:-1]
    corners = [[float(x), 3.0, float(z)] for x, z in coords]
    return legacy.TargetSplitPieceRecord(
        uuid=target.uuid,
        story=target.story,
        target_element_id=target.element_id,
        target_kind=target.target_kind,
        piece_id=f"{target.element_id}#supported:0:{idx}",
        piece_index=idx,
        piece_role="supported",
        area_xz_m2=float(poly.area),
        support_score=0.9,
        chain_ids=("chain-x",),
        corners=corners,
        holes=[],
    )


def test_intersection_seam_pieces_emitted_for_local_competitor_pair() -> None:
    target_a = _target(
        "b-test::ceiling-oblique::ceiling-oblique:1",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        azimuth_deg=0.0,
        kind="committed_oblique",
    )
    target_b = _target(
        "b-test::ceiling-oblique::ceiling-oblique:2",
        Polygon([(2.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0)]),
        azimuth_deg=90.0,
        kind="committed_oblique",
    )

    evidence_a_poly = Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)])
    evidence_b_poly = Polygon([(2.0, 1.0), (4.0, 1.0), (4.0, 4.0), (2.0, 4.0)])
    pieces = [
        _seam_supported_piece(target_a, evidence_a_poly, 0),
        _seam_supported_piece(target_b, evidence_b_poly, 0),
    ]

    relation = PlaneRelation(
        a_target_id=target_a.element_id,
        b_target_id=target_b.element_id,
        relation_kind="local_competitor",
        overlap_m2=4.0,
        azimuth_delta_deg=90.0,
        inclination_delta_deg=0.0,
        height_residual_m=0.0,
        same_story=True,
        same_part=False,
        shared_boundary_len_m=2.0,
    )
    graph = RelationGraph(
        contexts_by_target={target_a.element_id: None, target_b.element_id: None},
        relations=[relation],
    )

    seam_pieces, metadata = compute_intersection_seam_pieces(
        [target_a, target_b], pieces, graph
    )

    seam_by_target: dict[str, legacy.TargetSplitPieceRecord] = {}
    for piece in seam_pieces:
        assert piece.piece_role == "intersection_seam"
        seam_by_target.setdefault(piece.target_element_id, piece)
    assert set(seam_by_target) == {target_a.element_id, target_b.element_id}

    side_a = seam_by_target[target_a.element_id]
    side_b = seam_by_target[target_b.element_id]
    assert side_a.area_xz_m2 == pytest.approx(6.0, abs=1e-6)
    assert side_b.area_xz_m2 == pytest.approx(6.0, abs=1e-6)

    assert metadata[side_a.piece_id]["pair_partner_target_id"] == target_b.element_id
    assert metadata[side_b.piece_id]["pair_partner_target_id"] == target_a.element_id


def test_intersection_seam_pieces_skipped_when_no_local_competitor_relation() -> None:
    target_a = _target(
        "b-test::ceiling-oblique::ceiling-oblique:1",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        azimuth_deg=0.0,
        kind="committed_oblique",
    )
    target_b = _target(
        "b-test::ceiling-oblique::ceiling-oblique:2",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        azimuth_deg=180.0,
        kind="committed_oblique",
    )

    relation = PlaneRelation(
        a_target_id=target_a.element_id,
        b_target_id=target_b.element_id,
        relation_kind="mirror_pair",
        overlap_m2=8.0,
        azimuth_delta_deg=180.0,
        inclination_delta_deg=0.0,
        height_residual_m=0.0,
        same_story=True,
        same_part=True,
    )
    graph = RelationGraph(
        contexts_by_target={target_a.element_id: None, target_b.element_id: None},
        relations=[relation],
    )

    seam_pieces, metadata = compute_intersection_seam_pieces(
        [target_a, target_b], [], graph
    )
    assert seam_pieces == []
    assert metadata == {}


def test_intersection_seam_pieces_skipped_when_evidence_below_min_area() -> None:
    target_a = _target(
        "b-test::ceiling-oblique::ceiling-oblique:1",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        azimuth_deg=0.0,
        kind="committed_oblique",
    )
    target_b = _target(
        "b-test::ceiling-oblique::ceiling-oblique:2",
        Polygon([(2.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0)]),
        azimuth_deg=90.0,
        kind="committed_oblique",
    )

    tiny_b = Polygon([(2.5, 2.5), (2.7, 2.5), (2.7, 2.7), (2.5, 2.7)])
    big_a = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    pieces = [
        _seam_supported_piece(target_a, big_a),
        _seam_supported_piece(target_b, tiny_b),
    ]

    relation = PlaneRelation(
        a_target_id=target_a.element_id,
        b_target_id=target_b.element_id,
        relation_kind="local_competitor",
        overlap_m2=4.0,
        azimuth_delta_deg=90.0,
        inclination_delta_deg=0.0,
        height_residual_m=0.0,
        same_story=True,
        same_part=False,
    )
    graph = RelationGraph(
        contexts_by_target={target_a.element_id: None, target_b.element_id: None},
        relations=[relation],
    )

    seam_pieces, _metadata = compute_intersection_seam_pieces(
        [target_a, target_b], pieces, graph
    )
    assert seam_pieces == []


def _seam_residual_piece(
    target: legacy.TargetPlaneRecord,
    poly: Polygon,
    idx: int = 0,
) -> legacy.TargetSplitPieceRecord:
    coords = list(poly.exterior.coords)[:-1]
    corners = [[float(x), 3.0, float(z)] for x, z in coords]
    return legacy.TargetSplitPieceRecord(
        uuid=target.uuid,
        story=target.story,
        target_element_id=target.element_id,
        target_kind=target.target_kind,
        piece_id=f"{target.element_id}#residual:{idx}",
        piece_index=idx,
        piece_role="residual",
        area_xz_m2=float(poly.area),
        support_score=None,
        chain_ids=(),
        corners=corners,
        holes=[],
    )


def test_intersection_seam_pieces_extend_with_widened_target_polygon() -> None:
    # Both targets' poly_xz only barely overlap. Each side has been
    # widened upstream (supported + residual together cover a polygon
    # larger than poly_xz, simulating a successful eave widening). The
    # seam piece must use the widened polygon so it extends to the eave
    # rather than stopping at the fitted ring.
    target_a = _target(
        "b-test::ceiling-oblique::ceiling-oblique:1",
        Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        azimuth_deg=0.0,
        kind="committed_oblique",
    )
    target_b = _target(
        "b-test::ceiling-oblique::ceiling-oblique:2",
        Polygon([(2.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0)]),
        azimuth_deg=90.0,
        kind="committed_oblique",
    )

    evidence_a_poly = Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)])
    evidence_b_poly = Polygon([(2.0, 1.0), (4.0, 1.0), (4.0, 4.0), (2.0, 4.0)])
    # target_a has a residual piece extending to z=3 (beyond poly_xz at z=2)
    # -- i.e. widened to the eave. Without inheriting that widening, the
    # seam would stop at z=2.
    residual_a_extended = Polygon([(0.0, 2.0), (4.0, 2.0), (4.0, 3.0), (0.0, 3.0)])
    pieces = [
        _seam_supported_piece(target_a, evidence_a_poly, 0),
        _seam_residual_piece(target_a, residual_a_extended, 0),
        _seam_supported_piece(target_b, evidence_b_poly, 0),
    ]

    relation = PlaneRelation(
        a_target_id=target_a.element_id,
        b_target_id=target_b.element_id,
        relation_kind="local_competitor",
        overlap_m2=4.0,
        azimuth_delta_deg=90.0,
        inclination_delta_deg=0.0,
        height_residual_m=0.0,
        same_story=True,
        same_part=False,
        shared_boundary_len_m=2.0,
    )
    graph = RelationGraph(
        contexts_by_target={target_a.element_id: None, target_b.element_id: None},
        relations=[relation],
    )

    seam_pieces, _metadata = compute_intersection_seam_pieces(
        [target_a, target_b], pieces, graph
    )

    side_a = next(
        piece for piece in seam_pieces if piece.target_element_id == target_a.element_id
    )
    # Side-a seam = (target_a.poly_xz U residual_a_extended) - evidence_b.
    # Union spans xin[0,4], zin[0,3] = 12 m^2. evidence_b is xin[2,4], zin[1,4],
    # so overlap with the union is xin[2,4], zin[1,3] = 4 m^2. Result = 8 m^2.
    # Without seam extension the seam would be only poly_xz - evidence_b
    # = 8 - 2 = 6 m^2. The +2 m^2 comes from inheriting the residual.
    assert side_a.area_xz_m2 == pytest.approx(8.0, abs=1e-6)
    # The seam reaches up to z = 3 (the residual's far edge), past the
    # fitted poly_xz boundary at z = 2.
    assert max(corner[2] for corner in side_a.corners) >= 2.999
