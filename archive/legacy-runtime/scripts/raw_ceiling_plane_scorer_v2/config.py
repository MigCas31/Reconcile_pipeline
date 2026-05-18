from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationConfig:
    min_relation_overlap_m2: float = 0.05
    same_face_max_azimuth_delta_deg: float = 10.0
    same_face_max_inclination_delta_deg: float = 8.0
    same_face_max_height_residual_m: float = 0.5
    mirror_min_azimuth_delta_deg: float = 140.0
    mirror_max_azimuth_delta_deg: float = 220.0
    covering_min_overlap_fraction: float = 0.85
    continuous_run_max_inclination_delta_deg: float = 10.0
    continuous_run_max_height_residual_m: float = 0.25
    continuous_run_min_shared_boundary_m: float = 0.5


@dataclass(frozen=True)
class LayerPolicyConfig:
    min_source_edge_overlap_m: float = 0.5
    max_source_edge_alignment_deg: float = 35.0
    max_source_edge_height_residual_m: float = 0.75
    # ARKit depth precision at room-scale distances (~3-5 cm). 10 cm gives a
    # little slack for plane-fitting residuals while still rejecting planes
    # that float metres away from any observing chain.
    max_piece_anchor_height_residual_m: float = 0.10
    min_cross_part_source_overlap_fraction: float = 0.2
    same_face_shadow_min_overlap_fraction: float = 0.9
    xy_conflict_min_residual_area_m2: float = 0.1
    xy_conflict_min_residual_fraction: float = 0.01
    through_building_min_source_edge_count: int = 2
    through_building_min_piece_part_count: int = 2
    through_building_min_covering_azimuth_delta_deg: float = 70.0
    through_building_max_covering_azimuth_delta_deg: float = 110.0
    ridge_post_clip_source_edge_buffer_m: float = 0.5
    ridge_post_clip_max_source_edge_distance_m: float = 1.5
    ridge_post_clip_min_source_edge_overlap_m2: float = 0.05
    ridge_post_clip_overlap_check_min_area_m2: float = 1.0
    ridge_post_clip_rescue_max_visible_y_delta_m: float = 6.0
    ridge_post_clip_enable_rescue: bool = False


@dataclass(frozen=True)
class GlobalSelectionConfig:
    enabled: bool = False
    include_target_kinds: tuple[str, ...] = (
        "committed_oblique",
        "candidate_oblique",
        "ridge_eave_plane_group",
    )
    min_piece_area_m2: float = 0.05
    min_cell_area_m2: float = 0.03
    max_cells: int = 1200
    max_variables: int = 5000
    envelope_hard_buffer_m: float = 0.2
    envelope_soft_buffer_m: float = 0.05
    envelope_soft_outside_budget_fraction: float = 0.02
    max_cell_hard_outside_fraction: float = 0.05
    dormer_max_area_m2: float = 8.0
    dormer_min_inclination_deg: float = 10.0
    dormer_min_outside_fraction: float = 0.03
    dormer_max_outside_fraction: float = 0.65
    objective_weight_support: float = 2.0
    objective_weight_topness: float = 1.5
    objective_weight_prior_final: float = 0.5
    objective_weight_soft_outside: float = 3.0
    objective_weight_piece_activation: float = 0.03
    dormer_soft_outside_penalty_scale: float = 0.2
    solver_time_limit_s: float = 2.0
    hard_invalid_pre_reasons: tuple[str, ...] = (
        "ridge_eave_unanchored",
        "ridge_eave_covered_by_relation",
        "ridge_eave_relation_competitor",
        "ridge_eave_cross_part_unowned",
        "ridge_eave_post_clip_unanchored",
        "same_face_shadowed_by_committed",
        "same_face_chain_subset_shadowed",
        "committed_cross_part_unowned",
    )
    ridge_allowed_pre_reasons: tuple[str, ...] = (
        "ridge_eave_relation_owner",
        "ridge_eave_disjoint_extension_owner",
        "ridge_eave_through_building_owner",
        "ridge_eave_post_clip_rescued_for_coverage",
    )


@dataclass(frozen=True)
class ScorerV2Config:
    relation: RelationConfig = RelationConfig()
    layer_policy: LayerPolicyConfig = LayerPolicyConfig()
    global_selection: GlobalSelectionConfig = GlobalSelectionConfig()


DEFAULT_CONFIG = ScorerV2Config()
