"""Helpers feeding `_room_gable_candidates` / `_ceiling_candidates`.

Originally a 1255-line module; split by responsibility into:

- `_misc`            cross-cutting utilities (planarity, story labels, lid Y,
                     `_lower_plane_half`)
- `kink_detection`   ridge-artifact detection, per-room kink Y
- `flat_synthesis`   FLAT_EMIT emission, mirror-oblique synthesis
- `slope_ownership`  who claims a slope (gable / computed-arrangement /
                     unsupported), `_gable_planes_dip_into_room`
- `raw_ownership`    promote raw scan planes into room-owned candidates

This package re-exports the union of all sub-module symbols, so existing
imports (`from reconcile_tiers.build_internals.ceiling_helpers import _foo`)
continue to work unchanged.
"""

from __future__ import annotations

from reconcile_tiers.build_internals.ceiling_helpers._misc import (
    _ceiling_polygon_is_planar,
    _flat_lid_y,
    _hybrid_room_lid_y,
    _lower_plane_half,
    _parse_room_idx,
    _raw_plane_xz_area,
    _room_oblique_raw_coverage,
    _story_labels,
    _upper_floor_coverage_xz,
    _vec3_at_y_from_polygon,
)
from reconcile_tiers.build_internals.ceiling_helpers.flat_synthesis import (
    _flat_ceiling_candidates_for_domain,
    _flat_room_ceiling_candidates,
    _room_has_oblique_evidence_for_flat_synthesis,
    _split_flat_for_unmirrored_obliques,
    _synthesis_flat_xz_union,
    _synthesised_flat_candidate_for_room,
)
from reconcile_tiers.build_internals.ceiling_helpers.kink_detection import (
    _gable_building_kink_y,
    _kink_flat_conflicts_with_oblique_roof,
    _kink_flat_is_high_ridge_artifact,
    _room_kink_y,
)
from reconcile_tiers.build_internals.ceiling_helpers.raw_ownership import (
    _candidate_xz_polygon,
    _compatible_owner_coverage,
    _raw_oblique_owner_candidates_for_room,
    _raw_plane_owner_candidates_for_room,
)
from reconcile_tiers.build_internals.ceiling_helpers.slope_ownership import (
    _computed_oblique_arrangement_owns_kink_slope,
    _flat_lower_room_below_closed_gable,
    _gable_oblique_shell_coverage,
    _gable_owns_partial_kink_slope,
    _gable_planes_dip_into_room,
    _local_rooms_near_slope,
    _roof_oblique_supports_kink_slope,
    _room_has_roof_detail_support,
    _unsupported_low_flat_room_kink_slope,
)

__all__ = [
    "_candidate_xz_polygon",
    "_ceiling_polygon_is_planar",
    "_compatible_owner_coverage",
    "_computed_oblique_arrangement_owns_kink_slope",
    "_flat_ceiling_candidates_for_domain",
    "_flat_lid_y",
    "_flat_lower_room_below_closed_gable",
    "_flat_room_ceiling_candidates",
    "_gable_building_kink_y",
    "_gable_oblique_shell_coverage",
    "_gable_owns_partial_kink_slope",
    "_gable_planes_dip_into_room",
    "_hybrid_room_lid_y",
    "_kink_flat_conflicts_with_oblique_roof",
    "_kink_flat_is_high_ridge_artifact",
    "_local_rooms_near_slope",
    "_lower_plane_half",
    "_parse_room_idx",
    "_raw_oblique_owner_candidates_for_room",
    "_raw_plane_owner_candidates_for_room",
    "_raw_plane_xz_area",
    "_roof_oblique_supports_kink_slope",
    "_room_has_oblique_evidence_for_flat_synthesis",
    "_room_has_roof_detail_support",
    "_room_kink_y",
    "_room_oblique_raw_coverage",
    "_split_flat_for_unmirrored_obliques",
    "_story_labels",
    "_synthesis_flat_xz_union",
    "_synthesised_flat_candidate_for_room",
    "_unsupported_low_flat_room_kink_slope",
    "_upper_floor_coverage_xz",
    "_vec3_at_y_from_polygon",
]
