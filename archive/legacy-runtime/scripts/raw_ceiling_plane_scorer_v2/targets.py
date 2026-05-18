from __future__ import annotations

from typing import Any

from scripts import prototype_raw_ceiling_plane_scorer as legacy

TargetCollection = tuple[
    list[legacy.TargetPlaneRecord],
    list[legacy.TargetPlaneRecord],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[int, Any],
    dict[int, list[Any]],
]


def collect_targets_for_building(
    uuid: str,
    building: dict[str, Any],
    roof_result: dict[str, Any],
    ridge_eave_entry: dict[str, Any] | None,
    v3_building: dict[str, Any] | None,
) -> TargetCollection:
    story_extent_envelopes = legacy.build_story_extent_envelopes(building)
    story_gap_polygons = legacy.build_story_gap_polygons(building)

    targets = legacy.collect_target_plane_records(roof_result, uuid)
    ridge_eave_targets = legacy.collect_selected_ridge_eave_plane_group_targets(
        uuid,
        ridge_eave_entry,
        story_extent_envelopes,
    )
    ridge_eave_target_diagnostics = legacy.collect_ridge_eave_target_diagnostics(
        uuid,
        ridge_eave_entry,
        v3_building,
    )
    ridge_eave_target_anchor_masks = legacy.collect_ridge_eave_target_anchor_masks(
        uuid,
        ridge_eave_entry,
        building,
        v3_building,
        ridge_eave_target_diagnostics,
    )
    ridge_eave_targets = legacy.constrain_ridge_eave_targets_with_anchor_masks(
        ridge_eave_targets,
        ridge_eave_target_diagnostics,
        ridge_eave_target_anchor_masks,
    )
    split_targets = targets + ridge_eave_targets
    return (
        targets,
        split_targets,
        ridge_eave_target_diagnostics,
        ridge_eave_target_anchor_masks,
        story_extent_envelopes,
        story_gap_polygons,
    )
