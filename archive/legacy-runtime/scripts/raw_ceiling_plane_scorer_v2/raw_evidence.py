from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts import prototype_raw_ceiling_plane_scorer as legacy


@dataclass(frozen=True)
class RawEvidenceBundle:
    raw_records: list[legacy.RawPlaneRecord]
    raw_edges: list[legacy.RawEdgeRecord]
    conflicts: list[legacy.ConflictPairRecord]


def collect_raw_evidence(
    building: dict[str, Any],
    roof_result: dict[str, Any],
    split_targets: list[legacy.TargetPlaneRecord],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
) -> RawEvidenceBundle:
    raw_records = legacy.collect_raw_plane_records(
        building, roof_result, exposed_only=True
    )
    source_room_keys = legacy._source_room_keys_from_ridge_diagnostics(
        building,
        ridge_eave_target_diagnostics,
    )
    raw_records = legacy._augment_raw_records_with_source_rooms(
        building,
        roof_result,
        raw_records,
        source_room_keys,
    )
    raw_records = legacy.promote_raw_plane_support_records(raw_records, split_targets)
    raw_edges = legacy.collect_raw_edges(raw_records, roof_result)
    conflicts = legacy.collect_conflict_pairs(raw_records)
    return RawEvidenceBundle(
        raw_records=raw_records,
        raw_edges=raw_edges,
        conflicts=conflicts,
    )
