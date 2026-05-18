from __future__ import annotations

from scripts import prototype_raw_ceiling_plane_scorer as legacy


def score_targets(
    split_targets: list[legacy.TargetPlaneRecord],
    raw_records: list[legacy.RawPlaneRecord],
    raw_edges: list[legacy.RawEdgeRecord],
    conflicts: list[legacy.ConflictPairRecord],
    plane_chain_supports: list[legacy.PlaneEaveChainSupportRecord],
) -> list[dict[str, object]]:
    return [
        legacy.score_target(
            target,
            raw_records,
            raw_edges,
            conflicts,
            plane_chain_supports,
        )
        for target in split_targets
    ]
