from __future__ import annotations

from dataclasses import dataclass

from scripts import prototype_raw_ceiling_plane_scorer as legacy


@dataclass(frozen=True)
class ChainSupportBundle:
    eave_chains: list[legacy.EaveChainRecord]
    plane_chain_supports: list[legacy.PlaneEaveChainSupportRecord]


def collect_chain_support(
    uuid: str,
    split_targets: list[legacy.TargetPlaneRecord],
    raw_edges: list[legacy.RawEdgeRecord],
) -> ChainSupportBundle:
    eave_chains = legacy.build_eave_chains(uuid, raw_edges)
    plane_chain_supports = legacy.expand_plane_eave_chain_supports_by_facade_continuity(
        eave_chains,
        legacy.score_plane_eave_chain_supports(split_targets, eave_chains),
    )
    return ChainSupportBundle(
        eave_chains=eave_chains,
        plane_chain_supports=plane_chain_supports,
    )
