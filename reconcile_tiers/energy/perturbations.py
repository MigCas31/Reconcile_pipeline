"""Corrective perturbations used for flag-queue impact scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reconcile_tiers.payload.schema import AdjacencyKind, TierPayload


@dataclass(frozen=True, slots=True)
class FillCoverageGap:
    area_m2: float
    adjacency: AdjacencyKind = AdjacencyKind.UNHEATED_ATTIC

    @property
    def kind(self) -> str:
        return "fill_coverage_gap"

    def apply(self, payload: TierPayload) -> TierPayload:
        # The scorer computes this perturbation analytically to avoid inventing
        # geometry. The immutable no-op keeps the ADT usable by dry-run tooling.
        return payload


@dataclass(frozen=True, slots=True)
class RestoreInversion:
    area_m2: float

    @property
    def kind(self) -> str:
        return "restore_inversion"

    def apply(self, payload: TierPayload) -> TierPayload:
        return payload


@dataclass(frozen=True, slots=True)
class ClipOutOfEnvelope:
    area_m2: float

    @property
    def kind(self) -> str:
        return "clip_out_of_envelope"

    def apply(self, payload: TierPayload) -> TierPayload:
        return payload


@dataclass(frozen=True, slots=True)
class RemoveSilentDrop:
    area_m2: float = 0.0

    @property
    def kind(self) -> str:
        return "remove_silent_drop"

    def apply(self, payload: TierPayload) -> TierPayload:
        return payload


Perturbation = FillCoverageGap | RestoreInversion | ClipOutOfEnvelope | RemoveSilentDrop


def _float_evidence(flag: dict[str, Any], *names: str) -> float:
    evidence = flag.get("evidence") or {}
    for name in names:
        value = evidence.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def perturbation_for_flag(
    _payload: TierPayload, flag: dict[str, Any]
) -> Perturbation | None:
    rule = flag.get("rule")
    if rule == "ceiling_coverage_gap":
        floor_area = _float_evidence(flag, "floor_area_m2")
        covered_area = _float_evidence(flag, "covered_area_m2")
        return FillCoverageGap(area_m2=max(0.0, floor_area - covered_area))
    if rule == "ceiling_orientation_inverted":
        return RestoreInversion(area_m2=_float_evidence(flag, "area_m2"))
    if rule == "out_of_envelope":
        return ClipOutOfEnvelope(
            area_m2=_float_evidence(flag, "outside_area_m2", "area_m2")
        )
    if rule == "silent_drop":
        return RemoveSilentDrop(area_m2=_float_evidence(flag, "area_m2"))
    return None
