"""Dataclasses for the tier-payload heat-loss sensitivity proxy."""

from __future__ import annotations

from dataclasses import dataclass, field

from reconcile_tiers.payload.schema import AdjacencyKind


@dataclass(frozen=True, slots=True)
class ThermalProperties:
    """Thermal assumptions for one adjacency boundary."""

    adjacency: AdjacencyKind
    u_value_w_m2k: float
    label: str


@dataclass(frozen=True, slots=True)
class BuildingThermalSummary:
    """Building-level totals emitted into flag-queue/v2."""

    annual_kwh_proxy: float
    total_heat_loss_w: float
    envelope_area_m2: float
    floor_area_m2: float
    u_value_table_id: str
    hdd_k_days: float
    disclaimer: str = "sensitivity proxy, not DIN-compliant"


@dataclass(frozen=True, slots=True)
class SurfaceHeatLoss:
    """Heat-loss contribution for one payload surface."""

    locator_id: str | None
    adjacency: AdjacencyKind
    area_m2: float
    u_value_w_m2k: float
    heat_loss_w: float


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    """Full result of estimating a tier payload."""

    summary: BuildingThermalSummary
    surfaces: list[SurfaceHeatLoss] = field(default_factory=list)
