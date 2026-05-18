"""Heat-loss sensitivity estimator for tier payloads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from reconcile_tiers.energy.areas import (
    ceiling_piece_area,
    room_floor_area,
    surface_area,
    wall_area_minus_cutouts,
)
from reconcile_tiers.energy.properties import (
    BuildingThermalSummary,
    EstimatorResult,
    SurfaceHeatLoss,
)
from reconcile_tiers.energy.u_values import DEFAULT_DK_TABLE, UValueTable
from reconcile_tiers.payload.schema import AdjacencyKind, TierPayload, payload_from_dict


def _payload(payload: TierPayload | dict[str, Any]) -> TierPayload:
    return payload_from_dict(payload) if isinstance(payload, dict) else payload


def _surface(
    *,
    locator_id: str | None,
    adjacency: AdjacencyKind,
    area_m2: float,
    table: UValueTable,
) -> SurfaceHeatLoss:
    u_value = table.u_value(adjacency)
    return SurfaceHeatLoss(
        locator_id=locator_id,
        adjacency=adjacency,
        area_m2=area_m2,
        u_value_w_m2k=u_value,
        heat_loss_w=u_value * area_m2 * table.design_delta_t_k,
    )


def estimate(
    payload: TierPayload | dict[str, Any], table: UValueTable = DEFAULT_DK_TABLE
) -> EstimatorResult:
    """Estimate heat loss from the parametric tier payload.

    This is intentionally a sensitivity proxy: it ranks geometry defects by
    likely impact; it is not a standards-compliant energy calculation.
    """

    p = _payload(payload)
    surfaces: list[SurfaceHeatLoss] = []
    floor_area_m2 = 0.0

    for room in p.rooms:
        floor_area_m2 += room_floor_area(room)
        for floor in room.floor:
            surfaces.append(
                _surface(
                    locator_id=room.locator_id,
                    adjacency=floor.adjacency,
                    area_m2=surface_area(floor.corners),
                    table=table,
                )
            )
        for wall in room.walls:
            surfaces.append(
                _surface(
                    locator_id=wall.locator_id,
                    adjacency=wall.adjacency,
                    area_m2=wall_area_minus_cutouts(wall),
                    table=table,
                )
            )

    for gap in p.gaps:
        surfaces.append(
            _surface(
                locator_id=gap.locator_id,
                adjacency=gap.adjacency,
                area_m2=surface_area(gap.corners),
                table=table,
            )
        )
    for ceiling in p.ceiling:
        surfaces.append(
            _surface(
                locator_id=ceiling.locator_id,
                adjacency=ceiling.adjacency,
                area_m2=ceiling_piece_area(ceiling),
                table=table,
            )
        )
    for wall in p.knee_walls:
        surfaces.append(
            _surface(
                locator_id=wall.locator_id,
                adjacency=wall.adjacency,
                area_m2=surface_area(wall.corners),
                table=table,
            )
        )
    for face in p.dormer_faces:
        surfaces.append(
            _surface(
                locator_id=face.locator_id,
                adjacency=face.adjacency,
                area_m2=wall_area_minus_cutouts(face),
                table=table,
            )
        )
    for closure in p.gable_closures:
        surfaces.append(
            _surface(
                locator_id=closure.locator_id,
                adjacency=closure.adjacency,
                area_m2=surface_area(closure.corners),
                table=table,
            )
        )

    envelope_area_m2 = sum(s.area_m2 for s in surfaces if s.u_value_w_m2k > 0.0)
    total_heat_loss_w = sum(s.heat_loss_w for s in surfaces)
    annual_kwh_proxy = (
        total_heat_loss_w / table.design_delta_t_k * table.hdd_k_days * 24.0 / 1000.0
    )
    return EstimatorResult(
        summary=BuildingThermalSummary(
            annual_kwh_proxy=annual_kwh_proxy,
            total_heat_loss_w=total_heat_loss_w,
            envelope_area_m2=envelope_area_m2,
            floor_area_m2=floor_area_m2,
            u_value_table_id=table.table_id,
            hdd_k_days=table.hdd_k_days,
        ),
        surfaces=surfaces,
    )


def estimator_summary_dict(result: EstimatorResult) -> dict[str, Any]:
    return asdict(result.summary)


def perturb_and_estimate(
    payload: TierPayload | dict[str, Any],
    perturbation: Any,
    table: UValueTable = DEFAULT_DK_TABLE,
) -> EstimatorResult:
    return estimate(perturbation.apply(_payload(payload)), table)
