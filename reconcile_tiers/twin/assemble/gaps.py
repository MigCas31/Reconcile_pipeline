"""Step 8 (residual): Gap primitives from unclaimed regions of the
plan-view union.

Phase D scope: only floor-plan gaps — the interior holes of the
shapely union of room floor polygons. These are unclaimed regions
that no Room covers (gaps between rooms, missing corridors, scan
holes). Roof gaps and cross-story gaps land in a later phase.
"""

from __future__ import annotations

from collections.abc import Sequence

from reconcile_tiers.payload.schema import Vec3
from reconcile_tiers.twin.types import (
    Evidence,
    Gap,
    GapKind,
    InvariantViolation,
    Provenance,
    Wing,
)


def gaps_from_holes(
    holes: Sequence[tuple[float, list[tuple[float, float]]]],
    *,
    building_uuid: str,
    wings: Sequence[Wing],
) -> tuple[Gap, ...]:
    if not holes:
        return ()
    incident = tuple(w.id for w in wings)
    if not incident:
        return ()
    out: list[Gap] = []
    for idx, (y, ring) in enumerate(holes):
        polygon = tuple(Vec3(x=x, y=y, z=z) for x, z in ring)
        evidence = Evidence(
            provenance=Provenance(kind="computed", source="floor_polygon_union_hole"),
            geometry=polygon,
            parents=incident,
        )
        try:
            out.append(
                Gap(
                    id=f"{building_uuid}::gap::floor::{idx}",
                    kind=GapKind.FLOOR,
                    polygon=polygon,
                    incident_primitive_ids=incident,
                    evidence=(evidence,),
                )
            )
        except InvariantViolation:
            continue
    return tuple(out)
