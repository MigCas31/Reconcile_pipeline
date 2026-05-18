"""rule_wall_cutout_outside_outline: cutout protrudes past wall's outline.

These walls render as solid bands across windows because the renderer's
`THREE.ShapeUtils.triangulateShape` degenerates when a cutout corner sits
outside the wall outer polygon. The corpus had 669 such walls before the
cutout-clipping fixes (Fix A: 616 from the basement Y-split bug, Fix B:
~53 from gable-clip drift). After those fixes this rule should report ~0;
it stays in place as a regression guard.
"""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import (
    WALL_CUTOUT_DRIFT_TOL_M,
    FlagItem,
    _make_item,
    _y_range,
)


def rule_wall_cutout_outside_outline(payload: dict[str, Any]) -> list[FlagItem]:
    items: list[FlagItem] = []
    tol_m = WALL_CUTOUT_DRIFT_TOL_M
    for room in payload.get("rooms") or []:
        for wall in room.get("walls") or []:
            cutouts = wall.get("cutouts") or []
            if not cutouts:
                continue
            wall_corners = wall.get("corners") or []
            wy = _y_range(wall_corners)
            if wy is None:
                continue
            wy_lo, wy_hi = wy
            for cutout_idx, cutout in enumerate(cutouts):
                cy = _y_range(cutout.get("corners") or [])
                if cy is None:
                    continue
                cy_lo, cy_hi = cy
                excess = max(0.0, wy_lo - cy_lo) + max(0.0, cy_hi - wy_hi)
                if excess <= tol_m:
                    continue
                items.append(
                    _make_item(
                        wall.get("locator_id"),
                        rule="wall_cutout_outside_outline",
                        severity="medium",
                        evidence={
                            "wall_y_range": [round(wy_lo, 4), round(wy_hi, 4)],
                            "cutout_y_range": [round(cy_lo, 4), round(cy_hi, 4)],
                            "cutout_index": cutout_idx,
                            "excess_m": round(excess, 4),
                        },
                    )
                )
    return items
