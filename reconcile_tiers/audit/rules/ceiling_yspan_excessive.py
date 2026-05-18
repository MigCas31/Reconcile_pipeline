"""rule_ceiling_yspan_excessive: ceiling piece spans more than 2 m vertically.

A flat ceiling has near-zero Y span. A roof oblique can have several decimetres
to a meter or so of span across a single room. Anything above ~2 m means the
piece's corners belong to multiple physical surfaces, or it was clipped against
the wrong plane upstream. Cohort scan over 446 buildings (2026-05-08) found
this pattern in 127 buildings (28%) — it is the largest upstream contamination
class for plane-selection eval.
"""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import (
    CEILING_YSPAN_EXCESSIVE_M,
    FlagItem,
    _make_item,
    _polygon_area_3d,
    _y_range,
)


def rule_ceiling_yspan_excessive(payload: dict[str, Any]) -> list[FlagItem]:
    items: list[FlagItem] = []
    for piece in payload.get("ceiling") or []:
        corners = piece.get("corners") or []
        yr = _y_range(corners)
        if yr is None:
            continue
        span = yr[1] - yr[0]
        if span <= CEILING_YSPAN_EXCESSIVE_M:
            continue
        # Severity scales with how grossly the span exceeds the threshold.
        if span >= 5.0:
            severity = "high"
        elif span >= 3.0:
            severity = "medium"
        else:
            severity = "low"
        items.append(
            _make_item(
                piece.get("locator_id"),
                rule="ceiling_yspan_excessive",
                severity=severity,
                evidence={
                    "y_range": [float(yr[0]), float(yr[1])],
                    "y_span_m": float(span),
                    "threshold_m": float(CEILING_YSPAN_EXCESSIVE_M),
                    "n_corners": len(corners),
                    "area_m2": float(_polygon_area_3d(corners)),
                    "source": piece.get("source"),
                    "role": piece.get("role"),
                },
            )
        )
    return items
