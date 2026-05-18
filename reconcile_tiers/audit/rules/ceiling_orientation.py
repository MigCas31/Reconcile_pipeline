"""rule_ceiling_orientation_inverted: Newell normal Y < threshold."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import (
    CEILING_NORMAL_Y_MIN,
    FlagItem,
    _make_item,
    _newell_normal,
)


def rule_ceiling_orientation_inverted(payload: dict[str, Any]) -> list[FlagItem]:
    items: list[FlagItem] = []
    for piece in payload.get("ceiling") or []:
        n = _newell_normal(piece.get("corners"))
        if n is None:
            continue
        if n[1] < CEILING_NORMAL_Y_MIN:
            items.append(
                _make_item(
                    piece.get("locator_id"),
                    rule="ceiling_orientation_inverted",
                    severity="high",
                    evidence={
                        "normal": [float(x) for x in n],
                        "normal_y": float(n[1]),
                        "source": piece.get("source"),
                    },
                )
            )
    return items
