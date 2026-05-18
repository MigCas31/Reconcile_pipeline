"""rule_out_of_envelope: geometry extends past convex footprint + buffer."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import (
    ENVELOPE_BUFFER_M,
    MIN_OUTSIDE_AREA_M2,
    MIN_OUTSIDE_RATIO,
    FlagItem,
    _building_envelope,
    _corners_xz,
    _make_item,
    _safe_polygon,
)


def rule_out_of_envelope(payload: dict[str, Any]) -> list[FlagItem]:
    rooms = payload.get("rooms") or []
    envelope = _building_envelope(rooms)
    if envelope is None:
        return []

    items: list[FlagItem] = []
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    for room in rooms:
        for wall in room.get("walls") or []:
            sources.append(("room.wall", [wall]))
    sources.append(("ceiling", payload.get("ceiling") or []))
    sources.append(("gap", payload.get("gaps") or []))
    sources.append(("knee_wall", payload.get("knee_walls") or []))
    sources.append(("dormer_face", payload.get("dormer_faces") or []))

    for kind_label, group in sources:
        for item in group:
            corners = item.get("corners")
            poly = _safe_polygon(_corners_xz(corners))
            if poly is None:
                continue
            outside = poly.difference(envelope)
            if outside.is_empty:
                continue
            ratio = outside.area / max(poly.area, 1e-9)
            if outside.area < MIN_OUTSIDE_AREA_M2 or ratio < MIN_OUTSIDE_RATIO:
                continue
            cx, cy = outside.representative_point().coords[0]
            severity = "high" if ratio > 0.5 or outside.area > 5.0 else "med"
            items.append(
                _make_item(
                    item.get("locator_id"),
                    rule="out_of_envelope",
                    severity=severity,
                    evidence={
                        "category": kind_label,
                        "polygon_area_xz_m2": float(poly.area),
                        "outside_area_xz_m2": float(outside.area),
                        "outside_ratio": float(ratio),
                        "outside_at_xz": [float(cx), float(cy)],
                        "envelope_buffer_m": ENVELOPE_BUFFER_M,
                    },
                )
            )
    items.sort(key=lambda it: -it["evidence"]["outside_area_xz_m2"])
    return items
