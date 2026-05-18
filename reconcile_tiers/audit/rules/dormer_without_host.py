"""rule_dormer_without_host_slope: dormer face has no nearby roof oblique.

A real dormer cuts into a sloped roof — its corners should lie within a small
distance of an oblique ceiling piece. If no oblique ceiling sits close in xz
and within DORMER_HOST_SLOPE_MAX_DIST_M vertically, the dormer is orphaned.
"""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon

from reconcile_tiers.audit.rules._shared import (
    DORMER_HOST_SLOPE_MAX_DIST_M,
    FlagItem,
    _corners_xz,
    _make_item,
    _newell_normal,
    _safe_polygon,
    _y_range,
)


def rule_dormer_without_host_slope(payload: dict[str, Any]) -> list[FlagItem]:
    dormers = payload.get("dormer_faces") or []
    if not dormers:
        return []
    obliques: list[tuple[Polygon, tuple[float, float]]] = []
    for piece in payload.get("ceiling") or []:
        n = _newell_normal(piece.get("corners"))
        if n is None:
            continue
        if abs(n[1]) > 0.95:
            continue  # near-flat — not an oblique host candidate
        poly = _safe_polygon(_corners_xz(piece.get("corners")))
        if poly is None:
            continue
        yr = _y_range(piece.get("corners"))
        if yr is None:
            continue
        obliques.append((poly, yr))

    if not obliques:
        return [
            _make_item(
                d.get("locator_id"),
                rule="dormer_without_host_slope",
                severity="med",
                evidence={"reason": "no oblique ceiling pieces in payload"},
            )
            for d in dormers
        ]

    items: list[FlagItem] = []
    for dormer in dormers:
        corners = dormer.get("corners") or []
        dpoly = _safe_polygon(_corners_xz(corners))
        if dpoly is None:
            continue
        dyr = _y_range(corners)
        if dyr is None:
            continue
        nearest_dist = float("inf")
        for opoly, oyr in obliques:
            xz_dist = dpoly.distance(opoly)
            y_overlap = min(dyr[1], oyr[1]) - max(dyr[0], oyr[0])
            y_dist = 0.0 if y_overlap >= 0 else -y_overlap
            d = max(xz_dist, y_dist)
            if d < nearest_dist:
                nearest_dist = d
        if nearest_dist > DORMER_HOST_SLOPE_MAX_DIST_M:
            items.append(
                _make_item(
                    dormer.get("locator_id"),
                    rule="dormer_without_host_slope",
                    severity="med",
                    evidence={
                        "kind": dormer.get("kind"),
                        "nearest_oblique_distance_m": float(nearest_dist),
                        "threshold_m": DORMER_HOST_SLOPE_MAX_DIST_M,
                    },
                )
            )
    items.sort(key=lambda it: -it["evidence"]["nearest_oblique_distance_m"])
    return items
