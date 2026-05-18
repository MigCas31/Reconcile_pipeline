"""rule_ceiling_coverage_gap: room with <50% ceiling coverage above its floor."""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers.audit.rules._shared import (
    CEILING_COVERAGE_GAP_Y_SLACK_M,
    CEILING_COVERAGE_MIN,
    FlagItem,
    _corners_xz,
    _make_item,
    _room_floor_pieces,
    _safe_polygon,
    _y_range,
)


def rule_ceiling_coverage_gap(payload: dict[str, Any]) -> list[FlagItem]:
    rooms = payload.get("rooms") or []
    ceilings = payload.get("ceiling") or []
    items: list[FlagItem] = []
    for room in rooms:
        floor_polys: list[Polygon] = []
        for piece in _room_floor_pieces(room):
            poly = _safe_polygon(_corners_xz(piece.get("corners") or []))
            if poly is not None:
                floor_polys.append(poly)
        if not floor_polys:
            continue
        floor_poly = (
            floor_polys[0] if len(floor_polys) == 1 else unary_union(floor_polys)
        )
        if (
            not isinstance(floor_poly, Polygon)
            or floor_poly.is_empty
            or floor_poly.area < 1.0
        ):
            continue
        wall_y_max = None
        for wall in room.get("walls") or []:
            yr = _y_range(wall.get("corners") or [])
            if yr is None:
                continue
            wall_y_max = yr[1] if wall_y_max is None else max(wall_y_max, yr[1])

        slack_m = CEILING_COVERAGE_GAP_Y_SLACK_M
        candidate_polys = []
        for piece in ceilings:
            corners = piece.get("corners") or []
            yr = _y_range(corners)
            if yr is None:
                continue
            if wall_y_max is not None and yr[1] < wall_y_max - slack_m:
                continue
            poly = _safe_polygon(_corners_xz(corners))
            if poly:
                candidate_polys.append(poly)

        if not candidate_polys:
            covered_area = 0.0
        else:
            union = unary_union(candidate_polys)
            covered_area = floor_poly.intersection(union).area
        coverage = covered_area / floor_poly.area
        if coverage < CEILING_COVERAGE_MIN:
            severity = "high" if coverage < 0.20 else "med"
            items.append(
                _make_item(
                    room.get("locator_id"),
                    rule="ceiling_coverage_gap",
                    severity=severity,
                    evidence={
                        "story": room.get("story"),
                        "floor_area_m2": float(floor_poly.area),
                        "covered_area_m2": float(covered_area),
                        "coverage_ratio": float(coverage),
                    },
                )
            )
    items.sort(key=lambda it: it["evidence"]["coverage_ratio"])
    return items
