from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from reconcile_tiers._core.newell import polygon_area_3d
from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.classify.roof_type import (
    AZIMUTH_TOL_DEG as _GABLE_AZIMUTH_TOL_DEG,
)
from reconcile_tiers.classify.roof_type import (
    GABLE_AREA_FRACTION as _GABLE_AREA_FRACTION,
)
from reconcile_tiers.classify.roof_type import (
    GABLE_MIN_RELIEF_M as _GABLE_MIN_RELIEF_M,
)
from reconcile_tiers.classify.roof_type import (
    GABLE_RIDGE_ABOVE_TOP_TOL_M as _GABLE_RIDGE_ABOVE_TOP_TOL_M,
)
from reconcile_tiers.classify.roof_type import (
    GABLE_RIDGE_BELOW_TOP_TOL_M as _GABLE_RIDGE_BELOW_TOP_TOL_M,
)
from reconcile_tiers.classify.roof_type import (
    INCL_TOL_DEG as _GABLE_INCL_TOL_DEG,
)
from reconcile_tiers.classify.roof_type import (
    MANSARD_CAP_Y_TOL_M as _MANSARD_CAP_Y_TOL_M,
)
from reconcile_tiers.classify.roof_type import (
    NEAR_FLAT_PITCH_DEG as _NEAR_FLAT_PITCH_DEG,
)
from reconcile_tiers.classify.roof_type import (
    PITCH_BAND_GAP_DEG as _PITCH_BAND_GAP_DEG,
)

TIER_LABELS: dict[int, str] = {
    1: "1 storey · flat",
    2: "2 storeys · flat",
    3: "3 storeys · flat",
    4: "Multi-storey · half-height",
    5: "1 storey · slanted room",
    6: "Gable (ridge + eave)",
    7: "Mixed: gable + flat roof",
    8: "Other",
}

_MIN_GABLE_INCL_DEG = 8.0


def _angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b) % 360.0
    return min(diff, 360.0 - diff)


def _polygon_area_3d(corners: Iterable[Iterable[float]]) -> float:
    return polygon_area_3d([list(point) for point in corners])


def _oblique_meta(surface: dict[str, Any]) -> tuple[float | None, float | None, float]:
    cluster = surface.get("cluster") or {}
    az = cluster.get("avgAzimuth")
    incl = cluster.get("avgIncl")
    try:
        az = float(az) if az is not None else None
    except (TypeError, ValueError):
        az = None
    try:
        incl = float(incl) if incl is not None else None
    except (TypeError, ValueError):
        incl = None
    return az, incl, _polygon_area_3d(surface.get("corners") or [])


def _fit_surface_plane(surface: dict[str, Any]) -> Plane | None:
    plane = Plane.fit(surface.get("corners") or [])
    return None if isinstance(plane, FitFailure) else plane


def _ridge_y_for_pair(
    surface_a: dict[str, Any], surface_b: dict[str, Any]
) -> float | None:
    plane_a = _fit_surface_plane(surface_a)
    plane_b = _fit_surface_plane(surface_b)
    if plane_a is None or plane_b is None:
        return None
    b_sum = plane_a.b + plane_b.b
    if abs(b_sum) < 1e-6:
        return None
    ridge_y = (plane_a.d + plane_b.d) / b_sum
    return ridge_y if math.isfinite(ridge_y) else None


def _observed_y_bounds(
    surface_a: dict[str, Any], surface_b: dict[str, Any]
) -> tuple[float, float] | None:
    ys: list[float] = []
    for surface in (surface_a, surface_b):
        for corner in surface.get("corners") or []:
            try:
                ys.append(float(corner[1]))
            except (TypeError, ValueError, IndexError):
                continue
    return None if not ys else (min(ys), max(ys))


def _has_plausible_ridge(surface_a: dict[str, Any], surface_b: dict[str, Any]) -> bool:
    ridge_y = _ridge_y_for_pair(surface_a, surface_b)
    bounds = _observed_y_bounds(surface_a, surface_b)
    if ridge_y is None or bounds is None:
        return False
    min_y, max_y = bounds
    if ridge_y < max_y - _GABLE_RIDGE_BELOW_TOP_TOL_M:
        return False
    if ridge_y > max_y + _GABLE_RIDGE_ABOVE_TOP_TOL_M:
        return False
    return ridge_y >= min_y + _GABLE_MIN_RELIEF_M


def _is_opposing_gable_pair(candidate_a, candidate_b) -> bool:
    surface_a, az_a, incl_a, _area_a = candidate_a
    surface_b, az_b, incl_b, _area_b = candidate_b
    if abs(_angle_diff_deg(az_a, az_b) - 180.0) > _GABLE_AZIMUTH_TOL_DEG:
        return False
    if abs(incl_a - incl_b) > _GABLE_INCL_TOL_DEG:
        return False
    return _has_plausible_ridge(surface_a, surface_b)


def _same_gable_side(candidate_a, candidate_b) -> bool:
    _surface_a, az_a, _incl_a, _area_a = candidate_a
    _surface_b, az_b, _incl_b, _area_b = candidate_b
    return _angle_diff_deg(az_a, az_b) <= _GABLE_AZIMUTH_TOL_DEG


def _surface_mean_y(surface: dict[str, Any]) -> float | None:
    ys: list[float] = []
    for corner in surface.get("corners") or []:
        try:
            ys.append(float(corner[1]))
        except (TypeError, ValueError, IndexError):
            continue
    return sum(ys) / len(ys) if ys else None


def _strip_artefact_lower_band(
    valid: list[tuple[dict[str, Any], float, float, float]],
) -> list[tuple[dict[str, Any], float, float, float]]:
    """Drop a low-pitch band when it sits at or below a higher-pitch band.

    Keeps mansard caps (low-pitch surfaces above pitched slopes) intact —
    those raise the lower-band mean y above the upper-band mean y.
    """
    if len(valid) < 4:
        return valid
    sorted_valid = sorted(valid, key=lambda item: item[2])
    inclinations = [incl for _surface, _az, incl, _area in sorted_valid]
    best_gap = 0.0
    best_idx = -1
    for i in range(1, len(sorted_valid)):
        gap = inclinations[i] - inclinations[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    if best_gap < _PITCH_BAND_GAP_DEG or best_idx < 1:
        return valid
    lower = sorted_valid[:best_idx]
    upper = sorted_valid[best_idx:]
    if not all(incl < _NEAR_FLAT_PITCH_DEG for _surface, _az, incl, _area in lower):
        return valid  # legitimate shallow slope tier (e.g. gambrel) — keep
    lower_ys = [
        y
        for surface, _az, _incl, _area in lower
        if (y := _surface_mean_y(surface)) is not None
    ]
    upper_ys = [
        y
        for surface, _az, _incl, _area in upper
        if (y := _surface_mean_y(surface)) is not None
    ]
    if not lower_ys or not upper_ys:
        return valid
    lower_mean_y = sum(lower_ys) / len(lower_ys)
    upper_mean_y = sum(upper_ys) / len(upper_ys)
    if lower_mean_y > upper_mean_y + _MANSARD_CAP_Y_TOL_M:
        return valid  # cap above slopes — true mansard, keep everything
    return upper


def detect_gable(oblique_surfaces: list[dict[str, Any]]) -> bool:
    metas = [(surface, *_oblique_meta(surface)) for surface in oblique_surfaces]
    valid = [
        (surface, az, incl, area)
        for surface, az, incl, area in metas
        if az is not None and incl is not None and incl >= _MIN_GABLE_INCL_DEG
    ]
    if len(valid) < 2:
        return False
    valid = _strip_artefact_lower_band(valid)
    if len(valid) < 2:
        return False
    total_area = sum(area for _surface, _az, _incl, area in valid)
    if total_area <= 0.0:
        return False

    best_group_area = 0.0
    for idx, candidate_i in enumerate(valid):
        for jdx, candidate_j in enumerate(valid[idx + 1 :], start=idx + 1):
            if not _is_opposing_gable_pair(candidate_i, candidate_j):
                continue
            group = {idx, jdx}
            for kdx, candidate_k in enumerate(valid):
                if kdx in group:
                    continue
                same_side = _same_gable_side(
                    candidate_k, candidate_i
                ) or _same_gable_side(candidate_k, candidate_j)
                if not same_side:
                    continue
                connects_to_opposite = any(
                    not _same_gable_side(candidate_k, valid[member_idx])
                    and _is_opposing_gable_pair(candidate_k, valid[member_idx])
                    for member_idx in group
                )
                if connects_to_opposite:
                    group.add(kdx)
            group_area = sum(valid[group_idx][3] for group_idx in group)
            best_group_area = max(best_group_area, group_area)
    return best_group_area >= _GABLE_AREA_FRACTION * total_area


def _n_stories(building: dict[str, Any]) -> int:
    raw = building.get("stories_found")
    if isinstance(raw, int) and raw > 0:
        return raw
    stories = set()
    for room in building.get("rooms") or []:
        try:
            stories.add(int(room.get("story")))
        except (TypeError, ValueError):
            continue
    return len(stories)


def _top_story(surfaces: list[dict[str, Any]]) -> int | None:
    stories = []
    for surface in surfaces:
        for key in ("story", "dominant_story"):
            raw = surface.get(key)
            if raw is None:
                continue
            try:
                stories.append(int(raw))
                break
            except (TypeError, ValueError):
                continue
    return max(stories) if stories else None


def _count_top_flat(flat: list[dict[str, Any]], oblique: list[dict[str, Any]]) -> int:
    if not flat:
        return 0
    top = _top_story(oblique)
    if top is None:
        top = _top_story(flat)
    if top is None:
        return len(flat)
    count = 0
    for surface in flat:
        try:
            if surface.get("story") is not None and int(surface["story"]) >= top:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def classify_building(
    building: dict[str, Any], roof: dict[str, Any] | None
) -> dict[str, Any]:
    rooms = building.get("rooms") or []
    roof_surfaces = ((roof or {}).get("roof_surfaces") or {}) if roof else {}
    oblique = roof_surfaces.get("oblique") or []
    flat = roof_surfaces.get("flat") or []

    n_stories = _n_stories(building)
    n_oblique = len(oblique)
    n_flat = _count_top_flat(flat, oblique)
    has_half_height = bool(building.get("split_level"))
    has_gable = detect_gable(oblique) if n_oblique >= 2 else False

    if n_oblique == 0 and not has_half_height:
        if n_stories == 1:
            tier = 1
        elif n_stories == 2:
            tier = 2
        elif n_stories == 3:
            tier = 3
        else:
            tier = 8
    elif has_half_height and n_oblique == 0:
        tier = 4
    elif has_gable and n_flat == 0:
        tier = 6
    elif has_gable and n_flat >= 1:
        tier = 7
    elif n_stories == 1 and n_oblique >= 1 and not has_gable:
        tier = 5
    else:
        tier = 8

    return {
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "signals": {
            "n_stories": n_stories,
            "n_rooms": len(rooms),
            "n_oblique": n_oblique,
            "n_flat": n_flat,
            "has_half_height": has_half_height,
            "has_gable": has_gable,
        },
    }
