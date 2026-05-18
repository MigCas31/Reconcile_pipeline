from __future__ import annotations

import math

from reconcile_tiers._core.newell import polygon_area_3d
from reconcile_tiers.payload.schema import RoofType
from reconcile_tiers.roof.roof import ObliqueSurface

AZIMUTH_TOL_DEG = 30.0
INCL_TOL_DEG = 10.0
GABLE_AREA_FRACTION = 0.70
PITCH_BAND_GAP_DEG = 20.0
APEX_TOL_M = 0.25
GABLE_RIDGE_BELOW_TOP_TOL_M = 0.20
GABLE_RIDGE_ABOVE_TOP_TOL_M = 0.60
GABLE_MIN_RELIEF_M = 0.05
LOW_PITCH_MANSARD_MIN_DEG = 10.0
LOW_PITCH_MANSARD_MAX_DEG = 30.0
HIP_GROUP_MIN_INCL_DEG = 35.0
SAME_AXIS_TOL_DEG = 12.0
# A real mansard has a low-pitch upper deck sitting above steeper lower
# slopes. Misclassified gables show the inverse: low-pitch fragments (attic
# floor scraps, dormer pieces) sitting at or below the gable peaks. The
# tolerance is the minimum mean-y separation required to call the lower
# pitch band a true cap.
MANSARD_CAP_Y_TOL_M = 0.20
# Below this pitch a "slope" no longer drains as a roof — anything in this
# range is treated as near-flat (attic-floor scraps, dormer scraps, mansard
# cap). Above this we leave bands intact: a 17° lower tier is a real shallow
# slope (gambrel / two-pitch roof) and must not be silently dropped.
NEAR_FLAT_PITCH_DEG = 15.0


def _angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b) % 360.0
    return min(diff, 360.0 - diff)


def _axis_diff_deg(a: float, b: float) -> float:
    return _angle_diff_deg(a % 180.0, b % 180.0)


def _surface_area(surface: ObliqueSurface) -> float:
    return polygon_area_3d(surface.corners)


def _surface_meta(surface: ObliqueSurface) -> tuple[float, float, float]:
    return (
        float(surface.cluster.avg_azimuth) % 360.0,
        float(surface.cluster.avg_incl),
        _surface_area(surface),
    )


def _ridge_y_for_pair(
    surface_a: ObliqueSurface, surface_b: ObliqueSurface
) -> float | None:
    b_sum = float(surface_a.plane.b) + float(surface_b.plane.b)
    if abs(b_sum) < 1e-6:
        return None
    ridge_y = (float(surface_a.plane.d) + float(surface_b.plane.d)) / b_sum
    return ridge_y if math.isfinite(ridge_y) else None


def _observed_y_bounds(
    surface_a: ObliqueSurface, surface_b: ObliqueSurface
) -> tuple[float, float] | None:
    ys = [
        float(corner[1])
        for surface in (surface_a, surface_b)
        for corner in surface.corners
    ]
    return None if not ys else (min(ys), max(ys))


def _has_plausible_ridge(surface_a: ObliqueSurface, surface_b: ObliqueSurface) -> bool:
    ridge_y = _ridge_y_for_pair(surface_a, surface_b)
    bounds = _observed_y_bounds(surface_a, surface_b)
    if ridge_y is None or bounds is None:
        return False
    min_y, max_y = bounds
    if ridge_y < max_y - GABLE_RIDGE_BELOW_TOP_TOL_M:
        return False
    if ridge_y > max_y + GABLE_RIDGE_ABOVE_TOP_TOL_M:
        return False
    return ridge_y >= min_y + GABLE_MIN_RELIEF_M


def _opposing_pair_candidates(
    oblique: list[ObliqueSurface],
) -> list[tuple[int, int, float]]:
    metas = [_surface_meta(surface) for surface in oblique]
    candidates: list[tuple[int, int, float]] = []
    for idx, (az_i, incl_i, area_i) in enumerate(metas):
        for jdx, (az_j, incl_j, area_j) in enumerate(metas[idx + 1 :], start=idx + 1):
            if abs(_angle_diff_deg(az_i, az_j) - 180.0) > AZIMUTH_TOL_DEG:
                continue
            if abs(incl_i - incl_j) > INCL_TOL_DEG:
                continue
            if not _has_plausible_ridge(oblique[idx], oblique[jdx]):
                continue
            candidates.append((idx, jdx, area_i + area_j))
    candidates.sort(key=lambda item: -item[2])
    return candidates


def _opposing_pairs(oblique: list[ObliqueSurface]) -> list[tuple[int, int, float]]:
    candidates = _opposing_pair_candidates(oblique)
    used: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for idx, jdx, area in candidates:
        if idx in used or jdx in used:
            continue
        pairs.append((idx, jdx, area))
        used.update((idx, jdx))
    return pairs


def _dominant_gable_group_area(
    oblique: list[ObliqueSurface], candidates: list[tuple[int, int, float]]
) -> float:
    if not candidates:
        return 0.0
    metas = [_surface_meta(surface) for surface in oblique]
    best = 0.0
    for idx, jdx, _area in candidates:
        group = {idx, jdx}
        for kdx, (az_k, _incl_k, _area_k) in enumerate(metas):
            if kdx in group:
                continue
            same_side = any(
                _angle_diff_deg(az_k, metas[member_idx][0]) <= AZIMUTH_TOL_DEG
                for member_idx in group
            )
            if not same_side:
                continue
            connects_to_opposite = any(
                _angle_diff_deg(az_k, metas[member_idx][0]) > AZIMUTH_TOL_DEG
                and any(
                    {kdx, member_idx} == {cand_i, cand_j}
                    for cand_i, cand_j, _ in candidates
                )
                for member_idx in group
            )
            if connects_to_opposite:
                group.add(kdx)
        best = max(best, sum(metas[group_idx][2] for group_idx in group))
    return best


def _pair_axis(surface: ObliqueSurface) -> float:
    return float(surface.cluster.avg_azimuth) % 180.0


def _pairs_perpendicular(oblique: list[ObliqueSurface], pair_a, pair_b) -> bool:
    axis_a = _pair_axis(oblique[pair_a[0]])
    axis_b = _pair_axis(oblique[pair_b[0]])
    return abs(_angle_diff_deg(axis_a, axis_b) - 90.0) <= AZIMUTH_TOL_DEG


def _split_by_pitch_gap(
    oblique: list[ObliqueSurface],
) -> tuple[list[ObliqueSurface], list[ObliqueSurface]] | None:
    if len(oblique) < 4:
        return None
    sorted_obl = sorted(oblique, key=lambda surface: float(surface.cluster.avg_incl))
    inclinations = [float(surface.cluster.avg_incl) for surface in sorted_obl]
    best_gap = 0.0
    best_idx = -1
    for i in range(1, len(sorted_obl)):
        gap = inclinations[i] - inclinations[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    if best_gap < PITCH_BAND_GAP_DEG or best_idx < 1:
        return None
    return sorted_obl[:best_idx], sorted_obl[best_idx:]


def _all_near_flat(surfaces: list[ObliqueSurface]) -> bool:
    return bool(surfaces) and all(
        float(surface.cluster.avg_incl) < NEAR_FLAT_PITCH_DEG for surface in surfaces
    )


def _band_mean_y(surfaces: list[ObliqueSurface]) -> float | None:
    total = 0.0
    n = 0
    for surface in surfaces:
        for corner in surface.corners:
            total += float(corner[1])
            n += 1
    if n == 0:
        return None
    return total / n


def _is_mansard_cap_geometry(
    lower_band: list[ObliqueSurface], upper_band: list[ObliqueSurface]
) -> bool:
    lower_y = _band_mean_y(lower_band)
    upper_y = _band_mean_y(upper_band)
    if lower_y is None or upper_y is None:
        return False
    return lower_y > upper_y + MANSARD_CAP_Y_TOL_M


def _same_axis_low_pitch_mansard(oblique: list[ObliqueSurface]) -> bool:
    if len(oblique) < 4:
        return False
    axes = [_pair_axis(surface) for surface in oblique]
    ref = axes[0]
    if any(_axis_diff_deg(axis, ref) > SAME_AXIS_TOL_DEG for axis in axes[1:]):
        return False
    inclinations = [float(surface.cluster.avg_incl) for surface in oblique]
    return all(
        LOW_PITCH_MANSARD_MIN_DEG <= incl <= LOW_PITCH_MANSARD_MAX_DEG
        for incl in inclinations
    )


def _has_perpendicular_hip_groups(oblique: list[ObliqueSurface]) -> bool:
    groups: list[list[ObliqueSurface]] = []
    for surface in oblique:
        if float(surface.cluster.avg_incl) < HIP_GROUP_MIN_INCL_DEG:
            continue
        axis = _pair_axis(surface)
        for group in groups:
            if _axis_diff_deg(axis, _pair_axis(group[0])) <= AZIMUTH_TOL_DEG:
                group.append(surface)
                break
        else:
            groups.append([surface])
    strong_axes = [_pair_axis(group[0]) for group in groups if len(group) >= 2]
    for idx, axis in enumerate(strong_axes):
        for other in strong_axes[idx + 1 :]:
            if abs(_axis_diff_deg(axis, other) - 90.0) <= AZIMUTH_TOL_DEG:
                return True
    return False


def _converges_to_apex(oblique: list[ObliqueSurface]) -> bool:
    high_points = []
    for surface in oblique:
        if not surface.corners:
            return False
        max_y = max(corner[1] for corner in surface.corners)
        candidates = [
            corner for corner in surface.corners if abs(corner[1] - max_y) <= APEX_TOL_M
        ]
        if len(candidates) != 1:
            return False
        high_points.append(candidates[0])
    ref = high_points[0]
    for point in high_points[1:]:
        dx = float(point[0]) - float(ref[0])
        dy = float(point[1]) - float(ref[1])
        dz = float(point[2]) - float(ref[2])
        if (dx * dx + dy * dy + dz * dz) ** 0.5 > APEX_TOL_M:
            return False
    return True


def classify_oblique_roof(oblique: list[ObliqueSurface]) -> RoofType:
    if not oblique:
        return RoofType.NONE
    if len(oblique) == 1:
        return RoofType.SHED

    bands = _split_by_pitch_gap(oblique)
    if (
        bands is not None
        and _all_near_flat(bands[0])
        and not _is_mansard_cap_geometry(bands[0], bands[1])
    ):
        # Lower band is all near-flat and sits at/below the upper band —
        # scan artefacts (attic-floor scraps, dormer scraps), not a real
        # secondary slope tier or mansard cap. Re-classify against the
        # upper band only so the gable / hip / pyramid rules see the
        # actual roof structure.
        return classify_oblique_roof(bands[1])

    pair_candidates = _opposing_pair_candidates(oblique)
    pairs = _opposing_pairs(oblique)
    total_area = sum(_surface_area(surface) for surface in oblique) or 0.0
    if len(oblique) == 2 and pairs and pairs[0][2] >= GABLE_AREA_FRACTION * total_area:
        return RoofType.GABLE

    if len(oblique) == 4 and _converges_to_apex(oblique):
        return RoofType.PYRAMID

    if len(oblique) == 4 and len(pairs) == 2:
        return (
            RoofType.CROSS_GABLE
            if _pairs_perpendicular(oblique, pairs[0], pairs[1])
            else RoofType.HIP
        )

    if (
        _dominant_gable_group_area(oblique, pair_candidates)
        >= GABLE_AREA_FRACTION * total_area
    ):
        return RoofType.GABLE

    if _same_axis_low_pitch_mansard(oblique):
        return RoofType.MANSARD

    if _has_perpendicular_hip_groups(oblique):
        return RoofType.HIP

    if len(oblique) >= 4 and bands is not None:
        return RoofType.MANSARD

    return RoofType.COMPLEX
