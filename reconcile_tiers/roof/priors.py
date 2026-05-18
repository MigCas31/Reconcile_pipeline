"""Architectural-prior filter for oblique surfaces.

Real buildings follow architectural patterns: a gable has two mirror slopes,
a hip has four perpendicular slopes, a mansard has two pitch tiers, etc.
The bottom-up oblique extraction in `roof/` doesn't enforce these patterns,
so noise / mis-fit segments can produce stranded surfaces that "stick out
for no reason". This module classifies the building's roof type from its
unfiltered obliques, then drops surfaces that don't fit the architectural
prior.

Two safeguards:
  - Skip the filter when the building has too few obliques for the type to
    be meaningful (the type is more likely misclassified than violated).
  - Never strand: if applying the prior keeps zero obliques, fall back to
    keeping all of them.

Tolerances kept in lockstep with `build.py:_is_gable_partner_pair` and
`classify/roof_type.py`.
"""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.newell import polygon_area_3d
from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers.classify.roof_type import (
    APEX_TOL_M,
    PITCH_BAND_GAP_DEG,
    classify_oblique_roof,
)
from reconcile_tiers.classify.roof_type import (
    AZIMUTH_TOL_DEG as CLASSIFY_AZIMUTH_TOL_DEG,
)
from reconcile_tiers.classify.roof_type import (
    INCL_TOL_DEG as CLASSIFY_INCL_TOL_DEG,
)
from reconcile_tiers.payload.schema import RoofType
from reconcile_tiers.roof.geometry import angle_diff_deg
from reconcile_tiers.roof.roof import ObliqueSurface

GABLE_PARTNER_AZIMUTH_TOL_DEG = 40.0
GABLE_PARTNER_INCL_TOL_DEG = 10.0
COMPLEX_KEEP_AREA_M2 = 4.0
COMPLEX_STRANDED_OVERLAP_RATIO = 0.5
HIP_PARTNER_KEEP_AREA_M2 = 1.0


def _axis(surface: ObliqueSurface) -> float:
    return float(surface.cluster.avg_azimuth) % 180.0


def _gable_partner_pair(a: ObliqueSurface, b: ObliqueSurface) -> bool:
    az_diff = angle_diff_deg(a.cluster.avg_azimuth, b.cluster.avg_azimuth)
    if (
        not (180.0 - GABLE_PARTNER_AZIMUTH_TOL_DEG)
        <= az_diff
        <= (180.0 + GABLE_PARTNER_AZIMUTH_TOL_DEG)
    ):
        return False
    return abs(a.cluster.avg_incl - b.cluster.avg_incl) <= GABLE_PARTNER_INCL_TOL_DEG


def _has_partner(target: ObliqueSurface, all_obliques: list[ObliqueSurface]) -> bool:
    return any(
        _gable_partner_pair(target, other)
        for other in all_obliques
        if other is not target
    )


def _opposing_pairs(obliques: list[ObliqueSurface]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(obliques):
        for j, b in enumerate(obliques[i + 1 :], start=i + 1):
            az_diff = angle_diff_deg(a.cluster.avg_azimuth, b.cluster.avg_azimuth)
            if abs(az_diff - 180.0) > CLASSIFY_AZIMUTH_TOL_DEG:
                continue
            if abs(a.cluster.avg_incl - b.cluster.avg_incl) > CLASSIFY_INCL_TOL_DEG:
                continue
            pairs.append((i, j))
    return pairs


def _hip_perpendicular_partners(obliques: list[ObliqueSurface]) -> set[int]:
    pairs = _opposing_pairs(obliques)
    if len(pairs) < 2:
        return set()
    accepted: set[int] = set()
    for idx_a, pair_a in enumerate(pairs):
        for pair_b in pairs[idx_a + 1 :]:
            axis_a = _axis(obliques[pair_a[0]])
            axis_b = _axis(obliques[pair_b[0]])
            axis_diff = angle_diff_deg(axis_a % 180.0, axis_b % 180.0)
            if abs(axis_diff - 90.0) <= CLASSIFY_AZIMUTH_TOL_DEG:
                accepted.update(pair_a)
                accepted.update(pair_b)
    return accepted


def _mansard_tier_partners(obliques: list[ObliqueSurface]) -> set[int]:
    if len(obliques) < 2:
        return set()
    sorted_incl = sorted(
        ((float(o.cluster.avg_incl), idx) for idx, o in enumerate(obliques))
    )
    incls = [val for val, _ in sorted_incl]
    gaps = [(incls[i + 1] - incls[i], i) for i in range(len(incls) - 1)]
    if not gaps:
        return set()
    max_gap, split_idx = max(gaps)
    if max_gap < PITCH_BAND_GAP_DEG:
        return {i for i, o in enumerate(obliques) if _has_partner(o, obliques)}
    lower = {idx for _, idx in sorted_incl[: split_idx + 1]}
    upper = {idx for _, idx in sorted_incl[split_idx + 1 :]}
    accepted: set[int] = set()
    for tier in (lower, upper):
        tier_obs = [obliques[i] for i in tier]
        for i in tier:
            if _has_partner(obliques[i], tier_obs):
                accepted.add(i)
    return accepted


def _pyramid_partners(obliques: list[ObliqueSurface]) -> set[int]:
    if len(obliques) < 3:
        return set()
    apex_ys: list[float] = []
    for surface in obliques:
        if not surface.corners:
            return set()
        apex_ys.append(max(corner[1] for corner in surface.corners))
    median_y = sorted(apex_ys)[len(apex_ys) // 2]
    return {i for i, ay in enumerate(apex_ys) if abs(ay - median_y) <= APEX_TOL_M * 4.0}


def _xz_area(surface: ObliqueSurface) -> float:
    return polygon_area_3d(surface.corners)


def _xz_polygon(surface: ObliqueSurface):
    if len(surface.corners) < 3:
        return None
    poly = make_valid_polygon(
        Polygon([(float(corner[0]), float(corner[2])) for corner in surface.corners])
    )
    if poly is None or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _is_stranded_inside_partner_field(
    target: ObliqueSurface,
    partnered_surfaces: list[ObliqueSurface],
) -> bool:
    """A standalone complex-roof face should not repaint an established roof field.

    Complex roofs can include legitimate one-sided shed/extension slopes. Those
    should survive when they own a separate footprint region. The suspicious case
    is an unpartnered oblique whose XZ footprint is mostly already covered by
    surfaces that do form an opposing pair; visually that reads as a discontinuity
    through the same roof area rather than a separate roof wing.
    """
    target_poly = _xz_polygon(target)
    if target_poly is None:
        return False
    partner_polys = [
        poly
        for surface in partnered_surfaces
        if (poly := _xz_polygon(surface)) is not None
    ]
    if not partner_polys:
        return False
    overlap = target_poly.intersection(unary_union(partner_polys)).area
    return overlap / target_poly.area >= COMPLEX_STRANDED_OVERLAP_RATIO


def _min_obliques_for_type(roof_type: RoofType) -> int:
    return {
        RoofType.GABLE: 2,
        RoofType.HIP: 4,
        RoofType.CROSS_GABLE: 4,
        RoofType.MANSARD: 4,
        RoofType.PYRAMID: 4,
    }.get(roof_type, 0)


def _kept_indices_for_type(
    obliques: list[ObliqueSurface], roof_type: RoofType
) -> set[int] | None:
    """Return the per-type kept set, or None to skip filtering entirely."""
    n = len(obliques)
    if n < _min_obliques_for_type(roof_type):
        return None
    if roof_type == RoofType.NONE:
        # NONE means "no obliques observed by the classifier"; if we got here
        # with obliques, something's off — leave them alone for now.
        return None
    if roof_type == RoofType.SHED:
        if not obliques:
            return None
        return {max(range(n), key=lambda i: _xz_area(obliques[i]))}
    if roof_type == RoofType.GABLE:
        return {i for i, o in enumerate(obliques) if _has_partner(o, obliques)}
    if roof_type in (RoofType.HIP, RoofType.CROSS_GABLE):
        kept = _hip_perpendicular_partners(obliques)
        for i, o in enumerate(obliques):
            if _has_partner(o, obliques) and _xz_area(o) > HIP_PARTNER_KEEP_AREA_M2:
                kept.add(i)
        return kept
    if roof_type == RoofType.MANSARD:
        return _mansard_tier_partners(obliques)
    if roof_type == RoofType.PYRAMID:
        return _pyramid_partners(obliques)
    if roof_type == RoofType.COMPLEX:
        partnered_indices = {
            i for i, o in enumerate(obliques) if _has_partner(o, obliques)
        }
        partnered_surfaces = [obliques[i] for i in sorted(partnered_indices)]
        kept: set[int] = set(partnered_indices)
        for i, o in enumerate(obliques):
            if i in partnered_indices:
                continue
            if partnered_surfaces and _is_stranded_inside_partner_field(
                o, partnered_surfaces
            ):
                continue
            if _xz_area(o) >= COMPLEX_KEEP_AREA_M2:
                kept.add(i)
        return kept
    return None


def filter_obliques_by_roof_type(
    obliques: list[ObliqueSurface],
) -> tuple[list[ObliqueSurface], list[ObliqueSurface]]:
    """Drop obliques that don't fit the architectural prior implied by the
    classified roof type.

    Returns (kept, dropped). `dropped` is a debug aid; callers can ignore it.
    Order of kept is preserved from the input.

    Always classifies on the unfiltered set first so that the prior is
    applied based on the building's overall roof shape, not on a partial
    view shaped by an earlier filter.
    """
    if not obliques:
        return obliques, []
    roof_type = classify_oblique_roof(obliques)
    kept_indices = _kept_indices_for_type(obliques, roof_type)
    if kept_indices is None:
        return obliques, []
    if not kept_indices:
        # Never strand: if the prior would drop everything, keep everything.
        return obliques, []
    kept = [obliques[i] for i in range(len(obliques)) if i in kept_indices]
    dropped = [obliques[i] for i in range(len(obliques)) if i not in kept_indices]
    return kept, dropped
