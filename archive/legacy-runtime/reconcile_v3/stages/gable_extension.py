"""gable_extension — classify per-part roofspace as extension-grade gable.

Two-tier rule per V3Part:

Tier 1 (gable-geometry): exactly two slanted roofs on the part with opposing
azimuths, matching inclinations, horizontal ridge from plane intersection,
and ridge orientation consistent with the clustered slope direction.

Tier 2 (extension-grade): Tier 1 passes AND ridge runs along the part's
major axis AND footprint is elongated AND roof coverage below
``GABLE_EXTENSION_COVERAGE_MAX`` AND no dormers AND no architectural
flat ceilings on the top story (gap-fill flats excluded).

Status values:
- ``not_gable``: Tier 1 failed, or Tier 2 failed on multiple criteria.
- ``gable_complete``: Tier 1 passes, footprint already covered (nothing to extend).
- ``gable_along_extend``: all Tier 2 checks pass — safe to project planes
  across the full footprint.
- ``gable_cross_review``: Tier 1 passes but ridge runs across the short axis
  — flag for human review, do not auto-extend.
- ``gable_ambiguous``: classification could not be completed (bad footprint).

This is a pure post-processing stage over already-computed V3 fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import atan2, degrees, sqrt
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from ..audit import HypothesisTrace
from ..constants import (
    GABLE_AZIMUTH_OPPOSING_TOL_DEG,
    GABLE_EXTENSION_COVERAGE_MAX,
    GABLE_INCLINATION_MATCH_TOL_DEG,
    GABLE_MIN_ELONGATION,
    GABLE_RIDGE_HORIZONTAL_TOL,
    GABLE_RIDGE_ORIENTATION_TOL_DEG,
    GABLE_RIDGE_VS_MAJOR_TOL_DEG,
)
from ..models import (
    GableExtension,
    V3Building,
    V3FlatCeiling,
    V3Part,
    V3SlantedRoof,
)

V3Room = Any  # V3Room is referenced but not exported from ..models


def _plane_azimuth_inclination(
    plane: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Downslope azimuth (
        degrees,
        0=+Z,
        clockwise,
    ) and inclination from plane (a,b,c,d)."""
    a, b, c, _ = plane
    n = np.array([a, b, c], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return 0.0, 0.0
    n = n / norm
    if n[1] < 0:
        n = -n
    horiz = sqrt(float(n[0] * n[0] + n[2] * n[2]))
    incl = degrees(atan2(horiz, float(n[1])))
    # Downslope direction in XZ is opposite to the horizontal component of the
    # upward normal.
    azimuth = degrees(atan2(-float(n[0]), -float(n[2]))) % 360.0
    return azimuth, incl


def _angle_mod180(a: float, b: float) -> float:
    """Smallest angular difference for axis-like angles (mod 180°)."""
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


def _plane_intersection_direction(
    p0: tuple[float, float, float, float],
    p1: tuple[float, float, float, float],
) -> np.ndarray | None:
    n0 = np.array(p0[:3], dtype=float)
    n1 = np.array(p1[:3], dtype=float)
    d = np.cross(n0, n1)
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        return None
    return d / norm


def _plane_intersection_point(
    p0: tuple[float, float, float, float],
    p1: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Minimum-norm solution of the 2-plane system; any point on the ridge line."""
    A = np.array([p0[:3], p1[:3]], dtype=float)
    b = np.array([-p0[3], -p1[3]], dtype=float)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return sol.astype(float)


def _polygon_xz(footprint_xz: Iterable) -> ShapelyPolygon | None:
    pts = list(footprint_xz)
    if len(pts) < 3:
        return None
    try:
        poly = ShapelyPolygon([(p[0], p[2]) for p in pts])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _min_rotated_rect_axes(
    poly: ShapelyPolygon,
) -> tuple[float, float, float]:
    """Return (major_length, minor_length, major_axis_azimuth_deg_mod180)."""
    rect = poly.minimum_rotated_rectangle
    if rect.is_empty:
        return 0.0, 0.0, 0.0
    coords = list(rect.exterior.coords)[:4]
    edges: list[tuple[float, float]] = []
    for i in range(4):
        x0, z0 = coords[i]
        x1, z1 = coords[(i + 1) % 4]
        length = sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
        azimuth = degrees(atan2(x1 - x0, z1 - z0)) % 180.0
        edges.append((length, azimuth))
    edges.sort(reverse=True)
    major_len, major_az = edges[0]
    minor_len = edges[2][0]
    return float(major_len), float(minor_len), float(major_az)


def _part_roof_map(
    parts: list[V3Part], roofs: list[V3SlantedRoof]
) -> dict[str, list[V3SlantedRoof]]:
    part_room_sets = {p.id: set(p.room_ids) for p in parts}
    out: dict[str, list[V3SlantedRoof]] = {p.id: [] for p in parts}
    for roof in roofs:
        roof_room_ids = set(roof.trace.inputs.get("room_ids") or [])
        if not roof_room_ids:
            continue
        for pid, part_rooms in part_room_sets.items():
            if roof_room_ids & part_rooms:
                out[pid].append(roof)
    return out


def _part_dormer_counts(building: V3Building) -> dict[str, int]:
    roof_lookup = {r.id: r for r in building.slanted_roofs}
    counts = {p.id: 0 for p in building.parts}
    for dormer in building.dormers:
        roof = roof_lookup.get(dormer.roof_surface_id)
        if roof is None:
            continue
        rooms = set(roof.trace.inputs.get("room_ids") or [])
        for part in building.parts:
            if rooms & set(part.room_ids):
                counts[part.id] += 1
                break
    return counts


def _part_arch_flat_counts(building: V3Building, rooms: list[V3Room]) -> dict[str, int]:
    """Architectural flat ceilings on each part's top story.

    Gap-fill flats are excluded. Flats on lower stories are excluded because
    a gable on the top story is compatible with normal interior ceilings on
    lower stories of the same part.
    """
    room_story = {r.identifier: r.story for r in rooms}
    part_top_story = {
        p.id: (max(p.stories) if p.stories else None) for p in building.parts
    }
    counts = {p.id: 0 for p in building.parts}
    for fc in building.flat_ceilings:
        if _is_gap_fill_flat(fc):
            continue
        if fc.room_id is None:
            continue
        fc_story = room_story.get(fc.room_id)
        if fc_story is None:
            continue
        for part in building.parts:
            if fc.room_id not in part.room_ids:
                continue
            if part_top_story[part.id] is None:
                break
            if fc_story == part_top_story[part.id]:
                counts[part.id] += 1
            break
    return counts


def _is_gap_fill_flat(fc: V3FlatCeiling) -> bool:
    if fc.over == "gap":
        return True
    inputs = fc.trace.inputs or {}
    return bool(inputs.get("gap_id"))


def _ridge_endpoints(
    ridge_dir: np.ndarray,
    ridge_point: np.ndarray,
    footprint: ShapelyPolygon,
    major_len: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cx, cz = float(footprint.centroid.x), float(footprint.centroid.y)
    c = np.array([cx, float(ridge_point[1]), cz], dtype=float)
    t = float(np.dot(c - ridge_point, ridge_dir))
    mid = ridge_point + t * ridge_dir
    half = max(major_len, 1.0) * 0.5
    a = mid - half * ridge_dir
    b = mid + half * ridge_dir
    return (
        (float(a[0]), float(a[1]), float(a[2])),
        (float(b[0]), float(b[1]), float(b[2])),
    )


def _largest_polygon(geom) -> ShapelyPolygon | None:
    if geom.is_empty:
        return None
    if isinstance(geom, ShapelyPolygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return None


def classify_gable_extension(
    building: V3Building,
    rooms: list[V3Room],
) -> list[tuple[str, HypothesisTrace]]:
    """Attach ``GableExtension`` to every part. Returns audit entries."""
    audit: list[tuple[str, HypothesisTrace]] = []
    part_roofs = _part_roof_map(building.parts, building.slanted_roofs)
    part_dormers = _part_dormer_counts(building)
    part_arch_flats = _part_arch_flat_counts(building, rooms)

    for part in building.parts:
        result = _classify_single_part(
            part,
            roofs=part_roofs.get(part.id, []),
            n_dormers=part_dormers.get(part.id, 0),
            n_arch_flats=part_arch_flats.get(part.id, 0),
        )
        part.gable_extension = result
        audit.append(
            (
                f"{part.id}::gable",
                HypothesisTrace(
                    stage="gable_extension",
                    rule="gable-geometry+extension-grade",
                    inputs=dict(result.metrics),
                    decision_reason=result.decision_reason,
                ),
            )
        )
    return audit


def _classify_single_part(
    part: V3Part,
    *,
    roofs: list[V3SlantedRoof],
    n_dormers: int,
    n_arch_flats: int,
) -> GableExtension:
    metrics: dict = {"n_slanted_roofs": len(roofs)}

    if len(roofs) != 2:
        return GableExtension(
            status="not_gable",
            metrics=metrics,
            tier1_reasons=(f"n_slanted_roofs={len(roofs)}",),
            tier2_reasons=(),
            ridge_line=None,
            uncovered_region_xz=None,
            decision_reason=(
                f"Gable requires exactly 2 slanted roofs on part (got {len(roofs)})."
            ),
        )

    r0, r1 = roofs
    az0, incl0 = _plane_azimuth_inclination(r0.plane)
    az1, incl1 = _plane_azimuth_inclination(r1.plane)
    daz = (az0 - az1) % 360.0
    daz180 = abs(min(daz, 360.0 - daz) - 180.0)
    dincl = abs(incl0 - incl1)
    metrics.update(
        {
            "az0": round(az0, 2),
            "az1": round(az1, 2),
            "incl0": round(incl0, 2),
            "incl1": round(incl1, 2),
            "daz180": round(daz180, 2),
            "dincl": round(dincl, 2),
        }
    )

    tier1: list[str] = []
    if daz180 > GABLE_AZIMUTH_OPPOSING_TOL_DEG:
        tier1.append(f"daz180={daz180:.1f}")
    if dincl > GABLE_INCLINATION_MATCH_TOL_DEG:
        tier1.append(f"dincl={dincl:.1f}")

    ridge_dir = _plane_intersection_direction(r0.plane, r1.plane)
    ridge_point = _plane_intersection_point(r0.plane, r1.plane)
    if ridge_dir is None or ridge_point is None:
        tier1.append("ridge-degenerate")
    else:
        ridge_y_abs = abs(float(ridge_dir[1]))
        metrics["ridge_y_abs"] = round(ridge_y_abs, 4)
        if ridge_y_abs > GABLE_RIDGE_HORIZONTAL_TOL:
            tier1.append(f"ridge_y={ridge_dir[1]:.3f}")
        ridge_az = degrees(atan2(float(ridge_dir[0]), float(ridge_dir[2]))) % 180.0
        expected = (az0 + 90.0) % 180.0
        ridge_vs_expected = _angle_mod180(ridge_az, expected)
        metrics["ridge_az"] = round(ridge_az, 2)
        metrics["ridge_vs_expected"] = round(ridge_vs_expected, 2)
        if ridge_vs_expected > GABLE_RIDGE_ORIENTATION_TOL_DEG:
            tier1.append(f"ridge_vs_expected={ridge_vs_expected:.1f}")

    if tier1:
        return GableExtension(
            status="not_gable",
            metrics=metrics,
            tier1_reasons=tuple(tier1),
            tier2_reasons=(),
            ridge_line=None,
            uncovered_region_xz=None,
            decision_reason="Tier 1 failed: " + ", ".join(tier1),
        )

    footprint = _polygon_xz(part.footprint_xz)
    if footprint is None or footprint.area <= 0.0:
        return GableExtension(
            status="gable_ambiguous",
            metrics=metrics,
            tier1_reasons=(),
            tier2_reasons=("no-footprint",),
            ridge_line=None,
            uncovered_region_xz=None,
            decision_reason="Tier 1 passed but part footprint is invalid.",
        )

    major_len, minor_len, major_az = _min_rotated_rect_axes(footprint)
    elong = (major_len / minor_len) if minor_len > 1e-6 else float("inf")
    ridge_vs_major = _angle_mod180(float(metrics["ridge_az"]), major_az)
    metrics.update(
        {
            "major_m": round(major_len, 2),
            "minor_m": round(minor_len, 2),
            "major_az": round(major_az, 2),
            "elong": round(elong, 3),
            "ridge_vs_major": round(ridge_vs_major, 2),
        }
    )

    roof_polys = [p for p in (_polygon_xz(r.corners) for r in roofs) if p is not None]
    roof_union = unary_union(roof_polys) if roof_polys else None
    if roof_union is not None and not roof_union.is_empty:
        covered = roof_union.intersection(footprint).area
        coverage = covered / footprint.area
    else:
        coverage = 0.0
    metrics.update(
        {
            "coverage": round(coverage, 3),
            "n_dormers": n_dormers,
            "n_arch_flats": n_arch_flats,
        }
    )

    # Ridge endpoints for the viewer (after footprint is known).
    ridge_line = None
    if ridge_dir is not None and ridge_point is not None:
        ridge_line = _ridge_endpoints(ridge_dir, ridge_point, footprint, major_len)

    # Tier 2 checks; collect fail reasons.
    tier2: list[str] = []
    if ridge_vs_major > GABLE_RIDGE_VS_MAJOR_TOL_DEG:
        tier2.append(f"ridge_vs_major={ridge_vs_major:.1f}")
    if elong < GABLE_MIN_ELONGATION:
        tier2.append(f"elong={elong:.2f}")
    if coverage >= GABLE_EXTENSION_COVERAGE_MAX:
        tier2.append(f"coverage={coverage:.2f}")
    if n_dormers > 0:
        tier2.append(f"dormers={n_dormers}")
    if n_arch_flats > 0:
        tier2.append(f"arch_flats={n_arch_flats}")

    # Sub-case disambiguation: coverage-only → complete; ridge-only → cross-review.
    coverage_only = tier2 == [r for r in tier2 if r.startswith("coverage=")]
    ridge_only = tier2 == [r for r in tier2 if r.startswith("ridge_vs_major=")]

    uncovered_coords = None
    if (
        roof_union is not None
        and not roof_union.is_empty
        and coverage < GABLE_EXTENSION_COVERAGE_MAX
    ):
        uncovered = footprint.difference(roof_union)
        largest = _largest_polygon(uncovered)
        if largest is not None and largest.area > 0.1:
            y_ref = float(ridge_point[1]) if ridge_point is not None else 0.0
            uncovered_coords = [
                (float(x), y_ref, float(z)) for x, z in largest.exterior.coords
            ]

    if not tier2:
        return GableExtension(
            status="gable_along_extend",
            metrics=metrics,
            tier1_reasons=(),
            tier2_reasons=(),
            ridge_line=ridge_line,
            uncovered_region_xz=uncovered_coords,
            decision_reason=(
                "Extension-grade gable — safe to project planes across the full "
                "part footprint."
            ),
        )
    if coverage_only and len(tier2) == 1:
        return GableExtension(
            status="gable_complete",
            metrics=metrics,
            tier1_reasons=(),
            tier2_reasons=tuple(tier2),
            ridge_line=ridge_line,
            uncovered_region_xz=None,
            decision_reason=(
                "Gable-geometry passes and footprint is already covered — nothing to "
                "extend."
            ),
        )
    if ridge_only and len(tier2) == 1:
        return GableExtension(
            status="gable_cross_review",
            metrics=metrics,
            tier1_reasons=(),
            tier2_reasons=tuple(tier2),
            ridge_line=ridge_line,
            uncovered_region_xz=uncovered_coords,
            decision_reason=(
                "Ridge runs across the short axis — flag for review, do not "
                "auto-extend."
            ),
        )
    return GableExtension(
        status="not_gable",
        metrics=metrics,
        tier1_reasons=(),
        tier2_reasons=tuple(tier2),
        ridge_line=ridge_line,
        uncovered_region_xz=uncovered_coords,
        decision_reason="Tier 2 failed: " + ", ".join(tier2),
    )
