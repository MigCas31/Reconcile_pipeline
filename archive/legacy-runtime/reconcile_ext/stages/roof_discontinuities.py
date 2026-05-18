"""R1.2 — flag pairs of adjacent slanted roofs with inconsistent geometry.

Two roof surfaces sit on the same part of a building if their ridge heights
match (within tolerance), their eaves are aligned, and their azimuths /
inclinations are compatible. When *adjacent* roofs disagree on these — e.g.
one is 2m lower than the other at the ridge, or one faces a very different
direction — that disagreement is the physical signature of an extension.

Adjacency here is "XZ-close" — within ``ROOF_ADJACENCY_DISTANCE_M``. This
avoids needing a full roof-connection graph for the first cut; good enough
to surface the strong signals.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from ..constants import (
    AZIMUTH_MISMATCH_STRONG_DEG,
    EAVE_DELTA_STRONG_M,
    INCLINATION_MISMATCH_STRONG_DEG,
    RIDGE_DELTA_STRONG_M,
    RIDGE_DELTA_WEAK_M,
    ROOF_ADJACENCY_DISTANCE_M,
)
from ..models import ExtDiscontinuity, ExtSnapshot, SnapshotRoof


def _shapely_polygon(corners) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = Polygon([(c[0], c[2]) for c in corners])
    except Exception:
        return None
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _mid_xz(poly_a: Polygon, poly_b: Polygon) -> tuple[float, float]:
    """Midpoint of the closest points between the two roof footprints."""
    # Shapely's .representative_point is cheaper than centroid for our needs.
    pa = poly_a.representative_point()
    pb = poly_b.representative_point()
    return (float((pa.x + pb.x) / 2), float((pa.y + pb.y) / 2))


def _azimuth_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def detect_roof_discontinuities(snapshot: ExtSnapshot) -> list[ExtDiscontinuity]:
    roofs: list[tuple[SnapshotRoof, Polygon]] = []
    for r in snapshot.slanted_roofs:
        poly = _shapely_polygon(r.corners)
        if poly is None:
            continue
        roofs.append((r, poly))

    results: list[ExtDiscontinuity] = []
    for i in range(len(roofs)):
        roof_a, poly_a = roofs[i]
        for j in range(i + 1, len(roofs)):
            roof_b, poly_b = roofs[j]

            # Adjacency — XZ distance between polygons.
            dist = poly_a.distance(poly_b)
            if dist > ROOF_ADJACENCY_DISTANCE_M:
                continue

            mid = _mid_xz(poly_a, poly_b)

            # Ridge-height delta.
            ridge_delta = roof_a.ridge_y - roof_b.ridge_y
            abs_ridge_delta = abs(ridge_delta)
            if abs_ridge_delta >= RIDGE_DELTA_WEAK_M:
                strength = (
                    "strong" if abs_ridge_delta >= RIDGE_DELTA_STRONG_M else "weak"
                )
                results.append(
                    ExtDiscontinuity(
                        id=f"ridge-delta::{roof_a.id}::{roof_b.id}",
                        kind="ridge-delta",
                        roof_a_id=roof_a.id,
                        roof_b_id=roof_b.id,
                        delta=float(ridge_delta),
                        mid_xz=mid,
                        strength=strength,
                        rationale=(
                            f"Ridge heights differ by {abs_ridge_delta:.2f}m "
                            f"({roof_a.ridge_y:.2f} vs {roof_b.ridge_y:.2f})"
                        ),
                    )
                )

            # Eave-height delta — only if strong; weak is too noisy on its own.
            eave_delta = roof_a.eave_y - roof_b.eave_y
            abs_eave_delta = abs(eave_delta)
            if abs_eave_delta >= EAVE_DELTA_STRONG_M:
                results.append(
                    ExtDiscontinuity(
                        id=f"eave-delta::{roof_a.id}::{roof_b.id}",
                        kind="eave-delta",
                        roof_a_id=roof_a.id,
                        roof_b_id=roof_b.id,
                        delta=float(eave_delta),
                        mid_xz=mid,
                        strength="strong",
                        rationale=(
                            f"Eave heights differ by {abs_eave_delta:.2f}m "
                            f"({roof_a.eave_y:.2f} vs {roof_b.eave_y:.2f})"
                        ),
                    )
                )

            # Azimuth mismatch — only strong (weak mismatches are just a hip or gable
            # cap).
            azimuth_delta = _azimuth_diff_deg(roof_a.azimuth_deg, roof_b.azimuth_deg)
            if azimuth_delta >= AZIMUTH_MISMATCH_STRONG_DEG and azimuth_delta <= 135.0:
                # Skip ~180° (opposing) which is the classic gable pair.
                results.append(
                    ExtDiscontinuity(
                        id=f"azimuth-mismatch::{roof_a.id}::{roof_b.id}",
                        kind="azimuth-mismatch",
                        roof_a_id=roof_a.id,
                        roof_b_id=roof_b.id,
                        delta=float(azimuth_delta),
                        mid_xz=mid,
                        strength="strong",
                        rationale=(
                            f"Azimuths differ by {azimuth_delta:.1f}° "
                            f"({roof_a.azimuth_deg:.1f}° vs {roof_b.azimuth_deg:.1f}°)"
                        ),
                    )
                )

            # Inclination mismatch.
            incl_delta = abs(roof_a.inclination_deg - roof_b.inclination_deg)
            if incl_delta >= INCLINATION_MISMATCH_STRONG_DEG:
                results.append(
                    ExtDiscontinuity(
                        id=f"inclination-mismatch::{roof_a.id}::{roof_b.id}",
                        kind="inclination-mismatch",
                        roof_a_id=roof_a.id,
                        roof_b_id=roof_b.id,
                        delta=float(incl_delta),
                        mid_xz=mid,
                        strength="strong",
                        rationale=(
                            f"Inclinations differ by {incl_delta:.1f}° "
                            f"({roof_a.inclination_deg:.1f}° vs "
                            f"{roof_b.inclination_deg:.1f}°)"
                        ),
                    )
                )

    return results
