"""Orchestrator for the extension-detection pipeline.

Stages run in order; each takes the snapshot and returns its diagnostics.
Seam hypothesis synthesis (combining signals near the same XZ location)
happens at the end as a light aggregator — not a hard decision.
"""

from __future__ import annotations

import math

from .constants import (
    MIN_COMPOSITE_SCORE,
    TIER1_EVIDENCE_WEIGHT,
    TIER2_EVIDENCE_WEIGHT,
)
from .models import (
    ExtReport,
    ExtSeamHypothesis,
    ExtSnapshot,
)
from .stages.ceiling_floor_deltas import detect_height_deltas
from .stages.reflex_vertices import detect_reflex_vertices
from .stages.roof_discontinuities import detect_roof_discontinuities
from .stages.units import detect_units
from .stages.wall_plane_steps import detect_wall_plane_steps


def _synthesize_seams(report: ExtReport) -> list[ExtSeamHypothesis]:
    """Cluster nearby diagnostic signals into composite seam hypotheses.

    Reflex vertex = the strongest location signal (tier 1, ``w=1.0``). Every
    other signal (roof discontinuity, wall step, height delta) within
    ``RADIUS_M`` of a reflex vertex corroborates it (tier 2, ``w=0.5``).
    Composite score = sum of weights. Confidence tier from the score.

    Reflex vertices with no corroboration still emit a "weak" seam — useful
    as a viewer toggle to see every reflex in the building.
    """
    RADIUS_M = 4.0

    def near(p: tuple[float, float], q: tuple[float, float]) -> bool:
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= RADIUS_M

    hypotheses: list[ExtSeamHypothesis] = []
    for rv in report.reflex_vertices:
        anchor = (rv.x, rv.z)
        supporting: list[str] = [rv.id]
        score = TIER1_EVIDENCE_WEIGHT
        for d in report.discontinuities:
            if near(anchor, d.mid_xz):
                supporting.append(d.id)
                score += TIER2_EVIDENCE_WEIGHT * (
                    1.0 if d.strength == "strong" else 0.5
                )
        for w in report.wall_steps:
            if near(anchor, w.anchor_xz):
                supporting.append(w.id)
                score += TIER2_EVIDENCE_WEIGHT
        for h in report.height_deltas:
            if near(anchor, h.anchor_xz):
                supporting.append(h.id)
                score += TIER2_EVIDENCE_WEIGHT

        if score >= MIN_COMPOSITE_SCORE * 2:
            confidence = "strong"
        elif score >= MIN_COMPOSITE_SCORE:
            confidence = "medium"
        else:
            confidence = "weak"

        hypotheses.append(
            ExtSeamHypothesis(
                id=f"seam::{rv.id}",
                anchor_xz=anchor,
                supporting_signal_ids=tuple(supporting),
                composite_score=float(score),
                confidence=confidence,
                rationale=(
                    f"Reflex vertex (notch {rv.notch_depth_m:.2f}m) + "
                    f"{len(supporting) - 1} nearby corroborating signal(s)"
                ),
            )
        )
    return hypotheses


def run(snapshot: ExtSnapshot) -> ExtReport:
    report = ExtReport(
        building_uuid=snapshot.building_uuid,
        address=snapshot.address,
    )
    report.reflex_vertices = detect_reflex_vertices(snapshot)
    report.discontinuities = detect_roof_discontinuities(snapshot)
    report.wall_steps = detect_wall_plane_steps(snapshot)
    report.height_deltas = detect_height_deltas(snapshot)
    report.seam_hypotheses = _synthesize_seams(report)

    # Rendering helpers — footprints snapshotted so the viewer doesn't need
    # to cross-reference the source JSONs to draw outlines.
    report.parts_footprints_xz = [p.footprint_xz for p in snapshot.parts]

    # Merged footprint (outer outline) — reuse reflex_vertices' merger.
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = []
    for p in snapshot.parts:
        if len(p.footprint_xz) < 3:
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in p.footprint_xz])
        except Exception:
            continue
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)
    for gf in snapshot.gap_fill_footprints:
        if len(gf) < 3:
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in gf])
        except Exception:
            continue
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)
    if polys:
        merged = unary_union(polys)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        from .constants import FOOTPRINT_SIMPLIFY_TOL_M

        simplified = merged.simplify(FOOTPRINT_SIMPLIFY_TOL_M, preserve_topology=True)
        if simplified.is_valid and not simplified.is_empty:
            if simplified.geom_type == "MultiPolygon":
                simplified = max(simplified.geoms, key=lambda g: g.area)
            merged = simplified
        y = (
            snapshot.parts[0].footprint_xz[0][1]
            if snapshot.parts[0].footprint_xz
            else 0.0
        )
        coords = list(merged.exterior.coords)
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        report.building_footprint_xz = [
            (float(x), float(y), float(z)) for x, z in coords
        ]

    # Unit partitioning (R2 prototype) — cut the merged footprint at each
    # strong/medium seam and assign rooms to the resulting sub-polygons.
    units, cut_lines = detect_units(snapshot, report)
    report.units = units
    report.seam_cut_lines = cut_lines

    report.stats = {
        "reflex_count": len(report.reflex_vertices),
        "discontinuity_count": len(report.discontinuities),
        "wall_step_count": len(report.wall_steps),
        "height_delta_count": len(report.height_deltas),
        "seam_strong": sum(
            1 for s in report.seam_hypotheses if s.confidence == "strong"
        ),
        "seam_medium": sum(
            1 for s in report.seam_hypotheses if s.confidence == "medium"
        ),
        "seam_weak": sum(1 for s in report.seam_hypotheses if s.confidence == "weak"),
        "unit_count": len(report.units),
        "extension_count": sum(1 for u in report.units if u.role == "extension"),
    }
    return report
