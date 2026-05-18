"""Quality gates and building classification for reconciled output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import Building, MergeReport, Room


@dataclass
class BuildingQualityReport:
    uuid: str
    classification: str = "GREEN"  # GREEN / YELLOW / RED
    merged_room_count: int = 0
    scan_room_count: int = 0
    wall_count_merged: int = 0
    wall_count_scan: int = 0
    wall_reduction_pct: float = 0.0
    max_width_delta_cm: float = 0.0
    max_height_delta_cm: float = 0.0
    mean_width_delta_cm: float = 0.0
    mean_height_delta_cm: float = 0.0
    synthesized_count: int = 0
    flagged_count: int = 0
    floor_gap_count: int = 0
    floor_gap_mean_cm: float = 0.0
    floor_gap_median_cm: float = 0.0
    cross_floor_gap_count: int = 0
    cross_floor_gap_high_confidence: int = 0
    scan_session_count: int = 1
    median_wall_displacement_m: float = 0.0
    max_wall_displacement_m: float = 0.0
    p95_wall_displacement_m: float = 0.0
    story_change_count: int = 0
    story_change_ratio: float = 0.0
    walls_with_large_displacement: list[str] = field(default_factory=list)
    merge_reports: list[MergeReport] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def build_quality_report(
    uuid: str,
    building: Building,
    raw_rooms: list[Room],
    merge_reports: list[MergeReport],
    floor_gap_thicknesses: list | None = None,
    cross_floor_gaps: list | None = None,
    scan_session_count: int = 1,
    displacement_report=None,
) -> BuildingQualityReport:
    """Build a quality report for a reconciled building."""
    wall_reports = [r for r in merge_reports if r.surface_type == "wall"]

    # Wall counts
    merged_wall_count = sum(len(r.walls) for r in building.rooms)
    raw_wall_count = sum(len(r.walls) for r in raw_rooms)

    # Dimension deltas
    width_deltas = [
        r.divergences.get("dimensions.width", 0)
        for r in wall_reports
        if r.divergences.get("dimensions.width")
    ]
    height_deltas = [
        r.divergences.get("dimensions.height", 0)
        for r in wall_reports
        if r.divergences.get("dimensions.height")
    ]

    # Floor gap stats
    gap_vals = []
    if floor_gap_thicknesses:
        gap_vals = [t.thickness_cm for t in floor_gap_thicknesses]

    reduction_pct = 0.0
    if raw_wall_count > 0:
        reduction_pct = (1 - merged_wall_count / raw_wall_count) * 100

    # Cross-floor gap stats
    cf_gaps = cross_floor_gaps or []
    cf_high = sum(1 for g in cf_gaps if g.confidence == "high")

    # Displacement stats
    dr = displacement_report
    median_disp = dr.median_displacement_m if dr else 0.0
    max_disp = dr.max_displacement_m if dr else 0.0
    p95_disp = dr.p95_displacement_m if dr else 0.0
    story_changes = dr.story_change_count if dr else 0
    story_ratio = dr.story_change_ratio if dr else 0.0
    large_walls = dr.walls_with_large_displacement if dr else []

    report = BuildingQualityReport(
        uuid=uuid,
        merged_room_count=len(building.rooms),
        scan_room_count=len(raw_rooms),
        wall_count_merged=merged_wall_count,
        wall_count_scan=raw_wall_count,
        wall_reduction_pct=reduction_pct,
        max_width_delta_cm=max(width_deltas) if width_deltas else 0.0,
        max_height_delta_cm=max(height_deltas) if height_deltas else 0.0,
        mean_width_delta_cm=float(np.mean(width_deltas)) if width_deltas else 0.0,
        mean_height_delta_cm=float(np.mean(height_deltas)) if height_deltas else 0.0,
        synthesized_count=sum(
            1 for r in wall_reports if "synthesized_in_merge" in r.flags
        ),
        flagged_count=sum(1 for r in wall_reports if r.flags),
        floor_gap_count=len(gap_vals),
        floor_gap_mean_cm=float(np.mean(gap_vals)) if gap_vals else 0.0,
        floor_gap_median_cm=float(np.median(gap_vals)) if gap_vals else 0.0,
        cross_floor_gap_count=len(cf_gaps),
        cross_floor_gap_high_confidence=cf_high,
        scan_session_count=scan_session_count,
        median_wall_displacement_m=median_disp,
        max_wall_displacement_m=max_disp,
        p95_wall_displacement_m=p95_disp,
        story_change_count=story_changes,
        story_change_ratio=story_ratio,
        walls_with_large_displacement=large_walls,
        merge_reports=merge_reports,
    )

    report.classification = classify_building(report)
    return report


def classify_building(report: BuildingQualityReport) -> str:
    """GREEN / YELLOW / RED classification."""
    red_conditions = []
    yellow_conditions = []

    # RED: severe issues
    if report.wall_reduction_pct > 50:
        red_conditions.append("wall_reduction > 50%")
    if report.max_height_delta_cm > 50:
        red_conditions.append("max_height_delta > 50cm")
    if report.scan_session_count >= 3:
        red_conditions.append(
            f"multi_session ({report.scan_session_count} scan sessions)"
        )
    if report.median_wall_displacement_m > 10:
        red_conditions.append(
            f"median_wall_displacement > 10m ({report.median_wall_displacement_m:.1f}m)"
        )

    # YELLOW: moderate concerns
    if report.wall_reduction_pct > 30:
        yellow_conditions.append("wall_reduction > 30%")
    if report.max_height_delta_cm > 20:
        yellow_conditions.append("max_height_delta > 20cm")
    if report.max_width_delta_cm > 15:
        yellow_conditions.append("max_width_delta > 15cm")
    if report.synthesized_count > 5:
        yellow_conditions.append("synthesized_count > 5")
    if report.cross_floor_gap_high_confidence > 3:
        yellow_conditions.append(
            f"cross_floor_gaps_high > 3 ({report.cross_floor_gap_high_confidence})"
        )
    if report.story_change_ratio > 0.5:
        yellow_conditions.append(
            f"story_change_ratio > 50% ({report.story_change_ratio:.0%})"
        )
    if report.median_wall_displacement_m > 7:
        yellow_conditions.append(
            f"median_wall_displacement > 7m ({report.median_wall_displacement_m:.1f}m)"
        )

    if red_conditions:
        report.flags.extend([f"RED: {c}" for c in red_conditions])
        return "RED"
    elif yellow_conditions:
        report.flags.extend([f"YELLOW: {c}" for c in yellow_conditions])
        return "YELLOW"
    return "GREEN"
