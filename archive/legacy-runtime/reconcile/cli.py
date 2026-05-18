"""CLI entry point for the reconciliation pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .cross_floor_gaps import detect_cross_floor_gaps
from .floor_gaps import compute_wall_thicknesses
from .loader import (
    count_scan_sessions,
    load_merged,
    load_scan_cache,
)
from .matcher import compute_wall_displacement, match_elements, match_summary
from .output import write_reconciled
from .story_fix import fix_building_stories
from .trust_merge import apply_trust_table
from .validation import build_quality_report


def reconcile_building(
    merged_path: Path,
    scan_dir: Path,
    output_path: Path,
    uuid: str = "",
) -> dict:
    """Run full reconciliation pipeline on one building. Returns summary dict."""
    # 1. Load
    building = load_merged(merged_path)
    raw_rooms = load_scan_cache(scan_dir)

    # 1b. Scan session detection
    scan_sessions = count_scan_sessions(raw_rooms)

    # 2. Match
    matches = match_elements(building, raw_rooms)
    summary = match_summary(matches)

    # 2b. Wall displacement analysis
    displacement = compute_wall_displacement(matches, raw_rooms)

    # 3. Trust merge — apply to each matched surface
    all_reports = []
    reconciled_by_id = {}
    for match in matches:
        surface, report = apply_trust_table(match)
        all_reports.append(report)
        reconciled_by_id[surface.identifier] = surface

    # 4. Update building rooms with reconciled surfaces + fix stories
    reconciled_building = deepcopy(building)
    fix_building_stories(reconciled_building)
    for room in reconciled_building.rooms:
        room.walls = [reconciled_by_id.get(w.identifier, w) for w in room.walls]
        room.doors = [reconciled_by_id.get(d.identifier, d) for d in room.doors]
        room.windows = [reconciled_by_id.get(w.identifier, w) for w in room.windows]

    # Also update top-level surfaces
    reconciled_building.top_level_walls = [
        reconciled_by_id.get(w.identifier, w)
        for w in reconciled_building.top_level_walls
    ]
    reconciled_building.top_level_doors = [
        reconciled_by_id.get(d.identifier, d)
        for d in reconciled_building.top_level_doors
    ]
    reconciled_building.top_level_windows = [
        reconciled_by_id.get(w.identifier, w)
        for w in reconciled_building.top_level_windows
    ]

    # 4b. Cross-floor gap detection
    cross_floor_gaps, story_footprints = detect_cross_floor_gaps(reconciled_building)

    # 5. Floor polygon gap analysis
    thicknesses = compute_wall_thicknesses(reconciled_building)

    # 6. Quality report
    quality = build_quality_report(
        uuid=uuid,
        building=reconciled_building,
        raw_rooms=raw_rooms,
        merge_reports=all_reports,
        floor_gap_thicknesses=thicknesses,
        cross_floor_gaps=cross_floor_gaps,
        scan_session_count=scan_sessions,
        displacement_report=displacement,
    )

    # 7. Output
    write_reconciled(
        reconciled_building,
        thicknesses,
        quality,
        output_path,
        cross_floor_gaps=cross_floor_gaps,
        story_footprints=story_footprints,
    )

    return {
        "uuid": uuid,
        "classification": quality.classification,
        "match_rate": summary["match_rate"],
        "wall_count_merged": quality.wall_count_merged,
        "wall_count_scan": quality.wall_count_scan,
        "wall_reduction_pct": quality.wall_reduction_pct,
        "mean_width_delta_cm": quality.mean_width_delta_cm,
        "mean_height_delta_cm": quality.mean_height_delta_cm,
        "floor_gap_count": quality.floor_gap_count,
        "floor_gap_median_cm": quality.floor_gap_median_cm,
        "flagged_count": quality.flagged_count,
        "cross_floor_gap_count": quality.cross_floor_gap_count,
        "cross_floor_gap_high": quality.cross_floor_gap_high_confidence,
        "scan_session_count": quality.scan_session_count,
        "median_wall_displacement_m": quality.median_wall_displacement_m,
        "story_change_count": quality.story_change_count,
        "flags": quality.flags,
    }


def main():
    parser = argparse.ArgumentParser(description="Reconcile RoomPlan building models")
    parser.add_argument("--building", required=True, help="Path to merged.json")
    parser.add_argument(
        "--scan-cache", required=True, help="Path to scan-cache directory"
    )
    parser.add_argument("--output", required=True, help="Output reconciled.json path")
    parser.add_argument("--uuid", default="", help="Building UUID for reporting")

    args = parser.parse_args()
    result = reconcile_building(
        merged_path=Path(args.building),
        scan_dir=Path(args.scan_cache),
        output_path=Path(args.output),
        uuid=args.uuid,
    )

    print(f"[{result['classification']}] {result['uuid']}")
    print(f"  Match rate: {result['match_rate']:.1%}")
    print(
        f"  Walls: {result['wall_count_scan']} per-scan \u2192 "
        f"{result['wall_count_merged']} after merge"
    )
    print(
        f"  Dimension deltas: width={result['mean_width_delta_cm']:.1f}cm, "
        f"height={result['mean_height_delta_cm']:.1f}cm"
    )
    print(
        f"  Floor gaps: {result['floor_gap_count']} measurements, "
        f"median={result['floor_gap_median_cm']:.1f}cm"
    )
    print(
        f"  Scan sessions: {result['scan_session_count']}, wall displacement: "
        f"{result['median_wall_displacement_m']:.1f}m median"
    )
    if result["flags"]:
        for flag in result["flags"]:
            print(f"  ! {flag}")


if __name__ == "__main__":
    main()
