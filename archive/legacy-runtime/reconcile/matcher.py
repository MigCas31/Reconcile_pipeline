"""UUID-based element matching between merged and raw sources."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import Building, MatchResult, Room, Surface


def _build_raw_index(raw_rooms: list[Room]) -> dict[str, tuple[Room, Surface]]:
    """Build UUID -> (room, surface) index from raw room scans."""
    index: dict[str, tuple[Room, Surface]] = {}
    for room in raw_rooms:
        for surface in room.walls + room.doors + room.windows:
            index[surface.identifier] = (room, surface)
    return index


def match_elements(
    merged_building: Building,
    raw_rooms: list[Room],
) -> list[MatchResult]:
    """Match merged surfaces to raw room surfaces by UUID.

    Returns one MatchResult per merged room surface. Surfaces only in merged
    (synthesized during Apple's merge) get raw_surface=None.
    """
    raw_index = _build_raw_index(raw_rooms)

    results = []
    for room in merged_building.rooms:
        for surface_type, surfaces in [
            ("wall", room.walls),
            ("door", room.doors),
            ("window", room.windows),
        ]:
            for surface in surfaces:
                raw_entry = raw_index.get(surface.identifier)
                results.append(
                    MatchResult(
                        merged_surface=surface,
                        raw_surface=raw_entry[1] if raw_entry else None,
                        room_id=room.identifier,
                        surface_type=surface_type,
                    )
                )

    return results


def match_summary(matches: list[MatchResult]) -> dict:
    """Generate summary statistics for match results."""
    total = len(matches)
    matched = sum(1 for m in matches if m.raw_surface is not None)
    unmatched = total - matched

    by_type = {}
    for m in matches:
        t = m.surface_type
        if t not in by_type:
            by_type[t] = {"total": 0, "matched": 0, "unmatched": 0}
        by_type[t]["total"] += 1
        by_type[t]["matched"] += 1 if m.raw_surface else 0
        by_type[t]["unmatched"] += 1 if not m.raw_surface else 0

    return {
        "total": total,
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": matched / total if total > 0 else 0,
        "by_type": by_type,
    }


@dataclass
class DisplacementReport:
    """Wall displacement statistics between raw and merged positions."""

    median_displacement_m: float = 0.0
    max_displacement_m: float = 0.0
    p95_displacement_m: float = 0.0
    story_change_count: int = 0
    story_change_ratio: float = 0.0
    total_matched_walls: int = 0
    walls_with_large_displacement: list[str] = field(default_factory=list)


def compute_wall_displacement(
    matches: list[MatchResult],
    raw_rooms: list[Room],
    large_threshold_m: float = 5.0,
) -> DisplacementReport:
    """Compute 3D position displacement between raw and merged wall positions.

    For each matched wall, computes the Euclidean distance between
    the raw transform translation and the merged transform translation.
    """
    # Build room lookup: surface_id -> room
    surface_to_room: dict[str, Room] = {}
    for room in raw_rooms:
        for s in room.walls + room.doors + room.windows:
            surface_to_room[s.identifier] = room

    wall_matches = [
        m for m in matches if m.surface_type == "wall" and m.raw_surface is not None
    ]

    if not wall_matches:
        return DisplacementReport()

    displacements = []
    story_changes = 0
    large_walls = []

    for m in wall_matches:
        # Get world-space positions
        merged_pos = m.merged_surface.transform.translation.to_array()
        raw_pos = m.raw_surface.transform.translation.to_array()
        dist = float(np.linalg.norm(merged_pos - raw_pos))
        displacements.append(dist)

        if dist > large_threshold_m:
            large_walls.append(m.merged_surface.identifier)

        # Check story change
        if m.merged_surface.story != m.raw_surface.story:
            story_changes += 1

    displacements_arr = np.array(displacements)
    return DisplacementReport(
        median_displacement_m=float(np.median(displacements_arr)),
        max_displacement_m=float(np.max(displacements_arr)),
        p95_displacement_m=float(np.percentile(displacements_arr, 95)),
        story_change_count=story_changes,
        story_change_ratio=story_changes / len(wall_matches),
        total_matched_walls=len(wall_matches),
        walls_with_large_displacement=large_walls,
    )
