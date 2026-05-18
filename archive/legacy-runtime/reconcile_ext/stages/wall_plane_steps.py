"""R1.3 — find exterior-wall plane steps that cross a part seam.

Two exterior walls built for the same building usually either share a plane
(main-house walls on a single long facade) or are perpendicular. When two
walls have the *same* azimuth but sit on different offsets — separated by
more than a wall thickness — that's a plan-level "step" typical of an
extension's outer wall stepping out (or back) from the main house's wall.
"""

from __future__ import annotations

import math

from ..constants import (
    WALL_AZIMUTH_BUCKET_DEG,
    WALL_PROJECTION_OVERLAP_MIN_M,
    WALL_STEP_MAX_M,
    WALL_STEP_MIN_M,
)
from ..models import ExtSnapshot, ExtWallStep, SnapshotRoom, SnapshotWall


def _azimuth_bucket(deg: float) -> int:
    # Fold 180° — a wall facing +Z and a wall facing -Z share a plane class.
    folded = deg % 180.0
    return round(folded / WALL_AZIMUTH_BUCKET_DEG)


def _project_along(
    along_x: float, along_z: float, wall: SnapshotWall
) -> tuple[float, float]:
    """Return [min, max] of the wall's corners projected onto the along-axis."""
    projs = [c[0] * along_x + c[2] * along_z for c in wall.corners]
    return min(projs), max(projs)


def _along_dir(azimuth_deg: float) -> tuple[float, float]:
    """Unit vector along the wall direction (perpendicular to its normal)."""
    # Normal azimuth_deg is measured from +Z clockwise (via atan2(nx, nz)).
    # Wall runs perpendicular to normal.
    ar = math.radians(azimuth_deg)
    # Normal = (sin(ar), cos(ar))  (since azimuth = atan2(nx, nz))
    # Along = rotate 90° CCW in XZ = (cos(ar), -sin(ar))
    return math.cos(ar), -math.sin(ar)


def detect_wall_plane_steps(snapshot: ExtSnapshot) -> list[ExtWallStep]:
    # Build a flat list of (room, wall) for pairing.
    indexed: list[tuple[SnapshotRoom, SnapshotWall]] = []
    for room in snapshot.rooms:
        for wall in room.walls:
            if wall.length_m < 0.5:
                continue
            indexed.append((room, wall))

    # Bucket by azimuth (with 180° folding).
    buckets: dict[int, list[tuple[SnapshotRoom, SnapshotWall]]] = {}
    for item in indexed:
        azimuth = item[1].azimuth_deg
        b = _azimuth_bucket(azimuth)
        buckets.setdefault(b, []).append(item)
        # Also consider adjacent buckets — a wall at 87° and 93° are both
        # "east-facing" for our purposes.
        buckets.setdefault((b - 1) % int(180 / WALL_AZIMUTH_BUCKET_DEG), []).append(
            item
        )
        buckets.setdefault((b + 1) % int(180 / WALL_AZIMUTH_BUCKET_DEG), []).append(
            item
        )

    seen_pairs: set[frozenset[str]] = set()
    results: list[ExtWallStep] = []

    for wall_list in buckets.values():
        for i in range(len(wall_list)):
            room_a, wall_a = wall_list[i]
            for j in range(i + 1, len(wall_list)):
                room_b, wall_b = wall_list[j]
                if wall_a.id == wall_b.id:
                    continue
                pair_key = frozenset({wall_a.id, wall_b.id})
                if pair_key in seen_pairs:
                    continue

                # Skip walls in the same room; extension seams are between rooms.
                if room_a.id == room_b.id:
                    continue

                # Must be near-parallel.
                az_diff = min(
                    (wall_a.azimuth_deg - wall_b.azimuth_deg) % 180.0,
                    (wall_b.azimuth_deg - wall_a.azimuth_deg) % 180.0,
                )
                if az_diff > WALL_AZIMUTH_BUCKET_DEG * 2:
                    continue

                # Offset along the shared normal.
                # Fold 180° for the offset comparison — two walls facing opposite
                # ways can still be coplanar (same infinite plane).
                offset_a = wall_a.plane_offset_m
                offset_b = wall_b.plane_offset_m
                offset_diff = abs(offset_a - offset_b)
                if offset_diff < WALL_STEP_MIN_M or offset_diff > WALL_STEP_MAX_M:
                    continue

                # Overlap check along the wall direction.
                along_x, along_z = _along_dir(wall_a.azimuth_deg)
                a_min, a_max = _project_along(along_x, along_z, wall_a)
                b_min, b_max = _project_along(along_x, along_z, wall_b)
                overlap = min(a_max, b_max) - max(a_min, b_min)
                if overlap < WALL_PROJECTION_OVERLAP_MIN_M:
                    continue

                seen_pairs.add(pair_key)

                # Anchor midpoint: average of the walls' XZ centroids.
                ac = (
                    sum(c[0] for c in wall_a.corners) / len(wall_a.corners),
                    sum(c[2] for c in wall_a.corners) / len(wall_a.corners),
                )
                bc = (
                    sum(c[0] for c in wall_b.corners) / len(wall_b.corners),
                    sum(c[2] for c in wall_b.corners) / len(wall_b.corners),
                )
                anchor = ((ac[0] + bc[0]) / 2, (ac[1] + bc[1]) / 2)

                results.append(
                    ExtWallStep(
                        id=f"wall-step::{wall_a.id}::{wall_b.id}",
                        kind="parallel-step",
                        wall_a_id=wall_a.id,
                        wall_b_id=wall_b.id,
                        room_a_id=room_a.id,
                        room_b_id=room_b.id,
                        azimuth_deg=float(wall_a.azimuth_deg),
                        offset_m=float(offset_diff),
                        overlap_m=float(overlap),
                        anchor_xz=anchor,
                        rationale=(
                            f"Parallel exterior walls (Δaz {az_diff:.1f}°) "
                            f"with {offset_diff:.2f}m perpendicular offset, "
                            f"{overlap:.2f}m projection overlap"
                        ),
                    )
                )

    return results
