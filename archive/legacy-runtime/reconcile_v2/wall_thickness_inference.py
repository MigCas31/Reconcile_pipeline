"""Infer wall thickness proxies from gaps between adjacent room floor edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from reconcile.floor_gaps import find_adjacent_rooms, find_parallel_edges
from reconcile.models import Building, Room
from reconcile.transform import edge_angle_diff, edge_overlap_fraction


@dataclass
class InferredWallThickness:
    room_a_id: str
    room_b_id: str
    story: int
    thickness_cm: float
    p05_cm: float
    p50_cm: float
    p95_cm: float
    std_cm: float
    sample_count: int
    overlap_ratio: float
    angle_delta_deg: float
    opening_proximity_count: int
    confidence: str
    confidence_score: float
    centerline_wkt: str
    edge_a_wkt: str
    edge_b_wkt: str


def _sample_edge_distances(
    edge_a: LineString, edge_b: LineString, n: int = 15
) -> np.ndarray:
    """Sample point-to-segment distances from edge_a to edge_b within overlap."""
    if n < 3:
        n = 3

    t0, t1 = _overlap_interval_normalized(edge_a, edge_b)
    if t1 - t0 < 0.05:
        return np.asarray([], dtype=float)

    pad = min(0.03, (t1 - t0) * 0.2)
    ts = np.linspace(t0 + pad, t1 - pad, n)
    dists = []
    for t in ts:
        pa = edge_a.interpolate(float(t), normalized=True)
        pb = edge_b.interpolate(edge_b.project(Point(pa.x, pa.y)))
        d = float(np.hypot(pa.x - pb.x, pa.y - pb.y))
        dists.append(d)
    return np.asarray(dists, dtype=float)


def _overlap_interval_normalized(
    edge_ref: LineString, edge_other: LineString
) -> tuple[float, float]:
    """Estimate normalized overlap of edge_other on edge_ref."""
    ref_coords = list(edge_ref.coords)
    other_coords = list(edge_other.coords)
    if len(ref_coords) < 2 or len(other_coords) < 2:
        return 0.0, 0.0

    r0 = np.array(ref_coords[0], dtype=float)
    r1 = np.array(ref_coords[1], dtype=float)
    d = r1 - r0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return 0.0, 0.0
    u = d / length

    p0 = float(np.dot(np.array(other_coords[0], dtype=float) - r0, u))
    p1 = float(np.dot(np.array(other_coords[1], dtype=float) - r0, u))
    lo = max(0.0, min(p0, p1))
    hi = min(length, max(p0, p1))
    if hi <= lo:
        return 0.0, 0.0
    return lo / length, hi / length


def _room_opening_points_xz(room: Room) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for s in room.doors + room.windows:
        t = s.transform.translation
        pts.append((float(t.x), float(t.z)))
    return pts


def _opening_proximity_count(
    edge_a: LineString, edge_b: LineString, room_a: Room, room_b: Room
) -> int:
    """Count openings near either candidate boundary edge."""
    pts = _room_opening_points_xz(room_a) + _room_opening_points_xz(room_b)
    count = 0
    for x, z in pts:
        p = Point(x, z)
        if edge_a.distance(p) < 0.35 or edge_b.distance(p) < 0.35:
            count += 1
    return count


def _score_candidate(
    overlap_ratio: float,
    angle_delta_deg: float,
    std_cm: float,
    sample_count: int,
    opening_proximity_count: int,
) -> float:
    overlap_score = min(max(overlap_ratio, 0.0), 1.0)
    angle_score = max(0.0, 1.0 - angle_delta_deg / 20.0)
    variance_score = max(0.0, 1.0 - std_cm / 2.0)
    sample_score = min(sample_count / 12.0, 1.0)
    opening_penalty = min(opening_proximity_count * 0.08, 0.35)

    score = (
        0.35 * overlap_score
        + 0.25 * angle_score
        + 0.20 * variance_score
        + 0.20 * sample_score
        - opening_penalty
    )
    return min(max(score, 0.0), 1.0)


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def infer_wall_thicknesses(
    building: Building,
    *,
    angle_tolerance_deg: float = 15.0,
    max_distance_m: float = 0.4,
    min_overlap: float = 0.3,
) -> list[InferredWallThickness]:
    """Infer per-boundary wall thickness from room-floor edge gaps.

    This is a geometric proxy from 2D room boundary gaps and should not be treated
    as a structural wall assembly measurement.
    """
    out: list[InferredWallThickness] = []
    adjacent = find_adjacent_rooms(building.rooms, max_gap_m=max_distance_m)

    for room_a, room_b, poly_a, poly_b, _approx_dist in adjacent:
        if not room_a.identifier or not room_b.identifier:
            continue

        parallel = find_parallel_edges(
            poly_a,
            poly_b,
            angle_tolerance_deg=angle_tolerance_deg,
            max_distance_m=max_distance_m,
            min_overlap=min_overlap,
        )

        for edge_a, edge_b, _dist in parallel:
            dists_m = _sample_edge_distances(edge_a, edge_b, n=15)
            if dists_m.size == 0:
                continue

            p05_m = float(np.percentile(dists_m, 5))
            p50_m = float(np.percentile(dists_m, 50))
            p95_m = float(np.percentile(dists_m, 95))
            std_m = float(np.std(dists_m))

            angle_delta = float(edge_angle_diff(edge_a, edge_b))
            overlap_ab = float(edge_overlap_fraction(edge_a, edge_b))
            overlap_ba = float(edge_overlap_fraction(edge_b, edge_a))
            overlap = min(overlap_ab, overlap_ba)
            opening_hits = _opening_proximity_count(edge_a, edge_b, room_a, room_b)

            score = _score_candidate(
                overlap_ratio=overlap,
                angle_delta_deg=angle_delta,
                std_cm=std_m * 100.0,
                sample_count=int(dists_m.size),
                opening_proximity_count=opening_hits,
            )

            out.append(
                InferredWallThickness(
                    room_a_id=room_a.identifier,
                    room_b_id=room_b.identifier,
                    story=int(room_a.story),
                    thickness_cm=p50_m * 100.0,
                    p05_cm=p05_m * 100.0,
                    p50_cm=p50_m * 100.0,
                    p95_cm=p95_m * 100.0,
                    std_cm=std_m * 100.0,
                    sample_count=int(dists_m.size),
                    overlap_ratio=overlap,
                    angle_delta_deg=angle_delta,
                    opening_proximity_count=opening_hits,
                    confidence=_confidence_label(score),
                    confidence_score=score,
                    centerline_wkt=LineString(
                        [
                            (
                                (
                                    edge_a.interpolate(t, normalized=True).x
                                    + edge_b.interpolate(
                                        edge_b.project(
                                            edge_a.interpolate(t, normalized=True)
                                        )
                                    ).x
                                )
                                / 2.0,
                                (
                                    edge_a.interpolate(t, normalized=True).y
                                    + edge_b.interpolate(
                                        edge_b.project(
                                            edge_a.interpolate(t, normalized=True)
                                        )
                                    ).y
                                )
                                / 2.0,
                            )
                            for t in np.linspace(0.0, 1.0, 11)
                        ]
                    ).wkt,
                    edge_a_wkt=edge_a.wkt,
                    edge_b_wkt=edge_b.wkt,
                )
            )

    # Keep the best segment per unordered room pair + story.
    by_pair: dict[tuple[str, str, int], InferredWallThickness] = {}
    for rec in out:
        a, b = sorted([rec.room_a_id, rec.room_b_id])
        key = (a, b, rec.story)
        prev = by_pair.get(key)
        if prev is None:
            by_pair[key] = rec
            continue
        # Prefer stronger score; tie-breaker: larger overlap and lower variance.
        if rec.confidence_score > prev.confidence_score or (
            abs(rec.confidence_score - prev.confidence_score) < 1e-9
            and (rec.overlap_ratio > prev.overlap_ratio or rec.std_cm < prev.std_cm)
        ):
            by_pair[key] = rec

    return sorted(
        by_pair.values(),
        key=lambda r: (
            min(r.room_a_id, r.room_b_id),
            max(r.room_a_id, r.room_b_id),
            r.story,
        ),
    )
