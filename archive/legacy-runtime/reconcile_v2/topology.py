"""Topology extraction utilities for V2 graph generation."""

from __future__ import annotations

from dataclasses import dataclass

from reconcile.cross_floor_gaps import detect_cross_floor_gaps
from reconcile.models import Building

from .wall_thickness_inference import infer_wall_thicknesses


@dataclass
class IntraStoryAdjacency:
    room_a: str
    room_b: str
    story: int
    thickness_cm: float
    thickness_p05_cm: float
    thickness_p95_cm: float
    thickness_std_cm: float
    overlap_ratio: float
    angle_delta_deg: float
    opening_proximity_count: int
    centerline_wkt: str
    confidence_score: float
    confidence: str
    support_ratio: float
    relation_state: str
    floor_delta_m: float


@dataclass
class GapRecord:
    kind: str  # intra_story | cross_story
    story: int
    reference_story: int | None
    affected_room: str | None
    confidence: str
    confidence_score: float
    area_m2: float | None
    thickness_cm: float | None
    region_wkt: str | None
    support_ratio: float
    relation_state: str
    metrics: dict


def _story_floor_offsets(building: Building) -> dict[str, float]:
    room_floor_y: dict[str, float] = {}
    floor_values_by_story: dict[int, list[tuple[str, float]]] = {}
    for room in building.rooms:
        if not room.identifier or not room.floors:
            continue
        floor = room.floors[0]
        if not floor.polygon_corners:
            continue
        world = floor.transform.apply_corners(floor.polygon_corners)
        ys = [float(c.y) for c in world]
        if not ys:
            continue
        floor_y = sum(ys) / len(ys)
        room_floor_y[room.identifier] = floor_y
        floor_values_by_story.setdefault(int(room.story), []).append(
            (
                room.identifier,
                floor_y,
            )
        )

    offsets: dict[str, float] = {}
    for _story, entries in floor_values_by_story.items():
        values = sorted(v for _room_id, v in entries)
        if not values:
            continue
        mid = len(values) // 2
        median = (
            values[mid]
            if len(values) % 2 == 1
            else (values[mid - 1] + values[mid]) * 0.5
        )
        for room_id, floor_y in entries:
            offsets[room_id] = abs(floor_y - median)
    return offsets


def _adjacency_relation_state(
    *,
    overlap_ratio: float,
    confidence_score: float,
    floor_delta_m: float,
) -> tuple[float, str]:
    vertical_factor = (
        1.0 if floor_delta_m <= 0.15 else (0.25 if floor_delta_m <= 0.50 else 0.05)
    )
    support_ratio = min(max(overlap_ratio, 0.0), 1.0) * vertical_factor
    if support_ratio >= 0.35 and confidence_score >= 0.45:
        return support_ratio, "confirmed"
    if support_ratio >= 0.05:
        return support_ratio, "partial"
    return support_ratio, "weak"


def _gap_relation_state(
    confidence_score: float, compactness: float, perimeter_contact_pct: float
) -> tuple[float, str]:
    shape_penalty = 0.15 if compactness > 0.35 else (0.05 if compactness > 0.2 else 0.0)
    edge_bonus = min(max(perimeter_contact_pct, 0.0), 1.0) * 0.2
    support_ratio = min(max(confidence_score - shape_penalty + edge_bonus, 0.0), 1.0)
    if support_ratio >= 0.65:
        return support_ratio, "confirmed"
    if support_ratio >= 0.35:
        return support_ratio, "partial"
    return support_ratio, "weak"


def infer_intra_story_adjacency(building: Building) -> list[IntraStoryAdjacency]:
    """Infer adjacency from robust floor-gap wall thickness evidence."""
    thicknesses = infer_wall_thicknesses(building)
    floor_offsets = _story_floor_offsets(building)
    out: list[IntraStoryAdjacency] = []
    seen: set[tuple[str, str]] = set()

    for w in thicknesses:
        a, b = sorted([w.room_a_id, w.room_b_id])
        if (a, b) in seen:
            continue
        seen.add((a, b))
        floor_delta = max(floor_offsets.get(a, 0.0), floor_offsets.get(b, 0.0))
        support_ratio, relation_state = _adjacency_relation_state(
            overlap_ratio=float(w.overlap_ratio),
            confidence_score=float(w.confidence_score),
            floor_delta_m=float(floor_delta),
        )
        out.append(
            IntraStoryAdjacency(
                room_a=a,
                room_b=b,
                story=w.story,
                thickness_cm=float(w.thickness_cm),
                thickness_p05_cm=float(w.p05_cm),
                thickness_p95_cm=float(w.p95_cm),
                thickness_std_cm=float(w.std_cm),
                overlap_ratio=float(w.overlap_ratio),
                angle_delta_deg=float(w.angle_delta_deg),
                opening_proximity_count=int(w.opening_proximity_count),
                centerline_wkt=w.centerline_wkt,
                confidence_score=float(w.confidence_score),
                confidence=w.confidence,
                support_ratio=float(support_ratio),
                relation_state=relation_state,
                floor_delta_m=float(floor_delta),
            )
        )

    out.sort(key=lambda x: (x.story, x.room_a, x.room_b))
    return out


def infer_gap_records(building: Building) -> tuple[list[GapRecord], list[dict]]:
    """Create gap records from existing geometric gap detectors."""
    thicknesses = infer_wall_thicknesses(building)
    cross_floor_gaps, story_footprints = detect_cross_floor_gaps(building)
    floor_offsets = _story_floor_offsets(building)

    gaps: list[GapRecord] = []
    for i, w in enumerate(thicknesses):
        floor_delta = max(
            floor_offsets.get(w.room_a_id, 0.0),
            floor_offsets.get(w.room_b_id, 0.0),
        )
        support_ratio, relation_state = _adjacency_relation_state(
            overlap_ratio=float(w.overlap_ratio),
            confidence_score=float(w.confidence_score),
            floor_delta_m=float(floor_delta),
        )
        gaps.append(
            GapRecord(
                kind="intra_story",
                story=int(w.story),
                reference_story=None,
                affected_room=w.room_a_id,
                confidence=w.confidence,
                confidence_score=float(w.confidence_score),
                area_m2=None,
                thickness_cm=float(w.thickness_cm),
                region_wkt=w.centerline_wkt,
                support_ratio=float(support_ratio),
                relation_state=relation_state,
                metrics={
                    "index": i,
                    "room_a": w.room_a_id,
                    "room_b": w.room_b_id,
                    "p05_cm": w.p05_cm,
                    "p95_cm": w.p95_cm,
                    "std_cm": w.std_cm,
                    "overlap_ratio": w.overlap_ratio,
                    "support_ratio": round(support_ratio, 6),
                    "relation_state": relation_state,
                    "floor_delta_m": round(floor_delta, 6),
                    "angle_delta_deg": w.angle_delta_deg,
                    "opening_proximity_count": w.opening_proximity_count,
                    "edge_a_wkt": w.edge_a_wkt,
                    "edge_b_wkt": w.edge_b_wkt,
                },
            )
        )

    for g in cross_floor_gaps:
        support_ratio, relation_state = _gap_relation_state(
            confidence_score=float(g.confidence_score),
            compactness=float(g.compactness),
            perimeter_contact_pct=float(g.perimeter_contact_pct),
        )
        gaps.append(
            GapRecord(
                kind="cross_story",
                story=int(g.story),
                reference_story=int(g.reference_story),
                affected_room=g.affected_room_id,
                confidence=g.confidence,
                confidence_score=float(g.confidence_score),
                area_m2=float(g.area_m2),
                thickness_cm=None,
                region_wkt=g.region_wkt,
                support_ratio=float(support_ratio),
                relation_state=relation_state,
                metrics={
                    "compactness": g.compactness,
                    "perimeter_contact_pct": g.perimeter_contact_pct,
                    "support_ratio": round(support_ratio, 6),
                    "relation_state": relation_state,
                    "centroid_x": g.centroid_x,
                    "centroid_z": g.centroid_z,
                },
            )
        )

    fps = [
        {
            "story": sf.story,
            "area_m2": sf.area_m2,
            "room_count": sf.room_count,
            "footprint_wkt": sf.footprint_wkt,
        }
        for sf in story_footprints
    ]
    return gaps, fps
