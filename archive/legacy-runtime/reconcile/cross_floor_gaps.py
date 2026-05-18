"""Cross-floor gap detection: find missing walls by comparing story footprints.

Strategy: for each story, compute convex hull minus footprint to find concavities.
Then check which concavities are filled by other floors — those are likely scanning
gaps rather than real architectural features. A concavity that other floors fill
means the building geometry should extend there but the scan missed it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .models import Building, Room
from .transform import room_floor_polygon


@dataclass
class CrossFloorGap:
    story: int  # story where the gap exists
    reference_story: int  # best-filling other story (-1 if multiple)
    region_wkt: str  # WKT of the gap polygon (XZ plane)
    area_m2: float
    compactness: float  # 4*pi*area/perimeter^2
    perimeter_contact_pct: float  # % of boundary on story footprint exterior
    confidence: str  # "high" / "medium" / "low"
    confidence_score: float  # 0.0 - 1.0
    affected_room_id: str | None  # room most likely affected
    nearby_raw_walls: int  # count of raw walls near this gap
    centroid_x: float
    centroid_z: float


@dataclass
class StoryFootprint:
    story: int
    footprint_wkt: str
    area_m2: float
    room_count: int


def _polygon_compactness(poly: Polygon) -> float:
    if poly.length < 1e-10:
        return 0.0
    return 4 * math.pi * poly.area / (poly.length**2)


def _decompose(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def build_story_footprints(
    building: Building,
) -> dict[int, tuple[Polygon, StoryFootprint, list[Room]]]:
    rooms_by_story: dict[int, list[tuple[Room, Polygon]]] = defaultdict(list)
    for room in building.rooms:
        poly = room_floor_polygon(room)
        if poly is not None and poly.is_valid and poly.area > 0.01:
            rooms_by_story[room.story].append((room, poly))

    result = {}
    for story, room_polys in sorted(rooms_by_story.items()):
        polys = [p for _, p in room_polys]
        rooms = [r for r, _ in room_polys]
        union = unary_union(polys)
        footprint = union.buffer(0.02).buffer(-0.02)
        footprint = make_valid(footprint)
        if footprint.is_empty or footprint.area < 0.01:
            continue
        if isinstance(footprint, MultiPolygon):
            footprint = max(footprint.geoms, key=lambda g: g.area)
        sf = StoryFootprint(
            story=story,
            footprint_wkt=footprint.wkt,
            area_m2=footprint.area,
            room_count=len(rooms),
        )
        result[story] = (footprint, sf, rooms)
    return result


def _find_affected_room(region: Polygon, rooms: list[Room]) -> str | None:
    best_id, best_dist = None, float("inf")
    for room in rooms:
        poly = room_floor_polygon(room)
        if poly is None:
            continue
        dist = poly.distance(region)
        if dist < best_dist:
            best_dist = dist
            best_id = room.identifier if hasattr(room, "identifier") else None
    return best_id


def _find_concavities(footprint: Polygon, min_area: float = 0.1) -> list[Polygon]:
    """Find concave regions: convex hull minus actual footprint."""
    hull = footprint.convex_hull
    diff = make_valid(hull.difference(footprint))
    regions = _decompose(diff)
    return [r for r in regions if r.area >= min_area]


def _best_filling_story(
    region: Polygon,
    all_footprints: dict[int, Polygon],
    exclude_story: int,
    min_fill: float = 0.3,
) -> tuple[int, float]:
    """Find which other story best fills this concavity. Returns (
        story,
        fill_fraction,
    )."""
    best_story, best_fill = -1, 0.0
    for s, fp in all_footprints.items():
        if s == exclude_story:
            continue
        try:
            inter = region.intersection(fp)
            fill = inter.area / region.area if region.area > 0 else 0
        except Exception:
            fill = 0
        if fill > best_fill:
            best_story, best_fill = s, fill
    return best_story, best_fill


def _score_gap(
    region: Polygon,
    fill_fraction: float,
    story: int,
    story_footprint: Polygon,
    all_footprints: dict[int, Polygon],
) -> float:
    """Score how likely a concavity is a scanning gap (0-1)."""
    area = region.area

    # Fill score: how much other floors fill this concavity (most important signal)
    fill_score = min(fill_fraction / 0.8, 1.0)

    # Area score: smaller concavities more likely to be gaps
    area_score = max(0.1, 1.0 - area / 6.0) if area < 8 else 0.1

    # Compactness: less compact = more likely gap (elongated slivers from scan edge)
    compactness = _polygon_compactness(region)
    compactness_score = 1.0 - min(compactness / 0.4, 1.0)

    # Multi-floor agreement: how many other floors fill >50% of this region
    other = [s for s in all_footprints if s != story]
    if other:
        filling_count = sum(
            1 for s in other if all_footprints[s].intersection(region).area > area * 0.5
        )
        agreement_score = filling_count / len(other)
    else:
        agreement_score = fill_score  # fallback to fill

    return min(
        max(
            0.35 * fill_score
            + 0.20 * area_score
            + 0.15 * compactness_score
            + 0.30 * agreement_score,
            0.0,
        ),
        1.0,
    )


def detect_cross_floor_gaps(
    building: Building,
) -> tuple[list[CrossFloorGap], list[StoryFootprint]]:
    """Detect potential scanning gaps by comparing story footprints.

    Algorithm:
    1. Build per-story footprints
    2. For each story, find concavities (convex hull minus footprint)
    3. Check which concavities are filled by other floors
    4. Score based on fill fraction, area, shape, and multi-floor agreement
    """
    footprint_data = build_story_footprints(building)
    story_footprints_out = [sf for _, sf, _ in footprint_data.values()]

    if len(footprint_data) < 2:
        return [], story_footprints_out

    all_polys = {s: poly for s, (poly, _, _) in footprint_data.items()}
    gaps = []

    for story, (footprint, _sf, rooms) in footprint_data.items():
        concavities = _find_concavities(footprint)

        for region in concavities:
            best_story, fill_frac = _best_filling_story(region, all_polys, story)

            # Only report concavities that another floor fills significantly
            if fill_frac < 0.3:
                continue

            score = _score_gap(region, fill_frac, story, footprint, all_polys)

            if score >= 0.6:
                confidence = "high"
            elif score >= 0.35:
                confidence = "medium"
            else:
                confidence = "low"

            centroid = region.centroid
            compactness = _polygon_compactness(region)

            try:
                shared = region.boundary.intersection(footprint.exterior)
                contact = (
                    shared.length / region.boundary.length if not shared.is_empty else 0
                )
            except Exception:
                contact = 0

            affected = _find_affected_room(region, rooms)

            gaps.append(
                CrossFloorGap(
                    story=story,
                    reference_story=best_story,
                    region_wkt=region.wkt,
                    area_m2=region.area,
                    compactness=compactness,
                    perimeter_contact_pct=contact,
                    confidence=confidence,
                    confidence_score=score,
                    affected_room_id=affected,
                    nearby_raw_walls=0,
                    centroid_x=centroid.x,
                    centroid_z=centroid.y,
                )
            )

    return gaps, story_footprints_out
