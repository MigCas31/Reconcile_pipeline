"""Complexity-tier classification for the tier viewer.

Each building is placed into exactly one of eight tiers ordered by increasing
geometric complexity. The predicate list is evaluated top-to-bottom and the
first match wins.

    1. one storey, flat ceilings
    2. two stacked storeys, flat ceilings
    3. three stacked storeys, flat ceilings
    4. multiple storeys with half-heights (split-level)
    5. one storey with at least one slanted room
    6. classic gable (two opposite ridge/eave planes), no flat roof
    7. mixed: gable + flat roof surfaces
    8. rest

Pure functions only; the HTTP handler wraps these.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

TIER_LABELS: dict[int, str] = {
    1: "1 storey · flat",
    2: "2 storeys · flat",
    3: "3 storeys · flat",
    4: "Multi-storey · half-height",
    5: "1 storey · slanted room",
    6: "Gable (ridge + eave)",
    7: "Mixed: gable + flat roof",
    8: "Other",
}

# Gable detection -- two oblique surfaces opposite in azimuth, similar pitch.
_GABLE_AZIMUTH_TOL_DEG = 30.0
_GABLE_INCL_TOL_DEG = 10.0
_GABLE_AREA_FRACTION = 0.70

# Surfaces below this inclination are near-flat ceilings misclassified as
# oblique slopes (e.g. RoomPlan-noise tilts of 5-10 deg). They pollute gable
# detection on tier-6 buildings -- drop them before the pair search.
_MIN_GABLE_INCL_DEG = 10.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Shortest absolute difference between two azimuths, in degrees."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def _polygon_area_3d(corners: Iterable[Iterable[float]]) -> float:
    """Area of a planar polygon given as 3D corners. Uses Newell's method."""
    pts = [tuple(float(c) for c in p) for p in corners]
    if len(pts) < 3:
        return 0.0
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _oblique_meta(surface: dict[str, Any]) -> tuple[float | None, float | None, float]:
    """Return (azimuth_deg, inclination_deg, area) for an oblique surface."""
    cluster = surface.get("cluster") or {}
    az = cluster.get("avgAzimuth")
    incl = cluster.get("avgIncl")
    try:
        az = float(az) if az is not None else None
    except (TypeError, ValueError):
        az = None
    try:
        incl = float(incl) if incl is not None else None
    except (TypeError, ValueError):
        incl = None
    area = _polygon_area_3d(surface.get("corners") or [])
    return az, incl, area


def detect_gable(oblique_surfaces: list[dict[str, Any]]) -> bool:
    """True if the oblique roof surfaces form a classic gable.

    A gable is a pair of opposite-azimuth planes (≈ 180 deg apart) with similar
    inclinations (within 10 deg) whose combined area covers >= 70 % of the total
    oblique area. The 70 % threshold guards against dormers or small extra
    slopes while still accepting a dominant two-plane gable.
    """
    metas = [_oblique_meta(s) for s in oblique_surfaces]
    valid = [
        (az, incl, area)
        for az, incl, area in metas
        if az is not None and incl is not None and incl >= _MIN_GABLE_INCL_DEG
    ]
    if len(valid) < 2:
        return False

    total_area = (
        sum(
            area
            for _, incl, area in metas
            if incl is None or incl >= _MIN_GABLE_INCL_DEG
        )
        or 0.0
    )
    if total_area <= 0:
        return False

    best_pair_area = 0.0
    for i, (az_i, incl_i, area_i) in enumerate(valid):
        for az_j, incl_j, area_j in valid[i + 1 :]:
            az_delta = _angle_diff_deg(az_i, az_j)
            if abs(az_delta - 180.0) > _GABLE_AZIMUTH_TOL_DEG:
                continue
            if abs(incl_i - incl_j) > _GABLE_INCL_TOL_DEG:
                continue
            pair_area = area_i + area_j
            if pair_area > best_pair_area:
                best_pair_area = pair_area

    return best_pair_area >= _GABLE_AREA_FRACTION * total_area


def _n_stories(building: dict[str, Any]) -> int:
    """Prefer the precomputed `stories_found`, fall back to unique rooms[].story."""
    raw = building.get("stories_found")
    if isinstance(raw, int) and raw > 0:
        return raw
    rooms = building.get("rooms") or []
    stories = set()
    for r in rooms:
        s = r.get("story")
        if s is not None:
            try:
                stories.add(int(s))
            except (TypeError, ValueError):
                pass
    return len(stories)


def _has_half_height(building: dict[str, Any]) -> bool:
    flag = building.get("split_level")
    return bool(flag)


def _top_story(surfaces: list[dict[str, Any]]) -> int | None:
    """Highest `story` index found across a surface list.

    Roof surfaces carry the story key under different names: `story` for
    flats, `dominant_story` for oblique. Read both; skip values that cannot
    be parsed as integers (missing or null).
    """
    stories: list[int] = []
    for s in surfaces:
        for key in ("story", "dominant_story"):
            raw = s.get(key)
            if raw is None:
                continue
            try:
                stories.append(int(raw))
                break
            except (TypeError, ValueError):
                continue
    return max(stories) if stories else None


def _count_top_flat(flat: list[dict[str, Any]], oblique: list[dict[str, Any]]) -> int:
    """Count flat roof surfaces at the top story.

    The roof pipeline emits `roof_flat` entries for every ceiling in the
    stack, including inter-storey slabs. For tier classification we only
    care about flat regions that are actually on the roof -- i.e. at the
    same story as the highest oblique surface (or, if none, the highest
    flat). This keeps gable-only buildings out of the "mixed" tier.
    """
    if not flat:
        return 0
    top_from_oblique = _top_story(oblique)
    top = top_from_oblique if top_from_oblique is not None else _top_story(flat)
    if top is None:
        return len(flat)
    count = 0
    for s in flat:
        raw = s.get("story")
        try:
            if raw is not None and int(raw) >= top:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def classify_building(
    building: dict[str, Any], roof: dict[str, Any] | None
) -> dict[str, Any]:
    """Classify a single building into a tier 1-8 and return its signals."""
    rooms = building.get("rooms") or []
    roof_surfaces = ((roof or {}).get("roof_surfaces") or {}) if roof else {}
    oblique = roof_surfaces.get("oblique") or []
    flat = roof_surfaces.get("flat") or []

    n_stories = _n_stories(building)
    n_oblique = len(oblique)
    n_flat = _count_top_flat(flat, oblique)
    has_half_height = _has_half_height(building)
    has_gable = detect_gable(oblique) if n_oblique >= 2 else False

    if n_oblique == 0 and not has_half_height:
        if n_stories == 1:
            tier = 1
        elif n_stories == 2:
            tier = 2
        elif n_stories == 3:
            tier = 3
        else:
            tier = 8
    elif has_half_height and n_oblique == 0:
        tier = 4
    elif has_gable and n_flat == 0:
        tier = 6
    elif has_gable and n_flat >= 1:
        tier = 7
    elif n_stories == 1 and n_oblique >= 1 and not has_gable:
        tier = 5
    else:
        tier = 8

    return {
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "signals": {
            "n_stories": n_stories,
            "n_rooms": len(rooms),
            "n_oblique": n_oblique,
            "n_flat": n_flat,
            "has_half_height": has_half_height,
            "has_gable": has_gable,
        },
    }
