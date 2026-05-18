"""Pre-pass to detect simple room-level slanted ceilings.

Rooms with isolated slanted ceilings (1-2 per story) are handled here
instead of the full oblique clustering pipeline, which can produce
incorrect oblique walls for these simple cases.
"""

from __future__ import annotations

from math import atan2, sqrt

from .graph_support import (
    classify_graph_roof_room,
    graph_allows_roof_candidate,
    graph_requires_geometry_fallback,
)

MAX_SLOPED_ROOMS_PER_STORY = 2
MIN_Y_RANGE = 0.15
MAX_Y_RANGE = 2.0
_AZIMUTH_MULTI_THRESHOLD = 90.0


def _collect_oblique_azimuths(room: dict) -> list[float]:
    """Return azimuths of oblique wall segments (5° < incl < 80°) in the room."""
    azimuths: list[float] = []
    for w in room.get("walls_computed") or []:
        corners = w.get("corners", [])
        n = len(corners)
        for i in range(n):
            pa, pb = corners[i], corners[(i + 1) % n]
            dx, dy, dz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
            length = sqrt(dx * dx + dy * dy + dz * dz)
            if length < 0.3:
                continue
            horiz = sqrt(dx * dx + dz * dz)
            if horiz < 1e-6:
                continue
            incl = atan2(abs(dy), horiz) * 180.0 / 3.141592653589793
            if 5.0 < incl < 80.0:
                azimuths.append((atan2(dx, dz) * 180.0 / 3.141592653589793) % 360.0)
    return azimuths


def _is_simple_slant_geometry(room: dict) -> bool:
    """Return True only if the room has oblique wall segments all facing
    roughly the same direction — a genuine mono-pitch slant.

    Returns False (not simple slant) when:
    - No oblique segments at all (ceiling_type=sloped is just wall height
      variation, not an actual slope)
    - Oblique segments span multiple distinct azimuths (>90° apart),
      indicating a multi-faceted roof (gable, hip, etc.)
    """
    azimuths = _collect_oblique_azimuths(room)
    if not azimuths:
        return False
    for i in range(len(azimuths)):
        for j in range(i + 1, len(azimuths)):
            diff = abs(azimuths[i] - azimuths[j])
            if diff > 180.0:
                diff = 360.0 - diff
            if diff > _AZIMUTH_MULTI_THRESHOLD:
                return False
    return True


def identify_simple_slant_rooms(
    bldg: dict,
    has_floor_above,
    all_stories: list[int],
    graph=None,
) -> dict:
    """Identify rooms with simple slanted ceilings that can be handled
    directly from extraction data, bypassing the full roof pipeline.

    Returns dict with:
        simple_slant_rooms: set of room indices to exclude from oblique pipeline
        simple_slant_ceilings: list of ceiling surface dicts ready for output
    """
    candidates: list[dict] = []

    for room_idx, room in enumerate(bldg.get("rooms", [])):
        if room.get("ceiling_type") != "sloped":
            continue
        ceiling_polygon = room.get("ceiling_polygon")
        if not ceiling_polygon or len(ceiling_polygon) < 3:
            continue

        fp = room.get("floor_polygon")
        if not fp or len(fp) < 3:
            continue

        story = int(room.get("story", 0))
        fcx = sum(p[0] for p in fp) / len(fp)
        fcz = sum(p[2] for p in fp) / len(fp)
        _graph_room, roof_decision = classify_graph_roof_room(graph, story, fp)
        if roof_decision is not None and not graph_allows_roof_candidate(roof_decision):
            continue
        if graph_requires_geometry_fallback(roof_decision) and has_floor_above(
            fcx, fcz, story
        ):
            continue

        ridge = room.get("ceiling_ridge_height", 0)
        eave = room.get("ceiling_eave_height", 0)
        y_range = ridge - eave
        if y_range < MIN_Y_RANGE or y_range > MAX_Y_RANGE:
            continue

        if not _is_simple_slant_geometry(room):
            continue

        candidates.append({"room_idx": room_idx, "story": story, "room": room})

    # Group by story and filter: only stories with <= MAX_SLOPED_ROOMS_PER_STORY qualify
    story_groups: dict[int, list[dict]] = {}
    for cand in candidates:
        story_groups.setdefault(cand["story"], []).append(cand)

    simple_rooms: set[int] = set()
    simple_ceilings: list[dict] = []

    for story, group in story_groups.items():
        if len(group) > MAX_SLOPED_ROOMS_PER_STORY:
            continue
        for cand in group:
            room = cand["room"]
            ceiling_polygon = room["ceiling_polygon"]
            simple_rooms.add(cand["room_idx"])
            simple_ceilings.append(
                {
                    "kind": "simple_slant",
                    "story": story,
                    "poly": [(p[0], p[1], p[2]) for p in ceiling_polygon],
                    "room_index": cand["room_idx"],
                }
            )

    return {
        "simple_slant_rooms": simple_rooms,
        "simple_slant_ceilings": simple_ceilings,
    }
