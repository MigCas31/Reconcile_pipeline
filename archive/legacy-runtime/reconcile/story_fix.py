"""Fix story assignments by clustering rooms on floor Y position."""

from __future__ import annotations

import numpy as np

from .models import Building, Room
from .transform import corners_to_world


def get_floor_y(room: Room) -> float | None:
    """Get the Y position of a room's floor polygon in world coordinates."""
    if not room.floors or not room.floors[0].polygon_corners:
        return None
    wc = corners_to_world(room.floors[0].polygon_corners, room.floors[0].transform)
    if not wc:
        return None
    return float(np.mean([c.y for c in wc]))


def cluster_stories(
    floor_ys: list[float],
    min_gap: float = 1.0,
) -> list[int]:
    """Cluster floor Y values into story indices.

    Finds the natural gaps between floor Y values. Any gap > min_gap is
    treated as a story boundary.
    Returns story index (0-based, lowest floor = 0) for each input.
    """
    if not floor_ys:
        return []

    indexed = sorted(enumerate(floor_ys), key=lambda x: x[1])
    story_assignments = [0] * len(floor_ys)

    current_story = 0
    prev_y = indexed[0][1]

    for orig_idx, y in indexed:
        if y - prev_y > min_gap:
            current_story += 1
        story_assignments[orig_idx] = current_story
        prev_y = y

    return story_assignments


def fix_building_stories(building: Building) -> dict:
    """Re-assign story values based on floor Y clustering.

    Returns a summary of changes made.
    """
    floor_ys = []
    room_indices = []

    for i, room in enumerate(building.rooms):
        fy = get_floor_y(room)
        if fy is not None:
            floor_ys.append(fy)
            room_indices.append(i)

    if not floor_ys:
        return {"changed": 0, "stories_found": 0}

    new_stories = cluster_stories(floor_ys)

    changes = 0
    for room_idx, new_story in zip(room_indices, new_stories, strict=False):
        room = building.rooms[room_idx]
        if room.story != new_story:
            room.story = new_story
            # Also update wall/door/window story values
            for surface in room.walls + room.doors + room.windows:
                surface.story = new_story
            for floor in room.floors:
                floor.story = new_story
            changes += 1

    stories_found = max(new_stories) + 1 if new_stories else 0

    return {
        "changed": changes,
        "stories_found": stories_found,
        "story_floor_ys": {
            s: float(
                np.mean(
                    [
                        fy
                        for fy, ns in zip(floor_ys, new_stories, strict=False)
                        if ns == s
                    ]
                )
            )
            for s in range(stories_found)
        },
    }
