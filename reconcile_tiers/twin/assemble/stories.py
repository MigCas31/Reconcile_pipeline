"""Step 5 of the assembly: Story per room-group.

Anchor: extraction's `ExtractedRoom.story` (an int per room). The
canonical Story groups all Rooms with the same story_index. The
extraction's classification is treated as ScanEvidence — a future
phase can replace it with the structural definition (a maximal
connected set of co-storied Rooms by wall adjacency) without
changing the Story type.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from reconcile_tiers.twin.types import Room, Story


def stories_from_rooms(
    rooms: Iterable[Room], *, building_uuid: str
) -> tuple[Story, ...]:
    """Group `rooms` by `story_index` and emit one Story per group."""
    by_story: dict[int, list[Room]] = defaultdict(list)
    for r in rooms:
        by_story[r.story_index].append(r)

    return tuple(
        Story(
            id=f"{building_uuid}::story::{idx}",
            rooms=tuple(grouped),
        )
        for idx, grouped in sorted(by_story.items())
    )
