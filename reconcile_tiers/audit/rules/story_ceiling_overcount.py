"""rule_story_ceiling_overcount: story has many more ceilings than rooms.

Two patterns trigger this:
- Oversaturated: rooms <= 2 AND ceilings >= 5 in the same story.
- High ratio: non-top story with ceilings/rooms > 3.

Both indicate that ceiling pieces have been attributed to the wrong story (or
that the story splitter under-segmented rooms). Cohort scan over 446 buildings
(2026-05-08) found 8 + 9 + 27 buildings affected by these patterns; together
they are the second largest upstream contamination class for plane-selection
eval. The flag is attached to the first room in the offending story so the
viewer surfaces a clickable target.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from reconcile_tiers.audit.rules._shared import (
    STORY_OVERSAT_CEILS_MIN,
    STORY_OVERSAT_ROOMS_MAX,
    STORY_RATIO_MAX,
    FlagItem,
    _make_item,
    _story_y_bands,
    _y_range,
)


def rule_story_ceiling_overcount(payload: dict[str, Any]) -> list[FlagItem]:
    items: list[FlagItem] = []
    rooms = payload.get("rooms") or []
    if not rooms:
        return items

    bands = _story_y_bands(rooms)
    if not bands:
        return items

    rooms_by_story: dict[int, list[dict[str, Any]]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        rooms_by_story.setdefault(int(story), []).append(room)

    if not rooms_by_story:
        return items

    top_story = max(rooms_by_story.keys())

    # Attribute each ceiling piece to a story by mid-Y vs the bands. Pieces
    # outside any band are skipped (the dedicated ceiling_below_floor /
    # out_of_envelope rules already cover those failure modes).
    ceilings_by_story: Counter[int] = Counter()
    for piece in payload.get("ceiling") or []:
        corners = piece.get("corners") or []
        yr = _y_range(corners)
        if yr is None:
            continue
        mid_y = 0.5 * (yr[0] + yr[1])
        for story, (lo, hi) in bands.items():
            if lo - 0.5 <= mid_y <= hi + 0.5:
                ceilings_by_story[story] += 1
                break

    for story, room_list in rooms_by_story.items():
        n_rooms = len(room_list)
        n_ceilings = ceilings_by_story.get(story, 0)
        if n_rooms == 0:
            continue
        ratio = n_ceilings / n_rooms

        oversaturated = (
            n_rooms <= STORY_OVERSAT_ROOMS_MAX
            and n_ceilings >= STORY_OVERSAT_CEILS_MIN
        )
        high_ratio = story != top_story and ratio > STORY_RATIO_MAX

        if not (oversaturated or high_ratio):
            continue

        sub_pattern = "oversaturated" if oversaturated else "high_ratio"
        # Severity: oversaturated patterns are worse than high-ratio because
        # they imply room undercount; high_ratio on a non-top story implies
        # cross-story attribution.
        if oversaturated and ratio >= 8.0:
            severity = "high"
        elif oversaturated:
            severity = "medium"
        else:
            severity = "medium" if ratio > 5.0 else "low"

        anchor_locator = room_list[0].get("locator_id")
        items.append(
            _make_item(
                anchor_locator,
                rule="story_ceiling_overcount",
                severity=severity,
                evidence={
                    "story": int(story),
                    "is_top_story": story == top_story,
                    "rooms_in_story": n_rooms,
                    "ceilings_in_story": n_ceilings,
                    "ratio": float(ratio),
                    "sub_pattern": sub_pattern,
                    "rooms_by_story": {
                        str(s): len(rs) for s, rs in sorted(rooms_by_story.items())
                    },
                    "ceilings_by_story": {
                        str(s): n for s, n in sorted(ceilings_by_story.items())
                    },
                },
            )
        )
    return items
