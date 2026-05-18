from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from reconcile_tiers.extract.geometry import corners_to_world
from reconcile_tiers.ingest.merged import MergedDoc

STORY_Y_GAP_M = 1.0
SPLIT_LEVEL_DY_M = 2.0


@dataclass(frozen=True, slots=True)
class StoryAssignment:
    floor_ys: list[float]
    floor_polygons: list[list[list[float]]]
    room_stories: list[int]
    stories_found: int


def _floor_polygons_from_merged(merged: MergedDoc) -> list[list[list[float]]]:
    floors: list[list[list[float]]] = []
    for room in merged.rooms:
        floor_items = room.data.get("floors") or []
        if floor_items and floor_items[0].get("polygonCorners"):
            floors.append(
                corners_to_world(
                    floor_items[0]["polygonCorners"], floor_items[0]["transform"]
                )
            )
        else:
            floors.append([])
    return floors


def compute_room_stories(
    merged_or_floor_polygons: MergedDoc | Sequence[Sequence[Sequence[float]]],
) -> StoryAssignment:
    if isinstance(merged_or_floor_polygons, MergedDoc):
        floor_polygons = _floor_polygons_from_merged(merged_or_floor_polygons)
    else:
        floor_polygons = [
            [list(point) for point in polygon] for polygon in merged_or_floor_polygons
        ]

    floor_ys = [
        float(np.mean([corner[1] for corner in floor_polygon]))
        if floor_polygon
        else 0.0
        for floor_polygon in floor_polygons
    ]
    sorted_ys = sorted(set(floor_ys))
    story_map: dict[float, int] = {}
    story = 0
    for idx, y_value in enumerate(sorted_ys):
        if idx > 0 and abs(y_value - sorted_ys[idx - 1]) > STORY_Y_GAP_M:
            story += 1
        story_map[y_value] = story
    room_stories = []
    for floor_y in floor_ys:
        closest_y = (
            min(sorted_ys, key=lambda y_value: abs(y_value - floor_y))
            if sorted_ys
            else 0.0
        )
        room_stories.append(story_map.get(closest_y, 0))
    return StoryAssignment(
        floor_ys=floor_ys,
        floor_polygons=floor_polygons,
        room_stories=room_stories,
        stories_found=(max(room_stories) + 1 if room_stories else 0),
    )


def is_split_level(
    floor_polygons: Sequence[Sequence[Sequence[float]]], room_stories: Sequence[int]
) -> bool:
    if not room_stories:
        return False
    story_counts = {story: room_stories.count(story) for story in set(room_stories)}
    if len(story_counts) < 2:
        return False
    if any(count == 1 for count in story_counts.values()):
        return True

    ys_by_story: dict[int, list[float]] = {}
    for story, floor_polygon in zip(room_stories, floor_polygons, strict=False):
        if not floor_polygon:
            continue
        ys_by_story.setdefault(story, []).append(
            float(np.mean([corner[1] for corner in floor_polygon]))
        )
    mean_y = {
        story: float(np.mean(values)) for story, values in ys_by_story.items() if values
    }
    ordered = sorted(mean_y)
    deltas = [
        mean_y[ordered[idx + 1]] - mean_y[ordered[idx]]
        for idx in range(len(ordered) - 1)
    ]
    return any(delta < SPLIT_LEVEL_DY_M for delta in deltas)
