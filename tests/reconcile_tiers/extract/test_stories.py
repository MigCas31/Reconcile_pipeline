import json
from collections import Counter
from pathlib import Path

import pytest

from reconcile_tiers.extract.stories import compute_room_stories, is_split_level
from reconcile_tiers.ingest.merged import load_merged


@pytest.mark.parametrize(
    ("uuid", "stories_found", "story_counts", "split_level"),
    [
        ("c72ad855-9e52-46f1-886d-a9f37911521f", 2, {0: 6, 1: 4}, False),
        ("f40dcc9f-b97b-4bef-8b40-ba011aabf0bd", 1, {0: 9}, False),
        ("2ea3b759-e047-424c-8034-f8ee5b811fb4", 1, {0: 11}, False),
    ],
)
def test_compute_room_stories_matches_legacy_cohort(
    uuid, stories_found, story_counts, split_level
):
    merged = load_merged(uuid, Path("pipeline-outputs"))

    assignment = compute_room_stories(merged)

    assert assignment.stories_found == stories_found
    assert {
        story: assignment.room_stories.count(story)
        for story in sorted(set(assignment.room_stories))
    } == story_counts
    assert (
        is_split_level(assignment.floor_polygons, assignment.room_stories)
        is split_level
    )


SPLIT_LEVEL_LEGACY_DIVERGENCES = {
    # New is_split_level catches a half-floor that legacy extract_3d missed:
    # story-1 rooms straddle Y=0.78 and Y=0.025, putting the story-0/story-1
    # delta at 1.82m (< SPLIT_LEVEL_DY_M=2.0). Per "keep diagnostic signal",
    # the new positive is the correct read; pinning here so future regressions
    # in either direction surface clearly.
    "98697a13-7b1d-4813-82b0-90ef7483cdec",
}


def test_compute_room_stories_matches_legacy_corpus():
    legacy = {
        building["uuid"]: building
        for building in json.loads(Path("reconcile/buildings_3d.json").read_text())
    }

    for merged_path in sorted(Path("pipeline-outputs").glob("*/merged.json")):
        uuid = merged_path.parent.name
        legacy_building = legacy[uuid]

        assignment = compute_room_stories(load_merged(uuid, Path("pipeline-outputs")))

        assert assignment.stories_found == legacy_building.get("stories_found"), uuid
        if uuid not in SPLIT_LEVEL_LEGACY_DIVERGENCES:
            assert is_split_level(
                assignment.floor_polygons, assignment.room_stories
            ) is bool(legacy_building.get("split_level")), uuid
        assert Counter(assignment.room_stories) == Counter(
            int(room.get("story", 0)) for room in legacy_building.get("rooms") or []
        ), uuid


def test_story_gap_over_one_meter_starts_new_story():
    floor_polygons = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        [[0.0, 0.8, 0.0], [1.0, 0.8, 0.0], [1.0, 0.8, 1.0]],
        [[0.0, 2.1, 0.0], [1.0, 2.1, 0.0], [1.0, 2.1, 1.0]],
    ]

    assignment = compute_room_stories(floor_polygons)

    assert assignment.room_stories == [0, 0, 1]
    assert assignment.stories_found == 2
