import os
from collections import Counter
from pathlib import Path

import pytest

from reconcile_tiers.extract.building import extract_building_model

# Raw-plane counts shift under priors-ON because `merge_coplanar_raw_ceilings`
# runs in that path. Skip the legacy-cohort assertion in that mode.
_PRIORS_ON_SKIP = pytest.mark.skipif(
    os.environ.get("ARCHITECTURAL_PRIORS") == "1",
    reason="priors-ON merges coplanar raw ceilings; raw_plane counts shift",
)


@pytest.fixture(autouse=True)
def _force_legacy_priors_off(monkeypatch):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "0")


@_PRIORS_ON_SKIP
@pytest.mark.parametrize(
    (
        "uuid",
        "expected_rooms",
        "expected_stories",
        "expected_floor_count",
        "expected_merged_walls",
        "expected_computed_walls",
        "expected_sources",
        "expected_doors",
        "expected_windows",
        "expected_openings",
        "expected_storages",
        "expected_raw_ceiling_planes",
        "expected_raw_ceiling_source",
    ),
    [
        (
            "c72ad855-9e52-46f1-886d-a9f37911521f",
            10,
            2,
            10,
            50,
            51,
            {"scan-cache": 50, "scan-cache-dedup": 1},
            {"scan-cache": 14, "merged-room": 2},
            {"scan-cache": 9, "merged-room": 2},
            {"scan-cache": 2, "merged-room": 1},
            {"scan-cache": 11},
            21,
            {"noMesh": 10},
        ),
        (
            "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
            9,
            1,
            9,
            45,
            46,
            {"scan-cache": 45, "scan-cache-dedup": 1},
            {"scan-cache": 17, "merged-room": 2},
            {"scan-cache": 8, "merged-room": 2},
            {"scan-cache": 3},
            {"scan-cache": 20},
            9,
            {"noMesh": 9},
        ),
        (
            "2ea3b759-e047-424c-8034-f8ee5b811fb4",
            11,
            1,
            11,
            63,
            60,
            {"scan-cache": 60},
            {"scan-cache": 17, "merged-room": 3},
            {"scan-cache": 8, "merged-room": 3},
            {"scan-cache": 2},
            {"scan-cache": 28},
            12,
            {"noMesh": 11},
        ),
    ],
)
def test_extract_building_model_matches_legacy_wall_and_floor_counts(
    uuid,
    expected_rooms,
    expected_stories,
    expected_floor_count,
    expected_merged_walls,
    expected_computed_walls,
    expected_sources,
    expected_doors,
    expected_windows,
    expected_openings,
    expected_storages,
    expected_raw_ceiling_planes,
    expected_raw_ceiling_source,
):
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    assert model.uuid == uuid
    assert model.stories_found == expected_stories
    assert len(model.rooms) == expected_rooms
    assert sum(1 for room in model.rooms if room.floor_polygon) == expected_floor_count
    assert sum(len(room.walls_merged) for room in model.rooms) == expected_merged_walls
    assert (
        sum(len(room.walls_computed) for room in model.rooms) == expected_computed_walls
    )
    sources = Counter(
        wall.source for room in model.rooms for wall in room.walls_computed
    )
    assert sources == expected_sources
    assert (
        Counter(item.source for room in model.rooms for item in room.doors)
        == expected_doors
    )
    assert (
        Counter(item.source for room in model.rooms for item in room.windows)
        == expected_windows
    )
    assert (
        Counter(item.source for room in model.rooms for item in room.openings)
        == expected_openings
    )
    assert (
        Counter(item.source for room in model.rooms for item in room.storages)
        == expected_storages
    )
    assert (
        sum(len(room.raw_ceiling_planes) for room in model.rooms)
        == expected_raw_ceiling_planes
    )
    assert (
        Counter(
            room.raw_ceiling_source
            for room in model.rooms
            if room.raw_ceiling_source is not None
        )
        == expected_raw_ceiling_source
    )


def test_extract_layer_does_not_import_reconcile_v2_graph_builder():
    import reconcile_tiers.extract.building as building

    assert "reconcile_v2" not in repr(building.__dict__)
