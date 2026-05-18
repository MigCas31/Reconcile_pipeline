from pathlib import Path

import pytest

from reconcile_tiers.ingest.merged import load_merged

COHORT_ROOM_COUNTS = {
    "c72ad855-9e52-46f1-886d-a9f37911521f": 10,
    "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd": 9,
    "2ea3b759-e047-424c-8034-f8ee5b811fb4": 11,
}


@pytest.mark.parametrize(("uuid", "expected_count"), COHORT_ROOM_COUNTS.items())
def test_load_merged_returns_known_room_count(uuid: str, expected_count: int):
    merged = load_merged(uuid, Path("pipeline-outputs"))

    assert merged.uuid == uuid
    assert merged.path.name == "merged.json"
    assert len(merged.rooms) == expected_count
    assert len(merged.data["rooms"]) == expected_count


def test_load_merged_documents_room_shape():
    merged = load_merged(
        "c72ad855-9e52-46f1-886d-a9f37911521f", Path("pipeline-outputs")
    )

    assert set(merged.data) == {
        "doors",
        "floors",
        "objects",
        "openings",
        "rooms",
        "sections",
        "version",
        "walls",
        "windows",
    }
    first_room = merged.rooms[0]
    assert first_room.index == 0
    assert first_room.story == 0
    assert set(first_room.data) == {
        "coreModel",
        "doors",
        "floors",
        "objects",
        "openings",
        "referenceOriginTransform",
        "sections",
        "story",
        "version",
        "walls",
        "windows",
    }
    assert len(first_room.reference_origin_transform) == 16


def test_load_merged_raises_for_unknown_uuid(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing-uuid"):
        load_merged("missing-uuid", tmp_path)
