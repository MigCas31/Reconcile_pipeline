"""Tests for element lineage/provenance tracking."""

from reconcile.extract3d.lineage import (
    STEP_CLIP_WALLS_STORY,
    STEP_EXTRACT_WALLS,
    record,
)


def test_record_appends_to_empty_element():
    element = {"corners": [], "id": "w1", "source": "scan-cache"}
    record(element, STEP_EXTRACT_WALLS, "created", "method=floor-svd")

    assert "lineage" in element
    assert len(element["lineage"]) == 1
    entry = element["lineage"][0]
    assert entry["step"] == "extract_room_walls"
    assert entry["action"] == "created"
    assert entry["detail"] == "method=floor-svd"


def test_record_appends_multiple_entries():
    element = {"corners": [], "id": "w1", "source": "scan-cache"}
    record(element, STEP_EXTRACT_WALLS, "created", "method=floor-svd")
    record(element, STEP_CLIP_WALLS_STORY, "modified", "clipped y")

    assert len(element["lineage"]) == 2
    assert element["lineage"][0]["step"] == "extract_room_walls"
    assert element["lineage"][1]["step"] == "clip_walls_to_story_bounds"
    assert element["lineage"][1]["action"] == "modified"


def test_record_without_detail():
    element = {"corners": [], "id": "s1", "source": "scan-cache"}
    record(element, STEP_EXTRACT_WALLS, "created")

    entry = element["lineage"][0]
    assert "detail" not in entry
    assert entry["step"] == "extract_room_walls"
    assert entry["action"] == "created"


def test_record_preserves_existing_lineage():
    element = {
        "corners": [],
        "id": "w1",
        "lineage": [{"step": "previous", "action": "created"}],
    }
    record(element, STEP_CLIP_WALLS_STORY, "modified", "test")

    assert len(element["lineage"]) == 2
    assert element["lineage"][0]["step"] == "previous"
    assert element["lineage"][1]["step"] == "clip_walls_to_story_bounds"
