"""Tests for manifold-repair-steps export using segment-derived rooms."""

from __future__ import annotations

import json
from pathlib import Path

from reconcile_tiers.polyhedron.corpus_trace_export import (
    export_manifold_repair_steps_traces,
)


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _four_wall_tier_payload() -> dict:
    return {
        "uuid": "trace-four-wall",
        "rooms": [
            {
                "story": 0,
                "locator_id": "room-0",
                "walls": [
                    {
                        "locator_id": "w-south",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(4, 0, 0),
                            _pt(4, 2, 0),
                            _pt(0, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-east",
                        "corners": [
                            _pt(4, 0, 0),
                            _pt(4, 0, 3),
                            _pt(4, 2, 3),
                            _pt(4, 2, 0),
                        ],
                    },
                    {
                        "locator_id": "w-north",
                        "corners": [
                            _pt(4, 0, 3),
                            _pt(0, 0, 3),
                            _pt(0, 2, 3),
                            _pt(4, 2, 3),
                        ],
                    },
                    {
                        "locator_id": "w-west",
                        "corners": [
                            _pt(0, 0, 3),
                            _pt(0, 0, 0),
                            _pt(0, 2, 0),
                            _pt(0, 2, 3),
                        ],
                    },
                ],
                "floor": [
                    {
                        "locator_id": "floor-0",
                        "corners": [
                            _pt(0, 0, 0),
                            _pt(4, 0, 0),
                            _pt(4, 0, 3),
                            _pt(0, 0, 3),
                        ],
                    }
                ],
                "doors": [],
                "windows": [],
            }
        ],
        "ceiling": [],
    }


def test_manifold_steps_export_tier_rooms_unchanged(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    building_dir = pipeline_dir / "trace-four-wall"
    building_dir.mkdir(parents=True)
    (building_dir / "tier_payload.json").write_text(
        json.dumps(_four_wall_tier_payload())
    )

    out_dir = tmp_path / "traces-out-tier"
    index = export_manifold_repair_steps_traces(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
        max_buildings=1,
        room_source="tier",
        write_segment_payload=False,
    )

    assert index["records"][0]["room_source"] == "tier"
    assert not (building_dir / "tier_payload_segment_tier_rooms.json").exists()


def test_manifold_steps_export_segment_tier_rooms(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    building_dir = pipeline_dir / "trace-four-wall"
    building_dir.mkdir(parents=True)
    (building_dir / "tier_payload.json").write_text(
        json.dumps(_four_wall_tier_payload())
    )

    out_dir = tmp_path / "traces-out-segment-tier"
    index = export_manifold_repair_steps_traces(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
        max_buildings=1,
        room_source="segment-tier",
        segment_corner_tol=0.05,
        segment_adjacency_tol=0.5,
    )

    assert index["settings"]["room_source"] == "segment-tier"
    record = index["records"][0]
    assert record["room_source"] == "segment-tier"

    segment_path = building_dir / "tier_payload_segment_tier_rooms.json"
    assert segment_path.is_file()
    segment_payload = json.loads(segment_path.read_text())
    assert segment_payload["room_postprocessing_source"]["geometry_source"] == "perimeter_walls"
    room = segment_payload["rooms"][0]
    assert len(room["walls"]) >= 3
    wall_ids = {w["locator_id"] for w in room["walls"]}
    assert wall_ids == {"w-south", "w-east", "w-north", "w-west"}
    for wall in room["walls"]:
        assert len(wall.get("corners") or []) == 4
