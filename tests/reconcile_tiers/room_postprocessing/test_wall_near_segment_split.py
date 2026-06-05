"""Tests for splitting walls when another wall's segment is nearby."""

from __future__ import annotations

from typing import Any

from reconcile_tiers.room_postprocessing.export import build_corner_graph


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _segment_near_wall_without_corner_payload() -> dict[str, Any]:
    """Vertical wall at junction; passing wall has no corner at the junction."""

    return {
        "uuid": "near-seg-split",
        "rooms": [
            {
                "story": 0,
                "walls": [
                    {
                        "locator_id": "w-junction",
                        "corners": [
                            _pt(2.0, 0.0, 2.0),
                            _pt(2.1, 0.0, 2.0),
                            _pt(2.1, 2.0, 2.0),
                            _pt(2.0, 2.0, 2.0),
                        ],
                    },
                    {
                        "locator_id": "w-passing",
                        "corners": [
                            _pt(0.0, 0.0, 2.08),
                            _pt(4.0, 0.0, 2.08),
                            _pt(4.0, 2.0, 2.08),
                            _pt(0.0, 2.0, 2.08),
                        ],
                    },
                ],
            }
        ],
    }


def test_near_segment_splits_passing_wall_and_adds_vertical_segment() -> None:
    graph = build_corner_graph(
        _segment_near_wall_without_corner_payload(),
        corner_tol=0.05,
        adjacency_tol=0.5,
    )
    wall_ids = {n["id"] for n in graph["nodes"] if n["kind"] == "wall"}
    assert "w-passing::split::0" in wall_ids
    assert "w-passing::split::1" in wall_ids

    sg = graph["wall_segment_graph"]
    passing_segments = [s for s in sg["segments"] if "w-passing" in s["wall_id"]]
    assert len(passing_segments) >= 4
    junction_x = [s for s in passing_segments if abs(s["start"]["x"] - 2.0) < 0.15]
    assert junction_x
