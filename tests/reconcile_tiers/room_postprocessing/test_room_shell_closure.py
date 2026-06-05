"""Tests for half-closed floor+wall room shells."""

from __future__ import annotations

import math
from typing import Any

from reconcile_tiers.polyhedron.manifold_repair import collect_room_tiles
from reconcile_tiers.room_postprocessing.room_shell_closure import (
    build_half_closed_room_shell,
)
from reconcile_tiers.room_postprocessing.segment_room_payload import (
    HALF_CLOSED_SHELL,
    build_segment_room_tier_payload,
)
from tests.reconcile_tiers.room_postprocessing.test_segment_room_cycles import (
    _four_wall_room_payload,
)


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _quantize(
    coord: tuple[float, float, float],
    tol: float,
) -> tuple[float, float, float]:
    return (
        round(coord[0] / tol) * tol,
        round(coord[1] / tol) * tol,
        round(coord[2] / tol) * tol,
    )


def _undirected_edge(
    a: dict[str, float],
    b: dict[str, float],
    tol: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ka = _quantize((a["x"], a["y"], a["z"]), tol)
    kb = _quantize((b["x"], b["y"], b["z"]), tol)
    return tuple(sorted((ka, kb)))


def _edges_from_ring(corners: list[dict[str, float]], tol: float) -> set:
    out: set = set()
    n = len(corners)
    for i in range(n):
        out.add(_undirected_edge(corners[i], corners[(i + 1) % n], tol))
    return out


def _wall_bottom_edges(wall: dict[str, Any], floor_y: float, tol: float) -> set:
    corners = wall["corners"]
    bottom = [c for c in corners if abs(c["y"] - floor_y) <= tol]
    if len(bottom) < 2:
        ys = sorted(c["y"] for c in corners)
        cutoff = ys[0] + tol
        bottom = [c for c in corners if c["y"] <= cutoff]
    if len(bottom) < 2:
        return set()
    return _edges_from_ring(bottom, tol) if len(bottom) >= 3 else {
        _undirected_edge(bottom[0], bottom[1], tol)
    }


def test_half_closed_shell_shared_corners() -> None:
    out = build_segment_room_tier_payload(_four_wall_room_payload(), corner_tol=0.05)
    room = out["rooms"][0]
    floor_corners = room["floor"][0]["corners"]
    floor_y = floor_corners[0]["y"]
    tol = 0.02
    floor_edges = _edges_from_ring(floor_corners, tol)

    wall_bottom_edges: set = set()
    for wall in room["walls"]:
        wall_bottom_edges |= _wall_bottom_edges(wall, floor_y, tol)

    assert floor_edges <= wall_bottom_edges


def test_segment_payload_preserves_building_ceilings() -> None:
    payload = _four_wall_room_payload()
    ceiling_piece = {
        "locator_id": "ceil-0",
        "corners": [
            _pt(0, 2.5, 0),
            _pt(4, 2.5, 0),
            _pt(4, 2.5, 3),
            _pt(0, 2.5, 3),
        ],
    }
    payload["ceiling"] = [ceiling_piece]
    payload["visual_shells"] = [
        {
            "locator_id": "shell-0",
            "corners": ceiling_piece["corners"],
        }
    ]
    out = build_segment_room_tier_payload(payload, corner_tol=0.05)
    assert len(out["ceiling"]) == 1
    assert out["ceiling"][0]["locator_id"] == "ceil-0"
    assert len(out["visual_shells"]) == 1
    assert out["room_postprocessing_source"]["shell"] == HALF_CLOSED_SHELL

    tiles = collect_room_tiles(out, out["rooms"][0])
    sources = {t.source for t in tiles}
    assert "ceiling" in sources
    assert "visual_shell" in sources
    assert "floor" in sources
    assert "wall" in sources


def test_per_junction_wall_tops() -> None:
    from reconcile_tiers.room_postprocessing.room_shell_closure import (
        wall_dict_from_perimeter_side_closed,
    )

    floor_y = 0.0
    group_ids = ["g0", "g1", "g2"]
    polygon_xz = [
        {"x": 0.0, "z": 0.0},
        {"x": 4.0, "z": 0.0},
        {"x": 2.0, "z": 3.0},
    ]
    rep = {
        "g0": ["s0"],
        "g1": ["s1"],
        "g2": ["s2"],
    }
    segments_by_id = {
        "s0": {
            "id": "s0",
            "wall_id": "w-south",
            "start": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end": {"x": 0.0, "y": 2.0, "z": 0.0},
        },
        "s1": {
            "id": "s1",
            "wall_id": "w-south",
            "start": {"x": 4.0, "y": 0.0, "z": 0.0},
            "end": {"x": 4.0, "y": 3.5, "z": 0.0},
        },
        "s2": {
            "id": "s2",
            "wall_id": "w-east",
            "start": {"x": 4.0, "y": 0.0, "z": 0.0},
            "end": {"x": 4.0, "y": 2.5, "z": 0.0},
        },
    }
    wall = wall_dict_from_perimeter_side_closed(
        {
            "source_group": "g0",
            "target_group": "g1",
            "wall_id": "w-south",
        },
        group_ids=group_ids,
        polygon_xz=polygon_xz,
        floor_y=floor_y,
        representative_by_group=rep,
        segments_by_id=segments_by_id,
    )
    assert wall is not None
    top_ys = [c["y"] for c in wall["corners"] if c["y"] > floor_y + 0.01]
    assert len(top_ys) == 2
    assert not math.isclose(top_ys[0], top_ys[1], abs_tol=0.01)
