from __future__ import annotations

from reconcile.roof_algorithms_py.ceiling_clipping_caps import compute_plane_height_caps
from reconcile.roof_algorithms_py.flat_surface_generation import (
    build_flat_roof_surfaces,
)
from reconcile.roof_algorithms_py.footprint_derivation import build_building_footprint
from reconcile.roof_algorithms_py.math_utils import point_in_poly_2d, point_in_poly_xz
from reconcile_v2.models import GraphEdge, GraphNode, TopologyGraph


def _rect_floor(
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    *,
    y: float = 0.0,
) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def _flat_wall(x0: float, z0: float, x1: float, z1: float, top_y: float) -> dict:
    return {
        "corners": [
            [x0, 0.0, z0],
            [x1, 0.0, z1],
            [x1, top_y, z1],
            [x0, top_y, z0],
        ]
    }


def _flat_room(
    *,
    story: int,
    floor_polygon: list[list[float]],
    top_y: float,
    wall_count: int = 2,
) -> dict:
    x0, _, z0 = floor_polygon[0]
    x1, _, _ = floor_polygon[1]
    _, _, z1 = floor_polygon[2]
    walls = [
        _flat_wall(x0, z0, x1, z0, top_y),
        _flat_wall(x0, z1, x1, z1, top_y),
    ]
    return {
        "story": story,
        "floor_polygon": floor_polygon,
        "walls_computed": walls[:wall_count],
    }


def test_building_footprint_uses_only_exposed_room_set() -> None:
    exposed_rooms = [
        {"story": 0, "fp": _rect_floor(0.0, 0.0, 4.0, 4.0)},
        {"story": 0, "fp": _rect_floor(4.0, 0.0, 6.0, 2.0)},
    ]
    story_floor_polys = {
        1: [_rect_floor(100.0, 100.0, 104.0, 104.0, y=3.0)],
    }

    result = build_building_footprint(exposed_rooms, story_floor_polys)

    assert result["top_story"] == 0
    assert result["building_footprint"] is not None
    xs = [pt[0] for pt in result["building_footprint"]]
    zs = [pt[1] for pt in result["building_footprint"]]
    assert max(xs) < 10.0
    assert max(zs) < 10.0


def test_flat_roof_graph_can_include_non_max_story_roof_candidate() -> None:
    room0 = _flat_room(
        story=0,
        floor_polygon=_rect_floor(0.0, 0.0, 4.0, 4.0),
        top_y=2.8,
        wall_count=2,
    )
    room1 = _flat_room(
        story=1,
        floor_polygon=_rect_floor(10.0, 0.0, 12.0, 2.0, y=3.0),
        top_y=5.5,
        wall_count=1,
    )
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(
                id="room:r0",
                type="Room",
                story=0,
                bbox_xz=[0.0, 0.0, 4.0, 4.0],
                properties={"is_top_story": False},
            ),
            GraphNode(
                id="room:r1",
                type="Room",
                story=1,
                bbox_xz=[10.0, 0.0, 12.0, 2.0],
                properties={"is_top_story": True},
            ),
            GraphNode(
                id="cell:outside", type="Cell", properties={"cell_kind": "outside"}
            ),
        ],
        edges=[
            GraphEdge(
                id="exposes:r0-outside",
                type="EXPOSES_TO",
                from_id="room:r0",
                to_id="cell:outside",
                evidence={"relation_state": "confirmed"},
            )
        ],
    )

    flat_surfaces, _legend = build_flat_roof_surfaces(
        bldg={"rooms": [room0, room1]},
        all_stories=[0, 1],
        has_floor_above=lambda _x, _z, story: story == 0,
        bldg_min_y=0.0,
        graph=graph,
    )

    assert any(
        surface["kind"] == "top" and surface["story"] == 0 for surface in flat_surfaces
    )


def test_graph_cap_uses_explicit_upper_room_floor_and_skips_wall_top_fallback() -> None:
    lower_room = _flat_room(
        story=0,
        floor_polygon=_rect_floor(0.0, 0.0, 4.0, 4.0),
        top_y=2.6,
        wall_count=2,
    )
    upper_room = _flat_room(
        story=1,
        floor_polygon=_rect_floor(0.0, 0.0, 4.0, 4.0, y=3.0),
        top_y=9.0,
        wall_count=2,
    )
    graph = TopologyGraph(
        version="test",
        metadata={},
        nodes=[
            GraphNode(
                id="room:r0",
                type="Room",
                story=0,
                bbox_xz=[0.0, 0.0, 4.0, 4.0],
                properties={"is_top_story": False, "floor_height_y": 0.0},
            ),
            GraphNode(
                id="room:r1",
                type="Room",
                story=1,
                bbox_xz=[0.0, 0.0, 4.0, 4.0],
                properties={"is_top_story": True, "floor_height_y": 3.0},
            ),
            GraphNode(
                id="cell:room:r1",
                type="Cell",
                source_ids=["room:r1"],
                properties={
                    "cell_kind": "room",
                    "xz_footprint": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                    "bbox_xyz": [0.0, 3.0, 0.0, 4.0, 9.0, 4.0],
                },
            ),
        ],
        edges=[
            GraphEdge(
                id="above:r1-r0",
                type="ABOVE",
                from_id="room:r1",
                to_id="room:r0",
                evidence={"relation_state": "confirmed"},
            ),
            GraphEdge(
                id="below:r0-r1",
                type="BELOW",
                from_id="room:r0",
                to_id="room:r1",
                evidence={"relation_state": "confirmed"},
            ),
        ],
    )

    plane_max_y = compute_plane_height_caps(
        bldg={"rooms": [lower_room, upper_room]},
        ceiling_planes=[{"dominantStory": 0, "room_indices": [0]}],
        plane_clipped=[{"clipped": [(0.2, 0.2), (3.8, 0.2), (3.8, 3.8), (0.2, 3.8)]}],
        top_story=1,
        all_stories=[0, 1],
        floors_by_story={1: [upper_room["floor_polygon"]]},
        point_in_poly_xz=point_in_poly_xz,
        point_in_poly_2d=point_in_poly_2d,
        graph=graph,
    )

    assert plane_max_y == [3.0]
