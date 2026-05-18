from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers.extract.building import ExtractedRoom, ExtractedWall
from reconcile_tiers.extract.inferred_rooms import infer_enclosed_void_rooms


def _wall(wall_id, p0, p1, *, y_min=0.0, y_max=2.5):
    return ExtractedWall(
        id=wall_id,
        source="test",
        corners=[
            [p0[0], y_min, p0[1]],
            [p1[0], y_min, p1[1]],
            [p1[0], y_max, p1[1]],
            [p0[0], y_max, p0[1]],
        ],
    )


def _room(index, coords):
    floor = [[x, 0.0, z] for x, z in coords]
    walls = [
        _wall(f"r{index}w{idx}", coords[idx], coords[(idx + 1) % len(coords)])
        for idx in range(len(coords))
    ]
    return ExtractedRoom(
        index=index,
        story=0,
        floor_polygon=floor,
        walls_merged=[],
        walls_computed=walls,
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
        heating="radiators",
    )


def test_infer_enclosed_void_rooms_creates_synthetic_room_for_room_scale_hole():
    rooms = [
        _room(0, [(0, 0), (4, 0), (4, 1), (0, 1)]),
        _room(1, [(0, 3), (4, 3), (4, 5), (0, 5)]),
        _room(2, [(0, 1), (1, 1), (1, 3), (0, 3)]),
        _room(3, [(3, 1), (4, 1), (4, 3), (3, 3)]),
    ]

    inferred = infer_enclosed_void_rooms(rooms)

    assert len(inferred) == 5
    synthetic = inferred[-1]
    assert synthetic.index == 4
    assert synthetic.story == 0
    assert synthetic.heating == "radiators"
    assert synthetic.ceiling_type == "flat"
    assert len(synthetic.walls_computed) == 4
    assert all(wall.synthetic for wall in synthetic.walls_computed)

    floor = Polygon([(corner[0], corner[2]) for corner in synthetic.floor_polygon])
    assert round(floor.area, 3) == 4.0
    assert tuple(round(value, 3) for value in floor.bounds) == (1.0, 1.0, 3.0, 3.0)

    union = unary_union(
        [
            Polygon([(corner[0], corner[2]) for corner in room.floor_polygon])
            for room in inferred
        ]
    )
    assert isinstance(union, Polygon)
    assert len(union.interiors) == 0


def test_infer_enclosed_void_rooms_ignores_wall_thickness_sliver():
    rooms = [
        _room(0, [(0, 0), (2, 0), (2, 2), (0, 2)]),
        _room(1, [(2.4, 0), (4.4, 0), (4.4, 2), (2.4, 2)]),
        _room(2, [(0, 2.4), (4.4, 2.4), (4.4, 4.4), (0, 4.4)]),
    ]

    assert infer_enclosed_void_rooms(rooms) == rooms
