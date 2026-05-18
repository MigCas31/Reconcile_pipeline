from __future__ import annotations

from shapely.geometry import Polygon

from reconcile.roof_algorithms_py.ceiling_clipping_initial import (
    _per_plane_footprint,
    build_initial_plane_clips,
)


def _room(x0: float, z0: float, x1: float, z1: float) -> dict:
    return {
        "floor_polygon": [
            [x0, 0.0, z0],
            [x1, 0.0, z0],
            [x1, 0.0, z1],
            [x0, 0.0, z1],
        ]
    }


def _seg(x0: float, z0: float, x1: float, z1: float) -> dict:
    return {"a": [x0, 0.0, z0], "b": [x1, 0.0, z1]}


def test_per_plane_footprint_keeps_multi_wing_scan_evidence() -> None:
    rooms = [_room(0.0, 0.0, 2.0, 2.0), _room(6.0, 0.0, 8.0, 2.0)]
    wings = [
        Polygon([(0.0, -1.0), (3.0, -1.0), (3.0, 3.0), (0.0, 3.0)]),
        Polygon([(5.0, -1.0), (9.0, -1.0), (9.0, 3.0), (5.0, 3.0)]),
    ]
    plane = {
        "room_indices": [0, 1],
        "cl": {"segs": [_seg(0.5, 1.0, 1.5, 1.0), _seg(6.5, 1.0, 7.5, 1.0)]},
    }

    footprint = _per_plane_footprint(plane, rooms, buffer=0.1, wing_polygons=wings)

    assert footprint is not None
    bounds = Polygon(footprint).bounds
    assert bounds[0] <= 0.1
    assert bounds[2] >= 7.9


def test_per_plane_footprint_still_clips_single_wing_scan_evidence() -> None:
    rooms = [_room(0.0, 0.0, 2.0, 2.0), _room(6.0, 0.0, 8.0, 2.0)]
    wings = [
        Polygon([(0.0, -1.0), (3.0, -1.0), (3.0, 3.0), (0.0, 3.0)]),
        Polygon([(5.0, -1.0), (9.0, -1.0), (9.0, 3.0), (5.0, 3.0)]),
    ]
    plane = {
        "room_indices": [0, 1],
        "cl": {"segs": [_seg(0.5, 1.0, 1.5, 1.0)]},
    }

    footprint = _per_plane_footprint(plane, rooms, buffer=0.1, wing_polygons=wings)

    assert footprint is not None
    bounds = Polygon(footprint).bounds
    assert bounds[2] <= 3.0


def test_initial_clip_trusts_plane_footprint_outside_global_footprint() -> None:
    rooms = [_room(0.0, 0.0, 4.0, 2.0)]
    plane = {
        "cl": {"avgAzimuth": 0.0, "segs": [_seg(0.5, 1.0, 3.5, 1.0)]},
        "n": {"x": 0.0, "y": -1.0, "z": 0.0},
        "ref": {"x": 0.0, "y": 1.0, "z": 0.0},
        "ridgeX": 1.0,
        "ridgeZ": 0.0,
        "minRidge": 0.0,
        "maxRidge": 4.0,
        "slopeX": 0.0,
        "slopeZ": 1.0,
        "minSlope": 0.0,
        "maxSlope": 2.0,
        "dominantStory": 0,
        "room_indices": [0],
        "seed_room_indices": [0],
    }

    clipped = build_initial_plane_clips(
        ceiling_planes=[plane],
        building_footprint=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
        exposed_rooms=[],
        all_rooms=rooms,
    )[0]["clipped"]

    bounds = Polygon(clipped).bounds
    assert bounds[2] >= 3.9
