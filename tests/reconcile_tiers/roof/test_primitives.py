from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from reconcile_tiers._core.wing_decomposition import Wing
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedRoom,
    ExtractedWall,
    RawCeilingPlane,
)
from reconcile_tiers.roof_primitive.flat import classify_flat
from reconcile_tiers.roof_primitive.gable import classify_gable, synthesise_gable
from reconcile_tiers.roof_primitive.shed import (
    classify_oblique,
    classify_shed,
    synthesise_oblique,
)


def _room(
    index: int, x0: float, x1: float, *, ceiling_type: str, ceiling
) -> ExtractedRoom:
    floor = [[x0, 0.0, 0.0], [x1, 0.0, 0.0], [x1, 0.0, 4.0], [x0, 0.0, 4.0]]
    return ExtractedRoom(
        index=index,
        story=0,
        floor_polygon=floor,
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[RawCeilingPlane(corners=ceiling)],
        raw_ceiling_source="test",
        ceiling_polygon=ceiling,
        ceiling_type=ceiling_type,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )


def _model(rooms: list[ExtractedRoom]) -> BuildingModel:
    return BuildingModel(
        uuid="synthetic-primitive",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=rooms,
        scan_rooms_found=len(rooms),
        scan_rooms_transformed=len(rooms),
    )


def _wall(wall_id: str, corners: list[list[float]]) -> ExtractedWall:
    return ExtractedWall(id=wall_id, corners=corners, source="test")


def _wing() -> Wing:
    poly = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)])
    return Wing(
        index=0, polygon=poly, area_m2=poly.area, role="main", long_axis_math=0.0
    )


def test_oblique_primitive_snaps_one_wing_plane_to_part_axis():
    rooms = []
    for idx, (x0, x1) in enumerate(((0.0, 2.0), (2.0, 4.0), (4.0, 6.0))):
        ceiling = [
            [x0, 2.0 + 0.15 * x0 + 0.03 * 0.0, 0.0],
            [x1, 2.0 + 0.15 * x1 + 0.03 * 0.0, 0.0],
            [x1, 2.0 + 0.15 * x1 + 0.03 * 4.0, 4.0],
            [x0, 2.0 + 0.15 * x0 + 0.03 * 4.0, 4.0],
        ]
        rooms.append(_room(idx, x0, x1, ceiling_type="sloped", ceiling=ceiling))

    params = classify_oblique(_wing(), 0, _model(rooms))

    assert params is not None
    assert params.slope_axis_math_deg == pytest.approx(0.0)
    assert params.eave_axis_math_deg == pytest.approx(90.0)
    assert params.plane.c == pytest.approx(0.0, abs=1e-9)
    surfaces = synthesise_oblique(params)
    assert len(surfaces) == 1
    assert len(surfaces[0].corners) == 4
    assert surfaces[0].cluster.avg_incl == pytest.approx(8.7, abs=0.1)


def test_shed_primitive_respects_excluded_simple_slant_rooms():
    ceiling = [[0.0, 2.0, 0.0], [6.0, 2.9, 0.0], [6.0, 2.9, 4.0], [0.0, 2.0, 4.0]]
    room = _room(0, 0.0, 6.0, ceiling_type="sloped", ceiling=ceiling)

    assert classify_shed(_wing(), 0, _model([room]), exclude_room_indices={0}) is None


def test_flat_primitive_classifies_flat_wing_without_oblique_surfaces():
    rooms = []
    for idx, (x0, x1) in enumerate(((0.0, 3.0), (3.0, 6.0))):
        ceiling = [[x0, 2.5, 0.0], [x1, 2.5, 0.0], [x1, 2.5, 4.0], [x0, 2.5, 4.0]]
        rooms.append(_room(idx, x0, x1, ceiling_type="flat", ceiling=ceiling))

    params = classify_flat(_wing(), 0, _model(rooms))

    assert params is not None
    assert params.y == pytest.approx(2.5)
    assert len(params.polygon_xz) == 4


def test_gable_primitive_uses_pentagonal_wall_top_profile():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ],
        walls_merged=[],
        walls_computed=[
            _wall(
                "eave-south",
                [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            ),
            _wall(
                "eave-north",
                [[6.0, 0.0, 4.0], [0.0, 0.0, 4.0], [0.0, 2.0, 4.0], [6.0, 2.0, 4.0]],
            ),
            _wall(
                "gable-west",
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 0.0, 4.0],
                    [0.0, 0.0, 0.0],
                ],
            ),
            _wall(
                "gable-east",
                [
                    [6.0, 2.0, 4.0],
                    [6.0, 4.0, 2.0],
                    [6.0, 2.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [6.0, 0.0, 4.0],
                ],
            ),
        ],
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
    )

    params = classify_gable(_wing(), 0, _model([room]), wall_axis_math=0.0)

    assert params is not None
    assert params.eave_y == pytest.approx(2.0)
    assert params.ridge_y == pytest.approx(4.0)
    assert params.ridge_axis_math_deg == pytest.approx(0.0)
    surfaces = synthesise_gable(params)
    assert len(surfaces) == 2
    assert {round(surface.cluster.avg_azimuth) for surface in surfaces} == {0, 180}
    assert all(surface.cluster.avg_incl == pytest.approx(45.0) for surface in surfaces)
