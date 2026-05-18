from dataclasses import replace

import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from reconcile_tiers.extract.building import (
    ExtractedRoom,
    ExtractedWall,
    RawCeilingPlane,
)
from reconcile_tiers.roof.clipping import _ring, clip_planes_to_footprint
from reconcile_tiers.roof.clustering import cluster_oblique_segments
from reconcile_tiers.roof.footprint import build_building_footprint
from reconcile_tiers.roof.obliques import (
    build_oblique_surfaces,
    build_raw_oblique_surfaces,
    story_floor_y,
)
from reconcile_tiers.roof.planes import build_roof_planes
from reconcile_tiers.roof.segments import collect_oblique_segments
from tests.reconcile_tiers.roof.helpers import (
    make_gable_model,
    make_two_story_flat_model,
)


def test_ring_extracts_polygon_from_geometry_collection():
    polygon = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)])
    geometry = GeometryCollection(
        [
            polygon,
            LineString([(4.0, 0.0), (5.0, 0.0)]),
        ]
    )

    assert _ring(geometry) == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]


def test_planes_clip_and_lift_to_oblique_surfaces_on_the_same_plane():
    model = make_gable_model()
    footprint = build_building_footprint(model)
    clusters = cluster_oblique_segments(collect_oblique_segments(model))
    planes = build_roof_planes(clusters, footprint)

    assert len(planes) == 1
    assert planes[0].ridge_span >= 2.0

    clipped = clip_planes_to_footprint(planes, footprint)
    assert len(clipped) == 1
    fp_poly = Polygon(footprint.polygon_xz)
    assert all(
        fp_poly.buffer(1e-6).contains(Point(x, z)) or fp_poly.touches(Point(x, z))
        for x, z in clipped[0].polygon_xz
    )

    obliques = build_oblique_surfaces(clipped, story_floor_y(model))
    assert len(obliques) == 1
    for x, y, z in obliques[0].corners:
        assert y == pytest.approx(obliques[0].plane.y_at(x, z))


def test_plane_clipping_uses_contributing_room_support_not_full_footprint():
    base = make_gable_model()
    far_floor = [[12.0, 0.0, 0.0], [16.0, 0.0, 0.0], [16.0, 0.0, 4.0], [12.0, 0.0, 4.0]]
    far_wall = ExtractedWall(
        id="far-flat-wall",
        corners=[
            [12.0, 0.0, 0.0],
            [16.0, 0.0, 0.0],
            [16.0, 2.4, 0.0],
            [12.0, 2.4, 0.0],
        ],
        source="test",
    )
    far_room = ExtractedRoom(
        index=1,
        story=0,
        floor_polygon=far_floor,
        walls_merged=[far_wall],
        walls_computed=[far_wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type="flat",
        ceiling_eave_height=2.4,
        ceiling_ridge_height=2.4,
    )
    model = replace(
        base,
        rooms=[base.rooms[0], far_room],
        scan_rooms_found=2,
        scan_rooms_transformed=2,
    )
    footprint = build_building_footprint(model)
    clusters = cluster_oblique_segments(collect_oblique_segments(model))
    planes = build_roof_planes(clusters, footprint)

    clipped = clip_planes_to_footprint(planes, footprint, model)
    obliques = build_oblique_surfaces(clipped, story_floor_y(model))

    assert len(obliques) == 1
    assert max(point[0] for point in obliques[0].corners) <= 7.05
    assert max(point[1] for point in obliques[0].corners) <= 5.6


def test_raw_oblique_surfaces_require_clean_large_rectangles():
    model = make_two_story_flat_model()
    clean_room = replace(
        model.rooms[0],
        raw_ceiling_planes=[
            RawCeilingPlane(
                corners=[
                    [0.0, 2.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 3.0],
                    [0.0, 2.0, 3.0],
                ]
            )
        ],
    )
    small_room = replace(
        model.rooms[0],
        raw_ceiling_planes=[
            RawCeilingPlane(
                corners=[
                    [0.0, 2.0, 0.0],
                    [1.0, 2.3, 0.0],
                    [1.0, 2.3, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            )
        ],
    )

    assert (
        len(
            build_raw_oblique_surfaces(
                replace(model, rooms=[clean_room, model.rooms[1]]), []
            )
        )
        == 1
    )
    assert (
        build_raw_oblique_surfaces(
            replace(model, rooms=[small_room, model.rooms[1]]), []
        )
        == []
    )
