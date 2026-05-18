import warnings

from shapely.geometry import MultiPolygon, Polygon

import reconcile_tiers._core.shapely2 as shapely2
from reconcile_tiers._core.shapely2 import (
    coverage_union,
    make_valid,
    make_valid_polygon,
    oriented_rectangle_side_lengths,
    query_intersecting,
)


def test_make_valid_repairs_bowtie_to_non_empty_geometry():
    bowtie = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])

    repaired = make_valid(bowtie)

    assert repaired.is_valid
    assert not repaired.is_empty


def test_make_valid_polygon_returns_largest_polygonal_part():
    geometry = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
        ]
    )

    repaired = make_valid_polygon(geometry)

    assert repaired is not None
    assert repaired.geom_type == "Polygon"
    assert repaired.is_valid
    assert repaired.area == 4.0


def test_coverage_union_merges_non_overlapping_coverage():
    left = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    merged = coverage_union([left, right])

    assert merged.area == 2.0


def test_oriented_rectangle_side_lengths_suppresses_shapely_runtime_warnings():
    poly = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sides = oriented_rectangle_side_lengths(poly)

    assert sorted(round(side, 6) for side in sides) == [2.0, 2.0, 4.0, 4.0]
    assert recorded == []


def test_query_intersecting_returns_geometries_not_indices():
    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
    ]
    needle = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])

    hits = query_intersecting(geoms, needle)

    assert hits == [geoms[0]]


def test_query_intersecting_handles_no_hits_and_geometry_returning_strtree(monkeypatch):
    class FakeTree:
        def __init__(self, geoms):
            self.geoms = geoms

        def query(self, query_geometry):
            return self.geoms if query_geometry is not None else []

    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    monkeypatch.setattr(shapely2, "STRtree", FakeTree)

    assert shapely2.query_intersecting(geoms, geoms[0]) == geoms
    assert shapely2.query_intersecting(geoms, None) == []
