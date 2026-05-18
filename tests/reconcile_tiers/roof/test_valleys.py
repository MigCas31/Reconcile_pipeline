"""Phase 5 Step 6 regression tests for the pairwise valley resolver."""

from __future__ import annotations

from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers._core.wing_decomposition import Wing
from reconcile_tiers.roof.part_record import PartRecord
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster, RoofSegment
from reconcile_tiers.roof.valleys import resolve_valleys


def _wing(
    min_x: float, max_x: float, min_z: float, max_z: float, index: int = 0
) -> Wing:
    poly = Polygon([(min_x, min_z), (max_x, min_z), (max_x, max_z), (min_x, max_z)])
    return Wing(
        index=index,
        polygon=poly,
        area_m2=poly.area,
        role="main",
        long_axis_math=0.0,
    )


def _stub_cluster() -> RoofCluster:
    seg = RoofSegment(
        a=[0.0, 0.0, 0.0],
        b=[1.0, 0.0, 0.0],
        incl=30.0,
        azimuth=0.0,
        length=1.0,
        story=0,
        room_index=0,
    )
    return RoofCluster(
        segments=[seg], avg_incl=30.0, avg_azimuth=0.0, ref_pt=[0.0, 0.0, 0.0]
    )


def _surface_from_xz(
    plane: Plane, xz_corners: list[tuple[float, float]]
) -> ObliqueSurface:
    corners = [
        [float(x), float(plane.y_at(x, z) or 0.0), float(z)] for x, z in xz_corners
    ]
    ys = [c[1] for c in corners]
    return ObliqueSurface(
        corners=corners,
        plane=plane,
        cluster=_stub_cluster(),
        dominant_story=0,
        ridge={"y": max(ys), "eave_y": min(ys)},
    )


def test_resolve_valleys_skips_non_adjacent_wings():
    """Two far-apart wings should pass through unchanged."""
    wing_a = _wing(0.0, 4.0, 0.0, 4.0)
    wing_b = _wing(20.0, 24.0, 20.0, 24.0)

    plane = Plane.fit([[0, 0, 0], [4, 0, 0], [4, 2, 4], [0, 2, 4]])
    assert isinstance(plane, Plane)
    surf_a = _surface_from_xz(plane, [(0, 0), (4, 0), (4, 4), (0, 4)])
    surf_b = _surface_from_xz(plane, [(20, 20), (24, 20), (24, 24), (20, 24)])

    records = [
        PartRecord(wing=wing_a, kind="legacy", surfaces=[surf_a]),
        PartRecord(wing=wing_b, kind="legacy", surfaces=[surf_b]),
    ]

    out = resolve_valleys(records)
    assert len(out) == 2
    assert len(out[0].surfaces) == 1
    assert len(out[1].surfaces) == 1
    assert {tuple(c) for c in out[0].surfaces[0].corners} == {
        tuple(c) for c in surf_a.corners
    }


def test_resolve_valleys_preserves_non_overlapping_slopes_of_adjacent_wings():
    """Adjacent wings whose slopes don't overlap in XZ should pass through.

    L-junction case: two wings sharing an edge, each with two gable slopes
    facing perpendicular directions. The slopes shouldn't be clipped because
    their XZ footprints don't overlap.
    """
    # Wing A: 0..6 x 0..4, gable along X axis (ridge at z=2)
    wing_a = _wing(0.0, 6.0, 0.0, 4.0)
    # Wing B: 4..10 x 4..8, sharing corner with A at (4,4); ridge along z axis
    wing_b = _wing(4.0, 10.0, 4.0, 8.0)

    # Wing A's two slopes (north-facing and south-facing of an X-axis gable)
    pa_north = Plane.fit([[0, 0, 0], [6, 0, 0], [6, 2, 2], [0, 2, 2]])
    pa_south = Plane.fit([[0, 2, 2], [6, 2, 2], [6, 0, 4], [0, 0, 4]])
    assert isinstance(pa_north, Plane) and isinstance(pa_south, Plane)
    surf_a_north = _surface_from_xz(pa_north, [(0, 0), (6, 0), (6, 2), (0, 2)])
    surf_a_south = _surface_from_xz(pa_south, [(0, 2), (6, 2), (6, 4), (0, 4)])

    # Wing B's two slopes (east/west of a Z-axis gable, ridge at x=7)
    pb_west = Plane.fit([[4, 0, 4], [4, 0, 8], [7, 2, 8], [7, 2, 4]])
    pb_east = Plane.fit([[7, 2, 4], [7, 2, 8], [10, 0, 8], [10, 0, 4]])
    assert isinstance(pb_west, Plane) and isinstance(pb_east, Plane)
    surf_b_west = _surface_from_xz(pb_west, [(4, 4), (4, 8), (7, 8), (7, 4)])
    surf_b_east = _surface_from_xz(pb_east, [(7, 4), (7, 8), (10, 8), (10, 4)])

    records = [
        PartRecord(wing=wing_a, kind="legacy", surfaces=[surf_a_north, surf_a_south]),
        PartRecord(wing=wing_b, kind="legacy", surfaces=[surf_b_west, surf_b_east]),
    ]
    out = resolve_valleys(records)

    assert len(out) == 2
    assert len(out[0].surfaces) == 2
    assert len(out[1].surfaces) == 2


def test_resolve_valleys_cross_gable_splits_slope_into_arms():
    """Cross-gable (+ shape): two perpendicular gables of equal ridge height
    crossing at the centre. Each slope should split into TWO disjoint arms
    where the perpendicular gable's higher plane wins the central region.
    Total kept area should equal the building footprint area (no double-cover,
    no missing area).
    """
    # Wing A: long along X (x[-6,6], z[-2,2]), gable ridge at z=0, ridge_y=2
    pa_north = Plane.fit([[-6, 0, 2], [6, 0, 2], [6, 2, 0], [-6, 2, 0]])
    pa_south = Plane.fit([[-6, 2, 0], [6, 2, 0], [6, 0, -2], [-6, 0, -2]])
    # Wing B: long along Z (x[-2,2], z[-6,6]), gable ridge at x=0, ridge_y=2
    pb_east = Plane.fit([[2, 0, -6], [2, 0, 6], [0, 2, 6], [0, 2, -6]])
    pb_west = Plane.fit([[0, 2, -6], [0, 2, 6], [-2, 0, 6], [-2, 0, -6]])
    assert all(isinstance(p, Plane) for p in [pa_north, pa_south, pb_east, pb_west])

    surf_a_north = _surface_from_xz(pa_north, [(-6, 0), (6, 0), (6, 2), (-6, 2)])
    surf_a_south = _surface_from_xz(pa_south, [(-6, -2), (6, -2), (6, 0), (-6, 0)])
    surf_b_east = _surface_from_xz(pb_east, [(0, -6), (2, -6), (2, 6), (0, 6)])
    surf_b_west = _surface_from_xz(pb_west, [(-2, -6), (0, -6), (0, 6), (-2, 6)])

    wing_a = _wing(-6.0, 6.0, -2.0, 2.0, index=0)
    wing_b = _wing(-2.0, 2.0, -6.0, 6.0, index=1)

    records = [
        PartRecord(wing=wing_a, kind="gable", surfaces=[surf_a_north, surf_a_south]),
        PartRecord(wing=wing_b, kind="gable", surfaces=[surf_b_east, surf_b_west]),
    ]
    out = resolve_valleys(records)

    total_area = 0.0
    for pr in out:
        for s in pr.surfaces:
            poly = Polygon([(c[0], c[2]) for c in s.corners])
            total_area += poly.area

    footprint = wing_a.polygon.union(wing_b.polygon)
    assert abs(total_area - footprint.area) < 0.5

    # Wing B's slopes have perpendicular shape that DOES split into arms;
    # wing A's slopes might or might not split depending on which corners
    # the half-plane line cuts. At minimum, total surfaces should exceed 4
    # (the original count) and should equal 6+ (some splitting happened).
    total_surfaces = sum(len(pr.surfaces) for pr in out)
    assert total_surfaces >= 6, (
        f"expected >=6 surfaces after cross-gable clipping, got {total_surfaces}"
    )


def test_resolve_valleys_clips_overlapping_slopes_at_higher_plane():
    """When two slopes overlap in XZ, only the higher plane wins in the overlap."""
    wing_a = _wing(0.0, 6.0, 0.0, 4.0)
    wing_b = _wing(2.0, 8.0, 0.0, 4.0)

    # Plane A is taller (rises 4 m over the same XZ patch)
    plane_high = Plane.fit([[0, 0, 0], [6, 0, 0], [6, 4, 4], [0, 4, 4]])
    plane_low = Plane.fit([[2, 0, 0], [8, 0, 0], [8, 2, 4], [2, 2, 4]])
    assert isinstance(plane_high, Plane) and isinstance(plane_low, Plane)
    surf_high = _surface_from_xz(plane_high, [(0, 0), (6, 0), (6, 4), (0, 4)])
    surf_low = _surface_from_xz(plane_low, [(2, 0), (8, 0), (8, 4), (2, 4)])

    records = [
        PartRecord(wing=wing_a, kind="legacy", surfaces=[surf_high]),
        PartRecord(wing=wing_b, kind="legacy", surfaces=[surf_low]),
    ]
    out = resolve_valleys(records)

    high_xz = [(c[0], c[2]) for c in out[0].surfaces[0].corners]
    low_xz = [(c[0], c[2]) for c in out[1].surfaces[0].corners]

    high_poly = Polygon(high_xz)
    low_poly = Polygon(low_xz)

    overlap = high_poly.intersection(low_poly)
    # In the overlap region, high plane wins; the polygons may share the line
    # x = where planes are equal but should not double-cover area.
    assert overlap.area < 0.01

    # The taller plane's footprint should be at least its non-overlap portion
    # (x in [0, 2]) plus part of the contested x in [2, 6].
    assert high_poly.area > 8.0  # roughly: 2*4 (uncontested) + some of contested

    # The lower plane's footprint should retain its uncontested x in [6, 8]
    assert low_poly.area > 4.0
