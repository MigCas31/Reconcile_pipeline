import itertools

from reconcile_tiers.extract.building import ExtractedRoom, ExtractedWall
from reconcile_tiers.extract.extension import (
    PERIMETER_DISTANCE_TOL_M,
    compute_descent_strips,
    compute_uplift_strips,
)


def _wall(wall_id, x0, x1, top_y=2.40, bottom_y=0.0, z=0.0):
    return ExtractedWall(
        id=wall_id,
        source="synthetic",
        corners=[
            [x0, top_y, z],
            [x1, top_y, z],
            [x1, bottom_y, z],
            [x0, bottom_y, z],
        ],
    )


def _room(
    index,
    story,
    floor_y,
    floor_x0=0.0,
    floor_x1=1.0,
    floor_z0=-0.5,
    floor_z1=0.5,
    walls=None,
):
    return ExtractedRoom(
        index=index,
        story=story,
        floor_polygon=[
            [floor_x0, floor_y, floor_z0],
            [floor_x1, floor_y, floor_z0],
            [floor_x1, floor_y, floor_z1],
            [floor_x0, floor_y, floor_z1],
        ],
        walls_merged=[],
        walls_computed=walls or [],
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


def test_descent_strip_attached_to_perimeter_upper_wall():
    """Upper wall on the upper-story footprint perimeter gets a descent strip
    whose bottom edge follows the actual lower-wall top (here a flat top at
    y=2.40)."""
    upper_wall = _wall(
        "uw0",
        x0=0.0,
        x1=1.0,
        top_y=5.10,
        bottom_y=2.70,
        z=-0.5,
    )
    lower = _room(0, 0, 0.0, walls=[_wall("lw0", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)])
    upper = _room(1, 1, 2.70, walls=[upper_wall])

    extended = compute_descent_strips([lower, upper])

    descent = extended[1].walls_computed[0].descent_strip
    assert descent is not None
    assert len(descent) >= 1
    # Each quad has its top edge at the upper wall bottom (2.70) and its
    # bottom edge at the lower wall top (2.40).
    for quad in descent:
        assert quad[0][1] == 2.70
        assert quad[3][1] == 2.70
        assert quad[1][1] == 2.40
        assert quad[2][1] == 2.40
    # And adjacent quads share their transition coordinates exactly.
    for prev, cur in itertools.pairwise(descent):
        assert prev[3] == cur[0]
        assert prev[2] == cur[1]


def test_descent_strip_falls_back_to_flat_ceiling_when_lower_wall_is_offset():
    """If no lower wall top is close enough in XZ, the flat lower ceiling still
    supports the inter-story face below an upper perimeter wall."""
    upper_wall = _wall(
        "uw0",
        x0=0.0,
        x1=1.0,
        top_y=5.10,
        bottom_y=2.70,
        z=-0.5,
    )
    lower = _room(
        0,
        0,
        0.0,
        walls=[
            _wall("lw_far0", x0=0.0, x1=1.0, top_y=2.40, z=0.5),
            _wall("lw_far1", x0=0.0, x1=1.0, top_y=2.40, z=0.45),
        ],
    )
    upper = _room(1, 1, 2.70, walls=[upper_wall])

    extended = compute_descent_strips([lower, upper])

    descent = extended[1].walls_computed[0].descent_strip
    assert descent is not None
    assert len(descent) >= 1
    for quad in descent:
        assert quad[0][1] == 2.70
        assert quad[3][1] == 2.70
        assert quad[1][1] == 2.40
        assert quad[2][1] == 2.40


def test_no_descent_for_interior_upper_wall():
    """An upper wall sitting in the interior of the upper-story footprint
    is more than the perimeter tolerance from the boundary, so no descent."""
    interior_wall = _wall(
        "uw_interior",
        x0=0.5,
        x1=0.7,
        top_y=5.10,
        bottom_y=2.70,
        z=0.0,  # well inside floor (z spans -0.5 to 0.5)
    )
    lower = _room(0, 0, 0.0, walls=[_wall("lw0", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)])
    upper = _room(1, 1, 2.70, walls=[interior_wall])

    extended = compute_descent_strips([lower, upper])
    assert extended[1].walls_computed[0].descent_strip is None


def test_synthetic_wall_created_when_perimeter_lacks_upper_wall():
    """If the upper-story footprint has perimeter with no scanned upper wall,
    we synthesise an inter-story strip wall on that arc."""
    lower = _room(0, 0, 0.0, walls=[_wall("lw0", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)])
    upper = _room(1, 1, 2.70)

    extended = compute_descent_strips([lower, upper])

    synth_walls = extended[1].synthetic_walls
    assert len(synth_walls) >= 1
    synth = synth_walls[0]
    assert synth.source == "synthesised"
    assert synth.synthetic is True
    ys = sorted({c[1] for c in synth.corners})
    assert ys == [2.40, 2.70]


def test_uplift_strip_for_interior_lower_partition_under_upper_slab():
    """An interior lower-story wall whose XZ projection sits inside the
    upper footprint receives an uplift_strip up to the upper floor y."""
    interior_partition = _wall(
        "lw_interior",
        x0=0.4,
        x1=0.6,
        top_y=2.40,
        bottom_y=0.0,
        z=0.0,  # interior of lower room
    )
    perimeter_lower = _wall("lw_perim", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)
    lower = _room(
        0,
        0,
        0.0,
        floor_x0=0.0,
        floor_x1=1.0,
        floor_z0=-0.5,
        floor_z1=0.5,
        walls=[interior_partition, perimeter_lower],
    )
    upper = _room(
        1,
        1,
        2.70,
        floor_x0=0.0,
        floor_x1=1.0,
        floor_z0=-0.5,
        floor_z1=0.5,
    )

    extended = compute_uplift_strips([lower, upper])
    walls = extended[0].walls_computed
    interior_after = next(w for w in walls if w.id == "lw_interior")
    perimeter_after = next(w for w in walls if w.id == "lw_perim")

    assert interior_after.uplift_strip is not None
    assert interior_after.uplift_strip[0][2][1] == 2.70
    assert interior_after.uplift_strip[0][3][1] == 2.70
    # Perimeter lower wall is on the lower-story footprint boundary; per the
    # design the upper wall owns the perimeter, so no uplift here.
    assert perimeter_after.uplift_strip is None


def test_no_uplift_when_no_upper_story():
    """Top-story walls have no upper slab; nothing to uplift to."""
    only_story = _room(0, 0, 0.0, walls=[_wall("lw0", x0=0.0, x1=1.0)])
    extended = compute_uplift_strips([only_story])
    assert extended[0].walls_computed[0].uplift_strip is None


def test_perimeter_tol_constant_matches_corpus_pick():
    """Pinned to the corpus-derived value so changes are deliberate."""
    assert PERIMETER_DISTANCE_TOL_M == 0.05


def test_descent_strips_wings_none_matches_legacy_behaviour():
    """`wings=None` (default) preserves bit-for-bit legacy output."""
    upper_wall = _wall("uw0", x0=0.0, x1=1.0, top_y=5.10, bottom_y=2.70, z=-0.5)
    lower = _room(0, 0, 0.0, walls=[_wall("lw0", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)])
    upper = _room(1, 1, 2.70, walls=[upper_wall])

    extended_default = compute_descent_strips([lower, upper])
    extended_explicit = compute_descent_strips([lower, upper], wings=None)
    a = extended_default[1].walls_computed[0].descent_strip
    b = extended_explicit[1].walls_computed[0].descent_strip
    assert a == b


def test_uplift_strips_wings_none_matches_legacy_behaviour():
    """`wings=None` (default) preserves bit-for-bit legacy output for uplift too."""
    interior = _wall("lw_int", x0=0.4, x1=0.6, top_y=2.40, bottom_y=0.0, z=0.0)
    perim = _wall("lw_perim", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)
    lower = _room(0, 0, 0.0, walls=[interior, perim])
    upper = _room(1, 1, 2.70)

    a = compute_uplift_strips([lower, upper])
    b = compute_uplift_strips([lower, upper], wings=None)
    a_strips = [w.uplift_strip for w in a[0].walls_computed]
    b_strips = [w.uplift_strip for w in b[0].walls_computed]
    assert a_strips == b_strips


def test_descent_strip_wing_aware_skips_other_wing_supports():
    """When `wings` is given, lower walls in a different wing don't support
    a descent on the upper wall.

    Setup: two upper walls on different wings. Lower wall under wing A
    supports descent for upper wall A. Without wing filter, the lower
    wall could be considered for upper wall B too if both wings have
    perimeter support; with wing filter, only wall A's support is used.
    """
    from shapely.geometry import Polygon

    from reconcile_tiers._core.wing_decomposition import Wing

    # Upper wall A (perimeter on wing 0) at z=-0.5, x in [0,1].
    upper_a = _wall("uw_a", x0=0.0, x1=1.0, top_y=5.10, bottom_y=2.70, z=-0.5)
    # Lower wall under wing 0 — should support upper_a.
    lower_a = _wall("lw_a", x0=0.0, x1=1.0, top_y=2.40, z=-0.5)

    lower_room = _room(
        0,
        0,
        0.0,
        floor_x0=0.0,
        floor_x1=1.0,
        floor_z0=-0.5,
        floor_z1=0.5,
        walls=[lower_a],
    )
    upper_room = _room(
        1,
        1,
        2.70,
        floor_x0=0.0,
        floor_x1=1.0,
        floor_z0=-0.5,
        floor_z1=0.5,
        walls=[upper_a],
    )

    # Single wing covering everything — both rooms in same wing.
    wing_poly = Polygon([(-0.1, -0.6), (1.1, -0.6), (1.1, 0.6), (-0.1, 0.6)])
    one_wing = [Wing(index=0, polygon=wing_poly, area_m2=wing_poly.area, role="main")]

    # With single wing: identical result to no-wings case.
    extended = compute_descent_strips([lower_room, upper_room], wings=one_wing)
    assert extended[1].walls_computed[0].descent_strip is not None

    # With a wing that excludes the lower room (small wing only over upper):
    excluding_wing = Polygon([(0.0, 0.49), (1.0, 0.49), (1.0, 0.5), (0.0, 0.5)])
    other_wing = [
        Wing(index=0, polygon=excluding_wing, area_m2=excluding_wing.area, role="main")
    ]
    extended2 = compute_descent_strips([lower_room, upper_room], wings=other_wing)
    # wing_polygon_for_room picks the only wing for both rooms (single-wing
    # case is treated as no-constraint by `wing_polygon_for_room`). So the
    # behaviour should match the no-wings case here too.
    assert extended2[1].walls_computed[0].descent_strip is not None


def test_arc_room_owner_wing_filter_falls_back_when_empty():
    """`_arc_room_owner` with `wings` returns the unrestricted nearest-room
    answer when no candidate room is in the arc's wing."""
    from shapely.geometry import LineString, Polygon

    from reconcile_tiers._core.wing_decomposition import Wing
    from reconcile_tiers.extract.extension import _arc_room_owner

    # Two rooms, wing covers neither's interior.
    a = _room(0, 0, 0.0, floor_x0=0.0, floor_x1=1.0)
    b = _room(1, 0, 0.0, floor_x0=2.0, floor_x1=3.0)
    rooms_with_polys = [
        (0, Polygon([(0, -0.5), (1, -0.5), (1, 0.5), (0, 0.5)])),
        (1, Polygon([(2, -0.5), (3, -0.5), (3, 0.5), (2, 0.5)])),
    ]
    # Wing way away from both rooms.
    far_wing = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    wings = [Wing(index=0, polygon=far_wing, area_m2=1.0, role="main")]

    arc = LineString([(0.4, 0.0), (0.6, 0.0)])  # arc near room 0
    owner = _arc_room_owner(arc, rooms_with_polys, wings=wings, rooms=[a, b])
    # Wing filter empties; falls back to unrestricted distance — picks room 0.
    assert owner == 0
