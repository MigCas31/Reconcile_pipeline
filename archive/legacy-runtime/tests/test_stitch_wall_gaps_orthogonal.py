"""Orthogonal L-closure behaviour for `stitch_wall_gaps`.

The stitcher must never emit a wall whose in-plane direction is at an
oblique (non-right-angle) angle to either parent wall. When the parent
walls meet at something other than 90 deg, the closure is an L anchored
on one of the two wall axes (chosen to minimise total leg length).
"""

import math

from reconcile.extract3d import stitch


def _wall(corners):
    return {"corners": corners, "extension_strip": []}


def _wall_from_endpoints(p0_xz, p1_xz, y_bot=0.0, y_top=2.4):
    x0, z0 = p0_xz
    x1, z1 = p1_xz
    return _wall(
        [
            [x0, y_bot, z0],
            [x1, y_bot, z1],
            [x1, y_top, z1],
            [x0, y_top, z0],
        ]
    )


def _vertical_stitches(stitches):
    return [s for s in stitches if s.get("type") == "stitch"]


def _xz_dir(entry):
    corners = entry["corners"]
    dx = corners[1][0] - corners[0][0]
    dz = corners[1][2] - corners[0][2]
    length = math.hypot(dx, dz)
    assert length > 1e-4
    return (dx / length, dz / length)


def test_oblique_parent_walls_produce_orthogonal_l():
    # Wall A along +X from (0,0) to (1,0).
    # Wall B along the 45 deg diagonal starting at (2, 0.3).
    # Pair endpoints: A end1 = (1, 0), B end0 = (2, 0.3). Gap ~1.04 m.
    # With the old ray-intersection path, the two stitch legs would meet
    # at 45 deg (an oblique V). With the new path, legs must be at 90 deg.
    diag = math.sqrt(2) / 2
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (1.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((2.0, 0.3), (2.0 + diag, 0.3 + diag)),
            ],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    legs = _vertical_stitches(out)
    assert len(legs) == 2, f"expected two L legs, got {len(legs)}"

    d0 = _xz_dir(legs[0])
    d1 = _xz_dir(legs[1])
    dot = abs(d0[0] * d1[0] + d0[1] * d1[1])
    assert dot < 1e-3, f"legs not perpendicular: dot={dot:.4f}"

    # One leg must be parallel to wall A's axis (+/- X); the anchor choice
    # minimises total leg length, which picks wall A here.
    parallel_to_x = [d for d in (d0, d1) if abs(d[1]) < 1e-3]
    parallel_to_z = [d for d in (d0, d1) if abs(d[0]) < 1e-3]
    assert len(parallel_to_x) == 1 and len(parallel_to_z) == 1


def test_near_perpendicular_parents_collapse_to_single_leg():
    # Wall A along +X from (0,0) to (5,0). Wall B along +Z from (5.3, 0)
    # to (5.3, 5). Only the (5,0)<->(5.3,0) endpoint pair is within
    # max_gap (1.5 m); the other endpoints are >5 m away. The projection
    # along wall A's axis covers the full gap (parallel_len = 0.3,
    # perp_len = 0), so only one L leg survives.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (5.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((5.3, 0.0), (5.3, 5.0))],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    legs = _vertical_stitches(out)
    assert len(legs) == 1, f"expected a single collapsed leg, got {len(legs)}"
    d = _xz_dir(legs[0])
    assert abs(d[1]) < 1e-3, f"single leg should lie on X axis, dir={d}"


def test_inward_endpoint_is_not_stitched():
    # Wall A along +X from (0,0) to (5,0). Wall B along +Z from (2, 0.3)
    # to (2, 5). The paired endpoints are A end1 = (5,0) and B end0 =
    # (2, 0.3) at distance ~2.82 m -- outside max_gap=1.5. Use the other
    # ends instead: A end0 = (0,0) and B end0 = (2, 0.3) at ~2 m -- also
    # outside. Bring B closer so a pair falls inside max_gap but the
    # L corner would need to be on wall B's inward side.
    #
    # Setup that forces "inward" rejection: Wall A end1 = (1, 0) with
    # u1_out = +X. Wall B endpoint at (0.5, 0.3) with its +Z axis such
    # that u2_out = -Z. The natural corner sits at (0.5, 0) which is
    # BEHIND wall A's outward extension from (1,0) [along_A = -0.5 < 0].
    # Both candidates return None -> no stitch emitted.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (1.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.5, 0.3), (0.5, 1.3))],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    legs = _vertical_stitches(out)
    assert legs == [], f"expected no stitch for inward-corner pair, got {len(legs)}"
