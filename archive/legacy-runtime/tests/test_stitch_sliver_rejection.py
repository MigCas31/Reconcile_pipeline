"""Sliver-leg rejection in ``stitch_wall_gaps``.

A stitch leg whose XZ length is below ``_MIN_LEG_LENGTH_M`` renders as
a bow-tie / sliver flap in the viewer. Branch-level gates must drop
such legs at emission time, and the post-pass ``_drop_degenerate_stitches``
must catch any that slip through due to downstream corner displacement.
"""

import math

from reconcile.extract3d import stitch


def _wall_from_endpoints(p0_xz, p1_xz, y_bot=0.0, y_top=2.4):
    x0, z0 = p0_xz
    x1, z1 = p1_xz
    return {
        "corners": [
            [x0, y_bot, z0],
            [x1, y_bot, z1],
            [x1, y_top, z1],
            [x0, y_top, z0],
        ],
        "extension_strip": [],
    }


def _vertical_stitches(stitches):
    return [s for s in stitches if s.get("type") == "stitch"]


def test_near_collinear_walls_drop_sub_threshold_perp_leg():
    # Wall A along +X from (0,0) to (5,0). Wall B also along +X from
    # (5.2, 0.04) to (10.2, 0.04) — parallel, 4 cm perpendicular offset.
    # Pair endpoints A end1=(5,0) and B end0=(5.2, 0.04), gap ~0.201 m.
    # Anchored on either wall, parallel_len ~0.2 m, perp_len ~0.04 m.
    # The 4 cm perp leg is below _MIN_LEG_LENGTH_M (6 cm) and must be
    # dropped, leaving only the parallel leg.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (5.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((5.2, 0.04), (10.2, 0.04)),
            ],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    legs = _vertical_stitches(out)
    assert len(legs) == 1, (
        f"expected sub-threshold perp leg to be dropped, got {len(legs)} legs"
    )
    dx = legs[0]["corners"][1][0] - legs[0]["corners"][0][0]
    dz = legs[0]["corners"][1][2] - legs[0]["corners"][0][2]
    length = math.hypot(dx, dz)
    assert length >= stitch._MIN_LEG_LENGTH_M


def test_well_sized_legs_still_emitted():
    # Wall A along +X (0,0)→(5,0). Wall B along +X (5.5, 0.3)→(10.5, 0.3).
    # Gap 0.583 m; parallel_len 0.5, perp_len 0.3 — both comfortably
    # above the 6 cm threshold. Both legs must still be emitted.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (5.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((5.5, 0.3), (10.5, 0.3)),
            ],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    legs = _vertical_stitches(out)
    assert len(legs) == 2, f"expected two healthy legs, got {len(legs)}"
    for leg in legs:
        dx = leg["corners"][1][0] - leg["corners"][0][0]
        dz = leg["corners"][1][2] - leg["corners"][0][2]
        assert math.hypot(dx, dz) >= stitch._MIN_LEG_LENGTH_M


def test_drop_degenerate_catches_sub_threshold_xz_base():
    # Hand-crafted entry with a 4 vertices that are all unique in 3D
    # (vertical extrusion 2.4 m) but whose XZ footprint is only 5 mm
    # wide. Must be dropped by the post-pass check on the XZ bbox
    # diagonal.
    sliver = {
        "type": "stitch",
        "story": 0,
        "room_indices": [0, 1],
        "corners": [
            [0.000, 0.0, 0.000],
            [0.003, 0.0, 0.004],
            [0.003, 2.4, 0.004],
            [0.000, 2.4, 0.000],
        ],
    }
    healthy = {
        "type": "stitch",
        "story": 0,
        "room_indices": [0, 1],
        "corners": [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 2.4, 0.0],
            [0.0, 2.4, 0.0],
        ],
    }

    kept = stitch._drop_degenerate_stitches([sliver, healthy])
    assert len(kept) == 1
    assert kept[0] is healthy
