"""Regression tests for exterior-gap closures on walls with kinked tops.

The closure synthesiser previously collapsed the wall's upper outline to a
single flat Y by Y-mid-splitting the corners. For attic walls bounded by a
sloped roof eave, the kink corner (Y below mid_y but topologically on the
upper outline) was discarded, so the closure ceiling and side quads capped at
the ridge Y over regions where the physical wall topped out at the kink Y.

These tests pin the new behaviour: the closure follows the wall's actual
top polyline (kinks included).
"""

import numpy as np

from reconcile.extract_3d import (
    _compute_gap_closures,
    _interp_profile_y,
    _wall_top_bottom_profiles,
)


def _axis_origin_from_bottom_edge(corners, bot_left_idx, bot_right_idx):
    bl = corners[bot_left_idx]
    br = corners[bot_right_idx]
    v = np.array([br[0] - bl[0], br[2] - bl[2]])
    L = float(np.linalg.norm(v))
    return v / L, (float(bl[0]), float(bl[2])), L


def test_profiles_rectangular_wall_round_trip():
    rect = [
        [0.0, -1.0, 0.0],
        [3.0, -1.0, 0.0],
        [3.0, +1.0, 0.0],
        [0.0, +1.0, 0.0],
    ]
    axis2d = np.array([1.0, 0.0])
    top, bot = _wall_top_bottom_profiles(rect, axis2d, (0.0, 0.0))

    assert len(top) == 2 and len(bot) == 2
    assert top[0] == (0.0, 1.0) and top[1] == (3.0, 1.0)
    assert bot[0] == (0.0, -1.0) and bot[1] == (3.0, -1.0)


def test_profiles_kinked_attic_wall_preserves_kink():
    # Five-corner wall from building 016980bc story 2 (the case the user
    # flagged in the viewer): top is a polyline kink->ridge->ridge.
    wc = [
        [-1.5151969557136193, -1.1702169857995797, -4.050988751486993],  # kink
        [-0.28126173172983676, 0.898123630099976, -2.31664515973179],  # ridge inner
        [1.23093817735651, 0.898123630099976, -0.1911898110228418],  # ridge right
        [1.23093817735651, -1.4318766301, -0.1911898110228418],  # floor right
        [-1.5151969557136193, -1.4318766301, -4.050988751486993],  # floor left
    ]
    axis2d, origin_xz, L = _axis_origin_from_bottom_edge(wc, 4, 3)

    top, bot = _wall_top_bottom_profiles(wc, axis2d, origin_xz)

    # Top profile follows the kink: three points, Y rising from -1.17 to +0.90.
    assert len(top) == 3
    assert top[0][1] == -1.1702169857995797
    assert top[-1][1] == 0.898123630099976
    # Bottom profile is flat at the floor Y.
    assert len(bot) == 2
    assert bot[0][1] == -1.4318766301
    assert bot[-1][1] == -1.4318766301

    # Interp at the left end gives the kink, NOT the ridge.
    y_left_top = _interp_profile_y(top, 0.0)
    y_right_top = _interp_profile_y(top, L)
    assert abs(y_left_top - (-1.1702169857995797)) < 1e-9
    assert abs(y_right_top - 0.898123630099976) < 1e-9


def _kinked_indicator_and_rooms():
    """Synthetic case mirroring story 2 of building 016980bc.

    Parallel attic wall has a kinked top (5 corners). Door's parent wall is
    a plain 4-corner rectangle, parallel to the attic wall and offset 1 m
    perpendicular along the door side, spanning the full attic-wall length
    along the shared axis. Both walls share floor Y=-1.432 and ridge Y=+0.898.
    """
    door_id = "DOOR-1"
    parent_wall_id = "WALL-PARENT"
    parallel_wall_id = "WALL-PARALLEL"

    # Wall axis from the kinked wall's bottom edge.
    bl = np.array([-1.5151969557136193, -4.050988751486993])
    br = np.array([+1.23093817735651, -0.1911898110228418])
    axis = (br - bl) / np.linalg.norm(br - bl)  # (0.580, 0.815)
    perp = np.array([-axis[1], axis[0]])  # (-0.815, 0.580), points to door side

    floor_y = -1.4318766301
    ridge_y = 0.898123630099976

    # Parent wall: parallel to kinked wall, offset 1.0 m perpendicular,
    # same length so the overlap spans the entire kinked-wall axis.
    p_bl_xz = bl + perp * 1.0
    p_br_xz = br + perp * 1.0
    parent_corners = [
        [float(p_bl_xz[0]), floor_y, float(p_bl_xz[1])],
        [float(p_br_xz[0]), floor_y, float(p_br_xz[1])],
        [float(p_br_xz[0]), ridge_y, float(p_br_xz[1])],
        [float(p_bl_xz[0]), ridge_y, float(p_bl_xz[1])],
    ]

    # Door: 0.7 m wide, centred on the parent wall, 2.08 m tall.
    mid_xz = (p_bl_xz + p_br_xz) / 2.0
    half_w = 0.35
    d_left_xz = mid_xz - axis * half_w
    d_right_xz = mid_xz + axis * half_w
    door_top_y = 0.65
    door_corners = [
        [float(d_left_xz[0]), floor_y, float(d_left_xz[1])],
        [float(d_right_xz[0]), floor_y, float(d_right_xz[1])],
        [float(d_right_xz[0]), door_top_y, float(d_right_xz[1])],
        [float(d_left_xz[0]), door_top_y, float(d_left_xz[1])],
    ]

    # Parallel wall: 5 corners with a kink at the left top.
    parallel_corners = [
        [-1.5151969557136193, -1.1702169857995797, -4.050988751486993],
        [-0.28126173172983676, 0.898123630099976, -2.31664515973179],
        [1.23093817735651, 0.898123630099976, -0.1911898110228418],
        [1.23093817735651, -1.4318766301, -0.1911898110228418],
        [-1.5151969557136193, -1.4318766301, -4.050988751486993],
    ]

    room_with_door = {
        "story": 2,
        "walls_computed": [
            {"id": parent_wall_id, "corners": parent_corners},
        ],
        "doors": [{"id": door_id, "corners": door_corners}],
        "openings": [],
        "storages": [],
        "_parent_lookup": {door_id: parent_wall_id},
    }
    room_across_gap = {
        "story": 2,
        "walls_computed": [
            {"id": parallel_wall_id, "corners": parallel_corners},
        ],
        "doors": [],
        "openings": [],
        "storages": [],
        "_parent_lookup": {},
    }
    rooms_out = [room_with_door, room_across_gap]

    indicator = {
        "story": 2,
        "element_type": "door",
        "element_id": door_id,
        "element_corners": door_corners,
        "element_width_m": 0.7,
        "wall_id": parallel_wall_id,
        "wall_corners": parallel_corners,
        "wall_distance_m": 0.99,
        "angle_deg": 0.3,
        "confidence": "low",
    }
    return [indicator], rooms_out


def test_closure_ceiling_follows_kinked_top_not_flat_ridge():
    """Closure ceiling must drop to the kink Y at the kinked end."""
    indicators, rooms_out = _kinked_indicator_and_rooms()
    closures = _compute_gap_closures(indicators, rooms_out)

    by_type = {c["type"]: c for c in closures if c["type"] in {"floor", "ceiling"}}
    assert "ceiling" in by_type and "floor" in by_type

    ceiling = by_type["ceiling"]
    cy = sorted(c[1] for c in ceiling["corners"])
    # If the bug were still present, all four ceiling Y values would be the
    # ridge Y (~+0.90). With the fix, the kinked side drops far below — span
    # should be at least ~1 m of vertical drop.
    assert cy[-1] - cy[0] > 0.5, (
        f"ceiling closure should slope along the kinked top; got Y range {cy}"
    )
    assert cy[0] < 0.0, f"kinked end should dip below 0; got min Y {cy[0]}"
    assert cy[-1] > 0.5, f"ridge end should reach near +0.90; got max Y {cy[-1]}"

    floor = by_type["floor"]
    fy = [c[1] for c in floor["corners"]]
    # Floor should be flat-ish at the actual floor Y; previously the kink
    # bled into the floor closure and tilted it.
    assert max(fy) - min(fy) < 0.05, (
        f"floor closure should be near-flat at the real floor Y; got {fy}"
    )


def test_closure_side_on_kinked_end_caps_at_kink_y():
    indicators, rooms_out = _kinked_indicator_and_rooms()
    closures = _compute_gap_closures(indicators, rooms_out)

    sides = [c for c in closures if c["type"] == "side"]
    assert len(sides) == 2

    # The kinked end of the parallel wall is at XZ ~ (-1.515, -4.051). Pick
    # the side closure that has a corner near that XZ.
    def has_xz(side, x, z, tol=0.05):
        return any(abs(c[0] - x) < tol and abs(c[2] - z) < tol for c in side["corners"])

    kinked_side = next(s for s in sides if has_xz(s, -1.5152, -4.0510))
    # Among the two corners on the parallel-wall side of the kinked-end
    # closure, the upper one should be near the kink Y, not the ridge Y.
    parallel_corners = [
        c
        for c in kinked_side["corners"]
        if abs(c[0] - (-1.5152)) < 0.05 and abs(c[2] - (-4.0510)) < 0.05
    ]
    assert parallel_corners, (
        "expected two corners on the parallel wall at the kinked end"
    )
    top_y = max(c[1] for c in parallel_corners)
    assert abs(top_y - (-1.170)) < 0.05, (
        f"side closure top at kinked end should land on the kink "
        f"(~-1.17), not the ridge (+0.90); got {top_y}"
    )
