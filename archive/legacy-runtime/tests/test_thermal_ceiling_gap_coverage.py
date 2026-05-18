"""Tests for widened thermal cap footprint and wall barrier subtraction.

Covers the fix where ridge/eave caps now follow the building envelope under
the roof (room fp U within-story gap polygons) and get split by walls whose
top Y reaches the cap plane within BARRIER_REACH.
"""

from __future__ import annotations

from shapely.geometry import Point, Polygon

from reconcile.roof_algorithms_py.thermal_ceiling import (
    BARRIER_REACH,
    _build_story_thermal_context,
    _build_thermal_for_oblique_room,
)


def _rect_xz(x0: float, z0: float, x1: float, z1: float, y: float):
    return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]


def _wall_quad(x0: float, z0: float, x1: float, z1: float, y_bot: float, y_top: float):
    """A thin vertical wall quad (two end-points, bottom and top)."""
    return [
        [x0, y_bot, z0],
        [x1, y_bot, z1],
        [x1, y_top, z1],
        [x0, y_top, z0],
    ]


def _sloped_plane_rising_in_x(ref_x=0.0, ref_y=2.0, ref_z=0.0, slope=0.25):
    """Plane with y = ref_y + slope * (x - ref_x). Normal points up."""
    # From plane_y_at: y = ref_y - (n_x*(x-ref_x) + n_z*(z-ref_z)) / n_y
    # We want y = ref_y + slope*(x - ref_x), so n_x/n_y = -slope, n_z = 0.
    return {
        "n": {"x": -slope, "y": 1.0, "z": 0.0},
        "ref": {"x": ref_x, "y": ref_y, "z": ref_z},
        "dominantStory": 0,
    }


def _oblique_surface_covering(room_fp):
    xs = [p[0] for p in room_fp]
    zs = [p[2] for p in room_fp]
    return {
        "corners": _rect_xz(min(xs), min(zs), max(xs), max(zs), 3.0),
        "dominant_story": 0,
    }


def _room(room_index: int, fp, *, wall_top_y: float, wall_top_min: float, walls=None):
    xs = [p[0] for p in fp]
    zs = [p[2] for p in fp]
    return {
        "room": {"walls_merged": [], "walls_computed": walls or [], "story": 0},
        "room_index": room_index,
        "fp": fp,
        "fcx": sum(xs) / len(xs),
        "fcz": sum(zs) / len(zs),
        "floorY": 0.0,
        "wallTopY": wall_top_y,
        "wallTopMin": wall_top_min,
        "wallTopYs": [wall_top_min, wall_top_y],
        "story": 0,
    }


def _thermal_caps(caps):
    return [s for s in caps if s["kind"] == "thermal-cap"]


def _union_of(polys_2d):
    from shapely.ops import unary_union

    return unary_union([Polygon(p) for p in polys_2d])


def test_eave_cap_covers_within_story_gap_assigned_to_room():
    # Room A [0,0]-[4,4]. Gap [4,0]-[5,4] attached to room 0. Plane rises in x.
    fp = _rect_xz(0, 0, 4, 4, 0.0)
    room = _room(0, fp, wall_top_y=3.0, wall_top_min=2.0)
    plane = _sloped_plane_rising_in_x(ref_x=0, ref_y=2.0, slope=0.25)
    surface = _oblique_surface_covering(fp)
    gap_corners = [[4, 0.0, 0], [5, 0.0, 0], [5, 0.0, 4], [4, 0.0, 4]]
    rooms_payload = [{"story": 0, "walls_merged": [], "walls_computed": []}]
    gap_walls = [
        {"type": "within_story", "story": 0, "room_index": 0, "corners": gap_corners}
    ]
    ctx = _build_story_thermal_context([room], gap_walls, rooms_payload)

    caps = _build_thermal_for_oblique_room(
        room, [surface], [0], [plane], story_context=ctx[0]
    )

    cap_entries = _thermal_caps(caps)
    assert cap_entries, "expected at least one thermal-cap to cover the eave gap"
    cap_union = _union_of([[(p[0], p[2]) for p in e["poly"]] for e in cap_entries])
    # Gap centroid must be covered.
    assert cap_union.contains(Point(4.5, 2.0))


# Shared barrier-test setup: plane y = 2.0 + 0.25*x over fp [0,0]-[4,4].
# wallTopMin=2.5 puts the eave-cap plane where the slant's rendered extent is
# x in [2, 4], leaving a non-empty eave cap over x in [0, 2]. A probe wall at
# x=1 sits inside that eave-cap band.
_PROBE_WALL_X = 1.0
_EAVE_CAP_Y = 2.5


def _barrier_setup(wall_top_y: float):
    fp = _rect_xz(0, 0, 4, 4, 0.0)
    room = _room(0, fp, wall_top_y=3.0, wall_top_min=_EAVE_CAP_Y)
    plane = _sloped_plane_rising_in_x(ref_x=0, ref_y=2.0, slope=0.25)
    surface = _oblique_surface_covering(fp)
    wall = {
        "corners": _wall_quad(
            _PROBE_WALL_X, 0.0, _PROBE_WALL_X, 4.0, y_bot=0.0, y_top=wall_top_y
        )
    }
    rooms_payload = [{"story": 0, "walls_merged": [], "walls_computed": [wall]}]
    ctx = _build_story_thermal_context([room], [], rooms_payload)
    caps = _build_thermal_for_oblique_room(
        room, [surface], [0], [plane], story_context=ctx[0]
    )
    return _thermal_caps(caps)


def _probe_points(cap_entries, *, z: float = 2.0):
    cap_polys = [Polygon([(p[0], p[2]) for p in e["poly"]]) for e in cap_entries]
    return {
        "on_wall": any(p.contains(Point(_PROBE_WALL_X, z)) for p in cap_polys),
        "left_in": any(p.contains(Point(_PROBE_WALL_X - 0.5, z)) for p in cap_polys),
        "right_in": any(p.contains(Point(_PROBE_WALL_X + 0.5, z)) for p in cap_polys),
    }


def test_tall_interior_wall_splits_eave_cap():
    cap_entries = _barrier_setup(wall_top_y=_EAVE_CAP_Y)  # exactly at cap
    probes = _probe_points(cap_entries)
    assert not probes["on_wall"], "cap must not cross the wall strip"
    assert probes["left_in"] and probes["right_in"], "cap must remain on both sides"


def test_short_wall_does_not_split_eave_cap():
    short_top = _EAVE_CAP_Y - BARRIER_REACH - 0.10
    cap_entries = _barrier_setup(wall_top_y=short_top)
    probes = _probe_points(cap_entries)
    assert probes["on_wall"], "short wall (below REACH) must not split the cap"
    assert probes["left_in"] and probes["right_in"]


def test_almost_reaching_wall_splits_eave_cap():
    almost_top = _EAVE_CAP_Y - 0.20  # within REACH (0.30)
    cap_entries = _barrier_setup(wall_top_y=almost_top)
    probes = _probe_points(cap_entries)
    assert not probes["on_wall"], (
        "almost-reaching wall (within REACH) must split the cap"
    )
    assert probes["left_in"] and probes["right_in"]


def test_wall_above_cap_splits_eave_cap():
    above_top = _EAVE_CAP_Y + 0.40
    cap_entries = _barrier_setup(wall_top_y=above_top)
    probes = _probe_points(cap_entries)
    assert not probes["on_wall"], "wall extending above cap plane must split the cap"
    assert probes["left_in"] and probes["right_in"]
