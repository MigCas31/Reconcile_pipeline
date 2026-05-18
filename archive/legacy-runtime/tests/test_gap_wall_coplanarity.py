from __future__ import annotations

import math

from reconcile.extract3d.gaps import compute_gap_walls
from reconcile.extract_3d import _compute_gap_walls


def _poly_area_xz(corners):
    pts = [(c[0], c[2]) for c in corners]
    if len(pts) < 3:
        return 0.0
    area2 = 0.0
    for i in range(len(pts)):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % len(pts)]
        area2 += x0 * z1 - x1 * z0
    return abs(area2) * 0.5


def _base_rooms():
    # Two parallel room walls at x=0.0 and x=0.2, both long in z.
    return [
        {
            "story": 0,
            "walls_computed": [
                {
                    "id": "w-left",
                    "corners": [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [0.0, 2.4, 3.0],
                        [0.0, 2.4, 0.0],
                    ],
                }
            ],
        },
        {
            "story": 0,
            "walls_computed": [
                {
                    "id": "w-right",
                    "corners": [
                        [0.2, 0.0, 0.0],
                        [0.2, 0.0, 3.0],
                        [0.2, 2.1, 3.0],
                        [0.2, 2.1, 0.0],
                    ],
                }
            ],
        },
    ]


def _base_gap(ontology=False):
    gap = {
        "story": 0,
        "type": "within_story",
        "confidence": "high",
        "room_index": 0,
        "centroid": [0.1, 0.0, 1.5],
        "corners": [
            [0.08, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.12, 0.0, 3.0],
            [0.08, 0.0, 3.0],
            [0.08, 0.0, 0.0],
        ],
    }
    if ontology:
        gap["_ontology_gap_id"] = "gap:test"
    return [gap]


def _long_edges_on_room_planes(walls):
    long_edges = []
    for w in walls:
        if w.get("type") != "within_story":
            continue
        c0, c1 = w["corners"][0], w["corners"][1]
        length = math.hypot(c1[0] - c0[0], c1[2] - c0[2])
        if length > 1.0:
            long_edges.append((c0, c1))
    return long_edges


def _ceiling_lookup(walls):
    caps = [w for w in walls if w.get("type") == "gap_ceiling"]
    assert caps
    lookup = {}
    for cap in caps:
        for c in cap.get("corners", []):
            lookup[(round(c[0], 6), round(c[2], 6))] = c[1]
    return lookup


def test_coplanar_long_edges_and_coherent_caps_across_paths():
    gaps_mod = _base_gap()
    gaps_legacy = _base_gap()
    rooms = _base_rooms()
    story_y_map = {0: 0.0}

    mod_walls = compute_gap_walls(gaps_mod, rooms, story_y_map)
    leg_walls = _compute_gap_walls(gaps_legacy, rooms, story_y_map)

    for walls in (mod_walls, leg_walls):
        long_edges = _long_edges_on_room_planes(walls)
        assert long_edges, "expected long within-story walls after snapping"
        for c0, c1 in long_edges:
            # Long edges should lie on either supporting room-wall plane x=0 or x=0.2
            assert abs(c0[0] - c1[0]) <= 1e-6
            assert min(abs(c0[0] - 0.0), abs(c0[0] - 0.2)) <= 1e-6

        lookup = _ceiling_lookup(walls)
        for w in walls:
            if w.get("type") != "within_story":
                continue
            top_l = w["corners"][3]
            top_r = w["corners"][2]
            yl = lookup[(round(top_l[0], 6), round(top_l[2], 6))]
            yr = lookup[(round(top_r[0], 6), round(top_r[2], 6))]
            assert abs(top_l[1] - yl) <= 1e-6
            assert abs(top_r[1] - yr) <= 1e-6


def test_extract3d_keeps_ontology_gap_id_and_ids():
    walls = compute_gap_walls(_base_gap(ontology=True), _base_rooms(), {0: 0.0})
    assert walls
    for w in walls:
        assert w.get("ontology_gap_id") == "gap:test"
        assert str(w.get("id", "")).startswith("gw:gap:test:")


def test_gap_wall_ids_are_stable_without_ontology_ids():
    mod = compute_gap_walls(_base_gap(), _base_rooms(), {0: 0.0})
    leg = _compute_gap_walls(_base_gap(), _base_rooms(), {0: 0.0})

    mod_ids = [str(w.get("id", "")) for w in mod if ":cap:" not in str(w.get("id", ""))]
    leg_ids = [str(w.get("id", "")) for w in leg]

    assert mod_ids
    assert mod_ids == leg_ids
    assert all(gap_id.startswith("gw:gap:within_story:0:") for gap_id in mod_ids)
    assert len(set(mod_ids)) == len(mod_ids)


def test_degeneracy_fallback_keeps_nonzero_gap_caps():
    rooms = [
        {
            "story": 0,
            "walls_computed": [
                {
                    "id": "single",
                    "corners": [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [0.0, 2.4, 3.0],
                        [0.0, 2.4, 0.0],
                    ],
                }
            ],
        }
    ]
    gaps = [
        {
            "story": 0,
            "type": "within_story",
            "confidence": "high",
            "room_index": 0,
            "centroid": [0.1, 0.0, 1.5],
            "corners": [
                [0.08, 0.0, 0.0],
                [0.12, 0.0, 0.0],
                [0.12, 0.0, 3.0],
                [0.08, 0.0, 3.0],
                [0.08, 0.0, 0.0],
            ],
        }
    ]
    walls = compute_gap_walls(gaps, rooms, {0: 0.0})
    floor_caps = [w for w in walls if w.get("type") == "gap_floor"]
    assert floor_caps
    assert max(_poly_area_xz(w["corners"]) for w in floor_caps) > 1e-4


def test_paths_keep_long_edge_plane_parity():
    mod = compute_gap_walls(_base_gap(), _base_rooms(), {0: 0.0})
    leg = _compute_gap_walls(_base_gap(), _base_rooms(), {0: 0.0})
    mod_x = sorted(round(c0[0], 6) for c0, _ in _long_edges_on_room_planes(mod))
    leg_x = sorted(round(c0[0], 6) for c0, _ in _long_edges_on_room_planes(leg))
    assert mod_x == leg_x


def _absorbed_rooms():
    # Simulates rooms_out after `assign_gaps_to_rooms` — each floor_polygon
    # has been expanded to swallow the wall-thickness gap. The walls_computed
    # corners still track the original scan-derived wall planes.
    base = _base_rooms()
    base[0]["floor_polygon"] = [
        [0.0, 0.0, 0.0],
        [0.12, 0.0, 0.0],
        [0.12, 0.0, 3.0],
        [0.0, 0.0, 3.0],
    ]
    base[1]["floor_polygon"] = [
        [0.08, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.2, 0.0, 3.0],
        [0.08, 0.0, 3.0],
    ]
    return base


def _original_floor_polygons():
    # Pre-absorption: each room's floor stops at the scan wall plane.
    return [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.2, 0.0, 0.0],
            [0.2, 0.0, 3.0],
            [0.2, 0.0, 3.0],
            [0.2, 0.0, 0.0],
        ],
    ]


def test_pre_absorption_snapshot_keeps_side_walls_emitting_after_absorption():
    """Regression: within-story gap polygons absorbed into a room floor must
    still produce side walls. Without the pre-absorption snapshot the
    post-absorption room union swallows the snapped polygon and the clip
    used to drop the whole gap.
    """
    rooms = _absorbed_rooms()
    mod = compute_gap_walls(
        _base_gap(),
        rooms,
        {0: 0.0},
        pre_absorption_floor_polygons=_original_floor_polygons(),
    )
    leg = _compute_gap_walls(
        _base_gap(),
        rooms,
        {0: 0.0},
        pre_absorption_floor_polygons=_original_floor_polygons(),
    )
    for walls in (mod, leg):
        sides = [w for w in walls if w.get("type") == "within_story"]
        assert sides, "expected side walls with pre-absorption snapshot"


def _sloped_ceiling_rooms():
    # Two rooms straddling the within-story gap at x ≈ 0.1, both with a flat
    # wall-top at y=2.4. Each carries a `ceiling_polygon` that slopes from
    # y=2.4 at z=0 down to y=1.4 at z=3 — eave on the +z end.
    floor_left = [
        [-1.0, 0.0, 0.0],
        [0.12, 0.0, 0.0],
        [0.12, 0.0, 3.0],
        [-1.0, 0.0, 3.0],
    ]
    floor_right = [
        [0.08, 0.0, 0.0],
        [1.12, 0.0, 0.0],
        [1.12, 0.0, 3.0],
        [0.08, 0.0, 3.0],
    ]
    ceiling_left = [
        [-1.0, 2.4, 0.0],
        [0.12, 2.4, 0.0],
        [0.12, 1.4, 3.0],
        [-1.0, 1.4, 3.0],
    ]
    ceiling_right = [
        [0.08, 2.4, 0.0],
        [1.12, 2.4, 0.0],
        [1.12, 1.4, 3.0],
        [0.08, 1.4, 3.0],
    ]
    return [
        {
            "story": 0,
            "walls_computed": [
                {
                    "id": "w-left",
                    "corners": [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [0.0, 2.4, 3.0],
                        [0.0, 2.4, 0.0],
                    ],
                }
            ],
            "floor_polygon": floor_left,
            "ceiling_polygon": ceiling_left,
            "ceiling_type": "sloped",
        },
        {
            "story": 0,
            "walls_computed": [
                {
                    "id": "w-right",
                    "corners": [
                        [0.2, 0.0, 0.0],
                        [0.2, 0.0, 3.0],
                        [0.2, 2.4, 3.0],
                        [0.2, 2.4, 0.0],
                    ],
                }
            ],
            "floor_polygon": floor_right,
            "ceiling_polygon": ceiling_right,
            "ceiling_type": "sloped",
        },
    ]


def _slope_y(z, y_ridge=2.4, y_eave=1.4, z_ridge=0.0, z_eave=3.0):
    t = (z - z_ridge) / (z_eave - z_ridge)
    return y_ridge + t * (y_eave - y_ridge)


def test_gap_wall_top_clamped_to_sloped_ceiling():
    """Gap-wall ytop must follow the sloped ceiling instead of the flat
    wall-top, so gap geometry doesn't extend horizontally past the eave.
    """
    rooms = _sloped_ceiling_rooms()
    for fn in (compute_gap_walls, _compute_gap_walls):
        walls = fn(_base_gap(), rooms, {0: 0.0})
        long_edges = [w for w in walls if w.get("type") == "within_story"]
        assert long_edges, f"expected within-story gap walls from {fn.__name__}"
        for w in long_edges:
            for corner in w["corners"]:
                _, y, z = corner
                if y <= 0.5 + 1e-6:  # bottom corners
                    continue
                expected = _slope_y(z)
                # Clamp must lower flat 2.4 to the slope; allow plane-fit slack.
                assert y <= expected + 0.05, (
                    f"{fn.__name__}: top corner at z={z:.2f} got y={y:.3f}, "
                    f"expected <= {expected:.3f}"
                )
                assert y >= expected - 0.05


def test_flat_ceiling_rooms_unaffected_by_clamp():
    """Rooms with no `ceiling_type=sloped` must not be clamped — the existing
    wall-snap height should pass through unchanged.
    """
    rooms = _base_rooms()  # no ceiling_polygon / ceiling_type set
    for fn in (compute_gap_walls, _compute_gap_walls):
        walls = fn(_base_gap(), rooms, {0: 0.0})
        long_edges = [w for w in walls if w.get("type") == "within_story"]
        assert long_edges
        # Top corners on the left wall plane should match the flat 2.4 wall top.
        left_tops = [
            c[1]
            for w in long_edges
            for c in w["corners"]
            if abs(c[0] - 0.0) < 1e-6 and c[1] > 1.0
        ]
        assert left_tops
        for y in left_tops:
            assert abs(y - 2.4) <= 1e-3


def test_caps_suppressed_when_snapshot_fully_contains_gap():
    """When the pre-absorption room union already covers the gap area, side
    walls still emit (via the original snapped polygon) but floor/ceiling
    caps are skipped to avoid duplicating a real room floor/ceiling.
    """
    rooms = _absorbed_rooms()
    # Pre-absorption polygons that already contain the gap strip.
    covering_polys = [
        [
            [-0.1, 0.0, -0.5],
            [0.3, 0.0, -0.5],
            [0.3, 0.0, 3.5],
            [-0.1, 0.0, 3.5],
        ],
        [
            [-0.1, 0.0, -0.5],
            [0.3, 0.0, -0.5],
            [0.3, 0.0, 3.5],
            [-0.1, 0.0, 3.5],
        ],
    ]
    walls = compute_gap_walls(
        _base_gap(),
        rooms,
        {0: 0.0},
        pre_absorption_floor_polygons=covering_polys,
    )
    floor_caps = [w for w in walls if w.get("type") == "gap_floor"]
    ceil_caps = [w for w in walls if w.get("type") == "gap_ceiling"]
    assert not floor_caps, "gap floor cap should be suppressed when fully inside room"
    assert not ceil_caps, "gap ceiling cap should be suppressed when fully inside room"
