"""Short-gap direct-quad closure and stable stitch IDs.

When two wall endpoints are within ``_DIRECT_QUAD_MAX_GAP_M`` (15 cm),
the stitcher emits a single rectangular quad spanning P1 → P2 instead
of an L-closure plus cap triangles. Every emitted stitch also carries
an ``id`` field that round-trips through ``reconcile.element_locator``.
"""

from reconcile import element_locator
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


def _counts_by_type(stitches):
    counts = {}
    for s in stitches:
        counts[s.get("type")] = counts.get(s.get("type"), 0) + 1
    return counts


def test_sub_threshold_gap_emits_direct_quad_no_caps():
    # Wall A along +X (0,0)→(5,0). Wall B along +X (5.08, 0.03)→(10.08, 0.03).
    # Endpoint pair gap = hypot(0.08, 0.03) ≈ 0.0854 m < 0.15 m threshold.
    # Must emit a single 4-corner quad and zero cap triangles.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (5.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((5.08, 0.03), (10.08, 0.03)),
            ],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    counts = _counts_by_type(out)

    assert counts.get("stitch", 0) == 1, f"expected one direct quad, got {counts}"
    assert counts.get("stitch_floor", 0) == 0, f"unexpected floor cap: {counts}"
    assert counts.get("stitch_ceiling", 0) == 0, f"unexpected ceiling cap: {counts}"

    quad = next(s for s in out if s.get("type") == "stitch")
    assert len(quad["corners"]) == 4


def test_above_threshold_gap_still_uses_l_closure():
    # Gap ~0.58 m, well above 0.15 m — must fall through to L-closure.
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
    counts = _counts_by_type(out)

    # Two legs (parallel + perpendicular) plus one floor cap + one ceiling cap.
    assert counts.get("stitch", 0) == 2
    assert counts.get("stitch_floor", 0) == 1
    assert counts.get("stitch_ceiling", 0) == 1


def test_every_stitch_has_unique_id_matching_type_story_format():
    # Setup produces a mix: one short gap (direct quad) and one long gap
    # (L-closure + caps). Confirms IDs are stamped on every emission path
    # and that the (type, story, index) encoding is internally consistent.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (5.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((5.08, 0.03), (10.08, 0.03)),
            ],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((11.0, 0.5), (16.0, 0.5)),
            ],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    assert out, "expected at least one stitch"

    seen_ids = set()
    for entry in out:
        stitch_id = entry.get("id")
        assert stitch_id, f"stitch missing id: {entry}"
        assert stitch_id not in seen_ids, f"duplicate stitch id: {stitch_id}"
        seen_ids.add(stitch_id)

        # ID encodes type + story; type prefix disambiguates same-index caps.
        assert stitch_id.startswith(entry["type"]), (
            f"id {stitch_id!r} should start with type {entry['type']!r}"
        )
        assert f":{entry['story']}:" in stitch_id, (
            f"id {stitch_id!r} should encode story {entry['story']}"
        )


def test_stitch_id_round_trips_through_element_locator():
    # Emit a stitch, simulate a buildings_3d.json structure, then resolve
    # the token via element_locator.find_element. Must return the same
    # corners.
    rooms_out = [
        {
            "story": 0,
            "walls_computed": [_wall_from_endpoints((0.0, 0.0), (5.0, 0.0))],
        },
        {
            "story": 0,
            "walls_computed": [
                _wall_from_endpoints((5.08, 0.03), (10.08, 0.03)),
            ],
        },
    ]

    out = stitch.stitch_wall_gaps(rooms_out)
    quad = next(s for s in out if s.get("type") == "stitch")
    stitch_id = quad["id"]

    buildings = [
        {
            "uuid": "test-uuid",
            "stitch_walls": out,
        }
    ]
    token = f"test-uuid::wall-stitch::{stitch_id}"
    parsed = element_locator.parse_element_id(token)
    resolved = element_locator._find_in_building_collection(
        buildings[0], "stitch_walls", parsed.element_id
    )
    assert resolved is not None, f"failed to resolve stitch id {stitch_id!r}"
    assert resolved["element"]["corners"] == quad["corners"]


def test_post_snap_degeneracy_guard_drops_collapsed_quad():
    # Hand-craft a snap-collapsed quad where two corners coincide after
    # snap displacement would have pulled them together. The pipeline now
    # runs _drop_degenerate_stitches once right after snap, so the entry
    # is removed before prune/dedup/crossing-room see it.
    collapsed = {
        "type": "stitch",
        "story": 0,
        "room_indices": [0, 1],
        "id": "stitch:0:0",
        "corners": [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],  # collapsed onto first corner
            [1.0, 2.4, 1.0],
            [1.0, 2.4, 1.0],
        ],
    }
    healthy = {
        "type": "stitch",
        "story": 0,
        "room_indices": [0, 1],
        "id": "stitch:0:1",
        "corners": [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 2.4, 0.0],
            [0.0, 2.4, 0.0],
        ],
    }

    kept = stitch._drop_degenerate_stitches([collapsed, healthy])
    assert len(kept) == 1
    assert kept[0]["id"] == "stitch:0:1"
