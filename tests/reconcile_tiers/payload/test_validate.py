import dataclasses

import pytest

from reconcile_tiers.payload.schema import (
    CeilingPiece,
    GapKind,
    GapPiece,
    GapScope,
    HorizontalLid,
    Plane,
    Quad,
    Room,
    Vec3,
    Wall,
)
from reconcile_tiers.payload.validate import PayloadInvariantError, validate_payload
from tests.reconcile_tiers.payload.test_schema import sample_payload


def test_valid_payload_passes():
    validate_payload(sample_payload())


def test_payload_invariants_enforced_for_wrong_horizontal_lid_winding():
    payload = sample_payload()
    bad_floor = HorizontalLid(corners=list(reversed(payload.rooms[0].floor[0].corners)))
    bad_room = dataclasses.replace(payload.rooms[0], floor=[bad_floor])
    payload = dataclasses.replace(payload, rooms=[bad_room])

    with pytest.raises(PayloadInvariantError, match="rooms\\[0\\]\\.floor"):
        validate_payload(payload)


def test_payload_invariants_enforced_for_wrong_horizontal_gap_winding():
    payload = sample_payload()
    bad_gap = GapPiece(
        corners=list(reversed(payload.gaps[0].corners)),
        kind=GapKind.GAP_FLOOR,
        scope=GapScope.INTRA_STORY,
        locator_id="uuid-1::tier-gap::bad-floor",
    )
    payload = dataclasses.replace(payload, gaps=[bad_gap])

    with pytest.raises(PayloadInvariantError, match="gaps\\[0\\]\\.corners"):
        validate_payload(payload)


def test_non_planar_horizontal_lid_is_rejected():
    payload = sample_payload()
    bad_lid = HorizontalLid(
        corners=[
            Vec3(0.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(1.0, 0.01, 0.0),
            Vec3(0.0, 0.0, 0.0),
        ]
    )
    bad_room = dataclasses.replace(payload.rooms[0], floor=[bad_lid])
    payload = dataclasses.replace(payload, rooms=[bad_room])

    with pytest.raises(PayloadInvariantError, match="planar Y"):
        validate_payload(payload)


def test_self_intersecting_horizontal_lid_is_rejected():
    payload = sample_payload()
    bad_lid = HorizontalLid(
        corners=[
            Vec3(2.0, 0.0, 1.0),
            Vec3(0.0, 0.0, 2.0),
            Vec3(3.0, 0.0, 2.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(3.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
        ]
    )
    bad_room = dataclasses.replace(payload.rooms[0], floor=[bad_lid])
    payload = dataclasses.replace(payload, rooms=[bad_room])

    with pytest.raises(PayloadInvariantError, match="valid non-empty XZ polygon"):
        validate_payload(payload)


def test_self_intersecting_knee_wall_is_rejected():
    payload = sample_payload()
    bad_knee = dataclasses.replace(
        payload.knee_walls[0],
        corners=[
            Vec3(2.0, 1.0, 0.0),
            Vec3(0.0, 2.0, 0.0),
            Vec3(3.0, 2.0, 0.0),
            Vec3(1.0, 1.0, 0.0),
            Vec3(3.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
        ],
    )
    payload = dataclasses.replace(payload, knee_walls=[bad_knee])

    with pytest.raises(
        PayloadInvariantError, match="valid non-empty plane-local polygon"
    ):
        validate_payload(payload)


def test_adjacent_rooms_may_share_overlapping_wall_plane():
    payload = sample_payload()
    shared_wall_a = Wall(
        corners=[
            Vec3(1.0, 0.0, 0.0),
            Vec3(1.0, 2.0, 0.0),
            Vec3(1.0, 2.0, 1.0),
            Vec3(1.0, 0.0, 1.0),
        ],
        descent_strip=None,
        uplift_strip=None,
        cutouts=[],
        locator_id="uuid-1::tier-wall::0:shared",
    )
    shared_wall_b = dataclasses.replace(
        shared_wall_a, locator_id="uuid-1::tier-wall::1:shared"
    )
    room_a = Room(
        story=0,
        floor=[
            HorizontalLid(
                corners=[
                    Vec3(0.0, 0.0, 1.0),
                    Vec3(1.0, 0.0, 1.0),
                    Vec3(1.0, 0.0, 0.0),
                    Vec3(0.0, 0.0, 0.0),
                ]
            )
        ],
        walls=[shared_wall_a],
        doors=[],
        windows=[],
        locator_id="uuid-1::tier-room::0",
    )
    room_b = Room(
        story=0,
        floor=[
            HorizontalLid(
                corners=[
                    Vec3(1.0, 0.0, 1.0),
                    Vec3(2.0, 0.0, 1.0),
                    Vec3(2.0, 0.0, 0.0),
                    Vec3(1.0, 0.0, 0.0),
                ]
            )
        ],
        walls=[shared_wall_b],
        doors=[],
        windows=[],
        locator_id="uuid-1::tier-room::1",
    )
    payload = dataclasses.replace(payload, rooms=[room_a, room_b], knee_walls=[])

    validate_payload(payload)


def test_overlapping_room_floors_cannot_hide_wall_partition_overlap():
    payload = sample_payload()
    shared_wall_a = Wall(
        corners=[
            Vec3(0.75, 0.0, 0.0),
            Vec3(0.75, 2.0, 0.0),
            Vec3(0.75, 2.0, 1.0),
            Vec3(0.75, 0.0, 1.0),
        ],
        descent_strip=None,
        uplift_strip=None,
        cutouts=[],
        locator_id="uuid-1::tier-wall::0:shared",
    )
    shared_wall_b = dataclasses.replace(
        shared_wall_a, locator_id="uuid-1::tier-wall::1:shared"
    )
    room_a = Room(
        story=0,
        floor=[
            HorizontalLid(
                corners=[
                    Vec3(0.0, 0.0, 1.0),
                    Vec3(1.0, 0.0, 1.0),
                    Vec3(1.0, 0.0, 0.0),
                    Vec3(0.0, 0.0, 0.0),
                ]
            )
        ],
        walls=[shared_wall_a],
        doors=[],
        windows=[],
        locator_id="uuid-1::tier-room::0",
    )
    room_b = Room(
        story=0,
        floor=[
            HorizontalLid(
                corners=[
                    Vec3(0.5, 0.0, 1.0),
                    Vec3(1.5, 0.0, 1.0),
                    Vec3(1.5, 0.0, 0.0),
                    Vec3(0.5, 0.0, 0.0),
                ]
            )
        ],
        walls=[shared_wall_b],
        doors=[],
        windows=[],
        locator_id="uuid-1::tier-room::1",
    )
    payload = dataclasses.replace(payload, rooms=[room_a, room_b], knee_walls=[])

    with pytest.raises(PayloadInvariantError, match="no plane-local overlap"):
        validate_payload(payload)


def test_vertical_plane_is_rejected_with_path():
    payload = sample_payload()
    bad_piece = dataclasses.replace(
        payload.ceiling[0], plane=Plane(1.0, 0.01, 0.0, 0.0)
    )
    payload = dataclasses.replace(payload, ceiling=[bad_piece])

    with pytest.raises(PayloadInvariantError, match="ceiling\\[0\\]\\.plane"):
        validate_payload(payload)


def test_quad_must_have_four_corners():
    payload = sample_payload()
    bad_wall = dataclasses.replace(
        payload.rooms[0].walls[0],
        cutouts=[Quad(corners=[Vec3(0.0, 0.0, 0.0)] * 5)],
    )
    bad_room = dataclasses.replace(payload.rooms[0], walls=[bad_wall])
    payload = dataclasses.replace(payload, rooms=[bad_room])

    with pytest.raises(PayloadInvariantError, match="exactly 4"):
        validate_payload(payload)


def test_ceiling_corners_must_lie_on_plane():
    payload = sample_payload()
    bad_corners = [*payload.ceiling[0].corners]
    bad_corners[0] = Vec3(bad_corners[0].x, bad_corners[0].y + 0.05, bad_corners[0].z)
    bad_piece = dataclasses.replace(payload.ceiling[0], corners=bad_corners)
    payload = dataclasses.replace(payload, ceiling=[bad_piece])

    with pytest.raises(PayloadInvariantError, match=r"plane\.y_at"):
        validate_payload(payload)


def test_ceiling_hole_must_be_inside_main_polygon():
    payload = sample_payload()
    bad_piece = CeilingPiece(
        **{
            **dataclasses.asdict(payload.ceiling[0]),
            "corners": payload.ceiling[0].corners,
            "holes": [
                [
                    Vec3(10.0, 2.5, 10.0),
                    Vec3(11.0, 2.5, 10.0),
                    Vec3(11.0, 2.5, 11.0),
                    Vec3(10.0, 2.5, 11.0),
                ]
            ],
            "plane": payload.ceiling[0].plane,
            "source": payload.ceiling[0].source,
        }
    )
    payload = dataclasses.replace(payload, ceiling=[bad_piece])

    with pytest.raises(PayloadInvariantError, match="hole"):
        validate_payload(payload)


def test_ceiling_partition_accounts_for_holes():
    payload = sample_payload()
    plane = Plane(0.0, 1.0, 0.0, 2.5)
    ring_piece = dataclasses.replace(
        payload.ceiling[0],
        corners=[
            Vec3(0.0, 2.5, 3.0),
            Vec3(3.0, 2.5, 3.0),
            Vec3(3.0, 2.5, 0.0),
            Vec3(0.0, 2.5, 0.0),
        ],
        holes=[
            [
                Vec3(1.0, 2.5, 1.0),
                Vec3(2.0, 2.5, 1.0),
                Vec3(2.0, 2.5, 2.0),
                Vec3(1.0, 2.5, 2.0),
            ]
        ],
        plane=plane,
        locator_id="uuid-1::tier-ceiling-roof-arrangement::0",
    )
    hole_fill = dataclasses.replace(
        payload.ceiling[0],
        corners=[
            Vec3(1.2, 2.5, 1.8),
            Vec3(1.8, 2.5, 1.8),
            Vec3(1.8, 2.5, 1.2),
            Vec3(1.2, 2.5, 1.2),
        ],
        holes=[],
        plane=plane,
        locator_id="uuid-1::tier-ceiling-roof-arrangement::1",
    )
    payload = dataclasses.replace(payload, ceiling=[ring_piece, hole_fill])

    validate_payload(payload)
