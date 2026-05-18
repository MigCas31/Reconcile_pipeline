from __future__ import annotations

from reconcile_tiers.energy.areas import (
    ceiling_piece_area,
    room_floor_area,
    wall_area_minus_cutouts,
)
from reconcile_tiers.payload.schema import HorizontalLid, Quad, Room, Vec3, Wall


def _square(y: float = 0.0, size: float = 2.0) -> list[Vec3]:
    return [
        Vec3(0.0, y, 0.0),
        Vec3(size, y, 0.0),
        Vec3(size, y, size),
        Vec3(0.0, y, size),
    ]


def test_room_floor_area_sums_floor_pieces():
    room = Room(
        story=0,
        floor=[HorizontalLid(_square(size=2.0)), HorizontalLid(_square(size=1.0))],
        walls=[],
        doors=[],
        windows=[],
        locator_id="r",
    )

    assert room_floor_area(room) == 5.0


def test_wall_area_subtracts_cutouts():
    wall = Wall(
        corners=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 2.0, 0.0),
            Vec3(3.0, 2.0, 0.0),
            Vec3(3.0, 0.0, 0.0),
        ],
        descent_strip=None,
        uplift_strip=None,
        cutouts=[
            Quad(
                corners=[
                    Vec3(0.0, 0.0, 0.0),
                    Vec3(0.0, 1.0, 0.0),
                    Vec3(1.0, 1.0, 0.0),
                    Vec3(1.0, 0.0, 0.0),
                ]
            )
        ],
        locator_id="w",
    )

    assert wall_area_minus_cutouts(wall) == 5.0


def test_ceiling_piece_area_returns_zero_for_malformed_polygon():
    assert (
        ceiling_piece_area({"corners": [{"x": 0, "y": 0, "z": 0}], "holes": []}) == 0.0
    )
