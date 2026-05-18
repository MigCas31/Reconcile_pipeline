from __future__ import annotations

from math import radians, tan

from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
    RawCeilingPlane,
)


def _wall(wall_id: str, corners: list[list[float]]) -> ExtractedWall:
    return ExtractedWall(id=wall_id, corners=corners, source="test")


def make_gable_model(
    include_dormer: bool = False, room_depth: float = 4.0
) -> BuildingModel:
    slope = tan(radians(30.0))
    z = float(room_depth)
    floor = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 0.0, z], [0.0, 0.0, z]]
    roof_wall = _wall(
        "roof-plane",
        [
            [0.0, 2.0, 0.0],
            [6.0, 2.0 + slope * 6.0, 0.0],
            [6.0, 2.0 + slope * 6.0, z],
            [0.0, 2.0, z],
        ],
    )
    tall_reference = _wall(
        "tall-reference",
        [
            [6.0, 0.0, 0.0],
            [6.0, 0.0, z],
            [6.0, 4.0, z],
            [6.0, 4.0, 0.0],
        ],
    )
    walls = [roof_wall, tall_reference]
    windows: list[ExtractedElement] = []
    if include_dormer:
        base_y = 2.0 + slope * 3.0
        z_mid = z / 2.0
        front = [
            [3.0, base_y + 0.05, z_mid - 0.5],
            [3.0, base_y + 0.05, z_mid + 0.5],
            [3.0, base_y + 0.85, z_mid + 0.5],
            [3.0, base_y + 0.85, z_mid - 0.5],
        ]
        walls.extend(
            [
                _wall("dormer-front", front),
                _wall(
                    "dormer-left-cheek",
                    [
                        front[0],
                        [3.6, base_y + 0.05, z_mid - 0.5],
                        [3.6, base_y + 0.85, z_mid - 0.5],
                        front[3],
                    ],
                ),
                _wall(
                    "dormer-right-cheek",
                    [
                        front[1],
                        [3.6, base_y + 0.05, z_mid + 0.5],
                        [3.6, base_y + 0.85, z_mid + 0.5],
                        front[2],
                    ],
                ),
            ]
        )
        windows.append(
            ExtractedElement(
                id="dormer-window",
                source="test",
                corners=[
                    [3.0, base_y + 0.25, z_mid - 0.25],
                    [3.0, base_y + 0.25, z_mid + 0.25],
                    [3.0, base_y + 0.65, z_mid + 0.25],
                    [3.0, base_y + 0.65, z_mid - 0.25],
                ],
            )
        )
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=floor,
        walls_merged=list(walls),
        walls_computed=list(walls),
        doors=[],
        windows=windows,
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    return BuildingModel(
        uuid="synthetic-gable",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[room],
        scan_rooms_found=1,
        scan_rooms_transformed=1,
    )


def make_two_story_flat_model() -> BuildingModel:
    floor0 = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 4.0], [0.0, 0.0, 4.0]]
    floor1 = [[0.0, 3.1, 0.0], [4.0, 3.1, 0.0], [4.0, 3.1, 4.0], [0.0, 3.1, 4.0]]
    lower_wall = _wall(
        "lower-wall",
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 3.0, 0.0]],
    )
    top_wall = _wall(
        "top-wall", [[0.0, 3.1, 0.0], [4.0, 3.1, 0.0], [4.0, 3.4, 0.0], [0.0, 3.4, 0.0]]
    )
    rooms = [
        ExtractedRoom(
            index=0,
            story=0,
            floor_polygon=floor0,
            walls_merged=[lower_wall],
            walls_computed=[lower_wall],
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
        ),
        ExtractedRoom(
            index=1,
            story=1,
            floor_polygon=floor1,
            walls_merged=[top_wall],
            walls_computed=[top_wall],
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
        ),
    ]
    return BuildingModel("synthetic-flat", None, 2, False, rooms, 2, 2)


def make_simple_slant_model() -> BuildingModel:
    floor = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 3.0], [0.0, 0.0, 3.0]]
    wall = _wall(
        "plain-wall",
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.4, 0.0], [0.0, 2.4, 0.0]],
    )
    raw = RawCeilingPlane(
        corners=[
            [0.0, 2.0, 0.0],
            [3.0, 2.8, 0.0],
            [3.0, 2.8, 3.0],
            [0.0, 2.0, 3.0],
        ]
    )
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=floor,
        walls_merged=[wall],
        walls_computed=[wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[raw],
        raw_ceiling_source="test",
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    return BuildingModel("synthetic-simple-slant", None, 1, False, [room], 1, 1)
