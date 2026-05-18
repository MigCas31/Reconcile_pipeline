from reconcile_tiers.assemble.building_center import compute_building_center
from reconcile_tiers.extract.building import BuildingModel, ExtractedRoom, ExtractedWall


def test_building_center_averages_all_computed_wall_corners():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[],
        walls_merged=[],
        walls_computed=[
            ExtractedWall(
                id="w0",
                source="test",
                corners=[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
            ),
            ExtractedWall(
                id="w1",
                source="test",
                corners=[[2, 0, 2], [4, 0, 2], [4, 2, 2], [2, 2, 2]],
            ),
        ],
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
    model = BuildingModel(
        uuid="test",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[room],
        scan_rooms_found=0,
        scan_rooms_transformed=0,
    )

    center = compute_building_center(model)

    assert center.x == 2.0
    assert center.y == 1.0
    assert center.z == 1.0


def test_building_center_defaults_to_origin_when_no_wall_corners():
    model = BuildingModel(
        uuid="test",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[],
        scan_rooms_found=0,
        scan_rooms_transformed=0,
    )

    center = compute_building_center(model)

    assert center.x == 0.0
    assert center.y == 0.0
    assert center.z == 0.0
