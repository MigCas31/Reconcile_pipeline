from collections import Counter
from pathlib import Path

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers.assemble.gaps_to_pieces import assemble_gap_pieces
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedGapWall,
    ExtractedRoom,
    ExtractedStitchWall,
    GapClosure,
    extract_building_model,
)
from reconcile_tiers.payload.schema import GapKind, GapScope


def _model(gap_walls=None, stitch_walls=None, gap_closures=None):
    return BuildingModel(
        uuid="test",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[],
        scan_rooms_found=0,
        scan_rooms_transformed=0,
        gap_walls=gap_walls or [],
        stitch_walls=stitch_walls or [],
        gap_closures=gap_closures or [],
    )


def test_gap_floor_is_planarized_and_wound_upward():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="floor",
                type="gap_floor",
                story=0,
                confidence="high",
                corners=[[0, 0.0, 0], [0, 0.02, 1], [1, -0.01, 1], [1, 0.01, 0]],
            )
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert pieces[0].kind == GapKind.GAP_FLOOR
    assert pieces[0].scope == GapScope.INTRA_STORY
    assert (
        max(corner.y for corner in pieces[0].corners)
        - min(corner.y for corner in pieces[0].corners)
        < 1e-9
    )
    assert (
        newell_normal([[corner.x, corner.y, corner.z] for corner in pieces[0].corners])[
            1
        ]
        > 0.0
    )


def test_gap_ceiling_is_planarized_and_wound_upward():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="ceiling",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[[0, 3.0, 0], [1, 3.01, 0], [1, 2.99, 1], [0, 3.0, 1]],
            )
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert pieces[0].kind == GapKind.GAP_CEILING
    assert (
        newell_normal([[corner.x, corner.y, corner.z] for corner in pieces[0].corners])[
            1
        ]
        > 0.0
    )


def test_gap_ceiling_caps_with_small_overlap_are_retained():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="existing-cap",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[[0.0, 2.5, 0.0], [1.0, 2.5, 0.0], [0.0, 2.5, 1.0]],
            ),
            ExtractedGapWall(
                id="adjacent-cap",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[[0.85, 2.5, 0.0], [2.0, 2.5, 0.0], [0.85, 2.5, 1.0]],
            ),
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.locator_id for piece in pieces] == [
        "test::tier-gap::existing-cap",
        "test::tier-gap::adjacent-cap",
    ]


def test_mostly_duplicate_gap_ceiling_cap_is_suppressed():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="existing-cap",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[[0.0, 2.5, 0.0], [1.0, 2.5, 0.0], [0.0, 2.5, 1.0]],
            ),
            ExtractedGapWall(
                id="duplicate-cap",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[[0.02, 2.5, 0.0], [0.98, 2.5, 0.0], [0.02, 2.5, 0.98]],
            ),
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.locator_id for piece in pieces] == ["test::tier-gap::existing-cap"]


def test_sloped_gap_ceiling_lid_is_not_flattened_into_piece():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="sloped-ceiling",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[[0, 2.0, 0], [4, 3.2, 0], [4, 3.2, 1], [0, 2.0, 1]],
            )
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert len(pieces) == 1
    assert pieces[0].kind == GapKind.GAP_CEILING
    assert (
        max(corner.y for corner in pieces[0].corners)
        - min(corner.y for corner in pieces[0].corners)
        > 1.0
    )


def test_shallow_sloped_gap_ceiling_lid_is_not_flattened_into_piece():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="shallow-sloped-ceiling",
                type="gap_ceiling",
                story=0,
                confidence="high",
                corners=[
                    [0.5185, 0.2478, 1.1966],
                    [0.4680, 0.2426, 1.2105],
                    [0.1681, 0.2159, 0.1238],
                ],
            )
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert len(pieces) == 1
    assert pieces[0].kind == GapKind.GAP_CEILING
    ys = [corner.y for corner in pieces[0].corners]
    assert max(ys) - min(ys) > 0.03


def test_stitch_types_are_typed_junction_pieces_without_substring_dispatch():
    model = _model(
        stitch_walls=[
            ExtractedStitchWall(
                id="stitch",
                type="stitch",
                story=0,
                room_index=0,
                room_indices=[0, 1],
                corners=[[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]],
            ),
            ExtractedStitchWall(
                id="stitch-ceiling",
                type="stitch_ceiling",
                story=0,
                room_index=0,
                room_indices=[0, 1],
                corners=[[0, 2, 0], [1, 2, 0], [0, 2, 1]],
            ),
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.kind for piece in pieces] == [GapKind.STITCH, GapKind.STITCH_CEIL]
    assert all(piece.scope == GapScope.JUNCTION for piece in pieces)


def test_small_side_piece_does_not_suppress_larger_stitch_closure():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="small-side",
                type="within_story",
                story=0,
                confidence="high",
                corners=[[0, 0, 0], [0, 0, 0.1], [0, 2, 0.1], [0, 2, 0]],
            )
        ],
        stitch_walls=[
            ExtractedStitchWall(
                id="large-stitch",
                type="stitch",
                story=0,
                room_index=0,
                room_indices=[0, 1],
                corners=[[0, 0, 0], [0, 0, 1], [0, 2, 1], [0, 2, 0]],
            )
        ],
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.kind for piece in pieces] == [GapKind.STITCH]
    assert pieces[0].locator_id == "test::tier-gap::large-stitch"


def test_larger_exterior_side_replaces_duplicate_internal_side():
    model = _model(
        gap_walls=[
            ExtractedGapWall(
                id="internal-side",
                type="within_story",
                story=0,
                confidence="high",
                corners=[[0, 0, 0], [0, 0, 1], [0, 2, 1], [0, 2, 0]],
            )
        ],
        gap_closures=[
            GapClosure(
                type="side",
                story=0,
                indicator_element_id="storage-a",
                indicator_wall_id="wall-b",
                corners=[[0, -0.1, 0], [0, -0.1, 1], [0, 2, 1], [0, 2, 0]],
            )
        ],
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.kind for piece in pieces] == [GapKind.EXTERIOR_SIDE]
    assert pieces[0].locator_id == "test::tier-gap::storage-a:wall-b:side:0"


def test_one_sided_stitch_outside_story_footprint_is_not_rendered():
    uuid = "494a97c4-ab7e-4021-9345-c46fecfd1d67"
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    pieces = assemble_gap_pieces(model)
    locator_ids = {piece.locator_id for piece in pieces}

    assert f"{uuid}::tier-gap::stitch:0:6" not in locator_ids


def test_deep_internal_exterior_gap_closure_is_not_rendered():
    uuid = "b01824fc-be43-451f-bc6e-aaab701c144d"
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    pieces = assemble_gap_pieces(model)
    locator_ids = {piece.locator_id for piece in pieces}

    bad_prefix = (
        f"{uuid}::tier-gap::"
        "F89A576E-F3F9-4887-B36E-433E5A2E9238:DA017DDB-739E-4E51-A766-E7375F6E4731"
    )
    assert all(not locator_id.startswith(bad_prefix) for locator_id in locator_ids)


def test_unknown_gap_stitch_and_exterior_types_are_not_substring_dispatched():
    model = BuildingModel(
        uuid="test",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[],
        scan_rooms_found=0,
        scan_rooms_transformed=0,
        gap_walls=[
            ExtractedGapWall(
                id="mystery-floor-name",
                type="mystery_floor",
                story=0,
                confidence="low",
                corners=[[0, 0, 0], [1, 0, 0], [1, 0, 1]],
            )
        ],
        stitch_walls=[
            ExtractedStitchWall(
                id="mystery-stitch",
                type="stitch_unknown",
                story=0,
                room_index=0,
                room_indices=[0, 1],
                corners=[[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            )
        ],
        gap_closures=[
            GapClosure(
                type="floorish",
                story=0,
                indicator_element_id="door-a",
                indicator_wall_id="wall-b",
                corners=[[0, 0, 0], [1, 0, 0], [1, 0, 1]],
            )
        ],
    )

    assert assemble_gap_pieces(model) == []


def test_exterior_closures_are_typed_exterior_pieces():
    model = _model()
    model = BuildingModel(
        uuid=model.uuid,
        address=model.address,
        stories_found=model.stories_found,
        split_level=model.split_level,
        rooms=model.rooms,
        scan_rooms_found=model.scan_rooms_found,
        scan_rooms_transformed=model.scan_rooms_transformed,
        gap_closures=[
            GapClosure(
                type="side",
                story=0,
                indicator_element_id="door-a",
                indicator_wall_id="wall-b",
                corners=[[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]],
            ),
            GapClosure(
                type="ceiling",
                story=0,
                indicator_element_id="door-a",
                indicator_wall_id="wall-b",
                corners=[[0, 2, 0], [1, 2.02, 0], [1, 1.99, 1], [0, 2, 1]],
            ),
        ],
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.kind for piece in pieces] == [
        GapKind.EXTERIOR_SIDE,
        GapKind.EXTERIOR_CEIL,
    ]
    assert all(piece.scope == GapScope.EXTERIOR for piece in pieces)
    assert pieces[0].locator_id == "test::tier-gap::door-a:wall-b:side:0"
    assert (
        newell_normal([[corner.x, corner.y, corner.z] for corner in pieces[1].corners])[
            1
        ]
        > 0.0
    )


def test_internal_exterior_floor_closure_is_suppressed_when_inside_room_floor():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
        walls_merged=[],
        walls_computed=[],
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
    model = _model(
        gap_closures=[
            GapClosure(
                type="floor",
                story=0,
                indicator_element_id="door-a",
                indicator_wall_id="wall-b",
                corners=[[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
            )
        ]
    )
    model = BuildingModel(
        uuid=model.uuid,
        address=model.address,
        stories_found=model.stories_found,
        split_level=model.split_level,
        rooms=[room],
        scan_rooms_found=model.scan_rooms_found,
        scan_rooms_transformed=model.scan_rooms_transformed,
        gap_closures=model.gap_closures,
    )

    pieces = assemble_gap_pieces(model)

    assert pieces == []


def test_exterior_floor_closure_near_story_boundary_is_retained():
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
        walls_merged=[],
        walls_computed=[],
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
    model = _model(
        gap_closures=[
            GapClosure(
                type="floor",
                story=0,
                indicator_element_id="door-a",
                indicator_wall_id="wall-b",
                corners=[[0, 0, 0], [2, 0, 0], [2, 0, 0.2], [0, 0, 0.2]],
            )
        ]
    )
    model = BuildingModel(
        uuid=model.uuid,
        address=model.address,
        stories_found=model.stories_found,
        split_level=model.split_level,
        rooms=[room],
        scan_rooms_found=model.scan_rooms_found,
        scan_rooms_transformed=model.scan_rooms_transformed,
        gap_closures=model.gap_closures,
    )

    pieces = assemble_gap_pieces(model)

    assert [piece.kind for piece in pieces] == [GapKind.EXTERIOR_FLOOR]
    assert pieces[0].scope == GapScope.EXTERIOR


def test_sloped_exterior_ceiling_closure_is_not_flattened_into_piece():
    model = _model(
        gap_closures=[
            GapClosure(
                type="ceiling",
                story=0,
                indicator_element_id="door-a",
                indicator_wall_id="wall-b",
                corners=[
                    [0.0, 0.35, 0.0],
                    [1.0, 0.95, 0.0],
                    [1.0, 0.95, 1.0],
                    [0.0, 0.35, 1.0],
                ],
            )
        ]
    )

    pieces = assemble_gap_pieces(model)

    assert len(pieces) == 1
    assert pieces[0].kind == GapKind.EXTERIOR_CEIL
    ys = [corner.y for corner in pieces[0].corners]
    assert max(ys) - min(ys) > 0.5
    assert (
        newell_normal([[corner.x, corner.y, corner.z] for corner in pieces[0].corners])[
            1
        ]
        > 0.0
    )


def test_cohort_gap_piece_counts_include_gap_stitch_and_exterior_surfaces():
    expected = {
        # `c72ad855` shifted after `roof.priors.filter_obliques_by_roof_type`
        # dropped a "sticking out" oblique that didn't fit the gable pair
        # — extra floor area now becomes gap_ceiling/exterior pieces.
        # Local stitch top caps change one side-ownership duplicate decision:
        # the shorter stitch no longer fully yields to the exterior side. The
        # exterior air guard suppresses closure groups fully inside floor area.
        "c72ad855-9e52-46f1-886d-a9f37911521f": {
            "exterior_ceiling": 1,
            "exterior_side": 3,
            "exterior_floor": 1,
            "gap_ceiling": 15,
            "gap_floor": 2,
            "side": 23,
            "stitch": 8,
        },
        # The wall-crossing rule still rejects the L-stitch whose second leg
        # pierced room 6's wall 4. Without the legacy room-count cap, the
        # remaining non-conflicting wall-endpoint stitches are retained.
        # The exterior air guard suppresses the wide door+wall pairs that sit
        # inside occupied floor area rather than outside the building envelope.
        # Then the height_align flat-ceiling cap-aware merge collapsed three
        # flat rooms on story 0 whose LWM-snap stayed within CORNER_SNAP_TOL_M
        # of every member into a single group, unifying their floor_y and
        # eliminating the sub-cm gap_floor pieces between them (gap_floor 5→2).
        "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd": {
            "exterior_ceiling": 1,
            "exterior_floor": 1,
            "exterior_side": 4,
            "gap_ceiling": 17,
            "gap_floor": 2,
            "side": 29,
            "stitch": 7,
        },
        # `2ea3b759` gap_ceiling +3, side +2 from same prior-drop mechanism.
        # The internal exterior-closure guard drops the unsupported interior
        # door closure; the remaining valid endpoint closures are retained.
        # Then height_align cap-aware merge (with MEMBER_SHIFT_CAP_M = 10 cm)
        # collapsed all 11 flat-ceiling rooms on story 0 into one mode group
        # — the outlier room that was just past the old 5 cm cap now joins
        # the cluster, eliminating 2 sub-cm gap_floor pieces (gap_floor 3→1).
        "2ea3b759-e047-424c-8034-f8ee5b811fb4": {
            "exterior_ceiling": 1,
            "exterior_floor": 1,
            "exterior_side": 2,
            "gap_ceiling": 21,
            "gap_floor": 1,
            "side": 35,
            "stitch": 14,
        },
    }

    for uuid, expected_counts in expected.items():
        model = extract_building_model(
            uuid, Path("pipeline-outputs"), Path(".scan-cache")
        )
        pieces = assemble_gap_pieces(model)

        assert Counter(piece.kind.value for piece in pieces) == expected_counts
