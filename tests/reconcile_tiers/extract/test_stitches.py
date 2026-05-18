from math import isfinite
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point

from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.extract.building import (
    ExtractedRoom,
    ExtractedWall,
    extract_building_model,
)
from reconcile_tiers.extract.stitches import (
    _projected_xz_area,
    stitch_wall_gaps,
)


def _wall(
    wall_id: str, a: tuple[float, float], b: tuple[float, float]
) -> ExtractedWall:
    y_low = -1.60
    y_high = 0.56
    return ExtractedWall(
        id=wall_id,
        corners=[
            [a[0], y_high, a[1]],
            [b[0], y_high, b[1]],
            [b[0], y_low, b[1]],
            [a[0], y_low, a[1]],
        ],
        source="test",
    )


def _room(index: int, wall: ExtractedWall) -> ExtractedRoom:
    return ExtractedRoom(
        index=index,
        story=1,
        floor_polygon=[],
        walls_merged=[],
        walls_computed=[wall],
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


@pytest.mark.parametrize(
    "uuid",
    [
        "c72ad855-9e52-46f1-886d-a9f37911521f",
        "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
        "2ea3b759-e047-424c-8034-f8ee5b811fb4",
    ],
)
def test_stitch_wall_gaps_emit_valid_geometry_for_cohort(uuid):
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    assert model.stitch_walls
    for stitch in model.stitch_walls:
        assert stitch.id
        assert stitch.room_index is not None
        assert len(stitch.room_indices) >= 2
        assert len(stitch.corners) >= 3
        assert all(isfinite(coord) for corner in stitch.corners for coord in corner)
        if stitch.type == "stitch":
            assert len(stitch.corners) == 4
            ys = [corner[1] for corner in stitch.corners]
            assert max(ys) - min(ys) > 0.5


def test_f16973df_small_endpoint_wall_gaps_are_not_count_capped():
    model = extract_building_model(
        "f16973df-10b9-4369-9f0d-dd295a34d970",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    expected_edges = [
        (
            (6.250365298485082, -5.64580716983362),
            (6.338574164173684, -5.625859674997375),
        ),
        (
            (-1.0595048263982962, -9.201825727762724),
            (-1.1075024771428814, -9.250335449073388),
        ),
    ]
    actual_edges = [
        (
            (float(stitch.corners[0][0]), float(stitch.corners[0][2])),
            (float(stitch.corners[1][0]), float(stitch.corners[1][2])),
        )
        for stitch in model.stitch_walls
        if stitch.type == "stitch" and len(stitch.corners) >= 2
    ]

    def has_edge(a, b, tol=1e-4):
        def close(p, q):
            return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol

        return any(
            (close(edge_a, a) and close(edge_b, b))
            or (close(edge_a, b) and close(edge_b, a))
            for edge_a, edge_b in actual_edges
        )

    missing = [(a, b) for a, b in expected_edges if not has_edge(a, b)]
    assert missing == []


def test_l_stitch_side_stops_at_local_sloped_wall_top_on_motivating_building():
    model = extract_building_model(
        "466d0aa8-7021-4825-8f7b-b954c824c13a",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    stitch = next(s for s in model.stitch_walls if s.id == "stitch:1:20")
    top_ys = [corner[1] for corner in stitch.corners[2:]]

    assert max(top_ys) == pytest.approx(-0.7175577492)
    assert min(top_ys) == pytest.approx(-0.7175577492)


def test_stitches_do_not_cover_existing_wall_runs_on_motivating_building():
    payload = build_tier_payload(
        "0fe789ce-e653-4828-863c-0f6ce7fee21d",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )
    wall_lines = []
    for room in payload.rooms:
        for wall in room.walls:
            for idx, corner in enumerate(wall.corners):
                nxt = wall.corners[(idx + 1) % len(wall.corners)]
                line = LineString([(corner.x, corner.z), (nxt.x, nxt.z)])
                if line.length > 1e-6:
                    wall_lines.append(line)

    offenders = []
    for stitch in payload.gaps:
        if stitch.kind.value != "stitch" or len(stitch.corners) < 4:
            continue
        line = LineString(
            [
                (stitch.corners[0].x, stitch.corners[0].z),
                (stitch.corners[1].x, stitch.corners[1].z),
            ]
        )
        if line.length <= 1e-9:
            continue
        for wall_line in wall_lines:
            overlap_len = line.intersection(
                wall_line.buffer(0.03, cap_style="flat")
            ).length
            if overlap_len > 0.10 and overlap_len / line.length > 0.50:
                offenders.append(
                    (
                        stitch.locator_id,
                        round(line.length, 3),
                        round(overlap_len, 3),
                    )
                )

    assert offenders == []


def test_stitch_ceiling_cap_follows_lower_of_two_room_ceilings():
    # Two adjacent rooms whose walls form an L-stitch corner. Room 0 has a flat
    # ceiling at y=2.5 (full height). Room 1 has a knee-style ceiling at y=1.6
    # (low). The stitch_ceiling triangle must sit at y≈1.6 (the lower room's
    # ceiling), not at min(wall_yt) — the legacy bug parked it on the lowest
    # neighbouring wall-top, which produced caps floating below both ceilings.
    def make(index, walls, ceiling_y):
        return ExtractedRoom(
            index=index,
            story=0,
            floor_polygon=[
                [walls[0].corners[3][0], 0.0, walls[0].corners[3][2]],
                [walls[0].corners[2][0], 0.0, walls[0].corners[2][2]],
                [walls[0].corners[1][0], 0.0, walls[0].corners[1][2]],
                [walls[0].corners[0][0], 0.0, walls[0].corners[0][2]],
            ],
            walls_merged=[],
            walls_computed=walls,
            doors=[],
            windows=[],
            openings=[],
            storages=[],
            raw_ceiling_planes=[],
            raw_ceiling_source=None,
            ceiling_polygon=[
                [walls[0].corners[3][0], ceiling_y, walls[0].corners[3][2]],
                [walls[0].corners[2][0], ceiling_y, walls[0].corners[2][2]],
                [walls[0].corners[1][0], ceiling_y, walls[0].corners[1][2]],
                [walls[0].corners[0][0], ceiling_y, walls[0].corners[0][2]],
            ],
            ceiling_type="flat",
            ceiling_eave_height=ceiling_y,
            ceiling_ridge_height=ceiling_y,
        )

    wall0 = ExtractedWall(
        id="wall0",
        source="test",
        corners=[
            [0.0, 2.5, 0.0],
            [1.0, 2.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    wall1 = ExtractedWall(
        id="wall1",
        source="test",
        corners=[
            [2.0, 1.6, 1.0],
            [2.0, 1.6, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
        ],
    )
    room0 = make(0, [wall0], ceiling_y=2.5)
    room1 = make(1, [wall1], ceiling_y=1.6)

    stitches = stitch_wall_gaps([room0, room1])
    ceilings = [s for s in stitches if s.type == "stitch_ceiling"]

    assert ceilings, (
        "stitch_ceiling cap should be emitted when both rooms have ceilings"
    )
    for ceiling in ceilings:
        ys = [corner[1] for corner in ceiling.corners]
        # Cap must follow lower room (~1.6), not min(wall_yt) which would be
        # somewhere below 1.6 if any neighbour wall is partial-height.
        assert max(ys) == pytest.approx(1.6, abs=0.05)
        assert min(ys) == pytest.approx(1.6, abs=0.05)


def test_short_leg_l_stitch_still_emits_angle_cap_when_area_is_meaningful():
    def make_room(index, walls, ceiling_y, floor, ceiling):
        return ExtractedRoom(
            index=index,
            story=0,
            floor_polygon=[[x, 0.0, z] for x, z in floor],
            walls_merged=[],
            walls_computed=walls,
            doors=[],
            windows=[],
            openings=[],
            storages=[],
            raw_ceiling_planes=[],
            raw_ceiling_source=None,
            ceiling_polygon=[[x, ceiling_y, z] for x, z in ceiling],
            ceiling_type="flat",
            ceiling_eave_height=ceiling_y,
            ceiling_ridge_height=ceiling_y,
        )

    wall0 = ExtractedWall(
        id="short-leg-host",
        source="test",
        corners=[
            [0.0, 2.4, 0.0],
            [1.0, 2.4, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    wall1 = ExtractedWall(
        id="angle-host",
        source="test",
        corners=[
            [1.1, 2.0, 0.8],
            [1.1, 2.0, 1.8],
            [1.1, 0.0, 1.8],
            [1.1, 0.0, 0.8],
        ],
    )
    room0 = make_room(
        0,
        [wall0],
        2.4,
        floor=[(0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)],
        ceiling=[(0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)],
    )
    room1 = make_room(
        1,
        [wall1],
        2.0,
        floor=[(0.8, 0.0), (1.4, 0.0), (1.4, 1.8), (0.8, 1.8)],
        ceiling=[(0.8, 0.0), (1.4, 0.0), (1.4, 1.8), (0.8, 1.8)],
    )

    stitches = stitch_wall_gaps([room0, room1])

    ceiling_caps = [stitch for stitch in stitches if stitch.type == "stitch_ceiling"]
    floor_caps = [stitch for stitch in stitches if stitch.type == "stitch_floor"]
    assert ceiling_caps, "short-leg angle should still get ceiling coverage"
    assert floor_caps, "short-leg angle should still get floor coverage"
    assert max(point[1] for point in ceiling_caps[0].corners) == pytest.approx(2.0)
    assert min(point[1] for point in ceiling_caps[0].corners) == pytest.approx(2.0)
    assert _projected_xz_area(ceiling_caps[0].corners) == pytest.approx(0.04)


def test_l_stitch_rejects_parallel_offset_back_extension():
    # Parallel offset wall endpoints can look like an L gap numerically, but
    # physically the long leg would run behind an already-correct wall. The
    # small offset is wall-thickness noise, not a corner that needs a dogleg.
    wall0 = _wall("host", (0.0, 0.0), (2.0, 0.0))
    wall1 = _wall("parallel", (2.8, 0.2), (-3.0, 0.2))

    stitches = stitch_wall_gaps([_room(0, wall0), _room(1, wall1)])

    assert stitches == []


def test_stitch_ceiling_cap_skipped_when_no_room_has_ceiling_polygon():
    # When neither room exposes a ceiling polygon, there is no scan signal for
    # a cap height. The legacy code would synthesise one at min(wall_yt); the
    # fix drops the cap entirely (no scan, no synthesis).
    rooms = [
        _room(0, _wall("wA", (0.0, 0.0), (1.0, 0.0))),
        _room(1, _wall("wB", (1.5, 0.5), (1.5, 1.5))),
    ]
    stitches = stitch_wall_gaps(rooms)
    assert all(s.type != "stitch_ceiling" for s in stitches)


def test_l_stitch_rejected_when_a_leg_pierces_an_existing_wall():
    # Two terminating walls 1m apart, with a long gable wall sitting between
    # them at z=0.5. The straight-line stitch (or either L-leg) closing the
    # gap pierces the gable. The gable is intentionally long enough that its
    # own endpoints sit > MAX_GAP_M from wall_a/wall_b and so don't get paired
    # themselves — leaving wall_a↔wall_b as the only candidate pair.
    wall_a = _wall("a", (-0.5, 1.0), (0.5, 1.0))
    wall_gable = _wall("gable", (-3.0, 0.5), (3.0, 0.5))
    wall_b = _wall("b", (-0.5, 0.0), (0.5, 0.0))
    room0 = ExtractedRoom(
        index=0,
        story=1,
        floor_polygon=[],
        walls_merged=[],
        walls_computed=[wall_a, wall_gable],
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
    room1 = _room(1, wall_b)

    stitches = stitch_wall_gaps([room0, room1])

    # Any stitch closing the wall_a ↔ wall_b gap must straddle z=0.5 (the
    # gable's z-coord). With the wall-crossing rule, no such stitch survives.
    straddling = [
        s
        for s in stitches
        if any(c[2] < 0.5 for c in s.corners) and any(c[2] > 0.5 for c in s.corners)
    ]
    assert straddling == [], (
        f"stitch crossing the gable wall at z=0.5 still emitted: "
        f"{[(s.type, s.corners) for s in straddling]}"
    )


def test_e9dddee6_gable_piercing_l_stitch_no_longer_rendered():
    # Top-floor gable on this building: an L-stitch was paired across a real
    # gable wall, producing a wall quad + floor cap that pierced through the
    # gable. Both pieces must be absent from the rendered payload after the
    # wall-crossing rule lands. (Locator ids reshuffle once stitches are
    # dropped, so check by geometry — the specific corner anchors the user
    # called out.)
    payload = build_tier_payload(
        "e9dddee6-297a-43f5-b862-90091889ea39",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    bad_anchor = (-0.6898527345997825, -1.618459719209314, -6.172314274099293)
    other_anchor = (-1.1642176314563961, -1.618459719209314, -6.529445197900718)

    def _has_corner(piece, target):
        return any(
            corner.x == pytest.approx(target[0])
            and corner.y == pytest.approx(target[1])
            and corner.z == pytest.approx(target[2])
            for corner in piece.corners
        )

    offenders = [
        piece.locator_id
        for piece in payload.gaps
        if "stitch" in piece.kind.value
        and _has_corner(piece, bad_anchor)
        and _has_corner(piece, other_anchor)
    ]
    assert offenders == [], (
        "rendered payload still contains the gable-piercing L-stitch (or its "
        f"floor cap): {offenders}"
    )


def test_stitch_wall_gaps_closes_endpoint_to_wall_t_junction():
    support_wall = _wall(
        "support",
        (8.338526491221796, -2.3809284393171057),
        (5.97849683290302, -1.8326818605587387),
    )
    terminating_wall = _wall(
        "terminating",
        (7.617017446345975, -2.1793759402830957),
        (7.853351775514002, -1.1620315802321715),
    )

    stitches = stitch_wall_gaps([_room(0, support_wall), _room(1, terminating_wall)])

    t_stitches = [stitch for stitch in stitches if stitch.type == "stitch"]
    assert len(t_stitches) == 1
    stitch = t_stitches[0]
    assert stitch.room_indices == [1, 0]

    bottom_edge = LineString(
        [
            (stitch.corners[0][0], stitch.corners[0][2]),
            (stitch.corners[1][0], stitch.corners[1][2]),
        ]
    )
    support_line = LineString(
        [
            (support_wall.corners[0][0], support_wall.corners[0][2]),
            (support_wall.corners[1][0], support_wall.corners[1][2]),
        ]
    )
    assert bottom_edge.length == pytest.approx(0.0330619909)
    assert support_line.distance(Point(bottom_edge.coords[1])) == pytest.approx(0.0)
