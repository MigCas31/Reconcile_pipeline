import logging

import pytest
from shapely.geometry import Point, Polygon

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers.assemble.walls_to_rooms import (
    assemble_rooms,
    reclip_cutouts_to_wall,
)
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)
from reconcile_tiers.payload.schema import Quad, Vec3


def _room(*, walls, doors=None, windows=None, floor=None):
    return ExtractedRoom(
        index=0,
        story=2,
        floor_polygon=floor or [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
        walls_merged=[],
        walls_computed=walls,
        doors=doors or [],
        windows=windows or [],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )


def _model(room):
    return BuildingModel(
        uuid="uuid-1",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[room],
        scan_rooms_found=0,
        scan_rooms_transformed=0,
    )


def test_assemble_rooms_orients_floor_up_and_walls_away_from_room_centroid():
    wall = ExtractedWall(
        id="left",
        source="test",
        # This winds toward the room interior (+X) and must be reversed.
        corners=[[0, 0, 0], [0, 0, 2], [0, 2, 2], [0, 2, 0]],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall])))

    room = rooms[0]
    assert room.story == 2
    assert room.locator_id == "uuid-1::tier-room::0"
    assert newell_normal([[c.x, c.y, c.z] for c in room.floor[0].corners])[1] > 0
    normal = newell_normal([[c.x, c.y, c.z] for c in room.walls[0].corners])
    wall_center = [
        sum(getattr(c, axis) for c in room.walls[0].corners)
        / len(room.walls[0].corners)
        for axis in ("x", "y", "z")
    ]
    room_center = [1.0, 0.0, 1.0]
    outward = [wall_center[idx] - room_center[idx] for idx in range(3)]
    assert sum(normal[idx] * outward[idx] for idx in range(3)) > 0


def test_assemble_orients_exterior_when_centroid_ambig():
    floor = [
        [-2.1901907018080298, -1.5026966807907338, 4.355848723140788],
        [-4.2037381599920645, -1.5026966807907338, 6.99914539562997],
        [-5.928048997635852, -1.5026966807907338, 9.233702476137978],
        [0.2973636489412226, -1.5026966807907338, 14.037583881974292],
        [3.107166364056801, -1.5026966807907338, 10.39632507042108],
        [0.07726813251912201, -1.5026966807907338, 8.278928100343961],
        [1.3944713147787724, -1.5026966807907338, 6.571948542034722],
        [-1.9121210633104702, -1.5026966807907338, 4.029408676814009],
        [-3.191563695772775, -1.5026966807907338, 5.687455202450395],
        [-2.223093536197233, -1.5026966807907338, 4.432401992150203],
        [-2.23917437450682, -1.5026966807907338, 4.42015233740364],
    ]
    wall = ExtractedWall(
        id="door-host",
        source="test",
        # This is wound so the normal points into the room, even though the
        # wall bounds the exterior. The room centroid is almost exactly on
        # the wall plane, so a centroid sign test is not stable enough.
        corners=[
            [0.07726813782203679, -1.5026966807907338, 8.2789292816831],
            [3.1071663283076596, -1.5026966807907338, 10.396325845604157],
            [3.1071663283076596, 0.9498029739592764, 10.396325845604157],
            [0.07726813782203679, 0.9498029739592764, 8.2789292816831],
        ],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall], floor=floor)))

    corners = [[c.x, c.y, c.z] for c in rooms[0].walls[0].corners]
    normal = newell_normal(corners)
    xz_len = (normal[0] ** 2 + normal[2] ** 2) ** 0.5
    nx = normal[0] / xz_len
    nz = normal[2] / xz_len
    wall_center = [sum(c[axis] for c in corners) / len(corners) for axis in range(3)]
    room_poly = Polygon([(p[0], p[2]) for p in floor])

    assert not room_poly.covers(
        Point(wall_center[0] + nx * 0.10, wall_center[2] + nz * 0.10)
    )
    assert room_poly.covers(
        Point(wall_center[0] - nx * 0.10, wall_center[2] - nz * 0.10)
    )


def test_assemble_rooms_preserves_multi_quad_uplift_strip():
    wall = ExtractedWall(
        id="stepped",
        source="test",
        corners=[[0, 0, 2], [0, 0, 0], [0, 2, 0], [0, 2, 2]],
        uplift_strip=[
            [[0, 2.0, 0.0], [0, 2.0, 1.0], [0, 2.5, 1.0], [0, 2.5, 0.0]],
            [[0, 2.0, 1.0], [0, 2.2, 2.0], [0, 2.5, 2.0], [0, 2.5, 1.0]],
        ],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall])))

    strip = rooms[0].walls[0].uplift_strip
    assert strip is not None
    assert [[corner.x, corner.y, corner.z] for corner in strip] == [
        [0.0, 2.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, 2.2, 2.0],
        [0.0, 2.5, 2.0],
        [0.0, 2.5, 1.0],
        [0.0, 2.5, 0.0],
    ]


def test_assemble_rooms_preserves_descent_strip():
    wall = ExtractedWall(
        id="upper",
        source="test",
        corners=[[0, 2.7, 2], [0, 2.7, 0], [0, 5.0, 0], [0, 5.0, 2]],
        descent_strip=[
            [[0, 2.7, 0.0], [0, 2.4, 0.0], [0, 2.4, 2.0], [0, 2.7, 2.0]],
        ],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall])))

    strip = rooms[0].walls[0].descent_strip
    assert strip is not None
    assert sorted({round(c.y, 4) for c in strip}) == [2.4, 2.7]


def test_assemble_rooms_replaces_short_off_axis_corner_bevel_with_rectilinear_closure():
    walls = [
        ExtractedWall(
            id="northwest-run",
            source="test",
            corners=[
                [-1.029, 0.0, 3.730],
                [0.835, 0.0, 0.687],
                [0.835, 2.0, 0.687],
                [-1.029, 2.0, 3.730],
            ],
        ),
        ExtractedWall(
            id="bevel",
            source="test",
            corners=[
                [-1.029, 0.0, 3.730],
                [-2.138, 0.0, 4.077],
                [-2.138, 2.0, 4.077],
                [-1.029, 2.0, 3.730],
            ],
        ),
        ExtractedWall(
            id="southwest-run",
            source="test",
            corners=[
                [-2.138, 0.0, 4.077],
                [-3.874, 0.0, 3.013],
                [-3.874, 2.0, 3.013],
                [-2.138, 2.0, 4.077],
            ],
        ),
    ]

    rooms = assemble_rooms(
        _model(
            _room(
                walls=walls,
                floor=[
                    [0.835, 0.0, 0.687],
                    [-1.029, 0.0, 3.730],
                    [-2.138, 0.0, 4.077],
                    [-3.874, 0.0, 3.013],
                    [-1.882, 0.0, -0.238],
                ],
            )
        )
    )

    locators = [wall.locator_id for wall in rooms[0].walls]
    assert "uuid-1::tier-wall::0:bevel" not in locators
    closure_walls = [
        wall
        for wall in rooms[0].walls
        if wall.locator_id.startswith("uuid-1::tier-wall::0:bevel:rect-closure:")
    ]
    assert len(closure_walls) == 2
    assert all(wall.synthetic for wall in closure_walls)

    spans = []
    for wall in closure_walls:
        c0, c1 = wall.corners[0], wall.corners[1]
        spans.append(round(((c1.x - c0.x) ** 2 + (c1.z - c0.z) ** 2) ** 0.5, 3))
    assert sorted(spans) == [0.764, 0.875]


def test_assemble_rooms_snaps_short_parallel_cap_to_longer_outer_run_before_dedup():
    cap = ExtractedWall(
        id="inner-cap",
        source="test",
        corners=[
            [-4.638124996850417, 0.0, 4.8381974329127155],
            [-5.344678548073312, 0.0, 5.239770124551351],
            [-5.344678548073312, 2.0, 5.239770124551351],
            [-4.638124996850417, 2.0, 4.8381974329127155],
        ],
    )
    outer = ExtractedWall(
        id="outer-run",
        source="test",
        corners=[
            [-7.331, 0.0, 4.463],
            [-5.457, 0.0, 3.398],
            [-5.457, 2.0, 3.398],
            [-7.331, 2.0, 4.463],
        ],
    )

    rooms = assemble_rooms(
        _model(
            _room(
                walls=[cap, outer],
                floor=[
                    [-4.638124996850417, 0.0, 4.8381974329127155],
                    [-5.344678548073312, 0.0, 5.239770124551351],
                    [-7.331, 0.0, 4.463],
                    [-5.457, 0.0, 3.398],
                ],
            )
        )
    )

    by_locator = {wall.locator_id: wall for wall in rooms[0].walls}
    assert "uuid-1::tier-wall::0:outer-run" in by_locator
    if "uuid-1::tier-wall::0:inner-cap" in by_locator:
        snapped = by_locator["uuid-1::tier-wall::0:inner-cap"]
        outer_wall = by_locator["uuid-1::tier-wall::0:outer-run"]
        a, b = outer_wall.corners[0], outer_wall.corners[1]
        dx = b.x - a.x
        dz = b.z - a.z
        length = (dx * dx + dz * dz) ** 0.5
        for corner in snapped.corners:
            distance = abs(dx * (corner.z - a.z) - dz * (corner.x - a.x)) / length
            assert distance < 1e-9


def test_assemble_rooms_synthesises_long_missing_exterior_perimeter_wall():
    floor = [[0, 0, 0], [3, 0, 0], [3, 0, 3], [0, 0, 3]]
    walls = [
        ExtractedWall(
            id="south",
            source="test",
            corners=[[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0]],
        ),
        ExtractedWall(
            id="north",
            source="test",
            corners=[[3, 0, 3], [0, 0, 3], [0, 2, 3], [3, 2, 3]],
        ),
        ExtractedWall(
            id="west",
            source="test",
            corners=[[0, 0, 3], [0, 0, 0], [0, 2, 0], [0, 2, 3]],
        ),
    ]

    rooms = assemble_rooms(_model(_room(walls=walls, floor=floor)))

    synth = [
        wall
        for wall in rooms[0].walls
        if wall.locator_id == "uuid-1::tier-wall::0:perimeter-synth:1"
    ]
    assert len(synth) == 1
    assert synth[0].synthetic is True
    xz = {(round(c.x, 6), round(c.z, 6)) for c in synth[0].corners}
    assert xz == {(3.0, 0.0), (3.0, 3.0)}


def test_assemble_rooms_does_not_synthesise_two_meter_perimeter_wall():
    floor = [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]]
    walls = [
        ExtractedWall(
            id="south",
            source="test",
            corners=[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        ),
        ExtractedWall(
            id="north",
            source="test",
            corners=[[2, 0, 2], [0, 0, 2], [0, 2, 2], [2, 2, 2]],
        ),
        ExtractedWall(
            id="west",
            source="test",
            corners=[[0, 0, 2], [0, 0, 0], [0, 2, 0], [0, 2, 2]],
        ),
    ]

    rooms = assemble_rooms(_model(_room(walls=walls, floor=floor)))

    assert all("perimeter-synth" not in wall.locator_id for wall in rooms[0].walls)


def test_assemble_rooms_precollects_quad_cutouts_that_may_touch_wall_edges(caplog):
    wall = ExtractedWall(
        id="left",
        source="test",
        corners=[[0, 0, 2], [0, 0, 0], [0, 2, 0], [0, 2, 2]],
    )
    door = ExtractedElement(
        id="door",
        source="test",
        corners=[[0, 0.2, 0.6], [0, 0.2, 1.0], [0, 1.4, 1.0], [0, 1.4, 0.6]],
    )
    window = ExtractedElement(
        id="window",
        source="test",
        corners=[[0, 0.8, 1.2], [0, 0.8, 1.6], [0, 1.4, 1.6], [0, 1.4, 1.2]],
    )
    sill_touching = ExtractedElement(
        id="sill-touching",
        source="test",
        corners=[[0, 0.2, 0.0], [0, 0.2, 0.4], [0, 1.0, 0.4], [0, 1.0, 0.0]],
    )
    protruding = ExtractedElement(
        id="protruding",
        source="test",
        corners=[[0, 0.2, -0.2], [0, 0.2, 0.3], [0, 1.0, 0.3], [0, 1.0, -0.2]],
    )
    non_quad = ExtractedElement(
        id="triangle",
        source="test",
        corners=[[0, 0.2, 0.6], [0, 0.2, 1.0], [0, 1.4, 1.0]],
    )

    with caplog.at_level(
        logging.WARNING, logger="reconcile_tiers.assemble.walls_to_rooms"
    ):
        rooms = assemble_rooms(
            _model(
                _room(
                    walls=[wall],
                    doors=[door, sill_touching, protruding, non_quad],
                    windows=[window],
                )
            )
        )

    room = rooms[0]
    assert len(room.doors) == 3
    assert len(room.windows) == 1
    assert "Skipping non-quad door opening" in caplog.text
    assert "id=triangle" in caplog.text
    assert "corner_count=3" in caplog.text
    cutouts = [
        [[corner.x, corner.y, corner.z] for corner in cutout.corners]
        for cutout in room.walls[0].cutouts
    ]
    assert [
        coord for cutout in cutouts for corner in cutout for coord in corner
    ] == pytest.approx(
        [
            coord
            for cutout in [window.corners, door.corners, sill_touching.corners]
            for corner in cutout
            for coord in corner
        ]
    )


def test_assemble_rooms_clamps_cutout_to_actual_wall_polygon_not_bbox():
    wall = ExtractedWall(
        id="slanted-head",
        source="test",
        # Vertical wall in the X=0 plane with a sloped top edge:
        # y_top = 1 + 0.5*z. A bbox clamp would allow y up to 2 everywhere,
        # which is not the real wall face.
        corners=[[0, 0, 0], [0, 0, 2], [0, 2, 2], [0, 1, 0]],
    )
    window = ExtractedElement(
        id="near-sloped-head",
        source="test",
        # The left head corner is 5 mm above the sloped wall boundary. It is
        # inside the assignment tolerance, but must be clamped onto that edge.
        corners=[[0, 0.2, 0.2], [0, 0.2, 0.6], [0, 1.105, 0.6], [0, 1.105, 0.2]],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall], windows=[window])))

    cutout = [
        [corner.x, corner.y, corner.z]
        for corner in rooms[0].walls[0].cutouts[0].corners
    ]
    head_left = min(
        (corner for corner in cutout if corner[1] > 1.0), key=lambda corner: corner[2]
    )
    assert head_left[1] < 1.105
    assert head_left[1] == pytest.approx(1.0 + 0.5 * head_left[2], abs=1e-9)


def test_assemble_rooms_clamps_visible_parented_door_to_wall_polygon():
    wall = ExtractedWall(
        id="door-host",
        source="test",
        corners=[[0, 0, 0], [0, 0, 2], [0, 2, 2], [0, 2, 0]],
    )
    door = ExtractedElement(
        id="door",
        source="test",
        parent_wall_id="door-host",
        corners=[
            [0, -0.5, 0.25],
            [0, -0.5, 1.25],
            [0, 1.5, 1.25],
            [0, 1.5, 0.25],
        ],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall], doors=[door])))

    visible_door = [
        [corner.x, corner.y, corner.z] for corner in rooms[0].doors[0].corners
    ]
    cutout = [
        [corner.x, corner.y, corner.z]
        for corner in rooms[0].walls[0].cutouts[0].corners
    ]
    expected = [[0, 0, 0.25], [0, 0, 1.25], [0, 1.5, 1.25], [0, 1.5, 0.25]]
    assert [coord for corner in visible_door for coord in corner] == pytest.approx(
        [coord for corner in expected for coord in corner]
    )
    assert [coord for corner in cutout for coord in corner] == pytest.approx(
        [coord for corner in visible_door for coord in corner]
    )


def test_assemble_rooms_reprojects_inside_cutout_to_wall_plane():
    wall = ExtractedWall(
        id="window-host",
        source="test",
        corners=[[0, 0, 0], [0, 0, 4], [0, 3, 4], [0, 3, 0]],
    )
    window = ExtractedElement(
        id="win",
        source="test",
        parent_wall_id="window-host",
        corners=[
            [0.08, 1.0, 1.0],
            [0.08, 1.0, 2.0],
            [0.08, 2.0, 2.0],
            [0.08, 2.0, 1.0],
        ],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall], windows=[window])))

    cutout = rooms[0].walls[0].cutouts[0]
    visible_window = rooms[0].windows[0]
    assert [c.x for c in cutout.corners] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert [c.x for c in visible_window.corners] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert sorted((c.y, c.z) for c in cutout.corners) == sorted(
        [
            (1.0, 1.0),
            (1.0, 2.0),
            (2.0, 2.0),
            (2.0, 1.0),
        ]
    )


def test_reclip_cutouts_reprojects_in_bounds_quads_to_wall_plane():
    wall_corners = [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 3.0, 4.0], [0.0, 3.0, 0.0]]
    cutout = Quad(
        corners=[
            Vec3(0.15, 1.0, 1.0),
            Vec3(0.15, 1.0, 2.0),
            Vec3(0.15, 2.0, 2.0),
            Vec3(0.15, 2.0, 1.0),
        ]
    )
    out = reclip_cutouts_to_wall(wall_corners, [cutout])
    assert len(out) == 1
    assert out[0] is not cutout
    assert [
        coord for c in out[0].corners for coord in (c.x, c.y, c.z)
    ] == pytest.approx([0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0])


def test_reclip_cutouts_drops_quads_fully_outside():
    wall_corners = [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 3.0, 4.0], [0.0, 3.0, 0.0]]
    # Cutout entirely above the wall.
    cutout = Quad(
        corners=[
            Vec3(0.0, 5.0, 1.0),
            Vec3(0.0, 5.0, 2.0),
            Vec3(0.0, 6.0, 2.0),
            Vec3(0.0, 6.0, 1.0),
        ]
    )
    assert reclip_cutouts_to_wall(wall_corners, [cutout]) == []


def test_orphaned_window_with_parent_wall_id_attaches_despite_plane_drift():
    """A window whose parent wall has drifted >5cm (past PLANE_EPS_M) must
    still get attached as a cutout when its parent_wall_id matches a real
    wall — the scan link is ground truth and beats the tolerance gate."""
    # Wall in plane x=0, drifted: actual corners at x=0, opening at x=0.12.
    wall = ExtractedWall(
        id="host",
        source="test",
        corners=[[0, 0, 0], [0, 0, 4], [0, 3, 4], [0, 3, 0]],
    )
    # Window plane is 12 cm off the wall plane → first pass fails the 5 cm
    # gate (PLANE_EPS_M = 0.05) and the 10 cm parent-clamp gate too.
    window = ExtractedElement(
        id="win",
        source="test",
        parent_wall_id="host",
        corners=[
            [0.12, 1.0, 1.0],
            [0.12, 1.0, 2.0],
            [0.12, 2.0, 2.0],
            [0.12, 2.0, 1.0],
        ],
    )
    rooms = assemble_rooms(_model(_room(walls=[wall], windows=[window])))
    cutouts = rooms[0].walls[0].cutouts
    assert len(cutouts) == 1, "orphaned window with parent_wall_id must attach"
    # Cutout's wall-frame projection (Y, Z) must match the original window
    # rectangle, and its perpendicular offset must be removed so the renderer
    # receives a true coplanar wall-with-hole mesh.
    assert [c.x for c in cutouts[0].corners] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    yz = sorted((c.y, c.z) for c in cutouts[0].corners)
    assert yz == sorted([(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)])


def test_orphaned_window_no_parent_link_uses_xz_containment_fallback():
    """A window with no parent_wall_id but XZ-contained inside a wall must
    still attach when the wall plane has drifted past tolerance."""
    wall = ExtractedWall(
        id="host",
        source="test",
        corners=[[0, 0, 0], [0, 0, 4], [0, 3, 4], [0, 3, 0]],
    )
    window = ExtractedElement(
        id="win",
        source="test",
        parent_wall_id=None,
        corners=[
            [0.15, 1.0, 1.0],
            [0.15, 1.0, 2.0],
            [0.15, 2.0, 2.0],
            [0.15, 2.0, 1.0],
        ],
    )
    rooms = assemble_rooms(_model(_room(walls=[wall], windows=[window])))
    cutouts = rooms[0].walls[0].cutouts
    assert len(cutouts) == 1, "orphan with XZ containment must attach to that wall"
    assert [c.x for c in cutouts[0].corners] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_orphaned_opening_far_from_wall_plane_does_not_attach_by_projection_only():
    """Projected containment alone is not enough to attach an orphan opening.

    A stale parent link can leave a door with no surviving host wall. If its
    coordinates happen to project inside an unrelated wall frame, attaching it
    would punch a diagonal cutout across that wall in the viewer.
    """
    wall = ExtractedWall(
        id="candidate",
        source="test",
        corners=[[0, 0, 0], [0, 0, 4], [0, 3, 4], [0, 3, 0]],
    )
    door = ExtractedElement(
        id="door",
        source="test",
        parent_wall_id="missing-host",
        corners=[
            [2.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 1.0],
        ],
    )

    rooms = assemble_rooms(_model(_room(walls=[wall], doors=[door])))

    assert rooms[0].walls[0].cutouts == []
    assert len(rooms[0].doors) == 1


def test_reclip_cutouts_clamps_partial_overlap_to_wall_outline():
    """A gable-clipped wall with a triangular outline; the cutout pokes above
    the new wall top. After reclip, the cutout's top corners snap onto the
    wall's sloped edge — schema-valid 4-corner quad, every corner on or inside
    the new wall outline."""
    # Wall is a triangle (gable-clipped): apex at (z=2, y=3), base y=0.
    wall_corners = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0],
        [0.0, 3.0, 2.0],
    ]
    # Cutout originally a 1x2 rectangle whose top y=2.5 sits above the slope
    # at z=1 (slope y at z=1 is 3*1/2=1.5).
    cutout = Quad(
        corners=[
            Vec3(0.0, 0.5, 0.5),
            Vec3(0.0, 0.5, 1.5),
            Vec3(0.0, 2.5, 1.5),
            Vec3(0.0, 2.5, 0.5),
        ]
    )
    out = reclip_cutouts_to_wall(wall_corners, [cutout])
    assert len(out) == 1
    # Every corner must lie on or inside the wall triangle (in YZ since x=0).
    # The triangle has y >= 0, y <= 3*(1-|z-2|/2), i.e. y/3 + |z-2|/2 <= 1.
    for c in out[0].corners:
        # y-positivity tolerance for floating-point edge points
        assert c.y >= -1e-9
        assert c.y / 3.0 + abs(c.z - 2.0) / 2.0 <= 1.0 + 1e-6, (
            f"corner ({c.y:.3f},{c.z:.3f}) outside wall triangle"
        )
