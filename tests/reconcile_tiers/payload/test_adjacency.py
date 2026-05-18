"""Tests for the last-step adjacency tagger."""

from __future__ import annotations

import pytest

from reconcile_tiers.payload.adjacency import tag_payload
from reconcile_tiers.payload.schema import (
    AdjacencyKind,
    CeilingPiece,
    CeilingSource,
    DormerFace,
    DormerFaceKind,
    GableClosure,
    GableClosureKind,
    GapKind,
    GapPiece,
    GapScope,
    HorizontalLid,
    KneeWall,
    KneeWallKind,
    Plane,
    Quad,
    RoofType,
    Room,
    TierClassification,
    TierPayload,
    Vec3,
    Wall,
)


def _square_lid(
    y: float, *, size: float = 4.0, x0: float = 0.0, z0: float = 0.0
) -> HorizontalLid:
    return HorizontalLid(
        corners=[
            Vec3(x0, y, z0 + size),
            Vec3(x0 + size, y, z0 + size),
            Vec3(x0 + size, y, z0),
            Vec3(x0, y, z0),
        ]
    )


def _outward_wall(
    *,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    y_low: float,
    y_high: float,
    locator_id: str = "w",
) -> Wall:
    """Wall whose outward normal (Newell on these corners) points to the +X side
    when (x0,z0)→(x1,z1) runs along +Z, etc. Order: bottom-start, top-start,
    top-end, bottom-end so the Newell normal points perpendicular and outward
    from the room interior assumed to lie on the side opposite the corners.
    """
    return Wall(
        corners=[
            Vec3(x0, y_low, z0),
            Vec3(x0, y_high, z0),
            Vec3(x1, y_high, z1),
            Vec3(x1, y_low, z1),
        ],
        descent_strip=None,
        uplift_strip=None,
        cutouts=[],
        locator_id=locator_id,
    )


def _classification(
    roof_type: RoofType = RoofType.FLAT, n_stories: int = 1
) -> TierClassification:
    return TierClassification(
        tier=1,
        tier_label="t",
        roof_type=roof_type,
        n_stories=n_stories,
        n_rooms=1,
        n_oblique=0,
        n_flat=0,
        has_half_height=False,
        has_gable=False,
    )


def _empty_payload(rooms: list[Room], **kw) -> TierPayload:
    return TierPayload(
        schema_version="1",
        uuid="t",
        address=None,
        building_center=Vec3(0.0, 0.0, 0.0),
        classification=_classification(),
        rooms=rooms,
        gaps=kw.get("gaps", []),
        ceiling=kw.get("ceiling", []),
        knee_walls=kw.get("knee_walls", []),
        dormer_faces=kw.get("dormer_faces", []),
        gable_closures=kw.get("gable_closures", []),
    )


def _heated_room(
    story: int,
    *,
    heating: str | None = "radiators",
    x0: float = 0.0,
    z0: float = 0.0,
    size: float = 4.0,
    y: float = 0.0,
) -> Room:
    floor = _square_lid(y, size=size, x0=x0, z0=z0)
    walls = [
        _outward_wall(
            x0=x0,
            z0=z0,
            x1=x0 + size,
            z1=z0,
            y_low=y,
            y_high=y + 2.5,
            locator_id="w-south",
        ),
        _outward_wall(
            x0=x0 + size,
            z0=z0,
            x1=x0 + size,
            z1=z0 + size,
            y_low=y,
            y_high=y + 2.5,
            locator_id="w-east",
        ),
        _outward_wall(
            x0=x0 + size,
            z0=z0 + size,
            x1=x0,
            z1=z0 + size,
            y_low=y,
            y_high=y + 2.5,
            locator_id="w-north",
        ),
        _outward_wall(
            x0=x0,
            z0=z0 + size,
            x1=x0,
            z1=z0,
            y_low=y,
            y_high=y + 2.5,
            locator_id="w-west",
        ),
    ]
    return Room(
        story=story,
        floor=[floor],
        walls=walls,
        doors=[],
        windows=[],
        locator_id=f"r{story}",
        heating=heating,
    )


# ---------- Pass 2: kind defaults ----------


def test_knee_wall_default_is_unheated_attic():
    room = _heated_room(0)
    knee = KneeWall(
        corners=[
            Vec3(0.0, 2.5, 0.0),
            Vec3(4.0, 2.5, 0.0),
            Vec3(4.0, 4.0, 0.0),
            Vec3(0.0, 4.0, 0.0),
        ],
        kind=KneeWallKind.KNEE,
        locator_id="kw0",
    )
    payload = _empty_payload([room], knee_walls=[knee])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.knee_walls[0].adjacency == AdjacencyKind.UNHEATED_ATTIC


def test_dormer_face_default_is_external_air():
    room = _heated_room(0)
    df = DormerFace(
        corners=[
            Vec3(0.0, 2.5, 0.0),
            Vec3(1.0, 2.5, 0.0),
            Vec3(1.0, 3.0, 0.0),
            Vec3(0.0, 3.0, 0.0),
        ],
        kind=DormerFaceKind.DORMER_HEADER,
        locator_id="df0",
    )
    payload = _empty_payload([room], dormer_faces=[df])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.dormer_faces[0].adjacency == AdjacencyKind.EXTERNAL_AIR


def test_gable_closure_default_is_external_air():
    room = _heated_room(0)
    gc = GableClosure(
        corners=[Vec3(0.0, 2.5, 0.0), Vec3(4.0, 2.5, 0.0), Vec3(2.0, 4.0, 0.0)],
        kind=GableClosureKind.UPPER,
        locator_id="gc0",
    )
    payload = _empty_payload([room], gable_closures=[gc])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.gable_closures[0].adjacency == AdjacencyKind.EXTERNAL_AIR


# ---------- Pass 3: floor (table-driven ground-slab) ----------


@pytest.mark.parametrize(
    "story,has_basement,heating,expected",
    [
        (0, False, "radiators", AdjacencyKind.GROUND_SLAB),
        (0, False, "floorHeating", AdjacencyKind.GROUND_SLAB_UFH),
        (0, False, "radiatorsAndFloorHeating", AdjacencyKind.GROUND_SLAB_UFH),
        (
            0,
            True,
            "radiators",
            AdjacencyKind.GROUND_SLAB,
        ),  # heated basement still on grade
        (1, False, "radiators", AdjacencyKind.INTERNAL_TO_HEATED),
        (2, False, "radiators", AdjacencyKind.INTERNAL_TO_HEATED),
        (0, False, "unheated", AdjacencyKind.INTERNAL_TO_UNHEATED_HOST),
    ],
)
def test_floor_tags(story, has_basement, heating, expected):
    rooms = [_heated_room(s, y=s * 3.0) for s in range(story + 1)]
    rooms[story] = _heated_room(story, heating=heating, y=story * 3.0)
    payload = _empty_payload(rooms)
    out = tag_payload(payload, has_basement=has_basement, oblique_surface_corners=[])
    floor_pieces = out.rooms[story].floor
    assert all(piece.adjacency == expected for piece in floor_pieces)


def test_unheated_basement_floor_above_unheated_basement():
    basement = _heated_room(0, heating="unheated", y=0.0)
    ground = _heated_room(1, heating="radiators", y=3.0)
    payload = _empty_payload([basement, ground])
    out = tag_payload(payload, has_basement=True, oblique_surface_corners=[])
    assert all(
        piece.adjacency == AdjacencyKind.UNHEATED_BASEMENT_FLOOR
        for piece in out.rooms[1].floor
    )
    assert all(
        piece.adjacency == AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
        for piece in out.rooms[0].floor
    )


def test_floor_above_heated_basement_is_internal():
    basement = _heated_room(0, heating="radiators", y=0.0)
    ground = _heated_room(1, heating="radiators", y=3.0)
    payload = _empty_payload([basement, ground])
    out = tag_payload(payload, has_basement=True, oblique_surface_corners=[])
    assert all(
        piece.adjacency == AdjacencyKind.INTERNAL_TO_HEATED
        for piece in out.rooms[1].floor
    )


# ---------- Pass 4: walls ----------


def _walls_by_source(walls, source_id: str) -> list:
    return [
        w
        for w in walls
        if w.locator_id == source_id or w.locator_id.startswith(f"{source_id}/")
    ]


def test_partition_wall_between_two_heated_rooms_is_internal():
    a = _heated_room(0, x0=0.0, z0=0.0, size=4.0)
    b = _heated_room(0, x0=4.0, z0=0.0, size=4.0)  # touches a along x=4
    b = Room(
        story=b.story,
        floor=b.floor,
        walls=b.walls,
        doors=[],
        windows=[],
        locator_id="rB",
        heating="radiators",
    )
    payload = _empty_payload([a, b])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    # East wall of A and west wall of B face each other; every emitted piece
    # tagged internal.
    east_a_pieces = _walls_by_source(out.rooms[0].walls, "w-east")
    west_b_pieces = _walls_by_source(out.rooms[1].walls, "w-west")
    assert east_a_pieces and all(
        w.adjacency == AdjacencyKind.INTERNAL_TO_HEATED for w in east_a_pieces
    )
    assert west_b_pieces and all(
        w.adjacency == AdjacencyKind.INTERNAL_TO_HEATED for w in west_b_pieces
    )
    # Outward-facing walls remain external.
    south_a_pieces = _walls_by_source(out.rooms[0].walls, "w-south")
    assert south_a_pieces and all(
        w.adjacency == AdjacencyKind.EXTERNAL_AIR for w in south_a_pieces
    )


def test_basement_wall_split_by_depth():
    basement = _heated_room(0, heating="radiators", y=-3.0)
    ground = _heated_room(1, heating="radiators", y=0.0)
    shallow = _outward_wall(
        x0=0.0, z0=0.0, x1=4.0, z1=0.0, y_low=-1.5, y_high=0.0, locator_id="w-shallow"
    )
    deep = _outward_wall(
        x0=4.0, z0=0.0, x1=4.0, z1=4.0, y_low=-3.0, y_high=0.0, locator_id="w-deep"
    )
    basement = Room(
        story=0,
        floor=basement.floor,
        walls=[shallow, deep],
        doors=[],
        windows=[],
        locator_id="rB",
        heating="radiators",
    )
    payload = _empty_payload([basement, ground])
    out = tag_payload(payload, has_basement=True, oblique_surface_corners=[])
    shallow_pieces = _walls_by_source(out.rooms[0].walls, "w-shallow")
    deep_pieces = _walls_by_source(out.rooms[0].walls, "w-deep")
    # Shallow wall (y range -1.5..0) sits entirely above the deep threshold;
    # it stays in one BASEMENT_WALL_GROUND_SHALLOW piece.
    assert shallow_pieces and all(
        p.adjacency == AdjacencyKind.BASEMENT_WALL_GROUND_SHALLOW
        for p in shallow_pieces
    )
    # Deep wall (y range -3..0) crosses terrain_y - 2 = -2; it splits into a
    # SHALLOW upper piece (y -2..0) and a DEEP lower piece (y -3..-2).
    deep_tags = {p.adjacency for p in deep_pieces}
    assert AdjacencyKind.BASEMENT_WALL_GROUND_SHALLOW in deep_tags
    assert AdjacencyKind.BASEMENT_WALL_GROUND_DEEP in deep_tags


def test_basement_wall_protruding_above_grade_splits_at_terrain():
    basement = _heated_room(0, heating="radiators", y=-1.0)
    ground = _heated_room(1, heating="radiators", y=1.0)
    # Wall straddles terrain_y = 1.0: y range -1..1.5.
    straddler = _outward_wall(
        x0=0.0, z0=0.0, x1=4.0, z1=0.0, y_low=-1.0, y_high=1.5, locator_id="w-straddle"
    )
    basement = Room(
        story=0,
        floor=basement.floor,
        walls=[straddler],
        doors=[],
        windows=[],
        locator_id="rB",
        heating="radiators",
    )
    payload = _empty_payload([basement, ground])
    out = tag_payload(payload, has_basement=True, oblique_surface_corners=[])
    pieces = _walls_by_source(out.rooms[0].walls, "w-straddle")
    tags = {p.adjacency for p in pieces}
    assert AdjacencyKind.EXTERNAL_AIR in tags  # above-grade strip
    assert AdjacencyKind.BASEMENT_WALL_GROUND_SHALLOW in tags  # below-grade strip


def test_basement_split_clips_cutouts_to_each_piece_band():
    """A window cutout that straddles a basement Y split must be clipped to
    each piece's Y band — never inherited unclipped (would render as a
    visible band over the window in the viewer; see corpus measurement at
    616/727 split-walls-with-cutouts buggy before this fix)."""
    basement = _heated_room(0, heating="radiators", y=-1.0)
    ground = _heated_room(1, heating="radiators", y=1.0)
    # Wall straddles terrain_y = 1.0: y range -1..2. One window cutout
    # straddles that split: y range 0.5..1.5.
    wall = _outward_wall(
        x0=0.0, z0=0.0, x1=4.0, z1=0.0, y_low=-1.0, y_high=2.0, locator_id="w-host"
    )
    cutout = Quad(
        corners=[
            Vec3(1.0, 0.5, 0.0),
            Vec3(1.0, 1.5, 0.0),
            Vec3(2.0, 1.5, 0.0),
            Vec3(2.0, 0.5, 0.0),
        ]
    )
    wall = Wall(
        corners=wall.corners,
        descent_strip=None,
        uplift_strip=None,
        cutouts=[cutout],
        locator_id=wall.locator_id,
    )
    basement = Room(
        story=0,
        floor=basement.floor,
        walls=[wall],
        doors=[],
        windows=[],
        locator_id="rB",
        heating="radiators",
    )
    out = tag_payload(
        _empty_payload([basement, ground]),
        has_basement=True,
        oblique_surface_corners=[],
    )

    pieces = _walls_by_source(out.rooms[0].walls, "w-host")
    assert len(pieces) == 2, "wall must split at terrain_y=1.0 into two pieces"
    for piece in pieces:
        wy_lo = min(c.y for c in piece.corners)
        wy_hi = max(c.y for c in piece.corners)
        assert piece.cutouts, f"piece {piece.locator_id} lost its cutout"
        for cut in piece.cutouts:
            cy_lo = min(c.y for c in cut.corners)
            cy_hi = max(c.y for c in cut.corners)
            assert cy_lo >= wy_lo - 1e-6, (
                f"{piece.locator_id}: cutout y_min {cy_lo:.3f} below piece y_min "
                f"{wy_lo:.3f}"
            )
            assert cy_hi <= wy_hi + 1e-6, (
                f"{piece.locator_id}: cutout y_max {cy_hi:.3f} above piece y_max "
                f"{wy_hi:.3f}"
            )


def test_basement_split_drops_cutouts_outside_piece_band():
    """A cutout fully outside a piece's band must not appear on that piece."""
    basement = _heated_room(0, heating="radiators", y=-1.0)
    ground = _heated_room(1, heating="radiators", y=1.0)
    wall = _outward_wall(
        x0=0.0, z0=0.0, x1=4.0, z1=0.0, y_low=-1.0, y_high=2.0, locator_id="w-host"
    )
    # Cutout sits entirely above the split (terrain_y=1.0) — should appear on
    # the upper piece only, not the lower.
    cutout = Quad(
        corners=[
            Vec3(1.0, 1.2, 0.0),
            Vec3(1.0, 1.8, 0.0),
            Vec3(2.0, 1.8, 0.0),
            Vec3(2.0, 1.2, 0.0),
        ]
    )
    wall = Wall(
        corners=wall.corners,
        descent_strip=None,
        uplift_strip=None,
        cutouts=[cutout],
        locator_id=wall.locator_id,
    )
    basement = Room(
        story=0,
        floor=basement.floor,
        walls=[wall],
        doors=[],
        windows=[],
        locator_id="rB",
        heating="radiators",
    )
    out = tag_payload(
        _empty_payload([basement, ground]),
        has_basement=True,
        oblique_surface_corners=[],
    )

    pieces = _walls_by_source(out.rooms[0].walls, "w-host")
    pieces_by_band = {min(c.y for c in p.corners): p for p in pieces}
    lower = pieces_by_band[min(pieces_by_band)]
    upper = pieces_by_band[max(pieces_by_band)]
    assert lower.cutouts == [], "below-grade piece must drop the above-grade cutout"
    assert len(upper.cutouts) == 1


def test_adjacency_dedups_overlapping_wall_fragments_after_split():
    basement = _heated_room(0, heating="radiators", y=-3.0)
    ground = _heated_room(1, heating="radiators", y=0.0)
    long = _outward_wall(
        x0=0.0,
        z0=0.0,
        x1=4.0,
        z1=0.0,
        y_low=-3.0,
        y_high=0.0,
        locator_id="w-long",
    )
    subset = _outward_wall(
        x0=1.0,
        z0=0.0,
        x1=3.0,
        z1=0.0,
        y_low=-3.0,
        y_high=0.0,
        locator_id="w-subset",
    )
    basement = Room(
        story=0,
        floor=basement.floor,
        walls=[long, subset],
        doors=[],
        windows=[],
        locator_id="rB",
        heating="radiators",
    )

    out = tag_payload(
        _empty_payload([basement, ground]),
        has_basement=True,
        oblique_surface_corners=[],
    )

    assert _walls_by_source(out.rooms[0].walls, "w-long")
    assert not _walls_by_source(out.rooms[0].walls, "w-subset")


# ---------- Pass 4: ceilings (oblique-above test) ----------


def _flat_ceiling_at_centroid(
    cx: float,
    cz: float,
    *,
    source: CeilingSource = CeilingSource.FLAT_CEILING,
    y: float = 2.5,
    size: float = 4.0,
) -> CeilingPiece:
    return CeilingPiece(
        corners=[
            Vec3(cx - size / 2, y, cz - size / 2),
            Vec3(cx + size / 2, y, cz - size / 2),
            Vec3(cx + size / 2, y, cz + size / 2),
            Vec3(cx - size / 2, y, cz + size / 2),
        ],
        holes=[],
        plane=Plane(0.0, 1.0, 0.0, y),
        source=source,
        arrangement_cell_id=None,
        locator_id="ceil",
    )


def test_top_story_flat_ceiling_no_oblique_above_is_external_air():
    room = _heated_room(0)
    ceiling = _flat_ceiling_at_centroid(2.0, 2.0, source=CeilingSource.FLAT_CEILING)
    payload = _empty_payload([room], ceiling=[ceiling])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.ceiling[0].adjacency == AdjacencyKind.EXTERNAL_AIR


def test_top_story_flat_ceiling_under_oblique_is_unheated_attic():
    room = _heated_room(0)
    ceiling = _flat_ceiling_at_centroid(2.0, 2.0, source=CeilingSource.FLAT_CEILING)
    oblique = [
        [0.0, 3.0, 0.0],
        [4.0, 3.0, 0.0],
        [4.0, 5.0, 4.0],
        [0.0, 5.0, 4.0],
    ]
    payload = _empty_payload([room], ceiling=[ceiling])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[oblique])
    assert out.ceiling[0].adjacency == AdjacencyKind.UNHEATED_ATTIC


def test_top_story_sloped_ceiling_is_external_air_even_under_oblique():
    room = _heated_room(0)
    ceiling = _flat_ceiling_at_centroid(2.0, 2.0, source=CeilingSource.COMPUTED_OBLIQUE)
    oblique = [
        [0.0, 3.0, 0.0],
        [4.0, 3.0, 0.0],
        [4.0, 5.0, 4.0],
        [0.0, 5.0, 4.0],
    ]
    payload = _empty_payload([room], ceiling=[ceiling])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[oblique])
    assert out.ceiling[0].adjacency == AdjacencyKind.EXTERNAL_AIR


def test_top_story_ceiling_partially_under_oblique_splits():
    # Heated room covers x in [0,8], z in [0,4]. An oblique covers only x in [0,4].
    # Result: ceiling should split into one UNHEATED_ATTIC piece (under oblique)
    # and one EXTERNAL_AIR piece (under sky).
    room = _heated_room(0, size=8.0)
    ceiling = CeilingPiece(
        corners=[
            Vec3(0.0, 2.5, 0.0),
            Vec3(8.0, 2.5, 0.0),
            Vec3(8.0, 2.5, 4.0),
            Vec3(0.0, 2.5, 4.0),
        ],
        holes=[],
        plane=Plane(0.0, 1.0, 0.0, 2.5),
        source=CeilingSource.FLAT_CEILING,
        arrangement_cell_id=None,
        locator_id="ceil-source",
    )
    oblique = [
        [0.0, 3.0, 0.0],
        [4.0, 3.0, 0.0],
        [4.0, 5.0, 4.0],
        [0.0, 5.0, 4.0],
    ]
    payload = _empty_payload([room], ceiling=[ceiling])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[oblique])
    tags = {c.adjacency for c in out.ceiling}
    assert AdjacencyKind.UNHEATED_ATTIC in tags
    assert AdjacencyKind.EXTERNAL_AIR in tags
    # Each piece's locator suffixed with /N.
    locators = {c.locator_id for c in out.ceiling}
    assert all(loc.startswith("ceil-source/") for loc in locators)


def test_partial_basement_floor_splits_into_groundslab_and_unheated_basement_floor():
    # Story 0 has an unheated basement covering only the right half of story 1's
    # footprint.
    basement = Room(
        story=0,
        floor=[_square_lid(0.0, size=4.0, x0=4.0, z0=0.0)],
        walls=[],
        doors=[],
        windows=[],
        locator_id="rB",
        heating="unheated",
    )
    # Story 1 floor spans x=0..8, z=0..4. The right half sits above the unheated
    # basement;
    # the left half sticks out beyond → that part is on grade.
    upstairs = Room(
        story=1,
        floor=[_square_lid(3.0, size=8.0, x0=0.0, z0=0.0)],
        walls=[],
        doors=[],
        windows=[],
        locator_id="rU",
        heating="radiators",
    )
    payload = _empty_payload([basement, upstairs])
    out = tag_payload(payload, has_basement=True, oblique_surface_corners=[])
    tags = {piece.adjacency for piece in out.rooms[1].floor}
    assert AdjacencyKind.UNHEATED_BASEMENT_FLOOR in tags
    assert AdjacencyKind.GROUND_SLAB in tags


def test_lower_story_ceiling_is_internal_to_heated():
    ground = _heated_room(0, y=0.0)
    upstairs = _heated_room(1, y=3.0)
    # ceiling at top of ground floor (y=2.5)
    ceiling = _flat_ceiling_at_centroid(
        2.0, 2.0, source=CeilingSource.FLAT_CEILING, y=2.5
    )
    payload = _empty_payload([ground, upstairs], ceiling=[ceiling])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.ceiling[0].adjacency == AdjacencyKind.INTERNAL_TO_HEATED


# ---------- Pass 1: unheated host filter ----------


def test_unheated_room_floor_walls_and_ceiling_drop_out_of_envelope():
    room = _heated_room(0, heating="unheated")
    ceiling = _flat_ceiling_at_centroid(2.0, 2.0)
    payload = _empty_payload([room], ceiling=[ceiling])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert all(
        p.adjacency == AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
        for p in out.rooms[0].floor
    )
    for wall in out.rooms[0].walls:
        assert wall.adjacency == AdjacencyKind.INTERNAL_TO_UNHEATED_HOST
    assert out.ceiling[0].adjacency == AdjacencyKind.INTERNAL_TO_UNHEATED_HOST


# ---------- Gap pieces ----------


def test_exterior_side_gap_is_external_air():
    room = _heated_room(0)
    gap = GapPiece(
        corners=[
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 2.5, 0.0),
            Vec3(4.0, 2.5, 4.0),
            Vec3(4.0, 0.0, 4.0),
        ],
        kind=GapKind.EXTERIOR_SIDE,
        scope=GapScope.EXTERIOR,
        locator_id="g0",
    )
    payload = _empty_payload([room], gaps=[gap])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.gaps[0].adjacency == AdjacencyKind.EXTERNAL_AIR


def test_exterior_floor_gap_on_ground_story_is_ground_slab():
    room = _heated_room(0, y=0.0)
    gap = GapPiece(
        corners=[
            Vec3(0.0, 0.0, 4.0),
            Vec3(1.0, 0.0, 4.0),
            Vec3(1.0, 0.0, 5.0),
            Vec3(0.0, 0.0, 5.0),
        ],
        kind=GapKind.EXTERIOR_FLOOR,
        scope=GapScope.EXTERIOR,
        locator_id="g0",
    )
    payload = _empty_payload([room], gaps=[gap])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.gaps[0].adjacency == AdjacencyKind.GROUND_SLAB


def test_stitch_gap_is_left_unknown():
    room = _heated_room(0)
    gap = GapPiece(
        corners=[
            Vec3(0.0, 0.0, 4.0),
            Vec3(1.0, 0.0, 4.0),
            Vec3(1.0, 0.0, 5.0),
            Vec3(0.0, 0.0, 5.0),
        ],
        kind=GapKind.STITCH,
        scope=GapScope.JUNCTION,
        locator_id="g0",
    )
    payload = _empty_payload([room], gaps=[gap])
    out = tag_payload(payload, has_basement=False, oblique_surface_corners=[])
    assert out.gaps[0].adjacency == AdjacencyKind.UNKNOWN
