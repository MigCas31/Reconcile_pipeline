"""Tests for the tile-based manifold repair pipeline.

Increment 1 scope: ``collect_room_tiles`` + ``build_room_polyhedron``.
Tests are written to grow alongside subsequent increments (orphan
detection, filler ILP, FACE_CREATION_FROM_EDGE).
"""

from __future__ import annotations

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.manifold_repair import (
    TileFace,
    build_room_polyhedron,
    collect_room_tiles,
    extract_hole_chains,
    hypothesise_single_face_fillers,
    select_fillers,
)


def _pt(x: float, y: float, z: float) -> dict:
    return {"x": x, "y": y, "z": z}


def _cube_tiles() -> list[TileFace]:
    """Six tile faces forming a unit cube centred at the origin.

    Floor at y=0 (CCW from above = CW from below = outward-facing down),
    ceiling at y=1, four walls.
    """
    floor_corners = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    ceiling_corners = (
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
    )
    # Walls walked CCW from outside (right-hand rule with outward normal).
    wall_corners = (
        # +z wall (outward normal +z)
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        # +x wall (outward normal +x)
        (
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        # -z wall (outward normal -z)
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        # -x wall (outward normal -x)
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
        ),
    )
    tiles = [
        TileFace(
            face_id=0,
            corners=floor_corners,
            plane=Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
            source="floor",
            locator_id="cube::floor",
        ),
        TileFace(
            face_id=1,
            corners=ceiling_corners,
            plane=Plane(a=0.0, b=1.0, c=0.0, d=1.0),
            source="ceiling",
            locator_id="cube::ceiling",
        ),
    ]
    for i, corners in enumerate(wall_corners):
        normal = (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
        )[i]
        d = normal[0] * corners[0][0] + normal[1] * corners[0][1] + normal[2] * corners[0][2]
        tiles.append(
            TileFace(
                face_id=2 + i,
                corners=corners,
                plane=Plane(a=normal[0], b=normal[1], c=normal[2], d=d),
                source="wall",
                locator_id=f"cube::wall:{i}",
            )
        )
    return tiles


# ---------------------------------------------------------------------------
# build_room_polyhedron
# ---------------------------------------------------------------------------


def test_build_room_polyhedron_cube_no_orphans():
    """6-tile cube → all twins paired, 0 orphan half-edges."""
    tiles = _cube_tiles()
    result = build_room_polyhedron(tiles)
    assert len(result.poly.faces) == 6
    assert len(result.poly.vertices) == 8
    assert len(result.poly.half_edges) == 24  # 4 per face
    assert result.orphan_half_edges == []
    assert result.skipped_tiles == []
    # Each face has a tile mapping.
    for face_id in range(6):
        assert face_id in result.tile_face_by_id


def test_build_room_polyhedron_missing_wall_yields_orphans():
    """5-tile cube (one wall removed) → 4 orphan half-edges (the hole boundary)."""
    tiles = _cube_tiles()
    tiles_minus_one = tiles[:5]  # drop the last wall
    result = build_room_polyhedron(tiles_minus_one)
    assert len(result.poly.faces) == 5
    assert len(result.orphan_half_edges) == 4
    # The orphan half-edges should form a closed loop (a square hole).
    orphan_endpoints = {
        (he.origin.id, he.next.origin.id)
        for he in result.orphan_half_edges
        if he.next is not None
    }
    assert len(orphan_endpoints) == 4


def test_build_room_polyhedron_skips_degenerate():
    """A tile with <3 corners or zero area is reported, not crashed on."""
    tiles = _cube_tiles()
    bad = TileFace(
        face_id=99,
        corners=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # only 2 corners
        plane=Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
        source="floor",
        locator_id="bad",
    )
    result = build_room_polyhedron([*tiles, bad])
    assert any(reason == "fewer_than_3_corners" for _, reason in result.skipped_tiles)
    assert len(result.poly.faces) == 6  # all 6 valid cube tiles still build


def test_build_room_polyhedron_quantization_merges_close_corners():
    """Two tiles whose shared corners differ by < coord_tol must share a vertex."""
    floor = TileFace(
        face_id=0,
        corners=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ),
        plane=Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
        source="floor",
        locator_id="floor",
    )
    # A wall whose floor corners are 0.5 mm off — should still snap.
    wall = TileFace(
        face_id=1,
        corners=(
            (0.0005, 0.0, 0.0),  # 0.5 mm drift
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
        ),
        plane=Plane(a=-1.0, b=0.0, c=0.0, d=0.0),
        source="wall",
        locator_id="wall",
    )
    result = build_room_polyhedron([floor, wall], coord_tol=1e-3)
    # 1 mm tolerance snaps wall's (0.0005,0,0) → floor's (0,0,0) AND
    # wall's (0,0,1) → floor's (0,0,1). 4 floor + 4 wall - 2 shared = 6.
    assert len(result.poly.vertices) == 6


# ---------------------------------------------------------------------------
# collect_room_tiles
# ---------------------------------------------------------------------------


def _payload_with_one_room() -> dict:
    """A payload with one rectangular room at story 0 + one ceiling tile."""
    return {
        "rooms": [
            {
                "story": 0,
                "locator_id": "building::tier-room::0",
                "floor": [
                    {
                        "corners": [
                            _pt(0.0, 0.0, 0.0),
                            _pt(2.0, 0.0, 0.0),
                            _pt(2.0, 0.0, 2.0),
                            _pt(0.0, 0.0, 2.0),
                        ],
                        "plane": {"a": 0.0, "b": -1.0, "c": 0.0, "d": 0.0},
                    }
                ],
                "walls": [
                    {
                        "corners": [
                            _pt(0.0, 0.0, 0.0),
                            _pt(2.0, 0.0, 0.0),
                            _pt(2.0, 1.0, 0.0),
                            _pt(0.0, 1.0, 0.0),
                        ],
                        "plane": {"a": 0.0, "b": 0.0, "c": -1.0, "d": 0.0},
                    },
                ],
            }
        ],
        "ceiling": [
            {
                "locator_id": "building::tier-ceiling::0",
                "corners": [
                    _pt(0.0, 1.0, 0.0),
                    _pt(2.0, 1.0, 0.0),
                    _pt(2.0, 1.0, 2.0),
                    _pt(0.0, 1.0, 2.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0},
                "source": "flat_ceiling",
            },
            {
                # An unrelated ceiling tile FAR from this room — must be excluded.
                "locator_id": "building::tier-ceiling::elsewhere",
                "corners": [
                    _pt(50.0, 1.0, 50.0),
                    _pt(52.0, 1.0, 50.0),
                    _pt(52.0, 1.0, 52.0),
                    _pt(50.0, 1.0, 52.0),
                ],
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0},
                "source": "flat_ceiling",
            },
        ],
    }


def test_collect_room_tiles_picks_floor_walls_relevant_ceilings():
    payload = _payload_with_one_room()
    tiles = collect_room_tiles(payload, payload["rooms"][0])
    sources = [t.source for t in tiles]
    assert sources.count("floor") == 1
    assert sources.count("wall") == 1
    assert sources.count("ceiling") == 1  # only the relevant tile, not the far one
    # Story propagated.
    assert all(t.story == 0 for t in tiles)


def test_collect_room_tiles_face_ids_unique():
    payload = _payload_with_one_room()
    tiles = collect_room_tiles(payload, payload["rooms"][0])
    ids = [t.face_id for t in tiles]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# extract_hole_chains
# ---------------------------------------------------------------------------


def test_extract_hole_chains_no_orphans():
    """6-tile cube → 0 closed chains, 0 open chains, 0 ambiguous vertices."""
    result = build_room_polyhedron(_cube_tiles())
    extraction = extract_hole_chains(result)
    assert extraction.closed_chains == []
    assert extraction.open_chains == []
    assert extraction.ambiguous_vertices == []


def test_extract_hole_chains_missing_wall_one_closed_loop():
    """5-tile cube → 1 closed chain of 4 orphan half-edges (square hole)."""
    tiles = _cube_tiles()[:5]
    result = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(result)
    assert len(extraction.closed_chains) == 1
    chain = extraction.closed_chains[0]
    assert len(chain.half_edges) == 4
    assert len(chain.vertex_ids) == 4
    # The vertex IDs in a closed chain are unique (square boundary, 4 corners).
    assert len(set(chain.vertex_ids)) == 4
    # No open chains.
    assert extraction.open_chains == []


def test_extract_hole_chains_two_holes():
    """4 tiles (floor + ceiling + 2 opposing walls) → 2 closed chains."""
    cube = _cube_tiles()
    # Keep floor + ceiling + +z and -z walls, drop +x and -x walls.
    # That removes both x-side walls, leaving 2 separate square holes.
    tiles = [cube[0], cube[1], cube[2], cube[4]]  # floor, ceiling, +z, -z
    result = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(result)
    assert len(extraction.closed_chains) == 2
    for chain in extraction.closed_chains:
        assert len(chain.half_edges) == 4


def test_extract_hole_chains_ambiguous_pinched_topology():
    """When two orphan edges share an origin vertex (pinched topology),
    flag the vertex and stop the chain — caller handles the fallback."""
    # Build a degenerate case: one tile + one wall sharing a single vertex.
    floor = TileFace(
        face_id=0,
        corners=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ),
        plane=Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
        source="floor",
        locator_id="floor",
    )
    # A second floor tile sharing only the (0,0,0) corner — produces 8
    # orphan edges with a vertex shared between them.
    floor2 = TileFace(
        face_id=1,
        corners=(
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
        ),
        plane=Plane(a=0.0, b=-1.0, c=0.0, d=0.0),
        source="floor",
        locator_id="floor2",
    )
    result = build_room_polyhedron([floor, floor2])
    extraction = extract_hole_chains(result)
    # The shared vertex is reported as ambiguous.
    assert extraction.ambiguous_vertices  # non-empty


# ---------------------------------------------------------------------------
# hypothesise_single_face_fillers + select_fillers
# ---------------------------------------------------------------------------


def test_hypothesise_single_face_filler_for_missing_wall():
    """5-tile cube → 1 hole → 1 best-fit-plane filler covering it."""
    tiles = _cube_tiles()[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    assert len(fillers) == 1
    f = fillers[0]
    assert f.parent_hole_id == 0
    assert f.derivation == "best_fit_plane"
    assert len(f.corners) == 4  # square hole
    # Filler area should match a unit square.
    assert abs(f.area - 1.0) < 1e-3
    # Plane should be the missing wall's plane (-x wall in our cube).
    # Removed tile is index 5 (the -x wall in _cube_tiles); it has plane
    # x = 0 (a = -1 or 1, b = c = 0, d = 0). Best-fit via SVD will give
    # |a|=1, b≈0, c≈0.
    assert abs(abs(f.plane.a) - 1.0) < 1e-3
    assert abs(f.plane.b) < 1e-3
    assert abs(f.plane.c) < 1e-3


def test_hypothesise_single_face_filler_two_holes():
    """4-tile box (drop ±x walls) → 2 holes → 2 fillers."""
    cube = _cube_tiles()
    tiles = [cube[0], cube[1], cube[2], cube[4]]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    assert len(fillers) == 2
    parent_ids = {f.parent_hole_id for f in fillers}
    assert parent_ids == {0, 1}


def test_select_fillers_picks_one_per_hole():
    """Single-face hypothesis → ILP forces selection of all unique
    candidates because each hole has exactly one filler hypothesis."""
    tiles = _cube_tiles()[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    selection = select_fillers(build, extraction, fillers)
    assert len(selection.selected) == 1
    assert selection.unfilled_hole_ids == []
    assert selection.rejected == []


def test_select_fillers_no_holes_no_op():
    """Watertight cube → no fillers → empty selection."""
    build = build_room_polyhedron(_cube_tiles())
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    assert fillers == []
    selection = select_fillers(build, extraction, fillers)
    assert selection.selected == []
    assert selection.rejected == []
    assert selection.unfilled_hole_ids == []


# ---------------------------------------------------------------------------
# apply_fillers — Geniet FACE_CREATION_FROM_EDGE
# ---------------------------------------------------------------------------


def test_apply_fillers_closes_single_hole():
    """5-tile cube + 1 selected filler → polyhedron is watertight after apply."""
    from reconcile_tiers.polyhedron.manifold_repair import apply_fillers

    tiles = _cube_tiles()[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    selection = select_fillers(build, extraction, fillers)
    applied = apply_fillers(build, extraction, selection.selected)
    assert applied == 1
    # Manifold checks post-apply:
    assert len(build.poly.faces) == 6  # 5 tiles + 1 filler
    assert len(build.poly.half_edges) == 24  # 4 per face × 6
    assert build.orphan_half_edges == []
    # Every half-edge has a twin.
    assert all(he.opposite is not None for he in build.poly.half_edges)
    # Each pair of twins references each other reciprocally.
    for he in build.poly.half_edges:
        assert he.opposite.opposite is he


def test_apply_fillers_closes_two_holes():
    """4-tile box + 2 fillers → fully watertight cube."""
    from reconcile_tiers.polyhedron.manifold_repair import apply_fillers

    cube = _cube_tiles()
    tiles = [cube[0], cube[1], cube[2], cube[4]]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    selection = select_fillers(build, extraction, fillers)
    applied = apply_fillers(build, extraction, selection.selected)
    assert applied == 2
    assert len(build.poly.faces) == 6
    assert build.orphan_half_edges == []


def test_apply_fillers_no_op_when_no_selection():
    """Watertight cube + no fillers → apply is a no-op."""
    from reconcile_tiers.polyhedron.manifold_repair import apply_fillers

    build = build_room_polyhedron(_cube_tiles())
    extraction = extract_hole_chains(build)
    applied = apply_fillers(build, extraction, [])
    assert applied == 0
    assert len(build.poly.faces) == 6


# ---------------------------------------------------------------------------
# Increment 6e: filler insertion preserves strict 2-manifold invariants
# ---------------------------------------------------------------------------


def test_apply_fillers_strict_validate_single_hole():
    """5-tile cube + 1 filler → validate_polyhedron returns 0 issues
    related to topology wiring (face_orbit_mismatch, vertex_orbit_mismatch,
    missing_opposite, broken_opposite)."""
    from reconcile_tiers.polyhedron.manifold_repair import apply_fillers
    from reconcile_tiers.polyhedron.validity import validate_polyhedron

    tiles = _cube_tiles()[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    selection = select_fillers(build, extraction, fillers)
    apply_fillers(build, extraction, selection.selected)

    issues = validate_polyhedron(build.poly)
    wiring_kinds = {
        "missing_opposite",
        "broken_opposite",
        "face_orbit_open",
        "face_orbit_mismatch",
        "vertex_orbit_open",
        "vertex_orbit_mismatch",
    }
    wiring_issues = [i for i in issues if i.kind in wiring_kinds]
    assert wiring_issues == [], (
        f"filler insertion violated wiring invariants: {wiring_issues}"
    )


def test_apply_fillers_face_ids_unique_when_tile_ids_sparse():
    """When some tile face_idx fails both windings (sparse tile face ids),
    fillers must still get fresh ids — not collide with surviving tile ids.

    Regression for the face_orbit_mismatch class of defects: previously
    fillers got id = ``len(poly.faces)``, which equals an existing tile id
    when face_idx 3 (say) failed but 4..N succeeded.
    """
    from reconcile_tiers.polyhedron.manifold_repair import (
        FillerCandidate,
        apply_fillers,
        build_room_polyhedron,
    )

    # Build a normal 5-tile cube with one hole, then artificially renumber
    # the existing faces so they leave a gap in their id space.
    tiles = _cube_tiles()[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    selection = select_fillers(build, extraction, fillers)
    assert len(selection.selected) == 1

    # Simulate sparse tile ids: bump the first tile's id from 0 to 5 so the
    # set of tile ids is {1, 2, 3, 4, 5} with len = 5, max = 5.
    bumped = build.poly.faces[0]
    bumped.id = 5
    # Re-allocate the filler's face_id the same way `repair_room` does now:
    next_face_id = max((f.id for f in build.poly.faces), default=-1) + 1
    selection.selected[0] = FillerCandidate(
        face_id=next_face_id,
        parent_hole_id=selection.selected[0].parent_hole_id,
        corners=selection.selected[0].corners,
        plane=selection.selected[0].plane,
        derivation=selection.selected[0].derivation,
        area=selection.selected[0].area,
    )
    apply_fillers(build, extraction, selection.selected)
    ids = [f.id for f in build.poly.faces]
    assert len(ids) == len(set(ids)), f"duplicate face ids after fillers: {ids}"


def test_apply_fillers_strict_validate_two_holes():
    """4-tile box + 2 fillers → validate_polyhedron returns 0 wiring issues
    after sequentially inserting both fillers."""
    from reconcile_tiers.polyhedron.manifold_repair import apply_fillers
    from reconcile_tiers.polyhedron.validity import validate_polyhedron

    cube = _cube_tiles()
    tiles = [cube[0], cube[1], cube[2], cube[4]]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_single_face_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    selection = select_fillers(build, extraction, fillers)
    apply_fillers(build, extraction, selection.selected)

    issues = validate_polyhedron(build.poly)
    wiring_kinds = {
        "missing_opposite",
        "broken_opposite",
        "face_orbit_open",
        "face_orbit_mismatch",
        "vertex_orbit_open",
        "vertex_orbit_mismatch",
    }
    wiring_issues = [i for i in issues if i.kind in wiring_kinds]
    assert wiring_issues == [], (
        f"filler insertion violated wiring invariants: {wiring_issues}"
    )



def test_hypothesise_fillers_emits_best_fit_for_perpendicular_hole():
    """For a hole whose chain is perpendicular to every neighbour face's
    plane (cube fixture: -X wall removed → 4 perpendicular neighbours),
    neighbor-plane-extension projections collapse to degenerate polygons
    (zero area) and are correctly skipped — leaving the best-fit-plane
    as the only viable hypothesis."""
    from reconcile_tiers.polyhedron.manifold_repair import hypothesise_fillers

    tiles = _cube_tiles()[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    derivations = [f.derivation for f in fillers]
    assert "best_fit_plane" in derivations
    # No neighbour-extension candidates survive degenerate-area filter for
    # this fixture; that's correct behaviour, not a bug.
    assert all(d == "best_fit_plane" for d in derivations)


def test_hypothesise_fillers_emits_neighbor_extension_for_oblique_hole():
    """For a hole whose chain is NOT perpendicular to a neighbour face,
    that neighbour becomes a viable extension candidate. Uses a tilted
    fixture so the missing-face plane is not orthogonal to all four
    neighbour planes."""
    from math import cos, radians, sin

    from reconcile_tiers.polyhedron.manifold_repair import hypothesise_fillers

    # Take the standard cube and tilt the +Y wall (ceiling) by 30° about
    # the Z axis. Then drop the -X wall — the resulting hole's chain has
    # vertices at the tilted ceiling's edge, so projecting them onto
    # other neighbours' planes is no longer degenerate.
    angle = radians(30)
    c, s = cos(angle), sin(angle)
    tiles = list(_cube_tiles())
    # Replace ceiling (index 1) with a tilted version
    new_ceiling_corners = tuple(
        (
            x,
            y * c + z * s,
            -y * s + z * c,
        )
        for (x, y, z) in (
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    )
    from reconcile_tiers._core.plane import Plane

    tiles[1] = TileFace(
        face_id=1,
        corners=new_ceiling_corners,
        plane=Plane(a=0.0, b=c, c=s, d=c * 1.0),
        source="ceiling",
        locator_id="cube::tilted_ceiling",
    )
    # Drop -X wall (index 5)
    tiles = tiles[:5]
    build = build_room_polyhedron(tiles)
    extraction = extract_hole_chains(build)
    fillers = hypothesise_fillers(
        build, extraction, first_face_id=len(build.poly.faces)
    )
    derivations = [f.derivation for f in fillers]
    assert "best_fit_plane" in derivations
    assert any(d.startswith("neighbor_extension") for d in derivations)
