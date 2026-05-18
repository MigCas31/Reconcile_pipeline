import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from reconcile.inspect_building.audit import audit
from reconcile_tiers._core.plane import Plane
from reconcile_tiers.assemble.synthesis import (
    _extend_flat_to_obliques,
    _half_plane_polygon,
)
from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.build_internals.ceiling_helpers import _parse_room_idx
from reconcile_tiers.build_internals.gable_selection import _select_gable_oblique_pair
from reconcile_tiers.build_internals.io_index import (
    needs_rebuild,
    payload_json,
    write_tier_index,
)
from reconcile_tiers.build_internals.per_room_emission import _ceiling_candidates
from reconcile_tiers.build_internals.raw_snapping import (
    _raw_ceiling_crosses_upper_floor_slab,
)
from reconcile_tiers.extract.building import (
    extract_building_model,
)
from reconcile_tiers.payload.schema import (
    CeilingSource,
    DormerFaceKind,
    GapKind,
    payload_to_dict,
)
from reconcile_tiers.payload.validate import validate_payload
from reconcile_tiers.roof.roof import build_roof_model


@pytest.fixture(autouse=True)
def _force_legacy_priors_off(monkeypatch):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "0")


def _xz_piece_poly(piece):
    return Polygon(
        [(corner.x, corner.z) for corner in piece.corners],
        holes=[[(corner.x, corner.z) for corner in hole] for hole in piece.holes],
    )


def _xz_points_poly(points):
    return Polygon([(float(point[0]), float(point[2])) for point in points])


def _same_quad(corners_a, corners_b, tol: int = 4):
    points_a = {(round(c.x, tol), round(c.y, tol), round(c.z, tol)) for c in corners_a}
    points_b = {(round(c.x, tol), round(c.y, tol), round(c.z, tol)) for c in corners_b}
    return points_a == points_b


def _same_quad_within_m(corners_a, corners_b, tol_m: float):
    if len(corners_a) != len(corners_b):
        return False
    remaining = list(corners_b)
    for corner in corners_a:
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (
                (corner.x - remaining[idx].x) ** 2
                + (corner.y - remaining[idx].y) ** 2
                + (corner.z - remaining[idx].z) ** 2
            ),
        )
        best = remaining.pop(best_idx)
        distance_sq = (
            (corner.x - best.x) ** 2
            + (corner.y - best.y) ** 2
            + (corner.z - best.z) ** 2
        )
        if distance_sq > tol_m**2:
            return False
    return True


def _min_edge_m(corners) -> float:
    return min(
        math.sqrt(
            (corners[idx].x - corners[(idx + 1) % len(corners)].x) ** 2
            + (corners[idx].y - corners[(idx + 1) % len(corners)].y) ** 2
            + (corners[idx].z - corners[(idx + 1) % len(corners)].z) ** 2
        )
        for idx in range(len(corners))
    )


def _room_floor_union(payload):
    polys = [
        Polygon([(corner.x, corner.z) for corner in floor.corners])
        for room in payload.rooms
        for floor in room.floor
    ]
    return unary_union(polys)


def _room_top_coverage_ratio(payload, room_index: int) -> float:
    room = payload.rooms[room_index]
    floor = unary_union(
        [
            Polygon([(corner.x, corner.z) for corner in piece.corners])
            for piece in room.floor
        ]
    )
    top_polys = [
        Polygon([(corner.x, corner.z) for corner in piece.corners])
        for piece in payload.ceiling
    ]
    top_polys.extend(
        Polygon([(corner.x, corner.z) for corner in gap.corners])
        for gap in payload.gaps
        if gap.kind in {GapKind.GAP_CEILING, GapKind.STITCH_CEIL, GapKind.EXTERIOR_CEIL}
    )
    covered = unary_union(top_polys).intersection(floor)
    return float(covered.area) / float(floor.area)


def _synthetic_oblique(plane: Plane, x0: float, x1: float, *, azimuth: float):
    corners = [
        [x0, plane.y_at(x0, 0.0), 0.0],
        [x1, plane.y_at(x1, 0.0), 0.0],
        [x1, plane.y_at(x1, 4.0), 4.0],
        [x0, plane.y_at(x0, 4.0), 4.0],
    ]
    return SimpleNamespace(
        corners=corners,
        plane=plane,
        cluster=SimpleNamespace(avg_azimuth=azimuth, avg_incl=30.0),
    )


def _oblique_from_matching_plane(raw_corners, target_corners):
    from math import atan2, degrees, hypot

    raw_plane = Plane.fit(raw_corners)
    target_plane = Plane.fit(target_corners)
    raw_incl = degrees(atan2(hypot(raw_plane.a, raw_plane.c), abs(raw_plane.b)))
    raw_azimuth = degrees(atan2(-raw_plane.a, -raw_plane.c)) % 360.0
    return SimpleNamespace(
        corners=target_corners,
        plane=target_plane,
        cluster=SimpleNamespace(avg_azimuth=raw_azimuth, avg_incl=raw_incl),
    )


def test_build_infers_enclosed_void_room_for_unscanned_closet_regression():
    payload = build_tier_payload(
        "3ee81b85-a21f-4f55-a100-5995f36f84f9",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    assert len(payload.rooms) == 10
    assert not any(
        gap.locator_id == "3ee81b85-a21f-4f55-a100-5995f36f84f9::tier-gap::stitch:0:12"
        for gap in payload.gaps
    )

    union = _room_floor_union(payload)
    parts = [union] if isinstance(union, Polygon) else list(union.geoms)
    large_holes = [
        Polygon(ring)
        for part in parts
        for ring in part.interiors
        if Polygon(ring).area >= 0.5
    ]
    assert large_holes == []

    inferred_room = next(
        room
        for room in payload.rooms
        if room.locator_id == "3ee81b85-a21f-4f55-a100-5995f36f84f9::tier-room::9"
    )
    assert inferred_room.walls
    assert any(wall.synthetic for wall in inferred_room.walls)


def test_snap_raw_to_oblique_rejects_matching_orientation_with_wrong_height():
    from reconcile_tiers.build_internals.raw_snapping import _snap_raw_to_oblique

    raw = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]]
    target = [[p[0], p[1] - 10.0, p[2]] for p in raw]
    oblique = _oblique_from_matching_plane(raw, target)

    assert _snap_raw_to_oblique(raw, [oblique]) == raw


def test_snap_raw_to_oblique_keeps_matching_orientation_with_close_height():
    from reconcile_tiers.build_internals.raw_snapping import _snap_raw_to_oblique

    raw = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]]
    target = [[p[0], p[1] + 0.2, p[2]] for p in raw]
    oblique = _oblique_from_matching_plane(raw, target)

    snapped = _snap_raw_to_oblique(raw, [oblique])

    assert snapped != raw
    assert [round(p[1], 3) for p in snapped] == [2.2, 2.2, 4.2, 4.2]


def test_snap_raw_to_oblique_uses_slope_direction_not_plane_normal_sign():
    from reconcile_tiers.build_internals.raw_snapping import _snap_raw_to_oblique

    # Two opposing gable faces with the same pitch. The raw plane belongs to
    # the right/opposite face; using plane-normal azimuth instead of slope
    # direction matches the left face because the sign is 180 degrees off.
    raw = [[0.0, 4.0, 0.0], [4.0, 2.0, 0.0], [4.0, 2.0, 4.0], [0.0, 4.0, 4.0]]
    wrong_face = [[p[0], 2.0 + 0.5 * p[0], p[2]] for p in raw]
    right_face = [[p[0], 4.2 - 0.5 * p[0], p[2]] for p in raw]
    obliques = [
        _oblique_from_matching_plane(
            [[0.0, 2.0, 0.0], [4.0, 4.0, 0.0], [4.0, 4.0, 4.0], [0.0, 2.0, 4.0]],
            wrong_face,
        ),
        _oblique_from_matching_plane(raw, right_face),
    ]

    snapped = _snap_raw_to_oblique(raw, obliques)

    assert [round(p[1], 3) for p in snapped] == [4.2, 2.2, 2.2, 4.2]


def _gable_cross_section_for_room(piece, obliques, room):
    room_poly = _xz_points_poly(room.floor_polygon)
    selected = _select_gable_oblique_pair(obliques, room_poly)
    assert selected is not None
    selected_obliques, oblique_union = selected
    y_flat = sum(corner.y for corner in piece.corners) / len(piece.corners)
    piece_poly = _xz_piece_poly(piece)
    bounds_xs = [float(p[0]) for ob in selected_obliques for p in ob.corners] + [
        piece_poly.bounds[0],
        piece_poly.bounds[2],
    ]
    bounds_zs = [float(p[2]) for ob in selected_obliques for p in ob.corners] + [
        piece_poly.bounds[1],
        piece_poly.bounds[3],
    ]
    bbox = box(
        min(bounds_xs) - 50.0,
        min(bounds_zs) - 50.0,
        max(bounds_xs) + 50.0,
        max(bounds_zs) + 50.0,
    )
    cross_section = bbox
    for oblique in selected_obliques:
        a, b, c, d = oblique.plane.a, oblique.plane.b, oblique.plane.c, oblique.plane.d
        half = _half_plane_polygon(
            bbox, float(a), float(c), float(d) - float(b) * y_flat
        )
        assert half is not None
        cross_section = cross_section.intersection(half)
    return cross_section.intersection(oblique_union).intersection(room_poly)


def test_extend_flat_to_obliques_clips_to_selected_gable_pair():
    left = _synthetic_oblique(Plane(a=-0.4, b=1.0, c=0.0, d=2.0), 0.0, 5.0, azimuth=0.0)
    right = _synthetic_oblique(
        Plane(a=0.4, b=1.0, c=0.0, d=6.0), 5.0, 10.0, azimuth=180.0
    )
    ceiling = [[0.0, 3.0, 0.0], [10.0, 3.0, 0.0], [10.0, 3.0, 4.0], [0.0, 3.0, 4.0]]

    clipped = _extend_flat_to_obliques(ceiling, [left, right], ceiling)
    poly = _xz_points_poly(clipped)

    assert poly.area == pytest.approx(20.0)
    assert poly.bounds == pytest.approx((2.5, 0.0, 7.5, 4.0))


def test_extend_flat_to_obliques_preserves_flat_wing_outside_gable_pair():
    left = _synthetic_oblique(Plane(a=-0.4, b=1.0, c=0.0, d=2.0), 0.0, 5.0, azimuth=0.0)
    right = _synthetic_oblique(
        Plane(a=0.4, b=1.0, c=0.0, d=6.0), 5.0, 10.0, azimuth=180.0
    )
    flat_wing = [[11.0, 3.0, 0.0], [15.0, 3.0, 0.0], [15.0, 3.0, 4.0], [11.0, 3.0, 4.0]]

    assert _extend_flat_to_obliques(flat_wing, [left, right], flat_wing) == flat_wing


@pytest.mark.parametrize(
    "uuid",
    [
        "c72ad855-9e52-46f1-886d-a9f37911521f",
        "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
        "2ea3b759-e047-424c-8034-f8ee5b811fb4",
    ],
)
def test_validate_only_passes_on_cohort_uuid(uuid):
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    validate_payload(payload)


@pytest.mark.parametrize(
    "uuid",
    [
        "97571caa-1065-4ac5-a385-2ec9758026dd",
        "c87c1e25-ff00-44ec-b823-b0966c81af70",
    ],
)
def test_reported_corpus_validation_failures_build(uuid):
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    validate_payload(payload)


def test_payload_json_is_deterministic_for_cohort_uuid():
    uuid = "c72ad855-9e52-46f1-886d-a9f37911521f"

    first = payload_json(
        build_tier_payload(
            uuid, pipeline_dir=Path("pipeline-outputs"), scan_root=Path(".scan-cache")
        )
    )
    second = payload_json(
        build_tier_payload(
            uuid, pipeline_dir=Path("pipeline-outputs"), scan_root=Path(".scan-cache")
        )
    )

    assert first == second
    validate_payload(
        build_tier_payload(
            uuid, pipeline_dir=Path("pipeline-outputs"), scan_root=Path(".scan-cache")
        )
    )


def test_write_tier_index_preserves_existing_payload_corpus(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    existing_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    existing_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    new_uuid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    for uuid in [existing_a, existing_b]:
        payload_dir = pipeline_dir / uuid
        payload_dir.mkdir(parents=True)
        (payload_dir / "tier_payload.json").write_text("{}\n")

    write_tier_index(pipeline_dir, [(new_uuid, "written", None)])

    index = json.loads((pipeline_dir / "tier_index.json").read_text())
    assert index == {"buildings": [existing_a, existing_b, new_uuid]}


def test_roof_arrangement_ceilings_do_not_project_above_observed_roof_evidence():
    uuid = "0d3f2993-8386-4130-8f1c-b2938c410828"
    pipeline_dir = Path("pipeline-outputs")
    scan_root = Path(".scan-cache")

    model = extract_building_model(uuid, pipeline_dir, scan_root)
    observed_y = [
        float(point[1])
        for room in model.rooms
        for seq in [room.floor_polygon, room.ceiling_polygon]
        for point in seq
    ]
    observed_y.extend(
        float(point[1])
        for room in model.rooms
        for wall in room.walls_computed
        for point in wall.corners
    )
    observed_y.extend(
        float(point[1])
        for room in model.rooms
        for raw in room.raw_ceiling_planes
        for point in raw.corners
    )
    payload = build_tier_payload(uuid, pipeline_dir=pipeline_dir, scan_root=scan_root)
    roof_arrangement_y = [
        corner.y
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        for corner in piece.corners
    ]

    assert roof_arrangement_y
    assert max(roof_arrangement_y) <= max(observed_y) + 0.6


def test_roof_arrangement_ceilings_do_not_cross_room_stories():
    """Story consistency: each emitted ROOF_ARRANGEMENT cell sits in a room
    whose story matches the surface's dominant_story.

    The pre-Phase-5 implementation had a global cluster that produced
    `surface.dominant_story` from a building-wide vote; the test pinned
    specific cell IDs (cell:0/1/2:room:4:0) to verify that vote dropped
    cross-story emissions for room 4. Phase 5 per-wing fan-out reorders
    cell IDs (each wing makes its own surface index pool), so we can no
    longer pin specific cell IDs. The behavioural invariant — every
    emitted cell's story matches the surface's dominant_story — is what
    we still verify.
    """
    uuid = "52f91e67-3891-4729-8bf3-be2c0a6a0d04"
    pipeline_dir = Path("pipeline-outputs")
    scan_root = Path(".scan-cache")

    model = extract_building_model(uuid, pipeline_dir, scan_root)
    rooms_by_index = {room.index: room for room in model.rooms}
    split_by_cell = {
        surface.arrangement_cell_id: surface
        for surface in build_roof_model(model).oblique_split
    }
    payload = build_tier_payload(uuid, pipeline_dir=pipeline_dir, scan_root=scan_root)

    roof_arrangement_cells = {
        piece.arrangement_cell_id
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
    }

    for cell_id in roof_arrangement_cells:
        room_idx = _parse_room_idx(cell_id)
        if room_idx is None:
            continue
        room = rooms_by_index[room_idx]
        surface = split_by_cell[cell_id]
        assert room.story == surface.dominant_story, cell_id


def test_top_story_flat_rooms_keep_flat_ceiling_under_gable():
    """Flat-ceiling rooms under a gable keep their horizontal flat ceiling.

    Earlier the pipeline absorbed the gable into these rooms as their
    ceiling (suppressing FLAT_EMIT in favour of `_room_gable_candidates`).
    The gable now lives in `visual_shells` instead, and the wall-derived
    attic lid clip drops any in-room oblique sliver that would otherwise
    poke above the wall tops.
    """
    uuid = "019e1376-9762-42d6-8520-b664b8c752df"
    pipeline_dir = Path("pipeline-outputs")
    scan_root = Path(".scan-cache")

    payload = build_tier_payload(uuid, pipeline_dir=pipeline_dir, scan_root=scan_root)
    locators = {piece.locator_id for piece in payload.ceiling}

    # Rooms 3, 5 are type=flat; under the new design they keep their
    # horizontal flat ceiling. (Rooms 0, 1, 2, 4, 6 are type=sloped — no flat
    # locator expected for those. Room 6 has a 31cm wall-top variation that
    # classifies it as sloped, despite earlier being flat in this fixture.)
    for room_index in (3, 5):
        locator = f"{uuid}::tier-ceiling-flat::{room_index}"
        # Hybrid rooms emit `flat::N:hybrid-lid` (the residual after
        # subtracting oblique xz); pure flat rooms keep `flat::N` exact.
        assert any(
            piece_locator == locator
            or piece_locator.startswith(f"{locator}/")
            or piece_locator.startswith(f"{locator}:hybrid-lid")
            for piece_locator in locators
        ), f"{locator} should be emitted as a flat ceiling"

    assert not any(
        piece.locator_id.startswith(f"{uuid}::tier-ceiling-computed-oblique-room::5:")
        and piece.source == CeilingSource.COMPUTED_OBLIQUE
        for piece in payload.ceiling
    )


def test_gable_flat_room_emits_flat_lid_and_hybrid_oblique():
    """A flat-classified room under a gable whose plane physically dips into
    the room volume is hybrid: it emits a flat-lid piece for the area where
    the slope sits above eave Y, and a hybrid oblique piece for the area
    where the slope dips below it. The room's wall-tops happen to be uniform
    Y (so `ceiling_type` classifies it `flat`), but the gable's pair really
    does cross the room's airspace below the lid."""
    uuid = "2388d90c-8cc9-4cca-b232-d658e184074d"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )
    locators = {piece.locator_id for piece in payload.ceiling}

    # Hybrid lid (covers the residual flat area above the slope)
    assert f"{uuid}::tier-ceiling-flat::1:hybrid-lid" in locators

    # Hybrid oblique (the slope side that dips into the room)
    hybrid_obliques = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::1:hybrid"
        )
    ]
    assert hybrid_obliques


def test_top_sloped_gable_room_uses_two_gable_halves_not_raw_fragments():
    uuid = "0b75d30e-c50c-4fc6-88ff-fce983078aa4"
    room_index = 0

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    assert payload.classification.roof_type == "gable"
    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}:"
        )
    ]

    assert len(sloped) == 2
    assert {piece.locator_id.rsplit(":", 1)[1] for piece in sloped} == {
        "gable0_0",
        "gable1_0",
    }
    assert all(":raw" not in piece.locator_id for piece in sloped)
    assert min(max(corner.y for corner in piece.corners) for piece in sloped) > 5.5


def test_lower_flat_room_below_gable_keeps_flat_ceiling_not_kink_slope():
    """The building can be a gable while a lower room below it remains flat."""
    uuid = "52f91e67-3891-4729-8bf3-be2c0a6a0d04"
    room_index = 0
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    assert payload.classification.roof_type == "gable"

    flat = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.FLAT_CEILING
        and piece.locator_id.startswith(f"{uuid}::tier-ceiling-flat::{room_index}")
    ]
    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert flat
    assert sloped == []


def test_closed_lid_gable_room_piece_is_dropped():
    """A synthesized gable room piece is attic leakage when eave and lid meet."""
    uuid = "bb013161-a5cf-4dfc-9b88-d2c851f92aee"
    room_index = 3
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    leaked_locator = (
        f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}:gable1_0"
    )
    locators = {piece.locator_id for piece in payload.ceiling}

    assert leaked_locator not in locators


def test_gable_flat_room_keeps_full_ceiling_coverage():
    """Room 12 of d28b528a is a hybrid (flat-classified, slope dips in): the
    sum of flat-lid + hybrid-oblique pieces should still cover the full room
    footprint, even though no single full-room flat piece is emitted."""
    uuid = "d28b528a-475b-4ac0-a38d-ee992cd877db"
    room_index = 12
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    room = payload.rooms[room_index]
    flat_base = f"{uuid}::tier-ceiling-flat::{room_index}"
    oblique_base = f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
    ceiling_pieces = [
        piece
        for piece in payload.ceiling
        if piece.locator_id == flat_base
        or piece.locator_id.startswith(f"{flat_base}/")
        or piece.locator_id.startswith(f"{flat_base}:hybrid-lid")
        or piece.locator_id.startswith(f"{oblique_base}:hybrid")
    ]
    room_poly = Polygon(
        [(corner.x, corner.z) for piece in room.floor for corner in piece.corners]
    )
    ceiling_polys = [
        Polygon([(corner.x, corner.z) for corner in piece.corners])
        for piece in ceiling_pieces
    ]

    assert ceiling_pieces
    assert (
        sum(poly.intersection(room_poly).area for poly in ceiling_polys)
        / room_poly.area
        > 0.85
    )


def test_flat_ceiling_split_drops_self_overlap():
    uuid = "59b505e7-b384-451b-90b1-80f2654dd10d"

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    pieces = [
        piece
        for piece in payload.ceiling
        if piece.locator_id == f"{uuid}::tier-ceiling-flat::7"
        or piece.locator_id.startswith(f"{uuid}::tier-ceiling-flat::7/")
    ]
    polys = [
        Polygon([(corner.x, corner.z) for corner in piece.corners]) for piece in pieces
    ]
    for i, poly in enumerate(polys):
        for other in polys[i + 1 :]:
            assert poly.intersection(other).area <= 0.01


def test_flat_rooms_emit_floor_sized_ceiling_not_void_caps():
    uuid = "6e8a252f-fc38-4ffa-8691-3f43938a0a16"

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    room = payload.rooms[4]
    room_poly = unary_union(
        [
            Polygon([(corner.x, corner.z) for corner in floor.corners])
            for floor in room.floor
        ]
    )
    ceiling_union = unary_union(
        [
            Polygon([(corner.x, corner.z) for corner in piece.corners])
            for piece in payload.ceiling
        ]
    )
    room_ceiling_voids = [
        gap for gap in payload.gaps if "room_ceiling_void" in gap.locator_id
    ]

    assert room_poly.intersection(ceiling_union).area / room_poly.area > 0.99
    assert room_ceiling_voids == []


@pytest.mark.parametrize(
    ("uuid", "room_index"),
    [
        ("c9a43a7c-7171-484c-ac98-54f9ba59b6e6", 13),
    ],
)
def test_top_story_kinked_rooms_emit_flat_and_sloped_ceiling(uuid, room_index):
    """Top-story gable rooms can still be true knee-wall rooms.

    If RoomPlan directly observed a flat lid and a sloped patch in the same
    room, that scan evidence should win over the broader "gable owns top
    story" rule.
    """
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    flat = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.FLAT_CEILING
        and piece.locator_id.startswith(f"{uuid}::tier-ceiling-flat::{room_index}")
    ]
    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert flat
    assert sloped


@pytest.mark.parametrize(
    ("uuid", "room_index"),
    [
        ("5fdc8f48-e5d6-4951-915d-52ebf282eedb", 4),
        ("a492a5d6-ab9f-4b1e-bbfe-4762c1370a40", 10),
    ],
)
def test_gable_kink_keeps_flat_lid(uuid, room_index):
    """Rooms where the gable owns a partial kink slope but the slope does NOT
    dip below the room's eave: keep the flat lid only, no per-room oblique.
    The slope sits above the room's lid as attic, painted via visual_shells."""
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    flat = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.FLAT_CEILING
        and piece.locator_id.startswith(f"{uuid}::tier-ceiling-flat::{room_index}")
    ]
    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert flat
    assert sloped == []


@pytest.mark.parametrize(
    ("uuid", "room_index"),
    [
        ("c9a43a7c-7171-484c-ac98-54f9ba59b6e6", 11),
        ("670a8030-c451-4e80-badd-6702cef02a03", 11),
    ],
)
def test_gable_hybrid_partial_kink_room_emits_oblique_and_lid(uuid, room_index):
    """Rooms where the gable's plane physically dips below the room's eave Y
    inside the room footprint: emit both the hybrid lid (over the residual
    flat area) and the hybrid oblique (over the slope-dips-in area)."""
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    hybrid_lid = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.FLAT_CEILING
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-flat::{room_index}:hybrid-lid"
        )
    ]
    hybrid_oblique = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}:hybrid"
        )
    ]

    assert hybrid_lid, f"expected hybrid lid for {uuid} room {room_index}"
    assert hybrid_oblique, f"expected hybrid oblique for {uuid} room {room_index}"


@pytest.mark.parametrize(
    ("uuid", "room_index"),
    [
        ("0d3f2993-8386-4130-8f1c-b2938c410828", 4),
        ("1900be91-8684-4316-98e2-c4fef6e6296f", 0),
    ],
)
def test_room_kink_sloped_piece_survives_late_synthesis(uuid, room_index):
    """Bare per-room kink slopes are occupied-room ceilings, not attic shell.

    Late synthesis must not clip them away using roof.kinks attic-lid
    heuristics; those heuristics can sit below the directly observed sloped
    patch.
    """
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert sloped


def test_double_slanted_kink_room_emits_both_sloped_sides():
    """A flat-between-two-slopes room needs both sloped raw patches emitted.

    d8308 room 9 has two large, opposing sloped ceiling planes. Keeping only
    the largest leaves the other half uncovered in the tier model.
    """
    uuid = "d8308bfc-c2c1-42bd-8503-282571708b8c"
    room_index = 9
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert len(sloped) >= 2


def test_flat_lid_kink_room_keeps_observed_slope_when_roof_cell_clips_away():
    """Do not suppress scan-observed kink slopes behind droppable roof cells.

    a443 room 6 has an observed flat lid plus an observed sloped patch. The
    computed roof-cell oblique covers the sloped patch in XZ, but late
    synthesis drops that cell because the room's eave and lid are the same
    height. The observed room slope must therefore survive directly.
    """
    uuid = "a443b86f-c86a-47a7-abef-0a56893c99b0"
    room_index = 6
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert sloped


def test_top_story_gable_room_walls_clip_under_gable_plane():
    """Walls of top-story gable rooms must clip under the gable plane.

    Under v2-late-synth the attic-shell pieces are no longer part of the
    thermal envelope (they live in payload.visual_shells), so the painter
    no longer needs the priority-80 shell suppressing the raw plane. What
    remains testable is wall geometry: the room's wall corners must not
    poke above the room's gable obliques.
    """
    uuid = "0b75d30e-c50c-4fc6-88ff-fce983078aa4"

    model = extract_building_model(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_cache_root=Path(".scan-cache"),
    )
    roof = build_roof_model(model)
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    top_room = next(
        room for room in payload.rooms if room.locator_id.endswith("::tier-room::0")
    )
    model_room = next(room for room in model.rooms if room.index == 0)
    selected = _select_gable_oblique_pair(
        roof.oblique, _xz_points_poly(model_room.floor_polygon)
    )
    assert selected is not None
    selected_obliques, _oblique_union = selected
    for wall in top_room.walls:
        for corner in wall.corners:
            roof_y = min(
                oblique.plane.y_at(corner.x, corner.z) for oblique in selected_obliques
            )
            assert corner.y <= roof_y + 1e-4, wall.locator_id

    # The shell now lives in visual_shells (story-bucketed transparent
    # attic), not in payload.ceiling — so payload.ceiling should NOT contain
    # any ROOF_ARRANGEMENT_ATTIC piece (it was dropped by synthesis).
    assert all(
        piece.source != CeilingSource.ROOF_ARRANGEMENT_ATTIC
        for piece in payload.ceiling
    )


def test_top_sloped_gable_room_uses_two_gable_halves_not_kink_or_raw_fragments():
    uuid = "0b75d30e-c50c-4fc6-88ff-fce983078aa4"
    room_index = 0

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    sloped = [
        piece
        for piece in payload.ceiling
        if piece.source == CeilingSource.COMPUTED_OBLIQUE
        and piece.locator_id.startswith(
            f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}"
        )
    ]

    assert {piece.locator_id for piece in sloped} == {
        f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}:gable0_0",
        f"{uuid}::tier-ceiling-computed-oblique-room::{room_index}:gable1_0",
    }


def test_dormer_front_face_preserves_reported_window_cutout():
    uuid = "3a576e1b-4b3e-4f5d-8c20-b39b157fcf03"

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )
    room = next(
        room for room in payload.rooms if room.locator_id == f"{uuid}::tier-room::1"
    )
    reported_window = room.windows[0]
    front_faces = [
        face
        for face in payload.dormer_faces
        if face.kind == DormerFaceKind.DORMER_FRONT
    ]

    assert front_faces
    assert any(
        _same_quad(cutout.corners, reported_window.corners)
        for face in front_faces
        for cutout in face.cutouts
    )


def test_short_parent_ridge_span_dormer_preserves_reported_window_cutout():
    uuid = "90683bb0-28ca-42a8-82b8-09d82cd434d5"

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )
    room = next(
        room for room in payload.rooms if room.locator_id == f"{uuid}::tier-room::3"
    )
    reported_window = room.windows[0]
    front_faces = [
        face
        for face in payload.dormer_faces
        if face.kind == DormerFaceKind.DORMER_FRONT
    ]

    assert front_faces
    assert any(
        _same_quad_within_m(cutout.corners, reported_window.corners, tol_m=0.02)
        for face in front_faces
        for cutout in face.cutouts
    )


def test_lower_story_raw_ceiling_crossing_upper_slab_is_suppressed():
    uuid = "146ecf8b-ffa1-4239-ba58-040b61861fd9"

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )
    locators = {piece.locator_id for piece in payload.ceiling}

    assert f"{uuid}::tier-ceiling-raw::1:0" not in locators
    assert f"{uuid}::tier-ceiling-raw::1:2" not in locators


def test_raw_ceiling_upper_slab_guard_uses_overlap_and_height():
    lower = SimpleNamespace(story=0)
    upper = SimpleNamespace(
        story=1,
        floor_polygon=[
            [0.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 4.0],
            [0.0, 2.0, 4.0],
        ],
    )
    model = SimpleNamespace(rooms=[lower, upper])

    crossing = [
        [0.0, 1.9, 0.0],
        [4.0, 2.3, 0.0],
        [4.0, 2.3, 4.0],
        [0.0, 1.9, 4.0],
    ]
    below_slab = [
        [0.0, 1.8, 0.0],
        [4.0, 1.8, 0.0],
        [4.0, 1.8, 4.0],
        [0.0, 1.8, 4.0],
    ]
    mostly_uncovered = [
        [4.2, 2.3, 0.0],
        [8.2, 2.3, 0.0],
        [8.2, 2.3, 4.0],
        [4.2, 2.3, 4.0],
    ]

    assert _raw_ceiling_crosses_upper_floor_slab(lower, crossing, model)
    assert not _raw_ceiling_crosses_upper_floor_slab(lower, below_slab, model)
    assert not _raw_ceiling_crosses_upper_floor_slab(lower, mostly_uncovered, model)


def test_gap_absorption_preserves_valid_room_floor_rings():
    model = extract_building_model(
        "0430ebc2-236b-4b5d-991f-3e97ad246b78",
        pipeline_dir=Path("pipeline-outputs"),
        scan_cache_root=Path(".scan-cache"),
    )

    for room in model.rooms:
        poly = Polygon([(point[0], point[2]) for point in room.floor_polygon])
        assert poly.is_valid, room.index
        assert not poly.is_empty, room.index


def test_overlap_clipped_room_keeps_ceiling_and_wall_for_surviving_window():
    uuid = "24e8aaa7-ec15-4a72-be5f-c67b95a53411"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    room = payload.rooms[2]
    ceiling = next(
        piece
        for piece in payload.ceiling
        if piece.locator_id == f"{uuid}::tier-ceiling-flat::2"
    )
    room_poly = Polygon(
        [(corner.x, corner.z) for piece in room.floor for corner in piece.corners]
    )
    ceiling_poly = Polygon([(corner.x, corner.z) for corner in ceiling.corners])

    assert room_poly.intersection(ceiling_poly).area / room_poly.area > 0.95
    assert room.windows

    window = room.windows[0]
    window_center = Point(
        sum(corner.x for corner in window.corners) / len(window.corners),
        sum(corner.z for corner in window.corners) / len(window.corners),
    )
    wall_distances = [
        LineString([(corner.x, corner.z) for corner in wall.corners[:2]]).distance(
            window_center
        )
        for wall in room.walls
    ]
    assert min(wall_distances) < 0.05


@pytest.mark.parametrize(
    ("uuid", "room_index"),
    [
        ("2ea3b759-e047-424c-8034-f8ee5b811fb4", 4),
        ("a8aca518-ee19-4cef-8df9-1b574c2d43d1", 5),
        ("c7f2456f-91fe-4aaa-a9ba-d886ec8e0648", 7),
        ("c87c1e25-ff00-44ec-b823-b0966c81af70", 8),
        ("cb711a0b-6e8d-4ae6-b008-af3297446dcc", 4),
        ("e661e7b6-303d-415c-b378-2d9dd2fbfd6f", 4),
        ("fd6ee1a7-7243-4c9b-ad7a-5d146b7f8706", 2),
    ],
)
def test_nested_room_floor_is_preserved_after_overlap_clipping(uuid, room_index):
    model = extract_building_model(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_cache_root=Path(".scan-cache"),
    )
    room = model.rooms[room_index]
    poly = Polygon([(point[0], point[2]) for point in room.floor_polygon])

    assert len(room.floor_polygon) >= 3
    assert poly.is_valid
    assert not poly.is_empty


def test_mtime_gating_skips_current_payload_and_rebuilds_stale_payload(tmp_path):
    uuid = "uuid-1"
    pipeline_dir = tmp_path / "pipeline"
    building_dir = pipeline_dir / uuid
    building_dir.mkdir(parents=True)
    merged_path = building_dir / "merged.json"
    merged_path.write_text(json.dumps({"rooms": []}))
    scan_dir = tmp_path / "scan-root" / f"scans_address_{uuid}_suffix"
    scan_dir.mkdir(parents=True)
    (scan_dir / "room.json").write_text("{}")
    payload_path = building_dir / "tier_payload.json"
    payload_path.write_text("{}")

    old = 100.0
    current = 200.0
    newer = 300.0
    for path in (merged_path, scan_dir, scan_dir / "room.json"):
        path.touch()
    payload_path.touch()
    import os

    os.utime(merged_path, (current, current))
    os.utime(scan_dir / "room.json", (current, current))
    os.utime(scan_dir, (current, current))
    os.utime(payload_path, (newer, newer))

    assert not needs_rebuild(uuid, pipeline_dir, scan_dir.parent, payload_path)
    assert needs_rebuild(uuid, pipeline_dir, scan_dir.parent, payload_path, force=True)

    os.utime(payload_path, (old, old))
    assert needs_rebuild(uuid, pipeline_dir, scan_dir.parent, payload_path)


# --- raw-ceiling noise gate ----------------------------------------------------


def _raw_gate_make_candidate(corners, source, locator_id, story=0):
    from reconcile_tiers.assemble.ceiling_painter import CeilingCandidate

    plane = Plane.fit(corners)
    return CeilingCandidate(
        corners=corners,
        plane=plane,
        source=source,
        locator_id=locator_id,
        story=story,
    )


def _raw_gate_make_model_with_room(uuid, raw_planes, ceiling_type=None):
    from reconcile_tiers.extract.building import (
        BuildingModel,
        ExtractedRoom,
        RawCeilingPlane,
    )

    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ],
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[RawCeilingPlane(corners=p) for p in raw_planes],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=ceiling_type,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    return BuildingModel(
        uuid=uuid,
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[room],
        scan_rooms_found=1,
        scan_rooms_transformed=1,
    )


def _raw_gate_empty_roof():
    from reconcile_tiers.roof.roof import RoofKinks, RoofModel

    return RoofModel(
        simple_slant_room_indices=set(),
        segments=[],
        clusters=[],
        footprint=None,
        planes=[],
        clipped_planes=[],
        oblique=[],
        flat=[],
        oblique_split=[],
        dormer_candidates=[],
        thermal=[],
        kinks=RoofKinks(),
    )


def _raw_gate_roof_with_oblique(corners):
    from reconcile_tiers.roof.roof import ObliqueSurface

    plane = Plane.fit(corners)
    return replace(
        _raw_gate_empty_roof(),
        oblique=[
            ObliqueSurface(
                corners=corners,
                plane=plane,
                cluster=SimpleNamespace(avg_azimuth=0.0, avg_incl=30.0),
                dominant_story=0,
                ridge={},
                source_index=0,
            )
        ],
    )


def _kink_guard_room(
    index: int,
    x0: float,
    x1: float,
    *,
    ceiling_type: str = "flat",
    kinked: bool = False,
    flat_y: float = 2.4,
    flat_polygon: list[list[float]] | None = None,
    slope_polygon: list[list[float]] | None = None,
    raw_planes: list[list[list[float]]] | None = None,
):
    from reconcile_tiers.extract.building import ExtractedRoom, RawCeilingPlane

    floor = [[x0, 0.0, 0.0], [x1, 0.0, 0.0], [x1, 0.0, 4.0], [x0, 0.0, 4.0]]
    ceiling = [
        [x0, flat_y, 0.0],
        [x1, flat_y, 0.0],
        [x1, flat_y, 4.0],
        [x0, flat_y, 4.0],
    ]
    return ExtractedRoom(
        index=index,
        story=0,
        floor_polygon=floor,
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[RawCeilingPlane(corners=p) for p in (raw_planes or [])],
        raw_ceiling_source=None,
        ceiling_polygon=ceiling,
        ceiling_type=ceiling_type,
        ceiling_eave_height=flat_y,
        ceiling_ridge_height=flat_y,
        ceiling_is_kinked=kinked,
        ceiling_flat_polygon=(flat_polygon or ceiling) if kinked else [],
        ceiling_slope_polygon=slope_polygon or [],
        ceiling_slope_polygons=[slope_polygon] if slope_polygon else [],
    )


def _kink_guard_model(rooms):
    from reconcile_tiers.extract.building import BuildingModel

    return BuildingModel(
        uuid="00000000-0000-0000-0000-000000000000",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=rooms,
        scan_rooms_found=len(rooms),
        scan_rooms_transformed=len(rooms),
    )


def test_kink_guard_drops_unsupported_low_slope_in_flat_room_with_flat_neighbor():
    slope = [[2.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 1.1, 2.0], [2.0, 1.1, 2.0]]
    model = _kink_guard_model(
        [
            _kink_guard_room(0, 0.0, 4.0, kinked=True, slope_polygon=slope),
            _kink_guard_room(1, 4.0, 8.0, ceiling_type="flat"),
        ]
    )

    candidates = _ceiling_candidates(model, _raw_gate_empty_roof())
    locators = {candidate.locator_id for candidate in candidates}

    assert "00000000-0000-0000-0000-000000000000::tier-ceiling-flat::0" in locators
    assert (
        "00000000-0000-0000-0000-000000000000::tier-ceiling-roof-arrangement-room::0"
        not in locators
    )


def test_kink_guard_drops_shallow_unsupported_slope_in_flat_room():
    slope = [[0.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 2.07, 3.0], [0.0, 2.07, 3.0]]
    model = _kink_guard_model(
        [
            _kink_guard_room(0, 0.0, 4.0, kinked=True, slope_polygon=slope),
        ]
    )

    candidates = _ceiling_candidates(model, _raw_gate_empty_roof())
    locators = {candidate.locator_id for candidate in candidates}

    assert "00000000-0000-0000-0000-000000000000::tier-ceiling-flat::0" in locators
    assert (
        "00000000-0000-0000-0000-000000000000::tier-ceiling-roof-arrangement-room::0"
        not in locators
    )


def test_kink_guard_drops_unsupported_slope_in_unclassified_room():
    slope = [[0.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 1.2, 3.0], [0.0, 1.2, 3.0]]
    model = _kink_guard_model(
        [
            _kink_guard_room(
                0,
                0.0,
                4.0,
                ceiling_type=None,
                kinked=True,
                slope_polygon=slope,
            ),
        ]
    )

    candidates = _ceiling_candidates(model, _raw_gate_empty_roof())
    locators = {candidate.locator_id for candidate in candidates}

    assert "00000000-0000-0000-0000-000000000000::tier-ceiling-flat::0" in locators
    assert (
        "00000000-0000-0000-0000-000000000000::tier-ceiling-roof-arrangement-room::0"
        not in locators
    )


def test_kink_guard_keeps_low_slope_when_local_neighbor_is_sloped():
    slope = [[2.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 1.1, 2.0], [2.0, 1.1, 2.0]]
    model = _kink_guard_model(
        [
            _kink_guard_room(0, 0.0, 4.0, kinked=True, slope_polygon=slope),
            _kink_guard_room(1, 4.0, 8.0, ceiling_type="sloped"),
        ]
    )

    candidates = _ceiling_candidates(model, _raw_gate_empty_roof())
    locators = {candidate.locator_id for candidate in candidates}

    assert (
        "00000000-0000-0000-0000-000000000000::tier-ceiling-roof-arrangement-room::0"
        in locators
    )


def test_kink_guard_keeps_low_slope_when_dormer_supports_room():
    from reconcile_tiers.roof.roof import DormerCandidate

    slope = [[2.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 1.1, 2.0], [2.0, 1.1, 2.0]]
    model = _kink_guard_model(
        [
            _kink_guard_room(0, 0.0, 4.0, kinked=True, slope_polygon=slope),
            _kink_guard_room(1, 4.0, 8.0, ceiling_type="flat"),
        ]
    )
    roof = replace(
        _raw_gate_empty_roof(),
        dormer_candidates=[
            DormerCandidate(roof_surface_index=0, room_index=0, front_wall_id=None)
        ],
    )

    candidates = _ceiling_candidates(model, roof)
    locators = {candidate.locator_id for candidate in candidates}

    assert (
        "00000000-0000-0000-0000-000000000000::tier-ceiling-roof-arrangement-room::0"
        in locators
    )


def test_kink_guard_drops_flat_patch_above_oblique_roof():
    flat_y = 2.4
    flat_patch = [
        [0.0, flat_y, 0.0],
        [2.0, flat_y, 0.0],
        [2.0, flat_y, 2.0],
        [0.0, flat_y, 2.0],
    ]
    slope = [
        [0.0, 1.0, 0.0],
        [8.0, 2.4, 0.0],
        [8.0, 2.4, 4.0],
        [0.0, 1.0, 4.0],
    ]
    model = _kink_guard_model(
        [
            _kink_guard_room(
                0,
                0.0,
                8.0,
                kinked=True,
                flat_y=flat_y,
                flat_polygon=flat_patch,
                slope_polygon=slope,
                raw_planes=[flat_patch, slope],
            )
        ]
    )
    roof = _raw_gate_roof_with_oblique(
        [
            [0.0, 1.0, 0.0],
            [8.0, 1.6, 0.0],
            [8.0, 1.6, 4.0],
            [0.0, 1.0, 4.0],
        ]
    )

    candidates = _ceiling_candidates(model, roof)
    locators = {candidate.locator_id for candidate in candidates}

    assert "00000000-0000-0000-0000-000000000000::tier-ceiling-flat::0" not in locators
    assert any("::tier-ceiling-roof-arrangement" in locator for locator in locators)


def test_kink_guard_keeps_flat_patch_below_oblique_roof():
    flat_y = 2.4
    slope = [
        [0.0, 1.0, 0.0],
        [4.0, 2.4, 0.0],
        [4.0, 2.4, 4.0],
        [0.0, 1.0, 4.0],
    ]
    model = _kink_guard_model(
        [_kink_guard_room(0, 0.0, 4.0, kinked=True, flat_y=flat_y, slope_polygon=slope)]
    )
    roof = _raw_gate_roof_with_oblique(
        [
            [0.0, 3.0, 0.0],
            [4.0, 3.6, 0.0],
            [4.0, 3.6, 4.0],
            [0.0, 3.0, 4.0],
        ]
    )

    candidates = _ceiling_candidates(model, roof)
    locators = {candidate.locator_id for candidate in candidates}

    assert "00000000-0000-0000-0000-000000000000::tier-ceiling-flat::0" in locators


def test_raw_gate_drops_tilted_triangle():
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    triangle = [
        [0.0, 2.0, 0.0],
        [1.0, 2.5, 0.0],
        [0.5, 2.5, 1.0],
    ]  # 3 corners, ~0.5 m^2, y-span 0.5
    cand = _raw_gate_make_candidate(
        triangle, CeilingSource.RAW_FALLBACK, f"{uuid}::tier-ceiling-raw::0:0"
    )
    model = _raw_gate_make_model_with_room(uuid, [triangle], ceiling_type=None)
    drops: list = []
    kept = _filter_noisy_raw_candidates(
        [cand], model, _raw_gate_empty_roof(), drops_sink=drops
    )
    assert kept == []
    assert len(drops) == 1
    assert drops[0]["reason"] == "junk_triangle"


def test_raw_gate_drops_redundant_when_higher_priority_covers():
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [
        [0.0, 2.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 2.0, 4.0],
        [0.0, 2.0, 4.0],
    ]  # 16 m^2 quad
    higher = [
        [0.0, 2.1, 0.0],
        [4.0, 2.1, 0.0],
        [4.0, 2.1, 4.0],
        [0.0, 2.1, 4.0],
    ]  # 16 m^2 quad
    raw_cand = _raw_gate_make_candidate(
        raw, CeilingSource.RAW_FALLBACK, f"{uuid}::tier-ceiling-raw::0:0"
    )
    higher_cand = _raw_gate_make_candidate(
        higher, CeilingSource.FLAT_EMIT, f"{uuid}::tier-ceiling-flat::0"
    )
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type=None)
    drops: list = []
    kept = _filter_noisy_raw_candidates(
        [raw_cand, higher_cand], model, _raw_gate_empty_roof(), drops_sink=drops
    )
    kept_locators = {c.locator_id for c in kept}
    assert higher_cand.locator_id in kept_locators
    assert raw_cand.locator_id not in kept_locators
    assert any(d["reason"] == "redundant" for d in drops)


def test_raw_gate_drops_redundant_when_higher_priority_covers_large_fraction():
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [[0.0, 2.0, 0.0], [10.0, 2.0, 0.0], [10.0, 2.0, 10.0], [0.0, 2.0, 10.0]]
    higher = [[0.0, 2.1, 0.0], [8.0, 2.1, 0.0], [8.0, 2.1, 10.0], [0.0, 2.1, 10.0]]
    raw_cand = _raw_gate_make_candidate(
        raw, CeilingSource.RAW_FALLBACK, f"{uuid}::tier-ceiling-raw::0:0"
    )
    higher_cand = _raw_gate_make_candidate(
        higher,
        CeilingSource.ROOF_ARRANGEMENT,
        f"{uuid}::tier-ceiling-roof-arrangement::0",
    )
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type="sloped")
    drops: list = []
    kept = _filter_noisy_raw_candidates(
        [raw_cand, higher_cand], model, _raw_gate_empty_roof(), drops_sink=drops
    )

    assert [c.locator_id for c in kept] == [higher_cand.locator_id]
    assert drops[0]["reason"] == "redundant"
    assert drops[0]["area_xz_m2"] == 100.0


def test_raw_gate_keeps_sloped_raw_when_computed_overlap_is_too_low():
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]]
    lower = [[0.0, 1.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 4.0], [0.0, 2.0, 4.0]]
    raw_cand = _raw_gate_make_candidate(
        raw, CeilingSource.RAW_FALLBACK, f"{uuid}::tier-ceiling-raw::0:0"
    )
    lower_cand = _raw_gate_make_candidate(
        lower,
        CeilingSource.ROOF_ARRANGEMENT,
        f"{uuid}::tier-ceiling-roof-arrangement::0",
    )
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type="sloped")
    drops: list = []
    kept = _filter_noisy_raw_candidates(
        [raw_cand, lower_cand], model, _raw_gate_empty_roof(), drops_sink=drops
    )

    assert {c.locator_id for c in kept} == {raw_cand.locator_id, lower_cand.locator_id}
    assert drops == []


def test_raw_gate_keeps_sloped_raw_when_computed_misses_peak_edge():
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [[0.0, 2.0, 0.0], [10.0, 2.0, 0.0], [10.0, 4.0, 10.0], [0.0, 4.0, 10.0]]
    low_side = [[0.0, 2.0, 0.0], [10.0, 2.0, 0.0], [10.0, 3.4, 7.0], [0.0, 3.4, 7.0]]
    raw_cand = _raw_gate_make_candidate(
        raw, CeilingSource.RAW_FALLBACK, f"{uuid}::tier-ceiling-raw::0:0"
    )
    lower_cand = _raw_gate_make_candidate(
        low_side,
        CeilingSource.ROOF_ARRANGEMENT,
        f"{uuid}::tier-ceiling-roof-arrangement::0",
    )
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type="sloped")
    drops: list = []

    kept = _filter_noisy_raw_candidates(
        [raw_cand, lower_cand], model, _raw_gate_empty_roof(), drops_sink=drops
    )

    assert {c.locator_id for c in kept} == {raw_cand.locator_id, lower_cand.locator_id}
    assert drops == []


def test_raw_gate_keeps_orphan_raw_in_sloped_room():
    """In a sloped room with no higher-priority candidate, a raw plane must
    survive — it IS the ceiling source."""
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [
        [0.0, 2.0, 0.0],
        [4.0, 2.5, 0.0],
        [4.0, 2.5, 4.0],
        [0.0, 2.0, 4.0],
    ]  # 4 corners, 16 m^2, gentle slope
    raw_cand = _raw_gate_make_candidate(
        raw, CeilingSource.RAW_FALLBACK, f"{uuid}::tier-ceiling-raw::0:0"
    )
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type="sloped")
    drops: list = []
    kept = _filter_noisy_raw_candidates(
        [raw_cand], model, _raw_gate_empty_roof(), drops_sink=drops
    )
    assert [c.locator_id for c in kept] == [raw_cand.locator_id]
    assert drops == []


def test_raw_oblique_owner_promotes_sloped_room_raw_before_fallback():
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _raw_oblique_owner_candidates_for_room,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]]
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type="sloped")

    owners = _raw_oblique_owner_candidates_for_room(
        model,
        model.rooms[0],
        _raw_gate_empty_roof(),
        [],
        wall_axis_math=None,
        is_top_gable_room=False,
        gable_owns_room=False,
    )

    assert len(owners) == 1
    assert owners[0].source == CeilingSource.ROOF_ARRANGEMENT
    assert owners[0].locator_id == f"{uuid}::tier-ceiling-roof-arrangement-room::0:raw0"


def test_raw_oblique_owner_drops_unsupported_unclassified_kink_raw():
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _raw_oblique_owner_candidates_for_room,
    )

    flat = [[0.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 2.4, 4.0], [0.0, 2.4, 4.0]]
    slope = [[0.0, 2.4, 0.0], [4.0, 2.4, 0.0], [4.0, 1.2, 3.0], [0.0, 1.2, 3.0]]
    model = _kink_guard_model(
        [
            _kink_guard_room(
                0,
                0.0,
                4.0,
                ceiling_type=None,
                kinked=True,
                flat_polygon=flat,
                slope_polygon=slope,
                raw_planes=[slope],
            )
        ]
    )

    owners = _raw_oblique_owner_candidates_for_room(
        model,
        model.rooms[0],
        _raw_gate_empty_roof(),
        [],
        wall_axis_math=None,
        is_top_gable_room=False,
        gable_owns_room=False,
    )

    assert owners == []


def test_raw_oblique_owner_does_not_promote_plain_flat_room_raw():
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _raw_oblique_owner_candidates_for_room,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]]
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type="flat")

    owners = _raw_oblique_owner_candidates_for_room(
        model,
        model.rooms[0],
        _raw_gate_empty_roof(),
        [],
        wall_axis_math=None,
        is_top_gable_room=False,
        gable_owns_room=False,
    )

    assert owners == []


def test_raw_plane_owner_promotes_unclassified_flat_raw_before_fallback():
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _raw_plane_owner_candidates_for_room,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    raw = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 2.0, 4.0], [0.0, 2.0, 4.0]]
    model = _raw_gate_make_model_with_room(uuid, [raw], ceiling_type=None)

    owners = _raw_plane_owner_candidates_for_room(
        model,
        model.rooms[0],
        _raw_gate_empty_roof(),
        [],
        wall_axis_math=None,
        is_top_gable_room=False,
        gable_owns_room=False,
    )

    assert len(owners) == 1
    assert owners[0].source == CeilingSource.FLAT_EMIT
    assert owners[0].locator_id == f"{uuid}::tier-ceiling-flat::0:raw0"


def test_raw_plane_owner_merges_same_plane_fragments_before_fallback():
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _raw_plane_owner_candidates_for_room,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    left = [[0.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 4.0], [0.0, 2.0, 4.0]]
    right = [[2.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 4.0, 4.0], [2.0, 3.0, 4.0]]
    model = _raw_gate_make_model_with_room(uuid, [left, right], ceiling_type="sloped")

    owners = _raw_plane_owner_candidates_for_room(
        model,
        model.rooms[0],
        _raw_gate_empty_roof(),
        [],
        wall_axis_math=None,
        is_top_gable_room=False,
        gable_owns_room=False,
    )

    assert len(owners) == 1
    assert owners[0].source == CeilingSource.ROOF_ARRANGEMENT
    assert owners[0].locator_id == f"{uuid}::tier-ceiling-roof-arrangement-room::0:raw0"
    assert _xz_points_poly(owners[0].corners).area == pytest.approx(16.0)


def test_payload_closes_short_within_story_free_edge_with_v1_rule():
    uuid = "c1abe3ef-2eb5-49f4-b546-a5122f6c40a7"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )
    expected_bridge = LineString(
        [
            (-1.1258749695860761, -1.893409746092581),
            (-0.9586298298950959, -2.1911719320289578),
        ]
    )

    matching_sides = []
    for gap in payload.gaps:
        if gap.kind != GapKind.SIDE or gap.scope.value != "intra_story":
            continue
        edge = LineString(
            [
                (gap.corners[0].x, gap.corners[0].z),
                (gap.corners[1].x, gap.corners[1].z),
            ]
        )
        if edge.distance(expected_bridge) <= 0.03 and edge.length == pytest.approx(
            0.338, abs=0.02
        ):
            matching_sides.append(gap)

    assert matching_sides
    assert any(
        gap.locator_id.startswith(f"{uuid}::tier-gap::gw:gap:within_story")
        for gap in matching_sides
    )
    assert (
        max(corner.y for corner in matching_sides[0].corners)
        - min(corner.y for corner in matching_sides[0].corners)
        > 2.0
    )


def test_payload_does_not_close_long_unsupported_within_story_void_chord():
    uuid = "a7f4c712-2252-457d-b360-4fd9dbaffd0b"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    bad_locator = (
        f"{uuid}::tier-gap::gw:gap:within_story:0:d2587f7f5253a007:within_story:edge:1"
    )

    assert bad_locator not in {gap.locator_id for gap in payload.gaps}


def test_payload_does_not_close_short_unsupported_within_story_void_chord():
    uuid = "c1abe3ef-2eb5-49f4-b546-a5122f6c40a7"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    bad_locator = (
        f"{uuid}::tier-gap::gw:gap:within_story:0:89ddc7600af4304e:within_story:edge:8"
    )

    assert bad_locator not in {gap.locator_id for gap in payload.gaps}


def test_payload_does_not_close_deformed_within_story_free_edge_chord():
    uuid = "20fa9e17-7264-48c7-9ceb-2140d9cbc86e"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    bad_locator = (
        f"{uuid}::tier-gap::gw:gap:within_story:0:ee8c7352850b58f8:within_story:edge:2"
    )

    assert bad_locator not in {gap.locator_id for gap in payload.gaps}


@pytest.mark.parametrize(
    ("uuid", "bad_locator"),
    [
        (
            "16784bad-2cd9-4f4c-bb26-60355981cfe2",
            "16784bad-2cd9-4f4c-bb26-60355981cfe2::tier-ceiling-flat::2/1",
        ),
        (
            "74e87bcd-3989-4d5c-8f16-f7782dc3afbd",
            "74e87bcd-3989-4d5c-8f16-f7782dc3afbd::tier-ceiling-flat::1/2",
        ),
        (
            "670a8030-c451-4e80-badd-6702cef02a03",
            "670a8030-c451-4e80-badd-6702cef02a03::tier-ceiling-flat::10/2",
        ),
    ],
)
def test_high_ridge_flat_artifacts_are_not_emitted(monkeypatch, uuid, bad_locator):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "1")

    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    assert bad_locator not in {piece.locator_id for piece in payload.ceiling}


def test_flat_synthesis_guard_requires_mostly_oblique_raw_coverage():
    from reconcile_tiers.build_internals.ceiling_helpers import (
        _room_has_oblique_evidence_for_flat_synthesis,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    partial_slope = [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 4.0], [0.0, 3.0, 4.0]]
    full_slope = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 4.0], [0.0, 3.0, 4.0]]

    mixed_room = _raw_gate_make_model_with_room(
        uuid, [partial_slope], ceiling_type="sloped"
    ).rooms[0]
    fully_sloped_room = _raw_gate_make_model_with_room(
        uuid, [full_slope], ceiling_type=None
    ).rooms[0]

    assert not _room_has_oblique_evidence_for_flat_synthesis(mixed_room)
    assert _room_has_oblique_evidence_for_flat_synthesis(fully_sloped_room)


def test_computed_arrangement_suppresses_redundant_kink_slope_and_void_sides(
    monkeypatch,
):
    """6c29deb7 room 6 had a raw-derived kink slope promoted to computed
    oblique even though computed arrangement already covered it. That duplicate
    also exposed room_ceiling_void side panels through the room."""
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "1")

    uuid = "6c29deb7-51e6-437d-bfe4-0eb83e559881"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    ceiling_locators = {piece.locator_id for piece in payload.ceiling}
    gap_locators = {gap.locator_id for gap in payload.gaps}

    assert f"{uuid}::tier-ceiling-computed-oblique-room::6" not in ceiling_locators
    assert (
        f"{uuid}::tier-gap::gw:gap:room_ceiling_void:2:"
        "393081c4b7b500b5:room_ceiling_void:edge:5" not in gap_locators
    )
    assert any(
        locator.startswith(f"{uuid}::tier-ceiling-computed-oblique::6")
        for locator in ceiling_locators
    )


def test_computed_oblique_ceiling_rings_drop_collapsed_edges():
    uuid = "6f44f918-18e3-4978-a597-36e5f6865a18"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )

    raw5 = next(
        piece
        for piece in payload.ceiling
        if piece.locator_id == f"{uuid}::tier-ceiling-computed-oblique-room::4:raw5"
    )

    assert len(raw5.corners) < 8
    assert _min_edge_m(raw5.corners) >= 0.02 - 1e-9


def test_98472_roof_opt_in_priors_closes_fragmented_raw_fallback(monkeypatch):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "1")
    monkeypatch.setenv("TIER_RESIDUAL_CEILING_VOID_CLOSURE", "1")

    uuid = "98472f6b-45bc-4814-a4b8-914f8f6976dd"
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
    )
    payload_dict = payload_to_dict(payload)
    audit_result = audit(payload_dict)

    assert "ceiling_coverage_gaps" not in audit_result["flags"]
    assert not any(piece.source == CeilingSource.RAW_SCAN for piece in payload.ceiling)
    assert _room_top_coverage_ratio(payload, 2) >= 0.99


def test_raw_gate_keeps_attic_flat_lid_unconditionally():
    """ATTIC_FLAT_LID is synthesised geometry, not raw scan noise. Even when
    it's a small triangle that would trip junk_triangle for RAW_FALLBACK, it
    must survive the gate."""
    from reconcile_tiers.build_internals.raw_ceiling_filter import (
        _filter_noisy_raw_candidates,
    )

    uuid = "00000000-0000-0000-0000-000000000000"
    triangle = [[0.0, 2.0, 0.0], [1.0, 2.5, 0.0], [0.5, 2.5, 1.0]]
    cand = _raw_gate_make_candidate(
        triangle, CeilingSource.ATTIC_FLAT_LID, f"{uuid}::tier-ceiling-raw::0:0"
    )
    model = _raw_gate_make_model_with_room(uuid, [triangle], ceiling_type=None)
    kept = _filter_noisy_raw_candidates([cand], model, _raw_gate_empty_roof())
    assert [c.locator_id for c in kept] == [cand.locator_id]


@pytest.mark.parametrize(
    ("uuid", "expected_dropped_locators", "expected_reasons"),
    [
        (
            "e9dddee6-297a-43f5-b862-90091889ea39",
            [
                "e9dddee6-297a-43f5-b862-90091889ea39::tier-ceiling-raw::8:11",
                "e9dddee6-297a-43f5-b862-90091889ea39::tier-ceiling-raw::8:8",
            ],
            {"junk_triangle"},
        ),
        (
            "893d4535-3169-4907-a93a-b3ab8f66ec1c",
            [
                "893d4535-3169-4907-a93a-b3ab8f66ec1c::tier-ceiling-raw::4:9",
            ],
            {"redundant"},
        ),
    ],
)
def test_raw_gate_drops_user_cited_noisy_ceilings(
    uuid, expected_dropped_locators, expected_reasons
):
    drops: list = []
    payload = build_tier_payload(
        uuid,
        pipeline_dir=Path("pipeline-outputs"),
        scan_root=Path(".scan-cache"),
        drops_sink=drops,
    )
    payload_locators = {piece.locator_id for piece in payload.ceiling}
    # `drops` mixes two record schemas: the raw-gate writes `reason`, the
    # primitive-emission validator writes `kind`. We only care about raw-gate
    # drops here.
    drop_index = {d["locator_id"]: d["reason"] for d in drops if "reason" in d}
    for loc in expected_dropped_locators:
        assert loc not in payload_locators, f"{loc} should be filtered out"
        # If the raw-gate drops the raw, the locator appears in `drops_sink`
        # with the expected reason. After Phase 5 Step 5b enabled
        # ROOF_ARRANGEMENT_ATTIC for gable buildings, some user-cited raws
        # are now clipped to nothing by the painter's priority pass *before*
        # reaching the raw-gate — they don't appear in either set. The
        # behavioural invariant is that they're absent from the payload;
        # whether the gate or the painter suppresses them is a detail.
        if loc in drop_index:
            assert drop_index[loc] in expected_reasons, (
                f"{loc} dropped for unexpected reason {drop_index[loc]!r}"
            )
