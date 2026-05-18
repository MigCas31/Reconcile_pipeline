import dataclasses
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import Plane, fit_plane_any
from reconcile_tiers.assemble.ceiling_painter import (
    CeilingCandidate,
    assemble_ceiling,
)
from reconcile_tiers.assemble.dormer_reconstruction import reconstruct_dormers
from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.extract.building import ExtractedWall, extract_building_model
from reconcile_tiers.payload.schema import CeilingSource, DormerFaceKind
from reconcile_tiers.payload.validate import _project_corners_to_plane
from reconcile_tiers.roof.clipping import clip_planes_to_footprint
from reconcile_tiers.roof.clustering import cluster_oblique_segments
from reconcile_tiers.roof.dormers import cutout_and_trim, detect_dormers
from reconcile_tiers.roof.footprint import build_building_footprint
from reconcile_tiers.roof.obliques import build_oblique_surfaces, story_floor_y
from reconcile_tiers.roof.planes import build_roof_planes
from reconcile_tiers.roof.roof import build_roof_model
from reconcile_tiers.roof.segments import collect_oblique_segments
from tests.reconcile_tiers.roof.helpers import make_gable_model


@pytest.fixture(autouse=True)
def _force_legacy_priors_off(monkeypatch):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "0")


def _obliques(model):
    footprint = build_building_footprint(model)
    planes = build_roof_planes(
        cluster_oblique_segments(collect_oblique_segments(model)), footprint
    )
    clipped = clip_planes_to_footprint(planes, footprint)
    obliques = build_oblique_surfaces(clipped, story_floor_y(model))
    # The synthetic helper is only 4 m deep; production dormer detection now
    # requires a longer parent roof span so short local slants do not masquerade
    # as dormers.
    for oblique in obliques:
        if oblique.ridge["max"] - oblique.ridge["min"] < 5.0:
            oblique.ridge = {**oblique.ridge, "max": oblique.ridge["min"] + 6.0}
    return obliques


def _centroid(corners):
    return [
        sum(float(corner[idx]) for corner in corners) / len(corners) for idx in range(3)
    ]


def _assert_oriented_away(corners, reference):
    normal = newell_normal(corners)
    center = _centroid(corners)
    outward = [center[idx] - reference[idx] for idx in range(3)]
    assert sum(normal[idx] * outward[idx] for idx in range(3)) > 0.0


def test_detect_dormers_returns_lightweight_candidates_only():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)

    candidates = detect_dormers(model, obliques)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.front_wall_id == "dormer-front"
    assert candidate.front_opening_id == "dormer-window"
    assert candidate.room_index == 0
    assert candidate.roof_surface_index == 0


def test_detect_dormers_ignores_knee_wall_with_shifted_top_edge_barely_above_slant():
    model = make_gable_model(include_dormer=False)
    obliques = _obliques(model)
    plane = obliques[0].plane

    top_x = 3.8
    top_clearance = 0.05
    top_y = float(plane.y_at(top_x, 1.5)) + top_clearance
    knee_wall = ExtractedWall(
        id="shifted-knee-wall",
        source="test",
        corners=[
            [3.0, 0.0, 1.5],
            [3.0, 0.0, 2.5],
            [top_x, top_y, 2.5],
            [top_x, top_y, 1.5],
        ],
    )
    room = model.rooms[0]
    model.rooms[0] = dataclasses.replace(
        room,
        walls_merged=[*room.walls_merged, knee_wall],
        walls_computed=[*room.walls_computed, knee_wall],
    )

    candidates = detect_dormers(model, obliques)

    assert not any(
        candidate.front_wall_id == "shifted-knee-wall" for candidate in candidates
    )


def test_detect_dormers_requires_pitched_parent_roof():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    shallow = dataclasses.replace(
        obliques[0],
        cluster=dataclasses.replace(obliques[0].cluster, avg_incl=17.0),
    )

    candidates = detect_dormers(model, [shallow])

    assert candidates == []


def test_detect_dormers_ignores_short_room_local_slant():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    ridge = dict(obliques[0].ridge)
    ridge["max"] = ridge["min"] + 3.9
    short_span = dataclasses.replace(obliques[0], ridge=ridge)

    candidates = detect_dormers(model, [short_span])

    assert candidates == []


def test_detect_dormers_accepts_lower_story_roof_part_with_dormer():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    lower_room = model.rooms[0]
    upper_room = dataclasses.replace(
        lower_room,
        index=1,
        story=1,
        floor_polygon=[[x + 10.0, y + 3.0, z] for x, y, z in lower_room.floor_polygon],
    )
    model = dataclasses.replace(
        model,
        stories_found=2,
        rooms=[lower_room, upper_room],
        scan_rooms_found=2,
        scan_rooms_transformed=2,
    )

    candidates = detect_dormers(model, obliques)

    assert any(candidate.front_wall_id == "dormer-front" for candidate in candidates)


def test_detect_dormers_rejects_lower_story_roof_under_higher_floor():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    lower_room = model.rooms[0]
    upper_room = dataclasses.replace(lower_room, index=1, story=1)
    model = dataclasses.replace(
        model,
        stories_found=2,
        rooms=[lower_room, upper_room],
        scan_rooms_found=2,
        scan_rooms_transformed=2,
    )

    candidates = detect_dormers(model, obliques)

    assert not any(
        candidate.front_wall_id == "dormer-front" for candidate in candidates
    )


def test_detect_dormers_rejects_full_height_room_wall():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    plane = obliques[0].plane
    bottom_y = float(plane.y_at(3.0, 1.5)) + 0.05
    full_height_wall = ExtractedWall(
        id="full-height-false-dormer",
        source="test",
        corners=[
            [3.0, bottom_y, 0.5],
            [3.0, bottom_y, 1.2],
            [3.0, bottom_y + 3.9, 1.2],
            [3.0, bottom_y + 3.9, 0.5],
        ],
    )
    room = model.rooms[0]
    model.rooms[0] = dataclasses.replace(
        room,
        walls_merged=[*room.walls_merged, full_height_wall],
        walls_computed=[*room.walls_computed, full_height_wall],
    )

    candidates = detect_dormers(model, obliques)

    assert any(candidate.front_wall_id == "dormer-front" for candidate in candidates)
    assert not any(
        candidate.front_wall_id == "full-height-false-dormer"
        for candidate in candidates
    )


def test_detect_dormers_rejects_wall_far_from_parent_slant():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    plane = obliques[0].plane
    bottom_y = float(plane.y_at(3.0, 9.0)) + 0.05
    far_wall = ExtractedWall(
        id="far-from-parent-slant",
        source="test",
        corners=[
            [3.0, bottom_y, 8.5],
            [3.0, bottom_y, 9.2],
            [3.0, bottom_y + 0.85, 9.2],
            [3.0, bottom_y + 0.85, 8.5],
        ],
    )
    room = model.rooms[0]
    model.rooms[0] = dataclasses.replace(
        room,
        walls_computed=[*room.walls_computed, far_wall],
        walls_merged=[*room.walls_merged, far_wall],
    )

    candidates = detect_dormers(model, obliques)

    assert any(candidate.front_wall_id == "dormer-front" for candidate in candidates)
    assert not any(
        candidate.front_wall_id == "far-from-parent-slant" for candidate in candidates
    )


def test_detect_dormers_requires_front_opening():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    plane = obliques[0].plane
    bottom_y = float(plane.y_at(4.0, 1.5)) + 0.05
    unsupported_wall = ExtractedWall(
        id="unsupported-protruding-wall",
        source="test",
        corners=[
            [4.0, bottom_y, 1.2],
            [4.0, bottom_y, 1.9],
            [4.0, bottom_y + 0.85, 1.9],
            [4.0, bottom_y + 0.85, 1.2],
        ],
    )
    room = model.rooms[0]
    model.rooms[0] = dataclasses.replace(
        room,
        walls_computed=[*room.walls_computed, unsupported_wall],
        walls_merged=[*room.walls_merged, unsupported_wall],
    )

    candidates = detect_dormers(model, obliques)

    assert any(candidate.front_wall_id == "dormer-front" for candidate in candidates)
    assert not any(
        candidate.front_wall_id == "unsupported-protruding-wall"
        for candidate in candidates
    )


def test_detect_dormers_accepts_windowed_front_without_cheek_walls():
    model = make_gable_model(include_dormer=True)
    room = model.rooms[0]
    model.rooms[0] = dataclasses.replace(
        room,
        walls_computed=[
            wall
            for wall in room.walls_computed
            if not wall.id.startswith("dormer-") or wall.id == "dormer-front"
        ],
        walls_merged=[
            wall
            for wall in room.walls_merged
            if not wall.id.startswith("dormer-") or wall.id == "dormer-front"
        ],
    )
    obliques = _obliques(model)

    candidates = detect_dormers(model, obliques)

    assert any(candidate.front_wall_id == "dormer-front" for candidate in candidates)


def test_detect_dormers_accepts_tall_front_when_attached_to_slant():
    model = make_gable_model(include_dormer=True)
    room = model.rooms[0]
    dormer_wall = next(w for w in room.walls_computed if w.id == "dormer-front")
    base_top_y = max(p[1] for p in dormer_wall.corners)
    tall_wall = dataclasses.replace(
        dormer_wall,
        corners=[
            [p[0], p[1] + 2.0 if p[1] == base_top_y else p[1], p[2]]
            for p in dormer_wall.corners
        ],
    )
    model.rooms[0] = dataclasses.replace(
        room,
        walls_computed=[
            tall_wall if wall.id == tall_wall.id else wall
            for wall in room.walls_computed
        ],
        walls_merged=[
            tall_wall if wall.id == tall_wall.id else wall for wall in room.walls_merged
        ],
    )

    candidates = detect_dormers(model, _obliques(model))

    assert any(candidate.front_wall_id == "dormer-front" for candidate in candidates)


def test_detect_dormers_rejects_reported_facade_wall_detached_from_host_slant():
    model = extract_building_model(
        "59b505e7-b384-451b-90b1-80f2654dd10d",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )
    obliques = build_roof_model(model).oblique

    candidates = detect_dormers(model, obliques)

    assert not any(
        candidate.front_wall_id == "979085AE-3C8D-4A17-846B-7DBE5612B932"
        for candidate in candidates
    )


def test_cutout_and_trim_geometry_meets_slant_at_top_y():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    dormer_wall = next(
        w for w in model.rooms[0].walls_computed if w.id == "dormer-front"
    )
    plane = obliques[0].plane
    top_y = max(p[1] for p in dormer_wall.corners)

    trim = cutout_and_trim(plane, dormer_wall.corners)
    assert trim is not None
    cutout, cheeks, header = trim

    assert len(cutout) == 4
    assert len(cheeks) == 2
    assert len(header) == 4

    # Each cheek is a triangle whose back-apex sits on the slant at top_y.
    for cheek in cheeks:
        assert len(cheek) == 3
        back_apex = cheek[2]
        assert back_apex[1] == pytest.approx(top_y)
        assert plane.y_at(back_apex[0], back_apex[2]) == pytest.approx(top_y)

    # Header back edge sits on the slant at top_y; all four header corners are at top_y.
    for corner in header:
        assert corner[1] == pytest.approx(top_y)
    for back in header[2:]:
        assert plane.y_at(back[0], back[2]) == pytest.approx(top_y)

    # Cutout back corners coincide with header back corners (shared edge on the slant).
    assert cutout[2] == header[2]
    assert cutout[3] == header[3]
    # Cutout front corners lie on the slant at the dormer wall's bottom XZ.
    for front in cutout[:2]:
        assert plane.y_at(front[0], front[2]) == pytest.approx(front[1])


def test_cutout_and_trim_accepts_shallow_windowed_dormer_cutout():
    plane = Plane(a=-1.0, b=1.0, c=0.0, d=0.0)
    wall = [
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 1.0],
        [0.0, 0.12, 1.0],
        [0.0, 0.12, 0.0],
    ]

    trim = cutout_and_trim(plane, wall)

    assert trim is not None
    cutout, cheeks, header = trim
    assert len(cutout) == 4
    assert len(cheeks) == 2
    assert len(header) == 4


def test_reconstruct_dormers_punches_hole_and_emits_cheeks_and_header():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    candidates = detect_dormers(model, obliques)
    assert candidates

    # Build a synthetic CeilingPiece on the same slant via assemble_ceiling.
    surface = obliques[candidates[0].roof_surface_index]
    candidate = CeilingCandidate(
        corners=[[float(p[0]), float(p[1]), float(p[2])] for p in surface.corners],
        plane=surface.plane,
        source=CeilingSource.ROOF_ARRANGEMENT,
        locator_id="synthetic::tier-ceiling-roof-arrangement::0",
        story=surface.dominant_story,
    )
    ceilings = assemble_ceiling([candidate])
    assert ceilings and not any(piece.holes for piece in ceilings)

    updated, dormer_thermal = reconstruct_dormers(ceilings, candidates, obliques, model)

    # Cheek + header thermal surfaces emitted.
    front_thermals = [
        t for t in dormer_thermal if t.kind == DormerFaceKind.DORMER_FRONT
    ]
    cheek_thermals = [
        t for t in dormer_thermal if t.kind == DormerFaceKind.DORMER_CHEEK
    ]
    header_thermals = [
        t for t in dormer_thermal if t.kind == DormerFaceKind.DORMER_HEADER
    ]
    assert len(front_thermals) == 1
    assert len(cheek_thermals) == 2
    assert len(header_thermals) == 1

    # The matching ceiling piece now has an interior ring whose XZ matches the cutout.
    trim = cutout_and_trim(
        surface.plane,
        next(
            opening
            for opening in model.rooms[0].windows
            if opening.id == "dormer-window"
        ).corners,
    )
    assert trim is not None
    dormer_reference = _centroid([*trim[0], *trim[2]])
    for front in front_thermals:
        _assert_oriented_away(front.corners, dormer_reference)
    for cheek in cheek_thermals:
        _assert_oriented_away(cheek.corners, dormer_reference)
    for header in header_thermals:
        assert newell_normal(header.corners)[1] > 0.0

    cutout_xz = Polygon([(c[0], c[2]) for c in trim[0]])
    pierced = [piece for piece in updated if piece.holes]
    assert pierced, "expected at least one ceiling piece to carry an interior hole"
    matched = False
    for piece in pierced:
        for hole in piece.holes:
            hole_xz = Polygon([(c.x, c.z) for c in hole])
            if hole_xz.symmetric_difference(cutout_xz).area < 1e-6:
                matched = True
                break
        if matched:
            break
    assert matched, "no ceiling-piece interior ring matches the dormer cutout footprint"


def test_reconstruct_dormers_bounds_front_by_matched_opening_not_wide_wall():
    model = make_gable_model(include_dormer=True)
    room = model.rooms[0]
    dormer_wall = next(w for w in room.walls_computed if w.id == "dormer-front")
    wide_wall = dataclasses.replace(
        dormer_wall,
        corners=[
            [dormer_wall.corners[0][0], dormer_wall.corners[0][1], 0.2],
            [dormer_wall.corners[1][0], dormer_wall.corners[1][1], 3.8],
            [dormer_wall.corners[2][0], dormer_wall.corners[2][1], 3.8],
            [dormer_wall.corners[3][0], dormer_wall.corners[3][1], 0.2],
        ],
    )
    model.rooms[0] = dataclasses.replace(
        room,
        walls_computed=[
            wide_wall if wall.id == wide_wall.id else wall
            for wall in room.walls_computed
        ],
        walls_merged=[
            wide_wall if wall.id == wide_wall.id else wall for wall in room.walls_merged
        ],
    )
    obliques = _obliques(model)
    candidates = detect_dormers(model, obliques)
    assert candidates
    assert candidates[0].front_opening_id == "dormer-window"

    _updated, dormer_thermal = reconstruct_dormers([], candidates, obliques, model)

    front = next(
        surface
        for surface in dormer_thermal
        if surface.kind == DormerFaceKind.DORMER_FRONT
    )
    z_values = [corner[2] for corner in front.corners]
    assert max(z_values) - min(z_values) <= 0.75


def test_reconstruct_dormers_keeps_same_xz_cutout_on_own_story():
    model = make_gable_model(include_dormer=True)
    obliques = _obliques(model)
    candidates = detect_dormers(model, obliques)
    assert candidates

    lower_room = model.rooms[0]
    upper_room = dataclasses.replace(lower_room, index=1, story=1)
    model = dataclasses.replace(
        model,
        stories_found=2,
        rooms=[lower_room, upper_room],
        scan_rooms_found=2,
        scan_rooms_transformed=2,
    )

    # Both flat ceilings overlap the lower-story dormer cutout in XZ. The
    # cutout belongs to room 0, so it must not punch through room 1's ceiling.
    rect = [[2.5, 0.0, 1.0], [4.8, 0.0, 1.0], [4.8, 0.0, 3.0], [2.5, 0.0, 3.0]]
    ceiling_candidates = [
        CeilingCandidate(
            corners=[[x, 3.0, z] for x, _y, z in rect],
            plane=Plane(a=0.0, b=1.0, c=0.0, d=3.0),
            source=CeilingSource.FLAT_EMIT,
            locator_id=f"{model.uuid}::tier-ceiling-flat::0",
            story=0,
        ),
        CeilingCandidate(
            corners=[[x, 6.0, z] for x, _y, z in rect],
            plane=Plane(a=0.0, b=1.0, c=0.0, d=6.0),
            source=CeilingSource.FLAT_EMIT,
            locator_id=f"{model.uuid}::tier-ceiling-flat::1",
            story=1,
        ),
    ]
    ceilings = assemble_ceiling(ceiling_candidates)
    ceiling_stories = {
        candidate.locator_id: candidate.story for candidate in ceiling_candidates
    }
    before_area = 2.3 * 2.0

    updated, _dormer_thermal = reconstruct_dormers(
        ceilings,
        candidates,
        obliques,
        model,
        ceiling_stories,
    )

    by_locator = {piece.locator_id: piece for piece in updated}
    lower = by_locator[f"{model.uuid}::tier-ceiling-flat::0"]
    upper = by_locator[f"{model.uuid}::tier-ceiling-flat::1"]

    lower_poly = Polygon(
        [(corner.x, corner.z) for corner in lower.corners],
        holes=[[(corner.x, corner.z) for corner in hole] for hole in lower.holes],
    )
    upper_poly = Polygon(
        [(corner.x, corner.z) for corner in upper.corners],
        holes=[[(corner.x, corner.z) for corner in hole] for hole in upper.holes],
    )
    assert lower_poly.area < before_area
    assert upper_poly.area == pytest.approx(before_area)
    assert not upper.holes


def test_real_dormer_cheeks_are_valid_plane_local_quads():
    payload = build_tier_payload(
        "3a576e1b-4b3e-4f5d-8c20-b39b157fcf03",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )
    assert all(w.kind.value == "knee" for w in payload.knee_walls)
    cheeks = [w for w in payload.dormer_faces if w.kind == DormerFaceKind.DORMER_CHEEK]
    assert cheeks
    for cheek in cheeks:
        corners3 = [[c.x, c.y, c.z] for c in cheek.corners]
        plane = fit_plane_any(corners3)
        assert plane is not None
        poly = _project_corners_to_plane(cheek.corners, plane[:3])
        assert poly is not None
        assert poly.is_valid


def test_reported_shallow_dormer_on_gable_part_is_emitted():
    payload = build_tier_payload(
        "cf982769-994b-4193-a621-91d46d1a3344",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    fronts = [
        face
        for face in payload.dormer_faces
        if face.kind == DormerFaceKind.DORMER_FRONT
    ]

    assert len(fronts) == 1
    assert fronts[0].cutouts


def test_reported_lower_roof_part_dormers_are_emitted():
    payload = build_tier_payload(
        "e0155eef-34a5-4642-bca6-39b83ee42af1",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    fronts = [
        face
        for face in payload.dormer_faces
        if face.kind == DormerFaceKind.DORMER_FRONT
    ]

    assert len(fronts) >= 2


@pytest.mark.xfail(
    reason=(
        "Phase 5 per-wing fan-out splits the building's gable into per-wing "
        "pieces; dormers that sit near a wing boundary lose their host slope "
        "until the Step 6 valley resolver merges cross-wing slopes. Will be "
        "re-enabled after Step 6."
    ),
    strict=False,
)
def test_reported_windowed_dormer_fronts_are_emitted():
    cases = [
        ("0b75d30e-c50c-4fc6-88ff-fce983078aa4", 0, 3),
        ("0d3f2993-8386-4130-8f1c-b2938c410828", 0, 2),
        ("21af2a12-2a29-44b5-b703-fbaa208996e9", 10, 0),
    ]

    for uuid, room_idx, window_idx in cases:
        payload = build_tier_payload(
            uuid, Path("pipeline-outputs"), Path(".scan-cache")
        )
        target = payload.rooms[room_idx].windows[window_idx]
        target_center = _centroid([[c.x, c.y, c.z] for c in target.corners])

        matched = False
        for face in payload.dormer_faces:
            if face.kind != DormerFaceKind.DORMER_FRONT:
                continue
            for cutout in face.cutouts:
                center = _centroid([[c.x, c.y, c.z] for c in cutout.corners])
                if (
                    sum((center[idx] - target_center[idx]) ** 2 for idx in range(3))
                    < 0.25
                ):
                    matched = True
                    break
            if matched:
                break

        assert matched, (
            f"{uuid} room {room_idx} window {window_idx} has no dormer front cutout"
        )
