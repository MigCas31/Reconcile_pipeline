"""Synthesis utilities — geometry that is *invented* (not directly observed in scans).

Lives separately from the scan-driven assemble path so the ceiling painter can
stay focused on merging observed scan candidates. The long-term goal is to
collapse all synthesis into a single late `synthesise_thermal_envelope` pass
that runs after rooms/walls/roof are assembled. This module is the home for
that work as it migrates out of `reconcile_tiers/build.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from reconcile_tiers.config import architectural_priors_enabled

FLAT_EXTEND_BBOX_PADDING_M = 50.0
ENVELOPE_FLAT_ROOM_TOL_M = 0.10


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Output of the late-stage synthesis pass.

    Bundles the post-painter ceilings together with all geometry that is
    invented after scan-driven assembly: knee walls, gable closures, dormer
    faces, and visual shells.
    """

    ceilings: list
    knee_walls: list
    gable_closures: list
    dormer_faces: list
    visual_shells: list = field(default_factory=list)


def synthesise_thermal_envelope(
    rooms,
    walls,
    ceilings,
    roof,
    dormer_thermal,
    *,
    model,
) -> SynthesisResult:
    """Late-stage post-painter synthesis pass.

    Operates on already-painted ceilings (output of `assemble_ceiling`) and
    produces the final thermal envelope: filtered/relabeled ceilings + knee
    walls + gable closures + dormer faces + visual shells.

    The intent is to keep `assemble_ceiling` purely scan-driven (merge raw
    candidates by priority) and let synthesis run AFTER as a separate pass
    that has the assembled rooms + final roof envelope to work with. This
    aligns with the "roof first, ceilings default to flat, no circular
    references" principle.

    Parameters mirror the v2 thermal-envelope inputs but at piece-level
    (post-paint) instead of candidate-level (pre-paint):

    - rooms: assembled `Room` list (with walls).
    - walls: reserved for future late-stage wall synthesis; unused today.
    - ceilings: list[CeilingPiece] from `assemble_ceiling`.
    - roof: `RoofModel` (kinks, thermal, oblique surfaces).
    - dormer_thermal: thermal surfaces from `reconstruct_dormers`.
    - model: `BuildingModel` (raw planes, room geometry — needed for the
      raw-quality lookup applied to RAW_FALLBACK pieces).
    """
    # walls is reserved for a future late-stage wall-synthesis step; not
    # consumed today. Touch it so unused-arg linters stay quiet.
    _ = walls

    from reconcile_tiers.assemble.thermal_envelope import (
        RAW_QUALITY_THRESHOLD,
        _piece_raw_quality,
        _relabel_piece,
    )
    from reconcile_tiers.assemble.visual_shell import assemble_visual_shells
    from reconcile_tiers.build_internals.envelope_pieces import (
        _dormer_faces,
        _gable_closures,
        _knee_walls,
    )
    from reconcile_tiers.payload.schema import CeilingSource

    rooms_by_index = {room.index: room for room in model.rooms}

    out_ceilings = []
    for piece in ceilings:
        # Drop attic-shell pieces — visualisation-only, do not belong in the
        # thermal envelope. Mirrors v2's `_is_attic_shell` candidate filter.
        if piece.source == CeilingSource.ROOF_ARRANGEMENT_ATTIC:
            continue

        # Clip ROOF_ARRANGEMENT (computed oblique) at the room's eave when a
        # knee wall takes over below. Mirrors v2's `_clip_to_eave` candidate
        # transform but applied to the painted piece.
        if piece.source == CeilingSource.ROOF_ARRANGEMENT:
            clipped = _clip_piece_to_eave(piece, rooms_by_index, roof)
            if clipped is None:
                continue
            piece = clipped

        # Quality-gate RAW_FALLBACK pieces in flat-classified rooms. Mirrors
        # v2's per-locator quality gate.
        if piece.source == CeilingSource.RAW_FALLBACK:
            room_idx = _parse_room_idx_from_raw_locator(piece.locator_id)
            room = rooms_by_index.get(room_idx) if room_idx is not None else None
            if room is not None and room.ceiling_type == "flat":
                quality = _piece_raw_quality(piece, model, roof)
                if quality < RAW_QUALITY_THRESHOLD:
                    continue

        out_ceilings.append(_relabel_piece(piece, model, roof))

    return SynthesisResult(
        ceilings=out_ceilings,
        knee_walls=_knee_walls(model, roof, rooms),
        gable_closures=_gable_closures(model, roof),
        dormer_faces=_dormer_faces(model, rooms, dormer_thermal),
        visual_shells=assemble_visual_shells(model.uuid, roof, model),
    )


def _parse_room_idx_from_arrangement_locator(locator_id: str | None) -> int | None:
    """Per-room arrangement pieces (`_synthesise_room_arrangement_pieces`)
    encode the room index in the locator —
    `tier-ceiling-roof-arrangement-room::<r>:<s>:<p>`
    — and leave `arrangement_cell_id` as None. Pull it out of the locator instead."""
    if not locator_id:
        return None
    marker = "::tier-ceiling-roof-arrangement-room::"
    if marker not in locator_id:
        return None
    suffix = locator_id.rsplit(marker, 1)[1]
    head = suffix.split(":", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _clip_piece_to_eave(piece, rooms_by_index, roof):
    """Piece-level twin of thermal_envelope._clip_to_eave.

    Clips an in-room oblique piece to the room's vertical envelope:
    keeps only the slice between eave_y (bottom) and lid_y (top). Flat
    rooms (lid ≈ eave) drop the piece entirely — the gable oblique above
    a flat room is attic, handled by visual_shells.
    """
    from reconcile_tiers._core.shapely2 import split_polygon_at_y
    from reconcile_tiers.assemble.thermal_envelope import _parse_room_idx

    # Bare per-room arrangement pieces are observed/wall-derived room ceilings,
    # not roof-shell cells. Do not clip them to roof.kinks eave/lid; the kink
    # detector and wall-derived sloped-room path already describe the occupied
    # room ceiling, and roof.kinks can sit below that scan evidence. Kinked
    # double-slanted rooms may emit a second observed slope as `::<room>:1`.
    # Synthesised gable room pieces use `::<room>:gable...`; keep them only
    # when the room has real vertical gable relief.
    arrangement_tail = (
        piece.locator_id.rsplit("::tier-ceiling-roof-arrangement-room::", 1)[1]
        if piece.locator_id is not None
        and "::tier-ceiling-roof-arrangement-room::" in piece.locator_id
        else None
    )
    arrangement_tail_parts = (
        arrangement_tail.split(":") if arrangement_tail is not None else []
    )
    from_room_ceiling = (
        piece.arrangement_cell_id is None
        and arrangement_tail is not None
        and len(arrangement_tail_parts) <= 2
    )
    from_synthesised_gable_room = len(
        arrangement_tail_parts
    ) == 2 and arrangement_tail_parts[1].startswith("gable")

    room_idx = _parse_room_idx(piece.arrangement_cell_id)
    if room_idx is None:
        room_idx = _parse_room_idx_from_arrangement_locator(piece.locator_id)
    if room_idx is None:
        return piece
    room = rooms_by_index.get(room_idx)
    if room is None:
        return piece
    eave_y = roof.kinks.eave_y(room_idx)
    lid_y = roof.kinks.attic_lid_y(room_idx)
    if eave_y is None or lid_y is None:
        return piece
    if from_room_ceiling and not (
        from_synthesised_gable_room and lid_y - eave_y < ENVELOPE_FLAT_ROOM_TOL_M
    ):
        return piece
    # Vaulted-room override (priors-on only — goldens depend on the
    # priors-OFF heuristic). When the room's wall topology shows a peak
    # well above `attic_lid_y` (i.e. the room is open to a ridge), the
    # heuristic-derived `lid_y` underestimates the ceiling height and
    # would truncate a primitive-synthesised gable. Use the actual max
    # wall-top y as the effective lid for those rooms.
    if architectural_priors_enabled():
        walls = room.walls_computed or room.walls_merged
        wall_top_max = max(
            (max(float(c[1]) for c in w.corners) for w in walls if len(w.corners) >= 3),
            default=None,
        )
        if wall_top_max is not None and wall_top_max > lid_y + 0.10:
            lid_y = float(wall_top_max)
    if lid_y - eave_y < ENVELOPE_FLAT_ROOM_TOL_M:
        # Sloped non-top-story rooms with no synthesised oblique above store
        # `attic_lid_y == eave_y` (because there's no clear "lid"), but they
        # DO have a real wall-derived ceiling — passed in via Phase 3 Step 3
        # as the room's `tier-ceiling-roof-arrangement-room::<r>` piece (no
        # cell suffix, no arrangement_cell_id). Skip the drop for that
        # specific shape. Pre-existing gable per-room pieces (suffixed
        # `<r>:<s>:<p>` AND with arrangement_cell_id set) and gable arrangement
        # pieces above flat rooms are still dropped as attic.
        return None

    corners = [[c.x, c.y, c.z] for c in piece.corners]
    ys = [c[1] for c in corners]
    if max(ys) > lid_y + 1e-3:
        if min(ys) >= lid_y - 1e-3:
            return None
        below_lid, _above_lid = split_polygon_at_y(corners, lid_y)
        if len(below_lid) < 3:
            return None
        corners = below_lid
        ys = [c[1] for c in corners]

    if min(ys) < eave_y - 1e-3:
        if max(ys) <= eave_y + 1e-3:
            return None
        _below_eave, above_eave = split_polygon_at_y(corners, eave_y)
        if len(above_eave) < 3:
            return None
        corners = above_eave

    return _piece_with_clean_corners(piece, corners)


def _piece_with_clean_corners(piece, corners):
    """Return piece with clipped corners repaired to a valid XZ polygon.

    `split_polygon_at_y` can duplicate near-coincident vertices when a clip
    line grazes an existing vertex. That is harmless for drawing but invalid
    for the payload contract, so repair in XZ and re-lift to the piece plane.
    """
    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon
    from reconcile_tiers.payload.schema import Vec3

    if len(corners) < 3:
        return None
    poly = make_valid_polygon(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    if poly is None or poly.is_empty:
        return None
    if not isinstance(poly, Polygon):
        parts = [
            part
            for part in getattr(poly, "geoms", [])
            if isinstance(part, Polygon) and not part.is_empty and part.area > 0.0
        ]
        if not parts:
            return None
        poly = max(parts, key=lambda part: part.area)
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None
    plane = piece.plane
    if abs(float(plane.b)) < 1e-9:
        return None
    cleaned = []
    for x, z in coords:
        y = (
            float(plane.d) - float(plane.a) * float(x) - float(plane.c) * float(z)
        ) / float(plane.b)
        cleaned.append(Vec3(x=float(x), y=float(y), z=float(z)))
    return replace(piece, corners=cleaned, holes=[])


def _parse_room_idx_from_raw_locator(locator_id: str) -> int | None:
    marker = "::tier-ceiling-raw::"
    if marker not in locator_id:
        return None
    suffix = locator_id.rsplit(marker, 1)[1]
    parts = suffix.split(":")
    if len(parts) < 1:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def _extend_flat_to_obliques(
    corners: list[list[float]],
    obliques,
    room_envelope: list[list[float]] | None = None,
) -> list[list[float]]:
    if len(corners) < 3 or not obliques:
        return corners
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    from reconcile_tiers._core.shapely2 import make_valid_polygon
    from reconcile_tiers.build_internals.gable_selection import (
        _oblique_xz_polygon,
        _select_gable_oblique_pair,
    )

    y_flat = sum(float(p[1]) for p in corners) / len(corners)
    base = make_valid_polygon(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    if base is None or base.is_empty:
        return corners

    room_poly = None
    if room_envelope is not None and len(room_envelope) >= 3:
        room_poly = make_valid_polygon(
            Polygon([(float(p[0]), float(p[2])) for p in room_envelope])
        )
    if room_poly is not None and not room_poly.is_empty:
        selected = _select_gable_oblique_pair(obliques, room_poly)
        if selected is None:
            return corners
        selected_obliques, oblique_union = selected
    else:
        selected_obliques = list(obliques)
        oblique_xz_polys = [
            poly
            for ob in selected_obliques
            if (poly := _oblique_xz_polygon(ob)) is not None
        ]
        if not oblique_xz_polys:
            return corners
        oblique_union = unary_union(oblique_xz_polys)

    bounds_xs = [float(p[0]) for ob in selected_obliques for p in ob.corners] + [
        base.bounds[0],
        base.bounds[2],
    ]
    bounds_zs = [float(p[2]) for ob in selected_obliques for p in ob.corners] + [
        base.bounds[1],
        base.bounds[3],
    ]
    pad = FLAT_EXTEND_BBOX_PADDING_M
    bbox = box(
        min(bounds_xs) - pad,
        min(bounds_zs) - pad,
        max(bounds_xs) + pad,
        max(bounds_zs) + pad,
    )

    cross_section = bbox
    for ob in selected_obliques:
        a, b, c, d = (
            float(ob.plane.a),
            float(ob.plane.b),
            float(ob.plane.c),
            float(ob.plane.d),
        )
        if abs(b) < 1e-9:
            continue
        rhs = d - b * y_flat
        half = _half_plane_polygon(bbox, a, c, rhs)
        if half is None or half.is_empty:
            return corners
        cross_section = cross_section.intersection(half)
        if cross_section.is_empty:
            return corners

    cross_section = cross_section.intersection(oblique_union)
    if room_poly is not None and not room_poly.is_empty:
        cross_section = cross_section.intersection(room_poly)
    if cross_section.is_empty:
        return corners
    clipped = make_valid_polygon(cross_section)
    if clipped is None or clipped.is_empty:
        return corners
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)
    coords = list(clipped.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(x), float(y_flat), float(z)] for x, z in coords]


def _half_plane_polygon(bbox, a: float, c: float, rhs: float):
    from shapely.geometry import LineString
    from shapely.ops import split

    denom = a * a + c * c
    if denom < 1e-12:
        return bbox
    t = rhs / denom
    foot_x, foot_z = a * t, c * t
    far = max(abs(b) for b in bbox.bounds) * 4 + 100.0
    p1 = (foot_x - far * (-c), foot_z - far * a)
    p2 = (foot_x + far * (-c), foot_z + far * a)
    line = LineString([p1, p2])
    try:
        parts = list(split(bbox, line).geoms)
    except Exception:
        return None
    for part in parts:
        cx, cz = part.centroid.x, part.centroid.y
        if a * cx + c * cz <= rhs + 1e-6:
            return part
    return None
