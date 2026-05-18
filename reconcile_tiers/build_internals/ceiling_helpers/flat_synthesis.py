"""Flat ceiling synthesis: emit FLAT_EMIT candidates over rooms whose
floor is covered by `roof.flat`, plus mirror-oblique synthesis for
flat-overshoots and full per-room flat lids.

Mirror-oblique synthesis: many rooms emit a wall-top FLAT_EMIT lid that
blankets a footprint wider than the gable above. Where one chosen
oblique's plane extrapolates above flat_y but no opposing oblique was
reconstructed, the flat sits on the unmirrored side of the gable's ridge
- that area is physically sloped. Mirror the chosen plane across its
ridge to synthesise the missing slope, and clip the flat to what's left.

Audit cohort (`reconcile_tiers.audit.flat_overshoots_oblique`):
1022 / 2711 flats (38%) overshoot at least one chosen oblique's ridge
across 134 / 223 buildings.
"""

from __future__ import annotations

from typing import Any

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.assemble.ceiling_painter import CeilingCandidate
from reconcile_tiers.build_internals.ceiling_helpers._misc import (
    _room_oblique_raw_coverage,
    _vec3_at_y_from_polygon,
)
from reconcile_tiers.build_internals.ceiling_helpers.kink_detection import (
    _kink_flat_is_high_ridge_artifact,
)
from reconcile_tiers.build_internals.constants import (
    _NEAR_VERTICAL_NORMAL_Y_ABS_MAX,
    _SYNTHESIS_FLAT_COVERAGE_MIN,
    _SYNTHESIS_FLAT_OBLIQUE_RAW_COVERAGE_MIN,
    _SYNTHESIS_FLAT_OBLIQUE_RAW_MIN_AREA_M2,
    MIRROR_OBLIQUE_MIN_AREA_M2,
    MIRROR_OBLIQUE_MIN_DOMAIN_OVERLAP_M2,
)
from reconcile_tiers.build_internals.polygon_utils import (
    _polygon_parts_2d,
    _polygon_xz_from_corners,
    _room_floor_xz_polygon,
    _vec3_on_plane_from_polygon,
)
from reconcile_tiers.build_internals.raw_snapping import _fit_candidate
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingSource
from reconcile_tiers.roof.roof import RoofModel


def _half_plane_polygon(*args, **kwargs):
    from reconcile_tiers.assemble.synthesis import _half_plane_polygon as _hp

    return _hp(*args, **kwargs)


def _room_has_oblique_evidence_for_flat_synthesis(room) -> bool:
    """Guard against synthesising a flat ceiling over a room whose own scan
    says the room is mostly oblique.

    A single oblique fragment is not enough: mixed rooms often contain flat
    and sloped regions. This only blocks synthesis when oblique raw support
    explains most of the room footprint.
    """
    oblique_area, oblique_ratio = _room_oblique_raw_coverage(room)
    return (
        oblique_area >= _SYNTHESIS_FLAT_OBLIQUE_RAW_MIN_AREA_M2
        and oblique_ratio >= _SYNTHESIS_FLAT_OBLIQUE_RAW_COVERAGE_MIN
    )


def _synthesis_flat_xz_union(roof: RoofModel):
    """Union of all roof.flat XZ polygons, or None if there are none."""
    from shapely.geometry import Polygon as _ShPoly
    from shapely.ops import unary_union as _unary_union

    polys = []
    for surface in roof.flat:
        if len(surface.corners) < 3:
            continue
        try:
            poly = _ShPoly([(float(c[0]), float(c[2])) for c in surface.corners])
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            continue
        polys.append(poly)
    if not polys:
        return None
    return _unary_union(polys)


def _split_flat_for_unmirrored_obliques(
    domain_xz,
    flat_y: float,
    roof: RoofModel,
    story: int,
    extra_subtract_polys: list | None = None,
):
    """Clip ``domain_xz`` against unmirrored oblique ridges; return the
    clipped flat domain and synthesised opposing-slope candidates.

    For each oblique O in ``roof.oblique`` with matching ``dominant_story``
    whose physical XZ overlaps the domain:

    * ridge line in XZ:  a·x + c·z = d - b·flat_y
    * uphill half-plane (where O's plane > flat_y): a·x + c·z <= d - b·flat_y
    * unmirrored region = (uphill n domain) - O.xz_polygon
                          - (other obliques' XZ) - ``extra_subtract_polys``

    If the unmirrored region exceeds ``MIRROR_OBLIQUE_MIN_AREA_M2``, mirror
    O's plane across the ridge (mirror plane = (-a, b, -c, 2·b·flat_y - d))
    and emit it as a synthesised slope candidate. The unmirrored region is
    subtracted from the running flat domain.

    Returns ``(clipped_domain, [(corners_3d, plane), ...])``.
    """
    if domain_xz is None or domain_xz.is_empty:
        return domain_xz, []

    running = domain_xz
    synths: list[tuple[list[list[float]], Plane]] = []

    candidates: list[tuple[Plane, Any]] = []
    for surface in roof.oblique:
        if surface.dominant_story != story:
            continue
        if abs(surface.plane.b) >= _NEAR_VERTICAL_NORMAL_Y_ABS_MAX:
            continue
        if not surface.corners:
            continue
        ob_poly = _polygon_xz_from_corners(surface.corners)
        if ob_poly is None:
            continue
        try:
            overlap = ob_poly.intersection(domain_xz).area
        except Exception:
            overlap = 0.0
        if overlap < MIRROR_OBLIQUE_MIN_DOMAIN_OVERLAP_M2:
            continue
        candidates.append((surface.plane, ob_poly))

    extras = list(extra_subtract_polys or [])

    for plane, ob_poly in candidates:
        a, b, c, d = plane.a, plane.b, plane.c, plane.d
        if a * a + c * c < 1e-9:
            continue
        rhs = d - b * flat_y
        uphill = _half_plane_polygon(domain_xz, a, c, rhs)
        if uphill is None or uphill.is_empty:
            continue
        try:
            unmirrored = uphill.intersection(running)
        except Exception:
            continue
        if unmirrored.is_empty:
            continue
        for _other_plane, other_poly in candidates:
            if other_poly is ob_poly or other_poly is None:
                continue
            try:
                unmirrored = unmirrored.difference(other_poly)
            except Exception:
                pass
        try:
            unmirrored = unmirrored.difference(ob_poly)
        except Exception:
            pass
        for extra_poly in extras:
            if extra_poly is None or getattr(extra_poly, "is_empty", True):
                continue
            try:
                unmirrored = unmirrored.difference(extra_poly)
            except Exception:
                pass
        parts = _polygon_parts_2d(unmirrored)
        parts = [p for p in parts if p.area >= MIRROR_OBLIQUE_MIN_AREA_M2]
        if not parts:
            continue
        # Mirror plane across the ridge: same |slope|, opposite (a, c).
        # On the ridge line a·x + c·z = d - b·flat_y, mirror y = flat_y;
        # off-ridge it slopes downward away from the ridge.
        mirror_plane = Plane(a=-a, b=b, c=-c, d=2.0 * b * flat_y - d)
        for part in parts:
            corners = _vec3_on_plane_from_polygon(part, mirror_plane)
            if len(corners) < 3:
                continue
            synths.append((corners, mirror_plane))
            try:
                running = running.difference(part)
            except Exception:
                continue

    return running, synths


def _synthesised_flat_candidate_for_room(
    model: BuildingModel,
    room,
    synthesis_flat_xz,
    roof: RoofModel | None = None,
) -> list[CeilingCandidate]:
    """Emit per-room FLAT_EMIT candidates (and synthesised opposing-slope
    mirrors) when roof.flat covers the room.

    Y comes from the median of per-wall max-y (wall tops). XZ corners come
    from the room's floor polygon, which is already axis-aligned because the
    walls are. Suppresses raw_fallback fragmentation in the same XZ via the
    painter's by-story occupancy clip.

    With ``roof`` supplied, the floor footprint is split by
    ``_split_flat_for_unmirrored_obliques`` so unmirrored gable sides become
    synthesised opposing-slope candidates instead of flat blanket.
    """
    if _kink_flat_is_high_ridge_artifact(
        room
    ) or _room_has_oblique_evidence_for_flat_synthesis(room):
        return []
    if len(room.floor_polygon) < 3:
        return []
    from shapely.geometry import Polygon as _ShPoly

    try:
        floor_xz = _ShPoly([(float(p[0]), float(p[2])) for p in room.floor_polygon])
    except Exception:
        return []
    if not floor_xz.is_valid:
        floor_xz = floor_xz.buffer(0)
    if floor_xz.is_empty or floor_xz.area < 0.5:
        return []
    overlap = floor_xz.intersection(synthesis_flat_xz).area
    if overlap / floor_xz.area < _SYNTHESIS_FLAT_COVERAGE_MIN:
        return []

    walls = room.walls_computed or room.walls_merged
    wall_top_ys = [
        max(float(c[1]) for c in wall.corners)
        for wall in walls
        if len(wall.corners) >= 3
    ]
    if not wall_top_ys:
        return []
    wall_top_ys.sort()
    ceiling_y = wall_top_ys[len(wall_top_ys) // 2]

    flat_locator = f"{model.uuid}::tier-ceiling-flat::{room.index}"
    synth_prefix = (
        f"{model.uuid}::tier-ceiling-roof-arrangement-room::{room.index}:synth-mirror"
    )
    domain = floor_xz
    candidates: list[CeilingCandidate] = []
    if roof is not None:
        domain, synth_pieces = _split_flat_for_unmirrored_obliques(
            domain, ceiling_y, roof, room.story
        )
        for synth_idx, (synth_corners, _plane) in enumerate(synth_pieces):
            cand = _fit_candidate(
                synth_corners,
                CeilingSource.ROOF_ARRANGEMENT,
                f"{synth_prefix}:{synth_idx}",
                story=room.story,
            )
            if cand is not None:
                candidates.append(cand)
    if domain is not None and not getattr(domain, "is_empty", False):
        for part_idx, part in enumerate(_polygon_parts_2d(domain)):
            corners = _vec3_at_y_from_polygon(part, ceiling_y)
            cand = _fit_candidate(
                corners,
                CeilingSource.FLAT_EMIT,
                flat_locator if part_idx == 0 else f"{flat_locator}:covered:{part_idx}",
                story=room.story,
            )
            if cand is not None:
                candidates.append(cand)
        if candidates:
            return candidates
    if candidates:
        return candidates
    fallback_corners = [
        [float(p[0]), float(ceiling_y), float(p[2])] for p in room.floor_polygon
    ]
    cand = _fit_candidate(
        fallback_corners,
        CeilingSource.FLAT_EMIT,
        flat_locator,
        story=room.story,
    )
    return [cand] if cand is not None else []


def _flat_ceiling_candidates_for_domain(
    room,
    domain,
    locator_id: str,
) -> list[CeilingCandidate]:
    if len(room.ceiling_polygon) < 3:
        return []
    y = sum(float(p[1]) for p in room.ceiling_polygon) / len(room.ceiling_polygon)
    candidates: list[CeilingCandidate] = []
    for part_idx, part in enumerate(_polygon_parts_2d(domain)):
        corners = _vec3_at_y_from_polygon(part, y)
        candidate = _fit_candidate(
            corners,
            CeilingSource.FLAT_EMIT,
            locator_id if part_idx == 0 else f"{locator_id}:covered:{part_idx}",
            story=room.story,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _flat_room_ceiling_candidates(
    room,
    locator_id: str,
    roof: RoofModel | None = None,
    synth_locator_prefix: str | None = None,
) -> list[CeilingCandidate]:
    """Emit a flat room lid over the physical room footprint.

    RoomPlan/noMesh ceiling polygons can be smaller than the room floor even
    when the room is correctly classified as flat. For flat rooms, the physical
    ceiling should span the occupied floor footprint; any lower-priority raw
    scan fragments or gap caps can then be clipped by the painter.

    With ``roof`` supplied, the floor footprint is split against unmirrored
    oblique ridges via ``_split_flat_for_unmirrored_obliques``: areas on the
    uphill side of a chosen oblique's ridge are emitted as synthesised
    opposing-slope candidates (ROOF_ARRANGEMENT) instead of flat blanket.
    """
    if len(room.ceiling_polygon) < 3:
        return []
    flat_y = sum(float(p[1]) for p in room.ceiling_polygon) / len(room.ceiling_polygon)
    floor_poly = _room_floor_xz_polygon(room)
    candidates: list[CeilingCandidate] = []
    domain = floor_poly
    if domain is not None and roof is not None:
        domain, synth_pieces = _split_flat_for_unmirrored_obliques(
            domain, flat_y, roof, room.story
        )
        for synth_idx, (synth_corners, _plane) in enumerate(synth_pieces):
            synth_locator = (
                f"{synth_locator_prefix}:{synth_idx}"
                if synth_locator_prefix is not None
                else f"{locator_id}:synth-mirror:{synth_idx}"
            )
            cand = _fit_candidate(
                synth_corners,
                CeilingSource.ROOF_ARRANGEMENT,
                synth_locator,
                story=room.story,
            )
            if cand is not None:
                candidates.append(cand)
    if domain is not None and not getattr(domain, "is_empty", False):
        flat_cands = _flat_ceiling_candidates_for_domain(room, domain, locator_id)
        if flat_cands:
            candidates.extend(flat_cands)
            return candidates
    if candidates:
        return candidates
    candidate = _fit_candidate(
        room.ceiling_polygon,
        CeilingSource.FLAT_EMIT,
        locator_id,
        story=room.story,
    )
    return [candidate] if candidate is not None else []
