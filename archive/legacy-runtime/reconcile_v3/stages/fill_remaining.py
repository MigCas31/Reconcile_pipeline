"""fill_remaining — cover slab-above leftovers with the minimum needed geometry.

For each building part's top story, subtract existing slabs, flat
ceilings, and slanted roofs (projected to XZ) from the part footprint.
Each remaining piece is resolved by, in order:

1. **Slanted.** If a per-room ``SlopeHypothesis`` centroid lies inside
   the piece, emit ``V3SlantedRoof(rule="fill-remaining-slanted")``.
2. **Flat.** If the piece centroid is inside a top-story room whose
   wall-top spread ≤ ``FLAT_CEILING_SPREAD_M`` and outlier wall fraction
   < ``OUTLIER_FRACTION_MAX``, emit ``V3FlatCeiling(over="fill")``.
3. **Unresolved.** Otherwise emit ``V3UnresolvedRegion(reason=
   "uncovered-ambiguous")``. *Never invent geometry.*
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from ..audit import HypothesisTrace
from ..constants import FLAT_CEILING_SPREAD_M, OUTLIER_FRACTION_MAX, SCAN_NOISE_M
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import (
    V3FlatCeiling,
    V3Part,
    V3Slab,
    V3SlantedRoof,
    V3UnresolvedRegion,
)
from .flat_ceilings import _outlier_fraction
from .slanted_roofs import SlopeHypothesis, project_xz_onto_plane


def _safe_polygon_xz(corners: list) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = Polygon([(c[0], c[2]) for c in corners])
    except Exception:
        return None
    if poly.is_empty:
        return None
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return None
        if poly.is_empty or not poly.is_valid:
            return None
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    if poly.area <= 0:
        return None
    return poly


def _iter_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _room_wall_top_spread(room: V3Room) -> tuple[float | None, float]:
    tops = [
        max(c[1] for c in (w.get("corners") or []))
        for w in room.walls
        if w.get("corners") and len(w["corners"]) >= 3
    ]
    if len(tops) < 3:
        return None, 0.0
    return float(np.median(tops)), float(max(tops) - min(tops))


def cover_remaining_slab_above(
    building_uuid: str,
    rooms: list[V3Room],
    parts: list[V3Part],
    slabs: list[V3Slab],
    flat_ceilings: list[V3FlatCeiling],
    slanted_roofs: list[V3SlantedRoof],
    slope_hypotheses: dict[str, SlopeHypothesis] | None = None,
) -> tuple[list[V3FlatCeiling], list[V3SlantedRoof], list[V3UnresolvedRegion]]:
    slope_hypotheses = slope_hypotheses or {}

    ceilings: list[V3FlatCeiling] = []
    inferred_slants: list[V3SlantedRoof] = []
    unresolved: list[V3UnresolvedRegion] = []

    if not parts:
        return ceilings, inferred_slants, unresolved

    covered_components: list[Polygon] = []
    for s in slabs:
        p = _safe_polygon_xz(s.polygon)
        if p is not None:
            covered_components.append(p)
    for fc in flat_ceilings:
        p = _safe_polygon_xz(fc.footprint_xz)
        if p is not None:
            covered_components.append(p)
    for sr in slanted_roofs:
        p = _safe_polygon_xz(sr.corners)
        if p is not None:
            covered_components.append(p)
    covered = unary_union(covered_components) if covered_components else Polygon()

    hypothesis_points: list[tuple[Point, SlopeHypothesis, V3Room]] = []
    room_by_id: dict[str, V3Room] = {r.identifier: r for r in rooms}
    for room_id, hyp in slope_hypotheses.items():
        room = room_by_id.get(room_id)
        if room is None or not room.floor_polygon:
            continue
        cx = sum(c[0] for c in room.floor_polygon) / len(room.floor_polygon)
        cz = sum(c[2] for c in room.floor_polygon) / len(room.floor_polygon)
        hypothesis_points.append((Point(cx, cz), hyp, room))

    for part in parts:
        if not part.stories:
            continue
        top_story = max(part.stories)
        part_poly = _safe_polygon_xz(part.footprint_xz)
        if part_poly is None:
            continue

        try:
            leftovers = part_poly.difference(covered)
        except Exception as exc:
            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(
                        building_uuid,
                        "v3-unresolved",
                        f"fill-difference-failed-{part.id.rsplit('::', 1)[-1]}",
                    ),
                    room_id=None,
                    footprint_xz=list(part.footprint_xz),
                    y_range=None,
                    reason="fill-difference-failed",
                    context={"error": str(exc)[:200]},
                    trace=HypothesisTrace(
                        stage="fill_remaining",
                        rule="shapely-difference-failure",
                        inputs={"part_id": part.id},
                        decision_reason=(
                            "Shapely difference of part footprint vs covered geometry "
                            "raised — cannot compute leftovers for this part."
                        ),
                    ),
                )
            )
            continue

        top_story_rooms: list[tuple[V3Room, Polygon]] = []
        for room in rooms:
            if room.story != top_story:
                continue
            rp = _safe_polygon_xz(room.floor_polygon)
            if rp is not None:
                top_story_rooms.append((room, rp))

        for piece_idx, piece in enumerate(_iter_polygons(leftovers)):
            if piece.area < SCAN_NOISE_M * SCAN_NOISE_M:
                continue
            inner = f"{part.id.rsplit('::', 1)[-1]}-{piece_idx}"
            centroid = piece.centroid
            piece_corners_xz = [
                (float(x), 0.0, float(z)) for x, z in list(piece.exterior.coords)
            ]

            slanted_hit: tuple[Point, SlopeHypothesis, V3Room] | None = None
            for hp in hypothesis_points:
                if piece.contains(hp[0]):
                    slanted_hit = hp
                    break
            if slanted_hit is not None:
                _hp_pt, hyp, room = slanted_hit
                room_poly = _safe_polygon_xz(room.floor_polygon)
                try:
                    clipped = (
                        piece.intersection(room_poly)
                        if room_poly is not None
                        else piece
                    )
                except Exception:
                    clipped = piece
                clipped_pieces = _iter_polygons(clipped)
                if not clipped_pieces:
                    clipped_pieces = [piece]
                plane = hyp.plane or (0.0, 1.0, 0.0, 0.0)
                y_ref = _room_wall_top_spread(room)[0] or 0.0
                for sub_idx, sub in enumerate(clipped_pieces):
                    if sub.area < SCAN_NOISE_M * SCAN_NOISE_M:
                        continue
                    sub_corners = [
                        (float(x), 0.0, float(z)) for x, z in list(sub.exterior.coords)
                    ]
                    seeded = [
                        (float(x), float(y_ref), float(z)) for x, _, z in sub_corners
                    ]
                    corners_3d = project_xz_onto_plane(seeded, plane)
                    suffix = (
                        f"fill-{inner}" if sub_idx == 0 else f"fill-{inner}-{sub_idx}"
                    )
                    inferred_slants.append(
                        V3SlantedRoof(
                            id=make_element_id(
                                building_uuid, "v3-slanted-roof", suffix
                            ),
                            plane=plane,
                            corners=corners_3d,
                            source_segment_ids=(),
                            trace=HypothesisTrace(
                                stage="fill_remaining",
                                rule="fill-remaining-slanted",
                                inputs={
                                    "part_id": part.id,
                                    "room_id": room.identifier,
                                    "hypothesis_sources": list(hyp.sources),
                                    "confidence": hyp.confidence,
                                },
                                decision_reason=(
                                    "Uncovered piece contains a SlopeHypothesis "
                                    "centroid "
                                    "— inherit slope; polygon clipped to room "
                                    "footprint."
                                ),
                            ),
                        )
                    )
                continue

            containing_room: tuple[V3Room, Polygon] | None = None
            for room, rp in top_story_rooms:
                if rp.contains(centroid):
                    containing_room = (room, rp)
                    break
            if containing_room is None:
                min_dist = float("inf")
                for room, rp in top_story_rooms:
                    d = rp.distance(centroid)
                    if d < min_dist and d <= SCAN_NOISE_M:
                        min_dist = d
                        containing_room = (room, rp)

            if containing_room is not None:
                room, _ = containing_room
                median_top, spread = _room_wall_top_spread(room)
                top_ys = [
                    float(max(c[1] for c in (w.get("corners") or [])))
                    for w in room.walls
                    if w.get("corners") and len(w["corners"]) >= 3
                ]
                out_frac = _outlier_fraction(room.walls, top_ys)
                if (
                    median_top is not None
                    and spread <= FLAT_CEILING_SPREAD_M
                    and out_frac < OUTLIER_FRACTION_MAX
                ):
                    ceilings.append(
                        V3FlatCeiling(
                            id=make_element_id(
                                building_uuid, "v3-flat-ceiling", f"fill-{inner}"
                            ),
                            room_id=room.identifier,
                            footprint_xz=[
                                (float(x), float(median_top), float(z))
                                for x, _, z in piece_corners_xz
                            ],
                            y=float(median_top),
                            over="fill",
                            trace=HypothesisTrace(
                                stage="fill_remaining",
                                rule="fill-remaining-flat",
                                inputs={
                                    "part_id": part.id,
                                    "room_id": room.identifier,
                                    "wall_top_spread_m": round(spread, 3),
                                    "outlier_fraction": round(out_frac, 3),
                                    "y": round(median_top, 3),
                                },
                                decision_reason=(
                                    "Piece lies inside a top-story room with flat wall "
                                    "tops — "
                                    "copy that room's ceiling Y."
                                ),
                            ),
                        )
                    )
                    continue

            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(building_uuid, "v3-unresolved", f"fill-{inner}"),
                    room_id=None,
                    footprint_xz=piece_corners_xz,
                    y_range=None,
                    reason="uncovered-ambiguous",
                    context={
                        "part_id": part.id,
                        "piece_area_m2": round(float(piece.area), 3),
                        "containing_room_id": containing_room[0].identifier
                        if containing_room
                        else None,
                    },
                    trace=HypothesisTrace(
                        stage="fill_remaining",
                        rule="no-slope-no-flat",
                        inputs={
                            "part_id": part.id,
                            "piece_area_m2": round(float(piece.area), 3),
                        },
                        decision_reason=(
                            "Uncovered top-story piece has neither a SlopeHypothesis "
                            "nor a flat-wall-top room supporting a flat cap."
                        ),
                    ),
                )
            )

    return ceilings, inferred_slants, unresolved
