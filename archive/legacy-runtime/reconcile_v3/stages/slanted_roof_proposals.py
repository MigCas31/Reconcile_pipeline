"""slanted_roof_proposals -- permissive (slanted-segment x rain-exposed
slab piece) generator.

Human-in-the-loop labeling target. Each proposal pairs one slanted wall-edge
segment with one contiguous rain-exposed sub-region of a room floor. A floor
region is "rain-exposed" when no higher-floor room or closed gap sits above
it in XZ -- i.e. standing on that piece of floor, you'd see sky.

Slant threshold is deliberately very permissive (1 deg < incl < 89 deg) so edge
cases get surfaced for review. Each proposal carries a flat feature vector;
a best-effort ``heuristic_label`` compares against ``slanted_roofs``.

The authoritative slanted_roofs output is untouched; proposals live in a
parallel V3Building.roof_proposals list.
"""

from __future__ import annotations

from math import atan2, pi, sqrt
from typing import Any, Literal

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile.extract3d.overlaps import floor_polygon_to_shapely
from reconcile.roof_algorithms_py.math_utils import plane_normal

from ..audit import HypothesisTrace
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Gap, V3RoofProposal, V3Slab, V3SlantedRoof
from .slanted_roofs import project_xz_onto_plane

PERMISSIVE_MIN_INCL_DEG = 1.0
PERMISSIVE_MAX_INCL_DEG = 89.0
PERMISSIVE_MIN_SEG_LEN_M = 0.15
"""Floor is considered 'covered from above' when another room/closed gap has
its floor Y above this threshold (in metres) -- separates real upper stories
from same-story rooms with sub-cm scan noise."""
ABOVE_FLOOR_MIN_DELTA_M = 0.05
"""Minimum effective width (2*area/perimeter, metres) for an exposed piece
to be kept; this catches snaky slivers where the rotated bounding box's
short side stays large because the vertex span is wide but the actual
ribbon is thin."""
EXPOSED_PIECE_MIN_WIDTH_M = 0.10
"""Exposed pieces smaller than this area are dropped outright."""
EXPOSED_PIECE_MIN_AREA_M2 = 0.15


def _collect_permissive_slanted_segments(rooms: list[V3Room]) -> list[dict]:
    """Every slanted wall edge anywhere in the building, with room/wall origin.

    Much more permissive than the default oblique segment collector (1 deg vs 5 deg
    minimum inclination, 0.15 m vs 0.30 m minimum length).
    """
    segments: list[dict] = []
    seg_idx = 0
    for room in rooms:
        for wall in room.walls:
            corners = wall.get("corners") or []
            n = len(corners)
            if n < 3:
                continue
            for i in range(n):
                pa = corners[i]
                pb = corners[(i + 1) % n]
                if pa[1] < pb[1]:
                    pa, pb = pb, pa
                dx, dy, dz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
                length = sqrt(dx * dx + dy * dy + dz * dz)
                if length < PERMISSIVE_MIN_SEG_LEN_M:
                    continue
                horiz = sqrt(dx * dx + dz * dz)
                incl = atan2(abs(dy), horiz) * 180.0 / pi
                if incl <= PERMISSIVE_MIN_INCL_DEG or incl >= PERMISSIVE_MAX_INCL_DEG:
                    continue
                azimuth = (atan2(dx, dz) * 180.0 / pi) % 360.0
                segments.append(
                    {
                        "seg_idx": seg_idx,
                        "a": list(pa),
                        "b": list(pb),
                        "incl": incl,
                        "azimuth": azimuth,
                        "len": length,
                        "story": room.story,
                        "room_idx": room.index,
                        "room_id": room.identifier,
                        "wall_id": wall.get("id"),
                    }
                )
                seg_idx += 1
    return segments


def _plane_from_segment(segment: dict) -> tuple[float, float, float, float]:
    """Plane through segment midpoint with normal from segment's azimuth+incl."""
    n = plane_normal(segment["azimuth"], segment["incl"])
    mid = (
        (segment["a"][0] + segment["b"][0]) / 2.0,
        (segment["a"][1] + segment["b"][1]) / 2.0,
        (segment["a"][2] + segment["b"][2]) / 2.0,
    )
    d = -(n["x"] * mid[0] + n["y"] * mid[1] + n["z"] * mid[2])
    return (float(n["x"]), float(n["y"]), float(n["z"]), float(d))


def _polygon_compactness(poly: Polygon) -> float:
    p = float(poly.length)
    if p <= 0:
        return 0.0
    return float(4.0 * pi * poly.area / (p * p))


def _polygon_bbox_aspect(poly: Polygon) -> float:
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return 0.0
    return float(max(w, h) / min(w, h))


def _polygon_min_width(poly: Polygon) -> float:
    """Effective width for sliver detection: 2 x area / perimeter.

    For a true rectangle W x L this returns W exactly; for irregular ribbon
    shapes (L, Z, snake) it returns the ribbon's average thickness, which
    is what we actually want to threshold against. The minimum-rotated-
    rectangle short side is a worse metric here because it bounds the
    vertex span rather than the shape's ribbon width.
    """
    try:
        area = float(poly.area)
        perimeter = float(poly.length)
    except Exception:
        return 0.0
    if perimeter <= 0.0:
        return 0.0
    return 2.0 * area / perimeter


def _footprint_polygon_3d_to_shapely(footprint_xz) -> Polygon | None:
    """Convert a 3D XZ-plane footprint list (``[(x, y, z), ...]``) to a 2D
    Shapely polygon using (x, z)."""
    if not footprint_xz or len(footprint_xz) < 3:
        return None
    try:
        poly = Polygon([(float(p[0]), float(p[2])) for p in footprint_xz])
    except Exception:
        return None
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
        except Exception:
            return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if not isinstance(poly, Polygon) or poly.is_empty or not poly.is_valid:
        return None
    return poly


def _rain_exposure(
    room: V3Room,
    rooms: list[V3Room],
    gaps: list[V3Gap],
) -> Polygon | MultiPolygon | None:
    """Room floor polygon minus the XZ union of higher-floor rooms and closed
    gaps sitting above it.

    Returns None if the room has no valid floor polygon or nothing remains.
    Closed gaps are included in the "above" mask so that slivers beneath a
    sealed cross-floor gap (e.g. a gable end closure) don't leak into the
    rain-exposed set.
    """
    floor = floor_polygon_to_shapely(room.floor_polygon)
    if not isinstance(floor, Polygon) or floor.is_empty or not floor.is_valid:
        return None
    floor_y = float(room.floor_polygon[0][1]) if room.floor_polygon else 0.0

    above_polys: list[Polygon] = []
    for other in rooms:
        if other is room or not other.floor_polygon:
            continue
        other_y = float(other.floor_polygon[0][1])
        if other_y - floor_y <= ABOVE_FLOOR_MIN_DELTA_M:
            continue
        other_poly = floor_polygon_to_shapely(other.floor_polygon)
        if (
            isinstance(other_poly, Polygon)
            and not other_poly.is_empty
            and other_poly.is_valid
        ):
            above_polys.append(other_poly)

    for gap in gaps:
        if gap.status != "closed":
            continue
        if gap.floor_y - floor_y <= ABOVE_FLOOR_MIN_DELTA_M:
            continue
        gap_poly = _footprint_polygon_3d_to_shapely(gap.footprint_xz)
        if gap_poly is not None:
            above_polys.append(gap_poly)

    if not above_polys:
        return floor

    above = unary_union(above_polys)
    exposure = floor.difference(above)
    if exposure.is_empty:
        return None
    return exposure


def _exposed_pieces(
    exposure: Polygon | MultiPolygon | None,
) -> list[Polygon]:
    """Split a MultiPolygon exposure into individual Polygon pieces and drop
    slivers (below area or width threshold)."""
    if exposure is None or exposure.is_empty:
        return []
    candidates: list[Polygon]
    if isinstance(exposure, MultiPolygon):
        candidates = [
            g for g in exposure.geoms if isinstance(g, Polygon) and not g.is_empty
        ]
    elif isinstance(exposure, Polygon):
        candidates = [exposure]
    else:
        return []
    kept: list[Polygon] = []
    for piece in candidates:
        if not piece.is_valid or piece.is_empty:
            continue
        if float(piece.area) < EXPOSED_PIECE_MIN_AREA_M2:
            continue
        if _polygon_min_width(piece) < EXPOSED_PIECE_MIN_WIDTH_M:
            continue
        kept.append(piece)
    return kept


def _plane_y_at(
    point_xz: tuple[float, float], plane: tuple[float, float, float, float]
) -> float:
    a, b, c, d = plane
    if abs(b) < 1e-6:
        return 0.0
    return float(-(a * point_xz[0] + c * point_xz[1] + d) / b)


def _slant_delta_over_polygon(
    poly: Polygon, plane: tuple[float, float, float, float]
) -> float:
    a, b, c, d = plane
    if abs(b) < 1e-6:
        return 0.0
    ys = [-(a * float(x) + c * float(z) + d) / b for x, z in poly.exterior.coords]
    return float(max(ys) - min(ys)) if ys else 0.0


def _corners_on_plane(
    piece_poly: Polygon, plane: tuple[float, float, float, float]
) -> list[tuple[float, float, float]]:
    coords = list(piece_poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    corners_xz = [(float(x), 0.0, float(z)) for x, z in coords]
    return project_xz_onto_plane(corners_xz, plane)


def _heuristic_label_for_segment(
    segment: dict, slab_room_id: str, slanted_roofs: list[V3SlantedRoof]
) -> Literal["accepted", "rejected", "not_evaluated"]:
    """Best-effort comparison against the current pipeline's slanted_roofs.

    "accepted" = the segment's wall contributed to an existing roof AND that
    roof covers the slab's room. "rejected" = segment-in-roof but different
    room. "not_evaluated" = segment didn't clear the cluster threshold at all.
    """
    wall_id = segment.get("wall_id")
    wall_in_any_roof = False
    for roof in slanted_roofs:
        inputs = roof.trace.inputs if roof.trace else {}
        src_ids = roof.source_segment_ids or ()
        has_wall = any(
            isinstance(sid, str) and f"::{wall_id}::" in sid for sid in src_ids
        )
        if not has_wall:
            continue
        wall_in_any_roof = True
        ratifying_rooms = inputs.get("room_ids") or []
        if slab_room_id in ratifying_rooms:
            return "accepted"
    if wall_in_any_roof:
        return "rejected"
    return "not_evaluated"


def _compute_features(
    *,
    segment: dict,
    plane: tuple[float, float, float, float],
    target: dict,
    piece_poly: Polygon,
    piece_index: int,
    stories: list[int],
) -> dict[str, Any]:
    """Build the flat feature vector for a (segment x rain-exposed-piece) pair.

    ``target`` describes the piece's origin (room or closed gap): it carries
    ``kind``, ``slab_area_m2``, ``slab_vertex_count``, ``story``, ``floor_y``
    and (for rooms only) ``room_id`` -- so both origins produce the same schema.
    """
    slab_area = float(target["slab_area_m2"])
    piece_area = float(piece_poly.area)
    piece_perim = float(piece_poly.length)
    centroid = piece_poly.centroid
    floor_y = float(target["floor_y"])
    plane_y_centroid = _plane_y_at((float(centroid.x), float(centroid.y)), plane)
    seg_mid_xz = (
        (segment["a"][0] + segment["b"][0]) / 2.0,
        (segment["a"][2] + segment["b"][2]) / 2.0,
    )
    seg_mid_y = (segment["a"][1] + segment["b"][1]) / 2.0
    dxz = sqrt(
        (seg_mid_xz[0] - float(centroid.x)) ** 2
        + (seg_mid_xz[1] - float(centroid.y)) ** 2
    )
    piece_min_width = _polygon_min_width(piece_poly)
    target_story = int(target["story"])
    return {
        # segment geometry
        "segment_azimuth_deg": round(float(segment["azimuth"]), 2),
        "segment_incl_deg": round(float(segment["incl"]), 2),
        "segment_length_m": round(float(segment["len"]), 3),
        "segment_mid_y_m": round(float(seg_mid_y), 3),
        "segment_story": int(segment["story"]),
        # slab geometry (full target footprint, for context)
        "slab_area_m2": round(slab_area, 4),
        "slab_vertex_count": int(target["slab_vertex_count"]),
        "slab_story": target_story,
        "slab_kind": str(target["kind"]),
        # piece geometry (the rain-exposed sub-region this proposal covers)
        "piece_index": int(piece_index),
        "piece_area_m2": round(piece_area, 4),
        "piece_perimeter_m": round(piece_perim, 4),
        "piece_compactness": round(_polygon_compactness(piece_poly), 4),
        "piece_vertex_count": len(piece_poly.exterior.coords) - 1,
        "piece_bbox_aspect": round(_polygon_bbox_aspect(piece_poly), 4),
        "piece_min_width_m": round(piece_min_width, 4),
        "rain_exposure_ratio": round(piece_area / slab_area, 4)
        if slab_area > 0
        else 0.0,
        # relation
        "is_top_story_slab": bool(stories and target_story == max(stories)),
        "is_same_room": bool(
            target.get("room_id") is not None
            and segment.get("room_id") == target["room_id"]
        ),
        "story_delta": target_story - int(segment["story"]),
        "seg_mid_to_piece_centroid_xz_m": round(dxz, 3),
        # plane-vs-piece
        "slab_floor_y_m": round(floor_y, 4),
        "plane_y_at_piece_centroid_m": round(plane_y_centroid, 4),
        "plane_height_above_slab_m": round(plane_y_centroid - floor_y, 4),
        "slant_delta_over_piece_m": round(
            _slant_delta_over_polygon(piece_poly, plane), 4
        ),
    }


def _collect_target_pieces(
    rooms: list[V3Room],
    gaps: list[V3Gap],
) -> list[dict]:
    """Rain-exposed target pieces the proposer pairs with slanted segments.

    Each room contributes its floor polygon minus higher rooms + closed gaps
    above. Each closed gap contributes its own footprint as a single piece,
    because a sealed cross-floor gap (e.g. a gable-end closure) is itself a
    rain-exposed surface that can host a roof -- and without it the proposer
    never reaches into the XZ above that gap, so the merged stage never
    clusters there.
    """
    targets: list[dict] = []
    for room in rooms:
        full_poly = floor_polygon_to_shapely(room.floor_polygon)
        if (
            not isinstance(full_poly, Polygon)
            or full_poly.is_empty
            or not full_poly.is_valid
        ):
            continue
        exposure = _rain_exposure(room, rooms, gaps)
        pieces = _exposed_pieces(exposure)
        if not pieces:
            continue
        targets.append(
            {
                "kind": "room",
                "id_token": f"room-{room.index}",
                "room_id": room.identifier,
                "room_ref": room,
                "pieces": pieces,
                "slab_area_m2": float(full_poly.area),
                "slab_vertex_count": len(room.floor_polygon),
                "story": int(room.story),
                "floor_y": float(room.floor_polygon[0][1])
                if room.floor_polygon
                else 0.0,
            }
        )
    for gap in gaps:
        if gap.status != "closed":
            continue
        gap_poly = _footprint_polygon_3d_to_shapely(gap.footprint_xz)
        if gap_poly is None or gap_poly.is_empty:
            continue
        pieces = _exposed_pieces(gap_poly)
        if not pieces:
            continue
        gap_tail = (
            gap.id.rsplit("::", 1)[-1] if isinstance(gap.id, str) else str(gap.id)
        )
        targets.append(
            {
                "kind": "gap",
                "id_token": f"gap-{gap_tail}",
                "room_id": None,
                "room_ref": None,
                "gap_id": gap.id,
                "pieces": pieces,
                "slab_area_m2": float(gap_poly.area),
                "slab_vertex_count": len(gap.footprint_xz),
                "story": -1,
                "floor_y": float(gap.floor_y),
            }
        )
    return targets


def propose_slanted_roofs(
    building_uuid: str,
    rooms: list[V3Room],
    gaps: list[V3Gap],
    slabs: list[V3Slab],
    slanted_roofs: list[V3SlantedRoof],
) -> list[V3RoofProposal]:
    """Permissive (slanted-segment x rain-exposed piece) proposer.

    Target pieces are every room's rain-exposed floor region **plus every
    closed gap's footprint** -- sealed cross-floor gaps are rain-exposed too,
    and without them proposals never reach XZ areas covered by a gap-above
    (the merged stage then has no cluster footprint to intersect there, so
    extensions like Lodskovvej 2's lean-to lose all roof coverage).
    """
    segments = _collect_permissive_slanted_segments(rooms)
    if not segments:
        return []

    targets = _collect_target_pieces(rooms, gaps)
    if not targets:
        return []

    stories = sorted({r.story for r in rooms})
    slab_by_room = {s.room_id: s for s in slabs if s.room_id is not None}

    proposals: list[V3RoofProposal] = []
    for segment in segments:
        plane = _plane_from_segment(segment)
        for target in targets:
            slab_room_id = target["room_id"]
            slab = slab_by_room.get(slab_room_id) if slab_room_id else None
            heuristic_label = _heuristic_label_for_segment(
                segment, slab_room_id or "", slanted_roofs
            )
            for piece_index, piece_poly in enumerate(target["pieces"]):
                corners = _corners_on_plane(piece_poly, plane)
                if len(corners) < 3:
                    continue
                features = _compute_features(
                    segment=segment,
                    plane=plane,
                    target=target,
                    piece_poly=piece_poly,
                    piece_index=piece_index,
                    stories=stories,
                )
                inner = (
                    f"segment-{segment['seg_idx']}:{target['id_token']}:"
                    f"piece-{piece_index}"
                )
                proposals.append(
                    V3RoofProposal(
                        id=make_element_id(building_uuid, "v3-roof-proposal", inner),
                        segment_index=int(segment["seg_idx"]),
                        source_room_id=segment.get("room_id"),
                        source_wall_id=segment.get("wall_id"),
                        slab_room_id=slab_room_id,
                        slab_id=slab.id if slab else target.get("gap_id"),
                        plane=plane,
                        corners=corners,
                        features=features,
                        heuristic_label=heuristic_label,
                        trace=HypothesisTrace(
                            stage="slanted_roof_proposals",
                            rule="permissive-segment-x-rain-exposed-piece",
                            inputs={
                                "segment_index": int(segment["seg_idx"]),
                                "source_wall_id": segment.get("wall_id"),
                                "source_room_id": segment.get("room_id"),
                                "slab_room_id": slab_room_id,
                                "slab_kind": target["kind"],
                                "piece_index": int(piece_index),
                                "heuristic_label": heuristic_label,
                            },
                            decision_reason=(
                                "Every (slanted wall edge, rain-exposed piece) "
                                "pair is emitted as a candidate for human labeling. "
                                "Pieces come from rooms' rain-exposed floors AND "
                                "closed gap footprints -- a sealed gap is "
                                "rain-exposed, "
                                "so without it the merged stage has no cluster to "
                                "intersect above the gap and downstream roof coverage "
                                "vanishes. "
                                f"Pieces below {EXPOSED_PIECE_MIN_AREA_M2} m^2 area or "
                                f"{EXPOSED_PIECE_MIN_WIDTH_M} m effective width "
                                "(2*area/perimeter) are dropped. Permissive slant "
                                "threshold "
                                f"({PERMISSIVE_MIN_INCL_DEG} deg < incl < "
                                f"{PERMISSIVE_MAX_INCL_DEG} deg) to surface edge cases."
                            ),
                        ),
                    )
                )
    return proposals
