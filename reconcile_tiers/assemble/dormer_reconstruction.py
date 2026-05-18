from __future__ import annotations

from typing import Any

import numpy as np
from shapely import set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import fit_plane_any
from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.assemble.ceiling_painter import (
    OVERLAY_GRID_SIZE_M,
    _polygon_parts,
    _polygon_xz,
)
from reconcile_tiers.assemble.walls_to_rooms import (
    _orient_wall_outward,
    _project_to_plane,
    _room_center,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingPiece, DormerFaceKind, Vec3
from reconcile_tiers.roof.dormers import _opening_inside_wall, cutout_and_trim
from reconcile_tiers.roof.roof import DormerCandidate, ObliqueSurface, ThermalSurface


def _oblique_xz_polygon(surface: ObliqueSurface) -> Polygon | None:
    if len(surface.corners) < 3:
        return None
    poly = Polygon([(float(p[0]), float(p[2])) for p in surface.corners])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly


def _y_at(plane: Any, x: float, z: float) -> float | None:
    if abs(plane.b) < 1e-6:
        return None
    return (plane.d - plane.a * x - plane.c * z) / plane.b


def _ring_to_vec3(coords, plane: Any) -> list[Vec3] | None:
    ring = list(coords)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    out: list[Vec3] = []
    for x, z in ring:
        y = _y_at(plane, float(x), float(z))
        if y is None:
            return None
        out.append(Vec3(x=float(x), y=float(y), z=float(z)))
    return out if len(out) >= 3 else None


def _piece_polygon(piece: CeilingPiece) -> Polygon | None:
    if len(piece.corners) < 3:
        return None
    exterior = [(float(c.x), float(c.z)) for c in piece.corners]
    interiors = [
        [(float(h.x), float(h.z)) for h in hole]
        for hole in piece.holes
        if len(hole) >= 3
    ]
    try:
        poly = make_valid(Polygon(exterior, interiors))
        poly = set_precision(poly, OVERLAY_GRID_SIZE_M)
    except Exception:
        return None
    if poly.is_empty or not isinstance(poly, Polygon):
        return None
    return poly


def _piece_with_polygon(
    template: CeilingPiece, poly: Polygon, suffix: int
) -> CeilingPiece | None:
    corners = _ring_to_vec3(poly.exterior.coords, template.plane)
    if corners is None:
        return None
    if newell_normal([[c.x, c.y, c.z] for c in corners])[1] <= 0.0:
        corners = list(reversed(corners))
    holes: list[list[Vec3]] = []
    for interior in poly.interiors:
        hole = _ring_to_vec3(interior.coords, template.plane)
        if hole is not None:
            holes.append(hole)
    locator_id = (
        template.locator_id if suffix == 0 else f"{template.locator_id}:dormer:{suffix}"
    )
    return CeilingPiece(
        corners=corners,
        holes=holes,
        plane=template.plane,
        source=template.source,
        arrangement_cell_id=template.arrangement_cell_id,
        locator_id=locator_id,
    )


def _story_from_arrangement_cell(
    arrangement_cell_id: str | None, rooms_by_index: dict[int, Any]
) -> int | None:
    if not arrangement_cell_id:
        return None
    parts = arrangement_cell_id.split(":")
    if len(parts) < 5 or parts[2] != "room":
        return None
    try:
        room_idx = int(parts[3])
    except ValueError:
        return None
    room = rooms_by_index.get(room_idx)
    return None if room is None else int(room.story)


def _story_from_locator(locator_id: str, rooms_by_index: dict[int, Any]) -> int | None:
    for marker in ("::tier-ceiling-flat::", "::tier-ceiling-raw::"):
        if marker not in locator_id:
            continue
        tail = locator_id.split(marker, 1)[1]
        try:
            room_idx = int(tail.split(":", 1)[0])
        except ValueError:
            return None
        room = rooms_by_index.get(room_idx)
        return None if room is None else int(room.story)
    return None


def _story_from_mapping(
    locator_id: str, ceiling_stories: dict[str, int | None]
) -> int | None:
    matches = [
        (candidate_locator, story)
        for candidate_locator, story in ceiling_stories.items()
        if locator_id == candidate_locator
        or locator_id.startswith(f"{candidate_locator}:")
    ]
    if not matches:
        return None
    _locator, story = max(matches, key=lambda item: len(item[0]))
    return story


def _piece_story(
    piece: CeilingPiece,
    rooms_by_index: dict[int, Any],
    ceiling_stories: dict[str, int | None],
) -> int | None:
    mapped = _story_from_mapping(piece.locator_id, ceiling_stories)
    if mapped is not None:
        return mapped
    arranged = _story_from_arrangement_cell(piece.arrangement_cell_id, rooms_by_index)
    if arranged is not None:
        return arranged
    return _story_from_locator(piece.locator_id, rooms_by_index)


def _story_allows_cutout(piece_story: int | None, cutout_story: int | None) -> bool:
    if piece_story is None or cutout_story is None:
        return True
    if piece_story < 0:
        return True
    return piece_story == cutout_story


def _plane_allows_cutout_height(
    piece: CeilingPiece, cutout: Polygon, max_y: float
) -> bool:
    ys = [
        _y_at(piece.plane, float(x), float(z)) for x, z in list(cutout.exterior.coords)
    ]
    ys = [float(y) for y in ys if y is not None]
    if not ys:
        return True
    return max(ys) <= max_y + 0.25


def _clip_wall_above_roof_plane(
    corners: list[list[float]],
    plane: Any,
) -> list[list[float]]:
    if len(corners) < 3:
        return []

    def signed(point: list[float]) -> float:
        y = _y_at(plane, float(point[0]), float(point[2]))
        if y is None:
            return float("-inf")
        return float(point[1]) - y

    def intersect(
        p0: list[float], p1: list[float], s0: float, s1: float
    ) -> list[float]:
        denom = s0 - s1
        if abs(denom) <= 1e-12:
            return list(p0)
        t = max(0.0, min(1.0, s0 / denom))
        return [p0[idx] + t * (p1[idx] - p0[idx]) for idx in range(3)]

    out: list[list[float]] = []
    for idx, cur in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        s_cur = signed(cur)
        s_next = signed(nxt)
        cur_inside = s_cur >= -1e-6
        next_inside = s_next >= -1e-6
        if cur_inside and next_inside:
            out.append(list(nxt))
        elif cur_inside and not next_inside:
            out.append(intersect(cur, nxt, s_cur, s_next))
        elif not cur_inside and next_inside:
            out.append(intersect(cur, nxt, s_cur, s_next))
            out.append(list(nxt))

    deduped: list[list[float]] = []
    for point in out:
        if (
            not deduped
            or sum((point[i] - deduped[-1][i]) ** 2 for i in range(3)) > 1e-10
        ):
            deduped.append(point)
    if (
        len(deduped) >= 2
        and sum((deduped[0][i] - deduped[-1][i]) ** 2 for i in range(3)) <= 1e-10
    ):
        deduped.pop()
    return deduped


def _centroid(corners: list[list[float]]) -> list[float]:
    n = max(1, len(corners))
    return [sum(float(point[idx]) for point in corners) / n for idx in range(3)]


def _plane_frame(corners: list[list[float]]):
    plane = fit_plane_any(corners)
    if plane is None:
        return None
    normal = np.asarray(plane[:3], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal = normal / norm
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(normal[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(normal, helper)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        return None
    u = u / u_norm
    v = np.cross(normal, u)
    origin = np.asarray(corners[0], dtype=float)
    return origin, u, v


def _to_frame_polygon(corners: list[list[float]], frame) -> Polygon | None:
    origin, u, v = frame
    coords = []
    for corner in corners:
        point = np.asarray(corner, dtype=float) - origin
        coords.append((float(np.dot(point, u)), float(np.dot(point, v))))
    try:
        poly = make_valid(Polygon(coords))
    except Exception:
        return None
    return _largest_polygon(poly)


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 1e-6 else None
    polygons = [
        part
        for part in getattr(geom, "geoms", [])
        if isinstance(part, Polygon) and part.area > 1e-6
    ]
    if not polygons:
        return None
    return max(polygons, key=lambda part: part.area)


def _from_frame_ring(coords, frame) -> list[list[float]]:
    origin, u, v = frame
    ring = list(coords)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return [
        (origin + float(x) * u + float(y) * v).astype(float).tolist() for x, y in ring
    ]


def _repair_plane_local_polygon(corners: list[list[float]]) -> list[list[float]]:
    if len(corners) < 3:
        return []
    frame = _plane_frame(corners)
    if frame is None:
        return []
    poly = _to_frame_polygon(corners, frame)
    if poly is None:
        return []
    repaired = _from_frame_ring(poly.exterior.coords, frame)
    return repaired if len(repaired) >= 3 else []


def _orient_away_from_reference(
    corners: list[list[float]],
    reference: list[float],
) -> list[list[float]]:
    if len(corners) < 3:
        return corners
    normal = newell_normal(corners)
    center = _centroid(corners)
    outward = [center[idx] - reference[idx] for idx in range(3)]
    if sum(normal[idx] * outward[idx] for idx in range(3)) < 0.0:
        return list(reversed(corners))
    return corners


def _orient_header_up(corners: list[list[float]]) -> list[list[float]]:
    if len(corners) < 3:
        return corners
    if newell_normal(corners)[1] < 0.0:
        return list(reversed(corners))
    return corners


def _front_cutouts(room, front_corners: list[list[float]]) -> list[list[list[float]]]:
    cutouts: list[list[list[float]]] = []
    frame = _plane_frame(front_corners)
    front_poly = _to_frame_polygon(front_corners, frame) if frame is not None else None
    for opening in [*room.windows, *room.doors]:
        if _opening_inside_wall(front_corners, opening):
            opening_corners = [
                [float(value) for value in point] for point in opening.corners
            ]
            if frame is None or front_poly is None:
                cutouts.append(opening_corners)
                continue
            opening_poly = _to_frame_polygon(opening_corners, frame)
            if opening_poly is None:
                continue
            if front_poly.buffer(0.02).contains(opening_poly):
                cutouts.append(opening_corners)
                continue
            clipped = _largest_polygon(front_poly.intersection(opening_poly))
            if clipped is None:
                continue
            clipped_corners = _from_frame_ring(clipped.exterior.coords, frame)
            if len(clipped_corners) == 4:
                cutouts.append(clipped_corners)
    return cutouts


def _candidate_opening(room, opening_id: str | None):
    if opening_id is None:
        return None
    return next(
        (
            opening
            for opening in [*room.windows, *room.doors]
            if opening.id == opening_id
        ),
        None,
    )


def _is_valid_plane_local_polygon(corners: list[list[float]]) -> bool:
    plane = fit_plane_any(corners)
    if plane is None:
        return False
    poly = _project_to_plane(corners, plane[:3])
    return poly is not None and poly.is_valid and not poly.is_empty


def reconstruct_dormers(
    ceilings: list[CeilingPiece],
    candidates: list[DormerCandidate],
    obliques: list[ObliqueSurface],
    model: BuildingModel,
    ceiling_stories: dict[str, int | None] | None = None,
) -> tuple[list[CeilingPiece], list[ThermalSurface]]:
    """Subtract dormer cutouts from painted ceilings and emit dormer thermals.

    Runs after assemble_ceiling. For each candidate, reconstructs the
    cutout/cheek/header against the parent oblique's plane, subtracts the
    cutout XZ polygon from every painted ceiling piece on the matching plane,
    and returns the updated ceiling list together with the dormer thermal
    surfaces.
    """
    if not candidates:
        return list(ceilings), []

    rooms_by_index = {room.index: room for room in model.rooms}
    all_cutouts: list[tuple[int | None, Polygon, float]] = []
    thermal: list[ThermalSurface] = []
    ceiling_stories = ceiling_stories or {}

    for candidate in candidates:
        if not (0 <= candidate.roof_surface_index < len(obliques)):
            continue
        oblique = obliques[candidate.roof_surface_index]
        room = rooms_by_index.get(candidate.room_index)
        if room is None:
            continue
        wall = next(
            (
                w
                for w in [*room.walls_computed, *room.walls_merged]
                if w.id == candidate.front_wall_id and len(w.corners) >= 3
            ),
            None,
        )
        if wall is None:
            continue
        opening = _candidate_opening(room, candidate.front_opening_id)
        source_corners = opening.corners if opening is not None else wall.corners
        trim = cutout_and_trim(
            oblique.plane, source_corners, _oblique_xz_polygon(oblique)
        )
        if trim is None:
            continue
        cutout, cheeks, header = trim
        dormer_reference = _centroid([*cutout, *header])
        front = _orient_away_from_reference(
            _repair_plane_local_polygon(
                _clip_wall_above_roof_plane(
                    _orient_wall_outward(
                        [
                            [float(corner[0]), float(corner[1]), float(corner[2])]
                            for corner in source_corners
                        ],
                        _room_center(room),
                    ),
                    oblique.plane,
                )
            ),
            dormer_reference,
        )
        if len(front) >= 3 and _is_valid_plane_local_polygon(front):
            thermal.append(
                ThermalSurface(
                    corners=front,
                    kind=DormerFaceKind.DORMER_FRONT,
                    room_index=candidate.room_index,
                    source="dormer",
                    cutouts=_front_cutouts(room, front),
                )
            )
        cutout_poly = _polygon_xz(cutout)
        if cutout_poly is not None:
            all_cutouts.append(
                (int(room.story), cutout_poly, max(p[1] for p in header))
            )
        for cheek in cheeks:
            cheek = _orient_away_from_reference(cheek, dormer_reference)
            thermal.append(
                ThermalSurface(
                    corners=cheek,
                    kind=DormerFaceKind.DORMER_CHEEK,
                    room_index=candidate.room_index,
                    source="dormer",
                )
            )
        thermal.append(
            ThermalSurface(
                corners=_orient_header_up(header),
                kind=DormerFaceKind.DORMER_HEADER,
                room_index=candidate.room_index,
                source="dormer",
            )
        )

    if not all_cutouts:
        return list(ceilings), thermal

    # Keep the legacy cross-plane behavior within the same story: a dormer
    # cutout still punches through a gable shell on a different oblique plane,
    # but it must not remove ceilings from another storey with the same XZ.
    updated: list[CeilingPiece] = []
    for piece in ceilings:
        poly = _piece_polygon(piece)
        if poly is None:
            updated.append(piece)
            continue
        piece_story = _piece_story(piece, rooms_by_index, ceiling_stories)
        matching_cutouts = [
            cutout
            for cutout_story, cutout, cutout_max_y in all_cutouts
            if _story_allows_cutout(piece_story, cutout_story)
            and _plane_allows_cutout_height(piece, cutout, cutout_max_y)
        ]
        if not matching_cutouts:
            updated.append(piece)
            continue
        cutouts_union = unary_union(matching_cutouts)
        if not poly.intersects(cutouts_union) or poly.touches(cutouts_union):
            updated.append(piece)
            continue
        result = make_valid(poly.difference(cutouts_union))
        result = set_precision(result, OVERLAY_GRID_SIZE_M)
        parts = _polygon_parts(result)
        if not parts:
            continue
        for suffix, part in enumerate(parts):
            new_piece = _piece_with_polygon(piece, part, suffix)
            if new_piece is not None:
                updated.append(new_piece)

    return updated, thermal
