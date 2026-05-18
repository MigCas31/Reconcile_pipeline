from __future__ import annotations

from dataclasses import dataclass

from shapely import set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import Plane, PlaneKey, plane_key
from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.payload.schema import CeilingPiece, CeilingSource, Vec3
from reconcile_tiers.payload.schema import Plane as PayloadPlane

PRIORITY = {
    CeilingSource.FLAT_EMIT: 100,
    CeilingSource.ROOF_ARRANGEMENT: 80,
    CeilingSource.ROOF_ARRANGEMENT_ATTIC: 80,
    # Synthesised gable arrangement owns the (x,z) footprint where it exists.
    # The extended attic lid (build.py:_extend_flat_to_obliques) only fills
    # leftover area — extension wings, dormer flanks, gaps in the arrangement.
    CeilingSource.ATTIC_FLAT_LID: 75,
    CeilingSource.THERMAL_CAP: 70,
    CeilingSource.RAW_FALLBACK: 40,
}
MIN_PIECE_AREA_M2 = 0.05
OVERLAY_GRID_SIZE_M = 1e-8


@dataclass(frozen=True, slots=True)
class CeilingCandidate:
    corners: list[list[float]]
    plane: Plane
    source: CeilingSource
    locator_id: str
    arrangement_cell_id: str | None = None
    story: int | None = None


def _polygon_xz(corners: list[list[float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = make_valid(
            Polygon([(float(corner[0]), float(corner[2])) for corner in corners])
        )
        poly = set_precision(poly, OVERLAY_GRID_SIZE_M)
    except Exception:
        return None
    if poly.is_empty:
        return None
    if isinstance(poly, Polygon):
        return poly if poly.area >= MIN_PIECE_AREA_M2 else None
    parts = [
        part
        for part in getattr(poly, "geoms", [])
        if isinstance(part, Polygon) and part.area >= MIN_PIECE_AREA_M2
    ]
    if not parts:
        return None
    return set_precision(make_valid(unary_union(parts)), OVERLAY_GRID_SIZE_M)


def _polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    geom = make_valid(geom)
    if geom is None or geom.is_empty:
        return []
    geom = set_precision(geom, OVERLAY_GRID_SIZE_M)
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.is_valid and geom.area >= MIN_PIECE_AREA_M2 else []
    return [
        part
        for part in getattr(geom, "geoms", [])
        if isinstance(part, Polygon)
        and part.is_valid
        and part.area >= MIN_PIECE_AREA_M2
    ]


def _ring_to_vec3(coords, plane: Plane) -> list[Vec3] | None:
    ring = list(coords)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    out: list[Vec3] = []
    for x, z in ring:
        y = plane.y_at(float(x), float(z))
        if y is None:
            return None
        out.append(Vec3(x=float(x), y=float(y), z=float(z)))
    return out if len(out) >= 3 else None


def _to_payload_plane(plane: Plane) -> PayloadPlane:
    return PayloadPlane(a=plane.a, b=plane.b, c=plane.c, d=plane.d)


def _candidate_piece(
    candidate: CeilingCandidate, poly: Polygon, suffix: int
) -> CeilingPiece | None:
    corners = _ring_to_vec3(poly.exterior.coords, candidate.plane)
    if corners is None:
        return None
    if newell_normal([[corner.x, corner.y, corner.z] for corner in corners])[1] <= 0.0:
        corners = list(reversed(corners))
    holes: list[list[Vec3]] = []
    for interior in poly.interiors:
        hole = _ring_to_vec3(interior.coords, candidate.plane)
        if hole is not None:
            holes.append(hole)
    locator_id = (
        candidate.locator_id if suffix == 0 else f"{candidate.locator_id}:{suffix}"
    )
    return CeilingPiece(
        corners=corners,
        holes=holes,
        plane=_to_payload_plane(candidate.plane),
        source=candidate.source,
        arrangement_cell_id=candidate.arrangement_cell_id,
        locator_id=locator_id,
    )


def assemble_ceiling(candidates: list[CeilingCandidate]) -> list[CeilingPiece]:
    sorted_candidates = sorted(
        candidates, key=lambda candidate: -PRIORITY[candidate.source]
    )

    visible: list[CeilingPiece] = []
    occupied_by_story: dict[int | None, Polygon] = {}
    occupied_by_plane: dict[PlaneKey, Polygon] = {}
    for candidate in sorted_candidates:
        poly = _polygon_xz(candidate.corners)
        if poly is None:
            continue
        bucket = occupied_by_story.get(candidate.story)
        if bucket is not None and not bucket.is_empty:
            poly = set_precision(
                make_valid(poly.difference(bucket)), OVERLAY_GRID_SIZE_M
            )
        plane_bucket = occupied_by_plane.get(plane_key(candidate.plane))
        if plane_bucket is not None and not plane_bucket.is_empty:
            poly = set_precision(
                make_valid(poly.difference(plane_bucket)), OVERLAY_GRID_SIZE_M
            )

        emitted_parts = _polygon_parts(poly)
        for suffix, part in enumerate(emitted_parts):
            piece = _candidate_piece(candidate, part, suffix)
            if piece is not None:
                visible.append(piece)

        if emitted_parts:
            emitted_union = set_precision(
                make_valid(unary_union(emitted_parts)), OVERLAY_GRID_SIZE_M
            )
            occupied_by_story[candidate.story] = (
                emitted_union
                if bucket is None
                else set_precision(
                    make_valid(unary_union([bucket, emitted_union])),
                    OVERLAY_GRID_SIZE_M,
                )
            )
            plane_key_value = plane_key(candidate.plane)
            plane_bucket = occupied_by_plane.get(plane_key_value)
            occupied_by_plane[plane_key_value] = (
                emitted_union
                if plane_bucket is None
                else set_precision(
                    make_valid(unary_union([plane_bucket, emitted_union])),
                    OVERLAY_GRID_SIZE_M,
                )
            )

    return visible
