"""Envelope-assembly post-processing used by `reconcile_tiers.build`.

Converts per-roof gable closures / knee walls / dormer faces from the roof
model into payload schema objects, dedups vertical faces against claimed
walls, cleans ceiling polygons (rings, holes), and drops redundant split
pieces from a single candidate's `/0,/1,...` series. Re-exported from
`reconcile_tiers.build`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from reconcile_tiers.build_internals.constants import (
    _PIECE_DEDUP_INTERSECT_AREA_M2,
    _PIECE_DEDUP_INTERSECT_RATIO,
    _PIECE_REDUNDANT_INTERSECT_RATIO,
    CEILING_RING_MIN_AREA_M2,
    CEILING_RING_MIN_EDGE_M,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import (
    CeilingPiece,
    DormerFace,
    GableClosure,
    KneeWall,
    Quad,
    TierClassification,
    Vec3,
)
from reconcile_tiers.roof.gable_closures import build_gable_closures
from reconcile_tiers.roof.roof import RoofModel


def _gable_closures(model: BuildingModel, roof: RoofModel) -> list[GableClosure]:
    surfaces = build_gable_closures(model, roof)
    return [
        GableClosure(
            corners=[
                Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in surface.corners
            ],
            kind=surface.kind,
            locator_id=f"{model.uuid}::tier-gable-closure::{surface.kind.value}:{idx}",
        )
        for idx, surface in enumerate(surfaces)
        if len(surface.corners) >= 3
    ]


def _knee_walls(
    model: BuildingModel,
    roof: RoofModel,
    rooms: list,
) -> list[KneeWall]:
    candidates = [
        KneeWall(
            corners=[
                Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in surface.corners
            ],
            kind=surface.kind,
            locator_id=f"{model.uuid}::tier-knee-wall::{idx}",
        )
        for idx, surface in enumerate(roof.thermal)
        if len(surface.corners) >= 3
    ]
    return _dedup_vertical_faces(candidates, rooms)


def _dormer_faces(
    model: BuildingModel,
    rooms: list,
    dormer_thermal: list,
) -> list[DormerFace]:
    candidates = [
        DormerFace(
            corners=[
                Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in surface.corners
            ],
            kind=surface.kind,
            locator_id=f"{model.uuid}::tier-dormer-face::{surface.kind.value}:{idx}",
            cutouts=[
                Quad(
                    corners=[
                        Vec3(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                        for p in cutout
                    ]
                )
                for cutout in surface.cutouts
                if len(cutout) == 4
            ],
        )
        for idx, surface in enumerate(dormer_thermal)
        if len(surface.corners) >= 3
    ]
    return _dedup_vertical_faces(candidates, rooms)


def _dedup_vertical_faces(candidates: list, rooms: list) -> list:
    from shapely.geometry import Polygon

    from reconcile_tiers._core.plane import PlaneKey, fit_plane_any, plane_key
    from reconcile_tiers.assemble.walls_to_rooms import _project_to_plane

    claimed: dict[PlaneKey, list[Polygon]] = {}
    for room in rooms:
        for wall in room.walls:
            corners3 = [[c.x, c.y, c.z] for c in wall.corners]
            plane = fit_plane_any(corners3)
            if plane is None:
                continue
            poly2d = _project_to_plane(corners3, plane[:3])
            if poly2d is None:
                continue
            claimed.setdefault(plane_key(plane), []).append(poly2d)

    pending: list[tuple[int, PlaneKey, Polygon, object]] = []
    for idx, face in enumerate(candidates):
        corners3 = [[c.x, c.y, c.z] for c in face.corners]
        plane = fit_plane_any(corners3)
        if plane is None:
            continue
        poly2d = _project_to_plane(corners3, plane[:3])
        if poly2d is None:
            continue
        pending.append((idx, plane_key(plane), poly2d, face))

    drop: set[int] = set()
    for idx, key, poly2d, _knee in pending:
        prev_list = claimed.get(key, [])
        dropped = False
        for prev_poly in prev_list:
            try:
                inter = poly2d.intersection(prev_poly).area
            except Exception:
                inter = 0.0
            # Drop if EITHER the overlap is a meaningful fraction of the
            # candidate OR it crosses an absolute area threshold below the
            # validator's overlap budget. Without the absolute floor, large
            # synthetic dormer cheeks/headers slip past the % gate when their
            # own area is huge but the overlap is still well above the
            # validator's tolerance.
            if (
                inter > _PIECE_DEDUP_INTERSECT_RATIO * poly2d.area
                or inter > _PIECE_DEDUP_INTERSECT_AREA_M2
            ):
                drop.add(idx)
                dropped = True
                break
        if dropped:
            continue
        claimed.setdefault(key, []).append(poly2d)

    return [face for idx, face in enumerate(candidates) if idx not in drop]


def _ceiling_xz_polygon(piece: CeilingPiece):
    if len(piece.corners) < 3:
        return None
    from shapely.geometry import Polygon

    try:
        poly = Polygon([(float(c.x), float(c.z)) for c in piece.corners])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _clean_xz_ring(
    coords,
    *,
    min_edge_m: float = CEILING_RING_MIN_EDGE_M,
) -> list[tuple[float, float]]:
    ring = [(float(x), float(z)) for x, z in coords]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]

    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float]] = []
        min_edge_sq = min_edge_m * min_edge_m
        for point in ring:
            if not out:
                out.append(point)
                continue
            if (point[0] - out[-1][0]) ** 2 + (
                point[1] - out[-1][1]
            ) ** 2 < min_edge_sq:
                changed = True
                continue
            out.append(point)
        if (
            len(out) >= 2
            and (out[0][0] - out[-1][0]) ** 2 + (out[0][1] - out[-1][1]) ** 2
            < min_edge_sq
        ):
            out.pop()
            changed = True
        ring = out
    return ring


def _vec3_on_payload_plane(x: float, z: float, plane) -> Vec3 | None:
    b = float(plane.b)
    if abs(b) < 1e-9:
        return None
    y = (float(plane.d) - float(plane.a) * x - float(plane.c) * z) / b
    return Vec3(x=float(x), y=float(y), z=float(z))


def _clean_ceiling_ring_on_plane(coords, plane) -> list[Vec3] | None:
    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon

    ring = _clean_xz_ring(coords)
    if len(ring) < 3:
        return None
    poly = make_valid_polygon(Polygon(ring))
    if poly is None or poly.is_empty or float(poly.area) < CEILING_RING_MIN_AREA_M2:
        return None
    ring = _clean_xz_ring(poly.exterior.coords)
    if len(ring) < 3:
        return None
    if any(
        (ring[idx][0] - ring[(idx + 1) % len(ring)][0]) ** 2
        + (ring[idx][1] - ring[(idx + 1) % len(ring)][1]) ** 2
        < CEILING_RING_MIN_EDGE_M * CEILING_RING_MIN_EDGE_M
        for idx in range(len(ring))
    ):
        return None
    corners = [_vec3_on_payload_plane(x, z, plane) for x, z in ring]
    if any(corner is None for corner in corners):
        return None
    return [corner for corner in corners if corner is not None]


def _clean_ceiling_piece_geometry(piece: CeilingPiece) -> CeilingPiece | None:
    corners = _clean_ceiling_ring_on_plane(
        [(corner.x, corner.z) for corner in piece.corners],
        piece.plane,
    )
    if corners is None:
        return None
    from reconcile_tiers._core.newell import newell_normal

    if newell_normal([[corner.x, corner.y, corner.z] for corner in corners])[1] <= 0.0:
        corners = list(reversed(corners))

    holes: list[list[Vec3]] = []
    for hole in piece.holes:
        clean_hole = _clean_ceiling_ring_on_plane(
            [(corner.x, corner.z) for corner in hole],
            piece.plane,
        )
        if clean_hole is not None:
            holes.append(clean_hole)

    return replace(piece, corners=corners, holes=holes)


def _clean_ceiling_piece_geometries(ceilings: list[CeilingPiece]) -> list[CeilingPiece]:
    cleaned: list[CeilingPiece] = []
    for piece in ceilings:
        clean_piece = _clean_ceiling_piece_geometry(piece)
        if clean_piece is not None:
            cleaned.append(clean_piece)
    return cleaned


def _ceiling_base_locator(locator_id: str) -> str:
    return locator_id.split("/", 1)[0]


def _drop_redundant_ceiling_split_overlaps(
    ceilings: list[CeilingPiece],
) -> list[CeilingPiece]:
    """Drop duplicate split pieces emitted from the same ceiling candidate.

    Some late synthesis paths can split a flat candidate into `/0`, `/1`, ...
    pieces where a tiny part is fully contained in its sibling. Different
    locators still go through normal validation; this only cleans up
    self-overlap created by one candidate's own split.
    """
    from reconcile_tiers._core.plane import plane_key

    by_key: dict[tuple[str, object, object], list[tuple[int, object]]] = {}
    drop: set[int] = set()
    for idx, piece in enumerate(ceilings):
        poly = _ceiling_xz_polygon(piece)
        if poly is None:
            continue
        key = (
            _ceiling_base_locator(piece.locator_id),
            piece.source,
            plane_key(piece.plane),
        )
        for prev_idx, prev_poly in by_key.get(key, []):
            try:
                inter = poly.intersection(prev_poly).area
            except Exception:
                inter = 0.0
            smaller = min(float(poly.area), float(prev_poly.area))
            if smaller > 0.0 and inter / smaller >= _PIECE_REDUNDANT_INTERSECT_RATIO:
                if poly.area <= prev_poly.area:
                    drop.add(idx)
                else:
                    drop.add(prev_idx)
                break
        by_key.setdefault(key, []).append((idx, poly))
    if not drop:
        return ceilings
    return [piece for idx, piece in enumerate(ceilings) if idx not in drop]


def _building_dict(model: BuildingModel) -> dict[str, Any]:
    return {
        "uuid": model.uuid,
        "stories_found": model.stories_found,
        "split_level": model.split_level,
        "rooms": [{"story": room.story} for room in model.rooms],
    }


def _roof_dict(roof: RoofModel) -> dict[str, Any]:
    return {
        "roof_surfaces": {
            "oblique": [
                {
                    "corners": surface.corners,
                    "story": surface.dominant_story,
                    "dominant_story": surface.dominant_story,
                    "cluster": {
                        "avgAzimuth": surface.cluster.avg_azimuth,
                        "avgIncl": surface.cluster.avg_incl,
                    },
                }
                for surface in roof.oblique
            ],
            "flat": [
                {
                    "corners": surface.corners,
                    "story": surface.story,
                    "dominant_story": surface.dominant_story,
                }
                for surface in roof.flat
            ],
        }
    }


def _classification(model: BuildingModel, roof: RoofModel) -> TierClassification:
    from reconcile_tiers.classify.roof_type import classify_oblique_roof
    from reconcile_tiers.classify.tiers import TIER_LABELS, classify_building

    result = classify_building(_building_dict(model), _roof_dict(roof))
    signals = dict(result["signals"])
    tier = int(result["tier"])
    tier_label = str(result["tier_label"])
    if any(type(params).__name__ == "GableParams" for params in roof.primitive_records):
        signals["has_gable"] = True
        tier = 7 if int(signals["n_flat"]) >= 1 else 6
        tier_label = TIER_LABELS[tier]
    return TierClassification(
        tier=tier,
        tier_label=tier_label,
        roof_type=classify_oblique_roof(roof.oblique),
        n_stories=int(signals["n_stories"]),
        n_rooms=int(signals["n_rooms"]),
        n_oblique=int(signals["n_oblique"]),
        n_flat=int(signals["n_flat"]),
        has_half_height=bool(signals["has_half_height"]),
        has_gable=bool(signals["has_gable"]),
    )
