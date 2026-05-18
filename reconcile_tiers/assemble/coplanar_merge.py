"""Coplanar ceiling merge pass (Plan A).

Merges ceiling pieces whose planes match within tolerance into a single piece.
The motivation is upstream of the audit: today one physical roof slope often
gets split into N pieces, which inflates `roof_fragmented` flags, increases
the chance of any one piece landing `out_of_envelope` or `ceiling_below_floor`,
and degrades the visual rendering. Merging *before* the payload is emitted
collapses those failure modes at the source.

Source policy: only merge within ``{COMPUTED_OBLIQUE, RAW_SCAN}``. Flat
ceilings (``FLAT_CEILING``, ``FLAT_GAP_CLOSURE``) are tightly clipped to room
geometry and represent thermal-boundary structure that must not be silently
dilated. Already-merged pieces (``MERGED_COPLANAR``) are not re-merged.

Disjoint coplanar regions stay separate. ``unary_union`` returns a
MultiPolygon when pieces don't share footprint area; we keep the largest
component and emit smaller components as their own merged pieces. We never
fuse two pieces that don't actually overlap in plan view.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import (
    FitFailure,
    plane_key,
    planes_equivalent,
)
from reconcile_tiers._core.plane import (
    Plane as CorePlane,
)
from reconcile_tiers.payload.schema import (
    AdjacencyKind,
    CeilingPiece,
    CeilingRole,
    CeilingSource,
    Room,
    Vec3,
)
from reconcile_tiers.payload.schema import (
    Plane as SchemaPlane,
)

_log = logging.getLogger(__name__)

MERGEABLE_SOURCES: frozenset[CeilingSource] = frozenset(
    {CeilingSource.COMPUTED_OBLIQUE, CeilingSource.RAW_SCAN}
)
MIN_MERGED_AREA_M2 = 0.05
"""Don't emit merged components smaller than this area in m^2. Prevents tiny
slivers from a slightly mis-aligned union from cluttering the payload."""


def merge_coplanar_ceilings(
    pieces: list[CeilingPiece], building_uuid: str
) -> list[CeilingPiece]:
    """Return a new list with coplanar ``COMPUTED_OBLIQUE`` / ``RAW_SCAN``
    pieces merged. Other sources pass through unchanged. Order within
    each source class is preserved.
    """
    if not pieces:
        return []

    mergeable: list[tuple[int, CeilingPiece]] = []
    pass_through: list[tuple[int, CeilingPiece]] = []
    for idx, piece in enumerate(pieces):
        if piece.source in MERGEABLE_SOURCES and not _is_room_scope(piece.locator_id):
            mergeable.append((idx, piece))
        else:
            pass_through.append((idx, piece))

    if len(mergeable) < 2:
        return list(pieces)

    groups = _group_by_plane(mergeable)
    if not any(len(group) > 1 for group in groups):
        return list(pieces)

    merged: list[tuple[int, CeilingPiece]] = []
    group_idx = 0
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        new_pieces = _merge_group(
            [item[1] for item in group],
            building_uuid=building_uuid,
            group_idx=group_idx,
        )
        if not new_pieces:
            merged.extend(group)
            continue
        # Anchor the merged group at the first member's index so the output
        # ordering tracks the original payload roughly.
        anchor = min(item[0] for item in group)
        for offset, new_piece in enumerate(new_pieces):
            merged.append((anchor + offset / 1000.0, new_piece))
        group_idx += 1

    combined = pass_through + merged
    combined.sort(key=lambda item: item[0])
    return [item[1] for item in combined]


def _group_by_plane(
    indexed: list[tuple[int, CeilingPiece]],
) -> list[list[tuple[int, CeilingPiece]]]:
    """Group pieces by plane equivalence. Order: each group sorted by index;
    groups sorted by smallest member index."""
    groups: list[list[tuple[int, CeilingPiece]]] = []
    for item in indexed:
        _idx, piece = item
        for group in groups:
            if planes_equivalent(piece.plane, group[0][1].plane):
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def _merge_group(
    pieces: list[CeilingPiece],
    *,
    building_uuid: str,
    group_idx: int,
) -> list[CeilingPiece]:
    """Union the XZ footprints of ``pieces``, refit the plane on the combined
    corner cloud, and emit one piece per connected component (largest first).
    """
    polygons = [_to_xz_polygon(piece.corners) for piece in pieces]
    valid_polys = [p for p in polygons if p is not None]
    if len(valid_polys) < 2:
        return list(pieces)

    try:
        union = unary_union(valid_polys)
    except Exception as exc:
        _log.warning(
            "coplanar_merge: union failed for %s group %d: %s",
            building_uuid,
            group_idx,
            exc,
        )
        return list(pieces)

    components = _polygon_components(union)
    if not components:
        return list(pieces)

    all_corners: list[list[float]] = []
    for piece in pieces:
        for corner in piece.corners:
            all_corners.append([corner.x, corner.y, corner.z])
    fitted = CorePlane.fit(all_corners)
    if isinstance(fitted, FitFailure):
        _log.info(
            "coplanar_merge: plane refit failed for %s group %d (%s); keeping "
            "originals",
            building_uuid,
            group_idx,
            fitted,
        )
        return list(pieces)

    schema_plane = SchemaPlane(a=fitted.a, b=fitted.b, c=fitted.c, d=fitted.d)
    role = _common_or_default([piece.role for piece in pieces], CeilingRole.CEILING)
    adjacency = _common_or_default(
        [piece.adjacency for piece in pieces], AdjacencyKind.UNKNOWN
    )
    total_area = sum(p.area for p in valid_polys) or 1.0
    weighted_quality = sum(
        piece.support_quality * (poly.area / total_area)
        for piece, poly in zip(pieces, valid_polys, strict=False)
    )
    locator_sources = [piece.locator_id for piece in pieces if piece.locator_id]
    components.sort(key=lambda poly: poly.area, reverse=True)

    out: list[CeilingPiece] = []
    for component_idx, component in enumerate(components):
        if component.area < MIN_MERGED_AREA_M2:
            continue
        ring = _ring_to_vec3(list(component.exterior.coords), fitted)
        if ring is None:
            continue
        locator = (
            f"{building_uuid}::tier-ceiling-merged-coplanar::{group_idx}"
            if component_idx == 0
            else (
                f"{building_uuid}::tier-ceiling-merged-coplanar::"
                f"{group_idx}.{component_idx}"
            )
        )
        out.append(
            CeilingPiece(
                corners=ring,
                holes=[],
                plane=schema_plane,
                source=CeilingSource.MERGED_COPLANAR,
                arrangement_cell_id=None,
                locator_id=locator,
                support_quality=float(weighted_quality),
                role=role,
                adjacency=adjacency,
                merged_from=list(locator_sources),
            )
        )

    if not out:
        return list(pieces)
    return out


def _to_xz_polygon(corners: list[Vec3]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = Polygon([(c.x, c.z) for c in corners])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        candidates = [p for p in poly.geoms if isinstance(p, Polygon) and p.area > 0]
        if not candidates:
            return None
        poly = max(candidates, key=lambda p: p.area)
    if poly.area <= 0:
        return None
    return poly


def _polygon_components(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if isinstance(p, Polygon) and not p.is_empty]
    return [
        part
        for part in getattr(geom, "geoms", [])
        if isinstance(part, Polygon) and not part.is_empty
    ]


def _ring_to_vec3(
    coords: list[tuple[float, float]], plane: CorePlane
) -> list[Vec3] | None:
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


def _common_or_default(values: list, default):
    distinct = {v for v in values if v is not None}
    if len(distinct) == 1:
        return next(iter(distinct))
    return default


def _is_room_scope(locator_id: str | None) -> bool:
    """Per-room emissions (locators like ``::tier-ceiling-oblique-room::...``)
    are rendering-scope copies of building-level surfaces. The audit
    intentionally skips them when counting fragmentation, so the merge must
    too -- collapsing them would inflate `roof_fragmented` counts.
    """
    if not locator_id:
        return False
    parts = locator_id.split("::")
    if len(parts) < 2:
        return False
    return parts[1].endswith("-room")


# Tolerance for accepting an overlap as legitimate (matches the validator's
# `CEILING_OVERLAP_TOLERANCE_M2` invariant in `payload/validate.py`).
COPLANAR_OVERLAP_TOLERANCE_M2 = 1e-2

# Douglas-Peucker tolerance for the clipped polygon. Must stay strictly
# below `CEILING_RING_MIN_EDGE_M` (build.py = 0.02) so wall-thickness cuts
# (~5-10 mm offset from the original edge) survive -- at 2 cm tolerance
# Douglas-Peucker would collapse them straight back to the original edge.
_CLIP_RING_SIMPLIFY_TOL_M = 0.003


def clip_coplanar_overlaps(
    pieces: list[CeilingPiece],
    rooms: list[Room],
) -> list[CeilingPiece]:
    """Resolve overlapping coplanar ceiling pieces by subtracting the overlap
    from the smaller piece.

    Adjacent rooms whose floor polygons share a wall-thickness band emit flat
    ceilings on the same plane that overlap by 1-3 cm^2 along that band.
    Room-scope obliques can do the same. ``merge_coplanar_ceilings`` skips
    these sources by design (they're room-scoped, not building-level
    duplicates), so the validator's "no XZ overlap on same plane" invariant
    fires on what's really a synthesis-time leak.

    For each pair of pieces sharing a ``plane_key`` whose XZ polygons
    overlap by more than ``COPLANAR_OVERLAP_TOLERANCE_M2``, this pass
    subtracts the overlap from the smaller polygon and re-emits its
    corners from the clipped exterior. The larger piece is untouched.
    Pieces clipped below ``MIN_MERGED_AREA_M2`` are dropped entirely.

    Pairs the validator already exempts pass through:
      - either piece is ``ROOF_ARRANGEMENT_ATTIC``
      - the two pieces' room floors are nested (mezzanines etc.)
    """
    from reconcile_tiers.payload.validate import (
        _ceiling_piece_xz_polygon,
        _ceiling_room_index,
        _floors_are_nested,
        _room_floor_polygons,
    )

    if len(pieces) < 2:
        return list(pieces)

    room_floors = _room_floor_polygons(rooms)

    # Sort by area descending so each piece is clipped only against larger
    # pieces it has already been compared with (the larger piece "wins").
    polys: list[Polygon | None] = []
    areas: list[float] = []
    for piece in pieces:
        try:
            poly = _ceiling_piece_xz_polygon(piece)
        except Exception:
            poly = None
        if poly is None or not poly.is_valid or poly.is_empty:
            polys.append(None)
            areas.append(0.0)
        else:
            polys.append(poly)
            areas.append(float(poly.area))
    order = sorted(range(len(pieces)), key=lambda i: -areas[i])

    by_key: dict[object, list[int]] = {}
    drops: set[int] = set()
    out_pieces: list[CeilingPiece] = list(pieces)

    for idx in order:
        piece = out_pieces[idx]
        poly = polys[idx]
        if poly is None:
            continue
        key = plane_key(piece.plane)
        room_idx = _ceiling_room_index(piece)
        for prev_idx in by_key.get(key, []):
            if prev_idx in drops:
                continue
            prev_piece = out_pieces[prev_idx]
            if (
                piece.source == CeilingSource.ROOF_ARRANGEMENT_ATTIC
                or prev_piece.source == CeilingSource.ROOF_ARRANGEMENT_ATTIC
            ):
                continue
            prev_poly = polys[prev_idx]
            if prev_poly is None:
                continue
            try:
                inter_area = poly.intersection(prev_poly).area
            except Exception:
                continue
            if inter_area <= COPLANAR_OVERLAP_TOLERANCE_M2:
                continue
            prev_room_idx = _ceiling_room_index(prev_piece)
            if _floors_are_nested(room_floors, room_idx, prev_room_idx):
                continue
            try:
                diff = poly.difference(prev_poly)
            except Exception:
                continue
            largest = _largest_component(diff)
            if largest is None or largest.area < MIN_MERGED_AREA_M2:
                drops.add(idx)
                poly = None
                break
            # Light Douglas-Peucker simplification; tolerance must stay well
            # below the cut-strip width (~5-15 mm for wall-thickness leaks)
            # or the cut collapses straight back to the original edge.
            try:
                largest = largest.simplify(
                    _CLIP_RING_SIMPLIFY_TOL_M, preserve_topology=True
                )
            except Exception:
                pass
            if (
                not isinstance(largest, Polygon)
                or largest.is_empty
                or largest.area < MIN_MERGED_AREA_M2
            ):
                drops.add(idx)
                poly = None
                break
            new_corners = _ring_to_vec3(
                list(largest.exterior.coords),
                CorePlane(
                    a=float(piece.plane.a),
                    b=float(piece.plane.b),
                    c=float(piece.plane.c),
                    d=float(piece.plane.d),
                ),
            )
            if new_corners is None:
                drops.add(idx)
                poly = None
                break
            new_corners = _ensure_newell_up(new_corners)
            out_pieces[idx] = replace(piece, corners=new_corners, holes=[])
            polys[idx] = largest
            poly = largest
        if poly is not None:
            by_key.setdefault(key, []).append(idx)

    if not drops:
        return out_pieces
    return [p for i, p in enumerate(out_pieces) if i not in drops]


def _largest_component(geom) -> Polygon | None:
    components = _polygon_components(geom)
    if not components:
        return None
    return max(components, key=lambda p: p.area)


def _ensure_newell_up(corners: list[Vec3]) -> list[Vec3]:
    from reconcile_tiers._core.newell import newell_normal

    coords = [[c.x, c.y, c.z] for c in corners]
    if newell_normal(coords)[1] <= 0.0:
        return list(reversed(corners))
    return corners
