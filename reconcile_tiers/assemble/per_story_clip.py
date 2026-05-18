"""Plan H Phase 2: post-emit per-story clip for ceiling pieces that overshoot
their dominant story's footprint.

The Plan C audit fix made `_building_envelope` use the all-rooms buffer-union,
which catches overshoot beyond the BUILDING. But it's too permissive for
catching pieces that extend past a SPECIFIC STORY (e.g., an oblique that sits
over a kicked-out wing on the wrong story). The diagnostic at
`audit/geometry_overshoot_diagnostic.py` found 57 such cases corpus-wide,
concentrated in rated-1-2 buildings.

This module trims those overshoots in place — gated behind
``TIER_PER_STORY_CLIP=1`` so we can A/B in the viewer before flipping default.

Source policy: only clips ``COMPUTED_OBLIQUE``, ``MERGED_COPLANAR``, and
``RAW_SCAN`` pieces. Flat ceilings are tightly clipped to room geometry
already by `_clip_flat_by_room_obliques`; aggressive over-clip there has a
higher blast radius. The diagnostic showed 8 flat overshoots in R1-2 — small
enough to address separately if needed.
"""

from __future__ import annotations

import logging
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import FitFailure
from reconcile_tiers._core.plane import Plane as CorePlane
from reconcile_tiers.payload.schema import (
    CeilingPiece,
    CeilingSource,
    Vec3,
)
from reconcile_tiers.payload.schema import (
    Plane as SchemaPlane,
)

_log = logging.getLogger(__name__)

CLIPPABLE_SOURCES: frozenset[CeilingSource] = frozenset(
    {
        CeilingSource.COMPUTED_OBLIQUE,
        CeilingSource.MERGED_COPLANAR,
        CeilingSource.RAW_SCAN,
    }
)
PER_STORY_BUFFER_M = 0.4
"""Soft buffer matching the diagnostic — pieces inside this tolerance band
are not trimmed (legitimate eave overhangs)."""

MIN_RETAINED_AREA_M2 = 0.05


def clip_pieces_to_per_story(
    pieces: list[CeilingPiece],
    rooms_payload: list[dict[str, Any]],
    *,
    building_uuid: str,
) -> list[CeilingPiece]:
    """Return pieces with XZ extents trimmed to their dominant story's
    footprint. Pieces in non-clippable sources or with no story match pass
    through unchanged.
    """
    per_story = _per_story_envelopes(rooms_payload)
    bands = _story_y_bands(rooms_payload)
    if not per_story or not bands:
        return list(pieces)

    out: list[CeilingPiece] = []
    for piece in pieces:
        if piece.source not in CLIPPABLE_SOURCES:
            out.append(piece)
            continue
        clipped = _clip_one(piece, per_story, bands, building_uuid)
        out.extend(clipped)
    return out


def _clip_one(
    piece: CeilingPiece,
    per_story_env: dict[int, Polygon],
    bands: dict[int, tuple[float, float]],
    building_uuid: str,
) -> list[CeilingPiece]:
    poly = _to_xz_polygon(piece.corners)
    if poly is None:
        return [piece]

    story = _piece_dominant_story(piece, bands)
    if story is None or story not in per_story_env:
        return [piece]

    envelope = per_story_env[story]
    intersection = poly.intersection(envelope)
    if intersection.is_empty:
        # Whole piece outside its own story's envelope. Don't drop silently;
        # leaving the original is safer than a silent erase.
        return [piece]
    if intersection.area > poly.area - 1e-6:
        return [piece]  # nothing to trim

    components = _components(intersection)
    if not components:
        return [piece]
    components.sort(key=lambda p: p.area, reverse=True)

    # Re-fit the plane on whatever original corner cloud falls inside the
    # intersection — preserves the slope while shrinking the footprint.
    original_corners = [[c.x, c.y, c.z] for c in piece.corners]
    fitted = CorePlane.fit(original_corners)
    if isinstance(fitted, FitFailure):
        return [piece]

    out: list[CeilingPiece] = []
    for idx, component in enumerate(components):
        if component.area < MIN_RETAINED_AREA_M2:
            continue
        ring = _ring_to_vec3(list(component.exterior.coords), fitted)
        if ring is None:
            continue
        locator = piece.locator_id if idx == 0 else f"{piece.locator_id}.clip{idx}"
        out.append(
            CeilingPiece(
                corners=ring,
                holes=piece.holes,
                plane=SchemaPlane(a=fitted.a, b=fitted.b, c=fitted.c, d=fitted.d),
                source=piece.source,
                arrangement_cell_id=piece.arrangement_cell_id,
                locator_id=locator,
                support_quality=piece.support_quality,
                role=piece.role,
                adjacency=piece.adjacency,
                merged_from=list(piece.merged_from),
            )
        )
    if not out:
        return [piece]
    return out


def _per_story_envelopes(rooms: list[dict[str, Any]]) -> dict[int, Polygon]:
    by_story: dict[int, list[Polygon]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for piece in room.get("floor") or []:
            corners = piece.get("corners") or []
            poly = _safe_xz_polygon(corners)
            if poly is None:
                continue
            by_story.setdefault(int(story), []).append(
                poly.buffer(PER_STORY_BUFFER_M, join_style="mitre")
            )
    out: dict[int, Polygon] = {}
    for story, polys in by_story.items():
        merged = unary_union(polys)
        if merged.is_empty:
            continue
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        out[story] = merged
    return out


def _story_y_bands(rooms: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    bands: dict[int, list[tuple[float, float]]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for wall in room.get("walls") or []:
            corners = wall.get("corners") or []
            ys = [c.get("y") for c in corners if c.get("y") is not None]
            if not ys:
                continue
            bands.setdefault(int(story), []).append((min(ys), max(ys)))
    out: dict[int, tuple[float, float]] = {}
    for story, ranges in bands.items():
        if not ranges:
            continue
        out[story] = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    return out


def _piece_dominant_story(
    piece: CeilingPiece, bands: dict[int, tuple[float, float]]
) -> int | None:
    ys = [c.y for c in piece.corners]
    if not ys:
        return None
    mid = 0.5 * (min(ys) + max(ys))
    for story in sorted(bands.keys()):
        lo, hi = bands[story]
        if lo - 0.5 <= mid <= hi + 0.5:
            return story
    return None


def _to_xz_polygon(corners: list[Vec3]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = Polygon([(c.x, c.z) for c in corners])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    if isinstance(poly, MultiPolygon):
        candidates = [p for p in poly.geoms if isinstance(p, Polygon) and p.area > 0]
        if not candidates:
            return None
        poly = max(candidates, key=lambda p: p.area)
    return poly


def _safe_xz_polygon(corners) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = Polygon([(c["x"], c["z"]) for c in corners])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _components(geom) -> list[Polygon]:
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
