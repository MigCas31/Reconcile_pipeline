"""Stair-cutout post-processing for the tagged tier payload.

`_apply_stair_floor_holes` matches each reconstructed stair's
`slab_opening_polygon` to the upper-floor lid that contains its centroid
and appends the (clipped) opening to that lid's `holes`. Gated behind the
`TIER_STAIRS_ENABLED=1` env var by the caller in `build_tier_payload`.
"""

from __future__ import annotations

from dataclasses import replace

from shapely.geometry import Point as _ShPoint
from shapely.geometry import Polygon as _ShPolygon

from reconcile_tiers.payload.schema import HorizontalLid, Room, TierPayload, Vec3


def _apply_stair_floor_holes(payload: TierPayload, stairs: list) -> TierPayload:
    """Cut stair openings as holes in the upper floor lid that contains them.

    Each stair's `slab_opening_polygon` is matched to the room+lid whose
    XZ footprint contains the opening centroid, clipped to lie strictly
    inside the lid's outer ring (otherwise `validate_payload` rejects the
    hole), and appended to that lid's `holes`.
    """
    rooms_by_idx_with_hole: dict[int, list[list[Vec3]]] = {}
    for stair in stairs:
        if not stair.slab_opening_polygon:
            continue
        poly = stair.slab_opening_polygon
        cx = sum(v.x for v in poly) / len(poly)
        cz = sum(v.z for v in poly) / len(poly)
        centroid = _ShPoint(cx, cz)
        stair_poly_xz = _ShPolygon([(v.x, v.z) for v in poly])
        if not stair_poly_xz.is_valid:
            stair_poly_xz = stair_poly_xz.buffer(0)
        for ri, room in enumerate(payload.rooms):
            if room.story != stair.to_story:
                continue
            matched = False
            for lid in room.floor:
                if len(lid.corners) < 3:
                    continue
                poly_xz = _ShPolygon([(c.x, c.z) for c in lid.corners])
                if not poly_xz.is_valid:
                    poly_xz = poly_xz.buffer(0)
                if poly_xz.contains(centroid):
                    clipped = stair_poly_xz.intersection(poly_xz.buffer(-1e-3))
                    if clipped.is_empty:
                        matched = True
                        break
                    if clipped.geom_type == "MultiPolygon":
                        clipped = max(clipped.geoms, key=lambda g: g.area)
                    if clipped.area < 0.05:
                        matched = True
                        break
                    ring_y = sum(v.y for v in poly) / len(poly)
                    ring = list(clipped.exterior.coords)
                    if len(ring) > 1 and ring[0] == ring[-1]:
                        ring = ring[:-1]
                    clipped_vec = [
                        Vec3(x=float(x), y=float(ring_y), z=float(z)) for x, z in ring
                    ]
                    rooms_by_idx_with_hole.setdefault(ri, []).append(clipped_vec)
                    matched = True
                    break
            if matched:
                break

    if not rooms_by_idx_with_hole:
        return payload

    new_rooms = list(payload.rooms)
    for ri, holes in rooms_by_idx_with_hole.items():
        room = new_rooms[ri]
        if not room.floor:
            continue
        first = room.floor[0]
        merged_holes = list(first.holes) + holes
        new_first = HorizontalLid(
            corners=first.corners,
            adjacency=first.adjacency,
            holes=merged_holes,
        )
        new_floor = [new_first, *room.floor[1:]]
        new_rooms[ri] = Room(
            story=room.story,
            floor=new_floor,
            walls=room.walls,
            doors=room.doors,
            windows=room.windows,
            locator_id=room.locator_id,
            heating=room.heating,
        )
    return replace(payload, rooms=new_rooms)
