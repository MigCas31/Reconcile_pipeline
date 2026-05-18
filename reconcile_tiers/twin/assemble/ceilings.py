"""Step 3 of the assembly: Ceiling per Room.

Phase B-2 / B-3a scope: single-plane ceilings (flat or sloped) only.
Composite (kinked) ceilings — those with both a flat lid and a slope
held together by a knee wall — are deferred to Phase B-3b.

For both flat and sloped:
  - The canonical Ceiling polygon shares the host Floor's XZ extent.
  - Flat: Y = max wall-top Y; plane is horizontal.
  - Sloped: plane = best-fit through wall-top corners; polygon
    corners lifted onto that plane.
  - Evidence: ComputedEvidence anchored by wall ids + floor id, plus
    ScanEvidence for each `raw_ceiling_plane` in the room.
"""

from __future__ import annotations

from reconcile_tiers.extract.building import ExtractedRoom
from reconcile_tiers.payload.schema import Plane, Vec3
from reconcile_tiers.twin._geometry import (
    fit_plane,
    lift_xz_to_plane,
    plane_is_horizontal,
    plane_is_vertical,
)
from reconcile_tiers.twin.types import (
    Ceiling,
    CeilingKind,
    CeilingSeam,
    Evidence,
    Floor,
    Provenance,
    Wall,
)


def ceiling_for_room(
    room: ExtractedRoom,
    *,
    floor: Floor,
    walls: tuple[Wall, ...],
    building_uuid: str,
) -> tuple[Ceiling | None, tuple[Evidence, ...]]:
    """Construct a Ceiling for `room`: flat, oblique, or composite.

    Returns `(ceiling, scan_evidence)`.
    """
    if not walls:
        return None, ()
    if room.ceiling_is_kinked:
        return _composite_ceiling(
            room, floor=floor, walls=walls, building_uuid=building_uuid
        )
    if room.ceiling_type == "flat":
        return _flat_ceiling(
            room, floor=floor, walls=walls, building_uuid=building_uuid
        )
    if room.ceiling_type == "sloped":
        return _sloped_ceiling(
            room, floor=floor, walls=walls, building_uuid=building_uuid
        )
    return None, ()


# Backwards-compat alias used by Phase B-2 tests.
ceiling_for_flat_room = ceiling_for_room


def _flat_ceiling(
    room: ExtractedRoom,
    *,
    floor: Floor,
    walls: tuple[Wall, ...],
    building_uuid: str,
) -> tuple[Ceiling, tuple[Evidence, ...]]:
    ceiling_y = max(_wall_top_y(w) for w in walls)
    polygon = tuple(Vec3(x=c.x, y=ceiling_y, z=c.z) for c in floor.polygon)
    plane = Plane(a=0.0, b=1.0, c=0.0, d=-ceiling_y)
    return _build_ceiling(
        room=room,
        floor=floor,
        walls=walls,
        polygon=polygon,
        plane=plane,
        kind=CeilingKind.FLAT,
        building_uuid=building_uuid,
    )


def _sloped_ceiling(
    room: ExtractedRoom,
    *,
    floor: Floor,
    walls: tuple[Wall, ...],
    building_uuid: str,
) -> tuple[Ceiling | None, tuple[Evidence, ...]]:
    # The "top" of a Wall polygon now means every corner above the
    # bottom Y. Use the highest corners across walls as anchor points
    # for the slope plane fit (one per wall).
    wall_top_corners = []
    for w in walls:
        sorted_y = sorted(w.polygon, key=lambda c: -c.y)
        # Take the top 2 corners — sufficient for a fit anchor.
        wall_top_corners.extend(sorted_y[:2])
    plane = fit_plane(wall_top_corners)
    if plane is None:
        return None, ()
    if plane_is_horizontal(plane) or plane_is_vertical(plane):
        # Wall-tops are coplanar horizontal (or pathological): they don't
        # span an oblique surface. Phase B-3b's composite path or a later
        # roof-anchored step has to provide the slope; skip here.
        return None, ()
    polygon = lift_xz_to_plane(floor.polygon, plane)
    return _build_ceiling(
        room=room,
        floor=floor,
        walls=walls,
        polygon=polygon,
        plane=plane,
        kind=CeilingKind.OBLIQUE,
        building_uuid=building_uuid,
    )


def _build_ceiling(
    *,
    room: ExtractedRoom,
    floor: Floor,
    walls: tuple[Wall, ...],
    polygon: tuple[Vec3, ...],
    plane: Plane,
    kind: CeilingKind,
    building_uuid: str,
) -> tuple[Ceiling, tuple[Evidence, ...]]:
    parent_ids: tuple[str, ...] = (*tuple(w.id for w in walls), floor.id)
    computed = Evidence(
        provenance=Provenance(kind="computed", source="room_frame.wall_tops"),
        geometry=polygon,
        parents=parent_ids,
    )
    scan_evidence = tuple(
        Evidence(
            provenance=Provenance(
                kind="scan", source="extracted_room.raw_ceiling_planes"
            ),
            geometry=tuple(
                Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in raw.corners
            ),
        )
        for raw in room.raw_ceiling_planes
        if len(raw.corners) >= 3
    )
    ceiling = Ceiling(
        id=f"{building_uuid}::ceiling::{room.story}:{room.index}",
        kind=kind,
        polygon=polygon,
        plane=plane,
        evidence=(computed, *scan_evidence),
    )
    return ceiling, scan_evidence


def _wall_top_y(wall: Wall) -> float:
    return float(wall.top_y)


def _composite_ceiling(
    room: ExtractedRoom,
    *,
    floor: Floor,
    walls: tuple[Wall, ...],
    building_uuid: str,
) -> tuple[Ceiling | None, tuple[Evidence, ...]]:
    """Composite (kinked) ceiling: flat lid + ≥1 oblique sub-Ceilings,
    seamed by the slope(s)' high edge(s).

    Per the corpus survey: extraction's `ceiling_flat_polygon` is at one
    horizontal Y (the lid); each slope polygon's two highest-Y corners
    define its seam with the lid. The seam line lies on the lid plane
    and on the slope plane (their intersection); using the slope's own
    high edge as the seam avoids needing the flat polygon's corners to
    coincide with slope corners.
    """
    flat_polygon_in = room.ceiling_flat_polygon
    slope_polygons_in = (
        room.ceiling_slope_polygons
        if room.ceiling_slope_polygons
        else ([room.ceiling_slope_polygon] if room.ceiling_slope_polygon else [])
    )
    slope_polygons_in = [sp for sp in slope_polygons_in if len(sp) >= 3]
    if not flat_polygon_in or len(flat_polygon_in) < 3 or not slope_polygons_in:
        return None, ()

    flat_part, _flat_evidence = _build_flat_sub_ceiling(
        flat_polygon_in, room=room, building_uuid=building_uuid
    )
    if flat_part is None:
        return None, ()

    parts: list[Ceiling] = [flat_part]
    seams: list[CeilingSeam] = []
    for idx, slope_corners in enumerate(slope_polygons_in):
        oblique_part, seam = _build_oblique_sub_ceiling(
            slope_corners,
            flat_part=flat_part,
            slope_idx=idx,
            room=room,
            building_uuid=building_uuid,
        )
        if oblique_part is None or seam is None:
            return None, ()
        parts.append(oblique_part)
        seams.append(seam)

    parent_ids: tuple[str, ...] = (*tuple(w.id for w in walls), floor.id)
    composite_polygon = tuple(
        Vec3(x=c.x, y=flat_part.polygon[0].y, z=c.z) for c in floor.polygon
    )
    computed = Evidence(
        provenance=Provenance(kind="computed", source="room_frame.kinked_composite"),
        geometry=composite_polygon,
        parents=parent_ids,
    )
    scan_evidence = tuple(
        Evidence(
            provenance=Provenance(
                kind="scan", source="extracted_room.raw_ceiling_planes"
            ),
            geometry=tuple(
                Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in raw.corners
            ),
        )
        for raw in room.raw_ceiling_planes
        if len(raw.corners) >= 3
    )

    ceiling = Ceiling(
        id=f"{building_uuid}::ceiling::{room.story}:{room.index}",
        kind=CeilingKind.COMPOSITE,
        polygon=composite_polygon,
        plane=None,
        parts=tuple(parts),
        seams=tuple(seams),
        evidence=(computed, *scan_evidence),
    )
    return ceiling, scan_evidence


def _build_flat_sub_ceiling(
    corners_3d: list[list[float]],
    *,
    room: ExtractedRoom,
    building_uuid: str,
) -> tuple[Ceiling | None, tuple[Evidence, ...]]:
    import statistics

    canonical_y = statistics.median(c[1] for c in corners_3d)
    polygon = tuple(
        Vec3(x=float(c[0]), y=float(canonical_y), z=float(c[2])) for c in corners_3d
    )
    plane = Plane(a=0.0, b=1.0, c=0.0, d=-canonical_y)
    raw_geometry = tuple(
        Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners_3d
    )
    evidence = Evidence(
        provenance=Provenance(
            kind="scan", source="extracted_room.ceiling_flat_polygon"
        ),
        geometry=raw_geometry,
    )
    sub = Ceiling(
        id=f"{building_uuid}::ceiling::{room.story}:{room.index}::flat",
        kind=CeilingKind.FLAT,
        polygon=polygon,
        plane=plane,
        evidence=(evidence,),
    )
    return sub, (evidence,)


def _build_oblique_sub_ceiling(
    corners_3d: list[list[float]],
    *,
    flat_part: Ceiling,
    slope_idx: int,
    room: ExtractedRoom,
    building_uuid: str,
) -> tuple[Ceiling | None, CeilingSeam | None]:
    if len(corners_3d) < 3:
        return None, None
    raw = tuple(Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners_3d)
    plane = fit_plane(raw)
    if plane is None or plane_is_horizontal(plane) or plane_is_vertical(plane):
        return None, None
    polygon = lift_xz_to_plane(raw, plane)

    flat_y = float(flat_part.polygon[0].y)
    sorted_by_y = sorted(polygon, key=lambda v: v.y)
    high_a, high_b = sorted_by_y[-2], sorted_by_y[-1]
    # Seam = slope plane ∩ flat plane (a horizontal line at flat_y). The
    # endpoints are derived by setting Y = flat_y at the slope's two
    # high-corner XZ positions — i.e. project the slope's high edge onto
    # the flat lid plane. No scan-precision tolerance: the projection is
    # exact.
    seam_a = Vec3(x=high_a.x, y=flat_y, z=high_a.z)
    seam_b = Vec3(x=high_b.x, y=flat_y, z=high_b.z)
    flat_part_id = flat_part.id

    evidence = Evidence(
        provenance=Provenance(
            kind="scan", source="extracted_room.ceiling_slope_polygon"
        ),
        geometry=raw,
    )
    sub_id = f"{building_uuid}::ceiling::{room.story}:{room.index}::oblique:{slope_idx}"
    sub = Ceiling(
        id=sub_id,
        kind=CeilingKind.OBLIQUE,
        polygon=polygon,
        plane=plane,
        evidence=(evidence,),
    )
    seam_evidence = Evidence(
        provenance=Provenance(kind="computed", source="slope_high_edge"),
        geometry=(seam_a, seam_b),
        parents=(flat_part_id, sub_id),
    )
    seam = CeilingSeam(
        id=f"{building_uuid}::ceiling-seam::{room.story}:{room.index}:{slope_idx}",
        endpoint_a=seam_a,
        endpoint_b=seam_b,
        member_part_ids=(flat_part_id, sub_id),
        evidence=(seam_evidence,),
    )
    return sub, seam
