"""Step 2b of the assembly: Openings (doors, windows, passages) per Wall.

Anchor: each `ExtractedElement.parent_wall_id` ties the opening to a
specific Wall by id. The opening's corners are projected onto the
canonical wall plane (vertical, anchored at the bottom-edge XZ) so the
Wall constructor's coplanarity invariant holds. The original 3D corners
are preserved as ScanEvidence.

Openings whose parent_wall_id does not match any Wall in the room
become orphan evidence on the residual stream.
"""

from __future__ import annotations

from reconcile_tiers.extract.building import ExtractedElement, ExtractedRoom
from reconcile_tiers.payload.schema import Plane, Vec3
from reconcile_tiers.twin.types import Evidence, Opening, OpeningKind, Provenance


def openings_by_wall_id(
    room: ExtractedRoom, *, wall_planes: dict[str, Plane], building_uuid: str
) -> tuple[dict[str, tuple[Opening, ...]], tuple[Evidence, ...]]:
    """Group Openings by their host Wall id.

    `wall_planes` is keyed by raw wall id (= `ExtractedWall.id`, which is
    also what `ExtractedElement.parent_wall_id` references). Each plane
    is the canonical vertical plane the corresponding Wall primitive
    will use.

    Returns `(openings_by_wall_id, orphan_evidence)`.
    """
    openings: dict[str, list[Opening]] = {}
    orphans: list[Evidence] = []

    sources: list[tuple[ExtractedElement, OpeningKind, str]] = []
    for d in room.doors:
        sources.append((d, OpeningKind.DOOR, "extracted_room.doors"))
    for w in room.windows:
        sources.append((w, OpeningKind.WINDOW, "extracted_room.windows"))
    for p in room.openings:
        sources.append((p, OpeningKind.PASSAGE, "extracted_room.openings"))

    for elem, kind, source in sources:
        evidence = _opening_evidence(elem, source)
        host_plane = (
            wall_planes.get(elem.parent_wall_id) if elem.parent_wall_id else None
        )
        if host_plane is None:
            orphans.append(evidence)
            continue
        opening = _build_opening(
            elem,
            kind=kind,
            host_plane=host_plane,
            evidence=evidence,
            building_uuid=building_uuid,
            room=room,
        )
        if opening is None:
            orphans.append(evidence)
        else:
            openings.setdefault(elem.parent_wall_id, []).append(opening)

    return {k: tuple(v) for k, v in openings.items()}, tuple(orphans)


def _build_opening(
    elem: ExtractedElement,
    *,
    kind: OpeningKind,
    host_plane: Plane,
    evidence: Evidence,
    building_uuid: str,
    room: ExtractedRoom,
) -> Opening | None:
    if len(elem.corners) < 3:
        return None
    projected = tuple(_project_onto_plane(host_plane, c) for c in elem.corners)
    return Opening(
        id=f"{building_uuid}::opening::{room.story}:{room.index}::{elem.id}",
        kind=kind,
        polygon=projected,
        evidence=(evidence,),
    )


def _project_onto_plane(plane: Plane, corner: list[float]) -> Vec3:
    """Project a 3D point onto an a*x + b*y + c*z + d = 0 plane.

    The plane normal (a, b, c) is unit-length by construction (Wall
    builder normalises it). The signed distance is a*x + b*y + c*z + d.
    """
    x, y, z = float(corner[0]), float(corner[1]), float(corner[2])
    s = plane.a * x + plane.b * y + plane.c * z + plane.d
    return Vec3(
        x=x - s * plane.a,
        y=y - s * plane.b,
        z=z - s * plane.c,
    )


def _opening_evidence(elem: ExtractedElement, source: str) -> Evidence:
    geometry = tuple(
        Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in elem.corners
    )
    return Evidence(
        provenance=Provenance(kind="scan", source=source),
        geometry=geometry,
    )
