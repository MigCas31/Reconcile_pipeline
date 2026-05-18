"""reconcile_v3 orchestrator — linear composition, no cross-stage imports."""

from __future__ import annotations

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from .audit import AuditLog, note_geom_skip
from .io.load import V3Input
from .models import V3Building, V3SlantedRoof, V3UnresolvedRegion
from .stages.dormers import detect_dormers_v3
from .stages.fill_remaining import cover_remaining_slab_above
from .stages.flat_ceilings import flat_ceilings_unambiguous
from .stages.gable_extension import classify_gable_extension
from .stages.gaps import close_obvious_gaps
from .stages.merged_slanted_roof_proposals import build_merged_roof_segments
from .stages.parts import identify_building_parts
from .stages.slabs import build_room_slabs
from .stages.slanted_roof_proposals import propose_slanted_roofs
from .stages.slanted_roofs import cluster_oblique_to_roof_faces
from .stages.wall_extensions import extend_walls_to_slab_above


def _slanted_roof_footprint_union(slanted_roofs: list[V3SlantedRoof]):
    polys = []
    for roof in slanted_roofs:
        if len(roof.corners) < 3:
            continue
        try:
            p = ShapelyPolygon([(c[0], c[2]) for c in roof.corners])
        except Exception as exc:
            note_geom_skip(exc, "slanted_roof_footprint")
            continue
        if p.is_valid and not p.is_empty:
            polys.append(p)
    return unary_union(polys) if polys else None


def _drop_unresolved_under_slanted_roofs(
    unresolved: list[V3UnresolvedRegion],
    slanted_roofs: list[V3SlantedRoof],
) -> tuple[list[V3UnresolvedRegion], set[str]]:
    roof_union = _slanted_roof_footprint_union(slanted_roofs)
    if roof_union is None or roof_union.is_empty:
        return unresolved, set()
    kept: list[V3UnresolvedRegion] = []
    dropped_ids: set[str] = set()
    for u in unresolved:
        if len(u.footprint_xz) < 3:
            kept.append(u)
            continue
        try:
            fp = ShapelyPolygon([(c[0], c[2]) for c in u.footprint_xz])
        except Exception:
            kept.append(u)
            continue
        if not fp.is_valid or fp.is_empty:
            kept.append(u)
            continue
        inter = fp.intersection(roof_union)
        if inter.is_empty or fp.area <= 0:
            kept.append(u)
            continue
        if inter.area / fp.area < 0.5:
            kept.append(u)
        else:
            dropped_ids.add(u.id)
    return kept, dropped_ids


def run_pipeline(data: V3Input) -> V3Building:
    audit = AuditLog()
    building = V3Building(uuid=data.uuid, address=data.address)

    parts, part_unresolved = identify_building_parts(data.uuid, data.rooms)
    building.parts = parts
    building.unresolved.extend(part_unresolved)
    for p in parts:
        audit.record(p.id, p.trace)
    for u in part_unresolved:
        audit.record(u.id, u.trace)

    closed_gaps, gap_slabs, gap_ceilings, gap_unresolved = close_obvious_gaps(
        data.uuid, data.rooms
    )
    building.gaps.extend(closed_gaps)
    building.slabs.extend(gap_slabs)
    building.flat_ceilings.extend(gap_ceilings)
    building.unresolved.extend(gap_unresolved)
    for g in closed_gaps:
        audit.record(g.id, g.trace)
    for s in gap_slabs:
        audit.record(s.id, s.trace)
    for c in gap_ceilings:
        audit.record(c.id, c.trace)
    for u in gap_unresolved:
        audit.record(u.id, u.trace)

    room_slabs = build_room_slabs(data.uuid, data.rooms)
    building.slabs.extend(room_slabs)
    for s in room_slabs:
        audit.record(s.id, s.trace)

    slanted_roofs, slope_hypotheses, slope_unresolved = cluster_oblique_to_roof_faces(
        data.uuid, data.rooms, building.parts
    )
    building.slanted_roofs = slanted_roofs
    building.unresolved.extend(slope_unresolved)
    for r in slanted_roofs:
        audit.record(r.id, r.trace)
    for u in slope_unresolved:
        audit.record(u.id, u.trace)

    extensions, ext_unresolved = extend_walls_to_slab_above(
        data.uuid, data.rooms, building.slabs, slope_hypotheses
    )
    building.wall_extensions.extend(extensions)
    building.unresolved.extend(ext_unresolved)
    for e in extensions:
        audit.record(e.id, e.trace)
    for u in ext_unresolved:
        audit.record(u.id, u.trace)

    flat_ceilings, ceiling_unresolved = flat_ceilings_unambiguous(data.uuid, data.rooms)
    building.flat_ceilings.extend(flat_ceilings)
    building.unresolved.extend(ceiling_unresolved)
    for c in flat_ceilings:
        audit.record(c.id, c.trace)
    for u in ceiling_unresolved:
        audit.record(u.id, u.trace)

    fill_flats, fill_slants, fill_unresolved = cover_remaining_slab_above(
        data.uuid,
        data.rooms,
        building.parts,
        building.slabs,
        building.flat_ceilings,
        building.slanted_roofs,
        slope_hypotheses,
    )
    building.flat_ceilings.extend(fill_flats)
    building.slanted_roofs.extend(fill_slants)
    building.unresolved.extend(fill_unresolved)
    for c in fill_flats:
        audit.record(c.id, c.trace)
    for r in fill_slants:
        audit.record(r.id, r.trace)
    for u in fill_unresolved:
        audit.record(u.id, u.trace)

    building.roof_proposals = propose_slanted_roofs(
        data.uuid, data.rooms, building.gaps, building.slabs, building.slanted_roofs
    )
    for p in building.roof_proposals:
        audit.record(p.id, p.trace)

    building.merged_roof_segments = build_merged_roof_segments(
        data.uuid, data.rooms, building.gaps, building.slabs, building.roof_proposals
    )
    for s in building.merged_roof_segments:
        audit.record(s.id, s.trace)

    building.dormers = detect_dormers_v3(data.uuid, data.rooms, building.slanted_roofs)
    for d in building.dormers:
        audit.record(d.id, d.trace)

    for element_id, trace in classify_gable_extension(building, data.rooms):
        audit.record(element_id, trace)

    building.unresolved, dropped_ids = _drop_unresolved_under_slanted_roofs(
        building.unresolved, building.slanted_roofs
    )
    if dropped_ids:
        audit.entries = [e for e in audit.entries if e["element_id"] not in dropped_ids]

    building.audit_log = audit.to_list()
    return building
