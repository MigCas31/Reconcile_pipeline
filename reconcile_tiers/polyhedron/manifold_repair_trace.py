"""Viewer trace export for the manifold-repair tile pipeline (per-step frames)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdge,
    HalfEdgePolyhedron,
    Vertex,
)
from reconcile_tiers.polyhedron.manifold_repair import (
    FillerCandidate,
    RoomPolyhedronBuild,
    TileFace,
    _ceiling_tiles_lost_before_build,
    apply_fillers,
    build_room_polyhedron,
    collect_room_tiles,
    extract_hole_chains,
    hypothesise_fillers,
    prepare_room_tiles,
    repair_room,
    select_fillers,
)
from reconcile_tiers.polyhedron.roof_xz_clip import (
    clip_roof_tiles_to_floor_xz,
    footprint_edges_for_viewer,
)
from reconcile_tiers.polyhedron.tile_coherence import (
    RoomTileCoherenceResult,
    audit_room_tile_coherence,
    coherence_issues_to_segments,
    filter_unconnected_ceiling_tiles,
)

SCHEMA_VERSION = 2
SELECTION = "manifold-repair-steps"

PIPELINE_STEP_LABELS: dict[str, str] = {
    "tier_payload_input": "0. Input (tier_payload)",
    "input_tiles": "1. Collected tiles",
    "tile_coherence": "2. Tile coherence",
    "roof_xz_clip": "3. Roof XZ clip",
    "tiles_merged": "4. Snap + merge coplanar",
    "half_edge_built": "5. Half-edge build",
    "holes_detected": "6. Holes (orphan edges)",
    "filler_candidates": "7. Filler candidates",
    "fillers_selected": "8. Fillers selected (ILP)",
    "fillers_applied": "9. Fillers applied",
    "rooms_repaired": "1. All rooms (repaired)",
    "building_exterior": "2. Building exterior",
}


def _plane_dict(plane: Plane) -> dict[str, float]:
    return {
        "a": float(plane.a),
        "b": float(plane.b),
        "c": float(plane.c),
        "d": float(plane.d),
    }


def tiles_to_viewer_faces(
    tiles: Sequence[TileFace],
    *,
    role: str = "tile",
    selected: bool = True,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for tile in tiles:
        corners = [[float(c[0]), float(c[1]), float(c[2])] for c in tile.corners]
        if len(corners) < 3:
            continue
        faces.append(
            {
                "id": tile.face_id,
                "plane_id": tile.face_id,
                "selected": selected,
                "label": tile.source,
                "role": role,
                "corners": corners,
                "plane": _plane_dict(tile.plane),
            }
        )
    return faces


def build_to_viewer_faces(
    build: RoomPolyhedronBuild,
    *,
    role: str = "tile",
    selected: bool = True,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for face in build.poly.faces:
        if face.half_edge is None:
            continue
        corners = _face_corners_from_build(build, face)
        if len(corners) < 3:
            continue
        tile = build.tile_face_by_id.get(face.id)
        label = tile.source if tile is not None else role
        faces.append(
            {
                "id": face.id,
                "plane_id": face.id,
                "selected": selected,
                "label": label,
                "role": role,
                "corners": corners,
                "plane": _plane_dict(face.plane),
            }
        )
    return faces


def fillers_to_viewer_faces(
    fillers: Sequence[FillerCandidate],
    *,
    role: str,
    selected: bool = True,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for filler in fillers:
        corners = [[float(c[0]), float(c[1]), float(c[2])] for c in filler.corners]
        if len(corners) < 3:
            continue
        faces.append(
            {
                "id": filler.face_id,
                "plane_id": filler.face_id,
                "selected": selected,
                "label": filler.derivation,
                "role": role,
                "corners": corners,
                "plane": _plane_dict(filler.plane),
            }
        )
    return faces


def orphan_edges_for_frame(build: RoomPolyhedronBuild) -> list[dict[str, list[float]]]:
    segments: list[dict[str, list[float]]] = []
    for he in build.orphan_half_edges:
        if he.next is None:
            continue
        a = build.vertex_coords.get(he.origin.id)
        b = build.vertex_coords.get(he.next.origin.id)
        if a is None or b is None:
            continue
        segments.append(
            {
                "a": [float(a[0]), float(a[1]), float(a[2])],
                "b": [float(b[0]), float(b[1]), float(b[2])],
            }
        )
    return segments


def _face_corners_from_build(
    build: RoomPolyhedronBuild, face: Any
) -> list[list[float]]:
    corners: list[list[float]] = []
    he = face.half_edge
    if he is None:
        return corners
    guard = 0
    while True:
        guard += 1
        if guard > 1024:
            break
        coord = build.vertex_coords.get(he.origin.id)
        if coord is not None:
            corners.append([float(coord[0]), float(coord[1]), float(coord[2])])
        nxt = he.next
        if nxt is None or nxt is face.half_edge:
            break
        he = nxt
    return corners


def _pipeline_frame(
    index: int,
    pipeline_step: str,
    faces: list[dict[str, Any]],
    *,
    orphan_edges: list[dict[str, list[float]]] | None = None,
    coherence_edges: list[dict[str, Any]] | None = None,
    footprint_edges: list[dict[str, list[float]]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "label": PIPELINE_STEP_LABELS.get(pipeline_step, pipeline_step),
        "pipeline_step": pipeline_step,
        "counts": {
            "faces": len(faces),
            "orphan_edges": len(orphan_edges or []),
            "coherence_edges": len(coherence_edges or []),
            "footprint_edges": len(footprint_edges or []),
            "vertices": 0,
            "half_edges": 0,
        },
        "issues": [],
        "events": [],
        "event_error": None,
        "faces": faces,
        "orphan_edges": orphan_edges or [],
        "coherence_edges": coherence_edges or [],
        "footprint_edges": footprint_edges or [],
        "meta": meta or {},
    }


def _coherence_meta(result: RoomTileCoherenceResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "issue_count": len(result.issues),
        "component_count": result.component_count,
        "shared_edge_count": result.shared_edge_count,
        "ceiling_clearance_m": result.ceiling_clearance_m,
        "issues": [
            {
                "kind": issue.kind,
                "message": issue.message,
                "tile_locator_ids": list(issue.tile_locator_ids),
            }
            for issue in result.issues
        ],
    }


def build_manifold_repair_room_trace(
    payload: Mapping[str, Any],
    room: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
    snap_tol: float = 0.05,
) -> dict[str, Any]:
    """Run manifold repair and capture one viewer frame per pipeline stage."""
    from reconcile_tiers.polyhedron.manifold_repair import _room_index

    story = room.get("story")
    room_idx = _room_index(room)
    tiles_raw = collect_room_tiles(payload, room, corner_tol=corner_tol)

    frames: list[dict[str, Any]] = []

    input_faces = tiles_to_viewer_faces(tiles_raw, role="tier_payload")

    if len(tiles_raw) < 4:
        frames.append(_pipeline_frame(0, "tier_payload_input", input_faces))
        frames.append(
            _pipeline_frame(
                1,
                "input_tiles",
                input_faces,
                meta={"status": "skipped", "reason": "insufficient_tiles"},
            )
        )
        return _trace_document(
            frames,
            status="skipped",
            message="insufficient_tiles",
            room_index=room_idx,
            story=story,
        )

    frames.append(_pipeline_frame(0, "tier_payload_input", input_faces))

    frames.append(
        _pipeline_frame(1, "input_tiles", tiles_to_viewer_faces(tiles_raw))
    )

    coherence = audit_room_tile_coherence(tiles_raw, corner_tol=snap_tol)
    coherence_segments = coherence_issues_to_segments(coherence.issues)
    frames.append(
        _pipeline_frame(
            2,
            "tile_coherence",
            tiles_to_viewer_faces(tiles_raw),
            coherence_edges=coherence_segments,
            meta=_coherence_meta(coherence),
        )
    )

    clip_result = clip_roof_tiles_to_floor_xz(tiles_raw)
    tiles_clipped = list(clip_result.tiles)
    footprint_edges = footprint_edges_for_viewer(tiles_raw)
    frames.append(
        _pipeline_frame(
            3,
            "roof_xz_clip",
            tiles_to_viewer_faces(tiles_clipped),
            footprint_edges=footprint_edges,
            meta={
                "clipped_locator_ids": list(clip_result.clipped_locator_ids),
                "dropped_locator_ids": list(clip_result.dropped_locator_ids),
                "floor_area_m2": clip_result.floor_area_m2,
            },
        )
    )

    tiles_filtered, dropped_ceilings = filter_unconnected_ceiling_tiles(
        tiles_clipped, corner_tol=snap_tol
    )
    tiles_merged = prepare_room_tiles(
        tiles_filtered, coord_tol=coord_tol, snap_tol=snap_tol, merge_coplanar=True
    )
    pre_filter_ceilings = _ceiling_tiles_lost_before_build(
        tiles_raw=tiles_raw,
        tiles_clipped=tiles_clipped,
        build_tiles=tiles_merged,
    )
    frames.append(
        _pipeline_frame(
            4,
            "tiles_merged",
            tiles_to_viewer_faces(tiles_merged),
            meta={
                "tile_count_raw": len(tiles_raw),
                "tile_count_clipped": len(tiles_clipped),
                "tile_count_filtered": len(tiles_filtered),
                "tile_count_merged": len(tiles_merged),
                "dropped_ceiling_locators": list(dropped_ceilings),
                "roof_clip_dropped": list(clip_result.dropped_locator_ids),
            },
        )
    )

    build = build_room_polyhedron(
        tiles_merged, coord_tol=coord_tol, snap_tol=0.0, merge_coplanar=False
    )
    tile_faces = build_to_viewer_faces(build, role="tile")
    frames.append(
        _pipeline_frame(
            5,
            "half_edge_built",
            tile_faces,
            orphan_edges=[],
            meta={
                "skipped_tiles": len(build.skipped_tiles),
                "orphan_count": len(build.orphan_half_edges),
            },
        )
    )

    extraction = extract_hole_chains(build)
    frames.append(
        _pipeline_frame(
            6,
            "holes_detected",
            list(tile_faces),
            orphan_edges=orphan_edges_for_frame(build),
            meta={
                "closed_holes": len(extraction.closed_chains),
                "open_chains": len(extraction.open_chains),
                "ambiguous_vertices": len(extraction.ambiguous_vertices),
            },
        )
    )

    next_face_id = max((f.id for f in build.poly.faces), default=-1) + 1
    fillers = hypothesise_fillers(
        build,
        extraction,
        first_face_id=next_face_id,
        all_tiles=tiles_merged,
        skipped_tiles=build.skipped_tiles,
    )
    selection = select_fillers(build, extraction, fillers)

    frames.append(
        _pipeline_frame(
            7,
            "filler_candidates",
            tile_faces + fillers_to_viewer_faces(fillers, role="filler_candidate"),
            orphan_edges=orphan_edges_for_frame(build),
            meta={"candidate_count": len(fillers)},
        )
    )

    selected_ids = {f.face_id for f in selection.selected}
    rejected = [f for f in fillers if f.face_id not in selected_ids]
    frames.append(
        _pipeline_frame(
            8,
            "fillers_selected",
            tile_faces
            + fillers_to_viewer_faces(selection.selected, role="filler_selected")
            + fillers_to_viewer_faces(rejected, role="filler_candidate", selected=False),
            orphan_edges=orphan_edges_for_frame(build),
            meta={
                "selected_count": len(selection.selected),
                "rejected_count": len(rejected),
                "solver_status": selection.solver_status,
            },
        )
    )

    build_apply = _copy_build_for_apply(build)
    extraction_apply = extract_hole_chains(build_apply)
    fillers_applied = apply_fillers(
        build_apply, extraction_apply, selection.selected
    )
    applied_faces = build_to_viewer_faces(build_apply, role="tile")
    for face in applied_faces:
        if face["id"] not in build_apply.tile_face_by_id:
            face["role"] = "filler"
            face["label"] = "polyhedron_v3_filler"

    status = "watertight" if not build_apply.orphan_half_edges else "holes_remaining"
    frames.append(
        _pipeline_frame(
            9,
            "fillers_applied",
            applied_faces,
            orphan_edges=orphan_edges_for_frame(build_apply),
            meta={
                "fillers_applied": fillers_applied,
                "orphans_remaining": len(build_apply.orphan_half_edges),
            },
        )
    )

    return _trace_document(
        frames,
        status=status,
        message=status,
        room_index=room_idx,
        story=story,
        repair_summary={
            "status": status,
            "tile_count": len(tiles_raw),
            "tile_count_filtered": len(tiles_filtered),
            "dropped_ceiling_locators": list(dropped_ceilings),
            "roof_clip_clipped": list(clip_result.clipped_locator_ids),
            "roof_clip_dropped": list(clip_result.dropped_locator_ids),
            "coherence_ok": coherence.ok,
            "coherence_issue_count": len(coherence.issues),
            "fillers_applied": fillers_applied,
            "closed_holes": len(extraction.closed_chains),
            "orphans_remaining": len(build_apply.orphan_half_edges),
            "skipped_tiles": len(build.skipped_tiles),
        },
    )


def _copy_build_for_apply(build: RoomPolyhedronBuild) -> RoomPolyhedronBuild:
    """Shallow-copy topology so apply_fillers does not mutate the pre-apply build."""
    poly = HalfEdgePolyhedron()
    vertex_coords = dict(build.vertex_coords)
    old_to_new_vertex: dict[int, int] = {}
    for old_v in build.poly.vertices:
        new_id = len(poly.vertices)
        old_to_new_vertex[old_v.id] = new_id
        poly.vertices.append(Vertex(id=new_id))

    face_by_id: dict[int, Face] = {}
    for old_f in build.poly.faces:
        face = Face(id=old_f.id, plane=old_f.plane)
        poly.faces.append(face)
        face_by_id[old_f.id] = face

    old_to_new_he: dict[int, HalfEdge] = {}
    for old_he in build.poly.half_edges:
        new_he = HalfEdge(
            id=old_he.id,
            origin=poly.vertices[old_to_new_vertex[old_he.origin.id]],
        )
        old_to_new_he[old_he.id] = new_he
        poly.half_edges.append(new_he)

    for old_he in build.poly.half_edges:
        new_he = old_to_new_he[old_he.id]
        if old_he.opposite is not None:
            new_he.opposite = old_to_new_he[old_he.opposite.id]
        if old_he.next is not None:
            new_he.next = old_to_new_he[old_he.next.id]
        if old_he.face is not None:
            new_he.face = face_by_id.get(old_he.face.id)

    for old_f in build.poly.faces:
        face = face_by_id.get(old_f.id)
        if face is not None and old_f.half_edge is not None:
            face.half_edge = old_to_new_he.get(old_f.half_edge.id)

    for v in poly.vertices:
        outs = [he for he in poly.half_edges if he.origin.id == v.id]
        if outs:
            v.outgoing = min(outs, key=lambda h: h.id)

    tile_face_by_id = dict(build.tile_face_by_id)
    orphan_half_edges = [
        old_to_new_he[he.id]
        for he in build.orphan_half_edges
        if he.id in old_to_new_he
    ]
    return RoomPolyhedronBuild(
        poly=poly,
        tile_face_by_id=tile_face_by_id,
        orphan_half_edges=orphan_half_edges,
        skipped_tiles=list(build.skipped_tiles),
        vertex_coords=vertex_coords,
    )


def _trace_document(
    frames: list[dict[str, Any]],
    *,
    status: str,
    message: str,
    room_index: int | None,
    story: Any,
    repair_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": SELECTION,
        "room_index": room_index,
        "story": story,
        "tolerances": {},
        "frames": frames,
        "steps": [],
        "stop": {
            "reason": status,
            "message": message,
            "remaining_issues": [],
            "remaining_events": [],
        },
        "repair_summary": repair_summary,
    }


def envelope_faces_to_viewer(
    envelope: Any,
    *,
    role: str = "tile",
) -> list[dict[str, Any]]:
    if envelope is None:
        return []
    faces: list[dict[str, Any]] = []
    for i, face in enumerate(envelope.faces):
        corners = [[float(c[0]), float(c[1]), float(c[2])] for c in face.corners]
        if len(corners) < 3:
            continue
        face_role = role
        if getattr(face, "source", "").startswith("polyhedron_v3_filler"):
            face_role = "filler"
        faces.append(
            {
                "id": i,
                "plane_id": i,
                "selected": True,
                "label": face.source or face.kind,
                "role": face_role,
                "corners": corners,
                "plane": _plane_dict(face.plane),
            }
        )
    return faces


def build_manifold_repair_building_trace(
    payload: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
) -> dict[str, Any]:
    """Two-frame building trace: all repaired rooms + exterior envelope."""
    from reconcile_tiers.polyhedron.building_merge import (
        envelope_candidate_from_building,
        repair_building,
    )

    building = repair_building(
        payload, corner_tol=corner_tol, coord_tol=coord_tol
    )
    interior_envelope = envelope_candidate_from_building(
        building, include_interior=True
    )
    exterior_envelope = envelope_candidate_from_building(building)

    frames = [
        _pipeline_frame(0, "tier_payload_input", []),
        _pipeline_frame(
            1,
            "rooms_repaired",
            envelope_faces_to_viewer(interior_envelope, role="tile"),
            meta={"room_count": len(building.rooms), "face_count": len(building.faces)},
        ),
        _pipeline_frame(
            2,
            "building_exterior",
            envelope_faces_to_viewer(exterior_envelope, role="tile"),
            meta={
                "exterior_count": len(building.exterior_faces),
                "interior_count": len(building.faces) - len(building.exterior_faces),
            },
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": SELECTION,
        "room_index": None,
        "story": None,
        "tolerances": {},
        "frames": frames,
        "steps": [],
        "stop": {
            "reason": "watertight",
            "message": f"{len(building.rooms)} rooms",
            "remaining_issues": [],
            "remaining_events": [],
        },
    }
