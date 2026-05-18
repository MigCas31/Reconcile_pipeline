from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from reconcile_tiers.assemble.building_center import compute_building_center
from reconcile_tiers.assemble.ceiling_painter import assemble_ceiling
from reconcile_tiers.assemble.gaps_to_pieces import assemble_gap_pieces
from reconcile_tiers.assemble.walls_to_rooms import assemble_rooms
from reconcile_tiers.build_internals.cli import main
from reconcile_tiers.build_internals.envelope_pieces import (
    _classification,
    _clean_ceiling_piece_geometries,
    _drop_redundant_ceiling_split_overlaps,
)
from reconcile_tiers.build_internals.payload_polish import _polish_payload_ceilings
from reconcile_tiers.build_internals.per_room_emission import (
    _ceiling_candidates,
)
from reconcile_tiers.build_internals.raw_ceiling_filter import (
    _filter_noisy_raw_candidates,
)
from reconcile_tiers.build_internals.stair_holes import _apply_stair_floor_holes
from reconcile_tiers.build_internals.void_closure import (
    _close_residual_room_ceiling_voids,
    _close_within_story_free_edges,
    _inject_residual_void_walls,
)
from reconcile_tiers.build_internals.wall_clipping import (
    _clip_top_story_walls_to_gable_roof,
)
from reconcile_tiers.config import (
    residual_ceiling_void_closure_enabled,
)
from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.ingest.scan_cache import find_scan_cache_dir, load_scan_metadata
from reconcile_tiers.payload.adjacency import tag_payload
from reconcile_tiers.payload.schema import (
    TierPayload,
    payload_to_dict,
)
from reconcile_tiers.payload.validate import validate_payload
from reconcile_tiers.roof.roof import build_roof_model


def build_tier_payload(
    uuid: str,
    pipeline_dir: Path | str = Path("pipeline-outputs"),
    scan_root: Path | str | None = Path(".scan-cache"),
    *,
    drops_sink: list | None = None,
    plane_evidence_sink: list | None = None,
) -> TierPayload:
    from reconcile_tiers.assemble.dormer_reconstruction import reconstruct_dormers
    from reconcile_tiers.assemble.synthesis import synthesise_thermal_envelope
    from reconcile_tiers.build_internals.ceiling_helpers import _story_labels

    model = extract_building_model(uuid, pipeline_dir, scan_root)
    scan_dir = find_scan_cache_dir(uuid, scan_root) if scan_root else None
    scan_meta = load_scan_metadata(scan_dir)
    ys_by_story: dict[int, list[float]] = {}
    for room in model.rooms:
        if room.floor_polygon:
            y = sum(c[1] for c in room.floor_polygon) / len(room.floor_polygon)
            ys_by_story.setdefault(room.story, []).append(y)
    story_mean_ys = [
        sum(ys_by_story[i]) / len(ys_by_story[i]) if ys_by_story.get(i) else 0.0
        for i in range(model.stories_found)
    ]
    roof = build_roof_model(model)
    candidates = _ceiling_candidates(model, roof)
    pre_filter_candidates = list(candidates)
    ceiling_stories = {
        candidate.locator_id: candidate.story for candidate in candidates
    }

    model = _inject_residual_void_walls(model, roof, candidates)

    # Apply the noisy-raw-ceiling gate AFTER void detection but BEFORE the
    # painter. Void detection should see the full pre-gate set (a noisy raw
    # plane still indicates "the scan saw something here", so it shouldn't
    # trigger a synthetic void) -- but the painter / final payload should not
    # include those noisy fragments.
    candidates = _filter_noisy_raw_candidates(
        candidates, model, roof, drops_sink=drops_sink
    )
    if plane_evidence_sink is not None:
        from reconcile_tiers.build_internals.plane_evidence import (
            build_plane_evidence,
        )

        plane_evidence_sink.append(
            build_plane_evidence(
                model,
                roof,
                pre_filter_candidates=pre_filter_candidates,
                post_filter_candidates=candidates,
                raw_gate_drops=drops_sink,
            )
        )

    rooms = _clip_top_story_walls_to_gable_roof(assemble_rooms(model), model, roof)

    # Scan-driven painter first; synthesis runs as a separate post-paint pass
    # after dormer reconstruction.
    ceilings = assemble_ceiling(candidates)

    ceilings, dormer_thermal = reconstruct_dormers(
        ceilings, roof.dormer_candidates, roof.oblique, model, ceiling_stories
    )

    synth = synthesise_thermal_envelope(
        rooms,
        [w for r in rooms for w in r.walls],
        ceilings,
        roof,
        dormer_thermal,
        model=model,
    )
    ceilings = synth.ceilings
    knee_walls = synth.knee_walls
    gable_closures = synth.gable_closures
    dormer_faces = synth.dormer_faces
    visual_shells = synth.visual_shells

    ceilings = _drop_redundant_ceiling_split_overlaps(ceilings)
    ceilings = _clean_ceiling_piece_geometries(ceilings)
    if residual_ceiling_void_closure_enabled():
        model = _close_residual_room_ceiling_voids(model, ceilings)
    model = _close_within_story_free_edges(model)

    payload = TierPayload(
        schema_version="1",
        uuid=model.uuid,
        address=model.address,
        building_center=compute_building_center(model),
        classification=_classification(model, roof),
        rooms=rooms,
        gaps=assemble_gap_pieces(model),
        ceiling=ceilings,
        knee_walls=knee_walls,
        dormer_faces=dormer_faces,
        gable_closures=gable_closures,
        story_labels=_story_labels(scan_meta.has_basement, story_mean_ys),
        visual_shells=visual_shells,
    )
    payload = tag_payload(
        payload,
        has_basement=scan_meta.has_basement,
        oblique_surface_corners=[surface.corners for surface in roof.oblique],
    )
    payload = _polish_payload_ceilings(payload)
    # Stair reconstruction is gated behind an opt-in env var while the
    # geometry quality is iterated. Default off so payloads ship without
    # stairs (and the viewer renders nothing). Set TIER_STAIRS_ENABLED=1
    # to re-enable for development / experiments.
    if os.environ.get("TIER_STAIRS_ENABLED") == "1":
        from reconcile_tiers.assemble.stairs import reconstruct_stairs
        from reconcile_tiers.ingest.merged import load_merged

        payload_dict = payload_to_dict(payload)
        merged_doc = load_merged(uuid, pipeline_dir)
        stairs = reconstruct_stairs(
            merged_doc.data,
            payload_dict.get("rooms") or [],
            payload_dict.get("ceiling") or [],
            uuid=model.uuid,
            drops_sink=drops_sink,
        )
    else:
        stairs = []
    if stairs:
        payload = _apply_stair_floor_holes(payload, stairs)
    payload = replace(payload, stairs=stairs)
    validate_payload(payload)
    from reconcile_tiers.quality.score import score_building

    quality = score_building(payload)
    if quality is not None:
        payload = replace(payload, roof_quality=quality)
    if roof.primitive_records:
        from reconcile_tiers.roof_primitive.validation import (
            validate_primitive_emissions,
        )

        violations = validate_primitive_emissions(payload, roof.primitive_records)
        if violations and drops_sink is not None:
            drops_sink.extend(
                {
                    "kind": v.kind,
                    "wing_index": v.wing_index,
                    "primitive": v.primitive,
                    "detail": v.detail,
                    "locator_id": v.locator_id,
                }
                for v in violations
            )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
