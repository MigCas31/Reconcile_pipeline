from __future__ import annotations

from typing import Any

from .chains import collect_chain_support
from .config import DEFAULT_CONFIG, ScorerV2Config
from .global_selection import apply_global_xy_selection
from .intersection_seams import compute_intersection_seam_pieces
from .layer_policy import classify_split_piece_rows
from .models import BuildingResult, CorpusResult
from .ownership import annotate_split_rows_with_ownership
from .plane_relations import build_plane_relation_graph
from .raw_evidence import collect_raw_evidence
from .relation_context import build_plane_contexts
from .reporting import build_run_output
from .splitter import build_split_pieces, split_piece_rows
from .support_scoring import score_targets
from .targets import collect_targets_for_building


def _attach_context_fields(row: dict[str, Any], context: Any) -> dict[str, Any]:
    if context is None:
        return row
    out = dict(row)
    out.update(
        {
            "source_edge_ids": list(context.source_edge_ids),
            "source_edge_overlap_m": round(float(context.source_edge_overlap_m), 6),
            "source_edge_alignment_deg": (
                round(float(context.source_edge_alignment_deg), 6)
                if context.source_edge_alignment_deg is not None
                else None
            ),
            "source_edge_height_residual_m": (
                round(float(context.source_edge_height_residual_m), 6)
                if context.source_edge_height_residual_m is not None
                else None
            ),
            "building_part_ids": list(context.building_part_ids),
            "source_part_ids": list(context.source_part_ids),
            "crossed_part_ids": list(context.crossed_part_ids),
            "in_main_body": bool(context.in_main_body),
            "in_extension": bool(context.in_extension),
            "crosses_main_extension_boundary": bool(
                context.crosses_main_extension_boundary
            ),
        }
    )
    return out


def _attach_piece_anchor_fields(
    row: dict[str, Any],
    supports_by_key: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    # Per-piece anchor metrics: compute over this piece's own chain_ids rather
    # than over every chain the target plane-group aggregates. This is a
    # provenance fix — a ridge-eave plane group shares chain support across
    # its mirror-partner planes, so target-level medians mix residuals from
    # geometrically unrelated planes.
    target_id = str(row.get("target_element_id") or "")
    chain_ids = [str(cid) for cid in (row.get("chain_ids") or []) if cid]
    matched = [supports_by_key.get((target_id, cid)) for cid in chain_ids]
    matched = [s for s in matched if s is not None]
    if not matched:
        row["piece_anchor_chain_count"] = 0
        row["piece_anchor_height_residual_min_m"] = None
        return row
    residuals = [
        abs(float(s.height_residual_m))
        for s in matched
        if s.height_residual_m is not None
    ]
    row["piece_anchor_chain_count"] = len(matched)
    row["piece_anchor_height_residual_min_m"] = (
        round(min(residuals), 6) if residuals else None
    )
    return row


def score_buildings_v2(
    buildings: list[dict[str, Any]],
    roof_results: dict[str, Any],
    ridge_eave_scores_by_uuid: dict[str, dict[str, Any]] | None = None,
    v3_results_by_uuid: dict[str, dict[str, Any]] | None = None,
    config: ScorerV2Config = DEFAULT_CONFIG,
):
    per_target_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    relation_graph_by_uuid: dict[str, Any] = {}

    for building in buildings:
        uuid = str(building.get("uuid") or "")
        if not uuid:
            continue
        roof_result = roof_results.get(uuid)
        if roof_result is None:
            continue

        ridge_eave_entry = (ridge_eave_scores_by_uuid or {}).get(uuid)
        v3_building = (v3_results_by_uuid or {}).get(uuid)

        (
            _targets,
            split_targets,
            ridge_eave_target_diagnostics,
            ridge_eave_target_anchor_masks,
            story_extent_envelopes,
            story_gap_polygons,
        ) = collect_targets_for_building(
            uuid,
            building,
            roof_result,
            ridge_eave_entry,
            v3_building,
        )
        if not split_targets:
            continue

        evidence = collect_raw_evidence(
            building,
            roof_result,
            split_targets,
            ridge_eave_target_diagnostics,
        )
        chain_bundle = collect_chain_support(uuid, split_targets, evidence.raw_edges)

        extension_provenance: dict[str, dict[str, Any]] = {}
        pieces = build_split_pieces(
            split_targets,
            chain_bundle.eave_chains,
            chain_bundle.plane_chain_supports,
            story_extent_envelopes,
            story_gap_polygons,
            ridge_eave_target_anchor_masks,
            building,
            roof_result,
            ridge_eave_target_diagnostics,
            extension_provenance_out=extension_provenance,
        )
        piece_rows = split_piece_rows(
            pieces,
            split_targets,
            ridge_eave_target_diagnostics,
            chain_bundle.eave_chains,
            piece_metadata_by_piece_id=extension_provenance,
        )

        contexts_by_target = build_plane_contexts(
            split_targets,
            chain_bundle.plane_chain_supports,
            piece_rows,
            building,
            roof_result,
            ridge_eave_target_diagnostics,
        )
        graph = build_plane_relation_graph(
            split_targets, contexts_by_target, config.relation
        )
        relation_graph_by_uuid[uuid] = graph

        supports_by_key: dict[tuple[str, str], Any] = {}
        for support in chain_bundle.plane_chain_supports:
            if not support.supported:
                continue
            supports_by_key[(str(support.target_element_id), str(support.chain_id))] = (
                support
            )

        piece_rows = [
            _attach_context_fields(
                row, contexts_by_target.get(str(row.get("target_element_id") or ""))
            )
            for row in piece_rows
        ]
        piece_rows = [
            _attach_piece_anchor_fields(row, supports_by_key) for row in piece_rows
        ]
        piece_rows = annotate_split_rows_with_ownership(
            piece_rows, graph, building, roof_result
        )
        piece_rows = classify_split_piece_rows(piece_rows, graph, config.layer_policy)
        piece_rows = apply_global_xy_selection(
            piece_rows,
            building,
            config.global_selection,
            roof_result=roof_result,
            story_gap_polygons=story_gap_polygons,
            relation_graph=graph,
        )

        target_rows = score_targets(
            split_targets,
            evidence.raw_records,
            evidence.raw_edges,
            evidence.conflicts,
            chain_bundle.plane_chain_supports,
        )
        target_rows = [
            _attach_context_fields(
                row, contexts_by_target.get(str(row.get("element_id") or ""))
            )
            for row in target_rows
        ]

        seam_pieces, seam_metadata = compute_intersection_seam_pieces(
            split_targets,
            pieces,
            graph,
        )
        if seam_pieces:
            seam_rows = split_piece_rows(
                seam_pieces,
                split_targets,
                ridge_eave_target_diagnostics,
                chain_bundle.eave_chains,
                piece_metadata_by_piece_id=seam_metadata,
            )
            seam_rows = [
                _attach_context_fields(
                    row, contexts_by_target.get(str(row.get("target_element_id") or ""))
                )
                for row in seam_rows
            ]
            seam_rows = annotate_split_rows_with_ownership(
                seam_rows, graph, building, roof_result
            )
            piece_rows = piece_rows + seam_rows

        per_target_rows.extend(target_rows)
        split_rows.extend(piece_rows)

    output = build_run_output(per_target_rows, split_rows, relation_graph_by_uuid)
    return CorpusResult(
        per_target_rows=output.per_target_rows,
        per_story_rows=output.per_story_rows,
        split_piece_rows=output.split_piece_rows,
        summary=output.summary,
        relation_sidecar=output.relation_sidecar,
    )


def score_corpus_v2(
    buildings: list[dict[str, Any]],
    roof_results: dict[str, Any],
    ridge_eave_scores_by_uuid: dict[str, dict[str, Any]] | None = None,
    v3_results_by_uuid: dict[str, dict[str, Any]] | None = None,
    config: ScorerV2Config = DEFAULT_CONFIG,
) -> CorpusResult:
    return score_buildings_v2(
        buildings,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
        v3_results_by_uuid=v3_results_by_uuid,
        config=config,
    )


def score_building_v2(
    building: dict[str, Any],
    roof_results: dict[str, Any],
    ridge_eave_scores_by_uuid: dict[str, dict[str, Any]] | None = None,
    v3_results_by_uuid: dict[str, dict[str, Any]] | None = None,
    config: ScorerV2Config = DEFAULT_CONFIG,
) -> BuildingResult:
    uuid = str(building.get("uuid") or "")
    corpus = score_corpus_v2(
        [building],
        roof_results,
        ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
        v3_results_by_uuid=v3_results_by_uuid,
        config=config,
    )
    relation_sidecar = {
        "available": bool(corpus.relation_sidecar.get("available")),
        "buildings": {
            uuid: (corpus.relation_sidecar.get("buildings") or {}).get(uuid, {}),
        },
    }
    return BuildingResult(
        building_uuid=uuid,
        per_target_rows=[
            row for row in corpus.per_target_rows if str(row.get("uuid") or "") == uuid
        ],
        per_story_rows=[
            row for row in corpus.per_story_rows if str(row.get("uuid") or "") == uuid
        ],
        split_piece_rows=[
            row for row in corpus.split_piece_rows if str(row.get("uuid") or "") == uuid
        ],
        summary=corpus.summary,
        relation_sidecar=relation_sidecar,
    )
