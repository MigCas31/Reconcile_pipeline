from __future__ import annotations

from typing import Any

from shapely.ops import unary_union

from scripts import prototype_raw_ceiling_plane_scorer as legacy


def _source_part_ids_by_target(
    split_targets: list[legacy.TargetPlaneRecord],
    building: dict[str, Any],
    roof_result: dict[str, Any],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
    room_membership: dict[str, set[str]],
) -> dict[str, set[str]]:
    source_part_ids_by_target: dict[str, set[str]] = {}
    committed_targets = [
        target for target in split_targets if target.target_kind == "committed_oblique"
    ]
    committed_hypothesis_meta = legacy._committed_target_hypothesis_metadata(
        committed_targets,
        building,
        roof_result,
    )
    for target in split_targets:
        source_part_ids: set[str] = set()
        if target.target_kind == "ridge_eave_plane_group":
            diag = ridge_eave_target_diagnostics.get(target.element_id) or {}
            source_rooms = [
                str(room_id)
                for room_id in (diag.get("creator_source_room_ids") or [])
                if str(room_id)
            ]
            for room_id in source_rooms:
                source_part_ids.update(room_membership.get(room_id, set()))
        elif target.target_kind == "committed_oblique":
            meta = committed_hypothesis_meta.get(target.element_id) or {}
            source_part_ids.update(
                str(part_id)
                for part_id in (meta.get("hypothesis_part_ids") or [])
                if str(part_id)
            )
        if source_part_ids:
            source_part_ids_by_target[target.element_id] = source_part_ids
    return source_part_ids_by_target


def _clip_ridge_eave_supported_to_source_parts(
    split_pieces: list[legacy.TargetSplitPieceRecord],
    split_targets: list[legacy.TargetPlaneRecord],
    building: dict[str, Any],
    roof_result: dict[str, Any],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
) -> list[legacy.TargetSplitPieceRecord]:
    building_part_graph = roof_result.get("building_part_graph") or {}
    room_membership = legacy._room_part_membership(building_part_graph)
    source_part_ids_by_target = _source_part_ids_by_target(
        split_targets,
        building,
        roof_result,
        ridge_eave_target_diagnostics,
        room_membership,
    )
    if not source_part_ids_by_target:
        return split_pieces

    story_part_unions = legacy._story_part_region_unions(building, building_part_graph)
    all_story_part_unions = legacy._part_region_unions(building, building_part_graph)
    target_by_id = {target.element_id: target for target in split_targets}
    min_area = float(legacy.MIN_SPLIT_PIECE_AREA_M2)

    clipped: list[legacy.TargetSplitPieceRecord] = []
    for piece in split_pieces:
        if (
            piece.target_kind != "ridge_eave_plane_group"
            or piece.piece_role != "supported"
        ):
            clipped.append(piece)
            continue
        target = target_by_id.get(piece.target_element_id)
        source_part_ids = source_part_ids_by_target.get(piece.target_element_id, set())
        if target is None or not source_part_ids:
            clipped.append(piece)
            continue

        part_polys = [
            story_part_unions.get((piece.story, part_id))
            for part_id in sorted(source_part_ids)
        ]
        part_polys = [
            part_poly
            for part_poly in part_polys
            if part_poly is not None and not getattr(part_poly, "is_empty", True)
        ]
        has_source_story_part = bool(part_polys)
        if not part_polys:
            part_polys = [
                all_story_part_unions.get(part_id)
                for part_id in sorted(source_part_ids)
            ]
            part_polys = [
                part_poly
                for part_poly in part_polys
                if part_poly is not None and not getattr(part_poly, "is_empty", True)
            ]
        if not part_polys:
            clipped.append(piece)
            continue

        piece_poly = legacy._xz_polygon(piece.corners)
        if piece_poly is None or piece_poly.is_empty:
            clipped.append(piece)
            continue
        try:
            allowed_union = unary_union(part_polys)
            if not has_source_story_part:
                low_overlap_story_polys: list[Any] = []
                for (story, part_id), poly in story_part_unions.items():
                    if (
                        int(story) != int(piece.story)
                        or str(part_id) in source_part_ids
                    ):
                        continue
                    if poly is None or getattr(poly, "is_empty", True):
                        continue
                    try:
                        overlap_area = float(allowed_union.intersection(poly).area)
                        overlap_fraction = overlap_area / max(float(poly.area), 1e-9)
                    except Exception:
                        continue
                    # Fallback source masks can come from another story and overlap
                    # many same-mass rooms. Only subtract parts that barely overlap
                    # that source mask; those are typically extension intrusions.
                    if overlap_fraction <= 0.25:
                        low_overlap_story_polys.append(poly)
                if low_overlap_story_polys:
                    non_source_union = unary_union(low_overlap_story_polys)
                    if not getattr(non_source_union, "is_empty", True):
                        allowed_union = allowed_union.difference(non_source_union)
            clipped_geom = piece_poly.intersection(allowed_union)
        except Exception:
            clipped.append(piece)
            continue

        clipped_area = float(getattr(clipped_geom, "area", 0.0) or 0.0)
        original_area = max(float(piece_poly.area), 1e-9)
        if clipped_geom.is_empty or clipped_area < min_area:
            continue
        if clipped_area >= 0.995 * original_area:
            clipped.append(piece)
            continue

        clipped_polys = legacy._iter_polygons(clipped_geom)
        if not clipped_polys:
            clipped.append(piece)
            continue
        clipped_polys.sort(key=lambda poly: float(poly.area), reverse=True)
        for poly_index, poly in enumerate(clipped_polys):
            record = legacy._piece_records_from_polygon(
                target,
                poly,
                piece_id=(
                    piece.piece_id
                    if poly_index == 0
                    else f"{piece.piece_id}:source-part-clip:{poly_index}"
                ),
                piece_index=piece.piece_index + poly_index,
                piece_role=piece.piece_role,
                support_score=piece.support_score,
                chain_ids=piece.chain_ids,
            )
            if record is not None:
                clipped.append(record)

    return clipped


def build_split_pieces(
    split_targets: list[legacy.TargetPlaneRecord],
    eave_chains: list[legacy.EaveChainRecord],
    plane_chain_supports: list[legacy.PlaneEaveChainSupportRecord],
    story_extent_envelopes: dict[int, Any],
    story_gap_polygons: dict[int, list[Any]],
    target_segment_anchor_masks: dict[str, Any],
    building: dict[str, Any],
    roof_result: dict[str, Any],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
    apply_source_part_clipping: bool = True,
    extension_provenance_out: dict[str, dict[str, Any]] | None = None,
) -> list[legacy.TargetSplitPieceRecord]:
    building_part_graph = roof_result.get("building_part_graph") or {}
    room_membership = legacy._room_part_membership(building_part_graph)
    target_allowed_part_ids = _source_part_ids_by_target(
        split_targets,
        building,
        roof_result,
        ridge_eave_target_diagnostics,
        room_membership,
    )
    part_eave_envelopes = legacy._part_eave_envelopes(building, building_part_graph)
    story_eave_envelopes = legacy._story_eave_envelopes(building)

    split_pieces = legacy.build_plane_extent_split_pieces(
        split_targets,
        eave_chains,
        plane_chain_supports,
        story_extent_envelopes=story_extent_envelopes,
        story_gap_polygons=story_gap_polygons,
        target_segment_anchor_masks=target_segment_anchor_masks,
        part_eave_envelopes=part_eave_envelopes,
        target_allowed_part_ids=target_allowed_part_ids,
        story_eave_envelopes=story_eave_envelopes,
        extension_provenance_out=extension_provenance_out,
    )
    split_pieces = legacy.trim_ridge_eave_supported_pieces_to_chain_run_bands(
        split_targets,
        eave_chains,
        split_pieces,
    )
    if apply_source_part_clipping:
        split_pieces = _clip_ridge_eave_supported_to_source_parts(
            split_pieces,
            split_targets,
            building,
            roof_result,
            ridge_eave_target_diagnostics,
        )
    return split_pieces


def split_piece_rows(
    split_pieces: list[legacy.TargetSplitPieceRecord],
    split_targets: list[legacy.TargetPlaneRecord],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
    eave_chains: list[legacy.EaveChainRecord] | None = None,
    piece_metadata_by_piece_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    target_by_id = {target.element_id: target for target in split_targets}
    chain_by_id = {str(chain.chain_id): chain for chain in (eave_chains or [])}
    extra_metadata = piece_metadata_by_piece_id or {}
    rows: list[dict[str, Any]] = []
    for piece in split_pieces:
        target = target_by_id.get(piece.target_element_id)
        if target is None:
            continue
        target_plane_coeffs = legacy._target_plane_coeffs(target)
        source_edge_segments_xz: list[list[list[float]]] = []
        for chain_id in piece.chain_ids:
            chain = chain_by_id.get(str(chain_id))
            if chain is None:
                continue
            start_xz = chain.start_xz
            end_xz = chain.end_xz
            if start_xz is None or end_xz is None:
                continue
            source_edge_segments_xz.append(
                [
                    [round(float(start_xz[0]), 6), round(float(start_xz[1]), 6)],
                    [round(float(end_xz[0]), 6), round(float(end_xz[1]), 6)],
                ]
            )
        rows.append(
            {
                "uuid": piece.uuid,
                "story": piece.story,
                "target_element_id": piece.target_element_id,
                "target_kind": piece.target_kind,
                "piece_id": piece.piece_id,
                "piece_index": piece.piece_index,
                "piece_role": piece.piece_role,
                "area_xz_m2": round(piece.area_xz_m2, 6),
                "target_azimuth_deg": round(target.azimuth_deg, 6),
                "target_inclination_deg": round(target.inclination_deg, 6),
                "support_score": round(piece.support_score, 6)
                if piece.support_score is not None
                else None,
                "chain_ids": list(piece.chain_ids),
                "target_plane_point": [
                    round(float(target.plane_point[0]), 6),
                    round(float(target.plane_point[1]), 6),
                    round(float(target.plane_point[2]), 6),
                ]
                if target.plane_point is not None
                else None,
                "target_normal": [
                    round(float(target.normal[0]), 9),
                    round(float(target.normal[1]), 9),
                    round(float(target.normal[2]), 9),
                ]
                if target.normal is not None
                else None,
                "target_plane_coeffs": [
                    round(float(value), 9) for value in target_plane_coeffs
                ]
                if target_plane_coeffs is not None
                else None,
                "source_edge_segments_xz": source_edge_segments_xz,
                "corners": legacy._rounded_loop_3d(piece.corners),
                "holes": legacy._serialized_piece_holes(piece.holes),
                **ridge_eave_target_diagnostics.get(piece.target_element_id, {}),
                **extra_metadata.get(piece.piece_id, {}),
            }
        )
    return rows
