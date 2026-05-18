"""Intersection-seam pieces for `local_competitor` plane pairs.

When two oblique target planes have perpendicular azimuths but overlap in XZ
(an L-shape, T-shape, or any building where two wings meet at a hip), the
geometric plane-plane intersection used by v3's `_half_plane_for_opposing` is
brittle: it depends only on the fitted plane equations and ignores which part
of each plane was actually scanned.

This module derives per-side seam pieces from raw scan evidence. For each
`local_competitor` pair, each side's `intersection_seam` polygon is that
target's plane footprint minus the partner's evidence-claimed region. The
seam emerges from the difference: in the contested overlap area, A wins
where B's raw evidence is silent and vice versa.

Output is a new `piece_role="intersection_seam"`. Existing `supported` /
`residual` pieces are untouched so the viewer can show both side-by-side.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from scripts import prototype_raw_ceiling_plane_scorer as legacy

from .models import RelationGraph

MIN_EVIDENCE_AREA_M2 = 0.5
MIN_SEAM_PIECE_AREA_M2 = float(legacy.MIN_SPLIT_PIECE_AREA_M2)

# Restricted to oblique target kinds. Ridge-eave plane groups are mirror-pair
# aggregates whose downstream selection logic (final_layer_reason, layer
# policy) expects a single role per piece — emitting intersection_seam rows
# alongside their supported pieces would conflate the two semantics.
_ELIGIBLE_TARGET_KINDS: frozenset[str] = frozenset(
    {
        "committed_oblique",
        "candidate_oblique",
    }
)


def _supported_union_by_target(
    pieces: list[legacy.TargetSplitPieceRecord],
) -> dict[str, Any]:
    polys_by_target: dict[str, list[Polygon]] = {}
    for piece in pieces:
        if piece.piece_role != "supported":
            continue
        poly = legacy._xz_polygon(piece.corners)
        if poly is None or poly.is_empty:
            continue
        polys_by_target.setdefault(piece.target_element_id, []).append(poly)
    out: dict[str, Any] = {}
    for target_id, polys in polys_by_target.items():
        try:
            out[target_id] = unary_union(polys)
        except Exception:
            continue
    return out


def _extended_poly_by_target(
    pieces: list[legacy.TargetSplitPieceRecord],
) -> dict[str, Any]:
    """Per-target XZ polygon = union of ``supported`` + ``residual`` pieces.

    Equals the upstream ``support_base_poly`` (after eave-widening) up to
    polygon-emission rounding. Used as each side's "effective polygon"
    when computing intersection-seam pieces — without this, seams are
    bounded by the raw fitted ``target.poly_xz`` and stop short of the
    eave on buildings where the supported pieces have been widened.
    """

    polys_by_target: dict[str, list[Polygon]] = {}
    for piece in pieces:
        if piece.piece_role not in ("supported", "residual"):
            continue
        poly = legacy._xz_polygon(piece.corners)
        if poly is None or poly.is_empty:
            continue
        polys_by_target.setdefault(piece.target_element_id, []).append(poly)
    out: dict[str, Any] = {}
    for target_id, polys in polys_by_target.items():
        try:
            out[target_id] = unary_union(polys)
        except Exception:
            continue
    return out


def _local_competitor_pairs(
    graph: RelationGraph,
) -> Iterable[tuple[str, str, float, float]]:
    for relation in graph.relations:
        if relation.relation_kind != "local_competitor":
            continue
        if not relation.same_story:
            continue
        yield (
            relation.a_target_id,
            relation.b_target_id,
            float(relation.overlap_m2),
            float(relation.azimuth_delta_deg or 0.0),
        )


def _seam_piece(
    target: legacy.TargetPlaneRecord,
    poly: Polygon,
    *,
    pair_index: int,
    side_label: str,
    partner_target_id: str,
) -> legacy.TargetSplitPieceRecord | None:
    return legacy._piece_records_from_polygon(
        target,
        poly,
        piece_id=f"intersection-seam:{pair_index}:{side_label}:{partner_target_id}",
        piece_index=0,
        piece_role="intersection_seam",
        support_score=None,
        chain_ids=tuple(),
    )


def _seam_pieces_for_pair(
    a: legacy.TargetPlaneRecord,
    b: legacy.TargetPlaneRecord,
    evidence_a: Any,
    evidence_b: Any,
    extended_a: Any,
    extended_b: Any,
    *,
    pair_index: int,
) -> tuple[
    list[legacy.TargetSplitPieceRecord],
    dict[str, dict[str, Any]],
]:
    pieces: list[legacy.TargetSplitPieceRecord] = []
    metadata_by_piece_id: dict[str, dict[str, Any]] = {}

    # Use the union of ``poly_xz`` and each side's extended polygon
    # (supported + residual = widened ``support_base_poly`` from the
    # upstream splitter). Union-not-replace keeps the seam at least as
    # large as the original fitted polygon — important when a side has
    # only supported pieces (no residual) yet, e.g. in unit tests, so we
    # don't accidentally shrink below ``poly_xz``. In production where
    # widening fired, ``support_base_poly`` is strictly larger than
    # ``poly_xz`` and the union is just the widened polygon.
    def _seam_base(side, extended):
        if extended is None or getattr(extended, "is_empty", True):
            return side.poly_xz
        try:
            return unary_union([side.poly_xz, extended])
        except Exception:
            return side.poly_xz

    base_a = _seam_base(a, extended_a)
    base_b = _seam_base(b, extended_b)
    try:
        side_a_geom = base_a.difference(evidence_b)
        side_b_geom = base_b.difference(evidence_a)
    except Exception:
        return pieces, metadata_by_piece_id

    overlap_area_m2 = 0.0
    try:
        overlap_geom = base_a.intersection(base_b)
        overlap_area_m2 = float(getattr(overlap_geom, "area", 0.0) or 0.0)
    except Exception:
        overlap_geom = None

    for side_target, side_geom, side_label, partner_id, partner_evidence in (
        (a, side_a_geom, "a", b.element_id, evidence_b),
        (b, side_b_geom, "b", a.element_id, evidence_a),
    ):
        for poly_index, poly in enumerate(legacy._iter_polygons(side_geom)):
            if float(poly.area) < MIN_SEAM_PIECE_AREA_M2:
                continue
            piece = _seam_piece(
                side_target,
                poly,
                pair_index=pair_index,
                side_label=f"{side_label}:{poly_index}",
                partner_target_id=partner_id,
            )
            if piece is None:
                continue
            disputed_area = 0.0
            try:
                if overlap_geom is not None and not overlap_geom.is_empty:
                    disputed_area = float(poly.intersection(overlap_geom).area)
            except Exception:
                disputed_area = 0.0
            evidence_partner_area = 0.0
            try:
                evidence_partner_area = float(partner_evidence.area)
            except Exception:
                evidence_partner_area = 0.0
            metadata_by_piece_id[piece.piece_id] = {
                "pair_partner_target_id": partner_id,
                "pair_overlap_m2": round(overlap_area_m2, 6),
                "pair_disputed_overlap_in_piece_m2": round(disputed_area, 6),
                "pair_partner_evidence_area_m2": round(evidence_partner_area, 6),
            }
            pieces.append(piece)
    return pieces, metadata_by_piece_id


def compute_intersection_seam_pieces(
    split_targets: list[legacy.TargetPlaneRecord],
    split_pieces: list[legacy.TargetSplitPieceRecord],
    relation_graph: RelationGraph,
) -> tuple[list[legacy.TargetSplitPieceRecord], dict[str, dict[str, Any]]]:
    """Emit `intersection_seam` pieces for each `local_competitor` pair.

    Returns the list of new pieces plus a per-piece metadata dict keyed by
    `piece_id` so the row builder can attach pair-level fields (partner id,
    disputed area, etc.) without polluting `TargetSplitPieceRecord`.
    """
    target_by_id = {target.element_id: target for target in split_targets}
    evidence_by_target = _supported_union_by_target(split_pieces)
    extended_by_target = _extended_poly_by_target(split_pieces)
    new_pieces: list[legacy.TargetSplitPieceRecord] = []
    metadata_by_piece_id: dict[str, dict[str, Any]] = {}

    for pair_index, (a_id, b_id, _overlap_m2, _az_delta_deg) in enumerate(
        _local_competitor_pairs(relation_graph)
    ):
        a = target_by_id.get(a_id)
        b = target_by_id.get(b_id)
        if a is None or b is None:
            continue
        if (
            a.target_kind not in _ELIGIBLE_TARGET_KINDS
            or b.target_kind not in _ELIGIBLE_TARGET_KINDS
        ):
            continue
        evidence_a = evidence_by_target.get(a_id)
        evidence_b = evidence_by_target.get(b_id)
        if evidence_a is None or evidence_b is None:
            continue
        if float(evidence_a.area) < MIN_EVIDENCE_AREA_M2:
            continue
        if float(evidence_b.area) < MIN_EVIDENCE_AREA_M2:
            continue
        extended_a = extended_by_target.get(a_id)
        extended_b = extended_by_target.get(b_id)
        pair_pieces, pair_metadata = _seam_pieces_for_pair(
            a,
            b,
            evidence_a,
            evidence_b,
            extended_a,
            extended_b,
            pair_index=pair_index,
        )
        new_pieces.extend(pair_pieces)
        metadata_by_piece_id.update(pair_metadata)

    return new_pieces, metadata_by_piece_id
