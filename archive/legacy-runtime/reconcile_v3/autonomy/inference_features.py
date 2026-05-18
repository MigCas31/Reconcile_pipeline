"""Build the feature record for a merged roof segment at inference time.

Training read feature dicts out of label records stored by the viewer. At
inference, we walk ``reconcile_v3_results.json`` directly — every field
the label-time path read is already present on the ``V3MergedRoofSegment``
dict (``corners``, ``member_snapshots``, ``cluster_params``,
``building_boundary_xz``, ``opposing_planes``, etc). The only viewer-only
concept is ``side_pieces``; in 98 % of training labels it's a single
polygon whose area matches the segment's own area, so we reproduce that
dominant case here. The remaining three ``side_piece_*`` features carry
almost no signal in the trained GBM (they're below the top-40 importance).
"""

from __future__ import annotations

from typing import Any

from reconcile_v3.analysis import (
    advanced_features as advf,
)
from reconcile_v3.analysis import (
    context_features as ctxf,
)
from reconcile_v3.analysis import (
    exhaustive_features as exf,
)
from reconcile_v3.analysis import feature_expansion


def build_record(segment: dict, building_uuid: str) -> dict:
    """Shape a ``V3MergedRoofSegment`` dict for ``feature_expansion.expand()``.

    The returned dict mirrors the label-record schema written by
    ``viewer_server._handle_roof_proposal_label_post`` so the same
    downstream feature computers work unchanged.
    """
    # Segment.id already embeds the full element-locator prefix
    # ("<uuid>::v3-merged-roof-segment::<inner>") — do not re-prefix.
    proposal_id = segment.get("id") or ""
    member_snapshots = list(segment.get("member_snapshots") or [])
    corners = segment.get("corners") or []
    # Reproduce the dominant training-time side_pieces layout (one polygon
    # with the segment's own corners, no parent).
    side_pieces = (
        [{"parent_proposal_id": None, "piece_corners_xyz": [list(c) for c in corners]}]
        if corners
        else []
    )
    return {
        "building_uuid": building_uuid,
        "proposal_id": proposal_id,
        "cluster_canonical_id": segment.get("cluster_canonical_id"),
        "label": None,
        "heuristic_label": segment.get("heuristic_label"),
        "labeler": None,
        "ts": None,
        "merge_mode": True,
        "part_index": 0,
        "part_count": 1,
        "is_split_child": False,
        "parent_proposal_id": None,
        "kind": "v3-merged-roof-segment",
        "features_snapshot": dict(segment.get("features") or {}),
        "merged_plane": list(segment.get("merged_plane") or []),
        "segment_corners_xyz": [list(c) for c in corners],
        "building_boundary_xz": [
            list(c) for c in (segment.get("building_boundary_xz") or [])
        ],
        "opposing_planes": [list(p) for p in (segment.get("opposing_planes") or [])],
        "opposing_cluster_canonicals": list(
            segment.get("opposing_cluster_canonicals") or []
        ),
        "room_boundary_refs": list(segment.get("room_boundary_refs") or []),
        "member_proposal_ids": list(segment.get("member_proposal_ids") or []),
        "cluster_members": member_snapshots,
        "cluster_params": dict(segment.get("cluster_params") or {}),
        "side_pieces": side_pieces,
    }


def compute_features(
    record: dict,
    *,
    building_features: dict | None,
    wall_index: dict[str, dict] | None,
    building_context: dict | None,
    aux_context: dict | None = None,
) -> dict[str, Any]:
    """Run the four feature stages that ``analyze_labels.py`` runs, in order.

    Mirrors the sequence in ``scripts/analyze_labels.py`` so the output is
    a bit-for-bit match with ``artifacts/features_expanded.parquet`` rows.
    """
    from reconcile_v3.analysis import derived_features as derivedf

    row = feature_expansion.expand(record)
    if building_features is not None:
        row.update(building_features)
    row.update(ctxf.context_features(record, building_context))
    row.update(
        advf.advanced_features(
            record,
            wall_index=wall_index,
            building_context=building_context,
        )
    )
    row.update(
        exf.exhaustive_features(
            record,
            row,
            building_context=building_context,
            wall_index=wall_index,
            aux_context=aux_context,
        )
    )
    row.update(derivedf.derived_features(row))
    return row
