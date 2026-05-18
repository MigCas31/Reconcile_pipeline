#!/usr/bin/env python3
"""Phase 1 + 2 orchestrator: join labels and expand features.

Reads:
    .context/v3_roof_proposal_labels.jsonl
    .context/v3_roof_proposal_splits.jsonl

Writes:
    artifacts/labels_joined.parquet
    artifacts/features_expanded.parquet

Usage:
    python scripts/analyze_labels.py [--labels PATH] [--splits PATH]
                                     [--out-dir DIR] [--no-parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile_v3.analysis import (
    advanced_features as advf,
)
from reconcile_v3.analysis import (
    aux_context,
    building_features,
    derived_features,
    feature_expansion,
    join,
    offline_post_features,
    source_wall_index,
    v3_context,
)
from reconcile_v3.analysis import (
    context_features as ctxf,
)
from reconcile_v3.analysis import (
    exhaustive_features as exf,
)

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels",
        default=str(REPO / ".context" / "v3_roof_proposal_labels.jsonl"),
    )
    ap.add_argument(
        "--splits",
        default=str(REPO / ".context" / "v3_roof_proposal_splits.jsonl"),
    )
    ap.add_argument(
        "--buildings",
        default=str(REPO / "reconcile" / "buildings_3d.json"),
        help="Per-building context (footprint, stories, ...).",
    )
    ap.add_argument(
        "--v3-results",
        default=str(REPO / "reconcile" / "reconcile_v3_results.json"),
        help="Full V3 results (streamed once into a compact cache).",
    )
    ap.add_argument(
        "--v3-cache",
        default=str(REPO / "artifacts" / "v3_context.json"),
        help="Compact per-UUID context cache (regenerated when missing).",
    )
    ap.add_argument(
        "--rebuild-v3-cache",
        action="store_true",
        help="Re-stream the 2GB V3 results file even if a cache exists.",
    )
    ap.add_argument(
        "--skip-v3-context",
        action="store_true",
        help="Skip Band 3 context features entirely.",
    )
    ap.add_argument(
        "--aux-cache",
        default=str(REPO / "artifacts" / "aux_context.json"),
        help="Optional ontology/V2/cross-modal cache.",
    )
    ap.add_argument(
        "--with-aux-context",
        action="store_true",
        help="Build ontology / V2 / cross-modal context for labeled UUIDs.",
    )
    ap.add_argument(
        "--rebuild-aux-cache",
        action="store_true",
        help="Force rebuilding the ontology / V2 auxiliary cache.",
    )
    ap.add_argument(
        "--skip-advanced",
        action="store_true",
        help="Skip Band 2 source-wall + Band 4 advanced features.",
    )
    ap.add_argument(
        "--oof-predictions",
        default=str(REPO / "artifacts" / "oof_predictions.parquet"),
        help="OOF predictions used for offline model/meta features.",
    )
    ap.add_argument(
        "--gbm-pkl",
        default=str(REPO / "artifacts" / "model_gbm.pkl"),
        help="GBM bundle used for offline ensemble/SHAP features.",
    )
    ap.add_argument(
        "--reference-features",
        default=str(REPO / "artifacts" / "features_expanded.parquet"),
        help="Reference full-corpus feature parquet used to align OOF predictions by "
        "proposal id.",
    )
    ap.add_argument("--out-dir", default=str(REPO / "artifacts"))
    ap.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip parquet writes (still prints the resolution report).",
    )
    args = ap.parse_args()

    labels_path = Path(args.labels)
    splits_path = Path(args.splits)
    buildings_path = Path(args.buildings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = join.load_labels(labels_path, splits_path)
    report = join.resolution_report(records)
    print("=== Phase 1 -- join resolution ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    bld_index = building_features.load_building_features(buildings_path)
    print(
        f"=== Building context -- {len(bld_index)} buildings indexed "
        f"from {buildings_path.name} ==="
    )

    ctx_index: dict[str, dict] = {}
    if not args.skip_v3_context:
        labeled_uuids = {r["building_uuid"] for r in records}
        ctx_index = v3_context.load_or_build(
            Path(args.v3_results),
            Path(args.v3_cache),
            only_uuids=labeled_uuids,
            rebuild=args.rebuild_v3_cache,
        )
        print(
            f"=== V3 context -- {len(ctx_index)} buildings in "
            f"{Path(args.v3_cache).name} ==="
        )

    wall_idx: dict[str, dict] = {}
    if not args.skip_advanced:
        wall_idx = source_wall_index.build_wall_index(buildings_path)
        n_walls = sum(len(v) for v in wall_idx.values())
        print(f"=== Wall index -- {n_walls} walls across {len(wall_idx)} buildings ===")

    aux_idx: dict[str, dict] = {}
    if args.with_aux_context:
        labeled_uuids = {r["building_uuid"] for r in records}
        aux_idx = aux_context.load_or_build(
            Path(args.aux_cache),
            only_uuids=labeled_uuids,
            rebuild=args.rebuild_aux_cache,
        )
        print(
            f"=== Aux context -- {len(aux_idx)} buildings in "
            f"{Path(args.aux_cache).name} ==="
        )

    rows = feature_expansion.expand_all(records)
    if bld_index:
        any_bld_feats = next(iter(bld_index.values()))
        null_bld = {k: None for k in any_bld_feats}
        for row in rows:
            row.update(bld_index.get(row.get("building_uuid"), null_bld))

    # Band 3 -- per-record context features (parts / kneewalls / dormers / survival).
    if ctx_index:
        for row, rec in zip(rows, records, strict=False):
            row.update(ctxf.context_features(rec, ctx_index.get(row["building_uuid"])))

    # Bands 2+4 -- source-wall aggregates, shape descriptors, normal stats, IoU.
    if not args.skip_advanced:
        for row, rec in zip(rows, records, strict=False):
            row.update(
                advf.advanced_features(
                    rec,
                    wall_index=wall_idx.get(row["building_uuid"]),
                    building_context=ctx_index.get(row["building_uuid"])
                    if ctx_index
                    else None,
                )
            )

    for row, rec in zip(rows, records, strict=False):
        row.update(
            exf.exhaustive_features(
                rec,
                row,
                building_context=ctx_index.get(row["building_uuid"])
                if ctx_index
                else None,
                wall_index=wall_idx.get(row["building_uuid"]) if wall_idx else None,
                aux_context=aux_idx.get(row["building_uuid"]) if aux_idx else None,
                repo_root=REPO,
            )
        )

    # Tier A -- derived/combination features (catalogue v3 §N.1). Runs last so
    # every input column is already populated on the row.
    for row in rows:
        row.update(derived_features.derived_features(row))

    print(
        f"=== Phase 2 -- feature expansion: {len(rows)} rows x "
        f"{len(rows[0]) if rows else 0} features ==="
    )
    if rows:
        feature_names = sorted(rows[0].keys())
        print(f"sample features: {feature_names[:8]} ... ({len(feature_names)} total)")

    if args.no_parquet:
        return 0

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        print(f"pandas unavailable ({exc}); skipping parquet writes.", file=sys.stderr)
        return 0

    # Phase 1 -- preserve the raw record alongside the label for downstream
    # consumers; serialize nested structures as JSON strings so parquet is happy.
    joined_df = pd.DataFrame(
        [
            {
                "building_uuid": r["building_uuid"],
                "proposal_id": r["proposal_id"],
                "label": r.get("label"),
                "heuristic_label": r.get("heuristic_label"),
                "ts": r.get("ts"),
                "is_split_child": r.get("is_split_child"),
                "parent_proposal_id": r.get("parent_proposal_id"),
                "record_json": json.dumps(r, default=str),
            }
            for r in records
        ]
    )
    joined_out = out_dir / "labels_joined.parquet"
    joined_df.to_parquet(joined_out, index=False)
    print(f"wrote {joined_out} ({len(joined_df)} rows)")

    feat_df = pd.DataFrame(rows)
    feat_df = offline_post_features.augment_dataframe(
        feat_df,
        oof_predictions_path=args.oof_predictions,
        gbm_pkl_path=args.gbm_pkl,
        reference_features_path=args.reference_features,
    )
    feat_out = out_dir / "features_expanded.parquet"
    feat_df.to_parquet(feat_out, index=False)
    print(f"wrote {feat_out} ({len(feat_df)} rows x {len(feat_df.columns)} cols)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
