"""Phase 6 CLI: score every merged roof segment in reconcile_v3_results.json.

Streams the 1.8 GB results file one building at a time, scores each
``merged_roof_segment`` via the depth-4 reject tree + the calibrated GBM
ensemble, and writes a mirror JSON with ``score`` and ``autonomy_label``
added to every segment. Only those two fields are added; everything else
is copied through verbatim.

Thresholds follow the plan's starting values — θ_low=0.10, θ_high=0.90 —
so the middle 80 % still reaches the human review queue.

Usage::

    python -m reconcile_v3.autonomy.score_results \
        --input reconcile/reconcile_v3_results.json \
        --output reconcile/reconcile_v3_results_scored.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import ijson
import numpy as np
import pandas as pd

from reconcile_v3.analysis.aux_context import load_or_build as load_or_build_aux_context
from reconcile_v3.analysis.building_features import load_building_features
from reconcile_v3.analysis.source_wall_index import build_wall_index
from reconcile_v3.analysis.v3_context import load_or_build

from .inference_features import build_record, compute_features


def _vec(row: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    """Single-row DataFrame in the training-time column order, float32."""
    return pd.DataFrame(
        [[row.get(c, np.nan) for c in feature_names]],
        columns=feature_names,
    ).astype("float32")


def _score_one(
    row: dict[str, Any],
    *,
    tree_bundle: dict,
    gbm_bundle: dict,
    theta_low: float,
    theta_high: float,
) -> tuple[float, str, int, bool]:
    """Return (score, autonomy_label, tree_leaf_id, rule_fires)."""
    # ---- rule tree (hard auto-reject gate) ----
    tree_names = tree_bundle["feature_names"]
    medians = tree_bundle["medians"]
    approved = set(tree_bundle["approved_reject_leaves"])
    tree = tree_bundle["tree"]
    X_tree = _vec(row, tree_names)
    for c in tree_names:
        med = medians.get(c)
        if med is not None:
            X_tree[c] = X_tree[c].fillna(np.float32(med))
    X_tree = X_tree.fillna(0.0)  # median might itself be NaN for rare cols
    leaf = int(tree.apply(X_tree)[0])
    rule_fires = leaf in approved

    # ---- GBM ensemble + isotonic calibrator ----
    gbm_names = gbm_bundle["feature_names"]
    X_gbm = _vec(row, gbm_names)
    probs = np.mean(
        [m.predict_proba(X_gbm)[:, 1] for m in gbm_bundle["models"]], axis=0
    )
    iso = gbm_bundle.get("isotonic_calibrator")
    score = float(iso.transform(probs)[0]) if iso is not None else float(probs[0])

    if rule_fires or score < theta_low:
        label = "auto_reject"
    elif score > theta_high:
        label = "auto_accept"
    else:
        label = "review"
    return score, label, leaf, rule_fires


def score_results(
    input_path: Path,
    output_path: Path,
    *,
    tree_pkl: Path,
    gbm_pkl: Path,
    buildings_path: Path,
    v3_cache_path: Path,
    aux_cache_path: Path | None = None,
    theta_low: float = 0.10,
    theta_high: float = 0.90,
) -> dict[str, int]:
    """Stream-score every merged segment; write a mirror JSON."""
    with tree_pkl.open("rb") as f:
        tree_bundle = pickle.load(f)
    with gbm_pkl.open("rb") as f:
        gbm_bundle = pickle.load(f)

    # Load per-building context once; the streaming loop just looks up by uuid.
    bld = load_building_features(buildings_path)
    walls = build_wall_index(buildings_path)
    ctx = load_or_build(input_path, v3_cache_path)
    aux_ctx: dict[str, dict] = {}
    if aux_cache_path is not None:
        aux_ctx = load_or_build_aux_context(
            aux_cache_path,
            only_uuids=set(ctx.keys()),
        )

    hist = {
        "auto_accept": 0,
        "auto_reject": 0,
        "review": 0,
        "segments": 0,
        "buildings": 0,
    }
    out_f = output_path.open("w")
    out_f.write("[")
    first = True
    with input_path.open("rb") as in_f:
        for building in ijson.items(in_f, "item", use_float=True):
            uuid = building.get("building_uuid")
            bf = bld.get(uuid)
            wi = walls.get(uuid)
            bc = ctx.get(uuid)
            ac = aux_ctx.get(uuid)
            for seg in building.get("merged_roof_segments") or []:
                rec = build_record(seg, uuid)
                feats = compute_features(
                    rec,
                    building_features=bf,
                    wall_index=wi,
                    building_context=bc,
                    aux_context=ac,
                )
                score, label, leaf, fires = _score_one(
                    feats,
                    tree_bundle=tree_bundle,
                    gbm_bundle=gbm_bundle,
                    theta_low=theta_low,
                    theta_high=theta_high,
                )
                seg["score"] = score
                seg["autonomy_label"] = label
                seg["rule_leaf_id"] = leaf
                seg["rule_fires"] = bool(fires)
                hist[label] += 1
                hist["segments"] += 1
            hist["buildings"] += 1
            if not first:
                out_f.write(",")
            first = False
            json.dump(building, out_f)
            if hist["buildings"] % 20 == 0:
                print(
                    f"  scored {hist['buildings']} buildings / {hist['segments']} "
                    f"segments "
                    f"(review={hist['review']} reject={hist['auto_reject']} "
                    f"accept={hist['auto_accept']})",
                    flush=True,
                )
    out_f.write("]")
    out_f.close()
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="reconcile/reconcile_v3_results.json")
    ap.add_argument("--output", default="reconcile/reconcile_v3_results_scored.json")
    ap.add_argument("--tree-pkl", default="artifacts/rule_tree.pkl")
    ap.add_argument("--gbm-pkl", default="artifacts/model_gbm.pkl")
    ap.add_argument("--buildings", default="reconcile/buildings_3d.json")
    ap.add_argument("--v3-cache", default="artifacts/v3_context.json")
    ap.add_argument("--aux-cache", default=None)
    ap.add_argument("--theta-low", type=float, default=0.10)
    ap.add_argument("--theta-high", type=float, default=0.90)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found.", file=sys.stderr)
        return 2
    print(f"scoring {in_path} → {out_path}")
    print(f"  θ_low={args.theta_low}  θ_high={args.theta_high}")
    hist = score_results(
        in_path,
        out_path,
        tree_pkl=Path(args.tree_pkl),
        gbm_pkl=Path(args.gbm_pkl),
        buildings_path=Path(args.buildings),
        v3_cache_path=Path(args.v3_cache),
        aux_cache_path=Path(args.aux_cache) if args.aux_cache else None,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
    )
    total = hist["segments"] or 1
    print(
        f"done: {hist['buildings']} buildings, {hist['segments']} segments "
        f"→ accept {hist['auto_accept']} ({100 * hist['auto_accept'] / total:.1f}%) "
        f"reject {hist['auto_reject']} ({100 * hist['auto_reject'] / total:.1f}%) "
        f"review {hist['review']} ({100 * hist['review'] / total:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
