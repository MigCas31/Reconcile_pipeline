#!/usr/bin/env python3
"""Phase 4 trainer CLI -- reads features_expanded.parquet, runs GroupKFold
CV for decision tree + LightGBM, writes model artifacts + report.

Usage:
    python scripts/train_models.py [--features PATH] [--out-dir DIR] [--folds N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from reconcile_v3.analysis.modelling import run_phase4

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        default=str(REPO / "artifacts" / "features_expanded.parquet"),
    )
    ap.add_argument("--out-dir", default=str(REPO / "artifacts"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(
            f"ERROR: {feat_path} not found; run scripts/analyze_labels.py first.",
            file=sys.stderr,
        )
        return 2

    df = pd.read_parquet(feat_path)
    print(f"loaded {len(df)} rows x {len(df.columns)} cols from {feat_path.name}")

    metrics = run_phase4(df, Path(args.out_dir), n_splits=args.folds)

    gate = metrics["gate_heuristic_plus_10pct_f1_accept"]
    verdict = "PASS" if gate["passes"] else "FAIL"
    print()
    print("=" * 60)
    print(f"Gate (GBM F1_accept >= heuristic + 0.10): {verdict}")
    print(f"  heuristic F1_accept   = {gate['heuristic_f1']:.3f}")
    print(f"  gbm calibrated F1_acc = {gate['gbm_f1']:.3f}")
    print(f"  delta                 = {gate['delta']:+.3f}")
    print("=" * 60)
    return 0 if gate["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
