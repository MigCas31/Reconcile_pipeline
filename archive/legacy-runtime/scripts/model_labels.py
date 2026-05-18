#!/usr/bin/env python3
"""Phase 3+: feature ranking and (later) model training on the label set.

Subcommands:
    stats   -- per-feature effect size, MI, and percentile splits.
    model   -- (reserved) Phase 4 decision tree + LightGBM.
    rules   -- (reserved) Phase 5 rule extraction from tree leaves.
    audit   -- (reserved) Phase 6 disagreement audits.

The ``stats`` command reads ``artifacts/features_expanded.parquet`` (from
scripts/analyze_labels.py) and writes:

    artifacts/feature_ranking.csv
    artifacts/plots/<rank>_<feature>.png   (top-20 by effect size)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile_v3.analysis import ranking

REPO = Path(__file__).resolve().parent.parent


def _cmd_stats(args) -> int:
    import pandas as pd

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    df = pd.read_parquet(features_path)
    print(f"loaded {features_path} -- {len(df)} rows x {len(df.columns)} cols")

    rank = ranking.rank_features(df)
    ranking_path = out_dir / "feature_ranking.csv"
    rank.to_csv(ranking_path, index=False)
    print(f"wrote {ranking_path} -- {len(rank)} features ranked")

    top = rank.head(args.top_n)
    print("\n=== top-20 features by |effect size| ===")
    cols_to_show = [
        "rank",
        "feature",
        "kind",
        "cohens_d",
        "cramers_v",
        "mutual_info",
        "accept_mean",
        "reject_mean",
        "n_null",
    ]
    print(top[cols_to_show].to_string(index=False))

    if not args.no_plots:
        paths = ranking.plot_top_features(df, rank, plots_dir, top_n=args.top_n)
        print(f"\nwrote {len(paths)} plots to {plots_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    stats = sub.add_parser("stats", help="Phase 3 descriptive stats + effect sizes")
    stats.add_argument(
        "--features",
        default=str(REPO / "artifacts" / "features_expanded.parquet"),
    )
    stats.add_argument("--out-dir", default=str(REPO / "artifacts"))
    stats.add_argument("--top-n", type=int, default=20)
    stats.add_argument("--no-plots", action="store_true")
    stats.set_defaults(func=_cmd_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
