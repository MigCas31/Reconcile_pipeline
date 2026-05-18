#!/usr/bin/env python
"""Fit selector-v2 prior weights on a deterministic held-out payload subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "reconcile_tiers").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconcile_tiers.polyhedron.prior_fitting import (  # noqa: E402
    default_weight_grid,
    run_weight_sweep,
    select_holdout_payloads,
    sweep_report,
    write_fitted_priors,
    write_sweep_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".context/polyhedron-prior-fit/report.json"),
    )
    parser.add_argument("--max-buildings", type=int, default=30)
    parser.add_argument("--corner-tol", type=float, default=0.02)
    parser.add_argument("--time-budget-seconds", type=float, default=0.5)
    parser.add_argument("--max-intersections", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument(
        "--write-priors",
        type=Path,
        default=None,
        help="Optionally rewrite the given priors.py with the selected weights.",
    )
    args = parser.parse_args(argv)

    payload_paths = select_holdout_payloads(
        args.pipeline_dir,
        max_buildings=args.max_buildings,
    )
    results = run_weight_sweep(
        payload_paths,
        default_weight_grid(),
        corner_tol=args.corner_tol,
        time_budget_seconds=args.time_budget_seconds,
        max_intersections=args.max_intersections,
        max_candidates=args.max_candidates,
    )
    report = sweep_report(
        pipeline_dir=args.pipeline_dir,
        payload_paths=payload_paths,
        results=results,
    )
    write_sweep_report(args.out, report)
    if args.write_priors is not None:
        write_fitted_priors(args.write_priors, report)
    print(json.dumps(report["selected"], indent=2, sort_keys=True))
    print(f"Wrote report: {args.out}")
    if args.write_priors is not None:
        print(f"Updated priors: {args.write_priors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
