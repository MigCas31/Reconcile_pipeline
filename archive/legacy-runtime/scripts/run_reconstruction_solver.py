#!/usr/bin/env python3
"""Phase B.1 driver — run the BIP reconstruction solver over one
building or a full candidates corpus.

Inputs
------
- ``--candidates`` Phase A output (``reports/candidate_faces_<date>/candidates.json``).
  Must include ``scan_footprint_xz`` per building (see the edit to
  ``scripts/build_candidate_faces.py`` that persists it).
- Optional ``--hyperparams`` JSON overriding :class:`SolverConfig` defaults
  (used by B.5 to evaluate trials).

Outputs
-------
- ``<out-dir>/selections.json`` per-building solve results (one record
  per building — see :class:`SolveResult`).
- ``<out-dir>/summary.md``    aggregate stats.

Usage
-----
    python scripts/run_reconstruction_solver.py \
        --candidates reports/candidate_faces_20260419/candidates.json \
        --out-dir reports/reconstruction_20260419
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile_v3.reconstruction.solver import (
    SolverConfig,
    solve_building_with_zones,
)


def _load_config(path: Path | None) -> SolverConfig:
    if path is None:
        return SolverConfig()
    with path.open() as handle:
        payload = json.load(handle)
    return SolverConfig(**payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default="reports/candidate_faces_20260419/candidates.json",
        type=Path,
    )
    ap.add_argument(
        "--out-dir",
        default="reports/reconstruction_20260419",
        type=Path,
    )
    ap.add_argument("--hyperparams", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--building", type=str, default=None, help="Solve only this building UUID."
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Per-building solver wall-clock cap (seconds).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(args.hyperparams)
    if args.time_limit != cfg.time_limit_s:
        # override only if the CLI value differs from config default
        cfg = SolverConfig(**{**asdict(cfg), "time_limit_s": args.time_limit})

    print(f"Loading {args.candidates}...")
    with args.candidates.open() as handle:
        corpus = json.load(handle)

    if args.building:
        corpus = [b for b in corpus if b["building_uuid"] == args.building]
        if not corpus:
            print(
                f"Building {args.building} not found in {args.candidates}",
                file=sys.stderr,
            )
            return 2
    if args.limit:
        corpus = corpus[: args.limit]

    results: list[dict] = []
    t_start = time.perf_counter()
    for i, b in enumerate(corpus):
        uuid = b["building_uuid"]
        candidates = b.get("candidates") or []
        fp_xz = b.get("scan_footprint_xz")
        fp = [(float(x), float(z)) for x, z in fp_xz] if fp_xz else None
        res = solve_building_with_zones(
            uuid,
            candidates,
            fp,
            zones=b.get("zones") or [],
            config=cfg,
        )
        results.append(asdict(res))
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i + 1}/{len(corpus)} ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - t_start

    # ── Write outputs ────────────────────────────────────────────────
    sel_path = args.out_dir / "selections.json"
    with sel_path.open("w") as handle:
        json.dump(results, handle)

    # Aggregates
    total = len(results)
    solved = sum(1 for r in results if r["status"] == "solved")
    ambiguous = sum(1 for r in results if r["status"] == "ambiguous")
    infeasible = sum(1 for r in results if r["status"] == "infeasible")
    no_cands = sum(1 for r in results if r["status"] == "no_candidates")
    auto_accept = sum(1 for r in results if r["decision"] == "auto_accept")
    review = total - auto_accept
    coverage = [r["coverage_ratio"] for r in results if r["status"] == "solved"]
    solve_ms = [r["solve_time_ms"] for r in results]
    n_faces = [len(r["selected_face_ids"]) for r in results]

    def _pct(p: int, t: int) -> str:
        return f"{100 * p / t:.1f}%" if t else "—"

    md = [
        "# Phase B.1 — reconstruction solver summary",
        "",
        f"Corpus: `{args.candidates}` — {total} buildings (elapsed {elapsed:.1f}s).",
        f"Hyperparameters: `{args.hyperparams or '<defaults>'}` "
        f"(time_limit={cfg.time_limit_s:.0f}s).",
        "",
        "## Status distribution",
        "",
        f"- solved: **{solved}** ({_pct(solved, total)})",
        f"- ambiguous: {ambiguous} ({_pct(ambiguous, total)})",
        f"- infeasible: {infeasible} ({_pct(infeasible, total)})",
        f"- no_candidates: {no_cands} ({_pct(no_cands, total)})",
        "",
        "## Decisions",
        "",
        f"- auto_accept: **{auto_accept}** ({_pct(auto_accept, total)})",
        f"- review: **{review}** ({_pct(review, total)})",
        "",
    ]
    if coverage:
        md += [
            "## Coverage (solved only)",
            "",
            f"- mean: {statistics.mean(coverage):.3f}",
            f"- median: {statistics.median(coverage):.3f}",
            f"- min: {min(coverage):.3f}",
            f"- frac ≥ 0.95: "
            f"{_pct(sum(1 for c in coverage if c >= 0.95), len(coverage))}",
            "",
        ]
    if solve_ms:
        md += [
            "## Solver runtime (per building)",
            "",
            f"- median: {statistics.median(solve_ms):.0f} ms",
            f"- p90: {statistics.quantiles(solve_ms, n=10)[-1]:.0f} ms",
            f"- max: {max(solve_ms)} ms",
            "",
        ]
    if n_faces:
        md += [
            "## Faces selected per building",
            "",
            f"- mean: {statistics.mean(n_faces):.1f}",
            f"- median: {statistics.median(n_faces):.0f}",
            f"- max: {max(n_faces)}",
            "",
        ]

    md_path = args.out_dir / "summary.md"
    md_path.write_text("\n".join(md))

    print(f"\nWrote {sel_path}")
    print(f"Wrote {md_path}")
    print(
        f"{solved}/{total} solved, {auto_accept}/{total} auto-accept, "
        f"{review}/{total} review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
