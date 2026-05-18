#!/usr/bin/env python3
"""Phase B.5 -- random-search over BIP solver hyperparameters.

Samples :class:`SolverConfig` settings uniformly from the ranges in the
plan, runs the solver on a 20-building pilot set, scores each trial via
:func:`reconcile_v3.reconstruction.realism.score_building` against the
V3 full-model reference (``slanted_roofs`` + ``flat_ceilings``), and
tracks the top-3 trials.

Score
-----
    score = mean_iou + 0.3 * frac(iou >= 0.9) - 0.5 * review_rate

Inputs
------
- ``--candidates``   Phase A per-building candidates JSON (with
                     ``scan_footprint_xz``).
- ``--v3-results``   V3 full-model results (for the reference envelope).

Outputs
-------
- ``<out-dir>/trials.json`` -- one record per trial (config, aggregate
  realism, score, per-building outcomes).
- ``<out-dir>/top_trials.md`` -- human summary of the top-3.
- ``reconcile_v3/reconstruction/hyperparams.json`` (when ``--promote``
  is passed) -- the best trial's SolverConfig, checked in so later runs
  pick it up as the default.

Usage
-----
    python scripts/optimize_reconstruction.py \
        --n-trials 50 --seed 42 \
        --out-dir reports/reconstruction_trials_20260419
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile_v3.reconstruction.realism import (
    aggregate_scores,
    score_building,
)
from reconcile_v3.reconstruction.solver import (
    SolverConfig,
    solve_building_with_zones,
)

# Random-search ranges (inclusive). Keep in lock-step with the plan.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "w_fit": (0.5, 2.0),
    "w_prior": (0.0, 1.0),
    "w_complexity": (0.01, 0.5),
    "theta_cov": (0.80, 0.95),
    "theta_overlap": (0.05, 0.30),
    "theta_az_deg": (20.0, 90.0),
    "k_azimuth_bins": (2.0, 8.0),  # sampled, then rounded
}


def _sample_config(rng: random.Random, time_limit_s: float = 5.0) -> SolverConfig:
    def _u(key: str) -> float:
        lo, hi = SEARCH_SPACE[key]
        return rng.uniform(lo, hi)

    return SolverConfig(
        w_fit=_u("w_fit"),
        w_prior=_u("w_prior"),
        w_complexity=_u("w_complexity"),
        theta_cov=_u("theta_cov"),
        theta_overlap=_u("theta_overlap"),
        theta_az_deg=_u("theta_az_deg"),
        k_azimuth_bins=round(_u("k_azimuth_bins")),
        time_limit_s=time_limit_s,
    )


def _load_pilot(
    candidates_path: Path,
    v3_results_path: Path,
    pilot_n: int,
    seed: int,
) -> list[dict]:
    """Return a list of ``{uuid, candidates, scan_fp, v3}`` dicts."""
    print(f"Loading {candidates_path}...")
    with candidates_path.open() as h:
        cands = json.load(h)
    print(f"Loading {v3_results_path}...")
    with v3_results_path.open() as h:
        v3_rows = json.load(h)
    v3_by_uuid = {r["building_uuid"]: r for r in v3_rows}

    # Pilot = deterministic shuffle -> first pilot_n buildings that have
    # both a scan footprint AND a V3 reference envelope (otherwise the
    # realism score is None and the trial can't learn from them).
    rng = random.Random(seed)
    shuffled = list(cands)
    rng.shuffle(shuffled)

    pilot: list[dict] = []
    for entry in shuffled:
        uuid = entry["building_uuid"]
        v3 = v3_by_uuid.get(uuid)
        if not v3:
            continue
        if not entry.get("scan_footprint_xz"):
            continue
        # Phase B.5 only scores buildings where the solver has a slanted-
        # roof reconstruction problem to solve. Flat-only V3 references
        # produce zero candidates by design (no merged_roof_segments), and
        # including them masks the solver's actual behaviour.
        if not v3.get("slanted_roofs"):
            continue
        if not entry.get("candidates"):
            continue
        pilot.append(
            {
                "uuid": uuid,
                "candidates": entry.get("candidates") or [],
                "scan_fp": [
                    (float(x), float(z)) for x, z in entry["scan_footprint_xz"]
                ],
                "zones": entry.get("zones") or [],
                "v3": {
                    "slanted_roofs": v3.get("slanted_roofs") or [],
                    "flat_ceilings": v3.get("flat_ceilings") or [],
                },
            }
        )
        if len(pilot) >= pilot_n:
            break
    return pilot


def _run_trial(
    cfg: SolverConfig,
    pilot: list[dict],
) -> tuple[float, dict, list[dict]]:
    """Run the solver on the pilot set with ``cfg``, score each building's
    BIP output against its V3 reference, and compute the aggregate trial
    score. Returns ``(score, aggregate_stats, per_building_rows)``.
    """
    per_building: list[dict] = []
    realism_scores = []
    review_flags = 0
    for b in pilot:
        res = solve_building_with_zones(
            b["uuid"],
            b["candidates"],
            b["scan_fp"],
            zones=b.get("zones") or [],
            config=cfg,
        )
        selected = []
        if res.selected_face_ids:
            sel_ids = set(res.selected_face_ids)
            selected = [c for c in b["candidates"] if c["id"] in sel_ids]
        realism = score_building(b["uuid"], selected, b["v3"])
        realism_scores.append(realism)
        if res.decision != "auto_accept":
            review_flags += 1
        per_building.append(
            {
                "uuid": b["uuid"],
                "status": res.status,
                "decision": res.decision,
                "coverage": res.coverage_ratio,
                "iou": realism.iou_xz,
                "coverage_ref": realism.coverage_ref,
                "over_coverage": realism.over_coverage,
                "az_mismatch_frac": realism.azimuth_mismatch_frac,
                "solve_ms": res.solve_time_ms,
            }
        )

    aggregate = aggregate_scores(realism_scores)
    review_rate = review_flags / max(len(pilot), 1)
    score = (
        aggregate["mean_iou"] + 0.3 * aggregate["frac_iou_ge_0_9"] - 0.5 * review_rate
    )
    aggregate["review_rate"] = review_rate
    return score, aggregate, per_building


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default="reports/candidate_faces_20260419/candidates.json",
        type=Path,
    )
    ap.add_argument(
        "--v3-results",
        default="reconcile/reconcile_v3_results.json",
        type=Path,
    )
    ap.add_argument(
        "--out-dir",
        default="reports/reconstruction_trials_20260419",
        type=Path,
    )
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--pilot-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--promote",
        action="store_true",
        help="Write the winning config to "
        "reconcile_v3/reconstruction/hyperparams.json.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pilot = _load_pilot(args.candidates, args.v3_results, args.pilot_n, args.seed)
    if not pilot:
        print("Pilot set is empty -- check the input paths.", file=sys.stderr)
        return 2
    print(f"Pilot size: {len(pilot)} buildings")

    rng = random.Random(args.seed)
    trials: list[dict] = []
    t_start = time.perf_counter()
    for i in range(args.n_trials):
        cfg = _sample_config(rng)
        t0 = time.perf_counter()
        score, agg, per_bldg = _run_trial(cfg, pilot)
        dt = time.perf_counter() - t0
        trials.append(
            {
                "trial": i,
                "config": asdict(cfg),
                "score": score,
                "aggregate": agg,
                "per_building": per_bldg,
                "trial_wall_s": dt,
            }
        )
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"  trial {i + 1}/{args.n_trials} "
                f"score={score:.3f} mean_iou={agg['mean_iou']:.3f} "
                f"review={agg['review_rate']:.2f} "
                f"({dt:.1f}s -- total {elapsed:.0f}s)"
            )

    trials.sort(key=lambda t: t["score"], reverse=True)

    # ── Persist ──────────────────────────────────────────────────────
    trials_path = args.out_dir / "trials.json"
    with trials_path.open("w") as h:
        json.dump(trials, h, indent=2)

    top_path = args.out_dir / "top_trials.md"
    lines = [
        "# Phase B.5 -- hyperparameter search summary",
        "",
        f"Pilot: {len(pilot)} buildings; trials: {args.n_trials}; seed: {args.seed}.",
        f"Total wall time: {time.perf_counter() - t_start:.0f}s.",
        "",
        "## Top 3 trials",
        "",
    ]
    for rank, t in enumerate(trials[:3], 1):
        c = t["config"]
        a = t["aggregate"]
        lines += [
            f"### #{rank} -- score {t['score']:.3f}",
            "",
            f"- mean IoU: {a['mean_iou']:.3f}",
            f"- median IoU: {a['median_iou']:.3f}",
            f"- frac IoU >= 0.9: {a['frac_iou_ge_0_9']:.2f}",
            f"- review rate: {a['review_rate']:.2f}",
            f"- mean coverage_ref: {a['mean_coverage_ref']:.3f}",
            f"- mean over_coverage: {a['mean_over_coverage']:.3f}",
            "",
            "Config:",
            "```json",
            json.dumps(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in c.items()},
                indent=2,
            ),
            "```",
            "",
        ]
    top_path.write_text("\n".join(lines))

    print(f"\nWrote {trials_path}")
    print(f"Wrote {top_path}")
    best = trials[0]
    print(
        f"\nBest trial score={best['score']:.3f} "
        f"(mean_iou={best['aggregate']['mean_iou']:.3f}, "
        f"review={best['aggregate']['review_rate']:.2f})"
    )

    if args.promote:
        hp_path = Path("reconcile_v3/reconstruction/hyperparams.json")
        with hp_path.open("w") as h:
            # Drop time_limit_s -- runtime rather than a tuned knob.
            promoted = {k: v for k, v in best["config"].items() if k != "time_limit_s"}
            json.dump(promoted, h, indent=2)
        print(f"Promoted best config to {hp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
