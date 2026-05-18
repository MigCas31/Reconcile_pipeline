"""Fit the roof_quality calibration on manual ratings.

Run as:

    python -m reconcile_tiers.quality.fit_calibration

Reads ``.context/roof_ratings.json`` and the corresponding
``pipeline-outputs/<uuid>/tier_payload.json``. Drops ``upstream_error`` from
training (it labels failed scans, not pipeline-quality decisions). Splits
80/20 by deterministic uuid hash, fits closed-form OLS via numpy, picks the
model and writes ``calibration.json`` next to this module.

Refuses to fit if more than one distinct rater is detected in the data —
forces a deliberate decision rather than silently dilute a single-rater
preference predictor with a different rater's signal.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from reconcile_tiers.payload.schema import payload_from_dict
from reconcile_tiers.quality.features import extract_features, feature_names

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = WORKSPACE_ROOT / ".context" / "roof_ratings.json"
PIPELINE_DIR = WORKSPACE_ROOT / "pipeline-outputs"
OUT_PATH = Path(__file__).parent / "calibration.json"


def _split_holdout(uuid: str, holdout_frac: float = 0.2) -> bool:
    """Deterministic 80/20 split by SHA-256 of the uuid."""
    digest = hashlib.sha256(uuid.encode()).digest()
    bucket = digest[0] / 255.0
    return bucket < holdout_frac


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    ra = _ranks(a)
    rb = _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit roof_quality calibration")
    parser.add_argument("--ratings", default=str(RATINGS_PATH))
    parser.add_argument("--pipeline-dir", default=str(PIPELINE_DIR))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    ratings_path = Path(args.ratings)
    pipeline_dir = Path(args.pipeline_dir)

    raw = json.loads(ratings_path.read_text())

    # Single-rater guard. Today every record is rater-less — that's the
    # implicit "the only rater is the workspace owner" case. If a future
    # record carries a `rater` field with a different identity, refuse.
    raters = {entry.get("rater") for entry in raw.values() if isinstance(entry, dict)}
    raters.discard(None)
    if len(raters) > 1:
        print(
            "refusing to fit: multiple raters present "
            f"({sorted(raters)}). update fit_calibration.py to handle this.",
            file=sys.stderr,
        )
        return 2

    rows: list[tuple[str, int, dict[str, float]]] = []
    skipped = {
        "missing_payload": 0,
        "upstream_error": 0,
        "non_int_rating": 0,
        "load_fail": 0,
    }
    for uuid, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        rating = entry.get("rating")
        if rating == "upstream_error":
            skipped["upstream_error"] += 1
            continue
        if not isinstance(rating, int) or not (1 <= rating <= 5):
            skipped["non_int_rating"] += 1
            continue
        payload_path = pipeline_dir / uuid / "tier_payload.json"
        if not payload_path.exists():
            skipped["missing_payload"] += 1
            continue
        try:
            payload = payload_from_dict(json.loads(payload_path.read_text()))
            feats = extract_features(payload)
        except Exception as exc:
            skipped["load_fail"] += 1
            print(f"warn: load/extract failed for {uuid}: {exc}", file=sys.stderr)
            continue
        rows.append((uuid, int(rating), feats))

    if len(rows) < 30:
        print(f"refusing to fit: only {len(rows)} usable rows.", file=sys.stderr)
        return 2

    train_rows = [r for r in rows if not _split_holdout(r[0])]
    holdout_rows = [r for r in rows if _split_holdout(r[0])]
    print(
        f"loaded {len(rows)} rows  train={len(train_rows)}  holdout={len(holdout_rows)}"
    )
    print(f"skipped: {skipped}")

    names = feature_names()

    def matrix(
        subset: list[tuple[str, int, dict[str, float]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(
            [[row[2].get(name, 0.0) for name in names] for row in subset], dtype=float
        )
        y = np.array([row[1] for row in subset], dtype=float)
        return x, y

    x_train, y_train = matrix(train_rows)
    x_holdout, y_holdout = matrix(holdout_rows)

    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds[stds < 1e-9] = 1.0
    x_train_z = (x_train - means) / stds
    x_holdout_z = (x_holdout - means) / stds

    # Ridge regression with the intercept fit unregularized. Since features
    # are z-scored (mean 0), the optimal intercept is mean(y) and the weight
    # subproblem is independent.
    ridge = 1.0
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    a = x_train_z.T @ x_train_z + ridge * np.eye(x_train_z.shape[1])
    b = x_train_z.T @ y_centered
    weights = np.linalg.solve(a, b)
    intercept = y_mean

    train_pred = (x_train_z @ weights + intercept).clip(1.0, 5.0)
    holdout_pred = (x_holdout_z @ weights + intercept).clip(1.0, 5.0)
    train_rho = _spearman(list(y_train), list(train_pred))
    holdout_rho = _spearman(list(y_holdout), list(holdout_pred))
    train_mae = float(np.mean(np.abs(train_pred - y_train)))
    holdout_mae = float(np.mean(np.abs(holdout_pred - y_holdout)))
    print(f"  train  spearman={train_rho:.3f}  mae={train_mae:.3f}")
    print(f"  holdout spearman={holdout_rho:.3f}  mae={holdout_mae:.3f}")

    print("\n  predicted vs manual buckets (holdout):")
    bucket_counts: dict[tuple[int, int], int] = {}
    for actual, pred in zip(y_holdout, holdout_pred, strict=False):
        bucket_counts[(int(actual), round(float(pred)))] = (
            bucket_counts.get((int(actual), round(float(pred))), 0) + 1
        )
    for actual in range(1, 6):
        row = []
        for pred in range(1, 6):
            row.append(f"{bucket_counts.get((actual, pred), 0):3d}")
        print(f"   actual={actual} -> pred={'  '.join(row)}")

    sha = _git_commit()
    today = datetime.date.today().strftime("%Y-%m-%d")
    quality_version = f"rq-{today}-{sha[:7]}"

    out = {
        "quality_version": quality_version,
        "model": "linear_ols_ridge",
        "feature_means": {name: float(means[i]) for i, name in enumerate(names)},
        "feature_stds": {name: float(stds[i]) for i, name in enumerate(names)},
        "coefficients": {name: float(weights[i]) for i, name in enumerate(names)},
        "intercept": intercept,
        "metadata": {
            "git_commit": sha,
            "n_train": len(train_rows),
            "n_holdout": len(holdout_rows),
            "spearman_train": train_rho,
            "spearman_holdout": holdout_rho,
            "mae_train": train_mae,
            "mae_holdout": holdout_mae,
            "rater": "martin@lun.energy",
            "fit_date": datetime.datetime.now(datetime.UTC).isoformat(),
            "skipped": skipped,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}  ({quality_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
