"""Cohort calibration for energy-impact thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconcile_tiers.energy.score_flags import score_flag_queue


def _proposed_threshold(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) < 4:
        return max(values)
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[1] + 1.5 * (qs[2] - qs[0])


def calibrate(
    *,
    root: Path | str = Path("pipeline-outputs"),
    queues_root: Path | str = Path(".context/flag-queues"),
    out_root: Path | str = Path(".context/energy-calibration"),
) -> dict[str, Any]:
    root = Path(root)
    queues_root = Path(queues_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    by_rule: dict[str, list[float]] = defaultdict(list)
    results: list[dict[str, Any]] = []
    for building_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        uuid = building_dir.name
        if (
            not (building_dir / "tier_payload.json").exists()
            or not (queues_root / uuid / "auto-scan-latest.json").exists()
        ):
            continue
        try:
            result = score_flag_queue(uuid, root=root, queues_root=queues_root)
            scored = json.loads(
                (queues_root / uuid / "auto-scan-latest-scored.json").read_text()
            )
        except Exception as exc:
            results.append({"uuid": uuid, "error": str(exc)})
            continue
        results.append(result)
        for item in scored.get("items") or []:
            impact = item.get("impact") or {}
            by_rule[item.get("rule") or "unknown"].append(
                abs(float(impact.get("kwh_delta") or 0.0))
            )

    rules = {}
    for rule, values in sorted(by_rule.items()):
        csv_path = out_dir / f"{rule}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kwh_delta"])
            for value in values:
                writer.writerow([value])
        rules[rule] = {
            "count": len(values),
            "median_kwh_delta": statistics.median(values) if values else 0.0,
            "proposed_threshold_kwh": _proposed_threshold(values),
            "auto_fix_safe": (sum(1 for v in values if v > 200.0) / len(values) <= 0.05)
            if values
            else False,
            "histogram_csv": csv_path.name,
        }

    index = {
        "schema": "energy-calibration/v1",
        "created": timestamp,
        "rules": rules,
        "results": results,
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--queues-root", type=Path, default=Path(".context/flag-queues")
    )
    parser.add_argument("--out", type=Path, default=Path(".context/energy-calibration"))
    args = parser.parse_args(argv)
    index = calibrate(root=args.root, queues_root=args.queues_root, out_root=args.out)
    print(
        json.dumps(
            {"rules": index["rules"], "scored_buildings": len(index["results"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
