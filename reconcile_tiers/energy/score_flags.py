"""Score flag queues by estimated downstream energy impact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconcile_tiers.energy.estimator import estimate, estimator_summary_dict
from reconcile_tiers.energy.perturbations import (
    ClipOutOfEnvelope,
    FillCoverageGap,
    Perturbation,
    RemoveSilentDrop,
    RestoreInversion,
    perturbation_for_flag,
)
from reconcile_tiers.energy.u_values import DEFAULT_DK_TABLE, UValueTable
from reconcile_tiers.payload.schema import TierPayload, payload_from_dict


@dataclass(frozen=True, slots=True)
class ImpactPolicy:
    kwh_threshold: float = 100.0
    pct_threshold: float = 0.01


DEFAULT_IMPACT_POLICY = ImpactPolicy()


def _impact_for_perturbation(
    perturbation: Perturbation | None,
    *,
    baseline_kwh: float,
    table: UValueTable,
) -> dict[str, Any] | None:
    if perturbation is None:
        return {
            "kwh_delta": 0.0,
            "pct_of_total": 0.0,
            "confidence": 1.0,
            "perturbation_kind": "no_energy_impact",
            "perturbation_summary": (
                "No corrective energy perturbation is defined for this quality rule."
            ),
        }

    if isinstance(perturbation, FillCoverageGap):
        u_value = table.u_value(perturbation.adjacency)
        kwh_delta = u_value * perturbation.area_m2 * table.hdd_k_days * 24.0 / 1000.0
        confidence = 0.78
        summary = (
            f"Synthesise {perturbation.area_m2:.1f} m² ceiling at "
            f"{perturbation.adjacency.value}, U={u_value:.2f}."
        )
    elif isinstance(perturbation, RestoreInversion):
        kwh_delta = 0.0
        confidence = 0.90
        summary = "Restoring polygon winding changes visibility, not envelope area."
    elif isinstance(perturbation, ClipOutOfEnvelope):
        u_value = table.u_value("externalAir")
        kwh_delta = -(u_value * perturbation.area_m2 * table.hdd_k_days * 24.0 / 1000.0)
        confidence = 0.70
        summary = f"Clip {perturbation.area_m2:.1f} m² outside the envelope."
    elif isinstance(perturbation, RemoveSilentDrop):
        kwh_delta = 0.0
        confidence = 0.95
        summary = "Renderer-only silent drop; no measurable energy impact."
    else:
        kwh_delta = 0.0
        confidence = 1.0
        summary = "No energy impact."

    return {
        "kwh_delta": round(kwh_delta, 3),
        "pct_of_total": round(abs(kwh_delta) / baseline_kwh, 6)
        if baseline_kwh > 0
        else 0.0,
        "confidence": confidence,
        "perturbation_kind": perturbation.kind,
        "perturbation_summary": summary,
    }


def score_queue(
    payload: TierPayload | dict[str, Any],
    queue: dict[str, Any],
    *,
    table: UValueTable = DEFAULT_DK_TABLE,
) -> dict[str, Any]:
    tier_payload = payload_from_dict(payload) if isinstance(payload, dict) else payload
    baseline = estimate(tier_payload, table)
    baseline_summary = estimator_summary_dict(baseline)
    baseline_kwh = float(baseline.summary.annual_kwh_proxy)

    scored_items: list[dict[str, Any]] = []
    for item in queue.get("items") or []:
        if not isinstance(item, dict):
            continue
        perturbation = perturbation_for_flag(tier_payload, item)
        scored_items.append(
            {
                **item,
                "impact": _impact_for_perturbation(
                    perturbation, baseline_kwh=baseline_kwh, table=table
                ),
            }
        )

    return {
        **queue,
        "schema": "flag-queue/v2",
        "schema_min_consumer": "v1",
        "created": queue.get("created") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "energy_baseline": baseline_summary,
        "items": scored_items,
    }


def score_flag_queue(
    uuid: str,
    *,
    root: Path | str = Path("pipeline-outputs"),
    queues_root: Path | str = Path(".context/flag-queues"),
    table: UValueTable = DEFAULT_DK_TABLE,
) -> dict[str, Any]:
    root = Path(root)
    queues_root = Path(queues_root)
    payload_path = root / uuid / "tier_payload.json"
    queue_path = queues_root / uuid / "auto-scan-latest.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"No tier_payload.json under {root / uuid}")
    if not queue_path.exists():
        raise FileNotFoundError(f"No auto-scan-latest.json under {queues_root / uuid}")

    payload = json.loads(payload_path.read_text())
    queue = json.loads(queue_path.read_text())
    scored = score_queue(payload, queue, table=table)

    out_path = queues_root / uuid / "auto-scan-latest-scored.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")

    estimate_path = root / uuid / "energy_estimate.json"
    estimate_path.write_text(
        json.dumps(scored["energy_baseline"], indent=2, sort_keys=True) + "\n"
    )
    return {
        "uuid": uuid,
        "queue_path": str(out_path),
        "energy_estimate_path": str(estimate_path),
        "item_count": len(scored["items"]),
        "energy_baseline": scored["energy_baseline"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--queues-root", type=Path, default=Path(".context/flag-queues")
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            score_flag_queue(args.uuid, root=args.root, queues_root=args.queues_root),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
