"""Policy-gated silent-fix entry points.

The first implementation records what would be auto-routed without mutating
scan-derived room wall/floor locators. Actual geometry synthesis can be added
per calibrated rule without changing the build integration point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconcile_tiers.energy.score_flags import (
    DEFAULT_IMPACT_POLICY,
    ImpactPolicy,
    score_queue,
)
from reconcile_tiers.energy.u_values import DEFAULT_DK_TABLE, UValueTable
from reconcile_tiers.payload.schema import (
    TierPayload,
    payload_from_dict,
)


@dataclass(frozen=True, slots=True)
class FixPolicy:
    enabled: bool = False
    impact: ImpactPolicy = DEFAULT_IMPACT_POLICY

    @classmethod
    def from_env(cls) -> FixPolicy:
        return cls(enabled=os.environ.get("AUTOFIX_ENABLED") == "1")


@dataclass(frozen=True, slots=True)
class SilentFixOutcome:
    payload: TierPayload
    audit_log: dict[str, Any]


def _below_threshold(item: dict[str, Any], policy: ImpactPolicy) -> bool:
    impact = item.get("impact") or {}
    return (
        abs(float(impact.get("kwh_delta") or 0.0)) <= policy.kwh_threshold
        and float(impact.get("pct_of_total") or 0.0) <= policy.pct_threshold
    )


def apply_silent_fixes(
    payload: TierPayload,
    policy: FixPolicy,
    table: UValueTable = DEFAULT_DK_TABLE,
    *,
    queue: dict[str, Any] | None = None,
) -> SilentFixOutcome:
    items = []
    if queue:
        scored = score_queue(payload, queue, table=table)
        for item in scored.get("items") or []:
            rule = item.get("rule")
            if rule in {"silent_drop", "ceiling_coverage_gap"} and _below_threshold(
                item, policy.impact
            ):
                items.append(
                    {
                        "locator": item.get("locator"),
                        "rule": rule,
                        "impact": item.get("impact"),
                        "action": "recorded" if policy.enabled else "dry_run",
                    }
                )
    return SilentFixOutcome(
        payload=payload,
        audit_log={
            "schema": "tier-payload-autofix/v1",
            "created": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "enabled": policy.enabled,
            "mutated_payload": False,
            "items": items,
            "invariant": (
                "room_walls[].locator_id and room_floor[].locator_id are not modified"
            ),
        },
    )


def dry_run(
    uuid: str,
    *,
    root: Path | str = Path("pipeline-outputs"),
    queues_root: Path | str = Path(".context/flag-queues"),
) -> dict[str, Any]:
    root = Path(root)
    queues_root = Path(queues_root)
    payload_path = root / uuid / "tier_payload.json"
    queue_path = queues_root / uuid / "auto-scan-latest.json"
    payload = payload_from_dict(json.loads(payload_path.read_text()))
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else None
    outcome = apply_silent_fixes(payload, FixPolicy(enabled=False), queue=queue)
    out_path = root / uuid / "tier_payload_autofix.json"
    out_path.write_text(json.dumps(outcome.audit_log, indent=2, sort_keys=True) + "\n")
    return {
        "uuid": uuid,
        "audit_path": str(out_path),
        "item_count": len(outcome.audit_log["items"]),
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
            dry_run(args.uuid, root=args.root, queues_root=args.queues_root), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
