"""Run reconcile_tiers.audit rules over one or all buildings.

Per-building queues are written to .context/flag-queues/<uuid>/auto-scan-<ts>.json.
A pointer at .context/flag-queues/<uuid>/auto-scan-latest.json is updated to the
most recent run so the viewer can fetch it on building load.

Each scan honors `.context/flag-calibration/<uuid>.json` (dismissed
(locator, rule) pairs are suppressed by default) and emits a `diff` field
comparing against the prior auto-scan for the same building.

Usage:
    python -m reconcile_tiers.audit.cohort_scan --uuid <uuid>
    python -m reconcile_tiers.audit.cohort_scan --all [--root pipeline-outputs]
                                                     [--show-dismissed]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconcile_tiers.audit import calibration as calib
from reconcile_tiers.audit.rules import build_queue, run_all_rules
from reconcile_tiers.energy.score_flags import score_flag_queue


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _payload_path(building_dir: Path) -> Path | None:
    path = building_dir / "tier_payload.json"
    return path if path.exists() else None


def _diff_against_prior(
    new_items: list[dict[str, Any]],
    prior_queue: dict[str, Any] | None,
    prior_path: Path | None,
) -> dict[str, Any] | None:
    """Compute (locator, rule) bucket diff vs the prior auto-scan, if any."""
    if not prior_queue:
        return None

    def keyset(items):
        out = set()
        for item in items or []:
            rule = item.get("rule")
            locator = item.get("locator")
            if rule and locator:
                out.add((locator, rule))
        return out

    prior_keys = keyset(prior_queue.get("items"))
    new_keys = keyset(new_items)

    fixed = sorted(prior_keys - new_keys)
    persisting = sorted(prior_keys & new_keys)
    new = sorted(new_keys - prior_keys)

    return {
        "compared_to": prior_path.name if prior_path else None,
        "compared_to_created": prior_queue.get("created"),
        "fixed": [{"locator": loc, "rule": rule} for loc, rule in fixed],
        "persisting": [{"locator": loc, "rule": rule} for loc, rule in persisting],
        "new": [{"locator": loc, "rule": rule} for loc, rule in new],
    }


def _read_prior(latest_path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    if not latest_path.exists():
        return None, None
    try:
        data = json.loads(latest_path.read_text())
    except json.JSONDecodeError:
        return None, None
    return data, latest_path


def _scan_one(
    building_uuid: str,
    payload_path: Path,
    queues_root: Path,
    calibration_root: Path,
    timestamp: str,
    *,
    show_dismissed: bool = False,
) -> dict[str, Any]:
    payload = json.loads(payload_path.read_text())
    raw_items = run_all_rules(payload)

    calibration = calib.load(building_uuid, calibration_root)
    if show_dismissed:
        items = raw_items
        suppressed: list[dict[str, Any]] = []
    else:
        items, suppressed = calib.suppress_items(raw_items, calibration)

    queue_dir = queues_root / building_uuid
    queue_dir.mkdir(parents=True, exist_ok=True)
    latest_path = queue_dir / "auto-scan-latest.json"
    prior_queue, prior_path = _read_prior(latest_path)
    diff = _diff_against_prior(items, prior_queue, prior_path)

    queue = build_queue(
        building_uuid=building_uuid,
        items=items,
        source="auto-scan",
        created=timestamp,
    )
    if diff is not None:
        queue["diff"] = diff
    if suppressed:
        queue["suppressed_count"] = len(suppressed)

    queue_path = queue_dir / f"auto-scan-{timestamp}.json"
    queue_path.write_text(json.dumps(queue, indent=2))
    latest_path.write_text(json.dumps(queue, indent=2))
    if diff is not None:
        (queue_dir / f"auto-scan-{timestamp}-diff.json").write_text(
            json.dumps(diff, indent=2)
        )

    return {
        "uuid": building_uuid,
        "queue_path": str(queue_path),
        "item_count": len(items),
        "suppressed_count": len(suppressed),
        "rule_counts": dict(Counter(item["rule"] for item in items)),
        "diff_summary": (
            {
                "fixed": len(diff["fixed"]),
                "persisting": len(diff["persisting"]),
                "new": len(diff["new"]),
            }
            if diff is not None
            else None
        ),
    }


def scan_one(
    uuid: str,
    root: Path,
    queues_root: Path,
    calibration_root: Path,
    *,
    show_dismissed: bool = False,
    score_impact: bool = False,
) -> dict[str, Any]:
    building_dir = root / uuid
    if not building_dir.is_dir():
        raise FileNotFoundError(f"No building directory: {building_dir}")
    payload_path = _payload_path(building_dir)
    if payload_path is None:
        raise FileNotFoundError(f"No tier_payload.json under {building_dir}")
    result = _scan_one(
        uuid,
        payload_path,
        queues_root,
        calibration_root,
        _ts(),
        show_dismissed=show_dismissed,
    )
    if score_impact:
        result["impact"] = score_flag_queue(uuid, root=root, queues_root=queues_root)
    return result


def scan_all(
    root: Path,
    queues_root: Path,
    calibration_root: Path,
    *,
    show_dismissed: bool = False,
    score_impact: bool = False,
) -> dict[str, Any]:
    timestamp = _ts()
    results: list[dict[str, Any]] = []
    rule_counter: Counter[str] = Counter()
    skipped: list[str] = []
    persistent_offenders: list[dict[str, Any]] = []

    for building_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        uuid = building_dir.name
        payload_path = _payload_path(building_dir)
        if payload_path is None:
            skipped.append(uuid)
            continue
        try:
            result = _scan_one(
                uuid,
                payload_path,
                queues_root,
                calibration_root,
                timestamp,
                show_dismissed=show_dismissed,
            )
        except Exception as exc:
            results.append({"uuid": uuid, "error": str(exc)})
            continue
        results.append(result)
        rule_counter.update(result.get("rule_counts") or {})
        if score_impact:
            try:
                result["impact"] = score_flag_queue(
                    uuid, root=root, queues_root=queues_root
                )
            except Exception as exc:
                result["impact_error"] = str(exc)

        offenders = calib.persistent_offenders(calib.load(uuid, calibration_root))
        for off in offenders:
            persistent_offenders.append({**off, "uuid": uuid})

    index = {
        "schema": "cohort-defect-scan/v1",
        "created": timestamp,
        "buildings_scanned": len([r for r in results if "error" not in r]),
        "buildings_skipped": skipped,
        "rule_frequency": dict(rule_counter.most_common()),
        "persistent_dismissals": persistent_offenders,
        "results": results,
    }

    summary_dir = Path(".context/cohort-defect-scan") / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--uuid", type=str, help="Scan a single building UUID.")
    group.add_argument(
        "--all", action="store_true", help="Scan every building under --root."
    )
    parser.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--queues-root", type=Path, default=Path(".context/flag-queues")
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path(".context/flag-calibration"),
    )
    parser.add_argument(
        "--show-dismissed",
        action="store_true",
        help="Include items the user has previously dismissed via the viewer.",
    )
    parser.add_argument(
        "--score-impact",
        action="store_true",
        help="Also write auto-scan-latest-scored.json with energy-impact fields.",
    )
    args = parser.parse_args(argv)

    args.queues_root.mkdir(parents=True, exist_ok=True)
    args.calibration_root.mkdir(parents=True, exist_ok=True)

    if args.uuid:
        result = scan_one(
            args.uuid,
            args.root,
            args.queues_root,
            args.calibration_root,
            show_dismissed=args.show_dismissed,
            score_impact=args.score_impact,
        )
        print(json.dumps(result, indent=2))
        return 0

    summary = scan_all(
        args.root,
        args.queues_root,
        args.calibration_root,
        show_dismissed=args.show_dismissed,
        score_impact=args.score_impact,
    )
    print(
        json.dumps(
            {
                "buildings_scanned": summary["buildings_scanned"],
                "buildings_skipped": len(summary["buildings_skipped"]),
                "rule_frequency": summary["rule_frequency"],
                "persistent_dismissals": len(summary["persistent_dismissals"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
