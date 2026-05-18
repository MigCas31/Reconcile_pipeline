"""Persistent dismissal calibration per building.

When the user dismisses an auto-flag in the viewer (clicks the x next to a
row, then sends the queue), the (locator, rule) pair is recorded here so
the cohort scanner skips it next time. Manual flags (`rule is None`) are
not tracked -- dismissing one means "I changed my mind about this manual
flag", which has no learning value for the rule library.

Schema: `flag-calibration/v1`. Stored at
`.context/flag-calibration/<uuid>.json` (one file per building, alongside
the per-building flag queue directory at `.context/flag-queues/<uuid>/`).

The scanner uses these files only to **suppress** dismissed instances --
not to mutate rule thresholds. Per `feedback_keep_diagnostic_signal`, we
do not silently widen rules. Persistent dismissals (high `dismiss_count`)
are surfaced in the cohort summary so the human can decide whether to
refine the rule with more discriminating geometry.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "flag-calibration/v1"


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def calibration_path(uuid: str, root: Path) -> Path:
    return root / f"{uuid}.json"


def load(uuid: str, root: Path) -> dict[str, Any]:
    """Return the calibration dict for `uuid`, or an empty stub if missing."""
    path = calibration_path(uuid, root)
    if not path.exists():
        return {"schema": SCHEMA, "uuid": uuid, "updated": None, "dismissals": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"schema": SCHEMA, "uuid": uuid, "updated": None, "dismissals": []}
    data.setdefault("schema", SCHEMA)
    data.setdefault("uuid", uuid)
    data.setdefault("dismissals", [])
    return data


def save(uuid: str, root: Path, data: dict[str, Any]) -> Path:
    path = calibration_path(uuid, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def dismissed_keys(calibration: dict[str, Any]) -> set[tuple[str, str]]:
    """`(locator, rule)` pairs the user has previously vetoed."""
    out: set[tuple[str, str]] = set()
    for entry in calibration.get("dismissals") or []:
        locator = entry.get("locator")
        rule = entry.get("rule")
        if locator and rule:
            out.add((locator, rule))
    return out


def suppress_items(
    items: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split `items` into (kept, suppressed) using the calibration's dismissal set.

    Items without a rule (manual flags) are always kept -- calibration only
    applies to auto-scanner output.
    """
    keys = dismissed_keys(calibration)
    if not keys:
        return list(items), []
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for item in items:
        rule = item.get("rule")
        locator = item.get("locator")
        if rule and locator and (locator, rule) in keys:
            suppressed.append(item)
        else:
            kept.append(item)
    return kept, suppressed


def merge_dismissals(
    calibration: dict[str, Any],
    queue_items: list[dict[str, Any]],
    *,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Merge dismissed auto-flag items from a queue into the calibration in-place.

    A queue item is folded in iff `dismissed: True` AND `rule` is set. New
    pairs are appended; pairs already present have their `last_dismissed`
    refreshed and `dismiss_count` incremented.
    """
    ts = timestamp or _ts()
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in calibration.get("dismissals") or []:
        locator = entry.get("locator")
        rule = entry.get("rule")
        if locator and rule:
            by_key[(locator, rule)] = entry

    touched = False
    for item in queue_items:
        if not item.get("dismissed"):
            continue
        rule = item.get("rule")
        locator = item.get("locator")
        if not rule or not locator:
            continue
        key = (locator, rule)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = {
                "locator": locator,
                "rule": rule,
                "first_dismissed": ts,
                "last_dismissed": ts,
                "dismiss_count": 1,
                "note": item.get("note"),
            }
        else:
            existing["last_dismissed"] = ts
            existing["dismiss_count"] = int(existing.get("dismiss_count") or 0) + 1
            if item.get("note") and not existing.get("note"):
                existing["note"] = item["note"]
        touched = True

    if touched:
        calibration["dismissals"] = sorted(
            by_key.values(),
            key=lambda d: (d.get("rule"), d.get("locator")),
        )
        calibration["updated"] = ts
        calibration.setdefault("schema", SCHEMA)

    return calibration


def persistent_offenders(
    calibration: dict[str, Any],
    *,
    threshold: int = 3,
) -> list[dict[str, Any]]:
    """Dismissals seen more than `threshold` times -- candidates for rule refinement."""
    out: list[dict[str, Any]] = []
    for entry in calibration.get("dismissals") or []:
        if int(entry.get("dismiss_count") or 0) >= threshold:
            out.append(entry)
    return out
