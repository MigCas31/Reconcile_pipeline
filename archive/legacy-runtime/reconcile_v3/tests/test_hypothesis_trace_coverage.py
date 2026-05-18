"""Every V3 output element must trace to one audit_log entry."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = (
    Path(__file__).resolve().parent.parent.parent
    / "reconcile"
    / "reconcile_v3_results.json"
)

ELEMENT_KEYS = (
    "parts",
    "gaps",
    "slabs",
    "wall_extensions",
    "flat_ceilings",
    "slanted_roofs",
    "dormers",
    "unresolved",
)


def _iter_buildings() -> list[dict]:
    if not RESULTS.exists():
        return []
    return json.loads(RESULTS.read_text())


def test_each_element_has_trace():
    buildings = _iter_buildings()
    if not buildings:
        return
    missing: list[str] = []
    for b in buildings:
        for key in ELEMENT_KEYS:
            for elem in b.get(key) or []:
                trace = elem.get("trace") or {}
                if not trace.get("stage") or not trace.get("rule"):
                    missing.append(f"{b.get('building_uuid')}::{key}::{elem.get('id')}")
    assert not missing, f"Elements without trace: {missing[:10]}"


def test_audit_log_matches_elements():
    buildings = _iter_buildings()
    if not buildings:
        return
    for b in buildings:
        element_ids = set()
        for key in ELEMENT_KEYS:
            for elem in b.get(key) or []:
                if elem.get("id"):
                    element_ids.add(elem["id"])
        audit_ids = {entry.get("element_id") for entry in (b.get("audit_log") or [])}
        missing = element_ids - audit_ids
        assert not missing, (
            f"Building {b.get('building_uuid')} has elements without audit entries: "
            f"{sorted(missing)[:5]}"
        )
