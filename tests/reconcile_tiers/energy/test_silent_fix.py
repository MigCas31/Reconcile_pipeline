from __future__ import annotations

from reconcile_tiers.energy.silent_fix import FixPolicy, apply_silent_fixes
from tests.reconcile_tiers.energy.test_estimator import _cube_payload


def test_silent_fix_records_below_threshold_items_without_mutating_payload():
    payload = _cube_payload()
    queue = {
        "schema": "flag-queue/v1",
        "building_uuid": payload.uuid,
        "items": [
            {
                "id": "drop",
                "locator": "cube::ceiling::0",
                "rule": "silent_drop",
                "severity": "low",
                "evidence": {"area_m2": 0.00001},
            }
        ],
    }

    outcome = apply_silent_fixes(payload, FixPolicy(enabled=True), queue=queue)

    assert outcome.payload is payload
    assert outcome.audit_log["mutated_payload"] is False
    assert outcome.audit_log["items"][0]["action"] == "recorded"
