from __future__ import annotations

from reconcile_tiers.energy.score_flags import score_queue
from reconcile_tiers.energy.u_values import DEFAULT_DK_TABLE
from tests.reconcile_tiers.energy.test_estimator import _cube_payload


def test_score_queue_adds_impact_to_every_item_and_deprioritises_silent_drop():
    payload = _cube_payload()
    queue = {
        "schema": "flag-queue/v1",
        "building_uuid": payload.uuid,
        "created": "20260430T120000Z",
        "source": "auto-scan",
        "items": [
            {
                "id": "gap",
                "locator": "cube::room::0",
                "kind": "room",
                "rule": "ceiling_coverage_gap",
                "severity": "med",
                "evidence": {"floor_area_m2": 36, "covered_area_m2": 30},
            },
            {
                "id": "drop",
                "locator": "cube::ceiling::0",
                "kind": "ceiling",
                "rule": "silent_drop",
                "severity": "low",
                "evidence": {"area_m2": 0.00001},
            },
        ],
    }

    scored = score_queue(payload, queue, table=DEFAULT_DK_TABLE)

    assert scored["schema"] == "flag-queue/v2"
    assert "energy_baseline" in scored
    assert all("impact" in item for item in scored["items"])
    silent_drop = next(
        item for item in scored["items"] if item["rule"] == "silent_drop"
    )
    assert silent_drop["impact"]["kwh_delta"] < 1
