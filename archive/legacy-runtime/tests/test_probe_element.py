from __future__ import annotations

import json

from scripts.probe_element import _resolve_element

UUID = "11111111-2222-3333-4444-555555555555"


def test_resolve_element_loads_tier_payload_from_pipeline_dir(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    payload_dir = pipeline_dir / UUID
    payload_dir.mkdir(parents=True)
    token = f"{UUID}::tier-knee-wall::0"
    (payload_dir / "tier_payload.json").write_text(
        json.dumps(
            {
                "uuid": UUID,
                "address": "Testvej 1",
                "rooms": [],
                "gaps": [],
                "ceiling": [],
                "knee_walls": [
                    {
                        "locator_id": token,
                        "kind": "knee",
                        "corners": [
                            {"x": 0, "y": 1, "z": 0},
                            {"x": 1, "y": 1, "z": 0},
                            {"x": 1, "y": 2, "z": 0},
                        ],
                    }
                ],
            }
        )
    )

    result = _resolve_element(
        token,
        roof_results={},
        buildings_path=None,
        pipeline_dir=pipeline_dir,
    )

    assert result["kind"] == "tier-knee-wall"
    assert result["atom_kind"] == "knee"
    assert result["corners"] == [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
