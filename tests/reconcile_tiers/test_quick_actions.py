"""Unit tests for reconcile_tiers.quick_actions.

These exercise every entry in REGISTRY against a synthetic tier_payload
fixture so the tests don't depend on the live pipeline-outputs/ corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reconcile_tiers import quick_actions as qa

UUID = "00000000-0000-0000-0000-000000000001"


def _square(y: float, side: float = 4.0) -> list[dict[str, float]]:
    h = side / 2
    return [
        {"x": -h, "y": y, "z": -h},
        {"x": h, "y": y, "z": -h},
        {"x": h, "y": y, "z": h},
        {"x": -h, "y": y, "z": h},
    ]


@pytest.fixture
def fixture_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialise a minimal tier_payload.json under a tmp pipeline-outputs root."""
    flat_loc = f"{UUID}::tier-ceiling-flat::0"
    slanted_loc = f"{UUID}::tier-ceiling-slanted::0"
    knee_loc = f"{UUID}::tier-knee-wall::0"
    gap_loc = f"{UUID}::tier-gap-side::0"
    room_loc = f"{UUID}::tier-room::0"
    wall_loc = f"{UUID}::tier-wall::0:0"

    payload: dict[str, Any] = {
        "uuid": UUID,
        "address": "Test 1, 2300 Copenhagen",
        "schema_version": "tier_payload/1",
        "ceiling": [
            {
                "locator_id": flat_loc,
                "source": "flat_ceiling",
                "role": "ceiling",
                "corners": _square(2.5),
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": -2.5},
                "adjacency": [{"locator_id": slanted_loc}],
                "holes": [],
                "support_quality": 0.9,
            },
            {
                "locator_id": slanted_loc,
                "source": "computed_oblique",
                "role": "ceiling",
                "corners": _square(3.0),
                "plane": {"a": 0.2, "b": 0.95, "c": 0.0, "d": -3.0},
                "adjacency": [flat_loc],
                "holes": [],
                "support_quality": 0.7,
            },
        ],
        "knee_walls": [
            {
                "locator_id": knee_loc,
                "kind": "knee",
                "corners": _square(1.5, side=2.0),
                "plane": {"a": 0.0, "b": 0.0, "c": 1.0, "d": -1.0},
            }
        ],
        "gaps": [
            {
                "locator_id": gap_loc,
                "kind": "side",
                "scope": "intra_story",
                "corners": _square(1.0, side=1.0),
                "adjacency": [],
            }
        ],
        "rooms": [
            {
                "locator_id": room_loc,
                "story": 0,
                "floor": {"corners": _square(0.0)},
                "walls": [
                    {
                        "locator_id": wall_loc,
                        "corners": _square(1.5, side=2.0),
                        "synthetic": False,
                    }
                ],
                "doors": [],
                "windows": [],
            }
        ],
        "dormer_faces": [],
        "gable_closures": [],
    }

    outputs = tmp_path / "pipeline-outputs" / UUID
    outputs.mkdir(parents=True)
    payload_path = outputs / "tier_payload.json"
    payload_path.write_text(json.dumps(payload))

    monkeypatch.setattr(qa, "PIPELINE_OUTPUTS", tmp_path / "pipeline-outputs")
    return payload_path


def test_parse_locator_roundtrip() -> None:
    uid, scope, parts = qa.parse_locator(f"{UUID}::tier-ceiling-flat::3")
    assert uid == UUID
    assert scope == "ceiling-flat"
    assert parts == ["3"]


def test_parse_locator_rejects_legacy_kinds() -> None:
    with pytest.raises(ValueError):
        qa.parse_locator(f"{UUID}::roof-oblique::5")


def test_parse_locator_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        qa.parse_locator("not-a-locator")


def test_resolve_finds_ceiling(fixture_payload: Path) -> None:
    payload, entry, scope = qa.resolve(f"{UUID}::tier-ceiling-flat::0")
    assert payload["uuid"] == UUID
    assert entry["source"] == "flat_ceiling"
    assert scope == "ceiling-flat"


def test_resolve_finds_room_and_wall(fixture_payload: Path) -> None:
    _, room, _ = qa.resolve(f"{UUID}::tier-room::0")
    assert room["story"] == 0
    _, wall, _ = qa.resolve(f"{UUID}::tier-wall::0:0")
    assert wall["synthetic"] is False


def test_resolve_missing_raises(fixture_payload: Path) -> None:
    with pytest.raises(LookupError):
        qa.resolve(f"{UUID}::tier-ceiling-flat::999")


def test_preview_make_flat_zeroes_y_variance(fixture_payload: Path) -> None:
    locator = f"{UUID}::tier-ceiling-slanted::0"
    result = qa.preview_make_flat(locator)
    ys = [c["y"] for c in result["corners"]]
    assert max(ys) - min(ys) < 1e-9
    assert result["plane"]["b"] == pytest.approx(1.0)
    assert result["plane"]["a"] == pytest.approx(0.0)


def test_preview_make_slanted_introduces_y_variance(fixture_payload: Path) -> None:
    locator = f"{UUID}::tier-ceiling-flat::0"
    result = qa.preview_make_slanted(locator, slope_deg=30.0, azimuth_deg=0.0)
    ys = [c["y"] for c in result["corners"]]
    assert max(ys) - min(ys) > 1.0  # 4m square at 30° → ~2.3m delta
    # plane normal length ~1
    p = result["plane"]
    n2 = p["a"] ** 2 + p["b"] ** 2 + p["c"] ** 2
    assert n2 == pytest.approx(1.0, abs=1e-6)


def test_preview_make_slanted_zero_slope_is_flat(fixture_payload: Path) -> None:
    locator = f"{UUID}::tier-ceiling-flat::0"
    result = qa.preview_make_slanted(locator, slope_deg=0.0)
    ys = [c["y"] for c in result["corners"]]
    assert max(ys) - min(ys) < 1e-9


def test_preview_delete_marks_locator(fixture_payload: Path) -> None:
    result = qa.preview_delete(f"{UUID}::tier-knee-wall::0")
    assert result["deleted"] is True
    assert result["scope"] == "knee-wall"


def test_preview_toggle_gap_only_for_gap_scope(fixture_payload: Path) -> None:
    qa.preview_toggle_gap(f"{UUID}::tier-gap-side::0")  # ok
    with pytest.raises(ValueError):
        qa.preview_toggle_gap(f"{UUID}::tier-ceiling-flat::0")


def test_element_info_returns_summary(fixture_payload: Path) -> None:
    info = qa.element_info(f"{UUID}::tier-ceiling-flat::0")
    assert info["scope"] == "ceiling-flat"
    assert info["source"] == "flat_ceiling"
    assert info["corner_count"] == 4
    assert "bounds" in info
    assert info["bounds"]["y"] == [pytest.approx(2.5), pytest.approx(2.5)]
    assert info["address"] == "Test 1, 2300 Copenhagen"


def test_neighbors_handles_dict_and_string_adjacency(fixture_payload: Path) -> None:
    flat = qa.neighbors(f"{UUID}::tier-ceiling-flat::0")
    assert flat["neighbors"] == [f"{UUID}::tier-ceiling-slanted::0"]
    slanted = qa.neighbors(f"{UUID}::tier-ceiling-slanted::0")
    assert slanted["neighbors"] == [f"{UUID}::tier-ceiling-flat::0"]


def test_dispatch_routes_through_registry(fixture_payload: Path) -> None:
    info = qa.dispatch("element_info", locator=f"{UUID}::tier-ceiling-flat::0")
    assert info["scope"] == "ceiling-flat"


def test_dispatch_unknown_action_raises(fixture_payload: Path) -> None:
    with pytest.raises(KeyError):
        qa.dispatch("definitely_not_a_tool", locator="x")


def test_registry_has_all_documented_actions() -> None:
    expected = {
        "preview_make_flat",
        "preview_make_slanted",
        "preview_delete",
        "preview_toggle_gap",
        "element_info",
        "neighbors",
    }
    assert set(qa.REGISTRY) == expected
    for action in qa.REGISTRY.values():
        assert action.description
