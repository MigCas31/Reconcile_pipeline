"""Tests for reconcile_tiers.dev_tools.

Most of these verify the confirmation-gate envelope and the registry shape;
subprocess-actually-running tests are in test_jobs.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reconcile_tiers import dev_tools, snapshots

UUID = "00000000-0000-0000-0000-000000000777"


@pytest.fixture
def fake_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pipeline = tmp_path / "pipeline-outputs" / UUID
    pipeline.mkdir(parents=True)
    payload = {
        "uuid": UUID,
        "address": "Snapshot 1",
        "rooms": [{"locator_id": f"{UUID}::tier-room::0", "walls": []}],
        "ceiling": [],
        "knee_walls": [],
        "dormer_faces": [],
        "gable_closures": [],
        "gaps": [],
    }
    (pipeline / "tier_payload.json").write_text(json.dumps(payload))

    monkeypatch.setattr(snapshots, "PIPELINE_OUTPUTS", tmp_path / "pipeline-outputs")
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", tmp_path / ".context" / "snapshots")
    monkeypatch.setattr(dev_tools, "PIPELINE_OUTPUTS", tmp_path / "pipeline-outputs")
    monkeypatch.setattr(
        dev_tools, "TRACKING_PROGRESS", tmp_path / "tracking_progress.md"
    )
    (tmp_path / "tracking_progress.md").write_text(
        "Older note about cross-floor gaps.\n\nNewer note about ceiling parity.\n\n"
        "Another paragraph mentioning cross-floor gaps and threshold tuning.\n"
    )
    return tmp_path


def test_rebuild_building_returns_confirmation_envelope() -> None:
    out = dev_tools.rebuild_building(UUID)
    assert out["requires_confirmation"] is True
    assert "command" in out
    assert UUID in out["command"]


def test_rebuild_corpus_marks_destructive() -> None:
    out = dev_tools.rebuild_corpus()
    assert out["requires_confirmation"] is True
    assert out["destructive"] is True


def test_apply_threshold_tweak_returns_envelope_with_diff() -> None:
    out = dev_tools.apply_threshold_tweak("MAX_WALL_THICKNESS_M", 0.62)
    assert out["requires_confirmation"] is True
    assert "diff" in out
    assert out["diff"]["name"] == "MAX_WALL_THICKNESS_M"


def test_apply_threshold_tweak_validates_bounds() -> None:
    with pytest.raises(ValueError):
        dev_tools.apply_threshold_tweak("MAX_WALL_THICKNESS_M", 50.0)


def test_validate_building_starts_a_job() -> None:
    out = dev_tools.validate_building(UUID)
    assert "job_id" in out
    assert UUID in out["command"]


def test_audit_building_starts_a_job() -> None:
    out = dev_tools.audit_building(UUID)
    assert "job_id" in out
    assert "cohort_scan" in out["command"]


def test_inspect_building_returns_summary(fake_workspace: Path) -> None:
    info = dev_tools.inspect_building(UUID)
    assert info["uuid"] == UUID
    assert info["counts"]["rooms"] == 1


def test_read_tracking_progress_finds_paragraphs(fake_workspace: Path) -> None:
    out = dev_tools.read_tracking_progress("cross-floor")
    assert out["match_count"] == 2
    assert any("cross-floor" in m for m in out["matches"])


def test_take_snapshot_and_compare_before_after(fake_workspace: Path) -> None:
    take = dev_tools.take_snapshot(UUID)
    assert "snapshot" in take

    payload_path = fake_workspace / "pipeline-outputs" / UUID / "tier_payload.json"
    payload = json.loads(payload_path.read_text())
    payload["ceiling"] = [{"locator_id": f"{UUID}::tier-ceiling-flat::0"}]
    payload_path.write_text(json.dumps(payload))

    diff = dev_tools.compare_before_after(UUID)
    assert diff["deltas"]["ceiling"] == 1
    assert diff["added_locator_count"] == 1


def test_registry_dispatch_unknown_raises() -> None:
    with pytest.raises(KeyError):
        dev_tools.dispatch("not_a_real_tool")


def test_registry_dev_tools_disjoint_from_quick_actions() -> None:
    """Belt-and-braces: the two registries must not share any names."""
    from reconcile_tiers import quick_actions

    assert dev_tools.REGISTRY.keys().isdisjoint(quick_actions.REGISTRY.keys())


def test_known_pivots_documented() -> None:
    assert any("AZIMUTH_FILTER_THRESHOLD" in p for p in dev_tools.KNOWN_PIVOTS)
    assert any("GBM" in p for p in dev_tools.KNOWN_PIVOTS)
