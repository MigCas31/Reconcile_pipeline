"""Sanity tests for the threshold registry editor."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconcile_tiers import threshold_registry as tr


def test_registry_known_constants_resolve_in_source() -> None:
    """Every registered name must point at an actual module-level literal."""
    for name in tr.REGISTRY:
        value, path, line = tr.read_current(name)
        assert isinstance(value, float)
        assert Path(path).exists()
        assert line >= 1


def test_make_diff_respects_sanity_floor() -> None:
    with pytest.raises(ValueError):
        tr.make_diff("MAX_WALL_THICKNESS_M", 0.0)


def test_make_diff_respects_sanity_ceiling() -> None:
    with pytest.raises(ValueError):
        tr.make_diff("MAX_WALL_THICKNESS_M", 50.0)


def test_make_diff_returns_old_and_new_lines() -> None:
    diff = tr.make_diff("MAX_WALL_THICKNESS_M", 0.65)
    assert diff["name"] == "MAX_WALL_THICKNESS_M"
    assert diff["new"] == 0.65
    old_line = diff["diff"]["-"]
    new_line = diff["diff"]["+"]
    assert "MAX_WALL_THICKNESS_M" in old_line
    assert "0.65" in new_line
    assert old_line != new_line


def test_apply_writes_back_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    """Edit a temp copy of the source, confirm the new value reads back."""
    real_path = tr.WORKSPACE_ROOT / tr.REGISTRY["MAX_WALL_THICKNESS_M"].file
    fake_root = tmp_path
    fake_path = fake_root / tr.REGISTRY["MAX_WALL_THICKNESS_M"].file
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text(real_path.read_text())

    monkeypatch.setattr(tr, "WORKSPACE_ROOT", fake_root)

    before, _, _ = tr.read_current("MAX_WALL_THICKNESS_M")
    target = round(before + 0.01, 4)
    tr.apply("MAX_WALL_THICKNESS_M", target)
    after, _, _ = tr.read_current("MAX_WALL_THICKNESS_M")
    assert after == pytest.approx(target)
