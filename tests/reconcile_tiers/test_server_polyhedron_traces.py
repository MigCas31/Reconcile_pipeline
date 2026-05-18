"""Tests for polyhedron trace index URL helpers on the tier server."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconcile_tiers.server import (
    configure_polyhedron_traces_dir,
    trace_index_public_url,
)


def test_trace_index_public_url_relative_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    trace_dir = workspace / ".context" / "polyhedron-corpus-traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "index.json").write_text("{}", encoding="utf-8")

    url = trace_index_public_url(trace_dir, workspace_root=workspace)
    assert url == "/.context/polyhedron-corpus-traces/index.json"


def test_trace_index_public_url_missing_index(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    trace_dir = workspace / ".context" / "empty"
    trace_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        trace_index_public_url(trace_dir, workspace_root=workspace)


def test_configure_polyhedron_traces_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    trace_dir = workspace / ".context" / "polyhedron-repair-steps"
    trace_dir.mkdir(parents=True)
    (trace_dir / "index.json").write_text("{}", encoding="utf-8")

    url = configure_polyhedron_traces_dir(trace_dir, workspace_root=workspace)
    assert url == "/.context/polyhedron-repair-steps/index.json"
    configure_polyhedron_traces_dir(None, workspace_root=workspace)
