"""Filesystem layout for labeler runs.

```
.context/labeler/runs/<run_id>/
    meta.json          # one RunMeta
    cases.jsonl        # one Case per line
    labels.jsonl       # one Label per line, append-only, last-write-wins per case_id
```
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from reconcile_tiers.labeler.schema import (
    Case,
    Label,
    RunMeta,
    case_to_dict,
    label_to_dict,
    run_meta_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / ".context" / "labeler" / "runs"


def runs_root() -> Path:
    root = Path(os.environ.get("LABELER_RUNS_ROOT", RUNS_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id


def list_runs() -> list[RunMeta]:
    out: list[RunMeta] = []
    for child in sorted(runs_root().iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text())
            out.append(RunMeta(**data))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def write_run(meta: RunMeta, cases: list[Case]) -> Path:
    rd = run_dir(meta.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "meta.json").write_text(json.dumps(run_meta_to_dict(meta), indent=2))
    with (rd / "cases.jsonl").open("w") as f:
        for case in cases:
            f.write(json.dumps(case_to_dict(case)) + "\n")
    (rd / "labels.jsonl").touch(exist_ok=True)
    return rd


def iter_cases(run_id: str) -> Iterator[dict[str, Any]]:
    path = run_dir(run_id) / "cases.jsonl"
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def case_count(run_id: str) -> int:
    return sum(1 for _ in iter_cases(run_id))


def get_case(run_id: str, case_id: str) -> dict[str, Any] | None:
    for raw in iter_cases(run_id):
        if raw.get("case_id") == case_id:
            return raw
    return None


def get_case_at(run_id: str, index: int) -> dict[str, Any] | None:
    for i, raw in enumerate(iter_cases(run_id)):
        if i == index:
            return raw
    return None


def labels_path(run_id: str) -> Path:
    return run_dir(run_id) / "labels.jsonl"


def append_label(run_id: str, label: Label) -> None:
    path = labels_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(label_to_dict(label)) + "\n")


def latest_labels(run_id: str) -> dict[str, dict[str, Any]]:
    """Last-write-wins per case_id."""
    out: dict[str, dict[str, Any]] = {}
    path = labels_path(run_id)
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("case_id")
            if isinstance(cid, str):
                out[cid] = rec
    return out


def latest_label(run_id: str, case_id: str) -> dict[str, Any] | None:
    return latest_labels(run_id).get(case_id)
