"""Phase 1: join labels with split provenance.

Loads ``.context/v3_roof_proposal_labels.jsonl`` and returns one dict per
unique ``(building_uuid, proposal_id)`` using last-write-wins dedup. Skipped
labels are dropped. Split-child label records already carry their own
``segment_corners_xyz`` (the viewer/server snapshots the child polygon at
label time), so the splits file is not needed to resolve corners. We still
load it to compute the ``is_split_child`` flag and record ``parent_id``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_split_index(splits_path: Path) -> dict[str, str]:
    """Map child proposal_id → parent proposal_id from the splits file."""
    child_to_parent: dict[str, str] = {}
    if not splits_path.exists():
        return child_to_parent
    for rec in _iter_jsonl(splits_path):
        parent = rec.get("parent_id") or rec.get("original_id")
        for child in rec.get("children", []) or []:
            cid = child.get("id")
            if cid and parent:
                child_to_parent[cid] = parent
    return child_to_parent


def load_labels(
    labels_path: Path,
    splits_path: Path | None = None,
    *,
    drop_skipped: bool = True,
) -> list[dict]:
    """Read labels with last-write-wins dedup.

    Key is ``(building_uuid, proposal_id)``. When ``drop_skipped`` is True
    (default), records with ``label == "skipped"`` are removed after dedup.
    Attaches ``is_split_child`` and ``parent_proposal_id`` when the splits
    file is provided.
    """
    dedup: dict[tuple[str, str], dict] = {}
    for rec in _iter_jsonl(labels_path):
        key = (rec.get("building_uuid"), rec.get("proposal_id"))
        if not all(key):
            continue
        ts = rec.get("ts") or ""
        prev = dedup.get(key)
        if prev is None or (ts and ts >= (prev.get("ts") or "")):
            dedup[key] = rec

    child_to_parent: dict[str, str] = {}
    if splits_path is not None:
        child_to_parent = load_split_index(splits_path)

    out: list[dict] = []
    for rec in dedup.values():
        if drop_skipped and rec.get("label") == "skipped":
            continue
        pid = rec.get("proposal_id") or ""
        parent = child_to_parent.get(pid)
        rec = dict(rec)
        rec["is_split_child"] = bool(parent) or ("#" in pid)
        rec["parent_proposal_id"] = parent
        out.append(rec)
    return out


def resolution_report(labels: list[dict]) -> dict:
    """Summary counts useful for verifying Phase 1 completeness."""
    from collections import Counter

    by_label = Counter(r.get("label") for r in labels)
    by_kind = Counter(r.get("kind") for r in labels)
    have_corners = sum(1 for r in labels if r.get("segment_corners_xyz"))
    have_plane = sum(1 for r in labels if r.get("merged_plane"))
    have_cluster = sum(1 for r in labels if r.get("cluster_members"))
    have_boundary = sum(1 for r in labels if r.get("building_boundary_xz"))
    have_opposing = sum(1 for r in labels if r.get("opposing_planes"))
    have_side = sum(1 for r in labels if r.get("side_pieces"))
    n_buildings = len({r.get("building_uuid") for r in labels})
    n_split_children = sum(1 for r in labels if r.get("is_split_child"))
    return {
        "total": len(labels),
        "buildings": n_buildings,
        "by_label": dict(by_label),
        "by_kind": dict(by_kind),
        "have_segment_corners_xyz": have_corners,
        "have_merged_plane": have_plane,
        "have_cluster_members": have_cluster,
        "have_building_boundary_xz": have_boundary,
        "have_opposing_planes": have_opposing,
        "have_side_pieces": have_side,
        "split_children": n_split_children,
    }
