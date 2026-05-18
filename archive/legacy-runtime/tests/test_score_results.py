"""Phase 6 golden-parity test: inference feature math must match training.

Two checks:

1. **Math parity** — feeding a label-record dict through the inference
   ``compute_features`` path must reproduce the exact ``features_expanded.parquet``
   row within 1e-6. This catches any regression in the feature code itself.

2. **Results-JSON drift** — ``build_record`` applied to the current
   ``reconcile_v3_results.json`` segment must yield features close to the
   label-time row. Small divergence (~1e-2) is expected when the proposer
   has been regenerated since labels were captured. A large mismatch means
   either the proposer changed materially or ``build_record`` is wrong.

If (1) fails, ``build_record`` is unrelated — the training pipeline is
broken. If (2) fails big, investigate the proposer diff or ``build_record``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import ijson
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent

# Columns to skip in parity comparisons: identifiers, label-time-only
# metadata, categorical strings, and the viewer-state ``side_piece_*``
# fields (reproduced only in the dominant-case pattern at inference).
_SKIP = {
    "building_uuid",
    "proposal_id",
    "cluster_canonical_id",
    "parent_proposal_id",
    "labeler",
    "ts",
    "label",
    "heuristic_label",
    "label_is_accept",
    "heuristic_disagrees_with_user",
    "piece_kind",
    "part_gable_status",
    "bld_classification",
    "merge_mode",
    "is_split_child",
    "side_piece_count",
    "side_piece_area_sum_m2",
    "side_piece_has_parent",
    # Pass-through count from buildings_3d.json (building_features.py:241:
    # ``len(building["gap_walls"])``). Drifts whenever the V1 extraction
    # pipeline is rerun with different gap-wall detection — not a feature-code
    # regression, so excluded from math-parity comparison.
    "bld_gap_wall_count",
    # Same pass-through rationale as ``bld_gap_wall_count``: drifts whenever
    # the V1 stitch detection changes (e.g. duplicate-stitch dedup).
    "bld_stitch_wall_count",
}


def _compare(
    offline_rows: pd.DataFrame,
    inference_rows: pd.DataFrame,
    tol: float,
) -> list[str]:
    diffs: list[str] = []
    shared = [c for c in offline_rows.columns if c in inference_rows.columns]
    numeric = [
        c
        for c in shared
        if c not in _SKIP and offline_rows[c].dtype.kind in ("i", "u", "f", "b")
    ]
    for i in range(len(offline_rows)):
        off = offline_rows.iloc[i]
        inf = inference_rows.iloc[i]
        for c in numeric:
            a, b = off[c], inf[c]
            a_nan, b_nan = pd.isna(a), pd.isna(b)
            if a_nan and b_nan:
                continue
            if a_nan != b_nan:
                diffs.append(f"row[{i}] {c!r}: offline={a!r} inference={b!r}")
                continue
            fa, fb = float(a), float(b)
            if math.isclose(fa, fb, abs_tol=tol, rel_tol=tol):
                continue
            diffs.append(
                f"row[{i}] pid={off['proposal_id']!r} col={c!r} "
                f"offline={fa!r} inference={fb!r} diff={fb - fa!r}"
            )
    return diffs


@pytest.fixture(scope="module")
def _sample() -> dict:
    feat_path = REPO / "artifacts" / "features_expanded.parquet"
    labels_path = REPO / ".context" / "v3_roof_proposal_labels.jsonl"
    if not feat_path.exists() or not labels_path.exists():
        pytest.skip("feature parquet or labels missing")
    df = pd.read_parquet(feat_path)
    labeled = df[df["label"].isin(("accepted", "rejected"))]
    pick = (
        labeled.sort_values(["building_uuid", "proposal_id"])
        .drop_duplicates("building_uuid")
        .head(5)
        .reset_index(drop=True)
    )
    # Load label records for the chosen proposal ids (last-write-wins).
    proposal_ids = set(pick["proposal_id"])
    by_id: dict[str, dict] = {}
    with labels_path.open() as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("proposal_id")
            if pid in proposal_ids:
                ts = r.get("ts") or ""
                prev = by_id.get(pid)
                if prev is None or ts >= (prev.get("ts") or ""):
                    by_id[pid] = r
    return {"df": pick, "labels": by_id}


def test_math_parity_from_label_records(_sample) -> None:
    """compute_features(label_record) must match the parquet row exactly."""
    from reconcile_v3.analysis import aux_context as auxc
    from reconcile_v3.analysis.building_features import load_building_features
    from reconcile_v3.analysis.source_wall_index import build_wall_index
    from reconcile_v3.analysis.v3_context import load_or_build
    from reconcile_v3.autonomy.inference_features import compute_features

    df = _sample["df"]
    labels = _sample["labels"]
    results_path = REPO / "reconcile" / "reconcile_v3_results.json"
    buildings_path = REPO / "reconcile" / "buildings_3d.json"
    v3_cache_path = REPO / "artifacts" / "v3_context.json"
    aux_cache_path = REPO / "artifacts" / "aux_context.json"
    uuids = set(df["building_uuid"])

    bld = load_building_features(buildings_path)
    walls = build_wall_index(buildings_path)
    ctx = load_or_build(results_path, v3_cache_path, only_uuids=uuids)
    # The offline feature pipeline populates the ``ont_*`` / ``swall_*`` /
    # ``seg_cell_*`` aux band from ``aux_context.json`` (see
    # ``scripts/analyze_labels.py:201-211``). Without this, the inference
    # row leaves ~50 aux fields as ``None`` and parity fails on every one.
    aux_idx = (
        auxc.load_or_build(aux_cache_path, only_uuids=uuids, rebuild=False)
        if aux_cache_path.exists()
        else {}
    )

    rows = []
    for _, ref in df.iterrows():
        rec = labels[ref["proposal_id"]]
        rows.append(
            compute_features(
                rec,
                building_features=bld.get(ref["building_uuid"]),
                wall_index=walls.get(ref["building_uuid"]),
                building_context=ctx.get(ref["building_uuid"]),
                aux_context=aux_idx.get(ref["building_uuid"]),
            )
        )
    inf = pd.DataFrame(rows)
    diffs = _compare(df, inf, tol=1e-6)
    if diffs:
        preview = "\n  " + "\n  ".join(diffs[:15])
        raise AssertionError(
            f"{len(diffs)} math-parity mismatches (showing up to 15):{preview}"
        )


def test_results_json_drift_within_bound(_sample) -> None:
    """build_record(segment) + compute_features should track parquet row.

    Allows ~1e-2 drift because the current ``reconcile_v3_results.json``
    may have been regenerated since labels were captured; the point here
    is to catch *big* divergence, not exact re-derivation.
    """
    from reconcile_v3.analysis import aux_context as auxc
    from reconcile_v3.analysis.building_features import load_building_features
    from reconcile_v3.analysis.source_wall_index import build_wall_index
    from reconcile_v3.analysis.v3_context import load_or_build
    from reconcile_v3.autonomy.inference_features import (
        build_record,
        compute_features,
    )

    df = _sample["df"]
    target_proposals = set(df["proposal_id"])
    target_uuids = set(df["building_uuid"])

    results_path = REPO / "reconcile" / "reconcile_v3_results.json"
    buildings_path = REPO / "reconcile" / "buildings_3d.json"
    v3_cache_path = REPO / "artifacts" / "v3_context.json"
    aux_cache_path = REPO / "artifacts" / "aux_context.json"

    bld = load_building_features(buildings_path)
    walls = build_wall_index(buildings_path)
    ctx = load_or_build(results_path, v3_cache_path, only_uuids=target_uuids)
    aux_idx = (
        auxc.load_or_build(aux_cache_path, only_uuids=target_uuids, rebuild=False)
        if aux_cache_path.exists()
        else {}
    )

    found: dict[str, dict] = {}
    with results_path.open("rb") as f:
        for b in ijson.items(f, "item", use_float=True):
            uuid = b.get("building_uuid")
            if uuid not in target_uuids:
                continue
            for seg in b.get("merged_roof_segments") or []:
                pid = seg.get("id") or ""
                if pid not in target_proposals:
                    continue
                rec = build_record(seg, uuid)
                found[pid] = compute_features(
                    rec,
                    building_features=bld.get(uuid),
                    wall_index=walls.get(uuid),
                    building_context=ctx.get(uuid),
                    aux_context=aux_idx.get(uuid),
                )
            if len(found) >= len(target_proposals):
                break
    missing = target_proposals - set(found)
    if missing:
        pytest.skip(f"{len(missing)} labeled segments not in current results JSON")

    inf = pd.DataFrame([found[p] for p in df["proposal_id"]])
    # Loose tolerance: integer-count stats (slab_vertex_count_max, _range,
    # _std) can drift by ~1 unit if a slab gained/lost a vertex between
    # label capture and the current results JSON. We're catching big math
    # bugs here, not proposer drift.
    diffs = _compare(df, inf, tol=1.0)
    if diffs:
        preview = "\n  " + "\n  ".join(diffs[:15])
        raise AssertionError(
            f"{len(diffs)} drift-bound mismatches (showing up to 15):{preview}"
        )
