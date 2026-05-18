"""Sanity tests for roof_quality feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reconcile_tiers.payload.schema import payload_from_dict
from reconcile_tiers.quality.features import (
    MAX_PIECES_FOR_PAIRS,
    extract_features,
    feature_names,
)

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "pipeline-outputs"

# A handful of buildings spanning the rating x tier matrix. If the corpus
# isn't populated locally these tests skip (CI without scan data still passes).
EXEMPLARS = {
    "tier1_R5": "0430ebc2-236b-4b5d-991f-3e97ad246b78",
    "tier6_R5": "513a4b03-cfb1-4221-99ff-d9b7d1e1d3f6",
    "tier7_R2": "0d3f2993-8386-4130-8f1c-b2938c410828",
    "tier8_R1": "59b505e7-b384-451b-90b1-80f2654dd10d",
}


def _load(uuid: str):
    path = PIPELINE_DIR / uuid / "tier_payload.json"
    if not path.exists():
        pytest.skip(f"corpus not populated: {path}")
    return payload_from_dict(json.loads(path.read_text()))


def test_feature_names_match_extracted_keys() -> None:
    payload = _load(EXEMPLARS["tier1_R5"])
    feats = extract_features(payload)
    assert set(feats.keys()) == set(feature_names())


def test_features_are_floats() -> None:
    payload = _load(EXEMPLARS["tier7_R2"])
    feats = extract_features(payload)
    for key, value in feats.items():
        assert isinstance(value, float), f"{key} is {type(value).__name__}"


def test_tier_one_hot_exclusive() -> None:
    payload = _load(EXEMPLARS["tier1_R5"])
    feats = extract_features(payload)
    tier_cols = {f"tier_{i}": feats[f"tier_{i}"] for i in range(1, 9)}
    active = [k for k, v in tier_cols.items() if v == 1.0]
    assert len(active) == 1


def test_overlap_ratio_low_for_tier1() -> None:
    payload = _load(EXEMPLARS["tier1_R5"])
    feats = extract_features(payload)
    # Tier 1 single-storey flat buildings tile the bbox without overlap.
    assert feats["overlap_ratio"] < 1.2, feats["overlap_ratio"]


def test_overlap_ratio_or_pieces_high_for_bad_buildings() -> None:
    payload = _load(EXEMPLARS["tier8_R1"])
    feats = extract_features(payload)
    # The R1 building should fail at least one of these signals.
    bad = (
        feats["overlap_ratio"] > 1.2
        or feats["ceiling_piece_count"] > 25
        or feats["max_pair_normal_delta_deg"] > 60
    )
    assert bad, feats


def test_max_pair_delta_zero_for_flat() -> None:
    payload = _load(EXEMPLARS["tier1_R5"])
    feats = extract_features(payload)
    # Flat ceilings -- all normals point straight up, max Δ stays at 0.
    assert feats["max_pair_normal_delta_deg"] < 1.0


def test_pair_cap_bounds_runtime() -> None:
    # Direct contract: the cap exists and is a small positive number.
    assert 10 < MAX_PIECES_FOR_PAIRS < 200
