"""Sanity tests for roof_quality scoring + payload roundtrip."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from reconcile_tiers.payload.schema import (
    RoofQuality,
    payload_from_dict,
    payload_to_dict,
)
from reconcile_tiers.quality.score import score_building

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "pipeline-outputs"
RATINGS_PATH = Path(__file__).resolve().parents[3] / ".context" / "roof_ratings.json"


def _load_payload(uuid: str):
    path = PIPELINE_DIR / uuid / "tier_payload.json"
    if not path.exists():
        pytest.skip(f"corpus not populated: {path}")
    return payload_from_dict(json.loads(path.read_text()))


def test_roof_quality_roundtrip() -> None:
    payload = _load_payload("0430ebc2-236b-4b5d-991f-3e97ad246b78")
    rq = RoofQuality(
        score=0.81,
        predicted_rating=4.25,
        components={"a": 1.5, "b": -0.7},
        quality_version="rq-test-1",
    )
    payload_with = replace(payload, roof_quality=rq)
    data = payload_to_dict(payload_with)
    back = payload_from_dict(data)
    assert back.roof_quality == rq


def test_score_returns_predicted_rating_in_range() -> None:
    payload = _load_payload("0430ebc2-236b-4b5d-991f-3e97ad246b78")
    rq = score_building(payload)
    if rq is None:
        pytest.skip("calibration.json not present")
    assert 1.0 <= rq.predicted_rating <= 5.0
    assert 0.0 <= rq.score <= 1.0
    assert rq.quality_version
    assert rq.components, "expected at least one component contribution"


def test_holdout_spearman_above_floor() -> None:
    """End-to-end check: predicted ratings should track manual ratings.

    Computes Spearman rho on the held-out 20% (deterministic split by uuid hash)
    and asserts above the calibration's floor of 0.5.
    """
    if not RATINGS_PATH.exists():
        pytest.skip("roof_ratings.json not present")
    import hashlib

    ratings = json.loads(RATINGS_PATH.read_text())
    pairs: list[tuple[float, float]] = []
    for uuid, entry in ratings.items():
        if not isinstance(entry, dict):
            continue
        rating = entry.get("rating")
        if not isinstance(rating, int):
            continue
        digest = hashlib.sha256(uuid.encode()).digest()
        if digest[0] / 255.0 >= 0.2:
            continue
        path = PIPELINE_DIR / uuid / "tier_payload.json"
        if not path.exists():
            continue
        try:
            payload = payload_from_dict(json.loads(path.read_text()))
        except Exception:
            continue
        rq = score_building(payload)
        if rq is None:
            pytest.skip("calibration.json not present")
        pairs.append((float(rating), rq.predicted_rating))

    if len(pairs) < 10:
        pytest.skip(f"only {len(pairs)} holdout rows, not statistically meaningful")

    rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
    assert rho > 0.5, f"holdout Spearman {rho:.3f} below 0.5 floor"


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    import numpy as np

    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])
