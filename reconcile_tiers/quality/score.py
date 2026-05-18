"""Runtime scorer for the roof_quality predictor.

Loads ``calibration.json`` once (lru_cached) and applies the linear model to a
``TierPayload`` to produce a ``RoofQuality`` instance. Wrapped in graceful
degradation: if the checkpoint is missing or fails to load, ``score_building``
returns ``None`` and the build proceeds — the schema field is optional.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from reconcile_tiers.payload.schema import RoofQuality, TierPayload
from reconcile_tiers.quality.features import extract_features, feature_names

_log = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent / "calibration.json"

# Top components surfaced in the RoofQuality.components dict — the features
# whose contribution moves the score the most. Keeping this small avoids
# bloating the per-payload JSON.
_TOP_COMPONENTS_K = 6


@lru_cache(maxsize=1)
def _load_calibration() -> dict[str, Any] | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except Exception as exc:
        _log.warning("roof_quality calibration load failed: %s", exc)
        return None


def score_building(payload: TierPayload) -> RoofQuality | None:
    """Return predicted RoofQuality or None if the checkpoint is unavailable
    or scoring fails. Never raises — runtime safety > prediction completeness."""
    calibration = _load_calibration()
    if calibration is None:
        return None
    try:
        features = extract_features(payload)
        return _score_with(features, calibration)
    except Exception as exc:
        _log.warning("roof_quality scoring failed for %s: %s", payload.uuid, exc)
        return None


def _score_with(features: dict[str, float], calibration: dict[str, Any]) -> RoofQuality:
    means = calibration["feature_means"]
    stds = calibration["feature_stds"]
    weights = calibration["coefficients"]
    intercept = float(calibration["intercept"])
    quality_version = str(calibration["quality_version"])

    contributions: dict[str, float] = {}
    raw = intercept
    for name in feature_names():
        if name not in weights:
            continue
        std = stds.get(name, 1.0) or 1.0
        z = (features.get(name, 0.0) - means.get(name, 0.0)) / std
        contribution = z * float(weights[name])
        raw += contribution
        contributions[name] = contribution

    predicted = max(1.0, min(5.0, raw))
    score = (predicted - 1.0) / 4.0

    top = dict(
        sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[
            :_TOP_COMPONENTS_K
        ]
    )
    return RoofQuality(
        score=round(score, 4),
        predicted_rating=round(predicted, 3),
        components={k: round(v, 4) for k, v in top.items()},
        quality_version=quality_version,
    )
