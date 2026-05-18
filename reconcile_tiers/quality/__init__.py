"""Per-building roof_quality predictor.

Fits a model to manual roof ratings (`.context/roof_ratings.json`) and emits a
predicted rating + components dict on each TierPayload. Used as a regression
metric for pipeline changes (Plan A coplanar merge, Plan C envelope work) and
as a sortable signal in the viewer for triaging the next-worst buildings.

Important: the rater set is currently a single person. The fitted model is a
**personal-preference predictor**, not an objective quality metric. Internal
tooling only — never expose via public API. ``fit_calibration.py`` refuses
to fit if more than one distinct rater is detected.
"""

from reconcile_tiers.quality.features import extract_features
from reconcile_tiers.quality.score import score_building

__all__ = ["extract_features", "score_building"]
