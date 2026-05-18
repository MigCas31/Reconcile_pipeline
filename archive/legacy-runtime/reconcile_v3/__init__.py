"""reconcile_v3 — clean-slate building reconstruction pipeline.

Runs alongside the existing V1/roof/V2/ontology stack. Consumes the same
raw inputs that viewer.html steps 1 ("Apple merged") and 2 ("Room
segments") render — i.e. per-room floor polygons and wall segments — and
produces an explicit, hypothesis-traced reconstruction with first-class
unresolved regions instead of silent fallback geometry.
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
