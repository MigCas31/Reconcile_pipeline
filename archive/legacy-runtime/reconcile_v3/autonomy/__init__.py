"""Phase 6: scoring post-processor for merged roof segments.

Reads ``reconcile_v3_results.json``, computes the same 395-feature vector
used in training for every merged roof segment, evaluates the depth-4
reject rules + the calibrated LightGBM ensemble, and writes a mirror JSON
with ``score`` and ``autonomy_label`` added to each segment.
"""
