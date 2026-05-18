"""Offline analysis of V3 roof proposal labels.

Phase 1 (``join``) joins the HITL label store with split-child provenance.
Phase 2 (``feature_expansion``) computes the Band 1 + Band 2 feature vector
(~300 features per labeled segment) from fields embedded in each label record.
Both phases operate on ``.context/v3_roof_proposal_labels.jsonl`` — no need
to re-read ``reconcile_v3_results.json`` (which is ~2GB).
"""
