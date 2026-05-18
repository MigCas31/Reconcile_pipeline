"""Tolerances for the extension-detection pipeline.

Single source of truth. Every stage documents *why* a threshold has the
value it does; changing one should be justified by corpus evidence, not a
single building.
"""

# --- Reflex-vertex scanner --------------------------------------------------

REFLEX_MIN_EDGE_M = 0.5
"""Adjacent edges to a reflex vertex must each be at least this long for the
vertex to be considered an extension-candidate seam. Filters out bay windows,
bow-window notches, and scan-noise spurs. The merged footprint is simplified
first (``FOOTPRINT_SIMPLIFY_TOL_M``) so true scan-noise slivers don't survive
to this stage."""

REFLEX_MIN_INTERIOR_ANGLE_DEG = 200.0
"""Minimum interior angle (measured through the building interior) to count
as a "real" reflex. 180° = collinear; 200° avoids near-collinear noise while
still catching real L-shape and T-shape notches."""

REFLEX_MIN_NOTCH_DEPTH_M = 0.60
"""Minimum concavity depth — how far the reflex vertex sits from the convex
hull of the footprint. Extensions carve a notch at least this deep; anything
shallower is likely a bay window or scan artefact."""

FOOTPRINT_SIMPLIFY_TOL_M = 0.15
"""Douglas-Peucker tolerance applied to the merged footprint before reflex
enumeration. Collapses scan-noise jogs (typical amplitude 5-10cm) without
losing true corners (typical amplitude ≥ 20cm)."""

# --- Roof-discontinuity detector -------------------------------------------

ROOF_ADJACENCY_DISTANCE_M = 1.5
"""Two slanted roof footprints are considered "adjacent" if their XZ bounds
are within this distance. Pairs beyond this don't interact directly."""

RIDGE_DELTA_STRONG_M = 0.80
"""Ridge-height Δh that almost certainly indicates a part seam (main house
vs extension)."""

RIDGE_DELTA_WEAK_M = 0.30
"""Ridge-height Δh that is *suggestive* — needs corroboration from another
channel to count as a seam."""

EAVE_DELTA_STRONG_M = 0.50
"""Eave-height Δh that strongly suggests an extension's eave sits below the
main roof's eave."""

AZIMUTH_MISMATCH_STRONG_DEG = 45.0
"""Planar-normal azimuth difference between adjacent roofs that strongly
suggests independent roof systems."""

INCLINATION_MISMATCH_STRONG_DEG = 15.0
"""Pitch difference between adjacent roofs that suggests independent roof
systems (main roof at 35° vs shed at 15°, say)."""

# --- Wall-plane step detector ----------------------------------------------

WALL_AZIMUTH_BUCKET_DEG = 5.0
"""Walls within this azimuth range are considered "parallel enough" to be
candidates for a coplanarity / step check."""

WALL_STEP_MIN_M = 0.30
"""Below this, the wall offset is within wall-thickness / scan-noise and
probably just the inside vs. outside of the same wall."""

WALL_STEP_MAX_M = 2.5
"""Above this, the offset is too large to plausibly be an extension seam —
it's a separate building or a major plan feature."""

WALL_PROJECTION_OVERLAP_MIN_M = 0.50
"""Two walls must project onto their shared-azimuth axis with at least this
much overlap to count as collinear neighbours."""

# --- Ceiling / floor delta detector ----------------------------------------

CEILING_DELTA_STRONG_M = 0.40
"""Ceiling-Y difference between adjacent rooms that signals a structural
step — typical for a single-story extension below a two-story main house."""

FLOOR_DELTA_STRONG_M = 0.20
"""Floor-Y difference between adjacent rooms that suggests one is on a
slab-on-grade extension while the other is on the main foundation."""

# --- Aggregation / scoring --------------------------------------------------

TIER1_EVIDENCE_WEIGHT = 1.0
TIER2_EVIDENCE_WEIGHT = 0.5
MIN_COMPOSITE_SCORE = 0.75
"""Above this combined score, we call a seam "confident" in the report.
Below — diagnostic only."""
