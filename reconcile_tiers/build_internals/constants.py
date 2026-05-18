"""Module-level threshold constants used across `reconcile_tiers.build` and its
helpers. Extracted from `build.py` to keep the orchestrator file readable.

Constants are re-exported from `reconcile_tiers.build` so existing callers
continue to import them from there.
"""

from __future__ import annotations

KINK_MIN_SPLIT_HEIGHT_M = 0.05
# Top-story gable rooms often contain tiny inclined raw-plane fragments from
# the synthetic roof edge. Only let the scan-driven kink split override gable
# ownership when the sloped patch is large enough to be a real ceiling region.
TOP_GABLE_KINK_MIN_SLOPE_AREA_M2 = 1.0
ROOM_OBLIQUE_FLAT_LID_TOL_M = 0.10
# bucket the transparent gable shell separately so it isn't subtracted by
# interior solids or lids
ATTIC_SHELL_STORY = -1
UNSUPPORTED_KINK_FLAT_LID_MIN_COVERAGE_RATIO = 0.50
UNSUPPORTED_KINK_SLOPE_MIN_DROP_M = 0.90
UNSUPPORTED_KINK_SHALLOW_SLOPE_MAX_INCL_DEG = 10.0
UNSUPPORTED_KINK_LOCAL_NEIGHBOR_TOL_M = 0.15
UNSUPPORTED_KINK_LOCAL_NEIGHBOR_MIN_OVERLAP_M2 = 0.01
UNSUPPORTED_KINK_FLAT_NEIGHBOR_LID_TOL_M = 0.35

# Coverage gate for emission-time priors. Mirrors the gate used in
# `roof/roof.py` so Layers 1-3 turn on together for the same buildings.
_PRIORS_COVERAGE_MIN = 0.70

# Floor-coverage threshold for emitting a synthesised flat ceiling. If
# roof.flat covers less of the room than this fraction, we don't trust the
# synthesis and let the existing RAW_FALLBACK path handle it.
_SYNTHESIS_FLAT_COVERAGE_MIN = 0.70

# Max residual (m) for the wall-derived `ceiling_polygon` to count as planar.
# Above this, the polygon spans multiple ceiling planes (e.g., L-shaped room
# with a tall section) and a single Plane.fit produces a compromise plane
# that doesn't match either side. In that case Step 3 skips and lets
# RAW_FALLBACK keep the per-plane scan emissions.
_CEILING_POLYGON_PLANARITY_TOL_M = 0.10
_SYNTHESIS_FLAT_OBLIQUE_RAW_MIN_AREA_M2 = 0.5
_SYNTHESIS_FLAT_OBLIQUE_RAW_COVERAGE_MIN = 0.70
_WITHIN_STORY_FREE_EDGE_BOUNDARY_NEAR_M = 0.25
_WITHIN_STORY_FREE_EDGE_SUPPORT_BUFFER_M = 0.10
_WITHIN_STORY_FREE_EDGE_MIN_SUPPORT_RATIO = 0.50
_WITHIN_STORY_FREE_EDGE_MAX_SOURCE_DEFORMATION_M = 0.30
_WITHIN_STORY_FREE_EDGE_MAX_SOURCE_DEFORMATION_RATIO = 0.55

FLAT_HORIZONTAL_Y_SPAN_M = 0.05
RAW_TO_OBLIQUE_MAX_AZIMUTH_DELTA_DEG = 20.0
RAW_TO_OBLIQUE_MAX_INCL_DELTA_DEG = 15.0
RAW_TO_OBLIQUE_MAX_Y_DELTA_M = 0.75
GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG = 40.0  # deviation from 180° opposing
GABLE_PARTNER_INCL_TOLERANCE_DEG = 10.0
GABLE_PAIR_MIN_ROOM_OVERLAP_RATIO = 0.5
GABLE_RIDGE_Y_EPSILON_M = 0.05
ROOF_WALL_CLIP_EPS_M = 1e-5
RAW_UPPER_SLAB_MIN_OVERLAP_M2 = 0.10
RAW_UPPER_SLAB_MIN_OVERLAP_RATIO = 0.25
RAW_UPPER_SLAB_VERTICAL_EPS_M = 0.05
SYNTHETIC_GABLE_OBSERVED_Y_TOL_M = 0.6
SPLIT_LEVEL_EXPOSED_MIN_AREA_M2 = 0.05

# Noisy raw-ceiling gate. Phrased as physical thresholds, not curve fits:
# - REDUNDANT_LEFTOVER_MAX_M2: drop a raw plane when the painter would only
#   emit a sliver after subtracting higher-priority candidates on the same
#   story. Phrased as a leftover floor (not a coverage ratio) so a 16 m^2
#   plane survives at 70% coverage (4.8 m^2 leftover = real signal) while a
#   1 m^2 plane drops at 60% coverage (0.4 m^2 leftover = visual fringe).
# - JUNK_TRIANGLE_*: a 3-corner, sub-0.6 m^2, tilted (>30 cm Y-span) fragment
#   is a RoomPlan plane-fitting artifact. No real ceiling region looks like
#   this; the cohort histogram across 223 buildings shows 109 such fragments
#   and zero of them are anything a human would label as a ceiling.
# - LOW_QUALITY_THRESHOLD: mirror the v2 thermal-envelope gate
#   (assemble/thermal_envelope.py: RAW_QUALITY_THRESHOLD = 0.55) so v1 and v2
#   payloads agree on which raws are noise. Only applies to FLAT-classified
#   rooms -- sloped rooms keep raws ungated since they ARE the ceiling source.
RAW_CEILING_REDUNDANT_LEFTOVER_MAX_M2 = 0.5
# Relative-coverage rule: when synthesised/arrangement geometry covers a
# large fraction of a raw plane's XZ footprint, treat the raw as redundant
# even if the absolute leftover area is non-trivial. Multi-wing buildings
# produce many small arrangement fragments; each fragment can leave 1-2 m^2
# of leftover but together they reliably cover the raw, so the absolute
# 0.5 m^2 rule alone leaves the raw on top of the synthesis.
RAW_CEILING_REDUNDANT_COVERAGE_RATIO = 0.70
RAW_CEILING_REDUNDANT_Y_TOL_M = 0.50
RAW_CEILING_PEAK_YSPAN_MIN_M = 0.75
RAW_CEILING_JUNK_TRIANGLE_MAX_AREA_M2 = 1.0
RAW_CEILING_JUNK_TRIANGLE_MIN_YSPAN_M = 0.30
RAW_CEILING_LOW_QUALITY_THRESHOLD = 0.55
CEILING_RING_MIN_EDGE_M = 0.02
CEILING_RING_MIN_AREA_M2 = 0.01
ROOM_RAW_OWNER_MIN_AREA_M2 = 0.5
ROOM_RAW_OWNER_FLAT_MAX_INCL_DEG = 5.0
ROOM_RAW_OWNER_MIN_INCL_DEG = 5.0
ROOM_RAW_OWNER_MAX_INCL_DEG = 75.0
ROOM_RAW_OWNER_COMPATIBLE_COVERAGE = 0.70
GABLE_OWNED_SLOPE_MIN_SHELL_COVERAGE = 0.70
GABLE_OWNED_PARTIAL_SLOPE_MAX_ROOM_COVERAGE = 0.70
KINK_FLAT_RIDGE_ARTIFACT_WALLTOP_YSPAN_M = 0.50
KINK_FLAT_RIDGE_ARTIFACT_Y_TOL_M = 0.15
KINK_FLAT_RIDGE_ARTIFACT_MAX_FLAT_FLOOR_RATIO = 0.35
KINK_FLAT_RIDGE_ARTIFACT_MAX_SELECTED_SLOPE_RATIO = 0.25
KINK_FLAT_RIDGE_ARTIFACT_MIN_RAW_OBLIQUE_RATIO = 0.50
KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_M2 = 0.50
KINK_FLAT_RIDGE_ARTIFACT_MIN_ROOF_OVERLAP_RATIO = 0.25
KINK_FLAT_RIDGE_ARTIFACT_ROOF_Y_CLEARANCE_M = 0.30

MIRROR_OBLIQUE_MIN_AREA_M2 = 0.50
"""Skip mirror synthesis below this area -- slivers create render noise."""

MIRROR_OBLIQUE_MIN_DOMAIN_OVERLAP_M2 = 0.20
"""Oblique's physical XZ polygon must overlap the domain by at least this
much. Filters distant obliques whose plane happens to extrapolate over
the room -- only neighbouring gables qualify as ridge sources."""

_DIP_TEST_GRID_M = 0.30
_DIP_TEST_EPS_Y = 0.10

_DUPLICATE_WALL_DIRECTION_DOT_MIN = 0.98
_DUPLICATE_WALL_ENDPOINT_TOL_M = 0.03
_DUPLICATE_WALL_OVERLAP_RATIO_MIN = 0.80

_PIECE_DEDUP_INTERSECT_RATIO = 0.1
_PIECE_DEDUP_INTERSECT_AREA_M2 = 0.5
_PIECE_REDUNDANT_INTERSECT_RATIO = 0.95

_NEAR_VERTICAL_NORMAL_Y_ABS_MAX = 0.95

_CAVITY_CLOSURE_OVERLAP_RATIO_MIN = 0.70

_HYBRID_DOMAIN_MIN_AREA_M2 = 0.05
