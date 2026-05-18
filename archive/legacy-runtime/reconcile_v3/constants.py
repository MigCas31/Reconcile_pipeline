"""Global tolerances for reconcile_v3.

Single source of truth. Any stage needing a new threshold must add it
here (enforced by test_no_tolerance_drift).
"""

SCAN_NOISE_M = 0.05
"""Scan-capture noise floor; distances under this are not meaningful."""

FLAT_CEILING_SPREAD_M = 0.30
"""Wall-top spread under which a room can still be treated as flat, provided
the outlier wall is a minority of the perimeter (see OUTLIER_FRACTION_MAX).
Covers scan inaccuracies and minor architectural features (beams, steps, sills)."""

OUTLIER_FRACTION_MAX = 0.25
"""Max fraction of room perimeter allowed at the outlier Y for soft-flat
acceptance. Prevents treating a half-height or split-level room as flat."""

EAVE_OVERHANG_M = 0.30
"""Typical roof eave overhang beyond a wall line."""

CLUSTER_INCL_DEG = 15.0
"""Inclination tolerance for oblique plane clustering / slope fusion."""

KNEE_DELTA_M = 0.80
"""Wall-top height gap that flags a room as having asymmetric slope."""

CLUSTER_AZIMUTH_TOL_DEG = 30.0
"""
Max azimuth disagreement when fusing a SlopeHypothesis direction with an oblique
cluster.
"""

SLOPE_HYPOTHESIS_MIN_CONFIDENCE = 0.40
"""
Below this, a slope hypothesis emits V3UnresolvedRegion instead of boosting a cluster.
"""

MIN_OBLIQUE_SEGMENT_LEN_M = 0.30
"""Minimum oblique wall-segment length considered as roof-slope evidence."""

PART_BOUNDARY_SEARCH_M = 2.0
"""How far a room centroid may sit from a part boundary and still be
considered "on the boundary" for slope-evidence purposes."""

CONFIDENCE_CLUSTER = 0.80
"""Confidence of a SlopeHypothesis when the room has an oblique cluster hit."""

CONFIDENCE_ASYMMETRIC_ONLY = 0.55
"""Confidence when the only evidence is wall-height asymmetry on the exterior."""

CONFIDENCE_MISSING_WALL_ONLY = 0.50
"""Confidence when the only evidence is a missing exterior wall."""

CONFIDENCE_BOOST_ASYMMETRIC = 0.15
"""Added to cluster confidence when wall-height asymmetry corroborates it."""

CONFIDENCE_BOOST_MISSING_WALL = 0.10
"""Added to cluster confidence when a missing exterior wall corroborates it."""

GABLE_AZIMUTH_OPPOSING_TOL_DEG = 10.0
"""Max |Δazimuth - 180°| for two slopes to count as a symmetric gable pair."""

GABLE_INCLINATION_MATCH_TOL_DEG = 8.0
"""Max |Δinclination| between the two slopes of a gable."""

GABLE_RIDGE_HORIZONTAL_TOL = 0.08
"""Max |ridge_direction.y| (unit vector) — rejects non-horizontal ridges."""

GABLE_RIDGE_ORIENTATION_TOL_DEG = 15.0
"""Max angle between computed ridge azimuth and (slope_azimuth + 90°). Sanity
check that the plane intersection agrees with the clustered slope direction."""

GABLE_RIDGE_VS_MAJOR_TOL_DEG = 20.0
"""Max angle between the ridge and the part footprint's minimum-rotated-rectangle
major axis. Below the threshold: classical gable with extension along length.
Above: flagged as gable-cross-review (ridge runs across short axis)."""

GABLE_MIN_ELONGATION = 1.2
"""Min part footprint (major/minor) ratio for extension-grade classification.
Square-ish parts are more likely pyramid / hip roofs than gables."""

GABLE_EXTENSION_COVERAGE_MAX = 0.9
"""Max roof-corner XZ coverage of the part footprint for extension-grade.
Above this, the roof already covers the part — nothing to extend."""

GRID_BUCKET_FINE_M = 0.05
GRID_BUCKET_MEDIUM_M = 0.10
GRID_BUCKET_COARSE_M = 0.25
GRID_SNAP_FINE_TOL_M = 0.025
GRID_SNAP_COARSE_TOL_M = 0.125
VERTEX_Y_BAND_M = 0.15
VERTEX_Y_CLUSTER_SPLIT_M = 0.30
EDGE_ROLE_Y_BAND_M = 0.20
WALL_EDGE_ALIGN_DIST_M = 0.30
COPLANAR_D_TOL_M = 0.30
ROOF_REDUNDANT_AREA_M2 = 0.01
FLAT_CEILING_DOMINANCE_RATIO = 0.70
WALL_SUPPORT_BUFFER_M = 0.15
EAVE_OVERHANG_MIN_M = 0.25
THIN_SLIVER_WIDTH_M = 0.20
TINY_AREA_M2 = 0.25
PYRAMID_ASPECT_MAX = 1.50
RIDGE_HEIGHT_REASONABLE_MARGIN_M = 2.0
PLANE_INTERIOR_CLIP_MARGIN_M = 0.20
PART_SHAPE_L_SOLIDITY_MAX = 0.90
PART_SHAPE_T_SOLIDITY_MAX = 0.82
PART_SHAPE_U_SOLIDITY_MAX = 0.75
PART_RECTANGULARITY_MIN = 0.95
GABLE_UNIMODAL_RESULTANT_MIN = 0.80
