"""Soft-defect rules over tier_payload.json.

Each rule consumes the dict-form payload (as written to disk) and returns a
list of FlagItem dicts conforming to the flag-queue schema documented in
`.agents/skills/triage-queue/SKILL.md`.

Originally a 771-line module; split into one rule per sub-module plus a
`_shared` module for helpers and a `_runner` module for `run_all_rules`
and `build_queue`. This `__init__` re-exports the symbols that the audit
package's external consumers (`audit/__init__.py`, `cohort_scan.py`,
`roof_defects.py`, `geometry_overshoot_diagnostic.py`) and existing tests
import from `reconcile_tiers.audit.rules`.
"""

from __future__ import annotations

from reconcile_tiers.audit.rules._runner import build_queue, run_all_rules
from reconcile_tiers.audit.rules._shared import (
    CEILING_BELOW_FLOOR_Y_SLACK_M,
    CEILING_COVERAGE_GAP_Y_SLACK_M,
    CEILING_COVERAGE_MIN,
    CEILING_NORMAL_Y_MIN,
    CEILING_YSPAN_EXCESSIVE_M,
    DORMER_HOST_SLOPE_MAX_DIST_M,
    ENVELOPE_BUFFER_M,
    KNEE_WALL_TOP_STORY_TOL_M,
    MIN_OUTSIDE_AREA_M2,
    MIN_OUTSIDE_RATIO,
    MIN_POLYGON_AREA_M2,
    OPENING_MIN_DIM,
    RULE_HYPOTHESES,
    STORY_OVERSAT_CEILS_MIN,
    STORY_OVERSAT_ROOMS_MAX,
    STORY_RATIO_MAX,
    WALL_CUTOUT_DRIFT_TOL_M,
    FlagItem,
    _building_envelope,
    _corners_xyz,
    _corners_xz,
    _make_item,
    _newell_normal,
    _parse_locator,
    _polygon_area_3d,
    _room_floor_pieces,
    _safe_polygon,
    _story_y_bands,
    _y_range,
)
from reconcile_tiers.audit.rules.ceiling_below_floor import rule_ceiling_below_floor
from reconcile_tiers.audit.rules.ceiling_coverage_gap import rule_ceiling_coverage_gap
from reconcile_tiers.audit.rules.ceiling_orientation import (
    rule_ceiling_orientation_inverted,
)
from reconcile_tiers.audit.rules.ceiling_yspan_excessive import (
    rule_ceiling_yspan_excessive,
)
from reconcile_tiers.audit.rules.dormer_without_host import (
    rule_dormer_without_host_slope,
)
from reconcile_tiers.audit.rules.knee_wall_misplaced import rule_knee_wall_misplaced
from reconcile_tiers.audit.rules.out_of_envelope import rule_out_of_envelope
from reconcile_tiers.audit.rules.silent_drop import rule_silent_drop
from reconcile_tiers.audit.rules.story_ceiling_overcount import (
    rule_story_ceiling_overcount,
)
from reconcile_tiers.audit.rules.wall_cutout_outside import (
    rule_wall_cutout_outside_outline,
)

__all__ = [
    "CEILING_BELOW_FLOOR_Y_SLACK_M",
    "CEILING_COVERAGE_GAP_Y_SLACK_M",
    "CEILING_COVERAGE_MIN",
    "CEILING_NORMAL_Y_MIN",
    "CEILING_YSPAN_EXCESSIVE_M",
    "DORMER_HOST_SLOPE_MAX_DIST_M",
    "ENVELOPE_BUFFER_M",
    "KNEE_WALL_TOP_STORY_TOL_M",
    "MIN_OUTSIDE_AREA_M2",
    "MIN_OUTSIDE_RATIO",
    "MIN_POLYGON_AREA_M2",
    "OPENING_MIN_DIM",
    "RULE_HYPOTHESES",
    "STORY_OVERSAT_CEILS_MIN",
    "STORY_OVERSAT_ROOMS_MAX",
    "STORY_RATIO_MAX",
    "WALL_CUTOUT_DRIFT_TOL_M",
    "FlagItem",
    "_building_envelope",
    "_corners_xyz",
    "_corners_xz",
    "_make_item",
    "_newell_normal",
    "_parse_locator",
    "_polygon_area_3d",
    "_room_floor_pieces",
    "_safe_polygon",
    "_story_y_bands",
    "_y_range",
    "build_queue",
    "rule_ceiling_below_floor",
    "rule_ceiling_coverage_gap",
    "rule_ceiling_orientation_inverted",
    "rule_ceiling_yspan_excessive",
    "rule_dormer_without_host_slope",
    "rule_knee_wall_misplaced",
    "rule_out_of_envelope",
    "rule_silent_drop",
    "rule_story_ceiling_overcount",
    "rule_wall_cutout_outside_outline",
    "run_all_rules",
]
