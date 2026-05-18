"""Runs every rule + roof-defect rule and packages results into the
flag-queue schema documented in `.agents/skills/triage-queue/SKILL.md`.
"""

from __future__ import annotations

from typing import Any

from reconcile_tiers.audit.rules._shared import FlagItem
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

_RULES = (
    rule_ceiling_orientation_inverted,
    rule_ceiling_below_floor,
    rule_out_of_envelope,
    rule_ceiling_coverage_gap,
    rule_ceiling_yspan_excessive,
    rule_story_ceiling_overcount,
    rule_knee_wall_misplaced,
    rule_dormer_without_host_slope,
    rule_silent_drop,
    rule_wall_cutout_outside_outline,
)


def run_all_rules(payload: dict[str, Any]) -> list[FlagItem]:
    """Run every rule and return a flat list of FlagItems."""
    from reconcile_tiers.audit.roof_defects import ROOF_DEFECT_RULES

    items: list[FlagItem] = []
    for rule in _RULES + ROOF_DEFECT_RULES:
        items.extend(rule(payload))
    return items


def build_queue(
    building_uuid: str,
    items: list[FlagItem],
    *,
    source: str,
    created: str,
) -> dict[str, Any]:
    return {
        "schema": "flag-queue/v1",
        "building_uuid": building_uuid,
        "created": created,
        "source": source,
        "items": items,
    }
