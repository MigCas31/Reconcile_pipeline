"""Soft-defect audit over tier_payload.json.

Hard payload invariants live in `reconcile_tiers.payload.validate`. This package
flags *soft* defects — patterns the user would notice in the viewer but that do
not violate schema invariants (e.g. ceiling normal pointing sideways, geometry
extending past the building footprint). Output is a flag queue compatible with
the viewer's manual flag pipeline.
"""

from reconcile_tiers.audit.rules import FlagItem, build_queue, run_all_rules

__all__ = ["FlagItem", "build_queue", "run_all_rules"]
