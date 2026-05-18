"""reconcile_ext — independent extension-detection pipeline.

Consumes the partial V3Building snapshot (parts + gaps + slabs + raw rooms)
plus V3's slanted roofs, emits diagnostic overlays that flag where a
reconstruction likely contains an architectural extension (aisle, lean-to,
rear addition). Read-only with respect to V3. See the research plan at
~/.claude/plans/system-instruction-you-are-working-mighty-wilkes.md.
"""
