"""Field-level trust table application for reconciling merged and raw surfaces."""

from __future__ import annotations

from copy import deepcopy

from .models import MatchResult, MergeReport, Surface, Vec3

# Trust table: field -> (primary_source, fallback_source, condition)
# "raw" = per-room scan, "merged" = building-level merged, "either" = identical
TRUST_TABLE = {
    "dimensions.width": ("raw", "merged", None),
    "dimensions.height": ("raw", "merged", None),
    "dimensions.depth": ("merged", "raw", "merged_nonzero"),
    "transform": ("merged", None, None),
    "polygon_corners": ("raw", "merged", None),
    "parent_identifier": ("merged", None, None),
    "story": ("merged", None, None),
    "completed_edges": ("merged", None, None),
    "confidence": ("either", None, None),
    "category": ("either", None, None),
}

# Thresholds for flagging divergence (in cm)
DIVERGENCE_THRESHOLDS = {
    "dimensions.width": 15.0,
    "dimensions.height": 20.0,
    "dimensions.depth": 10.0,
}


def _get_field(surface: Surface, field: str):
    if field == "dimensions.width":
        return surface.dimensions.x
    elif field == "dimensions.height":
        return surface.dimensions.y
    elif field == "dimensions.depth":
        return surface.dimensions.z
    elif field == "transform":
        return surface.transform
    elif field == "polygon_corners":
        return surface.polygon_corners
    elif field == "parent_identifier":
        return surface.parent_identifier
    elif field == "story":
        return surface.story
    elif field == "completed_edges":
        return surface.completed_edges
    elif field == "confidence":
        return surface.confidence
    elif field == "category":
        return surface.category
    return None


def _set_field(surface: Surface, field: str, value):
    if field == "dimensions.width":
        surface.dimensions = Vec3(value, surface.dimensions.y, surface.dimensions.z)
    elif field == "dimensions.height":
        surface.dimensions = Vec3(surface.dimensions.x, value, surface.dimensions.z)
    elif field == "dimensions.depth":
        surface.dimensions = Vec3(surface.dimensions.x, surface.dimensions.y, value)
    elif field == "transform":
        surface.transform = value
    elif field == "polygon_corners":
        surface.polygon_corners = value
    elif field == "parent_identifier":
        surface.parent_identifier = value
    elif field == "story":
        surface.story = value
    elif field == "completed_edges":
        surface.completed_edges = value
    elif field == "confidence":
        surface.confidence = value
    elif field == "category":
        surface.category = value


def _compute_delta_cm(field: str, merged_val, raw_val) -> float | None:
    """Compute delta in cm for dimensional fields."""
    if field.startswith("dimensions."):
        if isinstance(merged_val, (int, float)) and isinstance(raw_val, (int, float)):
            return abs(merged_val - raw_val) * 100  # m -> cm
    return None


def apply_trust_table(match: MatchResult) -> tuple[Surface, MergeReport]:
    """Produce a reconciled surface by applying the trust table."""
    merged = match.merged_surface
    raw = match.raw_surface

    report = MergeReport(
        surface_id=merged.identifier,
        surface_type=match.surface_type,
    )

    if raw is None:
        # Surface only in merged (synthesized during merge) — keep as-is
        report.fields_from_merged = list(TRUST_TABLE.keys())
        report.flags.append("synthesized_in_merge")
        return deepcopy(merged), report

    reconciled = deepcopy(merged)  # Start with merged as base

    for field, (source, fallback, condition) in TRUST_TABLE.items():
        merged_val = _get_field(merged, field)
        raw_val = _get_field(raw, field)

        chosen_source = "merged"

        if source == "raw" and raw_val is not None:
            _set_field(reconciled, field, raw_val)
            chosen_source = "raw"
        elif source == "merged":
            if condition == "merged_nonzero":
                # Use merged if non-zero, otherwise fall back to raw
                is_zero = (
                    merged_val == 0
                    or merged_val is None
                    or (isinstance(merged_val, float) and abs(merged_val) < 1e-6)
                )
                if is_zero and fallback == "raw" and raw_val is not None:
                    _set_field(reconciled, field, raw_val)
                    chosen_source = "raw"
            # else keep merged value (already in reconciled)
        elif source == "either":
            # Both should be identical; keep merged
            pass

        if chosen_source == "raw":
            report.fields_from_raw.append(field)
        else:
            report.fields_from_merged.append(field)

        # Track divergence
        delta = _compute_delta_cm(field, merged_val, raw_val)
        if delta is not None:
            threshold = DIVERGENCE_THRESHOLDS.get(field)
            if threshold and delta > threshold:
                report.flags.append(f"{field}: {delta:.1f}cm divergence")
            report.divergences[field] = delta

    return reconciled, report


def merge_building(
    matches: list[MatchResult],
) -> tuple[list[Surface], list[MergeReport]]:
    """Apply trust table to all matched surfaces."""
    surfaces = []
    reports = []
    for match in matches:
        surface, report = apply_trust_table(match)
        surfaces.append(surface)
        reports.append(report)
    return surfaces, reports
