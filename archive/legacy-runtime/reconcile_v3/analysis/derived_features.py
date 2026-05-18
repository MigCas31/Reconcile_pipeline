"""Tier A derived features — §N.1 of ``reports/slanted_roof_feature_catalogue_v3.md``.

Runs *after* Bands 1-4 merge, so every input key is already on the row. We
only combine existing columns; no raw geometry / JSON is read here. Keeps
derivation auditable and cheap to regenerate.

Feature families (see catalogue §N.1 for the rationale behind each one):

* ONT.1/ONT.3 — V3-native roof-family guess + does the plane match?
* GAB.12-14 — gable-leg geometry: absolute azimuth offsets + delta to major axis.
* ADV.22 — cluster-level cross-story flag (any member bridges stories).
* SHED.2 — orphan architectural oblique (a lone pitched plane, no opposing).
* HIP.1-3 — slanted-roof-count signatures at the part level.
* PLN.8 — plane azimuth relative to the building's footprint major axis.
"""

from __future__ import annotations

from typing import Any


def _az_diff_180(a: float | None, b: float | None) -> float | None:
    """Smallest angular distance in 0..90 for ridge/major comparisons.

    We fold once into ``0..180`` (axial direction, sign-insensitive), then
    again into ``0..90`` because opposing gable legs differ by 180 °; a
    leg-vs-major comparison wants "how far from parallel."
    """
    if a is None or b is None:
        return None
    d = abs(float(a) - float(b)) % 180.0
    return float(d if d <= 90.0 else 180.0 - d)


def _family_guess(row: dict) -> str | None:
    """V3-native roof-family guess derived from gable_extension + slanted count.

    We deliberately do not run the V1 ontology pipeline — it's only in
    ``reconcile/roof_algorithms_py`` and V3Part doesn't carry the result.
    Instead we use two signals already on the row:

    * ``part_gable_status`` — the gable-extension stage's own verdict.
    * ``part_gable_metric_n_slanted_roofs`` — how many pitched planes the
      part carries.

    Values: ``"gable" | "hip" | "shed" | "flat" | "complex" | "unknown"``.
    """
    status = row.get("part_gable_status")
    n = row.get("part_gable_metric_n_slanted_roofs")
    if status == "gable_complete":
        return "gable"
    if status in {"gable_along_extend", "gable_cross_review", "gable_ambiguous"}:
        return "complex"
    try:
        k = int(n) if n is not None else None
    except (TypeError, ValueError):
        k = None
    if k is None:
        return "unknown"
    if k == 0:
        return "flat"
    if k == 1:
        return "shed"
    if k == 2:
        return "gable"
    if k == 3:
        return "complex"
    if k >= 4:
        return "hip"
    return "unknown"


def derived_features(row: dict) -> dict[str, Any]:
    """Return Tier A derived features keyed by ``derived_*``.

    ``row`` is the accumulated feature dict for a single proposal after
    Bands 1-4 have been merged — see ``scripts/analyze_labels.py``.
    """
    out: dict[str, Any] = {}

    family = _family_guess(row)
    out["derived_part_roof_family_guess"] = family
    plane_az = row.get("plane_azimuth_deg")
    major_az = row.get("part_gable_metric_major_az")
    az0 = row.get("part_gable_metric_az0")
    az1 = row.get("part_gable_metric_az1")
    ridge_az = row.get("part_gable_metric_ridge_az")

    if (
        family == "gable"
        and plane_az is not None
        and (az0 is not None or az1 is not None)
    ):
        d0 = _az_diff_180(plane_az, az0)
        d1 = _az_diff_180(plane_az, az1)
        candidates = [d for d in (d0, d1) if d is not None]
        nearest = min(candidates) if candidates else None
        out["derived_plane_matches_gable_family"] = bool(
            nearest is not None and nearest < 10.0
        )
        out["derived_plane_min_az_to_gable_leg_deg"] = nearest
    else:
        out["derived_plane_matches_gable_family"] = None
        out["derived_plane_min_az_to_gable_leg_deg"] = None

    out["derived_gable_leg_az_delta_deg"] = _az_diff_180(az0, az1)
    out["derived_ridge_vs_major_axis_deg"] = _az_diff_180(ridge_az, major_az)
    out["derived_gable_leg0_vs_major_deg"] = _az_diff_180(az0, major_az)
    out["derived_gable_leg1_vs_major_deg"] = _az_diff_180(az1, major_az)

    bld_major = row.get("bld_footprint_principal_axis_deg")
    out["derived_plane_az_vs_bld_major_deg"] = _az_diff_180(plane_az, bld_major)

    sd_max = row.get("member_story_delta_max")
    if sd_max is None:
        out["derived_cluster_any_cross_story"] = None
    else:
        try:
            out["derived_cluster_any_cross_story"] = bool(float(sd_max) > 0.0)
        except (TypeError, ValueError):
            out["derived_cluster_any_cross_story"] = None

    is_arch = row.get("plane_pitch_is_architectural")
    opp = row.get("opposing_count")
    try:
        opp_zero = opp is not None and int(opp) == 0
    except (TypeError, ValueError):
        opp_zero = False
    out["derived_is_orphan_oblique"] = (
        bool(is_arch) and opp_zero if is_arch is not None else None
    )

    try:
        k = (
            int(row.get("part_gable_metric_n_slanted_roofs"))
            if row.get("part_gable_metric_n_slanted_roofs") is not None
            else None
        )
    except (TypeError, ValueError):
        k = None
    out["derived_part_n_slanted_roofs_eq_1"] = (k == 1) if k is not None else None
    out["derived_part_n_slanted_roofs_eq_2"] = (k == 2) if k is not None else None
    out["derived_part_n_slanted_roofs_ge_4"] = (k >= 4) if k is not None else None

    return out
