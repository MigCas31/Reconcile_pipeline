#!/usr/bin/env python3
"""Audit non-external slanted-roof feature coverage against the exhaustive spec.

Parses ``reports/slanted_roof_features_exhaustive.md``, expands markdown table
feature shorthands, computes the currently emitted column set from the shared
feature pipeline, and writes both JSON and Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile_v3.analysis import advanced_features as advf
from reconcile_v3.analysis import context_features as ctxf
from reconcile_v3.analysis import derived_features as df
from reconcile_v3.analysis import exhaustive_features as exf
from reconcile_v3.analysis import feature_expansion as fe
from reconcile_v3.tests.test_exhaustive_features import (
    _building_context,
    _building_features,
    _record,
    _wall_index,
)

REPO = Path(__file__).resolve().parent.parent

STATUS_BLOCKED = {"NEEDS_PLUMBING", "TRAINING_ONLY"}
STATUS_SKIP = {"EXTERNAL"}
CODE_RE = re.compile(r"`([^`]+)`")
RANGE_RE = re.compile(r"^(.*?)(\d+)$")

ALIASES = {
    "plane_slope_ratio": "plane_rise_over_run",
    "drainage_azimuth_deg": "drainage_flow_azimuth_deg",
    "poly_area_m2": "poly_area_xz_m2",
    "poly_perimeter_m": "poly_perimeter_xz_m",
    "poly_solidity": "poly_convex_hull_ratio",
    "poly_aspect_ratio": "poly_bbox_aspect",
    "poly_mrr_length_m": "poly_min_rect_major_m",
    "poly_mrr_width_m": "poly_min_rect_minor_m",
    "segment_story_index_mean": "story_index_max",
    "plane_az_vs_bld_major_deg": "derived_plane_az_vs_bld_major_deg",
    "bld_footprint_aspect_ratio": "bld_footprint_bbox_aspect",
    "bld_footprint_elongation": "bld_footprint_elongation_ratio",
    "poly_outside_footprint_fraction": "poly_outside_footprint_fraction",
    "poly_centroid_utm_utm_n": "poly_centroid_utm_n",
    (
        "covered_side_count x member_heuristic_accepted_fraction"
    ): "interaction_covered_side_x_member_accept",
    (
        "drainage_to_building_center_cos x plane_incl_deg"
    ): "interaction_drainage_center_cos_x_incl",
    "member_story_delta_max x part_story_count": (
        "interaction_story_delta_x_part_story_count"
    ),
    (
        "edge_ridge_length_m x part_gable_metric_n_slanted_roofs"
    ): "interaction_ridge_len_x_part_n_slanted",
    (
        "swall_centroid_to_seg_mean_m x cluster_member_count"
    ): "interaction_swall_distance_x_member_count",
    (
        "poly_area x (1 - distance_to_footprint_edge_m/footprint_width)"
    ): "interaction_area_x_interiority",
    (
        "opposing_incl_diff_max_deg x opposing_count"
    ): "interaction_opposing_incl_diff_x_count",
    "normals_d_entropy x cluster_member_count": (
        "interaction_normals_entropy_x_cluster_member_count"
    ),
}

BLOCKED_PREFIXES = (
    "ont_",
    "v2_",
    "xm_",
    "pred_",
    "sib_",
    "lbl_",
    "meta_",
)

EXTERNAL_PREFIXES = (
    "bbr_",
    "orto_",
    "site_",
    "climate_",
    "clim_",
    "zone_",
)

TRAINING_ONLY_PREFIXES = ("label_",)

BLOCKED_EXACT = {
    "vtx_valence_max",
    "vtx_valence_min",
    "vtx_valence_distribution_entropy",
    "bld_accept_rate_history",
    "seg_cell_id",
    "seg_cell_is_pure_flat",
    "seg_cell_is_pure_oblique",
    "seg_crosses_cell_boundary_count",
    "seg_atom_sloped_state",
    "seg_subpart_semantic_kind",
    "seg_subpart_member_count",
    "seg_evidence_tier",
    "seg_projects_onto_ceiling_plane",
    "seg_ceiling_plane_azimuth_match_deg",
    "seg_hypothesis_match_selected",
    "seg_hypothesis_match_score",
    "seg_all_rooms_simple_slant",
    "seg_overlaps_flat_intermediate",
    "seg_partition_id",
    "seg_partition_area_m2",
    "trace_rule_accept_rate_prior",
}


def _expand_codes(cell: str) -> list[str]:
    codes = CODE_RE.findall(cell)
    out: list[str] = []
    prev: str | None = None
    i = 0
    while i < len(codes):
        tok = codes[i]
        if tok == "...":
            i += 1
            continue
        if i + 2 < len(codes) and codes[i + 1] == "...":
            m0 = RANGE_RE.match(tok)
            m1 = RANGE_RE.match(codes[i + 2])
            if m0 and m1 and m0.group(1) == m1.group(1):
                start = int(m0.group(2))
                stop = int(m1.group(2))
                step = 1 if start <= stop else -1
                expanded = [
                    f"{m0.group(1)}{n}" for n in range(start, stop + step, step)
                ]
                out.extend(expanded)
                prev = expanded[-1]
                i += 3
                continue
        if tok.startswith("_") and prev:
            full = prev.rsplit("_", 1)[0] + tok
            out.append(full)
            prev = full
        else:
            out.append(tok)
            prev = tok
        i += 1
    return out


def _parse_spec(path: Path) -> list[dict[str, Any]]:
    section = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section = line
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 2 or "`" not in parts[0]:
            continue
        status = None
        for part in reversed(parts):
            if part in {
                "LIVE",
                "READY",
                "NEEDS_PLUMBING",
                "EXTERNAL",
                "TRAINING_ONLY",
                "LIVE + READY",
                "LIVE (partial)",
                "LIVE (ADV)",
            }:
                status = part
                break
        for feature in _expand_codes(parts[0]):
            rows.append(
                {
                    "section": section,
                    "status": status or "UNSPECIFIED",
                    "feature": feature,
                }
            )
    return rows


def _current_columns() -> set[str]:
    record = _record()
    building_context = _building_context()
    row = fe.expand(record)
    row.update(_building_features())
    row.update(ctxf.context_features(record, building_context))
    row.update(
        advf.advanced_features(
            record, wall_index=_wall_index(), building_context=building_context
        )
    )
    row.update(
        exf.exhaustive_features(
            record,
            row,
            building_context=building_context,
            wall_index=_wall_index(),
            repo_root=REPO,
        )
    )
    row.update(df.derived_features(row))
    return set(row.keys())


def _effective_status(feature: str, status: str) -> str:
    if feature.startswith(EXTERNAL_PREFIXES):
        return "EXTERNAL"
    if feature.startswith(TRAINING_ONLY_PREFIXES):
        return "TRAINING_ONLY"
    if feature.startswith(BLOCKED_PREFIXES) or feature in BLOCKED_EXACT:
        return "NEEDS_PLUMBING"
    return status


def _is_present(feature: str, columns: set[str]) -> bool:
    if feature in columns or ALIASES.get(feature) in columns:
        return True
    if "*" in feature:
        pattern = "^" + re.escape(feature).replace("\\*", ".*") + "$"
        if any(re.match(pattern, c) for c in columns):
            return True
    if feature.endswith("_*"):
        return any(c.startswith(feature[:-1]) for c in columns)
    family_checks = {
        "poly_mrr_length_width_m": {"poly_mrr_length_m", "poly_mrr_width_m"},
        "poly_bbox_aligned_length_width_m": {
            "poly_bbox_aligned_length_m",
            "poly_bbox_aligned_width_m",
        },
        "opposing_incl_mean_min_deg": {
            "opposing_incl_mean_deg",
            "opposing_incl_min_deg",
        },
        "opposing_incl_mean_min_max_deg": {
            "opposing_incl_mean_deg",
            "opposing_incl_min_deg",
            "opposing_incl_max_deg",
        },
        "opposing_incl_mean_min_max_std_deg": {
            "opposing_incl_mean_deg",
            "opposing_incl_min_deg",
            "opposing_incl_max_deg",
            "opposing_incl_std_deg",
        },
        "swall_centroid_to_seg_mean_max_m": {
            "swall_centroid_to_seg_mean_m",
            "swall_centroid_to_seg_max_m",
        },
        "member_plane_*_spread_*": {
            "member_plane_azimuth_spread_deg",
            "member_plane_incl_spread_deg",
            "member_plane_d_spread_m",
        },
    }
    required = family_checks.get(feature)
    return required.issubset(columns) if required else False


def _render_markdown(results: list[dict[str, Any]]) -> str:
    counts = Counter(r["bucket"] for r in results)
    lines = [
        "# Exhaustive Feature Coverage Audit",
        "",
        f"- `implemented`: {counts['implemented']}",
        f"- `missing`: {counts['missing']}",
        f"- `blocked`: {counts['blocked']}",
        "",
        "| Bucket | Status | Feature | Section |",
        "|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['bucket']} | {row['status']} | `{row['feature']}` | "
            f"{row['section']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--spec", default=str(REPO / "reports" / "slanted_roof_features_exhaustive.md")
    )
    ap.add_argument(
        "--json-out",
        default=str(REPO / ".context" / "exhaustive_feature_coverage.json"),
    )
    ap.add_argument(
        "--md-out", default=str(REPO / ".context" / "exhaustive_feature_coverage.md")
    )
    args = ap.parse_args()

    spec_rows = _parse_spec(Path(args.spec))
    columns = _current_columns()
    results: list[dict[str, Any]] = []
    for row in spec_rows:
        status = _effective_status(row["feature"], row["status"])
        feature = row["feature"]
        if status in STATUS_SKIP:
            continue
        bucket = (
            "implemented"
            if _is_present(feature, columns)
            else ("blocked" if status in STATUS_BLOCKED else "missing")
        )
        results.append({**row, "bucket": bucket})

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2))
    md_out.write_text(_render_markdown(results))

    counts = Counter(r["bucket"] for r in results)
    print(json.dumps(dict(counts), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
