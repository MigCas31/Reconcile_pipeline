"""Cohort-wide roof-defect classifier and report.

For every building under `pipeline-outputs/`, compute:

    - building principal axis (from wall directions)
    - rectilinearity coverage (gate at 0.70 = "primitive-library candidate")
    - per-piece roof-defect flags (off-axis, fragmented, eave-not-parallel,
      cross-extension)
    - per-building roof_irregular_footprint flag (rectilinearity gate)

Aggregate counts to answer:

    Q1: What fraction of buildings have at least one off-axis oblique?
    Q2: What fraction of buildings would be fully resolved by yaw-snap +
        continuity merge?
    Q3: What fraction are pure-irregular (need mesh-patch fallback)?

Threshold sensitivity: the report sweeps off-axis tolerance over
{5, 10, 15, 20} deg so we see how aggressive the gate has to be before
the headline number changes.

Outputs three artifacts under `.context/cohort-defect-scan/<timestamp>/`:

    roof_defect_summary.json   cohort totals + thresholds
    per_building.csv           one row per UUID
    REPORT.md                  human-readable histogram + headline answer

Usage:
    python -m reconcile_tiers.audit.cohort_roof_report
    python -m reconcile_tiers.audit.cohort_roof_report --root pipeline-outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reconcile_tiers.audit.roof_defects import (
    OFF_AXIS_TOL_DEG,
    RECTILINEARITY_GATE,
    _angle_mod90_delta,
    _building_axis,
    _oblique_pieces,
    rule_roof_cross_extension,
    rule_roof_fragmented,
    rule_roof_irregular_footprint,
    rule_roof_off_axis,
)

OFF_AXIS_TOLERANCE_SWEEP_DEG = (5.0, 10.0, 15.0, 20.0)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _payload_path(building_dir: Path) -> Path | None:
    path = building_dir / "tier_payload.json"
    return path if path.exists() else None


def _per_building_metrics(payload: dict[str, Any]) -> dict[str, Any] | None:
    info = _building_axis(payload)
    if info is None:
        return None
    axis, coverage, _ = info
    pieces = _oblique_pieces(payload)
    n_obliques = len(pieces)
    is_irregular = coverage < RECTILINEARITY_GATE

    deltas = (
        [_angle_mod90_delta(p["azimuth"] % 90.0, axis) for p in pieces]
        if not is_irregular
        else []
    )
    off_axis_by_tol: dict[float, int] = {}
    for tol in OFF_AXIS_TOLERANCE_SWEEP_DEG:
        off_axis_by_tol[tol] = sum(1 for d in deltas if d > tol)

    fragmented = len(rule_roof_fragmented(payload))
    cross_extension = len(rule_roof_cross_extension(payload))
    irregular_flag = len(rule_roof_irregular_footprint(payload))
    off_axis_default = len(rule_roof_off_axis(payload))

    return {
        "axis_deg": float(axis),
        "wall_axis_coverage": float(coverage),
        "n_obliques": n_obliques,
        "is_irregular": is_irregular,
        "n_off_axis_by_tol": off_axis_by_tol,
        "n_off_axis_default": off_axis_default,
        "n_fragmented": fragmented,
        "n_cross_extension": cross_extension,
        "n_irregular_flag": irregular_flag,
        "median_slope_axis_delta_deg": (
            sorted(deltas)[len(deltas) // 2] if deltas else None
        ),
    }


def _scan_cohort(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cohort_off_axis = {tol: 0 for tol in OFF_AXIS_TOLERANCE_SWEEP_DEG}
    totals = Counter()
    skipped: list[str] = []
    errors: list[dict[str, str]] = []

    buildings = sorted(p for p in root.iterdir() if p.is_dir())
    for building_dir in buildings:
        uuid = building_dir.name
        payload_path = _payload_path(building_dir)
        if payload_path is None:
            skipped.append(uuid)
            continue
        try:
            payload = json.loads(payload_path.read_text())
            metrics = _per_building_metrics(payload)
        except Exception as exc:
            errors.append({"uuid": uuid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if metrics is None:
            skipped.append(uuid)
            continue
        metrics["uuid"] = uuid
        rows.append(metrics)
        for tol, count in metrics["n_off_axis_by_tol"].items():
            cohort_off_axis[tol] += count
        totals["n_obliques"] += metrics["n_obliques"]
        totals["n_fragmented"] += metrics["n_fragmented"]
        totals["n_cross_extension"] += metrics["n_cross_extension"]
        totals["n_irregular_buildings"] += int(metrics["is_irregular"])
        totals["n_buildings"] += 1
        totals["n_buildings_with_obliques"] += int(metrics["n_obliques"] > 0)

    n_buildings = totals["n_buildings"]
    n_buildings_with_obliques = totals["n_buildings_with_obliques"]
    n_irregular = totals["n_irregular_buildings"]

    # Buildings with at least one defect of each class (for "% buildings affected"
    # stats).
    affected: dict[str, int] = {
        "off_axis_default": sum(1 for r in rows if r["n_off_axis_default"] > 0),
        "fragmented": sum(1 for r in rows if r["n_fragmented"] > 0),
        "cross_extension": sum(1 for r in rows if r["n_cross_extension"] > 0),
        "irregular": n_irregular,
    }

    # Buildings whose only defects are in the priors-fixable set (off-axis,
    # fragmented, cross-extension) and which are NOT already irregular.
    # Cross-extension is included because per-part decomposition + per-part
    # primitive fitting is part of the planned primitive-library approach.
    priors_fixable = sum(
        1
        for r in rows
        if not r["is_irregular"]
        and (
            r["n_off_axis_default"] > 0
            or r["n_fragmented"] > 0
            or r["n_cross_extension"] > 0
        )
    )

    return {
        "schema": "cohort-roof-defect-report/v1",
        "thresholds": {
            "off_axis_tol_deg": OFF_AXIS_TOL_DEG,
            "off_axis_sweep_deg": list(OFF_AXIS_TOLERANCE_SWEEP_DEG),
            "rectilinearity_gate": RECTILINEARITY_GATE,
        },
        "totals": dict(totals),
        "off_axis_count_by_tolerance": cohort_off_axis,
        "buildings_affected": affected,
        "buildings_priors_fixable": priors_fixable,
        "buildings_with_obliques": n_buildings_with_obliques,
        "n_buildings": n_buildings,
        "n_skipped_no_payload": len(skipped),
        "n_errors": len(errors),
        "skipped": skipped[:20],
        "errors": errors[:20],
        "rows": rows,
    }


def _write_per_building_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = [
        "uuid",
        "axis_deg",
        "wall_axis_coverage",
        "is_irregular",
        "n_obliques",
        "n_off_axis_default",
        "n_off_axis_5deg",
        "n_off_axis_10deg",
        "n_off_axis_15deg",
        "n_off_axis_20deg",
        "n_fragmented",
        "n_cross_extension",
        "n_irregular_flag",
        "median_slope_axis_delta_deg",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            tol_counts = r["n_off_axis_by_tol"]
            writer.writerow(
                {
                    "uuid": r["uuid"],
                    "axis_deg": f"{r['axis_deg']:.2f}",
                    "wall_axis_coverage": f"{r['wall_axis_coverage']:.3f}",
                    "is_irregular": int(r["is_irregular"]),
                    "n_obliques": r["n_obliques"],
                    "n_off_axis_default": r["n_off_axis_default"],
                    "n_off_axis_5deg": tol_counts.get(5.0, 0),
                    "n_off_axis_10deg": tol_counts.get(10.0, 0),
                    "n_off_axis_15deg": tol_counts.get(15.0, 0),
                    "n_off_axis_20deg": tol_counts.get(20.0, 0),
                    "n_fragmented": r["n_fragmented"],
                    "n_cross_extension": r["n_cross_extension"],
                    "n_irregular_flag": r["n_irregular_flag"],
                    "median_slope_axis_delta_deg": (
                        f"{r['median_slope_axis_delta_deg']:.1f}"
                        if r["median_slope_axis_delta_deg"] is not None
                        else ""
                    ),
                }
            )


def _pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "0/0 (-)"
    return f"{numer}/{denom} ({100.0 * numer / denom:.1f}%)"


def _write_report_md(summary: dict[str, Any], path: Path) -> None:
    rows = summary["rows"]
    n_buildings = summary["n_buildings"]
    n_with_obliques = summary["buildings_with_obliques"]
    affected = summary["buildings_affected"]
    n_irregular = affected["irregular"]
    n_rectilinear = n_buildings - n_irregular
    priors_fixable = summary["buildings_priors_fixable"]
    sweep = summary["off_axis_count_by_tolerance"]

    top20 = sorted(
        rows,
        key=lambda r: (
            -(r["n_off_axis_default"] + r["n_fragmented"] + r["n_cross_extension"])
        ),
    )[:20]

    lines: list[str] = []
    lines.append("# Cohort roof-defect report")
    lines.append("")
    lines.append(f"Created: {datetime.now(UTC).isoformat()}")
    lines.append(f"Buildings scanned: {n_buildings}")
    lines.append(f"Buildings with oblique pieces: {n_with_obliques}")
    lines.append(
        f"Errors: {summary['n_errors']}, no-payload skipped: "
        f"{summary['n_skipped_no_payload']}"
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(
        f"- **Irregular footprints (need mesh-patch fallback):** "
        f"{_pct(n_irregular, n_buildings)}"
    )
    lines.append(
        f"- **Rectilinear, primitive-library candidates:** "
        f"{_pct(n_rectilinear, n_buildings)}"
    )
    lines.append(
        f"- **Buildings with at least one priors-fixable defect:** "
        f"{_pct(priors_fixable, n_buildings)}"
    )
    lines.append(
        f"- **Buildings priors-fixable / rectilinear:** "
        f"{_pct(priors_fixable, n_rectilinear)}"
    )
    lines.append("")
    lines.append("## Buildings affected, by defect class")
    lines.append("")
    lines.append("| Class | Buildings | % of all | % of rectilinear |")
    lines.append("|---|---|---|---|")
    for label, key, denom in (
        ("roof_off_axis (default 15deg)", "off_axis_default", n_rectilinear),
        ("roof_fragmented", "fragmented", n_buildings),
        ("roof_cross_extension", "cross_extension", n_buildings),
        ("roof_irregular_footprint", "irregular", n_buildings),
    ):
        n = affected[key]
        lines.append(
            f"| {label} | {n} | {_pct(n, n_buildings)} | "
            f"{_pct(n, denom) if denom else '-'} |"
        )
    lines.append("")
    lines.append("## Off-axis sensitivity sweep (cohort piece counts)")
    lines.append("")
    lines.append(
        "| Tolerance | Pieces flagged | % of obliques on rectilinear footprints |"
    )
    lines.append("|---|---|---|")
    n_obliques_on_rect = sum(r["n_obliques"] for r in rows if not r["is_irregular"])
    for tol in OFF_AXIS_TOLERANCE_SWEEP_DEG:
        n = sweep[tol]
        lines.append(f"| {int(tol)} deg | {n} | {_pct(n, n_obliques_on_rect)} |")
    lines.append("")
    lines.append("## Top 20 buildings by total defect count")
    lines.append("")
    lines.append("| UUID | obl | off_axis | frag | cross | irreg | axis_cov |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in top20:
        total = r["n_off_axis_default"] + r["n_fragmented"] + r["n_cross_extension"]
        if total == 0:
            continue
        lines.append(
            "| {uuid} | {obl} | {oa} | {fr} | {cx} | {ir} | {cov:.2f} |".format(
                uuid=r["uuid"],
                obl=r["n_obliques"],
                oa=r["n_off_axis_default"],
                fr=r["n_fragmented"],
                cx=r["n_cross_extension"],
                ir=int(r["is_irregular"]),
                cov=r["wall_axis_coverage"],
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- A `roof_off_axis` flag means the slope direction is not "
        f"parallel/perpendicular to the building's wall axis (delta > "
        f"{int(OFF_AXIS_TOL_DEG)} deg, mod 90). A yaw-snap prior would resolve it."
    )
    lines.append(
        "- A `roof_fragmented` flag means >=2 oblique pieces share the same "
        "plane within tolerance. A continuity-prior merge would resolve it."
    )
    lines.append(
        "- A `roof_cross_extension` flag means a single oblique polygon spans "
        "two wings. Per-part roof reconstruction would resolve it."
    )
    lines.append(
        "- A `roof_irregular_footprint` flag (per-building) means wall axis "
        "coverage < 0.70 — the building isn't a primitive-library candidate "
        "and needs a hybrid mesh-patch fallback."
    )

    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument("--out", type=Path, default=Path(".context/cohort-defect-scan"))
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"--root not a directory: {args.root}", file=sys.stderr)
        return 1

    summary = _scan_cohort(args.root)
    timestamp = _ts()
    out_dir = args.out / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "roof_defect_summary.json"
    csv_path = out_dir / "per_building.csv"
    md_path = out_dir / "REPORT.md"

    json_summary = {k: v for k, v in summary.items() if k != "rows"}
    json_summary["rows_count"] = len(summary["rows"])
    summary_path.write_text(json.dumps(json_summary, indent=2, default=float))
    _write_per_building_csv(summary["rows"], csv_path)
    _write_report_md(summary, md_path)

    print(json.dumps(json_summary, indent=2, default=float))
    print(f"\nReport: {md_path}")
    print(f"CSV:    {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
