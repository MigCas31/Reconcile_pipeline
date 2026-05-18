#!/usr/bin/env python3
"""
For every scan flagged RED in analysis_outputs/impacted_scans.csv, check the
state of its pipeline-outputs/<uuid>/ directory:

    - PIPELINE_BROKEN   no merged.json -> iOS merge crashed, never reached pipeline
    - RECON_BROKEN      merged.json present but no reconciled.json -> pipeline bailed
    - RECON_RED         reconciled.json says RED -> pipeline ran, geometry visibly bad
    - RECON_YELLOW      reconciled.json says YELLOW -> mild issues
    - RECON_GREEN       reconciled.json says GREEN -> "quietly recovered"
    - NO_OUTPUT         pipeline-outputs/<uuid> doesn't exist at all

Writes analysis_outputs/red_pipeline_audit.csv and prints a categorised summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).parent
IMPACTED = REPO / "analysis_outputs" / "impacted_scans.csv"
PIPE = REPO / "pipeline-outputs"
OUT = REPO / "analysis_outputs" / "red_pipeline_audit.csv"


def audit(uuid: str) -> dict:
    d = PIPE / uuid
    if not d.exists():
        return {
            "pipeline_state": "NO_OUTPUT",
            "tier_label": "",
            "recon_class": "",
            "recon_flags": "",
            "scan_sessions": "",
            "median_disp_m": "",
        }

    has_merged = (d / "merged.json").exists()
    has_recon = (d / "reconciled.json").exists()
    has_tier = (d / "tier_payload.json").exists()

    state = ""
    if not has_merged:
        state = "PIPELINE_BROKEN"
    elif not has_recon:
        state = "RECON_BROKEN"

    recon_class = ""
    flags: list[str] = []
    sessions = ""
    median_disp = ""
    if has_recon:
        try:
            with (d / "reconciled.json").open() as f:
                rec = json.load(f)
            r = rec.get("reconciliation", {})
            recon_class = r.get("classification", "")
            flags = r.get("flags", []) or []
            sessions = str(r.get("scan_session_count", ""))
            median_disp = str(r.get("median_wall_displacement_m", ""))
            if not state:
                state = f"RECON_{recon_class}" if recon_class else "RECON_UNKNOWN"
        except Exception as e:
            state = state or f"RECON_PARSE_ERR ({e})"

    tier_label = ""
    if has_tier:
        try:
            with (d / "tier_payload.json").open() as f:
                tier = json.load(f)
            tier_label = (tier.get("classification") or {}).get("tier_label", "")
        except Exception:
            pass

    return {
        "pipeline_state": state,
        "tier_label": tier_label,
        "recon_class": recon_class,
        "recon_flags": " ; ".join(flags),
        "scan_sessions": sessions,
        "median_disp_m": median_disp,
    }


def main() -> None:
    with IMPACTED.open() as f:
        rows = [r for r in csv.DictReader(f) if r["combined_severity"] == "RED"]
    print(f"{len(rows)} RED scans to audit")

    enriched = []
    for r in rows:
        a = audit(r["uuid"])
        enriched.append({**r, **a})

    fieldnames = [
        "uuid",
        "address",
        "pipeline_state",
        "recon_class",
        "tier_label",
        "scan_sessions",
        "median_disp_m",
        "ph_merge_errors",
        "ph_scan_errors",
        "ph_incomplete_scans",
        "ph_ar_reinit",
        "ph_scan_cancelled",
        "local_severity",
        "local_sessions",
        "recon_flags",
    ]
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)
    print(f"Wrote {OUT}")

    from collections import Counter

    state_counts = Counter(e["pipeline_state"] for e in enriched)
    print("\nPipeline-state distribution among RED scans:")
    for state, n in state_counts.most_common():
        print(f"  {state:18s} {n:3d}")

    print("\n--- PIPELINE_BROKEN (iOS merge crashed before output) ---")
    for e in enriched:
        if e["pipeline_state"] == "PIPELINE_BROKEN":
            print(
                f"  {e['uuid']}  merge_err={e['ph_merge_errors']:>2}  "
                f"scan_err={e['ph_scan_errors']:>1}  "
                f"incomplete={e['ph_incomplete_scans']:>2}  "
                f"reinit={e['ph_ar_reinit']:>2}  {e['address'][:60]}"
            )

    print("\n--- RECON_BROKEN (pipeline failed during reconcile) ---")
    for e in enriched:
        if e["pipeline_state"] == "RECON_BROKEN":
            print(f"  {e['uuid']}  {e['address'][:60]}")

    print("\n--- RECON_RED (pipeline ran, geometry visibly bad) ---")
    for e in enriched:
        if e["pipeline_state"] == "RECON_RED":
            print(
                f"  {e['uuid']}  sess={e['scan_sessions']:>2}  "
                f"disp={e['median_disp_m']:>5}  {e['address'][:50]}"
            )
            for flag in e["recon_flags"].split(" ; "):
                if flag:
                    print(f"      ! {flag}")

    print("\n--- RECON_GREEN/YELLOW (quietly recovered despite trouble) ---")
    for e in enriched:
        if e["pipeline_state"] in ("RECON_GREEN", "RECON_YELLOW"):
            print(
                f"  {e['pipeline_state']:13s} {e['uuid']}  "
                f"merge_err={e['ph_merge_errors']:>2}  "
                f"reinit={e['ph_ar_reinit']:>2}  cancel={e['ph_scan_cancelled']:>2}  "
                f"{e['address'][:50]}"
            )


if __name__ == "__main__":
    main()
