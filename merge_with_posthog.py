#!/usr/bin/env python3
"""
Join the local session-break audit (analysis_outputs/scan_session_audit.csv)
with the four PostHog HogQL exports the user downloads from
https://eu.posthog.com/project/2452/sql/, and emit a unified ranking of
scans / cases impacted by AR or session problems.

Expected input CSVs in analysis_outputs/posthog/ (any subset is fine —
missing files are treated as zero-evidence on those signals):

    scan_completion_by_home.csv        Q1  (started/stopped/completed/AR-opened)
    long_scan_gaps_by_home.csv         Q2  (max gap between scan events)
    ar_reopen_per_room.csv             Q3  (per-room AR retries — drill-down)
    scan_trouble_events_by_home.csv    Q5  (ar_reinit + cancel/retry/error events)
    rapid_trouble_bursts_by_home.csv   Q6  (rapid-succession failure bursts)

Each must have a `home_id` column (= building UUID). Other columns vary;
this script only reads the metric columns it knows about.

Output:
    analysis_outputs/impacted_scans.csv     unified per-UUID table
    analysis_outputs/impacted_scans.md      short human-readable summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).parent
OUT_DIR = REPO / "analysis_outputs"
LOCAL_CSV = OUT_DIR / "scan_session_audit.csv"
PH_DIR = OUT_DIR / "posthog"
PIPE_DIR = REPO / "pipeline-outputs"
OUT_CSV = OUT_DIR / "impacted_scans.csv"
OUT_MD = OUT_DIR / "impacted_scans.md"


def load_recon(uuid: str) -> dict:
    """Pull the reconciler's classification + scan_session_count, if present."""
    rec = PIPE_DIR / uuid / "reconciled.json"
    if not rec.exists():
        return {"recon_class": "MISSING", "recon_sessions": "", "recon_flags": ""}
    try:
        with rec.open() as f:
            r = json.load(f).get("reconciliation") or {}
    except Exception:
        return {"recon_class": "PARSE_ERR", "recon_sessions": "", "recon_flags": ""}
    return {
        "recon_class": r.get("classification", ""),
        "recon_sessions": str(r.get("scan_session_count", "")),
        "recon_flags": " ; ".join(r.get("flags") or []),
    }


def read_ph_csv(name: str, key_cols: list[str]) -> dict[str, dict]:
    """Load a PostHog export keyed by home_id.

    PostHog exports array-typed columns as numbered siblings (e.g.
    `error_texts.0`, `error_texts.1`). For any key_col we asked for, we
    also gather all matching `<key>.N` columns and join them with ' | '.
    """
    path = PH_DIR / name
    if not path.exists():
        print(f"  (skipped, not found: {path.name})")
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        array_siblings: dict[str, list[str]] = {}
        for k in key_cols:
            siblings = sorted(c for c in fieldnames if c == k or c.startswith(k + "."))
            if siblings:
                array_siblings[k] = siblings
        for row in reader:
            home = (row.get("home_id") or "").strip()
            if not home:
                continue
            entry: dict[str, str] = {}
            for k, cols in array_siblings.items():
                vals = [row.get(c, "") for c in cols if row.get(c)]
                entry[k] = (
                    " | ".join(vals) if len(cols) > 1 else (vals[0] if vals else "")
                )
            out[home] = entry
    print(f"  loaded {len(out):4d} rows from {path.name}")
    return out


def to_int(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def combined_severity(
    local: str, ph: dict, recon_class: str = "", recon_flags: str = ""
) -> str:
    """
    Thresholds calibrated from the Botjek cohort.

    `roomScanAROpened` was added to the app later, so older scans have
    AROpened << Started → negative `ar_reopens`. Only positive values
    are signal (88/475 homes), so we clamp to max(0, …).

    Calibrated tail thresholds (from real Botjek distributions):
      incomplete_scans         p95=2,  p99=4    → ORANGE >=2,  RED >=4
      stopped_not_done         p95=2,  p99=4    → ORANGE >=2,  RED >=4
      ar_reopens (clamp)       p95≈2,  p99≈5    → ORANGE >=2,  RED >=5
      gaps_over_1day           n=86 nz=15 max=2 → ORANGE >=1,  RED >=2
      ar_reinit                CORROBORATING ONLY (no solo promotion)
        ARKit re-init is recovery, not failure: tracking blip → ARKit
        re-establishes a frame → scan continues. Of buildings RED only via
        ar_reinit >= 6, 38% are recon-YELLOW (recoverable). So ar_reinit
        only escalates to RED when paired with another signal:
          ar_reinit >= 2 AND any other PostHog/local/recon signal -> RED
        Solo ar_reinit (any count) only escalates to ORANGE at >= 6.
        Calibration data (per reinit bucket, 229 local scans):
          0 reinits  -> 34% recon-RED (baseline — recon's bar is loose)
          2 reinits  -> 51% recon-RED (+17pp)
          6-8 reinit -> 64% recon-RED (+30pp)
          9-12 reinit-> 62% recon-RED + 12% PIPELINE_BROKEN
          13+ reinit -> 100% recon-RED
      merge_errors             nz=20  max=10    → RED >=1 (it's a NaN crash)
      scan_errors              nz=30  max=2     → RED >=1
      scan_retry_after_error   nz=13  max=1     → RED >=1
      upload_failed            nz=10  max=6     → RED >=1
      bursts_under_10s         n=273 nz=38 max=4  → RED >=1 (ARKit thrashing)
      bursts_under_60s         n=273 nz=99 max=9  → ORANGE >=2, RED >=4
      bursts_under_5min        n=273 nz=165 max=11 → ORANGE >=4

    NOT used to drive severity (normal UX flow, not AR failures):
      scan_cancelled           — surveyor explicitly stopped a scan
      scan_retake_requested    — surveyor asked to redo a scan
    Both are kept in the CSV for evidence, but they reflect surveyor
    decisions, not ARKit/iOS breaking.
    """
    incomplete = to_int(ph.get("incomplete_scans"))
    stopped = to_int(ph.get("stopped_not_done"))
    ar_reopens = max(0, to_int(ph.get("ar_reopens")))
    day_gaps = to_int(ph.get("gaps_over_1day"))

    ar_reinit = to_int(ph.get("ar_reinit"))
    scan_errors = to_int(ph.get("scan_errors"))
    merge_errors = to_int(ph.get("merge_errors"))
    to_int(ph.get("scan_cancelled"))
    to_int(ph.get("scan_retake_requested"))
    scan_retry = to_int(ph.get("scan_retry_after_error"))
    upload_failed = to_int(ph.get("upload_failed"))

    burst_10s = to_int(ph.get("bursts_under_10s"))
    burst_60s = to_int(ph.get("bursts_under_60s"))
    burst_5min = to_int(ph.get("bursts_under_5min"))

    # Hard-failure signals: each promotes to RED on its own.
    ph_red_solo = (
        incomplete >= 4
        or stopped >= 4
        or ar_reopens >= 5
        or day_gaps >= 2
        or merge_errors >= 1
        or scan_errors >= 1
        or scan_retry >= 1
        or upload_failed >= 1
        or burst_10s >= 1
        or burst_60s >= 4
    )

    # Mild signals (excluding ar_reinit): each promotes to ORANGE alone.
    ph_orange_non_reinit = (
        incomplete >= 2
        or stopped >= 2
        or ar_reopens >= 2
        or day_gaps >= 1
        or burst_60s >= 2
        or burst_5min >= 4
    )

    # ar_reinit is recovery, not failure — only RED if PAIRED with a non-reinit
    # signal. Solo it caps at ORANGE.
    has_non_reinit_signal = (
        ph_orange_non_reinit
        or ph_red_solo
        or local in ("RED", "ORANGE", "YELLOW")
        or "multi_session" in (recon_flags or "")
        or recon_class == "MISSING"
    )
    ph_red = ph_red_solo or (ar_reinit >= 6 and has_non_reinit_signal)
    ph_orange = ph_orange_non_reinit or ar_reinit >= 6

    # The reconciler's general RED/YELLOW bars are too coarse to fold in
    # directly (109 RED + 113 YELLOW out of 225 — only 1 GREEN). Most recon-RED
    # comes from `max_height_delta > 50cm`, which is geometry-quality, not
    # specifically AR/session. We honour the reconciler in two cases:
    #   - recon `multi_session` flag IS the AR-session signal -> auto RED
    #   - MISSING reconciled.json = iOS merge crashed -> always RED
    # Otherwise the recon class travels along as evidence in the CSV but
    # doesn't drive severity — we want the question "impacted by AR" to keep
    # its meaning, not collapse into "has any geometry deviation".
    recon_multi_session = "multi_session" in (recon_flags or "")
    if recon_class == "MISSING":
        return "RED"
    if recon_multi_session or local == "RED" or ph_red:
        return "RED"
    if local == "ORANGE" or ph_orange:
        return "ORANGE"
    if local == "YELLOW":
        return "YELLOW"
    return "GREEN"


def main() -> None:
    if not LOCAL_CSV.exists():
        raise SystemExit(f"Missing {LOCAL_CSV}; run detect_session_breaks_all.py first")

    PH_DIR.mkdir(exist_ok=True)
    print(f"Reading PostHog exports from {PH_DIR}/")
    completion = read_ph_csv(
        "scan_completion_by_home.csv",
        [
            "scans_started",
            "scans_stopped",
            "scans_completed",
            "ar_opens",
            "incomplete_scans",
            "stopped_not_done",
            "ar_reopens",
        ],
    )
    gaps = read_ph_csv(
        "long_scan_gaps_by_home.csv",
        ["max_gap_hours", "gaps_over_10min", "gaps_over_1hr", "gaps_over_1day"],
    )
    trouble = read_ph_csv(
        "scan_trouble_events_by_home.csv",
        [
            "ar_reinit",
            "scan_errors",
            "merge_errors",
            "scan_cancelled",
            "scan_retake_requested",
            "scan_retry_after_error",
            "upload_failed",
            "error_texts",
        ],
    )
    bursts = read_ph_csv(
        "rapid_trouble_bursts_by_home.csv",
        [
            "total_failures",
            "bursts_under_10s",
            "bursts_under_60s",
            "bursts_under_5min",
            "min_gap_seconds",
            "tightest_burst",
        ],
    )

    with LOCAL_CSV.open() as f:
        local_rows = list(csv.DictReader(f))
    print(f"Loaded {len(local_rows)} rows from {LOCAL_CSV.name}")

    rows = []
    for r in local_rows:
        uuid = r["uuid"]
        ph_combined = {}
        ph_combined.update(completion.get(uuid, {}))
        ph_combined.update(gaps.get(uuid, {}))
        ph_combined.update(trouble.get(uuid, {}))
        ph_combined.update(bursts.get(uuid, {}))

        recon = load_recon(uuid)
        sev = combined_severity(
            r["severity"], ph_combined, recon["recon_class"], recon["recon_flags"]
        )

        rows.append(
            {
                "uuid": uuid,
                "address": r["address"],
                "combined_severity": sev,
                "recon_class": recon["recon_class"],
                "recon_sessions": recon["recon_sessions"],
                "recon_flags": recon["recon_flags"],
                "local_severity": r["severity"],
                "local_sessions": r["num_sessions"],
                "local_max_yaw_deg": r["max_yaw_diff_deg"],
                "local_wall_disagreements": r["wall_disagreements"],
                "ph_incomplete_scans": to_int(ph_combined.get("incomplete_scans")),
                "ph_ar_reopens": max(0, to_int(ph_combined.get("ar_reopens"))),
                "ph_stopped_not_done": to_int(ph_combined.get("stopped_not_done")),
                "ph_day_gaps": to_int(ph_combined.get("gaps_over_1day")),
                "ph_ar_reinit": to_int(ph_combined.get("ar_reinit")),
                "ph_scan_errors": to_int(ph_combined.get("scan_errors")),
                "ph_merge_errors": to_int(ph_combined.get("merge_errors")),
                "ph_scan_cancelled": to_int(ph_combined.get("scan_cancelled")),
                "ph_scan_retake": to_int(ph_combined.get("scan_retake_requested")),
                "ph_scan_retry": to_int(ph_combined.get("scan_retry_after_error")),
                "ph_upload_failed": to_int(ph_combined.get("upload_failed")),
                "ph_burst_10s": to_int(ph_combined.get("bursts_under_10s")),
                "ph_burst_60s": to_int(ph_combined.get("bursts_under_60s")),
                "ph_burst_5min": to_int(ph_combined.get("bursts_under_5min")),
                "ph_min_gap_sec": ph_combined.get("min_gap_seconds", ""),
                "ph_tightest_burst": ph_combined.get("tightest_burst", ""),
                "ph_error_texts": ph_combined.get("error_texts", "")[:500],
                "local_recon_class": r["recon_class"],
            }
        )

    sev_order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
    rows.sort(
        key=lambda x: (
            sev_order[x["combined_severity"]],
            -x["ph_burst_10s"],
            -x["ph_merge_errors"],
            -x["ph_scan_errors"],
            -x["ph_burst_60s"],
            -x["ph_incomplete_scans"],
            -x["ph_ar_reinit"],
            -int(x["local_sessions"] or 0),
        )
    )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    from collections import Counter

    sev_counts = Counter(r["combined_severity"] for r in rows)
    impacted = [r for r in rows if r["combined_severity"] in ("RED", "ORANGE")]

    lines = [
        "# Scans impacted by AR / session problems",
        "",
        f"Combined ranking from local detector + PostHog event evidence ({len(rows)} "
        f"scans).",
        "",
        "## Severity distribution",
        "",
        "| Level  | Count |",
        "|--------|------:|",
    ]
    for level in ("RED", "ORANGE", "YELLOW", "GREEN"):
        lines.append(f"| {level:6s} | {sev_counts.get(level, 0):4d} |")
    lines += [
        "",
        f"## Impacted scans ({len(impacted)} RED + ORANGE)",
        "",
        "| UUID | Sev | Recon | B10s | B60s | MgErr | ScErr | Incompl | Reinit | "
        "TightestBurst | Address |",
        "|------|-----|-------|-----:|-----:|------:|------:|--------:|-------:|---------------|---------|",
    ]
    for r in impacted:
        lines.append(
            f"| `{r['uuid']}` | {r['combined_severity']} | {r['recon_class']} | "
            f"{r['ph_burst_10s']} | {r['ph_burst_60s']} | "
            f"{r['ph_merge_errors']} | {r['ph_scan_errors']} | "
            f"{r['ph_incomplete_scans']} | {r['ph_ar_reinit']} | "
            f"{r['ph_tightest_burst'][:42]} | {r['address']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")
    print("\nSeverity distribution:")
    for level in ("RED", "ORANGE", "YELLOW", "GREEN"):
        print(f"  {level:7s} {sev_counts.get(level, 0):4d}")


if __name__ == "__main__":
    main()
