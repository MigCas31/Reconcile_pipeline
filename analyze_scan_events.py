#!/usr/bin/env python3
"""
Reconstruct each home's scanning timeline from raw PostHog events
(analysis_outputs/posthog/all_scan_events.csv) and classify each individual
room-scan attempt as CLEAN / RECOVERED / BROKEN.

The key insight motivating this rewrite: aggregate per-home counts conflate
recovery with failure. ARKit re-initialising mid-scan AND the scan finishing
cleanly = recovery, not a problem. The same reinit followed by no completion
= a real failure. We can only tell the two apart by walking the timeline.

Per-scan classification:
  CLEAN     roomScanStarted -> roomScanCompleted, no reinit/error in window
  RECOVERED reinit OR error during window, but scan still completed
  BROKEN    roomScanStarted but no roomScanCompleted before the next Started
            (or before the timeline ends). The user gave up or the app died.
  ERRORED   scan ended with mergeRoomsError / scanUploadFailed / roomScanError
            attributed to it (within 60s of the close event)

Per-home severity:
  RED       >=1 BROKEN scan OR mergeRoomsError ever fired OR
            >=1 mid-scan reinit followed by Stopped (not Completed)
  ORANGE    >=2 RECOVERED scans (sustained tracking trouble that recovered)
            OR exactly 1 BROKEN scan with an active retry pattern
  GREEN     all scans CLEAN or single isolated recovery

Outputs:
  analysis_outputs/per_scan_classification.csv  one row per scan attempt
  analysis_outputs/per_home_summary.csv         one row per home, with verdict
  analysis_outputs/scan_event_findings.md       short report
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent
EVENTS = REPO / "analysis_outputs" / "posthog" / "all_scan_events.csv"
PIPE = REPO / "pipeline-outputs"
OUT_PER_SCAN = REPO / "analysis_outputs" / "per_scan_classification.csv"
OUT_PER_HOME = REPO / "analysis_outputs" / "per_home_summary.csv"
OUT_REPORT = REPO / "analysis_outputs" / "scan_event_findings.md"

SCAN_EVENTS = {
    "roomScanStarted",
    "roomScanAROpened",
    "roomScanStopped",
    "roomScanCompleted",
    "roomScanCancelled",
    "roomScanRetakeRequested",
    "roomScanRetryAfterError",
    "roomScanError",
    "mergeRoomsError",
    "scanUploadFailed",
    "scanUploadStarted",
    "scanUploaded",
    "initializingARStateOutOfTheBlue",
    "roomCreated",
    "roomDeleted",
    "ceilingScanStarted",
    "ceilingScanAROpened",
    "ceilingScanCompleted",
    "storyCreated",
}


def parse_ts(s: str) -> datetime:
    # PostHog format: "2025-09-02 10:40:15.307000+02:00"
    return datetime.fromisoformat(s)


def load_events() -> dict[str, list[dict]]:
    """Return events grouped by home_id, sorted by timestamp."""
    by_home: dict[str, list[dict]] = defaultdict(list)
    with EVENTS.open() as f:
        for r in csv.DictReader(f):
            if r["event"] not in SCAN_EVENTS:
                continue
            home = r["home_id"]
            if not home:
                continue
            r["ts"] = parse_ts(r["timestamp"])
            by_home[home].append(r)
    for home in by_home:
        by_home[home].sort(key=lambda x: x["ts"])
    return by_home


def classify_scans(events: list[dict]) -> list[dict]:
    """
    Walk the event sequence and emit one record per roomScanStarted, with
    the events between it and the next terminating event recorded.
    """
    scans = []
    open_scan: dict | None = None

    def close_open(close_event: str | None, close_idx: int):
        nonlocal open_scan
        if not open_scan:
            return
        open_scan["closed_by"] = close_event
        open_scan["close_idx"] = close_idx
        scans.append(open_scan)
        open_scan = None

    for i, e in enumerate(events):
        ev = e["event"]
        if ev == "roomScanStarted":
            # If a previous scan was still open, treat it as BROKEN (no Completed)
            if open_scan:
                close_open(None, i - 1)
            open_scan = {
                "home_id": e["home_id"],
                "lead_id": e["lead_id"],
                "room_id": e["room_id"],
                "story_id": e["story_id"],
                "started_at": e["ts"],
                "surveyor": e["distinct_id"],
                "ar_reinits_during": 0,
                "ar_opens_during": 0,
                "stops_during": 0,
                "errors_during": 0,
                "events_during": [],
                "start_idx": i,
            }
            continue

        if not open_scan:
            continue

        if ev in ("roomScanCompleted", "roomScanCancelled"):
            close_open(ev, i)
            continue

        # roomScanStopped is interesting: by itself it ends the scan, but iOS
        # often emits Stopped immediately followed by Completed (the user
        # tapped Stop after the scan was already capturing). So we record
        # the Stop and only close on a later Completed/another Started.
        if ev == "roomScanStopped":
            open_scan["stops_during"] += 1
            open_scan["events_during"].append(ev)
            continue

        if ev == "initializingARStateOutOfTheBlue":
            open_scan["ar_reinits_during"] += 1
            open_scan["events_during"].append(ev)
        elif ev == "roomScanAROpened":
            open_scan["ar_opens_during"] += 1
            open_scan["events_during"].append(ev)
        elif ev in (
            "roomScanError",
            "mergeRoomsError",
            "scanUploadFailed",
            "roomScanRetryAfterError",
        ):
            open_scan["errors_during"] += 1
            open_scan["events_during"].append(ev)
        else:
            open_scan["events_during"].append(ev)

    # If the timeline ended with an open scan, that's BROKEN
    if open_scan:
        close_open(None, len(events) - 1)
    return scans


def label_scan(scan: dict) -> str:
    cb = scan["closed_by"]
    if cb is None:
        return "BROKEN"  # never reached Completed/Cancelled
    if cb == "roomScanCancelled":
        # User cancellation; not an AR failure but worth tracking separately
        return "CANCELLED"
    # cb == "roomScanCompleted"
    if scan["errors_during"] > 0:
        return "ERRORED"
    if scan["ar_reinits_during"] > 0:
        return "RECOVERED"
    return "CLEAN"


def home_severity(scans: list[dict], home_extra: dict) -> tuple[str, list[str]]:
    """Decide RED/ORANGE/GREEN for a home and explain why."""
    labels = Counter(s["label"] for s in scans)
    reasons: list[str] = []

    broken = labels.get("BROKEN", 0)
    errored = labels.get("ERRORED", 0)
    recovered = labels.get("RECOVERED", 0)
    cancelled = labels.get("CANCELLED", 0)
    total = sum(labels.values())

    merge_errors = home_extra.get("merge_errors_total", 0)
    upload_failed = home_extra.get("upload_failed_total", 0)

    # RED conditions — actual failures that left damaged data
    if broken >= 1:
        reasons.append(f"{broken} broken-scan(s)")
    if errored >= 1:
        reasons.append(f"{errored} errored-scan(s)")
    if merge_errors >= 1:
        reasons.append(f"{merge_errors} mergeRoomsError")
    if upload_failed >= 1:
        reasons.append(f"{upload_failed} scanUploadFailed")

    if reasons:
        return ("RED", reasons)

    # ORANGE — recovery patterns (tracking trouble that came back)
    if recovered >= 3:
        return ("ORANGE", [f"{recovered} recovered-scan(s) (sustained reinit pattern)"])
    if recovered >= 1 and total >= 3:
        return ("ORANGE", [f"{recovered} recovered-scan(s)"])
    if cancelled >= 3:
        return ("ORANGE", [f"{cancelled} cancellation(s)"])

    return ("GREEN", [])


def recon_for(uuid: str) -> tuple[str, str]:
    p = PIPE / uuid / "reconciled.json"
    if not p.exists():
        return ("MISSING", "")
    try:
        r = json.load(p.open()).get("reconciliation", {})
        return (r.get("classification", "?"), " ; ".join(r.get("flags") or []))
    except Exception:
        return ("ERR", "")


def main() -> None:
    events_by_home = load_events()
    print(f"Loaded events for {len(events_by_home)} homes")

    all_scans: list[dict] = []
    home_records: list[dict] = []

    for home, events in events_by_home.items():
        scans = classify_scans(events)
        for s in scans:
            s["label"] = label_scan(s)
            s["events_during_str"] = ",".join(s["events_during"])
            all_scans.append(s)

        # Home-level extras
        merge_errs = sum(1 for e in events if e["event"] == "mergeRoomsError")
        upload_fails = sum(1 for e in events if e["event"] == "scanUploadFailed")
        home_extra = {
            "merge_errors_total": merge_errs,
            "upload_failed_total": upload_fails,
        }

        sev, reasons = home_severity(scans, home_extra)
        rc, rflags = recon_for(home)

        labels = Counter(s["label"] for s in scans)
        first = events[0]["ts"] if events else None
        last = events[-1]["ts"] if events else None
        home_records.append(
            {
                "home_id": home,
                "lead_id": events[0]["lead_id"] if events else "",
                "verdict": sev,
                "reasons": " ; ".join(reasons),
                "n_scans": sum(labels.values()),
                "n_clean": labels.get("CLEAN", 0),
                "n_recovered": labels.get("RECOVERED", 0),
                "n_errored": labels.get("ERRORED", 0),
                "n_broken": labels.get("BROKEN", 0),
                "n_cancelled": labels.get("CANCELLED", 0),
                "merge_errors": merge_errs,
                "upload_failed": upload_fails,
                "recon_class": rc,
                "recon_flags": rflags,
                "first_event": first.isoformat() if first else "",
                "last_event": last.isoformat() if last else "",
                "n_surveyors": len({e["distinct_id"] for e in events}),
            }
        )

    # Per-scan CSV
    fieldnames = [
        "home_id",
        "room_id",
        "story_id",
        "started_at",
        "surveyor",
        "label",
        "closed_by",
        "ar_reinits_during",
        "ar_opens_during",
        "errors_during",
        "stops_during",
        "events_during_str",
    ]
    with OUT_PER_SCAN.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for s in all_scans:
            row = {**s, "started_at": s["started_at"].isoformat()}
            w.writerow(row)
    print(f"Wrote {len(all_scans)} scans to {OUT_PER_SCAN.name}")

    # Per-home CSV
    sev_order = {"RED": 0, "ORANGE": 1, "GREEN": 2}
    home_records.sort(
        key=lambda x: (
            sev_order[x["verdict"]],
            -x["n_broken"],
            -x["merge_errors"],
            -x["n_errored"],
            -x["n_recovered"],
        )
    )
    with OUT_PER_HOME.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(home_records[0].keys()))
        w.writeheader()
        w.writerows(home_records)
    print(f"Wrote {len(home_records)} homes to {OUT_PER_HOME.name}")

    # Summary stats
    sev_counts = Counter(h["verdict"] for h in home_records)
    label_counts = Counter(s["label"] for s in all_scans)

    print("\nPer-scan label distribution:")
    for k, v in label_counts.most_common():
        print(f"  {k:10s} {v:5d}  ({100 * v / len(all_scans):4.1f}%)")

    print("\nPer-home verdict (n=", len(home_records), "):", sep="")
    for k in ("RED", "ORANGE", "GREEN"):
        n = sev_counts.get(k, 0)
        print(f"  {k:7s} {n:4d}  ({100 * n / len(home_records):4.1f}%)")

    # Cross-validate against reconciler
    print("\nReconciler outcomes by home verdict:")
    print(
        f"  {'verdict':<8} {'N':<5} {'%recon-RED':<12} {'%recon-MISS':<12} "
        f"{'%recon-YEL':<12} {'%recon-GREEN':<12}"
    )
    for v in ("RED", "ORANGE", "GREEN"):
        bucket = [h for h in home_records if h["verdict"] == v]
        n = len(bucket)
        if not n:
            continue
        cnt = Counter(h["recon_class"] for h in bucket)
        print(
            f"  {v:<8} {n:<5} "
            f"{100 * cnt.get('RED', 0) / n:>8.0f}%    "
            f"{100 * cnt.get('MISSING', 0) / n:>8.0f}%    "
            f"{100 * cnt.get('YELLOW', 0) / n:>8.0f}%    "
            f"{100 * cnt.get('GREEN', 0) / n:>8.0f}%"
        )

    # Restrict to the 229 buildings we have local cache for
    with (REPO / "analysis_outputs" / "scan_session_audit.csv").open() as f:
        local_uuids = {r["uuid"] for r in csv.DictReader(f)}
    local_homes = [h for h in home_records if h["home_id"] in local_uuids]
    local_sev = Counter(h["verdict"] for h in local_homes)
    print(f"\nRestricted to {len(local_homes)} homes that exist in our scan-cache:")
    for k in ("RED", "ORANGE", "GREEN"):
        n = local_sev.get(k, 0)
        print(f"  {k:7s} {n:4d}  ({100 * n / len(local_homes):4.1f}%)")

    # Top RED in our cohort
    print("\nTop 20 RED homes from our 229-cohort:")
    print(
        f"  {'home_id':<38} {'recon':<8} {'#brk':<5} {'#err':<5} {'#rec':<5} "
        f"{'merge':<6} {'reasons'}"
    )
    for h in [h for h in local_homes if h["verdict"] == "RED"][:20]:
        print(
            f"  {h['home_id']:<38} {h['recon_class']:<8} "
            f"{h['n_broken']:<5} {h['n_errored']:<5} {h['n_recovered']:<5} "
            f"{h['merge_errors']:<6} {h['reasons'][:70]}"
        )


if __name__ == "__main__":
    main()
