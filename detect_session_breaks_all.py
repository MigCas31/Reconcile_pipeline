#!/usr/bin/env python3
"""
Run the ARKit session-break detector across every scan in .scan-cache/.

Adapts the analysis from detect_session_breaks.py (which only ran against
8 known PROBLEM and 5 random CONTROL UUIDs) to the full corpus, and emits
a ranked CSV identifying which scans are likely impacted by AR tracking
or session continuity issues.

Local signals computed per scan:
  - num_sessions          distinct yaw clusters of referenceOriginTransform
  - max_yaw_diff_deg      largest pair-wise yaw gap between sessions
  - large_time_gaps       count of >10min pauses between consecutive room scans
  - y_offset_stories      stories where same-story rooms span >1m vertically
  - wall_disagreements    count of shared-wall transform mismatches
  - recon_sessions        scan_session_count from reconciler (if present)
  - recon_class           reconciler quality classification (if present)

Severity tiers (combining the signals):
  RED      num_sessions >= 3                       OR wall_disagreements >= 5
  ORANGE   num_sessions == 2 AND (large_time_gaps>=1 OR y_offset_stories>=1
                                  OR wall_disagreements>=1)
  YELLOW   num_sessions == 2                       (single benign session boundary)
  GREEN    num_sessions == 1
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# Reuse the analysis function from the existing script.
sys.path.insert(0, str(Path(__file__).parent))
from detect_session_breaks import analyze_building

REPO = Path(__file__).parent
CACHE = REPO / ".scan-cache"
OUT_DIR = REPO / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "scan_session_audit.csv"

UUID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
ADDRESS_RE = re.compile(r"^scans_[A-Za-z0-9]+_(.+)_[0-9a-f]{8}-")


def parse_cache_dir(name: str) -> tuple[str | None, str | None]:
    """Return (uuid, address) extracted from a .scan-cache dir name."""
    m = UUID_RE.search(name)
    uuid = m.group(1) if m else None
    a = ADDRESS_RE.match(name)
    address = a.group(1).replace("_", " ").strip() if a else None
    return uuid, address


def severity(num_sessions: int, gaps: int, y_off: int, wd: int) -> str:
    if num_sessions >= 3 or wd >= 5:
        return "RED"
    if num_sessions == 2 and (gaps >= 1 or y_off >= 1 or wd >= 1):
        return "ORANGE"
    if num_sessions == 2:
        return "YELLOW"
    return "GREEN"


def main() -> None:
    cache_dirs = sorted(d for d in CACHE.iterdir() if d.is_dir())
    print(f"Found {len(cache_dirs)} scan-cache directories")

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for i, d in enumerate(cache_dirs, 1):
        uuid, address = parse_cache_dir(d.name)
        if not uuid:
            skipped.append((d.name, "no UUID in dir name"))
            continue

        if i % 25 == 0 or i == len(cache_dirs):
            print(f"  [{i}/{len(cache_dirs)}] {uuid[:8]}…")

        r = analyze_building(uuid)
        if r.get("error"):
            skipped.append((uuid, r["error"]))
            continue

        gaps = len(r.get("large_time_gaps", []))
        y_off = len(r.get("y_offset_issues", {}))
        wd = len(r.get("wall_disagreements", []))
        sessions = r["num_sessions"]

        # Pair-wise max yaw diff between sessions
        max_yaw_diff = 0.0
        sess_yaws = [s["mean_yaw_deg"] for s in r.get("sessions", [])]
        for a in range(len(sess_yaws)):
            for b in range(a + 1, len(sess_yaws)):
                diff = abs(sess_yaws[a] - sess_yaws[b])
                diff = min(diff, 360 - diff)
                if diff > max_yaw_diff:
                    max_yaw_diff = diff

        rows.append(
            {
                "uuid": uuid,
                "address": address or "",
                "num_rooms": r.get("num_rooms", 0),
                "num_floors": r.get("num_floors", 0),
                "num_sessions": sessions,
                "max_yaw_diff_deg": round(max_yaw_diff, 2),
                "large_time_gaps": gaps,
                "y_offset_stories": y_off,
                "wall_disagreements": wd,
                "recon_sessions": r.get("recon_sessions", ""),
                "recon_class": r.get("recon_class", ""),
                "recon_gap_median_cm": r.get("recon_gap_median_cm", ""),
                "severity": severity(sessions, gaps, y_off, wd),
            }
        )

    # Rank: RED > ORANGE > YELLOW > GREEN, then by num_sessions, then wall_disagreements
    sev_order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
    rows.sort(
        key=lambda r: (
            sev_order[r["severity"]],
            -r["num_sessions"],
            -r["wall_disagreements"],
            -r["max_yaw_diff_deg"],
        )
    )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    if skipped:
        print(f"Skipped {len(skipped)} caches:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more")

    # Summary
    from collections import Counter

    sev_counts = Counter(r["severity"] for r in rows)
    print("\nSeverity distribution:")
    for level in ("RED", "ORANGE", "YELLOW", "GREEN"):
        print(f"  {level:7s} {sev_counts.get(level, 0):4d}")

    print("\nTop 15 RED/ORANGE scans:")
    print(
        f"{'UUID':<38} {'Sev':<7} {'Sess':<5} {'YawΔ':<7} {'Gaps':<5} {'YOff':<5} "
        f"{'WD':<4} {'Recon':<7} {'Address'}"
    )
    for r in rows[:15]:
        if r["severity"] not in ("RED", "ORANGE"):
            break
        print(
            f"{r['uuid']:<38} {r['severity']:<7} {r['num_sessions']:<5} "
            f"{r['max_yaw_diff_deg']:<7} {r['large_time_gaps']:<5} "
            f"{r['y_offset_stories']:<5} {r['wall_disagreements']:<4} "
            f"{r['recon_class']!s:<7} {r['address'][:50]}"
        )


if __name__ == "__main__":
    main()
