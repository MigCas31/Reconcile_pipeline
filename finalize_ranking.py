#!/usr/bin/env python3
"""
Final ranking of scans impacted by AR / session problems.

After several rounds of threshold tuning we found that no single signal —
neither cumulative PostHog metrics nor sequence-walk per-scan classification
— is precise on its own. The most discriminating signal is AGREEMENT
between the two: when both rankings flag the same home, 79% of the time the
downstream reconstruction is actually broken (vs ~32% baseline).

So this script combines them into three confidence tiers:

  HIGH       Both `impacted_scans.csv` (threshold-based) and
             `per_home_summary.csv` (sequence-based) flag it RED.
             Plus any home with no merged.json (iOS pipeline crash).
             -> 79% of these are recon-broken downstream.

  MEDIUM     RED in exactly one of the two rankings.
             -> ~50% downstream-broken, worth a manual look.

  LOW        Neither ranking flags it. 32% downstream-broken,
             ~ baseline rate for the whole corpus.

Outputs:
  analysis_outputs/final_impacted_ranking.csv
  analysis_outputs/final_impacted_ranking.md
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).parent
OUT = REPO / "analysis_outputs"
PIPE = REPO / "pipeline-outputs"


def recon_for(uuid: str) -> tuple[str, str]:
    p = PIPE / uuid / "reconciled.json"
    if not p.exists():
        return ("MISSING", "")
    try:
        r = json.load(p.open()).get("reconciliation", {})
        return (r.get("classification", "?"), " ; ".join(r.get("flags") or []))
    except Exception:
        return ("ERR", "")


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def main() -> None:
    with (OUT / "scan_session_audit.csv").open() as f:
        local = {r["uuid"]: r for r in csv.DictReader(f)}
    with (OUT / "impacted_scans.csv").open() as f:
        thresh = {r["uuid"]: r for r in csv.DictReader(f)}
    with (OUT / "per_home_summary.csv").open() as f:
        seq = {r["home_id"]: r for r in csv.DictReader(f)}

    rows = []
    for uuid, lr in local.items():
        t = thresh.get(uuid, {})
        s = seq.get(uuid, {})
        rc, rflags = recon_for(uuid)

        thresh_red = t.get("combined_severity") == "RED"
        seq_red = s.get("verdict") == "RED"
        pipeline_broken = rc == "MISSING"

        if pipeline_broken or (thresh_red and seq_red):
            tier = "HIGH"
        elif thresh_red or seq_red:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        # Reasons summary
        reasons: list[str] = []
        if pipeline_broken:
            reasons.append("PIPELINE_BROKEN (no merged.json)")
        if to_int(s.get("merge_errors")):
            reasons.append(f"{s['merge_errors']} mergeRoomsError")
        if to_int(s.get("upload_failed")):
            reasons.append(f"{s['upload_failed']} scanUploadFailed")
        if to_int(s.get("n_broken")):
            reasons.append(f"{s['n_broken']} broken-scan(s)")
        if to_int(s.get("n_errored")):
            reasons.append(f"{s['n_errored']} errored-scan(s)")
        if "multi_session" in (rflags or ""):
            reasons.append("recon multi_session")
        if to_int(t.get("ph_burst_10s")):
            reasons.append(f"burst<10s x{t['ph_burst_10s']}")

        rows.append(
            {
                "uuid": uuid,
                "address": lr["address"],
                "tier": tier,
                "thresh_severity": t.get("combined_severity", ""),
                "seq_verdict": s.get("verdict", ""),
                "recon_class": rc,
                "n_scans": s.get("n_scans", ""),
                "n_clean": s.get("n_clean", ""),
                "n_broken": s.get("n_broken", ""),
                "n_recovered": s.get("n_recovered", ""),
                "merge_errors": s.get("merge_errors", ""),
                "upload_failed": s.get("upload_failed", ""),
                "ar_reinit_total": t.get("ph_ar_reinit", ""),
                "tightest_burst": t.get("ph_tightest_burst", ""),
                "recon_flags": rflags,
                "reasons": " ; ".join(reasons),
            }
        )

    tier_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rows.sort(
        key=lambda x: (
            tier_order[x["tier"]],
            -to_int(x["merge_errors"]),
            -to_int(x["n_broken"]),
            -to_int(x["ar_reinit_total"]),
        )
    )

    out_csv = OUT / "final_impacted_ranking.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv.name}: {len(rows)} homes")

    # Validation
    tier_counts = Counter(r["tier"] for r in rows)
    print("\nTier distribution:")
    for t in ("HIGH", "MEDIUM", "LOW"):
        n = tier_counts.get(t, 0)
        print(f"  {t:7s} {n:4d}  ({100 * n / len(rows):4.1f}%)")

    print("\nValidation against reconciler outcomes:")
    print(
        f"  {'tier':<7} {'N':<5} {'recon-RED':<11} {'PIPE-BROKEN':<13} "
        f"{'recon-YEL':<11} {'broken+missing':<14}"
    )
    for t in ("HIGH", "MEDIUM", "LOW"):
        bucket = [r for r in rows if r["tier"] == t]
        n = len(bucket)
        if not n:
            continue
        cnt = Counter(r["recon_class"] for r in bucket)
        red = cnt.get("RED", 0)
        miss = cnt.get("MISSING", 0)
        yel = cnt.get("YELLOW", 0)
        broken = red + miss
        print(
            f"  {t:<7} {n:<5} "
            f"{100 * red / n:>7.0f}%    "
            f"{100 * miss / n:>9.0f}%    "
            f"{100 * yel / n:>7.0f}%    "
            f"{100 * broken / n:>10.0f}%"
        )

    # Markdown writeup
    high = [r for r in rows if r["tier"] == "HIGH"]
    [r for r in rows if r["tier"] == "MEDIUM"]
    lines = [
        "# Scans impacted by AR / session problems — final ranking",
        "",
        "229 scans, three confidence tiers based on agreement between the "
        "threshold-based PostHog ranking and the per-scan-sequence walk.",
        "",
        "## Tier definitions",
        "",
        "- **HIGH**: both rankings flag RED, or pipeline-outputs has no merged.json. "
        "**~80% of these are recon-broken downstream.**",
        "- **MEDIUM**: flagged RED in only one ranking. ~50% recon-broken, worth "
        "manual review.",
        "- **LOW**: clean in both rankings. ~32% recon-broken (≈ baseline noise).",
        "",
        "## Distribution",
        "",
        "| Tier | N | % recon-RED | % PIPELINE_BROKEN | % recon-YELLOW |",
        "|------|--:|------------:|------------------:|---------------:|",
    ]
    for t in ("HIGH", "MEDIUM", "LOW"):
        bucket = [r for r in rows if r["tier"] == t]
        n = len(bucket)
        if not n:
            continue
        cnt = Counter(r["recon_class"] for r in bucket)
        lines.append(
            f"| {t} | {n} | {100 * cnt.get('RED', 0) / n:.0f}% | "
            f"{100 * cnt.get('MISSING', 0) / n:.0f}% | "
            f"{100 * cnt.get('YELLOW', 0) / n:.0f}% |"
        )
    lines += [
        "",
        f"## HIGH-confidence impacted scans (n={len(high)})",
        "",
        "These have either the iOS NaN merge bug (no merged.json) or both "
        "the threshold and sequence rankings flagging them.",
        "",
        "| UUID | Recon | #broken | Merge errs | AR reinit | Reasons | Address |",
        "|------|-------|--------:|-----------:|----------:|---------|---------|",
    ]
    for r in high:
        lines.append(
            f"| `{r['uuid']}` | {r['recon_class']} | {r['n_broken']} | "
            f"{r['merge_errors']} | {r['ar_reinit_total']} | "
            f"{r['reasons'][:80]} | {r['address']} |"
        )
    out_md = OUT / "final_impacted_ranking.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md.name}")


if __name__ == "__main__":
    main()
