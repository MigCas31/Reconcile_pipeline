#!/usr/bin/env python3
"""
Produce pipeline-outputs/scan_health.json — a per-building summary of any
AR / scan failures observed in PostHog or in the reconciler, for the
viewer to surface.

Source data:
  analysis_outputs/broken_vs_clean_features.csv     (per-room labels)
  analysis_outputs/posthog/scan_trouble_events_by_home.csv  (merge/upload errors)
  .scan-cache/*/data.json                           (user-created storeys)
  pipeline-outputs/<uuid>/reconciled.json           (multi_session flag)
  pipeline-outputs/<uuid>/merged.json               (presence = pipeline ran)

Output schema:
  {
    "<uuid>": {
      "broken_rooms": int,        # rooms that never reached roomScanCompleted
      "merge_errors": int,        # mergeRoomsError events (Float.nan crashes)
      "upload_failed": int,       # scanUploadFailed events
      "pipeline_broken": bool,    # no merged.json on disk
      "multi_session": int,       # reconciler ARKit session count
      "user_storeys": int,        # raw homeMetadata.floors count from scan-cache
      "multi_session_excess": int,# sessions beyond expected one per user storey
      "summary": str              # short label for the UI
    },
    ...
  }
Only buildings with at least one issue are included.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

MULTI_SESSION_RE = re.compile(r"multi_session\s*\((\d+)\s*scan\s*sessions?\)")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

REPO = Path(__file__).parent
ANALYSIS = REPO / "analysis_outputs"
PIPE = REPO / "pipeline-outputs"
SCAN_CACHE = REPO / ".scan-cache"
OUT = PIPE / "scan_health.json"


def to_int(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def raw_user_storeys(uuid: str) -> int:
    """Count user-created storeys from raw scan-cache metadata."""
    if not SCAN_CACHE.exists():
        return 0
    for data_path in SCAN_CACHE.glob(f"*{uuid}*/data.json"):
        try:
            with data_path.open() as f:
                data = json.load(f)
            return len(data.get("homeMetadata", {}).get("floors", []) or [])
        except Exception:
            return 0
    return 0


def main() -> None:
    broken_per_home: dict[str, int] = defaultdict(int)
    with (ANALYSIS / "broken_vs_clean_features.csv").open() as f:
        for r in csv.DictReader(f):
            if r["label"] == "BROKEN_ROOM":
                broken_per_home[r["home_id"]] += 1

    trouble: dict[str, dict] = {}
    with (ANALYSIS / "posthog" / "scan_trouble_events_by_home.csv").open() as f:
        for r in csv.DictReader(f):
            trouble[r["home_id"]] = r

    out: dict[str, dict] = {}
    # Iterate every pipeline-output directory so we also catch buildings whose
    # only signal is reconciler `multi_session` (no PostHog rows, no broken
    # rooms — but ARKit clearly broke into multiple sessions).
    candidate_ids = {
        d.name for d in PIPE.iterdir() if d.is_dir() and UUID_RE.match(d.name)
    }
    candidate_ids |= set(broken_per_home) | set(trouble)
    candidate_ids = {uuid for uuid in candidate_ids if uuid and UUID_RE.match(uuid)}

    for uuid in candidate_ids:
        broken = broken_per_home.get(uuid, 0)
        t = trouble.get(uuid, {})
        merge_errs = to_int(t.get("merge_errors"))
        upload_fails = to_int(t.get("upload_failed"))

        building_dir = PIPE / uuid
        if not building_dir.is_dir():
            continue

        merged_path = building_dir / "merged.json"
        recon_path = building_dir / "reconciled.json"
        pipeline_broken = not merged_path.exists()

        # multi_session count from reconciler's own flags
        multi_session = 0
        if recon_path.exists():
            try:
                rec = json.load(recon_path.open()).get("reconciliation", {}) or {}
                for flag in rec.get("flags", []) or []:
                    m = MULTI_SESSION_RE.search(flag)
                    if m:
                        multi_session = int(m.group(1))
                        break
            except Exception:
                pass

        user_storeys = raw_user_storeys(uuid)
        extra_sessions = (
            max(0, multi_session - user_storeys) if user_storeys else multi_session
        )

        if not (
            broken
            or merge_errs
            or upload_fails
            or pipeline_broken
            or extra_sessions > 0
        ):
            continue

        parts: list[str] = []
        if pipeline_broken:
            parts.append("no merged.json")
        if broken:
            parts.append(f"{broken} broken room{'s' if broken != 1 else ''}")
        if merge_errs:
            parts.append(f"{merge_errs} mergeRoomsError")
        if upload_fails:
            parts.append(f"{upload_fails} scanUploadFailed")
        if extra_sessions:
            if user_storeys:
                session_word = "session" if extra_sessions == 1 else "sessions"
                storey_word = "storey" if user_storeys == 1 else "storeys"
                parts.append(
                    f"{extra_sessions} extra ARKit {session_word} "
                    f"({multi_session} for {user_storeys} user {storey_word})"
                )
            else:
                parts.append(f"{multi_session} ARKit sessions")

        out[uuid] = {
            "broken_rooms": broken,
            "merge_errors": merge_errs,
            "upload_failed": upload_fails,
            "pipeline_broken": pipeline_broken,
            "multi_session": multi_session,
            "user_storeys": user_storeys,
            "multi_session_excess": extra_sessions,
            "summary": " · ".join(parts),
        }

    with OUT.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"Wrote {OUT}: {len(out)} buildings flagged")
    for uuid, info in sorted(
        out.items(),
        key=lambda x: (
            -int(x[1]["pipeline_broken"]),
            -x[1]["merge_errors"],
            -x[1]["broken_rooms"],
        ),
    )[:10]:
        print(f"  {uuid}  {info['summary']}")


if __name__ == "__main__":
    main()
