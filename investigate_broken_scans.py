#!/usr/bin/env python3
"""
Stop classifying, start investigating.

For every roomScanStarted event, compute features about its CONTEXT:
  - gap_since_prev_scan_min      time since the previous Started in this home
  - story_change                 did the room story change since the last scan?
  - position_in_session          1-indexed room number within the current session
                                 (a new session starts after >30min idle)
  - session_room_count           total rooms in the session this scan belongs to
  - session_age_min              minutes since the first event in this session
  - reinits_in_session_so_far    cumulative ar_reinit events in this session
                                 before this scan started
  - scan_duration_sec            time from Started to terminating event
  - weekday, hour                when the scan started
  - surveyor_id                  the distinct_id

Then: compare BROKEN vs CLEAN distributions for each feature. The goal
is to find features that actually distinguish broken from clean — not
just describe the broken set.

Also: dump 5 full event sequences around BROKEN scans to read qualitatively.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).parent
EVENTS = REPO / "analysis_outputs" / "posthog" / "all_scan_events.csv"
OUT_FEATURES = REPO / "analysis_outputs" / "broken_vs_clean_features.csv"
OUT_REPORT = REPO / "analysis_outputs" / "broken_scan_investigation.md"

SESSION_GAP = timedelta(minutes=30)


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_events_by_home() -> dict[str, list[dict]]:
    by_home: dict[str, list[dict]] = defaultdict(list)
    with EVENTS.open() as f:
        for r in csv.DictReader(f):
            home = r["home_id"]
            if not home:
                continue
            r["ts"] = parse_ts(r["timestamp"])
            by_home[home].append(r)
    for h in by_home:
        by_home[h].sort(key=lambda x: x["ts"])
    return by_home


def assign_sessions(events: list[dict]) -> None:
    """Mark each event with session_idx (gap >SESSION_GAP starts new session)."""
    sid = 0
    prev = None
    for e in events:
        if prev and (e["ts"] - prev["ts"]) > SESSION_GAP:
            sid += 1
        e["session_idx"] = sid
        prev = e


def extract_scans_with_features(events: list[dict]) -> list[dict]:
    """One record per ROOM (not per scan attempt). A room is BROKEN iff it
    never reaches roomScanCompleted even after retakes."""
    # First pass: which (home, room) IDs ever completed?
    completed_rooms: set[tuple[str, str]] = set()
    for e in events:
        if e["event"] == "roomScanCompleted" and e["room_id"]:
            completed_rooms.add((e["home_id"], e["room_id"]))

    # Walk events: emit one record per FIRST roomScanStarted of each room
    rooms_seen: set[str] = set()
    out: list[dict] = []
    sessions = defaultdict(
        lambda: {
            "first_ts": None,
            "rooms_started": 0,
            "reinits_before_room": 0,
            "broken_so_far": 0,
        }
    )
    prev_room_ts: datetime | None = None
    prev_room_story: str | None = None

    # Track per-(home, room) attempt counts and timing
    attempts_by_room: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e["event"] in (
            "roomScanStarted",
            "roomScanRetakeRequested",
            "roomScanCompleted",
            "roomScanCancelled",
            "roomScanStopped",
            "roomScanError",
            "initializingARStateOutOfTheBlue",
        ):
            if e["room_id"]:
                attempts_by_room[e["room_id"]].append(e)

    for e in events:
        sid = e["session_idx"]
        s = sessions[sid]
        if s["first_ts"] is None:
            s["first_ts"] = e["ts"]

        ev = e["event"]

        if ev == "initializingARStateOutOfTheBlue":
            s["reinits_before_room"] += 1
            continue

        if ev != "roomScanStarted":
            continue

        room = e["room_id"]
        if not room or room in rooms_seen:
            continue
        rooms_seen.add(room)

        # Build features for this room
        room_attempts = attempts_by_room.get(room, [])
        starts = [a for a in room_attempts if a["event"] == "roomScanStarted"]
        completes = [a for a in room_attempts if a["event"] == "roomScanCompleted"]
        retakes = [a for a in room_attempts if a["event"] == "roomScanRetakeRequested"]
        errors = [a for a in room_attempts if a["event"] == "roomScanError"]
        reinits_during = [
            a
            for a in room_attempts
            if a["event"] == "initializingARStateOutOfTheBlue"
            and starts
            and a["ts"] >= starts[0]["ts"]
            and (not completes or a["ts"] <= completes[-1]["ts"])
        ]

        completed = (e["home_id"], room) in completed_rooms

        s["rooms_started"] += 1
        rec = {
            "home_id": e["home_id"],
            "lead_id": e["lead_id"],
            "room_id": room,
            "story_id": e["story_id"],
            "surveyor_id": e["distinct_id"],
            "started_at": e["ts"],
            "weekday": e["ts"].weekday(),
            "hour": e["ts"].hour,
            "n_starts": len(starts),
            "n_retakes": len(retakes),
            "n_completes": len(completes),
            "n_errors": len(errors),
            "n_reinits_during": len(reinits_during),
            "label": "COMPLETED" if completed else "BROKEN_ROOM",
            "session_idx": sid,
            "position_in_session": s["rooms_started"],
            "session_age_min_at_start": (
                (e["ts"] - s["first_ts"]).total_seconds() / 60
            ),
            "reinits_in_session_before_room": s["reinits_before_room"],
            "broken_in_session_so_far": s["broken_so_far"],
            "story_change": (
                prev_room_story is not None
                and e["story_id"] != ""
                and e["story_id"] != prev_room_story
            ),
            "gap_since_prev_room_min": (
                (e["ts"] - prev_room_ts).total_seconds() / 60 if prev_room_ts else None
            ),
        }

        # Time elapsed across all attempts of this room
        if completes:
            rec["time_to_complete_sec"] = (
                completes[-1]["ts"] - starts[0]["ts"]
            ).total_seconds()
        else:
            rec["time_to_complete_sec"] = None

        if not completed:
            s["broken_so_far"] += 1

        out.append(rec)
        prev_room_ts = e["ts"]
        if e["story_id"]:
            prev_room_story = e["story_id"]

    # Fill session_room_count after the fact
    counts: dict[tuple, int] = defaultdict(int)
    for sc in out:
        counts[(sc["home_id"], sc["session_idx"])] += 1
    for sc in out:
        sc["session_room_count"] = counts[(sc["home_id"], sc["session_idx"])]
    return out


# (no per-attempt finalize needed any more — see extract_scans_with_features)


def compare(scans: list[dict], feature: str, predicate=lambda v: v is not None) -> str:
    """Return a one-liner comparing the feature distribution for BROKEN vs COMPLETED."""
    broken = [
        s[feature]
        for s in scans
        if s["label"] == "BROKEN_ROOM" and predicate(s[feature])
    ]
    clean = [
        s[feature] for s in scans if s["label"] == "COMPLETED" and predicate(s[feature])
    ]
    if not broken or not clean:
        return f"  {feature}: not enough data"

    def stats(vs):
        return (
            statistics.median(vs),
            statistics.mean(vs),
            sum(1 for v in vs if v) / len(vs)
            if all(isinstance(v, bool) for v in vs)
            else statistics.quantiles(vs, n=4)[2]
            if len(vs) > 4
            else max(vs),
        )

    if all(isinstance(v, bool) for v in broken + clean):
        bb = sum(broken) / len(broken)
        cc = sum(clean) / len(clean)
        return (
            f"  {feature:<35s} BROKEN: {100 * bb:5.1f}% true (n={len(broken)})  "
            f"CLEAN: {100 * cc:5.1f}% true (n={len(clean)})  "
            f"lift {bb / cc if cc else 0:.2f}x"
        )

    bm = statistics.median(broken)
    cm = statistics.median(clean)
    bmean = statistics.mean(broken)
    cmean = statistics.mean(clean)
    bp75 = statistics.quantiles(broken, n=4)[2] if len(broken) > 4 else max(broken)
    cp75 = statistics.quantiles(clean, n=4)[2] if len(clean) > 4 else max(clean)
    return (
        f"  {feature:<35s} BROKEN: median={bm:7.1f} mean={bmean:7.1f} p75={bp75:7.1f}  "
        f"COMPLETED: median={cm:7.1f} mean={cmean:7.1f} p75={cp75:7.1f}"
    )


def dump_sequence_around(events: list[dict], scan: dict, n: int = 8) -> str:
    """Return a text block showing N events before & after the BROKEN scan's Started."""
    target_ts = scan["started_at"]
    idx = next((i for i, e in enumerate(events) if e["ts"] == target_ts), None)
    if idx is None:
        return f"(could not locate event at {target_ts})"
    lo, hi = max(0, idx - n), min(len(events), idx + n + 1)
    out = []
    for i in range(lo, hi):
        e = events[i]
        marker = " *START*" if i == idx else ""
        room = (e["room_id"] or "")[:8]
        story = e["story_id"] or "-"
        err = (e["error_text"] or "")[:60]
        out.append(
            f"    {e['ts'].strftime('%H:%M:%S')}  s{e['session_idx']}  "
            f"{e['event']:<35s} room={room:<8s} story={story} {err}{marker}"
        )
    return "\n".join(out)


def main() -> None:
    by_home = load_events_by_home()
    print(f"Loaded {len(by_home)} homes")

    all_scans: list[dict] = []
    home_to_events: dict[str, list[dict]] = {}
    for home, events in by_home.items():
        assign_sessions(events)
        scans = extract_scans_with_features(events)
        all_scans.extend(scans)
        home_to_events[home] = events

    from collections import Counter

    print(
        f"Scans: {len(all_scans)}  "
        f"labels={dict(Counter(s['label'] for s in all_scans))}"
    )

    # Write per-scan features CSV
    if all_scans:
        with OUT_FEATURES.open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(all_scans[0].keys()), extrasaction="ignore"
            )
            w.writeheader()
            for s in all_scans:
                row = {**s, "started_at": s["started_at"].isoformat()}
                w.writerow(row)
        print(f"Wrote {OUT_FEATURES.name}")

    # === COMPARISONS: BROKEN vs CLEAN on each feature ===
    print("\n=== HYPOTHESIS TESTS: features that distinguish BROKEN from CLEAN ===")
    print("(Looking for features where BROKEN distribution ≠ CLEAN distribution)\n")

    print("--- Time / pacing features ---")
    print(compare(all_scans, "gap_since_prev_room_min"))
    print(compare(all_scans, "session_age_min_at_start"))
    print(compare(all_scans, "time_to_complete_sec"))

    print("\n--- Retake / attempt features ---")
    print(compare(all_scans, "n_starts"))
    print(compare(all_scans, "n_retakes"))
    print(compare(all_scans, "n_errors"))
    print(compare(all_scans, "n_reinits_during"))

    print("\n--- Position in session ---")
    print(compare(all_scans, "position_in_session"))
    print(compare(all_scans, "session_room_count"))
    print(compare(all_scans, "broken_in_session_so_far"))

    print("\n--- Cross-room context ---")
    print(compare(all_scans, "story_change", predicate=lambda v: True))
    print(compare(all_scans, "reinits_in_session_before_room"))

    print("\n--- Time-of-day ---")
    print(compare(all_scans, "hour"))
    print(compare(all_scans, "weekday"))

    # === SURVEYOR breakdown: are some surveyors more impacted? ===
    print("\n=== SURVEYOR breakdown (top 10 by scan count) ===")
    by_surveyor = defaultdict(lambda: Counter())
    for s in all_scans:
        by_surveyor[s["surveyor_id"]][s["label"]] += 1
    surveyors = sorted(by_surveyor.items(), key=lambda x: -sum(x[1].values()))
    print(f"  {'surveyor':<40s} {'rooms':<7s} {'%broken_room':<14s}")
    for sid, c in surveyors[:15]:
        total = sum(c.values())
        if total < 10:
            continue
        b = c.get("BROKEN_ROOM", 0)
        print(f"  {sid:<40s} {total:<7d} {100 * b / total:>6.1f}%")

    # === BROKEN-IN-SESSION cascade: does one break predict more? ===
    print(
        "\n=== CASCADE: does a broken scan predict more breaks in the same session? ==="
    )
    sessions_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for s in all_scans:
        sessions_grouped[(s["home_id"], s["session_idx"])].append(s)

    sessions_with_break = 0
    sessions_total = 0
    breaks_after_break = 0
    breaks_total = 0
    for sess_scans in sessions_grouped.values():
        sessions_total += 1
        labels = [s["label"] for s in sess_scans]
        if "BROKEN_ROOM" in labels:
            sessions_with_break += 1
            first_break = labels.index("BROKEN_ROOM")
            breaks_after = sum(
                1 for lab in labels[first_break + 1 :] if lab == "BROKEN_ROOM"
            )
            breaks_total += sum(1 for lab in labels if lab == "BROKEN_ROOM")
            breaks_after_break += breaks_after
    print(
        f"  Sessions with >=1 break: {sessions_with_break}/{sessions_total} "
        f"({100 * sessions_with_break / sessions_total:.1f}%)"
    )
    if breaks_total:
        print(
            f"  Of all breaks, {breaks_after_break}/{breaks_total} "
            f"({100 * breaks_after_break / breaks_total:.1f}%) followed an earlier "
            f"break in the same session"
        )

    # === DUMP 5 BROKEN-scan event sequences ===
    print("\n=== EVENT SEQUENCES around 5 BROKEN scans ===")
    broken_scans = [s for s in all_scans if s["label"] == "BROKEN_ROOM"]
    # Pick 5 from different homes for variety
    seen_homes: set[str] = set()
    samples = []
    for s in broken_scans:
        if s["home_id"] not in seen_homes:
            samples.append(s)
            seen_homes.add(s["home_id"])
        if len(samples) >= 5:
            break

    for s in samples:
        gap = s["gap_since_prev_room_min"]
        gap_str = f"{gap:.1f}min" if gap is not None else "first room"
        print(
            f"\n  --- {s['home_id']} room={s['room_id'][:8]} story={s['story_id']} "
            f"surveyor={s['surveyor_id'][:8]} ---"
        )
        print(
            f"    pos={s['position_in_session']}/{s['session_room_count']}  "
            f"gap_since_prev={gap_str}  "
            f"story_change={s['story_change']}  "
            f"n_starts={s['n_starts']} n_retakes={s['n_retakes']} "
            f"n_errors={s['n_errors']} n_reinits_during={s['n_reinits_during']}"
        )
        print(dump_sequence_around(home_to_events[s["home_id"]], s, n=10))


if __name__ == "__main__":
    main()
