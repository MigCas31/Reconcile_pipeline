#!/usr/bin/env python3
"""
Detect ARKit session breaks in building scans by analysing raw room files
from .scan-cache.

Key insight: rooms scanned within the same ARKit session share the same
referenceOriginTransform *rotation* (yaw). A different yaw = different session.
The translation component shifts per room (device position), but rotation is fixed
per session.

Analyses:
1. Yaw clustering of referenceOriginTransform -> identifies distinct ARKit sessions
2. Per-session spatial analysis (room centroid spread, Y-offsets)
3. Time-gap analysis from room creation timestamps
4. Cross-session alignment quality assessment
5. Shared wall transform consistency
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(
    "/Users/martincollignon/conductor/workspaces/look-ma-no-hands/tirana/pipeline-outputs"
)
CACHE = Path(
    "/Users/martincollignon/conductor/workspaces/look-ma-no-hands/tirana/.scan-cache"
)

PROBLEM_UUIDS = [
    "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3",
    "1d26eda3-c927-4cdf-a3b0-218036828a55",
    "7dbc53a6-17e8-4806-83de-42286b95726c",
    "e661e7b6-303d-415c-b378-2d9dd2fbfd6f",
    "cb711a0b-6e8d-4ae6-b008-af3297446dcc",
    "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b",
    "feccbd0c-0420-4775-b5b7-49b99559947e",
    "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5",
]

random.seed(42)
all_uuids = sorted(
    [d.name for d in BASE.iterdir() if d.is_dir() and (d / "merged.json").exists()]
)
control_candidates = [u for u in all_uuids if u not in PROBLEM_UUIDS]
CONTROL_UUIDS = random.sample(control_candidates, 5)


def parse_transform_4x4(flat16):
    return np.array(flat16, dtype=np.float64).reshape(4, 4).T


def yaw_from_transform(flat16):
    mat = parse_transform_4x4(flat16)
    R = mat[:3, :3]
    return math.degrees(math.atan2(R[0, 2], R[0, 0]))


def extract_translation(flat16):
    mat = parse_transform_4x4(flat16)
    return mat[:3, 3]


def rotation_angle_between(R1, R2):
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle_rad = math.acos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(angle_rad)


def cluster_yaws(yaws, eps=2.0):
    """Cluster yaw angles (degrees), handling wrap-around at +/-180."""
    if not yaws:
        return []
    indexed = list(enumerate(yaws))
    indexed.sort(key=lambda x: x[1])
    clusters = []
    current = [indexed[0]]
    for i in range(1, len(indexed)):
        diff = abs(indexed[i][1] - indexed[i - 1][1])
        # Handle wrap-around
        diff = min(diff, 360 - diff)
        if diff <= eps:
            current.append(indexed[i])
        else:
            clusters.append(current)
            current = [indexed[i]]
    clusters.append(current)
    # Check wrap: can first and last cluster merge?
    if len(clusters) > 1:
        diff = abs(clusters[-1][-1][1] - clusters[0][0][1])
        diff = min(diff, 360 - diff)
        # Also check angular distance considering wrap
        last_yaw = clusters[-1][-1][1]
        first_yaw = clusters[0][0][1]
        wrap_diff = min(
            abs(last_yaw - first_yaw),
            abs(last_yaw - first_yaw + 360),
            abs(last_yaw - first_yaw - 360),
        )
        if wrap_diff <= eps:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
    return clusters


def find_cache_dir(uuid):
    for d in CACHE.iterdir():
        if uuid in d.name:
            return d
    return None


def analyze_building(uuid):
    results = {"uuid": uuid}

    cache_dir = find_cache_dir(uuid)
    if not cache_dir:
        results["error"] = "no scan-cache directory"
        return results

    data_path = cache_dir / "data.json"
    if not data_path.exists():
        results["error"] = "no data.json"
        return results

    with open(data_path) as f:
        metadata = json.load(f)

    floors = metadata.get("homeMetadata", {}).get("floors", [])
    results["num_floors"] = len(floors)

    # Load all rooms with their raw referenceOriginTransform
    rooms = []
    for fi, fl in enumerate(floors):
        for rm in fl.get("rooms", []):
            jp = rm.get("jsonPath", "")
            fid = jp.split("/")[-1].replace(".json", "") if jp else None
            if not fid:
                continue
            room_path = cache_dir / f"{fid}.json"
            if not room_path.exists():
                continue
            with open(room_path) as f:
                raw = json.load(f)

            ref = raw.get("referenceOriginTransform")
            if not ref or len(ref) != 16:
                continue

            walls = raw.get("walls", [])
            wall_positions = []
            wall_ids = []
            wall_transforms_raw = []
            for w in walls:
                t = w.get("transform")
                if t and len(t) == 16:
                    mat = parse_transform_4x4(t)
                    wall_positions.append(mat[:3, 3])
                    wall_ids.append(w.get("identifier", ""))
                    wall_transforms_raw.append(t)

            centroid = np.mean(wall_positions, axis=0) if wall_positions else None

            rooms.append(
                {
                    "name": rm.get("name", "?"),
                    "created": rm.get("created", 0),
                    "floor_type": fl.get("floorType", "?"),
                    "floor_idx": fi,
                    "story": raw.get("story", 0),
                    "ref_transform": ref,
                    "yaw": yaw_from_transform(ref),
                    "ref_translation": extract_translation(ref),
                    "centroid": centroid,
                    "wall_ids": wall_ids,
                    "wall_transforms_raw": wall_transforms_raw,
                    "num_walls": len(walls),
                }
            )

    if not rooms:
        results["error"] = "no valid rooms"
        return results

    results["num_rooms"] = len(rooms)
    rooms.sort(key=lambda r: r["created"])

    # ===== 1. YAW CLUSTERING -> Session identification =====
    yaws = [r["yaw"] for r in rooms]
    yaw_clusters = cluster_yaws(yaws, eps=2.0)
    results["num_sessions"] = len(yaw_clusters)

    session_info = []
    for si, cluster in enumerate(yaw_clusters):
        member_indices = [idx for idx, _ in cluster]
        member_yaws = [y for _, y in cluster]
        member_rooms = [rooms[idx] for idx in member_indices]
        member_floors = sorted(set(r["floor_type"] for r in member_rooms))
        member_names = [r["name"] for r in member_rooms]
        mean_yaw = np.mean(member_yaws)

        # Time span of this session
        times = sorted(r["created"] for r in member_rooms)
        time_span_min = (times[-1] - times[0]) / 60 if len(times) > 1 else 0

        session_info.append(
            {
                "session_id": si,
                "num_rooms": len(member_rooms),
                "mean_yaw_deg": round(float(mean_yaw), 2),
                "yaw_spread_deg": round(float(max(member_yaws) - min(member_yaws)), 3),
                "floor_types": member_floors,
                "room_names": member_names,
                "time_span_min": round(time_span_min, 1),
            }
        )
    results["sessions"] = session_info

    # ===== 2. Time-gap analysis =====
    time_gaps = []
    for i in range(1, len(rooms)):
        gap = rooms[i]["created"] - rooms[i - 1]["created"]
        time_gaps.append(
            {
                "from": rooms[i - 1]["name"],
                "to": rooms[i]["name"],
                "gap_min": round(gap / 60, 1),
                "yaw_change": round(abs(rooms[i]["yaw"] - rooms[i - 1]["yaw"]), 1),
                "same_session": abs(rooms[i]["yaw"] - rooms[i - 1]["yaw"]) < 2.0
                or abs(abs(rooms[i]["yaw"] - rooms[i - 1]["yaw"]) - 360) < 2.0,
            }
        )
    large_gaps = [g for g in time_gaps if g["gap_min"] > 10]
    session_switches = [g for g in time_gaps if not g["same_session"]]
    results["large_time_gaps"] = large_gaps
    results["session_switches"] = session_switches

    # ===== 3. Y-offset analysis within same story =====
    story_rooms = defaultdict(list)
    for r in rooms:
        if r["centroid"] is not None:
            story_rooms[r["story"]].append(r)

    y_offset_info = {}
    for story, srms in story_rooms.items():
        ys = [r["centroid"][1] for r in srms]
        if len(ys) < 2:
            continue
        y_range = max(ys) - min(ys)
        if y_range > 1.0:
            # Check if Y differences correlate with session differences
            session_yaw_map = {}
            for r in srms:
                for si in session_info:
                    if r["name"] in si["room_names"]:
                        session_yaw_map[r["name"]] = (
                            r["centroid"][1],
                            si["session_id"],
                        )
                        break
            y_offset_info[str(story)] = {
                "y_range_m": round(float(y_range), 3),
                "num_rooms": len(srms),
                "room_y_session": {
                    k: (round(float(v[0]), 3), v[1]) for k, v in session_yaw_map.items()
                },
            }
    results["y_offset_issues"] = y_offset_info

    # ===== 4. Cross-session alignment quality =====
    if len(session_info) > 1:
        # For each pair of sessions, check if rooms from different sessions
        # on the same floor have overlapping spatial extent
        alignment_checks = []
        for si_a in range(len(yaw_clusters)):
            for si_b in range(si_a + 1, len(yaw_clusters)):
                rooms_a = [
                    rooms[idx]
                    for idx, _ in yaw_clusters[si_a]
                    if rooms[idx]["centroid"] is not None
                ]
                rooms_b = [
                    rooms[idx]
                    for idx, _ in yaw_clusters[si_b]
                    if rooms[idx]["centroid"] is not None
                ]
                if rooms_a and rooms_b:
                    centroids_a = np.array([r["centroid"] for r in rooms_a])
                    centroids_b = np.array([r["centroid"] for r in rooms_b])
                    # Distance between nearest centroids
                    min_dist = float("inf")
                    for ca in centroids_a:
                        for cb in centroids_b:
                            d = float(np.linalg.norm(ca - cb))
                            min_dist = min(min_dist, d)
                    # Mean center distance
                    mean_a = np.mean(centroids_a, axis=0)
                    mean_b = np.mean(centroids_b, axis=0)
                    center_dist = float(np.linalg.norm(mean_a - mean_b))
                    # Yaw difference
                    yaw_diff = abs(
                        session_info[si_a]["mean_yaw_deg"]
                        - session_info[si_b]["mean_yaw_deg"]
                    )
                    yaw_diff = min(yaw_diff, 360 - yaw_diff)

                    alignment_checks.append(
                        {
                            "sessions": (si_a, si_b),
                            "yaw_diff_deg": round(yaw_diff, 2),
                            "min_centroid_dist_m": round(min_dist, 2),
                            "center_dist_m": round(center_dist, 2),
                        }
                    )
        results["cross_session_alignment"] = alignment_checks

    # ===== 5. Shared wall transform consistency =====
    wall_occurrences = defaultdict(list)
    for ri, r in enumerate(rooms):
        for wi, wid in enumerate(r["wall_ids"]):
            if wid and wi < len(r["wall_transforms_raw"]):
                wall_occurrences[wid].append((ri, r["wall_transforms_raw"][wi]))

    shared_walls = {k: v for k, v in wall_occurrences.items() if len(v) > 1}
    results["num_shared_walls"] = len(shared_walls)

    wall_disagreements = []
    for wid, occs in shared_walls.items():
        for a in range(len(occs)):
            for b in range(a + 1, len(occs)):
                ta = parse_transform_4x4(occs[a][1])
                tb = parse_transform_4x4(occs[b][1])
                t_diff = float(np.linalg.norm(ta[:3, 3] - tb[:3, 3]))
                r_diff = rotation_angle_between(ta[:3, :3], tb[:3, :3])
                if t_diff > 0.01 or r_diff > 0.5:
                    wall_disagreements.append(
                        {
                            "wall_id": wid[:12],
                            "rooms": (
                                rooms[occs[a][0]]["name"],
                                rooms[occs[b][0]]["name"],
                            ),
                            "trans_diff_m": round(t_diff, 4),
                            "rot_diff_deg": round(r_diff, 2),
                        }
                    )
    results["wall_disagreements"] = wall_disagreements[:10]

    # Get reconciliation metadata
    recon_path = BASE / uuid / "reconciled.json"
    if recon_path.exists():
        with open(recon_path) as f:
            recon = json.load(f)
        rc = recon.get("reconciliation", {})
        results["recon_sessions"] = rc.get("scan_session_count")
        results["recon_class"] = rc.get("classification")
        results["recon_gap_median_cm"] = rc.get("floor_gap_median_cm")
    else:
        results["recon_sessions"] = "?"
        results["recon_class"] = "?"
        results["recon_gap_median_cm"] = "?"

    return results


def print_report(label, r):
    uuid = r["uuid"]
    print(f"\n{'=' * 90}")
    print(f"  {label}: {uuid}")
    if r.get("error"):
        print(f"  ERROR: {r['error']}")
        return
    print(
        f"  Rooms: {r['num_rooms']} | Floors: {r['num_floors']} | "
        f"Detected sessions: {r['num_sessions']} | "
        f"Recon sessions: {r.get('recon_sessions', '?')} | "
        f"Class: {r.get('recon_class', '?')} | "
        f"Gap median: {r.get('recon_gap_median_cm', '?')}cm"
    )
    print(f"{'=' * 90}")

    # Sessions
    for s in r.get("sessions", []):
        print(
            f"\n  Session {s['session_id']}: {s['num_rooms']} rooms, "
            f"yaw={s['mean_yaw_deg']}deg (spread={s['yaw_spread_deg']}deg), "
            f"floors={s['floor_types']}, span={s['time_span_min']}min"
        )
        if len(s["room_names"]) <= 15:
            print(f"    rooms: {s['room_names']}")

    # Session switches (yaw changes over time)
    switches = r.get("session_switches", [])
    if switches:
        print(f"\n  Session switches in capture order ({len(switches)} transitions):")
        for sw in switches:
            tg = f"(gap={sw['gap_min']}min)" if sw["gap_min"] > 5 else ""
            print(
                f"    {sw['from']:20s} -> {sw['to']:20s}  "
                f"yaw_change={sw['yaw_change']}deg {tg}"
            )

    # Time gaps
    gaps = r.get("large_time_gaps", [])
    if gaps:
        print(f"\n  Large time gaps (>10min): {len(gaps)}")
        for g in gaps:
            print(f"    {g['from']:20s} -> {g['to']:20s}: {g['gap_min']}min")

    # Y-offset issues
    y_issues = r.get("y_offset_issues", {})
    if y_issues:
        print("\n  Y-offset issues (same story, different Y):")
        for story, info in y_issues.items():
            print(
                f"    Story {story}: Y range = {info['y_range_m']}m across "
                f"{info['num_rooms']} rooms"
            )
            for name, (y, sid) in info.get("room_y_session", {}).items():
                print(f"      {name:20s} Y={y:7.3f}m  session={sid}")

    # Cross-session alignment
    alignment = r.get("cross_session_alignment", [])
    if alignment:
        print("\n  Cross-session alignment:")
        for a in alignment:
            print(
                f"    Sessions {a['sessions']}: yaw_diff={a['yaw_diff_deg']}deg, "
                f"min_dist={a['min_centroid_dist_m']}m, "
                f"center_dist={a['center_dist_m']}m"
            )

    # Shared wall disagreements
    wd = r.get("wall_disagreements", [])
    if wd:
        print(f"\n  Shared wall disagreements: {len(wd)}")
        for d in wd[:5]:
            print(
                f"    {d['wall_id']} between {d['rooms']}: "
                f"trans={d['trans_diff_m']}m, rot={d['rot_diff_deg']}deg"
            )

    # Verdict
    sessions = r["num_sessions"]
    print("\n  VERDICT: ", end="")
    if sessions > 1:
        print(f"** {sessions} ARKit SESSIONS DETECTED **")
        if sessions != r.get("recon_sessions"):
            print(f"    (reconciler detected {r.get('recon_sessions')} sessions)")
    else:
        print("Single ARKit session (no session breaks)")


def main():
    print("=" * 90)
    print("  ARKit SESSION BREAK DETECTION")
    print("  Method: referenceOriginTransform yaw clustering")
    print("=" * 90)
    print(f"\n  Problem buildings: {len(PROBLEM_UUIDS)}")
    print(f"  Control buildings: {len(CONTROL_UUIDS)}")

    all_results = []
    for label, uuids in [("PROBLEM", PROBLEM_UUIDS), ("CONTROL", CONTROL_UUIDS)]:
        for uuid in uuids:
            r = analyze_building(uuid)
            all_results.append((label, r))
            print_report(label, r)

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(
        f"\n{'UUID':<40} {'Type':<8} {'Rm':<4} {'Sess':<5} {'RSes':<5} {'TGap':<5} "
        f"{'SwYaw':<6} {'YOff':<5} {'ShWD':<5} {'Class':<7} {'GapCm':<6}"
    )
    print("-" * 110)

    for label, r in all_results:
        if r.get("error"):
            print(f"{r['uuid']:<40} {label:<8} ERR")
            continue
        y_flag = "Y" if r.get("y_offset_issues") else "-"
        print(
            f"{r['uuid']:<40} {label:<8} "
            f"{r['num_rooms']:<4} "
            f"{r['num_sessions']:<5} "
            f"{r.get('recon_sessions', '?'):<5} "
            f"{len(r.get('large_time_gaps', [])):<5} "
            f"{len(r.get('session_switches', [])):<6} "
            f"{y_flag:<5} "
            f"{len(r.get('wall_disagreements', [])):<5} "
            f"{r.get('recon_class', '?'):<7} "
            f"{r.get('recon_gap_median_cm', '?')}"
        )

    # Key findings
    print(f"\n\n{'=' * 90}")
    print("  KEY FINDINGS")
    print(f"{'=' * 90}")

    prob_sess = [
        r["num_sessions"]
        for label, r in all_results
        if label == "PROBLEM" and not r.get("error")
    ]
    ctrl_sess = [
        r["num_sessions"]
        for label, r in all_results
        if label == "CONTROL" and not r.get("error")
    ]
    prob_multi = sum(1 for s in prob_sess if s > 1)
    ctrl_multi = sum(1 for s in ctrl_sess if s > 1)

    print(
        f"\n  Problem buildings with multiple sessions: {prob_multi}/{len(prob_sess)} "
        f"(avg {np.mean(prob_sess):.1f} sessions)"
    )
    print(
        f"  Control buildings with multiple sessions: {ctrl_multi}/{len(ctrl_sess)} "
        f"(avg {np.mean(ctrl_sess):.1f} sessions)"
    )

    prob_recon = [
        r.get("recon_sessions", 1)
        for label, r in all_results
        if label == "PROBLEM" and not r.get("error")
    ]
    ctrl_recon = [
        r.get("recon_sessions", 1)
        for label, r in all_results
        if label == "CONTROL" and not r.get("error")
    ]
    print("\n  Reconciler session counts:")
    print(f"    Problem: {prob_recon}")
    print(f"    Control: {ctrl_recon}")

    print("\n  Session count comparison (detected via yaw vs reconciler):")
    for label, r in all_results:
        if r.get("error"):
            continue
        match = "MATCH" if r["num_sessions"] == r.get("recon_sessions") else "MISMATCH"
        print(
            f"    {r['uuid'][:20]}.. {label:8s} yaw={r['num_sessions']} "
            f"recon={r.get('recon_sessions')} {match}"
        )

    # Session break impact analysis
    print("\n  Buildings by session count and classification:")
    for label, r in all_results:
        if r.get("error"):
            continue
        if r["num_sessions"] > 1:
            sess_yaws = [s["mean_yaw_deg"] for s in r["sessions"]]
            yaw_diffs = []
            for i in range(len(sess_yaws)):
                for j in range(i + 1, len(sess_yaws)):
                    d = abs(sess_yaws[i] - sess_yaws[j])
                    d = min(d, 360 - d)
                    yaw_diffs.append(round(d, 1))
            print(
                f"    {r['uuid'][:20]}.. {label:8s} "
                f"sessions={r['num_sessions']}, yaw_diffs={yaw_diffs}deg, "
                f"class={r.get('recon_class')}, "
                f"gap_median={r.get('recon_gap_median_cm')}cm"
            )


if __name__ == "__main__":
    main()
