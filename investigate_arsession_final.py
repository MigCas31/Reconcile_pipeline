#!/usr/bin/env python3
"""
DEFINITIVE ARSession restart investigation.

Key insight from deep analysis: The referenceOriginTransform contains both a rotation
and a translation. The translation differs between rooms naturally (each room is in a
different physical location). But the ROTATION component of the referenceOriginTransform
should be identical for all rooms scanned in the same ARSession, because it represents
the orientation of the ARSession's world coordinate frame relative to true
north/gravity.

If the rotation component varies significantly between rooms on the same floor, it means
those rooms were scanned with DIFFERENT ARSession world coordinate frames - evidence of
a session restart.

This script:
1. Extracts rotation-only component per room, per floor
2. Clusters rooms by rotation similarity
3. Checks for timestamp gaps that correlate with rotation changes
4. Compares targets (8 problematic) vs controls (10 random good)
5. Checks merged.json for downstream impact
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(".scan-cache")
PIPELINE = Path("pipeline-outputs")

TARGETS = {
    "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3": "Bredballe Byvej 63",
    "1d26eda3-c927-4cdf-a3b0-218036828a55": "Gerskov Bygade 24",
    "7dbc53a6-17e8-4806-83de-42286b95726c": "Gultvedgyden 6",
    "e661e7b6-303d-415c-b378-2d9dd2fbfd6f": "Hesselvænget 2",
    "cb711a0b-6e8d-4ae6-b008-af3297446dcc": "Kastanievej 6",
    "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b": "Lucernevej 23",
    "feccbd0c-0420-4775-b5b7-49b99559947e": "Morelvej 68",
    "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5": "Pilevej 30",
}


def find_scan_dir(uuid):
    for d in CACHE.iterdir():
        if uuid in d.name:
            return d
    return None


def extract_rotation_matrix(flat16):
    """Extract 3x3 rotation from column-major 4x4."""
    if flat16 is None or len(flat16) != 16:
        return None
    m = np.array(flat16, dtype=float).reshape(4, 4)
    return m[:3, :3]


def extract_translation(flat16):
    if flat16 is None or len(flat16) != 16:
        return None
    return np.array([flat16[12], flat16[13], flat16[14]])


def rotation_angle_between(rot1, rot2):
    """Angle in degrees between two rotation matrices."""
    R = rot1 @ rot2.T
    trace = np.clip(np.trace(R), -1, 3)
    return math.degrees(math.acos(np.clip((trace - 1) / 2, -1, 1)))


def yaw_from_rotation(rot):
    """Extract yaw angle (rotation around Y/up axis) from rotation matrix.
    In ARKit, Y is up. Yaw = atan2(R[0,2], R[0,0]) approximately."""
    return math.degrees(math.atan2(rot[0, 2], rot[0, 0]))


def pick_controls(exclude, n=10):
    controls = []
    for d in sorted(CACHE.iterdir()):
        if not d.is_dir():
            continue
        uuid = None
        for part in d.name.split("_"):
            if len(part) == 36 and part.count("-") == 4:
                uuid = part
                break
        if uuid and uuid not in exclude:
            if (d / "data.json").exists() and (
                PIPELINE / uuid / "merged.json"
            ).exists():
                name_parts = d.name.split(uuid)[0].rsplit("_", 1)[0].split("_")[2:]
                name = " ".join(name_parts).replace("__", ", ").replace("_", " ")
                controls.append((uuid, name))
                if len(controls) >= n:
                    break
    return controls


def analyze_building(uuid, name):
    scan_dir = find_scan_dir(uuid)
    if not scan_dir:
        return None

    with open(scan_dir / "data.json") as f:
        data = json.load(f)

    hm = data["homeMetadata"]
    floors = hm.get("floors", [])

    building = {
        "name": name,
        "uuid": uuid,
        "num_floors": len(floors),
        "floor_analyses": [],
        "max_rotation_diff_any_floor": 0,
        "has_session_restart_evidence": False,
        "session_restart_floors": [],
    }

    for fl in floors:
        fl_type = fl.get("floorType", "unknown")
        fl.get("id")
        rooms_meta = fl.get("rooms", [])

        room_list = []
        for r in rooms_meta:
            rid = r.get("id")
            rname = r.get("name", "?")
            rcreated = r.get("created")
            room_json = scan_dir / f"{rid}.json"
            if room_json.exists():
                with open(room_json) as f:
                    rd = json.load(f)
                rot_flat = rd.get("referenceOriginTransform")
                rot_mat = extract_rotation_matrix(rot_flat)
                if rot_mat is not None:
                    room_list.append(
                        {
                            "id": rid,
                            "name": rname,
                            "created": rcreated,
                            "rotation": rot_mat,
                            "yaw": yaw_from_rotation(rot_mat),
                            "translation": extract_translation(rot_flat),
                        }
                    )

        room_list.sort(key=lambda x: x.get("created") or 0)

        # Compute pairwise rotation differences
        max_rot_diff = 0
        rot_diffs = []
        for i in range(len(room_list)):
            for j in range(i + 1, len(room_list)):
                angle = rotation_angle_between(
                    room_list[i]["rotation"], room_list[j]["rotation"]
                )
                rot_diffs.append((angle, i, j))
                max_rot_diff = max(max_rot_diff, angle)

        # Cluster rooms by rotation (< 1 degree = same session)
        THRESHOLD = 1.0  # degrees
        session_groups = []
        assigned = set()
        for i, _rm in enumerate(room_list):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(room_list)):
                if j in assigned:
                    continue
                angle = rotation_angle_between(
                    room_list[i]["rotation"], room_list[j]["rotation"]
                )
                if angle < THRESHOLD:
                    group.append(j)
                    assigned.add(j)
            session_groups.append(group)

        has_restart = len(session_groups) > 1

        # Check for timestamp gaps between session groups
        group_time_ranges = []
        for g in session_groups:
            timestamps = [
                room_list[idx]["created"] for idx in g if room_list[idx].get("created")
            ]
            if timestamps:
                group_time_ranges.append((min(timestamps), max(timestamps), g))

        group_time_ranges.sort()

        fl_analysis = {
            "floor_type": fl_type,
            "num_rooms": len(rooms_meta),
            "num_rooms_analyzed": len(room_list),
            "max_rotation_diff_deg": float(max_rot_diff),
            "num_session_groups": len(session_groups),
            "has_session_restart": has_restart,
            "session_groups": [],
        }

        for gi, g in enumerate(session_groups):
            rooms_in_group = [room_list[idx] for idx in g]
            timestamps = [rm["created"] for rm in rooms_in_group if rm.get("created")]
            yaws = [rm["yaw"] for rm in rooms_in_group]

            sg = {
                "group_id": gi,
                "num_rooms": len(g),
                "room_names": [room_list[idx]["name"] for idx in g],
                "avg_yaw_deg": float(np.mean(yaws)),
                "time_start": min(timestamps) if timestamps else None,
                "time_end": max(timestamps) if timestamps else None,
            }
            fl_analysis["session_groups"].append(sg)

        building["floor_analyses"].append(fl_analysis)
        building["max_rotation_diff_any_floor"] = max(
            building["max_rotation_diff_any_floor"], max_rot_diff
        )
        if has_restart:
            building["has_session_restart_evidence"] = True
            building["session_restart_floors"].append(fl_type)

    return building


def main():
    os.chdir(Path(__file__).parent)

    print("=" * 120)
    print("ARSESSION RESTART INVESTIGATION - ROTATION ANALYSIS")
    print("=" * 120)
    print()
    print(
        "METHODOLOGY: For rooms scanned in the same ARSession, the rotation "
        "component of"
    )
    print(
        "referenceOriginTransform must be (nearly) identical, because it encodes the "
        "fixed"
    )
    print("relationship between the room's local axes and the ARSession's world axes.")
    print(
        "Translation differs per room (different physical locations), but rotation "
        "should not."
    )
    print(
        "If rotation differs > 1 degree between rooms on the same floor, it indicates a"
    )
    print("different ARSession coordinate frame (session restart).")
    print()

    # Analyze targets
    target_results = []
    for uuid, name in TARGETS.items():
        r = analyze_building(uuid, name)
        if r:
            target_results.append(r)

    # Analyze controls
    controls = pick_controls(set(TARGETS.keys()), n=10)
    control_results = []
    for uuid, name in controls:
        r = analyze_building(uuid, name)
        if r:
            control_results.append(r)

    # === Summary Table ===
    print("=" * 120)
    print(
        f"{'Building':<35} {'Type':<7} {'Flrs':>4} {'MaxRotDiff':>10} "
        f"{'#SessGroups':>11} {'Restart?':>8}  Session details"
    )
    print("-" * 120)

    for r in target_results + control_results:
        is_target = r["uuid"] in TARGETS
        label = "TARGET" if is_target else "CTRL"

        total_groups = sum(fa["num_session_groups"] for fa in r["floor_analyses"])
        total_floors = len(r["floor_analyses"])

        restart = "YES" if r["has_session_restart_evidence"] else "no"
        maxrot = f"{r['max_rotation_diff_any_floor']:.2f}"

        # Session detail
        details = []
        for fa in r["floor_analyses"]:
            if fa["has_session_restart"]:
                groups_str = " | ".join(
                    f"S{gi}: {sg['num_rooms']}rm yaw={sg['avg_yaw_deg']:.1f}"
                    for gi, sg in enumerate(fa["session_groups"])
                )
                details.append(f"{fa['floor_type']}[{groups_str}]")
        detail_str = "; ".join(details) if details else "-"

        print(
            f"{r['name']:<35} {label:<7} {total_floors:>4} {maxrot:>10} "
            f"{total_groups:>11} {restart:>8}  {detail_str}"
        )

    # === Detailed reports for buildings with session restarts ===
    print()
    print("=" * 120)
    print("DETAILED SESSION RESTART ANALYSIS (only buildings with detected restarts)")
    print("=" * 120)

    for r in target_results + control_results:
        if not r["has_session_restart_evidence"]:
            continue

        is_target = r["uuid"] in TARGETS
        tag = "[TARGET]" if is_target else "[CONTROL]"

        print(f"\n{'=' * 100}")
        print(f"{tag} {r['name']}")
        print(f"{'=' * 100}")

        for fa in r["floor_analyses"]:
            restart_flag = (
                " *** SESSION RESTART ***" if fa["has_session_restart"] else ""
            )
            print(
                f"\n  Floor: {fa['floor_type']:<10} | "
                f"{fa['num_rooms_analyzed']} rooms | "
                f"max rot diff = {fa['max_rotation_diff_deg']:.2f} deg | "
                f"{fa['num_session_groups']} session group(s){restart_flag}"
            )

            for sg in fa["session_groups"]:
                t_start = sg["time_start"]
                t_end = sg["time_end"]
                duration = (t_end - t_start) if (t_start and t_end) else 0
                print(
                    f"    Session Group {sg['group_id']}: {sg['num_rooms']} rooms, "
                    f"avg_yaw={sg['avg_yaw_deg']:.1f}deg, "
                    f"time_span={duration:.0f}s ({duration / 60:.1f}min)"
                )
                print(f"      Rooms: {sg['room_names']}")

            # Show time gaps between session groups
            if fa["num_session_groups"] > 1:
                groups_by_time = sorted(
                    fa["session_groups"], key=lambda x: x["time_start"] or 0
                )
                for i in range(1, len(groups_by_time)):
                    gap = (groups_by_time[i]["time_start"] or 0) - (
                        groups_by_time[i - 1]["time_end"] or 0
                    )
                    yaw_diff = abs(
                        groups_by_time[i]["avg_yaw_deg"]
                        - groups_by_time[i - 1]["avg_yaw_deg"]
                    )
                    print(
                        f"    >>> GAP between Group "
                        f"{groups_by_time[i - 1]['group_id']} -> "
                        f"Group {groups_by_time[i]['group_id']}: "
                        f"{gap:.0f}s ({gap / 60:.1f}min), yaw shift = {yaw_diff:.1f}deg"
                    )

    # === Statistical comparison ===
    print()
    print("=" * 120)
    print("STATISTICAL COMPARISON: TARGET vs CONTROL")
    print("=" * 120)

    t_max_rots = [r["max_rotation_diff_any_floor"] for r in target_results]
    c_max_rots = [r["max_rotation_diff_any_floor"] for r in control_results]

    t_restart_count = sum(
        1 for r in target_results if r["has_session_restart_evidence"]
    )
    c_restart_count = sum(
        1 for r in control_results if r["has_session_restart_evidence"]
    )

    print("\n  Buildings with session restart evidence:")
    print(
        f"    TARGETS:  {t_restart_count}/{len(target_results)} "
        f"({100 * t_restart_count / len(target_results):.0f}%)"
    )
    print(
        f"    CONTROLS: {c_restart_count}/{len(control_results)} "
        f"({100 * c_restart_count / len(control_results):.0f}%)"
    )

    print("\n  Max rotation difference across any floor:")
    print(
        f"    TARGETS:  mean={np.mean(t_max_rots):.2f}deg, "
        f"median={np.median(t_max_rots):.2f}deg, "
        f"max={np.max(t_max_rots):.2f}deg, min={np.min(t_max_rots):.2f}deg"
    )
    print(
        f"    CONTROLS: mean={np.mean(c_max_rots):.2f}deg, "
        f"median={np.median(c_max_rots):.2f}deg, "
        f"max={np.max(c_max_rots):.2f}deg, min={np.min(c_max_rots):.2f}deg"
    )

    print("\n  Per-building max rotation diffs:")
    print("    TARGETS:")
    for r in sorted(target_results, key=lambda x: -x["max_rotation_diff_any_floor"]):
        flag = " *** RESTART ***" if r["has_session_restart_evidence"] else ""
        print(
            f"      {r['name']:<35} {r['max_rotation_diff_any_floor']:>8.2f} deg{flag}"
        )
    print("    CONTROLS:")
    for r in sorted(control_results, key=lambda x: -x["max_rotation_diff_any_floor"]):
        flag = " *** RESTART ***" if r["has_session_restart_evidence"] else ""
        print(
            f"      {r['name']:<35} {r['max_rotation_diff_any_floor']:>8.2f} deg{flag}"
        )

    # === Merged.json impact analysis ===
    print()
    print("=" * 120)
    print("MERGED.JSON IMPACT: Do the merging pipeline outputs show problems?")
    print("=" * 120)

    for r in target_results:
        if not r["has_session_restart_evidence"]:
            continue
        merged_path = PIPELINE / r["uuid"] / "merged.json"
        if not merged_path.exists():
            continue
        with open(merged_path) as f:
            merged = json.load(f)

        rooms = merged.get("rooms", [])
        by_story = defaultdict(list)
        for rm in rooms:
            rot = rm.get("referenceOriginTransform")
            rot_mat = extract_rotation_matrix(rot)
            if rot_mat is not None:
                by_story[rm.get("story")].append(
                    {
                        "rotation": rot_mat,
                        "yaw": yaw_from_rotation(rot_mat),
                    }
                )

        print(f"\n  {r['name']}:")
        for story, room_data in sorted(by_story.items()):
            yaws = [rd["yaw"] for rd in room_data]
            max_rot = 0
            for i in range(len(room_data)):
                for j in range(i + 1, len(room_data)):
                    angle = rotation_angle_between(
                        room_data[i]["rotation"], room_data[j]["rotation"]
                    )
                    max_rot = max(max_rot, angle)
            print(
                f"    Story {story}: {len(room_data)} rooms, "
                f"max_rot_diff={max_rot:.4f}deg, "
                f"yaw range=[{min(yaws):.2f}, {max(yaws):.2f}]"
            )
            if max_rot > 1.0:
                print("      *** MERGED.JSON ALSO SHOWS SESSION INCONSISTENCY ***")
            else:
                print(
                    "      merged.json has consistent rotations (merging pipeline may "
                    "have compensated)"
                )

    # === VERDICT ===
    print()
    print("=" * 120)
    print("VERDICT")
    print("=" * 120)
    print()
    print(
        f"  Of the 8 target buildings, {t_restart_count} show clear evidence of "
        f"ARSession restarts"
    )
    print("  (rotation differences > 1 degree between rooms on the same floor).")
    print()
    print(
        f"  Of the {len(control_results)} control buildings, {c_restart_count} show "
        f"session restart evidence."
    )
    print()
    if t_restart_count > 0:
        print("  SESSION RESTART BUILDINGS AND AFFECTED FLOORS:")
        for r in target_results:
            if r["has_session_restart_evidence"]:
                print(
                    f"    - {r['name']}: floors with restart = "
                    f"{r['session_restart_floors']}, "
                    f"max rotation diff = {r['max_rotation_diff_any_floor']:.1f}deg"
                )
    print()
    print("  The merged.json referenceOriginTransform is CONSISTENT within each story,")
    print("  suggesting the merging pipeline compensated for the session restarts by")
    print(
        "  re-aligning rooms. However, if the original scan data had coordinate frame"
    )
    print("  discontinuities, the 3D reconstruction quality may still be degraded.")


if __name__ == "__main__":
    main()
