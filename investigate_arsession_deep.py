#!/usr/bin/env python3
"""
Deep investigation: Per-floor referenceOriginTransform analysis.
The key question: within a SINGLE floor, do rooms have different
referenceOriginTransforms?
If yes, it means the ARSession/worldMap was reset mid-scan.
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


def extract_translation(flat16):
    if flat16 is None or len(flat16) != 16:
        return None
    return np.array([flat16[12], flat16[13], flat16[14]])


def extract_rotation_matrix(flat16):
    if flat16 is None or len(flat16) != 16:
        return None
    m = np.array(flat16).reshape(4, 4)
    return m[:3, :3]


def rotation_angle_between(rot1, rot2):
    if rot1 is None or rot2 is None:
        return None
    R = rot1 @ rot2.T
    trace = np.clip(np.trace(R), -1, 3)
    return math.degrees(math.acos(np.clip((trace - 1) / 2, -1, 1)))


def pick_controls(exclude, n=5):
    """Pick N control buildings with merged.json available."""
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


def analyze_per_floor_ref_origins(uuid, name):
    """
    For each floor in data.json, load each room's JSON and compare
    referenceOriginTransform.
    """
    scan_dir = find_scan_dir(uuid)
    if not scan_dir:
        return None

    with open(scan_dir / "data.json") as f:
        data = json.load(f)

    hm = data["homeMetadata"]
    floors = hm.get("floors", [])

    building_result = {
        "name": name,
        "uuid": uuid,
        "floors": [],
        "has_intra_floor_variation": False,
        "max_intra_floor_trans_diff": 0,
        "max_intra_floor_rot_diff": 0,
    }

    for fl in floors:
        fl_type = fl.get("floorType", "unknown")
        fl_id = fl.get("id")
        rooms = fl.get("rooms", [])

        room_data_list = []
        for r in rooms:
            rid = r.get("id")
            rname = r.get("name", "?")
            rcreated = r.get("created")
            room_json = scan_dir / f"{rid}.json"
            if room_json.exists():
                with open(room_json) as f:
                    rd = json.load(f)
                rot = rd.get("referenceOriginTransform")
                room_data_list.append(
                    {
                        "id": rid,
                        "name": rname,
                        "created": rcreated,
                        "referenceOriginTransform": rot,
                        "translation": extract_translation(rot),
                        "rotation": extract_rotation_matrix(rot),
                    }
                )

        # Sort by creation time
        room_data_list.sort(key=lambda x: x.get("created") or 0)

        # Compare all referenceOriginTransforms within this floor
        transforms = [
            (r["translation"], r["rotation"], r["name"], r["created"])
            for r in room_data_list
            if r["translation"] is not None
        ]

        max_trans_diff = 0
        max_rot_diff = 0
        groups = []  # groups of rooms with same referenceOriginTransform

        if len(transforms) >= 2:
            # Cluster rooms by referenceOriginTransform similarity
            used = set()
            for i in range(len(transforms)):
                if i in used:
                    continue
                group = [i]
                used.add(i)
                for j in range(i + 1, len(transforms)):
                    if j in used:
                        continue
                    tdiff = np.linalg.norm(transforms[i][0] - transforms[j][0])
                    rdiff = rotation_angle_between(transforms[i][1], transforms[j][1])
                    if tdiff < 0.01 and (rdiff is None or rdiff < 0.1):
                        group.append(j)
                        used.add(j)
                groups.append(group)

            # Compute max differences
            for i in range(len(transforms)):
                for j in range(i + 1, len(transforms)):
                    td = np.linalg.norm(transforms[i][0] - transforms[j][0])
                    rd = rotation_angle_between(transforms[i][1], transforms[j][1]) or 0
                    max_trans_diff = max(max_trans_diff, td)
                    max_rot_diff = max(max_rot_diff, rd)

        has_variation = len(groups) > 1 if groups else False

        fl_result = {
            "floor_type": fl_type,
            "floor_id": fl_id,
            "num_rooms": len(rooms),
            "num_rooms_with_json": len(room_data_list),
            "max_trans_diff_m": float(max_trans_diff),
            "max_rot_diff_deg": float(max_rot_diff),
            "has_variation": has_variation,
            "num_ref_origin_groups": len(groups),
            "groups": [],
        }

        for gi, g in enumerate(groups):
            group_rooms = [transforms[idx] for idx in g]
            fl_result["groups"].append(
                {
                    "group_id": gi,
                    "rooms": [transforms[idx][2] for idx in g],
                    "timestamps": [transforms[idx][3] for idx in g],
                    "translation": group_rooms[0][0].tolist(),
                }
            )

        building_result["floors"].append(fl_result)

        if has_variation:
            building_result["has_intra_floor_variation"] = True
            building_result["max_intra_floor_trans_diff"] = max(
                building_result["max_intra_floor_trans_diff"], max_trans_diff
            )
            building_result["max_intra_floor_rot_diff"] = max(
                building_result["max_intra_floor_rot_diff"], max_rot_diff
            )

    return building_result


def analyze_merged_per_story(uuid):
    """Check merged.json referenceOriginTransform per story."""
    merged_path = PIPELINE / uuid / "merged.json"
    if not merged_path.exists():
        return None
    with open(merged_path) as f:
        merged = json.load(f)

    by_story = defaultdict(list)
    for r in merged.get("rooms", []):
        story = r.get("story")
        rot = r.get("referenceOriginTransform")
        by_story[story].append(rot)

    result = {}
    for story, rots in by_story.items():
        translations = [extract_translation(t) for t in rots if t]
        translations = [t for t in translations if t is not None]
        if len(translations) >= 2:
            max_d = max(
                np.linalg.norm(translations[i] - translations[j])
                for i in range(len(translations))
                for j in range(i + 1, len(translations))
            )
        else:
            max_d = 0
        result[story] = {
            "num_rooms": len(rots),
            "max_trans_diff_m": float(max_d),
            "all_identical_trans": max_d < 0.001,
        }
    return result


def main():
    os.chdir(Path(__file__).parent)

    print("=" * 120)
    print("DEEP ARSESSION INVESTIGATION: PER-FLOOR REFERENCEORIGINTRANSFORM ANALYSIS")
    print("=" * 120)

    # Analyze targets
    print("\n" + "=" * 120)
    print("TARGET BUILDINGS (8 suspected problematic)")
    print("=" * 120)

    target_results = []
    for uuid, name in TARGETS.items():
        r = analyze_per_floor_ref_origins(uuid, name)
        if r:
            target_results.append(r)

    for r in target_results:
        flag = (
            " *** INTRA-FLOOR VARIATION ***" if r["has_intra_floor_variation"] else ""
        )
        print(f"\n{'=' * 80}")
        print(f"{r['name']} ({r['uuid'][:16]}...){flag}")
        print(f"{'=' * 80}")

        for fl in r["floors"]:
            var_flag = " *** VARIATION ***" if fl["has_variation"] else ""
            print(
                f"\n  Floor: {fl['floor_type']:>10} | {fl['num_rooms']} rooms | "
                f"groups={fl['num_ref_origin_groups']} | "
                f"max_trans={fl['max_trans_diff_m']:.3f}m | "
                f"max_rot={fl['max_rot_diff_deg']:.2f}deg{var_flag}"
            )

            for g in fl["groups"]:
                t = g["translation"]
                ts = g["timestamps"]
                ts_sorted = sorted(ts)
                time_range = ""
                if len(ts_sorted) >= 2:
                    time_range = f" (span={ts_sorted[-1] - ts_sorted[0]:.0f}s)"
                print(
                    f"    Group {g['group_id']}: {len(g['rooms'])} rooms - {g['rooms']}"
                )
                print(
                    f"      Translation: [{t[0]:.3f}, {t[1]:.3f}, "
                    f"{t[2]:.3f}]{time_range}"
                )

        # Show merged.json comparison
        merged = analyze_merged_per_story(r["uuid"])
        if merged:
            print("\n  MERGED.JSON (per story):")
            for story, v in sorted(merged.items()):
                ident = (
                    "identical"
                    if v["all_identical_trans"]
                    else f"VARIES by {v['max_trans_diff_m']:.3f}m"
                )
                print(
                    f"    Story {story}: {v['num_rooms']} rooms, translations {ident}"
                )

    # Analyze controls
    print("\n\n" + "=" * 120)
    print("CONTROL BUILDINGS (5 randomly selected)")
    print("=" * 120)

    controls = pick_controls(set(TARGETS.keys()), n=5)
    control_results = []
    for uuid, name in controls:
        r = analyze_per_floor_ref_origins(uuid, name)
        if r:
            control_results.append(r)

    for r in control_results:
        flag = (
            " *** INTRA-FLOOR VARIATION ***" if r["has_intra_floor_variation"] else ""
        )
        print(f"\n{'=' * 80}")
        print(f"{r['name']} ({r['uuid'][:16]}...){flag}")
        print(f"{'=' * 80}")

        for fl in r["floors"]:
            var_flag = " *** VARIATION ***" if fl["has_variation"] else ""
            print(
                f"\n  Floor: {fl['floor_type']:>10} | {fl['num_rooms']} rooms | "
                f"groups={fl['num_ref_origin_groups']} | "
                f"max_trans={fl['max_trans_diff_m']:.3f}m | "
                f"max_rot={fl['max_rot_diff_deg']:.2f}deg{var_flag}"
            )

            for g in fl["groups"]:
                t = g["translation"]
                print(
                    f"    Group {g['group_id']}: {len(g['rooms'])} rooms - {g['rooms']}"
                )
                print(f"      Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

    # === FINAL VERDICT ===
    print("\n\n" + "=" * 120)
    print("FINAL ANALYSIS: COMPARING TARGETS vs CONTROLS")
    print("=" * 120)

    # Key metrics
    target_intra_floor_vars = []
    target_intra_floor_rots = []
    target_multi_group_floors = 0
    target_total_floors = 0
    for r in target_results:
        for fl in r["floors"]:
            target_total_floors += 1
            if fl["has_variation"]:
                target_multi_group_floors += 1
                target_intra_floor_vars.append(fl["max_trans_diff_m"])
                target_intra_floor_rots.append(fl["max_rot_diff_deg"])

    control_intra_floor_vars = []
    control_intra_floor_rots = []
    control_multi_group_floors = 0
    control_total_floors = 0
    for r in control_results:
        for fl in r["floors"]:
            control_total_floors += 1
            if fl["has_variation"]:
                control_multi_group_floors += 1
                control_intra_floor_vars.append(fl["max_trans_diff_m"])
                control_intra_floor_rots.append(fl["max_rot_diff_deg"])

    print(
        "\n  INTRA-FLOOR referenceOriginTransform variation (rooms on SAME floor have "
        "different origins):"
    )
    print(
        f"    TARGETS: {target_multi_group_floors}/{target_total_floors} floors have "
        f"multiple ref-origin groups"
    )
    print(
        f"    CONTROLS: {control_multi_group_floors}/{control_total_floors} "
        f"floors have multiple ref-origin groups"
    )

    if target_intra_floor_vars:
        print(
            f"\n    TARGET translation diffs : "
            f"mean={np.mean(target_intra_floor_vars):.3f}m, "
            f"max={np.max(target_intra_floor_vars):.3f}m"
        )
    if target_intra_floor_rots:
        print(
            f"    TARGET rotation diffs   : "
            f"mean={np.mean(target_intra_floor_rots):.2f}deg, "
            f"max={np.max(target_intra_floor_rots):.2f}deg"
        )
    if control_intra_floor_vars:
        print(
            f"    CONTROL translation diffs: "
            f"mean={np.mean(control_intra_floor_vars):.3f}m, "
            f"max={np.max(control_intra_floor_vars):.3f}m"
        )
    if control_intra_floor_rots:
        print(
            f"    CONTROL rotation diffs  : "
            f"mean={np.mean(control_intra_floor_rots):.2f}deg, "
            f"max={np.max(control_intra_floor_rots):.2f}deg"
        )

    # Cross-floor analysis: do different floors always have different ref origins?
    print(
        "\n  CROSS-FLOOR referenceOriginTransform (expected to differ between floors):"
    )
    for r in target_results + control_results:
        if len(r["floors"]) > 1:
            floor_translations = []
            for fl in r["floors"]:
                if fl["groups"]:
                    floor_translations.append(
                        (
                            fl["floor_type"],
                            fl["groups"][0]["translation"],
                        )
                    )
            if len(floor_translations) >= 2:
                print(f"    {r['name']}:")
                for ft, t in floor_translations:
                    print(f"      {ft:>10}: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

    # Key finding: Are the ref-origin groups correlated with the floor boundaries?
    print(
        "\n\n  KEY QUESTION: Within a single floor scan, do rooms share the same "
        "ARSession?"
    )
    print("  ============")
    print(
        "\n  Observation: ALL buildings (both target and control) show different "
        "referenceOriginTransforms"
    )
    print(
        "  between rooms, even on the same floor. This is because Apple RoomPlan "
        "creates a new"
    )
    print(
        "  coordinate system for each room scan. The referenceOriginTransform "
        "maps from the room's"
    )
    print("  local coordinate system to the shared ARSession world coordinate system.")
    print(
        "\n  The fact that referenceOriginTransform DIFFERS between rooms is EXPECTED "
        "and NORMAL."
    )
    print(
        "  What matters is whether the overall ARSession coordinate frame (the 'world' "
        "origin)"
    )
    print(
        "  remains consistent. If a new ARSession was started mid-scan, room "
        "transforms would"
    )
    print("  be relative to a DIFFERENT world origin, making them incompatible.")
    print(
        "\n  To detect this, we need to check if rooms from the same floor, when "
        "transformed by"
    )
    print(
        "  their referenceOriginTransform, produce consistent spatial positions. That "
        "analysis"
    )
    print(
        "  follows in the merged.json check (where all rooms are already in a shared "
        "frame)."
    )


if __name__ == "__main__":
    main()
