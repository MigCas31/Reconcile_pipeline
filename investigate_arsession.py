#!/usr/bin/env python3
"""
Investigate ARKit session failures/restarts in 8 problematic buildings.
Checks for:
1. referenceOriginTransform variation across rooms (session restart indicator)
2. Timestamp gaps between rooms/floors (app close/reopen)
3. capturedAzimuths transform discontinuities (coordinate frame jumps)
4. Multiple floors with timing anomalies
5. Room-level JSON coordinate frame consistency
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(".scan-cache")
PIPELINE = Path("pipeline-outputs")

# 8 target buildings
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
    """Find the scan-cache directory for a given UUID."""
    for d in CACHE.iterdir():
        if uuid in d.name:
            return d
    return None


def parse_4x4(flat16):
    """Parse flat list of 16 floats into 4x4 matrix (column-major as stored)."""
    if flat16 is None or len(flat16) != 16:
        return None
    return np.array(flat16, dtype=float).reshape(4, 4)  # column-major storage


def extract_translation(flat16):
    """Extract translation (x,y,z) from flat 4x4 column-major matrix."""
    if flat16 is None or len(flat16) != 16:
        return None
    # In column-major 4x4: indices 12,13,14 are tx,ty,tz
    return np.array([flat16[12], flat16[13], flat16[14]])


def extract_rotation_matrix(flat16):
    """Extract 3x3 rotation from flat 4x4 column-major matrix."""
    if flat16 is None or len(flat16) != 16:
        return None
    m = np.array(flat16).reshape(4, 4)
    return m[:3, :3]


def rotation_angle_between(rot1, rot2):
    """Compute rotation angle in degrees between two 3x3 rotation matrices."""
    if rot1 is None or rot2 is None:
        return None
    R = rot1 @ rot2.T
    trace = np.clip(np.trace(R), -1, 3)
    angle = math.degrees(math.acos(np.clip((trace - 1) / 2, -1, 1)))
    return angle


def analyze_building(uuid, name, is_control=False):
    """Analyze a single building for session restart indicators."""
    result = {"name": name, "uuid": uuid, "is_control": is_control}

    scan_dir = find_scan_dir(uuid)
    if not scan_dir:
        result["error"] = "scan-cache dir not found"
        return result

    # --- 1. Load data.json ---
    data_path = scan_dir / "data.json"
    if not data_path.exists():
        result["error"] = "data.json not found"
        return result

    with open(data_path) as f:
        data = json.load(f)

    hm = data.get("homeMetadata", {})

    # --- 2. Analyze floors and rooms from data.json ---
    floors = hm.get("floors", [])
    result["num_floors"] = len(floors)

    floor_info = []
    all_room_timestamps = []
    for fl in floors:
        fl_created = fl.get("created")
        fl_type = fl.get("floorType", "unknown")
        fl_id = fl.get("id")
        rooms = fl.get("rooms", [])
        room_timestamps = []
        for r in rooms:
            r_created = r.get("created")
            if r_created:
                room_timestamps.append(r_created)
                all_room_timestamps.append(
                    (
                        r_created,
                        r.get("name", "?"),
                        fl_type,
                        fl_id,
                        r.get("id"),
                    )
                )

        room_timestamps.sort()
        fl_info = {
            "type": fl_type,
            "id": fl_id,
            "created": fl_created,
            "num_rooms": len(rooms),
            "room_timestamps": room_timestamps,
        }
        if len(room_timestamps) >= 2:
            gaps = []
            for i in range(1, len(room_timestamps)):
                gap = room_timestamps[i] - room_timestamps[i - 1]
                gaps.append(gap)
            fl_info["max_intra_floor_gap_sec"] = max(gaps)
            fl_info["min_intra_floor_gap_sec"] = min(gaps)
        floor_info.append(fl_info)

    result["floors"] = floor_info

    # Sort all room timestamps globally
    all_room_timestamps.sort()
    result["all_room_timestamps"] = all_room_timestamps

    # Find inter-floor gaps
    if len(floors) >= 2:
        floor_created = [
            (fl.get("created"), fl.get("floorType"))
            for fl in floors
            if fl.get("created")
        ]
        floor_created.sort()
        inter_gaps = []
        for i in range(1, len(floor_created)):
            gap = floor_created[i][0] - floor_created[i - 1][0]
            inter_gaps.append((gap, floor_created[i - 1][1], floor_created[i][1]))
        result["inter_floor_gaps"] = inter_gaps

    # --- 3. Check timestamp gaps (> 5 min suggests app restart) ---
    big_gaps = []
    for i in range(1, len(all_room_timestamps)):
        gap = all_room_timestamps[i][0] - all_room_timestamps[i - 1][0]
        if gap > 300:  # > 5 minutes
            big_gaps.append(
                {
                    "gap_seconds": gap,
                    "gap_minutes": gap / 60,
                    "before": all_room_timestamps[i - 1],
                    "after": all_room_timestamps[i],
                }
            )
    result["timestamp_gaps_over_5min"] = big_gaps

    # --- 4. Analyze capturedAzimuths ---
    azimuths = hm.get("capturedAzimuths", [])
    result["num_azimuths"] = len(azimuths)

    if azimuths:
        # Group by storyId
        story_azimuths = defaultdict(list)
        for az in azimuths:
            sid = az.get("storyId", "unknown")
            story_azimuths[sid].append(az)

        result["azimuth_stories"] = list(story_azimuths.keys())

        # Check for transform jumps within each story's azimuths
        azimuth_jumps = []
        for sid, azs in story_azimuths.items():
            # Sort by azimuth value for consecutive analysis
            azs_sorted = sorted(azs, key=lambda x: x.get("azimuth", 0))
            for i in range(1, len(azs_sorted)):
                transforms_prev = azs_sorted[i - 1].get("transforms", [])
                transforms_curr = azs_sorted[i].get("transforms", [])

                if transforms_prev and transforms_curr:
                    # Compare first transform of each
                    t_prev = extract_translation(transforms_prev[0])
                    t_curr = extract_translation(transforms_curr[0])
                    if t_prev is not None and t_curr is not None:
                        dist = np.linalg.norm(t_curr - t_prev)
                        if dist > 5.0:  # > 5m jump suggests coordinate frame change
                            azimuth_jumps.append(
                                {
                                    "story": sid,
                                    "az_prev": azs_sorted[i - 1].get("azimuth"),
                                    "az_curr": azs_sorted[i].get("azimuth"),
                                    "translation_jump_m": float(dist),
                                }
                            )
        result["azimuth_transform_jumps_over_5m"] = azimuth_jumps

    # --- 5. Analyze arworldmap.json ---
    awm_path = scan_dir / "arworldmap.json"
    if awm_path.exists():
        with open(awm_path) as f:
            awm = json.load(f)
        result["arworldmap"] = awm

    # --- 6. Analyze room-level JSONs for referenceOriginTransform ---
    room_ref_origins = {}
    room_ref_origins_by_story = defaultdict(list)

    for fl in floors:
        fl_type = fl.get("floorType", "unknown")
        fl_id = fl.get("id")
        for r in fl.get("rooms", []):
            rid = r.get("id")
            rname = r.get("name", "?")
            room_json_path = scan_dir / f"{rid}.json"
            if room_json_path.exists():
                with open(room_json_path) as f:
                    room_data = json.load(f)
                rot = room_data.get("referenceOriginTransform")
                story = room_data.get("story")
                room_ref_origins[rid] = {
                    "name": rname,
                    "floor_type": fl_type,
                    "floor_id": fl_id,
                    "story": story,
                    "referenceOriginTransform": rot,
                }
                room_ref_origins_by_story[story].append(
                    {
                        "room_id": rid,
                        "room_name": rname,
                        "floor_type": fl_type,
                        "referenceOriginTransform": rot,
                    }
                )

    result["room_ref_origins"] = room_ref_origins

    # Check if referenceOriginTransform varies within same story
    ref_origin_variations = {}
    for story, rooms in room_ref_origins_by_story.items():
        if len(rooms) < 2:
            continue

        # Compare all pairs
        transforms = [
            r["referenceOriginTransform"]
            for r in rooms
            if r["referenceOriginTransform"]
        ]
        if len(transforms) < 2:
            continue

        # Check translation variance
        translations = [extract_translation(t) for t in transforms]
        translations = [t for t in translations if t is not None]

        if len(translations) >= 2:
            max_dist = 0
            for i in range(len(translations)):
                for j in range(i + 1, len(translations)):
                    d = np.linalg.norm(translations[i] - translations[j])
                    max_dist = max(max_dist, d)

            # Check rotation variance
            rotations = [extract_rotation_matrix(t) for t in transforms]
            rotations = [r for r in rotations if r is not None]
            max_rot_angle = 0
            for i in range(len(rotations)):
                for j in range(i + 1, len(rotations)):
                    angle = rotation_angle_between(rotations[i], rotations[j])
                    if angle is not None:
                        max_rot_angle = max(max_rot_angle, angle)

            ref_origin_variations[story] = {
                "num_rooms": len(rooms),
                "max_translation_diff_m": float(max_dist),
                "max_rotation_diff_deg": float(max_rot_angle),
                "all_identical": max_dist < 0.001 and max_rot_angle < 0.01,
                "room_names": [r["room_name"] for r in rooms],
            }

    result["ref_origin_variation_by_story"] = ref_origin_variations

    # --- 7. Analyze merged.json referenceOriginTransform ---
    merged_path = PIPELINE / uuid / "merged.json"
    if merged_path.exists():
        with open(merged_path) as f:
            merged = json.load(f)

        merged_rooms = merged.get("rooms", [])
        merged_ref_by_story = defaultdict(list)
        for r in merged_rooms:
            story = r.get("story")
            rot = r.get("referenceOriginTransform")
            merged_ref_by_story[story].append(rot)

        merged_variations = {}
        for story, rots in merged_ref_by_story.items():
            translations = [extract_translation(t) for t in rots if t]
            translations = [t for t in translations if t is not None]

            rotations = [extract_rotation_matrix(t) for t in rots if t]
            rotations = [r for r in rotations if r is not None]

            if len(translations) >= 2:
                max_dist = 0
                for i in range(len(translations)):
                    for j in range(i + 1, len(translations)):
                        d = np.linalg.norm(translations[i] - translations[j])
                        max_dist = max(max_dist, d)

                max_rot = 0
                for i in range(len(rotations)):
                    for j in range(i + 1, len(rotations)):
                        angle = rotation_angle_between(rotations[i], rotations[j])
                        if angle is not None:
                            max_rot = max(max_rot, angle)

                merged_variations[story] = {
                    "num_rooms": len(rots),
                    "max_translation_diff_m": float(max_dist),
                    "max_rotation_diff_deg": float(max_rot),
                    "all_identical": max_dist < 0.001 and max_rot < 0.01,
                }

        result["merged_ref_origin_variation_by_story"] = merged_variations

    return result


def pick_control_buildings(exclude_uuids, n=3):
    """Pick N buildings not in exclude list as controls."""
    controls = []
    for d in sorted(CACHE.iterdir()):
        if not d.is_dir():
            continue
        # Extract UUID from dir name
        parts = d.name.split("_")
        # UUID is typically the second-to-last segment before the timestamp
        uuid = None
        for part in parts:
            if len(part) == 36 and part.count("-") == 4:
                uuid = part
                break
        if uuid and uuid not in exclude_uuids:
            # Check it has data.json and merged.json
            if (d / "data.json").exists() and (
                PIPELINE / uuid / "merged.json"
            ).exists():
                # Extract address from dir name
                name = d.name.split(uuid)[0].rsplit("_", 1)[0]
                # Cleanup: take everything after userID
                name_parts = name.split("_")[1:]  # skip 'scans'
                name_parts = name_parts[1:]  # skip userID
                name = " ".join(name_parts).replace("__", ", ").replace("_", " ")
                controls.append((uuid, name))
                if len(controls) >= n:
                    break
    return controls


def print_report(results):
    """Print a comprehensive comparison report."""
    print("=" * 100)
    print("ARSESSION RESTART INVESTIGATION REPORT")
    print("=" * 100)

    # Separate targets and controls
    targets = [r for r in results if not r.get("is_control")]
    controls = [r for r in results if r.get("is_control")]

    # --- Summary Table ---
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (
        f"{'Building':<30} {'Type':<8} {'Floors':>6} {'Rooms':>6} {'Gaps>5m':>8} "
        f"{'RefOrigin Var':>14} {'AzJumps':>8} {'WorldMap':>20}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERR':<8}")
            continue

        label = "CTRL" if r.get("is_control") else "TARGET"
        num_floors = r.get("num_floors", 0)

        total_rooms = sum(fl.get("num_rooms", 0) for fl in r.get("floors", []))

        gaps = len(r.get("timestamp_gaps_over_5min", []))

        # Check ref origin variation
        ref_var = r.get("ref_origin_variation_by_story", {})
        has_variation = any(not v.get("all_identical") for v in ref_var.values())
        max_trans = max(
            (v.get("max_translation_diff_m", 0) for v in ref_var.values()), default=0
        )
        ref_str = f"{'YES' if has_variation else 'no':>4} ({max_trans:.3f}m)"

        az_jumps = len(r.get("azimuth_transform_jumps_over_5m", []))

        wm = r.get("arworldmap", {})
        wm_str = str(wm) if wm else "none"
        if len(wm_str) > 20:
            wm_str = wm_str[:17] + "..."

        print(
            f"{r['name']:<30} {label:<8} {num_floors:>6} {total_rooms:>6} {gaps:>8} "
            f"{ref_str:>14} {az_jumps:>8} {wm_str:>20}"
        )

    # --- Detailed per-building reports ---
    for r in results:
        if "error" in r:
            continue

        tag = "[CONTROL]" if r.get("is_control") else "[TARGET]"
        print(f"\n{'=' * 100}")
        print(f"{tag} {r['name']} ({r['uuid']})")
        print(f"{'=' * 100}")

        # Floors
        print(f"\n  FLOORS ({r.get('num_floors', 0)}):")
        for fl in r.get("floors", []):
            print(
                f"    {fl['type']:>10} (id={fl['id'][:20]}...): {fl['num_rooms']} "
                f"rooms, created={fl.get('created')}"
            )
            if "max_intra_floor_gap_sec" in fl:
                print(
                    f"      Intra-floor room gaps: "
                    f"min={fl['min_intra_floor_gap_sec']:.0f}s, "
                    f"max={fl['max_intra_floor_gap_sec']:.0f}s "
                    f"({fl['max_intra_floor_gap_sec'] / 60:.1f}min)"
                )

        # Inter-floor gaps
        if "inter_floor_gaps" in r:
            print("\n  INTER-FLOOR GAPS:")
            for gap, ft1, ft2 in r["inter_floor_gaps"]:
                flag = " *** LARGE GAP ***" if gap > 3600 else ""
                print(f"    {ft1} -> {ft2}: {gap:.0f}s ({gap / 60:.1f}min){flag}")

        # Timestamp gaps > 5 min
        gaps = r.get("timestamp_gaps_over_5min", [])
        if gaps:
            print(f"\n  TIMESTAMP GAPS > 5 MIN ({len(gaps)}):")
            for g in gaps:
                print(f"    {g['gap_minutes']:.1f} min gap")
                print(
                    f"      Before: room '{g['before'][1]}' on {g['before'][2]} floor "
                    f"(t={g['before'][0]})"
                )
                print(
                    f"      After:  room '{g['after'][1]}' on {g['after'][2]} floor "
                    f"(t={g['after'][0]})"
                )
        else:
            print("\n  TIMESTAMP GAPS > 5 MIN: None")

        # Reference origin variations (from room JSONs)
        ref_var = r.get("ref_origin_variation_by_story", {})
        print("\n  REFERENCE ORIGIN TRANSFORM VARIATION (room JSONs, per story):")
        if not ref_var:
            print("    No multi-room stories found")
        for story, v in sorted(ref_var.items()):
            flag = " *** DIFFERENT REF ORIGINS ***" if not v["all_identical"] else ""
            print(
                f"    Story {story}: {v['num_rooms']} rooms, "
                f"max_trans_diff={v['max_translation_diff_m']:.6f}m, "
                f"max_rot_diff={v['max_rotation_diff_deg']:.4f}deg{flag}"
            )
            if not v["all_identical"]:
                print(f"      Rooms: {v['room_names']}")

        # Merged.json ref origin variation
        merged_var = r.get("merged_ref_origin_variation_by_story", {})
        if merged_var:
            print("\n  MERGED.JSON REFERENCE ORIGIN VARIATION (per story):")
            for story, v in sorted(merged_var.items()):
                flag = " *** DIFFERENT ***" if not v["all_identical"] else ""
                print(
                    f"    Story {story}: {v['num_rooms']} rooms, "
                    f"max_trans_diff={v['max_translation_diff_m']:.6f}m, "
                    f"max_rot_diff={v['max_rotation_diff_deg']:.4f}deg{flag}"
                )

        # Azimuth jumps
        az_jumps = r.get("azimuth_transform_jumps_over_5m", [])
        if az_jumps:
            print(f"\n  AZIMUTH TRANSFORM JUMPS > 5m ({len(az_jumps)}):")
            for j in az_jumps:
                print(
                    f"    Story {j['story']}: azimuth {j['az_prev']} -> "
                    f"{j['az_curr']}: {j['translation_jump_m']:.2f}m jump"
                )
        else:
            print("\n  AZIMUTH TRANSFORM JUMPS > 5m: None")

        # ARWorldMap
        wm = r.get("arworldmap", {})
        print(f"\n  ARWORLDMAP: {wm if wm else 'not present'}")

    # --- Cross-building comparison ---
    print(f"\n{'=' * 100}")
    print("CROSS-BUILDING COMPARISON: TARGET vs CONTROL")
    print(f"{'=' * 100}")

    def stats_for_group(group):
        ref_vars = []
        gap_counts = []
        az_jump_counts = []
        for r in group:
            if "error" in r:
                continue
            ref_var = r.get("ref_origin_variation_by_story", {})
            max_trans = max(
                (v.get("max_translation_diff_m", 0) for v in ref_var.values()),
                default=0,
            )
            ref_vars.append(max_trans)
            gap_counts.append(len(r.get("timestamp_gaps_over_5min", [])))
            az_jump_counts.append(len(r.get("azimuth_transform_jumps_over_5m", [])))
        return ref_vars, gap_counts, az_jump_counts

    t_rv, t_gc, t_az = stats_for_group(targets)
    c_rv, c_gc, c_az = stats_for_group(controls)

    print(
        f"\n  Metric                    | Targets (n={len(targets)})        | Controls "
        f"(n={len(controls)})"
    )
    print(f"  {'':->28}|{'':->30}|{'':->30}")
    if t_rv:
        print(
            f"  Max refOrigin trans diff  | mean={np.mean(t_rv):.6f}m "
            f"max={np.max(t_rv):.6f}m | mean={np.mean(c_rv):.6f}m "
            f"max={np.max(c_rv):.6f}m"
        )
    if t_gc:
        print(
            f"  Timestamp gaps > 5min     | mean={np.mean(t_gc):.1f}  "
            f"max={np.max(t_gc)}              | mean={np.mean(c_gc):.1f}  "
            f"max={np.max(c_gc)}"
        )
    if t_az:
        print(
            f"  Azimuth transform jumps   | mean={np.mean(t_az):.1f}  "
            f"max={np.max(t_az)}              | mean={np.mean(c_az):.1f}  "
            f"max={np.max(c_az)}"
        )


def main():
    os.chdir(Path(__file__).parent)

    results = []

    # Analyze target buildings
    print("Analyzing 8 target buildings...")
    for uuid, name in TARGETS.items():
        print(f"  Processing {name}...")
        r = analyze_building(uuid, name, is_control=False)
        results.append(r)

    # Pick and analyze 3 control buildings
    print("\nPicking 3 control buildings...")
    controls = pick_control_buildings(set(TARGETS.keys()), n=3)
    for uuid, name in controls:
        print(f"  Processing {name} (control)...")
        r = analyze_building(uuid, name, is_control=True)
        results.append(r)

    print_report(results)

    # Also dump raw data for further analysis
    with open("arsession_investigation_raw.json", "w") as f:
        # Convert numpy arrays etc for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return str(obj)

        json.dump(results, f, indent=2, default=default_serializer)
    print("\nRaw data saved to arsession_investigation_raw.json")


if __name__ == "__main__":
    main()
