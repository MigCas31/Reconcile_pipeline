#!/usr/bin/env python3
"""
Investigate the 5 target buildings that did NOT show rotation-based session restarts.
Look for other anomalies:
1. Cross-floor coordinate frame consistency (Y offset between floors)
2. Unusual timestamp patterns
3. Room-to-room transform continuity (wall endpoints matching)
4. The arworldmap.json content
5. capturedAzimuths transform anomalies (more detailed)
6. Whether merged.json shows the rotation of the "wrong" session group
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(".scan-cache")
PIPELINE = Path("pipeline-outputs")

# These 5 did NOT show rotation-based restart
REMAINING = {
    "1d26eda3-c927-4cdf-a3b0-218036828a55": "Gerskov Bygade 24",
    "7dbc53a6-17e8-4806-83de-42286b95726c": "Gultvedgyden 6",
    "e661e7b6-303d-415c-b378-2d9dd2fbfd6f": "Hesselvænget 2",
    "cb711a0b-6e8d-4ae6-b008-af3297446dcc": "Kastanievej 6",
    "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5": "Pilevej 30",
}

# 3 that DID show restart (for comparison)
RESTART_BUILDINGS = {
    "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3": "Bredballe Byvej 63",
    "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b": "Lucernevej 23",
    "feccbd0c-0420-4775-b5b7-49b99559947e": "Morelvej 68",
}

ALL_TARGETS = {**REMAINING, **RESTART_BUILDINGS}


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
    return np.array(flat16, dtype=float).reshape(4, 4)[:3, :3]


def yaw_from_rotation(rot):
    return math.degrees(math.atan2(rot[0, 2], rot[0, 0]))


def analyze_cross_floor_consistency(uuid, name):
    """Check if floors have consistent Y-axis offsets (vertical alignment)."""
    scan_dir = find_scan_dir(uuid)
    if not scan_dir:
        return None

    with open(scan_dir / "data.json") as f:
        data = json.load(f)

    hm = data["homeMetadata"]
    floors = hm.get("floors", [])

    result = {"name": name, "uuid": uuid, "floors": []}

    for fl in floors:
        fl_type = fl.get("floorType", "unknown")
        fl_id = fl.get("id")
        rooms_meta = fl.get("rooms", [])

        room_y_values = []
        room_yaws = []
        for r in rooms_meta:
            rid = r.get("id")
            r.get("name", "?")
            room_json = scan_dir / f"{rid}.json"
            if room_json.exists():
                with open(room_json) as f:
                    rd = json.load(f)
                rot_flat = rd.get("referenceOriginTransform")
                trans = extract_translation(rot_flat)
                rot = extract_rotation_matrix(rot_flat)
                if trans is not None:
                    room_y_values.append(trans[1])  # Y = vertical in ARKit
                if rot is not None:
                    room_yaws.append(yaw_from_rotation(rot))

        fl_info = {
            "type": fl_type,
            "id": fl_id,
            "created": fl.get("created"),
            "num_rooms": len(rooms_meta),
            "y_values": room_y_values,
            "mean_y": float(np.mean(room_y_values)) if room_y_values else None,
            "std_y": float(np.std(room_y_values)) if room_y_values else None,
            "mean_yaw": float(np.mean(room_yaws)) if room_yaws else None,
            "yaw_std": float(np.std(room_yaws)) if room_yaws else None,
        }
        result["floors"].append(fl_info)

    # Sort floors by creation time
    result["floors"].sort(key=lambda x: x.get("created") or 0)

    # Check cross-floor Y consistency
    if len(result["floors"]) >= 2:
        y_means = [
            (fl["type"], fl["mean_y"])
            for fl in result["floors"]
            if fl["mean_y"] is not None
        ]
        yaw_means = [
            (fl["type"], fl["mean_yaw"])
            for fl in result["floors"]
            if fl["mean_yaw"] is not None
        ]

        # Different floors should have different Y values (vertical offset)
        # but the same yaw (same session orientation)
        if len(yaw_means) >= 2:
            yaw_range = max(y[1] for y in yaw_means) - min(y[1] for y in yaw_means)
            result["cross_floor_yaw_range"] = yaw_range
            result["cross_floor_yaw_consistent"] = yaw_range < 1.0

        if len(y_means) >= 2:
            result["cross_floor_y_values"] = y_means

    return result


def analyze_azimuth_detail(uuid, name):
    """Detailed analysis of capturedAzimuths transforms."""
    scan_dir = find_scan_dir(uuid)
    if not scan_dir:
        return None

    with open(scan_dir / "data.json") as f:
        data = json.load(f)

    hm = data["homeMetadata"]
    azimuths = hm.get("capturedAzimuths", [])

    result = {"name": name, "uuid": uuid, "num_azimuths": len(azimuths)}

    # Group by storyId
    by_story = defaultdict(list)
    for az in azimuths:
        by_story[az.get("storyId", "?")].append(az)

    result["stories"] = {}
    for sid, azs in by_story.items():
        azs_sorted = sorted(azs, key=lambda x: x.get("azimuth", 0))

        # Check transform consistency per story
        # Each azimuth entry has transforms - a list of 4x4 matrices
        # Check if the transforms show multiple coordinate frames
        first_transforms = []
        for az in azs_sorted:
            ts = az.get("transforms", [])
            if ts:
                first_transforms.append(
                    {
                        "azimuth": az.get("azimuth"),
                        "translation": extract_translation(ts[0]),
                        "rotation": extract_rotation_matrix(ts[0]),
                        "num_transforms": len(ts),
                    }
                )

        # Check translation range (how far the captures spread)
        if first_transforms:
            trans = [
                ft["translation"]
                for ft in first_transforms
                if ft["translation"] is not None
            ]
            if len(trans) >= 2:
                trans = np.array(trans)
                spread = np.max(trans, axis=0) - np.min(trans, axis=0)
                max_dist = max(
                    np.linalg.norm(trans[i] - trans[j])
                    for i in range(len(trans))
                    for j in range(i + 1, len(trans))
                )
                result["stories"][sid] = {
                    "num_azimuths": len(azs),
                    "translation_spread_xyz": spread.tolist(),
                    "max_pairwise_distance": float(max_dist),
                }

    return result


def analyze_merged_structure(uuid, name):
    """Check merged.json for structural anomalies."""
    merged_path = PIPELINE / uuid / "merged.json"
    if not merged_path.exists():
        return None

    with open(merged_path) as f:
        merged = json.load(f)

    rooms = merged.get("rooms", [])
    walls = merged.get("walls", [])
    doors = merged.get("doors", [])
    windows = merged.get("windows", [])
    floors = merged.get("floors", [])

    # Check story distribution
    stories = set()
    for r in rooms:
        stories.add(r.get("story"))

    # Check room sizes via wall extents
    room_areas = []
    for r in rooms:
        room_walls = r.get("walls", [])
        if room_walls:
            # Approximate room area from wall transforms
            wall_positions = []
            for w in room_walls:
                trans = extract_translation(w.get("transform"))
                if trans is not None:
                    wall_positions.append(trans)
            if len(wall_positions) >= 2:
                positions = np.array(wall_positions)
                extent = np.max(positions, axis=0) - np.min(positions, axis=0)
                room_areas.append(float(extent[0] * extent[2]))  # XZ plane area approx

    return {
        "name": name,
        "uuid": uuid,
        "num_rooms": len(rooms),
        "num_walls": len(walls),
        "num_doors": len(doors),
        "num_windows": len(windows),
        "num_floors": len(floors),
        "stories": sorted(stories),
    }


def main():
    os.chdir(Path(__file__).parent)

    print("=" * 120)
    print("INVESTIGATING 5 TARGET BUILDINGS WITHOUT ROTATION-BASED SESSION RESTARTS")
    print("=" * 120)
    print()
    print("These 5 buildings were flagged as problematic but do NOT show the rotation")
    print("signature of an ARSession restart. Checking for other anomalies...")
    print()

    # --- Cross-floor analysis ---
    print("=" * 120)
    print("1. CROSS-FLOOR COORDINATE FRAME CONSISTENCY (Y offsets and Yaw)")
    print("=" * 120)

    for uuid, name in ALL_TARGETS.items():
        r = analyze_cross_floor_consistency(uuid, name)
        if not r or len(r["floors"]) < 2:
            continue

        tag = "[RESTART]" if uuid in RESTART_BUILDINGS else "[NO-RESTART]"
        print(f"\n  {tag} {r['name']}:")
        for fl in r["floors"]:
            yaw_std = fl.get("yaw_std", 0)
            yaw_flag = " *** HIGH YAW SPREAD ***" if yaw_std > 1.0 else ""
            print(
                f"    {fl['type']:>10}: mean_y={fl['mean_y']:.3f}m "
                f"(std={fl['std_y']:.4f}), "
                f"mean_yaw={fl['mean_yaw']:.1f}deg "
                f"(std={fl.get('yaw_std', 0):.3f}){yaw_flag}"
            )

        if "cross_floor_yaw_range" in r:
            consistent = (
                "YES"
                if r["cross_floor_yaw_consistent"]
                else "NO *** DIFFERENT SESSIONS ***"
            )
            print(
                f"    Cross-floor yaw range: {r['cross_floor_yaw_range']:.2f}deg -> "
                f"Consistent: {consistent}"
            )

    # --- Azimuth analysis ---
    print()
    print("=" * 120)
    print("2. CAPTURED AZIMUTHS SPATIAL SPREAD")
    print("=" * 120)

    for uuid, name in ALL_TARGETS.items():
        r = analyze_azimuth_detail(uuid, name)
        if not r:
            continue

        tag = "[RESTART]" if uuid in RESTART_BUILDINGS else "[NO-RESTART]"
        print(f"\n  {tag} {r['name']} ({r['num_azimuths']} azimuths):")
        for sid, info in r.get("stories", {}).items():
            print(
                f"    Story {sid[:20]}...: {info['num_azimuths']} azimuths, "
                f"max_pairwise_dist={info['max_pairwise_distance']:.2f}m, "
                f"spread_XYZ=[{info['translation_spread_xyz'][0]:.1f}, "
                f"{info['translation_spread_xyz'][1]:.1f}, "
                f"{info['translation_spread_xyz'][2]:.1f}]"
            )

    # --- Merged structure ---
    print()
    print("=" * 120)
    print("3. MERGED.JSON STRUCTURAL SUMMARY")
    print("=" * 120)

    for uuid, name in ALL_TARGETS.items():
        r = analyze_merged_structure(uuid, name)
        if not r:
            continue
        tag = "[RESTART]" if uuid in RESTART_BUILDINGS else "[NO-RESTART]"
        print(
            f"  {tag} {r['name']}: {r['num_rooms']} rooms, {r['num_walls']} walls, "
            f"{r['num_doors']} doors, {r['num_windows']} windows, "
            f"stories={r['stories']}"
        )

    # --- Timestamp analysis for remaining 5 ---
    print()
    print("=" * 120)
    print("4. DETAILED TIMESTAMP ANALYSIS FOR 5 NON-RESTART BUILDINGS")
    print("=" * 120)

    for uuid, name in REMAINING.items():
        scan_dir = find_scan_dir(uuid)
        if not scan_dir:
            continue
        with open(scan_dir / "data.json") as f:
            data = json.load(f)

        hm = data["homeMetadata"]
        floors = hm.get("floors", [])

        print(f"\n  {name}:")

        all_events = []
        for fl in floors:
            fl_type = fl.get("floorType", "unknown")
            fl_created = fl.get("created")
            if fl_created:
                all_events.append((fl_created, f"FLOOR-START:{fl_type}", fl_type))
            for r in fl.get("rooms", []):
                rcreated = r.get("created")
                if rcreated:
                    all_events.append((rcreated, f"ROOM:{r.get('name', '?')}", fl_type))

        all_events.sort()

        total_scan_time = (
            all_events[-1][0] - all_events[0][0] if len(all_events) >= 2 else 0
        )
        print(
            f"    Total scan duration: {total_scan_time:.0f}s "
            f"({total_scan_time / 60:.1f}min)"
        )
        print("    Scan timeline:")

        prev_time = None
        for t, label, fl_type in all_events:
            gap = f"  [+{t - prev_time:.0f}s]" if prev_time else ""
            big_gap = " *** BIG GAP ***" if prev_time and (t - prev_time) > 300 else ""
            print(f"      {t:.0f}{gap:>10} {fl_type:>10} {label}{big_gap}")
            prev_time = t

    # --- Check for the "Badeværelse" pattern (Gerskov) ---
    print()
    print("=" * 120)
    print("5. SPECIAL PATTERN CHECK: ROOMS WITH ANOMALOUS Y-OFFSET (FLOOR MISLABEL?)")
    print("=" * 120)

    for uuid, name in REMAINING.items():
        scan_dir = find_scan_dir(uuid)
        if not scan_dir:
            continue
        with open(scan_dir / "data.json") as f:
            data = json.load(f)

        hm = data["homeMetadata"]
        floors = hm.get("floors", [])

        print(f"\n  {name}:")
        for fl in floors:
            fl_type = fl.get("floorType", "unknown")
            rooms_meta = fl.get("rooms", [])

            y_values = []
            for r in rooms_meta:
                rid = r.get("id")
                rname = r.get("name", "?")
                room_json = scan_dir / f"{rid}.json"
                if room_json.exists():
                    with open(room_json) as f:
                        rd = json.load(f)
                    rot_flat = rd.get("referenceOriginTransform")
                    trans = extract_translation(rot_flat)
                    if trans is not None:
                        y_values.append((rname, trans[1], rid))

            if not y_values:
                continue

            mean_y = np.mean([y[1] for y in y_values])
            print(f"    {fl_type:>10} floor (mean_y={mean_y:.3f}m):")
            for rname, y, _rid in y_values:
                deviation = abs(y - mean_y)
                flag = " *** OUTLIER ***" if deviation > 1.0 else ""
                print(f"      {rname:<25} y={y:.3f}m (dev={deviation:.3f}m){flag}")


if __name__ == "__main__":
    main()
