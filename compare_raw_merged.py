#!/usr/bin/env python3
"""
Compare raw room scan data with merged building data for 8 problematic buildings
to detect reconstruction issues.
"""

import json
import math
import os

BASE = "/Users/martincollignon/conductor/workspaces/look-ma-no-hands/tirana"
SCAN_CACHE = os.path.join(BASE, ".scan-cache")
PIPELINE = os.path.join(BASE, "pipeline-outputs")

BUILDINGS = {
    "1f03f6e0": {
        "name": "Bredballe Byvej 63",
        "uuid": "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3",
        "scan_dir": (
            "scans_imgyiHP46oPmBK2ZQLlx3kLxLAR2_Bredballe_Byvej_63__7120_Vejle__st_"
            "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3_1772204075_129672"
        ),
    },
    "1d26eda3": {
        "name": "Gerskov Bygade 24",
        "uuid": "1d26eda3-c927-4cdf-a3b0-218036828a55",
        "scan_dir": (
            "scans_ZbzSl5KqVsNNBG6I5p53yxeLoNT2_Gerskov_Bygade_24__5450_Otterup_"
            "1d26eda3-c927-4cdf-a3b0-218036828a55_1772632245_7456899"
        ),
    },
    "7dbc53a6": {
        "name": "Gultvedgyden 6",
        "uuid": "7dbc53a6-17e8-4806-83de-42286b95726c",
        "scan_dir": (
            "scans_Bmgf8ROc4ZcAtPoPIi5P8vxm2VI3_Gultvedgyden_6__5772_Kv_rndrup_"
            "7dbc53a6-17e8-4806-83de-42286b95726c_1773653476_089257"
        ),
    },
    "e661e7b6": {
        "name": "Hesselvænget 2",
        "uuid": "e661e7b6-303d-415c-b378-2d9dd2fbfd6f",
        "scan_dir": (
            "scans_ZbzSl5KqVsNNBG6I5p53yxeLoNT2_Hesselv_nget_2__5800_Nyborg_"
            "e661e7b6-303d-415c-b378-2d9dd2fbfd6f_1772704341_1621342"
        ),
    },
    "cb711a0b": {
        "name": "Kastanievej 6",
        "uuid": "cb711a0b-6e8d-4ae6-b008-af3297446dcc",
        "scan_dir": (
            "scans_aKeHkHmE5fM8kkPth63xGUGOfz63_Kastanievej_6__5400_Bogense_"
            "cb711a0b-6e8d-4ae6-b008-af3297446dcc_1774438563_092342"
        ),
    },
    "b4b6f3ed": {
        "name": "Lucernevej 23",
        "uuid": "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b",
        "scan_dir": (
            "scans_RzpAyFHQtPf6afmXkhqnIwPo9QU2_Lucernevej_23__8700_Horsens_"
            "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b_1772535414_466711"
        ),
    },
    "feccbd0c": {
        "name": "Morelvej 68",
        "uuid": "feccbd0c-0420-4775-b5b7-49b99559947e",
        "scan_dir": (
            "scans_ZbzSl5KqVsNNBG6I5p53yxeLoNT2_Morelvej_68__5250_Odense_SV_"
            "feccbd0c-0420-4775-b5b7-49b99559947e_1770801053_970685"
        ),
    },
    "99ce0ab2": {
        "name": "Pilevej 30",
        "uuid": "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5",
        "scan_dir": (
            "scans_ZbzSl5KqVsNNBG6I5p53yxeLoNT2_Pilevej_30__5792_A_rslev_"
            "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5_1775030332_775651"
        ),
    },
}

EXCLUDE_PREFIXES = ("data.", "arworldmap.", "ceiling_")


def get_translation(transform):
    """Extract translation (x,y,z) from column-major 4x4 transform."""
    if not transform or len(transform) < 16:
        return (0, 0, 0)
    return (transform[12], transform[13], transform[14])


def dist3d(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=False)))


def load_raw_rooms(scan_dir_path):
    """
    Load all room JSONs from scan-cache, excluding data.json, arworldmap.json,
    ceiling_*.json
    """
    rooms = {}
    for fname in os.listdir(scan_dir_path):
        if not fname.endswith(".json"):
            continue
        if any(fname.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        fpath = os.path.join(scan_dir_path, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            room_id = fname.replace(".json", "")
            rooms[room_id] = data
        except (OSError, json.JSONDecodeError):
            pass
    return rooms


def load_merged(uuid):
    """Load merged.json from pipeline-outputs."""
    mpath = os.path.join(PIPELINE, uuid, "merged.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as f:
        return json.load(f)


def analyze_building(short_uuid, info):
    print(f"\n{'=' * 80}")
    print(f"BUILDING: {info['name']} ({info['uuid']})")
    print(f"{'=' * 80}")

    scan_path = os.path.join(SCAN_CACHE, info["scan_dir"])
    if not os.path.exists(scan_path):
        print("  ERROR: Scan cache directory not found!")
        return

    raw_rooms = load_raw_rooms(scan_path)
    merged = load_merged(info["uuid"])

    if not merged:
        print("  ERROR: merged.json not found!")
        return

    merged_rooms = merged.get("rooms", [])

    print(f"\n  Raw room scans: {len(raw_rooms)}")
    print(f"  Merged rooms:   {len(merged_rooms)}")

    # --- Build wall index from raw rooms ---
    # wall_id -> { "room_id": ..., "raw_transform": ..., "raw_translation": ...,
    # "story": ... }
    raw_wall_index = {}
    raw_room_summaries = {}

    for room_id, room_data in raw_rooms.items():
        walls = room_data.get("walls", [])
        doors = room_data.get("doors", [])
        windows = room_data.get("windows", [])
        floors = room_data.get("floors", [])
        story = room_data.get("story", "?")

        floor_corners = []
        for fl in floors:
            if fl.get("polygonCorners"):
                floor_corners.extend(fl["polygonCorners"])

        raw_room_summaries[room_id] = {
            "story": story,
            "num_walls": len(walls),
            "num_doors": len(doors),
            "num_windows": len(windows),
            "num_floor_corners": len(floor_corners),
        }

        for w in walls:
            wid = w.get("identifier")
            if wid:
                raw_wall_index[wid] = {
                    "room_id": room_id,
                    "transform": w.get("transform"),
                    "translation": get_translation(w.get("transform")),
                    "story": story,
                }

    # --- Build wall index from merged rooms ---
    merged_wall_index = {}
    merged_room_wall_sets = {}  # room_idx -> set of wall ids

    for idx, mroom in enumerate(merged_rooms):
        mwalls = mroom.get("walls", [])
        mstory = mroom.get("story", "?")
        wall_ids_in_room = set()
        for w in mwalls:
            wid = w.get("identifier")
            if wid:
                merged_wall_index[wid] = {
                    "room_idx": idx,
                    "transform": w.get("transform"),
                    "translation": get_translation(w.get("transform")),
                    "story": mstory,
                }
                wall_ids_in_room.add(wid)
        merged_room_wall_sets[idx] = wall_ids_in_room

    # --- Analysis ---

    # 1) Check for raw rooms that got dropped entirely
    print("\n  --- Raw Room Summaries ---")
    for room_id, summary in sorted(raw_room_summaries.items()):
        print(
            f"    Room '{room_id}': story={summary['story']}, "
            f"walls={summary['num_walls']}, doors={summary['num_doors']}, "
            f"windows={summary['num_windows']}, "
            f"floor_corners={summary['num_floor_corners']}"
        )

    # 2) Check which raw walls are missing from merged
    raw_wall_ids = set(raw_wall_index.keys())
    merged_wall_ids = set(merged_wall_index.keys())

    missing_walls = raw_wall_ids - merged_wall_ids
    extra_walls = merged_wall_ids - raw_wall_ids
    common_walls = raw_wall_ids & merged_wall_ids

    print("\n  --- Wall UUID Comparison ---")
    print(f"    Raw wall UUIDs:    {len(raw_wall_ids)}")
    print(f"    Merged wall UUIDs: {len(merged_wall_ids)}")
    print(f"    Common:            {len(common_walls)}")
    print(f"    Missing from merged (dropped): {len(missing_walls)}")
    print(f"    Extra in merged (new):         {len(extra_walls)}")

    if missing_walls:
        # Group missing walls by room
        missing_by_room = {}
        for wid in missing_walls:
            rid = raw_wall_index[wid]["room_id"]
            missing_by_room.setdefault(rid, []).append(wid)
        print("\n    Walls DROPPED from merged, by source room:")
        for rid, wids in sorted(missing_by_room.items()):
            total_walls_in_room = raw_room_summaries[rid]["num_walls"]
            print(
                f"      Room '{rid}' (story {raw_room_summaries[rid]['story']}): "
                f"{len(wids)}/{total_walls_in_room} walls dropped"
            )

    # Check if any raw rooms were entirely dropped
    rooms_with_any_wall_in_merged = set()
    rooms_entirely_dropped = set()
    for room_id, _summary in raw_room_summaries.items():
        room_walls = [
            wid for wid, info in raw_wall_index.items() if info["room_id"] == room_id
        ]
        if any(wid in merged_wall_ids for wid in room_walls):
            rooms_with_any_wall_in_merged.add(room_id)
        elif room_walls:
            rooms_entirely_dropped.add(room_id)

    if rooms_entirely_dropped:
        print(f"\n    *** ROOMS ENTIRELY DROPPED ({len(rooms_entirely_dropped)}): ***")
        for rid in sorted(rooms_entirely_dropped):
            s = raw_room_summaries[rid]
            print(
                f"      Room '{rid}': story={s['story']}, walls={s['num_walls']}, "
                f"doors={s['num_doors']}, windows={s['num_windows']}"
            )

    # 3) For common walls: compute position differences
    print("\n  --- Wall Position Differences (raw vs merged) ---")
    large_diffs = []
    all_diffs = []

    for wid in common_walls:
        raw_t = raw_wall_index[wid]["translation"]
        merged_t = merged_wall_index[wid]["translation"]
        d = dist3d(raw_t, merged_t)
        all_diffs.append(d)

        raw_story = raw_wall_index[wid]["story"]
        merged_story = merged_wall_index[wid]["story"]
        room_id = raw_wall_index[wid]["room_id"]

        if d > 1.0:  # More than 1 meter difference
            large_diffs.append(
                {
                    "wall_id": wid,
                    "room_id": room_id,
                    "raw_pos": raw_t,
                    "merged_pos": merged_t,
                    "distance": d,
                    "raw_story": raw_story,
                    "merged_story": merged_story,
                }
            )

    if all_diffs:
        all_diffs.sort()
        print(f"    Total walls compared: {len(all_diffs)}")
        print(f"    Min position diff:    {all_diffs[0]:.3f} m")
        print(f"    Median position diff: {all_diffs[len(all_diffs) // 2]:.3f} m")
        print(f"    Max position diff:    {all_diffs[-1]:.3f} m")
        print(f"    Mean position diff:   {sum(all_diffs) / len(all_diffs):.3f} m")
        print(f"    Walls moved > 1m:     {len(large_diffs)}")
        print(f"    Walls moved > 3m:     {sum(1 for d in all_diffs if d > 3.0)}")
        print(f"    Walls moved > 5m:     {sum(1 for d in all_diffs if d > 5.0)}")
        print(f"    Walls moved > 10m:    {sum(1 for d in all_diffs if d > 10.0)}")

    if large_diffs:
        large_diffs.sort(key=lambda x: -x["distance"])
        print("\n    *** TOP WALLS WITH LARGE POSITION CHANGES (>1m): ***")
        for entry in large_diffs[:20]:
            dx = entry["merged_pos"][0] - entry["raw_pos"][0]
            dy = entry["merged_pos"][1] - entry["raw_pos"][1]
            dz = entry["merged_pos"][2] - entry["raw_pos"][2]
            print(f"      Wall {entry['wall_id'][:12]}... (room '{entry['room_id']}')")
            print(
                f"        raw_story={entry['raw_story']} -> "
                f"merged_story={entry['merged_story']}"
            )
            print(
                f"        dist={entry['distance']:.2f}m  dx={dx:.2f} dy={dy:.2f} "
                f"dz={dz:.2f}"
            )

    # 4) Check for rooms whose walls got scattered across multiple stories
    print("\n  --- Story Consistency Check ---")
    for room_id in sorted(raw_room_summaries.keys()):
        raw_story = raw_room_summaries[room_id]["story"]
        room_walls = [
            wid for wid, info in raw_wall_index.items() if info["room_id"] == room_id
        ]
        merged_stories = set()
        merged_room_indices = set()
        for wid in room_walls:
            if wid in merged_wall_index:
                merged_stories.add(merged_wall_index[wid]["story"])
                merged_room_indices.add(merged_wall_index[wid]["room_idx"])

        if len(merged_stories) > 1:
            print(
                f"    *** SCATTERED *** Room '{room_id}' (raw story={raw_story}): "
                f"walls ended up in merged stories {merged_stories}"
            )
        elif merged_stories and next(iter(merged_stories)) != raw_story:
            print(
                f"    STORY CHANGED: Room '{room_id}' raw_story={raw_story} -> "
                f"merged_story={next(iter(merged_stories))}"
            )

        if len(merged_room_indices) > 1:
            print(
                f"    *** SPLIT *** Room '{room_id}': walls split across "
                f"merged room indices {merged_room_indices}"
            )

    # 5) Per-room transform analysis: check if the room's referenceOriginTransform
    # changed dramatically
    print("\n  --- Room-Level Transform Comparison ---")
    # Try to match raw rooms to merged rooms by wall overlap
    for room_id in sorted(raw_room_summaries.items(), key=lambda x: x[0]):
        room_id = room_id[0]
        room_walls = {
            wid for wid, info in raw_wall_index.items() if info["room_id"] == room_id
        }

        # Find which merged room has most overlap
        best_merged_idx = None
        best_overlap = 0
        for midx, mwall_set in merged_room_wall_sets.items():
            overlap = len(room_walls & mwall_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_merged_idx = midx

        if best_merged_idx is not None and best_overlap > 0:
            raw_ref = raw_rooms[room_id].get("referenceOriginTransform")
            merged_ref = merged_rooms[best_merged_idx].get("referenceOriginTransform")
            if raw_ref and merged_ref:
                raw_ref_t = get_translation(raw_ref)
                merged_ref_t = get_translation(merged_ref)
                ref_dist = dist3d(raw_ref_t, merged_ref_t)
                if ref_dist > 2.0:
                    print(
                        f"    *** LARGE REF TRANSFORM SHIFT *** Room '{room_id}' -> "
                        f"merged room {best_merged_idx}: ref origin moved "
                        f"{ref_dist:.2f}m"
                    )
                    print(
                        f"      raw_ref_pos=({raw_ref_t[0]:.2f}, {raw_ref_t[1]:.2f}, "
                        f"{raw_ref_t[2]:.2f})"
                    )
                    print(
                        f"      merged_ref_pos=({merged_ref_t[0]:.2f}, "
                        f"{merged_ref_t[1]:.2f}, {merged_ref_t[2]:.2f})"
                    )

            # Also compute average wall displacement for this room
            displacements = []
            for wid in room_walls:
                if wid in merged_wall_index:
                    d = dist3d(
                        raw_wall_index[wid]["translation"],
                        merged_wall_index[wid]["translation"],
                    )
                    displacements.append(d)
            if displacements:
                avg_d = sum(displacements) / len(displacements)
                max_d = max(displacements)
                if avg_d > 2.0:
                    print(
                        f"    *** HIGH AVG WALL DISPLACEMENT *** Room '{room_id}' -> "
                        f"merged room {best_merged_idx}: avg={avg_d:.2f}m, "
                        f"max={max_d:.2f}m "
                        f"({len(displacements)} walls)"
                    )
        else:
            if room_walls:
                print(
                    f"    Room '{room_id}': NO matching merged room found (all walls "
                    f"dropped)"
                )


# Run analysis
print("BUILDING SCAN vs MERGED COMPARISON REPORT")
print(f"Analyzing {len(BUILDINGS)} buildings...")

for short_uuid, info in BUILDINGS.items():
    analyze_building(short_uuid, info)

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
