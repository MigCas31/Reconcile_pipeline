#!/usr/bin/env python3
"""Analyze 8 problem buildings vs 5 random GREEN/YELLOW buildings."""

import json
import math
import random
from collections import defaultdict

random.seed(42)

PROBLEM_UUIDS = [
    "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3",  # Bredballe Byvej 63
    "1d26eda3-c927-4cdf-a3b0-218036828a55",  # Gerskov Bygade 24
    "7dbc53a6-17e8-4806-83de-42286b95726c",  # Gultvedgyden 6
    "e661e7b6-303d-415c-b378-2d9dd2fbfd6f",  # Hesselvænget 2
    "cb711a0b-6e8d-4ae6-b008-af3297446dcc",  # Kastanievej 6
    "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b",  # Lucernevej 23
    "feccbd0c-0420-4775-b5b7-49b99559947e",  # Morelvej 68
    "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5",  # Pilevej 30
]


def load_data():
    with open("reconcile/buildings_3d.json") as f:
        buildings_3d = json.load(f)
    with open("reconcile/batch_results.json") as f:
        batch = json.load(f)
    b3d_map = {b["uuid"]: b for b in buildings_3d}
    batch_map = {b["uuid"]: b for b in batch["results"]}
    return b3d_map, batch_map


def get_all_coords(building):
    """Extract all XYZ coordinates from walls and floor polygons."""
    all_pts = []
    for room in building.get("rooms", []):
        # Floor polygon points
        for pt in room.get("floor_polygon", []):
            all_pts.append(pt)
        # Merged wall corners
        for wall in room.get("walls_merged", []):
            for pt in wall.get("corners", []):
                all_pts.append(pt)
        # Raw wall corners
        for wall in room.get("walls_computed", []):
            for pt in wall.get("corners", []):
                all_pts.append(pt)
    return all_pts


def compute_centroid(pts):
    if not pts:
        return (0, 0, 0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def dist_3d(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=False)))


def dist_2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)  # XZ plane


def analyze_building(b3d, batch_entry):
    """Full spatial + metric analysis of one building."""
    info = {}
    # Batch metrics
    if batch_entry:
        info["classification"] = batch_entry.get("classification", "?")
        info["flags"] = batch_entry.get("flags", [])
        info["match_rate"] = batch_entry.get("match_rate")
        info["wall_count_merged"] = batch_entry.get("wall_count_merged")
        info["wall_count_scan"] = batch_entry.get("wall_count_scan")
        info["wall_reduction_pct"] = batch_entry.get("wall_reduction_pct")
        info["mean_width_delta_cm"] = batch_entry.get("mean_width_delta_cm")
        info["mean_height_delta_cm"] = batch_entry.get("mean_height_delta_cm")
        info["floor_gap_count"] = batch_entry.get("floor_gap_count")
        info["floor_gap_median_cm"] = batch_entry.get("floor_gap_median_cm")
        info["flagged_count"] = batch_entry.get("flagged_count")
    else:
        info["classification"] = b3d.get("classification", "?") if b3d else "?"
        info["flags"] = []

    if not b3d:
        info["error"] = "NOT FOUND in buildings_3d.json"
        return info

    info["address"] = b3d.get("address", "?")
    info["stories_found"] = b3d.get("stories_found")
    info["stories_changed"] = b3d.get("stories_changed")
    info["computed_walls_total"] = b3d.get("computed_walls_total")
    info["merged_walls_total"] = b3d.get("merged_walls_total")
    info["room_count"] = len(b3d.get("rooms", []))
    info["cross_floor_gaps"] = len(b3d.get("cross_floor_gaps", []))

    rooms = b3d.get("rooms", [])
    all_pts = get_all_coords(b3d)

    if not all_pts:
        info["error"] = "No coordinate data"
        return info

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    info["bbox_x"] = (min(xs), max(xs))
    info["bbox_y"] = (min(ys), max(ys))
    info["bbox_z"] = (min(zs), max(zs))
    info["extent_x"] = max(xs) - min(xs)
    info["extent_y"] = max(ys) - min(ys)
    info["extent_z"] = max(zs) - min(zs)

    centroid = compute_centroid(all_pts)
    info["centroid"] = centroid

    # Per-room analysis
    room_analyses = []
    for i, room in enumerate(rooms):
        r_pts = []
        for pt in room.get("floor_polygon", []):
            r_pts.append(pt)
        for wall in room.get("walls_merged", []):
            for pt in wall.get("corners", []):
                r_pts.append(pt)
        if not r_pts:
            continue
        r_centroid = compute_centroid(r_pts)
        r_dist = dist_2d(r_centroid, centroid)
        r_xs = [p[0] for p in r_pts]
        r_zs = [p[2] for p in r_pts]
        r_extent = max(max(r_xs) - min(r_xs), max(r_zs) - min(r_zs))
        room_analyses.append(
            {
                "room_idx": i,
                "story": room.get("story"),
                "dist_from_centroid_2d": r_dist,
                "room_extent": r_extent,
                "n_merged_walls": len(room.get("walls_merged", [])),
                "n_computed_walls": len(room.get("walls_computed", [])),
                "n_floor_pts": len(room.get("floor_polygon", [])),
            }
        )

    info["room_analyses"] = room_analyses

    # Outlier rooms: distance from centroid > 2x median
    dists = [r["dist_from_centroid_2d"] for r in room_analyses]
    if dists:
        median_dist = sorted(dists)[len(dists) // 2]
        info["median_room_dist"] = median_dist
        info["max_room_dist"] = max(dists)
        outlier_rooms = [
            r
            for r in room_analyses
            if r["dist_from_centroid_2d"] > 2 * median_dist and median_dist > 0.5
        ]
        info["outlier_rooms_count"] = len(outlier_rooms)
        info["outlier_rooms"] = outlier_rooms
    else:
        info["median_room_dist"] = 0
        info["max_room_dist"] = 0
        info["outlier_rooms_count"] = 0
        info["outlier_rooms"] = []

    # Story height analysis
    story_ys = defaultdict(list)
    for room in rooms:
        story = room.get("story", -1)
        for pt in room.get("floor_polygon", []):
            story_ys[story].append(pt[1])
        for wall in room.get("walls_merged", []):
            for pt in wall.get("corners", []):
                story_ys[story].append(pt[1])

    story_ranges = {}
    for s, yvs in sorted(story_ys.items()):
        story_ranges[s] = {
            "min_y": min(yvs),
            "max_y": max(yvs),
            "range_y": max(yvs) - min(yvs),
        }
    info["story_ranges"] = story_ranges

    # Check for inconsistent story heights (stories that overlap in Y)
    sorted_stories = sorted(story_ranges.keys())
    overlaps = []
    for i_s in range(len(sorted_stories) - 1):
        s1 = sorted_stories[i_s]
        s2 = sorted_stories[i_s + 1]
        if story_ranges[s1]["max_y"] > story_ranges[s2]["min_y"]:
            overlaps.append((s1, s2))
    info["story_overlaps"] = overlaps

    # Check for walls far from building centroid
    far_walls = []
    for room in rooms:
        for wall in room.get("walls_merged", []):
            corners = wall.get("corners", [])
            if not corners:
                continue
            w_centroid = compute_centroid(corners)
            d = dist_2d(w_centroid, centroid)
            if d > info["max_room_dist"] * 1.5 and d > 10:
                far_walls.append(
                    {
                        "id": wall.get("id", "?"),
                        "dist": d,
                        "story": room.get("story"),
                    }
                )
    info["far_walls_count"] = len(far_walls)

    # Check for duplicate room floor polygons (overlapping rooms)
    floor_sigs = []
    for i_r, room in enumerate(rooms):
        fp = room.get("floor_polygon", [])
        if fp:
            c = compute_centroid(fp)
            floor_sigs.append(
                (
                    i_r,
                    room.get("story"),
                    round(c[0], 1),
                    round(c[1], 1),
                    round(c[2], 1),
                    len(fp),
                )
            )

    duplicates = []
    for i_a in range(len(floor_sigs)):
        for i_b in range(i_a + 1, len(floor_sigs)):
            a, b = floor_sigs[i_a], floor_sigs[i_b]
            if (
                abs(a[2] - b[2]) < 0.5
                and abs(a[3] - b[3]) < 0.5
                and abs(a[4] - b[4]) < 0.5
            ):
                duplicates.append((a[0], b[0], a[1], b[1]))
    info["duplicate_rooms"] = duplicates

    return info


def fmt(val, decimals=2):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def print_building_report(uuid, info, label=""):
    addr = info.get("address", "?")
    print(f"\n{'=' * 90}")
    print(f"  {label}{addr}  [{uuid[:12]}...]")
    print(
        f"  Classification: {info.get('classification', '?')}  |  Flags: "
        f"{', '.join(info.get('flags', [])) or 'none'}"
    )
    print(f"{'=' * 90}")

    if "error" in info:
        print(f"  ERROR: {info['error']}")
        return

    print(
        f"  Stories: {info.get('stories_found')} found, {info.get('stories_changed')} "
        f"changed"
    )
    print(
        f"  Rooms: {info.get('room_count')}  |  Cross-floor gaps: "
        f"{info.get('cross_floor_gaps')}"
    )
    print(
        f"  Walls: {info.get('merged_walls_total')} merged / "
        f"{info.get('computed_walls_total')} computed"
    )

    if info.get("match_rate") is not None:
        print(
            f"  Match rate: {fmt(info['match_rate'])}  |  Wall reduction: "
            f"{fmt(info.get('wall_reduction_pct'))}%"
        )
        print(
            f"  Mean width delta: {fmt(info.get('mean_width_delta_cm'))} cm  |  Mean "
            f"height delta: {fmt(info.get('mean_height_delta_cm'))} cm"
        )
        print(
            f"  Floor gap count: {info.get('floor_gap_count')}  |  Floor gap median: "
            f"{fmt(info.get('floor_gap_median_cm'))} cm"
        )
        print(f"  Flagged count: {info.get('flagged_count')}")

    print("\n  --- Spatial Extent ---")
    print(
        f"  BBox X: [{fmt(info['bbox_x'][0])}, {fmt(info['bbox_x'][1])}]  "
        f"extent={fmt(info['extent_x'])} m"
    )
    print(
        f"  BBox Y: [{fmt(info['bbox_y'][0])}, {fmt(info['bbox_y'][1])}]  "
        f"extent={fmt(info['extent_y'])} m (vertical)"
    )
    print(
        f"  BBox Z: [{fmt(info['bbox_z'][0])}, {fmt(info['bbox_z'][1])}]  "
        f"extent={fmt(info['extent_z'])} m"
    )
    print(
        f"  Centroid: ({fmt(info['centroid'][0])}, {fmt(info['centroid'][1])}, "
        f"{fmt(info['centroid'][2])})"
    )

    print("\n  --- Story Height Ranges (Y axis) ---")
    for s, sr in sorted(info.get("story_ranges", {}).items()):
        print(
            f"    Story {s}: Y=[{fmt(sr['min_y'])}, {fmt(sr['max_y'])}]  "
            f"range={fmt(sr['range_y'])} m"
        )
    if info.get("story_overlaps"):
        print(f"  *** STORY OVERLAPS: {info['story_overlaps']}")

    print("\n  --- Room Distance from Centroid ---")
    print(
        f"  Median room dist: {fmt(info.get('median_room_dist'))} m  |  Max: "
        f"{fmt(info.get('max_room_dist'))} m"
    )
    print(f"  Outlier rooms (>2x median): {info.get('outlier_rooms_count')}")
    for r in info.get("outlier_rooms", []):
        print(
            f"    Room {r['room_idx']} (story {r['story']}): "
            f"dist={fmt(r['dist_from_centroid_2d'])} m, "
            f"extent={fmt(r['room_extent'])} m"
        )

    if info.get("far_walls_count", 0) > 0:
        print(
            f"  *** FAR WALLS (>1.5x max room dist & >10m): {info['far_walls_count']}"
        )

    if info.get("duplicate_rooms"):
        print(
            f"  *** POTENTIALLY OVERLAPPING ROOMS: {len(info['duplicate_rooms'])} pairs"
        )
        for a, b, sa, sb in info["duplicate_rooms"]:
            print(f"    Rooms {a} & {b} (stories {sa} & {sb})")


def main():
    b3d_map, batch_map = load_data()

    # Analyze problem buildings
    problem_results = {}
    for uuid in PROBLEM_UUIDS:
        b3d = b3d_map.get(uuid)
        batch = batch_map.get(uuid)
        problem_results[uuid] = analyze_building(b3d, batch)

    # Pick 5 random GREEN or YELLOW buildings for comparison
    green_yellow = [
        u
        for u, b in batch_map.items()
        if b.get("classification") in ("GREEN", "YELLOW") and u not in PROBLEM_UUIDS
    ]
    comparison_uuids = random.sample(green_yellow, min(5, len(green_yellow)))

    comparison_results = {}
    for uuid in comparison_uuids:
        b3d = b3d_map.get(uuid)
        batch = batch_map.get(uuid)
        comparison_results[uuid] = analyze_building(b3d, batch)

    # Print reports
    print("\n" + "#" * 90)
    print("#  PROBLEM BUILDINGS (8)")
    print("#" * 90)
    for uuid in PROBLEM_UUIDS:
        print_building_report(uuid, problem_results[uuid], label="[PROBLEM] ")

    print("\n\n" + "#" * 90)
    print("#  COMPARISON BUILDINGS (5 random GREEN/YELLOW)")
    print("#" * 90)
    for uuid in comparison_uuids:
        print_building_report(uuid, comparison_results[uuid], label="[COMPARISON] ")

    # Summary comparison table
    print("\n\n" + "=" * 130)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 130)
    header = (
        f"{'Address':<30} {'Class':>6} {'Rooms':>5} {'Stories':>7} {'ExtentX':>8} "
        f"{'ExtentY':>8} {'ExtentZ':>8} {'MaxRmDist':>9} {'Outliers':>8} "
        f"{'Overlaps':>8} {'FarWalls':>8} {'Flags':>5}"
    )
    print(header)
    print("-" * 130)

    def print_summary_row(uuid, info, tag=""):
        addr = info.get("address", "?")[:28]
        if tag:
            addr = tag + addr
        cls = info.get("classification", "?")
        rooms = info.get("room_count", "?")
        stories = info.get("stories_found", "?")
        ex = fmt(info.get("extent_x", 0))
        ey = fmt(info.get("extent_y", 0))
        ez = fmt(info.get("extent_z", 0))
        mrd = fmt(info.get("max_room_dist", 0))
        outl = info.get("outlier_rooms_count", 0)
        ovlp = len(info.get("duplicate_rooms", []))
        fw = info.get("far_walls_count", 0)
        fl = info.get("flagged_count", "?")
        print(
            f"{addr:<30} {cls:>6} {rooms:>5} {stories:>7} {ex:>8} {ey:>8} {ez:>8} "
            f"{mrd:>9} {outl:>8} {ovlp:>8} {fw:>8} {fl:>5}"
        )

    for uuid in PROBLEM_UUIDS:
        print_summary_row(uuid, problem_results[uuid], "* ")
    print("-" * 130)
    for uuid in comparison_uuids:
        print_summary_row(uuid, comparison_results[uuid], "  ")

    # Aggregate stats
    print("\n\nAGGREGATE STATISTICS")
    print("=" * 80)
    for label, results in [
        ("PROBLEM", problem_results),
        ("COMPARISON", comparison_results),
    ]:
        extents_x = [r["extent_x"] for r in results.values() if "extent_x" in r]
        extents_y = [r["extent_y"] for r in results.values() if "extent_y" in r]
        extents_z = [r["extent_z"] for r in results.values() if "extent_z" in r]
        max_dists = [
            r["max_room_dist"] for r in results.values() if "max_room_dist" in r
        ]
        outliers = [
            r["outlier_rooms_count"]
            for r in results.values()
            if "outlier_rooms_count" in r
        ]
        overlaps = [len(r.get("duplicate_rooms", [])) for r in results.values()]

        print(f"\n  {label} (n={len(results)}):")
        if extents_x:
            print(
                f"    Extent X:  mean={sum(extents_x) / len(extents_x):.2f}  "
                f"max={max(extents_x):.2f}"
            )
            print(
                f"    Extent Y:  mean={sum(extents_y) / len(extents_y):.2f}  "
                f"max={max(extents_y):.2f}"
            )
            print(
                f"    Extent Z:  mean={sum(extents_z) / len(extents_z):.2f}  "
                f"max={max(extents_z):.2f}"
            )
            print(
                f"    MaxRmDist: mean={sum(max_dists) / len(max_dists):.2f}  "
                f"max={max(max_dists):.2f}"
            )
            print(f"    Outlier rooms: total={sum(outliers)}")
            print(f"    Overlapping room pairs: total={sum(overlaps)}")


if __name__ == "__main__":
    main()
