#!/usr/bin/env python3
"""Compare problem buildings against reference buildings using wall-level transforms."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("pipeline-outputs")

PROBLEM_BUILDINGS = {
    "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3": "Bredballe Byvej 63",
    "1d26eda3-c927-4cdf-a3b0-218036828a55": "Gerskov Bygade 24",
    "7dbc53a6-17e8-4806-83de-42286b95726c": "Gultvedgyden 6",
    "e661e7b6-303d-415c-b378-2d9dd2fbfd6f": "Hesselvænget 2",
    "cb711a0b-6e8d-4ae6-b008-af3297446dcc": "Kastanievej 6",
    "b4b6f3ed-7bfd-43a8-aeed-520b558bfa2b": "Lucernevej 23",
    "feccbd0c-0420-4775-b5b7-49b99559947e": "Morelvej 68",
    "99ce0ab2-eaac-406c-9fa2-707d7dbe30c5": "Pilevej 30",
}

REFERENCE_BUILDINGS = {
    "e9a5d5e2-17c8-4879-89f2-f6a3af5cfd38": "Elme Alle 30 (GREEN)",
    "019e1376-9762-42d6-8520-b664b8c752df": "Kielshusvej 18 (YELLOW)",
    "0430ebc2-236b-4b5d-991f-3e97ad246b78": "Vesterled 13 (YELLOW)",
    "05cecad4-119e-4dd2-beaf-d4af36973644": "Majsvej 11 (YELLOW)",
    "0b75d30e-c50c-4fc6-88ff-fce983078aa4": "Østerbrogade 63 (YELLOW)",
    "107e8496-9bff-42bb-b776-720f44b70e55": "Serridslevvej 75 (YELLOW)",
    "1825a812-09d0-4407-9265-182a07053cfc": "Morelvej 7 (YELLOW)",
    "191e11c0-fd47-489c-9e3d-c8072713a761": "Heimdalsvej 46 (YELLOW)",
}


def extract_translation(t):
    """Extract tx, ty, tz from column-major 4x4 transform."""
    if not t or len(t) < 16:
        return None
    return (t[12], t[13], t[14])


def extract_rotation_angle(t):
    """Extract rotation angle in degrees from column-major 4x4 transform."""
    if not t or len(t) < 16:
        return 0
    trace = t[0] + t[5] + t[10]
    cos_theta = max(-1, min(1, (trace - 1) / 2))
    return math.degrees(math.acos(cos_theta))


def analyze_building(uuid):
    """Analyze a single building from its merged.json using wall-level transforms."""
    merged_path = BASE / uuid / "merged.json"
    if not merged_path.exists():
        return None

    with open(merged_path) as f:
        data = json.load(f)

    rooms = data.get("rooms", [])
    all_walls_global = data.get("walls", [])

    result = {
        "num_rooms": len(rooms),
        "num_walls_global": len(all_walls_global),
        "stories": defaultdict(int),
        "room_details": [],
    }

    # ---- Per-room analysis using wall transforms ----
    all_wall_positions = []  # all wall center positions across all rooms
    all_wall_heights = []
    all_wall_widths = []
    room_centroids = []

    for i, room in enumerate(rooms):
        story = room.get("story", "unknown")
        result["stories"][story] += 1

        walls = room.get("walls", [])
        room_wall_positions = []
        room_wall_heights = []

        for wall in walls:
            t = wall.get("transform")
            pos = extract_translation(t)
            if pos:
                room_wall_positions.append(pos)
                all_wall_positions.append(pos)

            dims = wall.get("dimensions", [0, 0, 0])
            if len(dims) >= 2:
                all_wall_widths.append(dims[0])
                all_wall_heights.append(dims[1])
                room_wall_heights.append(dims[1])

        # Room centroid (from wall positions)
        if room_wall_positions:
            rc = (
                np.mean([p[0] for p in room_wall_positions]),
                np.mean([p[1] for p in room_wall_positions]),
                np.mean([p[2] for p in room_wall_positions]),
            )
            room_centroids.append(rc)

            # Room spatial extent
            xs = [p[0] for p in room_wall_positions]
            ys = [p[1] for p in room_wall_positions]
            zs = [p[2] for p in room_wall_positions]
        else:
            rc = (0, 0, 0)
            xs = ys = zs = [0]

        ref_t = room.get("referenceOriginTransform")
        ref_rot = extract_rotation_angle(ref_t) if ref_t else 0

        result["room_details"].append(
            {
                "index": i,
                "story": story,
                "wall_count": len(walls),
                "centroid": rc,
                "x_extent": max(xs) - min(xs) if xs else 0,
                "y_extent": max(ys) - min(ys) if ys else 0,
                "z_extent": max(zs) - min(zs) if zs else 0,
                "wall_height_mean": np.mean(room_wall_heights)
                if room_wall_heights
                else 0,
                "wall_height_std": np.std(room_wall_heights)
                if room_wall_heights
                else 0,
                "ref_rotation": ref_rot,
            }
        )

    # ---- Building-level spatial analysis ----
    if all_wall_positions:
        wxs = [p[0] for p in all_wall_positions]
        wys = [p[1] for p in all_wall_positions]
        wzs = [p[2] for p in all_wall_positions]
        result["wall_x_spread"] = max(wxs) - min(wxs)
        result["wall_y_spread"] = max(wys) - min(wys)  # vertical
        result["wall_z_spread"] = max(wzs) - min(wzs)
        result["wall_x_range"] = (min(wxs), max(wxs))
        result["wall_y_range"] = (min(wys), max(wys))
        result["wall_z_range"] = (min(wzs), max(wzs))

        building_centroid = (np.mean(wxs), np.mean(wys), np.mean(wzs))
        result["building_centroid"] = building_centroid

        # Distance of each wall from building centroid
        dists = [
            math.sqrt(
                (p[0] - building_centroid[0]) ** 2
                + (p[1] - building_centroid[1]) ** 2
                + (p[2] - building_centroid[2]) ** 2
            )
            for p in all_wall_positions
        ]
        result["wall_max_dist"] = max(dists)
        result["wall_mean_dist"] = np.mean(dists)

        # Check room centroids vs building centroid
        if room_centroids:
            room_dists = [
                math.sqrt(
                    (rc[0] - building_centroid[0]) ** 2
                    + (rc[1] - building_centroid[1]) ** 2
                    + (rc[2] - building_centroid[2]) ** 2
                )
                for rc in room_centroids
            ]
            result["room_max_dist_from_bldg_centroid"] = max(room_dists)
            result["room_mean_dist_from_bldg_centroid"] = np.mean(room_dists)

            # Find outlier rooms (>10m from building centroid)
            result["rooms_far_from_centroid"] = [
                (result["room_details"][i], d)
                for i, d in enumerate(room_dists)
                if d > 10
            ]
            # Statistical outliers
            if len(room_dists) > 2:
                std_d = np.std(room_dists)
                mean_d = np.mean(room_dists)
                result["room_outliers_2std"] = [
                    (result["room_details"][i], d)
                    for i, d in enumerate(room_dists)
                    if d > mean_d + 2 * std_d
                ]
            else:
                result["room_outliers_2std"] = []
        else:
            result["room_max_dist_from_bldg_centroid"] = 0
            result["room_mean_dist_from_bldg_centroid"] = 0
            result["rooms_far_from_centroid"] = []
            result["room_outliers_2std"] = []
    else:
        for k in [
            "wall_x_spread",
            "wall_y_spread",
            "wall_z_spread",
            "wall_max_dist",
            "wall_mean_dist",
            "room_max_dist_from_bldg_centroid",
            "room_mean_dist_from_bldg_centroid",
        ]:
            result[k] = 0
        result["building_centroid"] = (0, 0, 0)
        result["rooms_far_from_centroid"] = []
        result["room_outliers_2std"] = []

    # Wall height stats
    if all_wall_heights:
        result["wall_height_min"] = min(all_wall_heights)
        result["wall_height_max"] = max(all_wall_heights)
        result["wall_height_mean"] = np.mean(all_wall_heights)
        result["wall_height_std"] = np.std(all_wall_heights)
        result["wall_height_range"] = max(all_wall_heights) - min(all_wall_heights)
    else:
        result["wall_height_min"] = result["wall_height_max"] = 0
        result["wall_height_mean"] = result["wall_height_std"] = 0
        result["wall_height_range"] = 0

    # Wall count stats per room
    wc = [rd["wall_count"] for rd in result["room_details"]]
    if wc:
        result["wc_min"] = min(wc)
        result["wc_max"] = max(wc)
        result["wc_mean"] = np.mean(wc)

    # Per-room Y positions grouped by story (to check vertical spread within story)
    story_ys = defaultdict(list)
    for rd in result["room_details"]:
        story_ys[rd["story"]].append(rd["centroid"][1])
    result["story_y_ranges"] = {}
    for s, ys in sorted(story_ys.items()):
        result["story_y_ranges"][s] = (min(ys), max(ys), max(ys) - min(ys))

    return result


def get_reconciliation(uuid):
    """Get reconciliation data for a building."""
    recon_path = BASE / uuid / "reconciled.json"
    if not recon_path.exists():
        return None
    with open(recon_path) as f:
        data = json.load(f)
    return data.get("reconciliation", {})


def sep(char="=", length=130):
    print(char * length)


def main():
    sep()
    print("BUILDING COMPARISON: PROBLEM (RED) vs REFERENCE (GREEN/YELLOW)")
    sep()

    problem_results = {}
    reference_results = {}

    for uuid, name in PROBLEM_BUILDINGS.items():
        r = analyze_building(uuid)
        if r:
            r["name"] = name
            problem_results[uuid] = r

    for uuid, name in REFERENCE_BUILDINGS.items():
        r = analyze_building(uuid)
        if r:
            r["name"] = name
            reference_results[uuid] = r

    # =========================================================================
    # SECTION 1: Summary comparison table
    # =========================================================================
    sep()
    print("SECTION 1: SPATIAL SUMMARY TABLE")
    sep()

    header = (
        f"{'Building':<28} {'Rooms':>5} {'Walls':>5} {'Stry':>4} "
        f"{'X-spr':>7} {'Y-spr':>7} {'Z-spr':>7} "
        f"{'WallMaxD':>8} {'RmMaxD':>7} "
        f"{'WH-mean':>7} {'WH-std':>7} {'WH-rng':>7} "
        f"{'WC-min':>6} {'WC-max':>6}"
    )
    print(header)
    print("-" * 130)

    print("--- PROBLEM BUILDINGS ---")
    for _uuid, r in sorted(problem_results.items(), key=lambda x: x[1]["name"]):
        print(
            f"{r['name']:<28} {r['num_rooms']:>5} {r['num_walls_global']:>5} "
            f"{len(r['stories']):>4} "
            f"{r['wall_x_spread']:>7.2f} {r['wall_y_spread']:>7.2f} "
            f"{r['wall_z_spread']:>7.2f} "
            f"{r['wall_max_dist']:>8.2f} "
            f"{r.get('room_max_dist_from_bldg_centroid', 0):>7.2f} "
            f"{r.get('wall_height_mean', 0):>7.2f} {r.get('wall_height_std', 0):>7.3f} "
            f"{r.get('wall_height_range', 0):>7.2f} "
            f"{r.get('wc_min', 0):>6} {r.get('wc_max', 0):>6}"
        )

    print("\n--- REFERENCE BUILDINGS ---")
    for _uuid, r in sorted(reference_results.items(), key=lambda x: x[1]["name"]):
        print(
            f"{r['name']:<28} {r['num_rooms']:>5} {r['num_walls_global']:>5} "
            f"{len(r['stories']):>4} "
            f"{r['wall_x_spread']:>7.2f} {r['wall_y_spread']:>7.2f} "
            f"{r['wall_z_spread']:>7.2f} "
            f"{r['wall_max_dist']:>8.2f} "
            f"{r.get('room_max_dist_from_bldg_centroid', 0):>7.2f} "
            f"{r.get('wall_height_mean', 0):>7.2f} {r.get('wall_height_std', 0):>7.3f} "
            f"{r.get('wall_height_range', 0):>7.2f} "
            f"{r.get('wc_min', 0):>6} {r.get('wc_max', 0):>6}"
        )

    # =========================================================================
    # SECTION 2: Aggregate statistics
    # =========================================================================
    sep()
    print("SECTION 2: AGGREGATE STATISTICS")
    sep()

    def agg(results, field):
        vals = [
            r[field] for r in results.values() if field in r and r[field] is not None
        ]
        if not vals:
            return (0, 0, 0, 0)
        return (np.mean(vals), np.median(vals), min(vals), max(vals))

    metrics = [
        ("num_rooms", "Room count"),
        ("num_walls_global", "Wall count (global)"),
        ("wall_x_spread", "X spread (m)"),
        ("wall_y_spread", "Y spread (m) [vertical]"),
        ("wall_z_spread", "Z spread (m)"),
        ("wall_max_dist", "Wall max dist from centroid (m)"),
        ("wall_mean_dist", "Wall mean dist from centroid (m)"),
        ("room_max_dist_from_bldg_centroid", "Room max dist from bldg centroid (m)"),
        ("room_mean_dist_from_bldg_centroid", "Room mean dist from bldg centroid (m)"),
        ("wall_height_mean", "Wall height mean (m)"),
        ("wall_height_std", "Wall height std (m)"),
        ("wall_height_range", "Wall height range (m)"),
    ]

    print(
        f"{'Metric':<42} {'PROB mean':>10} {'PROB med':>10} {'REF mean':>10} "
        f"{'REF med':>10} {'Diff':>10}"
    )
    print("-" * 95)
    for field, label in metrics:
        p = agg(problem_results, field)
        r = agg(reference_results, field)
        diff = p[0] - r[0]
        print(
            f"{label:<42} {p[0]:>10.3f} {p[1]:>10.3f} {r[0]:>10.3f} {r[1]:>10.3f} "
            f"{diff:>+10.3f}"
        )

    # =========================================================================
    # SECTION 3: Story distribution
    # =========================================================================
    sep()
    print("SECTION 3: STORY DISTRIBUTION & VERTICAL RANGES PER STORY")
    sep()

    print("\n--- PROBLEM BUILDINGS ---")
    for _uuid, r in sorted(problem_results.items(), key=lambda x: x[1]["name"]):
        stories = dict(sorted(r["stories"].items()))
        y_ranges = r.get("story_y_ranges", {})
        print(f"  {r['name']:<28} stories: {dict(stories)}")
        for s, (ymin, ymax, yrange) in sorted(y_ranges.items()):
            print(
                f"    story {s}: Y range [{ymin:.2f}, {ymax:.2f}] spread={yrange:.2f}m"
            )

    print("\n--- REFERENCE BUILDINGS ---")
    for _uuid, r in sorted(reference_results.items(), key=lambda x: x[1]["name"]):
        stories = dict(sorted(r["stories"].items()))
        y_ranges = r.get("story_y_ranges", {})
        print(f"  {r['name']:<28} stories: {dict(stories)}")
        for s, (ymin, ymax, yrange) in sorted(y_ranges.items()):
            print(
                f"    story {s}: Y range [{ymin:.2f}, {ymax:.2f}] spread={yrange:.2f}m"
            )

    # =========================================================================
    # SECTION 4: Reconciliation flags
    # =========================================================================
    sep()
    print("SECTION 4: RECONCILIATION FLAGS")
    sep()

    print("\n--- PROBLEM BUILDINGS ---")
    for uuid, name in sorted(PROBLEM_BUILDINGS.items(), key=lambda x: x[1]):
        recon = get_reconciliation(uuid)
        if recon:
            print(f"\n  {name} ({uuid[:8]})")
            print(f"    Classification: {recon.get('classification')}")
            print(
                f"    Walls: {recon.get('wall_count_merged')} merged / "
                f"{recon.get('wall_count_scan')} per-scan "
                f"({recon.get('wall_reduction_pct')}% reduction)"
            )
            print(f"    Width delta: {recon.get('mean_width_delta_cm')} cm")
            print(f"    Height delta: {recon.get('mean_height_delta_cm')} cm")
            print(
                f"    Floor gap: {recon.get('floor_gap_median_cm')} cm "
                f"({recon.get('floor_gap_measurements')} measurements)"
            )
            print(f"    Flags: {recon.get('flags', [])}")

    print("\n--- REFERENCE BUILDINGS ---")
    for uuid, name in sorted(REFERENCE_BUILDINGS.items(), key=lambda x: x[1]):
        recon = get_reconciliation(uuid)
        if recon:
            print(
                f"  {name}: class={recon.get('classification')}, "
                f"ht_delta={recon.get('mean_height_delta_cm')}cm, "
                f"wd_delta={recon.get('mean_width_delta_cm')}cm, "
                f"gap={recon.get('floor_gap_median_cm')}cm, "
                f"flags={recon.get('flags', [])}"
            )

    # =========================================================================
    # SECTION 5: Position anomalies
    # =========================================================================
    sep()
    print("SECTION 5: POSITION ANOMALIES IN PROBLEM BUILDINGS")
    sep()

    print("\n--- Rooms far from building centroid (>10m) ---")
    any_found = False
    for _uuid, r in sorted(problem_results.items(), key=lambda x: x[1]["name"]):
        for rd, d in r.get("rooms_far_from_centroid", []):
            any_found = True
            print(
                f"  {r['name']} room[{rd['index']}] story={rd['story']}: "
                f"centroid=({rd['centroid'][0]:.2f}, {rd['centroid'][1]:.2f}, "
                f"{rd['centroid'][2]:.2f}) "
                f"dist={d:.2f}m"
            )
    if not any_found:
        print("  None found.")

    print("\n--- Statistical outlier rooms (>2 std from building centroid) ---")
    any_found = False
    for _uuid, r in sorted(problem_results.items(), key=lambda x: x[1]["name"]):
        for rd, d in r.get("room_outliers_2std", []):
            any_found = True
            print(
                f"  {r['name']} room[{rd['index']}] story={rd['story']}: "
                f"centroid=({rd['centroid'][0]:.2f}, {rd['centroid'][1]:.2f}, "
                f"{rd['centroid'][2]:.2f}) "
                f"dist={d:.2f}m walls={rd['wall_count']}"
            )
    if not any_found:
        print("  None found.")

    # Also check reference buildings for comparison
    print("\n--- Statistical outlier rooms in REFERENCE buildings ---")
    any_found = False
    for _uuid, r in sorted(reference_results.items(), key=lambda x: x[1]["name"]):
        for rd, d in r.get("room_outliers_2std", []):
            any_found = True
            print(
                f"  {r['name']} room[{rd['index']}] story={rd['story']}: "
                f"centroid=({rd['centroid'][0]:.2f}, {rd['centroid'][1]:.2f}, "
                f"{rd['centroid'][2]:.2f}) "
                f"dist={d:.2f}m walls={rd['wall_count']}"
            )
    if not any_found:
        print("  None found.")

    # =========================================================================
    # SECTION 6: Per-room detail for problem buildings
    # =========================================================================
    sep()
    print("SECTION 6: ROOM-LEVEL DETAILS (PROBLEM BUILDINGS)")
    sep()

    for _uuid, r in sorted(problem_results.items(), key=lambda x: x[1]["name"]):
        bc = r.get("building_centroid", (0, 0, 0))
        print(
            f"\n  {r['name']} ({r['num_rooms']} rooms, bldg centroid=({bc[0]:.2f}, "
            f"{bc[1]:.2f}, {bc[2]:.2f}))"
        )
        print(
            f"    {'Rm':>3} {'Stry':>4} {'Walls':>5} {'CentX':>8} {'CentY':>8} "
            f"{'CentZ':>8} "
            f"{'DistBC':>7} {'Xext':>6} {'Yext':>6} {'Zext':>6} {'WHmean':>6} "
            f"{'WHstd':>6}"
        )
        for rd in r["room_details"]:
            dist = math.sqrt(
                (rd["centroid"][0] - bc[0]) ** 2
                + (rd["centroid"][1] - bc[1]) ** 2
                + (rd["centroid"][2] - bc[2]) ** 2
            )
            marker = " <<<" if dist > 10 else ""
            print(
                f"    {rd['index']:>3} {rd['story']:>4} {rd['wall_count']:>5} "
                f"{rd['centroid'][0]:>8.2f} {rd['centroid'][1]:>8.2f} "
                f"{rd['centroid'][2]:>8.2f} "
                f"{dist:>7.2f} {rd['x_extent']:>6.2f} {rd['y_extent']:>6.2f} "
                f"{rd['z_extent']:>6.2f} "
                f"{rd['wall_height_mean']:>6.2f} {rd['wall_height_std']:>6.3f}{marker}"
            )

    # =========================================================================
    # SECTION 7: Vertical spread comparison (Y dimension)
    # =========================================================================
    sep()
    print("SECTION 7: Y-SPREAD (VERTICAL) COMPARISON")
    sep()

    prob_y = [r["wall_y_spread"] for r in problem_results.values()]
    ref_y = [r["wall_y_spread"] for r in reference_results.values()]
    print("\n  Problem buildings Y-spread:")
    print(
        f"    mean={np.mean(prob_y):.3f}m, median={np.median(prob_y):.3f}m, "
        f"range=[{min(prob_y):.3f}, {max(prob_y):.3f}]"
    )
    for _uuid, r in sorted(
        problem_results.items(), key=lambda x: x[1]["wall_y_spread"], reverse=True
    ):
        print(
            f"    {r['name']:<28} Y-spread={r['wall_y_spread']:.3f}m  "
            f"Y-range=[{r.get('wall_y_range', (0, 0))[0]:.2f}, "
            f"{r.get('wall_y_range', (0, 0))[1]:.2f}]"
        )

    print("\n  Reference buildings Y-spread:")
    print(
        f"    mean={np.mean(ref_y):.3f}m, median={np.median(ref_y):.3f}m, "
        f"range=[{min(ref_y):.3f}, {max(ref_y):.3f}]"
    )
    for _uuid, r in sorted(
        reference_results.items(), key=lambda x: x[1]["wall_y_spread"], reverse=True
    ):
        print(
            f"    {r['name']:<28} Y-spread={r['wall_y_spread']:.3f}m  "
            f"Y-range=[{r.get('wall_y_range', (0, 0))[0]:.2f}, "
            f"{r.get('wall_y_range', (0, 0))[1]:.2f}]"
        )

    # =========================================================================
    # SECTION 8: DIAGNOSTIC SUMMARY
    # =========================================================================
    sep()
    print("SECTION 8: DIAGNOSTIC SUMMARY")
    sep()

    print("\nKey differences between PROBLEM and REFERENCE buildings:\n")

    def compare(label, prob_vals, ref_vals):
        pm, rm = np.mean(prob_vals), np.mean(ref_vals)
        ratio = pm / rm if rm > 0.001 else float("inf")
        print(f"  {label}:")
        print(
            f"    Problem: mean={pm:.3f}  Reference: mean={rm:.3f}  Ratio={ratio:.2f}x"
        )

    compare(
        "1. Room count",
        [r["num_rooms"] for r in problem_results.values()],
        [r["num_rooms"] for r in reference_results.values()],
    )

    compare(
        "2. Wall count",
        [r["num_walls_global"] for r in problem_results.values()],
        [r["num_walls_global"] for r in reference_results.values()],
    )

    compare(
        "3. Number of stories",
        [len(r["stories"]) for r in problem_results.values()],
        [len(r["stories"]) for r in reference_results.values()],
    )

    compare(
        "4. X spread (horizontal)",
        [r["wall_x_spread"] for r in problem_results.values()],
        [r["wall_x_spread"] for r in reference_results.values()],
    )

    compare(
        "5. Y spread (vertical)",
        [r["wall_y_spread"] for r in problem_results.values()],
        [r["wall_y_spread"] for r in reference_results.values()],
    )

    compare(
        "6. Z spread (depth)",
        [r["wall_z_spread"] for r in problem_results.values()],
        [r["wall_z_spread"] for r in reference_results.values()],
    )

    compare(
        "7. Max wall dist from centroid",
        [r["wall_max_dist"] for r in problem_results.values()],
        [r["wall_max_dist"] for r in reference_results.values()],
    )

    compare(
        "8. Wall height std (uniformity)",
        [r["wall_height_std"] for r in problem_results.values()],
        [r["wall_height_std"] for r in reference_results.values()],
    )

    compare(
        "9. Wall height range",
        [r["wall_height_range"] for r in problem_results.values()],
        [r["wall_height_range"] for r in reference_results.values()],
    )

    compare(
        "10. Max room-centroid distance from building centroid",
        [r["room_max_dist_from_bldg_centroid"] for r in problem_results.values()],
        [r["room_max_dist_from_bldg_centroid"] for r in reference_results.values()],
    )

    # Multi-story analysis
    print("\n  11. MULTI-STORY ANALYSIS:")
    prob_multi = sum(1 for r in problem_results.values() if len(r["stories"]) > 1)
    ref_multi = sum(1 for r in reference_results.values() if len(r["stories"]) > 1)
    print(
        f"    Problem: {prob_multi}/{len(problem_results)} multi-story "
        f"({prob_multi / len(problem_results) * 100:.0f}%)"
    )
    print(
        f"    Reference: {ref_multi}/{len(reference_results)} multi-story "
        f"({ref_multi / len(reference_results) * 100:.0f}%)"
    )

    # Flag analysis
    print("\n  12. RECONCILIATION FLAG ANALYSIS:")
    prob_flags = defaultdict(int)
    ref_flags = defaultdict(int)
    for uuid in PROBLEM_BUILDINGS:
        recon = get_reconciliation(uuid)
        if recon:
            for f in recon.get("flags", []):
                prob_flags[f] += 1
    for uuid in REFERENCE_BUILDINGS:
        recon = get_reconciliation(uuid)
        if recon:
            for f in recon.get("flags", []):
                ref_flags[f] += 1
    print("    Problem flags:")
    for f, c in sorted(prob_flags.items(), key=lambda x: -x[1]):
        print(f"      {f}: {c}/{len(PROBLEM_BUILDINGS)}")
    print("    Reference flags:")
    for f, c in sorted(ref_flags.items(), key=lambda x: -x[1]):
        print(f"      {f}: {c}/{len(REFERENCE_BUILDINGS)}")


if __name__ == "__main__":
    main()
