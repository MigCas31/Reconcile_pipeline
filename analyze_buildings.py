#!/usr/bin/env python3
"""Analyze merged.json files for 8 problem buildings + 3 control buildings."""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = str(Path(__file__).parent / "pipeline-outputs")

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

problem_uuids = set(PROBLEM_BUILDINGS.keys())

# Pick 3 random "good" buildings
all_uuids = [
    d
    for d in os.listdir(BASE)
    if os.path.isdir(os.path.join(BASE, d)) and d not in problem_uuids
]
random.seed(42)
control_uuids = random.sample(all_uuids, 3)

# Try to get names from datapack.json
CONTROL_BUILDINGS = {}
for uuid in control_uuids:
    dp = os.path.join(BASE, uuid, "datapack.json")
    name = uuid[:12]
    if os.path.exists(dp):
        try:
            with open(dp) as f:
                d = json.load(f)
            name = (
                d.get("address", {}).get("street", "") or d.get("name", "") or uuid[:12]
            )
        except Exception:
            pass
    CONTROL_BUILDINGS[uuid] = f"CONTROL: {name}"


def parse_transform(t):
    """
    Parse a flat 16-element list into a 4x4 matrix (column-major as used by
    RoomPlan).
    """
    if not t or len(t) != 16:
        return None
    return np.array(t).reshape(4, 4, order="F")  # column-major


def extract_position(t):
    """Extract translation (x, y, z) from flat 16-element transform."""
    if not t or len(t) != 16:
        return None
    # In column-major 4x4, translation is elements [12, 13, 14]
    return np.array([t[12], t[13], t[14]])


def extract_scale(mat):
    """Extract scale factors from 4x4 matrix."""
    if mat is None:
        return None
    sx = np.linalg.norm(mat[:3, 0])
    sy = np.linalg.norm(mat[:3, 1])
    sz = np.linalg.norm(mat[:3, 2])
    return np.array([sx, sy, sz])


def analyze_building(uuid, name):
    """Analyze a single building's merged.json."""
    path = os.path.join(BASE, uuid, "merged.json")
    if not os.path.exists(path):
        return {"name": name, "uuid": uuid, "error": "merged.json not found"}

    with open(path) as f:
        data = json.load(f)

    result = {"name": name, "uuid": uuid, "issues": []}

    # --- 1. Room-level analysis ---
    rooms = data.get("rooms", [])
    result["num_rooms"] = len(rooms)

    stories = defaultdict(list)
    room_positions = []  # (story, y_pos, room_idx, ref_origin_pos)
    ref_origins = []
    room_wall_counts = []
    room_floor_counts = []

    for i, room in enumerate(rooms):
        story = room.get("story", -1)
        ref_t = room.get("referenceOriginTransform")
        ref_pos = extract_position(ref_t)

        if ref_pos is not None:
            ref_origins.append(ref_pos)
            stories[story].append((i, ref_pos))
            room_positions.append((story, ref_pos[1], i, ref_pos))

        # Count walls/floors per room
        room_wall_counts.append(len(room.get("walls", [])))
        room_floor_counts.append(len(room.get("floors", [])))

        # Analyze walls within room
        for _w_idx, wall in enumerate(room.get("walls", [])):
            wt = wall.get("transform")
            w_pos = extract_position(wt)
            w_dim = wall.get("dimensions", [0, 0, 0])

    result["stories"] = {s: len(v) for s, v in sorted(stories.items())}

    # --- 2. referenceOriginTransform analysis ---
    if ref_origins:
        ref_arr = np.array(ref_origins)
        result["ref_origin_stats"] = {
            "mean": ref_arr.mean(axis=0).tolist(),
            "std": ref_arr.std(axis=0).tolist(),
            "min": ref_arr.min(axis=0).tolist(),
            "max": ref_arr.max(axis=0).tolist(),
            "range": (ref_arr.max(axis=0) - ref_arr.min(axis=0)).tolist(),
        }

        # Check for outlier rooms (position > 3 std from mean)
        mean = ref_arr.mean(axis=0)
        std = ref_arr.std(axis=0)
        for i, pos in enumerate(ref_origins):
            dist = np.abs(pos - mean)
            if std.max() > 0:
                z_scores = dist / (std + 1e-10)
                if z_scores.max() > 3:
                    axis = ["X", "Y", "Z"][z_scores.argmax()]
                    result["issues"].append(
                        f"Room {i} refOrigin outlier: {axis}-axis "
                        f"z-score={z_scores.max():.1f}, pos={pos.tolist()}"
                    )

    # --- 3. Story overlap / Y-coordinate analysis ---
    if len(stories) > 1:
        story_y_ranges = {}
        for s, room_list in sorted(stories.items()):
            ys = [pos[1] for _, pos in room_list]
            story_y_ranges[s] = (min(ys), max(ys), np.mean(ys))

        result["story_y_ranges"] = {
            s: {"min_y": v[0], "max_y": v[1], "mean_y": v[2]}
            for s, v in story_y_ranges.items()
        }

        # Check for overlapping Y ranges between stories
        sorted_stories = sorted(story_y_ranges.keys())
        for i_s in range(len(sorted_stories)):
            for j_s in range(i_s + 1, len(sorted_stories)):
                s1, s2 = sorted_stories[i_s], sorted_stories[j_s]
                r1 = story_y_ranges[s1]
                r2 = story_y_ranges[s2]
                # Check overlap
                overlap = min(r1[1], r2[1]) - max(r1[0], r2[0])
                if overlap > 0:
                    result["issues"].append(
                        f"Story {s1} and {s2} Y-ranges OVERLAP by {overlap:.3f}m "
                        f"(story{s1}: [{r1[0]:.3f}, {r1[1]:.3f}], story{s2}: "
                        f"[{r2[0]:.3f}, {r2[1]:.3f}])"
                    )
                else:
                    gap = -overlap
                    # Typical floor height is ~2.5-3m. Gap < 1m or > 5m is suspicious
                    if gap < 0.5:
                        result["issues"].append(
                            f"Story {s1} to {s2}: suspiciously small Y-gap of "
                            f"{gap:.3f}m"
                        )
                    elif gap > 6.0:
                        result["issues"].append(
                            f"Story {s1} to {s2}: suspiciously large Y-gap of "
                            f"{gap:.3f}m"
                        )

        # Check if higher stories actually have higher Y (they should)
        for i_s in range(len(sorted_stories) - 1):
            s_lo = sorted_stories[i_s]
            s_hi = sorted_stories[i_s + 1]
            if story_y_ranges[s_lo][2] > story_y_ranges[s_hi][2]:
                result["issues"].append(
                    f"INVERTED stories: story {s_lo} "
                    f"mean_y={story_y_ranges[s_lo][2]:.3f} > "
                    f"story {s_hi} mean_y={story_y_ranges[s_hi][2]:.3f}"
                )

    # --- 4. Top-level wall analysis ---
    walls = data.get("walls", [])
    result["num_walls"] = len(walls)
    wall_positions = []
    wall_dimensions = []

    for w in walls:
        wt = w.get("transform")
        w_pos = extract_position(wt)
        w_dim = w.get("dimensions", [0, 0, 0])
        w_story = w.get("story", -1)

        if w_pos is not None:
            wall_positions.append((w_story, w_pos))
            wall_dimensions.append(w_dim)

            # Check for extreme wall dimensions
            if w_dim and max(w_dim[:2]) > 20:
                result["issues"].append(
                    f"Wall {w.get('identifier', '?')[:8]}: extreme dimension {w_dim} "
                    f"(story {w_story})"
                )

            # Check for walls at extreme positions
            if abs(w_pos[0]) > 50 or abs(w_pos[1]) > 20 or abs(w_pos[2]) > 50:
                result["issues"].append(
                    f"Wall {w.get('identifier', '?')[:8]}: extreme position "
                    f"{w_pos.tolist()} (story {w_story})"
                )

    if wall_positions:
        wp_arr = np.array([p for _, p in wall_positions])
        result["wall_pos_stats"] = {
            "mean": wp_arr.mean(axis=0).tolist(),
            "std": wp_arr.std(axis=0).tolist(),
            "range": (wp_arr.max(axis=0) - wp_arr.min(axis=0)).tolist(),
        }

        # Check for wall position outliers
        mean = wp_arr.mean(axis=0)
        std = wp_arr.std(axis=0)
        outlier_count = 0
        for _s, pos in wall_positions:
            z = np.abs(pos - mean) / (std + 1e-10)
            if z.max() > 4:
                outlier_count += 1
        if outlier_count > 0:
            result["issues"].append(
                f"{outlier_count} wall position outliers (>4 std from mean)"
            )

    if wall_dimensions:
        wd_arr = np.array(wall_dimensions)
        result["wall_dim_stats"] = {
            "mean": wd_arr.mean(axis=0).tolist(),
            "max": wd_arr.max(axis=0).tolist(),
        }

    # --- 5. Floor analysis ---
    floors = data.get("floors", [])
    result["num_floors"] = len(floors)
    floor_positions = []
    for fl in floors:
        ft = fl.get("transform")
        f_pos = extract_position(ft)
        if f_pos is not None:
            floor_positions.append((fl.get("story", -1), f_pos))

    if floor_positions:
        fp_arr = np.array([p for _, p in floor_positions])
        result["floor_pos_stats"] = {
            "mean": fp_arr.mean(axis=0).tolist(),
            "range": (fp_arr.max(axis=0) - fp_arr.min(axis=0)).tolist(),
        }

    # --- 6. Transform matrix sanity checks ---
    # Check all transforms for non-orthogonal rotation matrices or bad scale
    bad_transforms = 0
    for entity_type in ["walls", "doors", "windows", "floors"]:
        for item in data.get(entity_type, []):
            t = item.get("transform")
            mat = parse_transform(t)
            if mat is not None:
                scale = extract_scale(mat)
                # Scale should be ~1.0 for each axis
                if scale is not None and (np.abs(scale - 1.0) > 0.01).any():
                    bad_transforms += 1

    if bad_transforms > 0:
        result["issues"].append(f"{bad_transforms} transforms with non-unit scale")

    # --- 7. Spatial spread / bounding box ---
    all_positions = []
    for entity_type in ["walls", "doors", "windows", "floors"]:
        for item in data.get(entity_type, []):
            t = item.get("transform")
            pos = extract_position(t)
            if pos is not None:
                all_positions.append(pos)

    if all_positions:
        all_pos = np.array(all_positions)
        bbox = all_pos.max(axis=0) - all_pos.min(axis=0)
        result["bounding_box"] = bbox.tolist()
        result["centroid"] = all_pos.mean(axis=0).tolist()

        # A typical single house bbox should be < 30m in X/Z
        if bbox[0] > 30 or bbox[2] > 30:
            result["issues"].append(
                f"Very large spatial extent: bbox={bbox.tolist()} (X or Z > 30m)"
            )
        if bbox[1] > 10:
            result["issues"].append(f"Very tall building: Y-range={bbox[1]:.2f}m")

    # --- 8. Per-room referenceOriginTransform consistency ---
    if len(ref_origins) >= 2:
        # Check pairwise distances between room reference origins
        dists = []
        for i in range(len(ref_origins)):
            for j in range(i + 1, len(ref_origins)):
                d = np.linalg.norm(ref_origins[i] - ref_origins[j])
                dists.append(d)
        dists = np.array(dists)
        result["ref_origin_pairwise_dist"] = {
            "mean": float(dists.mean()),
            "max": float(dists.max()),
            "std": float(dists.std()),
        }

        # If some room origins are very far from others, that's suspicious
        if dists.max() > 30:
            result["issues"].append(
                f"Max pairwise refOrigin distance: {dists.max():.2f}m (>30m, rooms may "
                f"be misaligned)"
            )

    # --- 9. Check referenceOriginTransform rotation consistency ---
    # All rooms in the same scan session should share similar orientation
    ref_rotations = []
    for room in rooms:
        ref_t = room.get("referenceOriginTransform")
        mat = parse_transform(ref_t)
        if mat is not None:
            # Extract yaw angle from rotation matrix (around Y axis)
            # R = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
            yaw = np.arctan2(mat[0, 2], mat[0, 0])
            ref_rotations.append(yaw)

    if ref_rotations:
        yaws = np.array(ref_rotations)
        # Circular std for angles
        result["ref_yaw_range"] = float(yaws.max() - yaws.min())
        unique_yaws = np.unique(np.round(yaws, 2))
        result["unique_ref_yaws"] = len(unique_yaws)
        if len(unique_yaws) > 1:
            result["ref_yaw_values"] = unique_yaws.tolist()

    # --- 10. Rooms with zero walls or zero floors ---
    empty_rooms = sum(1 for r in rooms if not r.get("walls"))
    no_floor_rooms = sum(1 for r in rooms if not r.get("floors"))
    if empty_rooms:
        result["issues"].append(f"{empty_rooms} rooms with no walls")
    if no_floor_rooms:
        result["issues"].append(f"{no_floor_rooms} rooms with no floors")

    return result


def print_report(result):
    """Print a formatted report for one building."""
    name = result["name"]
    uuid = result["uuid"]
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"  UUID: {uuid}")
    print(f"{'=' * 80}")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    print(
        f"  Rooms: {result['num_rooms']}  |  Walls: {result['num_walls']}  |  Floors: "
        f"{result['num_floors']}"
    )
    print(f"  Stories: {result.get('stories', {})}")

    if "bounding_box" in result:
        bb = result["bounding_box"]
        print(f"  Bounding box (X,Y,Z): [{bb[0]:.2f}, {bb[1]:.2f}, {bb[2]:.2f}] m")

    if "centroid" in result:
        c = result["centroid"]
        print(f"  Centroid: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]")

    if "story_y_ranges" in result:
        print("  Story Y-ranges:")
        for s, v in sorted(result["story_y_ranges"].items()):
            print(
                f"    Story {s}: Y=[{v['min_y']:.3f}, {v['max_y']:.3f}], "
                f"mean={v['mean_y']:.3f}"
            )

    if "ref_origin_stats" in result:
        st = result["ref_origin_stats"]
        print(
            f"  RefOrigin range (X,Y,Z): [{st['range'][0]:.2f}, {st['range'][1]:.2f}, "
            f"{st['range'][2]:.2f}]"
        )
        print(
            f"  RefOrigin std (X,Y,Z): [{st['std'][0]:.2f}, {st['std'][1]:.2f}, "
            f"{st['std'][2]:.2f}]"
        )

    if "ref_origin_pairwise_dist" in result:
        d = result["ref_origin_pairwise_dist"]
        print(
            f"  RefOrigin pairwise dist: mean={d['mean']:.2f}, max={d['max']:.2f}, "
            f"std={d['std']:.2f}"
        )

    if "ref_yaw_range" in result:
        print(
            f"  RefOrigin yaw range: {np.degrees(result['ref_yaw_range']):.1f} deg, "
            f"{result['unique_ref_yaws']} unique orientations"
        )
        if "ref_yaw_values" in result:
            print(f"    Yaw values (rad): {result['ref_yaw_values']}")

    if "wall_pos_stats" in result:
        ws = result["wall_pos_stats"]
        print(
            f"  Wall position range (X,Y,Z): [{ws['range'][0]:.2f}, "
            f"{ws['range'][1]:.2f}, {ws['range'][2]:.2f}]"
        )

    if "wall_dim_stats" in result:
        wd = result["wall_dim_stats"]
        print(
            f"  Wall max dimensions: [{wd['max'][0]:.2f}, {wd['max'][1]:.2f}, "
            f"{wd['max'][2]:.2f}]"
        )

    issues = result.get("issues", [])
    if issues:
        print(f"\n  ** ISSUES ({len(issues)}) **")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("\n  No issues detected.")


# ============================================================
# MAIN
# ============================================================
print("=" * 80)
print("  BUILDING RECONSTRUCTION ANOMALY ANALYSIS")
print("=" * 80)

all_buildings = {**PROBLEM_BUILDINGS, **CONTROL_BUILDINGS}
results = {}

for uuid, name in all_buildings.items():
    results[uuid] = analyze_building(uuid, name)

# Print individual reports
print("\n\n" + "#" * 80)
print("  PROBLEM BUILDINGS")
print("#" * 80)
for uuid in PROBLEM_BUILDINGS:
    print_report(results[uuid])

print("\n\n" + "#" * 80)
print("  CONTROL BUILDINGS (for comparison)")
print("#" * 80)
for uuid in CONTROL_BUILDINGS:
    print_report(results[uuid])

# ============================================================
# COMPARATIVE SUMMARY
# ============================================================
print("\n\n" + "#" * 80)
print("  COMPARATIVE SUMMARY: PROBLEM vs CONTROL")
print("#" * 80)


def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


print(
    f"\n{'Building':<30} {'Rooms':>5} {'Walls':>5} {'Stories':>8} {'BBox X':>7} "
    f"{'BBox Y':>7} {'BBox Z':>7} {'RefDist':>8} {'Yaw Rng':>8} {'Issues':>6}"
)
print("-" * 110)

for uuid, name in {**PROBLEM_BUILDINGS, **CONTROL_BUILDINGS}.items():
    r = results[uuid]
    if "error" in r:
        print(f"{name[:30]:<30} ERROR")
        continue
    bb = r.get("bounding_box", [0, 0, 0])
    rd = safe_get(r, "ref_origin_pairwise_dist", "max", default=0)
    yr = np.degrees(r.get("ref_yaw_range", 0))
    issues = len(r.get("issues", []))
    label = "CTRL" if "CONTROL" in name else "PROB"
    print(
        f"[{label}] {name[:26]:<26} {r['num_rooms']:>5} {r['num_walls']:>5} "
        f"{r.get('stories', {})!s:>8} {bb[0]:>7.2f} {bb[1]:>7.2f} {bb[2]:>7.2f} "
        f"{rd:>8.2f} {yr:>7.1f}d {issues:>6}"
    )

# Final issue tally
print(f"\n{'Building':<30} Issues")
print("-" * 60)
for uuid, name in {**PROBLEM_BUILDINGS, **CONTROL_BUILDINGS}.items():
    r = results[uuid]
    issues = r.get("issues", [])
    label = "CTRL" if "CONTROL" in name else "PROB"
    print(f"[{label}] {name[:26]:<26} {len(issues):>3} issues")
    for iss in issues:
        print(f"      {iss}")
