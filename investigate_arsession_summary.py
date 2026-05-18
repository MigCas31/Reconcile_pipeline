#!/usr/bin/env python3
"""
Final comprehensive summary combining all findings.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

CACHE = Path(".scan-cache")
PIPELINE = Path("pipeline-outputs")


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


def rotation_angle_between(rot1, rot2):
    R = rot1 @ rot2.T
    trace = np.clip(np.trace(R), -1, 3)
    return math.degrees(math.acos(np.clip((trace - 1) / 2, -1, 1)))


def get_floor_yaws(uuid):
    """Get per-floor yaw values from room JSONs."""
    scan_dir = find_scan_dir(uuid)
    if not scan_dir:
        return {}
    with open(scan_dir / "data.json") as f:
        data = json.load(f)
    floors = data["homeMetadata"].get("floors", [])
    result = {}
    for fl in floors:
        fl_type = fl.get("floorType", "unknown")
        yaws = []
        y_vals = []
        for r in fl.get("rooms", []):
            rid = r.get("id")
            room_json = scan_dir / f"{rid}.json"
            if room_json.exists():
                with open(room_json) as f:
                    rd = json.load(f)
                rot_flat = rd.get("referenceOriginTransform")
                rot_mat = extract_rotation_matrix(rot_flat)
                trans = extract_translation(rot_flat)
                if rot_mat is not None:
                    yaws.append(yaw_from_rotation(rot_mat))
                if trans is not None:
                    y_vals.append(trans[1])
        if yaws:
            result[fl_type] = {
                "mean_yaw": float(np.mean(yaws)),
                "yaw_std": float(np.std(yaws)),
                "yaws": yaws,
                "mean_y": float(np.mean(y_vals)) if y_vals else None,
                "y_vals": y_vals,
                "num_rooms": len(fl.get("rooms", [])),
                "created": fl.get("created"),
            }
    return result


def main():
    os.chdir(Path(__file__).parent)

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

    print("=" * 120)
    print("COMPREHENSIVE ARSESSION INVESTIGATION - FINAL REPORT")
    print("=" * 120)
    print()
    print("INVESTIGATION METHODOLOGY")
    print("-" * 60)
    print("1. Extracted referenceOriginTransform from each room's JSON in .scan-cache")
    print("2. Decomposed into rotation (yaw) and translation components")
    print("3. Compared rotations WITHIN each floor (same ARSession = same yaw)")
    print(
        "4. Compared rotations ACROSS floors (different floor = may have different "
        "session)"
    )
    print("5. Checked for Y-offset outliers (rooms placed at wrong height)")
    print("6. Analyzed timestamp gaps for correlation with session changes")
    print("7. Compared against 10 randomly selected control buildings")
    print()

    print("=" * 120)
    print("BUILDING-BY-BUILDING FINDINGS")
    print("=" * 120)

    findings = []

    for uuid, name in TARGETS.items():
        floor_data = get_floor_yaws(uuid)
        if not floor_data:
            continue

        print(f"\n{'=' * 100}")
        print(f"  {name} ({uuid})")
        print(f"{'=' * 100}")

        # Classify the type of issue
        issues = []

        # Check intra-floor yaw variation (session restart within a floor)
        for ft, fd in floor_data.items():
            if fd["yaw_std"] > 1.0:
                max_yaw_diff = max(fd["yaws"]) - min(fd["yaws"])
                issues.append(
                    f"INTRA-FLOOR SESSION RESTART on {ft} floor (yaw "
                    f"spread={max_yaw_diff:.1f}deg)"
                )

        # Check cross-floor yaw variation
        floor_types = list(floor_data.keys())
        if len(floor_types) >= 2:
            cross_floor_yaw_diffs = []
            for i in range(len(floor_types)):
                for j in range(i + 1, len(floor_types)):
                    ft1, ft2 = floor_types[i], floor_types[j]
                    yaw_diff = abs(
                        floor_data[ft1]["mean_yaw"] - floor_data[ft2]["mean_yaw"]
                    )
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    if yaw_diff > 1.0:
                        cross_floor_yaw_diffs.append((ft1, ft2, yaw_diff))
                        issues.append(
                            f"CROSS-FLOOR SESSION RESTART: {ft1}->{ft2} (yaw "
                            f"diff={yaw_diff:.1f}deg)"
                        )

        # Check Y-offset outliers within a floor
        for ft, fd in floor_data.items():
            if fd["y_vals"]:
                mean_y = np.mean(fd["y_vals"])
                for _i, y in enumerate(fd["y_vals"]):
                    if abs(y - mean_y) > 1.0:
                        issues.append(
                            f"Y-OFFSET OUTLIER on {ft} floor (room at y={y:.2f}m vs "
                            f"mean={mean_y:.2f}m)"
                        )
                        break  # One note per floor is enough

        # Print floor details
        for ft, fd in sorted(floor_data.items(), key=lambda x: x[1].get("created", 0)):
            yaw_flag = " *** MULTI-SESSION ***" if fd["yaw_std"] > 1.0 else ""
            y_outlier = (
                any(abs(y - np.mean(fd["y_vals"])) > 1.0 for y in fd["y_vals"])
                if fd["y_vals"]
                else False
            )
            y_flag = " *** Y-OUTLIER ***" if y_outlier else ""
            print(
                f"    {ft:>10}: {fd['num_rooms']:>2} rooms | "
                f"yaw={fd['mean_yaw']:>7.1f}deg (std={fd['yaw_std']:.3f}) | "
                f"mean_y={fd['mean_y']:>6.3f}m{yaw_flag}{y_flag}"
            )

        # Print issues
        if issues:
            print("\n    ISSUES DETECTED:")
            for iss in issues:
                print(f"      - {iss}")
        else:
            print("\n    No anomalies detected in rotation or Y-offset data.")

        findings.append({"name": name, "uuid": uuid, "issues": issues})

    # === CLASSIFICATION ===
    print()
    print("=" * 120)
    print("CLASSIFICATION OF ALL 8 BUILDINGS")
    print("=" * 120)
    print()
    print(
        f"{'Building':<25} {'Intra-Floor':>12} {'Cross-Floor':>12} {'Y-Outlier':>10} "
        f"{'Category':<40}"
    )
    print("-" * 120)

    for f in findings:
        intra = "YES" if any("INTRA-FLOOR" in i for i in f["issues"]) else "no"
        cross = "YES" if any("CROSS-FLOOR" in i for i in f["issues"]) else "no"
        y_out = "YES" if any("Y-OFFSET" in i for i in f["issues"]) else "no"

        if intra == "YES":
            cat = "SESSION RESTART WITHIN FLOOR"
        elif cross == "YES" and y_out == "YES":
            cat = "CROSS-FLOOR SESSION RESTART + Y-OUTLIER"
        elif cross == "YES":
            cat = "CROSS-FLOOR SESSION RESTART"
        elif y_out == "YES":
            cat = "Y-OFFSET ANOMALY ONLY"
        else:
            cat = "NO ANOMALY DETECTED"

        print(f"{f['name']:<25} {intra:>12} {cross:>12} {y_out:>10} {cat:<40}")

    print()
    print("=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)
    print("""
    1. INTRA-FLOOR SESSION RESTARTS (3 buildings):
       - Bredballe Byvej 63: 'Kokken 1' on first floor has a 260.5deg yaw shift from
         the other 12 rooms. This room was scanned first, then a 32-minute gap before
         the remaining rooms were scanned in a different ARSession.
       - Lucernevej 23: 3 rooms (Gang, Vaerelse 1, Sovevarelse) have ~168deg yaw
         shift from the other 9 rooms. These 3 were scanned mid-way through the
         session but with a completely different coordinate frame.
       - Morelvej 68: 'Sovevarelse 2' has a 39.5deg yaw shift from the other 9 rooms.
         This was the last room scanned (4.6min gap before it).

    2. CROSS-FLOOR SESSION RESTARTS (4 additional buildings):
       - Gerskov Bygade 24: First floor yaw=82.5deg, ground floor yaw=72.9deg (9.6deg
       diff).
         Each floor was scanned in a separate ARSession.
       - Gultvedgyden 6: Ground/first/second all at ~107.6deg, but basement at 26.2deg
         (81.5deg difference). The basement was scanned in a completely different
         session.
       - Hesselvænget 2: Basement yaw=89.8deg, ground+first yaw=-45.3deg (135deg diff).
         The basement used a different ARSession than the upper floors.
       - Kastanievej 6: Ground yaw=34.4deg, first yaw=174.0deg (139.6deg diff).
         Each floor was a different ARSession. Additionally, the first floor has
         rooms at two distinct Y-levels (-0.3m and -2.8m), suggesting mixed floor data
         or an additional sub-session.

    3. NO ANOMALY DETECTED (1 building):
       - Pilevej 30: Both floors have consistent yaw (95.9deg) and no Y-outliers.
         The scan appears to have been done in a single continuous ARSession across
         both floors. If there are reconstruction issues, they are NOT caused by
         session restarts.

    4. MERGED.JSON COMPENSATION:
       The merging pipeline successfully compensated for all session restarts -
       merged.json shows consistent referenceOriginTransform per story (max rotation
       diff < 0.05deg). However, the underlying room scan data may still have
       quality issues from the coordinate frame discontinuities.

    5. CONTROL BUILDINGS (10 tested):
       ZERO control buildings showed any session restart evidence. All had yaw
       standard deviations of < 0.06deg within and across floors.
""")


if __name__ == "__main__":
    main()
