#!/usr/bin/env python3
"""
Deeper analysis: per-story wall/floor Y positions, room-level geometry, cross-floor
gaps.
"""

import json
import os
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

CONTROL_UUIDS = {
    "85bc077a-8953-40e3-87b4-9c42f8727e6e": "CTRL: 85bc077a",
    "945b16b9-0ebf-409e-af08-aab7672801d2": "CTRL: 945b16b9",
    "d7f1aa19-5ca9-4e1b-ac1a-c86d63f9ede7": "CTRL: d7f1aa19",
}


def pos(t):
    """Extract position from flat 16-element transform."""
    if not t or len(t) != 16:
        return None
    return np.array([t[12], t[13], t[14]])


def analyze(uuid, name):
    path = os.path.join(BASE, uuid, "merged.json")
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'=' * 80}")
    print(f"  {name} ({uuid})")
    print(f"{'=' * 80}")

    rooms = data.get("rooms", [])
    stories_set = sorted(set(r.get("story", -1) for r in rooms))
    print(
        f"  {len(rooms)} rooms, {len(data.get('walls', []))} top-level walls, stories: "
        f"{stories_set}"
    )

    # Per-room analysis: wall Y positions grouped by story
    print("\n  --- Per-Room Wall Y-coordinate Analysis (grouped by story) ---")
    story_wall_ys = defaultdict(list)
    story_floor_ys = defaultdict(list)
    story_room_centroids = defaultdict(list)

    for i, room in enumerate(rooms):
        story = room.get("story", -1)
        room_walls = room.get("walls", [])
        room_floors = room.get("floors", [])

        wall_ys = []
        wall_xzs = []
        for w in room_walls:
            p = pos(w.get("transform"))
            if p is not None:
                wall_ys.append(p[1])
                wall_xzs.append((p[0], p[2]))
                story_wall_ys[story].append(p[1])

        floor_ys = []
        for fl in room_floors:
            p = pos(fl.get("transform"))
            if p is not None:
                floor_ys.append(p[1])
                story_floor_ys[story].append(p[1])

        if wall_ys:
            centroid_y = np.mean(wall_ys)
            story_room_centroids[story].append(
                (
                    i,
                    centroid_y,
                    min(wall_ys),
                    max(wall_ys),
                )
            )

    # Print per-story summary
    for s in stories_set:
        ys = story_wall_ys.get(s, [])
        fys = story_floor_ys.get(s, [])
        centroids = story_room_centroids.get(s, [])
        if ys:
            print(f"\n  Story {s}:")
            print(
                f"    Wall Y: min={min(ys):.3f}, max={max(ys):.3f}, "
                f"mean={np.mean(ys):.3f}, range={max(ys) - min(ys):.3f}"
            )
            if fys:
                print(
                    f"    Floor Y: min={min(fys):.3f}, max={max(fys):.3f}, "
                    f"mean={np.mean(fys):.3f}"
                )
            print("    Room centroids (Y):")
            for idx, cy, miny, maxy in centroids:
                n_walls = len(rooms[idx].get("walls", []))
                print(
                    f"      Room {idx}: centroid_y={cy:.3f}, wall_y_range=[{miny:.3f}, "
                    f"{maxy:.3f}], {n_walls} walls"
                )

    # Cross-story overlap check using actual wall positions
    if len(stories_set) > 1:
        print("\n  --- Cross-Story Y Overlap Analysis ---")
        story_ranges = {}
        for s in stories_set:
            ys = story_wall_ys.get(s, [])
            if ys:
                story_ranges[s] = (min(ys), max(ys), np.mean(ys))

        for i in range(len(stories_set)):
            for j in range(i + 1, len(stories_set)):
                s1, s2 = stories_set[i], stories_set[j]
                if s1 in story_ranges and s2 in story_ranges:
                    r1, r2 = story_ranges[s1], story_ranges[s2]
                    overlap = min(r1[1], r2[1]) - max(r1[0], r2[0])
                    gap = r2[2] - r1[2]  # mean Y difference
                    if overlap > 0:
                        print(
                            f"    Story {s1} <-> {s2}: OVERLAP = {overlap:.3f}m  (mean "
                            f"gap = {gap:.3f}m)"
                        )
                        print(
                            f"      Story {s1}: Y=[{r1[0]:.3f}, {r1[1]:.3f}], "
                            f"mean={r1[2]:.3f}"
                        )
                        print(
                            f"      Story {s2}: Y=[{r2[0]:.3f}, {r2[1]:.3f}], "
                            f"mean={r2[2]:.3f}"
                        )
                    else:
                        sep = -overlap
                        print(
                            f"    Story {s1} <-> {s2}: separation = {sep:.3f}m, mean "
                            f"gap = {gap:.3f}m"
                        )

    # Top-level wall analysis per story
    print("\n  --- Top-Level Walls Per Story ---")
    top_wall_story_ys = defaultdict(list)
    top_wall_story_xz = defaultdict(list)
    for w in data.get("walls", []):
        s = w.get("story", -1)
        p = pos(w.get("transform"))
        if p is not None:
            top_wall_story_ys[s].append(p[1])
            top_wall_story_xz[s].append((p[0], p[2]))

    for s in sorted(top_wall_story_ys.keys()):
        ys = top_wall_story_ys[s]
        xzs = np.array(top_wall_story_xz[s])
        print(
            f"  Story {s}: {len(ys)} walls, Y=[{min(ys):.3f}, {max(ys):.3f}], "
            f"mean_Y={np.mean(ys):.3f}"
        )
        print(
            f"    X-range=[{xzs[:, 0].min():.2f}, {xzs[:, 0].max():.2f}], "
            f"Z-range=[{xzs[:, 1].min():.2f}, {xzs[:, 1].max():.2f}]"
        )

    # XZ footprint overlap between stories (do rooms on different stories share the same
    # floorplan area?)
    if len(stories_set) > 1:
        print("\n  --- XZ Footprint Per Story ---")
        for s in stories_set:
            xzs = np.array(top_wall_story_xz.get(s, []))
            if len(xzs) > 0:
                print(
                    f"    Story {s}: X=[{xzs[:, 0].min():.2f}, {xzs[:, 0].max():.2f}], "
                    f"Z=[{xzs[:, 1].min():.2f}, {xzs[:, 1].max():.2f}]"
                )

    # Check sections data for story info
    sections = data.get("sections", [])
    if sections:
        print("\n  --- Sections ---")
        for sec in sections:
            center = sec.get("center", [])
            label = sec.get("label", "?")
            story = sec.get("story", "?")
            print(f"    Section '{label}', story={story}, center={center}")

    # Look for rooms with extreme wall spreads (might indicate bad merging)
    print("\n  --- Rooms with Large Wall Spread (potential merge artifacts) ---")
    for i, room in enumerate(rooms):
        story = room.get("story", -1)
        walls = room.get("walls", [])
        if not walls:
            continue
        positions = [
            pos(w.get("transform"))
            for w in walls
            if pos(w.get("transform")) is not None
        ]
        if not positions:
            continue
        parr = np.array(positions)
        spread = parr.max(axis=0) - parr.min(axis=0)
        if spread[0] > 15 or spread[2] > 15 or spread[1] > 4:
            print(
                f"    Room {i} (story {story}): wall spread X={spread[0]:.2f}, "
                f"Y={spread[1]:.2f}, Z={spread[2]:.2f}"
            )

    # Look for walls with thickness dimension = 0 (missing thickness)
    zero_thick = 0
    for w in data.get("walls", []):
        dim = w.get("dimensions", [0, 0, 0])
        if len(dim) >= 3 and dim[2] == 0:
            zero_thick += 1
    if zero_thick > 0:
        print(
            f"\n  ** {zero_thick}/{len(data.get('walls', []))} walls have zero "
            f"thickness"
        )


ALL = {**PROBLEM_BUILDINGS, **CONTROL_UUIDS}
for uuid, name in ALL.items():
    try:
        analyze(uuid, name)
    except Exception as e:
        print(f"\nERROR analyzing {name}: {e}")
