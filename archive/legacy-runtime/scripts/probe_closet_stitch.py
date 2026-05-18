"""Replay stitch_wall_gaps gates on the target closet pair and report the blocking gate.

Target: building 24e8aaa7-ec15-4a72-be5f-c67b95a53411, story 0, Room 1, endpoint B of
wall
E5C3F73F (target end at (8.118, -8.662)) paired with endpoint A of wall 572EE06C
(at (8.813, -8.507)).

Prints the endpoint pool that would be built by stitch_wall_gaps, the best match chosen
for the target endpoint, and the resulting pair_key. This identifies whether the pair is
(a) not in the pool (height filter), (b) displaced by a closer neighbor, or (c) rejected
by the mutual-best reverse check.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_JSON = REPO / "reconcile" / "buildings_3d.json"

TARGET_BUILDING = "24e8aaa7-ec15-4a72-be5f-c67b95a53411"
TARGET_STORY = 0
TARGET_WALL_A = "E5C3F73F-DC69-423B-940D-A48A7B88F542"  # Room 1 target wall
TARGET_WALL_B = "572EE06C-A367-4261-AD97-CE7194313BD6"  # Room 1 colinear partner

# Reproduce stitch_wall_gaps constants from reconcile/extract3d/stitch.py
MAX_GAP = 1.50
MIN_GAP = 0.06
MIN_WALL_HEIGHT = 0.5
HEIGHT_RATIO = 0.65
MUTUAL_THRESH = 0.60


def main() -> int:
    data = json.loads(BUILDINGS_JSON.read_text())
    bldg = next((b for b in data if b["uuid"] == TARGET_BUILDING), None)
    if bldg is None:
        print(f"building {TARGET_BUILDING} not found")
        return 1

    rooms_out = bldg["rooms"]

    story_rooms = defaultdict(list)
    for room_idx, room in enumerate(rooms_out):
        story_rooms[room["story"]].append((room_idx, room))

    rooms = story_rooms[TARGET_STORY]

    heights = []
    for _ri, room in rooms:
        for wall in room["walls_computed"]:
            corners = wall["corners"]
            if len(corners) < 4:
                continue
            h = abs(corners[2][1] - corners[0][1])
            if h >= MIN_WALL_HEIGHT:
                heights.append(h)
    median_h = float(np.median(heights))
    cutoff = median_h * HEIGHT_RATIO
    print(
        f"story {TARGET_STORY}: wall-height median={median_h:.3f} m, "
        f"cutoff={cutoff:.3f} m"
    )

    endpoints = []
    target_ep_by_wall: dict[str, list[int]] = {}  # wall_id -> [ei=0 idx, ei=1 idx]
    for room_idx, room in rooms:
        for wall_idx, wall in enumerate(room["walls_computed"]):
            corners = wall["corners"]
            if len(corners) < 4:
                continue
            h = abs(corners[2][1] - corners[0][1])
            if h < MIN_WALL_HEIGHT or h < cutoff:
                print(
                    f"  FILTERED room={room_idx} wall_idx={wall_idx} "
                    f"id={wall.get('id', '')[:8]} h={h:.2f}"
                )
                continue
            wid = wall.get("id", "")
            wall_eps = []
            for ei in (0, 1):
                yt = corners[3 if ei == 0 else 2][1]
                endpoints.append(
                    (
                        corners[ei][0],
                        corners[ei][2],
                        corners[ei][1],
                        yt,
                        (room_idx, wall_idx),
                        ei,
                        wid,
                    )
                )
                wall_eps.append(len(endpoints) - 1)
            if wid in (TARGET_WALL_A, TARGET_WALL_B):
                target_ep_by_wall[wid] = wall_eps

    # Pick the two endpoints that are actually facing each other (smallest cross-pair
    # distance)
    target_eps = target_ep_by_wall.get(TARGET_WALL_A, [])
    partner_eps = target_ep_by_wall.get(TARGET_WALL_B, [])
    if not target_eps or not partner_eps:
        print("target or partner wall filtered out of endpoint pool")
        return 2
    best_pair = None
    best_d = float("inf")
    for ti in target_eps:
        for pi in partner_eps:
            te = endpoints[ti]
            pe = endpoints[pi]
            d = math.hypot(te[0] - pe[0], te[1] - pe[1])
            if d < best_d:
                best_d = d
                best_pair = (ti, pi)
    target_a_endpoint_idx, target_b_endpoint_idx = best_pair

    if target_a_endpoint_idx is None:
        print(
            f"target wall {TARGET_WALL_A[:8]} not in pool — height filter excluded it"
        )
        return 2
    if target_b_endpoint_idx is None:
        print(
            f"partner wall {TARGET_WALL_B[:8]} not in pool — height filter excluded it"
        )
        return 2

    print(f"\npool size: {len(endpoints)} endpoints across {len(rooms)} rooms")
    ta = endpoints[target_a_endpoint_idx]
    tb = endpoints[target_b_endpoint_idx]
    print(
        f"target A (E5C3F73F B-end): x={ta[0]:.3f} z={ta[1]:.3f}  room={ta[4][0]} "
        f"wall_idx={ta[4][1]} end={ta[5]}"
    )
    print(
        f"partner (572EE06C A-end): x={tb[0]:.3f} z={tb[1]:.3f}  room={tb[4][0]} "
        f"wall_idx={tb[4][1]} end={tb[5]}"
    )
    expected_dist = math.hypot(ta[0] - tb[0], ta[1] - tb[1])
    print(
        f"expected pair distance: {expected_dist:.3f} m  (max_gap={MAX_GAP}, "
        f"mutual_thresh={MUTUAL_THRESH})"
    )

    # Replay the inner best-match loop for the target endpoint
    x1, z1, _yb1, _yt1, wk1, _end1, _wid1 = ta
    best_dist = MAX_GAP
    best_idx = -1
    candidates = []
    for jdx, ep in enumerate(endpoints):
        if jdx == target_a_endpoint_idx:
            continue
        x2, z2, _yb, _yt, wk2, _e2, wid2 = ep
        if wk2 == wk1:
            continue
        d = math.hypot(x1 - x2, z1 - z2)
        if MIN_GAP <= d < best_dist:
            best_dist = d
            best_idx = jdx
        if d < MAX_GAP:
            candidates.append((d, jdx, wid2, ep[4]))

    candidates.sort()
    print(f"\nall candidates within {MAX_GAP}m of target:")
    for d, jdx, wid2, key in candidates[:10]:
        marker = " <-- EXPECTED PARTNER" if wid2 == TARGET_WALL_B else ""
        marker += " <-- CHOSEN BEST" if jdx == best_idx else ""
        print(f"  d={d:.3f}  wall={wid2[:8]}  room={key[0]} wall_idx={key[1]}{marker}")

    if best_idx < 0:
        print("\nBLOCK GATE: no candidate within max_gap")
        return 0

    chosen = endpoints[best_idx]
    print(
        f"\nbest match for target endpoint: wall={chosen[6][:8]} dist={best_dist:.3f}"
    )

    if chosen[6] != TARGET_WALL_B:
        print(
            "\nBLOCK GATE: a CLOSER endpoint on a DIFFERENT wall displaced the "
            "expected partner"
        )
        print(
            f"  target picked {chosen[6][:8]} at {best_dist:.3f} m instead of 572EE06C "
            f"at {expected_dist:.3f} m"
        )
        return 0

    # Mutual check
    if best_dist <= MUTUAL_THRESH:
        # reverse best
        x2, z2, _yb, _yt, wk2, _e2, _wid2 = chosen
        reverse_best = -1
        reverse_dist = MAX_GAP
        for kdx, ep in enumerate(endpoints):
            if kdx == best_idx:
                continue
            xk, zk, _, _, wkk, _, _ = ep
            if wkk == wk2:
                continue
            d = math.hypot(x2 - xk, z2 - zk)
            if MIN_GAP <= d < reverse_dist:
                reverse_dist = d
                reverse_best = kdx
        if reverse_best != target_a_endpoint_idx:
            print("\nBLOCK GATE: mutual-best reverse check failed")
            print(
                f"  partner's best is index {reverse_best}, not target "
                f"({target_a_endpoint_idx})"
            )
            return 0
    else:
        print(
            f"\nmutual-best gate SKIPPED (best_dist {best_dist:.3f} > mutual_thresh "
            f"{MUTUAL_THRESH})"
        )

    print(
        "\nNO BLOCK: pair should have been stitched; check that stitch output was "
        "actually generated in buildings_3d.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
