"""Audit wall_computed entries for missing extension_strip across stories.

Purpose: classify every wall in every building by the reason its wall-extension
either fires or doesn't. Used to (a) confirm the `story + 1`-only slab pool is
the dominant failure mode on the half-floor cohort, and (b) measure before/after
counts when re-run after the fix.

Reason codes:
  already_extended          -- extension_strip already present
  no_slab_anywhere_above    -- no story-above slab covers wall midpoint in XZ
  slab_s+1_ok               -- story+1 slab covers wall, gap within max_gap
  slab_s+k_ok               -- story+k (k>=2) slab covers wall, gap within max_gap
                               (no viable slab at story+1; THIS is the bug)
  gap_exceeds_max           -- best viable slab above gap > MAX_GAP
  wall_too_short            -- wall_height < MIN_WALL_HEIGHT
  empty_corners             -- wall has no corners

Usage:
    python scripts/audit_half_floor_wall_extensions.py [--input PATH] [--csv OUT]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reconcile" / "buildings_3d.json"

EPSILON = 0.05  # matches ceilings.extend_wall_to_slab
MAX_GAP = 0.80  # matches ceilings.extend_wall_to_slab
MIN_WALL_HEIGHT = 0.10
MIN_MARGIN = 0.05  # matches ceilings.find_best_slab_above
XZ_NEAR_TOL = 0.30  # count a wall midpoint within this XZ distance as "under" the slab


def _shapely():
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    return Point, Polygon, unary_union


def _xz_polygon(floor_polygon, Polygon):
    if not floor_polygon or len(floor_polygon) < 3:
        return None
    pts = [(c[0], c[2]) for c in floor_polygon]
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _classify_building(building, Point, Polygon, unary_union):
    rooms = building.get("rooms") or []

    # Build per-story slab polygons and mean floor_y, matching builder.py.
    slabs_by_story = defaultdict(list)
    floor_y_by_story = defaultdict(list)
    for room in rooms:
        poly = _xz_polygon(room.get("floor_polygon") or [], Polygon)
        if poly is None:
            continue
        fy = sum(c[1] for c in room["floor_polygon"]) / len(room["floor_polygon"])
        slabs_by_story[room["story"]].append((poly, fy))
        floor_y_by_story[room["story"]].append(fy)

    rows = []
    for room in rooms:
        src_story = room["story"]
        # Build candidate pool: all stories strictly above src_story (the fix's pool).
        broad_candidates = []
        for s in sorted(slabs_by_story):
            if s > src_story:
                for poly, fy in slabs_by_story[s]:
                    broad_candidates.append((s, poly, fy))

        for wall in room.get("walls_computed") or []:
            corners = wall.get("corners") or []
            if len(corners) < 2:
                rows.append(
                    {
                        "uuid": building["uuid"],
                        "story": src_story,
                        "wall_id": wall.get("id"),
                        "wall_top_y": None,
                        "has_extension": bool(wall.get("extension_strip")),
                        "best_story_above": None,
                        "best_slab_y": None,
                        "gap": None,
                        "reason": "empty_corners",
                    }
                )
                continue

            ys = [c[1] for c in corners]
            wall_top_y = max(ys)
            wall_height = wall_top_y - min(ys)
            has_ext = bool(wall.get("extension_strip"))

            if has_ext:
                rows.append(
                    {
                        "uuid": building["uuid"],
                        "story": src_story,
                        "wall_id": wall.get("id"),
                        "wall_top_y": round(wall_top_y, 3),
                        "has_extension": True,
                        "best_story_above": None,
                        "best_slab_y": None,
                        "gap": None,
                        "reason": "already_extended",
                    }
                )
                continue

            if wall_height < MIN_WALL_HEIGHT:
                rows.append(
                    {
                        "uuid": building["uuid"],
                        "story": src_story,
                        "wall_id": wall.get("id"),
                        "wall_top_y": round(wall_top_y, 3),
                        "has_extension": False,
                        "best_story_above": None,
                        "best_slab_y": None,
                        "gap": None,
                        "reason": "wall_too_short",
                    }
                )
                continue

            # Find best slab matching find_best_slab_above semantics,
            # but over the broad candidate pool.
            wx = sum(c[0] for c in corners) / len(corners)
            wz = sum(c[2] for c in corners) / len(corners)
            pt = Point(wx, wz)
            best = None  # (dist, story, slab_y)
            for story_above, poly, slab_y in broad_candidates:
                if slab_y <= wall_top_y + MIN_MARGIN:
                    continue
                dist = float(poly.distance(pt))
                if dist > XZ_NEAR_TOL:
                    continue
                if (
                    best is None
                    or dist < best[0]
                    or (dist == best[0] and story_above < best[1])
                ):
                    best = (dist, story_above, slab_y)

            if best is None:
                rows.append(
                    {
                        "uuid": building["uuid"],
                        "story": src_story,
                        "wall_id": wall.get("id"),
                        "wall_top_y": round(wall_top_y, 3),
                        "has_extension": False,
                        "best_story_above": None,
                        "best_slab_y": None,
                        "gap": None,
                        "reason": "no_slab_anywhere_above",
                    }
                )
                continue

            _, tgt_story, slab_y = best
            gap = slab_y - wall_top_y
            if gap > MAX_GAP:
                reason = "gap_exceeds_max"
            elif tgt_story == src_story + 1:
                reason = "slab_s+1_ok"
            else:
                reason = "slab_s+k_ok"
            rows.append(
                {
                    "uuid": building["uuid"],
                    "story": src_story,
                    "wall_id": wall.get("id"),
                    "wall_top_y": round(wall_top_y, 3),
                    "has_extension": False,
                    "best_story_above": tgt_story,
                    "best_slab_y": round(slab_y, 3),
                    "gap": round(gap, 3),
                    "reason": reason,
                }
            )

    # Building-level split-level signal.
    rooms_per_story = Counter(r["story"] for r in rooms)
    has_single_room_story = any(c == 1 for c in rooms_per_story.values())
    ordered_stories = sorted(floor_y_by_story)
    deltas = []
    for i in range(len(ordered_stories) - 1):
        lo = sum(floor_y_by_story[ordered_stories[i]]) / len(
            floor_y_by_story[ordered_stories[i]]
        )
        hi = sum(floor_y_by_story[ordered_stories[i + 1]]) / len(
            floor_y_by_story[ordered_stories[i + 1]]
        )
        deltas.append(hi - lo)
    has_tight_spacing = any(d < 2.0 for d in deltas)
    split_level = has_single_room_story or has_tight_spacing

    return rows, split_level, rooms_per_story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--csv", default=None, help="Optional per-wall CSV dump path")
    ap.add_argument("--only-split-level", action="store_true")
    args = ap.parse_args()

    Point, Polygon, unary_union = _shapely()

    with open(args.input) as fh:
        buildings = json.load(fh)

    by_building = {}
    split_level_uuids = set()
    for b in buildings:
        rows, is_split, rooms_per_story = _classify_building(
            b, Point, Polygon, unary_union
        )
        by_building[b["uuid"]] = (rows, is_split, rooms_per_story)
        if is_split:
            split_level_uuids.add(b["uuid"])

    Counter()
    # fixable = slab_s+k_ok (k>=2) or slab_s+1_ok (but already_extended=False
    # means untriggered)
    Counter()

    print(f"buildings total: {len(buildings)}")
    print(
        f"split-level (single-room-story OR story spacing < 2.0m): "
        f"{len(split_level_uuids)}"
    )
    print()

    # Per-reason totals across the whole corpus and on split-level subset.
    reason_all = Counter()
    reason_split = Counter()
    fixable_uuids = Counter()  # uuid -> number of walls with slab_s+k_ok (k>=2)
    for uuid, (rows, is_split, _rpp) in by_building.items():
        for r in rows:
            reason_all[r["reason"]] += 1
            if is_split:
                reason_split[r["reason"]] += 1
            if r["reason"] == "slab_s+k_ok":
                fixable_uuids[uuid] += 1

    print("Reason distribution (all buildings):")
    for reason, count in reason_all.most_common():
        print(f"  {reason:28s} {count:6d}")
    print()
    print(f"Reason distribution (split-level subset, n={len(split_level_uuids)}):")
    for reason, count in reason_split.most_common():
        print(f"  {reason:28s} {count:6d}")
    print()

    # Per-building summary (split-level only).
    print(
        "Per-building (split-level) with slab_s+k_ok (k>=2) walls -- these will "
        "be FIXED by the Phase B change:"
    )
    items = sorted(fixable_uuids.items(), key=lambda kv: -kv[1])
    for uuid, n in items:
        rows = by_building[uuid][0]
        exts_now = sum(1 for r in rows if r["reason"] == "already_extended")
        total = len(rows)
        print(
            f"  {uuid[:8]}  walls_total={total:4d}  ext_now={exts_now:4d}  "
            f"fixable_s+k={n:4d}"
        )
    print()
    print(f"Total fixable walls across corpus: {sum(fixable_uuids.values())}")
    print(f"Buildings with >=1 fixable wall: {len(fixable_uuids)}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "uuid",
                    "story",
                    "wall_id",
                    "wall_top_y",
                    "has_extension",
                    "best_story_above",
                    "best_slab_y",
                    "gap",
                    "reason",
                ],
            )
            writer.writeheader()
            for rows, _, _ in by_building.values():
                for r in rows:
                    writer.writerow(r)
        print(f"Per-wall CSV written to {out}")


if __name__ == "__main__":
    sys.exit(main())
