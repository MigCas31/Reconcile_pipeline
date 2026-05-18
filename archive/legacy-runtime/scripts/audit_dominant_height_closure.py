"""Audit the dominant-height wall-extension heuristic across buildings_3d.json.

Reports how many stories are eligible (have a dominant top-Y cohort covering
>= MIN_COHORT_COVERAGE of per-story wall perimeter) and how many walls would
be promoted under three progressively tighter gates:

    raw  : wall is a short outlier within delta
    floor: raw + wall bottom matches cohort floor-Y
    all  : floor + wall is colinear with a dominant-cohort neighbour

Output is a summary + histograms. No pipeline state is modified.

Usage:
    python scripts/audit_dominant_height_closure.py \
        [--input reconcile/buildings_3d.json] \
        [--building <uuid>]

The script is intentionally self-contained — it reimplements the cohort and
guard logic locally so it can run without the final ceilings.py helpers
existing yet. Once the pipeline helpers land the numbers should match.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

COHORT_TOLERANCE_M = 0.15
MIN_COHORT_COVERAGE = 0.70
MAX_OUTLIER_SPAN_FRAC = 0.25
MAX_DELTA_M = 0.40
# Minimum delta (dominant_y - wall_top_y) required to promote a wall. Deltas
# below this are indistinguishable from scan noise so we leave them alone.
MIN_PROMOTION_DELTA_M = 0.10
FLOOR_TOLERANCE_M = 0.10
COLINEAR_ANGLE_DEG = 8.0
COLINEAR_OFFSET_M = 0.15
COLINEAR_GAP_M = 0.80


def wall_span_xz(corners: list) -> float:
    """Max pairwise XZ distance among the wall's bottom corners.

    Robust to non-canonical / 5+-corner walls — takes corners within 1 cm of
    min(y) and returns the longest chord.
    """
    if len(corners) < 2:
        return 0.0
    ys = [c[1] for c in corners]
    min_y = min(ys)
    bot = [c for c in corners if c[1] < min_y + 0.01]
    if len(bot) < 2:
        return 0.0
    best = 0.0
    for i in range(len(bot)):
        for j in range(i + 1, len(bot)):
            dx = bot[i][0] - bot[j][0]
            dz = bot[i][2] - bot[j][2]
            d = math.hypot(dx, dz)
            if d > best:
                best = d
    return best


def wall_axis_xz(corners: list) -> tuple[float, float] | None:
    """Return a unit XZ direction vector for the wall's longest bottom chord."""
    if len(corners) < 2:
        return None
    ys = [c[1] for c in corners]
    min_y = min(ys)
    bot = [c for c in corners if c[1] < min_y + 0.01]
    if len(bot) < 2:
        return None
    best = 0.0
    best_pair = None
    for i in range(len(bot)):
        for j in range(i + 1, len(bot)):
            dx = bot[i][0] - bot[j][0]
            dz = bot[i][2] - bot[j][2]
            d = math.hypot(dx, dz)
            if d > best:
                best = d
                best_pair = (bot[i], bot[j])
    if best < 1e-4 or best_pair is None:
        return None
    dx = best_pair[1][0] - best_pair[0][0]
    dz = best_pair[1][2] - best_pair[0][2]
    return (dx / best, dz / best)


def wall_bottom_midpoint_xz(corners: list) -> tuple[float, float] | None:
    if len(corners) < 2:
        return None
    ys = [c[1] for c in corners]
    min_y = min(ys)
    bot = [c for c in corners if c[1] < min_y + 0.01]
    if not bot:
        return None
    return (
        sum(c[0] for c in bot) / len(bot),
        sum(c[2] for c in bot) / len(bot),
    )


def cluster_heights(
    tops_and_weights: list[tuple[float, float]], tol: float
) -> list[tuple[float, float]]:
    """1D clustering: sort by top-Y, greedy-extend a cluster while gap <= tol.

    Returns list of (weighted_center_y, total_weight) per cluster.
    """
    if not tops_and_weights:
        return []
    tops_and_weights = sorted(tops_and_weights, key=lambda t: t[0])
    clusters: list[list[tuple[float, float]]] = [[tops_and_weights[0]]]
    for top, w in tops_and_weights[1:]:
        if top - clusters[-1][-1][0] <= tol:
            clusters[-1].append((top, w))
        else:
            clusters.append([(top, w)])
    out: list[tuple[float, float]] = []
    for cluster in clusters:
        total_w = sum(w for _, w in cluster)
        if total_w <= 0:
            continue
        center = sum(top * w for top, w in cluster) / total_w
        out.append((center, total_w))
    return out


def collect_story_walls(rooms: list[dict]) -> dict[int, list[dict]]:
    """Return {story_idx: [wall_records]} with top_y, bottom_y, span, axis, mid."""
    per_story: dict[int, list[dict]] = defaultdict(list)
    for ridx, room in enumerate(rooms):
        story = room.get("story", 0)
        for widx, wall in enumerate(room.get("walls_computed") or []):
            corners = wall.get("corners") or []
            if len(corners) < 3:
                continue
            ys = [c[1] for c in corners]
            top_y = max(ys)
            bot_y = min(ys)
            if top_y - bot_y < 0.10:
                continue
            span = wall_span_xz(corners)
            if span < 0.05:
                continue
            axis = wall_axis_xz(corners)
            mid = wall_bottom_midpoint_xz(corners)
            per_story[story].append(
                {
                    "room_idx": ridx,
                    "wall_idx": widx,
                    "id": wall.get("id"),
                    "top_y": top_y,
                    "bot_y": bot_y,
                    "span": span,
                    "axis": axis,
                    "mid": mid,
                    "corners": corners,
                    "has_extension": bool(wall.get("extension_strip")),
                }
            )
    return per_story


def compute_cohort(walls: list[dict]) -> dict | None:
    """Identify the dominant top-Y cohort weighted by wall horizontal span.

    Returns {dominant_y, coverage_frac, total_perimeter, cohort_floor_y} or None.
    """
    if not walls:
        return None
    total_w = sum(w["span"] for w in walls)
    if total_w <= 0:
        return None
    clusters = cluster_heights(
        [(w["top_y"], w["span"]) for w in walls], COHORT_TOLERANCE_M
    )
    if not clusters:
        return None
    dominant_y, dominant_w = max(clusters, key=lambda c: c[1])
    coverage = dominant_w / total_w
    cohort_bottoms = [
        (w["bot_y"], w["span"])
        for w in walls
        if abs(w["top_y"] - dominant_y) <= COHORT_TOLERANCE_M
    ]
    cohort_floor_y = weighted_median(cohort_bottoms) if cohort_bottoms else None
    return {
        "dominant_y": dominant_y,
        "coverage_frac": coverage,
        "total_perimeter": total_w,
        "cohort_floor_y": cohort_floor_y,
    }


def weighted_median(items: list[tuple[float, float]]) -> float:
    """Weighted median over (value, weight) pairs."""
    items = sorted(items, key=lambda t: t[0])
    total = sum(w for _, w in items)
    if total <= 0:
        return items[len(items) // 2][0]
    half = total / 2.0
    acc = 0.0
    for val, w in items:
        acc += w
        if acc >= half:
            return val
    return items[-1][0]


def is_colinear_neighbour(target: dict, other: dict) -> bool:
    """Direction parallel, perpendicular offset small, endpoints close along line."""
    if target is other:
        return False
    if target["axis"] is None or other["axis"] is None:
        return False
    if target["mid"] is None or other["mid"] is None:
        return False
    ax, az = target["axis"]
    bx, bz = other["axis"]
    cos_a = abs(ax * bx + az * bz)
    cos_a = min(1.0, max(-1.0, cos_a))
    angle_deg = math.degrees(math.acos(cos_a))
    if angle_deg > COLINEAR_ANGLE_DEG:
        return False
    nx, nz = -az, ax
    tx, tz = target["mid"]
    ox, oz = other["mid"]
    perp = abs((ox - tx) * nx + (oz - tz) * nz)
    if perp > COLINEAR_OFFSET_M:
        return False
    along = (ox - tx) * ax + (oz - tz) * az
    t_half = target["span"] / 2.0
    o_half = other["span"] / 2.0
    gap = abs(along) - (t_half + o_half)
    if gap > COLINEAR_GAP_M:
        return False
    return True


def evaluate_wall(wall: dict, cohort: dict, story_walls: list[dict]) -> dict:
    """Return which gate levels this wall passes."""
    top_y = wall["top_y"]
    dominant_y = cohort["dominant_y"]
    delta = dominant_y - top_y
    raw_pass = (
        delta >= MIN_PROMOTION_DELTA_M
        and delta <= MAX_DELTA_M
        and wall["span"] <= MAX_OUTLIER_SPAN_FRAC * cohort["total_perimeter"]
    )
    floor_pass = False
    all_pass = False
    if raw_pass:
        cfy = cohort["cohort_floor_y"]
        floor_pass = cfy is not None and abs(wall["bot_y"] - cfy) <= FLOOR_TOLERANCE_M
    if floor_pass:
        for other in story_walls:
            if abs(other["top_y"] - cohort["dominant_y"]) > COHORT_TOLERANCE_M:
                continue
            if is_colinear_neighbour(wall, other):
                all_pass = True
                break
    return {"raw": raw_pass, "floor": floor_pass, "all": all_pass, "delta": delta}


def format_hist(values: list[float], edges: list[float]) -> str:
    if not values:
        return "  (empty)"
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, e in enumerate(edges):
            if v < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = []
    for e in edges:
        labels.append(f"<{e:.2f}")
    labels.append(f">={edges[-1]:.2f}")
    lines = []
    for lab, c in zip(labels, counts, strict=False):
        bar = "#" * min(60, c)
        lines.append(f"  {lab:>8} : {c:5d} {bar}")
    return "\n".join(lines)


def run(input_path: Path, building_filter: str | None) -> int:
    with open(input_path) as fh:
        buildings = json.load(fh)

    stories_total = 0
    stories_with_walls = 0
    stories_eligible = 0
    stories_with_raw_candidates = 0
    stories_with_floor_candidates = 0
    stories_with_all_candidates = 0
    walls_raw = 0
    walls_floor = 0
    walls_all = 0
    already_extended = 0

    raw_deltas: list[float] = []
    raw_spans: list[float] = []
    coverage_fracs: list[float] = []
    promoted_per_building: Counter = Counter()

    example_promotions: list[str] = []

    for bldg in buildings:
        uuid = bldg.get("uuid", "?")
        if building_filter and uuid != building_filter:
            continue
        rooms = bldg.get("rooms") or []
        per_story = collect_story_walls(rooms)
        for story, walls in per_story.items():
            stories_total += 1
            if not walls:
                continue
            stories_with_walls += 1
            cohort = compute_cohort(walls)
            if cohort is None:
                continue
            coverage_fracs.append(cohort["coverage_frac"])
            if cohort["coverage_frac"] < MIN_COHORT_COVERAGE:
                continue
            stories_eligible += 1
            any_raw = any_floor = any_all = False
            for wall in walls:
                if wall["has_extension"]:
                    already_extended += 1
                    continue
                res = evaluate_wall(wall, cohort, walls)
                if res["raw"]:
                    walls_raw += 1
                    raw_deltas.append(res["delta"])
                    raw_spans.append(wall["span"])
                    any_raw = True
                if res["floor"]:
                    walls_floor += 1
                    any_floor = True
                if res["all"]:
                    walls_all += 1
                    any_all = True
                    promoted_per_building[uuid] += 1
                    if len(example_promotions) < 20:
                        example_promotions.append(
                            f"{uuid}::wall-computed::{wall['id']}:{story}:"
                            f"{wall['room_idx']} "
                            f"top={wall['top_y']:.3f} dom={cohort['dominant_y']:.3f} "
                            f"delta={res['delta']:.3f} span={wall['span']:.2f}m"
                        )
            if any_raw:
                stories_with_raw_candidates += 1
            if any_floor:
                stories_with_floor_candidates += 1
            if any_all:
                stories_with_all_candidates += 1

    print("=" * 72)
    print("Audit: dominant-height wall extension")
    print(f"Input: {input_path}")
    if building_filter:
        print(f"Building filter: {building_filter}")
    print(
        f"Thresholds: cohort_tol={COHORT_TOLERANCE_M}, min_cov={MIN_COHORT_COVERAGE},"
    )
    print(
        f"            max_span_frac={MAX_OUTLIER_SPAN_FRAC}, max_delta={MAX_DELTA_M},"
    )
    print(f"            min_promotion_delta={MIN_PROMOTION_DELTA_M},")
    print(
        f"            floor_tol={FLOOR_TOLERANCE_M}, colin_angle={COLINEAR_ANGLE_DEG}°,"
    )
    print(f"            colin_offset={COLINEAR_OFFSET_M}, colin_gap={COLINEAR_GAP_M}")
    print("-" * 72)
    print(f"Buildings scanned           : {len(buildings)}")
    print(f"Stories total               : {stories_total}")
    print(f"Stories with computed walls : {stories_with_walls}")
    print(
        f"Stories eligible (coverage >= {MIN_COHORT_COVERAGE:.2f}) : {stories_eligible}"
    )
    print(f"Stories with raw candidates : {stories_with_raw_candidates}")
    print(f"Stories with floor-passing  : {stories_with_floor_candidates}")
    print(f"Stories with all-passing    : {stories_with_all_candidates}")
    print(f"Walls raw-pass              : {walls_raw}")
    print(f"Walls floor-pass            : {walls_floor}")
    print(f"Walls all-pass (promoted)   : {walls_all}")
    print(f"Walls already-extended      : {already_extended}")
    print(
        f"Unique buildings promoted   : {len(promoted_per_building)}"
        f"  (of {len(buildings)})"
    )
    if promoted_per_building:
        print("Top 10 buildings by promotions:")
        for uuid, n in promoted_per_building.most_common(10):
            print(f"  {uuid} : {n}")
    print()
    print("Story coverage_frac distribution:")
    print(format_hist(coverage_fracs, [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]))
    print("Raw-pass delta (dominant_y - top_y) distribution:")
    print(format_hist(raw_deltas, [0.08, 0.12, 0.18, 0.25, 0.32]))
    print("Raw-pass wall span distribution (m):")
    print(format_hist(raw_spans, [0.3, 0.6, 1.0, 1.5, 2.5]))
    if example_promotions:
        print("Example promotions (up to 20):")
        for line in example_promotions:
            print(f"  {line}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = (
        Path(__file__).resolve().parent.parent / "reconcile" / "buildings_3d.json"
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--building", type=str, default=None)
    args = parser.parse_args()
    return run(args.input, args.building)


if __name__ == "__main__":
    raise SystemExit(main())
