"""Analyse residual per-room floor Y / wall-height variation in tier_payload.json.

The user's hypothesis: rooms within a building show small differences in floor
Y and wall height that could be harmonised for visual cleanliness, *without*
changing total wall area per building.

This script:
  1. Reads every pipeline-outputs/<uuid>/tier_payload.json.
  2. Reverse-engineers per-story alignment groups using the same tolerances as
     reconcile_tiers/extract/height_align.py (FLOOR_ALIGN_TOL_M = 6 cm,
     WALL_HEIGHT_ALIGN_TOL_M = 5 cm).
  3. Per group, computes total wall horizontal length and the (already-snapped)
     wall height -> wall area = length * height.
  4. Per story, reports:
       - n_groups
       - between-group floor_y spread, ceiling_y spread, height spread
       - the "story length-weighted mean" target wall height that would
         preserve total wall area exactly across the whole story
  5. Flags false-split candidates: pairs of groups within the same story whose
     floor gap AND height gap are both < 10 cm (likely sub-tolerance noise the
     current grouping rejected, NOT real architectural breaks).
  6. Computes total per-building wall area as a sanity-check baseline.
  7. Aggregates corpus-level histograms and writes a markdown report.

No code paths in reconcile_tiers/ are modified by this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pipeline-outputs").is_dir() and (
            parent / "reconcile_tiers"
        ).is_dir():
            return parent
    return Path.cwd()


ROOT = _find_workspace_root()
DEFAULT_OUTPUTS = ROOT / "pipeline-outputs"
DEFAULT_REPORT = ROOT / ".context" / "floor_y_harmonization_report.md"

FLOOR_ALIGN_TOL_M = 0.06
WALL_HEIGHT_ALIGN_TOL_M = 0.05
MIN_WALL_HEIGHT_M = 0.10
MIN_WALL_SPAN_M = 0.05

FALSE_SPLIT_FLOOR_GAP_M = 0.10
FALSE_SPLIT_HEIGHT_GAP_M = 0.10

FLAT_CEILING_TOL_M = 0.05


@dataclass
class RoomMetrics:
    floor_y: float
    ceiling_y: float
    height: float
    total_wall_length: float
    n_walls: int
    wall_top_spread: float
    flat_ceiling: bool


@dataclass
class GroupSummary:
    rooms: int
    floor_y: float
    ceiling_y: float
    height: float
    total_wall_length: float
    total_wall_area: float


MODE_GAP_THRESHOLD_M = 0.10


@dataclass
class StorySummary:
    story: int
    n_rooms: int
    groups: list[GroupSummary]
    total_wall_area: float
    target_height_lwm: float | None
    floor_spread_cm: float
    ceiling_spread_cm: float
    height_spread_cm: float
    false_split_pairs: int
    room_floor_ys: list[float]
    room_heights: list[float]
    floor_max_gap_cm: float
    height_max_gap_cm: float
    floor_modes: int
    height_modes: int


@dataclass
class BuildingSummary:
    uuid: str
    address: str | None
    total_wall_area: float
    n_stories: int
    stories: list[StorySummary]


def _wall_horizontal_length(corners: list[dict[str, float]]) -> float:
    if len(corners) < 2:
        return 0.0
    pts = [(c["x"], c["z"]) for c in corners]
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dz = pts[i][1] - pts[j][1]
            d = math.sqrt(dx * dx + dz * dz)
            if d > best:
                best = d
    return best


def _wall_metrics(wall: dict) -> tuple[float, float, float, float] | None:
    """Return (min_y, max_y, height, horizontal_length) or None if the wall is
    too small to count. Mirrors the gates in
    reconcile_tiers/extract/height_align.py."""
    corners = wall.get("corners") or []
    if len(corners) < 3:
        return None
    span = _wall_horizontal_length(corners)
    if span < MIN_WALL_SPAN_M:
        return None
    ys = [c["y"] for c in corners]
    h = max(ys) - min(ys)
    if h < MIN_WALL_HEIGHT_M:
        return None
    return min(ys), max(ys), h, span


def _room_metrics(room: dict) -> RoomMetrics | None:
    walls = room.get("walls") or []
    floor_ys: list[float] = []
    ceil_ys: list[float] = []
    total_len = 0.0
    n = 0
    for w in walls:
        m = _wall_metrics(w)
        if m is None:
            continue
        wmin, wmax, _h, span = m
        floor_ys.append(wmin)
        ceil_ys.append(wmax)
        total_len += span
        n += 1
    if not floor_ys:
        return None
    floor_y = statistics.median(floor_ys)
    ceiling_y = statistics.median(ceil_ys)
    wall_top_spread = max(ceil_ys) - min(ceil_ys)
    return RoomMetrics(
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        height=ceiling_y - floor_y,
        total_wall_length=total_len,
        n_walls=n,
        wall_top_spread=wall_top_spread,
        flat_ceiling=wall_top_spread <= FLAT_CEILING_TOL_M,
    )


def _group_rooms(
    metrics: list[tuple[RoomMetrics, int]],
) -> list[list[tuple[RoomMetrics, int]]]:
    """Replicate the grouping in height_align.py:_group_rooms."""
    sorted_entries = sorted(metrics, key=lambda item: item[0].floor_y)
    groups: list[list[tuple[RoomMetrics, int]]] = []
    current: list[tuple[RoomMetrics, int]] = []
    for entry in sorted_entries:
        if not current:
            current.append(entry)
            continue
        rm = entry[0]
        floor_ys = [item[0].floor_y for item in current]
        spread_ok = (
            rm.floor_y - min(floor_ys) <= FLOOR_ALIGN_TOL_M
            and max(rm.floor_y - min(floor_ys), max(floor_ys) - rm.floor_y)
            <= FLOOR_ALIGN_TOL_M
        )
        median_h = statistics.median(item[0].height for item in current)
        height_ok = abs(rm.height - median_h) <= WALL_HEIGHT_ALIGN_TOL_M
        if spread_ok and height_ok:
            current.append(entry)
        else:
            groups.append(current)
            current = [entry]
    if current:
        groups.append(current)
    return groups


def _summarise_group(rooms: list[tuple[RoomMetrics, int]]) -> GroupSummary:
    floor_ys = [rm.floor_y for rm, _ in rooms]
    ceiling_ys = [rm.ceiling_y for rm, _ in rooms]
    heights = [rm.height for rm, _ in rooms]
    total_len = sum(rm.total_wall_length for rm, _ in rooms)
    floor_y = statistics.median(floor_ys)
    ceiling_y = statistics.median(ceiling_ys)
    height = statistics.median(heights)
    return GroupSummary(
        rooms=len(rooms),
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        height=height,
        total_wall_length=total_len,
        total_wall_area=total_len * height,
    )


def _gap_modes(values: list[float], threshold: float) -> tuple[float, int]:
    """Return (max_consecutive_gap, mode_count). A 'mode' is a maximal run of
    sorted values whose consecutive gaps are all < threshold."""
    if len(values) < 2:
        return 0.0, max(1, len(values))
    sorted_vals = sorted(values)
    gaps = [b - a for a, b in itertools.pairwise(sorted_vals)]
    max_gap = max(gaps)
    modes = 1 + sum(1 for g in gaps if g >= threshold)
    return max_gap, modes


def _length_weighted_mean_height(groups: list[GroupSummary]) -> float | None:
    total_len = sum(g.total_wall_length for g in groups)
    if total_len <= 1e-9:
        return None
    return sum(g.total_wall_length * g.height for g in groups) / total_len


def _building_total_wall_area(building: dict) -> float:
    total = 0.0
    for room in building.get("rooms") or []:
        for w in room.get("walls") or []:
            m = _wall_metrics(w)
            if m is None:
                continue
            _wmin, _wmax, h, span = m
            total += h * span
    return total


def analyse_building(payload: dict, *, flat_only: bool = False) -> BuildingSummary:
    rooms = payload.get("rooms") or []
    by_story: dict[int, list[tuple[RoomMetrics, int]]] = {}
    for idx, room in enumerate(rooms):
        rm = _room_metrics(room)
        if rm is None:
            continue
        if flat_only and not rm.flat_ceiling:
            continue
        by_story.setdefault(int(room.get("story", 0)), []).append((rm, idx))

    stories: list[StorySummary] = []
    for story_idx in sorted(by_story.keys()):
        story_entries = by_story[story_idx]
        groups = _group_rooms(story_entries)
        group_summaries = [_summarise_group(g) for g in groups]
        if not group_summaries:
            continue

        floor_ys = [g.floor_y for g in group_summaries]
        ceiling_ys = [g.ceiling_y for g in group_summaries]
        heights = [g.height for g in group_summaries]
        floor_spread = (max(floor_ys) - min(floor_ys)) * 100.0
        ceiling_spread = (max(ceiling_ys) - min(ceiling_ys)) * 100.0
        height_spread = (max(heights) - min(heights)) * 100.0

        false_split = 0
        for i in range(len(group_summaries)):
            for j in range(i + 1, len(group_summaries)):
                gi = group_summaries[i]
                gj = group_summaries[j]
                if (
                    abs(gi.floor_y - gj.floor_y) < FALSE_SPLIT_FLOOR_GAP_M
                    and abs(gi.height - gj.height) < FALSE_SPLIT_HEIGHT_GAP_M
                ):
                    false_split += 1

        target_h = _length_weighted_mean_height(group_summaries)
        total_area = sum(g.total_wall_area for g in group_summaries)

        room_floor_ys = sorted(rm.floor_y for rm, _ in story_entries)
        room_heights = sorted(rm.height for rm, _ in story_entries)
        floor_max_gap, floor_modes = _gap_modes(room_floor_ys, MODE_GAP_THRESHOLD_M)
        height_max_gap, height_modes = _gap_modes(room_heights, MODE_GAP_THRESHOLD_M)

        stories.append(
            StorySummary(
                story=story_idx,
                n_rooms=sum(g.rooms for g in group_summaries),
                groups=group_summaries,
                total_wall_area=total_area,
                target_height_lwm=target_h,
                floor_spread_cm=floor_spread,
                ceiling_spread_cm=ceiling_spread,
                height_spread_cm=height_spread,
                false_split_pairs=false_split,
                room_floor_ys=room_floor_ys,
                room_heights=room_heights,
                floor_max_gap_cm=floor_max_gap * 100.0,
                height_max_gap_cm=height_max_gap * 100.0,
                floor_modes=floor_modes,
                height_modes=height_modes,
            )
        )

    return BuildingSummary(
        uuid=str(payload.get("uuid", "<unknown>")),
        address=payload.get("address"),
        total_wall_area=_building_total_wall_area(payload),
        n_stories=len(stories),
        stories=stories,
    )


def _bucket(value_cm: float, edges: list[float]) -> str:
    for i, edge in enumerate(edges):
        if value_cm < edge:
            if i == 0:
                return f"<{edge:.0f}cm"
            return f"{edges[i - 1]:.0f}-{edge:.0f}cm"
    return f">={edges[-1]:.0f}cm"


def _corpus_rollup_lines(label: str, buildings: list[BuildingSummary]) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {label}\n")

    n_buildings = sum(1 for b in buildings if b.n_stories > 0)
    total_stories = sum(b.n_stories for b in buildings)
    all_groups = [g for b in buildings for s in b.stories for g in s.groups]
    total_groups = len(all_groups)
    if total_groups == 0:
        lines.append("(no rooms matched this filter — nothing to report)\n")
        return lines

    singleton_groups = sum(1 for g in all_groups if g.rooms == 1)
    total_rooms = sum(g.rooms for g in all_groups)
    rooms_in_singleton_groups = sum(g.rooms for g in all_groups if g.rooms == 1)
    multi_group_stories = [s for b in buildings for s in b.stories if len(s.groups) > 1]
    false_split_stories = [s for s in multi_group_stories if s.false_split_pairs > 0]

    floor_spreads_cm = [s.floor_spread_cm for s in multi_group_stories]
    height_spreads_cm = [s.height_spread_cm for s in multi_group_stories]

    edges = [1, 5, 10, 20, 50, 100]
    floor_buckets = Counter(_bucket(v, edges) for v in floor_spreads_cm)
    height_buckets = Counter(_bucket(v, edges) for v in height_spreads_cm)

    lines.append(f"- Buildings with rooms in this slice: **{n_buildings}**")
    lines.append(f"- Stories: **{total_stories}**")
    lines.append(f"- Rooms: **{total_rooms}**")
    lines.append(f"- Alignment groups: **{total_groups}**")
    lines.append(
        f"- Singleton groups (1 room only): "
        f"**{singleton_groups}** ({singleton_groups / total_groups * 100:.1f}% of "
        f"groups)"
    )
    lines.append(
        f"- Rooms left as their own group: **{rooms_in_singleton_groups}** "
        f"({rooms_in_singleton_groups / total_rooms * 100:.1f}% of rooms)"
    )
    lines.append(f"- Average rooms per group: **{total_rooms / total_groups:.2f}**")
    if total_stories > 0:
        lines.append(
            f"- Stories with >1 group: **{len(multi_group_stories)}** "
            f"({len(multi_group_stories) / total_stories * 100:.1f}% of stories)"
        )
        lines.append(
            f"- Stories with >=1 false-split pair (< 10 cm in BOTH floor and height): "
            f"**{len(false_split_stories)}** "
            f"({len(false_split_stories) / total_stories * 100:.1f}% of stories)\n"
        )

    if floor_spreads_cm:
        lines.append("### Between-group floor-Y spread per multi-group story (cm)")
        lines.append(
            f"- min={min(floor_spreads_cm):.1f}  "
            f"median={statistics.median(floor_spreads_cm):.1f}  "
            f"mean={statistics.fmean(floor_spreads_cm):.1f}  "
            f"max={max(floor_spreads_cm):.1f}"
        )
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|---|---|")
        for bucket in sorted(floor_buckets.keys(), key=lambda s: (len(s), s)):
            lines.append(f"| {bucket} | {floor_buckets[bucket]} |")
        lines.append("")

    if height_spreads_cm:
        lines.append("### Between-group wall-height spread per multi-group story (cm)")
        lines.append(
            f"- min={min(height_spreads_cm):.1f}  "
            f"median={statistics.median(height_spreads_cm):.1f}  "
            f"mean={statistics.fmean(height_spreads_cm):.1f}  "
            f"max={max(height_spreads_cm):.1f}"
        )
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|---|---|")
        for bucket in sorted(height_buckets.keys(), key=lambda s: (len(s), s)):
            lines.append(f"| {bucket} | {height_buckets[bucket]} |")
        lines.append("")

    multi_or_more = [s for b in buildings for s in b.stories if s.n_rooms >= 2]
    if multi_or_more:
        floor_max_gaps = [s.floor_max_gap_cm for s in multi_or_more]
        height_max_gaps = [s.height_max_gap_cm for s in multi_or_more]
        floor_unimodal = sum(1 for s in multi_or_more if s.floor_modes == 1)
        height_unimodal = sum(1 for s in multi_or_more if s.height_modes == 1)
        floor_mode_dist = Counter(s.floor_modes for s in multi_or_more)
        height_mode_dist = Counter(s.height_modes for s in multi_or_more)

        lines.append(
            f"### Distribution shape per story (>= 2 rooms, gap threshold "
            f"{MODE_GAP_THRESHOLD_M * 100:.0f} cm)"
        )
        lines.append(
            "Sort each story's per-room values, classify a 'mode' as a maximal "
            f"run of sorted values whose consecutive gaps are all < "
            f"{MODE_GAP_THRESHOLD_M * 100:.0f} cm. A unimodal story is "
            f"alignable noise; "
            "a multimodal story carries real architectural structure that must "
            "be preserved.\n"
        )
        lines.append(
            f"- Floor-Y unimodal stories: **{floor_unimodal} / {len(multi_or_more)}** "
            f"({floor_unimodal / len(multi_or_more) * 100:.0f}%)"
        )
        lines.append(
            f"- Wall-height unimodal stories: **{height_unimodal} / "
            f"{len(multi_or_more)}** "
            f"({height_unimodal / len(multi_or_more) * 100:.0f}%)"
        )
        lines.append("")
        lines.append("Floor-Y mode count distribution:")
        lines.append("")
        lines.append("| modes | stories |")
        lines.append("|---|---|")
        for k in sorted(floor_mode_dist.keys()):
            lines.append(f"| {k} | {floor_mode_dist[k]} |")
        lines.append("")
        lines.append("Wall-height mode count distribution:")
        lines.append("")
        lines.append("| modes | stories |")
        lines.append("|---|---|")
        for k in sorted(height_mode_dist.keys()):
            lines.append(f"| {k} | {height_mode_dist[k]} |")
        lines.append("")

        gap_buckets_floor = Counter(_bucket(v, edges) for v in floor_max_gaps)
        gap_buckets_height = Counter(_bucket(v, edges) for v in height_max_gaps)
        lines.append("Max consecutive gap in sorted per-room floor-Y (cm):")
        lines.append(
            f"- min={min(floor_max_gaps):.1f}  "
            f"median={statistics.median(floor_max_gaps):.1f}  "
            f"mean={statistics.fmean(floor_max_gaps):.1f}  "
            f"max={max(floor_max_gaps):.1f}"
        )
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|---|---|")
        for bucket in sorted(gap_buckets_floor.keys(), key=lambda s: (len(s), s)):
            lines.append(f"| {bucket} | {gap_buckets_floor[bucket]} |")
        lines.append("")
        lines.append("Max consecutive gap in sorted per-room wall-height (cm):")
        lines.append(
            f"- min={min(height_max_gaps):.1f}  "
            f"median={statistics.median(height_max_gaps):.1f}  "
            f"mean={statistics.fmean(height_max_gaps):.1f}  "
            f"max={max(height_max_gaps):.1f}"
        )
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|---|---|")
        for bucket in sorted(gap_buckets_height.keys(), key=lambda s: (len(s), s)):
            lines.append(f"| {bucket} | {gap_buckets_height[bucket]} |")
        lines.append("")

    return lines


def write_report(
    all_buildings: list[BuildingSummary],
    flat_buildings: list[BuildingSummary],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    multi_group_stories = [
        s for b in all_buildings for s in b.stories if len(s.groups) > 1
    ]
    false_split_stories = [s for s in multi_group_stories if s.false_split_pairs > 0]

    lines: list[str] = []
    lines.append("# Floor-Y / wall-height harmonization analysis\n")
    lines.append(
        "Reverse-engineers post-`align_room_heights` groups from "
        "`tier_payload.json`. Reports residual between-group variation per "
        "story and identifies candidates for further harmonization. "
        "Compares the full corpus against the subset of rooms classified as "
        f"flat-ceiling (intra-room wall-top-Y spread <= {FLAT_CEILING_TOL_M * 100:.0f} "
        f"cm).\n"
    )

    lines.extend(_corpus_rollup_lines("Corpus rollup — all rooms", all_buildings))
    lines.extend(
        _corpus_rollup_lines("Corpus rollup — flat-ceiling rooms only", flat_buildings)
    )

    lines.append("## False-split candidates (all rooms)\n")
    lines.append(
        "Stories where TWO groups are < 10 cm apart in BOTH floor-Y and "
        "wall-height. These are the borderline cases the current 6 cm / 5 cm "
        "tolerance just barely rejects. If we were ever to widen tolerances or "
        "merge cross-group within a story, these are the rooms it would "
        "affect.\n"
    )
    if not false_split_stories:
        lines.append("(none in corpus)\n")
    else:
        lines.append(
            "| Building | Story | n_groups | floor_spread cm | height_spread cm | "
            "false_split_pairs |"
        )
        lines.append("|---|---|---|---|---|---|")
        for b in all_buildings:
            for s in b.stories:
                if s.false_split_pairs == 0:
                    continue
                lines.append(
                    f"| `{b.uuid[:8]}` | {s.story} | {len(s.groups)} | "
                    f"{s.floor_spread_cm:.1f} | {s.height_spread_cm:.1f} | "
                    f"{s.false_split_pairs} |"
                )
        lines.append("")

    lines.append("## What this means\n")

    flat_multi_or_more = [
        s for b in flat_buildings for s in b.stories if s.n_rooms >= 2
    ]
    all_multi_or_more = [s for b in all_buildings for s in b.stories if s.n_rooms >= 2]
    flat_floor_unimodal = sum(1 for s in flat_multi_or_more if s.floor_modes == 1)
    flat_height_unimodal = sum(1 for s in flat_multi_or_more if s.height_modes == 1)
    flat_both_unimodal = sum(
        1 for s in flat_multi_or_more if s.floor_modes == 1 and s.height_modes == 1
    )
    [s for b in flat_buildings for s in b.stories if len(s.groups) > 1]

    def _pct(num: int, denom: int) -> str:
        if denom == 0:
            return "n/a"
        return f"{num / denom * 100:.0f}%"

    lines.append(
        "1. **Spread is the wrong primitive — distribution shape is the right "
        "one.** A story with 9 rooms tightly clustered around floor_y = -1.50 "
        "plus 1 outlier at -1.85 has a 35 cm 'spread' but is *unimodal*: 9 "
        "rooms want to merge, 1 doesn't. A genuine split-level looks "
        "different — two clusters with a clear gap between them. "
        f"Using a {MODE_GAP_THRESHOLD_M * 100:.0f} cm gap threshold to define "
        "modes:\n"
        f"   - All rooms: floor-Y unimodal "
        f"{
            _pct(
                sum(1 for s in all_multi_or_more if s.floor_modes == 1),
                len(all_multi_or_more),
            )
        }, "
        f"height unimodal "
        f"{
            _pct(
                sum(1 for s in all_multi_or_more if s.height_modes == 1),
                len(all_multi_or_more),
            )
        }\n"
        f"   - Flat-ceiling only: floor-Y unimodal "
        f"**{_pct(flat_floor_unimodal, len(flat_multi_or_more))}**, "
        f"height unimodal "
        f"**{_pct(flat_height_unimodal, len(flat_multi_or_more))}**, "
        f"BOTH unimodal **{_pct(flat_both_unimodal, len(flat_multi_or_more))}**\n"
    )
    lines.append(
        "2. **Restricting to flat-ceiling rooms collapses the long tail.** "
        "Median wall-height max-gap drops from "
        f"{statistics.median([s.height_max_gap_cm for s in all_multi_or_more]):.1f} cm "
        f"(all rooms) to "
        f"{statistics.median([s.height_max_gap_cm for s in flat_multi_or_more]):.1f}"
        f" cm "
        "(flat-ceiling only). The >= 100 cm tail goes from "
        f"{sum(1 for s in all_multi_or_more if s.height_max_gap_cm >= 100)} stories to "
        f"{sum(1 for s in flat_multi_or_more if s.height_max_gap_cm >= 100)} stories. "
        "Vaulted ceilings, attic transitions, half-floors all live in oblique "
        "rooms — exactly where they should.\n"
    )
    lines.append(
        "3. **The flat-ceiling flag is the missing physical signal.** "
        f"Derivable from wall corners alone (intra-room max-Y - min-Y of "
        f"wall tops <= {FLAT_CEILING_TOL_M * 100:.0f} cm). No topology graph "
        "or doorway-without-stair check needed. Vaulted / mezzanine / "
        "half-floor rooms are never touched. Satisfies 'keep diagnostic "
        "signal' and 'synthesis follows scan'.\n"
    )
    lines.append(
        "4. **Concrete proposal (mode-based merge):** in "
        "`reconcile_tiers/extract/height_align.py`, replace the cumulative "
        "tolerance grouping with mode-based grouping for flat-ceiling rooms. "
        "Per story: sort flat-ceiling rooms by floor_y, find consecutive "
        f"gaps, break into modes wherever a gap >= "
        f"{MODE_GAP_THRESHOLD_M * 100:.0f} cm. Each mode is a group; rooms "
        "within a mode snap to the **length-weighted mean** of (floor_y, "
        "height) — preserves total wall area exactly per mode, and therefore "
        "per story and per building. Oblique-ceiling rooms keep the current "
        "6 cm / 5 cm tolerance behaviour.\n"
    )
    lines.append(
        f"5. **Estimated impact:** {flat_both_unimodal} flat-ceiling stories "
        f"({_pct(flat_both_unimodal, len(flat_multi_or_more))} of those with "
        ">=2 flat rooms) collapse to a single group (visually flat floor and "
        f"ceiling per story). The remaining "
        f"{len(flat_multi_or_more) - flat_both_unimodal} stories keep "
        f"multiple groups but each group itself becomes flat. Before "
        "shipping: pytest covering bit-exact area preservation; visual "
        "spot-check 5-10 buildings; confirm audits in "
        "`archive/legacy-runtime/reconcile/inspect_building/audit.py` "
        "still pass.\n"
    )

    lines.append(
        "## Per-building detail — flat-ceiling rooms only (multi-group stories)\n"
    )
    lines.append(
        "For each story with >1 alignment group **after restricting to "
        "flat-ceiling rooms**, shows per-group floor_y, height, total wall "
        "length, area, and the length-weighted-mean story-collapsed target "
        "height (the only single height that preserves total wall area "
        "exactly when collapsing all groups in this story).\n"
    )

    for b in sorted(
        flat_buildings, key=lambda x: -sum(1 for s in x.stories if len(s.groups) > 1)
    ):
        multi = [s for s in b.stories if len(s.groups) > 1]
        if not multi:
            continue
        addr = f" — {b.address}" if b.address else ""
        lines.append(f"### `{b.uuid[:8]}`{addr}")
        lines.append(
            f"Total wall area: **{b.total_wall_area:.1f} m²** across {b.n_stories} "
            f"stories\n"
        )
        for s in multi:
            lines.append(
                f"**Story {s.story}** — {s.n_rooms} rooms, "
                f"{len(s.groups)} groups, floor spread {s.floor_spread_cm:.1f} cm, "
                f"height spread {s.height_spread_cm:.1f} cm, "
                f"false-split pairs: {s.false_split_pairs}"
            )
            lines.append("")
            lines.append(
                "- Per-room floor-Y (sorted, m): "
                + ", ".join(f"{v:.3f}" for v in s.room_floor_ys)
            )
            floor_gaps_cm = [
                (b - a) * 100.0
                for a, b in zip(s.room_floor_ys, s.room_floor_ys[1:], strict=False)
            ]
            lines.append(
                "- Floor-Y consecutive gaps (cm): "
                + ", ".join(f"{v:.1f}" for v in floor_gaps_cm)
                + f"  -> max={s.floor_max_gap_cm:.1f}, modes={s.floor_modes}"
            )
            lines.append(
                "- Per-room wall height (sorted, m): "
                + ", ".join(f"{v:.3f}" for v in s.room_heights)
            )
            height_gaps_cm = [
                (b - a) * 100.0
                for a, b in zip(s.room_heights, s.room_heights[1:], strict=False)
            ]
            lines.append(
                "- Wall-height consecutive gaps (cm): "
                + ", ".join(f"{v:.1f}" for v in height_gaps_cm)
                + f"  -> max={s.height_max_gap_cm:.1f}, modes={s.height_modes}"
            )
            lines.append("")
            lines.append(
                "| Group | rooms | floor_y | ceiling_y | height | length | area |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for i, g in enumerate(s.groups):
                lines.append(
                    f"| {i} | {g.rooms} | {g.floor_y:.3f} | {g.ceiling_y:.3f} | "
                    f"{g.height:.3f} | {g.total_wall_length:.2f} | "
                    f"{g.total_wall_area:.2f} |"
                )
            target = s.target_height_lwm or 0.0
            collapsed_area = target * sum(g.total_wall_length for g in s.groups)
            lines.append(
                f"\nLength-weighted-mean target height across this story: "
                f"**{target:.3f} m** -> story-collapsed area "
                f"**{collapsed_area:.2f} m²** "
                f"(current per-group sum: **{s.total_wall_area:.2f} m²**, "
                f"delta: **{collapsed_area - s.total_wall_area:+.4f} m²**)\n"
            )
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=DEFAULT_OUTPUTS, help="pipeline-outputs/ root"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="markdown report path"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="if >0, only process N buildings"
    )
    args = parser.parse_args()

    payload_paths = sorted(args.inputs.glob("*/tier_payload.json"))
    if args.limit > 0:
        payload_paths = payload_paths[: args.limit]

    all_buildings: list[BuildingSummary] = []
    flat_buildings: list[BuildingSummary] = []
    for path in payload_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {path}: {exc}")
            continue
        try:
            all_buildings.append(analyse_building(payload, flat_only=False))
            flat_buildings.append(analyse_building(payload, flat_only=True))
        except Exception as exc:
            print(f"error in {path.parent.name}: {exc}")
            continue

    write_report(all_buildings, flat_buildings, args.report)
    print(f"analysed {len(all_buildings)} buildings; report written to {args.report}")


if __name__ == "__main__":
    main()
