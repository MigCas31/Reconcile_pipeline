"""Audit wall_clipped rooms whose own floor polygon already agreed with the
walls' original bottom y. Those rooms are split-level / extension rooms whose
walls were clipped up to the story-aggregate floor, breaking floor/wall
correspondence (see reconcile/extract3d/overlaps.py::clip_walls_to_story_bounds).

Usage:
    python scripts/audit_wall_clip_splitlevel.py [--input <path>] [--top N]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reconcile" / "buildings_3d.json"
FLOOR_MATCH_TOL = 0.05
DELTA_BINS = [0.00, 0.15, 0.30, 0.50, 1.00, float("inf")]


def _min_y(corners):
    return min(c[1] for c in corners)


def _scan_room(room):
    walls = room.get("walls_computed") or []
    clipped = [
        w
        for w in walls
        if w.get("wall_clipped") and len(w.get("corners_original") or []) >= 3
    ]
    if not clipped:
        return None

    floor_poly = room.get("floor_polygon") or []
    if len(floor_poly) < 3:
        return None
    floor_mean_y = sum(c[1] for c in floor_poly) / len(floor_poly)

    orig_bottoms = [_min_y(w["corners_original"]) for w in clipped]
    new_bottoms = [
        _min_y(w["corners"]) for w in clipped if len(w.get("corners") or []) >= 3
    ]
    orig_bottom_median = statistics.median(orig_bottoms)
    new_bottom_median = (
        statistics.median(new_bottoms) if new_bottoms else orig_bottom_median
    )

    matches = abs(floor_mean_y - orig_bottom_median) <= FLOOR_MATCH_TOL
    return {
        "n_walls_clipped": len(clipped),
        "n_walls_total": len(walls),
        "floor_mean_y": floor_mean_y,
        "orig_bottom_median": orig_bottom_median,
        "new_bottom_median": new_bottom_median,
        "delta_median": new_bottom_median - orig_bottom_median,
        "flagged": matches,
    }


def _bin_index(value):
    for idx in range(len(DELTA_BINS) - 1):
        if DELTA_BINS[idx] <= value < DELTA_BINS[idx + 1]:
            return idx
    return len(DELTA_BINS) - 2


def _bin_label(idx):
    lo, hi = DELTA_BINS[idx], DELTA_BINS[idx + 1]
    hi_txt = "inf" if hi == float("inf") else f"{hi:.2f}"
    return f"[{lo:.2f},{hi_txt})"


def run(input_path: Path, top: int) -> int:
    data = json.loads(input_path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"expected list of buildings in {input_path}")

    rows = []
    for building in data:
        uuid = building.get("uuid") or ""
        address = building.get("address") or ""
        for idx, room in enumerate(building.get("rooms") or []):
            info = _scan_room(room)
            if info is None:
                continue
            rows.append(
                {
                    "uuid": uuid,
                    "address": address,
                    "story": room.get("story"),
                    "room_index": idx,
                    **info,
                }
            )

    flagged = [r for r in rows if r["flagged"]]
    flagged_buildings = {r["uuid"] for r in flagged}

    print(f"input: {input_path}")
    print(f"rooms with wall_clipped walls: {len(rows)}")
    print(f"flagged rooms (floor polygon agreed with orig wall bottom): {len(flagged)}")
    print(f"flagged buildings: {len(flagged_buildings)}")

    if flagged:
        bins = [0] * (len(DELTA_BINS) - 1)
        for r in flagged:
            bins[_bin_index(r["delta_median"])] += 1
        print("\ndelta_median histogram (new_bottom - orig_bottom, m):")
        for i, count in enumerate(bins):
            print(f"  {_bin_label(i)}: {count}")

        ratio_bins = {"0-25": 0, "25-50": 0, "50-75": 0, "75-99": 0, "100": 0}
        for r in flagged:
            ratio = (
                r["n_walls_clipped"] / r["n_walls_total"] if r["n_walls_total"] else 0
            )
            if ratio >= 1.0:
                ratio_bins["100"] += 1
            elif ratio >= 0.75:
                ratio_bins["75-99"] += 1
            elif ratio >= 0.5:
                ratio_bins["50-75"] += 1
            elif ratio >= 0.25:
                ratio_bins["25-50"] += 1
            else:
                ratio_bins["0-25"] += 1
        print("\nclipped-wall-ratio histogram (n_clipped / n_walls):")
        for k, v in ratio_bins.items():
            print(f"  {k}%: {v}")

        flagged_sorted = sorted(flagged, key=lambda r: r["delta_median"], reverse=True)
        print(f"\ntop {min(top, len(flagged_sorted))} flagged rows by delta_median:")
        for r in flagged_sorted[:top]:
            print(
                f"  {r['uuid']} story={r['story']} room={r['room_index']:>2}"
                f" clipped={r['n_walls_clipped']}/{r['n_walls_total']}"
                f" floor_y={r['floor_mean_y']:+.3f}"
                f" orig_bottom={r['orig_bottom_median']:+.3f}"
                f" new_bottom={r['new_bottom_median']:+.3f}"
                f" delta={r['delta_median']:+.3f}"
                f" — {r['address']}"
            )

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    raise SystemExit(run(args.input, args.top))


if __name__ == "__main__":
    main()
