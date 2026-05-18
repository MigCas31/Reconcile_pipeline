"""Plan J Phase 1: read-only diagnostic for flat ceilings that overshoot
the rooms they cover.

User-visible failure mode: a flat ceiling extends past the gable into XZ
areas where there is no actual horizontal ceiling at that Y. The pipeline
emits the flat over a single room (correctly), but the polygon also covers
adjacent rooms whose actual ceiling is much lower. The kink line is the
boundary between rooms-with-this-flat-as-ceiling and rooms-with-a-lower-lid.

Detection signal per flat piece F at Y = flat_y, covering XZ polygon flat_xz:
1. For each room R whose floor footprint intersects flat_xz:
2. Look up R.attic_lid_y from the building's RoofKinks (scan-derived).
3. If flat_y - R.attic_lid_y > LID_OVERSHOOT_TOL_M, the flat's portion over
   R is overgenerous — that area should be clipped from this flat.

Classifications (per flat piece):
- ``room_overshoot`` — at least one room is overshot (intersection area >
   MIN_OVERSHOOT_AREA_M2). Most reliable signal — driven by scan-derived
   kink data, not extrapolation.
- ``correct``       — every covered room has attic_lid_y ≈ flat_y. The
   flat is the actual ceiling for its rooms.
- ``no_kink_data``  — couldn't resolve any room's lid (kinks unavailable
   or room indices not parseable from locator IDs).
- ``degenerate``    — polygon failed validity / area too small.

Run:

    python -m reconcile_tiers.audit.kink_flat_diagnostic --all
    python -m reconcile_tiers.audit.kink_flat_diagnostic --uuid <uuid>

Outputs ``analysis_outputs/kink_flat_<ts>.csv`` and a per-class tally.
The CSV has one row per (flat, overshot-room) pair so Phase 2 can clip the
specific intersection.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = WORKSPACE_ROOT / "pipeline-outputs"
OUT_DIR = WORKSPACE_ROOT / "analysis_outputs"

LID_OVERSHOOT_TOL_M = 0.30
"""If flat_y - room.attic_lid_y exceeds this tolerance, the flat is treated as
overgenerous over that room. 0.30m is the slack that allows flat ceiling at
the wall-top + a small ridge fudge."""

MIN_OVERSHOOT_AREA_M2 = 0.30
"""Don't flag tiny slivers (rounding / room-edge buffers)."""

MIN_FLAT_AREA_M2 = 0.5


def _xz_polygon(corners) -> Polygon | None:
    if len(corners) < 3:
        return None
    pts = [(c["x"], c["z"]) for c in corners]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _y_range(corners) -> tuple[float, float] | None:
    if not corners:
        return None
    ys = [c["y"] for c in corners]
    return (min(ys), max(ys))


def _is_oblique(piece: dict[str, Any]) -> bool:
    plane = piece.get("plane") or {}
    b = plane.get("b", 1.0)
    src = (piece.get("source") or "").lower()
    if "oblique" in src or "merged_coplanar" in src:
        return True
    return abs(b) < 0.95


def _is_flat(piece: dict[str, Any]) -> bool:
    plane = piece.get("plane") or {}
    return abs(plane.get("b", 0.0)) >= 0.95 and (piece.get("source") == "flat_ceiling")


def _oblique_low_edge_centroid(piece: dict[str, Any]) -> tuple[float, float] | None:
    """Return XZ centroid of the lowest-Y edge of the oblique piece. That's
    the eave (or kink-line candidate)."""
    corners = piece.get("corners") or []
    if len(corners) < 3:
        return None
    ys = [c["y"] for c in corners]
    y_lo = min(ys)
    low_pts = [c for c in corners if c["y"] <= y_lo + 0.05]
    if not low_pts:
        return None
    cx = sum(c["x"] for c in low_pts) / len(low_pts)
    cz = sum(c["z"] for c in low_pts) / len(low_pts)
    return (float(cx), float(cz))


def _downhill_direction(plane: dict[str, Any]) -> tuple[float, float] | None:
    """XZ unit vector pointing in the oblique's downhill direction."""
    a = plane.get("a", 0.0)
    c = plane.get("c", 0.0)
    horiz = math.hypot(a, c)
    if horiz < 1e-6:
        return None
    return (-a / horiz, -c / horiz)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def _room_lid_y(room: dict[str, Any]) -> float | None:
    """Approximate ``RoofKinks.attic_lid_y_by_room``: p10 of wall-top Y values
    (drop kneewalls — they sit lower). p10 is robust to a single tall wall.
    """
    wall_tops: list[float] = []
    for wall in room.get("walls") or []:
        ys = [c.get("y") for c in (wall.get("corners") or []) if c.get("y") is not None]
        if not ys:
            continue
        wall_tops.append(max(ys))
    if not wall_tops:
        return None
    return _percentile(wall_tops, 0.10)


def _room_floor_polys(
    rooms: list[dict[str, Any]],
) -> list[tuple[Polygon, dict[str, Any], float | None]]:
    """One entry per room floor piece: (xz_polygon, room_dict, room_lid_y)."""
    out: list[tuple[Polygon, dict[str, Any], float | None]] = []
    for room in rooms:
        lid = _room_lid_y(room)
        for floor in room.get("floor") or []:
            poly = _xz_polygon(floor.get("corners") or [])
            if poly is None:
                continue
            out.append((poly, room, lid))
    return out


def _classify_flat(
    flat: dict[str, Any],
    room_floors: list[tuple[Polygon, dict[str, Any], float | None]],
) -> list[dict[str, Any]]:
    """Return one row per (flat, overshot_room) pair. Empty list means the
    flat is correctly placed (or no kink data was usable).
    """
    flat_poly = _xz_polygon(flat.get("corners") or [])
    if flat_poly is None or flat_poly.area < MIN_FLAT_AREA_M2:
        return [{"classification": "degenerate", "flat_area_xz_m2": 0.0}]

    yr = _y_range(flat.get("corners") or [])
    if yr is None:
        return [
            {"classification": "degenerate", "flat_area_xz_m2": float(flat_poly.area)}
        ]
    flat_y = 0.5 * (yr[0] + yr[1])

    overshoots: list[dict[str, Any]] = []
    any_room_with_lid = False
    correct_coverage = 0.0

    for floor_poly, room, lid in room_floors:
        if not floor_poly.intersects(flat_poly):
            continue
        if lid is None:
            continue
        any_room_with_lid = True
        try:
            inter_area = floor_poly.intersection(flat_poly).area
        except Exception:
            continue
        if inter_area < MIN_OVERSHOOT_AREA_M2:
            continue
        delta = flat_y - lid
        if delta > LID_OVERSHOOT_TOL_M:
            overshoots.append(
                {
                    "classification": "room_overshoot",
                    "flat_area_xz_m2": float(flat_poly.area),
                    "flat_y": flat_y,
                    "room_locator_id": room.get("locator_id"),
                    "room_story": room.get("story"),
                    "room_lid_y": lid,
                    "flat_minus_lid_m": delta,
                    "overshoot_intersection_xz_m2": float(inter_area),
                }
            )
        else:
            correct_coverage += inter_area

    if overshoots:
        return overshoots
    if not any_room_with_lid:
        return [
            {"classification": "no_kink_data", "flat_area_xz_m2": float(flat_poly.area)}
        ]
    return [
        {
            "classification": "correct",
            "flat_area_xz_m2": float(flat_poly.area),
            "flat_y": flat_y,
            "correct_coverage_xz_m2": float(correct_coverage),
        }
    ]


def diagnose_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = payload.get("rooms") or []
    ceilings = payload.get("ceiling") or []
    flats = [piece for piece in ceilings if _is_flat(piece)]
    room_floors = _room_floor_polys(rooms)
    rows: list[dict[str, Any]] = []
    for flat in flats:
        for entry in _classify_flat(flat, room_floors):
            entry["rule"] = "kink_flat"
            entry["locator_id"] = flat.get("locator_id")
            entry["adjacency"] = flat.get("adjacency")
            rows.append(entry)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--uuid", type=str)
    parser.add_argument("--root", type=Path, default=PIPELINE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rating-cohort", type=str, default=None)
    args = parser.parse_args(argv)

    if args.uuid:
        targets = [args.uuid]
    else:
        targets = sorted(p.parent.name for p in args.root.glob("*/tier_payload.json"))
    if args.rating_cohort:
        ratings_path = WORKSPACE_ROOT / ".context" / "roof_ratings.json"
        if ratings_path.exists():
            ratings = json.loads(ratings_path.read_text())
            wanted = {s.strip() for s in args.rating_cohort.split(",") if s.strip()}
            targets = [
                u
                for u in targets
                if str((ratings.get(u) or {}).get("rating")) in wanted
            ]

    out_dir = args.out or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"kink_flat_{timestamp}.csv"

    counts: Counter[str] = Counter()
    by_uuid_overshoot: Counter[str] = Counter()
    by_uuid_overshoot_area: dict[str, float] = {}
    rows_total = 0
    fields = [
        "uuid",
        "rule",
        "classification",
        "locator_id",
        "adjacency",
        "flat_area_xz_m2",
        "flat_y",
        "room_locator_id",
        "room_story",
        "room_lid_y",
        "flat_minus_lid_m",
        "overshoot_intersection_xz_m2",
        "correct_coverage_xz_m2",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for uuid in targets:
            payload_path = args.root / uuid / "tier_payload.json"
            if not payload_path.exists():
                continue
            try:
                payload = json.loads(payload_path.read_text())
            except Exception:
                continue
            for row in diagnose_payload(payload):
                row["uuid"] = uuid
                writer.writerow(row)
                counts[row["classification"]] += 1
                if row["classification"] == "room_overshoot":
                    by_uuid_overshoot[uuid] += 1
                    by_uuid_overshoot_area[uuid] = by_uuid_overshoot_area.get(
                        uuid, 0.0
                    ) + row.get("overshoot_intersection_xz_m2", 0.0)
                rows_total += 1

    print(f"\ncsv: {csv_path}")
    print(f"buildings: {len(targets)}, rows: {rows_total}\n")
    print(f"{'classification':<22} {'count':>6}")
    for cls, n in counts.most_common():
        print(f"{cls:<22} {n:>6}")
    if by_uuid_overshoot:
        print("\n=== top buildings by total overshoot area (m²) ===")
        ranked = sorted(by_uuid_overshoot_area.items(), key=lambda kv: -kv[1])
        for uuid, total_area in ranked[:10]:
            n = by_uuid_overshoot[uuid]
            print(
                f"  {uuid[:8]} {n:>3} (flat,room) overshoot pairs, total "
                f"{total_area:.1f} m²"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
