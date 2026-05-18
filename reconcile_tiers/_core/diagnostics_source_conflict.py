"""Diagnose oblique-vs-raw plane disagreement in tier payload ceilings.

For each `pipeline-outputs/<uuid>/tier_payload.json`, groups `payload.ceiling[]`
by room and reports the angle between the dominant `roof_arrangement*` plane
and any coexisting `raw_fallback` plane.

Run: `python -m reconcile_tiers._core.diagnostics_source_conflict
[--root pipeline-outputs] [--out pipeline-outputs/_diagnostics/source_conflict.json]`
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

from reconcile_tiers._core.shapely2 import make_valid

ARRANGEMENT_SOURCES = {"roof_arrangement", "roof_arrangement_attic"}
RAW_SOURCE = "raw_fallback"
FLAT_SOURCE = "flat_emit"
ATTIC_LID_SOURCE = "attic_flat_lid"

LOCATOR_RAW_RE = re.compile(r"::tier-ceiling-raw::(\d+):")
LOCATOR_FLAT_RE = re.compile(r"::tier-ceiling-flat::(\d+)")
ARRANGEMENT_CELL_RE = re.compile(r"^cell:\d+:room:(\d+):\d+$")

ANGLE_BINS_DEG = (1.0, 3.0, 5.0, 10.0, 20.0, 45.0, 90.0)
OFFSET_BINS_M = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def _room_idx_for_piece(piece: dict[str, Any]) -> int | None:
    """Recover the originating room index for a ceiling piece, or None for
    building-scope pieces (e.g. roof-arrangement-attic-full).
    """
    cell = piece.get("arrangement_cell_id")
    if cell:
        match = ARRANGEMENT_CELL_RE.match(cell)
        if match:
            return int(match.group(1))
    locator = piece.get("locator_id", "")
    for pattern in (LOCATOR_RAW_RE, LOCATOR_FLAT_RE):
        match = pattern.search(locator)
        if match:
            return int(match.group(1))
    return None


def _xz_polygon(corners: list[dict[str, float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = make_valid(Polygon([(float(c["x"]), float(c["z"])) for c in corners]))
    except Exception:
        return None
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _signed_normal(plane: dict[str, float]) -> tuple[float, float, float, float]:
    a = float(plane["a"])
    b = float(plane["b"])
    c = float(plane["c"])
    d = float(plane["d"])
    norm = math.sqrt(a * a + b * b + c * c) or 1.0
    a, b, c, d = a / norm, b / norm, c / norm, d / norm
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    return a, b, c, d


def _angle_deg(plane_a: dict, plane_b: dict) -> float:
    ax, ay, az, _ = _signed_normal(plane_a)
    bx, by, bz, _ = _signed_normal(plane_b)
    dot = max(-1.0, min(1.0, ax * bx + ay * by + az * bz))
    return math.degrees(math.acos(abs(dot)))


def _offset_at_centroid(plane_a: dict, plane_b: dict, poly: Polygon) -> float:
    """Distance between plane A's height and plane B's height at poly's
    centroid. If either plane is near-vertical, returns NaN.
    """
    ax, ay, az, ad = _signed_normal(plane_a)
    bx, by, bz, bd = _signed_normal(plane_b)
    if abs(ay) < 1e-3 or abs(by) < 1e-3:
        return float("nan")
    cx = float(poly.centroid.x)
    cz = float(poly.centroid.y)
    y_a = (ad - ax * cx - az * cz) / ay
    y_b = (bd - bx * cx - bz * cz) / by
    return abs(y_a - y_b)


def _xz_overlap_area(poly_a: Polygon, poly_b: Polygon) -> float:
    inter = poly_a.intersection(poly_b)
    if inter.is_empty:
        return 0.0
    if isinstance(inter, Polygon):
        return float(inter.area)
    return float(sum(part.area for part in getattr(inter, "geoms", [])))


def _bin_index(value: float, bins: tuple[float, ...]) -> int:
    if math.isnan(value):
        return -1
    for idx, edge in enumerate(bins):
        if value <= edge:
            return idx
    return len(bins)


def _bin_label(idx: int, bins: tuple[float, ...], unit: str) -> str:
    if idx == -1:
        return "nan"
    if idx == 0:
        return f"<= {bins[0]}{unit}"
    if idx == len(bins):
        return f"> {bins[-1]}{unit}"
    return f"<= {bins[idx]}{unit}"


def analyse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rooms_by_idx: dict[int, dict[str, Any]] = {}
    for room in payload.get("rooms", []):
        locator = room.get("locator_id", "")
        match = re.search(r"::tier-room::(\d+)", locator)
        if match:
            rooms_by_idx[int(match.group(1))] = room

    by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    building_scope: list[dict[str, Any]] = []
    for piece in payload.get("ceiling", []):
        room_idx = _room_idx_for_piece(piece)
        if room_idx is None:
            building_scope.append(piece)
            continue
        by_room[room_idx].append(piece)

    conflicts: list[dict[str, Any]] = []
    arrangement_only_rooms = 0
    raw_only_rooms = 0
    for room_idx, pieces in sorted(by_room.items()):
        sources = {p["source"] for p in pieces}
        has_arr = bool(sources & ARRANGEMENT_SOURCES)
        has_raw = RAW_SOURCE in sources
        if has_arr and not has_raw:
            arrangement_only_rooms += 1
            continue
        if has_raw and not has_arr:
            raw_only_rooms += 1
            continue
        if not (has_arr and has_raw):
            continue

        arr_pieces = [p for p in pieces if p["source"] in ARRANGEMENT_SOURCES]
        raw_pieces = [p for p in pieces if p["source"] == RAW_SOURCE]
        arr_polys = [(_xz_polygon(p["corners"]), p) for p in arr_pieces]
        arr_polys = [(poly, p) for poly, p in arr_polys if poly is not None]
        if not arr_polys:
            continue
        # Dominant arrangement = largest xz area among arrangement pieces.
        dominant_poly, dominant_piece = max(arr_polys, key=lambda pair: pair[0].area)
        dominant_plane = dominant_piece["plane"]

        for raw in raw_pieces:
            raw_poly = _xz_polygon(raw["corners"])
            if raw_poly is None:
                continue
            angle = _angle_deg(dominant_plane, raw["plane"])
            offset = _offset_at_centroid(dominant_plane, raw["plane"], raw_poly)
            overlap = _xz_overlap_area(dominant_poly, raw_poly)
            conflicts.append(
                {
                    "room_idx": room_idx,
                    "story": rooms_by_idx.get(room_idx, {}).get("story"),
                    "raw_locator": raw["locator_id"],
                    "dominant_locator": dominant_piece["locator_id"],
                    "dominant_source": dominant_piece["source"],
                    "angle_deg": angle,
                    "centroid_offset_m": offset,
                    "raw_area_m2": float(raw_poly.area),
                    "dominant_area_m2": float(dominant_poly.area),
                    "xz_overlap_m2": overlap,
                    "snappable": angle <= 3.0
                    and not math.isnan(offset)
                    and offset <= 0.05,
                }
            )

    source_counts: dict[str, int] = defaultdict(int)
    for piece in payload.get("ceiling", []):
        source_counts[piece["source"]] += 1

    return {
        "uuid": payload.get("uuid"),
        "n_rooms": len(rooms_by_idx),
        "n_ceiling_pieces": len(payload.get("ceiling", [])),
        "source_counts": dict(source_counts),
        "n_arrangement_only_rooms": arrangement_only_rooms,
        "n_raw_only_rooms": raw_only_rooms,
        "n_conflict_rooms": len({c["room_idx"] for c in conflicts}),
        "conflicts": conflicts,
    }


def aggregate(per_payload: Iterable[dict[str, Any]]) -> dict[str, Any]:
    angle_hist: dict[str, int] = defaultdict(int)
    offset_hist: dict[str, int] = defaultdict(int)
    snappable = 0
    not_snappable = 0
    total_conflicts = 0
    buildings_with_conflicts = 0
    for record in per_payload:
        if record["conflicts"]:
            buildings_with_conflicts += 1
        for conflict in record["conflicts"]:
            total_conflicts += 1
            angle_hist[
                _bin_label(
                    _bin_index(conflict["angle_deg"], ANGLE_BINS_DEG),
                    ANGLE_BINS_DEG,
                    "deg",
                )
            ] += 1
            offset_hist[
                _bin_label(
                    _bin_index(conflict["centroid_offset_m"], OFFSET_BINS_M),
                    OFFSET_BINS_M,
                    "m",
                )
            ] += 1
            if conflict["snappable"]:
                snappable += 1
            else:
                not_snappable += 1
    return {
        "buildings_with_conflicts": buildings_with_conflicts,
        "total_conflicts": total_conflicts,
        "snappable": snappable,
        "not_snappable": not_snappable,
        "angle_histogram": dict(angle_hist),
        "offset_histogram": dict(offset_hist),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("pipeline-outputs"),
        help="Root containing per-uuid subdirectories with tier_payload.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pipeline-outputs/_diagnostics/source_conflict.json"),
        help="Output JSON path for the per-building report.",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for uuid_dir in sorted(args.root.iterdir()):
        if not uuid_dir.is_dir() or uuid_dir.name.startswith("_"):
            continue
        payload_path = uuid_dir / "tier_payload.json"
        if not payload_path.is_file():
            continue
        try:
            payload = json.loads(payload_path.read_text())
        except Exception as exc:
            skipped.append(f"{payload_path}: {exc}")
            continue
        record = analyse_payload(payload)
        record["uuid_dir"] = uuid_dir.name
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "buildings_examined": len(records),
        "aggregate": aggregate(records),
        "per_building": records,
    }
    if skipped:
        summary["skipped"] = skipped
    args.out.write_text(json.dumps(summary, indent=2, default=float))

    agg = summary["aggregate"]
    print(f"\n=== {summary['buildings_examined']} buildings ===")
    print(f"  buildings_with_conflicts: {agg['buildings_with_conflicts']}")
    print(f"  total room-conflicts:     {agg['total_conflicts']}")
    print(f"    snappable (<=3deg, <=5cm): {agg['snappable']}")
    print(f"    not snappable:             {agg['not_snappable']}")
    print("  angle histogram:")
    for k in sorted(agg["angle_histogram"], key=lambda s: (s != "nan", s)):
        print(f"    {k:>14s}: {agg['angle_histogram'][k]}")
    print("  centroid-offset histogram:")
    for k in sorted(agg["offset_histogram"], key=lambda s: (s != "nan", s)):
        print(f"    {k:>14s}: {agg['offset_histogram'][k]}")
    if skipped:
        print(f"\n  skipped {len(skipped)} payloads (see {args.out} 'skipped')")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
