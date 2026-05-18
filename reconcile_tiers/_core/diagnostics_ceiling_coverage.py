"""Diagnose post-assembly ceiling water-tightness in tier payloads.

For each `pipeline-outputs/<uuid>/tier_payload[_v2].json`, compute per-room
the area of the room's floor polygon that has no overhead ceiling piece
covering it (in xz). Output a per-building report plus a histogram of gap
fractions.

A "ceiling" for this purpose is the union of:
  - `payload.ceiling[]` xz polygons
  - `payload.gable_closures[]` xz polygons
  - `payload.dormer_faces[]` xz polygons (face kind == ceiling)
  - `payload.visual_shells[]` xz polygons (v2 only)

Run:
  python -m reconcile_tiers._core.diagnostics_ceiling_coverage \\
    [--root pipeline-outputs] \\
    [--out pipeline-outputs/_diagnostics/ceiling_coverage.json]
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
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid

ROOM_LOCATOR_RE = re.compile(r"::tier-room::(\d+)")

GAP_FRACTION_BINS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50)
ABS_AREA_BINS_M2 = (0.05, 0.25, 1.0, 5.0, 20.0)


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


def _xz_union_with_holes(corner_lists: list[list[dict[str, float]]]) -> Polygon | None:
    polys = [
        poly
        for poly in (_xz_polygon(corners) for corners in corner_lists)
        if poly is not None
    ]
    if not polys:
        return None
    union = make_valid(unary_union(polys))
    if union.is_empty:
        return None
    return union


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
        match = ROOM_LOCATOR_RE.search(room.get("locator_id", ""))
        if match:
            rooms_by_idx[int(match.group(1))] = room

    ceiling_corner_lists = [piece["corners"] for piece in payload.get("ceiling", [])]
    closure_corner_lists = [
        piece["corners"] for piece in payload.get("gable_closures", [])
    ]
    dormer_corner_lists = [
        piece["corners"]
        for piece in payload.get("dormer_faces", [])
        if piece.get("kind") in (None, "ceiling", "horizontal_lid")
    ]
    shell_corner_lists = [
        piece["corners"]
        for piece in payload.get("visual_shells", [])
        if piece.get("tag") in (None, "shell_outer", "shell_inner")
    ]

    coverage_corner_lists = (
        ceiling_corner_lists
        + closure_corner_lists
        + dormer_corner_lists
        + shell_corner_lists
    )
    coverage_union = _xz_union_with_holes(coverage_corner_lists)

    rooms_report: list[dict[str, Any]] = []
    total_room_area = 0.0
    total_gap_area = 0.0
    gappy_rooms = 0

    for room_idx, room in sorted(rooms_by_idx.items()):
        floor = room.get("floor", {})
        floor_corners = floor.get("corners") or floor.get("polygon")
        if not floor_corners:
            continue
        floor_poly = _xz_polygon(floor_corners)
        if floor_poly is None:
            continue
        total_room_area += float(floor_poly.area)
        if coverage_union is None:
            void = floor_poly
        else:
            try:
                void = make_valid(floor_poly.difference(coverage_union))
            except Exception:
                void = floor_poly
        gap_area = float(void.area) if not void.is_empty else 0.0
        gap_fraction = gap_area / floor_poly.area if floor_poly.area > 0 else 0.0
        total_gap_area += gap_area
        if gap_area > 0.05:  # 5 cm² is below scan precision; ignore micro-slivers
            gappy_rooms += 1
            gap_components: list[float] = []
            if isinstance(void, Polygon):
                gap_components.append(float(void.area))
            else:
                gap_components.extend(
                    float(part.area) for part in getattr(void, "geoms", [])
                )
            rooms_report.append(
                {
                    "room_idx": room_idx,
                    "story": room.get("story"),
                    "floor_area_m2": float(floor_poly.area),
                    "gap_area_m2": gap_area,
                    "gap_fraction": gap_fraction,
                    "gap_components": sorted(gap_components, reverse=True)[:8],
                    "n_gap_components": len(gap_components),
                }
            )

    return {
        "uuid": payload.get("uuid"),
        "n_rooms": len(rooms_by_idx),
        "n_gappy_rooms": gappy_rooms,
        "total_room_area_m2": total_room_area,
        "total_gap_area_m2": total_gap_area,
        "total_gap_fraction": (total_gap_area / total_room_area)
        if total_room_area > 0
        else 0.0,
        "rooms": rooms_report,
    }


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    fraction_hist: dict[str, int] = defaultdict(int)
    area_hist: dict[str, int] = defaultdict(int)
    buildings_with_gaps = 0
    total_rooms = 0
    total_gappy_rooms = 0
    sum_room_area = 0.0
    sum_gap_area = 0.0
    for record in records:
        if record["n_gappy_rooms"] > 0:
            buildings_with_gaps += 1
        total_rooms += record["n_rooms"]
        total_gappy_rooms += record["n_gappy_rooms"]
        sum_room_area += record["total_room_area_m2"]
        sum_gap_area += record["total_gap_area_m2"]
        for room in record["rooms"]:
            fraction_hist[
                _bin_label(
                    _bin_index(room["gap_fraction"], GAP_FRACTION_BINS),
                    GAP_FRACTION_BINS,
                    "",
                )
            ] += 1
            area_hist[
                _bin_label(
                    _bin_index(room["gap_area_m2"], ABS_AREA_BINS_M2),
                    ABS_AREA_BINS_M2,
                    "m2",
                )
            ] += 1
    return {
        "buildings_with_gaps": buildings_with_gaps,
        "total_rooms": total_rooms,
        "total_gappy_rooms": total_gappy_rooms,
        "sum_room_area_m2": sum_room_area,
        "sum_gap_area_m2": sum_gap_area,
        "global_gap_fraction": (sum_gap_area / sum_room_area)
        if sum_room_area > 0
        else 0.0,
        "fraction_histogram": dict(fraction_hist),
        "area_histogram": dict(area_hist),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pipeline-outputs/_diagnostics/ceiling_coverage.json"),
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=10,
        help="Print the N buildings with the largest absolute gap area.",
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
    print(f"  buildings_with_gaps:    {agg['buildings_with_gaps']}")
    print(f"  total rooms:            {agg['total_rooms']}")
    print(f"  rooms with gaps:        {agg['total_gappy_rooms']}")
    print(f"  sum room area:          {agg['sum_room_area_m2']:.1f} m²")
    print(f"  sum gap area:           {agg['sum_gap_area_m2']:.2f} m²")
    print(f"  global gap fraction:    {agg['global_gap_fraction']:.4%}")
    print("  per-room gap-fraction histogram:")
    for k in sorted(agg["fraction_histogram"], key=lambda s: (s != "nan", s)):
        print(f"    {k:>14s}: {agg['fraction_histogram'][k]}")
    print("  per-room absolute gap-area histogram:")
    for k in sorted(agg["area_histogram"], key=lambda s: (s != "nan", s)):
        print(f"    {k:>14s}: {agg['area_histogram'][k]}")
    worst = sorted(
        summary["per_building"],
        key=lambda r: r["total_gap_area_m2"],
        reverse=True,
    )[: args.worst]
    print(f"  top {args.worst} buildings by absolute gap area:")
    for record in worst:
        if record["total_gap_area_m2"] <= 0:
            continue
        print(
            f"    {record['uuid_dir']}: gap={record['total_gap_area_m2']:.2f} m² "
            f"({record['total_gap_fraction']:.2%}), "
            f"{record['n_gappy_rooms']}/{record['n_rooms']} rooms"
        )
    if skipped:
        print(f"\n  skipped {len(skipped)} payloads (see {args.out} 'skipped')")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
