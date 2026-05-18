#!/usr/bin/env python3
"""Summarize duplicate/shadow zone artifacts in candidate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import Polygon


def _poly_from_zone(zone: dict) -> Polygon | None:
    ring = zone.get("footprint_xz") or []
    if len(ring) < 3:
        return None
    try:
        poly = Polygon([(float(pt[0]), float(pt[1])) for pt in ring])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0.0:
        return None
    return poly


def _artifact_summary(rows: list[dict], overlap_ratio: float) -> dict:
    n_no_part_zones = 0
    n_shadow_zones = 0
    buildings_with_shadow: set[str] = set()
    overlaps: list[dict] = []
    for row in rows:
        uuid = str(row.get("building_uuid") or "")
        zones = []
        for zone in row.get("zones") or []:
            poly = _poly_from_zone(zone)
            if poly is None:
                continue
            zone = dict(zone)
            zone["_poly"] = poly
            zones.append(zone)
            if not zone.get("part_id"):
                n_no_part_zones += 1
        for zone in zones:
            if zone.get("part_id"):
                continue
            for other in zones:
                if zone["id"] == other["id"] or not other.get("part_id"):
                    continue
                overlap_area = zone["_poly"].intersection(other["_poly"]).area
                min_area = min(zone["_poly"].area, other["_poly"].area)
                if min_area <= 0.0:
                    continue
                ratio = overlap_area / min_area
                if ratio < overlap_ratio:
                    continue
                n_shadow_zones += 1
                buildings_with_shadow.add(uuid)
                overlaps.append(
                    {
                        "building_uuid": uuid,
                        "shadow_zone_id": zone["id"],
                        "real_zone_id": other["id"],
                        "overlap_ratio": round(ratio, 6),
                    }
                )
                break
    return {
        "buildings": len(rows),
        "no_part_zones": n_no_part_zones,
        "shadow_zones": n_shadow_zones,
        "buildings_with_shadow": len(buildings_with_shadow),
        "examples": overlaps[:10],
    }


def _load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "artifact", type=Path, help="candidate_faces/candidates.json artifact"
    )
    ap.add_argument(
        "--compare", type=Path, default=None, help="optional second artifact to compare"
    )
    ap.add_argument("--overlap-ratio", type=float, default=0.80)
    args = ap.parse_args()

    baseline = _artifact_summary(_load_rows(args.artifact), args.overlap_ratio)
    print(json.dumps({"artifact": str(args.artifact), **baseline}, indent=2))

    if args.compare is not None:
        comparison = _artifact_summary(_load_rows(args.compare), args.overlap_ratio)
        print(
            json.dumps(
                {
                    "artifact": str(args.compare),
                    **comparison,
                    "delta": {
                        "no_part_zones": comparison["no_part_zones"]
                        - baseline["no_part_zones"],
                        "shadow_zones": comparison["shadow_zones"]
                        - baseline["shadow_zones"],
                        "buildings_with_shadow": comparison["buildings_with_shadow"]
                        - baseline["buildings_with_shadow"],
                    },
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
