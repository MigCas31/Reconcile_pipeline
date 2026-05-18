#!/usr/bin/env python3
"""Audit every rendered surface against the per-building ceiling footprint.

For each building we:
  1. Load the XZ footprint from roof_algorithms_py_results.json
     (falling back to the buffered union of room floor polygons).
  2. Walk every surface source the viewer renders (roof flats/obliques,
     ceilings, thermal ceilings, cross-floor gaps, stitch walls) and
     compute how much of its XZ extent lies outside the footprint.
  3. Flag surfaces with `outside_area_m2 >= AREA_THRESHOLD_M2` OR
     `outside_ratio >= RATIO_THRESHOLD`.

Output: stdout summary + /tmp/surfaces_outside_footprint.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_JSON = REPO / "reconcile" / "buildings_3d.json"
ROOF_JSON = REPO / "reconcile" / "roof_algorithms_py_results.json"

FOOTPRINT_BUFFER_M = 0.80  # eave/overhang tolerance on top of the room union
AREA_THRESHOLD_M2 = 1.00
RATIO_THRESHOLD = 0.10


def _poly_from_xz(corners):
    if not corners or len(corners) < 3:
        return None
    pts = [
        (c[0], c[2]) for c in corners if isinstance(c, (list, tuple)) and len(c) >= 3
    ]
    if len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            return None
        return p
    except Exception:
        return None


def _footprint_poly(bldg, py_entry):
    """
    Union of every room's floor polygon across all stories (building ground
    footprint).
    """
    polys = []
    for room in bldg.get("rooms") or []:
        fp = room.get("floor_polygon") or []
        if len(fp) < 3:
            continue
        pts = [(p[0], p[2]) for p in fp]
        try:
            poly = Polygon(pts).buffer(0.3, join_style="mitre")
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            continue
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged


def _iter_surfaces(bldg, py_entry):
    """Yield (kind, locator_id, corners) for every renderable XZ polygon."""
    # roof flats + obliques (roof_algorithms output)
    roof = (py_entry or {}).get("roof_surfaces") or {}
    for i, s in enumerate(roof.get("flat") or []):
        yield ("roof-flat", f"flat:{i}", s.get("corners"))
    for i, s in enumerate(roof.get("oblique") or []):
        yield ("roof-oblique", f"oblique:{i}", s.get("corners"))
    # ceilings
    ceiling = (py_entry or {}).get("ceiling") or {}
    for i, c in enumerate(ceiling.get("flat") or []):
        yield ("ceiling-flat", f"ceiling-flat:{i}", c.get("poly"))
    for i, c in enumerate(ceiling.get("simple_slant") or []):
        yield ("ceiling-simple-slant", f"ceiling-slant:{i}", c.get("poly"))
    for i, c in enumerate(ceiling.get("thermal") or []):
        yield (
            f"thermal:{c.get('kind', '?')}",
            f"thermal:{c.get('kind', '?')}:{i}",
            c.get("poly"),
        )
    # dormers
    for di, d in enumerate((py_entry or {}).get("dormers") or []):
        for ci, cheek in enumerate(d.get("cheeks") or []):
            yield ("dormer-cheek", f"dormer-cheek:{di}:{ci}", cheek.get("corners"))
        if d.get("header"):
            yield ("dormer-header", f"dormer-header:{di}", d["header"].get("corners"))
    # cross-floor gaps (thermal path: only non-skipped)
    thermal_count = len(ceiling.get("thermal") or [])
    has_thermal = thermal_count > 0
    for gi, g in enumerate(bldg.get("cross_floor_gaps") or []):
        if has_thermal and g.get("type") == "cross_story":
            continue
        corners = g.get("ceiling_corners") or g.get("corners")
        yield (f"gap-thermal-{g.get('type')}", f"thermal:gap:{gi}", corners)
    # stitch walls
    for i, sw in enumerate(bldg.get("stitch_walls") or []):
        yield ("wall-stitch", f"stitch:{i}", sw.get("corners"))


def audit_building(bldg, py_entry):
    fp_poly = _footprint_poly(bldg, py_entry)
    if fp_poly is None:
        return None, []
    fp_padded = fp_poly.buffer(FOOTPRINT_BUFFER_M, join_style="mitre")
    offenders = []
    for kind, locator_id, corners in _iter_surfaces(bldg, py_entry):
        sp = _poly_from_xz(corners)
        if sp is None:
            continue
        outside = sp.difference(fp_padded)
        out_area = outside.area
        if out_area <= 0:
            continue
        total = sp.area
        ratio = out_area / total if total > 0 else 0.0
        if out_area >= AREA_THRESHOLD_M2 or ratio >= RATIO_THRESHOLD:
            ob = outside.bounds if not outside.is_empty else None
            offenders.append(
                {
                    "kind": kind,
                    "id": locator_id,
                    "area_m2": round(total, 3),
                    "outside_area_m2": round(out_area, 3),
                    "outside_ratio": round(ratio, 3),
                    "outside_bbox_xz": [round(v, 3) for v in ob] if ob else None,
                }
            )
    return fp_poly.area, offenders


def main() -> int:
    buildings_data = json.loads(BUILDINGS_JSON.read_text())
    buildings = (
        buildings_data.get("buildings")
        if isinstance(buildings_data, dict)
        else buildings_data
    )
    py_all = json.loads(ROOF_JSON.read_text())
    if not isinstance(py_all, dict):
        print("roof_algorithms_py_results.json is not a dict", file=sys.stderr)
        return 1

    total = len(buildings)
    print(f"Scanning {total} buildings…\n")

    per_building = []
    for idx, bldg in enumerate(buildings, 1):
        uuid = bldg.get("uuid")
        if not uuid:
            continue
        py_entry = py_all.get(uuid)
        fp_area, offenders = audit_building(bldg, py_entry)
        per_building.append(
            {
                "uuid": uuid,
                "footprint_area_m2": fp_area,
                "offenders": offenders,
            }
        )
        if offenders:
            worst = max(o["outside_area_m2"] for o in offenders)
            print(
                f"[{idx}/{total}] {uuid}  offenders={len(offenders):3d}  "
                f"worst_outside={worst:.2f} m²"
            )
        elif idx % 25 == 0:
            print(f"[{idx}/{total}] scanned, no offenders so far in this batch…")

    total_offenders = sum(len(b["offenders"]) for b in per_building)
    buildings_with_offenders = sum(1 for b in per_building if b["offenders"])

    # Aggregate by kind
    by_kind = {}
    for b in per_building:
        for o in b["offenders"]:
            by_kind[o["kind"]] = by_kind.get(o["kind"], 0) + 1

    print("\n" + "=" * 66)
    print("SCAN COMPLETE")
    print("=" * 66)
    print(f"Buildings scanned:                 {total}")
    print(f"Buildings with offenders:          {buildings_with_offenders}")
    print(f"Total offending surfaces:          {total_offenders}")
    print()
    print("By kind:")
    for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:30s} {n:6d}")
    print()
    print("Top 15 buildings by offending surfaces:")
    ranked = sorted(per_building, key=lambda b: -len(b["offenders"]))
    for b in ranked[:15]:
        if not b["offenders"]:
            break
        worst = max(o["outside_area_m2"] for o in b["offenders"])
        print(
            f"  {b['uuid']}  n={len(b['offenders']):3d}  worst_outside={worst:.2f} m²  "
            f"footprint={b['footprint_area_m2']:.1f} m²"
        )

    out = Path("/tmp/surfaces_outside_footprint.json")
    out.write_text(
        json.dumps(
            {
                "thresholds": {
                    "area_m2": AREA_THRESHOLD_M2,
                    "ratio": RATIO_THRESHOLD,
                    "footprint_buffer_m": FOOTPRINT_BUFFER_M,
                },
                "per_building": per_building,
            },
            indent=2,
        )
    )
    print(f"\nDetails saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
