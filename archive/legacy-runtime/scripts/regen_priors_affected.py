"""Regenerate tier payloads for the 16 priors-filter-affected buildings and
diff vs the saved pre-filter copy.

For each UUID:
  1. Move pipeline-outputs/<uuid>/tier_payload[_v2].json -> .prefilter (once;
     idempotent — keeps the original snapshot for diffing).
  2. Re-run `build_tier_payload(..., envelope_version="v1"|"v2")` and write the
     new output.
  3. Compute side-by-side metrics: number of obliques (parsed from arrangement
     cell ids), number of ceiling pieces, number of gap pieces, classified
     roof_type, total floor-vs-ceiling gap area in xz.

Output: pipeline-outputs/_diagnostics/priors_regen_compare.json plus a printed
table.
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.payload.schema import payload_to_dict

warnings.filterwarnings("ignore")

UUIDS = [
    "1f03f6e0-dfe6-4b25-bb45-f44ad146c0a3",
    "107e8496-9bff-42bb-b776-720f44b70e55",
    "1d26eda3-c927-4cdf-a3b0-218036828a55",
    "287808db-3826-4351-b9a1-6f9831bdc870",
    "49762ea7-8131-4141-ad52-e8012b46f630",
    "5831141f-829c-4adb-9bae-289c2140c72d",
    "5a97cb51-78cd-40c7-9dd5-423bdea306a6",
    "66a72e63-8b3c-4e57-977b-32f5119a9d09",
    "72122129-7ee2-4a14-a645-23d44df3d2b5",
    "873952bc-1159-43dd-b4bd-6535a55c57cf",
    "8c0ef0cf-51a7-4ac0-8bbf-735f88a799cc",
    "9bdb3966-1a42-44b8-a033-07cdcab74bbb",
    "a317a543-06cf-4f1b-97d2-139c26c1cb13",
    "a6cb04fa-e84a-4641-a667-b4dd05dd7d41",
    "ad0ee87f-5986-4d5e-9aa9-3bf1a5acf789",
    "e661e7b6-303d-415c-b378-2d9dd2fbfd6f",
]

ROOT = Path("pipeline-outputs")
SCAN_ROOT = Path(".scan-cache")


def _xz_polygon(corners):
    if len(corners) < 3:
        return None
    try:
        poly = make_valid(Polygon([(float(c["x"]), float(c["z"])) for c in corners]))
    except Exception:
        return None
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _watertightness_gap_m2(payload):
    rooms_by_idx = {}
    import re

    for room in payload.get("rooms", []):
        m = re.search(r"::tier-room::(\d+)", room.get("locator_id", ""))
        if m:
            rooms_by_idx[int(m.group(1))] = room
    coverage = []
    for piece in payload.get("ceiling", []):
        poly = _xz_polygon(piece["corners"])
        if poly is not None:
            coverage.append(poly)
    for piece in payload.get("gable_closures", []):
        poly = _xz_polygon(piece["corners"])
        if poly is not None:
            coverage.append(poly)
    for piece in payload.get("dormer_faces", []):
        poly = _xz_polygon(piece["corners"])
        if poly is not None:
            coverage.append(poly)
    for piece in payload.get("visual_shells", []):
        poly = _xz_polygon(piece["corners"])
        if poly is not None:
            coverage.append(poly)
    cov_union = make_valid(unary_union(coverage)) if coverage else None
    total_floor = 0.0
    total_gap = 0.0
    for room in rooms_by_idx.values():
        floor = room.get("floor", {})
        floor_corners = floor.get("corners") or floor.get("polygon")
        if not floor_corners:
            continue
        floor_poly = _xz_polygon(floor_corners)
        if floor_poly is None:
            continue
        total_floor += floor_poly.area
        if cov_union is None:
            total_gap += floor_poly.area
            continue
        try:
            void = make_valid(floor_poly.difference(cov_union))
        except Exception:
            continue
        if not void.is_empty:
            total_gap += void.area
    return total_floor, total_gap


def _summarize(payload):
    cls = payload.get("classification", {}) or {}
    floor, gap = _watertightness_gap_m2(payload)
    arr_idxs = set()
    import re

    for piece in payload.get("ceiling", []):
        cell = piece.get("arrangement_cell_id") or ""
        m = re.match(r"^cell:(\d+)", cell)
        if m:
            arr_idxs.add(int(m.group(1)))
    return {
        "roof_type": cls.get("roof_type"),
        "n_oblique_classify": cls.get("n_oblique"),
        "n_oblique_in_payload": len(arr_idxs),
        "n_ceiling": len(payload.get("ceiling", [])),
        "n_gap": len(payload.get("gaps", [])),
        "floor_area_m2": round(floor, 2),
        "gap_area_m2": round(gap, 2),
        "gap_fraction": round(gap / floor, 4) if floor > 0 else None,
    }


def _backup_and_load_pre(uuid: str, filename: str):
    src = ROOT / uuid / filename
    pre = ROOT / uuid / f"{filename}.prefilter"
    if not pre.exists() and src.exists():
        shutil.copy2(src, pre)
    if pre.exists():
        return json.loads(pre.read_text())
    return None


def main() -> int:
    rows = []
    for uuid in UUIDS:
        for variant, filename in (
            ("v1", "tier_payload.json"),
            ("v2", "tier_payload_v2.json"),
        ):
            pre = _backup_and_load_pre(uuid, filename)
            try:
                new_payload = build_tier_payload(
                    uuid, ROOT, SCAN_ROOT, envelope_version=variant
                )
            except Exception as exc:
                rows.append(
                    {
                        "uuid": uuid,
                        "variant": variant,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            new_dict = payload_to_dict(new_payload)
            (ROOT / uuid / filename).write_text(
                json.dumps(new_dict, indent=2, sort_keys=True) + "\n"
            )
            row = {
                "uuid": uuid,
                "variant": variant,
                "pre": _summarize(pre) if pre else None,
                "post": _summarize(new_dict),
            }
            rows.append(row)
            pre = row["pre"]
            post = row["post"]
            obl_pre = pre["n_oblique_classify"] if pre else "?"
            type_pre = pre["roof_type"] if pre else "?"
            ceil_pre = pre["n_ceiling"] if pre else "?"
            gap_pre = pre["gap_area_m2"] if pre else "?"
            print(
                f"  {uuid} ({variant}): "
                f"obl {obl_pre}->{post['n_oblique_classify']} "
                f"type {type_pre}->{post['roof_type']} "
                f"ceil {ceil_pre}->{post['n_ceiling']} "
                f"gap {gap_pre}->{post['gap_area_m2']} m²"
            )

    out = ROOT / "_diagnostics" / "priors_regen_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
