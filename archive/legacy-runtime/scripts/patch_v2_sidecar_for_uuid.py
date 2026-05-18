"""Re-score a single building with the v2 sidecar and patch its rows into the
existing `.context/raw_ceiling_plane_scorer_v2_full/plane_extent_splits.json`.

This avoids re-loading the 2 GB v3 results file for every building when we just
want to refresh one UUID (e.g. to surface new intersection_seam pieces in the
viewer). All other buildings in the JSON are left untouched.

Usage:
  python scripts/patch_v2_sidecar_for_uuid.py [UUID ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ijson

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import prototype_raw_ceiling_plane_scorer as legacy
from scripts.raw_ceiling_plane_scorer_v2.runner import score_building_v2

REPO = Path(__file__).resolve().parent.parent
SIDECAR_JSON = (
    REPO / ".context" / "raw_ceiling_plane_scorer_v2_full" / "plane_extent_splits.json"
)


def _load_building(uuid: str) -> dict:
    with open(legacy.BUILDINGS_PATH) as fh:
        for building in ijson.items(fh, "item"):
            if building.get("uuid") == uuid:
                return json.loads(json.dumps(building, default=float))
    raise SystemExit(f"Building {uuid} not in {legacy.BUILDINGS_PATH}")


def _load_roof_result(uuid: str) -> dict:
    with open(legacy.ROOF_RESULTS_PATH, "rb") as fh:
        for key, value in ijson.kvitems(fh, ""):
            if key == uuid:
                return json.loads(json.dumps(value, default=float))
    raise SystemExit(f"Roof result {uuid} not in {legacy.ROOF_RESULTS_PATH}")


def _load_v3_entry(uuid: str) -> dict | None:
    if not legacy.V3_RESULTS_PATH.exists():
        return None
    with open(legacy.V3_RESULTS_PATH, "rb") as fh:
        for entry in ijson.items(fh, "item"):
            if str(entry.get("building_uuid")) == uuid:
                return json.loads(json.dumps(entry, default=float))
    return None


def _load_ridge_eave_entry(uuid: str) -> dict | None:
    if not legacy.RIDGE_EAVE_SCORES_PATH.exists():
        return None
    with open(legacy.RIDGE_EAVE_SCORES_PATH) as fh:
        payload = json.load(fh)
    for entry in payload.get("buildings") or []:
        if str(entry.get("building_uuid")) == uuid:
            return entry
    return None


def patch_uuid(uuid: str, sidecar: dict) -> int:
    print(f"== {uuid} ==")
    print("  loading inputs ...")
    building = _load_building(uuid)
    roof_result = _load_roof_result(uuid)
    v3_entry = _load_v3_entry(uuid)
    ridge_eave_entry = _load_ridge_eave_entry(uuid)

    print("  scoring ...")
    result = score_building_v2(
        building,
        {uuid: roof_result},
        ridge_eave_scores_by_uuid={uuid: ridge_eave_entry}
        if ridge_eave_entry
        else None,
        v3_results_by_uuid={uuid: v3_entry} if v3_entry else None,
    )

    rows = result.split_piece_rows
    sidecar.setdefault("buildings", {})[uuid] = rows
    seam_count = sum(1 for r in rows if r.get("piece_role") == "intersection_seam")
    print(f"  wrote {len(rows)} pieces ({seam_count} intersection_seam)")
    return seam_count


def main() -> None:
    uuids = sys.argv[1:] or ["16784bad-2cd9-4f4c-bb26-60355981cfe2"]

    print(f"loading existing sidecar from {SIDECAR_JSON} ...")
    with SIDECAR_JSON.open() as fh:
        sidecar = json.load(fh)

    total_seams = 0
    for uuid in uuids:
        total_seams += patch_uuid(uuid, sidecar)

    sidecar["available"] = bool(sidecar.get("buildings"))
    print(f"writing patched sidecar ({total_seams} new seam pieces total) ...")
    SIDECAR_JSON.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print("done.")


if __name__ == "__main__":
    main()
