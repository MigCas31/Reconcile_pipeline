"""Smoke test: run v2 sidecar on a single building and inspect seam pieces.

Streams the heavy JSON inputs (v3 results is ~2GB) so this is fast for a
single UUID. Prints a summary of any `intersection_seam` pieces produced.

Usage:
  python scripts/smoke_intersection_seams.py [UUID]

Default UUID is the L-shape target building 16784bad-2cd9-4f4c-bb26-60355981cfe2.
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


def main() -> None:
    uuid = sys.argv[1] if len(sys.argv) > 1 else "16784bad-2cd9-4f4c-bb26-60355981cfe2"
    print(f"loading building {uuid} ...")
    building = _load_building(uuid)
    print("loading roof result ...")
    roof_result = _load_roof_result(uuid)
    print("loading v3 entry ...")
    v3_entry = _load_v3_entry(uuid)
    print("loading ridge-eave entry ...")
    ridge_eave_entry = _load_ridge_eave_entry(uuid)

    roof_results = {uuid: roof_result}
    v3_results_by_uuid = {uuid: v3_entry} if v3_entry else None
    ridge_eave_scores_by_uuid = {uuid: ridge_eave_entry} if ridge_eave_entry else None

    print("scoring ...")
    result = score_building_v2(
        building,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
        v3_results_by_uuid=v3_results_by_uuid,
    )

    seam_rows = [
        r for r in result.split_piece_rows if r.get("piece_role") == "intersection_seam"
    ]
    by_role: dict[str, int] = {}
    for r in result.split_piece_rows:
        role = str(r.get("piece_role") or "")
        by_role[role] = by_role.get(role, 0) + 1
    print(f"\n=== building {uuid} ===")
    print(f"total split pieces: {len(result.split_piece_rows)}")
    print(f"by piece_role: {by_role}")
    print(f"intersection_seam pieces: {len(seam_rows)}")

    if seam_rows:
        print("\nseam pieces:")
        for r in seam_rows:
            print(
                f"  target={r['target_element_id']}  "
                f"piece_id={r['piece_id']}  "
                f"area_xz_m2={r['area_xz_m2']:.2f}  "
                f"partner={r.get('pair_partner_target_id')}  "
                f"disputed_in_piece={r.get('pair_disputed_overlap_in_piece_m2')}  "
                f"partner_evidence={r.get('pair_partner_evidence_area_m2')}"
            )


if __name__ == "__main__":
    main()
