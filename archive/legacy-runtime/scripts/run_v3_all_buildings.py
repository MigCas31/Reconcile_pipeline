"""Run the v3 pipeline across every building, tolerating pre-existing errors.

Writes the aggregate to reconcile/reconcile_v3_results.json. Prints a summary
of OK / failed builds and total roof_proposals emitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile_v3.io.load import load_building
from reconcile_v3.io.write import to_dict as building_to_dict
from reconcile_v3.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
BUILDINGS_JSON = ROOT / "reconcile" / "buildings_3d.json"
OUT_PATH = ROOT / "reconcile" / "reconcile_v3_results.json"


def main() -> int:
    with open(BUILDINGS_JSON) as f:
        buildings = json.load(f)
    if not isinstance(buildings, list):
        buildings = buildings.get("buildings", [])

    uuids = [b.get("uuid") for b in buildings if b.get("uuid")]
    results = []
    ok = 0
    failed: list[tuple[str, str]] = []
    total_proposals = 0

    for i, uuid in enumerate(uuids):
        try:
            data = load_building(BUILDINGS_JSON, uuid)
            building = run_pipeline(data)
            results.append(building_to_dict(building))
            total_proposals += len(building.roof_proposals)
            ok += 1
        except Exception as exc:
            failed.append((uuid, f"{type(exc).__name__}: {exc}"))
        if (i + 1) % 20 == 0:
            print(
                f"[{i + 1}/{len(uuids)}] ok={ok} failed={len(failed)} "
                f"props={total_proposals}"
            )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nwrote {OUT_PATH} ({len(results)} buildings)")
    print(f"ok: {ok} / {len(uuids)}")
    print(f"failed: {len(failed)}")
    print(f"total roof_proposals: {total_proposals}")
    if failed:
        print("\nfirst 10 failures:")
        for u, msg in failed[:10]:
            print(f"  {u[:8]}  {msg[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
