"""CLI: python -m reconcile_v3.cli --uuid <uuid>  |  --all"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io.load import iter_building_uuids, load_building
from .io.write import write_collection
from .pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILDINGS_JSON = REPO_ROOT / "reconcile" / "buildings_3d.json"
DEFAULT_OUTPUT = REPO_ROOT / "reconcile" / "reconcile_v3_results.json"


def _run_one(buildings_json: Path, uuid: str):
    data = load_building(buildings_json, uuid)
    return run_pipeline(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reconcile_v3")
    parser.add_argument("--buildings-json", type=Path, default=DEFAULT_BUILDINGS_JSON)
    parser.add_argument("--uuid", help="Process a single building UUID")
    parser.add_argument(
        "--all", action="store_true", help="Process every building in --buildings-json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the aggregate results (default: "
        "reconcile/reconcile_v3_results.json)",
    )
    args = parser.parse_args(argv)

    if not args.buildings_json.exists():
        print(f"Missing input: {args.buildings_json}", file=sys.stderr)
        return 2

    if args.uuid:
        building = _run_one(args.buildings_json, args.uuid)
        from .io.write import to_dict

        existing: list = []
        if args.output.exists():
            with open(args.output) as _fh:
                existing = json.load(_fh)
            existing = [e for e in existing if e.get("building_uuid") != args.uuid]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as _fh:
            json.dump([*existing, to_dict(building)], _fh, indent=2)
        print(
            f"Wrote {args.output} ({len(existing) + 1} buildings total, "
            f"{len(building.slabs)} slabs, "
            f"{len(building.flat_ceilings)} flat ceilings, "
            f"{len(building.unresolved)} unresolved regions)"
        )
        return 0

    if args.all:
        uuids = iter_building_uuids(args.buildings_json)
        results = []
        failures: list[tuple[str, str]] = []
        for idx, uuid in enumerate(uuids, 1):
            try:
                data = load_building(args.buildings_json, uuid)
                results.append(run_pipeline(data))
            except Exception as exc:
                failures.append((uuid, f"{type(exc).__name__}: {exc}"))
                print(
                    f"  ! skipped {uuid}: {type(exc).__name__}: {exc}", file=sys.stderr
                )
            if idx % 25 == 0:
                print(f"... {idx}/{len(uuids)}")
        write_collection(args.output, results)
        print(f"Wrote {args.output} ({len(results)} buildings, {len(failures)} failed)")
        if failures:
            print("Failures:", file=sys.stderr)
            for uuid, msg in failures:
                print(f"  {uuid}: {msg}", file=sys.stderr)
        return 0

    parser.error("Pass either --uuid <uuid> or --all")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
