"""CLI: ``python -m reconcile_ext --uuid <uuid>`` or ``--all``.

Reads ``reconcile/reconcile_v3_results.json`` + ``reconcile/buildings_3d.json``,
emits ``reconcile/reconcile_ext_results.json`` with one entry per building.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from .io.v3_snapshot import build_snapshot, iter_v3_buildings, load_v3_building
from .pipeline import run

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V3_RESULTS = REPO_ROOT / "reconcile" / "reconcile_v3_results.json"
DEFAULT_BUILDINGS_JSON = REPO_ROOT / "reconcile" / "buildings_3d.json"
DEFAULT_OUTPUT = REPO_ROOT / "reconcile" / "reconcile_ext_results.json"


def _asdict(item: Any) -> Any:
    if dataclasses.is_dataclass(item):
        return {k: _asdict(v) for k, v in dataclasses.asdict(item).items()}
    if isinstance(item, (list, tuple)):
        return [_asdict(v) for v in item]
    if isinstance(item, dict):
        return {k: _asdict(v) for k, v in item.items()}
    return item


def _run_one(v3_results: dict[str, Any], buildings_json: Path) -> dict[str, Any]:
    snapshot = build_snapshot(v3_results, buildings_json)
    report = run(snapshot)
    return _asdict(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reconcile_ext")
    parser.add_argument("--v3-results", type=Path, default=DEFAULT_V3_RESULTS)
    parser.add_argument("--buildings-json", type=Path, default=DEFAULT_BUILDINGS_JSON)
    parser.add_argument("--uuid", help="Process a single building UUID")
    parser.add_argument("--all", action="store_true", help="Process every building")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.v3_results.exists():
        print(f"Missing V3 results: {args.v3_results}", file=sys.stderr)
        return 2
    if not args.buildings_json.exists():
        print(f"Missing buildings JSON: {args.buildings_json}", file=sys.stderr)
        return 2

    existing: list[dict[str, Any]] = []
    if args.uuid and args.output.exists():
        with open(args.output) as _fh:
            existing = json.load(_fh)

    results: list[dict[str, Any]] = []

    if args.uuid:
        v3 = load_v3_building(args.v3_results, args.uuid)
        report = _run_one(v3, args.buildings_json)
        results = [e for e in existing if e.get("building_uuid") != args.uuid]
        results.append(report)
    elif args.all:
        failures: list[tuple[str, str]] = []
        total = 0
        for v3 in iter_v3_buildings(args.v3_results):
            uuid = v3.get("building_uuid") or "?"
            total += 1
            try:
                results.append(_run_one(v3, args.buildings_json))
            except Exception as exc:
                failures.append((uuid, f"{type(exc).__name__}: {exc}"))
                print(f"  ! skipped {uuid}: {exc}", file=sys.stderr)
            if total % 25 == 0:
                print(f"... {total} buildings processed")
        if failures:
            print(f"{len(failures)} failures", file=sys.stderr)
    else:
        parser.error("Pass either --uuid <uuid> or --all")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Wrote {args.output} ({len(results)} building(s))")
    for r in results[:5]:
        stats = r.get("stats", {})
        addr = r.get("address") or r.get("building_uuid")
        print(
            f"  {addr}: "
            f"reflex={stats.get('reflex_count', 0)} "
            f"roof_disc={stats.get('discontinuity_count', 0)} "
            f"wall_steps={stats.get('wall_step_count', 0)} "
            f"height_deltas={stats.get('height_delta_count', 0)} "
            f"seams(strong/med/weak)={stats.get('seam_strong', 0)}/"
            f"{stats.get('seam_medium', 0)}/{stats.get('seam_weak', 0)} "
            f"units={stats.get('unit_count', 0)} ext={stats.get('extension_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
