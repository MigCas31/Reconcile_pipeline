#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconcile.extract3d.builder import extract_building
from reconcile.roof_algorithms_py import run_roof_algorithms
from reconcile_v2.graph_builder import build_topology_graph

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "pipeline-outputs"
SCAN_CACHE_ROOT = ROOT / ".scan-cache"
OUTPUT_PATH = ROOT / "reconcile" / "roof_algorithms_py_results.json"


def _discover_uuids() -> list[str]:
    uuids: list[str] = []
    for child in sorted(PIPELINE_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if (child / "merged.json").exists():
            uuids.append(child.name)
    return uuids


def _build_result(uuid: str) -> dict:
    merged_path = PIPELINE_ROOT / uuid / "merged.json"
    graph = build_topology_graph(
        merged_path=merged_path,
        scan_dir=SCAN_CACHE_ROOT,
        uuid=uuid,
    )
    building = extract_building(
        uuid=uuid,
        pipeline_dir=PIPELINE_ROOT,
        scan_cache_root=SCAN_CACHE_ROOT,
        load_topology_graph=False,
    )
    if not building:
        raise RuntimeError(f"extract_building returned no building for {uuid}")
    return run_roof_algorithms(building, graph=graph)


def _write_checkpoint(path: Path, results: dict[str, dict]) -> None:
    tmp_path = path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(results), encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild reconcile/roof_algorithms_py_results.json from the current "
            "graph-aware roof pipeline."
        )
    )
    parser.add_argument(
        "uuids",
        nargs="*",
        help="Optional building UUIDs to rebuild. Defaults to all "
        "pipeline-outputs/*/merged.json buildings.",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write partial results every N successful buildings.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    uuids = list(args.uuids) if args.uuids else _discover_uuids()
    if not uuids:
        raise SystemExit("No pipeline buildings found.")

    results: dict[str, dict] = {}
    failures: list[dict[str, str]] = []
    started = perf_counter()

    for index, uuid in enumerate(uuids, start=1):
        t0 = perf_counter()
        try:
            results[uuid] = _build_result(uuid)
            elapsed = perf_counter() - t0
            print(f"[{index}/{len(uuids)}] ok {uuid} {elapsed:.2f}s", flush=True)
            if args.checkpoint_every > 0 and (index % args.checkpoint_every) == 0:
                _write_checkpoint(output_path, results)
        except Exception as exc:
            failures.append(
                {
                    "uuid": uuid,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"[{index}/{len(uuids)}] error {uuid} {exc}",
                file=sys.stderr,
                flush=True,
            )

    _write_checkpoint(output_path, results)
    total_elapsed = perf_counter() - started
    print(
        json.dumps(
            {
                "written": len(results),
                "requested": len(uuids),
                "failures": len(failures),
                "elapsed_s": round(total_elapsed, 3),
                "output": str(output_path),
            },
            indent=2,
        )
    )
    if failures:
        failure_path = output_path.with_suffix(".failures.json")
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Wrote failure details to {failure_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
