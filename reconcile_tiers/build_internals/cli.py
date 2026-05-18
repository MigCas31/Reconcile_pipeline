"""CLI for `reconcile_tiers.build`: argparse + parallel/serial dispatch.

`build_many` runs `_build_one` across a UUID list (process pool above
worker > 1). `main` is the entry point — exposed as `reconcile_tiers.build.main`
via re-export and used by the existing `reconcile_tiers.cli` shim.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from reconcile_tiers.build_internals.io_index import (
    _build_one,
    list_uuids,
    write_tier_index,
)


def build_many(
    uuids: list[str],
    *,
    pipeline_dir: Path | str = Path("pipeline-outputs"),
    scan_root: Path | str | None = Path(".scan-cache"),
    force: bool = False,
    validate_only: bool = False,
    workers: int = 1,
) -> list[tuple[str, str, str | None]]:
    worker_args = [
        (
            uuid,
            str(pipeline_dir),
            str(scan_root) if scan_root is not None else None,
            force,
            validate_only,
        )
        for uuid in sorted(uuids)
    ]
    if workers <= 1:
        results = []
        for args in worker_args:
            try:
                results.append(_build_one(args))
            except Exception as exc:
                results.append((args[0], "failed", str(exc)))
        return results
    results: list[tuple[str, str, str | None]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, args): args[0] for args in worker_args}
        for future in as_completed(futures):
            uuid = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append((uuid, "failed", str(exc)))
    return sorted(results, key=lambda item: item[0])


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--uuid", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-j", "--workers", type=int, default=1)
    parser.add_argument("--pipeline-dir", default="pipeline-outputs")
    parser.add_argument("--scan-root", default=".scan-cache")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    uuids = list_uuids(args.pipeline_dir) if args.all else sorted(args.uuid)
    if not uuids:
        parser.error("pass --all or at least one --uuid")
    results = build_many(
        uuids,
        pipeline_dir=args.pipeline_dir,
        scan_root=args.scan_root,
        force=args.force,
        validate_only=args.validate_only,
        workers=max(1, args.workers),
    )
    failures = [result for result in results if result[1] == "failed"]
    if failures:
        failure_path = Path(args.pipeline_dir) / "tier_build_failures.log"
        failure_path.write_text(
            "\n".join(f"{uuid}: {message}" for uuid, _status, message in failures)
            + "\n"
        )
        return 1
    failure_path = Path(args.pipeline_dir) / "tier_build_failures.log"
    if failure_path.exists():
        failure_path.unlink()
    if not args.validate_only:
        write_tier_index(args.pipeline_dir, results)
    return 0
