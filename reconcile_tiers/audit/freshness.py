"""Report tier_payload.json files older than their inputs.

A payload is stale when its `build_meta.built_at` predates the newest input
that fed it — scan-cache, merged.json, or any file under `reconcile_tiers/`.
Run this after a code change to find which buildings need a rebuild, or in
CI to catch a stale `pipeline-outputs/` checkout.

    python -m reconcile_tiers.audit.freshness
    python -m reconcile_tiers.audit.freshness --pipeline-dir pipeline-outputs
    --scan-root .scan-cache

Exit code is 1 when any payload is stale (or missing build_meta), 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reconcile_tiers.build_internals.io_index import _latest_mtime
from reconcile_tiers.ingest.merged import find_merged_path
from reconcile_tiers.ingest.scan_cache import find_scan_cache_dir

# Subdirectories of reconcile_tiers/ that don't affect payload contents:
# `audit/` is post-hoc analysis on already-built payloads, `web/` is the
# Three.js viewer. Touching either should not invalidate every payload.
_SOURCE_EXCLUDED = {"audit", "web", "__pycache__"}


def _build_source_mtime(source_root: Path) -> float:
    if not source_root.exists():
        return 0.0
    latest = source_root.stat().st_mtime
    for entry in source_root.iterdir():
        if entry.name in _SOURCE_EXCLUDED:
            continue
        latest = max(latest, _latest_mtime(entry))
    return latest


def _payload_built_at(payload_path: Path) -> float | None:
    try:
        data = json.loads(payload_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    meta = data.get("build_meta")
    if not isinstance(meta, dict):
        return None
    built_at = meta.get("built_at")
    return float(built_at) if isinstance(built_at, (int, float)) else None


def stale_uuids(
    pipeline_dir: Path,
    scan_root: Path | None,
    source_root: Path,
) -> list[tuple[str, str]]:
    """Return [(uuid, reason), ...] for each stale or unstamped payload."""
    source_mtime = _build_source_mtime(source_root)
    out: list[tuple[str, str]] = []
    for payload_path in sorted(pipeline_dir.glob("*/tier_payload.json")):
        uuid = payload_path.parent.name
        built_at = _payload_built_at(payload_path)
        if built_at is None:
            out.append((uuid, "missing build_meta.built_at"))
            continue
        merged_mtime = _latest_mtime(find_merged_path(uuid, pipeline_dir))
        scan_dir = find_scan_cache_dir(uuid, scan_root) if scan_root else None
        scan_mtime = _latest_mtime(scan_dir) if scan_dir else 0.0
        newest_input = max(merged_mtime, scan_mtime, source_mtime)
        if newest_input > built_at:
            reason = (
                f"built_at={built_at:.0f} < newest_input={newest_input:.0f} "
                f"(merged={merged_mtime:.0f} scan={scan_mtime:.0f} "
                f"source={source_mtime:.0f})"
            )
            out.append((uuid, reason))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", default="pipeline-outputs", type=Path)
    parser.add_argument("--scan-root", default=".scan-cache", type=Path)
    parser.add_argument("--source-root", default="reconcile_tiers", type=Path)
    args = parser.parse_args(argv)

    stale = stale_uuids(args.pipeline_dir, args.scan_root, args.source_root)
    if not stale:
        print(f"All tier_payload.json files are up-to-date in {args.pipeline_dir}/")
        return 0
    print(f"{len(stale)} stale tier_payload.json file(s) in {args.pipeline_dir}/:")
    for uuid, reason in stale:
        print(f"  {uuid}  ({reason})")
    print("\nRebuild with: python -m reconcile_tiers.cli --all --force")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
