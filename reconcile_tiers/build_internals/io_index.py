"""Output-path resolution, freshness checks, and tier-payload index writing.

Per-UUID workers (`_build_one`) and debug sidecars also live here so the
orchestrator only owns model assembly, not file I/O. Re-exported from
`reconcile_tiers.build` for backwards-compatible imports.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

from reconcile_tiers.energy.silent_fix import FixPolicy, apply_silent_fixes
from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.ingest.merged import find_merged_path
from reconcile_tiers.ingest.scan_cache import find_scan_cache_dir
from reconcile_tiers.payload.schema import BuildMeta, TierPayload, payload_to_dict


def payload_json(payload: TierPayload) -> str:
    return json.dumps(payload_to_dict(payload), indent=2, sort_keys=True) + "\n"


def _latest_mtime(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
    return latest


def output_path_for_uuid(uuid: str, pipeline_dir: Path | str) -> Path:
    return Path(pipeline_dir) / uuid / "tier_payload.json"


def needs_rebuild(
    uuid: str,
    pipeline_dir: Path | str,
    scan_root: Path | str | None,
    output_path: Path | str | None = None,
    *,
    force: bool = False,
) -> bool:
    if force:
        return True
    out_path = (
        Path(output_path)
        if output_path is not None
        else output_path_for_uuid(uuid, pipeline_dir)
    )
    if not out_path.exists():
        return True
    merged_path = find_merged_path(uuid, pipeline_dir)
    scan_dir = find_scan_cache_dir(uuid, scan_root) if scan_root is not None else None
    newest_input = max(_latest_mtime(merged_path), _latest_mtime(scan_dir))
    return newest_input > out_path.stat().st_mtime


def list_uuids(pipeline_dir: Path | str) -> list[str]:
    root = Path(pipeline_dir)
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "merged.json").exists()
    )


def list_tier_payload_uuids(pipeline_dir: Path | str) -> list[str]:
    root = Path(pipeline_dir)
    if not root.exists():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "tier_payload.json").exists()
    )


def write_tier_index(
    pipeline_dir: Path | str,
    results: list[tuple[str, str, str | None]],
) -> None:
    root = Path(pipeline_dir)
    uuids = set(list_tier_payload_uuids(root))
    uuids.update(uuid for uuid, status, _message in results if status != "failed")
    index_path = root / "tier_index.json"
    index_path.write_text(
        json.dumps({"buildings": sorted(uuids)}, indent=2, sort_keys=True) + "\n"
    )


def _build_one(
    args: tuple[str, str, str | None, bool, bool],
) -> tuple[str, str, str | None]:
    from reconcile_tiers.build import build_tier_payload

    uuid, pipeline_dir, scan_root, force, validate_only = args
    out_path = output_path_for_uuid(uuid, pipeline_dir)
    if not validate_only and not needs_rebuild(
        uuid, pipeline_dir, scan_root, out_path, force=force
    ):
        return uuid, "skipped", None
    drops: list = []
    plane_evidence: list = []
    payload = build_tier_payload(
        uuid,
        pipeline_dir,
        scan_root,
        drops_sink=drops,
        plane_evidence_sink=plane_evidence,
    )
    if validate_only:
        return uuid, "validated", None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = replace(payload, build_meta=BuildMeta(built_at=time.time()))
    fix_policy = FixPolicy.from_env()
    if fix_policy.enabled:
        outcome = apply_silent_fixes(payload, fix_policy)
        payload = outcome.payload
        (out_path.parent / "tier_payload_autofix.json").write_text(
            json.dumps(outcome.audit_log, indent=2, sort_keys=True) + "\n"
        )
    out_path.write_text(payload_json(payload))
    drops_path = out_path.parent / "tier_payload_drops.json"
    if drops:
        drops_path.write_text(
            json.dumps({"raw_ceilings": drops}, indent=2, sort_keys=True) + "\n"
        )
    elif drops_path.exists():
        # Stale audit log from a previous run that dropped raws; clear it so
        # the file always reflects the latest build.
        drops_path.unlink()
    evidence_path = out_path.parent / "plane_evidence.json"
    if plane_evidence:
        evidence_path.write_text(
            json.dumps(plane_evidence[0], indent=2, sort_keys=True) + "\n"
        )
    elif evidence_path.exists():
        evidence_path.unlink()
    if os.environ.get("EXTERIOR_DEBUG") == "1":
        from reconcile_tiers.extract.exterior import drain_rejection_log

        rejections, accepted = drain_rejection_log()
        rej_path = out_path.parent / "exterior_debug.json"
        rej_path.write_text(
            json.dumps(
                {"uuid": uuid, "rejections": rejections, "accepted": accepted},
                indent=2,
            )
            + "\n"
        )
    if os.environ.get("WING_DETECTION_DEBUG") == "1":
        _write_wings_debug_sidecar(uuid, pipeline_dir, scan_root, out_path.parent)
    return uuid, "written", None


def _write_wings_debug_sidecar(
    uuid: str,
    pipeline_dir: Path | str,
    scan_root: Path | str | None,
    out_dir: Path,
) -> None:
    """Emit `wings_debug.json` with v1 and v2 wing partitions side by side.

    Called only when `WING_DETECTION_DEBUG=1`. Re-extracts the model so the
    wings are computed independently of the in-flight build (which has
    already cached one variant). Failures are silently swallowed; this is
    a developer aid, not a build invariant.
    """
    try:
        from reconcile_tiers._core.wing_decomposition import compute_wings_for_model
        from reconcile_tiers._core.wing_decomposition_v2 import (
            compute_wings_for_model_v2,
        )

        scan_root_path = Path(scan_root) if scan_root is not None else None
        model = extract_building_model(
            uuid,
            pipeline_dir=str(pipeline_dir),
            scan_root=str(scan_root_path) if scan_root_path is not None else None,
        )
        wings_v1 = compute_wings_for_model(model)
        wings_v2 = compute_wings_for_model_v2(model)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wings_debug.json").write_text(
            json.dumps(
                {
                    "uuid": uuid,
                    "wings_v1": [_wing_to_dict(w) for w in wings_v1],
                    "wings_v2": [_wing_to_dict(w) for w in wings_v2],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except Exception:
        # Debug-only; never fail a build because the sidecar couldn't be written.
        return


def _wing_to_dict(wing) -> dict:
    coords = (
        list(wing.polygon.exterior.coords)
        if wing.polygon and not wing.polygon.is_empty
        else []
    )
    return {
        "index": wing.index,
        "role": wing.role,
        "area_m2": float(wing.area_m2),
        "long_axis_math": float(wing.long_axis_math),
        "confidence": getattr(wing, "confidence", "high"),
        "disagreement": list(getattr(wing, "disagreement", ())),
        "polygon_xz": [[float(x), float(z)] for x, z in coords],
    }
