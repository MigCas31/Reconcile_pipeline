"""Stream-extract compact per-building V3 context from the results JSON.

The full ``reconcile_v3_results.json`` is large because each building carries
verbose traces and proposer payloads. Analysis only needs a compact slice with
enough structure to reproduce training and inference features bit-for-bit.
"""

from __future__ import annotations

import decimal
import json
from pathlib import Path
from typing import Any

import ijson

_SCHEMA_VERSION = 2

_KEEP_PART_FIELDS = (
    "id",
    "room_ids",
    "footprint_xz",
    "stories",
    "gable_extension",
    "trace",
)
_KEEP_WE_FIELDS = (
    "id",
    "wall_id",
    "room_id",
    "strip_corners",
    "behind_knee_wall",
    "trace",
)
_KEEP_SR_FIELDS = ("id", "plane", "corners", "source_segment_ids", "trace")
_KEEP_SEG_FIELDS = (
    "id",
    "cluster_canonical_id",
    "merged_plane",
    "corners",
    "footprint_xz",
    "features",
    "heuristic_label",
    "room_boundary_refs",
    "member_proposal_ids",
    "trace",
)
_KEEP_SLAB_FIELDS = ("id", "room_id", "polygon", "trace")
_KEEP_FLAT_FIELDS = ("id", "room_id", "footprint_xz", "y", "over", "trace")
_KEEP_GAP_FIELDS = (
    "id",
    "footprint_xz",
    "floor_y",
    "ceiling_y",
    "between_rooms",
    "status",
    "trace",
)
_KEEP_UNRESOLVED_FIELDS = (
    "id",
    "room_id",
    "footprint_xz",
    "y_range",
    "reason",
    "context",
    "trace",
)
_KEEP_DORMER_FIELDS = ("id", "front_wall_id", "roof_surface_id", "corners", "trace")


def _coerce(obj: Any) -> Any:
    """Recursively convert Decimal -> float so JSON serialization works."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(v) for v in obj]
    if isinstance(obj, tuple):
        return [_coerce(v) for v in obj]
    return obj


def _pick(d: dict, keys: tuple[str, ...]) -> dict:
    return {k: _coerce(d[k]) for k in keys if k in d}


def extract(results_path: Path, only_uuids: set[str] | None = None) -> dict[str, dict]:
    """Stream ``results_path`` and return a compact per-UUID context dict."""
    out: dict[str, dict] = {}
    with results_path.open("rb") as f:
        for b in ijson.items(f, "item", use_float=True):
            uuid = b.get("building_uuid")
            if uuid is None:
                continue
            if only_uuids is not None and uuid not in only_uuids:
                continue
            out[uuid] = {
                "context_schema_version": _SCHEMA_VERSION,
                "parts": [_pick(p, _KEEP_PART_FIELDS) for p in b.get("parts") or []],
                "wall_extensions": [
                    _pick(w, _KEEP_WE_FIELDS) for w in b.get("wall_extensions") or []
                ],
                "dormers": [
                    _pick(d, _KEEP_DORMER_FIELDS) for d in b.get("dormers") or []
                ],
                "slanted_roofs": [
                    _pick(s, _KEEP_SR_FIELDS) for s in b.get("slanted_roofs") or []
                ],
                "merged_roof_segments": [
                    _pick(s, _KEEP_SEG_FIELDS)
                    for s in b.get("merged_roof_segments") or []
                ],
                "slabs": [_pick(s, _KEEP_SLAB_FIELDS) for s in b.get("slabs") or []],
                "flat_ceilings": [
                    _pick(c, _KEEP_FLAT_FIELDS) for c in b.get("flat_ceilings") or []
                ],
                "gaps": [_pick(g, _KEEP_GAP_FIELDS) for g in b.get("gaps") or []],
                "unresolved": [
                    _pick(u, _KEEP_UNRESOLVED_FIELDS) for u in b.get("unresolved") or []
                ],
                "roof_proposals_count": len(b.get("roof_proposals") or []),
                "merged_roof_segments_count": len(b.get("merged_roof_segments") or []),
                "slab_count": len(b.get("slabs") or []),
                "flat_ceiling_count": len(b.get("flat_ceilings") or []),
                "slanted_roof_count": len(b.get("slanted_roofs") or []),
                "dormer_count": len(b.get("dormers") or []),
                "unresolved_region_count": len(b.get("unresolved") or []),
                "part_count": len(b.get("parts") or []),
            }
    return out


def _cache_is_compatible(ctx: dict[str, dict]) -> bool:
    if not ctx:
        return False
    try:
        sample = next(iter(ctx.values()))
    except StopIteration:
        return False
    if not isinstance(sample, dict):
        return False
    version = sample.get("context_schema_version")
    required_keys = {
        "merged_roof_segments",
        "slabs",
        "flat_ceilings",
        "gaps",
        "unresolved",
        "part_count",
    }
    return version == _SCHEMA_VERSION and required_keys.issubset(sample.keys())


def load_or_build(
    results_path: Path,
    cache_path: Path,
    *,
    only_uuids: set[str] | None = None,
    rebuild: bool = False,
) -> dict[str, dict]:
    """Load the cached context; rebuild by streaming results if missing/stale."""
    if not rebuild and cache_path.exists():
        with cache_path.open() as f:
            ctx = json.load(f)
        if _cache_is_compatible(ctx):
            if only_uuids is None:
                return ctx
            return {uuid: val for uuid, val in ctx.items() if uuid in only_uuids}
    if not results_path.exists():
        raise FileNotFoundError(
            f"V3 results file not found at {results_path} and no cache at {cache_path}"
        )
    ctx = extract(results_path, only_uuids=only_uuids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(ctx, f)
    return ctx
