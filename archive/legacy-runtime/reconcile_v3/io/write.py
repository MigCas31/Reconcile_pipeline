"""Serialize a V3Building to ``reconcile_v3_results.json``."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ..constants import CLUSTER_INCL_DEG, EAVE_OVERHANG_M, KNEE_DELTA_M, SCAN_NOISE_M
from ..models import V3Building

SCHEMA_VERSION = 1


def _asdict(item: Any) -> Any:
    if dataclasses.is_dataclass(item):
        return {k: _asdict(v) for k, v in dataclasses.asdict(item).items()}
    if isinstance(item, (list, tuple)):
        return [_asdict(v) for v in item]
    if isinstance(item, dict):
        return {k: _asdict(v) for k, v in item.items()}
    return item


def to_dict(building: V3Building) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "building_uuid": building.uuid,
        "address": building.address,
        "constants": {
            "scan_noise_m": SCAN_NOISE_M,
            "eave_overhang_m": EAVE_OVERHANG_M,
            "cluster_incl_deg": CLUSTER_INCL_DEG,
            "knee_delta_m": KNEE_DELTA_M,
        },
        "parts": [_asdict(x) for x in building.parts],
        "gaps": [_asdict(x) for x in building.gaps],
        "slabs": [_asdict(x) for x in building.slabs],
        "wall_extensions": [_asdict(x) for x in building.wall_extensions],
        "flat_ceilings": [_asdict(x) for x in building.flat_ceilings],
        "slanted_roofs": [_asdict(x) for x in building.slanted_roofs],
        "roof_proposals": [_asdict(x) for x in building.roof_proposals],
        "merged_roof_segments": [_asdict(x) for x in building.merged_roof_segments],
        "dormers": [_asdict(x) for x in building.dormers],
        "unresolved": [_asdict(x) for x in building.unresolved],
        "audit_log": building.audit_log,
    }


def write_building(path: Path, building: V3Building) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_dict(building), handle, indent=2)


def write_collection(path: Path, buildings: list[V3Building]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump([to_dict(b) for b in buildings], handle, indent=2)
