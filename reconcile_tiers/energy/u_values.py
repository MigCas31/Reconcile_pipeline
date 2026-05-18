"""U-value table for the first-pass Danish energy sensitivity proxy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reconcile_tiers.payload.schema import AdjacencyKind


@dataclass(frozen=True, slots=True)
class UValueTable:
    """A lookup table keyed by existing tier-payload adjacency kinds."""

    table_id: str
    values: dict[AdjacencyKind, float]
    hdd_k_days: float = 3000.0
    design_delta_t_k: float = 20.0

    def u_value(self, adjacency: AdjacencyKind | str | None) -> float:
        if adjacency is None:
            kind = AdjacencyKind.UNKNOWN
        elif isinstance(adjacency, AdjacencyKind):
            kind = adjacency
        else:
            try:
                kind = AdjacencyKind(adjacency)
            except ValueError:
                kind = AdjacencyKind.UNKNOWN
        return float(self.values.get(kind, self.values[AdjacencyKind.UNKNOWN]))


DEFAULT_DK_TABLE = UValueTable(
    table_id="default-dk",
    hdd_k_days=3000.0,
    design_delta_t_k=20.0,
    values={
        AdjacencyKind.EXTERNAL_AIR: 0.30,
        AdjacencyKind.GROUND_SLAB: 0.12,
        AdjacencyKind.GROUND_SLAB_UFH: 0.10,
        AdjacencyKind.UNHEATED_ATTIC: 0.20,
        AdjacencyKind.UNHEATED_BASEMENT_FLOOR: 0.25,
        AdjacencyKind.BASEMENT_WALL_GROUND_SHALLOW: 0.20,
        AdjacencyKind.BASEMENT_WALL_GROUND_DEEP: 0.15,
        AdjacencyKind.INTERNAL_TO_HEATED: 0.0,
        AdjacencyKind.INTERNAL_TO_UNHEATED_HOST: 0.60,
        AdjacencyKind.PARTY_WALL: 0.0,
        AdjacencyKind.UNKNOWN: 0.30,
    },
)


def _adjacency_key(value: str) -> AdjacencyKind:
    try:
        return AdjacencyKind(value)
    except ValueError:
        return AdjacencyKind[value]


def table_from_dict(data: dict[str, Any]) -> UValueTable:
    values = dict(DEFAULT_DK_TABLE.values)
    raw_values = data.get("values") or data.get("u_values") or {}
    for key, value in raw_values.items():
        values[_adjacency_key(key)] = float(value)
    return UValueTable(
        table_id=str(
            data.get("table_id") or data.get("id") or DEFAULT_DK_TABLE.table_id
        ),
        values=values,
        hdd_k_days=float(data.get("hdd_k_days", DEFAULT_DK_TABLE.hdd_k_days)),
        design_delta_t_k=float(
            data.get("design_delta_t_k", DEFAULT_DK_TABLE.design_delta_t_k)
        ),
    )


def load_u_value_table(
    country: str = "dk", *, context_root: Path | str = ".context"
) -> UValueTable:
    """Load `.context/u-values/<country>.json` when present, otherwise default DK."""

    path = Path(context_root) / "u-values" / f"{country}.json"
    if not path.exists():
        return DEFAULT_DK_TABLE
    return table_from_dict(json.loads(path.read_text()))
