from __future__ import annotations

from reconcile_tiers.energy.u_values import DEFAULT_DK_TABLE, table_from_dict
from reconcile_tiers.payload.schema import AdjacencyKind


def test_default_table_zeroes_internal_and_party_walls():
    assert DEFAULT_DK_TABLE.u_value(AdjacencyKind.INTERNAL_TO_HEATED) == 0.0
    assert DEFAULT_DK_TABLE.u_value(AdjacencyKind.PARTY_WALL) == 0.0


def test_table_override_accepts_adjacency_values():
    table = table_from_dict(
        {
            "id": "custom",
            "values": {"externalAir": 0.42},
            "hdd_k_days": 2500,
        }
    )

    assert table.table_id == "custom"
    assert table.hdd_k_days == 2500
    assert table.u_value(AdjacencyKind.EXTERNAL_AIR) == 0.42
