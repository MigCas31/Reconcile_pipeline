from __future__ import annotations

from reconcile_tiers.energy.estimator import estimate
from reconcile_tiers.energy.u_values import UValueTable
from reconcile_tiers.payload.schema import (
    AdjacencyKind,
    HorizontalLid,
    RoofType,
    Room,
    TierClassification,
    TierPayload,
    Vec3,
    Wall,
)


def _classification() -> TierClassification:
    return TierClassification(
        tier=1,
        tier_label="simple",
        roof_type=RoofType.FLAT,
        n_stories=1,
        n_rooms=1,
        n_oblique=0,
        n_flat=0,
        has_half_height=False,
        has_gable=False,
    )


def _cube_payload(size: float = 6.0, height: float = 3.0) -> TierPayload:
    floor = HorizontalLid(
        corners=[
            Vec3(0, 0, 0),
            Vec3(size, 0, 0),
            Vec3(size, 0, size),
            Vec3(0, 0, size),
        ],
        adjacency=AdjacencyKind.GROUND_SLAB,
    )
    walls = [
        Wall(
            [
                Vec3(0, 0, 0),
                Vec3(0, height, 0),
                Vec3(size, height, 0),
                Vec3(size, 0, 0),
            ],
            None,
            None,
            [],
            "w0",
            adjacency=AdjacencyKind.EXTERNAL_AIR,
        ),
        Wall(
            [
                Vec3(size, 0, 0),
                Vec3(size, height, 0),
                Vec3(size, height, size),
                Vec3(size, 0, size),
            ],
            None,
            None,
            [],
            "w1",
            adjacency=AdjacencyKind.EXTERNAL_AIR,
        ),
        Wall(
            [
                Vec3(size, 0, size),
                Vec3(size, height, size),
                Vec3(0, height, size),
                Vec3(0, 0, size),
            ],
            None,
            None,
            [],
            "w2",
            adjacency=AdjacencyKind.EXTERNAL_AIR,
        ),
        Wall(
            [
                Vec3(0, 0, size),
                Vec3(0, height, size),
                Vec3(0, height, 0),
                Vec3(0, 0, 0),
            ],
            None,
            None,
            [],
            "w3",
            adjacency=AdjacencyKind.EXTERNAL_AIR,
        ),
    ]
    return TierPayload(
        schema_version="1",
        uuid="cube",
        address=None,
        building_center=Vec3(3, 1.5, 3),
        classification=_classification(),
        rooms=[
            Room(
                story=0,
                floor=[floor],
                walls=walls,
                doors=[],
                windows=[],
                locator_id="r",
                heating="radiators",
            )
        ],
        gaps=[],
        ceiling=[],
        knee_walls=[],
        dormer_faces=[],
        gable_closures=[],
    )


def test_cube_matches_hand_computed_heat_loss():
    table = UValueTable(
        table_id="test",
        values={kind: 0.0 for kind in AdjacencyKind},
        hdd_k_days=3000,
        design_delta_t_k=20,
    )
    table.values[AdjacencyKind.EXTERNAL_AIR] = 0.30
    table.values[AdjacencyKind.GROUND_SLAB] = 0.12

    result = estimate(_cube_payload(), table)

    expected_w = (4 * 6 * 3 * 0.30 + 6 * 6 * 0.12) * 20
    expected_kwh = expected_w / 20 * 3000 * 24 / 1000
    assert result.summary.total_heat_loss_w == expected_w
    assert result.summary.annual_kwh_proxy == expected_kwh
