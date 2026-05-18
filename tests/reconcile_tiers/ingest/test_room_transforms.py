from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from reconcile_tiers.ingest.merged import load_merged
from reconcile_tiers.ingest.room_transforms import (
    METHOD_RANK,
    RoomTransform,
    compute_room_transforms,
)
from reconcile_tiers.ingest.scan_cache import find_scan_cache_dir, load_raw_rooms


@pytest.mark.parametrize(
    ("uuid", "expected_methods", "max_residual_cm"),
    [
        (
            "c72ad855-9e52-46f1-886d-a9f37911521f",
            {"floor-svd": 10},
            0.01,
        ),
        (
            "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
            {"floor-svd": 9},
            0.01,
        ),
        (
            "2ea3b759-e047-424c-8034-f8ee5b811fb4",
            {"floor-svd": 11},
            0.01,
        ),
    ],
)
def test_compute_room_transforms_matches_cohort_method_distribution(
    uuid: str, expected_methods: dict[str, int], max_residual_cm: float
):
    merged = load_merged(uuid, Path("pipeline-outputs"))
    raw_rooms = load_raw_rooms(find_scan_cache_dir(uuid, Path(".scan-cache")))

    transforms = compute_room_transforms(raw_rooms, merged)

    assert (
        Counter(transform.method for transform in transforms.values())
        == expected_methods
    )
    assert (
        max(transform.residual_cm for transform in transforms.values())
        <= max_residual_cm + 0.01
    )
    assert all(
        transform.residual_cm < 50.0
        for transform in transforms.values()
        if transform.method == "floor-svd"
    )
    assert all(
        transform.residual_cm < 200.0
        for transform in transforms.values()
        if transform.method == "wall-center-svd"
    )


def test_room_transform_applies_rigid_transform_to_corners():
    transform = RoomTransform(
        rotation=np.eye(3),
        translation=np.array([1.0, 2.0, 3.0]),
        residual_cm=0.0,
        method="floor-svd",
    )

    assert transform.apply([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]]) == [
        [1.0, 2.0, 3.0],
        [3.0, 3.0, 2.0],
    ]


def test_method_rank_prefers_floor_svd_over_wall_center():
    assert (
        METHOD_RANK["floor-svd"]
        > METHOD_RANK["hybrid"]
        > METHOD_RANK["wall-center-svd"]
    )
