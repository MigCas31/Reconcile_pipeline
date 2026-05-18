import numpy as np
import pytest

from reconcile_tiers._core.transforms import (
    corners_to_world,
    hybrid_wall_corners,
    parse_transform,
)


def test_parse_transform_returns_row_major_matrix():
    flat = list(range(16))

    transform = parse_transform(flat)

    assert transform.shape == (4, 4)
    assert transform[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert transform[3].tolist() == [12.0, 13.0, 14.0, 15.0]


def test_parse_transform_rejects_non_4x4():
    with pytest.raises(ValueError, match="16"):
        parse_transform([1.0, 2.0, 3.0])


def test_corners_to_world_applies_homogeneous_transform():
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert corners_to_world([[1.0, 2.0, 3.0]], transform) == [[3.0, 5.0, 7.0]]


def test_corners_to_world_rejects_bad_transform_and_corner_shape():
    with pytest.raises(ValueError, match="4x4"):
        corners_to_world([[1.0, 2.0, 3.0]], np.eye(3))
    with pytest.raises(ValueError, match="corner"):
        corners_to_world([[1.0, 2.0]], np.eye(4))


def test_hybrid_wall_corners_prefers_merged_polygon_and_aligns_floor_y():
    merged_wall = {
        "transform": [
            1.0,
            0.0,
            0.0,
            10.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -2.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "polygonCorners": [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
    }
    raw_wall = {
        "dimensions": [8.0, 8.0],
        "polygonCorners": [[99.0, 99.0, 99.0], [98.0, 99.0, 99.0], [98.0, 98.0, 99.0]],
    }

    corners = hybrid_wall_corners(merged_wall, raw_wall, floor_y=0.25)

    assert min(c[1] for c in corners) == pytest.approx(0.25)
    assert corners[0] == pytest.approx([10.0, 0.25, -2.0])
    assert corners[2] == pytest.approx([12.0, 2.25, -2.0])


def test_hybrid_wall_corners_uses_raw_polygon_then_dimensions_without_floor_alignment():
    identity = np.eye(4).reshape(-1).tolist()
    merged_wall = {"transform": identity, "polygonCorners": []}
    raw_with_polygon = {
        "dimensions": [10.0, 10.0],
        "polygonCorners": [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 3.0, 0.0]],
    }
    raw_from_dimensions = {"dimensions": [4.0, 6.0], "polygonCorners": []}

    assert (
        hybrid_wall_corners(merged_wall, raw_with_polygon)
        == raw_with_polygon["polygonCorners"]
    )
    assert hybrid_wall_corners(merged_wall, raw_from_dimensions) == [
        [-2.0, -3.0, 0.0],
        [2.0, -3.0, 0.0],
        [2.0, 3.0, 0.0],
        [-2.0, 3.0, 0.0],
    ]
