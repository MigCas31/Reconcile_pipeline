"""
Tests for the should-be-flat classifier and noisy-plane drop in extract3d/ceilings.py.
"""

from __future__ import annotations

from reconcile.extract3d.ceilings import (
    NOISY_PLANE_DROP_TAU_M,
    apply_flat_classification,
    classify_should_be_flat,
    drop_noisy_raw_ceiling_planes,
    infer_ceilings,
)


def _rect_wall(x0: float, z0: float, x1: float, z1: float, top_y: float = 1.17) -> dict:
    return {
        "corners": [
            [x0, 0.0, z0],
            [x1, 0.0, z1],
            [x1, top_y, z1],
            [x0, top_y, z0],
        ],
    }


def _square_floor(side: float = 4.0) -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [side, 0.0, side],
        [0.0, 0.0, side],
    ]


def _flat_room(top_y: float = 1.17, *, story: int = 0) -> dict:
    return {
        "story": story,
        "floor_polygon": _square_floor(),
        "walls_merged": [
            _rect_wall(0, 0, 4, 0, top_y),
            _rect_wall(4, 0, 4, 4, top_y),
            _rect_wall(4, 4, 0, 4, top_y),
            _rect_wall(0, 4, 0, 0, top_y),
        ],
        "raw_ceiling_planes": [],
    }


def _ceiling_plane(corners: list[list[float]]) -> dict:
    return {"corners": corners}


def test_clean_flat_room_classified_flat():
    room = _flat_room(top_y=1.17)
    room["raw_ceiling_planes"] = [
        _ceiling_plane([[0, 1.16, 0], [4, 1.16, 0], [4, 1.16, 4], [0, 1.16, 4]]),
    ]
    cls = classify_should_be_flat([room])
    assert cls[0][0] is True, "uniform wall tops + clean ceiling plane → should be flat"
    assert abs(cls[0][1] - 1.17) < 0.01


def test_noisy_wall_polygons_dropped_in_flat_room():
    """Target case mirror: flat room, raw planes contain wall-shaped polygons that
    span floor-to-ceiling. The wall polygons must be dropped, the clean ones kept.
    """
    room = _flat_room(top_y=1.17)
    room["raw_ceiling_planes"] = [
        _ceiling_plane([[0, 1.16, 0], [4, 1.16, 0], [4, 1.16, 4], [0, 1.16, 4]]),
        _ceiling_plane([[0, 0.0, 0], [4, 0.0, 0], [4, 1.16, 0], [0, 1.16, 0]]),
        _ceiling_plane([[4, 0.0, 0], [4, 0.0, 4], [4, 1.16, 4], [4, 1.16, 0]]),
    ]
    cls = classify_should_be_flat([room])
    assert cls[0][0] is True
    drop_noisy_raw_ceiling_planes([room], cls)
    kept = room["raw_ceiling_planes"]
    assert len(kept) == 1, f"expected 1 plane kept, got {len(kept)}"
    ys = [c[1] for c in kept[0]["corners"]]
    assert max(ys) - min(ys) < 0.05


def test_attic_with_high_ceiling_not_flat():
    """A room with low knee walls but raw ceiling reaching far above wall tops
    is an attic — must not be classified as flat.
    """
    room = _flat_room(top_y=1.0)
    room["raw_ceiling_planes"] = [
        _ceiling_plane([[0, 1.0, 0], [4, 1.0, 0], [4, 3.0, 4], [0, 3.0, 4]]),
    ]
    cls = classify_should_be_flat([room])
    assert cls[0][0] is False, "ceiling reaches 2m above wall tops → not flat"


def test_slightly_sloped_room_vetoed_by_ridge_eave_guard():
    """When prior pass set ridge_eave > 0.10m, override the wall-top consensus —
    the slope was already detected, don't flatten it.
    """
    room = _flat_room(top_y=1.17)
    room["ceiling_eave_height"] = 0.70
    room["ceiling_ridge_height"] = 0.94
    cls = classify_should_be_flat([room])
    assert cls[0][0] is False


def test_neighbour_consensus_reinforces_room_with_few_walls():
    """A room with one wall (wall-top spread undefined) can still be classified
    flat when a neighbour on the same story has consistent wall-top y.
    """
    rooms = [_flat_room(top_y=1.17, story=0), _flat_room(top_y=1.17, story=0)]
    rooms[1]["walls_merged"] = [_rect_wall(0, 0, 4, 0, 1.17)]
    cls = classify_should_be_flat(rooms)
    assert cls[0][0] is True
    assert cls[1][0] is False


def test_apply_flat_classification_preserves_existing_ceiling_polygon():
    room = _flat_room(top_y=1.18)
    room["ceiling_polygon"] = [
        [0.0, 1.16, 0.0],
        [4.0, 1.16, 0.0],
        [4.0, 1.16, 4.0],
        [0.0, 1.16, 4.0],
    ]
    room["raw_ceiling_planes"] = [
        _ceiling_plane([[0, 1.16, 0], [4, 1.16, 0], [4, 1.16, 4], [0, 1.16, 4]]),
    ]
    cls = classify_should_be_flat([room])
    apply_flat_classification([room], cls)
    assert room["ceiling_type"] == "flat"
    assert len(room["ceiling_polygon"]) == 4
    assert all(abs(p[1] - 1.16) < 1e-6 for p in room["ceiling_polygon"])
    assert room["ceiling_eave_height"] == 1.16
    assert room["ceiling_ridge_height"] == 1.16


def test_apply_flat_classification_lifts_floor_when_no_existing_polygon():
    room = _flat_room(top_y=1.17)
    room["raw_ceiling_planes"] = [
        _ceiling_plane([[0, 1.16, 0], [4, 1.16, 0], [4, 1.16, 4], [0, 1.16, 4]]),
    ]
    assert "ceiling_polygon" not in room
    cls = classify_should_be_flat([room])
    apply_flat_classification([room], cls)
    assert room["ceiling_type"] == "flat"
    assert len(room["ceiling_polygon"]) == 4
    ys = [p[1] for p in room["ceiling_polygon"]]
    assert max(ys) - min(ys) < 1e-6


def test_drop_threshold_constant():
    """Per-face drop threshold matches the documented 0.30 m."""
    assert NOISY_PLANE_DROP_TAU_M == 0.30


def test_infer_ceilings_full_pipeline_runs_classifier_first():
    """End-to-end through infer_ceilings — flat classifier runs before slant
    fallback and noisy planes get dropped.
    """
    room = _flat_room(top_y=1.17)
    room["raw_ceiling_planes"] = [
        _ceiling_plane([[0, 1.16, 0], [4, 1.16, 0], [4, 1.16, 4], [0, 1.16, 4]]),
        _ceiling_plane([[0, 0.0, 0], [4, 0.0, 0], [4, 1.16, 0], [0, 1.16, 0]]),
    ]
    infer_ceilings([room])
    assert room["ceiling_type"] == "flat"
    assert len(room["raw_ceiling_planes"]) == 1
