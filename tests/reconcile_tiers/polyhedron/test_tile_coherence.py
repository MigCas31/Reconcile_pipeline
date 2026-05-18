"""Tests for pre-build room tile coherence checks."""

from __future__ import annotations

from reconcile_tiers.polyhedron.manifold_repair import TileFace
from reconcile_tiers.polyhedron.tile_coherence import (
    audit_room_tile_coherence,
    filter_unconnected_ceiling_tiles,
)
from tests.reconcile_tiers.polyhedron.test_manifold_repair import _cube_tiles


def test_cube_tiles_are_coherent():
    result = audit_room_tile_coherence(_cube_tiles(), corner_tol=0.05)
    assert result.ok
    assert result.issues == ()
    assert result.component_count == 1
    assert result.shared_edge_count > 0
    assert result.ceiling_clearance_m is not None
    assert result.ceiling_clearance_m < 0.01


def test_floating_ceiling_fails_clearance_and_wall_ceiling_gap():
    tiles = _cube_tiles()
    # Same footprint as cube ceiling but 2 m too high — mis-assigned roof lid.
    high_ceiling = TileFace(
        face_id=99,
        corners=(
            (0.0, 3.0, 0.0),
            (0.0, 3.0, 1.0),
            (1.0, 3.0, 1.0),
            (1.0, 3.0, 0.0),
        ),
        plane=tiles[1].plane,
        source="ceiling",
        locator_id="floating::ceiling",
    )
    tiles_with_float = [tiles[0], *tiles[2:], high_ceiling]

    result = audit_room_tile_coherence(
        tiles_with_float,
        corner_tol=0.05,
        max_ceiling_clearance_m=1.0,
    )
    assert not result.ok
    kinds = {issue.kind for issue in result.issues}
    assert "ceiling_far_from_wall_tops" in kinds
    assert "wall_ceiling_gap" in kinds


def test_missing_wall_reports_missing_walls():
    tiles = [t for t in _cube_tiles() if t.source != "wall"]
    result = audit_room_tile_coherence(tiles)
    assert not result.ok
    assert any(issue.kind == "missing_walls" for issue in result.issues)


def test_filter_drops_floating_ceiling_keeps_connected():
    tiles = _cube_tiles()
    high_ceiling = TileFace(
        face_id=99,
        corners=(
            (0.0, 3.0, 0.0),
            (0.0, 3.0, 1.0),
            (1.0, 3.0, 1.0),
            (1.0, 3.0, 0.0),
        ),
        plane=tiles[1].plane,
        source="ceiling",
        locator_id="floating::ceiling",
    )
    mixed = [tiles[0], tiles[1], *tiles[2:], high_ceiling]
    filtered, dropped = filter_unconnected_ceiling_tiles(mixed, corner_tol=0.05)
    assert "floating::ceiling" in dropped
    assert not any(t.locator_id == "floating::ceiling" for t in filtered)
    assert any(t.locator_id == "cube::ceiling" for t in filtered)
