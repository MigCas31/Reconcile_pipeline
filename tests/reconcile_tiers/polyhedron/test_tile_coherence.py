"""Tests for pre-build room tile coherence checks."""

from __future__ import annotations

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.manifold_repair import TileFace
from reconcile_tiers.polyhedron.tile_coherence import (
    audit_room_tile_coherence,
    ceiling_connects_to_walls,
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


def test_filter_keeps_oblique_ceiling_chained_to_wall_anchored_flat():
    tiles = _cube_tiles()
    flat = tiles[1]
    oblique = TileFace(
        face_id=50,
        corners=(
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.5, 1.0),
            (0.0, 1.5, 1.0),
        ),
        plane=Plane(a=0.0, b=0.6, c=0.0, d=-0.6),
        source="ceiling",
        locator_id="hybrid::oblique",
    )
    mixed = [tiles[0], flat, *tiles[2:], oblique]
    filtered, dropped = filter_unconnected_ceiling_tiles(mixed, corner_tol=0.05)
    assert "hybrid::oblique" not in dropped
    assert any(t.locator_id == "hybrid::oblique" for t in filtered)
    assert any(t.locator_id == "cube::ceiling" for t in filtered)


def test_filter_drops_ceiling_island_with_no_wall_anchor():
    tiles = _cube_tiles()
    flat = tiles[1]
    island_a = TileFace(
        face_id=60,
        corners=(
            (2.0, 1.0, 0.0),
            (3.0, 1.0, 0.0),
            (3.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
        ),
        plane=flat.plane,
        source="ceiling",
        locator_id="island::a",
    )
    island_b = TileFace(
        face_id=61,
        corners=(
            (2.0, 1.0, 1.0),
            (3.0, 1.0, 1.0),
            (3.0, 1.5, 2.0),
            (2.0, 1.5, 2.0),
        ),
        plane=Plane(a=0.0, b=0.5, c=0.0, d=-0.5),
        source="ceiling",
        locator_id="island::b",
    )
    mixed = [tiles[0], flat, *tiles[2:], island_a, island_b]
    filtered, dropped = filter_unconnected_ceiling_tiles(mixed, corner_tol=0.05)
    assert "island::a" in dropped
    assert "island::b" in dropped
    assert any(t.locator_id == "cube::ceiling" for t in filtered)


def test_ceiling_point_contact_anchors_sloped_tile_without_rim_edge():
    tiles = _cube_tiles()
    walls = [t for t in tiles if t.source == "wall"]
    oblique = TileFace(
        face_id=70,
        corners=(
            (1.0, 1.0, 0.0),
            (0.7, 1.2, 0.0),
            (0.7, 1.2, 0.4),
            (1.0, 1.01, 0.4),
        ),
        plane=Plane(a=0.0, b=0.5, c=0.0, d=-0.5),
        source="ceiling",
        locator_id="hybrid::oblique_point",
    )
    assert ceiling_connects_to_walls(
        oblique, walls, corner_tol=0.05, rim_y_tol=0.08
    )
    mixed = [tiles[0], *walls, oblique]
    filtered, dropped = filter_unconnected_ceiling_tiles(mixed, corner_tol=0.05)
    assert "hybrid::oblique_point" not in dropped
