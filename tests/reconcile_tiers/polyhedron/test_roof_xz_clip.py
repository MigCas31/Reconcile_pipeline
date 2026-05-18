"""Tests for clipping ceiling tiles to the room floor XZ footprint."""

from __future__ import annotations

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.manifold_repair import TileFace
from reconcile_tiers.polyhedron.roof_xz_clip import (
    clip_roof_tiles_to_floor_xz,
    floor_polygon_from_tiles,
)
from tests.reconcile_tiers.polyhedron.test_manifold_repair import _cube_tiles


def test_floor_polygon_from_cube():
    poly = floor_polygon_from_tiles(_cube_tiles())
    assert poly is not None
    assert abs(float(poly.area) - 1.0) < 0.01


def test_oversized_sloped_ceiling_is_clipped_to_floor():
    tiles = _cube_tiles()
    floor = tiles[0]
    # Sloped ceiling extending 1 m beyond the unit cube in +x (XZ overhang).
    oversized = TileFace(
        face_id=99,
        corners=(
            (0.0, 1.0, 0.0),
            (2.0, 1.5, 0.0),
            (2.0, 1.5, 1.0),
            (0.0, 1.0, 1.0),
        ),
        plane=Plane(a=0.0, b=0.5, c=0.0, d=0.5),
        source="ceiling",
        locator_id="oversized::ceiling",
    )
    mixed = [floor, oversized, *tiles[2:]]

    result = clip_roof_tiles_to_floor_xz(mixed)
    assert "oversized::ceiling" in result.clipped_locator_ids
    ceilings = [t for t in result.tiles if t.source == "ceiling"]
    assert len(ceilings) == 1
    xs = [c[0] for c in ceilings[0].corners]
    assert max(xs) <= 1.0 + 0.05
    assert min(xs) >= -0.05


def test_ceiling_inside_footprint_is_unchanged():
    tiles = _cube_tiles()
    result = clip_roof_tiles_to_floor_xz(tiles)
    assert result.clipped_locator_ids == ()
    assert result.dropped_locator_ids == ()
    ceiling = next(t for t in result.tiles if t.source == "ceiling")
    original = next(t for t in tiles if t.source == "ceiling")
    assert ceiling.corners == original.corners
