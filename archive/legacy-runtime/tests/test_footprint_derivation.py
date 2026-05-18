"""Unit tests for reconcile/roof_algorithms_py/footprint_derivation.py.

Covers the cases the D4 audit flagged as latent risks for the
reconstruction BIP: simple rectangles, L-shapes, multi-room union with
wall-thickness gaps, degenerate/empty inputs, and the convex-hull
fallback path when Shapely is unavailable.
"""

from __future__ import annotations

import math

import pytest

from reconcile.roof_algorithms_py.footprint_derivation import (
    build_building_footprint,
)


def _rect(
    x0: float, z0: float, x1: float, z1: float, *, y: float = 0.0, story: int = 0
):
    return {
        "story": story,
        "fp": [
            [x0, y, z0],
            [x1, y, z0],
            [x1, y, z1],
            [x0, y, z1],
        ],
    }


def _polygon_area_2d(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    area = 0.0
    for i in range(n):
        xi, zi = pts[i]
        xj, zj = pts[(i + 1) % n]
        area += xi * zj - xj * zi
    return abs(area) / 2.0


def test_single_rectangle_round_trips() -> None:
    result = build_building_footprint([_rect(0.0, 0.0, 4.0, 3.0)], {})

    assert result["top_story"] == 0
    footprint = result["building_footprint"]
    assert footprint is not None and len(footprint) >= 4

    area = _polygon_area_2d(footprint)
    assert area == pytest.approx(12.0, abs=0.01)


def test_adjacent_rooms_merge_into_single_ring() -> None:
    # Two rooms abutting along x=4 — should union into a single 6x4 rectangle.
    rooms = [
        _rect(0.0, 0.0, 4.0, 4.0),
        _rect(4.0, 0.0, 6.0, 4.0),
    ]
    result = build_building_footprint(rooms, {})
    footprint = result["building_footprint"]

    assert footprint is not None
    area = _polygon_area_2d(footprint)
    assert area == pytest.approx(24.0, abs=0.5)


def test_rooms_with_small_wall_gap_are_bridged_by_buffer() -> None:
    # 0.2 m gap between rooms — below the 0.3 m room buffer so should merge.
    rooms = [
        _rect(0.0, 0.0, 4.0, 4.0),
        _rect(4.2, 0.0, 6.0, 4.0),
    ]
    result = build_building_footprint(rooms, {})
    footprint = result["building_footprint"]

    assert footprint is not None
    area = _polygon_area_2d(footprint)
    assert area >= 22.0  # close to the combined ~24 m² without gap
    # Gap should not survive: the union occupies a contiguous region.
    xs = [p[0] for p in footprint]
    assert max(xs) - min(xs) == pytest.approx(6.0, abs=0.2)


def test_l_shape_preserves_concavity() -> None:
    # L-shape: 4x4 square plus a 2x2 wing. Convex hull would be a 6x4 rect.
    rooms = [
        _rect(0.0, 0.0, 4.0, 4.0),
        _rect(4.0, 0.0, 6.0, 2.0),
    ]
    result = build_building_footprint(rooms, {})
    footprint = result["building_footprint"]

    area = _polygon_area_2d(footprint)
    # L-shape area = 16 + 4 = 20 m². Convex hull would be 6*4 = 24 m².
    assert 18.0 < area < 22.0, f"Expected ~20 m² for L-shape, got {area:.2f}"
    assert area < 23.0, "Concavity lost — footprint approached hull area"


def test_top_story_is_max_story_of_exposed_rooms() -> None:
    rooms = [
        _rect(0.0, 0.0, 4.0, 4.0, story=0),
        _rect(0.0, 0.0, 4.0, 4.0, y=3.0, story=2),
        _rect(0.0, 0.0, 4.0, 4.0, y=6.0, story=1),
    ]
    result = build_building_footprint(rooms, {})
    assert result["top_story"] == 2


def test_empty_exposed_rooms_returns_none() -> None:
    result = build_building_footprint([], {})
    assert result["building_footprint"] is None
    assert result["top_story"] == -math.inf


def test_degenerate_room_with_under_three_points_is_skipped() -> None:
    # One good room + one degenerate (line segment) — still get a footprint.
    good = _rect(0.0, 0.0, 4.0, 4.0)
    bad = {"story": 0, "fp": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    result = build_building_footprint([good, bad], {})
    assert result["building_footprint"] is not None
    area = _polygon_area_2d(result["building_footprint"])
    assert area == pytest.approx(16.0, abs=0.01)


def test_footprint_is_closed_ring_without_duplicate_closing_point() -> None:
    result = build_building_footprint([_rect(0.0, 0.0, 4.0, 3.0)], {})
    footprint = result["building_footprint"]
    assert footprint is not None
    assert footprint[0] != footprint[-1], (
        "footprint_derivation should drop the duplicate closing vertex that "
        "Shapely adds to exterior rings"
    )


def test_convex_hull_fallback_when_shapely_missing(monkeypatch) -> None:
    """If Shapely is unimportable, the fallback convex-hull path runs.

    Setting ``sys.modules["shapely"] = None`` causes any subsequent
    ``import shapely`` statement to raise ``ImportError`` — the exact
    failure mode ``_union_room_footprint`` guards against.
    """
    import sys

    monkeypatch.setitem(sys.modules, "shapely", None)
    monkeypatch.setitem(sys.modules, "shapely.geometry", None)
    monkeypatch.setitem(sys.modules, "shapely.ops", None)

    rooms = [_rect(0.0, 0.0, 4.0, 4.0), _rect(4.0, 0.0, 6.0, 2.0)]
    result = build_building_footprint(rooms, {})

    footprint = result["building_footprint"]
    assert footprint is not None
    # Fallback uses convex hull of all fp points. For this L-shape the hull
    # is the pentagon (0,0)-(6,0)-(6,2)-(4,4)-(0,4) with area 22 m², which
    # is strictly larger than Shapely's concavity-preserving union (20 m²)
    # — that gap is the cost of the fallback and why the D4 audit flags
    # buildings that end up on it.
    area = _polygon_area_2d(footprint)
    assert area == pytest.approx(22.0, abs=0.5)
    assert area > 20.5, "fallback must differ from Shapely concave union"
