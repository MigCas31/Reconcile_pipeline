"""Unit tests for the post-split continuity-merge.

Phase 1's post-edit yaw-snap and `ObliqueSurface` continuity-merge were
retired (see `architectural_priors.py` docstring). What remains here is the
post-split continuity-merge for `SplitObliqueSurface` cells.
"""

from __future__ import annotations

import math

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.roof.architectural_priors import clean_split_oblique_surfaces
from reconcile_tiers.roof.roof import SplitObliqueSurface


def _split_surface(
    a: float,
    b: float,
    c: float,
    d: float,
    corners: list[list[float]],
    *,
    cell_id: str,
    source_idx: int,
) -> SplitObliqueSurface:
    return SplitObliqueSurface(
        corners=corners,
        plane=Plane(a=a, b=b, c=c, d=d),
        arrangement_cell_id=cell_id,
        source_oblique_index=source_idx,
        dominant_story=0,
    )


def test_clean_split_merges_within_room_same_source_fragments() -> None:
    """Same source oblique, same room, abutting → merge (within-room
    fragmentation from arrangement.split is what the cleaner targets)."""
    plane = (0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 1.0)
    p = Plane(a=plane[0], b=plane[1], c=plane[2], d=plane[3])
    s1 = _split_surface(
        *plane,
        corners=[
            [0.0, p.y_at(0.0, 0.0) or 0.0, 0.0],
            [4.0, p.y_at(4.0, 0.0) or 0.0, 0.0],
            [4.0, p.y_at(4.0, 4.0) or 0.0, 4.0],
            [0.0, p.y_at(0.0, 4.0) or 0.0, 4.0],
        ],
        cell_id="cell:0:room:0:0",
        source_idx=0,
    )
    s2 = _split_surface(
        *plane,
        corners=[
            [4.0, p.y_at(4.0, 0.0) or 0.0, 0.0],
            [8.0, p.y_at(8.0, 0.0) or 0.0, 0.0],
            [8.0, p.y_at(8.0, 4.0) or 0.0, 4.0],
            [4.0, p.y_at(4.0, 4.0) or 0.0, 4.0],
        ],
        cell_id="cell:0:room:0:1",
        source_idx=0,
    )
    merged = clean_split_oblique_surfaces([s1, s2])
    assert len(merged) == 1


def test_clean_split_keeps_same_source_per_room_arrangement() -> None:
    """Same source oblique, *different* rooms → DO NOT merge.
    Regression guard: Layer 2 makes per-room slices share an exact plane;
    merging them would collapse arrangement.split's per-room fan-out."""
    plane = (0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 1.0)
    p = Plane(a=plane[0], b=plane[1], c=plane[2], d=plane[3])
    s1 = _split_surface(
        *plane,
        corners=[
            [0.0, p.y_at(0.0, 0.0) or 0.0, 0.0],
            [4.0, p.y_at(4.0, 0.0) or 0.0, 0.0],
            [4.0, p.y_at(4.0, 4.0) or 0.0, 4.0],
            [0.0, p.y_at(0.0, 4.0) or 0.0, 4.0],
        ],
        cell_id="cell:0:room:0:0",
        source_idx=0,
    )
    s2 = _split_surface(
        *plane,
        corners=[
            [4.0, p.y_at(4.0, 0.0) or 0.0, 0.0],
            [8.0, p.y_at(8.0, 0.0) or 0.0, 0.0],
            [8.0, p.y_at(8.0, 4.0) or 0.0, 4.0],
            [4.0, p.y_at(4.0, 4.0) or 0.0, 4.0],
        ],
        cell_id="cell:0:room:1:0",
        source_idx=0,
    )
    merged = clean_split_oblique_surfaces([s1, s2])
    assert len(merged) == 2


def test_clean_split_merges_cross_source_coplanar_abutting() -> None:
    """Different source obliques, coplanar, abutting → merge (the original
    `roof_cross_extension` cohort fix)."""
    plane = (0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 1.0)
    p = Plane(a=plane[0], b=plane[1], c=plane[2], d=plane[3])
    s1 = _split_surface(
        *plane,
        corners=[
            [0.0, p.y_at(0.0, 0.0) or 0.0, 0.0],
            [4.0, p.y_at(4.0, 0.0) or 0.0, 0.0],
            [4.0, p.y_at(4.0, 4.0) or 0.0, 4.0],
            [0.0, p.y_at(0.0, 4.0) or 0.0, 4.0],
        ],
        cell_id="cell:0:room:0:0",
        source_idx=0,
    )
    s2 = _split_surface(
        *plane,
        corners=[
            [4.0, p.y_at(4.0, 0.0) or 0.0, 0.0],
            [8.0, p.y_at(8.0, 0.0) or 0.0, 0.0],
            [8.0, p.y_at(8.0, 4.0) or 0.0, 4.0],
            [4.0, p.y_at(4.0, 4.0) or 0.0, 4.0],
        ],
        cell_id="cell:1:room:1:0",
        source_idx=1,
    )
    merged = clean_split_oblique_surfaces([s1, s2])
    assert len(merged) == 1


def test_clean_split_keeps_disjoint_coplanar_cells_separate() -> None:
    plane = (0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 1.0)
    p = Plane(a=plane[0], b=plane[1], c=plane[2], d=plane[3])
    s1 = _split_surface(
        *plane,
        corners=[
            [0.0, p.y_at(0.0, 0.0) or 0.0, 0.0],
            [3.0, p.y_at(3.0, 0.0) or 0.0, 0.0],
            [3.0, p.y_at(3.0, 3.0) or 0.0, 3.0],
            [0.0, p.y_at(0.0, 3.0) or 0.0, 3.0],
        ],
        cell_id="cell:0:room:0:0",
        source_idx=0,
    )
    s2 = _split_surface(
        *plane,
        corners=[
            [10.0, p.y_at(10.0, 0.0) or 0.0, 0.0],
            [13.0, p.y_at(13.0, 0.0) or 0.0, 0.0],
            [13.0, p.y_at(13.0, 3.0) or 0.0, 3.0],
            [10.0, p.y_at(10.0, 3.0) or 0.0, 3.0],
        ],
        cell_id="cell:1:room:1:0",
        source_idx=1,
    )
    merged = clean_split_oblique_surfaces([s1, s2])
    assert len(merged) == 2  # disjoint -> kept separate


def test_clean_split_disabled_is_identity() -> None:
    plane = (0.0, 0.866, 0.5, 1.0)
    s = _split_surface(
        *plane,
        corners=[[0, 0, 0], [4, 0, 0], [4, 1, 4], [0, 1, 4]],
        cell_id="cell:0:room:0:0",
        source_idx=0,
    )
    out = clean_split_oblique_surfaces([s, s], enabled=False)
    assert len(out) == 2
