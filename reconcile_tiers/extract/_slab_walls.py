"""Slab-quad emission for facing wall pairs.

Builds `ExtractedGapWall` surfaces from `SlabQuad`s produced by
`find_facing_pairs`. Each slab is subdivided at every breakpoint in
either wall's top-edge profile, so the emitted wall geometry follows
descent strips and uplift strips precisely.

Extracted from `wall_pairs.py`; the public entry points
(`compute_slab_walls`, `slab_locator_id`) are re-exported there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from reconcile_tiers.extract.building import ExtractedGapWall, ExtractedRoom
from reconcile_tiers.extract.wall_pairs import (
    ENDPOINT_SNAP_TOL_M,
    SlabQuad,
    WallRun,
    _drop_perpendicular,
    _snap_to_endpoint,
    _t_along,
    find_facing_pairs,
)


def slab_locator_id(slab: SlabQuad, kind: str, idx: int = 0) -> str:
    """Stable id for a slab-derived gap surface. Uses the lower
    (room_index, wall_index) pair to keep IDs deterministic regardless of
    which wall was wall_a in the iteration."""
    a = (slab.wall_a.room_index, slab.wall_a.wall_index)
    b = (slab.wall_b.room_index, slab.wall_b.wall_index)
    lo, hi = sorted((a, b))
    return f"slab:r{lo[0]}w{lo[1]}-r{hi[0]}w{hi[1]}:{kind}:{idx}"


def _slab_breakpoints(slab: SlabQuad) -> list[float]:
    """Tangent positions (in wall_a's frame) where the slab's top profile
    can change slope. Includes:
      - The two slab end positions (a0 / a1 on wall_a).
      - Every wall_a top-profile breakpoint inside the slab.
      - Every wall_b top-profile breakpoint, projected onto wall_a's line
        and converted to wall_a's tangent parameter.
    Sorted, deduped (within 1 cm)."""
    ta_start = _t_along(slab.a0, slab.wall_a)
    ta_end = _t_along(slab.a1, slab.wall_a)
    if ta_end < ta_start:
        ta_start, ta_end = ta_end, ta_start
    candidates: list[float] = [ta_start, ta_end]
    for t_a, _y in slab.wall_a.top_profile:
        if ta_start - 0.001 < t_a < ta_end + 0.001:
            candidates.append(t_a)
    # Map wall_b breakpoints into wall_a's tangent space by dropping the
    # perpendicular from each (wall_b point at t_b) onto wall_a.
    for t_b, _y in slab.wall_b.top_profile:
        pt_b = (
            slab.wall_b.p0[0] + t_b * slab.wall_b.tangent[0],
            slab.wall_b.p0[1] + t_b * slab.wall_b.tangent[1],
        )
        foot = _drop_perpendicular(pt_b, slab.wall_a)
        t_a = _t_along(foot, slab.wall_a)
        if ta_start - 0.001 < t_a < ta_end + 0.001:
            candidates.append(t_a)
    candidates.sort()
    out: list[float] = []
    tol = 0.01
    for t in candidates:
        if not out or abs(t - out[-1]) > tol:
            out.append(t)
    return out


def _slab_corner_at_t(
    slab: SlabQuad, t_a: float
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """At tangent position t_a along wall_a, return (a, b, t_b, top_y).
      - a = the point on wall_a's line at t_a.
      - b = the perpendicular foot of a on wall_b's line.
      - t_b = b's tangent parameter on wall_b.
      - top_y = lower of the two wall tops at this slab cross-section.
    Snaps a / b to wall endpoints when within ENDPOINT_SNAP_TOL_M."""
    a = (
        slab.wall_a.p0[0] + t_a * slab.wall_a.tangent[0],
        slab.wall_a.p0[1] + t_a * slab.wall_a.tangent[1],
    )
    a = _snap_to_endpoint(a, slab.wall_a)
    b = _drop_perpendicular(a, slab.wall_b)
    b = _snap_to_endpoint(b, slab.wall_b)
    t_b = _t_along(b, slab.wall_b)
    top_y = min(slab.wall_a.top_y_at(t_a), slab.wall_b.top_y_at(t_b))
    return a, b, t_b, top_y


def _slab_cap_y_at(
    slab: SlabQuad,
    xz: tuple[float, float],
    fallback_top_y: float,
) -> float:
    ceiling_candidates: list[float] = []
    for run in (slab.wall_a, slab.wall_b):
        y = run.ceiling_y_at(xz[0], xz[1])
        if y is not None:
            ceiling_candidates.append(float(y))
    if ceiling_candidates:
        return min(ceiling_candidates)
    return fallback_top_y


def _run_bottom_y_at_xz(
    run: WallRun, xz: tuple[float, float], fallback_floor_y: float
) -> float:
    t = _t_along(xz, run)
    bottom_y = float(run.bottom_y_at(t))
    if math.hypot(xz[0] - run.p0[0], xz[1] - run.p0[1]) <= ENDPOINT_SNAP_TOL_M:
        bottom_y = min(bottom_y, run.endpoint_bottoms[0])
    if math.hypot(xz[0] - run.p1[0], xz[1] - run.p1[1]) <= ENDPOINT_SNAP_TOL_M:
        bottom_y = min(bottom_y, run.endpoint_bottoms[1])
    return min(float(fallback_floor_y), bottom_y)


def compute_slab_walls(
    rooms: Sequence[ExtractedRoom],
    *,
    confidence: str = "high",
) -> list[ExtractedGapWall]:
    """Top-level: detect facing wall pairs across the building and emit
    ``ExtractedGapWall`` surfaces. Each slab is **subdivided** at every
    breakpoint in either wall's top-edge profile (a profile breakpoint =
    a wall corner or descent/uplift-strip sample point that introduces a
    slope change). Each sub-strip emits its own four surfaces:

      - gap_floor (horizontal quad at floor y).
      - gap_ceiling (quad following the lower of the two facing walls'
        top profiles at this sub-strip).
      - within_story:edge:0/1 (vertical short-end caps).

    A wall with multiple slope changes (e.g., flat-then-sloped, or two
    different slopes meeting at a kink) produces multiple sub-strips,
    each capped by the wall's actual top-edge profile at that section —
    no straight-line interpolation across kinks."""
    slabs = find_facing_pairs(rooms)
    out: list[ExtractedGapWall] = []
    for slab in slabs:
        breakpoints = _slab_breakpoints(slab)
        if len(breakpoints) < 2:
            continue
        floor_y = slab.floor_y
        story = rooms[slab.wall_a.room_index].story
        sub_corners: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        for t in breakpoints:
            a, b, _t_b, top_y = _slab_corner_at_t(slab, t)
            sub_corners.append((a, b, top_y))
        for sub_idx in range(len(sub_corners) - 1):
            a0, b0, top_y0 = sub_corners[sub_idx]
            a1, b1, top_y1 = sub_corners[sub_idx + 1]
            # Skip degenerate sub-strips (zero tangent length).
            if math.hypot(a1[0] - a0[0], a1[1] - a0[1]) < ENDPOINT_SNAP_TOL_M:
                continue
            a0_3d = [a0[0], floor_y, a0[1]]
            a1_3d = [a1[0], floor_y, a1[1]]
            b0_3d = [b0[0], floor_y, b0[1]]
            b1_3d = [b1[0], floor_y, b1[1]]
            a0_bottom = [a0[0], _run_bottom_y_at_xz(slab.wall_a, a0, floor_y), a0[1]]
            a1_bottom = [a1[0], _run_bottom_y_at_xz(slab.wall_a, a1, floor_y), a1[1]]
            b0_bottom = [b0[0], _run_bottom_y_at_xz(slab.wall_b, b0, floor_y), b0[1]]
            b1_bottom = [b1[0], _run_bottom_y_at_xz(slab.wall_b, b1, floor_y), b1[1]]
            a0_top = [a0[0], _slab_cap_y_at(slab, a0, top_y0), a0[1]]
            a1_top = [a1[0], _slab_cap_y_at(slab, a1, top_y1), a1[1]]
            b0_top = [b0[0], _slab_cap_y_at(slab, b0, top_y0), b0[1]]
            b1_top = [b1[0], _slab_cap_y_at(slab, b1, top_y1), b1[1]]
            out.append(
                ExtractedGapWall(
                    id=slab_locator_id(slab, "gap_floor", sub_idx),
                    corners=[a0_3d, a1_3d, b1_3d, b0_3d],
                    type="gap_floor",
                    story=story,
                    confidence=confidence,
                )
            )
            out.append(
                ExtractedGapWall(
                    id=slab_locator_id(slab, "gap_ceiling", sub_idx),
                    corners=[a0_top, a1_top, b1_top, b0_top],
                    type="gap_ceiling",
                    story=story,
                    confidence=confidence,
                )
            )
            # Short-end caps only at the slab's outer ends (sub_idx == 0
            # for the start, sub_idx == last for the end). Internal
            # sub-strip boundaries don't get short-end caps because the
            # neighbouring sub-strip is on the same side of the slab.
            if sub_idx == 0:
                out.append(
                    ExtractedGapWall(
                        id=slab_locator_id(slab, "within_story", 0),
                        corners=[a0_bottom, b0_bottom, b0_top, a0_top],
                        type="within_story",
                        story=story,
                        confidence=confidence,
                    )
                )
            if sub_idx == len(sub_corners) - 2:
                out.append(
                    ExtractedGapWall(
                        id=slab_locator_id(slab, "within_story", 1),
                        corners=[a1_bottom, b1_bottom, b1_top, a1_top],
                        type="within_story",
                        story=story,
                        confidence=confidence,
                    )
                )
    return out
