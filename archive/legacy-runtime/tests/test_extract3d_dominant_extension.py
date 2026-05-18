"""Unit tests for the dominant-height wall extension fallback.

Covers the helpers in ``reconcile/extract3d/ceilings.py`` that power the
"close the top-of-wall gap when the perimeter is uniform" heuristic (see
``scripts/audit_dominant_height_closure.py`` and the tracking_progress
entry from 2026-04-20).
"""

from __future__ import annotations

import pytest

from reconcile.extract3d.ceilings import (
    COHORT_TOLERANCE_M,
    MIN_DOMINANT_DELTA_M,
    compute_story_wall_top_cohort,
    extend_wall_to_dominant,
    should_extend_wall_to_dominant,
)

FLOOR_Y = 0.0
DOMINANT_TOP_Y = 2.50


def _wall_quad(
    x0: float, z0: float, x1: float, z1: float, top_y: float, bot_y: float = FLOOR_Y
) -> dict:
    """Return a canonical ``[bl, br, tr, tl]`` wall along the (x0,z0)→(x1,z1) line."""
    return {
        "corners": [
            [x0, bot_y, z0],
            [x1, bot_y, z1],
            [x1, top_y, z1],
            [x0, top_y, z0],
        ],
        "id": f"wall-{x0:.2f}-{z0:.2f}-{x1:.2f}-{z1:.2f}",
    }


def _rooms_with_walls(walls: list[dict], story: int = 0) -> list[dict]:
    return [{"story": story, "walls_computed": walls}]


def _uniform_cohort_walls(short: dict) -> list[dict]:
    """Six dominant-height walls forming a closed loop + one short wall inside the loop.

    Layout (top view, X right, Z down):
        (0,0) ──A── (4,0) ──B── (8,0)
          │                         │
          F            short        C
          │                         │
        (0,6) ──E── (4,6) ──D── (8,6)

    A,B,C,D,E,F are tall (top=2.50) — total perimeter 28 m.
    ``short`` is the wall under test.
    """
    return [
        _wall_quad(0.0, 0.0, 4.0, 0.0, DOMINANT_TOP_Y),  # A
        _wall_quad(4.0, 0.0, 8.0, 0.0, DOMINANT_TOP_Y),  # B
        _wall_quad(8.0, 0.0, 8.0, 6.0, DOMINANT_TOP_Y),  # C
        _wall_quad(8.0, 6.0, 4.0, 6.0, DOMINANT_TOP_Y),  # D
        _wall_quad(4.0, 6.0, 0.0, 6.0, DOMINANT_TOP_Y),  # E
        _wall_quad(0.0, 6.0, 0.0, 0.0, DOMINANT_TOP_Y),  # F
        short,
    ]


# ---------------------------------------------------------------------------
# Fixture A — uniform cohort, short wall aligned with dominant neighbours
# ---------------------------------------------------------------------------


def test_fixtureA_short_wall_colinear_with_neighbour_is_promoted():
    """A 2 m wall colinear with neighbour A (top=2.50), 0.25 m shorter, same floor.

    The short wall sits on the top edge y=0, just offset slightly inside (z=0.10)
    so it's a plausible scan-gap remnant alongside A/B.
    """
    short = _wall_quad(1.0, 0.10, 3.0, 0.10, 2.25)  # delta = 0.25
    short["id"] = "short-target"
    walls = _uniform_cohort_walls(short)
    cohort = compute_story_wall_top_cohort(_rooms_with_walls(walls), 0)
    assert cohort is not None
    assert cohort["coverage_frac"] == pytest.approx(28.0 / 30.0, abs=1e-6)
    assert cohort["dominant_y"] == pytest.approx(DOMINANT_TOP_Y, abs=1e-6)

    target = should_extend_wall_to_dominant(short["corners"], cohort)
    assert target == pytest.approx(DOMINANT_TOP_Y)

    ext = extend_wall_to_dominant(short["corners"], target)
    assert ext is not None
    lifted = [
        c for c in ext["extended_corners"] if c[1] == pytest.approx(DOMINANT_TOP_Y)
    ]
    assert len(lifted) == 2
    assert ext["extension_strip"]


# ---------------------------------------------------------------------------
# Fixture B — non-uniform cohort: no cohort exceeds min_cov
# ---------------------------------------------------------------------------


def test_fixtureB_non_uniform_cohort_returns_none():
    """When heights are scattered with no ≥70 % cluster, cohort lookup returns None."""
    walls = [
        _wall_quad(0.0, 0.0, 3.0, 0.0, 2.50),
        _wall_quad(3.0, 0.0, 6.0, 0.0, 2.00),
        _wall_quad(6.0, 0.0, 6.0, 3.0, 1.80),
        _wall_quad(6.0, 3.0, 3.0, 3.0, 2.20),
        _wall_quad(3.0, 3.0, 0.0, 3.0, 1.50),
        _wall_quad(0.0, 3.0, 0.0, 0.0, 2.40),
    ]
    cohort = compute_story_wall_top_cohort(_rooms_with_walls(walls), 0)
    assert cohort is None


# ---------------------------------------------------------------------------
# Fixture C — uniform cohort, but target's floor is stepped 0.30 m higher
# ---------------------------------------------------------------------------


def test_fixtureC_stepped_floor_blocks_promotion():
    """Floor guard blocks promotion when the target sits on a different level."""
    short = _wall_quad(
        1.0, 0.10, 3.0, 0.10, 2.25, bot_y=0.30
    )  # top delta OK, floor delta = 0.30
    short["id"] = "short-on-step"
    walls = _uniform_cohort_walls(short)
    cohort = compute_story_wall_top_cohort(_rooms_with_walls(walls), 0)
    assert cohort is not None

    target = should_extend_wall_to_dominant(short["corners"], cohort)
    assert target is None


# ---------------------------------------------------------------------------
# Fixture D — uniform cohort but target is offset from any dominant neighbour
# ---------------------------------------------------------------------------


def test_fixtureD_offset_wall_blocks_promotion():
    """Colinearity guard blocks promotion when no dominant neighbour is aligned.

    The short wall sits deep inside the room (not near any exterior wall) so no
    dominant-cohort neighbour is colinear within the perp/offset thresholds.
    """
    short = _wall_quad(2.0, 3.0, 4.0, 3.0, 2.25)
    short["id"] = "short-interior"
    walls = _uniform_cohort_walls(short)
    cohort = compute_story_wall_top_cohort(_rooms_with_walls(walls), 0)
    assert cohort is not None

    target = should_extend_wall_to_dominant(short["corners"], cohort)
    assert target is None


# ---------------------------------------------------------------------------
# Extra guard: noise-level deltas (< MIN_DOMINANT_DELTA_M) are NOT promoted.
# ---------------------------------------------------------------------------


def test_noise_delta_not_promoted():
    """A wall only 0.05 m shorter than dominant should stay put (scan noise)."""
    short = _wall_quad(1.0, 0.10, 3.0, 0.10, DOMINANT_TOP_Y - 0.05)
    walls = _uniform_cohort_walls(short)
    cohort = compute_story_wall_top_cohort(_rooms_with_walls(walls), 0)
    assert cohort is not None
    assert MIN_DOMINANT_DELTA_M > 0.05
    assert should_extend_wall_to_dominant(short["corners"], cohort) is None


def test_cohort_tolerance_used_for_clustering():
    """Two tops separated by less than the cohort tolerance still cluster as one."""
    walls = [
        _wall_quad(0.0, 0.0, 4.0, 0.0, DOMINANT_TOP_Y),
        _wall_quad(4.0, 0.0, 8.0, 0.0, DOMINANT_TOP_Y + COHORT_TOLERANCE_M - 0.01),
        _wall_quad(8.0, 0.0, 8.0, 6.0, DOMINANT_TOP_Y),
        _wall_quad(8.0, 6.0, 0.0, 6.0, DOMINANT_TOP_Y),
        _wall_quad(0.0, 6.0, 0.0, 0.0, DOMINANT_TOP_Y),
    ]
    cohort = compute_story_wall_top_cohort(_rooms_with_walls(walls), 0)
    assert cohort is not None
    assert cohort["coverage_frac"] == pytest.approx(1.0, abs=1e-6)
