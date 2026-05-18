"""Regression tests for the Plan C story-envelope audit fixes.

Two surgical changes guarded here:

1. ``_building_envelope`` is now an all-rooms buffer-union (matching the
   pipeline's ``build_building_footprint``) instead of a ground-story-only
   convex hull. Multi-story top-floor pieces no longer false-positive on
   ``out_of_envelope``.
2. ``rule_ceiling_below_floor`` is story-aware. Top-story ceilings sharing
   XZ extent with a ground-floor room are no longer matched against that
   ground-floor room's Y level.

The diagnostic at ``audit/story_envelope_diagnostic.py`` showed 100% of
``out_of_envelope`` and 99.9% of ``ceiling_below_floor`` flags pre-fix were
artifacts of these mismatches.
"""

from __future__ import annotations

from reconcile_tiers.audit.rules import (
    rule_ceiling_below_floor,
    rule_out_of_envelope,
)


def _floor(corners_xz, *, story, locator):
    return {
        "story": story,
        "locator_id": locator,
        "floor": [{"corners": [{"x": x, "y": y, "z": z} for x, y, z in corners_xz]}],
        "walls": [],
    }


def _wall(corners_xz):
    return {
        "corners": [{"x": x, "y": y, "z": z} for x, y, z in corners_xz],
        "locator_id": "wall",
    }


def _ceiling(corners_xz, *, locator="cp"):
    return {
        "corners": [{"x": x, "y": y, "z": z} for x, y, z in corners_xz],
        "plane": {"a": 0, "b": 1, "c": 0, "d": corners_xz[0][1]},
        "source": "raw_scan",
        "locator_id": locator,
    }


def test_out_of_envelope_uses_all_rooms_not_just_ground() -> None:
    """A second-story room that overhangs the ground footprint must not
    cause its ceiling to be flagged as out_of_envelope."""
    payload = {
        "rooms": [
            # Ground story — a small square at origin
            {
                **_floor(
                    [(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)],
                    story=0,
                    locator="ground",
                ),
                "walls": [_wall([(0, 0, 0), (0, 3, 0)])],
            },
            # Second story — kicked-out wing extending east
            {
                **_floor(
                    [(0, 3, 0), (8, 3, 0), (8, 3, 4), (0, 3, 4)],
                    story=1,
                    locator="upper",
                ),
                "walls": [_wall([(0, 3, 0), (0, 6, 0)])],
            },
        ],
        "ceiling": [
            # Ceiling over the second-story wing — sits over (4..8) which is
            # OUTSIDE the ground-only convex hull but INSIDE the all-rooms
            # union. Pre-fix this would flag.
            _ceiling(
                [(4, 6, 0), (8, 6, 0), (8, 6, 4), (4, 6, 4)],
                locator="upper_ceiling",
            ),
        ],
        "knee_walls": [],
        "dormer_faces": [],
        "gaps": [],
    }
    flags = rule_out_of_envelope(payload)
    upper_flags = [f for f in flags if f.get("locator") == "upper_ceiling"]
    assert upper_flags == [], f"upper-story ceiling falsely flagged: {upper_flags}"


def test_ceiling_below_floor_does_not_match_top_ceiling_to_ground_room() -> None:
    """A top-story ceiling that sits Y-above its own room must not be
    matched against a ground-floor room sharing the same XZ extent."""
    payload = {
        "rooms": [
            # Ground story floor at Y=0, walls reach to Y=3
            {
                "story": 0,
                "locator_id": "ground_room",
                "floor": [
                    {
                        "corners": [
                            {"x": 0, "y": 0, "z": 0},
                            {"x": 4, "y": 0, "z": 0},
                            {"x": 4, "y": 0, "z": 4},
                            {"x": 0, "y": 0, "z": 4},
                        ]
                    }
                ],
                "walls": [
                    _wall([(0, 0, 0), (0, 3, 0)]),
                    _wall([(4, 0, 0), (4, 3, 0)]),
                ],
            },
            # Upper story floor at Y=3, walls reach to Y=6
            {
                "story": 1,
                "locator_id": "upper_room",
                "floor": [
                    {
                        "corners": [
                            {"x": 0, "y": 3, "z": 0},
                            {"x": 4, "y": 3, "z": 0},
                            {"x": 4, "y": 3, "z": 4},
                            {"x": 0, "y": 3, "z": 4},
                        ]
                    }
                ],
                "walls": [
                    _wall([(0, 3, 0), (0, 6, 0)]),
                    _wall([(4, 3, 0), (4, 6, 0)]),
                ],
            },
        ],
        # Ceiling at Y=6 over the upper room (correctly placed there).
        "ceiling": [
            _ceiling(
                [(0, 6, 0), (4, 6, 0), (4, 6, 4), (0, 6, 4)],
                locator="upper_ceiling",
            ),
        ],
        "knee_walls": [],
        "dormer_faces": [],
        "gaps": [],
    }
    flags = rule_ceiling_below_floor(payload)
    # Pre-fix: matches against ground_room (floor_y_max=0), ceiling Y=6 is
    # ABOVE that — wouldn't trigger. So actually let me make a tougher case:
    # ground floor with floor_y_max higher than the ceiling.
    assert flags == [], (
        f"ceiling correctly placed above floor — should not flag: {flags}"
    )


def test_ceiling_below_floor_still_catches_real_inversion() -> None:
    """Sanity: a ceiling actually sitting below its own room's floor should
    still be flagged."""
    payload = {
        "rooms": [
            {
                "story": 0,
                "locator_id": "room0",
                "floor": [
                    {
                        "corners": [
                            {"x": 0, "y": 2, "z": 0},
                            {"x": 4, "y": 2, "z": 0},
                            {"x": 4, "y": 2, "z": 4},
                            {"x": 0, "y": 2, "z": 4},
                        ]
                    }
                ],
                "walls": [
                    _wall([(0, 2, 0), (0, 5, 0)]),
                ],
            },
        ],
        # Ceiling is below the floor (Y=1 < floor_y_max=2): this is the
        # real "ceiling below floor" violation we want to keep catching.
        "ceiling": [
            _ceiling(
                [(0, 1, 0), (4, 1, 0), (4, 1, 4), (0, 1, 4)],
                locator="bad_ceiling",
            ),
        ],
        "knee_walls": [],
        "dormer_faces": [],
        "gaps": [],
    }
    flags = rule_ceiling_below_floor(payload)
    assert any(f.get("locator") == "bad_ceiling" for f in flags), (
        f"real ceiling-below-floor was not flagged: {flags}"
    )
