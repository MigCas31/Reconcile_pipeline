"""Unit tests for the architectural-priors roof-defect classifiers.

The synthetic payloads here are minimal — just enough to exercise each rule's
gate. Real-cohort signal lives in the cohort_roof_report output; this file
locks in the rule semantics so refactors don't silently change them.
"""

from __future__ import annotations

import math

from reconcile_tiers.audit.roof_defects import (
    rule_roof_cross_extension,
    rule_roof_fragmented,
    rule_roof_irregular_footprint,
    rule_roof_off_axis,
)


def _square_floor_corners(size: float = 8.0) -> list[dict]:
    """Axis-aligned square floor, ground story."""
    return [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": size, "y": 0.0, "z": 0.0},
        {"x": size, "y": 0.0, "z": size},
        {"x": 0.0, "y": 0.0, "z": size},
    ]


def _wall_corners(
    x1: float, z1: float, x2: float, z2: float, height: float = 2.5
) -> list[dict]:
    return [
        {"x": x1, "y": 0.0, "z": z1},
        {"x": x2, "y": 0.0, "z": z2},
        {"x": x2, "y": height, "z": z2},
        {"x": x1, "y": height, "z": z1},
    ]


def _square_walls(size: float = 8.0) -> list[dict]:
    return [
        {"corners": _wall_corners(0, 0, size, 0), "locator_id": "wall::0"},
        {"corners": _wall_corners(size, 0, size, size), "locator_id": "wall::1"},
        {"corners": _wall_corners(size, size, 0, size), "locator_id": "wall::2"},
        {"corners": _wall_corners(0, size, 0, 0), "locator_id": "wall::3"},
    ]


def _slope_piece(
    locator_kind: str,
    idx: int,
    plane_a: float,
    plane_b: float,
    plane_c: float,
    plane_d: float,
    corners: list[dict],
) -> dict:
    return {
        "locator_id": f"u::tier-ceiling-{locator_kind}::{idx}",
        "source": "computed_oblique",
        "plane": {"a": plane_a, "b": plane_b, "c": plane_c, "d": plane_d},
        "corners": corners,
    }


def _axis_aligned_payload() -> dict:
    """Square footprint, single axis-aligned gable slope (
        ridge runs along X,
        slope faces +Z,
    )."""
    incl = math.radians(30.0)
    a, b, c = 0.0, math.cos(incl), math.sin(incl)
    corners = [
        {"x": 0.0, "y": 2.5, "z": 0.0},
        {"x": 8.0, "y": 2.5, "z": 0.0},
        {"x": 8.0, "y": 4.5, "z": 4.0},
        {"x": 0.0, "y": 4.5, "z": 4.0},
    ]
    return {
        "uuid": "u",
        "rooms": [
            {
                "story": 0,
                "floor": [{"corners": _square_floor_corners()}],
                "walls": _square_walls(),
                "locator_id": "u::room::0",
            }
        ],
        "ceiling": [_slope_piece("computed-oblique", 0, a, b, c, 0.0, corners)],
    }


def test_axis_aligned_slope_does_not_fire_off_axis() -> None:
    payload = _axis_aligned_payload()
    assert rule_roof_off_axis(payload) == []


def test_rotated_slope_fires_off_axis() -> None:
    payload = _axis_aligned_payload()
    # Rotate slope normal by 30 degrees in XZ -> az shifts away from axis
    incl = math.radians(30.0)
    rot = math.radians(30.0)
    a = math.cos(incl) * math.sin(rot) * 0  # keep b dominant
    a = math.sin(incl) * math.sin(rot)
    b = math.cos(incl)
    c = math.sin(incl) * math.cos(rot)
    payload["ceiling"][0]["plane"] = {"a": a, "b": b, "c": c, "d": 0.0}
    flags = rule_roof_off_axis(payload)
    assert len(flags) == 1
    assert flags[0]["rule"] == "roof_off_axis"
    assert flags[0]["evidence"]["delta_mod90_deg"] > 15.0


def test_off_axis_does_not_fire_on_aligned_slope_with_rotated_walls() -> None:
    """Regression for the compass-vs-math convention bug.

    Walls at math angle 60 deg → compass slope direction perpendicular = 30 deg
    (because compass = 90 - math). slope_az mod 90 = 30 vs wall axis mod 90 = 60
    looks like a 30 deg misalignment if you compare the two values directly,
    but they're in different conventions; the slope is actually aligned.
    The fix (`_axis_misalignment_deg`) takes (slope + wall) mod 90 → 0.
    """
    payload = _axis_aligned_payload()
    # Rotate walls to math 60 deg / 150 deg (perpendicular pair).
    rot = math.radians(60.0)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)

    def rot_xz(x: float, z: float) -> tuple[float, float]:
        return (x * cos_r - z * sin_r, x * sin_r + z * cos_r)

    new_walls = []
    for wall in payload["rooms"][0]["walls"]:
        rotated = []
        for c in wall["corners"]:
            x2, z2 = rot_xz(c["x"], c["z"])
            rotated.append({"x": x2, "y": c["y"], "z": z2})
        new_walls.append({"corners": rotated, "locator_id": wall["locator_id"]})
    payload["rooms"][0]["walls"] = new_walls
    new_floor = []
    for c in payload["rooms"][0]["floor"][0]["corners"]:
        x2, z2 = rot_xz(c["x"], c["z"])
        new_floor.append({"x": x2, "y": c["y"], "z": z2})
    payload["rooms"][0]["floor"][0]["corners"] = new_floor

    # Rotate the slope plane normal by the same angle so it stays aligned.
    incl = math.radians(30.0)
    # Original normal: (0, cos(incl), sin(incl)). Rotate XZ components.
    a_orig, c_orig = 0.0, math.sin(incl)
    a_rot = a_orig * cos_r - c_orig * sin_r
    c_rot = a_orig * sin_r + c_orig * cos_r
    payload["ceiling"][0]["plane"] = {
        "a": a_rot,
        "b": math.cos(incl),
        "c": c_rot,
        "d": 0.0,
    }
    new_corners = []
    for c in payload["ceiling"][0]["corners"]:
        x2, z2 = rot_xz(c["x"], c["z"])
        new_corners.append({"x": x2, "y": c["y"], "z": z2})
    payload["ceiling"][0]["corners"] = new_corners

    assert rule_roof_off_axis(payload) == [], (
        "rotated-but-aligned slope should not fire off-axis"
    )


def test_two_pieces_same_plane_fires_fragmented() -> None:
    payload = _axis_aligned_payload()
    plane = payload["ceiling"][0]["plane"]
    payload["ceiling"].append(
        _slope_piece(
            "computed-oblique",
            1,
            plane["a"],
            plane["b"],
            plane["c"],
            plane["d"],
            payload["ceiling"][0]["corners"],
        )
    )
    flags = rule_roof_fragmented(payload)
    # Both pieces flagged (group_size = 2)
    assert len(flags) == 2
    assert all(f["rule"] == "roof_fragmented" for f in flags)
    assert all(f["evidence"]["group_size"] == 2 for f in flags)


def test_per_room_emission_is_not_counted_as_fragment() -> None:
    """Pieces with locator kind ending in `-room` are deduplicated out."""
    payload = _axis_aligned_payload()
    plane = payload["ceiling"][0]["plane"]
    payload["ceiling"].append(
        _slope_piece(
            "computed-oblique-room",
            0,
            plane["a"],
            plane["b"],
            plane["c"],
            plane["d"],
            payload["ceiling"][0]["corners"],
        )
    )
    assert rule_roof_fragmented(payload) == []


def test_irregular_footprint_fires_when_walls_span_many_angles() -> None:
    """
    Hexagon-ish wall set spreads length across many angles -> coverage < 0.70 ->
    irregular.
    """
    # Six walls at 0/30/60/90/120/150 degrees, equal length. Any single axis
    # captures only ~1/3 of total wall length, well below the 0.70 gate.
    angles = [0, 30, 60, 90, 120, 150]
    walls = []
    length = 5.0
    for idx, deg in enumerate(angles):
        rad = math.radians(deg)
        x2 = length * math.cos(rad)
        z2 = length * math.sin(rad)
        walls.append(
            {
                "corners": _wall_corners(0.0, 0.0, x2, z2),
                "locator_id": f"wall::{idx}",
            }
        )
    payload = _axis_aligned_payload()
    payload["rooms"][0]["walls"] = walls
    flags = rule_roof_irregular_footprint(payload)
    assert len(flags) == 1
    assert flags[0]["rule"] == "roof_irregular_footprint"
    assert flags[0]["evidence"]["rectilinearity_coverage"] < 0.70


def test_single_wing_does_not_fire_cross_extension() -> None:
    """Square footprint -> single wing -> no cross-extension flag possible."""
    payload = _axis_aligned_payload()
    assert rule_roof_cross_extension(payload) == []


def test_extension_strip_crosses_wing_flags_strip_spanning_two_wings() -> None:
    """L-shape footprint with a descent strip whose XZ projection spans both
    wings should fire the new rule. A strip confined to one wing should not.
    """
    from reconcile_tiers.audit.roof_defects import rule_extension_strip_crosses_wing

    # L-shape footprint: vertical limb (0,0)-(4,8) + horizontal extension (4,0)-(8,4).
    # Walls trace the L-shape boundary.
    walls_l = [
        {"corners": _wall_corners(0, 0, 8, 0), "locator_id": "wall::0"},
        {"corners": _wall_corners(8, 0, 8, 4), "locator_id": "wall::1"},
        {"corners": _wall_corners(8, 4, 4, 4), "locator_id": "wall::2"},
        {"corners": _wall_corners(4, 4, 4, 8), "locator_id": "wall::3"},
        {"corners": _wall_corners(4, 8, 0, 8), "locator_id": "wall::4"},
        {"corners": _wall_corners(0, 8, 0, 0), "locator_id": "wall::5"},
    ]
    floor_l = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 8.0, "y": 0.0, "z": 0.0},
        {"x": 8.0, "y": 0.0, "z": 4.0},
        {"x": 4.0, "y": 0.0, "z": 4.0},
        {"x": 4.0, "y": 0.0, "z": 8.0},
        {"x": 0.0, "y": 0.0, "z": 8.0},
    ]

    # Strip A (cross-wing): a quad spanning x in [2, 6] at z=2, with non-trivial
    # XZ extent that intersects both the vertical limb (x<4) and the horizontal
    # extension (x>4) by > CROSS_EXTENSION_MIN_OVERLAP_M2 = 1.0 m².
    cross_strip = [
        {"x": 2.0, "y": 2.5, "z": 1.5},
        {"x": 6.0, "y": 2.5, "z": 1.5},
        {"x": 6.0, "y": 2.5, "z": 2.5},
        {"x": 2.0, "y": 2.5, "z": 2.5},
    ]
    # Strip B (single-wing, sits entirely in the vertical limb).
    inner_strip = [
        {"x": 1.0, "y": 2.5, "z": 5.0},
        {"x": 3.0, "y": 2.5, "z": 5.0},
        {"x": 3.0, "y": 2.5, "z": 6.5},
        {"x": 1.0, "y": 2.5, "z": 6.5},
    ]
    walls_l[0] = {**walls_l[0], "descent_strip": cross_strip, "uplift_strip": None}
    walls_l[3] = {**walls_l[3], "descent_strip": inner_strip, "uplift_strip": None}

    payload = {
        "uuid": "u",
        "rooms": [
            {
                "story": 0,
                "floor": [{"corners": floor_l}],
                "walls": walls_l,
                "locator_id": "u::room::0",
            }
        ],
        "ceiling": [],
    }

    flags = rule_extension_strip_crosses_wing(payload)
    cross_flags = [f for f in flags if f["locator"] == "wall::0"]
    inner_flags = [f for f in flags if f["locator"] == "wall::3"]
    assert len(cross_flags) == 1, "cross-wing descent strip should fire the rule"
    assert cross_flags[0]["evidence"]["strip_kind"] == "descent_strip"
    assert len(cross_flags[0]["evidence"]["wing_indices"]) >= 2
    assert inner_flags == [], "single-wing strip should not fire the rule"


def test_extension_strip_crosses_wing_no_flags_when_single_wing() -> None:
    """Square footprint -> single wing -> never fires regardless of strips."""
    from reconcile_tiers.audit.roof_defects import rule_extension_strip_crosses_wing

    payload = _axis_aligned_payload()
    payload["rooms"][0]["walls"][0] = {
        **payload["rooms"][0]["walls"][0],
        "descent_strip": [
            {"x": 0.0, "y": 2.5, "z": 0.0},
            {"x": 8.0, "y": 2.5, "z": 0.0},
            {"x": 8.0, "y": 2.5, "z": 8.0},
            {"x": 0.0, "y": 2.5, "z": 8.0},
        ],
        "uplift_strip": None,
    }
    assert rule_extension_strip_crosses_wing(payload) == []
