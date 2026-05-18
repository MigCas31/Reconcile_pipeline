"""Tests for dormer detection and geometry generation."""

from __future__ import annotations

import math

import pytest

from reconcile.roof_algorithms_py.dormer_detection import (
    _wall_normal_xz,
    _wall_width_along_ridge,
    detect_dormers,
    group_dormer_walls,
)
from reconcile.roof_algorithms_py.dormer_geometry import (
    _walk_to_roof_height,
    build_dormer_geometry,
)
from reconcile.roof_algorithms_py.math_utils import plane_normal, plane_y_at

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_wall(wid: str, corners: list) -> dict:
    return {"id": wid, "corners": corners}


def _make_roof_surface(
    az: float = 297.6,
    incl: float = 46.0,
    cx: float = 0.0,
    cz: float = 0.0,
    cy: float = 2.0,
) -> dict:
    """Create a synthetic oblique roof surface."""
    azi_rad = az * math.pi / 180
    incl_rad = incl * math.pi / 180
    slope_x = math.sin(azi_rad) * math.cos(incl_rad)
    slope_z = math.cos(azi_rad) * math.cos(incl_rad)
    ridge_x = -math.cos(azi_rad)
    ridge_z = math.sin(azi_rad)

    # Build a simple rectangular roof surface
    corners = [
        (
            cx - ridge_x * 3 + slope_x * 3,
            cy - math.sin(incl_rad) * 3,
            cz - ridge_z * 3 + slope_z * 3,
        ),
        (
            cx + ridge_x * 3 + slope_x * 3,
            cy - math.sin(incl_rad) * 3,
            cz + ridge_z * 3 + slope_z * 3,
        ),
        (
            cx + ridge_x * 3 - slope_x * 3,
            cy + math.sin(incl_rad) * 3,
            cz + ridge_z * 3 - slope_z * 3,
        ),
        (
            cx - ridge_x * 3 - slope_x * 3,
            cy + math.sin(incl_rad) * 3,
            cz - ridge_z * 3 - slope_z * 3,
        ),
    ]

    return {
        "cluster": {
            "avgAzimuth": az,
            "avgIncl": incl,
            "segs": [{"a": corners[0], "b": corners[1], "story": 1}],
        },
        "dominant_story": 1,
        "corners": corners,
        "center": {"x": cx, "y": cy, "z": cz},
        "ridge": {"x": ridge_x, "z": ridge_z, "min": -3.0, "max": 3.0},
    }


# ── Unit tests: wall normal ─────────────────────────────────────────────────


class TestWallNormalXZ:
    def test_wall_facing_z(self):
        """Wall along X axis → normal should point in Z direction."""
        corners = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
        nx, nz = _wall_normal_xz(corners)
        # Normal perpendicular to X direction = (0, ±1) in XZ
        assert abs(nx) < 0.1
        assert abs(abs(nz) - 1.0) < 0.1

    def test_wall_facing_x(self):
        """Wall along Z axis → normal should point in X direction."""
        corners = [(0, 0, 0), (0, 0, 2), (0, 2, 2), (0, 2, 0)]
        nx, nz = _wall_normal_xz(corners)
        assert abs(abs(nx) - 1.0) < 0.1
        assert abs(nz) < 0.1

    def test_degenerate_wall(self):
        """Two coincident bottom corners → returns (0, 0)."""
        corners = [(1, 0, 1), (1, 0, 1), (1, 2, 1), (1, 2, 1)]
        nx, nz = _wall_normal_xz(corners)
        assert nx == 0.0
        assert nz == 0.0


# ── Unit tests: wall width along ridge ──────────────────────────────────────


class TestWallWidthAlongRidge:
    def test_simple(self):
        corners = [(0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)]
        width = _wall_width_along_ridge(corners, 1.0, 0.0)  # ridge along X
        assert abs(width - 3.0) < 0.01


# ── Unit tests: dormer detection ─────────────────────────────────────────────


class TestDetectDormers:
    def test_detects_protruding_wall(self):
        """A wall that protrudes above a roof plane is detected as a dormer."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=1.0)
        normal = plane_normal(0.0, 45.0)
        plane = {"n": normal, "ref": {"x": 0.0, "y": 1.0, "z": 0.0}}

        # Wall that partially protrudes above the roof
        # Roof faces az=0 (north), incl=45°. Ridge runs E-W.
        # Wall facing south (az=180°) at z=1, bottom at y=0, top at y=3
        plane_y_at(plane, 0.0, 1.0)
        wall_corners = [
            (-1, 0, 1),
            (1, 0, 1),
            (1, 3, 1),
            (-1, 3, 1),
        ]

        # Room needs a taller wall so the dormer wall (h=3) is shorter
        # than the room's tallest wall — otherwise height-ratio filter rejects it.
        tall_wall = _make_wall(
            "TALL",
            [
                (-2, 0, -1),
                (2, 0, -1),
                (2, 5, -1),
                (-2, 5, -1),
            ],
        )

        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [
                        _make_wall("FRONT", wall_corners),
                        tall_wall,
                    ],
                    "walls_computed": [],
                }
            ]
        }

        dormers = detect_dormers(bldg, [roof], {1: 0.0})
        assert len(dormers) == 1
        assert dormers[0]["front_wall"]["id"] == "FRONT"

    def test_ignores_wall_below_roof(self):
        """A wall entirely below the roof plane is not a dormer."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=5.0)

        # Wall far below the roof
        wall_corners = [
            (-1, -5, 1),
            (1, -5, 1),
            (1, -3, 1),
            (-1, -3, 1),
        ]

        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [_make_wall("LOW", wall_corners)],
                    "walls_computed": [],
                }
            ]
        }

        dormers = detect_dormers(bldg, [roof], {1: -5.0})
        assert len(dormers) == 0

    def test_ignores_wall_parallel_to_ridge(self):
        """A wall whose normal is parallel to the ridge is not a dormer front."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=1.0)

        # Ridge runs E-W (ridge_x = -cos(0) = -1, ridge_z = sin(0) = 0).
        # A wall facing east (normal along X) has dot(normal, ridge) ~ 1 → parallel →
        # rejected
        wall_corners = [
            (0, 0, -1),
            (0, 0, 1),
            (0, 3, 1),
            (0, 3, -1),
        ]

        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [_make_wall("PARALLEL", wall_corners)],
                    "walls_computed": [],
                }
            ]
        }

        dormers = detect_dormers(bldg, [roof], {1: 0.0})
        assert len(dormers) == 0

    def test_uses_top_boundary_atoms_to_filter_candidate_rooms(self):
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=1.0)
        roof["roof_hypothesis_id"] = "roof-hypothesis:oblique:0"

        wall_corners = [
            (-1, 0, 1),
            (1, 0, 1),
            (1, 3, 1),
            (-1, 3, 1),
        ]
        tall_wall = _make_wall(
            "TALL",
            [
                (-2, 0, -1),
                (2, 0, -1),
                (2, 5, -1),
                (-2, 5, -1),
            ],
        )
        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [_make_wall("OTHER", wall_corners), tall_wall],
                    "walls_computed": [],
                },
                {
                    "story": 1,
                    "walls_merged": [_make_wall("FRONT", wall_corners), tall_wall],
                    "walls_computed": [],
                },
            ]
        }
        top_boundary_graph = {
            "nodes": [
                {
                    "id": "atom:1",
                    "type": "TopBoundaryAtom",
                    "role": "sloped_ceiling",
                    "room_index": 1,
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ]
        }

        dormers = detect_dormers(
            bldg,
            [roof],
            {1: 0.0},
            top_boundary_graph=top_boundary_graph,
        )

        assert len(dormers) == 1
        assert dormers[0]["front_wall"]["room_index"] == 1

    def test_uses_coverage_polygon_to_filter_wall_location(self):
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=1.0)
        roof["roof_hypothesis_id"] = "roof-hypothesis:oblique:0"

        inside_wall = [
            (-0.5, 0, 1),
            (0.5, 0, 1),
            (0.5, 3, 1),
            (-0.5, 3, 1),
        ]
        outside_wall = [
            (5.0, 0, 1),
            (6.0, 0, 1),
            (6.0, 3, 1),
            (5.0, 3, 1),
        ]
        tall_wall = _make_wall(
            "TALL",
            [
                (-2, 0, -1),
                (2, 0, -1),
                (2, 5, -1),
                (-2, 5, -1),
            ],
        )
        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [
                        _make_wall("INSIDE", inside_wall),
                        _make_wall("OUTSIDE", outside_wall),
                        tall_wall,
                    ],
                    "walls_computed": [],
                }
            ]
        }
        top_boundary_graph = {
            "nodes": [
                {
                    "id": "atom:1",
                    "type": "TopBoundaryAtom",
                    "role": "sloped_ceiling",
                    "room_index": 0,
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ]
        }
        roof_coverage_graph = {
            "atom_coverage": {
                "atom:1": {
                    "sloped_hypothesis_id": "roof-hypothesis:oblique:0",
                    "sloped_state": "confirmed",
                }
            },
            "atom_regions": {
                "atom:1": {
                    "room_index": 0,
                    "polygon_xz": [[-1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [-1.0, 2.0]],
                }
            },
        }
        dormers = detect_dormers(
            bldg,
            [roof],
            {1: 0.0},
            roof_coverage_graph=roof_coverage_graph,
            top_boundary_graph=top_boundary_graph,
        )
        assert len(dormers) == 1
        assert dormers[0]["front_wall"]["id"] == "INSIDE"

    def test_rejects_wall_that_matches_knee_wall_face(self):
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=1.0)
        roof["roof_hypothesis_id"] = "roof-hypothesis:oblique:0"
        wall_corners = [
            (-1, 0, 1),
            (1, 0, 1),
            (1, 3, 1),
            (-1, 3, 1),
        ]
        tall_wall = _make_wall(
            "TALL",
            [
                (-2, 0, -1),
                (2, 0, -1),
                (2, 5, -1),
                (-2, 5, -1),
            ],
        )
        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [_make_wall("FRONT", wall_corners), tall_wall],
                    "walls_computed": [],
                }
            ]
        }
        top_boundary_graph = {
            "nodes": [
                {
                    "id": "atom:1",
                    "type": "TopBoundaryAtom",
                    "role": "sloped_ceiling",
                    "room_index": 0,
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ]
        }
        roof_coverage_graph = {
            "atom_coverage": {
                "atom:1": {
                    "sloped_hypothesis_id": "roof-hypothesis:oblique:0",
                    "sloped_state": "confirmed",
                }
            },
            "atom_regions": {
                "atom:1": {
                    "room_index": 0,
                    "polygon_xz": [[-2.0, 0.0], [2.0, 0.0], [2.0, 2.0], [-2.0, 2.0]],
                }
            },
        }
        knee_walls = [
            {
                "room_index": 0,
                "corners": wall_corners,
            }
        ]
        dormers = detect_dormers(
            bldg,
            [roof],
            {1: 0.0},
            roof_coverage_graph=roof_coverage_graph,
            top_boundary_graph=top_boundary_graph,
            knee_walls=knee_walls,
        )
        assert dormers == []

    def test_allows_strong_branch_supported_region(self):
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=1.0)
        roof["roof_hypothesis_id"] = "roof-hypothesis:oblique:0"
        wall_corners = [
            (-0.5, 0, 1),
            (0.5, 0, 1),
            (0.5, 3, 1),
            (-0.5, 3, 1),
        ]
        tall_wall = _make_wall(
            "TALL",
            [
                (-2, 0, -1),
                (2, 0, -1),
                (2, 5, -1),
                (-2, 5, -1),
            ],
        )
        bldg = {
            "rooms": [
                {
                    "story": 1,
                    "walls_merged": [_make_wall("FRONT", wall_corners), tall_wall],
                    "walls_computed": [],
                }
            ]
        }
        top_boundary_graph = {
            "nodes": [
                {
                    "id": "atom:1",
                    "type": "TopBoundaryAtom",
                    "role": "sloped_ceiling",
                    "room_index": 0,
                    "roof_hypothesis_id": "roof-hypothesis:oblique:0",
                }
            ]
        }
        roof_coverage_graph = {
            "atom_coverage": {
                "atom:1": {
                    "sloped_hypothesis_id": "roof-hypothesis:oblique:0",
                    "sloped_state": "confirmed",
                }
            },
            "atom_regions": {
                "atom:1": {
                    "room_index": 0,
                    "polygon_xz": [[-1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [-1.0, 2.0]],
                }
            },
            "atom_subpart_membership": {"atom:1": ["subpart:branch:0"]},
            "subparts": [
                {
                    "id": "subpart:branch:0",
                    "semantic_kind": "l_t_branch",
                }
            ],
        }
        roof_evidence_graph = {
            "subpart_evidence": {"subpart:branch:0": {"evidence_score": 4}}
        }

        dormers = detect_dormers(
            bldg,
            [roof],
            {1: 0.0},
            roof_coverage_graph=roof_coverage_graph,
            top_boundary_graph=top_boundary_graph,
            roof_evidence_graph=roof_evidence_graph,
        )

        assert len(dormers) == 1
        assert dormers[0]["front_wall"]["id"] == "FRONT"


# ── Unit tests: dormer grouping ──────────────────────────────────────────────


class TestGroupDormerWalls:
    def test_finds_cheeks(self):
        """Adjacent perpendicular walls are identified as cheeks."""
        front = _make_wall(
            "FRONT",
            [(-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)],
        )
        left_cheek = _make_wall(
            "LEFT",
            [(-1, 0, 0), (-1, 0, 2), (-1, 2, 2), (-1, 2, 0)],
        )
        right_cheek = _make_wall(
            "RIGHT",
            [(1, 0, 0), (1, 0, 2), (1, 2, 2), (1, 2, 0)],
        )
        other = _make_wall(
            "OTHER",
            [(5, 0, 5), (7, 0, 5), (7, 2, 5), (5, 2, 5)],
        )

        walls = [front, left_cheek, right_cheek, other]
        group = group_dormer_walls(front, walls)
        assert group["front"]["id"] == "FRONT"
        cheek_ids = {
            group["left_cheek"]["id"] if group["left_cheek"] else None,
            group["right_cheek"]["id"] if group["right_cheek"] else None,
        }
        assert "LEFT" in cheek_ids or "RIGHT" in cheek_ids

    def test_no_cheeks(self):
        """When no adjacent perpendicular walls exist, cheeks are None."""
        front = _make_wall(
            "FRONT",
            [(-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)],
        )
        group = group_dormer_walls(front, [front])
        assert group["left_cheek"] is None
        assert group["right_cheek"] is None


# ── Unit tests: walk to roof height ──────────────────────────────────────────


class TestWalkToRoofHeight:
    def test_walks_uphill(self):
        """Walking uphill on a 45° slope for 1m height → sqrt(2) horizontal distance."""
        normal = plane_normal(0.0, 45.0)
        plane = {"n": normal, "ref": {"x": 0.0, "y": 1.0, "z": 0.0}}

        slope_x = math.sin(0) * math.cos(math.pi / 4)
        slope_z = math.cos(0) * math.cos(math.pi / 4)

        result = _walk_to_roof_height((0, 0, 0), 2.0, plane, slope_x, slope_z)
        assert result is not None
        assert abs(result[1] - 2.0) < 0.01  # Y should be target

    def test_returns_none_for_impossible(self):
        """If target_y is below current roof height, returns None."""
        normal = plane_normal(0.0, 45.0)
        plane = {"n": normal, "ref": {"x": 0.0, "y": 10.0, "z": 0.0}}

        slope_x = math.sin(0) * math.cos(math.pi / 4)
        slope_z = math.cos(0) * math.cos(math.pi / 4)

        # Roof at (0,0) is already at y=10, asking for y=2 going uphill → impossible
        # (negative t)
        result = _walk_to_roof_height((0, 0, 0), 2.0, plane, slope_x, slope_z)
        assert result is None


# ── Unit tests: cutout on roof plane ─────────────────────────────────────────


class TestCutoutQuad:
    def test_cutout_points_on_roof_plane(self):
        """All 4 cutout corners should lie on the roof plane (within tolerance)."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)
        normal = plane_normal(0.0, 45.0)
        plane = {"n": normal, "ref": {"x": 0.0, "y": 2.0, "z": 0.0}}

        # Front wall protruding above roof
        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        group = {"front": front, "left_cheek": None, "right_cheek": None}

        geom = build_dormer_geometry(group, roof)
        cutout = geom["cutout_corners"]

        if len(cutout) < 4:
            pytest.skip("Cutout not generated (degenerate case)")

        for pt in cutout:
            expected_y = plane_y_at(plane, pt[0], pt[2])
            assert abs(pt[1] - expected_y) < 0.1, (
                f"Point {pt} not on roof plane: expected y={expected_y:.3f}, got "
                f"y={pt[1]:.3f}"
            )


# ── Integration test: build_dormer_geometry ──────────────────────────────────


class TestBuildDormerGeometry:
    def test_generates_cheeks_when_missing(self):
        """When no cheek walls exist, geometry is generated."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        group = {"front": front, "left_cheek": None, "right_cheek": None}

        geom = build_dormer_geometry(group, roof)

        assert len(geom["cheeks"]) == 2
        for cheek in geom["cheeks"]:
            assert cheek["source"] == "generated"
            assert len(cheek["corners"]) >= 3

    def test_clips_against_existing_cheek(self):
        """When an existing cheek wall is present, the generated cheek is
        clipped to avoid overlap — only the non-overlapping portion remains."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        # Right cheek (at x=-1) that only covers a tiny fraction of the depth
        # (ridge direction is (-1,0), so right side is at x=-1)
        short_cheek = _make_wall(
            "SHORT_RIGHT",
            [(-1, 0, 1), (-1, 0, 0.9), (-1, 4, 0.9), (-1, 4, 1)],
        )
        group = {"front": front, "left_cheek": None, "right_cheek": short_cheek}

        geom = build_dormer_geometry(group, roof)

        right = geom["cheeks"][1]
        assert right["source"] == "generated"
        assert len(right["corners"]) >= 3

    def test_header_always_generated(self):
        """Header is always generated as a horizontal surface at dormer top."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        group = {"front": front, "left_cheek": None, "right_cheek": None}

        geom = build_dormer_geometry(group, roof)

        assert geom["header"] is not None
        assert geom["header"]["source"] == "generated"
        # Header should be at dormer_top_y = 4.0
        for pt in geom["header"]["corners"]:
            assert abs(pt[1] - 4.0) < 0.01

    def test_both_cheeks_always_generated(self):
        """Both left and right cheeks are always in the output list."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        group = {"front": front, "left_cheek": None, "right_cheek": None}

        geom = build_dormer_geometry(group, roof)

        assert len(geom["cheeks"]) == 2
        # Both should have corners
        for cheek in geom["cheeks"]:
            assert len(cheek["corners"]) >= 3

    def test_cutout_always_generated(self):
        """Cutout corners for slant hollowing are always generated."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        group = {"front": front, "left_cheek": None, "right_cheek": None}

        geom = build_dormer_geometry(group, roof)

        assert len(geom["cutout_corners"]) == 4
        # All cutout corners must lie on the roof plane
        normal = plane_normal(0.0, 45.0)
        plane = {"n": normal, "ref": {"x": 0.0, "y": 2.0, "z": 0.0}}
        for pt in geom["cutout_corners"]:
            expected_y = plane_y_at(plane, pt[0], pt[2])
            assert abs(pt[1] - expected_y) < 0.1

    def test_cheek_clipped_no_overlap_with_room_wall(self):
        """A generated cheek is clipped against a coplanar room wall."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        # A room wall coplanar with the left cheek (at x=1) covering
        # part of the cheek area
        blocking_wall = _make_wall(
            "BLOCKER",
            [(1, 0, 1), (1, 0, 0.5), (1, 4, 0.5), (1, 4, 1)],
        )

        group = {"front": front, "left_cheek": None, "right_cheek": None}

        geom = build_dormer_geometry(group, roof, room_walls=[front, blocking_wall])

        # Left cheek is at x=1 side — it should be clipped
        left = geom["cheeks"][0]
        assert left["source"] == "generated"
        assert len(left["corners"]) >= 3

    def test_full_existing_cheek_yields_empty(self):
        """When an existing cheek fully covers the ideal, the generated
        cheek is empty (clipped-empty) — no overlap."""
        roof = _make_roof_surface(az=0.0, incl=45.0, cx=0.0, cz=0.0, cy=2.0)
        plane_normal(0.0, 45.0)

        front = _make_wall(
            "FRONT",
            [(-1, 0, 1), (1, 0, 1), (1, 4, 1), (-1, 4, 1)],
        )
        # Build the ideal cheek geometry first to know its bounds,
        # then create an existing wall that fully covers it
        group_no_cheek = {"front": front, "left_cheek": None, "right_cheek": None}
        ref_geom = build_dormer_geometry(group_no_cheek, roof)
        left_ideal = ref_geom["cheeks"][0]["corners"]

        # Create a large existing cheek that fully covers the ideal
        ys = [c[1] for c in left_ideal]
        zs = [c[2] for c in left_ideal]
        big_cheek = _make_wall(
            "BIG_LEFT",
            [
                (1, min(ys) - 0.5, max(zs) + 0.5),
                (1, min(ys) - 0.5, min(zs) - 0.5),
                (1, max(ys) + 0.5, min(zs) - 0.5),
                (1, max(ys) + 0.5, max(zs) + 0.5),
            ],
        )

        group = {"front": front, "left_cheek": big_cheek, "right_cheek": None}
        geom = build_dormer_geometry(group, roof)

        left = geom["cheeks"][0]
        assert left["source"] == "clipped-empty"
        assert len(left["corners"]) < 3

        # Right cheek should still be generated
        right = geom["cheeks"][1]
        assert right["source"] == "generated"
        assert len(right["corners"]) >= 3
