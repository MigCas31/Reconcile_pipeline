"""Tests for L-junction detection and ridge extension."""

from __future__ import annotations

import math

import pytest

from reconcile.roof_algorithms_py.math_utils import (
    plane_normal,
    ridge_line_intersection_2d,
)


def _make_plane(
    azi: float, incl: float, ref_x: float, ref_z: float, ref_y: float = 3.0
):
    """Build a minimal plane dict for testing."""
    n = plane_normal(azi, incl)
    azi_rad = azi * math.pi / 180.0
    ridge_x, ridge_z = -math.cos(azi_rad), math.sin(azi_rad)
    slope_x, slope_z = math.sin(azi_rad), math.cos(azi_rad)
    return {
        "cl": {"avgAzimuth": azi, "avgIncl": incl, "segs": []},
        "n": n,
        "ref": {"x": ref_x, "y": ref_y, "z": ref_z},
        "ridgeX": ridge_x,
        "ridgeZ": ridge_z,
        "slopeX": slope_x,
        "slopeZ": slope_z,
        "minRidge": -3.0,
        "maxRidge": 3.0,
        "minSlope": -1.0,
        "maxSlope": 1.0,
        "dominantStory": 1,
        "room_indices": set(),
    }


# --- ridge_line_intersection_2d ---


class TestRidgeLineIntersection2D:
    def test_perpendicular_planes(self):
        pi = _make_plane(90, 45, 3.0, 0.0)
        pj = _make_plane(0, 45, 0.0, 3.0)
        result = ridge_line_intersection_2d(pi, pj)
        assert result is not None
        # Both ridges should cross somewhere near (0, 0) or at the intersection
        assert isinstance(result["x"], float)
        assert isinstance(result["z"], float)

    def test_parallel_planes_returns_none(self):
        pi = _make_plane(90, 45, 0.0, 0.0)
        pj = _make_plane(90, 45, 5.0, 0.0)
        result = ridge_line_intersection_2d(pi, pj)
        assert result is None

    def test_opposing_planes_returns_none(self):
        pi = _make_plane(90, 45, 0.0, 0.0)
        pj = _make_plane(270, 45, 0.0, 0.0)
        # Same ridge direction → parallel → None
        result = ridge_line_intersection_2d(pi, pj)
        assert result is None


# --- detect_and_expand_l_junctions ---


class TestDetectAndExpandLJunctions:
    def _make_l_shaped_setup(self):
        """Create an L-shaped building with 4 planes (two opposing pairs)."""
        from reconcile.roof_algorithms_py.ceiling_clipping_l_junction import (
            detect_and_expand_l_junctions,
        )
        from reconcile.roof_algorithms_py.math_utils import (
            clip_poly_by_ridge,
            point_in_poly_2d,
        )

        # Main wing runs along z-axis, perp wing along x-axis
        planes = [
            _make_plane(90, 45, 3.0, 0.0),  # 0: east-facing (main wing)
            _make_plane(270, 45, 3.0, 0.0),  # 1: west-facing (main wing)
            _make_plane(180, 45, 0.0, -2.0),  # 2: south-facing (perp wing)
            _make_plane(0, 45, 0.0, -2.0),  # 3: north-facing (perp wing)
        ]

        # L-shaped footprint
        footprint = [
            (-4.0, -4.0),
            (6.0, -4.0),
            (6.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
            (-4.0, 0.0),
        ]

        # Initial clips: just ridge-clip the footprint
        plane_clipped = []
        for p in planes:
            clipped = list(footprint)
            clipped = clip_poly_by_ridge(
                clipped,
                p["ridgeX"],
                p["ridgeZ"],
                p["ref"]["x"],
                p["ref"]["z"],
                p["minRidge"],
                True,
            )
            clipped = clip_poly_by_ridge(
                clipped,
                p["ridgeX"],
                p["ridgeZ"],
                p["ref"]["x"],
                p["ref"]["z"],
                p["maxRidge"],
                False,
            )
            plane_clipped.append(
                {
                    "clipped": clipped,
                    "ridgeMin": p["minRidge"],
                    "ridgeMax": p["maxRidge"],
                }
            )

        return (
            planes,
            plane_clipped,
            footprint,
            detect_and_expand_l_junctions,
            point_in_poly_2d,
        )

    def test_l_junctions_detected(self):
        planes, plane_clipped, fp, detect_fn, pip_fn = self._make_l_shaped_setup()
        l_junctions = detect_fn(
            ceiling_planes=planes,
            plane_clipped=plane_clipped,
            building_footprint=fp,
            point_in_poly_2d=pip_fn,
        )
        # Should detect perpendicular pairs (0↔2, 0↔3, 1↔2, 1↔3)
        assert len(l_junctions) > 0
        pair_set = {(lj["plane_a"], lj["plane_b"]) for lj in l_junctions}
        # At least some perpendicular pairs detected
        assert any(a < 2 and b >= 2 for a, b in pair_set)

    def test_ridge_bounds_extended(self):
        planes, plane_clipped, fp, detect_fn, pip_fn = self._make_l_shaped_setup()
        orig_bounds = [(pc["ridgeMin"], pc["ridgeMax"]) for pc in plane_clipped]
        detect_fn(
            ceiling_planes=planes,
            plane_clipped=plane_clipped,
            building_footprint=fp,
            point_in_poly_2d=pip_fn,
        )
        # At least one plane should have extended ridge bounds
        any_extended = False
        for i, pc in enumerate(plane_clipped):
            if (
                pc["ridgeMin"] < orig_bounds[i][0] - 0.01
                or pc["ridgeMax"] > orig_bounds[i][1] + 0.01
            ):
                any_extended = True
                break
        assert any_extended

    def test_no_junction_for_straight_gable(self):
        """A simple gable building (no L) should produce no L-junctions."""
        from reconcile.roof_algorithms_py.ceiling_clipping_l_junction import (
            detect_and_expand_l_junctions,
        )
        from reconcile.roof_algorithms_py.math_utils import (
            clip_poly_by_ridge,
            point_in_poly_2d,
        )

        planes = [
            _make_plane(90, 45, 0.0, 0.0),  # east-facing
            _make_plane(270, 45, 0.0, 0.0),  # west-facing (opposing)
        ]
        footprint = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]

        plane_clipped = []
        for p in planes:
            clipped = list(footprint)
            clipped = clip_poly_by_ridge(
                clipped,
                p["ridgeX"],
                p["ridgeZ"],
                p["ref"]["x"],
                p["ref"]["z"],
                p["minRidge"],
                True,
            )
            clipped = clip_poly_by_ridge(
                clipped,
                p["ridgeX"],
                p["ridgeZ"],
                p["ref"]["x"],
                p["ref"]["z"],
                p["maxRidge"],
                False,
            )
            plane_clipped.append(
                {
                    "clipped": clipped,
                    "ridgeMin": p["minRidge"],
                    "ridgeMax": p["maxRidge"],
                }
            )

        l_junctions = detect_and_expand_l_junctions(
            ceiling_planes=planes,
            plane_clipped=plane_clipped,
            building_footprint=footprint,
            point_in_poly_2d=point_in_poly_2d,
        )
        assert len(l_junctions) == 0


# --- Integration test on real building ---


class TestIntegration16784bad:
    @pytest.fixture()
    def roof_result(self):
        from pathlib import Path

        from reconcile.extract3d import extract_building
        from reconcile.roof_algorithms_py import run_roof_algorithms

        result = extract_building(
            "16784bad-2cd9-4f4c-bb26-60355981cfe2",
            Path("pipeline-outputs"),
            Path(".scan-cache"),
        )
        if not result:
            pytest.skip("Building data not available")
        return result, run_roof_algorithms(result)

    def test_all_y_ranges_reasonable(self, roof_result):
        bldg, roof = roof_result
        oblique = roof["roof_surfaces"]["oblique"]
        structure_y_max = max(
            (
                float(corner[1])
                for room in (bldg.get("rooms") or [])
                for poly in [room.get("floor_polygon") or []]
                for corner in poly
                if isinstance(corner, (list, tuple)) and len(corner) >= 3
            ),
            default=0.0,
        )
        for room in bldg.get("rooms") or []:
            for wall in room.get("walls_computed") or []:
                for corner in wall.get("corners") or []:
                    if isinstance(corner, (list, tuple)) and len(corner) >= 3:
                        structure_y_max = max(structure_y_max, float(corner[1]))
                for strip in wall.get("extension_strip") or []:
                    for corner in strip:
                        if isinstance(corner, (list, tuple)) and len(corner) >= 3:
                            structure_y_max = max(structure_y_max, float(corner[1]))
        allowable_roof_peak = structure_y_max + 4.0
        for i, o in enumerate(oblique):
            ys = [p[1] for p in o["corners"]]
            assert min(ys) > -1.0, f"oblique[{i}] y_min={min(ys):.2f} is below ground"
            assert max(ys) < allowable_roof_peak, (
                f"oblique[{i}] y_max={max(ys):.2f} exceeds structure_y_max+4.0 "
                f"({allowable_roof_peak:.2f})"
            )

    def test_at_least_4_oblique_surfaces(self, roof_result):
        _, roof = roof_result
        oblique = roof["roof_surfaces"]["oblique"]
        assert len(oblique) >= 4
