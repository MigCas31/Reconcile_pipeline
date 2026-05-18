"""Phase B.1 tests for reconcile_v3/reconstruction/solver.py.

We synthesize the same gable fixtures used in ``test_candidate_faces`` then
push the emitted candidates through :func:`solve_building` and assert on
the BIP behaviour: unique optimal selection on a gable, infeasibility when
coverage is starved, azimuth-bin capping, and topology rejection of
isolated faces.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict

from reconcile_v3.reconstruction.candidate_faces import (
    CandidateFace,
    build_candidate_faces,
)
from reconcile_v3.reconstruction.solver import (
    SolverConfig,
    solve_building,
    solve_building_with_zones,
)

_BLDG = "test-building"


def _plane_from_normal_and_point(
    nx: float, ny: float, nz: float, px: float, py: float, pz: float
) -> tuple[float, float, float, float]:
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    d = -(nx * px + ny * py + nz * pz)
    return nx, ny, nz, d


def _segment(
    seg_id: str,
    plane,
    footprint_xz_pairs,
    *,
    opposing_planes=None,
    opposing_canonicals=None,
    cluster_canonical_id: str = "cluster-A",
    area_m2: float | None = None,
) -> dict:
    fp = [[float(x), 0.0, float(z)] for (x, z) in footprint_xz_pairs]
    return {
        "id": seg_id,
        "cluster_canonical_id": cluster_canonical_id,
        "merged_plane": list(plane),
        "footprint_xz": fp,
        "opposing_planes": [list(p) for p in (opposing_planes or [])],
        "opposing_cluster_canonicals": list(opposing_canonicals or []),
        "features": {"area_m2": area_m2 if area_m2 is not None else 1.0},
    }


def _gable_candidates() -> tuple[list[dict], list[tuple[float, float]]]:
    plane_s = _plane_from_normal_and_point(0.0, 1.0, -1.0, 0.0, 2.0, 2.0)
    plane_n = _plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)
    south_fp = [(0.0, 0.0), (6.0, 0.0), (6.0, 1.5), (0.0, 1.5)]
    north_fp = [(0.0, 2.5), (6.0, 2.5), (6.0, 4.0), (0.0, 4.0)]
    segments = [
        _segment(
            f"{_BLDG}::v3-merged-roof-segment::south",
            plane_s,
            south_fp,
            opposing_planes=[plane_n],
            opposing_canonicals=["cluster-north"],
            cluster_canonical_id="cluster-south",
            area_m2=12.0,
        ),
        _segment(
            f"{_BLDG}::v3-merged-roof-segment::north",
            plane_n,
            north_fp,
            opposing_planes=[plane_s],
            opposing_canonicals=["cluster-south"],
            cluster_canonical_id="cluster-north",
            area_m2=12.0,
        ),
    ]
    footprint = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]
    faces = build_candidate_faces(_BLDG, segments, footprint)
    return [asdict(f) for f in faces], footprint


def test_gable_two_planes_unique_optimal() -> None:
    # On a clean gable both faces must be selected (each covers half the
    # footprint so only the union satisfies coverage), the solver should
    # report "solved", and the selection should auto-accept.
    cands, fp = _gable_candidates()
    assert len(cands) == 2, "fixture regression: gable should produce 2 candidates"

    t0 = time.perf_counter()
    res = solve_building(_BLDG, cands, fp)
    elapsed = time.perf_counter() - t0

    assert res.status == "solved", f"got {res.status}: {res.reason}"
    assert len(res.selected_face_ids) == 2
    assert set(res.selected_face_ids) == {c["id"] for c in cands}
    assert res.coverage_ratio >= 0.99
    assert res.decision == "auto_accept", f"expected auto_accept, got {res.decision}"
    assert elapsed < 2.0, f"gable solve took {elapsed:.2f}s"


def test_forced_infeasible_high_coverage_threshold() -> None:
    # Only one small sliver candidate; theta_cov=0.99 of a 24 m^2 footprint
    # demands far more coverage than the candidate can provide.
    plane = _plane_from_normal_and_point(0.0, 1.0, -1.0, 0.0, 2.0, 2.0)
    seg_fp = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]  # 1 m^2 sliver
    segments = [
        _segment(
            f"{_BLDG}::v3-merged-roof-segment::sliver", plane, seg_fp, area_m2=1.0
        ),
    ]
    footprint = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]  # 24 m^2
    cands = [asdict(f) for f in build_candidate_faces(_BLDG, segments, footprint)]
    assert cands, "fixture regression: sliver should produce at least one candidate"

    res = solve_building(
        _BLDG,
        cands,
        footprint,
        config=SolverConfig(theta_cov=0.99),
    )
    assert res.status == "infeasible", f"expected infeasible, got {res.status}"
    assert res.selected_face_ids == []


def test_no_candidates_returns_no_candidates_status() -> None:
    fp = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]
    res = solve_building(_BLDG, [], fp)
    assert res.status == "no_candidates"
    assert res.decision == "review"
    assert res.selected_face_ids == []


def test_azimuth_bin_constraint_limits_selection() -> None:
    # The gable has two candidates with opposed azimuths (~180 deg apart).
    # With azimuth_bin_width_deg=45 they fall in different bins; capping
    # k_azimuth_bins=1 forces the solver to pick AT MOST one -- and, with
    # theta_cov high enough that one face can't satisfy coverage, the
    # whole problem turns infeasible.
    cands, fp = _gable_candidates()

    res = solve_building(
        _BLDG,
        cands,
        fp,
        config=SolverConfig(k_azimuth_bins=1, theta_cov=0.85),
    )
    # Coverage constraint (0.85 x 24 = 20.4 m^2) cannot be met with only
    # one 12 m^2 half, so the model is infeasible under the single-bin cap.
    assert res.status == "infeasible", (
        f"expected infeasible, got {res.status}: {res.reason}"
    )


def test_azimuth_bin_cap_two_allows_gable() -> None:
    # Same fixture but with k_azimuth_bins=2 -- must solve cleanly.
    cands, fp = _gable_candidates()
    res = solve_building(
        _BLDG,
        cands,
        fp,
        config=SolverConfig(k_azimuth_bins=2),
    )
    assert res.status == "solved"
    assert len(res.selected_face_ids) == 2


def test_zone_local_solves_do_not_compete_for_global_azimuth_budget() -> None:
    west = {
        "id": "building-part:west",
        "part_id": "building-part:west",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
    }
    east = {
        "id": "building-part:east",
        "part_id": "building-part:east",
        "source": "roof-pipeline",
        "footprint_xz": [[4.0, 0.0], [8.0, 0.0], [8.0, 4.0], [4.0, 4.0]],
    }
    scan_fp = [(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)]
    candidates = [
        {
            "id": f"{_BLDG}::candidate::west",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::west",
            "plane": list(_plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)),
            "footprint_xz": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "area_m2": 16.0,
            "azimuth_deg": 0.0,
            "inclination_deg": 45.0,
            "neighbors": [],
            "support_m2": 16.0,
            "extended": False,
            "gbm_prior": None,
            "zone_id": west["id"],
            "part_id": west["part_id"],
            "zone_source": west["source"],
            "zone_room_ids": ["room:0"],
        },
        {
            "id": f"{_BLDG}::candidate::east",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::east",
            "plane": list(_plane_from_normal_and_point(1.0, 1.0, 0.0, 6.0, 2.0, 2.0)),
            "footprint_xz": [(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0)],
            "area_m2": 16.0,
            "azimuth_deg": 90.0,
            "inclination_deg": 45.0,
            "neighbors": [],
            "support_m2": 16.0,
            "extended": False,
            "gbm_prior": None,
            "zone_id": east["id"],
            "part_id": east["part_id"],
            "zone_source": east["source"],
            "zone_room_ids": ["room:1"],
        },
    ]
    cfg = SolverConfig(theta_cov=0.85, k_azimuth_bins=1, azimuth_bin_width_deg=45.0)

    global_res = solve_building(_BLDG, candidates, scan_fp, config=cfg)
    assert global_res.status == "infeasible"

    zoned_res = solve_building_with_zones(
        _BLDG,
        candidates,
        scan_fp,
        zones=[west, east],
        config=cfg,
    )
    assert zoned_res.status == "solved", zoned_res.reason
    assert zoned_res.decision == "auto_accept"
    assert set(zoned_res.selected_face_ids) == {c["id"] for c in candidates}
    assert math.isclose(zoned_res.coverage_ratio, 1.0, abs_tol=1e-6)
    assert len(zoned_res.zone_results) == 2
    assert {row["zone_id"] for row in zoned_res.zone_results} == {
        "building-part:west",
        "building-part:east",
    }
    assert all(row["status"] == "solved" for row in zoned_res.zone_results)


def test_low_confidence_zone_forces_review() -> None:
    zone = {
        "id": "building-part:west",
        "part_id": "building-part:west",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        "confidence": 0.2,
        "fallback_kind": None,
    }
    candidate = {
        "id": f"{_BLDG}::candidate::west",
        "building_uuid": _BLDG,
        "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::west",
        "plane": list(_plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)),
        "footprint_xz": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        "area_m2": 16.0,
        "azimuth_deg": 0.0,
        "inclination_deg": 45.0,
        "neighbors": [],
        "support_m2": 16.0,
        "extended": False,
        "gbm_prior": None,
        "zone_id": zone["id"],
        "part_id": zone["part_id"],
        "zone_source": zone["source"],
        "zone_room_ids": ["room:0"],
        "zone_confidence": zone["confidence"],
        "zone_fallback_kind": zone["fallback_kind"],
    }
    fp = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]

    res = solve_building_with_zones(
        _BLDG,
        [candidate],
        fp,
        zones=[zone],
    )
    assert res.status == "solved"
    assert res.decision == "review"
    assert "low_confidence_zones" in res.reason
    assert res.zone_confidence_summary == [
        {
            "zone_id": "building-part:west",
            "confidence": 0.2,
            "fallback_kind": None,
        }
    ]


def test_fallback_zone_with_many_selected_slices_is_ambiguous() -> None:
    zone = {
        "id": "building-part:fallback",
        "part_id": "building-part:fallback",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [9.0, 0.0], [9.0, 1.0], [0.0, 1.0]],
        "confidence": 0.8,
        "fallback_kind": "support_component_without_subpart",
    }
    candidates = []
    for idx in range(9):
        candidates.append(
            {
                "id": f"{_BLDG}::candidate::{idx}",
                "building_uuid": _BLDG,
                "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::{idx}",
                "plane": list(
                    _plane_from_normal_and_point(0.0, 1.0, 1.0, float(idx), 2.0, 0.0)
                ),
                "footprint_xz": [
                    (float(idx), 0.0),
                    (float(idx + 1), 0.0),
                    (float(idx + 1), 1.0),
                    (float(idx), 1.0),
                ],
                "area_m2": 1.0,
                "azimuth_deg": 0.0,
                "inclination_deg": 45.0,
                "neighbors": [],
                "support_m2": 1.0,
                "extended": False,
                "gbm_prior": None,
                "zone_id": zone["id"],
                "part_id": zone["part_id"],
                "zone_source": zone["source"],
                "zone_room_ids": ["room:0"],
                "zone_confidence": zone["confidence"],
                "zone_fallback_kind": zone["fallback_kind"],
            }
        )
    fp = [(0.0, 0.0), (9.0, 0.0), (9.0, 1.0), (0.0, 1.0)]

    res = solve_building_with_zones(
        _BLDG,
        candidates,
        fp,
        zones=[zone],
        config=SolverConfig(theta_cov=0.99, k_azimuth_bins=1),
    )
    assert res.status == "ambiguous"
    assert res.decision == "review"
    assert "fallback_zone_selected_many_slices" in res.reason
    assert res.zone_results[0]["status"] == "ambiguous"


def test_elongated_fallback_zone_prefers_perpendicular_axis_pair() -> None:
    zone = {
        "id": "building-part:fallback",
        "part_id": "building-part:fallback",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [6.0, 0.0], [6.0, 2.0], [0.0, 2.0]],
        "confidence": 0.8,
        "fallback_kind": "support_component_without_subpart",
        "seed_subpart_ids": [],
    }
    fp = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)]
    candidates = []
    for name, azimuth_deg in (
        ("perp-a", 0.0),
        ("perp-b", 180.0),
        ("major-a", 90.0),
        ("major-b", 270.0),
    ):
        candidates.append(
            {
                "id": f"{_BLDG}::candidate::{name}",
                "building_uuid": _BLDG,
                "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::{name}",
                "plane": list(
                    _plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 0.0)
                ),
                "footprint_xz": fp,
                "area_m2": 12.0,
                "azimuth_deg": azimuth_deg,
                "inclination_deg": 45.0,
                "neighbors": [],
                "support_m2": 12.0,
                "extended": False,
                "gbm_prior": None,
                "zone_id": zone["id"],
                "part_id": zone["part_id"],
                "zone_source": zone["source"],
                "zone_room_ids": ["room:0"],
                "zone_confidence": zone["confidence"],
                "zone_fallback_kind": zone["fallback_kind"],
            }
        )

    res = solve_building_with_zones(
        _BLDG,
        candidates,
        fp,
        zones=[zone],
        config=SolverConfig(theta_cov=0.85, theta_overlap=2.0),
    )
    assert res.status == "solved"
    assert set(res.selected_face_ids) == {
        f"{_BLDG}::candidate::perp-a",
        f"{_BLDG}::candidate::perp-b",
    }


def test_gap_axis_guides_fallback_zone_even_when_zone_is_not_elongated() -> None:
    zone = {
        "id": "building-part:fallback",
        "part_id": "building-part:fallback",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        "confidence": 0.8,
        "fallback_kind": "support_component_without_subpart",
        "seed_subpart_ids": [],
    }
    fp = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    candidates = []
    for name, azimuth_deg in (
        ("perp-a", 0.0),
        ("perp-b", 180.0),
        ("major-a", 90.0),
        ("major-b", 270.0),
    ):
        candidates.append(
            {
                "id": f"{_BLDG}::candidate::{name}",
                "building_uuid": _BLDG,
                "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::{name}",
                "plane": list(
                    _plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 0.0)
                ),
                "footprint_xz": fp,
                "area_m2": 16.0,
                "azimuth_deg": azimuth_deg,
                "inclination_deg": 45.0,
                "neighbors": [],
                "support_m2": 16.0,
                "extended": False,
                "gbm_prior": None,
                "zone_id": zone["id"],
                "part_id": zone["part_id"],
                "zone_source": zone["source"],
                "zone_room_ids": ["room:0"],
                "zone_confidence": zone["confidence"],
                "zone_fallback_kind": zone["fallback_kind"],
                "source_gap_major_axis_azimuth_deg": 90.0,
            }
        )

    res = solve_building_with_zones(
        _BLDG,
        candidates,
        fp,
        zones=[zone],
        config=SolverConfig(theta_cov=0.85, theta_overlap=2.0),
    )
    assert res.status == "solved"
    assert set(res.selected_face_ids) == {
        f"{_BLDG}::candidate::perp-a",
        f"{_BLDG}::candidate::perp-b",
    }


def test_tiny_unassigned_remainder_is_ignored() -> None:
    zone = {
        "id": "building-part:west",
        "part_id": "building-part:west",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [3.99, 0.0], [3.99, 4.0], [0.0, 4.0]],
        "confidence": 0.9,
        "fallback_kind": None,
    }
    scan_fp = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    candidates = [
        {
            "id": f"{_BLDG}::candidate::west",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::west",
            "plane": list(_plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)),
            "footprint_xz": [(0.0, 0.0), (3.99, 0.0), (3.99, 4.0), (0.0, 4.0)],
            "area_m2": 15.96,
            "azimuth_deg": 0.0,
            "inclination_deg": 45.0,
            "neighbors": [],
            "support_m2": 15.96,
            "extended": False,
            "gbm_prior": None,
            "zone_id": zone["id"],
            "part_id": zone["part_id"],
            "zone_source": zone["source"],
            "zone_room_ids": ["room:0"],
            "zone_confidence": zone["confidence"],
            "zone_fallback_kind": zone["fallback_kind"],
        },
        {
            "id": f"{_BLDG}::candidate::outside",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::outside",
            "plane": list(_plane_from_normal_and_point(1.0, 1.0, 0.0, 10.0, 2.0, 2.0)),
            "footprint_xz": [(10.0, 0.0), (11.0, 0.0), (11.0, 1.0), (10.0, 1.0)],
            "area_m2": 1.0,
            "azimuth_deg": 90.0,
            "inclination_deg": 45.0,
            "neighbors": [],
            "support_m2": 1.0,
            "extended": False,
            "gbm_prior": None,
        },
    ]

    res = solve_building_with_zones(
        _BLDG,
        candidates,
        scan_fp,
        zones=[zone],
        config=SolverConfig(theta_cov=0.99),
    )
    assert res.status == "solved", res.reason
    assert res.decision == "auto_accept"
    assert {row["zone_id"] for row in res.zone_results} == {zone["id"]}


def test_meaningful_unassigned_remainder_becomes_local_subproblem() -> None:
    zone = {
        "id": "building-part:west",
        "part_id": "building-part:west",
        "source": "roof-pipeline",
        "footprint_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        "confidence": 0.9,
        "fallback_kind": None,
    }
    scan_fp = [(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)]
    candidates = [
        {
            "id": f"{_BLDG}::candidate::west",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::west",
            "plane": list(_plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)),
            "footprint_xz": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "area_m2": 16.0,
            "azimuth_deg": 0.0,
            "inclination_deg": 45.0,
            "neighbors": [],
            "support_m2": 16.0,
            "extended": False,
            "gbm_prior": None,
            "zone_id": zone["id"],
            "part_id": zone["part_id"],
            "zone_source": zone["source"],
            "zone_room_ids": ["room:0"],
            "zone_confidence": zone["confidence"],
            "zone_fallback_kind": zone["fallback_kind"],
        },
        {
            "id": f"{_BLDG}::candidate::east",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::east",
            "plane": list(_plane_from_normal_and_point(1.0, 1.0, 0.0, 6.0, 2.0, 2.0)),
            "footprint_xz": [(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0)],
            "area_m2": 16.0,
            "azimuth_deg": 90.0,
            "inclination_deg": 45.0,
            "neighbors": [],
            "support_m2": 16.0,
            "extended": False,
            "gbm_prior": None,
        },
    ]

    res = solve_building_with_zones(
        _BLDG,
        candidates,
        scan_fp,
        zones=[zone],
        config=SolverConfig(theta_cov=0.99),
    )
    assert res.status == "solved", res.reason
    assert res.decision == "auto_accept"
    assert set(res.selected_face_ids) == {c["id"] for c in candidates}
    assert {row["zone_id"] for row in res.zone_results} == {
        zone["id"],
        "__unassigned__",
    }


def test_topology_constraint_requires_neighbour_when_available() -> None:
    # A face with in-set neighbours can only be picked if at least one of
    # its neighbours is also picked. Build a 3-face linear chain A-B-C
    # where B is the only one satisfying coverage; if the solver picks B
    # it MUST also pick a neighbour (A or C) to honour topology.
    plane_s = _plane_from_normal_and_point(0.0, 1.0, -1.0, 0.0, 2.0, 2.0)
    plane_n = _plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)
    a_id = f"{_BLDG}::candidate::a"
    b_id = f"{_BLDG}::candidate::b"
    c_id = f"{_BLDG}::candidate::c"
    cands = [
        {
            "id": a_id,
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::a",
            "plane": list(plane_s),
            "footprint_xz": [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            "area_m2": 4.0,
            "azimuth_deg": 180.0,
            "inclination_deg": 45.0,
            "neighbors": [b_id],
            "support_m2": 4.0,
            "extended": False,
            "gbm_prior": None,
        },
        {
            "id": b_id,
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::b",
            "plane": list(plane_n),
            "footprint_xz": [(0.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)],
            "area_m2": 4.0,
            "azimuth_deg": 0.0,
            "inclination_deg": 45.0,
            "neighbors": [a_id, c_id],
            "support_m2": 4.0,
            "extended": False,
            "gbm_prior": None,
        },
        {
            "id": c_id,
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::c",
            "plane": list(plane_s),
            "footprint_xz": [(0.0, 4.0), (2.0, 4.0), (2.0, 6.0), (0.0, 6.0)],
            "area_m2": 4.0,
            "azimuth_deg": 180.0,
            "inclination_deg": 45.0,
            "neighbors": [b_id],
            "support_m2": 4.0,
            "extended": False,
            "gbm_prior": None,
        },
    ]
    fp = [(0.0, 0.0), (2.0, 0.0), (2.0, 6.0), (0.0, 6.0)]
    res = solve_building(_BLDG, cands, fp, config=SolverConfig(k_azimuth_bins=2))
    assert res.status == "solved"
    picked = set(res.selected_face_ids)
    if b_id in picked:
        assert picked & {a_id, c_id}, (
            "B has in-set neighbours, so at least one neighbour must also be picked"
        )


def test_isolated_face_is_selectable() -> None:
    # A face with no in-set neighbours must remain selectable -- forcing
    # it to 0 made buildings whose Phase A slices lack cross-references
    # spuriously infeasible (32/34 of the 20260419 corpus 'infeasible'
    # bucket). A single positive-utility isolated face should be picked.
    lone = {
        "id": f"{_BLDG}::candidate::lone",
        "building_uuid": _BLDG,
        "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::lone",
        "plane": list(_plane_from_normal_and_point(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        "footprint_xz": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        "area_m2": 16.0,
        "azimuth_deg": 0.0,
        "inclination_deg": 10.0,
        "neighbors": [],
        "support_m2": 16.0,
        "extended": False,
        "gbm_prior": None,
    }
    fp = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    other = {
        **lone,
        "id": f"{_BLDG}::candidate::other",
        "footprint_xz": [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)],
        "area_m2": 1.0,
        "support_m2": 0.0,
        "neighbors": [],
    }
    res = solve_building(_BLDG, [lone, other], fp, config=SolverConfig())
    assert res.status == "solved"
    assert lone["id"] in res.selected_face_ids


def test_runner_up_captured_when_alternatives_exist() -> None:
    # Three candidates where two equivalent subsets satisfy coverage.
    # The runner-up objective should be positive and close to the primary
    # -- triggering the ambiguity flag.
    plane_s = _plane_from_normal_and_point(0.0, 1.0, -1.0, 0.0, 2.0, 2.0)
    plane_n = _plane_from_normal_and_point(0.0, 1.0, 1.0, 0.0, 2.0, 2.0)

    # Two near-identical south faces (call them "south_a" and "south_b")
    # both cover the same south half -- the BIP can pick either. Both have
    # identical fit. Plus a required north face.
    south_fp = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)]
    north_fp = [(0.0, 2.0), (6.0, 2.0), (6.0, 4.0), (0.0, 4.0)]
    north_id = f"{_BLDG}::candidate::north"
    cands = [
        {
            "id": f"{_BLDG}::candidate::south_a",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::south_a",
            "plane": list(plane_s),
            "footprint_xz": south_fp,
            "area_m2": 12.0,
            "azimuth_deg": 180.0,
            "inclination_deg": 45.0,
            "neighbors": [north_id],
            "support_m2": 12.0,
            "extended": False,
            "gbm_prior": None,
        },
        {
            "id": f"{_BLDG}::candidate::south_b",
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::south_b",
            "plane": list(plane_s),
            "footprint_xz": south_fp,
            "area_m2": 12.0,
            "azimuth_deg": 180.0,
            "inclination_deg": 45.0,
            "neighbors": [north_id],
            "support_m2": 12.0,
            "extended": False,
            "gbm_prior": None,
        },
        {
            "id": north_id,
            "building_uuid": _BLDG,
            "parent_segment_id": f"{_BLDG}::v3-merged-roof-segment::north",
            "plane": list(plane_n),
            "footprint_xz": north_fp,
            "area_m2": 12.0,
            "azimuth_deg": 0.0,
            "inclination_deg": 45.0,
            "neighbors": [
                f"{_BLDG}::candidate::south_a",
                f"{_BLDG}::candidate::south_b",
            ],
            "support_m2": 12.0,
            "extended": False,
            "gbm_prior": None,
        },
    ]
    fp = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]
    # Low theta_overlap so the duplicated souths conflict (same plane,
    # same footprint -> trivially > 10% overlap, azimuth diff 0 deg).
    res = solve_building(
        _BLDG,
        cands,
        fp,
        config=SolverConfig(theta_overlap=0.10, theta_az_deg=45.0),
    )
    assert res.status in ("solved", "ambiguous")
    assert res.runner_up_objective > 0.0
    # Runner-up should be close enough to primary to trigger ambiguity
    # -- both alternative selections (south_a+north vs south_b+north) have
    # identical objective, so runner_up == primary.
    assert math.isclose(res.objective_value, res.runner_up_objective, rel_tol=1e-3), (
        f"expected tied runner-up, got obj={res.objective_value} "
        f"runner={res.runner_up_objective}"
    )


def test_candidate_face_dataclass_round_trip_through_asdict() -> None:
    # Regression guard: `solve_building` takes dicts, but Phase A emits
    # `CandidateFace` dataclasses. `asdict` must produce the exact keys
    # the solver reads. If this test fails with a KeyError the dataclass
    # has drifted from the solver contract.
    face = CandidateFace(
        id="x::candidate::a",
        building_uuid="x",
        parent_segment_id="x::v3-merged-roof-segment::a",
        plane=(0.0, 1.0, 0.0, 0.0),
        footprint_xz=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        area_m2=1.0,
        azimuth_deg=0.0,
        inclination_deg=0.0,
        neighbors=[],
        support_m2=1.0,
        extended=False,
        gbm_prior=None,
    )
    d = asdict(face)
    for key in (
        "id",
        "footprint_xz",
        "area_m2",
        "azimuth_deg",
        "support_m2",
        "gbm_prior",
        "neighbors",
    ):
        assert key in d, f"CandidateFace drift: missing {key}"
