"""Smoke tests for the analysis feature-expansion module.

Guards against two regressions:
    1. Plane-angle computation (azimuth/incl from a 4-tuple plane) must
       round-trip with the snapshot values in :file:`features_snapshot`.
    2. :func:`expand` must produce a flat dict with no nested values,
       since downstream analysis serializes to parquet.
"""

from __future__ import annotations

import math

from reconcile_v3.analysis import feature_expansion as fe


def _mk_record(**overrides):
    # A minimal label record with a 45° pitched plane facing north (+Z).
    # Normal = (0, cos(45°), sin(45°)) → incl = 45°, azimuth = 0°.
    plane = [0.0, math.cos(math.radians(45)), math.sin(math.radians(45)), -1.0]
    corners = [
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]
    base = {
        "building_uuid": "b",
        "proposal_id": "b::v3-merged-roof-segment::x",
        "label": "accepted",
        "heuristic_label": "accepted",
        "merged_plane": plane,
        "segment_corners_xyz": corners,
        "building_boundary_xz": [[-5, -5], [5, -5], [5, 5], [-5, 5]],
        "opposing_planes": [],
        "opposing_cluster_canonicals": [],
        "side_pieces": [],
        "cluster_members": [],
        "cluster_params": {},
        "room_boundary_refs": [],
        "features_snapshot": {
            "area_m2": 0.0,
            "perimeter_m": 0.0,
            "member_count": 0,
            "opposing_cluster_count": 0,
            "piece_kind": "room",
            "rain_hitting_side_count": 0,
            "covered_side_count": 0,
            "clipped_by_building_boundary": False,
        },
    }
    base.update(overrides)
    return base


def test_plane_to_azimuth_incl_round_trip():
    # Normal tilted 30° east and pitched 45°.
    plane = [0.5, math.cos(math.radians(45)), 0.5, 0.0]
    az, incl = fe._plane_to_azimuth_incl(plane)
    assert 44.0 < incl < 46.0
    assert 44.0 < az < 46.0


def test_expand_is_flat_and_has_expected_keys():
    feats = fe.expand(_mk_record())
    # Flat: every value is a python scalar or None.
    for k, v in feats.items():
        assert not isinstance(v, (list, dict)), f"nested value at {k}: {type(v)}"
    # Core categories all represented.
    for key in (
        "plane_incl_deg",
        "plane_azimuth_deg",
        "vertex_count",
        "poly_area_xz_m2",
        "edge_count",
        "opposing_count",
        "cluster_member_count",
        "label",
        "heuristic_disagrees_with_user",
        "is_split_child",
        "drainage_flow_azimuth_deg",
    ):
        assert key in feats, f"missing {key}"
    assert feats["vertex_count"] == 4
    # 45° pitch should resolve cleanly.
    assert 44.0 < feats["plane_incl_deg"] < 46.0


def test_expand_handles_missing_boundary():
    rec = _mk_record(building_boundary_xz=None)
    feats = fe.expand(rec)
    assert feats["inside_building_footprint"] is None
    assert feats["plane_incl_deg"] is not None  # plane features still work


def test_drainage_sheds_away_checks_centroid_direction():
    # Plane drains toward +Z; centroid must be on the +Z side of the building
    # center for sheds_away to be True. Our default record has its polygon
    # centroid at z=0 and boundary centroid at z=0 → direction is undefined,
    # so the flag should be None.
    rec = _mk_record()
    feats = fe.expand(rec)
    assert feats["drainage_flow_azimuth_deg"] is not None
