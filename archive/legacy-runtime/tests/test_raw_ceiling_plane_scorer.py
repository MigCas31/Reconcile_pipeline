from __future__ import annotations

import math
from dataclasses import replace

from shapely.geometry import LineString, Polygon

from scripts.prototype_raw_ceiling_plane_scorer import (
    EaveChainRecord,
    PlaneEaveChainSupportRecord,
    RawEdgeRecord,
    RawPlaneRecord,
    TargetPlaneRecord,
    TargetSplitPieceRecord,
    _candidate_polygon,
    _fit_plane_svd,
    _piece_records_from_polygon,
    _serialized_piece_holes,
    _source_room_keys_from_ridge_diagnostics,
    annotate_committed_supported_pieces_with_hypothesis_part_overlap,
    annotate_rows_with_target_face_runs,
    annotate_split_piece_rows_with_precedence,
    build_eave_chains,
    build_plane_extent_split_pieces,
    build_story_extent_envelopes,
    build_story_gap_polygons,
    classify_split_piece_final_layer,
    clip_unpaired_ridge_run_supported_pieces_to_support_union,
    collect_conflict_pairs,
    collect_raw_plane_records,
    collect_ridge_eave_target_anchor_masks,
    collect_ridge_eave_target_diagnostics,
    collect_selected_ridge_eave_plane_group_targets,
    collect_target_plane_records,
    diagnose_ridge_eave_supported_piece_ownership,
    expand_plane_eave_chain_supports_by_facade_continuity,
    merge_same_plane_committed_oblique_cores,
    merge_split_piece_rows_with_ownership,
    promote_raw_plane_support_records,
    prune_ridge_eave_rows_with_unreliable_mirror_pairs,
    score_plane_eave_chain_supports,
    score_target,
    select_locally_owned_ridge_eave_chain_supports,
    trim_ridge_eave_rows_to_local_mirror_pieces,
    trim_ridge_eave_supported_pieces_to_chain_run_bands,
    trim_ridge_eave_supported_pieces_to_mirror_partner,
    trim_ridge_eave_supported_pieces_to_room_ownership,
)


def _make_plane_corners() -> list[list[float]]:
    return [
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 3.0, 1.0],
        [0.0, 3.0, 1.0],
    ]


def _make_trusted_building(
    raw_corners: list[list[float]] | None = None,
) -> tuple[dict, dict]:
    corners = raw_corners or _make_plane_corners()
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                "walls_computed": [
                    {
                        "corners": [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 2.0, 0.0],
                            [0.0, 2.0, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [2.0, 0.0, 0.0],
                            [2.0, 0.0, 1.0],
                            [2.0, 3.0, 1.0],
                            [2.0, 2.0, 0.0],
                        ]
                    },
                    {
                        "corners": [
                            [2.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 3.0, 1.0],
                            [2.0, 3.0, 1.0],
                        ]
                    },
                    {
                        "corners": [
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 3.0, 1.0],
                        ]
                    },
                ],
                "raw_ceiling_planes": [{"corners": corners}],
            }
        ],
    }
    roof_result = {
        "ceiling": {
            "footprint": [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            "exposed_rooms": [{"story": 0, "room_index": 0}],
            "planes": [
                {
                    "dominantStory": 0,
                    "minRidge": -1.0,
                    "maxRidge": 1.0,
                    "minSlope": -0.5,
                    "maxSlope": 0.5,
                    "ridgeX": 1.0,
                    "ridgeZ": 0.0,
                    "slopeX": 0.0,
                    "slopeZ": 1.0,
                    "ref": {"x": 1.0, "y": 2.5, "z": 0.5},
                    "n": {"x": 0.0, "y": 1.0, "z": -1.0},
                    "cl": {"avgAzimuth": 180.0, "avgIncl": 45.0},
                }
            ],
        },
        "roof_surfaces": {"oblique": []},
    }
    return building, roof_result


def _make_target(
    poly_corners: list[list[float]] | None = None, *, story: int = 0
) -> TargetPlaneRecord:
    corners = poly_corners or _make_plane_corners()
    poly = _candidate_polygon(
        {
            "minRidge": -1.0,
            "maxRidge": 1.0,
            "minSlope": -0.5,
            "maxSlope": 0.5,
            "ridgeX": 1.0,
            "ridgeZ": 0.0,
            "slopeX": 0.0,
            "slopeZ": 1.0,
            "ref": {"x": 1.0, "y": 2.5, "z": 0.5},
        }
    )
    assert poly is not None
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    return TargetPlaneRecord(
        uuid="b-test",
        story=story,
        target_kind="candidate_oblique",
        target_index=0,
        element_id="b-test::ceiling-oblique::ceiling-oblique:0",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(1.0, 2.5, 0.5),
    )


def _make_raw_record(
    element_id: str,
    corners: list[list[float]],
    *,
    story: int = 0,
    room_trust_score: float = 1.0,
    usable_for_support: bool = True,
) -> RawPlaneRecord:
    from scripts.prototype_raw_ceiling_plane_scorer import (
        _normal_to_azimuth_inclination,
        _xz_polygon,
    )

    poly = _xz_polygon(corners)
    assert poly is not None
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    azimuth_deg, inclination_deg = _normal_to_azimuth_inclination(normal)
    return RawPlaneRecord(
        uuid="b-test",
        story=story,
        room_index=0,
        plane_index=0,
        element_id=element_id,
        corners=corners,
        poly_xz=poly,
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        normal=normal,
        azimuth_deg=azimuth_deg,
        inclination_deg=inclination_deg,
        area_xz_m2=float(poly.area),
        room_trust_score=room_trust_score,
        usable_for_support=usable_for_support,
    )


def test_exact_match_candidate_plane_has_high_coverage_and_normal_agreement() -> None:
    building, roof_result = _make_trusted_building()
    raw_records = collect_raw_plane_records(building, roof_result)
    targets = collect_target_plane_records(roof_result, "b-test")

    assert len(raw_records) == 1
    assert raw_records[0].usable_for_support is True
    result = score_target(targets[0], raw_records, [], [])

    assert result["raw_match_count"] == 1
    assert math.isclose(result["raw_xz_coverage"], 1.0, abs_tol=1e-6)
    assert math.isclose(result["raw_normal_dot_p50"], 1.0, abs_tol=1e-6)
    assert result["orientation_flag"] == "strong"


def test_low_trust_room_raw_planes_can_be_promoted_by_strong_local_target_match() -> (
    None
):
    good = [[c[0], c[1] + 0.2, c[2]] for c in _make_plane_corners()]
    noisy = [[c[0], c[1] + 5.0, c[2]] for c in _make_plane_corners()]
    building, roof_result = _make_trusted_building(raw_corners=good)
    building["rooms"][0]["raw_ceiling_planes"] = [{"corners": good}, {"corners": noisy}]
    raw_records = collect_raw_plane_records(building, roof_result)
    targets = collect_target_plane_records(roof_result, "b-test")
    promoted = promote_raw_plane_support_records(raw_records, targets)

    assert len(raw_records) == 2
    assert raw_records[0].usable_for_support is False
    assert promoted[0].usable_for_support is True

    result = score_target(targets[0], promoted, [], [])

    assert result["raw_match_count"] == 1
    assert result["raw_normal_dot_p50"] is not None
    assert result["orientation_flag"] in {"medium", "strong"}


def test_low_trust_room_raw_planes_stay_excluded_when_height_is_wrong() -> None:
    shifted = [[c[0], c[1] + 5.0, c[2]] for c in _make_plane_corners()]
    building, roof_result = _make_trusted_building(raw_corners=shifted)
    raw_records = collect_raw_plane_records(building, roof_result)
    targets = collect_target_plane_records(roof_result, "b-test")
    promoted = promote_raw_plane_support_records(raw_records, targets)

    assert len(raw_records) == 1
    assert raw_records[0].usable_for_support is False
    assert promoted[0].usable_for_support is False

    result = score_target(targets[0], promoted, [], [])

    assert result["raw_match_count"] == 0
    assert result["raw_normal_dot_p50"] is None
    assert result["retention_flag"] == "drop_candidate"


def test_ridge_edge_support_only_counts_aligned_edges() -> None:
    target = _make_target()
    aligned = RawEdgeRecord(
        story=0,
        plane_element_id="raw:1",
        label="ridge_or_hip",
        length_m=2.0,
        midpoint_xz=(1.0, 0.5),
        edge_azimuth_xz_deg=90.0,
    )
    misaligned = RawEdgeRecord(
        story=0,
        plane_element_id="raw:2",
        label="ridge_or_hip",
        length_m=5.0,
        midpoint_xz=(1.0, 0.5),
        edge_azimuth_xz_deg=0.0,
    )

    result = score_target(target, [], [aligned, misaligned], [])

    assert math.isclose(result["ridge_edge_support_len_m"], 2.0, abs_tol=1e-6)


def test_eave_edge_support_only_counts_edges_near_boundary() -> None:
    target = _make_target()
    near_boundary = RawEdgeRecord(
        story=0,
        plane_element_id="raw:1",
        label="eave",
        length_m=1.5,
        midpoint_xz=(0.2, 0.5),
        edge_azimuth_xz_deg=90.0,
    )
    far_inside = RawEdgeRecord(
        story=0,
        plane_element_id="raw:2",
        label="eave",
        length_m=4.0,
        midpoint_xz=(1.0, 2.0),
        edge_azimuth_xz_deg=90.0,
    )

    result = score_target(target, [], [near_boundary, far_inside], [])

    assert math.isclose(result["eave_edge_support_len_m"], 1.5, abs_tol=1e-6)


def test_conflicting_overlapping_raw_planes_trigger_split_flag() -> None:
    target = _make_target()
    left = _make_raw_record(
        "raw:left",
        [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 3.0, 1.0]],
    )
    right = _make_raw_record(
        "raw:right",
        [[1.0, 2.5, 0.0], [3.0, 2.5, 0.0], [3.0, 2.5, 1.0], [1.0, 2.5, 1.0]],
    )
    conflicts = collect_conflict_pairs([left, right])

    assert len(conflicts) == 1
    result = score_target(target, [left, right], [], conflicts)

    assert result["conflicting_raw_pair_count"] == 1
    assert result["split_flag"] is True


def test_candidate_plane_polygon_reconstruction_is_correct() -> None:
    poly = _candidate_polygon(
        {
            "minRidge": -1.0,
            "maxRidge": 1.0,
            "minSlope": -0.5,
            "maxSlope": 0.5,
            "ridgeX": 1.0,
            "ridgeZ": 0.0,
            "slopeX": 0.0,
            "slopeZ": 1.0,
            "ref": {"x": 1.0, "y": 2.5, "z": 0.5},
        }
    )

    assert poly is not None
    coords = list(poly.exterior.coords)[:-1]
    assert set(coords) == {(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)}
    assert math.isclose(poly.area, 2.0, abs_tol=1e-6)


def test_same_story_filtering_prevents_cross_story_support_leakage() -> None:
    target = _make_target(story=0)
    raw = _make_raw_record("raw:other-story", _make_plane_corners(), story=1)

    result = score_target(target, [raw], [], [])

    assert result["raw_match_count"] == 0
    assert result["raw_xz_coverage"] == 0.0


def test_missing_raw_support_yields_weak_and_drop_candidate() -> None:
    target = _make_target()

    result = score_target(target, [], [], [])

    assert result["raw_match_count"] == 0
    assert result["orientation_flag"] == "weak"
    assert result["retention_flag"] == "drop_candidate"


def test_build_eave_chains_merges_colinear_segments_on_same_story() -> None:
    edges = [
        RawEdgeRecord(
            story=0,
            plane_element_id="raw:1",
            label="eave",
            length_m=2.0,
            midpoint_xz=(1.0, 0.0),
            edge_azimuth_xz_deg=90.0,
            start_xz=(0.0, 0.0),
            end_xz=(2.0, 0.0),
            y_mid=2.0,
        ),
        RawEdgeRecord(
            story=0,
            plane_element_id="raw:2",
            label="eave",
            length_m=2.0,
            midpoint_xz=(3.0, 0.05),
            edge_azimuth_xz_deg=90.0,
            start_xz=(2.1, 0.05),
            end_xz=(4.1, 0.05),
            y_mid=2.05,
        ),
        RawEdgeRecord(
            story=0,
            plane_element_id="raw:3",
            label="eave",
            length_m=2.0,
            midpoint_xz=(1.0, 2.0),
            edge_azimuth_xz_deg=90.0,
            start_xz=(0.0, 2.0),
            end_xz=(2.0, 2.0),
            y_mid=2.0,
        ),
    ]

    chains = build_eave_chains("b-test", edges)

    assert len(chains) == 2
    merged = max(chains, key=lambda chain: chain.total_length_m)
    assert math.isclose(merged.total_length_m, 4.0, abs_tol=1e-6)
    assert merged.edge_count == 2


def test_plane_eave_chain_support_prefers_facade_aligned_chain() -> None:
    target = _make_target()
    good_chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="good",
        edge_count=2,
        total_length_m=4.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(2.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (2.0, 0.0)]),
        member_plane_ids=("raw:1", "raw:2"),
    )
    bad_chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="bad",
        edge_count=1,
        total_length_m=2.0,
        azimuth_deg=0.0,
        y_mean=2.0,
        start_xz=(1.0, 0.5),
        end_xz=(1.0, 1.5),
        line_xz=LineString([(1.0, 0.5), (1.0, 1.5)]),
        member_plane_ids=("raw:3",),
    )

    supports = score_plane_eave_chain_supports([target], [good_chain, bad_chain])
    by_id = {support.chain_id: support for support in supports}

    assert by_id["good"].supported is True
    assert by_id["good"].support_score > by_id["bad"].support_score


def test_build_plane_extent_split_pieces_closes_gap_between_supported_stripes() -> None:
    poly = _candidate_polygon(
        {
            "minRidge": -5.0,
            "maxRidge": 5.0,
            "minSlope": -1.0,
            "maxSlope": 1.0,
            "ridgeX": 1.0,
            "ridgeZ": 0.0,
            "slopeX": 0.0,
            "slopeZ": 1.0,
            "ref": {"x": 5.0, "y": 3.0, "z": 1.0},
        }
    )
    assert poly is not None
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="candidate_oblique",
        target_index=0,
        element_id="b-test::ceiling-oblique::ceiling-oblique:0",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    left_chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="left",
        edge_count=1,
        total_length_m=3.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(3.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (3.0, 0.0)]),
        member_plane_ids=("raw:left",),
    )
    right_chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="right",
        edge_count=1,
        total_length_m=3.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(7.0, 0.0),
        end_xz=(10.0, 0.0),
        line_xz=LineString([(7.0, 0.0), (10.0, 0.0)]),
        member_plane_ids=("raw:right",),
    )
    supports = [
        score_plane_eave_chain_supports([target], [left_chain, right_chain])[0],
        score_plane_eave_chain_supports([target], [left_chain, right_chain])[1],
    ]

    pieces = build_plane_extent_split_pieces(
        [target], [left_chain, right_chain], supports
    )

    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]
    residual_pieces = [piece for piece in pieces if piece.piece_role == "residual"]

    assert len(supported_pieces) == 1
    assert len(residual_pieces) == 0
    assert math.isclose(supported_pieces[0].area_xz_m2, 20.0, abs_tol=1e-6)


def _make_committed_oblique_target(
    poly_xz: Polygon,
    corners: list[list[float]],
    *,
    element_id: str = "b-test::roof-oblique::oblique:0",
) -> TargetPlaneRecord:
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    return TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="committed_oblique",
        target_index=0,
        element_id=element_id,
        poly_xz=poly_xz,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly_xz.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )


def _make_eave_chain(
    chain_id: str,
    start_xz: tuple[float, float],
    end_xz: tuple[float, float],
    *,
    y_mean: float,
) -> EaveChainRecord:
    return EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id=chain_id,
        edge_count=1,
        total_length_m=float(LineString([start_xz, end_xz]).length),
        azimuth_deg=90.0,
        y_mean=y_mean,
        start_xz=start_xz,
        end_xz=end_xz,
        line_xz=LineString([start_xz, end_xz]),
        member_plane_ids=(f"raw:{chain_id}",),
    )


def test_build_plane_extent_split_pieces_extends_to_part_eave_envelope() -> None:
    # Small target (2 x 1 XZ) whose plane slopes so +z is down-slope.
    # Its poly_xz barely reaches the eave.
    target_poly = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
    plane_corners = [
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
    target = _make_committed_oblique_target(target_poly, plane_corners)
    # Eave envelope extends far down-slope (z up to 3) and a bit on the ridge
    # side (z down to -1). Widening must add only the down-slope portion.
    envelope = Polygon([(0.0, -1.0), (2.0, -1.0), (2.0, 3.0), (0.0, 3.0)])
    eave_chain = _make_eave_chain("c1", (0.0, 1.0), (2.0, 1.0), y_mean=2.0)
    supports = score_plane_eave_chain_supports([target], [eave_chain])

    baseline = build_plane_extent_split_pieces([target], [eave_chain], supports)
    baseline_supported = [p for p in baseline if p.piece_role == "supported"]
    assert baseline_supported, "fixture must produce a supported piece"
    baseline_area = sum(p.area_xz_m2 for p in baseline_supported)

    provenance: dict[str, dict] = {}
    extended = build_plane_extent_split_pieces(
        [target],
        [eave_chain],
        supports,
        part_eave_envelopes={"part-A": envelope},
        target_allowed_part_ids={target.element_id: {"part-A"}},
        extension_provenance_out=provenance,
    )
    extended_supported = [p for p in extended if p.piece_role == "supported"]
    extended_area = sum(p.area_xz_m2 for p in extended_supported)

    assert extended_area > baseline_area + 0.1
    # Up-slope bound unchanged: no extended supported vertex above z = 0
    # (the ridge edge of the original target). Tolerance for the 0.01 anchor
    # offset used inside _downslope_halfplane_polygon.
    for piece in extended_supported:
        for x, _y, z in piece.corners:
            assert z >= -0.05, f"piece vertex leaked up-slope: ({x}, {z})"
    # Provenance recorded for the supported piece
    pid = extended_supported[0].piece_id
    assert pid in provenance
    assert provenance[pid]["extended_to_eave"] is True
    assert provenance[pid]["extended_by_m2"] > 0
    assert provenance[pid]["eave_envelope_parts"] == ["part-A"]


def test_build_plane_extent_split_pieces_extension_scoped_to_allowed_parts() -> None:
    target_poly = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
    plane_corners = [
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
    target = _make_committed_oblique_target(target_poly, plane_corners)
    # Two envelopes: A sits downslope of target; B is offset far away (simulating
    # another zone). If 'B' is not allowed, it must not contribute area.
    envelope_a = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)])
    envelope_b = Polygon([(10.0, 0.0), (12.0, 0.0), (12.0, 3.0), (10.0, 3.0)])
    eave_chain = _make_eave_chain("c1", (0.0, 1.0), (2.0, 1.0), y_mean=2.0)
    supports = score_plane_eave_chain_supports([target], [eave_chain])

    provenance: dict[str, dict] = {}
    pieces = build_plane_extent_split_pieces(
        [target],
        [eave_chain],
        supports,
        part_eave_envelopes={"part-A": envelope_a, "part-B": envelope_b},
        target_allowed_part_ids={target.element_id: {"part-A"}},
        extension_provenance_out=provenance,
    )
    supported = [p for p in pieces if p.piece_role == "supported"]
    assert supported
    # Extended piece should not contain vertices in B's x-range [10, 12].
    for piece in supported:
        for x, _y, _z in piece.corners:
            assert x <= 5.0, f"piece leaked into disallowed part: x={x}"
    pid = supported[0].piece_id
    assert provenance[pid]["eave_envelope_parts"] == ["part-A"]


def test_build_plane_extent_split_pieces_skips_extension_without_allowed_parts() -> (
    None
):
    target_poly = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
    plane_corners = [
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
    target = _make_committed_oblique_target(target_poly, plane_corners)
    envelope = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)])
    eave_chain = _make_eave_chain("c1", (0.0, 1.0), (2.0, 1.0), y_mean=2.0)
    supports = score_plane_eave_chain_supports([target], [eave_chain])

    provenance: dict[str, dict] = {}
    pieces = build_plane_extent_split_pieces(
        [target],
        [eave_chain],
        supports,
        part_eave_envelopes={"part-A": envelope},
        target_allowed_part_ids={},  # no allowed parts for this target
        extension_provenance_out=provenance,
    )
    supported = [p for p in pieces if p.piece_role == "supported"]
    assert supported
    # No provenance emitted when no parts were allowed (= no widening happened)
    for piece in supported:
        assert piece.piece_id not in provenance
    # Baseline area only (target.poly_xz.area = 2)
    assert sum(p.area_xz_m2 for p in supported) <= 2.0 + 0.01


def test_build_plane_extent_split_pieces_falls_back_to_story_envelope() -> None:
    target_poly = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
    plane_corners = [
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
    target = _make_committed_oblique_target(target_poly, plane_corners)
    story_env = Polygon([(0.0, -1.0), (2.0, -1.0), (2.0, 3.0), (0.0, 3.0)])
    eave_chain = _make_eave_chain("c1", (0.0, 1.0), (2.0, 1.0), y_mean=2.0)
    supports = score_plane_eave_chain_supports([target], [eave_chain])

    provenance: dict[str, dict] = {}
    pieces = build_plane_extent_split_pieces(
        [target],
        [eave_chain],
        supports,
        # No part_eave_envelopes / allowed_parts — simulates a building
        # with no resolved part graph. Widening must still fire via the
        # story-level fallback.
        story_eave_envelopes={0: story_env},
        extension_provenance_out=provenance,
    )
    supported = [p for p in pieces if p.piece_role == "supported"]
    assert supported
    pid = supported[0].piece_id
    assert pid in provenance
    assert provenance[pid]["extended_to_eave"] is True
    assert provenance[pid]["extended_by_m2"] > 0
    assert provenance[pid]["eave_envelope_source"] == "story_slabs+neighbouring_gaps"
    assert provenance[pid]["eave_envelope_parts"] == []
    # Up-slope bound preserved by the down-slope half-plane.
    for piece in supported:
        for x, _y, z in piece.corners:
            assert z >= -0.05, f"piece vertex leaked up-slope: ({x}, {z})"


def test_build_plane_extent_split_pieces_skips_extension_for_candidate_oblique() -> (
    None
):
    # Candidate obliques should NEVER widen via either per-part or
    # story-level envelope — they exist as raw evidence and the
    # partitioner uses them to carve area away from the committed face,
    # not to claim new area.
    target_poly = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
    plane_corners = [
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
    centroid, normal = _fit_plane_svd(plane_corners)
    assert centroid is not None and normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="candidate_oblique",
        target_index=0,
        element_id="b-test::ceiling-oblique::ceiling-oblique:0",
        poly_xz=target_poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(target_poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    story_env = Polygon([(0.0, -1.0), (2.0, -1.0), (2.0, 3.0), (0.0, 3.0)])
    eave_chain = _make_eave_chain("c1", (0.0, 1.0), (2.0, 1.0), y_mean=2.0)
    supports = score_plane_eave_chain_supports([target], [eave_chain])

    provenance: dict[str, dict] = {}
    pieces = build_plane_extent_split_pieces(
        [target],
        [eave_chain],
        supports,
        part_eave_envelopes={"part-A": story_env},
        target_allowed_part_ids={target.element_id: {"part-A"}},
        story_eave_envelopes={0: story_env},
        extension_provenance_out=provenance,
    )
    supported = [p for p in pieces if p.piece_role == "supported"]
    assert supported
    for piece in supported:
        assert piece.piece_id not in provenance
    # Baseline area only — no widening.
    assert sum(p.area_xz_m2 for p in supported) <= 2.0 + 0.01


def _make_ridge_eave_plane_group_target(
    poly_xz: Polygon,
    plane_corners: list[list[float]],
    *,
    element_id: str = "b-test::ridge-eave-candidate::plane-group::abc",
) -> TargetPlaneRecord:
    centroid, normal = _fit_plane_svd(plane_corners)
    assert centroid is not None and normal is not None
    return TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id=element_id,
        poly_xz=poly_xz,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly_xz.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )


def test_build_plane_extent_split_pieces_widens_ridge_eave_plane_group() -> None:
    # Ridge/eave plane group: poly_xz is short of the eave (z up to 0.5
    # only). Widening must extend it to reach the eave at z=1.0 via the
    # story envelope.
    target_poly = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 0.5)])
    plane_corners = [
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, 2.5, 0.5],
        [0.0, 2.5, 0.5],
    ]
    target = _make_ridge_eave_plane_group_target(target_poly, plane_corners)
    story_env = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.5), (0.0, 1.5)])
    eave_chain = _make_eave_chain("c1", (0.0, 0.5), (2.0, 0.5), y_mean=2.5)
    supports = score_plane_eave_chain_supports([target], [eave_chain])

    provenance: dict[str, dict] = {}
    pieces = build_plane_extent_split_pieces(
        [target],
        [eave_chain],
        supports,
        story_eave_envelopes={0: story_env},
        extension_provenance_out=provenance,
    )
    supported = [p for p in pieces if p.piece_role == "supported"]
    assert supported
    # The widened ridge/eave group reaches into the story envelope.
    pid = supported[0].piece_id
    assert pid in provenance
    assert provenance[pid]["extended_to_eave"] is True
    assert provenance[pid]["extended_by_m2"] > 0
    # Ridge/eave widening uses no down-slope half-plane — it spans both
    # slopes around the ridge.
    assert provenance[pid]["directional_trim"] is False


def test_build_extent_split_merges_small_gap():
    target = _make_target()
    chain_a = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="a",
        edge_count=1,
        total_length_m=0.8,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(0.8, 0.0),
        line_xz=LineString([(0.0, 0.0), (0.8, 0.0)]),
        member_plane_ids=("raw:a",),
    )
    chain_b = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="b",
        edge_count=1,
        total_length_m=0.7,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(1.0, 0.0),
        end_xz=(1.7, 0.0),
        line_xz=LineString([(1.0, 0.0), (1.7, 0.0)]),
        member_plane_ids=("raw:b",),
    )
    supports = score_plane_eave_chain_supports([target], [chain_a, chain_b])

    pieces = build_plane_extent_split_pieces([target], [chain_a, chain_b], supports)
    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]

    assert len(supported_pieces) == 1
    assert supported_pieces[0].chain_ids == ("a", "b")


def test_build_extent_keeps_disconnected_cross_slope() -> None:
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 4.0],
        [0.0, 4.0, 4.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::x",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    lower_chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="lower",
        edge_count=1,
        total_length_m=10.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(10.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (10.0, 0.0)]),
        member_plane_ids=("raw:lower",),
    )
    upper_chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="upper",
        edge_count=1,
        total_length_m=10.0,
        azimuth_deg=90.0,
        y_mean=4.0,
        start_xz=(0.0, 4.0),
        end_xz=(10.0, 4.0),
        line_xz=LineString([(0.0, 4.0), (10.0, 4.0)]),
        member_plane_ids=("raw:upper",),
    )
    supports = score_plane_eave_chain_supports([target], [lower_chain, upper_chain])

    pieces = build_plane_extent_split_pieces(
        [target], [lower_chain, upper_chain], supports
    )
    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]

    assert len(supported_pieces) == 2
    assert sorted(piece.chain_ids for piece in supported_pieces) == [
        ("lower",),
        ("upper",),
    ]


def test_select_chain_supports_same_facing_family() -> None:
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    family_a = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::family-a",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=136.0,
        inclination_deg=43.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    family_b = replace(
        family_a,
        element_id="b-test::ridge-eave-candidate::plane-group::family-b",
    )
    opposite = replace(
        family_a,
        element_id="b-test::ridge-eave-candidate::plane-group::opposite",
        azimuth_deg=316.0,
        inclination_deg=39.0,
    )
    base_kwargs = {
        "uuid": "b-test",
        "story": 0,
        "target_kind": "ridge_eave_plane_group",
        "chain_id": "b-test::eave-chain::0:0",
        "chain_azimuth_deg": 90.0,
        "ridge_azimuth_deg": 90.0,
        "angle_delta_deg": 0.0,
        "boundary_distance_m": 0.0,
        "supported": True,
        "chain_length_m": 4.0,
    }
    supports = [
        PlaneEaveChainSupportRecord(
            **base_kwargs,
            target_element_id=family_a.element_id,
            overlap_fraction=0.7,
            height_residual_m=0.8,
            support_score=0.8,
        ),
        PlaneEaveChainSupportRecord(
            **base_kwargs,
            target_element_id=family_b.element_id,
            overlap_fraction=0.6,
            height_residual_m=0.1,
            support_score=0.7,
        ),
        PlaneEaveChainSupportRecord(
            **base_kwargs,
            target_element_id=opposite.element_id,
            overlap_fraction=1.0,
            height_residual_m=0.2,
            support_score=0.9,
        ),
    ]

    filtered = select_locally_owned_ridge_eave_chain_supports(
        [family_a, family_b, opposite],
        supports,
    )
    by_target = {support.target_element_id: support.supported for support in filtered}

    assert by_target[family_a.element_id] is False
    assert by_target[family_b.element_id] is True
    assert by_target[opposite.element_id] is True


def test_select_chain_breaks_tie_by_overlap() -> None:
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target_a = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::one",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=136.0,
        inclination_deg=43.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    target_b = replace(
        target_a,
        element_id="b-test::ridge-eave-candidate::plane-group::two",
    )
    base_kwargs = {
        "uuid": "b-test",
        "story": 0,
        "target_kind": "ridge_eave_plane_group",
        "chain_id": "b-test::eave-chain::0:0",
        "chain_azimuth_deg": 90.0,
        "ridge_azimuth_deg": 90.0,
        "angle_delta_deg": 0.0,
        "boundary_distance_m": 0.0,
        "height_residual_m": 0.05,
        "supported": True,
        "chain_length_m": 4.0,
    }
    supports = [
        PlaneEaveChainSupportRecord(
            **base_kwargs,
            target_element_id=target_a.element_id,
            overlap_fraction=0.25,
            support_score=0.7,
        ),
        PlaneEaveChainSupportRecord(
            **base_kwargs,
            target_element_id=target_b.element_id,
            overlap_fraction=1.0,
            support_score=0.7,
        ),
    ]

    filtered = select_locally_owned_ridge_eave_chain_supports(
        [target_a, target_b],
        supports,
    )
    by_target = {support.target_element_id: support.supported for support in filtered}

    assert by_target[target_a.element_id] is False
    assert by_target[target_b.element_id] is True


def test_build_plane_extent_split_pieces_absorb_cross_floor_gap_residual() -> None:
    poly = _candidate_polygon(
        {
            "minRidge": -5.0,
            "maxRidge": 5.0,
            "minSlope": -1.0,
            "maxSlope": 1.0,
            "ridgeX": 1.0,
            "ridgeZ": 0.0,
            "slopeX": 0.0,
            "slopeZ": 1.0,
            "ref": {"x": 5.0, "y": 3.0, "z": 1.0},
        }
    )
    assert poly is not None
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=1,
        target_kind="candidate_oblique",
        target_index=0,
        element_id="b-test::ceiling-oblique::ceiling-oblique:0",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    left_chain = EaveChainRecord(
        uuid="b-test",
        story=1,
        chain_id="left",
        edge_count=1,
        total_length_m=3.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(3.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (3.0, 0.0)]),
        member_plane_ids=("raw:left",),
    )
    supports = score_plane_eave_chain_supports([target], [left_chain])

    pieces = build_plane_extent_split_pieces(
        [target],
        [left_chain],
        supports,
        story_gap_polygons={
            1: [Polygon([(3.0, 0.0), (7.0, 0.0), (7.0, 2.0), (3.0, 2.0)])],
        },
    )

    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]
    residual_pieces = [piece for piece in pieces if piece.piece_role == "residual"]

    assert len(supported_pieces) == 1
    assert len(residual_pieces) == 1
    assert math.isclose(supported_pieces[0].area_xz_m2, 14.0, abs_tol=1e-6)
    assert math.isclose(residual_pieces[0].area_xz_m2, 6.0, abs_tol=1e-6)


def test_story_extent_envelope_includes_room_slabs_and_gaps() -> None:
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [3.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                ],
            },
        ],
        "cross_floor_gaps": [
            {
                "story": 0,
                "corners": [
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                ],
            }
        ],
    }

    envelopes = build_story_extent_envelopes(building)

    assert 0 in envelopes
    assert math.isclose(float(envelopes[0].area), 5.0, abs_tol=1e-6)


def test_story_extent_envelope_promotes_gap_ceiling_to_story_above() -> None:
    building = {
        "rooms": [
            {
                "story": 1,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        ],
        "gap_walls": [
            {
                "story": 0,
                "type": "gap_ceiling",
                "corners": [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ],
            }
        ],
    }

    envelopes = build_story_extent_envelopes(building)

    assert 1 in envelopes
    assert math.isclose(float(envelopes[1].area), 2.0, abs_tol=1e-6)


def test_story_gap_polygons_promote_gap_ceiling_to_story_above() -> None:
    building = {
        "cross_floor_gaps": [
            {
                "story": 0,
                "type": "within_story",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        ],
        "gap_walls": [
            {
                "story": 0,
                "type": "gap_ceiling",
                "corners": [
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                ],
            }
        ],
    }

    by_story = build_story_gap_polygons(building)

    assert 0 in by_story
    assert 1 in by_story
    assert any(
        math.isclose(float(poly.area), 1.0, abs_tol=1e-6) for poly in by_story[1]
    )


def test_collect_selected_ridge_eave_plane_group_targets_prefers_top_story_on_tie() -> (
    None
):
    story_extent = build_story_extent_envelopes(
        {
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [
                        [0.0, 0.0, 0.0],
                        [4.0, 0.0, 0.0],
                        [4.0, 0.0, 2.0],
                        [0.0, 0.0, 2.0],
                    ],
                },
                {
                    "story": 1,
                    "floor_polygon": [
                        [0.0, 1.0, 0.0],
                        [4.0, 1.0, 0.0],
                        [4.0, 1.0, 2.0],
                        [0.0, 1.0, 2.0],
                    ],
                },
            ],
        }
    )
    ridge_entry = {
        "plane_groups": [
            {
                "id": "b-test::plane-group::abcd1234",
                "selected": True,
                "plane": [0.0, 1.0, -1.0, 0.0],
                "union_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
            },
            {
                "id": "b-test::plane-group::skip",
                "selected": False,
                "plane": [0.0, 1.0, -1.0, 0.0],
                "union_xz": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            },
        ],
    }

    targets = collect_selected_ridge_eave_plane_group_targets(
        "b-test", ridge_entry, story_extent
    )

    assert len(targets) == 1
    assert targets[0].story == 1
    assert targets[0].target_kind == "ridge_eave_plane_group"
    assert (
        targets[0].element_id == "b-test::ridge-eave-candidate::plane-group::abcd1234"
    )
    assert math.isclose(targets[0].area_xz_m2, 8.0, abs_tol=1e-6)


def test_collect_ridge_eave_target_diagnostics_flags_interior_slice_suspect() -> None:
    ridge_entry = {
        "candidates": [
            {
                "id": "b-test::candidate::one",
                "parent_segment_id": "b-test::segment::covered-big",
                "area_m2": 9.0,
                "extended": True,
            },
            {
                "id": "b-test::candidate::two",
                "parent_segment_id": "b-test::segment::covered-big",
                "area_m2": 9.0,
                "extended": True,
            },
            {
                "id": "b-test::candidate::three",
                "parent_segment_id": "b-test::segment::rain-small",
                "area_m2": 1.0,
                "extended": False,
            },
        ],
        "plane_groups": [
            {
                "id": "b-test::plane-group::abcd1234",
                "selected": True,
                "member_ids": [
                    "b-test::candidate::one",
                    "b-test::candidate::two",
                    "b-test::candidate::three",
                ],
                "best_partner_plane_group_id": None,
            }
        ],
    }
    v3_building = {
        "merged_roof_segments": [
            {
                "id": "b-test::segment::covered-big",
                "features": {"rain_hitting_side_count": 0, "covered_side_count": 2},
                "member_snapshots": [
                    {
                        "source_room_id": "room:4",
                        "slab_room_id": "room:4",
                        "features": {
                            "is_top_story_slab": True,
                            "slab_kind": "room",
                            "plane_height_above_slab_m": -0.4,
                        },
                    },
                    {
                        "source_room_id": "room:4",
                        "slab_room_id": "room:5",
                        "features": {
                            "is_top_story_slab": True,
                            "slab_kind": "room",
                            "plane_height_above_slab_m": -0.3,
                        },
                    },
                ],
            },
            {
                "id": "b-test::segment::rain-small",
                "features": {"rain_hitting_side_count": 1, "covered_side_count": 1},
                "member_snapshots": [
                    {
                        "source_room_id": "room:4",
                        "slab_room_id": "room:6",
                        "features": {
                            "is_top_story_slab": True,
                            "slab_kind": "room",
                            "plane_height_above_slab_m": 0.6,
                        },
                    }
                ],
            },
        ]
    }

    diagnostics = collect_ridge_eave_target_diagnostics(
        "b-test", ridge_entry, v3_building
    )
    diag = diagnostics["b-test::ridge-eave-candidate::plane-group::abcd1234"]

    assert math.isclose(diag["creator_rain_area_fraction"], 0.052632, abs_tol=1e-6)
    assert diag["creator_covered_segment_count"] == 2
    assert diag["creator_rain_segment_count"] == 1
    assert diag["provenance_relevance_flag"] == "suspect_interior_slice"
    assert "weak_creator_rain_area" in diag["provenance_relevance_reasons"]
    assert "covered_creators_dominate" in diag["provenance_relevance_reasons"]
    assert "unpaired" in diag["provenance_relevance_reasons"]


def test_collect_ridge_eave_target_diagnostics_leaves_rain_facing_plane_normal() -> (
    None
):
    ridge_entry = {
        "candidates": [
            {
                "id": "b-test::candidate::one",
                "parent_segment_id": "b-test::segment::rain-a",
                "area_m2": 4.0,
                "extended": False,
            },
            {
                "id": "b-test::candidate::two",
                "parent_segment_id": "b-test::segment::rain-b",
                "area_m2": 3.0,
                "extended": False,
            },
        ],
        "plane_groups": [
            {
                "id": "b-test::plane-group::ok1234",
                "selected": True,
                "member_ids": [
                    "b-test::candidate::one",
                    "b-test::candidate::two",
                ],
                "best_partner_plane_group_id": "b-test::plane-group::partner",
            }
        ],
    }
    v3_building = {
        "merged_roof_segments": [
            {
                "id": "b-test::segment::rain-a",
                "features": {"rain_hitting_side_count": 1, "covered_side_count": 0},
                "member_snapshots": [
                    {
                        "source_room_id": "room:7",
                        "slab_room_id": "room:7",
                        "features": {
                            "is_top_story_slab": True,
                            "slab_kind": "room",
                            "plane_height_above_slab_m": 1.0,
                        },
                    }
                ],
            },
            {
                "id": "b-test::segment::rain-b",
                "features": {"rain_hitting_side_count": 2, "covered_side_count": 0},
                "member_snapshots": [
                    {
                        "source_room_id": "room:8",
                        "slab_room_id": "room:8",
                        "features": {
                            "is_top_story_slab": True,
                            "slab_kind": "room",
                            "plane_height_above_slab_m": 0.8,
                        },
                    }
                ],
            },
        ]
    }

    diagnostics = collect_ridge_eave_target_diagnostics(
        "b-test", ridge_entry, v3_building
    )
    diag = diagnostics["b-test::ridge-eave-candidate::plane-group::ok1234"]

    assert math.isclose(diag["creator_rain_area_fraction"], 1.0, abs_tol=1e-6)
    assert diag["provenance_relevance_flag"] == "normal"


def test_plane_extent_split_pieces_clip_to_story_extent_envelope() -> None:
    poly = _candidate_polygon(
        {
            "minRidge": -5.0,
            "maxRidge": 5.0,
            "minSlope": -1.0,
            "maxSlope": 1.0,
            "ridgeX": 1.0,
            "ridgeZ": 0.0,
            "slopeX": 0.0,
            "slopeZ": 1.0,
            "ref": {"x": 5.0, "y": 3.0, "z": 1.0},
        }
    )
    assert poly is not None
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="candidate_oblique",
        target_index=0,
        element_id="b-test::ceiling-oblique::ceiling-oblique:0",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="full",
        edge_count=1,
        total_length_m=5.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(5.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (5.0, 0.0)]),
        member_plane_ids=("raw:full",),
    )
    supports = score_plane_eave_chain_supports([target], [chain])
    story_extent = build_story_extent_envelopes(
        {
            "rooms": [
                {
                    "story": 0,
                    "floor_polygon": [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 0.0, 2.0],
                        [0.0, 0.0, 2.0],
                    ],
                },
                {
                    "story": 0,
                    "floor_polygon": [
                        [3.0, 0.0, 0.0],
                        [4.0, 0.0, 0.0],
                        [4.0, 0.0, 2.0],
                        [3.0, 0.0, 2.0],
                    ],
                },
            ],
            "cross_floor_gaps": [
                {
                    "story": 0,
                    "corners": [
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                        [3.0, 0.0, 2.0],
                        [2.0, 0.0, 2.0],
                    ],
                }
            ],
        }
    )

    pieces = build_plane_extent_split_pieces(
        [target],
        [chain],
        supports,
        story_extent_envelopes=story_extent,
    )

    assert len(pieces) == 1
    assert math.isclose(pieces[0].area_xz_m2, 8.0, abs_tol=1e-6)


def test_plane_extent_split_pieces_do_not_clip_supported_to_segment_anchor_mask_early():
    poly = _candidate_polygon(
        {
            "minRidge": -5.0,
            "maxRidge": 5.0,
            "minSlope": -1.0,
            "maxSlope": 1.0,
            "ridgeX": 1.0,
            "ridgeZ": 0.0,
            "slopeX": 0.0,
            "slopeZ": 1.0,
            "ref": {"x": 5.0, "y": 3.0, "z": 1.0},
        }
    )
    assert poly is not None
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::anchor-test",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="full",
        edge_count=1,
        total_length_m=10.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(10.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (10.0, 0.0)]),
        member_plane_ids=("raw:full",),
    )
    supports = score_plane_eave_chain_supports([target], [chain])
    pieces = build_plane_extent_split_pieces(
        [target],
        [chain],
        supports,
        target_segment_anchor_masks={
            target.element_id: Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)])
        },
        segment_anchor_buffer_m=0.0,
    )

    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]
    assert len(supported_pieces) == 1
    assert supported_pieces[0].chain_ids == ("full",)
    assert math.isclose(supported_pieces[0].area_xz_m2, 20.0, abs_tol=1e-6)


def test_ridge_eave_anchor_mask_can_restore_local_supported_domain() -> None:
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::anchor-union",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="full",
        edge_count=1,
        total_length_m=10.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(0.0, 0.0),
        end_xz=(10.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (10.0, 0.0)]),
        member_plane_ids=("raw:full",),
    )
    supports = score_plane_eave_chain_supports([target], [chain])
    pieces = build_plane_extent_split_pieces(
        [target],
        [chain],
        supports,
        target_segment_anchor_masks={
            target.element_id: Polygon(
                [
                    (4.0, 0.0),
                    (10.0, 0.0),
                    (10.0, 2.0),
                    (4.0, 2.0),
                ]
            )
        },
        segment_anchor_buffer_m=0.0,
    )

    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]
    residual_pieces = [piece for piece in pieces if piece.piece_role == "residual"]
    assert len(supported_pieces) == 1
    assert math.isclose(supported_pieces[0].area_xz_m2, 20.0, abs_tol=1e-6)
    assert len(residual_pieces) == 0


def test_ridge_eave_sweep_falls_back_to_nonempty_side_when_plane_side_is_empty() -> (
    None
):
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 4.0],
        [0.0, 4.0, 4.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::fallback",
        poly_xz=poly,
        normal=-normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    chain = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="tip",
        edge_count=1,
        total_length_m=2.0,
        azimuth_deg=90.0,
        y_mean=2.0,
        start_xz=(8.0, 4.0),
        end_xz=(10.0, 4.0),
        line_xz=LineString([(8.0, 4.0), (10.0, 4.0)]),
        member_plane_ids=("raw:tip",),
    )
    supports = [
        PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id=target.element_id,
            target_kind=target.target_kind,
            chain_id="tip",
            chain_azimuth_deg=90.0,
            ridge_azimuth_deg=0.0,
            angle_delta_deg=90.0,
            boundary_distance_m=0.0,
            overlap_fraction=1.0,
            height_residual_m=0.0,
            support_score=1.0,
            supported=True,
            chain_length_m=2.0,
        )
    ]

    pieces = build_plane_extent_split_pieces(
        [target],
        [chain],
        supports,
    )

    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]
    assert len(supported_pieces) == 1
    supported_poly = Polygon(
        [(corner[0], corner[2]) for corner in supported_pieces[0].corners]
    )
    assert math.isclose(float(supported_poly.bounds[1]), 0.0, abs_tol=1e-6)
    assert supported_poly.bounds[3] > 3.5


def test_trim_ridge_eave_supported_pieces_to_chain_run_bands_splits_same_family_runs():
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 4.0],
        [0.0, 4.0, 4.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    left_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::left",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    right_target = replace(
        left_target,
        target_index=1,
        element_id="b-test::ridge-eave-candidate::plane-group::right",
    )
    left_piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=left_target.element_id,
        target_kind=left_target.target_kind,
        piece_id=f"{left_target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=28.0,
        support_score=1.0,
        chain_ids=("left:a", "left:b"),
        corners=[
            [0.0, 2.0, 0.0],
            [7.0, 2.0, 0.0],
            [7.0, 4.0, 4.0],
            [0.0, 4.0, 4.0],
        ],
        holes=[],
    )
    right_piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=right_target.element_id,
        target_kind=right_target.target_kind,
        piece_id=f"{right_target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=28.0,
        support_score=1.0,
        chain_ids=("right:a", "right:b"),
        corners=[
            [3.0, 2.0, 0.0],
            [10.0, 2.0, 0.0],
            [10.0, 4.0, 4.0],
            [3.0, 4.0, 4.0],
        ],
        holes=[],
    )
    chains = [
        EaveChainRecord(
            uuid="b-test",
            story=0,
            chain_id="left:a",
            edge_count=1,
            total_length_m=1.0,
            azimuth_deg=90.0,
            y_mean=2.0,
            start_xz=(0.5, 0.0),
            end_xz=(2.5, 0.0),
            line_xz=LineString([(0.5, 0.0), (2.5, 0.0)]),
            member_plane_ids=("raw:left:a",),
        ),
        EaveChainRecord(
            uuid="b-test",
            story=0,
            chain_id="left:b",
            edge_count=1,
            total_length_m=1.0,
            azimuth_deg=90.0,
            y_mean=2.0,
            start_xz=(1.0, 0.0),
            end_xz=(3.0, 0.0),
            line_xz=LineString([(1.0, 0.0), (3.0, 0.0)]),
            member_plane_ids=("raw:left:b",),
        ),
        EaveChainRecord(
            uuid="b-test",
            story=0,
            chain_id="right:a",
            edge_count=1,
            total_length_m=1.0,
            azimuth_deg=90.0,
            y_mean=2.0,
            start_xz=(7.0, 0.0),
            end_xz=(9.0, 0.0),
            line_xz=LineString([(7.0, 0.0), (9.0, 0.0)]),
            member_plane_ids=("raw:right:a",),
        ),
        EaveChainRecord(
            uuid="b-test",
            story=0,
            chain_id="right:b",
            edge_count=1,
            total_length_m=1.0,
            azimuth_deg=90.0,
            y_mean=2.0,
            start_xz=(7.5, 0.0),
            end_xz=(9.5, 0.0),
            line_xz=LineString([(7.5, 0.0), (9.5, 0.0)]),
            member_plane_ids=("raw:right:b",),
        ),
    ]

    trimmed = trim_ridge_eave_supported_pieces_to_chain_run_bands(
        [left_target, right_target],
        chains,
        [left_piece, right_piece],
    )

    trimmed_by_id = {piece.piece_id: piece for piece in trimmed}
    left_poly = Polygon(
        [
            (corner[0], corner[2])
            for corner in trimmed_by_id[left_piece.piece_id].corners
        ]
    )
    right_poly = Polygon(
        [
            (corner[0], corner[2])
            for corner in trimmed_by_id[right_piece.piece_id].corners
        ]
    )
    assert left_poly.bounds[2] < 6.0
    assert right_poly.bounds[0] > 4.0
    assert float(left_poly.intersection(right_poly).area) < 1e-6


def test_collect_ridge_eave_target_anchor_masks_prefers_source_room_component() -> None:
    uuid = "b-test"
    ridge_entry = {
        "candidates": [
            {"id": "cand:left", "parent_segment_id": "seg:left", "extended": False},
            {"id": "cand:right", "parent_segment_id": "seg:right", "extended": False},
        ],
        "plane_groups": [
            {
                "id": f"{uuid}::plane-group::pg",
                "selected": True,
                "member_ids": ["cand:left", "cand:right"],
            }
        ],
    }
    building = {
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [8.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [8.0, 0.0, 2.0],
                ],
            },
        ],
    }
    v3_building = {
        "merged_roof_segments": [
            {
                "id": "seg:left",
                "footprint_xz": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            },
            {
                "id": "seg:right",
                "footprint_xz": [[8.0, 0.0], [10.0, 0.0], [10.0, 2.0], [8.0, 2.0]],
            },
        ]
    }
    diagnostics = {
        f"{uuid}::ridge-eave-candidate::plane-group::pg": {
            "provenance_relevance_flag": "suspect_interior_slice",
            "creator_extended_area_fraction": 1.0,
            "creator_source_room_ids": ["room:1"],
        }
    }

    masks = collect_ridge_eave_target_anchor_masks(
        uuid,
        ridge_entry,
        building,
        v3_building,
        diagnostics,
    )

    mask = masks[f"{uuid}::ridge-eave-candidate::plane-group::pg"]
    assert math.isclose(mask.area, 4.0, abs_tol=1e-6)
    assert tuple(round(v, 6) for v in mask.bounds) == (8.0, 0.0, 10.0, 2.0)


def test_trim_to_room_clips_late() -> None:
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [5.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [5.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [5.0, 0.0, 2.0],
                ],
            },
        ],
    }
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    ridge_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="target:ridge",
        poly_xz=Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0)]),
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=20.0,
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    committed_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="committed_oblique",
        target_index=1,
        element_id="target:committed",
        poly_xz=Polygon([(5.0, 0.0), (10.0, 0.0), (10.0, 2.0), (5.0, 2.0)]),
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=10.0,
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    split_targets = [ridge_target, committed_target]
    split_target_score_rows = [
        {
            "element_id": ridge_target.element_id,
            "target_kind": ridge_target.target_kind,
            "retention_support_score": 0.75,
        },
        {
            "element_id": committed_target.element_id,
            "target_kind": committed_target.target_kind,
            "retention_support_score": 0.95,
        },
    ]
    split_pieces = [
        TargetSplitPieceRecord(
            uuid="b-test",
            story=0,
            target_element_id=ridge_target.element_id,
            target_kind=ridge_target.target_kind,
            piece_id="target:ridge#supported:0:0",
            piece_index=0,
            piece_role="supported",
            area_xz_m2=20.0,
            support_score=0.8,
            chain_ids=("chain:a",),
            corners=corners,
            holes=[],
        ),
        TargetSplitPieceRecord(
            uuid="b-test",
            story=0,
            target_element_id=committed_target.element_id,
            target_kind=committed_target.target_kind,
            piece_id="target:committed#supported:0:0",
            piece_index=1,
            piece_role="supported",
            area_xz_m2=10.0,
            support_score=0.95,
            chain_ids=("chain:b",),
            corners=[
                [5.0, 3.0, 0.0],
                [10.0, 3.0, 0.0],
                [10.0, 4.0, 2.0],
                [5.0, 4.0, 2.0],
            ],
            holes=[],
        ),
    ]

    trimmed = trim_ridge_eave_supported_pieces_to_room_ownership(
        building,
        split_targets,
        split_target_score_rows,
        split_pieces,
        room_buffer_m=0.0,
    )

    by_piece = {piece.piece_id: piece for piece in trimmed}
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"].area_xz_m2, 10.0, abs_tol=1e-6
    )
    assert math.isclose(
        by_piece["target:committed#supported:0:0"].area_xz_m2, 10.0, abs_tol=1e-6
    )


def test_trim_mirror_partner_clips_equal_seam() -> None:
    own_corners = [
        [0.0, 0.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 2.0, 2.0],
        [0.0, 0.0, 2.0],
    ]
    partner_corners = [
        [0.0, 2.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
    ]
    own_centroid, own_normal = _fit_plane_svd(own_corners)
    partner_centroid, partner_normal = _fit_plane_svd(partner_corners)
    assert own_centroid is not None
    assert own_normal is not None
    assert partner_centroid is not None
    assert partner_normal is not None

    own_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::own",
        poly_xz=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        normal=own_normal,
        azimuth_deg=90.0,
        inclination_deg=45.0,
        ridge_dir_xz=(0.0, 1.0),
        area_xz_m2=8.0,
        plane_point=(
            float(own_centroid[0]),
            float(own_centroid[1]),
            float(own_centroid[2]),
        ),
    )
    partner_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=1,
        element_id="b-test::ridge-eave-candidate::plane-group::partner",
        poly_xz=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        normal=partner_normal,
        azimuth_deg=270.0,
        inclination_deg=45.0,
        ridge_dir_xz=(0.0, 1.0),
        area_xz_m2=8.0,
        plane_point=(
            float(partner_centroid[0]),
            float(partner_centroid[1]),
            float(partner_centroid[2]),
        ),
    )
    split_pieces = [
        TargetSplitPieceRecord(
            uuid="b-test",
            story=0,
            target_element_id=own_target.element_id,
            target_kind=own_target.target_kind,
            piece_id="b-test::ridge-eave-candidate::plane-group::own#supported:0:0",
            piece_index=0,
            piece_role="supported",
            area_xz_m2=8.0,
            support_score=0.8,
            chain_ids=("chain:a",),
            corners=own_corners,
            holes=[],
        )
    ]
    ridge_entry = {
        "plane_groups": [
            {
                "id": "b-test::plane-group::own",
                "best_partner_plane_group_id": "b-test::plane-group::partner",
            },
            {
                "id": "b-test::plane-group::partner",
                "best_partner_plane_group_id": "b-test::plane-group::own",
            },
        ],
    }

    trimmed = trim_ridge_eave_supported_pieces_to_mirror_partner(
        "b-test",
        ridge_entry,
        [own_target, partner_target],
        split_pieces,
        pad_m=0.0,
    )

    assert len(trimmed) == 1
    assert math.isclose(trimmed[0].area_xz_m2, 4.0, abs_tol=1e-6)


def test_trim_ridge_eave_rows_to_local_mirror_pieces_clips_reciprocal_supported_pair():
    left_row = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::left",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::left#supported:0:0",
        "piece_role": "supported",
        "target_azimuth_deg": 90.0,
        "target_inclination_deg": 45.0,
        "chain_signature_id": "sig:shared",
        "mirror_partner_plane_group_id": "b-test::plane-group::right",
        "chain_ids": ["chain:a"],
        "corners": [
            [0.0, 0.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        "holes": [],
    }
    right_row = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::right",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::right#supported:0:0",
        "piece_role": "supported",
        "target_azimuth_deg": 270.0,
        "target_inclination_deg": 45.0,
        "chain_signature_id": "sig:shared",
        "mirror_partner_plane_group_id": "b-test::plane-group::left",
        "chain_ids": ["chain:a"],
        "corners": [
            [0.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
        ],
        "holes": [],
    }

    trimmed = trim_ridge_eave_rows_to_local_mirror_pieces(
        [left_row, right_row], pad_m=0.0
    )

    by_piece = {row["piece_id"]: row for row in trimmed}
    left_poly = Polygon(
        [(corner[0], corner[2]) for corner in by_piece[left_row["piece_id"]]["corners"]]
    )
    right_poly = Polygon(
        [
            (corner[0], corner[2])
            for corner in by_piece[right_row["piece_id"]]["corners"]
        ]
    )
    assert math.isclose(float(left_poly.area), 4.0, abs_tol=1e-6)
    assert math.isclose(float(right_poly.area), 4.0, abs_tol=1e-6)
    assert float(left_poly.intersection(right_poly).area) < 1e-6


def test_prune_ridge_eave_rows_drops_final_row_with_tiny_same_signature_overlap() -> (
    None
):
    row = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::left",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::left#supported:0:0",
        "piece_role": "supported",
        "chain_signature_id": "sig:shared",
        "mirror_partner_plane_group_id": "b-test::plane-group::right",
        "mirror_support_score": 0.99,
        "through_ratio": 0.5,
        "final_layer": True,
        "ownership_redundant": False,
        "corners": [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    }
    partner = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::right",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::right#supported:0:0",
        "piece_role": "supported",
        "chain_signature_id": "sig:shared",
        "mirror_partner_plane_group_id": "b-test::plane-group::left",
        "mirror_support_score": 0.99,
        "through_ratio": 0.95,
        "final_layer": True,
        "ownership_redundant": False,
        # Overlap strip is 0.001 x 2 = 0.002 m2 (< 0.005 threshold).
        "corners": [
            [3.999, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 2.0],
            [3.999, 0.0, 2.0],
        ],
    }

    pruned, diag = prune_ridge_eave_rows_with_unreliable_mirror_pairs([row, partner])
    piece_ids = {str(item.get("piece_id") or "") for item in pruned}

    assert row["piece_id"] not in piece_ids
    assert partner["piece_id"] in piece_ids
    assert diag["drop_reason_counts"]["final_same_signature_tiny_partner_overlap"] == 1


def test_prune_ridge_eave_rows_drops_final_row_with_missing_partner() -> None:
    row = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::solo",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::solo#supported:0:0",
        "piece_role": "supported",
        "chain_signature_id": "sig:solo",
        "mirror_partner_plane_group_id": "b-test::plane-group::missing",
        "mirror_support_score": 0.91,
        "through_ratio": 0.6,
        "final_layer": True,
        "ownership_redundant": False,
        "corners": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    }

    pruned, diag = prune_ridge_eave_rows_with_unreliable_mirror_pairs([row])
    piece_ids = {str(item.get("piece_id") or "") for item in pruned}

    assert row["piece_id"] not in piece_ids
    assert diag["drop_reason_counts"]["final_missing_mirror_partner"] == 1
    assert not piece_ids


def test_prune_redundant_mismatched_signature() -> None:
    row = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::left",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::left#supported:0:0",
        "piece_role": "supported",
        "chain_signature_id": "sig:left",
        "mirror_partner_plane_group_id": "b-test::plane-group::right",
        "mirror_support_score": 0.99,
        "through_ratio": 1.0,
        "final_layer": False,
        "ownership_redundant": True,
        "corners": [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    }
    partner = {
        "target_element_id": "b-test::ridge-eave-candidate::plane-group::right",
        "target_kind": "ridge_eave_plane_group",
        "piece_id": "b-test::ridge-eave-candidate::plane-group::right#supported:0:0",
        "piece_role": "supported",
        "chain_signature_id": "sig:right",
        "mirror_partner_plane_group_id": "b-test::plane-group::left",
        "mirror_support_score": 0.99,
        "through_ratio": 1.0,
        "final_layer": False,
        "ownership_redundant": False,
        # Overlap strip is 0.01 x 2 = 0.02 m2 (< 0.05 threshold).
        "corners": [
            [3.99, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 2.0],
            [3.99, 0.0, 2.0],
        ],
    }

    pruned, diag = prune_ridge_eave_rows_with_unreliable_mirror_pairs([row, partner])
    piece_ids = {str(item.get("piece_id") or "") for item in pruned}

    assert row["piece_id"] not in piece_ids
    assert partner["piece_id"] in piece_ids
    assert (
        diag["drop_reason_counts"][
            "redundant_mismatched_signature_tiny_partner_overlap"
        ]
        == 1
    )


def test_trim_to_room_clips_face_run_parts() -> None:
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [4.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [6.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                ],
            },
        ],
    }
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::subject",
        poly_xz=Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)]),
        normal=(0.0, 1.0, 0.0),
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=12.0,
        plane_point=(0.0, 2.0, 0.0),
    )
    piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=target.element_id,
        target_kind=target.target_kind,
        piece_id=f"{target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=12.0,
        support_score=0.9,
        chain_ids=("chain:a",),
        corners=[[0.0, 2.0, 0.0], [6.0, 2.0, 0.0], [6.0, 2.0, 2.0], [0.0, 2.0, 2.0]],
        holes=[],
    )

    trimmed = trim_ridge_eave_supported_pieces_to_room_ownership(
        building,
        [target],
        [
            {
                "element_id": target.element_id,
                "retention_support_score": 0.9,
                "face_run_hypothesis_part_ids": ["part:left"],
            }
        ],
        [piece],
        building_part_graph={
            "room_membership": {
                "room:0": ["part:left"],
                "room:1": ["part:left"],
                "room:2": ["part:right"],
            }
        },
        room_buffer_m=0.0,
    )

    assert len(trimmed) == 1
    trimmed_poly = Polygon([(corner[0], corner[2]) for corner in trimmed[0].corners])
    assert math.isclose(float(trimmed_poly.area), 8.0, abs_tol=1e-6)
    assert math.isclose(float(trimmed_poly.bounds[2]), 4.0, abs_tol=1e-6)


def test_trim_to_room_falls_back_to_cross_story_part_union() -> None:
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [6.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "story": 2,
                "floor_polygon": [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                ],
            },
        ],
    }
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::subject",
        poly_xz=Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)]),
        normal=(0.0, 1.0, 0.0),
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=12.0,
        plane_point=(0.0, 2.0, 0.0),
    )
    piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=target.element_id,
        target_kind=target.target_kind,
        piece_id=f"{target.element_id}#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=12.0,
        support_score=0.9,
        chain_ids=("chain:a",),
        corners=[[0.0, 2.0, 0.0], [6.0, 2.0, 0.0], [6.0, 2.0, 2.0], [0.0, 2.0, 2.0]],
        holes=[],
    )

    trimmed = trim_ridge_eave_supported_pieces_to_room_ownership(
        building,
        [target],
        [
            {
                "element_id": target.element_id,
                "retention_support_score": 0.9,
                "face_run_hypothesis_part_ids": ["part:upper"],
            }
        ],
        [piece],
        building_part_graph={
            "room_membership": {
                "room:1": ["part:upper"],
            }
        },
        room_buffer_m=0.0,
    )

    assert len(trimmed) == 1
    trimmed_poly = Polygon([(corner[0], corner[2]) for corner in trimmed[0].corners])
    assert math.isclose(float(trimmed_poly.area), 4.0, abs_tol=1e-6)
    assert trimmed_poly.bounds == (2.0, 0.0, 4.0, 2.0)


def test_annotate_rows_with_target_face_runs_uses_element_id_for_score_rows() -> None:
    rows = annotate_rows_with_target_face_runs(
        [{"element_id": "target:ridge"}],
        {
            "target:ridge": {
                "face_run_id": "face-run:1",
                "face_run_hypothesis_part_ids": ["part:a"],
            }
        },
    )

    assert rows[0]["face_run_id"] == "face-run:1"
    assert rows[0]["face_run_hypothesis_part_ids"] == ["part:a"]


def test_piece_records_from_polygon_normalizes_degenerate_holes() -> None:
    target = _make_target()
    poly = Polygon(
        [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)],
        holes=[[(1.0, 1.0), (1.0, 1.0), (2.0, 1.0)]],
    )

    record = _piece_records_from_polygon(
        target,
        poly,
        piece_id="b-test::piece",
        piece_index=0,
        piece_role="supported",
        support_score=1.0,
        chain_ids=(),
    )

    assert record is not None
    assert record.holes == []


def test_serialized_piece_holes_drops_loops_that_collapse_after_rounding() -> None:
    holes = [
        [
            [-2.1686604, 0.6900271, -3.3762784],
            [-2.1686604, 0.6900271, -3.3762784],
            [-2.5534157, 0.5442555, -3.0106156],
        ],
        [
            [-0.534866, 0.622711, -1.304385],
            [-1.012485, 0.622862, -1.806944],
            [-0.4803, 0.824524, -2.3129],
            [-0.0026, 0.8244, -1.8104],
        ],
    ]

    serialized = _serialized_piece_holes(holes)

    assert serialized == [holes[1]]


def test_ridge_eave_plane_group_keeps_single_strongest_supported_component() -> None:
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0)])
    corners = [
        [0.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
    ]
    centroid, normal = _fit_plane_svd(corners)
    assert centroid is not None
    assert normal is not None
    target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::single-best",
        poly_xz=poly,
        normal=normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=float(poly.area),
        plane_point=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )
    chains = [
        EaveChainRecord(
            uuid="b-test",
            story=0,
            chain_id="c-high",
            edge_count=1,
            total_length_m=4.0,
            azimuth_deg=90.0,
            y_mean=2.0,
            start_xz=(0.0, 0.0),
            end_xz=(4.0, 0.0),
            line_xz=LineString([(0.0, 0.0), (4.0, 0.0)]),
            member_plane_ids=("raw:0",),
        ),
        EaveChainRecord(
            uuid="b-test",
            story=0,
            chain_id="c-low",
            edge_count=1,
            total_length_m=4.0,
            azimuth_deg=90.0,
            y_mean=2.0,
            start_xz=(6.0, 0.0),
            end_xz=(10.0, 0.0),
            line_xz=LineString([(6.0, 0.0), (10.0, 0.0)]),
            member_plane_ids=("raw:1",),
        ),
    ]
    supports = [
        PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id=target.element_id,
            target_kind=target.target_kind,
            chain_id="c-high",
            chain_azimuth_deg=90.0,
            ridge_azimuth_deg=90.0,
            angle_delta_deg=0.0,
            boundary_distance_m=0.0,
            overlap_fraction=1.0,
            height_residual_m=0.0,
            support_score=0.9,
            supported=True,
            chain_length_m=4.0,
        ),
        PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id=target.element_id,
            target_kind=target.target_kind,
            chain_id="c-low",
            chain_azimuth_deg=90.0,
            ridge_azimuth_deg=90.0,
            angle_delta_deg=0.0,
            boundary_distance_m=0.0,
            overlap_fraction=1.0,
            height_residual_m=0.0,
            support_score=0.8,
            supported=True,
            chain_length_m=4.0,
        ),
    ]

    anchor = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]).union(
        Polygon([(6.0, 0.0), (10.0, 0.0), (10.0, 2.0), (6.0, 2.0)])
    )
    pieces = build_plane_extent_split_pieces(
        [target],
        chains,
        supports,
        target_segment_anchor_masks={target.element_id: anchor},
        segment_anchor_buffer_m=0.0,
    )

    supported_pieces = [piece for piece in pieces if piece.piece_role == "supported"]
    residual_pieces = [piece for piece in pieces if piece.piece_role == "residual"]
    assert len(supported_pieces) == 2
    assert sorted(piece.chain_ids for piece in supported_pieces) == [
        ("c-high",),
        ("c-low",),
    ]
    assert len(residual_pieces) == 1
    assert all(piece.area_xz_m2 > 0.0 for piece in supported_pieces)
    assert residual_pieces[0].area_xz_m2 > 0.0
    assert math.isclose(
        sum(piece.area_xz_m2 for piece in supported_pieces)
        + residual_pieces[0].area_xz_m2,
        20.0,
        abs_tol=1e-6,
    )


def test_diag_chain_owner_reports_through_and_competitor_loss() -> None:
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "story": 0,
                "floor_polygon": [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                ],
            },
        ],
    }
    subject_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::subject",
        poly_xz=Polygon([(1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (1.0, 3.0)]),
        normal=_make_target().normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(0.0, 1.0),
        area_xz_m2=8.0,
        plane_point=(2.0, 2.0, 1.0),
    )
    competitor_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="candidate_oblique",
        target_index=1,
        element_id="b-test::ceiling-oblique::ceiling-oblique:1",
        poly_xz=Polygon([(2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0)]),
        normal=_make_target().normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=4.0,
        plane_point=(3.0, 2.0, 1.0),
    )
    piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=subject_target.element_id,
        target_kind="ridge_eave_plane_group",
        piece_id="b-test::ridge-eave-candidate::plane-group::subject#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=4.0,
        support_score=0.8,
        chain_ids=("b-test::eave-chain::0:0",),
        corners=[[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 2.0, 1.0], [0.0, 2.0, 1.0]],
        holes=[],
    )
    support = score_plane_eave_chain_supports(
        [subject_target],
        [
            EaveChainRecord(
                uuid="b-test",
                story=0,
                chain_id="b-test::eave-chain::0:0",
                edge_count=1,
                total_length_m=2.0,
                azimuth_deg=0.0,
                y_mean=0.0,
                start_xz=(1.0, 0.0),
                end_xz=(3.0, 0.0),
                line_xz=LineString([(1.0, 0.0), (3.0, 0.0)]),
                member_plane_ids=("b-test::ceiling-raw::0:0:0",),
            )
        ],
    )[0]
    row = diagnose_ridge_eave_supported_piece_ownership(
        building,
        piece,
        targets_by_id={
            subject_target.element_id: subject_target,
            competitor_target.element_id: competitor_target,
        },
        target_scores_by_id={
            subject_target.element_id: {
                "element_id": subject_target.element_id,
                "retention_support_score": 0.4,
            },
            competitor_target.element_id: {
                "element_id": competitor_target.element_id,
                "retention_support_score": 0.9,
            },
        },
        supported_chain_by_target={subject_target.element_id: support},
        supported_ridge_eave_pieces_by_signature={},
        ridge_eave_meta_by_target={
            subject_target.element_id: {
                "best_partner_plane_group_id": None,
                "best_score": None,
                "creator_eave_proximity": 0.02,
            }
        },
        ridge_eave_target_diagnostics={
            subject_target.element_id: {
                "creator_source_room_ids": ["room:0"],
                "creator_touch_room_ids": ["room:0", "room:1"],
                "creator_source_room_count": 1,
                "creator_touch_room_count": 2,
            }
        },
    )

    assert row is not None
    assert row["crossed_room_count"] == 2
    assert row["through_ratio"] > 1.0
    assert row["local_competitor_loss_room_count"] == 1
    assert row["local_competitor_loss_area_m2"] > 0.0
    assert row["local_top_competitor_ids"] == [competitor_target.element_id]
    assert row["mirror_partner_plane_group_id"] is None


def test_diag_chain_owner_ignores_opposite_facing_overlap() -> None:
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ],
    }
    subject_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::subject",
        poly_xz=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        normal=_make_target().normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=8.0,
        plane_point=(2.0, 2.0, 1.0),
    )
    opposite_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=1,
        element_id="b-test::ridge-eave-candidate::plane-group::opposite",
        poly_xz=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        normal=_make_target().normal,
        azimuth_deg=0.0,
        inclination_deg=44.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=8.0,
        plane_point=(2.0, 2.0, 1.0),
    )
    piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=subject_target.element_id,
        target_kind="ridge_eave_plane_group",
        piece_id="b-test::ridge-eave-candidate::plane-group::subject#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=8.0,
        support_score=0.8,
        chain_ids=("b-test::eave-chain::0:0",),
        corners=[[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 2.0, 2.0], [0.0, 2.0, 2.0]],
        holes=[],
    )
    support = score_plane_eave_chain_supports(
        [subject_target],
        [
            EaveChainRecord(
                uuid="b-test",
                story=0,
                chain_id="b-test::eave-chain::0:0",
                edge_count=1,
                total_length_m=4.0,
                azimuth_deg=90.0,
                y_mean=0.0,
                start_xz=(0.0, 0.0),
                end_xz=(4.0, 0.0),
                line_xz=LineString([(0.0, 0.0), (4.0, 0.0)]),
                member_plane_ids=("b-test::ceiling-raw::0:0:0",),
            )
        ],
    )[0]

    row = diagnose_ridge_eave_supported_piece_ownership(
        building,
        piece,
        targets_by_id={
            subject_target.element_id: subject_target,
            opposite_target.element_id: opposite_target,
        },
        target_scores_by_id={
            subject_target.element_id: {
                "element_id": subject_target.element_id,
                "retention_support_score": 0.4,
            },
            opposite_target.element_id: {
                "element_id": opposite_target.element_id,
                "retention_support_score": 0.95,
            },
        },
        supported_chain_by_target={subject_target.element_id: support},
        supported_ridge_eave_pieces_by_signature={},
        ridge_eave_meta_by_target={subject_target.element_id: {}},
        ridge_eave_target_diagnostics={subject_target.element_id: {}},
    )

    assert row is not None
    assert row["local_competitor_loss_room_count"] == 0
    assert row["local_competitor_loss_area_m2"] == 0.0
    assert row["local_top_competitor_ids"] == []


def test_merge_split_piece_rows_with_ownership_attaches_piece_metrics() -> None:
    merged_rows = merge_split_piece_rows_with_ownership(
        [
            {
                "piece_id": "piece:a",
                "target_element_id": "target:a",
                "support_score": 0.8,
            },
            {
                "piece_id": "piece:b",
                "target_element_id": "target:b",
                "support_score": 0.6,
            },
        ],
        [
            {
                "piece_id": "piece:a",
                "through_ratio": 1.75,
                "local_competitor_loss_fraction": 1.0,
                "mirror_partner_plane_group_id": "target:mirror",
            }
        ],
    )

    assert merged_rows[0]["through_ratio"] == 1.75
    assert merged_rows[0]["local_competitor_loss_fraction"] == 1.0
    assert merged_rows[0]["mirror_partner_plane_group_id"] == "target:mirror"
    assert "through_ratio" not in merged_rows[1]


def test_classify_final_keeps_extension_owner_pair_final() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:committed",
                "target_kind": "committed_oblique",
                "piece_id": "t:committed#supported:0:0",
                "piece_role": "supported",
            },
            {
                "target_element_id": "t:candidate",
                "target_kind": "candidate_oblique",
                "piece_id": "t:candidate#supported:0:0",
                "piece_role": "supported",
            },
            {
                "target_element_id": "t:ridge:owner",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge:owner#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.2,
                "roof_surface_cover_fraction": 0.2,
            },
            {
                "target_element_id": "t:ridge:owner",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge:owner#residual:0",
                "piece_role": "residual",
            },
            {
                "target_element_id": "t:ridge:through",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge:through#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 1.0,
                "roof_surface_cover_fraction": 0.2,
            },
            {
                "target_element_id": "t:ridge:through",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge:through#residual:0",
                "piece_role": "residual",
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:committed#supported:0:0"]["final_layer"] is True
    assert (
        by_piece["t:committed#supported:0:0"]["final_layer_reason"]
        == "committed_oblique"
    )
    assert by_piece["t:candidate#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:candidate#supported:0:0"]["final_layer_reason"]
        == "candidate_oblique"
    )
    assert by_piece["t:ridge:owner#supported:0:0"]["final_layer"] is True
    assert (
        by_piece["t:ridge:owner#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_local_ownership"
    )
    assert by_piece["t:ridge:owner#residual:0"]["final_layer"] is False
    assert by_piece["t:ridge:through#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:ridge:through#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_competitor_loss"
    )
    assert by_piece["t:ridge:through#residual:0"]["final_layer"] is False


def test_classify_final_demotes_committed_piece_in_wrong_building_part() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_kind": "committed_oblique",
                "piece_role": "supported",
                "piece_id": "t:committed#supported:0:0",
                "hypothesis_part_misaligned": True,
            }
        ]
    )

    assert rows[0]["final_layer"] is False
    assert rows[0]["final_layer_reason"] == "committed_wrong_building_part"


def test_classify_final_keeps_single_committed_owner_per_target() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "uuid": "b-test",
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_role": "supported",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:0:0",
                "area_xz_m2": 9.0,
                "support_score": 0.99,
            },
            {
                "uuid": "b-test",
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_role": "supported",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:1:0",
                "area_xz_m2": 13.0,
                "support_score": 0.9,
            },
            {
                "uuid": "b-test",
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_role": "supported",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:2:0",
                "area_xz_m2": 11.0,
                "support_score": 0.95,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:1:0"]["final_layer"] is True
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:1:0"]["final_layer_reason"]
        == "committed_oblique"
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:0:0"]["final_layer"]
        is False
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:0:0"]["final_layer_reason"]
        == "committed_union_demoted"
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:2:0"]["final_layer"]
        is False
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:2:0"]["final_layer_reason"]
        == "committed_union_demoted"
    )


def test_classify_final_committed_owner_pass_respects_misaligned_drop() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "uuid": "b-test",
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_role": "supported",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:0:0",
                "area_xz_m2": 20.0,
                "support_score": 0.99,
                "hypothesis_part_misaligned": True,
            },
            {
                "uuid": "b-test",
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_role": "supported",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:1:0",
                "area_xz_m2": 8.0,
                "support_score": 0.8,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:0:0"]["final_layer"]
        is False
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:0:0"]["final_layer_reason"]
        == "committed_wrong_building_part"
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:1:0"]["final_layer"] is True
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:1:0"]["final_layer_reason"]
        == "committed_oblique"
    )


def test_classify_final_ignores_redundant_ridge_eave_piece() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "ownership_redundant": True,
            },
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#residual:0",
                "piece_role": "residual",
                "ownership_redundant": True,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:ridge#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:ridge#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_competitor_loss"
    )


def test_classify_final_keeps_redundant_piece_for_strict_disjoint_extension_pattern():
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "ownership_redundant": True,
                "creator_source_room_ids": ["room:6"],
                "crossed_room_ids": ["room:7", "room:9", "room:10"],
                "creator_source_room_count": 1,
                "provenance_relevance_flag": "suspect_interior_slice",
                "provenance_relevance_reasons": [
                    "weak_creator_rain_area",
                    "covered_creators_dominate",
                    "mostly_extended",
                    "cuts_below_top_story",
                ],
                "through_ratio": 1.2,
                "roof_surface_cover_fraction": 0.2,
            }
        ]
    )

    assert rows[0]["final_layer"] is True
    assert rows[0]["final_layer_reason"] == "ridge_eave_local_ownership"


def test_classify_final_suppress_zero_overlap_with_sibling() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "roof_surface_cover_fraction": 0.2,
                "creator_source_part_overlap_area_m2": 0.0,
                "creator_source_room_overlap_area_m2": 0.0,
            },
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0:1",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "roof_surface_cover_fraction": 0.2,
                "creator_source_part_overlap_area_m2": 7.0,
                "creator_source_room_overlap_area_m2": 4.0,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:ridge#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:ridge#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_source_part_mismatch"
    )
    assert by_piece["t:ridge#supported:0:0:1"]["final_layer"] is True
    assert (
        by_piece["t:ridge#supported:0:0:1"]["final_layer_reason"]
        == "ridge_eave_local_ownership"
    )


def test_classify_final_promotes_sibling_when_demoted() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "ownership_redundant": True,
                "creator_source_room_ids": ["room:6"],
                "crossed_room_ids": ["room:7", "room:9"],
                "creator_source_room_count": 1,
                "provenance_relevance_flag": "suspect_interior_slice",
                "provenance_relevance_reasons": [
                    "weak_creator_rain_area",
                    "covered_creators_dominate",
                    "mostly_extended",
                    "cuts_below_top_story",
                ],
                "through_ratio": 1.2,
                "roof_surface_cover_fraction": 0.2,
                "creator_source_part_overlap_area_m2": 0.0,
                "creator_source_room_overlap_area_m2": 0.0,
            },
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#residual:0",
                "piece_role": "residual",
                "creator_source_part_overlap_area_m2": 5.0,
                "creator_source_room_overlap_area_m2": 2.5,
                "area_xz_m2": 5.0,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:ridge#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:ridge#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_source_part_mismatch"
    )
    assert by_piece["t:ridge#residual:0"]["final_layer"] is True
    assert (
        by_piece["t:ridge#residual:0"]["final_layer_reason"]
        == "ridge_eave_source_part_owner"
    )


def test_classify_final_demotes_zero_roof_cover_support() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "committed_cover_fraction": 0.0,
                "roof_surface_cover_fraction": 0.0,
                "local_roof_cover_fraction": 0.0,
            }
        ]
    )

    assert rows[0]["final_layer"] is False
    assert rows[0]["final_layer_reason"] == "ridge_eave_no_roof_cover_support"


def test_clip_unpaired_ridge_run_supported_piece_to_actual_support_union() -> None:
    rows = clip_unpaired_ridge_run_supported_pieces_to_support_union(
        [
            {
                "uuid": "b-test",
                "story": 1,
                "target_element_id": (
                    "b-test::ridge-eave-candidate::plane-group::subject"
                ),
                "target_kind": "ridge_eave_plane_group",
                "piece_id": (
                    "b-test::ridge-eave-candidate::plane-group::subject#supported:0:0"
                ),
                "piece_role": "supported",
                "face_run_role": "ridge_run",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [6.0, 0.0, 4.0],
                    [0.0, 0.0, 4.0],
                ],
                "holes": [],
            },
            {
                "uuid": "b-test",
                "story": 1,
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 4.0],
                    [0.0, 0.0, 4.0],
                ],
                "holes": [],
            },
            {
                "uuid": "b-test",
                "story": 1,
                "target_element_id": "b-test::ceiling-oblique::ceiling-oblique:0",
                "target_kind": "candidate_oblique",
                "piece_id": "b-test::ceiling-oblique::ceiling-oblique:0#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [4.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [6.0, 0.0, 4.0],
                    [4.0, 0.0, 4.0],
                ],
                "holes": [],
            },
        ]
    )

    subject_rows = [
        row
        for row in rows
        if row["target_element_id"]
        == "b-test::ridge-eave-candidate::plane-group::subject"
    ]
    assert len(subject_rows) == 2
    by_piece = {row["piece_id"]: row for row in subject_rows}
    assert math.isclose(
        by_piece["b-test::ridge-eave-candidate::plane-group::subject#supported:0:0"][
            "area_xz_m2"
        ],
        8.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        by_piece["b-test::ridge-eave-candidate::plane-group::subject#supported:0:0:1"][
            "area_xz_m2"
        ],
        8.0,
        abs_tol=1e-6,
    )
    assert by_piece["b-test::ridge-eave-candidate::plane-group::subject#supported:0:0"][
        "support_clip_target_ids"
    ] == [
        "b-test::ceiling-oblique::ceiling-oblique:0",
        "b-test::roof-oblique::oblique:0",
    ]


def test_classify_final_demotes_redundant_piece_when_disjoint_pattern_is_not_strict():
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "local_competitor_loss_fraction": 0.0,
                "ownership_redundant": True,
                "creator_source_room_ids": ["room:6"],
                "crossed_room_ids": ["room:7", "room:9", "room:10"],
                "creator_source_room_count": 2,
                "provenance_relevance_flag": "normal",
                "provenance_relevance_reasons": [],
                "through_ratio": 0.9,
            }
        ]
    )

    assert rows[0]["final_layer"] is False
    assert rows[0]["final_layer_reason"] == "ridge_eave_competitor_loss"


def test_classify_final_demotes_small_mirror_sliver() -> None:
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 6.0,
                "through_ratio": 0.2,
                "local_competitor_loss_fraction": 0.0,
                "mirror_support_score": 0.9,
                "roof_surface_cover_fraction": 0.2,
            },
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0:1",
                "piece_role": "supported",
                "area_xz_m2": 0.7,
                "through_ratio": 1.2,
                "local_competitor_loss_fraction": 0.0,
                "mirror_support_score": 0.9,
                "roof_surface_cover_fraction": 0.2,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:ridge#supported:0:0"]["final_layer"] is True
    assert by_piece["t:ridge#supported:0:0:1"]["final_layer"] is False
    assert (
        by_piece["t:ridge#supported:0:0:1"]["final_layer_reason"]
        == "ridge_eave_mirror_sliver"
    )


def test_classify_final_demotes_unpaired_lower_story_suspect_slice():
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 7.0,
                "local_competitor_loss_fraction": 0.0,
                "roof_surface_cover_fraction": 0.2,
                "provenance_relevance_flag": "suspect_interior_slice",
                "provenance_relevance_reasons": [
                    "weak_creator_rain_area",
                    "covered_creators_dominate",
                    "cuts_below_top_story",
                    "unpaired",
                ],
                "creator_source_room_count": 1,
                "mirror_support_score": None,
            },
            {
                "target_element_id": "t:ridge:anchored",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge:anchored#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 7.0,
                "local_competitor_loss_fraction": 0.0,
                "roof_surface_cover_fraction": 0.2,
                "provenance_relevance_flag": "suspect_interior_slice",
                "provenance_relevance_reasons": [
                    "weak_creator_rain_area",
                    "covered_creators_dominate",
                    "cuts_below_top_story",
                    "unpaired",
                ],
                "creator_source_room_count": 1,
                "mirror_support_score": 0.95,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:ridge#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:ridge#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_suspect_interior_slice"
    )
    assert by_piece["t:ridge:anchored#supported:0:0"]["final_layer"] is True
    assert (
        by_piece["t:ridge:anchored#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_local_ownership"
    )


def test_classify_final_demotes_creator_disconnected_continuation():
    rows = classify_split_piece_final_layer(
        [
            {
                "target_element_id": "t:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 7.0,
                "local_competitor_loss_fraction": 0.0,
                "roof_surface_cover_fraction": 0.2,
                "creator_touch_room_ids": ["room:0", "room:1"],
                "crossed_room_ids": ["room:6", "room:7"],
                "provenance_relevance_reasons": [
                    "covered_creators_dominate",
                    "mostly_extended",
                    "cuts_below_top_story",
                ],
                "through_ratio": 1.4,
                "mirror_support_score": 0.95,
            },
            {
                "target_element_id": "t:ridge:local",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "t:ridge:local#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 7.0,
                "local_competitor_loss_fraction": 0.0,
                "roof_surface_cover_fraction": 0.2,
                "creator_touch_room_ids": ["room:0", "room:1"],
                "crossed_room_ids": ["room:1", "room:7"],
                "provenance_relevance_reasons": [
                    "covered_creators_dominate",
                    "mostly_extended",
                    "cuts_below_top_story",
                ],
                "through_ratio": 1.4,
                "mirror_support_score": 0.95,
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["t:ridge#supported:0:0"]["final_layer"] is False
    assert (
        by_piece["t:ridge#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_creator_disconnected"
    )
    assert by_piece["t:ridge:local#supported:0:0"]["final_layer"] is True
    assert (
        by_piece["t:ridge:local#supported:0:0"]["final_layer_reason"]
        == "ridge_eave_local_ownership"
    )


def test_source_room_keys_from_ridge_diagnostics_resolve_story_and_index() -> None:
    building = {
        "rooms": [
            {"story": 0},
            {"story": 0},
            {"story": 1},
        ]
    }
    diagnostics = {
        "target:a": {
            "creator_source_room_ids": ["room:1", "room:2", "room:not-an-int", "foo"],
        },
        "target:b": {
            "creator_source_room_ids": ["room:99"],
        },
    }

    keys = _source_room_keys_from_ridge_diagnostics(building, diagnostics)

    assert keys == {(0, 1), (1, 2)}


def test_facade_continuity_promotes_adjacent_chain_from_same_raw_plane() -> None:
    chain_a = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="chain:a",
        edge_count=1,
        total_length_m=2.0,
        azimuth_deg=90.0,
        y_mean=3.0,
        start_xz=(0.0, 0.0),
        end_xz=(2.0, 0.0),
        line_xz=LineString([(0.0, 0.0), (2.0, 0.0)]),
        member_plane_ids=("b-test::ceiling-raw::0:2:1",),
    )
    chain_b = EaveChainRecord(
        uuid="b-test",
        story=0,
        chain_id="chain:b",
        edge_count=1,
        total_length_m=2.0,
        azimuth_deg=89.5,
        y_mean=3.02,
        start_xz=(2.05, 0.0),
        end_xz=(4.0, 0.0),
        line_xz=LineString([(2.05, 0.0), (4.0, 0.0)]),
        member_plane_ids=("b-test::ceiling-raw::0:2:1",),
    )
    supports = [
        PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id="target:a",
            target_kind="committed_oblique",
            chain_id="chain:a",
            chain_azimuth_deg=90.0,
            ridge_azimuth_deg=90.0,
            angle_delta_deg=0.0,
            boundary_distance_m=0.1,
            overlap_fraction=1.0,
            height_residual_m=0.0,
            support_score=0.95,
            supported=True,
            chain_length_m=2.0,
        ),
        PlaneEaveChainSupportRecord(
            uuid="b-test",
            story=0,
            target_element_id="target:a",
            target_kind="committed_oblique",
            chain_id="chain:b",
            chain_azimuth_deg=89.5,
            ridge_azimuth_deg=90.0,
            angle_delta_deg=0.5,
            boundary_distance_m=0.8,
            overlap_fraction=0.0,
            height_residual_m=0.0,
            support_score=0.55,
            supported=False,
            chain_length_m=2.0,
        ),
    ]

    expanded = expand_plane_eave_chain_supports_by_facade_continuity(
        [chain_a, chain_b], supports
    )
    by_chain = {support.chain_id: support for support in expanded}

    assert by_chain["chain:a"].supported is True
    assert by_chain["chain:b"].supported is True
    assert math.isclose(by_chain["chain:b"].support_score, 0.55, abs_tol=1e-9)


def test_precedence_marks_lower_priority_supported_piece_redundant() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:final",
                "target_kind": "committed_oblique",
                "piece_id": "target:final#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:coarse",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "target:coarse#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:final#supported:0:0"]["ownership_redundant"] is False
    assert math.isclose(
        by_piece["target:final#supported:0:0"]["higher_priority_cover_fraction"],
        0.0,
        abs_tol=1e-9,
    )
    assert by_piece["target:coarse#supported:0:0"]["ownership_redundant"] is True
    assert math.isclose(
        by_piece["target:coarse#supported:0:0"]["higher_priority_cover_fraction"],
        1.0,
        abs_tol=1e-9,
    )
    assert by_piece["target:coarse#supported:0:0"][
        "higher_priority_covering_target_ids"
    ] == ["target:final"]
    assert math.isclose(
        by_piece["target:coarse#supported:0:0"]["committed_cover_fraction"],
        1.0,
        abs_tol=1e-9,
    )
    assert by_piece["target:coarse#supported:0:0"]["committed_covering_target_ids"] == [
        "target:final"
    ]


def test_precedence_marks_weaker_same_priority_peer_redundant() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:strong",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "target:strong#supported:0:0",
                "piece_role": "supported",
                "support_score": 0.95,
                "local_competitor_loss_fraction": 0.2,
                "best_supported_chain_height_residual_m": 0.1,
                "chain_ids": ["chain:a"],
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:weak",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "target:weak#supported:0:0",
                "piece_role": "supported",
                "support_score": 0.7,
                "local_competitor_loss_fraction": 0.8,
                "best_supported_chain_height_residual_m": 4.0,
                "chain_ids": ["chain:a"],
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:strong#supported:0:0"]["ownership_redundant"] is False
    assert by_piece["target:weak#supported:0:0"]["ownership_redundant"] is True
    assert by_piece["target:weak#supported:0:0"][
        "higher_priority_covering_target_ids"
    ] == ["target:strong"]


def test_precedence_does_not_mark_opposite_facing_ridge_piece_redundant() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:extension:a",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:extension:a#supported:0:0",
                "piece_role": "supported",
                "support_score": 0.95,
                "local_competitor_loss_fraction": 0.0,
                "best_supported_chain_height_residual_m": 0.1,
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:extension:b",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 0.0,
                "target_inclination_deg": 44.0,
                "piece_id": "target:extension:b#supported:0:0",
                "piece_role": "supported",
                "support_score": 0.8,
                "local_competitor_loss_fraction": 0.0,
                "best_supported_chain_height_residual_m": 0.1,
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:extension:a#supported:0:0"]["ownership_redundant"] is False
    assert by_piece["target:extension:b#supported:0:0"]["ownership_redundant"] is False
    assert (
        by_piece["target:extension:b#supported:0:0"]["higher_priority_cover_fraction"]
        == 0.0
    )


def test_precedence_keeps_different_signature_ridge_eave_pieces_non_redundant() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:run:a",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:run:a#supported:0:0",
                "piece_role": "supported",
                "support_score": 0.95,
                "local_competitor_loss_fraction": 0.0,
                "best_supported_chain_height_residual_m": 0.1,
                "chain_ids": ["chain:a"],
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:run:b",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:run:b#supported:0:0",
                "piece_role": "supported",
                "support_score": 0.7,
                "local_competitor_loss_fraction": 0.0,
                "best_supported_chain_height_residual_m": 0.1,
                "chain_ids": ["chain:b"],
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:run:a#supported:0:0"]["ownership_redundant"] is False
    assert by_piece["target:run:b#supported:0:0"]["ownership_redundant"] is False
    assert (
        by_piece["target:run:b#supported:0:0"]["higher_priority_cover_fraction"] == 0.0
    )


def test_redundant_committed_oblique_covers_most_of_it() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:committed",
                "target_kind": "committed_oblique",
                "piece_id": "target:committed#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [9.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:ridge#supported:0:0"]["ownership_redundant"] is True
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["committed_cover_fraction"],
        1.0,
        abs_tol=1e-9,
    )
    assert by_piece["target:ridge#supported:0:0"]["committed_covering_target_ids"] == [
        "target:committed"
    ]


def test_no_redundant_committed_oblique_only_partially_covers_it() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:committed",
                "target_kind": "committed_oblique",
                "piece_id": "target:committed#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [7.0, 0.0, 0.0],
                    [7.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:ridge#supported:0:0"]["ownership_redundant"] is False
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["committed_cover_fraction"],
        0.7,
        abs_tol=1e-9,
    )
    assert by_piece["target:ridge#supported:0:0"]["committed_covering_target_ids"] == [
        "target:committed"
    ]


def test_redundant_local_roof_cover_is_nearly_total() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:committed",
                "target_kind": "committed_oblique",
                "target_azimuth_deg": 230.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:committed#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [8.3, 0.0, 0.0],
                    [8.3, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": (
                    "b-test::ridge-eave-candidate::plane-group::mirror"
                ),
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 230.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:mirror#supported:0:0",
                "piece_role": "supported",
                "chain_ids": ["chain:a"],
                "mirror_partner_plane_group_id": "b-test::plane-group::ridge",
                "corners": [
                    [8.25, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [8.25, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "b-test::ridge-eave-candidate::plane-group::ridge",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 50.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "chain_ids": ["chain:a"],
                "mirror_partner_plane_group_id": "b-test::plane-group::mirror",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:ridge#supported:0:0"]["ownership_redundant"] is True
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["committed_cover_fraction"],
        0.83,
        abs_tol=1e-9,
    )
    assert by_piece["target:ridge#supported:0:0"]["local_roof_cover_fraction"] >= 0.98
    assert by_piece["target:ridge#supported:0:0"]["local_roof_covering_target_ids"] == [
        "target:committed",
        "b-test::ridge-eave-candidate::plane-group::mirror",
    ]


def test_redundant_supported_roof_surfaces_cover_most_of_it() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:committed",
                "target_kind": "committed_oblique",
                "piece_id": "target:committed#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [4.3, 0.0, 0.0],
                    [4.3, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:candidate",
                "target_kind": "candidate_oblique",
                "piece_id": "target:candidate#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [4.3, 0.0, 0.0],
                    [8.7, 0.0, 0.0],
                    [8.7, 0.0, 2.0],
                    [4.3, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:ridge",
                "target_kind": "ridge_eave_plane_group",
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:ridge#supported:0:0"]["ownership_redundant"] is True
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["committed_cover_fraction"],
        0.43,
        abs_tol=1e-9,
    )
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["roof_surface_cover_fraction"],
        0.87,
        abs_tol=1e-9,
    )
    assert by_piece["target:ridge#supported:0:0"][
        "roof_surface_covering_target_ids"
    ] == [
        "target:candidate",
        "target:committed",
    ]


def test_redundant_unpaired_same_side_superset_run_covers_it() -> None:
    rows = annotate_split_piece_rows_with_precedence(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": (
                    "b-test::ridge-eave-candidate::plane-group::coverer"
                ),
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 264.0,
                "target_inclination_deg": 11.0,
                "support_score": 0.99,
                "piece_id": "target:coverer#supported:0:0",
                "piece_role": "supported",
                "chain_ids": ["chain:a", "chain:b", "chain:c"],
                "corners": [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "b-test::ridge-eave-candidate::plane-group::ridge",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 261.0,
                "target_inclination_deg": 25.0,
                "support_score": 0.85,
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "chain_ids": ["chain:a", "chain:b"],
                "mirror_partner_plane_group_id": "b-test::plane-group::mirror",
                "corners": [
                    [0.0, 0.0, 0.0],
                    [9.9, 0.0, 0.0],
                    [9.9, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert by_piece["target:ridge#supported:0:0"]["ownership_redundant"] is True
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["same_side_superset_cover_fraction"],
        1.0,
        abs_tol=1e-9,
    )
    assert by_piece["target:ridge#supported:0:0"][
        "same_side_superset_covering_target_ids"
    ] == ["b-test::ridge-eave-candidate::plane-group::coverer"]


def test_annotate_committed_supported_piece_marks_zero_part_overlap_as_misaligned() -> (
    None
):
    rows = annotate_committed_supported_pieces_with_hypothesis_part_overlap(
        [
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "b-test::roof-oblique::oblique:0",
                "target_kind": "committed_oblique",
                "piece_id": "b-test::roof-oblique::oblique:0#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [4.0, 2.0, 0.0],
                    [6.0, 2.0, 0.0],
                    [6.0, 3.0, 2.0],
                    [4.0, 3.0, 2.0],
                ],
                "holes": [],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "b-test::roof-oblique::oblique:1",
                "target_kind": "committed_oblique",
                "piece_id": "b-test::roof-oblique::oblique:1#supported:0:0",
                "piece_role": "supported",
                "corners": [
                    [0.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
                "holes": [],
            },
        ],
        buildings_by_uuid={
            "b-test": {
                "uuid": "b-test",
                "rooms": [
                    {
                        "floor_polygon": [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 0.0, 2.0],
                            [0.0, 0.0, 2.0],
                        ],
                    },
                    {
                        "floor_polygon": [
                            [4.0, 0.0, 0.0],
                            [6.0, 0.0, 0.0],
                            [6.0, 0.0, 2.0],
                            [4.0, 0.0, 2.0],
                        ],
                    },
                ],
            }
        },
        roof_results_by_uuid={
            "b-test": {
                "roof_surfaces": {
                    "oblique": [
                        {"roof_hypothesis_id": "roof-hypothesis:oblique:0"},
                        {"roof_hypothesis_id": "roof-hypothesis:oblique:1"},
                    ]
                },
                "building_part_graph": {
                    "room_membership": {
                        "room:0": ["part:left"],
                        "room:1": ["part:right"],
                    },
                    "hypothesis_membership": {
                        "roof-hypothesis:oblique:0": ["part:left"],
                        "roof-hypothesis:oblique:1": ["part:left"],
                    },
                },
            }
        },
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:0:0"][
            "hypothesis_part_overlap_fraction"
        ]
        == 0.0
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:0#supported:0:0"][
            "hypothesis_part_misaligned"
        ]
        is True
    )
    assert math.isclose(
        by_piece["b-test::roof-oblique::oblique:1#supported:0:0"][
            "hypothesis_part_overlap_fraction"
        ],
        1.0,
        abs_tol=1e-9,
    )
    assert (
        by_piece["b-test::roof-oblique::oblique:1#supported:0:0"][
            "hypothesis_part_misaligned"
        ]
        is False
    )


def test_merge_same_plane_committed_oblique_cores_keeps_only_ridge_tail() -> None:
    rows = merge_same_plane_committed_oblique_cores(
        [
            {
                "uuid": "b-test",
                "story": 1,
                "target_element_id": "target:committed",
                "target_kind": "committed_oblique",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:committed#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 8.0,
                "corners": [
                    [0.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
                "holes": [],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:ridge",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 16.0,
                "corners": [
                    [0.0, 2.0, 0.0],
                    [8.0, 2.0, 0.0],
                    [8.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
                "holes": [],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    ridge = by_piece["target:ridge#supported:0:0"]
    committed = by_piece["target:committed#supported:0:0"]
    assert math.isclose(ridge["area_xz_m2"], 8.0, abs_tol=1e-6)
    assert ridge["same_plane_committed_core_target_ids"] == ["target:committed"]
    assert math.isclose(ridge["same_plane_committed_core_fraction"], 0.5, abs_tol=1e-6)
    assert ridge["face_run_id"] == committed["face_run_id"]
    assert ridge["face_run_role"] == "ridge_continuation"
    assert committed["face_run_role"] == "committed_core"
    xs = sorted({round(corner[0], 6) for corner in ridge["corners"]})
    assert xs == [4.0, 8.0]


def test_merge_same_plane_committed_oblique_cores_skips_parallel_offset_rows() -> None:
    rows = merge_same_plane_committed_oblique_cores(
        [
            {
                "uuid": "b-test",
                "story": 1,
                "target_element_id": "target:committed",
                "target_kind": "committed_oblique",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:committed#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 8.0,
                "corners": [
                    [0.0, 4.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [4.0, 5.0, 2.0],
                    [0.0, 5.0, 2.0],
                ],
                "holes": [],
            },
            {
                "uuid": "b-test",
                "story": 0,
                "target_element_id": "target:ridge",
                "target_kind": "ridge_eave_plane_group",
                "target_azimuth_deg": 180.0,
                "target_inclination_deg": 45.0,
                "piece_id": "target:ridge#supported:0:0",
                "piece_role": "supported",
                "area_xz_m2": 16.0,
                "corners": [
                    [0.0, 2.0, 0.0],
                    [8.0, 2.0, 0.0],
                    [8.0, 3.0, 2.0],
                    [0.0, 3.0, 2.0],
                ],
                "holes": [],
            },
        ]
    )

    by_piece = {row["piece_id"]: row for row in rows}
    assert math.isclose(
        by_piece["target:ridge#supported:0:0"]["area_xz_m2"], 16.0, abs_tol=1e-6
    )
    assert by_piece["target:ridge#supported:0:0"].get(
        "same_plane_committed_core_target_ids"
    ) in (None, [])


def test_diag_chain_owner_uses_same_signature_class() -> None:
    building = {
        "uuid": "b-test",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            },
        ],
    }
    subject_target = TargetPlaneRecord(
        uuid="b-test",
        story=0,
        target_kind="ridge_eave_plane_group",
        target_index=0,
        element_id="b-test::ridge-eave-candidate::plane-group::subject",
        poly_xz=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        normal=_make_target().normal,
        azimuth_deg=180.0,
        inclination_deg=45.0,
        ridge_dir_xz=(1.0, 0.0),
        area_xz_m2=8.0,
        plane_point=(2.0, 2.0, 1.0),
    )
    same_signature_target = replace(
        subject_target,
        element_id="b-test::ridge-eave-candidate::plane-group::same",
        target_index=1,
    )
    different_signature_target = replace(
        subject_target,
        element_id="b-test::ridge-eave-candidate::plane-group::different",
        target_index=2,
    )
    piece = TargetSplitPieceRecord(
        uuid="b-test",
        story=0,
        target_element_id=subject_target.element_id,
        target_kind="ridge_eave_plane_group",
        piece_id="b-test::ridge-eave-candidate::plane-group::subject#supported:0:0",
        piece_index=0,
        piece_role="supported",
        area_xz_m2=8.0,
        support_score=0.6,
        chain_ids=("chain:a",),
        corners=[[0.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 2.0, 2.0], [0.0, 2.0, 2.0]],
        holes=[],
    )
    same_signature_piece = replace(
        piece,
        target_element_id=same_signature_target.element_id,
        piece_id="b-test::ridge-eave-candidate::plane-group::same#supported:0:0",
        support_score=0.95,
    )
    different_signature_piece = replace(
        piece,
        target_element_id=different_signature_target.element_id,
        piece_id="b-test::ridge-eave-candidate::plane-group::different#supported:0:0",
        support_score=0.99,
        chain_ids=("chain:b",),
    )
    support = score_plane_eave_chain_supports(
        [subject_target],
        [
            EaveChainRecord(
                uuid="b-test",
                story=0,
                chain_id="chain:a",
                edge_count=1,
                total_length_m=4.0,
                azimuth_deg=90.0,
                y_mean=0.0,
                start_xz=(0.0, 0.0),
                end_xz=(4.0, 0.0),
                line_xz=LineString([(0.0, 0.0), (4.0, 0.0)]),
                member_plane_ids=("b-test::ceiling-raw::0:0:0",),
            )
        ],
    )[0]

    row = diagnose_ridge_eave_supported_piece_ownership(
        building,
        piece,
        targets_by_id={
            subject_target.element_id: subject_target,
            same_signature_target.element_id: same_signature_target,
            different_signature_target.element_id: different_signature_target,
        },
        target_scores_by_id={
            subject_target.element_id: {
                "element_id": subject_target.element_id,
                "retention_support_score": 0.6,
            },
            same_signature_target.element_id: {
                "element_id": same_signature_target.element_id,
                "retention_support_score": 0.95,
            },
            different_signature_target.element_id: {
                "element_id": different_signature_target.element_id,
                "retention_support_score": 0.99,
            },
        },
        supported_chain_by_target={subject_target.element_id: support},
        supported_ridge_eave_pieces_by_signature={
            (0, ("chain:a",)): [piece, same_signature_piece],
            (0, ("chain:b",)): [different_signature_piece],
        },
        ridge_eave_meta_by_target={subject_target.element_id: {}},
        ridge_eave_target_diagnostics={subject_target.element_id: {}},
    )

    assert row is not None
    assert row["local_competitor_loss_room_count"] == 1
    assert row["local_top_competitor_ids"] == [same_signature_target.element_id]
    assert row["local_top_competitor_piece_ids"] == [same_signature_piece.piece_id]
    assert row["chain_signature"] == ["chain:a"]
    assert row["local_signature_competitor_target_ids"] == [
        same_signature_target.element_id,
        subject_target.element_id,
    ]
