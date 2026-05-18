from __future__ import annotations

import math

from shapely.geometry import Polygon

from reconcile_tiers.polyhedron import cell_selector as cs
from reconcile_tiers.polyhedron import payload_adapter as pa
from reconcile_tiers.polyhedron.cell_selector import (
    ROOM_COVERED_RATIO,
    audit_payload_rooms,
    select_payload_cells,
    select_payload_cells_v2,
)
from reconcile_tiers.polyhedron.face_selection import FaceSelectionResult
from reconcile_tiers.polyhedron.payload_adapter import (
    build_envelope_polyhedra_from_tier_payload,
    payload_faces_from_tier_payload,
)


def _pt(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _rect(
    x0: float,
    y: float,
    z0: float,
    x1: float,
    z1: float,
) -> list[dict[str, float]]:
    return [
        _pt(x0, y, z0),
        _pt(x1, y, z0),
        _pt(x1, y, z1),
        _pt(x0, y, z1),
    ]


def _payload_with_rooms(room_count: int = 3) -> dict:
    payload = {"rooms": [], "ceiling": []}
    for room_index in range(room_count):
        x0 = float(room_index * 2)
        x1 = x0 + 2.0
        payload["rooms"].append(
            {
                "story": 0,
                "locator_id": f"building::tier-room::{room_index}",
                "floor": [{"corners": _rect(x0, 0.0, 0.0, x1, 2.0), "holes": []}],
                "walls": [],
                "doors": [],
                "windows": [],
            }
        )
    return payload


def _footprint(payload: dict) -> Polygon:
    polys = []
    for room in payload["rooms"]:
        corners = room["floor"][0]["corners"]
        polys.append(Polygon([(p["x"], p["z"]) for p in corners]))
    return Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)])


def _add_gable(payload: dict, *, weak_middle: bool = False) -> None:
    # Left slope y = 2 + 0.5x, right slope y = 5 - 0.5x, ridge at x=3.
    left_ridge_x = 2.0 if weak_middle else 3.0
    payload["ceiling"].extend(
        [
            {
                "locator_id": "building::tier-ceiling-oblique::left-main",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(left_ridge_x, 2.0 + 0.5 * left_ridge_x, 2.0),
                    _pt(left_ridge_x, 2.0 + 0.5 * left_ridge_x, 0.0),
                ],
                "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "computed_oblique",
            },
            {
                "locator_id": "building::tier-ceiling-oblique::right-main",
                "corners": [
                    _pt(3.0, 3.5, 0.0),
                    _pt(3.0, 3.5, 2.0),
                    _pt(6.0, 2.0, 2.0),
                    _pt(6.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.5, "b": 1.0, "c": 0.0, "d": 5.0},
                "source": "computed_oblique",
            },
        ]
    )
    if weak_middle:
        payload["ceiling"].append(
            {
                "locator_id": "building::tier-ceiling-oblique::left-weak-middle",
                "corners": [
                    _pt(2.0, 3.0, 0.0),
                    _pt(2.0, 3.0, 2.0),
                    _pt(2.2, 3.1, 2.0),
                    _pt(2.2, 3.1, 0.0),
                ],
                "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "computed_oblique",
            }
        )


def test_weak_room_oblique_support_promoted_by_part_plane():
    payload = _payload_with_rooms()
    _add_gable(payload, weak_middle=True)
    faces = payload_faces_from_tier_payload(payload)
    ceiling_faces = [face for face in faces if face.kind == "ceiling"]

    result = select_payload_cells(
        payload,
        footprint=_footprint(payload),
        ceiling_faces=ceiling_faces,
        min_top_overlap_ratio=0.35,
    )

    domain_promoted = [
        selected
        for selected in result.selected_cells
        if 1 in selected.cell.source_room_indices
        and selected.top.reason == "domain_gable_pair"
    ]
    assert domain_promoted
    assert any(group.support_ratio >= 0.35 for group in result.plane_groups)
    audit = audit_payload_rooms(payload)
    middle_room = audit["rooms"][1]
    assert middle_room["selected_cell_debug"]
    assert middle_room["best_part_plane_groups"]
    assert "strict_candidate_coverage_ratio" in middle_room
    assert "build_attempts" in audit
    assert audit["build_attempts"]


def test_select_payload_cells_v2_emits_diagnostic_face_selection_trace():
    payload = _payload_with_rooms(room_count=1)
    payload["ceiling"].append(
        {
            "locator_id": "building::tier-ceiling-flat::main",
            "corners": _rect(0.0, 2.5, 0.0, 2.0, 2.0),
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.5},
            "source": "flat_ceiling",
        }
    )
    faces = payload_faces_from_tier_payload(payload)

    result = select_payload_cells_v2(
        payload,
        footprint=Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
    )

    assert result.selector == "new"
    assert result.candidates
    assert result.candidates[0].selector == "new"
    assert result.candidates[0].top_overlap_ratio == 1.0
    assert result.domain_traces
    assert any(trace["status"] == "ok" for trace in result.domain_traces)
    ok = next(trace for trace in result.domain_traces if trace["status"] == "ok")
    assert ok["candidate_count"] == 6
    assert ok["selection"]["selected_count"] == 6
    assert ok["selection"]["energy_breakdown"]["coverage_ratio"] == 1.0
    assert ok["emitted_candidate"] == "envelope-v2-domain:0"


def test_select_payload_cells_v2_uses_best_available_top_height_fallback():
    payload = _payload_with_rooms(room_count=1)
    payload["ceiling"].append(
        {
            "locator_id": "building::tier-ceiling-flat::small",
            "corners": _rect(0.0, 2.5, 0.0, 0.5, 0.5),
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.5},
            "source": "flat_ceiling",
        }
    )
    faces = payload_faces_from_tier_payload(payload)

    result = select_payload_cells_v2(
        payload,
        footprint=Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
    )

    ok = next(trace for trace in result.domain_traces if trace["status"] == "ok")
    assert ok["emitted_candidate"] == "envelope-v2-domain:0"
    assert ok["plane_groups"][0]["included"] is True
    assert ok["plane_groups"][0]["reason"] == "best_available_top_fallback"


def test_select_payload_cells_v2_emits_valid_oblique_selection():
    payload = _payload_with_rooms(room_count=1)
    payload["ceiling"].append(
        {
            "locator_id": "building::tier-ceiling-oblique::single",
            "corners": [
                _pt(0.0, 2.0, 0.0),
                _pt(2.0, 2.5, 0.0),
                _pt(2.0, 2.5, 2.0),
                _pt(0.0, 2.0, 2.0),
            ],
            "plane": {"a": -0.25, "b": 1.0, "c": 0.0, "d": 2.0},
            "source": "computed_oblique",
        }
    )
    faces = payload_faces_from_tier_payload(payload)

    result = select_payload_cells_v2(
        payload,
        footprint=Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
    )

    assert result.candidates
    candidate = result.candidates[0]
    assert candidate.selector == "new"
    assert candidate.top_overlap_ratio == 1.0
    assert "single-oblique" in candidate.top_source
    top_face = next(face for face in candidate.faces if face.kind == "ceiling")
    y_by_x = {round(corner[0], 1): round(corner[1], 1) for corner in top_face.corners}
    assert y_by_x == {0.0: 2.0, 2.0: 2.5}


def test_select_payload_cells_v2_rejects_low_coverage_emission():
    selection = FaceSelectionResult(
        selected=(object(),),  # type: ignore[arg-type]
        objective=0.0,
        energy_breakdown={"coverage_ratio": 0.5},
        solver_status="test",
        elapsed_seconds=0.0,
    )

    assert (
        cs._v2_envelope_candidate_from_selection(
            selection,
            domain_index=0,
            domain=Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
        )
        is None
    )


def test_flat_ceiling_under_gable_keeps_ceiling_and_roof_distinct():
    payload = _payload_with_rooms()
    _add_gable(payload)
    payload["ceiling"].append(
        {
            "locator_id": "building::tier-ceiling-flat::middle",
            "corners": _rect(2.0, 2.4, 0.0, 4.0, 2.0),
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.4},
            "source": "flat_ceiling",
        }
    )
    faces = payload_faces_from_tier_payload(payload)

    result = select_payload_cells(
        payload,
        footprint=_footprint(payload),
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
        min_top_overlap_ratio=0.35,
    )

    middle_tops = [
        selected.top.label
        for selected in result.selected_cells
        if 1 in selected.cell.source_room_indices
    ]
    assert "gable-pair" in middle_tops
    assert any(pair.support_ratio > 0.0 for pair in result.gable_pairs)
    assert any(
        group.label == "flat-ceiling"
        and any(locator.endswith("::middle") for locator in group.source_locators)
        for group in result.plane_groups
    )


def test_coherent_near_gate_gable_can_win_domain_choice():
    payload = _payload_with_rooms()
    # Two under-observed eave strips cover just under the normal 0.60 strict
    # support gate, but their planes define a coherent ridge over the domain.
    payload["ceiling"].extend(
        [
            {
                "locator_id": "building::tier-ceiling-oblique::left-partial",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(1.7, 2.85, 2.0),
                    _pt(1.7, 2.85, 0.0),
                ],
                "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "computed_oblique",
            },
            {
                "locator_id": "building::tier-ceiling-oblique::right-partial",
                "corners": [
                    _pt(4.3, 2.85, 0.0),
                    _pt(4.3, 2.85, 2.0),
                    _pt(6.0, 2.0, 2.0),
                    _pt(6.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.5, "b": 1.0, "c": 0.0, "d": 5.0},
                "source": "computed_oblique",
            },
            {
                "locator_id": "building::tier-ceiling-flat::competing",
                "corners": _rect(0.0, 2.4, 0.0, 3.55, 2.0),
                "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": 2.4},
                "source": "flat_ceiling",
            },
        ]
    )
    faces = payload_faces_from_tier_payload(payload)
    footprint = _footprint(payload)
    domains = [footprint]
    groups = cs._plane_groups_for_domains(
        payload,
        domains,
        [face for face in faces if face.kind == "ceiling"],
        corner_tol=0.02,
    )
    pairs = cs._gable_pairs_for_domains(domains, groups)

    choices = cs._select_domain_top_choices(
        domains,
        groups,
        pairs,
        room_polys=cs._room_polygons(payload, corner_tol=0.02),
    )

    assert pairs
    assert max(pair.support_ratio for pair in pairs) < cs.STRICT_TOP_SUPPORT_GATE
    assert choices
    assert choices[0].label == "gable-pair"
    assert choices[0].reason == "domain_coherent_gable_pair"


def test_local_flat_ceiling_partitions_under_generated_roof_plane():
    payload = _payload_with_rooms(room_count=1)
    flat = pa.PayloadFace(
        kind="ceiling",
        locator_id="building::raw-ceiling-plane::0:0",
        corners=[
            [0.0, 2.4, 0.0],
            [1.0, 2.4, 0.0],
            [1.0, 2.4, 2.0],
            [0.0, 2.4, 2.0],
        ],
        plane=pa.Plane(a=0.0, b=1.0, c=0.0, d=2.4),
        source="raw_observed_ceiling_plane",
        room_index=0,
        story=0,
    )
    roof = pa.PayloadFace(
        kind="ceiling",
        locator_id="building::tier-ceiling-roof-arrangement-attic-full::0",
        corners=[
            [0.0, 2.7, 0.0],
            [2.0, 3.1, 0.0],
            [2.0, 3.1, 2.0],
            [0.0, 2.7, 2.0],
        ],
        plane=pa.Plane(a=-0.2, b=1.0, c=0.0, d=2.7),
        source="roof_arrangement_attic",
        room_index=None,
        story=0,
    )

    result = select_payload_cells(
        payload,
        footprint=Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
        ceiling_faces=[flat, roof],
        min_top_overlap_ratio=0.35,
    )

    assert result.selected_cells
    assert {selected.top.label for selected in result.selected_cells} == {
        "flat-ceiling",
        "single-oblique",
    }
    assert any(
        selected.top.label == "flat-ceiling"
        and selected.top.reason == "domain_flat-ceiling"
        for selected in result.selected_cells
    )


def test_merge_compatible_single_side_cells_into_gable_component():
    left_face = pa.PayloadFace(
        kind="ceiling",
        locator_id="left",
        corners=[[0.0, 2.0, 0.0], [1.0, 2.5, 0.0], [1.0, 2.5, 2.0]],
        plane=pa.Plane(a=-0.5, b=1.0, c=0.0, d=2.0),
    )
    right_face = pa.PayloadFace(
        kind="ceiling",
        locator_id="right",
        corners=[[1.0, 2.5, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 2.0]],
        plane=pa.Plane(a=0.5, b=1.0, c=0.0, d=3.0),
    )
    gable_top = cs.SelectedTop(
        label="gable-pair",
        signature=("gable-pair", ("left-key",), ("right-key",)),
        faces=(left_face, right_face),
        score=10.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="domain_gable_pair",
    )
    single_top = cs.SelectedTop(
        label="single-oblique",
        signature=("single-oblique", ("left-key",)),
        faces=(left_face,),
        score=5.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="domain_single-oblique",
    )
    selected = [
        cs.SelectedCell(
            cell=cs.PlanCell(
                cell_id="left",
                story=0,
                polygon=Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)]),
                source_room_indices=(0,),
                floor_y=0.0,
            ),
            top=gable_top,
        ),
        cs.SelectedCell(
            cell=cs.PlanCell(
                cell_id="right",
                story=0,
                polygon=Polygon([(1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 2.0)]),
                source_room_indices=(1,),
                floor_y=0.0,
            ),
            top=single_top,
        ),
    ]

    assert cs._merge_selected_cells(selected) == [[0, 1]]
    assert cs._representative_top_for_selected_component(selected).label == "gable-pair"


def test_compatible_component_union_closes_tolerance_gap():
    face = pa.PayloadFace(
        kind="ceiling",
        locator_id="flat",
        corners=[[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 2.0]],
        plane=pa.Plane(a=0.0, b=1.0, c=0.0, d=2.0),
    )
    top = cs.SelectedTop(
        label="flat-ceiling",
        signature=("flat-ceiling", (0.0, 1.0, 0.0, -2.0)),
        faces=(face,),
        score=1.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="candidate_covered",
    )
    selected = [
        cs.SelectedCell(
            cell=cs.PlanCell(
                cell_id="left",
                story=0,
                polygon=Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)]),
                source_room_indices=(0,),
                floor_y=0.0,
            ),
            top=top,
        ),
        cs.SelectedCell(
            cell=cs.PlanCell(
                cell_id="right",
                story=0,
                polygon=Polygon([(1.04, 0.0), (2.04, 0.0), (2.04, 2.0), (1.04, 2.0)]),
                source_room_indices=(1,),
                floor_y=0.0,
            ),
            top=top,
        ),
    ]

    merged = cs._union_selected_component_polygons(selected)

    assert len(cs._polygon_components(merged)) == 1
    assert math.isclose(merged.area, 4.08, rel_tol=0.03)


def test_large_room_keeps_roof_plane_over_partial_flat_ceiling():
    payload = _payload_with_rooms(room_count=1)
    payload["rooms"][0]["floor"][0]["corners"] = _rect(0.0, 0.0, 0.0, 5.0, 2.0)
    flat = pa.PayloadFace(
        kind="ceiling",
        locator_id="building::raw-ceiling-plane::0:0",
        corners=[
            [0.0, 2.4, 0.0],
            [2.0, 2.4, 0.0],
            [2.0, 2.4, 2.0],
            [0.0, 2.4, 2.0],
        ],
        plane=pa.Plane(a=0.0, b=1.0, c=0.0, d=2.4),
        source="raw_observed_ceiling_plane",
        room_index=0,
        story=0,
    )
    roof = pa.PayloadFace(
        kind="ceiling",
        locator_id="building::tier-ceiling-roof-arrangement-attic-full::0",
        corners=[
            [0.0, 2.7, 0.0],
            [5.0, 3.7, 0.0],
            [5.0, 3.7, 2.0],
            [0.0, 2.7, 2.0],
        ],
        plane=pa.Plane(a=-0.2, b=1.0, c=0.0, d=2.7),
        source="roof_arrangement_attic",
        room_index=0,
        story=0,
    )

    result = select_payload_cells(
        payload,
        footprint=Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)]),
        ceiling_faces=[flat, roof],
        min_top_overlap_ratio=0.35,
    )

    assert result.selected_cells
    assert any(
        selected.top.label == "single-oblique"
        for selected in result.selected_cells
    )


def test_multi_room_gable_builds_parts_and_preserves_room_coverage():
    payload = _payload_with_rooms()
    _add_gable(payload)

    built = build_envelope_polyhedra_from_tier_payload(
        payload,
        wing_level=True,
        min_top_overlap_ratio=0.35,
        coord_tol=1e-6,
    )

    assert built
    assert all(candidate.selector == "cell-selector" for candidate, _poly in built)
    coverage = built[0][0].room_coverage
    assert coverage is not None
    assert coverage["rooms_ge80"] >= 2
    assert coverage["rooms_ge80"] / coverage["rooms_total"] >= ROOM_COVERED_RATIO - 0.15


def test_gable_build_attempt_records_footprint_coherence():
    payload = _payload_with_rooms()
    _add_gable(payload)
    faces = payload_faces_from_tier_payload(payload)

    result = select_payload_cells(
        payload,
        footprint=_footprint(payload),
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
        min_top_overlap_ratio=0.35,
    )

    diagnostic = next(
        attempt["gable_footprint_coherence"]
        for attempt in result.build_attempts
        if attempt.get("top_label") == "gable-pair"
    )
    assert diagnostic["status"] == "ok"
    assert diagnostic["uses_both_roof_faces"] is True
    assert diagnostic["split_region_count"] == 2
    assert diagnostic["fragmented_side_count"] == 0
    assert diagnostic["side_area_balance"] > 0.95
    assert diagnostic["covered_area_ratio"] > 0.99
    assert diagnostic["local_width_samples"]["sample_count"] > 0


def test_gable_footprint_coherence_exposes_local_width_field_for_notched_part():
    polygon = Polygon(
        [
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 1.0),
            (4.0, 1.0),
            (4.0, 3.0),
            (6.0, 3.0),
            (6.0, 4.0),
            (0.0, 4.0),
            (0.0, 3.0),
            (2.0, 3.0),
            (2.0, 1.0),
            (0.0, 1.0),
        ]
    )
    left = pa.PayloadFace(
        kind="ceiling",
        locator_id="left",
        corners=[],
        plane=pa.Plane(a=-0.5, b=1.0, c=0.0, d=2.0),
    )
    right = pa.PayloadFace(
        kind="ceiling",
        locator_id="right",
        corners=[],
        plane=pa.Plane(a=0.5, b=1.0, c=0.0, d=5.0),
    )
    top = cs.SelectedTop(
        label="gable-pair",
        signature=("gable-pair",),
        faces=(left, right),
        score=1.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="candidate_covered",
    )

    diagnostic = cs._gable_footprint_coherence_json(polygon, top)

    assert diagnostic is not None
    assert diagnostic["status"] == "ok"
    assert diagnostic["uses_both_roof_faces"] is True
    assert diagnostic["split_region_count"] == 2
    widths = diagnostic["local_width_samples"]
    assert widths["sample_count"] > 0
    assert widths["along_ridge_m"]["min"] < widths["along_ridge_m"]["max"]
    assert widths["across_ridge_m"]["min"] < widths["across_ridge_m"]["max"]


def test_missing_top_support_is_audited_as_no_top_support():
    payload = _payload_with_rooms(room_count=1)

    audit = audit_payload_rooms(payload)

    assert audit["summary"]["dropped_rooms"] == 1
    assert audit["rooms"][0]["reason"] == "no_top_support"


def test_overextended_gable_support_is_clipped_to_owned_ridge_side():
    payload = _payload_with_rooms(room_count=2)
    payload["ceiling"].extend(
        [
            {
                "locator_id": "building::tier-ceiling-oblique-room::0:gable0_0",
                "corners": [
                    _pt(0.0, 2.0, 0.0),
                    _pt(0.0, 2.0, 2.0),
                    _pt(4.0, 4.0, 2.0),
                    _pt(4.0, 4.0, 0.0),
                ],
                "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
                "source": "computed_oblique",
            },
            {
                "locator_id": "building::tier-ceiling-oblique-room::1:gable1_0",
                "corners": [
                    _pt(2.0, 3.0, 0.0),
                    _pt(2.0, 3.0, 2.0),
                    _pt(4.0, 2.0, 2.0),
                    _pt(4.0, 2.0, 0.0),
                ],
                "plane": {"a": 0.5, "b": 1.0, "c": 0.0, "d": 4.0},
                "source": "computed_oblique",
            },
        ]
    )
    faces = payload_faces_from_tier_payload(payload)

    result = select_payload_cells(
        payload,
        footprint=Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]),
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
        min_top_overlap_ratio=0.35,
    )

    left_group = next(
        group
        for group in result.plane_groups
        if group.representative.locator_id.endswith("gable0_0")
    )
    assert left_group.footprint.bounds[2] <= 2.0 + 1e-6


def test_small_lone_oblique_fragment_without_topology_is_not_selected():
    payload = _payload_with_rooms(room_count=1)
    payload["ceiling"].append(
        {
            "locator_id": "building::tier-ceiling-oblique-room::0:fragment",
            "corners": [
                _pt(0.0, 2.0, 0.0),
                _pt(0.0, 2.0, 2.0),
                _pt(0.5, 2.25, 2.0),
                _pt(0.5, 2.25, 0.0),
            ],
            "plane": {"a": -0.5, "b": 1.0, "c": 0.0, "d": 2.0},
            "source": "computed_oblique",
        }
    )

    audit = audit_payload_rooms(payload)

    assert audit["rooms"][0]["status"] == "dropped"
    assert audit["rooms"][0]["reason"] == "top_coverage_too_low"


def test_top_below_floor_is_not_selected():
    payload = _payload_with_rooms(room_count=1)
    payload["ceiling"].append(
        {
            "locator_id": "building::tier-ceiling-flat::below",
            "corners": _rect(0.0, -0.2, 0.0, 2.0, 2.0),
            "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": -0.2},
            "source": "flat_ceiling",
        }
    )

    audit = audit_payload_rooms(payload)

    assert audit["rooms"][0]["coverage_ratio"] < 0.5
    assert audit["rooms"][0]["status"] == "dropped"


def test_selected_part_cleanup_removes_precision_hairpins_for_strict_candidate():
    polygon = Polygon(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0 + 1e-12, 2.0),
            (0.0, 2.0),
        ]
    )
    top_face = pa.PayloadFace(
        kind="ceiling",
        locator_id="top",
        corners=[[0.0, 3.0, 0.0], [4.0, 3.0, 0.0], [4.0, 3.0, 2.0], [0.0, 3.0, 2.0]],
        plane=pa.Plane(a=0.0, b=1.0, c=0.0, d=3.0),
    )
    top = cs.SelectedTop(
        label="flat-ceiling",
        signature=("flat-ceiling", "top"),
        faces=(top_face,),
        score=1.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="candidate_covered",
    )

    candidate = cs._candidate_for_selected_part(
        polygon,
        top,
        floor_y=0.0,
        locator_id="hairpin",
        min_top_overlap_ratio=0.60,
    )

    assert candidate is not None
    pa.build_from_planar_polygons(
        [(face.corners, face.plane) for face in candidate.faces]
    )


def test_gable_pair_cell_on_one_side_collapses_to_single_plane_candidate():
    polygon = Polygon([(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)])
    left = pa.PayloadFace(
        kind="ceiling",
        locator_id="left",
        corners=[[1.0, 3.5, 0.0], [3.0, 4.5, 0.0], [3.0, 4.5, 2.0], [1.0, 3.5, 2.0]],
        plane=pa.Plane(a=-0.5, b=1.0, c=0.0, d=3.0),
    )
    right = pa.PayloadFace(
        kind="ceiling",
        locator_id="right",
        corners=[[1.0, 2.5, 0.0], [3.0, 1.5, 0.0], [3.0, 1.5, 2.0], [1.0, 2.5, 2.0]],
        plane=pa.Plane(a=0.5, b=1.0, c=0.0, d=3.0),
    )
    top = cs.SelectedTop(
        label="gable-pair",
        signature=("gable-pair",),
        faces=(left, right),
        score=1.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="candidate_covered",
    )

    candidate = cs._candidate_for_selected_part(
        polygon,
        top,
        floor_y=0.0,
        locator_id="single-side",
        min_top_overlap_ratio=0.60,
    )

    assert candidate is not None
    assert candidate.top_source.startswith("right")
    assert "single side of selected gable pair" in candidate.top_source
    pa.build_from_planar_polygons(
        [(face.corners, face.plane) for face in candidate.faces]
    )


def test_split_cleanup_merges_sub_meter_fragments_back_into_cells():
    original = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    pieces = [
        Polygon([(0.0, 0.0), (3.7, 0.0), (3.7, 2.0), (0.0, 2.0)]),
        Polygon([(3.7, 0.0), (4.0, 0.0), (4.0, 2.0), (3.7, 2.0)]),
    ]

    cleaned = cs._merge_tiny_split_fragments(original, pieces)

    assert len(cleaned) == 1
    assert math.isclose(cleaned[0].area, original.area)


def test_candidate_assembly_closes_tolerance_gaps_between_compatible_cells():
    payload = _payload_with_rooms(room_count=2)
    payload["rooms"][1]["floor"][0]["corners"] = _rect(2.05, 0.0, 0.0, 4.05, 2.0)
    top_face = pa.PayloadFace(
        kind="ceiling",
        locator_id="top",
        corners=[[0.0, 3.0, 0.0], [4.05, 3.0, 0.0], [4.05, 3.0, 2.0], [0.0, 3.0, 2.0]],
        plane=pa.Plane(a=0.0, b=1.0, c=0.0, d=3.0),
    )
    top = cs.SelectedTop(
        label="flat-ceiling",
        signature=("flat-ceiling", (0.0, 1.0, 0.0, 3.0)),
        faces=(top_face,),
        score=1.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="candidate_covered",
    )
    selected = [
        cs.SelectedCell(
            cell=cs.PlanCell(
                cell_id=f"room:{index}:cell:0",
                story=0,
                polygon=pa._polygon_xz(
                    pa._corners(payload["rooms"][index]["floor"][0])
                ),
                source_room_indices=(index,),
                floor_y=0.0,
            ),
            top=top,
        )
        for index in range(2)
    ]

    candidates, _attempts = cs._candidates_from_selected_cells(
        payload,
        selected,
        ceiling_faces=[top_face],
        min_top_overlap_ratio=0.60,
        corner_tol=0.02,
        plane_groups=[],
    )

    assert len(candidates) == 1
    assert candidates[0].room_coverage == {
        "rooms_total": 2,
        "rooms_ge80": 2,
        "rooms_ge50": 2,
    }


def test_candidate_floor_height_comes_from_selected_story_not_lowest_overlap():
    payload = _payload_with_rooms(room_count=2)
    payload["rooms"][1]["story"] = 1
    payload["rooms"][1]["floor"][0]["corners"] = _rect(0.0, 3.0, 0.0, 2.0, 2.0)
    top_face = pa.PayloadFace(
        kind="ceiling",
        locator_id="upper-top",
        corners=[[0.0, 5.0, 0.0], [2.0, 5.0, 0.0], [2.0, 5.0, 2.0], [0.0, 5.0, 2.0]],
        plane=pa.Plane(a=0.0, b=1.0, c=0.0, d=5.0),
    )
    top = cs.SelectedTop(
        label="flat-ceiling",
        signature=("flat-ceiling", (0.0, 1.0, 0.0, 5.0)),
        faces=(top_face,),
        score=1.0,
        local_coverage=1.0,
        part_support_ratio=1.0,
        reason="candidate_covered",
    )
    selected = [
        cs.SelectedCell(
            cell=cs.PlanCell(
                cell_id="room:1:cell:0",
                story=1,
                polygon=pa._polygon_xz(pa._corners(payload["rooms"][1]["floor"][0])),
                source_room_indices=(1,),
                floor_y=3.0,
            ),
            top=top,
        )
    ]

    candidates, _attempts = cs._candidates_from_selected_cells(
        payload,
        selected,
        ceiling_faces=[top_face],
        min_top_overlap_ratio=0.60,
        corner_tol=0.02,
        plane_groups=[],
    )

    floor_face = next(face for face in candidates[0].faces if face.kind == "floor")
    assert all(math.isclose(corner[1], 3.0) for corner in floor_face.corners)


def test_room_coverage_threshold_constant_matches_plan():
    assert math.isclose(ROOM_COVERED_RATIO, 0.80)
