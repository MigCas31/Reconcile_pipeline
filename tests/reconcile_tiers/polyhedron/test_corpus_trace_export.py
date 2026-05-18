from __future__ import annotations

import json

from reconcile_tiers.polyhedron.corpus_trace_export import (
    _annotate_selector_v2_assembly,
    _selector_v2_building_summary,
    export_corpus_envelope_traces,
    export_corpus_room_audit,
    export_corpus_room_shell_traces,
    export_manifold_repair_steps_traces,
    export_selector_v2_traces,
)
from reconcile_tiers.polyhedron.manifold_repair_trace import SELECTION
from tests.reconcile_tiers.polyhedron.test_payload_adapter import _cube_payload


def test_export_corpus_room_shell_traces_writes_index_and_sample(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    building_dir = pipeline_dir / "building-a"
    building_dir.mkdir(parents=True)
    (building_dir / "tier_payload.json").write_text(json.dumps(_cube_payload()))

    out_dir = tmp_path / "traces"
    index = export_corpus_room_shell_traces(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
        valid_sample_limit=1,
    )

    assert index["summary"]["buildings"] == 1
    assert index["summary"]["attempted_rooms"] == 1
    assert index["summary"]["built_rooms"] == 1
    assert index["summary"]["failed_rooms"] == 0
    assert index["summary"]["written_traces"] == 1
    assert index["summary"]["stop_reason_counts"] == {"valid": 1}

    written_index = json.loads((out_dir / "index.json").read_text())
    assert written_index["records"][0]["uuid"] == "building-a"
    trace_relpath = written_index["records"][0]["trace"]
    assert trace_relpath == "traces/building-a__room_0.json"

    trace = json.loads((out_dir / trace_relpath).read_text())
    assert trace["stop"]["reason"] == "valid"
    assert trace["frames"][0]["counts"] == {
        "faces": 6,
        "vertices": 8,
        "half_edges": 24,
    }


def test_export_corpus_room_shell_traces_records_build_failures(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    building_dir = pipeline_dir / "building-a"
    building_dir.mkdir(parents=True)
    payload = _cube_payload()
    payload["ceiling"] = []
    (building_dir / "tier_payload.json").write_text(json.dumps(payload))

    index = export_corpus_room_shell_traces(
        pipeline_dir=pipeline_dir,
        output_dir=tmp_path / "traces",
    )

    assert index["summary"]["attempted_rooms"] == 1
    assert index["summary"]["built_rooms"] == 0
    assert index["summary"]["failed_rooms"] == 1
    assert index["build_failure_samples"][0]["uuid"] == "building-a"
    assert "no overlapping ceiling" in index["build_failure_samples"][0]["error"]


def test_export_corpus_envelope_traces_writes_building_part_trace(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    building_dir = pipeline_dir / "building-a"
    building_dir.mkdir(parents=True)
    (building_dir / "tier_payload.json").write_text(json.dumps(_cube_payload()))

    out_dir = tmp_path / "envelope-traces"
    index = export_corpus_envelope_traces(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
        wing_level=False,
        valid_sample_limit=1,
    )

    assert index["domain"] == "envelope"
    assert index["summary"]["buildings"] == 1
    assert index["summary"]["built_parts"] == 1
    assert index["summary"]["stop_reason_counts"] == {"valid": 1}
    trace_relpath = index["records"][0]["trace"]
    assert trace_relpath == "traces/building-a__envelope-wing_0.json"
    trace = json.loads((out_dir / trace_relpath).read_text())
    assert trace["frames"][0]["counts"] == {
        "faces": 6,
        "vertices": 8,
        "half_edges": 24,
    }


def test_export_corpus_room_audit_writes_room_reasons(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    building_dir = pipeline_dir / "building-a"
    building_dir.mkdir(parents=True)
    payload = _cube_payload()
    payload["ceiling"] = []
    (building_dir / "tier_payload.json").write_text(json.dumps(payload))

    out_dir = tmp_path / "room-audit"
    index = export_corpus_room_audit(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
    )

    assert index["domain"] == "room-audit"
    assert index["summary"]["audited_buildings"] == 1
    assert index["summary"]["dropped_rooms"] == 1
    assert index["summary"]["reason_counts"] == {"no_top_support": 1}
    assert "build_attempt_counts" in index["summary"]
    assert index["records"][0]["build_attempt_reasons"] == {}
    audit_relpath = index["records"][0]["audit_path"]
    audit = json.loads((out_dir / audit_relpath).read_text())
    assert audit["rooms"][0]["reason"] == "no_top_support"
    assert "build_attempts" in audit


def test_export_selector_v2_traces_writes_viewer_trace(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    building_dir = pipeline_dir / "building-a"
    building_dir.mkdir(parents=True)
    (building_dir / "tier_payload.json").write_text(json.dumps(_cube_payload()))

    out_dir = tmp_path / "selector-v2"
    index = export_selector_v2_traces(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
        max_buildings=1,
        time_budget_seconds=0.0,
    )

    assert index["domain"] == "selector-v2"
    assert index["summary"]["buildings"] == 1
    assert index["summary"]["status_counts"] == {"ok": 1}
    assert index["summary"]["candidate_cap_exceeded_domains"] == 0
    assert index["summary"]["emitted_candidates"] == 1
    assert index["summary"]["building_summary"]["bucket_counts"] == {"complete": 1}
    assert index["summary"]["building_summary"]["complete_buildings"] == ["building-a"]

    record = index["records"][0]
    assert record["candidate_count"] == 6
    assert record["selected_count"] == 6
    assert record["coverage_ratio"] == 1.0
    assert record["emitted_candidate"] == "envelope-v2-domain:0"
    assert record["assembly_eligible_candidate"] == "envelope-v2-domain:0"
    assert record["assembly_candidate"] is True
    assert record["assembly_coverage_ratio"] == 1.0
    domain_points = {
        tuple(round(value, 6) for value in point)
        for point in record["domain_polygon"]
    }
    assert domain_points == {
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
    }

    trace = json.loads((out_dir / record["trace"]).read_text())
    assert trace["selection"] == "selector-v2"
    assert [frame["counts"]["faces"] for frame in trace["frames"]] == [6, 6]
    assert trace["frames"][0]["faces"][0]["corners"]


def test_selector_v2_assembly_chooses_coverage_set_over_all_domains():
    records = [
        _selector_record(
            "building-a",
            0,
            domain=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            emitted=True,
        ),
        _selector_record(
            "building-a",
            1,
            domain=[[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0]],
            emitted=True,
        ),
        _selector_record(
            "building-a",
            2,
            domain=[[2.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 4.0]],
            emitted=False,
        ),
    ]

    _annotate_selector_v2_assembly(records)
    summary = _selector_v2_building_summary(records)

    assert records[0]["assembly_candidate"] is True
    assert records[1]["assembly_candidate"] is False
    assert records[2]["assembly_candidate"] is False
    assert records[0]["assembly_coverage_ratio"] == 1.0
    assert summary["bucket_counts"] == {"complete": 1}


def test_selector_v2_assembly_can_use_partial_eligible_fragments():
    records = [
        _selector_record(
            "building-a",
            0,
            domain=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            emitted=False,
            assembly_eligible=True,
            selected_coverage=[[[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0]]],
        ),
        _selector_record(
            "building-a",
            1,
            domain=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            emitted=False,
            assembly_eligible=True,
            selected_coverage=[[[2.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 4.0]]],
        ),
    ]

    _annotate_selector_v2_assembly(records)

    assert [record["assembly_candidate"] for record in records] == [True, True]
    assert records[0]["assembly_coverage_ratio"] == 1.0


def test_export_manifold_repair_steps_traces_writes_index(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    building_dir = pipeline_dir / "building-a"
    building_dir.mkdir(parents=True)
    (building_dir / "tier_payload.json").write_text(json.dumps(_cube_payload()))

    out_dir = tmp_path / "steps-traces"
    index = export_manifold_repair_steps_traces(
        pipeline_dir=pipeline_dir,
        output_dir=out_dir,
        max_buildings=1,
    )

    assert index["domain"] == SELECTION
    assert index["summary"]["records"] == 1
    record = index["records"][0]
    assert record["uuid"] == "building-a"
    assert record["frame_count"] == 10

    trace = json.loads((out_dir / record["trace"]).read_text())
    assert trace["selection"] == SELECTION
    steps = [f["pipeline_step"] for f in trace["frames"]]
    assert steps[3] == "roof_xz_clip"
    assert trace["stop"]["reason"] == "watertight"


def _selector_record(
    uuid: str,
    part_index: int,
    *,
    domain: list[list[float]],
    emitted: bool,
    assembly_eligible: bool | None = None,
    selected_coverage: list[list[list[float]]] | None = None,
) -> dict:
    if assembly_eligible is None:
        assembly_eligible = emitted
    return {
        "uuid": uuid,
        "part_index": part_index,
        "stop_reason": "ok",
        "coverage_ratio": 1.0,
        "low_coverage": False,
        "emitted_candidate": f"envelope-v2-domain:{part_index}" if emitted else None,
        "assembly_eligible_candidate": (
            f"envelope-v2-domain:{part_index}" if assembly_eligible else None
        ),
        "domain_polygon": domain,
        "selected_coverage_polygons": selected_coverage or [],
    }
