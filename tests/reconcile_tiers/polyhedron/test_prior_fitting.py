from __future__ import annotations

import json
from pathlib import Path

from reconcile_tiers.polyhedron.prior_fitting import (
    SweepResult,
    SweepSetting,
    choose_pareto_knee,
    default_weight_grid,
    render_fitted_priors_source,
    select_holdout_payloads,
    sweep_report,
)


def test_default_weight_grid_is_normalized():
    grid = default_weight_grid()

    assert grid
    for setting in grid:
        assert round(setting.data_fit + setting.complexity + setting.coverage, 12) == 1


def test_select_holdout_payloads_is_deterministic_spread(tmp_path):
    pipeline_dir = tmp_path / "pipeline-outputs"
    for index in range(10):
        building_dir = pipeline_dir / f"building-{index:02d}"
        building_dir.mkdir(parents=True)
        (building_dir / "tier_payload.json").write_text("{}")

    selected = select_holdout_payloads(pipeline_dir, max_buildings=3)

    assert [path.parent.name for path in selected] == [
        "building-00",
        "building-03",
        "building-06",
    ]


def test_choose_pareto_knee_prefers_high_score_then_completeness():
    weaker = _result(score=1.0, complete=2, emitted=20)
    stronger = _result(score=1.0, complete=3, emitted=10)

    assert choose_pareto_knee([weaker, stronger]) is stronger


def test_sweep_report_is_json_serializable(tmp_path):
    payload = tmp_path / "pipeline-outputs" / "building-a" / "tier_payload.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}")

    report = sweep_report(
        pipeline_dir=tmp_path / "pipeline-outputs",
        payload_paths=[payload],
        results=[_result(score=1.0, complete=1, emitted=1)],
    )

    encoded = json.dumps(report)
    assert "building-a" in encoded
    assert report["selected"]["score"] == 1.0


def test_render_fitted_priors_source_updates_default_block():
    source = '''"""priors"""

from dataclasses import dataclass


DEFAULT = Priors(
    epsilon_meters=0.05,
    sharp_edge_radians=0.35,
    alpha_shape_radius=0.50,
    weight_data_fit=0.43,
    weight_complexity=0.27,
    weight_coverage=0.30,
    min_support_points=0,
)


@dataclass(frozen=True, slots=True)
class SelectionWeights:
    pass
'''
    report = sweep_report(
        pipeline_dir=Path("pipeline-outputs"),
        payload_paths=[],
        results=[_result(score=1.25, complete=1, emitted=1)],
    )

    rendered = render_fitted_priors_source(source, report)

    assert "selected score 1.250000" in rendered
    assert "weight_data_fit=0.4" in rendered
    assert "weight_complexity=0.3" in rendered
    assert "weight_coverage=0.3" in rendered
    assert "class SelectionWeights" in rendered


def _result(*, score: float, complete: int, emitted: int) -> SweepResult:
    return SweepResult(
        setting=SweepSetting(data_fit=0.4, complexity=0.3, coverage=0.3),
        building_count=5,
        domain_count=20,
        ok_count=20,
        error_count=0,
        skipped_count=0,
        emitted_count=emitted,
        complete_building_count=complete,
        low_coverage_emitted_count=0,
        max_candidates=6,
        emitted_coverage_p50=1.0,
        emitted_coverage_p95=1.0,
        score=score,
    )
