"""Corpus fitting helpers for polyhedron selector priors."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from reconcile_tiers.polyhedron import payload_adapter as pa
from reconcile_tiers.polyhedron.cell_selector import select_payload_cells_v2
from reconcile_tiers.polyhedron.priors import DEFAULT, Priors, SelectionWeights


@dataclass(frozen=True, slots=True)
class SweepSetting:
    data_fit: float
    complexity: float
    coverage: float

    def weights(self) -> SelectionWeights:
        return SelectionWeights(
            data_fit=self.data_fit,
            complexity=self.complexity,
            coverage=self.coverage,
        )


@dataclass(frozen=True, slots=True)
class SweepResult:
    setting: SweepSetting
    building_count: int
    domain_count: int
    ok_count: int
    error_count: int
    skipped_count: int
    emitted_count: int
    complete_building_count: int
    low_coverage_emitted_count: int
    max_candidates: int
    emitted_coverage_p50: float | None
    emitted_coverage_p95: float | None
    score: float


def default_weight_grid() -> list[SweepSetting]:
    """Small normalized grid around the PolyFit default weights."""

    raw = [
        (0.43, 0.27, 0.30),
        (0.30, 0.30, 0.40),
        (0.25, 0.25, 0.50),
        (0.20, 0.20, 0.60),
        (0.50, 0.20, 0.30),
        (0.35, 0.15, 0.50),
        (0.15, 0.35, 0.50),
    ]
    return [SweepSetting(*_normalize_weights(item)) for item in raw]


def select_holdout_payloads(
    pipeline_dir: Path,
    *,
    max_buildings: int = 30,
) -> list[Path]:
    """Select a deterministic held-out subset of payload paths."""

    if max_buildings <= 0:
        return []
    payloads = sorted(pipeline_dir.glob("*/tier_payload.json"))
    if len(payloads) <= max_buildings:
        return payloads
    step = max(1, len(payloads) // max_buildings)
    selected = payloads[::step][:max_buildings]
    return selected


def run_weight_sweep(
    payload_paths: Iterable[Path],
    settings: Iterable[SweepSetting] | None = None,
    *,
    priors: Priors = DEFAULT,
    corner_tol: float = 0.02,
    time_budget_seconds: float = 0.5,
    max_intersections: int = 10_000,
    max_candidates: int = 500,
    low_coverage_threshold: float = 0.95,
) -> list[SweepResult]:
    """Evaluate selector-v2 over payloads for each weight setting."""

    paths = list(payload_paths)
    active_settings = (
        list(settings) if settings is not None else default_weight_grid()
    )
    return [
        _evaluate_setting(
            paths,
            setting,
            priors=priors,
            corner_tol=corner_tol,
            time_budget_seconds=time_budget_seconds,
            max_intersections=max_intersections,
            max_candidates=max_candidates,
            low_coverage_threshold=low_coverage_threshold,
        )
        for setting in active_settings
    ]


def choose_pareto_knee(results: list[SweepResult]) -> SweepResult | None:
    """Pick the highest-scoring setting with deterministic tie-breaking."""

    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            item.score,
            item.complete_building_count,
            item.emitted_count,
            -item.error_count,
            item.setting.coverage,
        ),
    )


def sweep_report(
    *,
    pipeline_dir: Path,
    payload_paths: list[Path],
    results: list[SweepResult],
) -> dict[str, Any]:
    knee = choose_pareto_knee(results)
    return {
        "schema_version": 1,
        "pipeline_dir": str(pipeline_dir),
        "payload_count": len(payload_paths),
        "payload_uuids": [path.parent.name for path in payload_paths],
        "results": [_result_json(result) for result in results],
        "selected": _result_json(knee) if knee is not None else None,
    }


def write_sweep_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))


def write_fitted_priors(
    path: Path,
    report: dict[str, Any],
    *,
    epsilon_meters: float = DEFAULT.epsilon_meters,
    sharp_edge_radians: float = DEFAULT.sharp_edge_radians,
    alpha_shape_radius: float = DEFAULT.alpha_shape_radius,
    min_support_points: int = DEFAULT.min_support_points,
) -> None:
    """Replace the committed ``DEFAULT`` block with the selected sweep knee."""

    source = path.read_text()
    path.write_text(
        render_fitted_priors_source(
            source,
            report,
            epsilon_meters=epsilon_meters,
            sharp_edge_radians=sharp_edge_radians,
            alpha_shape_radius=alpha_shape_radius,
            min_support_points=min_support_points,
        )
    )


def render_fitted_priors_source(
    source: str,
    report: dict[str, Any],
    *,
    epsilon_meters: float = DEFAULT.epsilon_meters,
    sharp_edge_radians: float = DEFAULT.sharp_edge_radians,
    alpha_shape_radius: float = DEFAULT.alpha_shape_radius,
    min_support_points: int = DEFAULT.min_support_points,
) -> str:
    """Return ``priors.py`` source with ``DEFAULT`` updated from a sweep report."""

    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("sweep report has no selected result")
    setting = selected.get("setting")
    if not isinstance(setting, dict):
        raise ValueError("selected sweep result has no setting")
    start = source.index("DEFAULT = Priors(")
    end_marker = "\n\n\n@dataclass(frozen=True, slots=True)\nclass SelectionWeights"
    end = source.index(end_marker, start)
    payload_uuids = report.get("payload_uuids") or []
    sample = ", ".join(str(item) for item in payload_uuids[:5])
    if len(payload_uuids) > 5:
        sample += f", ... (+{len(payload_uuids) - 5} more)"
    score = float(selected.get("score", 0.0))
    payload_count = report.get("payload_count", 0)
    block = f"""# Fitted on pipeline-outputs held-out subset, 2026-05-10.
# Report: {payload_count} payloads; selected score {score:.6f}.
# Payload sample: {sample or "none"}.
DEFAULT = Priors(
    epsilon_meters={float(epsilon_meters):.6g},
    sharp_edge_radians={float(sharp_edge_radians):.6g},
    alpha_shape_radius={float(alpha_shape_radius):.6g},
    weight_data_fit={float(setting["data_fit"]):.6g},
    weight_complexity={float(setting["complexity"]):.6g},
    weight_coverage={float(setting["coverage"]):.6g},
    min_support_points={int(min_support_points)},
)"""
    return source[:start] + block + source[end:]


def _evaluate_setting(
    payload_paths: list[Path],
    setting: SweepSetting,
    *,
    priors: Priors,
    corner_tol: float,
    time_budget_seconds: float,
    max_intersections: int,
    max_candidates: int,
    low_coverage_threshold: float,
) -> SweepResult:
    building_count = 0
    domain_count = 0
    ok_count = 0
    error_count = 0
    skipped_count = 0
    emitted_count = 0
    complete_building_count = 0
    low_coverage_emitted_count = 0
    max_seen_candidates = 0
    emitted_coverages: list[float] = []

    for payload_path in payload_paths:
        building_count += 1
        try:
            traces = _run_selector_for_payload(
                payload_path,
                weights=setting.weights(),
                corner_tol=corner_tol,
                time_budget_seconds=time_budget_seconds,
                max_intersections=max_intersections,
                max_candidates=max_candidates,
            )
        except Exception:
            error_count += 1
            continue
        building_complete = bool(traces)
        for trace in traces:
            domain_count += 1
            status = str(trace.get("status") or "unknown")
            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skipped_count += 1
                building_complete = False
            else:
                error_count += 1
                building_complete = False
            candidate_count = int(trace.get("candidate_count") or 0)
            max_seen_candidates = max(max_seen_candidates, candidate_count)
            coverage = _trace_coverage(trace)
            emitted = bool(trace.get("emitted_candidate"))
            if emitted:
                emitted_count += 1
                if coverage is not None:
                    emitted_coverages.append(coverage)
                    if coverage < low_coverage_threshold:
                        low_coverage_emitted_count += 1
                        building_complete = False
            else:
                building_complete = False
        if building_complete:
            complete_building_count += 1

    return SweepResult(
        setting=setting,
        building_count=building_count,
        domain_count=domain_count,
        ok_count=ok_count,
        error_count=error_count,
        skipped_count=skipped_count,
        emitted_count=emitted_count,
        complete_building_count=complete_building_count,
        low_coverage_emitted_count=low_coverage_emitted_count,
        max_candidates=max_seen_candidates,
        emitted_coverage_p50=_percentile(emitted_coverages, 0.50),
        emitted_coverage_p95=_percentile(emitted_coverages, 0.95),
        score=_score(
            building_count=building_count,
            domain_count=domain_count,
            emitted_count=emitted_count,
            complete_building_count=complete_building_count,
            error_count=error_count,
            low_coverage_emitted_count=low_coverage_emitted_count,
        ),
    )


def _run_selector_for_payload(
    payload_path: Path,
    *,
    weights: SelectionWeights,
    corner_tol: float,
    time_budget_seconds: float,
    max_intersections: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    payload = json.loads(payload_path.read_text())
    evidence_path = payload_path.parent / "plane_evidence.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
    faces = pa.payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
    faces.extend(
        pa.payload_faces_from_plane_evidence(
            evidence,
            corner_tol=corner_tol,
            include_filtered_candidates=True,
        )
    )
    footprint = pa._payload_footprint_polygon(
        payload,
        room_buffer_m=0.3,
        footprint_shrink_m=0.3,
        corner_tol=corner_tol,
    )
    if footprint is None:
        raise ValueError("could not derive payload footprint")
    result = select_payload_cells_v2(
        payload,
        footprint=footprint,
        ceiling_faces=[face for face in faces if face.kind == "ceiling"],
        corner_tol=corner_tol,
        time_budget_seconds=time_budget_seconds,
        max_intersections=max_intersections,
        max_candidates=max_candidates,
        weights=weights,
    )
    return result.domain_traces


def _trace_coverage(trace: dict[str, Any]) -> float | None:
    selection = trace.get("selection")
    if not isinstance(selection, dict):
        return None
    energy = selection.get("energy_breakdown")
    if not isinstance(energy, dict):
        return None
    value = energy.get("coverage_ratio")
    return float(value) if value is not None else None


def _score(
    *,
    building_count: int,
    domain_count: int,
    emitted_count: int,
    complete_building_count: int,
    error_count: int,
    low_coverage_emitted_count: int,
) -> float:
    if building_count <= 0 or domain_count <= 0:
        return float("-inf")
    complete_rate = complete_building_count / building_count
    emitted_rate = emitted_count / domain_count
    error_rate = error_count / domain_count
    low_rate = low_coverage_emitted_count / max(emitted_count, 1)
    return complete_rate * 2.0 + emitted_rate - error_rate * 2.0 - low_rate


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if fraction == 0.5:
        return float(median(values))
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _normalize_weights(
    weights: tuple[float, float, float],
) -> tuple[float, float, float]:
    total = sum(float(item) for item in weights)
    if total <= 0.0:
        raise ValueError("weight sum must be positive")
    return tuple(float(item) / total for item in weights)


def _result_json(result: SweepResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    out = asdict(result)
    out["setting"] = asdict(result.setting)
    return out
