"""Bounded scan-fit refinement loop for half-edge polyhedra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from reconcile_tiers.polyhedron.face_fit import (
    CoordinateDescentResult,
    ScanPointEvidence,
    coordinate_descent_face_shifts,
    evidence_from_scan_points,
    signed_distance_cost,
)
from reconcile_tiers.polyhedron.half_edge import HalfEdgePolyhedron
from reconcile_tiers.polyhedron.priors import DEFAULT, Priors
from reconcile_tiers.polyhedron.topology_events import (
    TopologyResolutionTrace,
    resolve_supported_topology_events,
)
from reconcile_tiers.polyhedron.validity import (
    TopologicalEvent,
    ValidityIssue,
    detect_topological_events,
    validate_polyhedron,
)

RefinementStopReason = Literal[
    "converged",
    "max_steps",
    "topology_blocked",
    "invalid_after_refinement",
]


@dataclass(frozen=True, slots=True)
class RefinementResult:
    initial_cost: float
    final_cost: float
    steps: int
    accepted_shifts: int
    stop_reason: RefinementStopReason
    stop_message: str
    evidence: tuple[ScanPointEvidence, ...]
    shift_results: tuple[CoordinateDescentResult, ...]
    topology_traces: tuple[TopologyResolutionTrace, ...]
    remaining_issues: tuple[ValidityIssue, ...]
    remaining_events: tuple[TopologicalEvent, ...]


def refine_polyhedron(
    polyhedron: HalfEdgePolyhedron,
    scan_points: np.ndarray,
    priors: Priors = DEFAULT,
    *,
    max_steps: int = 10,
    max_shift_iterations_per_step: int = 8,
    initial_step_m: float = 0.05,
    min_step_m: float = 0.001,
    topology_steps_per_pass: int = 8,
    topology_selection: Literal["unique", "first"] = "first",
    convergence_tol: float = 1e-8,
) -> RefinementResult:
    """Refine plane offsets against scan points while preserving topology bounds.

    The polyhedron is mutated in place, matching the lower-level face-shift and
    topology APIs. The loop is intentionally bounded so experimental corpus runs
    cannot spin indefinitely.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    evidence = tuple(
        evidence_from_scan_points(
            polyhedron,
            scan_points,
            epsilon=priors.epsilon_meters,
        )
    )
    initial_cost = signed_distance_cost(polyhedron, list(evidence))
    previous_cost = initial_cost
    accepted_shifts = 0
    shift_results: list[CoordinateDescentResult] = []
    topology_traces: list[TopologyResolutionTrace] = []

    for step_index in range(max_steps):
        shift_result = coordinate_descent_face_shifts(
            polyhedron,
            list(evidence),
            initial_step_m=initial_step_m,
            min_step_m=min_step_m,
            max_iterations=max_shift_iterations_per_step,
        )
        shift_results.append(shift_result)
        accepted_shifts += shift_result.accepted_shifts

        topology_trace = resolve_supported_topology_events(
            polyhedron,
            max_steps=topology_steps_per_pass,
            selection=topology_selection,
        )
        topology_traces.append(topology_trace)
        if topology_trace.stop_reason not in ("valid", "max_steps"):
            remaining_issues, remaining_events = _remaining_state(polyhedron)
            return RefinementResult(
                initial_cost=initial_cost,
                final_cost=signed_distance_cost(polyhedron, list(evidence)),
                steps=step_index + 1,
                accepted_shifts=accepted_shifts,
                stop_reason="topology_blocked",
                stop_message=topology_trace.stop_message,
                evidence=evidence,
                shift_results=tuple(shift_results),
                topology_traces=tuple(topology_traces),
                remaining_issues=remaining_issues,
                remaining_events=remaining_events,
            )

        current_cost = signed_distance_cost(polyhedron, list(evidence))
        issues, events = _remaining_state(polyhedron)
        if issues:
            return RefinementResult(
                initial_cost=initial_cost,
                final_cost=current_cost,
                steps=step_index + 1,
                accepted_shifts=accepted_shifts,
                stop_reason="invalid_after_refinement",
                stop_message="validity issues remain after refinement",
                evidence=evidence,
                shift_results=tuple(shift_results),
                topology_traces=tuple(topology_traces),
                remaining_issues=issues,
                remaining_events=events,
            )
        if (
            topology_trace.stop_reason == "valid"
            and shift_result.accepted_shifts == 0
            and abs(previous_cost - current_cost) <= convergence_tol
        ):
            return RefinementResult(
                initial_cost=initial_cost,
                final_cost=current_cost,
                steps=step_index + 1,
                accepted_shifts=accepted_shifts,
                stop_reason="converged",
                stop_message="scan-fit shifts and topology resolution converged",
                evidence=evidence,
                shift_results=tuple(shift_results),
                topology_traces=tuple(topology_traces),
                remaining_issues=issues,
                remaining_events=events,
            )
        previous_cost = current_cost

    issues, events = _remaining_state(polyhedron)
    return RefinementResult(
        initial_cost=initial_cost,
        final_cost=signed_distance_cost(polyhedron, list(evidence)),
        steps=max_steps,
        accepted_shifts=accepted_shifts,
        stop_reason="max_steps",
        stop_message=f"stopped after {max_steps} refinement step(s)",
        evidence=evidence,
        shift_results=tuple(shift_results),
        topology_traces=tuple(topology_traces),
        remaining_issues=issues,
        remaining_events=events,
    )


def _remaining_state(
    polyhedron: HalfEdgePolyhedron,
) -> tuple[tuple[ValidityIssue, ...], tuple[TopologicalEvent, ...]]:
    issues = tuple(validate_polyhedron(polyhedron))
    if issues:
        return issues, ()
    return issues, tuple(detect_topological_events(polyhedron))
