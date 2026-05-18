"""Topological event resolvers for half-edge polyhedra.

These are intentionally narrow implementations of the ISPRS 2024 paper's
event-resolution layer. Each resolver mutates topology only after checking the
local shape it understands; unsupported cases raise ``TopologyResolutionError``
instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdge,
    HalfEdgePolyhedron,
    Vertex,
    build_from_planar_polygons,
)
from reconcile_tiers.polyhedron.validity import (
    TopologicalEvent,
    ValidityIssue,
    detect_topological_events,
    validate_polyhedron,
)

SelectionMode = Literal["unique", "first"]
TopologyAction = Literal[
    "edge_collapse",
    "edge_flip",
    "face_creation",
    "triangle_face_collapse",
    "adjacent_coplanar_face_merge",
]
TopologyStopReason = Literal[
    "valid",
    "max_steps",
    "ambiguous_events",
    "ambiguous_issues",
    "unsupported_issues",
    "resolver_error",
]


class TopologyResolutionError(ValueError):
    """The requested topological event cannot be safely resolved yet."""


@dataclass(frozen=True, slots=True)
class EdgeCollapseResult:
    kept_vertex_id: int
    removed_vertex_id: int
    removed_half_edge_ids: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FaceCollapseResult:
    removed_face_id: int
    kept_vertex_id: int
    removed_vertex_id: int
    removed_half_edge_ids: tuple[int, ...]
    paired_half_edge_ids: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CoplanarFaceMergeResult:
    kept_face_id: int
    removed_face_id: int
    removed_half_edge_ids: tuple[int, int]


@dataclass(frozen=True, slots=True)
class EdgeFlipResult:
    applied: bool
    old_half_edge_ids: tuple[int, int]
    old_face_ids: tuple[int, int]
    new_face_ids: tuple[int, int]
    new_diagonal: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class FaceCreationResult:
    applied: bool
    target_face_id: int
    created_face_ids: tuple[int, int]
    split_vertices: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class TopologyCounts:
    faces: int
    vertices: int
    half_edges: int


@dataclass(frozen=True, slots=True)
class TopologyResolutionStep:
    index: int
    action: TopologyAction
    trigger_ids: tuple[int, ...]
    before: TopologyCounts
    after: TopologyCounts
    result: (
        EdgeCollapseResult
        | EdgeFlipResult
        | FaceCollapseResult
        | FaceCreationResult
        | CoplanarFaceMergeResult
    )


@dataclass(frozen=True, slots=True)
class TopologyResolutionTrace:
    steps: tuple[TopologyResolutionStep, ...]
    stop_reason: TopologyStopReason
    stop_message: str
    remaining_issues: tuple[ValidityIssue, ...]
    remaining_events: tuple[TopologicalEvent, ...]


def resolve_supported_topology_events(
    polyhedron: HalfEdgePolyhedron,
    *,
    max_steps: int = 32,
    selection: SelectionMode = "unique",
    edge_tol_m: float = 1e-6,
    face_area_tol_m2: float = 1e-8,
) -> TopologyResolutionTrace:
    """Apply supported topology updates until valid or blocked.

    The default ``selection="unique"`` mode is deliberately conservative: if
    several supported events or coplanar-face issues are present, it stops and
    returns a trace instead of picking one. ``selection="first"`` is useful for
    experiments and visualization because it chooses a deterministic supported
    candidate, but callers should treat that mode as diagnostic.
    """

    if max_steps < 0:
        raise TopologyResolutionError("max_steps must be non-negative")
    if selection not in ("unique", "first"):
        raise TopologyResolutionError(f"unsupported selection mode: {selection}")

    steps: list[TopologyResolutionStep] = []
    for step_index in range(max_steps):
        issues = tuple(validate_polyhedron(polyhedron))
        unsupported_issues = tuple(
            issue
            for issue in issues
            if issue.kind != "adjacent_coplanar_faces"
        )
        if unsupported_issues:
            return _trace_stop(
                steps,
                stop_reason="unsupported_issues",
                stop_message=(
                    "unsupported validity issue(s): "
                    + ", ".join(issue.kind for issue in unsupported_issues)
                ),
                remaining_issues=issues,
                remaining_events=(),
            )

        coplanar_issues = tuple(
            issue for issue in issues if issue.kind == "adjacent_coplanar_faces"
        )
        if coplanar_issues:
            if len(coplanar_issues) > 1 and selection == "unique":
                return _trace_stop(
                    steps,
                    stop_reason="ambiguous_issues",
                    stop_message=(
                        "multiple adjacent coplanar face pairs require "
                        "explicit selection"
                    ),
                    remaining_issues=issues,
                    remaining_events=(),
                )
            issue = _select_coplanar_issue(coplanar_issues)
            before = _topology_counts(polyhedron)
            try:
                result = resolve_single_adjacent_coplanar_face_merge(
                    polyhedron,
                    issue.ids,
                )
            except TopologyResolutionError as exc:
                return _trace_stop(
                    steps,
                    stop_reason="resolver_error",
                    stop_message=str(exc),
                    remaining_issues=issues,
                    remaining_events=(),
                )
            steps.append(
                TopologyResolutionStep(
                    index=step_index,
                    action="adjacent_coplanar_face_merge",
                    trigger_ids=issue.ids,
                    before=before,
                    after=_topology_counts(polyhedron),
                    result=result,
                )
            )
            continue

        events = tuple(
            detect_topological_events(
                polyhedron,
                edge_tol_m=edge_tol_m,
                face_area_tol_m2=face_area_tol_m2,
            )
        )
        if not events:
            return _trace_stop(
                steps,
                stop_reason="valid",
                stop_message="no validity issues or supported events remain",
                remaining_issues=(),
                remaining_events=(),
            )
        if len(events) > 1 and selection == "unique":
            return _trace_stop(
                steps,
                stop_reason="ambiguous_events",
                stop_message="multiple topological events require explicit selection",
                remaining_issues=(),
                remaining_events=events,
            )

        event = _select_topological_event(events)
        before = _topology_counts(polyhedron)
        try:
            result = _resolve_topological_event(
                polyhedron,
                event,
                edge_tol_m=edge_tol_m,
                face_area_tol_m2=face_area_tol_m2,
            )
        except TopologyResolutionError as exc:
            return _trace_stop(
                steps,
                stop_reason="resolver_error",
                stop_message=str(exc),
                remaining_issues=(),
                remaining_events=events,
            )
        steps.append(
            TopologyResolutionStep(
                index=step_index,
                action=_action_for_event(event),
                trigger_ids=event.ids,
                before=before,
                after=_topology_counts(polyhedron),
                result=result,
            )
        )

    remaining_issues, remaining_events = _remaining_state(
        polyhedron,
        edge_tol_m=edge_tol_m,
        face_area_tol_m2=face_area_tol_m2,
    )
    return _trace_stop(
        steps,
        stop_reason="max_steps",
        stop_message=f"stopped after {max_steps} topology step(s)",
        remaining_issues=remaining_issues,
        remaining_events=remaining_events,
    )


def resolve_single_edge_collapse(
    polyhedron: HalfEdgePolyhedron,
    event: TopologicalEvent | None = None,
    *,
    edge_tol_m: float = 1e-6,
) -> EdgeCollapseResult:
    """Resolve one zero-length edge by merging its two endpoint vertices.

    This implements the simplest Section 3.3.1 case: delete the two half-edges
    representing the collapsed edge, splice each adjacent face ring over the
    deleted half-edge, and retarget all outgoing half-edges from the removed
    vertex to the kept vertex.

    It deliberately refuses cases where either adjacent face is already a
    triangle. Those require the paper's face-collapse resolver (Section 3.3.3).
    """

    selected = _select_edge_collapse_event(polyhedron, event, edge_tol_m=edge_tol_m)
    half_edge = _half_edge_by_id(polyhedron, selected.ids[0])
    opposite = half_edge.opposite
    if opposite is None:
        raise TopologyResolutionError("collapsed half-edge has no opposite")
    if selected.ids[1] != opposite.id:
        half_edge = _half_edge_by_id(polyhedron, selected.ids[1])
        opposite = half_edge.opposite
        if opposite is None or opposite.id != selected.ids[0]:
            raise TopologyResolutionError("event ids are not opposite half-edges")

    if half_edge.next is None or opposite.next is None:
        raise TopologyResolutionError("collapsed edge has an open face ring")
    if half_edge.face is None or opposite.face is None:
        raise TopologyResolutionError("collapsed edge is missing adjacent faces")
    if _face_size(half_edge) <= 3 or _face_size(opposite) <= 3:
        raise TopologyResolutionError(
            "edge collapse adjacent to a triangle requires face-collapse resolver"
        )

    previous = _previous_in_face(half_edge)
    opposite_previous = _previous_in_face(opposite)
    if previous is None or opposite_previous is None:
        raise TopologyResolutionError("could not find previous half-edge")

    kept = half_edge.origin
    removed = opposite.origin
    removed_ids = (half_edge.id, opposite.id)

    previous.next = half_edge.next
    opposite_previous.next = opposite.next
    if half_edge.face.half_edge is half_edge:
        half_edge.face.half_edge = half_edge.next
    if opposite.face.half_edge is opposite:
        opposite.face.half_edge = opposite.next

    for candidate in polyhedron.half_edges:
        if candidate.origin is removed and candidate not in (half_edge, opposite):
            candidate.origin = kept

    polyhedron.half_edges = [
        candidate
        for candidate in polyhedron.half_edges
        if candidate not in (half_edge, opposite)
    ]
    polyhedron.vertices = [
        vertex for vertex in polyhedron.vertices if vertex is not removed
    ]

    _refresh_outgoing(polyhedron, preferred=kept)

    return EdgeCollapseResult(
        kept_vertex_id=kept.id,
        removed_vertex_id=removed.id,
        removed_half_edge_ids=removed_ids,
    )


def resolve_single_triangle_face_collapse(
    polyhedron: HalfEdgePolyhedron,
    event: TopologicalEvent | None = None,
    *,
    edge_tol_m: float = 1e-6,
    face_area_tol_m2: float = 1e-8,
) -> FaceCollapseResult:
    """Resolve one triangle face that has collapsed to a segment.

    Narrow Section 3.3.3 case: the selected face is a triangle, exactly one of
    its three edges is zero-length, and that zero edge has an adjacent outside
    face. The triangle is removed, the outside zero half-edge is spliced out of
    its face, the zero-edge endpoints are merged, and the two outside half-edges
    along the surviving segment are paired as opposites.
    """

    selected = _select_triangle_face_collapse_event(
        polyhedron,
        event,
        face_area_tol_m2=face_area_tol_m2,
    )
    face = _face_by_id(polyhedron, selected.ids[0])
    ring = _face_ring(face)
    if len(ring) != 3:
        raise TopologyResolutionError("face-collapse resolver requires a triangle")

    zero_edges = [
        half_edge
        for half_edge in ring
        if _edge_length(polyhedron, half_edge) <= edge_tol_m
    ]
    if len(zero_edges) != 1:
        raise TopologyResolutionError(
            "expected exactly one zero edge on collapsed triangle, "
            f"found {len(zero_edges)}"
        )

    zero = zero_edges[0]
    zero_opposite = zero.opposite
    if zero.next is None or zero_opposite is None:
        raise TopologyResolutionError("collapsed triangle has open zero edge")
    zero_outside_previous = _previous_in_face(zero_opposite)
    if zero_outside_previous is None or zero_opposite.next is None:
        raise TopologyResolutionError("could not splice outside zero half-edge")

    first_segment = zero.next
    second_segment = zero.next.next
    if second_segment is None:
        raise TopologyResolutionError("collapsed triangle ring is open")
    first_outside = first_segment.opposite
    second_outside = second_segment.opposite
    if first_outside is None or second_outside is None:
        raise TopologyResolutionError("collapsed triangle segment lacks outside twins")

    kept = zero.next.origin
    removed = zero.origin
    removed_half_edges = (
        zero.id,
        first_segment.id,
        second_segment.id,
        zero_opposite.id,
    )

    zero_outside_previous.next = zero_opposite.next
    if zero_opposite.face is not None and zero_opposite.face.half_edge is zero_opposite:
        zero_opposite.face.half_edge = zero_opposite.next

    for candidate in polyhedron.half_edges:
        if candidate.origin is removed:
            candidate.origin = kept

    first_outside.opposite = second_outside
    second_outside.opposite = first_outside

    removed_half_edge_set = set(removed_half_edges)
    polyhedron.half_edges = [
        candidate
        for candidate in polyhedron.half_edges
        if candidate.id not in removed_half_edge_set
    ]
    polyhedron.faces = [
        candidate for candidate in polyhedron.faces if candidate is not face
    ]
    polyhedron.vertices = [
        candidate for candidate in polyhedron.vertices if candidate is not removed
    ]

    _refresh_outgoing(polyhedron, preferred=kept)

    return FaceCollapseResult(
        removed_face_id=face.id,
        kept_vertex_id=kept.id,
        removed_vertex_id=removed.id,
        removed_half_edge_ids=removed_half_edges,
        paired_half_edge_ids=(first_outside.id, second_outside.id),
    )


def resolve_single_adjacent_coplanar_face_merge(
    polyhedron: HalfEdgePolyhedron,
    face_ids: tuple[int, int] | None = None,
) -> CoplanarFaceMergeResult:
    """Merge two adjacent faces sharing the same supporting plane.

    The shared half-edge pair is removed, the two boundary rings are spliced
    into one ring, and all surviving half-edges from the removed face are
    reassigned to the kept face. This is the cleanup needed after some collapse
    sequences leave two faces on the same support plane.
    """

    kept, removed = _select_adjacent_coplanar_faces(polyhedron, face_ids)
    shared = _shared_half_edge_pair(kept, removed)
    if shared is None:
        raise TopologyResolutionError(
            f"faces {kept.id} and {removed.id} do not share an edge"
        )
    kept_shared, removed_shared = shared
    if kept_shared.next is None or removed_shared.next is None:
        raise TopologyResolutionError("shared coplanar edge has an open ring")
    removed_ring = _face_ring(removed)
    kept_previous = _previous_in_face(kept_shared)
    removed_previous = _previous_in_face(removed_shared)
    if kept_previous is None or removed_previous is None:
        raise TopologyResolutionError("could not find previous shared half-edge")

    kept_previous.next = removed_shared.next
    removed_previous.next = kept_shared.next

    for half_edge in removed_ring:
        if half_edge is not removed_shared:
            half_edge.face = kept
    if kept.half_edge is kept_shared:
        kept.half_edge = kept_shared.next

    possible_two_face_vertices = (kept_shared.origin, removed_shared.origin)
    removed_ids = (kept_shared.id, removed_shared.id)
    polyhedron.half_edges = [
        half_edge for half_edge in polyhedron.half_edges if half_edge not in shared
    ]
    polyhedron.faces = [face for face in polyhedron.faces if face is not removed]

    for vertex in possible_two_face_vertices:
        _dissolve_two_face_vertex_if_needed(polyhedron, vertex)

    _refresh_outgoing(polyhedron)

    return CoplanarFaceMergeResult(
        kept_face_id=kept.id,
        removed_face_id=removed.id,
        removed_half_edge_ids=removed_ids,
    )


def resolve_single_edge_flip(
    polyhedron: HalfEdgePolyhedron,
    half_edge_id: int,
    *,
    coord_tol: float = 1e-3,
) -> EdgeFlipResult:
    """Flip the diagonal shared by two triangular faces.

    This is the conservative local form of the ISPRS edge-flip operator: the
    selected half-edge and its opposite must bound two triangles. The operator
    rebuilds the half-edge graph from the current face polygons after replacing
    the old diagonal with the alternate diagonal.
    """

    half_edge = _half_edge_by_id(polyhedron, half_edge_id)
    opposite = half_edge.opposite
    if opposite is None:
        raise TopologyResolutionError("edge flip requires an opposite half-edge")
    if half_edge.face is None or opposite.face is None:
        raise TopologyResolutionError("edge flip requires two adjacent faces")
    ring_a = _face_ring_starting_at(half_edge)
    ring_b = _face_ring_starting_at(opposite)
    if len(ring_a) != 3 or len(ring_b) != 3:
        raise TopologyResolutionError("edge flip currently requires two triangles")

    a = _vertex_tuple(polyhedron, ring_a[0].origin)
    b = _vertex_tuple(polyhedron, ring_a[1].origin)
    c = _vertex_tuple(polyhedron, ring_a[2].origin)
    d = _vertex_tuple(polyhedron, ring_b[2].origin)
    old_face_ids = (half_edge.face.id, opposite.face.id)
    base = _polyhedron_polygons(polyhedron, exclude_face_ids=set(old_face_ids))
    candidate_a = [c, d, b]
    candidate_b = [d, c, a]
    rebuilt = _build_with_oriented_replacements(
        base,
        (candidate_a, candidate_b),
        coord_tol=coord_tol,
    )
    _replace_polyhedron(polyhedron, rebuilt)
    new_face_ids = (polyhedron.faces[-2].id, polyhedron.faces[-1].id)
    return EdgeFlipResult(
        applied=True,
        old_half_edge_ids=(half_edge.id, opposite.id),
        old_face_ids=old_face_ids,
        new_face_ids=new_face_ids,
        new_diagonal=(c, d),
    )


def resolve_single_face_creation(
    polyhedron: HalfEdgePolyhedron,
    splitting_plane: Plane,
    target_face_id: int,
    *,
    coord_tol: float = 1e-3,
) -> FaceCreationResult:
    """Split one face by a plane that passes through two existing vertices.

    The fully general face-creation operator can introduce new boundary
    vertices and split neighbouring faces. This first implementation handles
    the explicit local case needed by the refinement tests: the split line
    intersects the target polygon at exactly two existing vertices, so no
    adjacent face needs to be subdivided.
    """

    target = _face_by_id(polyhedron, target_face_id)
    corners = [_as_tuple(point) for point in polyhedron.face_polygon(target)]
    if len(corners) < 4:
        raise TopologyResolutionError("face creation requires a 4+ vertex face")
    distances = [
        _signed_distance_to_plane(point, splitting_plane) for point in corners
    ]
    split_indices = [
        index for index, distance in enumerate(distances) if abs(distance) <= coord_tol
    ]
    if len(split_indices) != 2:
        raise TopologyResolutionError(
            "face creation currently requires the splitting plane to pass "
            f"through exactly two existing vertices, found {len(split_indices)}"
        )
    first, second = sorted(split_indices)
    if (second - first) % len(corners) in (0, 1, len(corners) - 1):
        raise TopologyResolutionError("splitting vertices must be non-adjacent")

    first_polygon = corners[first : second + 1]
    second_polygon = corners[second:] + corners[: first + 1]
    if len(first_polygon) < 3 or len(second_polygon) < 3:
        raise TopologyResolutionError("face creation produced a degenerate split")

    base = _polyhedron_polygons(polyhedron, exclude_face_ids={target.id})
    rebuilt = build_from_planar_polygons(
        [
            *base,
            (first_polygon, target.plane),
            (second_polygon, target.plane),
        ],
        coord_tol=coord_tol,
    )
    _replace_polyhedron(polyhedron, rebuilt)
    created_face_ids = (polyhedron.faces[-2].id, polyhedron.faces[-1].id)
    return FaceCreationResult(
        applied=True,
        target_face_id=target_face_id,
        created_face_ids=created_face_ids,
        split_vertices=(corners[first], corners[second]),
    )


def _select_edge_collapse_event(
    polyhedron: HalfEdgePolyhedron,
    event: TopologicalEvent | None,
    *,
    edge_tol_m: float,
) -> TopologicalEvent:
    if event is not None:
        if event.kind != "edge_collapse":
            raise TopologyResolutionError(f"unsupported event kind: {event.kind}")
        return event

    events = [
        candidate
        for candidate in detect_topological_events(polyhedron, edge_tol_m=edge_tol_m)
        if candidate.kind == "edge_collapse"
    ]
    if not events:
        raise TopologyResolutionError("no edge-collapse event detected")
    if len(events) > 1:
        raise TopologyResolutionError(
            f"expected one edge-collapse event, found {len(events)}"
        )
    return events[0]


def _select_triangle_face_collapse_event(
    polyhedron: HalfEdgePolyhedron,
    event: TopologicalEvent | None,
    *,
    face_area_tol_m2: float,
) -> TopologicalEvent:
    if event is not None:
        if event.kind != "triangle_face_collapse":
            raise TopologyResolutionError(f"unsupported event kind: {event.kind}")
        return event

    events = [
        candidate
        for candidate in detect_topological_events(
            polyhedron,
            face_area_tol_m2=face_area_tol_m2,
        )
        if candidate.kind == "triangle_face_collapse"
    ]
    if not events:
        raise TopologyResolutionError("no triangle-face-collapse event detected")
    if len(events) > 1:
        raise TopologyResolutionError(
            f"expected one triangle-face-collapse event, found {len(events)}"
        )
    return events[0]


def _select_adjacent_coplanar_faces(
    polyhedron: HalfEdgePolyhedron,
    face_ids: tuple[int, int] | None,
) -> tuple[Face, Face]:
    if face_ids is not None:
        if len(face_ids) != 2:
            raise TopologyResolutionError("face_ids must contain exactly two ids")
        kept = _face_by_id(polyhedron, face_ids[0])
        removed = _face_by_id(polyhedron, face_ids[1])
        if not _faces_share_support(kept, removed):
            raise TopologyResolutionError(
                f"faces {kept.id} and {removed.id} are not coplanar"
            )
        return kept, removed

    issues = [
        issue
        for issue in validate_polyhedron(polyhedron)
        if issue.kind == "adjacent_coplanar_faces"
    ]
    if not issues:
        raise TopologyResolutionError("no adjacent coplanar faces detected")
    if len(issues) > 1:
        raise TopologyResolutionError(
            f"expected one adjacent coplanar face pair, found {len(issues)}"
        )
    return _face_by_id(polyhedron, issues[0].ids[0]), _face_by_id(
        polyhedron,
        issues[0].ids[1],
    )


def _resolve_topological_event(
    polyhedron: HalfEdgePolyhedron,
    event: TopologicalEvent,
    *,
    edge_tol_m: float,
    face_area_tol_m2: float,
) -> EdgeCollapseResult | FaceCollapseResult:
    if event.kind == "triangle_face_collapse":
        return resolve_single_triangle_face_collapse(
            polyhedron,
            event,
            edge_tol_m=edge_tol_m,
            face_area_tol_m2=face_area_tol_m2,
        )
    if event.kind == "edge_collapse":
        return resolve_single_edge_collapse(
            polyhedron,
            event,
            edge_tol_m=edge_tol_m,
        )
    raise TopologyResolutionError(f"unsupported event kind: {event.kind}")


def _action_for_event(event: TopologicalEvent) -> TopologyAction:
    if event.kind == "triangle_face_collapse":
        return "triangle_face_collapse"
    if event.kind == "edge_collapse":
        return "edge_collapse"
    raise TopologyResolutionError(f"unsupported event kind: {event.kind}")


def _select_topological_event(
    events: tuple[TopologicalEvent, ...],
) -> TopologicalEvent:
    priority = {"triangle_face_collapse": 0, "edge_collapse": 1}
    return min(events, key=lambda event: (priority[event.kind], event.ids))


def _select_coplanar_issue(
    issues: tuple[ValidityIssue, ...],
) -> ValidityIssue:
    return min(issues, key=lambda issue: issue.ids)


def _topology_counts(polyhedron: HalfEdgePolyhedron) -> TopologyCounts:
    return TopologyCounts(
        faces=len(polyhedron.faces),
        vertices=len(polyhedron.vertices),
        half_edges=len(polyhedron.half_edges),
    )


def _remaining_state(
    polyhedron: HalfEdgePolyhedron,
    *,
    edge_tol_m: float,
    face_area_tol_m2: float,
) -> tuple[tuple[ValidityIssue, ...], tuple[TopologicalEvent, ...]]:
    issues = tuple(validate_polyhedron(polyhedron))
    if issues:
        return issues, ()
    events = tuple(
        detect_topological_events(
            polyhedron,
            edge_tol_m=edge_tol_m,
            face_area_tol_m2=face_area_tol_m2,
        )
    )
    return issues, events


def _trace_stop(
    steps: list[TopologyResolutionStep],
    *,
    stop_reason: TopologyStopReason,
    stop_message: str,
    remaining_issues: tuple[ValidityIssue, ...],
    remaining_events: tuple[TopologicalEvent, ...],
) -> TopologyResolutionTrace:
    return TopologyResolutionTrace(
        steps=tuple(steps),
        stop_reason=stop_reason,
        stop_message=stop_message,
        remaining_issues=remaining_issues,
        remaining_events=remaining_events,
    )


def _half_edge_by_id(polyhedron: HalfEdgePolyhedron, half_edge_id: int) -> HalfEdge:
    for half_edge in polyhedron.half_edges:
        if half_edge.id == half_edge_id:
            return half_edge
    raise TopologyResolutionError(f"half-edge {half_edge_id} does not exist")


def _face_by_id(polyhedron: HalfEdgePolyhedron, face_id: int) -> Face:
    for face in polyhedron.faces:
        if face.id == face_id:
            return face
    raise TopologyResolutionError(f"face {face_id} does not exist")


def _face_ring(face: Face) -> list[HalfEdge]:
    if face.half_edge is None:
        return []
    ring: list[HalfEdge] = []
    half_edge = face.half_edge
    for _ in range(512):
        ring.append(half_edge)
        if half_edge.next is None:
            return []
        half_edge = half_edge.next
        if half_edge is face.half_edge:
            return ring
    return []


def _face_ring_starting_at(start: HalfEdge) -> list[HalfEdge]:
    if start.face is None:
        return []
    ring: list[HalfEdge] = []
    half_edge = start
    for _ in range(512):
        ring.append(half_edge)
        if half_edge.next is None:
            return []
        half_edge = half_edge.next
        if half_edge is start:
            return ring
    return []


def _shared_half_edge_pair(
    face_a: Face,
    face_b: Face,
) -> tuple[HalfEdge, HalfEdge] | None:
    for half_edge in _face_ring(face_a):
        opposite = half_edge.opposite
        if opposite is not None and opposite.face is face_b:
            return half_edge, opposite
    return None


def _faces_share_support(face_a: Face, face_b: Face) -> bool:
    pa = face_a.plane
    pb = face_b.plane
    na = (pa.a, pa.b, pa.c)
    nb = (pb.a, pb.b, pb.c)
    dot = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]
    if dot < 0.0:
        nb = (-nb[0], -nb[1], -nb[2])
        pb_d = -pb.d
    else:
        pb_d = pb.d
    return (
        abs(na[0] - nb[0]) <= 1e-6
        and abs(na[1] - nb[1]) <= 1e-6
        and abs(na[2] - nb[2]) <= 1e-6
        and abs(pa.d - pb_d) <= 1e-6
    )


def _edge_length(polyhedron: HalfEdgePolyhedron, half_edge: HalfEdge) -> float:
    if half_edge.next is None:
        return float("inf")
    start = polyhedron.vertex_position(half_edge.origin)
    end = polyhedron.vertex_position(half_edge.next.origin)
    return float(((end - start) ** 2).sum() ** 0.5)


def _polyhedron_polygons(
    polyhedron: HalfEdgePolyhedron,
    *,
    exclude_face_ids: set[int],
) -> list[tuple[list[tuple[float, float, float]], Plane]]:
    polygons: list[tuple[list[tuple[float, float, float]], Plane]] = []
    for face in polyhedron.faces:
        if face.id in exclude_face_ids:
            continue
        polygons.append(
            (
                [_as_tuple(point) for point in polyhedron.face_polygon(face)],
                face.plane,
            )
        )
    return polygons


def _build_with_oriented_replacements(
    base: list[tuple[list[tuple[float, float, float]], Plane]],
    replacements: tuple[
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
    ],
    *,
    coord_tol: float,
) -> HalfEdgePolyhedron:
    last_error: Exception | None = None
    for reverse_first in (False, True):
        for reverse_second in (False, True):
            first = (
                list(reversed(replacements[0]))
                if reverse_first
                else replacements[0]
            )
            second = (
                list(reversed(replacements[1]))
                if reverse_second
                else replacements[1]
            )
            try:
                return build_from_planar_polygons(
                    [
                        *base,
                        (first, _plane_from_polygon(first)),
                        (second, _plane_from_polygon(second)),
                    ],
                    coord_tol=coord_tol,
                )
            except ValueError as exc:
                last_error = exc
                continue
    raise TopologyResolutionError(
        f"edge flip produced non-manifold replacement faces: {last_error}"
    )


def _plane_from_polygon(corners: list[tuple[float, float, float]]) -> Plane:
    normal = np.zeros(3, dtype=float)
    for index, current in enumerate(corners):
        nxt = corners[(index + 1) % len(corners)]
        normal[0] += (current[1] - nxt[1]) * (current[2] + nxt[2])
        normal[1] += (current[2] - nxt[2]) * (current[0] + nxt[0])
        normal[2] += (current[0] - nxt[0]) * (current[1] + nxt[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise TopologyResolutionError("replacement face is degenerate")
    normal /= norm
    point = np.asarray(corners[0], dtype=float)
    return Plane(
        a=float(normal[0]),
        b=float(normal[1]),
        c=float(normal[2]),
        d=float(np.dot(normal, point)),
    )


def _replace_polyhedron(
    target: HalfEdgePolyhedron,
    source: HalfEdgePolyhedron,
) -> None:
    target.vertices = source.vertices
    target.half_edges = source.half_edges
    target.faces = source.faces


def _vertex_tuple(
    polyhedron: HalfEdgePolyhedron,
    vertex: Vertex,
) -> tuple[float, float, float]:
    return _as_tuple(polyhedron.vertex_position(vertex))


def _as_tuple(
    point: np.ndarray | tuple[float, float, float],
) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def _signed_distance_to_plane(
    point: tuple[float, float, float],
    plane: Plane,
) -> float:
    return (
        plane.a * point[0]
        + plane.b * point[1]
        + plane.c * point[2]
        - plane.d
    )


def _dissolve_two_face_vertex_if_needed(
    polyhedron: HalfEdgePolyhedron,
    vertex: Vertex,
) -> None:
    if vertex not in polyhedron.vertices:
        return
    outgoing = [
        half_edge for half_edge in polyhedron.half_edges if half_edge.origin is vertex
    ]
    if len(outgoing) != 2:
        return
    first, second = outgoing
    first_previous = _previous_in_face(first)
    second_previous = _previous_in_face(second)
    if first_previous is None or second_previous is None:
        return
    if first_previous.opposite is not second or second_previous.opposite is not first:
        return
    if first.next is None or second.next is None:
        return

    first_previous.next = first.next
    second_previous.next = second.next
    if first.face is not None and first.face.half_edge is first:
        first.face.half_edge = first.next
    if second.face is not None and second.face.half_edge is second:
        second.face.half_edge = second.next

    first_previous.opposite = second_previous
    second_previous.opposite = first_previous

    polyhedron.half_edges = [
        half_edge
        for half_edge in polyhedron.half_edges
        if half_edge not in (first, second)
    ]
    polyhedron.vertices = [
        candidate for candidate in polyhedron.vertices if candidate is not vertex
    ]


def _previous_in_face(target: HalfEdge) -> HalfEdge | None:
    if target.face is None or target.face.half_edge is None:
        return None
    half_edge = target.face.half_edge
    for _ in range(512):
        if half_edge.next is target:
            return half_edge
        if half_edge.next is None:
            return None
        half_edge = half_edge.next
        if half_edge is target.face.half_edge:
            break
    return None


def _face_size(start: HalfEdge) -> int:
    if start.face is None or start.face.half_edge is None:
        return 0
    count = 0
    half_edge = start.face.half_edge
    for _ in range(512):
        count += 1
        if half_edge.next is None:
            return 0
        half_edge = half_edge.next
        if half_edge is start.face.half_edge:
            return count
    return 0


def _refresh_outgoing(
    polyhedron: HalfEdgePolyhedron,
    *,
    preferred: Vertex | None = None,
) -> None:
    by_origin: dict[int, list[HalfEdge]] = {}
    for half_edge in polyhedron.half_edges:
        by_origin.setdefault(half_edge.origin.id, []).append(half_edge)

    for vertex in polyhedron.vertices:
        outgoing = by_origin.get(vertex.id, [])
        vertex.outgoing = (
            min(outgoing, key=lambda half_edge: half_edge.id) if outgoing else None
        )
    if preferred is not None and preferred.id in by_origin:
        preferred.outgoing = min(
            by_origin[preferred.id],
            key=lambda half_edge: half_edge.id,
        )
