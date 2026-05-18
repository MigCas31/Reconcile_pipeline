"""Candidate-first building-part selector for tier payload envelopes.

The selector deliberately separates three decisions that were previously
coupled:

* collect room floor cells,
* aggregate roof/ceiling plane evidence over footprint/wing domains,
* merge labelled cells into strict ``EnvelopeCandidate`` inputs.

This is a deterministic v1 inspired by PolyFit/Kinetic Shape Reconstruction:
over-generate plausible planar support, then choose a compact supported set.
It does not introduce an ILP/min-cut dependency.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any, Literal

import numpy as np
from shapely import set_precision
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split, unary_union

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron import payload_adapter as pa
from reconcile_tiers.polyhedron.face_selection import (
    FaceSelectionResult,
    assemble_polyhedron,
    build_edge_incidence,
    face_selection_trace,
    generate_candidates,
    generate_tile_candidates,
    solve_face_selection_ilp,
)
from reconcile_tiers.polyhedron.priors import SelectionWeights

TopLabel = Literal[
    "flat-ceiling",
    "single-oblique",
    "gable-pair",
    "multi-piece",
    "ambiguous-top",
    "no-supported-top",
]

MIN_CELL_AREA_M2 = 1.0
DIAGNOSTIC_TOP_SUPPORT_GATE = 0.35
STRICT_TOP_SUPPORT_GATE = 0.60
COHERENT_GABLE_SUPPORT_GATE = 0.55
COHERENT_GABLE_SIDE_BALANCE_GATE = 0.35
TOP_ABOVE_FLOOR_SLACK_M = 0.10
ROOM_COVERED_RATIO = 0.80
ADJACENCY_TOL_M = 0.08
MIN_SHARED_BOUNDARY_M = 0.20
DOMAIN_GABLE_FRAME_SUPPORT_RATIO = 0.25
MIN_GABLE_SIDE_SUPPORT_RATIO = 0.10
MIN_GABLE_CELL_SUPPORT_RATIO = 0.10
MIN_REALISTIC_CEILING_CLEARANCE_M = 1.60
V2_MIN_EMIT_COVERAGE_RATIO = 0.95
V2_MIN_ASSEMBLY_COVERAGE_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class PlaneGroupEvidence:
    key: tuple[float, float, float, float]
    label: TopLabel
    representative: pa.PayloadFace
    footprint: Polygon
    domain_index: int
    domain_area_m2: float
    support_area_m2: float
    support_ratio: float
    source_locators: tuple[str, ...]
    stories: tuple[int, ...]
    source_room_indices: tuple[int, ...]
    source_face_room_indices: tuple[int, ...]
    inclination_deg: float


@dataclass(frozen=True, slots=True)
class GablePairEvidence:
    signature: tuple[Any, ...]
    left: PlaneGroupEvidence
    right: PlaneGroupEvidence
    footprint: Polygon
    domain_index: int
    support_area_m2: float
    support_ratio: float
    ridge_line: LineString


@dataclass(frozen=True, slots=True)
class DomainTopChoice:
    domain_index: int
    label: TopLabel
    signature: tuple[Any, ...]
    faces: tuple[pa.PayloadFace, ...]
    footprint: Polygon
    score: float
    support_ratio: float
    source_room_ratio: float
    reason: str


@dataclass(frozen=True, slots=True)
class PlanCell:
    cell_id: str
    story: int
    polygon: Polygon
    source_room_indices: tuple[int, ...]
    floor_y: float
    exposed_ratio: float = 1.0
    source_room_area_m2: float = 0.0


@dataclass(frozen=True, slots=True)
class SelectedTop:
    label: TopLabel
    signature: tuple[Any, ...]
    faces: tuple[pa.PayloadFace, ...]
    score: float
    local_coverage: float
    part_support_ratio: float
    reason: str


@dataclass(frozen=True, slots=True)
class SelectedCell:
    cell: PlanCell
    top: SelectedTop


@dataclass(frozen=True, slots=True)
class CellSelectorResult:
    candidates: list[pa.EnvelopeCandidate]
    cells: list[PlanCell]
    selected_cells: list[SelectedCell]
    build_attempts: list[dict[str, Any]]
    room_audit: dict[str, Any]
    plane_groups: list[PlaneGroupEvidence]
    gable_pairs: list[GablePairEvidence]
    domain_top_choices: list[DomainTopChoice]
    top_label_summary: dict[str, int]


@dataclass(frozen=True, slots=True)
class CellSelectorV2Result:
    candidates: list[pa.EnvelopeCandidate]
    domain_traces: list[dict[str, Any]]
    selector: str = "new"


def payload_cell_selector_candidates(
    payload: dict[str, Any],
    *,
    footprint: Polygon,
    ceiling_faces: list[pa.PayloadFace],
    min_top_overlap_ratio: float,
    corner_tol: float,
) -> list[pa.EnvelopeCandidate]:
    """Return strict envelope candidates from the cell selector path."""

    result = select_payload_cells(
        payload,
        footprint=footprint,
        ceiling_faces=ceiling_faces,
        min_top_overlap_ratio=min_top_overlap_ratio,
        corner_tol=corner_tol,
    )
    return result.candidates


def select_payload_cells_v2(
    payload: dict[str, Any],
    *,
    footprint: Polygon,
    ceiling_faces: list[pa.PayloadFace],
    corner_tol: float = 0.02,
    time_budget_seconds: float = 1.0,
    max_intersections: int = 50_000,
    max_candidates: int = 2_000,
    weights: SelectionWeights | None = None,
) -> CellSelectorV2Result:
    """Run the experimental Stage A selector per support domain.

    This remains diagnostic-only for now: emitted ``EnvelopeCandidate`` objects
    are returned to the caller for validation/probing, but this path is not
    wired into production selection.
    """

    domains, plane_groups, domain_story = _v2_story_aware_decomposition(
        payload,
        footprint,
        ceiling_faces,
        corner_tol=corner_tol,
    )
    scan_points = _v2_scan_points_from_payload(payload, corner_tol=corner_tol)
    rooms_by_story: dict[Any, list[dict[str, Any]]] = {}
    for room in payload.get("rooms") or []:
        rooms_by_story.setdefault(room.get("story"), []).append(room)
    traces: list[dict[str, Any]] = []
    traces_candidate_list: list[pa.EnvelopeCandidate] = []
    for domain_index, domain in enumerate(domains):
        domain_groups = [
            group for group in plane_groups if group.domain_index == domain_index
        ]
        if not domain_groups:
            traces.append(
                {
                    "domain_index": domain_index,
                    "status": "skipped",
                    "reason": "no_plane_groups",
                    "domain_area": float(domain.area),
                    "domain_polygon": _v2_domain_polygon_json(domain),
                }
            )
            continue
        try:
            (
                planes,
                labels,
                bounding_prism,
                plane_group_trace,
                plane_support_ratios,
                plane_footprints,
            ) = _v2_planes_for_domain(
                payload,
                domain,
                domain_groups,
                corner_tol=corner_tol,
                story=domain_story[domain_index],
            )
            # Primary path (§12 IMPLEMENTATION_PLAN.md): tile-based candidates
            # straight from tier_payload polygons. Falls through to the
            # synthesis path only when the tile pool is too small to form a
            # closed polyhedron — kept as a temporary safety net while the
            # synthesis code is being deleted.
            tile_floor_y = _v2_floor_y_for_domain(
                payload,
                domain,
                corner_tol=corner_tol,
                story=domain_story[domain_index],
            )
            candidates = generate_tile_candidates(
                payload,
                domain_polygon=domain,
                domain_story=domain_story[domain_index],
                domain_floor_y=tile_floor_y,
                rooms_by_story=rooms_by_story,
                corner_tol=corner_tol,
            )
            if not candidates:
                candidates = generate_candidates(
                    planes,
                    domain_polygon=domain,
                    bounding_prism=bounding_prism,
                    scan_points=np.empty((0, 3), dtype=float),
                    confidence_labels=labels,
                    max_intersections=max_intersections,
                    plane_support_ratios=plane_support_ratios,
                    plane_footprints=plane_footprints,
                )
            incidence = build_edge_incidence(candidates)
            selection = solve_face_selection_ilp(
                candidates,
                incidence,
                weights=weights,
                time_budget_seconds=time_budget_seconds,
                max_candidates=max_candidates,
            )
            candidate = _v2_envelope_candidate_from_selection(
                selection,
                domain_index=domain_index,
                domain=domain,
            )
            assembly_eligible = _v2_selection_is_assembly_eligible(
                selection,
                domain_index=domain_index,
            )
            trace = face_selection_trace(candidates, incidence, selection)
            trace.update(
                {
                    "domain_index": domain_index,
                    "status": "ok",
                    "domain_area": float(domain.area),
                    "domain_polygon": _v2_domain_polygon_json(domain),
                    "plane_count": len(planes),
                    "labels": labels,
                    "plane_groups": plane_group_trace,
                    "emitted_candidate": candidate.locator_id
                    if candidate is not None
                    else None,
                    "assembly_eligible_candidate": (
                        f"envelope-v2-domain:{domain_index}"
                        if assembly_eligible
                        else None
                    ),
                }
            )
            if candidate is not None:
                traces_candidate_list.append(candidate)
            traces.append(trace)
        except Exception as exc:
            traces.append(
                {
                    "domain_index": domain_index,
                    "status": "error",
                    "domain_area": float(domain.area),
                    "domain_polygon": _v2_domain_polygon_json(domain),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
    return CellSelectorV2Result(candidates=traces_candidate_list, domain_traces=traces)


def _v2_domain_polygon_json(domain: Polygon) -> list[list[float]]:
    return [[float(x), float(z)] for x, z in domain.exterior.coords[:-1]]


def _v2_story_aware_decomposition(
    payload: dict[str, Any],
    footprint: Polygon,
    ceiling_faces: list[pa.PayloadFace],
    *,
    corner_tol: float,
) -> tuple[list[Polygon], list[PlaneGroupEvidence], list[Any]]:
    """Decompose the building into per-(story, wing) domains.

    Without this, every upper-story domain inherits the basement floor (all
    rooms project onto the same XZ wings), producing prisms whose walls
    extend through the lower story. Splitting by story gives each domain the
    correct floor height and prevents the lower-story footprint from
    bleeding into the upper-story top extrusion.
    """
    from dataclasses import replace as dc_replace

    rooms_by_story: dict[Any, list[dict[str, Any]]] = {}
    for room in payload.get("rooms") or []:
        rooms_by_story.setdefault(room.get("story"), []).append(room)

    if not rooms_by_story:
        return ([], [], [])

    all_domains: list[Polygon] = []
    all_groups: list[PlaneGroupEvidence] = []
    domain_story: list[Any] = []

    stories = sorted(
        rooms_by_story.keys(),
        key=lambda s: (s is None, s if s is not None else 0),
    )
    for story_id in stories:
        story_rooms = rooms_by_story[story_id]
        story_footprint = _v2_story_footprint(
            story_rooms, footprint, corner_tol=corner_tol
        )
        if story_footprint is None or story_footprint.is_empty:
            continue
        story_domains = _support_domains(
            payload, story_footprint, corner_tol=corner_tol
        )
        # `payload.ceiling[]` entries arrive with story=None. Assign each to
        # the story whose floor sits below its plane (the one it physically
        # caps) by y-distance. Without this, every story sees every ceiling
        # — producing phantom oblique tops at wrong y-elevations
        # (2026-05-10: spiky orange polygons in viewer screenshots).
        story_ceilings: list[pa.PayloadFace] = []
        for face in ceiling_faces:
            if face.story == story_id:
                story_ceilings.append(face)
                continue
            if face.story is None and _ceiling_belongs_to_story(
                face, story_id, rooms_by_story, corner_tol=corner_tol
            ):
                story_ceilings.append(face)
        story_groups = _plane_groups_for_domains(
            payload,
            story_domains,
            story_ceilings,
            corner_tol=corner_tol,
        )
        offset = len(all_domains)
        rebased = [
            dc_replace(group, domain_index=group.domain_index + offset)
            for group in story_groups
        ]
        all_domains.extend(story_domains)
        all_groups.extend(rebased)
        domain_story.extend([story_id] * len(story_domains))

    return all_domains, all_groups, domain_story


def _ceiling_belongs_to_story(
    face: pa.PayloadFace,
    story_id: Any,
    rooms_by_story: Mapping[Any, list[dict[str, Any]]],
    *,
    corner_tol: float,
) -> bool:
    """Assign a story-less ceiling face to the closest story whose floor sits
    just below the ceiling's average y. Used to bucket
    ``payload.ceiling[]`` entries (which carry story=None) into per-story
    plane vocabularies.
    """
    ys = [c[1] for c in face.corners] if face.corners else []
    if not ys:
        return False
    ceiling_y = float(sum(ys) / len(ys))

    def story_floor_y(s: Any) -> float | None:
        floors_y = []
        for room in rooms_by_story.get(s, []):
            for floor in room.get("floor", []) or []:
                corners = pa._clean_ring(pa._corners(floor), tol=corner_tol)
                if corners:
                    floors_y.append(
                        float(sum(c[1] for c in corners) / len(corners))
                    )
        if not floors_y:
            return None
        return float(sum(floors_y) / len(floors_y))

    candidate_floor = story_floor_y(story_id)
    if candidate_floor is None:
        return False
    if candidate_floor > ceiling_y:
        return False  # ceiling is below this story's floor — not its lid
    # Pick the story whose floor is closest below the ceiling.
    best_story = None
    best_gap = float("inf")
    for s in rooms_by_story:
        sf = story_floor_y(s)
        if sf is None or sf > ceiling_y:
            continue
        gap = ceiling_y - sf
        if gap < best_gap:
            best_gap = gap
            best_story = s
    return best_story == story_id


def _v2_story_footprint(
    rooms: list[dict[str, Any]],
    building_footprint: Polygon,
    *,
    corner_tol: float,
) -> Polygon | None:
    polys: list[Polygon] = []
    for room in rooms:
        for floor in room.get("floor", []) or []:
            corners = pa._clean_ring(pa._corners(floor), tol=corner_tol)
            poly = pa._polygon_xz(corners)
            if poly is None or poly.is_empty:
                continue
            polys.append(poly)
    if not polys:
        return None
    union = unary_union(polys)
    # Restrict to the global building footprint so wings stay coherent.
    try:
        clipped = union.intersection(building_footprint)
    except Exception:
        clipped = union
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)
    if not isinstance(clipped, Polygon):
        return None
    return clipped


def _v2_scan_points_from_payload(
    payload: dict[str, Any],
    *,
    corner_tol: float,
) -> np.ndarray:
    """Pull raw scan-derived points from the tier_payload as evidence for the
    PolyFit data-fit term. Sources: the per-building ``ceiling[]`` polygons
    (raw scan ceiling tiles), ``visual_shells[]`` (gable-end / oblique scan
    shells), plus per-room floor and wall corners. Each returned row is an
    (x, y, z) point used by ``_supporting_points`` to count per-face support.
    """
    pts: list[tuple[float, float, float]] = []
    for ceiling in payload.get("ceiling", []) or []:
        for corner in ceiling.get("corners", []) or []:
            try:
                pts.append(
                    (float(corner["x"]), float(corner["y"]), float(corner["z"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    for shell in payload.get("visual_shells", []) or []:
        for corner in shell.get("corners", []) or []:
            try:
                pts.append(
                    (float(corner["x"]), float(corner["y"]), float(corner["z"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    for room in payload.get("rooms", []) or []:
        for kind in ("floor", "walls"):
            for piece in room.get(kind, []) or []:
                for corner in piece.get("corners", []) or []:
                    try:
                        pts.append(
                            (
                                float(corner["x"]),
                                float(corner["y"]),
                                float(corner["z"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


def select_payload_cells(
    payload: dict[str, Any],
    *,
    footprint: Polygon,
    ceiling_faces: list[pa.PayloadFace],
    min_top_overlap_ratio: float = STRICT_TOP_SUPPORT_GATE,
    corner_tol: float = 0.02,
) -> CellSelectorResult:
    """Build labelled plan cells and strict envelope candidates."""

    domains = _support_domains(payload, footprint, corner_tol=corner_tol)
    plane_groups = _plane_groups_for_domains(
        payload,
        domains,
        ceiling_faces,
        corner_tol=corner_tol,
    )
    gable_pairs = _gable_pairs_for_domains(domains, plane_groups)
    room_polys = _room_polygons(payload, corner_tol=corner_tol)
    domain_top_choices = _select_domain_top_choices(
        domains,
        plane_groups,
        gable_pairs,
        room_polys=room_polys,
    )
    exposed_masks = _exposed_masks_by_room(payload, footprint, corner_tol=corner_tol)
    cells = _plan_cells_for_rooms(
        payload,
        footprint,
        gable_pairs,
        plane_groups=plane_groups,
        exposed_masks=exposed_masks,
        corner_tol=corner_tol,
    )
    selected = [
        selected
        for cell in cells
        if (
            selected := _select_top_for_cell(
                cell,
                domains=domains,
                plane_groups=plane_groups,
                gable_pairs=gable_pairs,
                domain_top_choices=domain_top_choices,
                exposed_masks=exposed_masks,
            )
        )
        is not None
    ]
    candidates, build_attempts = _candidates_from_selected_cells(
        payload,
        selected,
        ceiling_faces=ceiling_faces,
        min_top_overlap_ratio=min_top_overlap_ratio,
        corner_tol=corner_tol,
        plane_groups=plane_groups,
    )
    room_audit = _room_audit(
        payload,
        cells=cells,
        selected_cells=selected,
        candidates=candidates,
        ceiling_faces=ceiling_faces,
        footprint=footprint,
        corner_tol=corner_tol,
        plane_groups=plane_groups,
        gable_pairs=gable_pairs,
        domain_top_choices=domain_top_choices,
    )
    top_label_summary = dict(Counter(cell.top.label for cell in selected))
    return CellSelectorResult(
        candidates=candidates,
        cells=cells,
        selected_cells=selected,
        build_attempts=build_attempts,
        room_audit=room_audit,
        plane_groups=plane_groups,
        gable_pairs=gable_pairs,
        domain_top_choices=domain_top_choices,
        top_label_summary=top_label_summary,
    )


def audit_payload_rooms(
    payload: dict[str, Any],
    *,
    footprint: Polygon | None = None,
    ceiling_faces: list[pa.PayloadFace] | None = None,
    corner_tol: float = 0.02,
) -> dict[str, Any]:
    """Return diagnostic JSON for room coverage and part-level plane evidence."""

    if footprint is None:
        footprint = pa._payload_footprint_polygon(
            payload,
            room_buffer_m=0.3,
            footprint_shrink_m=0.3,
            corner_tol=corner_tol,
        )
    if footprint is None:
        return {
            "schema_version": 1,
            "rooms": [],
            "summary": {"rooms_total": 0, "rooms_ge80": 0, "dropped_rooms": 0},
            "plane_groups": [],
            "gable_pairs": [],
            "cells": [],
        }
    if ceiling_faces is None:
        faces = pa.payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
        ceiling_faces = [face for face in faces if face.kind == "ceiling"]
    result = select_payload_cells(
        payload,
        footprint=footprint,
        ceiling_faces=ceiling_faces,
        corner_tol=corner_tol,
    )
    return _audit_json(result)


def _v2_planes_for_domain(
    payload: dict[str, Any],
    domain: Polygon,
    groups: list[PlaneGroupEvidence],
    *,
    corner_tol: float,
    story: Any = None,
) -> tuple[
    list[Plane],
    list[str],
    tuple[float, float],
    list[dict[str, Any]],
    dict[int, float],
    dict[int, Polygon],
]:
    floor_y = _v2_floor_y_for_domain(
        payload, domain, corner_tol=corner_tol, story=story
    )
    if floor_y is None:
        raise ValueError("could not infer domain floor height")
    # Composite ceilings on upper-story rooms typically have a dominant flat
    # plane (~0.7 support) with smaller oblique strips (~0.20 each); the
    # 0.35 diagnostic gate drops the obliques. The arrangement generator
    # needs them, so widen here and let downstream filtering take over.
    selected_groups = _v2_arrangement_top_groups(groups)
    group_trace = _v2_plane_group_trace(groups, selected_groups)
    top_y_values = _v2_top_y_values_for_domain(domain, selected_groups)
    if not top_y_values:
        raise ValueError("could not infer domain top height")

    planes: list[Plane] = [Plane(a=0.0, b=-1.0, c=0.0, d=-floor_y)]
    labels: list[str] = ["floor"]
    plane_support_ratios: dict[int, float] = {}
    plane_footprints: dict[int, Polygon] = {}
    for group in selected_groups:
        plane_id = len(planes)
        planes.append(group.representative.plane)
        labels.append(group.label)
        plane_support_ratios[plane_id] = float(group.support_ratio)
        # `group.footprint` is the union of supporting ceiling tiles' XZ
        # projections clipped to the domain — used by the planar-arrangement
        # composite-ceiling generator (IMPLEMENTATION_PLAN.md §11).
        plane_footprints[plane_id] = group.footprint
    if len(planes) == 1:
        raise ValueError("no supported top planes for domain")
    for index, plane in enumerate(_v2_vertical_boundary_planes(domain)):
        planes.append(plane)
        labels.append(f"domain-wall:{index}")
    return (
        planes,
        labels,
        (floor_y - 0.20, max(top_y_values) + 0.20),
        group_trace,
        plane_support_ratios,
        plane_footprints,
    )


_ARRANGEMENT_MIN_SUPPORT_RATIO = 0.10


def _v2_arrangement_top_groups(
    groups: list[PlaneGroupEvidence],
) -> list[PlaneGroupEvidence]:
    """Pick top plane groups for the composite-ceiling arrangement.

    Wider than the strict ``DIAGNOSTIC_TOP_SUPPORT_GATE`` (0.35) — composite
    upper-story ceilings often have a dominant flat plane (~0.7) with
    smaller oblique strips (~0.15–0.30) under the gable slopes. The strict
    gate drops the obliques and the v2 selector then can only pick a
    single-piece top, missing the gable. 0.10 admits the strips while
    still suppressing trivial scan noise.
    """

    supported = [
        g
        for g in groups
        if g.support_ratio >= _ARRANGEMENT_MIN_SUPPORT_RATIO
    ]
    if supported:
        return supported
    if not groups:
        return []
    return [
        max(
            groups,
            key=lambda group: (group.support_ratio, group.support_area_m2),
        )
    ]


def _v2_diagnostic_top_groups(
    groups: list[PlaneGroupEvidence],
) -> list[PlaneGroupEvidence]:
    supported = [
        group
        for group in groups
        if group.support_ratio >= DIAGNOSTIC_TOP_SUPPORT_GATE
    ]
    if supported:
        return supported
    if not groups:
        return []
    return [
        max(
            groups,
            key=lambda group: (group.support_ratio, group.support_area_m2),
        )
    ]


def _v2_plane_group_trace(
    groups: list[PlaneGroupEvidence],
    selected_groups: list[PlaneGroupEvidence],
) -> list[dict[str, Any]]:
    selected_ids = {id(group) for group in selected_groups}
    supported_ids = {
        id(group)
        for group in groups
        if group.support_ratio >= DIAGNOSTIC_TOP_SUPPORT_GATE
    }
    return [
        {
            "label": group.label,
            "support_area_m2": group.support_area_m2,
            "support_ratio": group.support_ratio,
            "domain_area_m2": group.domain_area_m2,
            "inclination_deg": group.inclination_deg,
            "source_locators": list(group.source_locators),
            "included": id(group) in selected_ids,
            "reason": (
                "support_gate"
                if id(group) in supported_ids
                else "best_available_top_fallback"
                if id(group) in selected_ids
                else "below_support_gate"
            ),
        }
        for group in groups
    ]


def _v2_envelope_candidate_from_selection(
    selection: FaceSelectionResult,
    *,
    domain_index: int,
    domain: Polygon,
) -> pa.EnvelopeCandidate | None:
    coverage_ratio = float(selection.energy_breakdown.get("coverage_ratio") or 0.0)
    if coverage_ratio < V2_MIN_EMIT_COVERAGE_RATIO:
        return None
    envelope = _v2_selection_envelope_payload(selection, domain_index=domain_index)
    if envelope is None:
        return None
    faces, top_sources = envelope
    return pa.EnvelopeCandidate(
        locator_id=f"envelope-v2-domain:{domain_index}",
        faces=faces,
        footprint_area_m2=float(domain.area),
        top_source=" + ".join(top_sources) or "polyhedron_v2",
        top_overlap_ratio=coverage_ratio,
        selector="new",
    )


def _v2_selection_is_assembly_eligible(
    selection: FaceSelectionResult,
    *,
    domain_index: int,
) -> bool:
    coverage_ratio = float(selection.energy_breakdown.get("coverage_ratio") or 0.0)
    return (
        coverage_ratio >= V2_MIN_ASSEMBLY_COVERAGE_RATIO
        and _v2_selection_envelope_payload(
            selection,
            domain_index=domain_index,
        )
        is not None
    )


_TILE_CEILING_LABELS = {
    "flat-ceiling",
    "single-oblique",
    "flat_ceiling",
    "computed_oblique",
    "merged_coplanar",
    "raw_scan",
    "gable_end",
    "gable_closure",
}


def _v2_selection_envelope_payload(
    selection: FaceSelectionResult,
    *,
    domain_index: int,
) -> tuple[list[pa.PayloadFace], list[str]] | None:
    if not selection.selected:
        return None
    # Permissive label vocabulary covering both the synthesis-path labels
    # and the tier_payload tile labels (§12).
    accepted = {"floor", "wall", *_TILE_CEILING_LABELS}
    labels = {candidate.confidence_label for candidate in selection.selected}
    if not labels or any(
        label not in accepted and not label.startswith("domain-wall:")
        for label in labels
    ):
        return None
    try:
        assemble_polyhedron(selection)
    except ValueError:
        pass
    faces: list[pa.PayloadFace] = []
    top_sources: list[str] = []
    for candidate in selection.selected:
        label = candidate.confidence_label
        if label == "floor":
            kind: pa.FaceKind = "floor"
        elif label in _TILE_CEILING_LABELS:
            kind = "ceiling"
            top_sources.append(f"{label}:plane:{candidate.plane_id}")
        elif label == "wall" or label.startswith("domain-wall:"):
            kind = "wall"
        else:
            return None
        faces.append(
            pa.PayloadFace(
                kind=kind,
                locator_id=(
                    f"envelope-v2-domain:{domain_index}"
                    f"::{label}::{candidate.face_id}"
                ),
                corners=[list(corner) for corner in candidate.corners],
                plane=candidate.plane,
                source="polyhedron_v2",
            )
        )
    return faces, top_sources


def _v2_floor_y_for_domain(
    payload: dict[str, Any],
    domain: Polygon,
    *,
    corner_tol: float,
    story: Any = None,
) -> float | None:
    """Return the most-overlapping floor y for the domain, restricted to the
    given story when provided. Without the story filter, multi-story buildings
    collapse onto the basement floor for upper-story domains because every
    upper-story room's XZ projection also overlaps the basement floor below
    it (2026-05-10 visual finding)."""

    candidates: list[tuple[float, float]] = []
    for room in payload.get("rooms", []) or []:
        if story is not None and room.get("story") != story:
            continue
        for floor in room.get("floor", []) or []:
            corners = pa._clean_ring(pa._corners(floor), tol=corner_tol)
            poly = pa._polygon_xz(corners)
            if poly is None or not corners:
                continue
            try:
                overlap = float(poly.intersection(domain).area)
            except Exception:
                continue
            if overlap <= 0.05:
                continue
            y = float(sum(corner[1] for corner in corners) / len(corners))
            candidates.append((overlap, y))
    if not candidates:
        return None
    # Pick the floor with the largest XZ overlap with the domain, ties broken
    # by lower y. This is more robust than min(y) when multiple floors overlap.
    best = max(candidates, key=lambda item: (item[0], -item[1]))
    return best[1]


def _v2_top_y_values_for_domain(
    domain: Polygon,
    groups: list[PlaneGroupEvidence],
) -> list[float]:
    coords = [(float(x), float(z)) for x, z in domain.exterior.coords[:-1]]
    values: list[float] = []
    for group in groups:
        plane = group.representative.plane
        if abs(plane.b) <= 1e-9:
            continue
        for x, z in coords:
            values.append(float((plane.d - plane.a * x - plane.c * z) / plane.b))
    return values


def _v2_vertical_boundary_planes(domain: Polygon) -> list[Plane]:
    oriented = pa.orient_polygon(domain, sign=1.0)
    coords = [(float(x), float(z)) for x, z in oriented.exterior.coords[:-1]]
    planes: list[Plane] = []
    for index, start in enumerate(coords):
        end = coords[(index + 1) % len(coords)]
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = float((dx * dx + dz * dz) ** 0.5)
        if length <= 1e-9:
            continue
        nx = dz / length
        nz = -dx / length
        planes.append(
            Plane(
                a=nx,
                b=0.0,
                c=nz,
                d=nx * start[0] + nz * start[1],
            )
        )
    return planes


def _support_domains(
    payload: dict[str, Any],
    footprint: Polygon,
    *,
    corner_tol: float,
) -> list[Polygon]:
    domains: list[Polygon] = []
    try:
        domains.extend(wing.polygon for wing in pa.decompose_to_wings(footprint))
    except Exception:
        pass
    try:
        domains.extend(
            pa._payload_room_graph_wing_polygons(
                payload,
                footprint=footprint,
                corner_tol=corner_tol,
            )
        )
    except Exception:
        pass
    domains.append(footprint)

    deduped: list[Polygon] = []
    for domain in domains:
        domain = _clean_polygon_ring(domain, tol=1e-6)
        if domain.is_empty or domain.area < MIN_CELL_AREA_M2:
            continue
        if any(
            float(domain.symmetric_difference(existing).area)
            / max(float(domain.area), float(existing.area), 1e-9)
            < 0.02
            for existing in deduped
        ):
            continue
        deduped.append(domain)
    return deduped or [_clean_polygon_ring(footprint, tol=1e-6)]


def _clean_polygon_ring(
    polygon: Polygon,
    *,
    tol: float,
    collinear_tol: float = 1e-4,
) -> Polygon:
    """Drop coincident and collinear consecutive vertices from the ring.

    Shapely buffer/union artifacts leave (a) duplicate vertices, which make
    `_match_domain_boundary_planes` see a zero-length edge and silently bail,
    and (b) collinear vertices that survive as distinct ring entries. Two
    collinear edges produce two same-plane wall faces during prism extrusion,
    which then trip `validate_polyhedron`'s adjacent-coplanar-faces check at
    assembly time and silently reject the domain (~62% of high-coverage
    non-emit domains, .context/zerocand-diagnosis-2026-05-10.md follow-up).
    """

    if polygon.is_empty:
        return polygon
    coords = list(polygon.exterior.coords)
    if len(coords) < 4:
        return polygon
    cleaned: list[tuple[float, float]] = []
    for x, y in coords[:-1]:
        if cleaned:
            px, py = cleaned[-1]
            if (x - px) * (x - px) + (y - py) * (y - py) <= tol * tol:
                continue
        cleaned.append((float(x), float(y)))
    while len(cleaned) >= 2:
        x0, y0 = cleaned[0]
        xn, yn = cleaned[-1]
        if (x0 - xn) * (x0 - xn) + (y0 - yn) * (y0 - yn) <= tol * tol:
            cleaned.pop()
            continue
        break
    if len(cleaned) < 3:
        return Polygon()

    cleaned = _drop_collinear_ring_vertices(cleaned, collinear_tol=collinear_tol)
    if len(cleaned) < 3:
        return Polygon()
    return Polygon(cleaned)


def _drop_collinear_ring_vertices(
    coords: list[tuple[float, float]],
    *,
    collinear_tol: float,
) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return coords
    n = len(coords)
    keep = [True] * n
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        x0, y0 = coords[prev_i]
        x1, y1 = coords[i]
        x2, y2 = coords[next_i]
        dx_a = x1 - x0
        dy_a = y1 - y0
        dx_b = x2 - x1
        dy_b = y2 - y1
        len_a = (dx_a * dx_a + dy_a * dy_a) ** 0.5
        len_b = (dx_b * dx_b + dy_b * dy_b) ** 0.5
        if len_a <= 1e-12 or len_b <= 1e-12:
            continue
        cross = abs(dx_a * dy_b - dy_a * dx_b)
        if cross / (len_a * len_b) <= collinear_tol:
            keep[i] = False
    return [coords[i] for i in range(n) if keep[i]]


def _plane_groups_for_domains(
    payload: dict[str, Any],
    domains: list[Polygon],
    ceiling_faces: list[pa.PayloadFace],
    *,
    corner_tol: float,
) -> list[PlaneGroupEvidence]:
    room_polys = _room_polygons(payload, corner_tol=corner_tol)
    groups: list[PlaneGroupEvidence] = []
    for domain_index, domain in enumerate(domains):
        grouped: dict[
            tuple[float, float, float, float],
            list[tuple[pa.PayloadFace, Polygon]],
        ] = {}
        for face in ceiling_faces:
            if abs(face.plane.b) <= 1e-6:
                continue
            poly = pa._polygon_xz(face.corners)
            if poly is None:
                continue
            try:
                clipped = poly.intersection(domain)
            except Exception:
                continue
            if clipped.is_empty or clipped.area <= 0.05:
                continue
            grouped.setdefault(pa._top_plane_group_key(face.plane), []).append(
                (face, clipped)
            )

        for key, items in grouped.items():
            try:
                footprint = pa._largest_polygon(
                    unary_union([poly for _face, poly in items]).intersection(domain)
                )
            except Exception:
                footprint = None
            if footprint is None or footprint.area <= 0.05:
                continue
            representative = max(items, key=lambda item: float(item[1].area))[0]
            stories: set[int] = set()
            room_indices: set[int] = set()
            source_face_room_indices: set[int] = set()
            source_face_stories: set[int] = set()
            for face, _poly in items:
                if face.room_index is not None:
                    source_face_room_indices.add(face.room_index)
                if face.story is not None:
                    source_face_stories.add(face.story)
            for room_index, story, room_poly in room_polys:
                if float(room_poly.intersection(footprint).area) > 0.05:
                    room_indices.add(room_index)
                    stories.add(story)
            stories.update(source_face_stories)
            inclination = _plane_inclination_deg(representative.plane)
            label: TopLabel = (
                "single-oblique" if 5.0 < inclination < 80.0 else "flat-ceiling"
            )
            groups.append(
                PlaneGroupEvidence(
                    key=key,
                    label=label,
                    representative=representative,
                    footprint=footprint,
                    domain_index=domain_index,
                    domain_area_m2=float(domain.area),
                    support_area_m2=float(footprint.area),
                    support_ratio=float(footprint.area)
                    / max(float(domain.area), 1e-9),
                    source_locators=tuple(
                        sorted({face.locator_id for face, _poly in items})
                    ),
                    stories=tuple(sorted(stories)),
                    source_room_indices=tuple(sorted(room_indices)),
                    source_face_room_indices=tuple(sorted(source_face_room_indices)),
                    inclination_deg=inclination,
                )
            )
    return _clip_oblique_groups_to_ridge_ownership(domains, groups)


def _clip_oblique_groups_to_ridge_ownership(
    domains: list[Polygon],
    plane_groups: list[PlaneGroupEvidence],
) -> list[PlaneGroupEvidence]:
    """Limit oblique finite support to the side of a valid opposing roof plane.

    A gable plane can geometrically continue after the ridge, but physically it
    does not own that area. PolyFit/Kinetic-style assembly treats finite support
    and partition ownership separately; this is the lightweight equivalent for
    our XZ selector.
    """

    by_domain: dict[int, list[PlaneGroupEvidence]] = {}
    for group in plane_groups:
        if group.label == "single-oblique":
            by_domain.setdefault(group.domain_index, []).append(group)

    clipped: list[PlaneGroupEvidence] = []
    for group in plane_groups:
        if group.label != "single-oblique":
            clipped.append(group)
            continue
        domain = domains[group.domain_index]
        owned: Polygon | None = group.footprint
        constraints = 0
        for other in by_domain.get(group.domain_index, []):
            if other.key == group.key:
                continue
            if (
                min(group.support_ratio, other.support_ratio)
                < MIN_GABLE_SIDE_SUPPORT_RATIO
            ):
                continue
            if not pa._roof_planes_are_opposing(
                group.representative.plane,
                other.representative.plane,
            ):
                continue
            line = pa._ridge_split_line_for_planes(
                domain,
                pa._top_plane_up(group.representative.plane),
                pa._top_plane_up(other.representative.plane),
            )
            if line is None:
                continue
            lower_side = _lower_side_polygon_for_group(domain, group, other)
            if lower_side is None or lower_side.area <= 0.05:
                continue
            try:
                next_owned = pa._largest_polygon(owned.intersection(lower_side))
            except Exception:
                next_owned = None
            if next_owned is None or next_owned.area <= 0.05:
                continue
            owned = next_owned
            constraints += 1
        if constraints == 0 or owned is None:
            clipped.append(group)
            continue
        clipped.append(_replace_group_footprint(group, owned))
    return clipped


def _replace_group_footprint(
    group: PlaneGroupEvidence,
    footprint: Polygon,
) -> PlaneGroupEvidence:
    support_area = float(footprint.area)
    return PlaneGroupEvidence(
        key=group.key,
        label=group.label,
        representative=group.representative,
        footprint=footprint,
        domain_index=group.domain_index,
        domain_area_m2=group.domain_area_m2,
        support_area_m2=support_area,
        support_ratio=support_area / max(group.domain_area_m2, 1e-9),
        source_locators=group.source_locators,
        stories=group.stories,
        source_room_indices=group.source_room_indices,
        source_face_room_indices=group.source_face_room_indices,
        inclination_deg=group.inclination_deg,
    )


def _gable_pairs_for_domains(
    domains: list[Polygon],
    plane_groups: list[PlaneGroupEvidence],
) -> list[GablePairEvidence]:
    pairs: list[GablePairEvidence] = []
    by_domain: dict[int, list[PlaneGroupEvidence]] = {}
    for group in plane_groups:
        if group.label == "single-oblique":
            by_domain.setdefault(group.domain_index, []).append(group)

    for domain_index, groups in by_domain.items():
        domain = domains[domain_index]
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                if (
                    min(left.support_ratio, right.support_ratio)
                    < MIN_GABLE_SIDE_SUPPORT_RATIO
                ):
                    continue
                if not pa._roof_planes_are_opposing(
                    left.representative.plane,
                    right.representative.plane,
                ):
                    continue
                line = pa._ridge_split_line_for_planes(
                    domain,
                    pa._top_plane_up(left.representative.plane),
                    pa._top_plane_up(right.representative.plane),
                )
                if line is None:
                    continue
                try:
                    footprint = unary_union([left.footprint, right.footprint])
                    footprint = footprint.intersection(domain)
                except Exception:
                    continue
                if footprint.is_empty or footprint.area <= MIN_CELL_AREA_M2:
                    continue
                support_ratio = float(footprint.area) / max(float(domain.area), 1e-9)
                pairs.append(
                    GablePairEvidence(
                        signature=("gable-pair", *sorted((left.key, right.key))),
                        left=left,
                        right=right,
                        footprint=footprint,
                        domain_index=domain_index,
                        support_area_m2=float(footprint.area),
                        support_ratio=support_ratio,
                        ridge_line=line,
                    )
                )
    return sorted(pairs, key=lambda pair: pair.support_area_m2, reverse=True)


def _select_domain_top_choices(
    domains: list[Polygon],
    plane_groups: list[PlaneGroupEvidence],
    gable_pairs: list[GablePairEvidence],
    *,
    room_polys: list[tuple[int, int, Polygon]],
) -> list[DomainTopChoice]:
    """Select one roof/envelope hypothesis per support domain.

    This is the deterministic, no-solver equivalent of a PolyFit-style global
    model selection step: evaluate complete domain hypotheses first, then let
    cells inherit the chosen domain top. Interior flat ceilings can still be
    used for non-exposed cells, but they do not erase a selected roof domain.
    """

    choices: list[DomainTopChoice] = []
    groups_by_domain: dict[int, list[PlaneGroupEvidence]] = {}
    pairs_by_domain: dict[int, list[GablePairEvidence]] = {}
    for group in plane_groups:
        groups_by_domain.setdefault(group.domain_index, []).append(group)
    for pair in gable_pairs:
        pairs_by_domain.setdefault(pair.domain_index, []).append(pair)

    for domain_index, domain in enumerate(domains):
        domain_rooms = _domain_room_indices(domain, room_polys)
        candidates: list[DomainTopChoice] = []
        for pair in pairs_by_domain.get(domain_index, []):
            coherent_near_gate = (
                pair.support_ratio < STRICT_TOP_SUPPORT_GATE
                and _gable_pair_is_coherent_for_domain(pair, domain)
            )
            if (
                pair.support_ratio < STRICT_TOP_SUPPORT_GATE
                and not coherent_near_gate
            ):
                continue
            source_rooms = set(pair.left.source_face_room_indices).union(
                pair.right.source_face_room_indices
            )
            source_ratio = _source_room_ratio(source_rooms, domain_rooms)
            side_support = min(pair.left.support_ratio, pair.right.support_ratio)
            score = (
                pair.support_ratio * 6.0
                + source_ratio * 4.0
                + side_support * 2.0
                - 0.5
            )
            if coherent_near_gate:
                score += 0.75
            candidates.append(
                DomainTopChoice(
                    domain_index=domain_index,
                    label="gable-pair",
                    signature=pair.signature,
                    faces=(pair.left.representative, pair.right.representative),
                    footprint=pair.footprint,
                    score=score,
                    support_ratio=pair.support_ratio,
                    source_room_ratio=source_ratio,
                    reason="domain_coherent_gable_pair"
                    if coherent_near_gate
                    else "domain_gable_pair",
                )
            )
        for group in groups_by_domain.get(domain_index, []):
            source_ratio = _source_room_ratio(
                set(group.source_face_room_indices),
                domain_rooms,
            )
            if group.label == "flat-ceiling":
                if (
                    group.support_ratio < DIAGNOSTIC_TOP_SUPPORT_GATE
                    and source_ratio <= 0.0
                ):
                    continue
                score = (
                    group.support_ratio * 4.0
                    + source_ratio * 4.0
                    + min(group.support_area_m2 / 20.0, 1.0)
                )
            else:
                if group.support_ratio < DIAGNOSTIC_TOP_SUPPORT_GATE:
                    continue
                score = (
                    group.support_ratio * 5.0
                    + source_ratio * 4.0
                    + min(group.support_area_m2 / 20.0, 1.0)
                )
            candidates.append(
                DomainTopChoice(
                    domain_index=domain_index,
                    label=group.label,
                    signature=(group.label, group.key),
                    faces=(group.representative,),
                    footprint=group.footprint,
                    score=score,
                    support_ratio=group.support_ratio,
                    source_room_ratio=source_ratio,
                    reason=f"domain_{group.label}",
                )
            )
        if candidates:
            choices.append(max(candidates, key=lambda candidate: candidate.score))
    return choices


def _gable_pair_is_coherent_for_domain(
    pair: GablePairEvidence,
    domain: Polygon,
) -> bool:
    if pair.support_ratio < COHERENT_GABLE_SUPPORT_GATE:
        return False
    top = SelectedTop(
        label="gable-pair",
        signature=pair.signature,
        faces=(pair.left.representative, pair.right.representative),
        score=0.0,
        local_coverage=1.0,
        part_support_ratio=pair.support_ratio,
        reason="domain_gable_pair",
    )
    diagnostic = _gable_footprint_coherence_json(domain, top)
    if diagnostic is None or diagnostic.get("status") != "ok":
        return False
    return (
        diagnostic.get("uses_both_roof_faces") is True
        and int(diagnostic.get("split_region_count") or 0) == 2
        and int(diagnostic.get("fragmented_side_count") or 0) == 0
        and float(diagnostic.get("side_area_balance") or 0.0)
        >= COHERENT_GABLE_SIDE_BALANCE_GATE
    )


def _domain_room_indices(
    domain: Polygon,
    room_polys: list[tuple[int, int, Polygon]],
) -> set[int]:
    out: set[int] = set()
    for room_index, _story, room_poly in room_polys:
        try:
            overlap_ratio = float(room_poly.intersection(domain).area) / max(
                float(room_poly.area),
                1e-9,
            )
        except Exception:
            continue
        if overlap_ratio >= 0.50:
            out.add(room_index)
    return out


def _source_room_ratio(source_rooms: set[int], domain_rooms: set[int]) -> float:
    if not domain_rooms:
        return 0.0
    return len(source_rooms.intersection(domain_rooms)) / len(domain_rooms)


def _plan_cells_for_rooms(
    payload: dict[str, Any],
    footprint: Polygon,
    gable_pairs: list[GablePairEvidence],
    *,
    plane_groups: list[PlaneGroupEvidence],
    exposed_masks: dict[int, Polygon],
    corner_tol: float,
) -> list[PlanCell]:
    cells: list[PlanCell] = []
    for room_index, room in enumerate(payload.get("rooms", []) or []):
        story = pa._int_or_none(room.get("story")) or 0
        room_floor_y = None
        for floor in room.get("floor", []) or []:
            corners = pa._clean_ring(pa._corners(floor), tol=corner_tol)
            poly = pa._polygon_xz(corners)
            if poly is None:
                continue
            try:
                clipped = pa._largest_polygon(poly.intersection(footprint))
            except Exception:
                clipped = None
            if clipped is None or clipped.area < MIN_CELL_AREA_M2:
                continue
            room_floor_y = float(sum(c[1] for c in corners) / len(corners))
            parts = [clipped]
            for pair in gable_pairs:
                next_parts: list[Polygon] = []
                for part in parts:
                    if (
                        pair.support_ratio < DOMAIN_GABLE_FRAME_SUPPORT_RATIO
                        and float(part.intersection(pair.footprint).area) <= 0.05
                    ):
                        next_parts.append(part)
                        continue
                    try:
                        split_parts = [
                            p
                            for p in split(part, pair.ridge_line).geoms
                            if isinstance(p, Polygon) and p.area > 0.01
                        ]
                    except Exception:
                        split_parts = []
                    if split_parts:
                        split_parts = _merge_tiny_split_fragments(part, split_parts)
                    next_parts.extend(split_parts or [part])
                parts = next_parts
            parts = _split_parts_by_support_footprints(
                parts,
                room_index=room_index,
                plane_groups=plane_groups,
            )
            cell_index = 0
            for part in parts:
                exposure_parts = _split_cell_by_exposure(
                    part,
                    room_index=room_index,
                    exposed_masks=exposed_masks,
                )
                if not exposure_parts:
                    exposure_parts = [
                        (
                            part,
                            _cell_exposed_ratio(
                                part,
                                room_index=room_index,
                                exposed_masks=exposed_masks,
                            ),
                        )
                    ]
                for exposure_part, exposed_ratio in exposure_parts:
                    if exposure_part.area < MIN_CELL_AREA_M2:
                        continue
                    cells.append(
                        PlanCell(
                            cell_id=f"room:{room_index}:cell:{cell_index}",
                            story=story,
                            polygon=exposure_part,
                            source_room_indices=(room_index,),
                            floor_y=room_floor_y,
                            exposed_ratio=exposed_ratio,
                            source_room_area_m2=float(clipped.area),
                        )
                    )
                    cell_index += 1
            break
    return cells


def _split_parts_by_support_footprints(
    parts: list[Polygon],
    *,
    room_index: int,
    plane_groups: list[PlaneGroupEvidence],
) -> list[Polygon]:
    relevant_groups = [
        group
        for group in plane_groups
        if room_index in group.source_room_indices
        or room_index in group.source_face_room_indices
    ]
    if not relevant_groups:
        return parts

    out = parts
    for group in sorted(
        relevant_groups,
        key=lambda item: (item.label != "flat-ceiling", -item.support_area_m2),
    ):
        next_parts: list[Polygon] = []
        for part in out:
            try:
                inside = part.intersection(group.footprint)
                outside = part.difference(group.footprint)
            except Exception:
                next_parts.append(part)
                continue
            components = [
                *[
                    poly
                    for poly in _polygon_components(inside)
                    if poly.area > 0.01
                ],
                *[
                    poly
                    for poly in _polygon_components(outside)
                    if poly.area > 0.01
                ],
            ]
            if components:
                next_parts.extend(_merge_tiny_split_fragments(part, components))
            else:
                next_parts.append(part)
        out = next_parts
    return out


def _merge_tiny_split_fragments(
    original: Polygon,
    pieces: list[Polygon],
) -> list[Polygon]:
    """Preserve room coverage when partition lines create sub-1m2 slivers.

    ``MIN_CELL_AREA_M2`` is a good lower bound for standalone cells, but using
    it while splitting can delete many small fragments from the same room. Those
    fragments add up to visible missing rooms. Merge them back into the adjacent
    retained fragment so the partition stays compact without punching holes in
    the source floor polygon.
    """

    cleaned = [
        poly
        for piece in pieces
        for poly in _polygon_components(piece.buffer(0))
        if poly.area > 0.01
    ]
    if not cleaned:
        return [original]

    large = [poly for poly in cleaned if poly.area >= MIN_CELL_AREA_M2]
    tiny = [poly for poly in cleaned if poly.area < MIN_CELL_AREA_M2]
    if not tiny:
        return large
    if not large:
        return [original] if original.area >= MIN_CELL_AREA_M2 else cleaned

    for fragment in sorted(tiny, key=lambda poly: poly.area, reverse=True):
        best_index = max(
            range(len(large)),
            key=lambda index: (
                _shared_boundary_length(large[index], fragment),
                -float(large[index].distance(fragment)),
                large[index].area,
            ),
        )
        try:
            merged = _safe_unary_union([large[best_index], fragment])
        except Exception:
            continue
        merged_parts = _polygon_components(merged)
        if len(merged_parts) == 1:
            large[best_index] = merged_parts[0]
        else:
            largest = max(merged_parts, key=lambda poly: poly.area)
            if largest.area >= large[best_index].area + fragment.area * 0.95:
                large[best_index] = largest
    return large


def _shared_boundary_length(left: Polygon, right: Polygon) -> float:
    try:
        return float(left.boundary.intersection(right.boundary).length)
    except Exception:
        return 0.0


def _select_top_for_cell(
    cell: PlanCell,
    *,
    domains: list[Polygon],
    plane_groups: list[PlaneGroupEvidence],
    gable_pairs: list[GablePairEvidence],
    domain_top_choices: list[DomainTopChoice],
    exposed_masks: dict[int, Polygon],
) -> SelectedCell | None:
    _ = exposed_masks
    domain_index = _best_domain_index(cell.polygon, domains)
    if cell.exposed_ratio >= 0.20:
        domain_choice = _domain_choice_for_cell(
            cell,
            domain_index=domain_index,
            domain_top_choices=domain_top_choices,
        )
        if domain_choice is not None:
            selected = _selected_cell_from_domain_choice(cell, domain_choice)
            if selected is not None:
                return selected

    candidates: list[SelectedTop] = []
    dominant_gable = _dominant_gable_pair_for_cell(
        cell,
        domain_index=domain_index,
        gable_pairs=gable_pairs,
    )

    for group in plane_groups:
        if dominant_gable is not None and group.label == "single-oblique":
            continue
        side_coverage = _lower_side_coverage_for_group(
            cell.polygon,
            group,
            gable_pairs,
        )
        if group.label == "single-oblique" and side_coverage < 0.50:
            continue
        if group.label == "single-oblique" and cell.exposed_ratio < 0.20:
            continue
        local_coverage = _safe_overlap_ratio(cell.polygon, group.footprint)
        can_promote_local_flat = _can_promote_local_flat_ceiling(
            cell,
            group,
            local_coverage=local_coverage,
        )
        can_fill_gable_owned_side = (
            group.label == "single-oblique"
            and side_coverage >= ROOM_COVERED_RATIO
            and group.support_ratio >= DOMAIN_GABLE_FRAME_SUPPORT_RATIO
        )
        if (
            local_coverage < 0.05
            and not can_fill_gable_owned_side
            and not can_promote_local_flat
        ):
            continue
        if group.label == "single-oblique" and not _single_oblique_topology_allowed(
            cell,
            group,
            local_coverage=local_coverage,
            side_coverage=side_coverage,
            gable_pairs=gable_pairs,
        ):
            continue
        top_plane = pa._top_plane_up(group.representative.plane)
        if not _top_is_above_cell(cell, [(cell.polygon, top_plane)]):
            continue
        reason = "candidate_covered"
        if can_promote_local_flat and local_coverage < DIAGNOSTIC_TOP_SUPPORT_GATE:
            reason = "weak_room_support_promoted_by_part_plane"
        elif (
            local_coverage < DIAGNOSTIC_TOP_SUPPORT_GATE
            and group.support_ratio >= DIAGNOSTIC_TOP_SUPPORT_GATE
        ):
            reason = "weak_room_support_promoted_by_part_plane"
        score_coverage = 0.85 if can_promote_local_flat else local_coverage
        score = (
            score_coverage * 4.0
            + group.support_ratio * 2.0
            + side_coverage * 1.5
            + min(group.support_area_m2 / 50.0, 1.0)
            + min(len(group.source_locators) / 5.0, 1.0) * 0.3
        )
        if (
            group.label == "flat-ceiling"
            and local_coverage >= DIAGNOSTIC_TOP_SUPPORT_GATE
        ):
            score += 2.5
        if group.domain_index == domain_index:
            score += 0.2
        candidates.append(
            SelectedTop(
                label=group.label,
                signature=(group.label, group.key),
                faces=(group.representative,),
                score=score,
                local_coverage=local_coverage,
                part_support_ratio=group.support_ratio,
                reason=reason,
            )
        )

    for pair in gable_pairs:
        if (
            pair.domain_index != domain_index
            and (
                dominant_gable is None
                or pair.signature != dominant_gable.signature
            )
        ):
            continue
        if dominant_gable is not None and pair.signature != dominant_gable.signature:
            continue
        if any(
            candidate.label == "flat-ceiling"
            and candidate.local_coverage >= ROOM_COVERED_RATIO
            for candidate in candidates
        ):
            continue
        if cell.exposed_ratio < 0.20:
            continue
        if not _gable_pair_uses_both_faces(cell.polygon, pair):
            continue
        support_coverage = _safe_overlap_ratio(cell.polygon, pair.footprint)
        if support_coverage < MIN_GABLE_CELL_SUPPORT_RATIO:
            continue
        local_coverage = (
            1.0
            if dominant_gable is not None and pair.signature == dominant_gable.signature
            else support_coverage
        )
        if local_coverage < 0.05:
            continue
        split_regions = pa._split_footprint_by_plane_intersection(
            cell.polygon,
            pa._top_plane_up(pair.left.representative.plane),
            pa._top_plane_up(pair.right.representative.plane),
        )
        top_regions: list[tuple[Polygon, Any]] = []
        if split_regions is None:
            face = _lower_face_for_region(cell.polygon, pair.left, pair.right)
            top_regions = [
                (
                    cell.polygon,
                    pa._top_plane_up(face.representative.plane),
                )
            ]
        else:
            for region in split_regions:
                face = _lower_face_for_region(region, pair.left, pair.right)
                top_regions.append(
                    (region, pa._top_plane_up(face.representative.plane))
                )
        if not top_regions or not _top_is_above_cell(cell, top_regions):
            continue
        reason = "candidate_covered"
        if (
            support_coverage < DIAGNOSTIC_TOP_SUPPORT_GATE
            and pair.support_ratio >= DOMAIN_GABLE_FRAME_SUPPORT_RATIO
        ):
            reason = "weak_room_support_promoted_by_part_plane"
        score = local_coverage * 4.0 + pair.support_ratio * 4.0 + 1.5
        if dominant_gable is not None and pair.signature == dominant_gable.signature:
            score += 3.0
        if local_coverage >= DIAGNOSTIC_TOP_SUPPORT_GATE:
            score += 1.0
        candidates.append(
            SelectedTop(
                label="gable-pair",
                signature=pair.signature,
                faces=(pair.left.representative, pair.right.representative),
                score=score,
                local_coverage=local_coverage,
                part_support_ratio=pair.support_ratio,
                reason=reason,
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return SelectedCell(cell=cell, top=candidates[0])


def _domain_choice_for_cell(
    cell: PlanCell,
    *,
    domain_index: int,
    domain_top_choices: list[DomainTopChoice],
) -> DomainTopChoice | None:
    matching = [
        choice
        for choice in domain_top_choices
        if choice.domain_index == domain_index
    ]
    return matching[0] if matching else None


def _selected_cell_from_domain_choice(
    cell: PlanCell,
    choice: DomainTopChoice,
) -> SelectedCell | None:
    if not choice.faces:
        return None
    if choice.label in ("flat-ceiling", "single-oblique"):
        support_overlap = _safe_overlap_ratio(cell.polygon, choice.footprint)
        if support_overlap < 0.50:
            return None
        plane = pa._top_plane_up(choice.faces[0].plane)
        if not _top_is_above_cell(cell, [(cell.polygon, plane)]):
            return None
        top = SelectedTop(
            label=choice.label,
            signature=choice.signature,
            faces=choice.faces,
            score=choice.score,
            local_coverage=support_overlap,
            part_support_ratio=choice.support_ratio,
            reason=choice.reason,
        )
        return SelectedCell(cell=cell, top=top)
    if choice.label == "gable-pair" and len(choice.faces) >= 2:
        top_regions = _domain_gable_top_regions(cell.polygon, choice.faces)
        if not top_regions or not _top_is_above_cell(cell, top_regions):
            return None
        top = SelectedTop(
            label="gable-pair",
            signature=choice.signature,
            faces=choice.faces,
            score=choice.score,
            local_coverage=1.0,
            part_support_ratio=choice.support_ratio,
            reason=choice.reason,
        )
        return SelectedCell(cell=cell, top=top)
    return None


def _domain_gable_top_regions(
    polygon: Polygon,
    faces: tuple[pa.PayloadFace, ...],
) -> list[tuple[Polygon, pa.Plane]]:
    if len(faces) < 2:
        return []
    left = pa._top_plane_up(faces[0].plane)
    right = pa._top_plane_up(faces[1].plane)
    split_regions = pa._split_footprint_by_plane_intersection(polygon, left, right)
    if not split_regions:
        face = (
            faces[0]
            if _plane_y_at_polygon(left, polygon)
            <= _plane_y_at_polygon(right, polygon)
            else faces[1]
        )
        return [(polygon, pa._top_plane_up(face.plane))]
    regions = []
    for region in split_regions:
        plane = (
            left
            if _plane_y_at_polygon(left, region)
            <= _plane_y_at_polygon(right, region)
            else right
        )
        regions.append((region, plane))
    return regions


def _plane_y_at_polygon(plane: pa.Plane, polygon: Polygon) -> float:
    point = polygon.representative_point()
    return pa._plane_y_at(plane, float(point.x), float(point.y))


def _can_promote_local_flat_ceiling(
    cell: PlanCell,
    group: PlaneGroupEvidence,
    *,
    local_coverage: float,
) -> bool:
    if group.label != "flat-ceiling":
        return False
    if not set(cell.source_room_indices).intersection(group.source_face_room_indices):
        return False
    if group.support_area_m2 < MIN_CELL_AREA_M2:
        return False
    if local_coverage >= DIAGNOSTIC_TOP_SUPPORT_GATE:
        return True
    plane = pa._top_plane_up(group.representative.plane)
    return _min_top_clearance(cell, plane) >= MIN_REALISTIC_CEILING_CLEARANCE_M


def _min_top_clearance(cell: PlanCell, plane: pa.Plane) -> float:
    coords = pa._polygon_exterior_coords(cell.polygon)
    if not coords:
        return -float("inf")
    samples = coords
    point = cell.polygon.representative_point()
    samples.append((float(point.x), float(point.y)))
    return min(pa._plane_y_at(plane, x, z) - cell.floor_y for x, z in samples)


def _single_oblique_topology_allowed(
    cell: PlanCell,
    group: PlaneGroupEvidence,
    *,
    local_coverage: float,
    side_coverage: float,
    gable_pairs: list[GablePairEvidence],
) -> bool:
    """Reject oblique projections that have overlap but no roof ownership."""

    has_source_face = bool(
        set(cell.source_room_indices).intersection(group.source_face_room_indices)
    )
    in_strong_gable = _group_participates_in_strong_gable(group, gable_pairs)
    if in_strong_gable:
        return (
            local_coverage >= DIAGNOSTIC_TOP_SUPPORT_GATE
            or (
                side_coverage >= ROOM_COVERED_RATIO
                and group.support_ratio >= DOMAIN_GABLE_FRAME_SUPPORT_RATIO
            )
            or (has_source_face and local_coverage >= 0.20)
        )
    if group.support_ratio >= STRICT_TOP_SUPPORT_GATE:
        return local_coverage >= DIAGNOSTIC_TOP_SUPPORT_GATE
    if has_source_face:
        return (
            local_coverage >= STRICT_TOP_SUPPORT_GATE
            and group.support_ratio >= DIAGNOSTIC_TOP_SUPPORT_GATE
        )
    return False


def _group_participates_in_strong_gable(
    group: PlaneGroupEvidence,
    gable_pairs: list[GablePairEvidence],
) -> bool:
    for pair in gable_pairs:
        if pair.domain_index != group.domain_index:
            continue
        if pair.support_ratio < DOMAIN_GABLE_FRAME_SUPPORT_RATIO:
            continue
        if pair.left.key == group.key or pair.right.key == group.key:
            return True
    return False


def _candidates_from_selected_cells(
    payload: dict[str, Any],
    selected_cells: list[SelectedCell],
    *,
    ceiling_faces: list[pa.PayloadFace],
    min_top_overlap_ratio: float,
    corner_tol: float,
    plane_groups: list[PlaneGroupEvidence],
) -> tuple[list[pa.EnvelopeCandidate], list[dict[str, Any]]]:
    merged = _merge_selected_cells(selected_cells)
    candidates: list[pa.EnvelopeCandidate] = []
    build_attempts: list[dict[str, Any]] = []
    label_summary = dict(Counter(cell.top.label for cell in selected_cells))
    serialized_groups = [_plane_group_json(group) for group in plane_groups]
    part_index = 0
    for component in merged:
        selected_component_all = [selected_cells[index] for index in component]
        raw_union = _union_selected_component_polygons(selected_component_all)
        component_polygons = [
            cleaned
            for polygon in _polygon_components(raw_union)
            if (cleaned := _clean_selected_part_polygon(polygon)) is not None
            and cleaned.area >= MIN_CELL_AREA_M2
        ]
        if not component_polygons:
            attempt = _build_attempt_json(
                part_index=part_index,
                selected_component=selected_component_all,
                polygon=None,
                floor_y=None,
            )
            attempt["result"] = "rejected"
            attempt["reason"] = "tiny_or_invalid_polygon"
            build_attempts.append(attempt)
            part_index += 1
            continue
        for polygon in component_polygons:
            selected_component = [
                selected
                for selected in selected_component_all
                if _safe_intersection_area(selected.cell.polygon, polygon) > 0.05
            ] or selected_component_all
            candidate_count_before = len(candidates)
            part_index = _append_candidate_for_selected_component(
                payload,
                selected_component,
                polygon=polygon,
                part_index=part_index,
                ceiling_faces=ceiling_faces,
                min_top_overlap_ratio=min_top_overlap_ratio,
                corner_tol=corner_tol,
                serialized_groups=serialized_groups,
                label_summary=label_summary,
                candidates=candidates,
                build_attempts=build_attempts,
            )
            if (
                len(candidates) == candidate_count_before
                and len(selected_component) > 1
            ):
                for selected in selected_component:
                    part_index = _append_candidate_for_selected_component(
                        payload,
                        [selected],
                        polygon=selected.cell.polygon,
                        part_index=part_index,
                        ceiling_faces=ceiling_faces,
                        min_top_overlap_ratio=min_top_overlap_ratio,
                        corner_tol=corner_tol,
                        serialized_groups=serialized_groups,
                        label_summary=label_summary,
                        candidates=candidates,
                        build_attempts=build_attempts,
                    )
    if candidates:
        aggregate_coverage = _room_coverage_for_candidates(
            payload,
            candidates,
            corner_tol,
        )
        candidates = [
            pa.EnvelopeCandidate(
                locator_id=candidate.locator_id,
                faces=candidate.faces,
                footprint_area_m2=candidate.footprint_area_m2,
                top_source=candidate.top_source,
                top_overlap_ratio=candidate.top_overlap_ratio,
                selector=candidate.selector,
                room_coverage=aggregate_coverage,
                part_plane_groups=candidate.part_plane_groups,
                top_label_summary=candidate.top_label_summary,
            )
            for candidate in candidates
        ]
    return candidates, build_attempts


def _append_candidate_for_selected_component(
    payload: dict[str, Any],
    selected_component: list[SelectedCell],
    *,
    polygon: Polygon,
    part_index: int,
    ceiling_faces: list[pa.PayloadFace],
    min_top_overlap_ratio: float,
    corner_tol: float,
    serialized_groups: list[dict[str, Any]],
    label_summary: dict[str, int],
    candidates: list[pa.EnvelopeCandidate],
    build_attempts: list[dict[str, Any]],
) -> int:
    attempt = _build_attempt_json(
        part_index=part_index,
        selected_component=selected_component,
        polygon=polygon,
        floor_y=None,
    )
    if polygon is None or polygon.area < MIN_CELL_AREA_M2:
        attempt["result"] = "rejected"
        attempt["reason"] = "tiny_or_invalid_polygon"
        build_attempts.append(attempt)
        return part_index + 1
    top = _representative_top_for_selected_component(selected_component)
    floor_y = _floor_y_for_selected_component(selected_component)
    if floor_y is None:
        floor_y = pa._floor_y_for_part(payload, polygon, corner_tol=corner_tol)
    attempt = _build_attempt_json(
        part_index=part_index,
        selected_component=selected_component,
        polygon=polygon,
        floor_y=floor_y,
    )
    gable_diagnostic = _gable_footprint_coherence_json(polygon, top)
    if gable_diagnostic is not None:
        attempt["gable_footprint_coherence"] = gable_diagnostic
    if floor_y is None:
        attempt["result"] = "rejected"
        attempt["reason"] = "no_floor_height"
        build_attempts.append(attempt)
        return part_index + 1
    locator_id = f"envelope-cell-selector:{part_index}"
    candidate = _candidate_for_selected_part(
        polygon,
        top,
        floor_y=floor_y,
        locator_id=locator_id,
        min_top_overlap_ratio=min_top_overlap_ratio,
    )
    attempt["direct_candidate"] = _candidate_build_status(candidate)
    final_candidate = candidate
    if final_candidate is None or not attempt["direct_candidate"]["strict_builds"]:
        fallback_candidate = pa._envelope_candidate_for_part(
            payload,
            polygon,
            ceiling_faces,
            min_top_overlap_ratio=min_top_overlap_ratio,
            corner_tol=corner_tol,
            locator_id=locator_id,
            strict_gate_all=True,
            floor_y_override=floor_y,
        )
        attempt["fallback_candidate"] = _candidate_build_status(fallback_candidate)
        final_candidate = fallback_candidate
    else:
        attempt["fallback_candidate"] = None
    if final_candidate is None:
        attempt["result"] = "rejected"
        attempt["reason"] = "candidate_not_constructed"
        build_attempts.append(attempt)
        return part_index + 1
    final_status = _candidate_build_status(final_candidate)
    top_clearance_m = _candidate_min_top_clearance_m(final_candidate)
    if (
        top_clearance_m is not None
        and top_clearance_m < TOP_ABOVE_FLOOR_SLACK_M
    ):
        attempt["result"] = "rejected"
        attempt["reason"] = "top_clearance_too_low"
        attempt["top_clearance_m"] = top_clearance_m
        attempt["final_candidate"] = final_status
        build_attempts.append(attempt)
        return part_index + 1
    if not final_status["strict_builds"]:
        attempt["result"] = "rejected"
        attempt["reason"] = "strict_build_failed"
        attempt["final_candidate"] = final_status
        build_attempts.append(attempt)
        return part_index + 1
    attempt["result"] = "accepted"
    attempt["reason"] = "strict_build_succeeded"
    attempt["final_candidate"] = final_status
    build_attempts.append(attempt)
    coverage = _room_coverage_for_candidates(payload, [final_candidate], corner_tol)
    candidates.append(
        pa.EnvelopeCandidate(
            locator_id=final_candidate.locator_id,
            faces=final_candidate.faces,
            footprint_area_m2=final_candidate.footprint_area_m2,
            top_source=final_candidate.top_source,
            top_overlap_ratio=final_candidate.top_overlap_ratio,
            selector="cell-selector",
            room_coverage=coverage,
            part_plane_groups=serialized_groups,
            top_label_summary=label_summary,
        )
    )
    return part_index + 1


def _floor_y_for_selected_component(
    selected_component: list[SelectedCell],
) -> float | None:
    total_area = 0.0
    weighted_y = 0.0
    for selected in selected_component:
        area = float(selected.cell.polygon.area)
        if area <= 0.0:
            continue
        total_area += area
        weighted_y += selected.cell.floor_y * area
    if total_area <= 0.0:
        return None
    return weighted_y / total_area


def _build_attempt_json(
    *,
    part_index: int,
    selected_component: list[SelectedCell],
    polygon: Polygon | None,
    floor_y: float | None,
) -> dict[str, Any]:
    top = (
        _representative_top_for_selected_component(selected_component)
        if selected_component
        else None
    )
    room_indices = sorted(
        {
            room_index
            for selected in selected_component
            for room_index in selected.cell.source_room_indices
        }
    )
    return {
        "part_index": part_index,
        "selected_cell_ids": [selected.cell.cell_id for selected in selected_component],
        "room_indices": room_indices,
        "top_label": top.label if top is not None else None,
        "top_signature": str(top.signature) if top is not None else None,
        "top_sources": [
            face.locator_id
            for face in (top.faces if top is not None else [])
        ],
        "footprint_area_m2": float(polygon.area)
        if polygon is not None and not polygon.is_empty
        else 0.0,
        "floor_y": floor_y,
        "selected_support": [
            {
                "cell_id": selected.cell.cell_id,
                "room_indices": list(selected.cell.source_room_indices),
                "exposed_ratio": selected.cell.exposed_ratio,
                "top_label": selected.top.label,
                "score": selected.top.score,
                "local_coverage": selected.top.local_coverage,
                "part_support_ratio": selected.top.part_support_ratio,
                "reason": selected.top.reason,
            }
            for selected in selected_component
        ],
    }


def _candidate_build_status(
    candidate: pa.EnvelopeCandidate | None,
) -> dict[str, Any]:
    if candidate is None:
        return {"constructed": False, "strict_builds": False}
    try:
        pa.build_from_planar_polygons(
            [(face.corners, face.plane) for face in candidate.faces]
        )
    except ValueError as exc:
        return {
            "constructed": True,
            "strict_builds": False,
            "locator_id": candidate.locator_id,
            "face_count": len(candidate.faces),
            "top_overlap_ratio": candidate.top_overlap_ratio,
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0],
        }
    return {
        "constructed": True,
        "strict_builds": True,
        "locator_id": candidate.locator_id,
        "face_count": len(candidate.faces),
        "top_overlap_ratio": candidate.top_overlap_ratio,
    }


def _gable_footprint_coherence_json(
    polygon: Polygon,
    top: SelectedTop | None,
) -> dict[str, Any] | None:
    """Return diagnostic evidence for how coherently a ridge partitions a part.

    PolyFit's useful lesson here is the separation between hypothesis generation
    and supported/watertight selection. Arnis adds a practical footprint-field
    signal: local edge distances reveal whether a gable behaves like one coherent
    wing or a fragmented set of plan cells. This diagnostic deliberately does not
    reject candidates; it only records evidence for corpus analysis.
    """

    if (
        top is None
        or top.label != "gable-pair"
        or len(top.faces) < 2
        or polygon is None
        or polygon.is_empty
    ):
        return None
    left, right = top.faces[:2]
    left_plane = pa._top_plane_up(left.plane)
    right_plane = pa._top_plane_up(right.plane)
    line_coeffs = pa._plane_equal_height_line_xz(left_plane, right_plane)
    ridge_line = pa._ridge_split_line_for_planes(polygon, left_plane, right_plane)
    if line_coeffs is None or ridge_line is None:
        return {
            "schema_version": 1,
            "status": "no_ridge_line",
            "footprint_area_m2": float(polygon.area),
        }

    a, b, _c = line_coeffs
    normal_len = max(hypot(a, b), 1e-12)
    normal_dir = (a / normal_len, b / normal_len)
    axis_dir = (normal_dir[1], -normal_dir[0])
    axis_deg = (degrees(atan2(axis_dir[1], axis_dir[0])) + 360.0) % 180.0

    split_regions = _split_polygon_by_line(polygon, ridge_line)
    assigned: list[tuple[Polygon, pa.PayloadFace]] = []
    for region in split_regions:
        clean_region = _clean_top_region_polygon(region)
        if clean_region is None or clean_region.area <= 0.05:
            continue
        assigned.append(
            (clean_region, _lower_payload_face_for_region(clean_region, left, right))
        )

    side_areas: dict[str, float] = {left.locator_id: 0.0, right.locator_id: 0.0}
    side_regions: dict[str, list[Polygon]] = {left.locator_id: [], right.locator_id: []}
    region_records: list[dict[str, Any]] = []
    for region, face in assigned:
        area = float(region.area)
        side_areas[face.locator_id] = side_areas.get(face.locator_id, 0.0) + area
        side_regions.setdefault(face.locator_id, []).append(region)
        centroid = region.representative_point()
        region_records.append(
            {
                "area_m2": area,
                "top_source": face.locator_id,
                "signed_ridge_distance_m": _signed_distance_to_line_coeffs(
                    line_coeffs,
                    float(centroid.x),
                    float(centroid.y),
                ),
            }
        )

    nonzero_side_areas = [area for area in side_areas.values() if area > 1e-6]
    side_area_balance = (
        min(nonzero_side_areas) / max(nonzero_side_areas)
        if len(nonzero_side_areas) >= 2
        else 0.0
    )
    side_component_counts: dict[str, int] = {}
    for source, regions in side_regions.items():
        side_component_counts[source] = (
            len(_polygon_components(_safe_unary_union(regions))) if regions else 0
        )

    covered_area = (
        float(_safe_unary_union([region for region, _face in assigned]).area)
        if assigned
        else 0.0
    )
    axis_widths = _sample_local_widths(polygon, axis_dir)
    perp_widths = _sample_local_widths(polygon, normal_dir)
    return {
        "schema_version": 1,
        "status": "ok",
        "footprint_area_m2": float(polygon.area),
        "covered_area_ratio": covered_area / max(float(polygon.area), 1e-9),
        "ridge_axis_deg": axis_deg,
        "ridge_intersection_length_m": float(ridge_line.intersection(polygon).length),
        "split_region_count": len(assigned),
        "uses_both_roof_faces": len(nonzero_side_areas) >= 2,
        "side_area_m2": side_areas,
        "side_area_balance": side_area_balance,
        "side_component_counts": side_component_counts,
        "fragmented_side_count": sum(
            1 for count in side_component_counts.values() if count > 1
        ),
        "regions": sorted(
            region_records,
            key=lambda item: item["area_m2"],
            reverse=True,
        ),
        "local_width_samples": {
            "sample_count": min(len(axis_widths), len(perp_widths)),
            "along_ridge_m": _summary_stats(axis_widths),
            "across_ridge_m": _summary_stats(perp_widths),
        },
    }


def _split_polygon_by_line(
    polygon: Polygon,
    line: LineString,
) -> list[Polygon]:
    try:
        parts = [
            part
            for part in split(polygon, line).geoms
            if isinstance(part, Polygon) and part.area > 0.05
        ]
    except Exception:
        return [polygon]
    return parts or [polygon]


def _signed_distance_to_line_coeffs(
    coeffs: tuple[float, float, float],
    x: float,
    z: float,
) -> float:
    a, b, c = coeffs
    return float((a * x + b * z + c) / max(hypot(a, b), 1e-12))


def _sample_local_widths(
    polygon: Polygon,
    direction: tuple[float, float],
    *,
    grid_size: int = 7,
) -> list[float]:
    dx, dz = direction
    norm = max(hypot(dx, dz), 1e-12)
    dx /= norm
    dz /= norm
    minx, minz, maxx, maxz = polygon.bounds
    span = max(maxx - minx, maxz - minz, 1.0) * 3.0
    widths: list[float] = []
    for ix in range(1, grid_size + 1):
        x = minx + (maxx - minx) * ix / (grid_size + 1)
        for iz in range(1, grid_size + 1):
            z = minz + (maxz - minz) * iz / (grid_size + 1)
            point = Point(x, z)
            if not polygon.covers(point):
                continue
            line = LineString(
                [
                    (x - dx * span, z - dz * span),
                    (x + dx * span, z + dz * span),
                ]
            )
            width = _line_component_length_at_point(polygon.intersection(line), point)
            if width is not None and width > 0.0:
                widths.append(width)
    return widths


def _line_component_length_at_point(geometry: Any, point: Point) -> float | None:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, LineString):
        return float(geometry.length) if geometry.distance(point) <= 1e-6 else None
    if hasattr(geometry, "geoms"):
        lengths = [
            float(part.length)
            for part in geometry.geoms
            if isinstance(part, LineString) and part.distance(point) <= 1e-6
        ]
        return max(lengths, default=None)
    return None


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) * 0.5
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def _candidate_min_top_clearance_m(
    candidate: pa.EnvelopeCandidate,
) -> float | None:
    floor_y = _candidate_floor_y(candidate)
    if floor_y is None:
        return None
    clearances: list[float] = []
    for face in candidate.faces:
        if face.kind != "ceiling":
            continue
        clearances.extend(float(corner[1]) - floor_y for corner in face.corners)
    if not clearances:
        return None
    return min(clearances)


def _safe_overlap_ratio(left: Polygon, right: Polygon) -> float:
    try:
        overlap = float(left.intersection(right).area)
    except Exception:
        try:
            overlap = float(left.buffer(0).intersection(right.buffer(0)).area)
        except Exception:
            return 0.0
    return overlap / max(float(left.area), 1e-9)


def _safe_intersection_area(left: Polygon, right: Polygon) -> float:
    try:
        return float(left.intersection(right).area)
    except Exception:
        try:
            return float(left.buffer(0).intersection(right.buffer(0)).area)
        except Exception:
            return 0.0


def _safe_unary_union(polygons: list[Any]) -> Any:
    if not polygons:
        return Polygon()
    try:
        return unary_union(polygons)
    except Exception:
        cleaned = []
        for poly in polygons:
            if poly is None or poly.is_empty:
                continue
            try:
                cleaned.append(poly.buffer(0))
            except Exception:
                continue
        if not cleaned:
            return Polygon()
        return unary_union(cleaned)


def _union_selected_component_polygons(selected_component: list[SelectedCell]) -> Any:
    polygons = [selected.cell.polygon for selected in selected_component]
    raw_union = _safe_unary_union(polygons)
    raw_components = _polygon_components(raw_union)
    if len(raw_components) <= 1:
        return raw_union
    gap = ADJACENCY_TOL_M * 0.5
    try:
        expanded = [poly.buffer(gap, join_style="mitre") for poly in polygons]
        closed = unary_union(expanded).buffer(-gap, join_style="mitre")
    except Exception:
        return raw_union
    closed_components = _polygon_components(closed)
    if not closed_components or len(closed_components) >= len(raw_components):
        return raw_union
    raw_area = sum(float(poly.area) for poly in raw_components)
    closed_area = sum(float(poly.area) for poly in closed_components)
    area_ratio = closed_area / max(raw_area, 1e-9)
    if 0.98 <= area_ratio <= 1.08:
        return closed
    return raw_union


def _candidate_for_selected_part(
    polygon: Polygon,
    top: SelectedTop,
    *,
    floor_y: float,
    locator_id: str,
    min_top_overlap_ratio: float,
) -> pa.EnvelopeCandidate | None:
    polygon = _clean_selected_part_polygon(polygon) or polygon
    if top.label in ("flat-ceiling", "single-oblique") and top.faces:
        top_face = top.faces[0]
        top_plane = pa._top_plane_up(top_face.plane)
        if not pa._top_plane_is_above_floor(
            polygon,
            top_plane,
            floor_y=floor_y,
            min_clearance_m=TOP_ABOVE_FLOOR_SLACK_M,
        ):
            return None
        return pa._envelope_candidate_from_top_plane(
            polygon,
            top_face,
            floor_y=floor_y,
            locator_id=locator_id,
            overlap_ratio=max(top.local_coverage, min_top_overlap_ratio),
            top_source=" + ".join(face.locator_id for face in top.faces),
        )
    if top.label == "gable-pair" and len(top.faces) >= 2:
        partition = pa._ridge_aware_top_partition(
            polygon,
            list(top.faces),
            min_coverage_ratio=DIAGNOSTIC_TOP_SUPPORT_GATE,
        )
        if partition is None:
            partition = pa._two_plane_top_partition(
                polygon,
                list(top.faces),
                min_coverage_ratio=DIAGNOSTIC_TOP_SUPPORT_GATE,
            )
        if partition is None:
            partition = _ridge_split_top_partition_without_support_gate(
                polygon,
                top.faces[0],
                top.faces[1],
            )
        if partition is None:
            return _candidate_from_single_side_of_gable_pair(
                polygon,
                top.faces[0],
                top.faces[1],
                floor_y=floor_y,
                locator_id=locator_id,
                min_top_overlap_ratio=min_top_overlap_ratio,
            )
        top_regions, coverage_ratio = partition
        top_regions = _clean_top_regions(top_regions)
        if not top_regions:
            return None
        if not pa._top_regions_are_above_floor(
            top_regions,
            floor_y=floor_y,
            min_clearance_m=TOP_ABOVE_FLOOR_SLACK_M,
        ):
            return None
        return pa._envelope_candidate_from_top_regions(
            top_regions,
            floor_y=floor_y,
            locator_id=locator_id,
            coverage_ratio=max(coverage_ratio, min_top_overlap_ratio),
            footprint_override=polygon,
        )
    return None


def _clean_selected_part_polygon(poly: Polygon | None) -> Polygon | None:
    """Remove precision noise before building strict candidate faces.

    Shapely unions from room cells can preserve near-duplicate vertices or tiny
    hairpin edges. Those are visually harmless in plan, but the half-edge
    constructor correctly rejects the resulting 3D faces as duplicate vertices
    or duplicate directed edges. A 1e-6m grid is far below scan precision while
    still enough to collapse floating-point duplicates.
    """

    if poly is None or poly.is_empty:
        return None
    original_area = float(poly.area)
    if original_area <= 0.0:
        return None
    candidates: list[Polygon] = []
    cleanup_candidates = [
        (poly, 1e-5),
        (_precision_polygon(poly, grid_size=1e-4), 1e-4),
        (_precision_polygon(poly, grid_size=1e-3), 1e-3),
        (_simplify_polygon(poly, tolerance=1e-3), 1e-3),
        (_simplify_polygon(poly, tolerance=2e-3), 2e-3),
    ]
    for candidate, tol in cleanup_candidates:
        cleaned = _polygon_from_clean_coords(candidate, tol=tol)
        if cleaned is not None:
            candidates.append(cleaned)
    if not candidates:
        return poly
    candidates.sort(
        key=lambda candidate: (
            0
            if 0.99 <= float(candidate.area) / max(original_area, 1e-9) <= 1.01
            else 1,
            len(pa._polygon_exterior_coords(candidate)),
            abs(float(candidate.area) - original_area),
        )
    )
    best = candidates[0]
    area_ratio = float(best.area) / max(original_area, 1e-9)
    if 0.80 <= area_ratio <= 1.20:
        return best
    return poly


def _precision_polygon(poly: Polygon, *, grid_size: float = 1e-6) -> Polygon:
    try:
        snapped = set_precision(poly, grid_size, mode="valid_output")
        return pa._largest_polygon(snapped) or poly
    except Exception:
        return poly


def _simplify_polygon(poly: Polygon, *, tolerance: float) -> Polygon:
    try:
        simplified = poly.simplify(tolerance, preserve_topology=True)
        return pa._largest_polygon(simplified) or poly
    except Exception:
        return poly


def _polygon_from_clean_coords(
    poly: Polygon | None,
    *,
    tol: float,
) -> Polygon | None:
    if poly is None or poly.is_empty:
        return None
    coords = _remove_close_coords(pa._polygon_exterior_coords(poly), tol=tol)
    coords = pa._remove_collinear_coords(coords, tol=tol)
    if len(coords) < 3:
        return None
    try:
        cleaned = pa._largest_polygon(Polygon(coords).buffer(0))
    except Exception:
        return None
    if cleaned is None or cleaned.area < MIN_CELL_AREA_M2:
        return None
    return cleaned


def _remove_close_coords(
    coords: list[tuple[float, float]],
    *,
    tol: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for coord in coords:
        if out and hypot(coord[0] - out[-1][0], coord[1] - out[-1][1]) <= tol:
            continue
        out.append(coord)
    if len(out) > 1 and hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def _clean_top_regions(
    top_regions: list[tuple[Polygon, pa.PayloadFace]],
) -> list[tuple[Polygon, pa.PayloadFace]]:
    cleaned: list[tuple[Polygon, pa.PayloadFace]] = []
    for region, face in top_regions:
        clean_region = _clean_top_region_polygon(region)
        if clean_region is None or clean_region.area < MIN_CELL_AREA_M2:
            continue
        cleaned.append((clean_region, face))
    return cleaned


def _clean_top_region_polygon(poly: Polygon | None) -> Polygon | None:
    if poly is None or poly.is_empty:
        return None
    original_area = float(poly.area)
    if original_area <= 0.0:
        return None
    coords = _remove_close_coords(pa._polygon_exterior_coords(poly), tol=1e-3)
    if len(coords) < 3:
        return None
    try:
        cleaned = pa._largest_polygon(Polygon(coords).buffer(0))
    except Exception:
        return None
    if cleaned is None or cleaned.area < MIN_CELL_AREA_M2:
        return None
    area_ratio = float(cleaned.area) / max(original_area, 1e-9)
    if 0.80 <= area_ratio <= 1.20:
        return cleaned
    return poly


def _lower_side_coverage_for_group(
    polygon: Polygon,
    group: PlaneGroupEvidence,
    gable_pairs: list[GablePairEvidence],
) -> float:
    relevant = [
        pair
        for pair in gable_pairs
        if pair.domain_index == group.domain_index
        and (pair.left.key == group.key or pair.right.key == group.key)
    ]
    if not relevant:
        return 1.0
    best = 0.0
    for pair in relevant:
        lower_regions = _lower_side_regions_for_pair(polygon, pair)
        if not lower_regions:
            continue
        matching_area = sum(
            float(region.area)
            for region, face in lower_regions
            if face.key == group.key
        )
        best = max(best, matching_area / max(float(polygon.area), 1e-9))
    return best


def _dominant_gable_pair_for_cell(
    cell: PlanCell,
    *,
    domain_index: int,
    gable_pairs: list[GablePairEvidence],
) -> GablePairEvidence | None:
    candidates: list[tuple[float, GablePairEvidence]] = []
    for pair in gable_pairs:
        if pair.support_ratio < DOMAIN_GABLE_FRAME_SUPPORT_RATIO:
            continue
        try:
            support_coverage = float(cell.polygon.intersection(pair.footprint).area)
        except Exception:
            continue
        support_coverage /= max(float(cell.polygon.area), 1e-9)
        if support_coverage < MIN_GABLE_CELL_SUPPORT_RATIO:
            continue
        if not _gable_pair_uses_both_faces(cell.polygon, pair):
            continue
        regions = _lower_side_regions_for_pair(cell.polygon, pair)
        if not regions:
            continue
        if not _top_is_above_cell(
            cell,
            [
                (region, pa._top_plane_up(face.representative.plane))
                for region, face in regions
            ],
        ):
            continue
        same_domain_bonus = 0.10 if pair.domain_index == domain_index else 0.0
        score = support_coverage * 2.0 + pair.support_ratio + same_domain_bonus
        candidates.append((score, pair))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _lower_side_regions_for_pair(
    polygon: Polygon,
    pair: GablePairEvidence,
) -> list[tuple[Polygon, PlaneGroupEvidence]]:
    split_regions = pa._split_footprint_by_plane_intersection(
        polygon,
        pa._top_plane_up(pair.left.representative.plane),
        pa._top_plane_up(pair.right.representative.plane),
    )
    regions = split_regions if split_regions is not None else [polygon]
    return [
        (region, _lower_face_for_region(region, pair.left, pair.right))
        for region in regions
        if isinstance(region, Polygon) and region.area > 0.05
    ]


def _lower_side_polygon_for_group(
    domain: Polygon,
    group: PlaneGroupEvidence,
    other: PlaneGroupEvidence,
) -> Polygon | None:
    split_regions = pa._split_footprint_by_plane_intersection(
        domain,
        pa._top_plane_up(group.representative.plane),
        pa._top_plane_up(other.representative.plane),
    )
    if split_regions is None:
        return domain
    owned_regions = [
        region
        for region in split_regions
        if isinstance(region, Polygon)
        and region.area > 0.05
        and _lower_face_for_region(region, group, other).key == group.key
    ]
    if not owned_regions:
        return None
    try:
        return pa._largest_polygon(unary_union(owned_regions).intersection(domain))
    except Exception:
        return None


def _gable_pair_uses_both_faces(
    polygon: Polygon,
    pair: GablePairEvidence,
) -> bool:
    regions = _lower_side_regions_for_pair(polygon, pair)
    if len(regions) < 2:
        return False
    areas = {pair.left.key: 0.0, pair.right.key: 0.0}
    for region, face in regions:
        areas[face.key] = areas.get(face.key, 0.0) + float(region.area)
    min_area = max(MIN_CELL_AREA_M2, float(polygon.area) * 0.10)
    return areas.get(pair.left.key, 0.0) >= min_area and areas.get(
        pair.right.key,
        0.0,
    ) >= min_area


def _lower_face_for_region(
    region: Polygon,
    left: PlaneGroupEvidence,
    right: PlaneGroupEvidence,
) -> PlaneGroupEvidence:
    point = region.representative_point()
    left_y = pa._plane_y_at(
        pa._top_plane_up(left.representative.plane),
        float(point.x),
        float(point.y),
    )
    right_y = pa._plane_y_at(
        pa._top_plane_up(right.representative.plane),
        float(point.x),
        float(point.y),
    )
    return left if left_y <= right_y else right


def _ridge_split_top_partition_without_support_gate(
    polygon: Polygon,
    left: pa.PayloadFace,
    right: pa.PayloadFace,
) -> tuple[list[tuple[Polygon, pa.PayloadFace]], float] | None:
    """Split a selected gable part by ridge even when local supports are sparse.

    The selected-cell stage has already chosen this gable pair using
    building-part plane evidence. At candidate-build time we should not require
    each room-local face footprint to cover its side again; doing that
    reintroduces the weak-local-signal failure mode. The equal-height ridge line
    gives a deterministic top partition, and each side gets the higher plane at
    its representative point.
    """

    left_plane = pa._top_plane_up(left.plane)
    right_plane = pa._top_plane_up(right.plane)
    split_regions = pa._split_footprint_by_plane_intersection(
        polygon,
        left_plane,
        right_plane,
    )
    if split_regions is None:
        return None
    assigned: list[tuple[Polygon, pa.PayloadFace]] = []
    used: set[str] = set()
    for region in split_regions:
        clean_region = _clean_selected_part_polygon(region)
        if clean_region is None:
            continue
        face = _lower_payload_face_for_region(clean_region, left, right)
        used.add(face.locator_id)
        assigned.append((clean_region, face))
    if len(assigned) < 2 or len(used) < 2:
        return None
    covered = unary_union([region for region, _face in assigned]).intersection(polygon)
    coverage_ratio = float(covered.area) / max(float(polygon.area), 1e-9)
    if coverage_ratio < DIAGNOSTIC_TOP_SUPPORT_GATE:
        return None
    return assigned, coverage_ratio


def _candidate_from_single_side_of_gable_pair(
    polygon: Polygon,
    left: pa.PayloadFace,
    right: pa.PayloadFace,
    *,
    floor_y: float,
    locator_id: str,
    min_top_overlap_ratio: float,
) -> pa.EnvelopeCandidate | None:
    face = _lower_payload_face_for_region(polygon, left, right)
    top_plane = pa._top_plane_up(face.plane)
    if not pa._top_plane_is_above_floor(
        polygon,
        top_plane,
        floor_y=floor_y,
        min_clearance_m=TOP_ABOVE_FLOOR_SLACK_M,
    ):
        return None
    return pa._envelope_candidate_from_top_plane(
        polygon,
        face,
        floor_y=floor_y,
        locator_id=locator_id,
        overlap_ratio=max(1.0, min_top_overlap_ratio),
        top_source=f"{face.locator_id} (single side of selected gable pair)",
    )


def _lower_payload_face_for_region(
    region: Polygon,
    left: pa.PayloadFace,
    right: pa.PayloadFace,
) -> pa.PayloadFace:
    point = region.representative_point()
    left_plane = pa._top_plane_up(left.plane)
    right_plane = pa._top_plane_up(right.plane)
    left_y = pa._plane_y_at(left_plane, float(point.x), float(point.y))
    right_y = pa._plane_y_at(right_plane, float(point.x), float(point.y))
    return left if left_y <= right_y else right


def _merge_selected_cells(selected_cells: list[SelectedCell]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {
        index: set() for index in range(len(selected_cells))
    }
    for i, left in enumerate(selected_cells):
        for j in range(i + 1, len(selected_cells)):
            right = selected_cells[j]
            if left.cell.story != right.cell.story:
                continue
            if not _selected_tops_are_merge_compatible(left.top, right.top):
                continue
            if abs(left.cell.floor_y - right.cell.floor_y) > 0.15:
                continue
            try:
                shared = float(
                    left.cell.polygon.boundary.intersection(
                        right.cell.polygon.boundary
                    ).length
                )
                distance = float(left.cell.polygon.distance(right.cell.polygon))
            except Exception:
                continue
            if shared < MIN_SHARED_BOUNDARY_M and distance > ADJACENCY_TOL_M:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(selected_cells)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        component: list[int] = []
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _representative_top_for_selected_component(
    selected_component: list[SelectedCell],
) -> SelectedTop:
    """Pick the strongest top model for a merged compatible component."""

    def key(selected: SelectedCell) -> tuple[int, float, float]:
        priority = {
            "flat-ceiling": 0,
            "single-oblique": 1,
            "gable-pair": 2,
            "multi-piece": 2,
        }.get(selected.top.label, -1)
        return (priority, float(selected.cell.polygon.area), selected.top.score)

    return max(selected_component, key=key).top


def _selected_tops_are_merge_compatible(
    left: SelectedTop,
    right: SelectedTop,
) -> bool:
    if left.signature == right.signature:
        return True
    if left.label == "gable-pair" and right.label == "gable-pair":
        return set(left.signature[1:]) == set(right.signature[1:])
    if left.label == "gable-pair" and right.label == "single-oblique":
        return _single_oblique_signature_in_gable_pair(right.signature, left.signature)
    if right.label == "gable-pair" and left.label == "single-oblique":
        return _single_oblique_signature_in_gable_pair(left.signature, right.signature)
    return False


def _single_oblique_signature_in_gable_pair(
    single_signature: tuple[Any, ...],
    gable_signature: tuple[Any, ...],
) -> bool:
    if len(single_signature) < 2 or len(gable_signature) < 3:
        return False
    return single_signature[1] in set(gable_signature[1:])


def _room_audit(
    payload: dict[str, Any],
    *,
    cells: list[PlanCell],
    selected_cells: list[SelectedCell],
    candidates: list[pa.EnvelopeCandidate],
    ceiling_faces: list[pa.PayloadFace],
    footprint: Polygon,
    corner_tol: float,
    plane_groups: list[PlaneGroupEvidence],
    gable_pairs: list[GablePairEvidence],
    domain_top_choices: list[DomainTopChoice],
) -> dict[str, Any]:
    _ = domain_top_choices
    rooms = []
    selected_by_room: dict[int, list[SelectedCell]] = {}
    for selected in selected_cells:
        for room_index in selected.cell.source_room_indices:
            selected_by_room.setdefault(room_index, []).append(selected)
    candidate_polys = [
        (poly, _candidate_floor_y(candidate))
        for candidate in candidates
        if (poly := pa._candidate_footprint_polygon(candidate)) is not None
    ]

    for room_index, room in enumerate(payload.get("rooms", []) or []):
        floor_poly = _room_floor_polygon(room, footprint, corner_tol=corner_tol)
        if floor_poly is None:
            rooms.append(
                _room_audit_record(
                    room,
                    room_index,
                    0.0,
                    0.0,
                    "dropped",
                    "no_floor_polygon",
                )
            )
            continue
        if floor_poly.area < MIN_CELL_AREA_M2:
            rooms.append(
                _room_audit_record(
                    room,
                    room_index,
                    float(floor_poly.area),
                    0.0,
                    "dropped",
                    "floor_outside_footprint",
                )
            )
            continue
        selected_for_room = selected_by_room.get(room_index, [])
        selected_union = (
            _safe_unary_union([selected.cell.polygon for selected in selected_for_room])
            if selected_for_room
            else None
        )
        selected_ratio = (
            _safe_overlap_ratio(floor_poly, selected_union)
            if selected_union is not None and not selected_union.is_empty
            else 0.0
        )
        floor_y = _room_floor_y(room, corner_tol=corner_tol)
        floor_matched_candidate_polys = [
            poly
            for poly, candidate_floor_y in candidate_polys
            if floor_y is None
            or candidate_floor_y is None
            or abs(candidate_floor_y - floor_y) <= 0.15
        ]
        candidate_union = (
            _safe_unary_union(floor_matched_candidate_polys)
            if floor_matched_candidate_polys
            else None
        )
        candidate_ratio = (
            _safe_overlap_ratio(floor_poly, candidate_union)
            if candidate_union is not None and not candidate_union.is_empty
            else 0.0
        )
        coverage_ratio = candidate_ratio if candidates else selected_ratio
        status = (
            "covered"
            if coverage_ratio >= ROOM_COVERED_RATIO
            else "partial"
            if coverage_ratio >= 0.50
            else "dropped"
        )
        best_top_candidates = _best_room_top_candidates(floor_poly, ceiling_faces)
        best_part_plane_groups = _best_room_part_plane_groups(
            floor_poly,
            floor_y=floor_y,
            plane_groups=plane_groups,
        )
        best_part_gable_pairs = _best_room_gable_pairs(
            floor_poly,
            floor_y=floor_y,
            gable_pairs=gable_pairs,
        )
        if any(
            selected.top.reason == "weak_room_support_promoted_by_part_plane"
            for selected in selected_for_room
        ):
            reason = "weak_room_support_promoted_by_part_plane"
        elif status == "covered":
            reason = "candidate_covered"
        elif not best_top_candidates:
            reason = "no_top_support"
        elif _room_top_candidates_below_floor(
            room,
            floor_poly,
            ceiling_faces,
            corner_tol=corner_tol,
        ):
            reason = "top_below_floor"
        else:
            reason = "top_coverage_too_low"
        rooms.append(
            _room_audit_record(
                room,
                room_index,
                float(floor_poly.area),
                coverage_ratio,
                status,
                reason,
                selected_cells=selected_for_room,
                best_top_candidates=best_top_candidates,
                selected_coverage_ratio=selected_ratio,
                strict_candidate_coverage_ratio=candidate_ratio,
                best_part_plane_groups=best_part_plane_groups,
                best_part_gable_pairs=best_part_gable_pairs,
            )
        )

    rooms_ge80 = sum(
        1 for room in rooms if room["coverage_ratio"] >= ROOM_COVERED_RATIO
    )
    rooms_ge50 = sum(1 for room in rooms if room["coverage_ratio"] >= 0.50)
    return {
        "schema_version": 1,
        "summary": {
            "rooms_total": len(rooms),
            "rooms_ge80": rooms_ge80,
            "rooms_ge50": rooms_ge50,
            "dropped_rooms": sum(1 for room in rooms if room["status"] == "dropped"),
            "reasons": dict(Counter(room["reason"] for room in rooms)),
        },
        "rooms": rooms,
        "cells": [_cell_json(cell) for cell in cells],
    }


def _audit_json(result: CellSelectorResult) -> dict[str, Any]:
    audit = dict(result.room_audit)
    audit["plane_groups"] = [_plane_group_json(group) for group in result.plane_groups]
    audit["gable_pairs"] = [_gable_pair_json(pair) for pair in result.gable_pairs]
    audit["domain_top_choices"] = [
        _domain_top_choice_json(choice) for choice in result.domain_top_choices
    ]
    audit["selected_cells"] = [
        _selected_cell_json(selected) for selected in result.selected_cells
    ]
    audit["build_attempts"] = result.build_attempts
    audit["top_label_summary"] = result.top_label_summary
    return audit


def _room_coverage_for_candidates(
    payload: dict[str, Any],
    candidates: list[pa.EnvelopeCandidate],
    corner_tol: float,
) -> dict[str, Any]:
    candidate_polys = [
        (poly, _candidate_floor_y(candidate))
        for candidate in candidates
        if (poly := pa._candidate_footprint_polygon(candidate)) is not None
    ]
    rooms_total = 0
    rooms_ge80 = 0
    rooms_ge50 = 0
    for room in payload.get("rooms", []) or []:
        floor = _room_floor_polygon(room, None, corner_tol=corner_tol)
        if floor is None or floor.area < MIN_CELL_AREA_M2:
            continue
        rooms_total += 1
        floor_y = _room_floor_y(room, corner_tol=corner_tol)
        polys = [
            poly
            for poly, candidate_floor_y in candidate_polys
            if floor_y is None
            or candidate_floor_y is None
            or abs(candidate_floor_y - floor_y) <= 0.15
        ]
        unioned = unary_union(polys) if polys else None
        coverage = (
            float(floor.intersection(unioned).area) / max(float(floor.area), 1e-9)
            if unioned is not None and not unioned.is_empty
            else 0.0
        )
        if coverage >= ROOM_COVERED_RATIO:
            rooms_ge80 += 1
        if coverage >= 0.50:
            rooms_ge50 += 1
    return {
        "rooms_total": rooms_total,
        "rooms_ge80": rooms_ge80,
        "rooms_ge50": rooms_ge50,
    }


def _candidate_floor_y(candidate: pa.EnvelopeCandidate) -> float | None:
    for face in candidate.faces:
        if face.kind != "floor" or not face.corners:
            continue
        try:
            return float(sum(corner[1] for corner in face.corners) / len(face.corners))
        except Exception:
            return None
    return None


def _room_polygons(
    payload: dict[str, Any],
    *,
    corner_tol: float,
) -> list[tuple[int, int, Polygon]]:
    out = []
    for room_index, room in enumerate(payload.get("rooms", []) or []):
        poly = _room_floor_polygon(room, None, corner_tol=corner_tol)
        if poly is None:
            continue
        out.append((room_index, pa._int_or_none(room.get("story")) or 0, poly))
    return out


def _exposed_masks_by_room(
    payload: dict[str, Any],
    footprint: Polygon,
    *,
    corner_tol: float,
) -> dict[int, Polygon]:
    room_polys = _room_polygons(payload, corner_tol=corner_tol)
    by_story: dict[int, list[tuple[int, Polygon]]] = {}
    for room_index, story, poly in room_polys:
        try:
            clipped = pa._largest_polygon(poly.intersection(footprint))
        except Exception:
            clipped = None
        if clipped is not None and clipped.area >= MIN_CELL_AREA_M2:
            by_story.setdefault(story, []).append((room_index, clipped))

    out: dict[int, Polygon] = {}
    for room_index, story, poly in room_polys:
        above_polys = [
            above_poly
            for above_story, items in by_story.items()
            if above_story > story
            for _above_index, above_poly in items
        ]
        if above_polys:
            try:
                covered = unary_union(above_polys).buffer(0.03)
                exposed = poly.difference(covered)
            except Exception:
                exposed = poly
        else:
            exposed = poly
        exposed_poly = pa._largest_polygon(exposed)
        if exposed_poly is not None and exposed_poly.area >= MIN_CELL_AREA_M2:
            out[room_index] = exposed_poly
    return out


def _cell_exposed_ratio(
    polygon: Polygon,
    *,
    room_index: int,
    exposed_masks: dict[int, Polygon],
) -> float:
    exposed = exposed_masks.get(room_index)
    if exposed is None:
        return 0.0
    try:
        return float(polygon.intersection(exposed).area) / max(
            float(polygon.area),
            1e-9,
        )
    except Exception:
        return 0.0


def _split_cell_by_exposure(
    polygon: Polygon,
    *,
    room_index: int,
    exposed_masks: dict[int, Polygon],
) -> list[tuple[Polygon, float]]:
    exposed = exposed_masks.get(room_index)
    if exposed is None:
        return [(polygon, 0.0)]
    try:
        exposed_geom = polygon.intersection(exposed)
        covered_geom = polygon.difference(exposed)
    except Exception:
        return [
            (
                polygon,
                _cell_exposed_ratio(
                    polygon,
                    room_index=room_index,
                    exposed_masks=exposed_masks,
                ),
            )
        ]

    out: list[tuple[Polygon, float]] = []
    for part in _polygon_components(exposed_geom):
        if part.area >= MIN_CELL_AREA_M2:
            out.append((part, 1.0))
    for part in _polygon_components(covered_geom):
        if part.area >= MIN_CELL_AREA_M2:
            out.append((part, 0.0))
    return out


def _polygon_components(geom: Any) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [part for part in getattr(geom, "geoms", []) if isinstance(part, Polygon)]


def _room_floor_polygon(
    room: dict[str, Any],
    footprint: Polygon | None,
    *,
    corner_tol: float,
) -> Polygon | None:
    for floor in room.get("floor", []) or []:
        poly = pa._polygon_xz(pa._clean_ring(pa._corners(floor), tol=corner_tol))
        if poly is None:
            continue
        if footprint is not None:
            try:
                poly = pa._largest_polygon(poly.intersection(footprint))
            except Exception:
                return None
        return poly
    return None


def _room_floor_y(room: dict[str, Any], *, corner_tol: float) -> float | None:
    for floor in room.get("floor", []) or []:
        corners = pa._clean_ring(pa._corners(floor), tol=corner_tol)
        if len(corners) >= 3:
            return float(sum(corner[1] for corner in corners) / len(corners))
    return None


def _room_top_candidates_below_floor(
    room: dict[str, Any],
    floor_poly: Polygon,
    ceiling_faces: list[pa.PayloadFace],
    *,
    corner_tol: float,
) -> bool:
    floor_y = _room_floor_y(room, corner_tol=corner_tol)
    if floor_y is None:
        return False
    saw_overlap = False
    for face in ceiling_faces:
        poly = pa._polygon_xz(face.corners)
        if poly is None:
            continue
        try:
            if float(floor_poly.intersection(poly).area) <= 0.05:
                continue
        except Exception:
            continue
        saw_overlap = True
        plane = pa._top_plane_up(face.plane)
        point = floor_poly.representative_point()
        try:
            if (
                pa._plane_y_at(plane, float(point.x), float(point.y))
                >= floor_y + TOP_ABOVE_FLOOR_SLACK_M
            ):
                return False
        except Exception:
            continue
    return saw_overlap


def _best_domain_index(poly: Polygon, domains: list[Polygon]) -> int:
    best: tuple[int, float, float, int] | None = None
    for index, domain in enumerate(domains):
        try:
            overlap = float(poly.intersection(domain).area)
        except Exception:
            continue
        overlap_ratio = overlap / max(float(poly.area), 1e-9)
        if overlap_ratio < 0.50:
            continue
        priority = 0 if overlap_ratio >= 0.80 else 1
        candidate = (priority, float(domain.area), -overlap_ratio, index)
        if best is None or candidate < best:
            best = candidate
    if best is not None:
        return best[3]
    fallback = (0.0, len(domains) - 1)
    for index, domain in enumerate(domains):
        try:
            overlap = float(poly.intersection(domain).area)
        except Exception:
            continue
        if overlap > fallback[0]:
            fallback = (overlap, index)
    return fallback[1]


def _top_is_above_cell(
    cell: PlanCell,
    regions: list[tuple[Polygon, Any]],
) -> bool:
    for region, plane_or_face in regions:
        plane = (
            pa._top_plane_up(plane_or_face.plane)
            if hasattr(plane_or_face, "plane")
            else pa._top_plane_up(plane_or_face)
        )
        points = [
            region.representative_point(),
            *[pa.Point(x, z) for x, z in region.exterior.coords[:-1]],
        ]
        for point in points:
            if (
                pa._plane_y_at(plane, float(point.x), float(point.y))
                < cell.floor_y + TOP_ABOVE_FLOOR_SLACK_M
            ):
                return False
    return True


def _plane_inclination_deg(plane: pa.Plane) -> float:
    plane = pa._top_plane_up(plane)
    normal_len = max(hypot(hypot(plane.a, plane.c), plane.b), 1e-12)
    vertical = min(1.0, max(0.0, abs(plane.b) / normal_len))
    import math

    return float(math.degrees(math.acos(vertical)))


def _best_room_top_candidates(
    floor_poly: Polygon,
    ceiling_faces: list[pa.PayloadFace],
) -> list[dict[str, Any]]:
    out = []
    for face in ceiling_faces:
        poly = pa._polygon_xz(face.corners)
        if poly is None:
            continue
        try:
            overlap = float(floor_poly.intersection(poly).area)
        except Exception:
            continue
        ratio = overlap / max(float(floor_poly.area), 1e-9)
        if ratio <= 0.0:
            continue
        out.append(
            {
                "locator_id": face.locator_id,
                "source": face.source,
                "overlap_ratio": ratio,
                "plane_key": pa._top_plane_group_key(face.plane),
                "inclination_deg": _plane_inclination_deg(face.plane),
            }
        )
    return sorted(out, key=lambda item: item["overlap_ratio"], reverse=True)[:3]


def _best_room_part_plane_groups(
    floor_poly: Polygon,
    *,
    floor_y: float | None,
    plane_groups: list[PlaneGroupEvidence],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in plane_groups:
        try:
            overlap = float(floor_poly.intersection(group.footprint).area)
        except Exception:
            continue
        ratio = overlap / max(float(floor_poly.area), 1e-9)
        if ratio <= 0.0:
            continue
        clearance = (
            _plane_clearance_stats(
                floor_poly,
                [pa._top_plane_up(group.representative.plane)],
                floor_y=floor_y,
            )
            if floor_y is not None
            else None
        )
        out.append(
            {
                "key": list(group.key),
                "label": group.label,
                "representative": group.representative.locator_id,
                "domain_index": group.domain_index,
                "room_overlap_ratio": ratio,
                "support_ratio": group.support_ratio,
                "support_area_m2": group.support_area_m2,
                "source_count": len(group.source_locators),
                "source_room_indices": list(group.source_room_indices),
                "source_face_room_indices": list(group.source_face_room_indices),
                "top_clearance": clearance,
            }
        )
    return sorted(
        out,
        key=lambda item: (
            float(item["room_overlap_ratio"]),
            float(item["support_ratio"]),
            float(item["support_area_m2"]),
        ),
        reverse=True,
    )[:5]


def _best_room_gable_pairs(
    floor_poly: Polygon,
    *,
    floor_y: float | None,
    gable_pairs: list[GablePairEvidence],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pair in gable_pairs:
        try:
            overlap = float(floor_poly.intersection(pair.footprint).area)
        except Exception:
            continue
        ratio = overlap / max(float(floor_poly.area), 1e-9)
        if ratio <= 0.0:
            continue
        clearance = (
            _plane_clearance_stats(
                floor_poly,
                [
                    pa._top_plane_up(pair.left.representative.plane),
                    pa._top_plane_up(pair.right.representative.plane),
                ],
                floor_y=floor_y,
                use_highest=True,
            )
            if floor_y is not None
            else None
        )
        out.append(
            {
                "signature": str(pair.signature),
                "domain_index": pair.domain_index,
                "room_overlap_ratio": ratio,
                "support_ratio": pair.support_ratio,
                "support_area_m2": pair.support_area_m2,
                "left_representative": pair.left.representative.locator_id,
                "right_representative": pair.right.representative.locator_id,
                "top_clearance": clearance,
            }
        )
    return sorted(
        out,
        key=lambda item: (
            float(item["room_overlap_ratio"]),
            float(item["support_ratio"]),
            float(item["support_area_m2"]),
        ),
        reverse=True,
    )[:3]


def _plane_clearance_stats(
    poly: Polygon,
    planes: list[pa.Plane],
    *,
    floor_y: float | None,
    use_highest: bool = False,
) -> dict[str, float] | None:
    if floor_y is None or not planes:
        return None
    points = [
        poly.representative_point(),
        *[pa.Point(x, z) for x, z in poly.exterior.coords[:-1]],
    ]
    clearances: list[float] = []
    for point in points:
        ys = [
            pa._plane_y_at(plane, float(point.x), float(point.y))
            for plane in planes
        ]
        top_y = max(ys) if use_highest else ys[0]
        clearances.append(float(top_y - floor_y))
    if not clearances:
        return None
    return {
        "min_m": min(clearances),
        "max_m": max(clearances),
        "passes_slack": min(clearances) >= TOP_ABOVE_FLOOR_SLACK_M,
    }


def _room_audit_record(
    room: dict[str, Any],
    room_index: int,
    floor_area_m2: float,
    coverage_ratio: float,
    status: str,
    reason: str,
    *,
    selected_cells: list[SelectedCell] | None = None,
    best_top_candidates: list[dict[str, Any]] | None = None,
    selected_coverage_ratio: float | None = None,
    strict_candidate_coverage_ratio: float | None = None,
    best_part_plane_groups: list[dict[str, Any]] | None = None,
    best_part_gable_pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "room_index": room_index,
        "room_locator_id": room.get("locator_id"),
        "story": room.get("story"),
        "floor_area_m2": floor_area_m2,
        "coverage_ratio": coverage_ratio,
        "selected_coverage_ratio": selected_coverage_ratio
        if selected_coverage_ratio is not None
        else coverage_ratio,
        "strict_candidate_coverage_ratio": strict_candidate_coverage_ratio
        if strict_candidate_coverage_ratio is not None
        else 0.0,
        "status": status,
        "reason": reason,
        "selected_cell_ids": [
            selected.cell.cell_id for selected in selected_cells or []
        ],
        "selected_top_labels": [
            selected.top.label for selected in selected_cells or []
        ],
        "selected_cell_debug": [
            _selected_cell_room_debug(selected) for selected in selected_cells or []
        ],
        "best_top_candidates": best_top_candidates or [],
        "best_part_plane_groups": best_part_plane_groups or [],
        "best_part_gable_pairs": best_part_gable_pairs or [],
    }


def _selected_cell_room_debug(selected: SelectedCell) -> dict[str, Any]:
    return {
        "cell_id": selected.cell.cell_id,
        "area_m2": float(selected.cell.polygon.area),
        "floor_y": selected.cell.floor_y,
        "exposed_ratio": selected.cell.exposed_ratio,
        "top_label": selected.top.label,
        "score": selected.top.score,
        "local_coverage": selected.top.local_coverage,
        "part_support_ratio": selected.top.part_support_ratio,
        "reason": selected.top.reason,
        "top_sources": [face.locator_id for face in selected.top.faces],
    }


def _cell_json(cell: PlanCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "story": cell.story,
        "room_indices": list(cell.source_room_indices),
        "floor_y": cell.floor_y,
        "exposed_ratio": cell.exposed_ratio,
        "area_m2": float(cell.polygon.area),
        "polygon": [
            [float(x), float(z)]
            for x, z in pa._polygon_exterior_coords(cell.polygon)
        ],
    }


def _selected_cell_json(selected: SelectedCell) -> dict[str, Any]:
    data = _cell_json(selected.cell)
    data.update(
        {
            "top_label": selected.top.label,
            "top_signature": str(selected.top.signature),
            "score": selected.top.score,
            "local_coverage": selected.top.local_coverage,
            "part_support_ratio": selected.top.part_support_ratio,
            "reason": selected.top.reason,
            "top_sources": [face.locator_id for face in selected.top.faces],
        }
    )
    return data


def _plane_group_json(group: PlaneGroupEvidence) -> dict[str, Any]:
    return {
        "key": list(group.key),
        "label": group.label,
        "representative": group.representative.locator_id,
        "domain_index": group.domain_index,
        "domain_area_m2": group.domain_area_m2,
        "support_area_m2": group.support_area_m2,
        "support_ratio": group.support_ratio,
        "source_locators": list(group.source_locators),
        "stories": list(group.stories),
        "source_room_indices": list(group.source_room_indices),
        "source_face_room_indices": list(group.source_face_room_indices),
        "inclination_deg": group.inclination_deg,
        "footprint": [
            [float(x), float(z)]
            for x, z in pa._polygon_exterior_coords(group.footprint)
        ],
    }


def _gable_pair_json(pair: GablePairEvidence) -> dict[str, Any]:
    footprint = pa._largest_polygon(pair.footprint)
    return {
        "signature": str(pair.signature),
        "left_key": list(pair.left.key),
        "right_key": list(pair.right.key),
        "domain_index": pair.domain_index,
        "support_area_m2": pair.support_area_m2,
        "support_ratio": pair.support_ratio,
        "footprint": [
            [float(x), float(z)] for x, z in pa._polygon_exterior_coords(footprint)
        ]
        if footprint is not None
        else [],
    }


def _domain_top_choice_json(choice: DomainTopChoice) -> dict[str, Any]:
    return {
        "domain_index": choice.domain_index,
        "label": choice.label,
        "signature": str(choice.signature),
        "top_sources": [face.locator_id for face in choice.faces],
        "score": choice.score,
        "support_ratio": choice.support_ratio,
        "source_room_ratio": choice.source_room_ratio,
        "reason": choice.reason,
    }
