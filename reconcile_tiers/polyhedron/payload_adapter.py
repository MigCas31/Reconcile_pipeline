"""Adapters from tier_payload-style dicts to half-edge polyhedron inputs.

This module deliberately stops at producing planar face candidates and a
strict constructor wrapper. A full ``tier_payload`` is not guaranteed to be a
single watertight manifold: it contains per-room floors/walls plus top-level
ceiling pieces, including duplicate/internal room surfaces. The wrapper
therefore surfaces manifold violations from ``build_from_planar_polygons``
instead of silently repairing topology.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import atan2, degrees, hypot
from typing import Any, Literal

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient as orient_polygon
from shapely.ops import split, unary_union

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import Plane, fit_plane_any
from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers._core.wing_decomposition import decompose_to_wings
from reconcile_tiers._core.wing_decomposition_v2 import decompose_to_wings_v2
from reconcile_tiers.polyhedron.half_edge import (
    HalfEdgePolyhedron,
    build_from_planar_polygons,
)

FaceKind = Literal["floor", "wall", "ceiling"]


@dataclass(frozen=True, slots=True)
class PayloadFace:
    """One payload polygon converted to the half-edge constructor's convention."""

    kind: FaceKind
    locator_id: str
    corners: list[list[float]]
    plane: Plane
    source: str | None = None
    room_index: int | None = None
    story: int | None = None


@dataclass(frozen=True, slots=True)
class EnvelopeCandidate:
    """One building/wing-level envelope candidate for strict polyhedron import."""

    locator_id: str
    faces: list[PayloadFace]
    footprint_area_m2: float
    top_source: str
    top_overlap_ratio: float
    selector: str = "legacy"
    room_coverage: dict[str, Any] | None = None
    part_plane_groups: list[dict[str, Any]] | None = None
    top_label_summary: dict[str, int] | None = None


Point2 = tuple[float, float]
BoundarySegment = tuple[Point2, Point2, PayloadFace]


@dataclass(frozen=True, slots=True)
class TopSupport:
    face: PayloadFace
    footprint: Polygon
    area: float


@dataclass(frozen=True, slots=True)
class RoomPartitionCell:
    room_index: int
    story: int
    polygon: Polygon
    top_signature: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class RoomPartitionPart:
    polygon: Polygon
    top_signature: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class RoofPartitionFamily:
    signature: tuple[Any, ...]
    footprint: Any
    area: float


@dataclass(frozen=True, slots=True)
class RidgeFrame:
    """Local ridge reference for one building part in XZ coordinates."""

    axis_deg: float
    line_a: float
    line_b: float
    line_c: float
    support_length_m: float


@dataclass(frozen=True, slots=True)
class RidgeStoryTransform:
    """Rigid XZ correction for one story in a local ridge frame."""

    story: int
    origin_x: float
    origin_z: float
    yaw_delta_deg: float
    shift_x: float
    shift_z: float


def payload_faces_from_tier_payload(
    payload: dict[str, Any],
    *,
    include: Iterable[FaceKind] = ("floor", "wall", "ceiling"),
    corner_tol: float = 0.02,
) -> list[PayloadFace]:
    """Extract floor, wall, and ceiling polygons as oriented planar faces.

    Payload points may be dicts (``{"x","y","z"}``) or raw triples. Rings are
    cleaned for consecutive near-duplicates because real floor polygons can
    contain zero-length sliver edges after clipping.
    """

    include_set = set(include)
    out: list[PayloadFace] = []

    for room_index, room in enumerate(payload.get("rooms", []) or []):
        floor_rings = [
            _clean_ring(_corners(floor), tol=corner_tol)
            for floor in room.get("floor", []) or []
        ]
        primary_floor = next((ring for ring in floor_rings if len(ring) >= 3), None)
        room_center = _room_center(room)
        story = _int_or_none(room.get("story"))
        room_id = str(room.get("locator_id") or f"room:{room_index}")

        if "floor" in include_set:
            for floor_index, corners in enumerate(floor_rings):
                if len(corners) < 3:
                    continue
                plane = _floor_plane(corners)
                out.append(
                    PayloadFace(
                        kind="floor",
                        locator_id=f"{room_id}::floor::{floor_index}",
                        corners=_orient_to_plane(corners, plane),
                        plane=plane,
                        room_index=room_index,
                        story=story,
                    )
                )

        if "wall" in include_set:
            for wall_index, wall in enumerate(room.get("walls", []) or []):
                corners = _clean_ring(_corners(wall), tol=corner_tol)
                if len(corners) < 3:
                    continue
                corners = _orient_wall_outward(
                    corners,
                    room_center=room_center,
                    floor_corners=primary_floor,
                )
                plane = _plane_from_oriented_polygon(corners)
                if plane is None:
                    continue
                out.append(
                    PayloadFace(
                        kind="wall",
                        locator_id=str(
                            wall.get("locator_id")
                            or f"{room_id}::wall::{wall_index}"
                        ),
                        corners=_orient_to_plane(corners, plane),
                        plane=plane,
                        room_index=room_index,
                        story=story,
                    )
                )

    if "ceiling" in include_set:
        for ceiling_index, ceiling in enumerate(payload.get("ceiling", []) or []):
            corners = _clean_ring(_corners(ceiling), tol=corner_tol)
            if len(corners) < 3:
                continue
            plane_dict = ceiling.get("plane")
            if not isinstance(plane_dict, dict):
                continue
            plane = _plane_from_dict(plane_dict)
            out.append(
                PayloadFace(
                    kind="ceiling",
                    locator_id=str(
                        ceiling.get("locator_id") or f"ceiling::{ceiling_index}"
                    ),
                    corners=_orient_to_plane(corners, plane),
                    plane=plane,
                    source=(
                        str(ceiling.get("source"))
                        if ceiling.get("source") is not None
                        else None
                    ),
                    room_index=_room_index_from_locator(ceiling.get("locator_id")),
                    story=None,
                )
            )

    return out


def payload_faces_from_plane_evidence(
    evidence: dict[str, Any] | None,
    *,
    corner_tol: float = 0.02,
    include_filtered_candidates: bool = True,
) -> list[PayloadFace]:
    """Extract ceiling candidate faces from `plane_evidence.json`.

    The sidecar is intentionally broader than `tier_payload.ceiling`: it holds
    pre-painter candidates and raw observed planes that may have been filtered
    out of the final payload. These faces are diagnostic/candidate evidence,
    not final geometry.
    """

    if not isinstance(evidence, dict):
        return []
    out: list[PayloadFace] = []
    for item in evidence.get("raw_ceiling_planes") or []:
        face = _payload_face_from_evidence_item(
            item,
            locator_prefix="raw-evidence",
            source_override="raw_observed_ceiling_plane",
            corner_tol=corner_tol,
        )
        if face is not None:
            out.append(face)
    for item in evidence.get("ceiling_candidates") or []:
        if not include_filtered_candidates and not item.get("kept_after_raw_gate"):
            continue
        face = _payload_face_from_evidence_item(
            item,
            locator_prefix="candidate-evidence",
            source_override=None,
            corner_tol=corner_tol,
        )
        if face is not None:
            out.append(face)
    return out


def _payload_face_from_evidence_item(
    item: Any,
    *,
    locator_prefix: str,
    source_override: str | None,
    corner_tol: float,
) -> PayloadFace | None:
    if not isinstance(item, dict):
        return None
    corners = _clean_ring(_corners(item), tol=corner_tol)
    if len(corners) < 3:
        return None
    plane_data = item.get("plane")
    if not isinstance(plane_data, dict):
        plane = _plane_from_oriented_polygon(corners)
        if plane is None:
            return None
    else:
        plane = _plane_from_dict(plane_data)
    locator_id = str(item.get("locator_id") or f"{locator_prefix}:{len(corners)}")
    return PayloadFace(
        kind="ceiling",
        locator_id=locator_id,
        corners=_orient_to_plane(corners, plane),
        plane=plane,
        source=source_override
        if source_override is not None
        else str(item.get("source")) if item.get("source") is not None else None,
        room_index=_room_index_from_locator(locator_id)
        if _room_index_from_locator(locator_id) is not None
        else _int_or_none(item.get("room_index")),
        story=_int_or_none(item.get("story")),
    )


def payload_envelope_candidates_from_tier_payload(
    payload: dict[str, Any],
    *,
    ceiling_faces: list[PayloadFace] | None = None,
    force_cell_selector: bool = False,
    wing_level: bool = True,
    min_top_overlap_ratio: float = 0.60,
    room_buffer_m: float = 0.3,
    footprint_shrink_m: float = 0.3,
    corner_tol: float = 0.02,
) -> list[EnvelopeCandidate]:
    """Build building/wing envelope candidates from footprint + top planes.

    Unlike ``payload_faces_for_room_shell``, this derives the vertical boundary
    from the unioned building footprint. It first emits candidates when one
    non-vertical top plane covers most of the footprint/wing. If no dominant
    top exists, it tries a two-plane split at the equal-height line between
    roof planes, which covers simple gable-like building parts.
    """

    footprint = _payload_footprint_polygon(
        payload,
        room_buffer_m=room_buffer_m,
        footprint_shrink_m=footprint_shrink_m,
        corner_tol=corner_tol,
    )
    if footprint is None:
        return []

    if ceiling_faces is None:
        faces = payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
        ceiling_faces = [face for face in faces if face.kind == "ceiling"]
    candidates: list[EnvelopeCandidate] = []

    if wing_level:
        try:
            from reconcile_tiers.polyhedron.cell_selector import (
                select_payload_cells,
            )

            cell_result = select_payload_cells(
                payload,
                footprint=footprint,
                ceiling_faces=ceiling_faces,
                min_top_overlap_ratio=min_top_overlap_ratio,
                corner_tol=corner_tol,
            )
        except Exception:
            cell_result = None
        if cell_result is not None and (
            force_cell_selector
            or _cell_selector_result_is_safe(
                cell_result,
                ceiling_faces,
            )
        ):
            return cell_result.candidates

    if wing_level:
        geometric_parts = [wing.polygon for wing in decompose_to_wings(footprint)]
        if not geometric_parts:
            geometric_parts = [footprint]
        cell_parts = _payload_roof_labeled_room_parts(
            payload,
            footprint=footprint,
            ceiling_faces=ceiling_faces,
            corner_tol=corner_tol,
        )
        cell_part_polygons = [part.polygon for part in cell_parts]
        cell_refines_geometric = _partition_refines(
            cell_part_polygons,
            geometric_parts,
        )
        if cell_refines_geometric:
            candidates.extend(
                _envelope_candidates_for_parts(
                    payload,
                    cell_part_polygons,
                    ceiling_faces,
                    part_ceiling_faces=[
                        _ceiling_faces_for_partition_signature(
                            ceiling_faces,
                            part.top_signature,
                        )
                        for part in cell_parts
                    ],
                    min_top_overlap_ratio=min_top_overlap_ratio,
                    corner_tol=corner_tol,
                    locator_prefix="envelope-cell",
                    strict_gate_all=True,
                )
            )
            geometric_parts = _parts_not_covered_by_candidates(
                geometric_parts,
                candidates,
            )
        graph_parts = _payload_room_graph_wing_polygons(
            payload,
            footprint=footprint,
            corner_tol=corner_tol,
        )
        graph_refines_geometric = _partition_refines(graph_parts, geometric_parts)
        if graph_refines_geometric:
            candidates.extend(
                _envelope_candidates_for_parts(
                    payload,
                    graph_parts,
                    ceiling_faces,
                    min_top_overlap_ratio=min_top_overlap_ratio,
                    corner_tol=corner_tol,
                    locator_prefix="envelope-part",
                    strict_gate_all=True,
                )
            )
            geometric_parts = _parts_not_covered_by_candidates(
                geometric_parts,
                candidates,
            )
        candidates.extend(
            _envelope_candidates_for_parts(
                payload,
                geometric_parts,
                ceiling_faces,
                min_top_overlap_ratio=min_top_overlap_ratio,
                corner_tol=corner_tol,
                locator_prefix="envelope-wing",
                strict_gate_all=False,
            )
        )
        if not graph_refines_geometric:
            graph_parts = _parts_not_covered_by_candidates(
                graph_parts,
                candidates,
            )
            candidates.extend(
                _envelope_candidates_for_parts(
                    payload,
                    graph_parts,
                    ceiling_faces,
                    min_top_overlap_ratio=min_top_overlap_ratio,
                    corner_tol=corner_tol,
                    locator_prefix="envelope-wing-v2",
                    strict_gate_all=True,
                )
            )
    else:
        candidates = _envelope_candidates_for_parts(
            payload,
            [footprint],
            ceiling_faces,
            min_top_overlap_ratio=min_top_overlap_ratio,
            corner_tol=corner_tol,
            locator_prefix="envelope-wing",
            strict_gate_all=False,
        )
    return candidates


def _cell_selector_result_is_safe(
    result: Any,
    ceiling_faces: list[PayloadFace],
) -> bool:
    candidates = getattr(result, "candidates", None) or []
    if not candidates:
        return False

    room_audit = getattr(result, "room_audit", None) or {}
    room_coverage = getattr(
        candidates[0],
        "room_coverage",
        None,
    ) or room_audit.get("summary") or {}
    total_rooms = int(room_coverage.get("rooms_total") or 0)
    rooms_ge80 = int(room_coverage.get("rooms_ge80") or 0)

    has_oblique_roof_evidence = any(
        _top_plane_is_oblique(face.plane) for face in ceiling_faces
    )
    if (
        has_oblique_roof_evidence
        and total_rooms > 0
        and rooms_ge80 / total_rooms < 0.75
    ):
        return False

    label_summary = getattr(result, "top_label_summary", None) or {}
    sloped_count = int(label_summary.get("single-oblique") or 0) + int(
        label_summary.get("gable-pair") or 0
    )
    if has_oblique_roof_evidence and sloped_count == 0:
        return False

    return True


def build_envelope_polyhedra_from_tier_payload(
    payload: dict[str, Any],
    *,
    wing_level: bool = True,
    min_top_overlap_ratio: float = 0.60,
    coord_tol: float = 1e-3,
    corner_tol: float = 0.02,
) -> list[tuple[EnvelopeCandidate, HalfEdgePolyhedron]]:
    """Build strict building/wing-level envelope polyhedra where supported."""

    out: list[tuple[EnvelopeCandidate, HalfEdgePolyhedron]] = []
    for candidate in payload_envelope_candidates_from_tier_payload(
        payload,
        wing_level=wing_level,
        min_top_overlap_ratio=min_top_overlap_ratio,
        corner_tol=corner_tol,
    ):
        polyhedron = build_from_planar_polygons(
            [(face.corners, face.plane) for face in candidate.faces],
            coord_tol=coord_tol,
        )
        out.append((candidate, polyhedron))
    return out


def _candidate_builds_strictly(candidate: EnvelopeCandidate) -> bool:
    try:
        build_from_planar_polygons(
            [(face.corners, face.plane) for face in candidate.faces],
            coord_tol=1e-3,
        )
    except ValueError:
        return False
    return True


def _partition_refines(parts: list[Polygon], baseline: list[Polygon]) -> bool:
    if len(parts) <= len(baseline) or not parts or not baseline:
        return False
    baseline_union = unary_union(baseline)
    parts_union = unary_union(parts)
    if parts_union.intersection(baseline_union).area <= 0.0:
        return False
    for part in parts:
        if part.is_empty or part.area <= 0.0:
            continue
        best_overlap = max(
            (float(part.intersection(base).area) for base in baseline),
            default=0.0,
        )
        if best_overlap / max(float(part.area), 1e-9) < 0.70:
            return False
    return True


def _payload_roof_labeled_room_part_polygons(
    payload: dict[str, Any],
    *,
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    corner_tol: float,
    min_room_top_coverage: float = 0.45,
    min_part_area_m2: float = 1.0,
) -> list[Polygon]:
    return [
        part.polygon
        for part in _payload_roof_labeled_room_parts(
            payload,
            footprint=footprint,
            ceiling_faces=ceiling_faces,
            corner_tol=corner_tol,
            min_room_top_coverage=min_room_top_coverage,
            min_part_area_m2=min_part_area_m2,
        )
    ]


def _payload_roof_labeled_room_parts(
    payload: dict[str, Any],
    *,
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    corner_tol: float,
    min_room_top_coverage: float = 0.45,
    min_part_area_m2: float = 1.0,
) -> list[RoomPartitionPart]:
    roof_families = _roof_partition_families(footprint, ceiling_faces)
    cells: list[RoomPartitionCell] = []
    for room_index, room in enumerate(payload.get("rooms", []) or []):
        room_poly = None
        for floor in room.get("floor", []) or []:
            poly = _polygon_xz(_clean_ring(_corners(floor), tol=corner_tol))
            if poly is not None:
                try:
                    clipped = poly.intersection(footprint)
                except Exception:
                    continue
                room_poly = _largest_polygon(clipped)
                break
        if room_poly is None or room_poly.area <= min_part_area_m2:
            continue
        signature = _building_part_signature_for_region(
            room_poly,
            ceiling_faces,
            roof_families=roof_families,
            min_coverage_ratio=min_room_top_coverage,
        )
        if not signature:
            continue
        cells.append(
            RoomPartitionCell(
                room_index=room_index,
                story=_int_or_none(room.get("story")) or 0,
                polygon=room_poly,
                top_signature=signature,
            )
        )

    if len(cells) < 2:
        return []

    components = _connected_room_partition_components(cells)
    parts: list[RoomPartitionPart] = []
    for component in components:
        try:
            unioned = _largest_polygon(
                unary_union([cells[index].polygon for index in component])
            )
        except Exception:
            continue
        if unioned is not None and unioned.area >= min_part_area_m2:
            parts.append(
                RoomPartitionPart(
                    polygon=unioned,
                    top_signature=cells[component[0]].top_signature,
                )
            )
    return parts


def _ceiling_faces_for_partition_signature(
    ceiling_faces: list[PayloadFace],
    signature: tuple[Any, ...],
) -> list[PayloadFace]:
    if not signature:
        return ceiling_faces
    label = signature[0]
    if label == "roof-gable":
        roof_keys = {key for key in signature[1:] if isinstance(key, tuple)}
    elif label == "roof-plane" and len(signature) > 1:
        roof_keys = {signature[1]} if isinstance(signature[1], tuple) else set()
    else:
        return ceiling_faces
    roof_faces = [
        face
        for face in ceiling_faces
        if _top_plane_group_key(face.plane) in roof_keys
    ]
    return roof_faces or ceiling_faces


def _building_part_signature_for_region(
    region: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    roof_families: list[RoofPartitionFamily],
    min_coverage_ratio: float,
) -> tuple[Any, ...]:
    roof_signature = _roof_family_signature_for_region(region, roof_families)
    if roof_signature:
        return roof_signature
    ceiling_signature = _top_signature_for_region(
        region,
        ceiling_faces,
        min_coverage_ratio=min_coverage_ratio,
    )
    if ceiling_signature:
        return ("ceiling", *ceiling_signature)
    return ()


def _roof_family_signature_for_region(
    region: Polygon,
    roof_families: list[RoofPartitionFamily],
    *,
    min_overlap_ratio: float = 0.35,
) -> tuple[Any, ...]:
    if not roof_families:
        return ()
    region_area = max(float(region.area), 1e-9)
    best: tuple[float, RoofPartitionFamily] | None = None
    for family in roof_families:
        try:
            overlap_ratio = (
                float(region.intersection(family.footprint).area) / region_area
            )
        except Exception:
            continue
        if overlap_ratio < min_overlap_ratio:
            continue
        if best is None or overlap_ratio > best[0]:
            best = (overlap_ratio, family)
    return best[1].signature if best is not None else ()


def _roof_partition_families(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_family_area_m2: float = 4.0,
) -> list[RoofPartitionFamily]:
    try:
        supports = _top_supports_for_footprint(footprint, ceiling_faces)
    except Exception:
        return []
    oblique_supports = [
        support
        for support in supports
        if _top_plane_is_partition_roof(support.face.plane)
    ]
    candidates: list[
        tuple[float, RoofPartitionFamily, set[tuple[float, float, float, float]]]
    ] = []
    for left_index, left in enumerate(oblique_supports):
        left_key = _top_plane_group_key(left.face.plane)
        for right in oblique_supports[left_index + 1 :]:
            right_key = _top_plane_group_key(right.face.plane)
            if left_key == right_key:
                continue
            if not _roof_planes_are_opposing(left.face.plane, right.face.plane):
                continue
            try:
                family_footprint = unary_union([left.footprint, right.footprint])
                family_area = float(family_footprint.area)
            except Exception:
                continue
            if family_area < min_family_area_m2:
                continue
            keys = tuple(sorted((left_key, right_key)))
            family = RoofPartitionFamily(
                signature=("roof-gable", *keys),
                footprint=family_footprint,
                area=family_area,
            )
            candidates.append((family_area, family, set(keys)))

    families: list[RoofPartitionFamily] = []
    used_keys: set[tuple[float, float, float, float]] = set()
    for _area, family, keys in sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        if keys & used_keys:
            continue
        families.append(family)
        used_keys.update(keys)

    for support in oblique_supports:
        key = _top_plane_group_key(support.face.plane)
        if key in used_keys or support.area < min_family_area_m2:
            continue
        families.append(
            RoofPartitionFamily(
                signature=("roof-plane", key),
                footprint=support.footprint,
                area=support.area,
            )
        )
    return sorted(families, key=lambda family: family.area, reverse=True)


def _top_plane_is_partition_roof(plane: Plane) -> bool:
    top_plane = _top_plane_up(plane)
    normal_len = max(
        float(
            (
                top_plane.a * top_plane.a
                + top_plane.b * top_plane.b
                + top_plane.c * top_plane.c
            )
            ** 0.5
        ),
        1e-12,
    )
    horizontal_normal = float(
        (top_plane.a * top_plane.a + top_plane.c * top_plane.c) ** 0.5
    )
    return horizontal_normal / normal_len >= 0.25 and _top_plane_is_oblique(plane)


def _roof_planes_are_opposing(
    left: Plane,
    right: Plane,
    *,
    max_dot: float = -0.60,
) -> bool:
    left_plane = _top_plane_up(left)
    right_plane = _top_plane_up(right)
    left_vec = np.asarray([left_plane.a, left_plane.c], dtype=float)
    right_vec = np.asarray([right_plane.a, right_plane.c], dtype=float)
    left_len = float(np.linalg.norm(left_vec))
    right_len = float(np.linalg.norm(right_vec))
    if left_len <= 1e-9 or right_len <= 1e-9:
        return False
    dot = float(np.dot(left_vec / left_len, right_vec / right_len))
    return (
        dot <= max_dot
        and _plane_equal_height_line_xz(left_plane, right_plane) is not None
    )


def _top_signature_for_region(
    region: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_coverage_ratio: float,
    max_planes: int = 3,
) -> tuple[tuple[float, float, float, float], ...]:
    try:
        supports = _top_supports_for_footprint(region, ceiling_faces)
    except Exception:
        return ()
    if not supports:
        return ()
    region_area = max(float(region.area), 1e-9)
    selected: list[tuple[float, tuple[float, float, float, float]]] = []
    for support in supports:
        try:
            overlap = float(region.intersection(support.footprint).area)
        except Exception:
            continue
        overlap_ratio = overlap / region_area
        if overlap_ratio < min_coverage_ratio:
            continue
        selected.append((overlap_ratio, _top_plane_group_key(support.face.plane)))
    if not selected:
        return ()
    selected.sort(reverse=True)
    keys = sorted({key for _ratio, key in selected[:max_planes]})
    return tuple(keys)


def _connected_room_partition_components(
    cells: list[RoomPartitionCell],
    *,
    adjacency_tol_m: float = 0.08,
    min_shared_boundary_m: float = 0.20,
) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(cells))}
    for i, left in enumerate(cells):
        for j in range(i + 1, len(cells)):
            right = cells[j]
            if left.story != right.story or left.top_signature != right.top_signature:
                continue
            try:
                shared = float(
                    left.polygon.boundary.intersection(right.polygon.boundary).length
                )
                distance = float(left.polygon.distance(right.polygon))
            except Exception:
                continue
            if shared < min_shared_boundary_m and distance > adjacency_tol_m:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(cells)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        component: list[int] = []
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                queue.append(neighbour)
        components.append(component)
    return components


def _envelope_candidates_for_parts(
    payload: dict[str, Any],
    parts: list[Polygon],
    ceiling_faces: list[PayloadFace],
    *,
    part_ceiling_faces: list[list[PayloadFace]] | None = None,
    min_top_overlap_ratio: float,
    corner_tol: float,
    locator_prefix: str,
    strict_gate_all: bool,
) -> list[EnvelopeCandidate]:
    candidates: list[EnvelopeCandidate] = []
    for part_index, part in enumerate(parts):
        active_ceiling_faces = (
            part_ceiling_faces[part_index]
            if part_ceiling_faces is not None and part_index < len(part_ceiling_faces)
            else ceiling_faces
        )
        locator_id = f"{locator_prefix}:{part_index}"
        for candidate_part, is_aligned_part in _ridge_aligned_part_variants(
            payload,
            part,
            active_ceiling_faces,
            corner_tol=corner_tol,
        ):
            try:
                candidate = _envelope_candidate_for_part(
                    payload,
                    candidate_part,
                    active_ceiling_faces,
                    min_top_overlap_ratio=min_top_overlap_ratio,
                    corner_tol=corner_tol,
                    locator_id=locator_id,
                    strict_gate_all=strict_gate_all or is_aligned_part,
                )
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
                break
    return candidates


def _envelope_candidate_for_part(
    payload: dict[str, Any],
    part: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_top_overlap_ratio: float,
    corner_tol: float,
    locator_id: str,
    strict_gate_all: bool,
    floor_y_override: float | None = None,
) -> EnvelopeCandidate | None:
    ridge_partition = _ridge_aware_top_partition(
        part,
        ceiling_faces,
        min_coverage_ratio=min_top_overlap_ratio,
    )
    if ridge_partition is not None:
        top_regions, coverage_ratio = ridge_partition
        floor_y = (
            floor_y_override
            if floor_y_override is not None
            else _floor_y_for_part(payload, part, corner_tol=corner_tol)
        )
        if floor_y is not None and _top_regions_are_above_floor(
            top_regions,
            floor_y=floor_y,
        ):
            candidate = _envelope_candidate_from_top_regions(
                top_regions,
                floor_y=floor_y,
                locator_id=locator_id,
                coverage_ratio=coverage_ratio,
                footprint_override=part,
            )
            if _candidate_builds_strictly(candidate):
                return candidate
            atomic_candidate = _envelope_candidate_from_top_regions(
                top_regions,
                floor_y=floor_y,
                locator_id=locator_id,
                coverage_ratio=coverage_ratio,
                footprint_override=part,
                atomic_external_walls=True,
            )
            if _candidate_builds_strictly(atomic_candidate):
                return atomic_candidate

    selected = _dominant_top_face(
        part,
        ceiling_faces,
        min_overlap_ratio=min_top_overlap_ratio,
    )
    if selected is not None:
        top_face, overlap_ratio = selected
        floor_y = (
            floor_y_override
            if floor_y_override is not None
            else _floor_y_for_part(payload, part, corner_tol=corner_tol)
        )
        if floor_y is not None:
            top_plane = _top_plane_up(top_face.plane)
            if _top_plane_is_above_floor(part, top_plane, floor_y=floor_y):
                candidate = _envelope_candidate_from_top_plane(
                    part,
                    top_face,
                    floor_y=floor_y,
                    locator_id=locator_id,
                    overlap_ratio=overlap_ratio,
                )
                if not strict_gate_all or _candidate_builds_strictly(candidate):
                    return candidate

    selected_group = _dominant_top_face_group(
        part,
        ceiling_faces,
        min_overlap_ratio=min_top_overlap_ratio,
    )
    if selected_group is not None:
        top_face, overlap_ratio, top_source = selected_group
        floor_y = (
            floor_y_override
            if floor_y_override is not None
            else _floor_y_for_part(payload, part, corner_tol=corner_tol)
        )
        if floor_y is not None:
            top_plane = _top_plane_up(top_face.plane)
            if _top_plane_is_above_floor(part, top_plane, floor_y=floor_y):
                candidate = _envelope_candidate_from_top_plane(
                    part,
                    top_face,
                    floor_y=floor_y,
                    locator_id=locator_id,
                    overlap_ratio=overlap_ratio,
                    top_source=top_source,
                )
                if not strict_gate_all or _candidate_builds_strictly(candidate):
                    return candidate

    require_strict_candidate = False
    partition = _two_plane_top_partition(
        part,
        ceiling_faces,
        min_coverage_ratio=min_top_overlap_ratio,
    )
    if partition is None:
        partition = _multi_piece_top_partition(
            part,
            ceiling_faces,
            min_coverage_ratio=min_top_overlap_ratio,
        )
        require_strict_candidate = True
    if partition is None:
        return None
    top_regions, coverage_ratio = partition
    floor_y = (
        floor_y_override
        if floor_y_override is not None
        else _floor_y_for_part(payload, part, corner_tol=corner_tol)
    )
    if floor_y is None:
        return None
    if not _top_regions_are_above_floor(top_regions, floor_y=floor_y):
        return None
    candidate = _envelope_candidate_from_top_regions(
        top_regions,
        floor_y=floor_y,
        locator_id=locator_id,
        coverage_ratio=coverage_ratio,
        footprint_override=part,
    )
    needs_strict_gate = require_strict_candidate or strict_gate_all
    if needs_strict_gate and not _candidate_builds_strictly(candidate):
        atomic_candidate = _envelope_candidate_from_top_regions(
            top_regions,
            floor_y=floor_y,
            locator_id=locator_id,
            coverage_ratio=coverage_ratio,
            footprint_override=part,
            atomic_external_walls=True,
        )
        if _candidate_builds_strictly(atomic_candidate):
            return atomic_candidate
        return None
    return candidate


RIDGE_ALIGN_MIN_STORIES = 2
RIDGE_ALIGN_MIN_AXIS_COVERAGE = 0.70
RIDGE_ALIGN_MAX_YAW_DEG = 8.0
RIDGE_ALIGN_MIN_YAW_DEG = 0.25
RIDGE_ALIGN_MAX_PERP_SHIFT_RIDGE_RATIO = 0.35
RIDGE_ALIGN_MIN_PERP_SHIFT_M = 0.05
RIDGE_ALIGN_MIN_RIDGE_LENGTH_M = 2.0
RIDGE_ALIGN_MIN_AREA_RATIO = 0.80
RIDGE_ALIGN_MAX_AREA_RATIO = 1.20


def _ridge_aligned_part_variants(
    payload: dict[str, Any],
    part: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    corner_tol: float,
) -> list[tuple[Polygon, bool]]:
    aligned = _ridge_aligned_part_footprint(
        payload,
        part,
        ceiling_faces,
        corner_tol=corner_tol,
    )
    if aligned is None:
        return [(part, False)]
    return [(aligned, True), (part, False)]


def _ridge_aligned_part_footprint(
    payload: dict[str, Any],
    part: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    corner_tol: float,
) -> Polygon | None:
    """Return a per-building-part footprint normalized to its local ridge.

    The transform is intentionally weak: yaw is corrected only when each story's
    own floor axis is already close to the ridge axis, and XZ translation only
    moves the story footprint perpendicular to the ridge. Translation along the
    ridge remains untouched because a single ridge line cannot observe it.
    """

    result = _ridge_story_transforms_for_part(
        payload,
        part,
        ceiling_faces,
        corner_tol=corner_tol,
    )
    if result is None:
        return None
    _frame, story_polys, transforms = result
    aligned_polys = [
        _apply_ridge_transform_to_polygon(poly, transforms[story])
        for story, poly in story_polys
        if story in transforms
    ]
    return _valid_aligned_part_union(part, aligned_polys)


def _ridge_aligned_payload_for_part(
    payload: dict[str, Any],
    part: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    corner_tol: float,
) -> tuple[dict[str, Any], Polygon] | None:
    result = _ridge_story_transforms_for_part(
        payload,
        part,
        ceiling_faces,
        corner_tol=corner_tol,
    )
    if result is None:
        return None
    _frame, story_polys, transforms = result
    aligned_polys = [
        _apply_ridge_transform_to_polygon(poly, transforms[story])
        for story, poly in story_polys
        if story in transforms
    ]
    aligned_part = _valid_aligned_part_union(part, aligned_polys)
    if aligned_part is None:
        return None

    aligned_payload = deepcopy(payload)
    for room_index, room in enumerate(aligned_payload.get("rooms", []) or []):
        original_room = (payload.get("rooms", []) or [])[room_index]
        story = _int_or_none(original_room.get("story")) or 0
        transform = transforms.get(story)
        if transform is None:
            continue
        if not _room_intersects_part(original_room, part, corner_tol=corner_tol):
            continue
        _transform_room_geometry_xz(room, transform)
    return aligned_payload, aligned_part


def _ridge_story_transforms_for_part(
    payload: dict[str, Any],
    part: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    corner_tol: float,
) -> (
    tuple[RidgeFrame, list[tuple[int, Polygon]], dict[int, RidgeStoryTransform]]
    | None
):
    if part.is_empty or part.area <= 0.0:
        return None
    frame = _ridge_frame_for_part(part, ceiling_faces)
    if frame is None:
        return None

    story_polys = _story_floor_polygons_for_part(
        payload,
        part,
        corner_tol=corner_tol,
    )
    if len(story_polys) < RIDGE_ALIGN_MIN_STORIES:
        return None

    prepared: list[tuple[int, Polygon, float, float, float, Point]] = []
    for story, poly in story_polys:
        axis_info = _polygon_axis_and_coverage(poly)
        if axis_info is None:
            return None
        axis_deg, coverage = axis_info
        if coverage < RIDGE_ALIGN_MIN_AXIS_COVERAGE:
            return None
        yaw_delta = _signed_delta_mod90(axis_deg, frame.axis_deg)
        if abs(yaw_delta) > RIDGE_ALIGN_MAX_YAW_DEG:
            return None
        prepared.append((story, poly, axis_deg, coverage, yaw_delta, poly.centroid))

    rotated: list[tuple[int, Polygon, float, float]] = []
    for story, poly, _axis_deg, _coverage, yaw_delta, centroid in prepared:
        candidate = poly
        if abs(yaw_delta) >= RIDGE_ALIGN_MIN_YAW_DEG:
            candidate = affinity.rotate(
                poly,
                yaw_delta,
                origin=(float(centroid.x), float(centroid.y)),
                use_radians=False,
            )
        distance = _signed_distance_to_ridge_frame(frame, candidate.centroid)
        rotated.append((story, candidate, float(candidate.area), distance))

    target_distance = _weighted_median(
        [(distance, area) for _story, _poly, area, distance in rotated]
    )
    transforms: dict[int, RidgeStoryTransform] = {}
    changed = False
    max_perp_shift = _max_ridge_perp_shift_for_frame(frame)
    centroid_by_story = {story: centroid for story, *_rest, centroid in prepared}
    yaw_by_story = {
        story: yaw_delta
        for story, *_rest, yaw_delta, _centroid in prepared
    }
    for story, _poly, _area, distance in rotated:
        shift = target_distance - distance
        if abs(shift) > max_perp_shift:
            return None
        yaw_delta = yaw_by_story[story]
        if (
            abs(shift) >= RIDGE_ALIGN_MIN_PERP_SHIFT_M
            or abs(yaw_delta) >= RIDGE_ALIGN_MIN_YAW_DEG
        ):
            changed = True
        centroid = centroid_by_story[story]
        transforms[story] = RidgeStoryTransform(
            story=story,
            origin_x=float(centroid.x),
            origin_z=float(centroid.y),
            yaw_delta_deg=float(yaw_delta),
            shift_x=float(frame.line_a * shift),
            shift_z=float(frame.line_b * shift),
        )

    if not changed:
        return None
    return frame, story_polys, transforms


def _max_ridge_perp_shift_for_frame(frame: RidgeFrame) -> float:
    return float(frame.support_length_m * RIDGE_ALIGN_MAX_PERP_SHIFT_RIDGE_RATIO)


def _valid_aligned_part_union(
    original_part: Polygon,
    aligned_polys: list[Polygon],
) -> Polygon | None:
    if not aligned_polys:
        return None

    try:
        unioned = _largest_polygon(unary_union(aligned_polys).buffer(0))
    except Exception:
        return None
    if unioned is None or unioned.area <= 0.0:
        return None
    area_ratio = float(unioned.area) / max(float(original_part.area), 1e-9)
    if not (RIDGE_ALIGN_MIN_AREA_RATIO <= area_ratio <= RIDGE_ALIGN_MAX_AREA_RATIO):
        return None
    return unioned


def _room_intersects_part(
    room: dict[str, Any],
    part: Polygon,
    *,
    corner_tol: float,
    min_overlap_area_m2: float = 0.05,
) -> bool:
    for floor in room.get("floor", []) or []:
        poly = _polygon_xz(_clean_ring(_corners(floor), tol=corner_tol))
        if poly is None:
            continue
        try:
            if float(poly.intersection(part).area) >= min_overlap_area_m2:
                return True
        except Exception:
            continue
    return False


def _transform_room_geometry_xz(
    room: dict[str, Any],
    transform: RidgeStoryTransform,
) -> None:
    for key in ("floor", "walls", "doors", "windows"):
        for piece in room.get(key, []) or []:
            _transform_piece_geometry_xz(piece, transform)


def _transform_piece_geometry_xz(
    piece: dict[str, Any],
    transform: RidgeStoryTransform,
) -> None:
    if isinstance(piece.get("corners"), list):
        piece["corners"] = [
            _point3_to_payload_dict(
                _apply_ridge_transform_to_point(_point3(c), transform)
            )
            for c in piece["corners"]
        ]
    holes = piece.get("holes")
    if isinstance(holes, list):
        piece["holes"] = [_transform_hole_xz(hole, transform) for hole in holes]


def _transform_hole_xz(hole: Any, transform: RidgeStoryTransform) -> Any:
    if isinstance(hole, dict):
        out = dict(hole)
        if isinstance(out.get("corners"), list):
            out["corners"] = [
                _point3_to_payload_dict(
                    _apply_ridge_transform_to_point(_point3(c), transform)
                )
                for c in out["corners"]
            ]
        return out
    if isinstance(hole, list):
        return [
            _point3_to_payload_dict(
                _apply_ridge_transform_to_point(_point3(c), transform)
            )
            for c in hole
        ]
    return hole


def _apply_ridge_transform_to_polygon(
    poly: Polygon,
    transform: RidgeStoryTransform,
) -> Polygon:
    candidate = poly
    if abs(transform.yaw_delta_deg) >= RIDGE_ALIGN_MIN_YAW_DEG:
        candidate = affinity.rotate(
            candidate,
            transform.yaw_delta_deg,
            origin=(transform.origin_x, transform.origin_z),
            use_radians=False,
        )
    if (
        abs(transform.shift_x) >= 1e-9
        or abs(transform.shift_z) >= 1e-9
    ):
        candidate = affinity.translate(
            candidate,
            xoff=transform.shift_x,
            yoff=transform.shift_z,
        )
    return candidate


def _apply_ridge_transform_to_point(
    point: list[float],
    transform: RidgeStoryTransform,
) -> list[float]:
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    if abs(transform.yaw_delta_deg) >= RIDGE_ALIGN_MIN_YAW_DEG:
        theta = np.radians(transform.yaw_delta_deg)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        dx = x - transform.origin_x
        dz = z - transform.origin_z
        x = transform.origin_x + dx * cos_t - dz * sin_t
        z = transform.origin_z + dx * sin_t + dz * cos_t
    return [x + transform.shift_x, y, z + transform.shift_z]


def _point3_to_payload_dict(point: list[float]) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def _ridge_frame_for_part(
    part: Polygon,
    ceiling_faces: list[PayloadFace],
) -> RidgeFrame | None:
    supports = [
        support
        for support in _top_supports_for_footprint(part, ceiling_faces)
        if _top_plane_is_oblique(support.face.plane)
    ]
    if len(supports) < 2:
        return None

    best: tuple[float, float, float, float, float] | None = None
    for left_index, left in enumerate(supports):
        for right in supports[left_index + 1 :]:
            if not _planes_have_opposing_slope(left.face.plane, right.face.plane):
                continue
            coeffs = _plane_equal_height_line_xz(
                _top_plane_up(left.face.plane),
                _top_plane_up(right.face.plane),
            )
            if coeffs is None:
                continue
            line = _ridge_split_line_for_planes(
                part,
                _top_plane_up(left.face.plane),
                _top_plane_up(right.face.plane),
            )
            if line is None:
                continue
            length = float(line.intersection(part).length)
            if length < RIDGE_ALIGN_MIN_RIDGE_LENGTH_M:
                continue
            a, b, c = coeffs
            norm = hypot(a, b)
            if norm <= 1e-12:
                continue
            a /= norm
            b /= norm
            c /= norm
            if a < -1e-9 or (abs(a) <= 1e-9 and b < -1e-9):
                a = -a
                b = -b
                c = -c
            axis_deg = degrees(atan2(-a, b)) % 90.0
            candidate = (length, axis_deg, a, b, c)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None
    length, axis_deg, a, b, c = best
    return RidgeFrame(
        axis_deg=float(axis_deg),
        line_a=float(a),
        line_b=float(b),
        line_c=float(c),
        support_length_m=float(length),
    )


def _planes_have_opposing_slope(left: Plane, right: Plane) -> bool:
    left = _top_plane_up(left)
    right = _top_plane_up(right)
    if abs(left.b) <= 1e-9 or abs(right.b) <= 1e-9:
        return False
    left_grad = np.asarray([-left.a / left.b, -left.c / left.b], dtype=float)
    right_grad = np.asarray([-right.a / right.b, -right.c / right.b], dtype=float)
    left_norm = float(np.linalg.norm(left_grad))
    right_norm = float(np.linalg.norm(right_grad))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return False
    cos = float(np.dot(left_grad, right_grad) / (left_norm * right_norm))
    return cos <= -0.35


def _story_floor_polygons_for_part(
    payload: dict[str, Any],
    part: Polygon,
    *,
    corner_tol: float,
) -> list[tuple[int, Polygon]]:
    by_story: dict[int, list[Polygon]] = {}
    for room in payload.get("rooms", []) or []:
        story = _int_or_none(room.get("story")) or 0
        for floor in room.get("floor", []) or []:
            poly = _polygon_xz(_clean_ring(_corners(floor), tol=corner_tol))
            if poly is None:
                continue
            try:
                clipped = _largest_polygon(poly.intersection(part))
            except Exception:
                continue
            if clipped is not None and clipped.area > 0.05:
                by_story.setdefault(story, []).append(clipped)

    out: list[tuple[int, Polygon]] = []
    for story, polys in sorted(by_story.items()):
        try:
            merged = _largest_polygon(unary_union(polys).buffer(0))
        except Exception:
            continue
        if merged is not None and merged.area > 0.50:
            out.append((story, merged))
    return out


def _polygon_axis_and_coverage(poly: Polygon) -> tuple[float, float] | None:
    if poly.is_empty:
        return None
    bins: dict[int, float] = {}
    samples: list[tuple[float, float]] = []
    for part in _polygon_parts(poly):
        coords = list(part.exterior.coords)
        for start, end in pairwise(coords):
            dx = float(end[0] - start[0])
            dz = float(end[1] - start[1])
            length = hypot(dx, dz)
            if length < 0.30:
                continue
            axis = degrees(atan2(dz, dx)) % 90.0
            samples.append((axis, length))
            bin_key = round(axis / 2.0) % 45
            for offset, weight in ((0, 1.0), (1, 0.5), (-1, 0.5)):
                key = (bin_key + offset) % 45
                bins[key] = bins.get(key, 0.0) + weight * length
    if not bins:
        return None
    axis = float(max(bins.items(), key=lambda item: item[1])[0]) * 2.0
    total = sum(length for _axis, length in samples)
    aligned = sum(
        length
        for sample_axis, length in samples
        if abs(_signed_delta_mod90(sample_axis, axis)) <= 12.0
    )
    coverage = aligned / total if total > 0.0 else 0.0
    return axis, coverage


def _signed_delta_mod90(source_deg: float, target_deg: float) -> float:
    return float(((target_deg - source_deg + 45.0) % 90.0) - 45.0)


def _signed_distance_to_ridge_frame(frame: RidgeFrame, point: Point) -> float:
    return float(frame.line_a * point.x + frame.line_b * point.y + frame.line_c)


def _weighted_median(samples: list[tuple[float, float]]) -> float:
    total = sum(max(0.0, weight) for _value, weight in samples)
    if total <= 0.0:
        return 0.0
    midpoint = total * 0.5
    running = 0.0
    ordered = sorted(samples)
    for index, (value, weight) in enumerate(ordered):
        running += max(0.0, weight)
        if abs(running - midpoint) <= 1e-12 and index + 1 < len(ordered):
            return float((value + ordered[index + 1][0]) * 0.5)
        if running >= midpoint:
            return float(value)
    return float(samples[-1][0])


def _payload_room_graph_wing_polygons(
    payload: dict[str, Any],
    *,
    footprint: Polygon,
    corner_tol: float,
) -> list[Polygon]:
    rooms = _payload_rooms_for_wing_detection(payload, corner_tol=corner_tol)
    if not rooms:
        return []
    try:
        from reconcile_tiers._core.room_graph import build_room_graph

        room_graph = build_room_graph(rooms)
        wings = decompose_to_wings_v2(
            footprint,
            room_graph=room_graph,
            rooms=rooms,
        )
    except Exception:
        return []
    return [wing.polygon for wing in wings if wing.polygon.area > 0.0]


def _parts_not_covered_by_candidates(
    parts: list[Polygon],
    candidates: list[EnvelopeCandidate],
    *,
    max_covered_ratio: float = 0.80,
    min_uncovered_area_m2: float = 0.5,
) -> list[Polygon]:
    if not parts:
        return []
    covered_polys = [
        poly
        for candidate in candidates
        if (poly := _candidate_footprint_polygon(candidate)) is not None
    ]
    if not covered_polys:
        return parts
    covered = unary_union(covered_polys)
    out: list[Polygon] = []
    for part in parts:
        if part.is_empty or part.area <= 0.0:
            continue
        covered_area = float(part.intersection(covered).area)
        covered_ratio = covered_area / max(float(part.area), 1e-9)
        uncovered_area = max(0.0, float(part.area) - covered_area)
        if (
            covered_ratio <= max_covered_ratio
            and uncovered_area >= min_uncovered_area_m2
        ):
            out.append(part)
    return out


def _candidate_footprint_polygon(candidate: EnvelopeCandidate) -> Polygon | None:
    floor = next((face for face in candidate.faces if face.kind == "floor"), None)
    if floor is None:
        return None
    return _polygon_xz(floor.corners)


def _payload_rooms_for_wing_detection(
    payload: dict[str, Any],
    *,
    corner_tol: float,
) -> list[Any]:
    from reconcile_tiers.extract.building import (
        ExtractedElement,
        ExtractedRoom,
        ExtractedWall,
    )

    rooms: list[Any] = []
    for room_index, room in enumerate(payload.get("rooms", []) or []):
        floor_corners: list[list[float]] = []
        for floor in room.get("floor", []) or []:
            corners = _clean_ring(_corners(floor), tol=corner_tol)
            if len(corners) >= 3:
                floor_corners = corners
                break
        if len(floor_corners) < 3:
            continue

        walls: list[Any] = []
        for wall_index, wall in enumerate(room.get("walls", []) or []):
            corners = _clean_ring(_corners(wall), tol=corner_tol)
            if len(corners) < 3:
                continue
            walls.append(
                ExtractedWall(
                    id=str(
                        wall.get("locator_id")
                        or f"room:{room_index}:wall:{wall_index}"
                    ),
                    corners=corners,
                    source="tier_payload",
                )
            )

        doors: list[Any] = []
        for door_index, door in enumerate(room.get("doors", []) or []):
            corners = _clean_ring(_corners(door), tol=corner_tol)
            if len(corners) < 3:
                continue
            doors.append(
                ExtractedElement(
                    id=str(
                        door.get("locator_id")
                        or f"room:{room_index}:door:{door_index}"
                    ),
                    corners=corners,
                    source="tier_payload",
                    parent_wall_id=door.get("parent_wall_id"),
                )
            )

        rooms.append(
            ExtractedRoom(
                index=room_index,
                story=_int_or_none(room.get("story")) or 0,
                floor_polygon=floor_corners,
                walls_merged=walls,
                walls_computed=walls,
                doors=doors,
                windows=[],
                openings=[],
                storages=[],
                raw_ceiling_planes=[],
                raw_ceiling_source=None,
                ceiling_polygon=[],
                ceiling_type=None,
                ceiling_eave_height=None,
                ceiling_ridge_height=None,
            )
        )
    return rooms


def build_polyhedron_from_tier_payload(
    payload: dict[str, Any],
    *,
    coord_tol: float = 1e-3,
    corner_tol: float = 0.02,
    include: Iterable[FaceKind] = ("floor", "wall", "ceiling"),
) -> HalfEdgePolyhedron:
    """Strictly build a half-edge polyhedron from payload faces.

    Raises ``ValueError`` when the selected payload faces are not one closed
    2-manifold. That is expected for many full payloads and is useful diagnostic
    signal for Phase 3: the caller should narrow to a room/wing shell or add an
    explicit topology-repair step before optimization.
    """

    faces = payload_faces_from_tier_payload(
        payload,
        include=include,
        corner_tol=corner_tol,
    )
    return build_from_planar_polygons(
        [(face.corners, face.plane) for face in faces],
        coord_tol=coord_tol,
    )


def payload_faces_for_room_shell(
    payload: dict[str, Any],
    room_index: int,
    *,
    ceiling_limit: int = 1,
    min_ceiling_overlap_area_m2: float = 0.01,
    corner_tol: float = 0.02,
) -> list[PayloadFace]:
    """Select the floor, walls, and best ceiling pieces for one room shell.

    This is Phase 3b's narrow optimization domain. Full payloads often contain
    internal/duplicate room surfaces, but a single simple room can already be a
    closed manifold. Ceiling pieces are ranked by XZ overlap with the room's
    floor polygon because flat ceilings do not always encode a room id in their
    locator.
    """

    faces = payload_faces_from_tier_payload(payload, corner_tol=corner_tol)
    room_faces = [
        face
        for face in faces
        if face.room_index == room_index and face.kind in ("floor", "wall")
    ]
    floor = next((face for face in room_faces if face.kind == "floor"), None)
    if floor is None:
        raise ValueError(f"room {room_index} has no floor face")

    floor_poly = _polygon_xz(floor.corners)
    if floor_poly is None:
        raise ValueError(f"room {room_index} has invalid floor XZ polygon")

    candidates: list[tuple[float, PayloadFace]] = []
    for face in faces:
        if face.kind != "ceiling":
            continue
        ceiling_poly = _polygon_xz(face.corners)
        if ceiling_poly is None:
            continue
        overlap = float(floor_poly.intersection(ceiling_poly).area)
        if face.room_index == room_index:
            # Direct locator evidence should beat equal-area incidental overlap.
            overlap += 1e-6
        if overlap >= min_ceiling_overlap_area_m2:
            candidates.append((overlap, face))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_ceilings = [face for _overlap, face in candidates[:ceiling_limit]]
    if not selected_ceilings:
        raise ValueError(f"room {room_index} has no overlapping ceiling face")
    return [*room_faces, *selected_ceilings]


def build_room_shell_from_tier_payload(
    payload: dict[str, Any],
    room_index: int,
    *,
    coord_tol: float = 1e-3,
    corner_tol: float = 0.02,
    ceiling_limit: int = 1,
    min_ceiling_overlap_area_m2: float = 0.01,
) -> HalfEdgePolyhedron:
    """Strictly build one room shell from a payload.

    This intentionally uses the same watertight constructor as the whole-payload
    path. If a room's walls/ceiling do not close, the caller gets the manifold
    violation that must be handled before coordinate descent.
    """

    faces = payload_faces_for_room_shell(
        payload,
        room_index,
        ceiling_limit=ceiling_limit,
        min_ceiling_overlap_area_m2=min_ceiling_overlap_area_m2,
        corner_tol=corner_tol,
    )
    return build_from_planar_polygons(
        [(face.corners, face.plane) for face in faces],
        coord_tol=coord_tol,
    )


def _payload_footprint_polygon(
    payload: dict[str, Any],
    *,
    room_buffer_m: float,
    footprint_shrink_m: float,
    corner_tol: float,
) -> Polygon | None:
    polygons: list[Polygon] = []
    points: list[tuple[float, float]] = []
    for room in payload.get("rooms", []) or []:
        for floor in room.get("floor", []) or []:
            corners = _clean_ring(_corners(floor), tol=corner_tol)
            poly = _polygon_xz(corners)
            if poly is None:
                continue
            polygons.append(poly.buffer(room_buffer_m, join_style="mitre"))
            points.extend((float(c[0]), float(c[2])) for c in corners)
    if not polygons:
        return None
    merged = _largest_polygon(unary_union(polygons))
    if merged is None:
        return None
    shrunk = _largest_polygon(merged.buffer(-footprint_shrink_m, join_style="mitre"))
    if shrunk is None or shrunk.area <= 0.0:
        if len(points) < 3:
            return None
        shrunk = _largest_polygon(Polygon(points).convex_hull)
    if shrunk is None or shrunk.area <= 0.0:
        return None
    return shrunk


def _largest_polygon(geometry: Any) -> Polygon | None:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        poly = make_valid_polygon(geometry)
        return poly if poly is not None and not poly.is_empty else None
    if geometry.geom_type == "MultiPolygon":
        return max(
            (poly for poly in geometry.geoms if not poly.is_empty),
            key=lambda poly: poly.area,
            default=None,
        )
    return None


def _dominant_top_face(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_overlap_ratio: float,
) -> tuple[PayloadFace, float] | None:
    footprint_area = max(float(footprint.area), 1e-9)
    best: tuple[PayloadFace, float] | None = None
    for face in ceiling_faces:
        if abs(face.plane.b) <= 1e-6:
            continue
        poly = _polygon_xz(face.corners)
        if poly is None:
            continue
        overlap_ratio = float(footprint.intersection(poly).area) / footprint_area
        if overlap_ratio < min_overlap_ratio:
            continue
        if best is None or overlap_ratio > best[1]:
            best = (face, overlap_ratio)
    return best


def _dominant_top_face_group(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_overlap_ratio: float,
) -> tuple[PayloadFace, float, str] | None:
    footprint_area = max(float(footprint.area), 1e-9)
    grouped: dict[
        tuple[float, float, float, float],
        list[tuple[PayloadFace, Any]],
    ] = {}
    for face in ceiling_faces:
        if abs(face.plane.b) <= 1e-6:
            continue
        poly = _polygon_xz(face.corners)
        if poly is None:
            continue
        try:
            clipped = poly.intersection(footprint)
        except Exception:
            continue
        if clipped.is_empty or clipped.area <= 0.05:
            continue
        grouped.setdefault(_top_plane_group_key(face.plane), []).append((face, clipped))

    best: tuple[PayloadFace, float, str] | None = None
    for group in grouped.values():
        if len(group) < 2:
            continue
        try:
            unioned = unary_union([clipped for _face, clipped in group]).intersection(
                footprint
            )
        except Exception:
            continue
        overlap_ratio = float(unioned.area) / footprint_area
        if overlap_ratio < min_overlap_ratio:
            continue
        group = sorted(
            group,
            key=lambda item: float(item[1].area),
            reverse=True,
        )
        representative = group[0][0]
        source = " + ".join(face.locator_id for face, _clipped in group)
        if best is None or overlap_ratio > best[1]:
            best = (representative, overlap_ratio, source)
    return best


def _top_plane_group_key(plane: Plane) -> tuple[float, float, float, float]:
    top_plane = _top_plane_up(plane)
    return (
        round(top_plane.a, 2),
        round(top_plane.b, 2),
        round(top_plane.c, 2),
        round(top_plane.d, 1),
    )


def _two_plane_top_partition(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_coverage_ratio: float,
) -> tuple[list[tuple[Polygon, PayloadFace]], float] | None:
    supports: list[tuple[PayloadFace, Polygon]] = []
    for face in ceiling_faces:
        if abs(face.plane.b) <= 1e-6:
            continue
        poly = _polygon_xz(face.corners)
        if poly is None:
            continue
        clipped = _largest_polygon(poly.intersection(footprint))
        if clipped is None or clipped.area <= 0.05:
            continue
        supports.append((face, clipped))

    best: tuple[list[tuple[Polygon, PayloadFace]], float] | None = None
    footprint_area = max(float(footprint.area), 1e-9)
    for left_index, (left_face, left_support) in enumerate(supports):
        for right_face, right_support in supports[left_index + 1 :]:
            split_regions = _split_footprint_by_plane_intersection(
                footprint,
                _top_plane_up(left_face.plane),
                _top_plane_up(right_face.plane),
            )
            if split_regions is None:
                continue
            union_support = _largest_polygon(
                unary_union([left_support, right_support]).intersection(footprint)
            )
            if union_support is None:
                continue
            coverage_ratio = float(union_support.area) / footprint_area
            if coverage_ratio < min_coverage_ratio:
                continue
            assigned = _assign_split_regions_to_faces(
                split_regions,
                (left_face, left_support),
                (right_face, right_support),
            )
            if assigned is None:
                continue
            if best is None or coverage_ratio > best[1]:
                best = (assigned, coverage_ratio)
    return best


def _ridge_aware_top_partition(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_coverage_ratio: float,
    min_cell_area_m2: float = 0.05,
    max_supports: int = 8,
) -> tuple[list[tuple[Polygon, PayloadFace]], float] | None:
    supports = _top_supports_for_footprint(footprint, ceiling_faces)
    if len(supports) < 2:
        return None
    if not any(_top_plane_is_oblique(support.face.plane) for support in supports):
        return None
    if len(supports) > max_supports:
        supports = sorted(
            supports,
            key=lambda support: (
                _top_plane_is_oblique(support.face.plane),
                support.area,
            ),
            reverse=True,
        )[:max_supports]

    regions = [footprint]
    split_count = 0
    for left_index, left in enumerate(supports):
        for right in supports[left_index + 1 :]:
            line = _ridge_split_line_for_planes(
                footprint,
                _top_plane_up(left.face.plane),
                _top_plane_up(right.face.plane),
            )
            if line is None:
                continue
            next_regions: list[Polygon] = []
            for region in regions:
                try:
                    split_parts = [
                        part
                        for part in split(region, line).geoms
                        if isinstance(part, Polygon) and part.area > min_cell_area_m2
                    ]
                except Exception:
                    split_parts = [region]
                if len(split_parts) > 1:
                    split_count += 1
                    next_regions.extend(split_parts)
                else:
                    next_regions.append(region)
            regions = next_regions

    if split_count == 0:
        return None

    assigned: list[tuple[Polygon, PayloadFace]] = []
    for region in regions:
        if region.area <= min_cell_area_m2:
            continue
        support = _highest_supported_plane_for_region(region, supports)
        if support is None:
            continue
        assigned.append((region, support.face))

    if not assigned:
        return None
    top_regions = _merge_regions_by_top_plane(assigned)
    if len({_top_plane_group_key(face.plane) for _region, face in top_regions}) < 2:
        return None

    covered = unary_union([region for region, _face in top_regions]).intersection(
        footprint
    )
    coverage_ratio = float(covered.area) / max(float(footprint.area), 1e-9)
    if coverage_ratio < min_coverage_ratio:
        return None
    return top_regions, coverage_ratio


def _top_plane_is_oblique(plane: Plane) -> bool:
    top_plane = _top_plane_up(plane)
    normal_len = max(
        float(
            (
                top_plane.a * top_plane.a
                + top_plane.b * top_plane.b
                + top_plane.c * top_plane.c
            )
            ** 0.5
        ),
        1e-12,
    )
    vertical_normal = min(1.0, max(0.0, abs(top_plane.b) / normal_len))
    inclination = float(np.degrees(np.arccos(vertical_normal)))
    return 5.0 < inclination < 80.0


def _top_supports_for_footprint(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
) -> list[TopSupport]:
    grouped: dict[
        tuple[float, float, float, float],
        list[tuple[PayloadFace, Any]],
    ] = {}
    for face in ceiling_faces:
        if abs(face.plane.b) <= 1e-6:
            continue
        poly = _polygon_xz(face.corners)
        if poly is None:
            continue
        try:
            clipped = poly.intersection(footprint)
        except Exception:
            continue
        if clipped.is_empty or clipped.area <= 0.05:
            continue
        grouped.setdefault(_top_plane_group_key(face.plane), []).append((face, clipped))

    supports: list[TopSupport] = []
    for group in grouped.values():
        group = sorted(group, key=lambda item: float(item[1].area), reverse=True)
        representative = group[0][0]
        try:
            unioned = unary_union([clipped for _face, clipped in group])
            unioned = unioned.intersection(footprint)
        except Exception:
            continue
        if unioned.is_empty or unioned.area <= 0.05:
            continue
        supports.append(
            TopSupport(
                face=representative,
                footprint=unioned,
                area=float(unioned.area),
            )
        )
    return sorted(supports, key=lambda support: support.area, reverse=True)


def _ridge_split_line_for_planes(
    footprint: Polygon,
    left: Plane,
    right: Plane,
) -> LineString | None:
    line_coeffs = _plane_equal_height_line_xz(left, right)
    if line_coeffs is None:
        return None
    a, b, c = line_coeffs
    minx, minz, maxx, maxz = footprint.bounds
    span = max(maxx - minx, maxz - minz, 1.0) * 4.0
    if abs(a) >= abs(b):
        point_z = (minz + maxz) * 0.5
        point = ((-b * point_z - c) / a, point_z)
    else:
        point_x = (minx + maxx) * 0.5
        point = (point_x, (-a * point_x - c) / b)
    direction = np.asarray([b, -a], dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    line = LineString(
        [
            (point[0] - direction[0] * span, point[1] - direction[1] * span),
            (point[0] + direction[0] * span, point[1] + direction[1] * span),
        ]
    )
    if line.intersection(footprint).length <= 1e-6:
        return None
    return line


def _highest_supported_plane_for_region(
    region: Polygon,
    supports: list[TopSupport],
    *,
    min_overlap_ratio: float = 0.05,
) -> TopSupport | None:
    point = region.representative_point()
    x = float(point.x)
    z = float(point.y)
    best: tuple[float, float, TopSupport] | None = None
    for support in supports:
        overlap = float(region.intersection(support.footprint).area)
        overlap_ratio = overlap / max(float(region.area), 1e-9)
        if overlap_ratio < min_overlap_ratio:
            continue
        y = _plane_y_at(_top_plane_up(support.face.plane), x, z)
        score = (y, overlap_ratio)
        if best is None or score > (best[0], best[1]):
            best = (y, overlap_ratio, support)
    return best[2] if best is not None else None


def _merge_regions_by_top_plane(
    regions: list[tuple[Polygon, PayloadFace]],
) -> list[tuple[Polygon, PayloadFace]]:
    grouped: dict[
        tuple[float, float, float, float],
        list[tuple[Polygon, PayloadFace]],
    ] = {}
    for region, face in regions:
        grouped.setdefault(_top_plane_group_key(face.plane), []).append((region, face))

    out: list[tuple[Polygon, PayloadFace]] = []
    for group in grouped.values():
        face = group[0][1]
        unioned = unary_union([region for region, _face in group])
        for part in _polygon_parts(unioned):
            if part.area > 0.05:
                out.append((part, face))
    return out


def _multi_piece_top_partition(
    footprint: Polygon,
    ceiling_faces: list[PayloadFace],
    *,
    min_coverage_ratio: float,
) -> tuple[list[tuple[Polygon, PayloadFace]], float] | None:
    grouped: dict[
        tuple[float, float, float, float],
        list[tuple[PayloadFace, Any]],
    ] = {}
    for face in ceiling_faces:
        if abs(face.plane.b) <= 1e-6:
            continue
        poly = _polygon_xz(face.corners)
        if poly is None:
            continue
        clipped = poly.intersection(footprint)
        if clipped.is_empty or clipped.area <= 0.05:
            continue
        grouped.setdefault(_top_plane_group_key(face.plane), []).append((face, clipped))

    supports: list[tuple[float, PayloadFace, Any]] = []
    for group in grouped.values():
        if not group:
            continue
        group = sorted(group, key=lambda item: float(item[1].area), reverse=True)
        unioned = unary_union([clipped for _face, clipped in group]).intersection(
            footprint
        )
        if unioned.is_empty or unioned.area <= 0.05:
            continue
        supports.append((float(unioned.area), group[0][0], unioned))

    if len(supports) < 2:
        return None

    top_regions: list[tuple[Polygon, PayloadFace]] = []
    covered = None
    sorted_supports = sorted(supports, reverse=True, key=lambda item: item[0])
    for _area, face, clipped in sorted_supports:
        remaining = clipped if covered is None else clipped.difference(covered)
        for part in _polygon_parts(remaining):
            if part.area <= 0.05:
                continue
            top_regions.append((part, face))
        covered = clipped if covered is None else unary_union([covered, clipped])

    if covered is None:
        return None
    coverage_ratio = float(covered.intersection(footprint).area) / max(
        float(footprint.area),
        1e-9,
    )
    if coverage_ratio < min_coverage_ratio:
        return None
    if len(top_regions) < 2:
        return None
    return top_regions, coverage_ratio


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        poly = make_valid_polygon(geometry)
        return [poly] if poly is not None and not poly.is_empty else []
    if geometry.geom_type == "MultiPolygon":
        return [
            poly
            for poly in geometry.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
    if hasattr(geometry, "geoms"):
        out: list[Polygon] = []
        for part in geometry.geoms:
            out.extend(_polygon_parts(part))
        return out
    return []


def _split_footprint_by_plane_intersection(
    footprint: Polygon,
    left: Plane,
    right: Plane,
) -> list[Polygon] | None:
    line_coeffs = _plane_equal_height_line_xz(left, right)
    if line_coeffs is None:
        return None
    a, b, c = line_coeffs
    minx, minz, maxx, maxz = footprint.bounds
    span = max(maxx - minx, maxz - minz, 1.0) * 4.0
    if abs(a) >= abs(b):
        point_z = (minz + maxz) * 0.5
        point = ((-b * point_z - c) / a, point_z)
    else:
        point_x = (minx + maxx) * 0.5
        point = (point_x, (-a * point_x - c) / b)
    direction = np.asarray([b, -a], dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    line = [
        (point[0] - direction[0] * span, point[1] - direction[1] * span),
        (point[0] + direction[0] * span, point[1] + direction[1] * span),
    ]
    try:
        parts = [
            part
            for part in split(footprint, LineString(line)).geoms
            if isinstance(part, Polygon) and part.area > 0.05
        ]
    except Exception:
        return None
    if len(parts) != 2:
        return None
    return parts


def _plane_equal_height_line_xz(
    left: Plane,
    right: Plane,
) -> tuple[float, float, float] | None:
    if abs(left.b) <= 1e-9 or abs(right.b) <= 1e-9:
        return None
    a = -right.b * left.a + left.b * right.a
    b = -right.b * left.c + left.b * right.c
    c = right.b * left.d - left.b * right.d
    if (a * a + b * b) <= 1e-18:
        return None
    return float(a), float(b), float(c)


def _assign_split_regions_to_faces(
    regions: list[Polygon],
    left: tuple[PayloadFace, Polygon],
    right: tuple[PayloadFace, Polygon],
) -> list[tuple[Polygon, PayloadFace]] | None:
    left_face, left_support = left
    right_face, right_support = right
    assignments: list[tuple[Polygon, PayloadFace]] = []
    used: set[str] = set()
    for region in regions:
        left_overlap = float(region.intersection(left_support).area)
        right_overlap = float(region.intersection(right_support).area)
        if max(left_overlap, right_overlap) / max(region.area, 1e-9) < 0.35:
            return None
        face = left_face if left_overlap >= right_overlap else right_face
        if face.locator_id in used:
            return None
        used.add(face.locator_id)
        assignments.append((region, face))
    return assignments


def _floor_y_for_part(
    payload: dict[str, Any],
    footprint: Polygon,
    *,
    corner_tol: float,
) -> float | None:
    ys: list[float] = []
    for room in payload.get("rooms", []) or []:
        for floor in room.get("floor", []) or []:
            corners = _clean_ring(_corners(floor), tol=corner_tol)
            poly = _polygon_xz(corners)
            if poly is None:
                continue
            overlap = float(footprint.intersection(poly).area)
            if overlap <= 1e-6:
                continue
            y = float(sum(c[1] for c in corners) / len(corners))
            ys.append(y)
    if not ys:
        return None
    return min(ys)


def _envelope_candidate_from_top_plane(
    footprint: Polygon,
    top_face: PayloadFace,
    *,
    floor_y: float,
    locator_id: str,
    overlap_ratio: float,
    top_source: str | None = None,
) -> EnvelopeCandidate:
    coords = _polygon_exterior_coords(footprint)
    top_plane = _top_plane_up(top_face.plane)
    floor_plane = Plane(a=0.0, b=-1.0, c=0.0, d=-float(floor_y))

    floor_corners = [[x, floor_y, z] for x, z in coords]
    top_corners = [[x, _plane_y_at(top_plane, x, z), z] for x, z in coords]

    faces = [
        PayloadFace(
            kind="floor",
            locator_id=f"{locator_id}::floor",
            corners=_orient_to_plane(floor_corners, floor_plane),
            plane=floor_plane,
        ),
        PayloadFace(
            kind="ceiling",
            locator_id=f"{locator_id}::top",
            corners=_orient_to_plane(top_corners, top_plane),
            plane=top_plane,
            source=top_face.source,
        ),
    ]

    for edge_index, ((x0, z0), (x1, z1)) in enumerate(_polygon_edges(coords)):
        wall_corners = [
            [x0, floor_y, z0],
            [x0, _plane_y_at(top_plane, x0, z0), z0],
            [x1, _plane_y_at(top_plane, x1, z1), z1],
            [x1, floor_y, z1],
        ]
        wall_plane = _plane_from_oriented_polygon(wall_corners)
        if wall_plane is None:
            continue
        faces.append(
            PayloadFace(
                kind="wall",
                locator_id=f"{locator_id}::wall::{edge_index}",
                corners=_orient_to_plane(wall_corners, wall_plane),
                plane=wall_plane,
            )
        )

    return EnvelopeCandidate(
        locator_id=locator_id,
        faces=faces,
        footprint_area_m2=float(footprint.area),
        top_source=top_source or top_face.locator_id,
        top_overlap_ratio=overlap_ratio,
    )


def _envelope_candidate_from_top_regions(
    top_regions: list[tuple[Polygon, PayloadFace]],
    *,
    floor_y: float,
    locator_id: str,
    coverage_ratio: float,
    footprint_override: Polygon | None = None,
    atomic_external_walls: bool = False,
) -> EnvelopeCandidate:
    if footprint_override is not None and not footprint_override.is_empty:
        footprint = footprint_override
    else:
        footprint_geom = unary_union([region for region, _face in top_regions])
        footprint = (
            footprint_geom
            if isinstance(footprint_geom, Polygon)
            and footprint_geom.is_valid
            and not footprint_geom.is_empty
            else _largest_polygon(footprint_geom)
        )
    if footprint is None:
        raise ValueError("top regions do not form a footprint")
    boundary_points = _top_region_boundary_points(top_regions)
    floor_coords = (
        _noded_region_exterior_coords(footprint, boundary_points)
        if atomic_external_walls
        else _remove_collinear_coords(_polygon_exterior_coords(footprint))
    )
    floor_plane = Plane(a=0.0, b=-1.0, c=0.0, d=-float(floor_y))
    floor_corners = [[x, floor_y, z] for x, z in floor_coords]
    faces = [
        PayloadFace(
            kind="floor",
            locator_id=f"{locator_id}::floor",
            corners=_orient_to_plane(floor_corners, floor_plane),
            plane=floor_plane,
        )
    ]

    wall_chains = _external_partition_wall_chains(
        top_regions,
        atomic=atomic_external_walls,
    )
    for top_index, (region, top_face) in enumerate(top_regions):
        top_plane = _top_plane_up(top_face.plane)
        coords = (
            _noded_region_exterior_coords(region, boundary_points)
            if atomic_external_walls
            else _remove_collinear_coords(_polygon_exterior_coords(region))
        )
        top_corners = [[x, _plane_y_at(top_plane, x, z), z] for x, z in coords]
        faces.append(
            PayloadFace(
                kind="ceiling",
                locator_id=f"{locator_id}::top::{top_index}",
                corners=_orient_to_plane(top_corners, top_plane),
                plane=top_plane,
                source=top_face.source,
            )
        )

    wall_index = 0
    for chain in wall_chains:
        top_corners = _wall_chain_top_corners(chain)
        x0, z0 = chain[0][0]
        x1, z1 = chain[-1][1]
        wall_corners = [
            [x0, floor_y, z0],
            *top_corners,
            [x1, floor_y, z1],
        ]
        wall_plane = _plane_from_oriented_polygon(wall_corners)
        if wall_plane is None:
            continue
        faces.append(
            PayloadFace(
                kind="wall",
                locator_id=f"{locator_id}::wall::{wall_index}",
                corners=_orient_to_plane(wall_corners, wall_plane),
                plane=wall_plane,
            )
        )
        wall_index += 1

    for step_index, corners in enumerate(
        _internal_partition_step_faces(
            top_regions,
            atomic=atomic_external_walls,
        )
    ):
        step_plane = _plane_from_oriented_polygon(corners)
        if step_plane is None:
            continue
        faces.append(
            PayloadFace(
                kind="wall",
                locator_id=f"{locator_id}::top-step::{step_index}",
                corners=_orient_to_plane(corners, step_plane),
                plane=step_plane,
            )
        )

    return EnvelopeCandidate(
        locator_id=locator_id,
        faces=faces,
        footprint_area_m2=float(footprint.area),
        top_source=" + ".join(face.locator_id for _region, face in top_regions),
        top_overlap_ratio=coverage_ratio,
    )


def _wall_chain_top_corners(
    chain: list[BoundarySegment],
    *,
    y_tol_m: float = 0.02,
) -> list[list[float]]:
    first_start, _first_end, first_face = chain[0]
    first_plane = _top_plane_up(first_face.plane)
    top_corners = [
        [
            first_start[0],
            _plane_y_at(first_plane, first_start[0], first_start[1]),
            first_start[1],
        ]
    ]
    for index, (_start, end, face) in enumerate(chain):
        plane = _top_plane_up(face.plane)
        current_y = _plane_y_at(plane, end[0], end[1])
        top_corners.append([end[0], current_y, end[1]])
        if index + 1 >= len(chain):
            continue
        _next_start, _next_end, next_face = chain[index + 1]
        next_plane = _top_plane_up(next_face.plane)
        next_y = _plane_y_at(next_plane, end[0], end[1])
        if abs(next_y - current_y) > y_tol_m:
            top_corners.append([end[0], next_y, end[1]])
    return top_corners


def _external_partition_wall_chains(
    top_regions: list[tuple[Polygon, PayloadFace]],
    *,
    atomic: bool = False,
) -> list[list[BoundarySegment]]:
    edge_counts: Counter[tuple[Point2, Point2]] = Counter()
    directed: list[BoundarySegment] = []
    if atomic:
        segments = _noded_top_region_boundary_segments(top_regions)
    else:
        segments = []
        for region, face in top_regions:
            coords = _remove_collinear_coords(_polygon_exterior_coords(region))
            segments.extend((start, end, face) for start, end in _polygon_edges(coords))
    for start, end, face in segments:
        key = _undirected_edge_key(start, end)
        edge_counts[key] += 1
        directed.append((start, end, face))
    boundary_edges = [
        (start, end, face)
        for start, end, face in directed
        if edge_counts[_undirected_edge_key(start, end)] == 1
    ]
    if atomic:
        return [[edge] for edge in boundary_edges]

    by_line: dict[tuple[float, float, float], list[BoundarySegment]] = {}
    for edge in boundary_edges:
        key = _line_key(edge[0], edge[1])
        if key is not None:
            by_line.setdefault(key, []).append(edge)

    chains: list[list[BoundarySegment]] = []
    for segments in by_line.values():
        chains.extend(_ordered_edge_chains(segments))
    return chains


def _internal_partition_step_faces(
    top_regions: list[tuple[Polygon, PayloadFace]],
    *,
    atomic: bool = False,
    y_tol_m: float = 0.02,
) -> list[list[list[float]]]:
    grouped: dict[tuple[Point2, Point2], list[BoundarySegment]] = {}
    if atomic:
        segments = _noded_top_region_boundary_segments(top_regions)
    else:
        segments = []
        for region, face in top_regions:
            coords = _remove_collinear_coords(_polygon_exterior_coords(region))
            segments.extend((start, end, face) for start, end in _polygon_edges(coords))
    for start, end, face in segments:
        grouped.setdefault(_undirected_edge_key(start, end), []).append(
            (start, end, face)
        )

    faces: list[list[list[float]]] = []
    for segments in grouped.values():
        if len(segments) != 2:
            continue
        (a_start, a_end, a_face), (b_start, b_end, b_face) = segments
        if not (_same_point2(a_start, b_end) and _same_point2(a_end, b_start)):
            continue
        a_plane = _top_plane_up(a_face.plane)
        b_plane = _top_plane_up(b_face.plane)
        a_start_y = _plane_y_at(a_plane, a_start[0], a_start[1])
        a_end_y = _plane_y_at(a_plane, a_end[0], a_end[1])
        b_start_y = _plane_y_at(b_plane, b_start[0], b_start[1])
        b_end_y = _plane_y_at(b_plane, b_end[0], b_end[1])
        if (
            abs(a_start_y - b_end_y) <= y_tol_m
            and abs(a_end_y - b_start_y) <= y_tol_m
        ):
            continue
        faces.append(
            [
                [b_start[0], b_start_y, b_start[1]],
                [b_end[0], b_end_y, b_end[1]],
                [a_start[0], a_start_y, a_start[1]],
                [a_end[0], a_end_y, a_end[1]],
            ]
        )
    return faces


def _top_region_boundary_points(
    top_regions: list[tuple[Polygon, PayloadFace]],
) -> list[Point2]:
    points: list[Point2] = []
    seen: set[Point2] = set()
    for region, _face in top_regions:
        for point in _polygon_exterior_coords(region):
            rounded = (round(point[0], 9), round(point[1], 9))
            if rounded in seen:
                continue
            seen.add(rounded)
            points.append(point)
    return points


def _noded_top_region_boundary_segments(
    top_regions: list[tuple[Polygon, PayloadFace]],
) -> list[BoundarySegment]:
    boundary_points = _top_region_boundary_points(top_regions)
    segments: list[BoundarySegment] = []
    for region, face in top_regions:
        coords = _polygon_exterior_coords(region)
        for start, end in _polygon_edges(coords):
            for split_start, split_end in _split_segment_at_points(
                start,
                end,
                boundary_points,
            ):
                segments.append((split_start, split_end, face))
    return segments


def _noded_region_exterior_coords(
    region: Polygon,
    boundary_points: list[Point2],
) -> list[Point2]:
    coords = _polygon_exterior_coords(region)
    if not coords:
        return coords
    noded: list[Point2] = []
    for start, end in _polygon_edges(coords):
        pieces = _split_segment_at_points(start, end, boundary_points)
        if not pieces:
            continue
        if not noded:
            noded.append(pieces[0][0])
        for _piece_start, piece_end in pieces:
            if not _same_point2(noded[-1], piece_end):
                noded.append(piece_end)
    if len(noded) > 1 and _same_point2(noded[0], noded[-1]):
        noded.pop()
    return noded


def _split_segment_at_points(
    start: Point2,
    end: Point2,
    points: list[Point2],
    *,
    tol: float = 1e-3,
) -> list[tuple[Point2, Point2]]:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length_sq = dx * dx + dz * dz
    if length_sq <= tol * tol:
        return []
    candidates: list[tuple[float, Point2]] = [(0.0, start), (1.0, end)]
    for point in points:
        t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz) / length_sq
        if t <= tol or t >= 1.0 - tol:
            continue
        projected = (start[0] + dx * t, start[1] + dz * t)
        if hypot(point[0] - projected[0], point[1] - projected[1]) > tol:
            continue
        candidates.append((float(t), projected))

    candidates.sort(key=lambda item: item[0])
    unique: list[tuple[float, Point2]] = []
    for t, point in candidates:
        if unique and abs(t - unique[-1][0]) <= tol:
            continue
        unique.append((t, point))

    pieces: list[tuple[Point2, Point2]] = []
    for (_t0, p0), (_t1, p1) in pairwise(unique):
        if hypot(p1[0] - p0[0], p1[1] - p0[1]) <= tol:
            continue
        pieces.append((p0, p1))
    return pieces


def _ordered_edge_chains(
    segments: list[BoundarySegment],
) -> list[list[BoundarySegment]]:
    unused = list(segments)
    chains: list[list[BoundarySegment]] = []
    while unused:
        chain = [unused.pop(0)]
        changed = True
        while changed:
            changed = False
            for index, segment in enumerate(unused):
                start, end, _face = segment
                if _same_point2(chain[-1][1], start):
                    chain.append(segment)
                    unused.pop(index)
                    changed = True
                    break
                if _same_point2(end, chain[0][0]):
                    chain.insert(0, segment)
                    unused.pop(index)
                    changed = True
                    break
        chains.append(chain)
    return chains


def _line_key(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float] | None:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length = (dx * dx + dz * dz) ** 0.5
    if length <= 1e-9:
        return None
    a = dz / length
    b = -dx / length
    c = a * start[0] + b * start[1]
    if a < -1e-9 or (abs(a) <= 1e-9 and b < -1e-9):
        a = -a
        b = -b
        c = -c
    return (round(a, 6), round(b, 6), round(c, 6))


def _same_point2(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return abs(left[0] - right[0]) <= 1e-6 and abs(left[1] - right[1]) <= 1e-6


def _undirected_edge_key(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    a = (round(start[0], 6), round(start[1], 6))
    b = (round(end[0], 6), round(end[1], 6))
    return (a, b) if a <= b else (b, a)


def _top_plane_up(plane: Plane) -> Plane:
    if plane.b >= 0.0:
        return plane
    return Plane(a=-plane.a, b=-plane.b, c=-plane.c, d=-plane.d)


def _plane_y_at(plane: Plane, x: float, z: float) -> float:
    if abs(plane.b) <= 1e-9:
        raise ValueError("cannot evaluate vertical plane as y(x,z)")
    return float((plane.d - plane.a * x - plane.c * z) / plane.b)


def _top_plane_is_above_floor(
    footprint: Polygon,
    top_plane: Plane,
    *,
    floor_y: float,
    min_clearance_m: float = 0.01,
) -> bool:
    return all(
        _plane_y_at(top_plane, x, z) >= floor_y + min_clearance_m
        for x, z in _polygon_exterior_coords(footprint)
    )


def _top_regions_are_above_floor(
    top_regions: list[tuple[Polygon, PayloadFace]],
    *,
    floor_y: float,
    min_clearance_m: float = 0.01,
) -> bool:
    for region, face in top_regions:
        top_plane = _top_plane_up(face.plane)
        for x, z in _polygon_exterior_coords(region):
            if _plane_y_at(top_plane, x, z) < floor_y + min_clearance_m:
                return False
    return True


def _polygon_exterior_coords(poly: Polygon) -> list[tuple[float, float]]:
    oriented = orient_polygon(poly, sign=1.0)
    coords = [(float(x), float(z)) for x, z in oriented.exterior.coords]
    if coords and coords[0] == coords[-1]:
        coords.pop()
    cleaned: list[tuple[float, float]] = []
    for coord in coords:
        if cleaned and _same_point2(cleaned[-1], coord):
            continue
        cleaned.append(coord)
    if len(cleaned) > 1 and _same_point2(cleaned[0], cleaned[-1]):
        cleaned.pop()
    coords = cleaned
    return coords


def _remove_collinear_coords(
    coords: list[tuple[float, float]],
    *,
    tol: float = 1e-8,
) -> list[tuple[float, float]]:
    if len(coords) <= 3:
        return coords
    out: list[tuple[float, float]] = []
    for index, point in enumerate(coords):
        prev = coords[index - 1]
        nxt = coords[(index + 1) % len(coords)]
        ux = point[0] - prev[0]
        uz = point[1] - prev[1]
        vx = nxt[0] - point[0]
        vz = nxt[1] - point[1]
        cross = ux * vz - uz * vx
        if abs(cross) <= tol:
            dot = ux * vx + uz * vz
            if dot >= -tol:
                continue
        out.append(point)
    return out if len(out) >= 3 else coords


def _polygon_edges(
    coords: list[tuple[float, float]],
) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    for index, coord in enumerate(coords):
        yield coord, coords[(index + 1) % len(coords)]


def _corners(piece: dict[str, Any]) -> list[list[float]]:
    return [_point3(p) for p in piece.get("corners", []) or []]


def _point3(point: Any) -> list[float]:
    if isinstance(point, dict):
        return [float(point["x"]), float(point["y"]), float(point["z"])]
    return [float(point[0]), float(point[1]), float(point[2])]


def _clean_ring(corners: list[list[float]], *, tol: float) -> list[list[float]]:
    if not corners:
        return []
    cleaned: list[list[float]] = []
    for corner in corners:
        if cleaned and _distance(cleaned[-1], corner) <= tol:
            continue
        cleaned.append(corner)
    if len(cleaned) > 1 and _distance(cleaned[0], cleaned[-1]) <= tol:
        cleaned.pop()
    return cleaned


def _distance(a: list[float], b: list[float]) -> float:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.linalg.norm(delta))


def _floor_plane(corners: list[list[float]]) -> Plane:
    y = float(sum(c[1] for c in corners) / len(corners))
    return Plane(a=0.0, b=-1.0, c=0.0, d=-y)


def _plane_from_dict(plane: dict[str, Any]) -> Plane:
    return Plane(
        a=float(plane["a"]),
        b=float(plane["b"]),
        c=float(plane["c"]),
        d=float(plane["d"]),
    )


def _plane_from_oriented_polygon(corners: list[list[float]]) -> Plane | None:
    normal = np.asarray(newell_normal(corners), dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-12:
        normal /= norm
        centroid = np.asarray(corners, dtype=float).mean(axis=0)
        return Plane(
            a=float(normal[0]),
            b=float(normal[1]),
            c=float(normal[2]),
            d=float(normal @ centroid),
        )
    fitted = fit_plane_any(corners)
    if fitted is None:
        return None
    return Plane(
        a=float(fitted[0]),
        b=float(fitted[1]),
        c=float(fitted[2]),
        d=float(fitted[3]),
    )


def _orient_wall_outward(
    corners: list[list[float]],
    *,
    room_center: np.ndarray | None,
    floor_corners: list[list[float]] | None,
) -> list[list[float]]:
    normal = np.asarray(newell_normal(corners), dtype=float)
    wall_center = np.asarray(corners, dtype=float).mean(axis=0)
    nxz_len = float(np.linalg.norm(normal[[0, 2]]))
    floor_poly = _floor_polygon_xz(floor_corners)
    if floor_poly is not None and nxz_len > 1e-12:
        step_m = 0.10
        nx = float(normal[0] / nxz_len)
        nz = float(normal[2] / nxz_len)
        along = Point(wall_center[0] + nx * step_m, wall_center[2] + nz * step_m)
        opposite = Point(wall_center[0] - nx * step_m, wall_center[2] - nz * step_m)
        normal_inside = floor_poly.covers(along)
        opposite_inside = floor_poly.covers(opposite)
        if normal_inside != opposite_inside:
            return list(reversed(corners)) if normal_inside else corners

    fallback_dot = (
        float(np.dot(normal, wall_center - room_center))
        if room_center is not None
        else 0.0
    )
    if room_center is not None and fallback_dot < 0.0:
        return list(reversed(corners))
    return corners


def _floor_polygon_xz(corners: list[list[float]] | None) -> Polygon | None:
    return _polygon_xz(corners)


def _polygon_xz(corners: list[list[float]] | None) -> Polygon | None:
    if corners is None or len(corners) < 3:
        return None
    try:
        poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _room_center(room: dict[str, Any]) -> np.ndarray | None:
    pts: list[list[float]] = []
    for floor in room.get("floor", []) or []:
        pts.extend(_corners(floor))
    if not pts:
        return None
    return np.asarray(pts, dtype=float).mean(axis=0)


def _orient_to_plane(corners: list[list[float]], plane: Plane) -> list[list[float]]:
    """Make polygon winding agree with the plane normal via right-hand rule."""

    target = np.asarray([plane.a, plane.b, plane.c], dtype=float)
    if float(np.linalg.norm(target)) <= 1e-12:
        return corners
    target = target / float(np.linalg.norm(target))
    normal = np.asarray(newell_normal(corners), dtype=float)
    if float(np.linalg.norm(normal)) > 1e-12:
        if float(np.dot(normal, target)) < 0.0:
            return [corners[0], *reversed(corners[1:])]
        return corners

    p0 = np.asarray(corners[0], dtype=float)
    for i in range(1, len(corners) - 1):
        u = np.asarray(corners[i], dtype=float) - p0
        v = np.asarray(corners[i + 1], dtype=float) - p0
        cross = np.cross(u, v)
        if float(np.linalg.norm(cross)) <= 1e-12:
            continue
        if float(np.dot(cross, target)) < 0.0:
            return [corners[0], *reversed(corners[1:])]
        return corners
    return corners


def _room_index_from_locator(locator_id: Any) -> int | None:
    if not isinstance(locator_id, str):
        return None
    marker = "::tier-ceiling-computed-oblique-room::"
    if marker in locator_id:
        tail = locator_id.split(marker, 1)[1]
        return _int_or_none(tail.split(":", 1)[0])
    marker = "::tier-ceiling-raw::"
    if marker in locator_id:
        tail = locator_id.split(marker, 1)[1]
        return _int_or_none(tail.split(":", 1)[0])
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
