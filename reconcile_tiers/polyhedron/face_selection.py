"""Stage A PolyFit-style face candidate generation and selection.

This module is deliberately standalone while the legacy cell selector remains
the production path. It implements the first useful subset of the approved
plan: closed-boundary plane vocabularies are converted into planar face
candidates, edge incidence is encoded as a hard manifold constraint, and a
binary MILP chooses a watertight face set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import TypeAlias

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient as orient_polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.half_edge import (
    HalfEdgePolyhedron,
    build_from_planar_polygons,
    three_plane_intersection,
)
from reconcile_tiers.polyhedron.priors import DEFAULT, SelectionWeights
from reconcile_tiers.polyhedron.validity import validate_polyhedron

EdgeKey: TypeAlias = tuple[int, int]

_PLANE_TOL = 1e-7
_POINT_KEY_SCALE = 1_000_000
_HORIZONTAL_NORMAL_MIN = 0.9961946980917455  # cos(5 degrees)
_OBLIQUE_TOP_B_MIN = 0.17364817766693  # sin(10 degrees) — minimum upward slope
_ENVELOPE_HEIGHT_REFERENCE_M = 0.0  # bonus measured above this y; 0 → bonus = 0.05·avg_y
_ENVELOPE_HEIGHT_WEIGHT = 20.0  # multiplier on area·avg_y for outer-envelope prior
_COMPETING_OBLIQUE_AZIMUTH_MIN_COS = -0.5  # angles ≥ 120° apart count as competing
_MIN_PRISM_HEIGHT_M = 0.05
DEFAULT_MAX_INTERSECTIONS = 50_000
DEFAULT_MAX_CANDIDATES = 2_000


@dataclass(frozen=True, slots=True)
class CandidateFace:
    face_id: int
    plane_id: int
    polygon: Polygon
    edge_keys: tuple[EdgeKey, ...]
    supporting_points: np.ndarray
    support_density: float
    confidence_label: str
    corners: tuple[tuple[float, float, float], ...]
    plane: Plane
    area: float
    support_score: float
    coverage_polygon: Polygon
    domain_area: float
    prism_id: int | None = None  # paper-aligned: faces in the same prism move together


@dataclass(frozen=True, slots=True)
class FaceSelectionResult:
    selected: tuple[CandidateFace, ...]
    objective: float
    energy_breakdown: dict[str, float]
    solver_status: str
    elapsed_seconds: float


def generate_tile_candidates(
    tier_payload: Mapping[str, Any],
    *,
    domain_polygon: Polygon,
    domain_story: Any,
    domain_floor_y: float | None,
    rooms_by_story: Mapping[Any, list[dict[str, Any]]] | None = None,
    sample_density_per_m2: float = 4.0,
    corner_tol: float = 0.02,
    grid_mm: float = 1.0,
) -> list[CandidateFace]:
    """Build candidate faces directly from `tier_payload` polygons (§12 of
    IMPLEMENTATION_PLAN.md). Each ceiling tile, room wall, room floor,
    visual_shell, and gable_closure becomes one CandidateFace anchored to
    its actual scan-derived corners. Edge keys are derived from
    vertex-coincidence (quantized to a `grid_mm` grid) so two faces share
    an edge iff they share a vertex pair — sidesteps the plane-pair
    collisions hit by the synthesis-based generator.
    """

    grid_m = float(grid_mm) * 1e-3
    vertex_map: dict[tuple[int, int, int], int] = {}
    out: list[CandidateFace] = []
    domain = orient_polygon(domain_polygon, sign=1.0)

    next_face_id = 0

    def add_tile(
        corners3d: Sequence[Sequence[float]],
        plane: Plane | None,
        label: str,
    ) -> None:
        nonlocal next_face_id
        if not corners3d or len(corners3d) < 3:
            return
        corners_tuple: tuple[tuple[float, float, float], ...] = tuple(
            (float(c[0]), float(c[1]), float(c[2])) for c in corners3d
        )
        if plane is None:
            plane = _plane_from_oriented_polygon_local(corners_tuple)
            if plane is None:
                return
        try:
            normalized_plane = _normalize_plane(plane)
        except ValueError:
            return
        area = _polygon_area_3d(corners_tuple)
        if area <= 1e-9:
            return
        try:
            _origin, u, v = _plane_frame(normalized_plane)
            poly2d = Polygon(
                [
                    (
                        float(np.asarray(c, dtype=float) @ u),
                        float(np.asarray(c, dtype=float) @ v),
                    )
                    for c in corners_tuple
                ]
            )
        except Exception:
            return
        if poly2d.is_empty or poly2d.area <= 1e-9:
            return
        coverage_polygon = _coverage_polygon_xz(corners_tuple, domain)
        # XZ-coverage gating only applies to roughly-horizontal faces
        # (floors and ceilings). Walls and other steeply-tilted faces
        # project to a line in XZ — coverage area is meaningless there.
        if abs(float(normalized_plane.b)) > 0.5:
            if coverage_polygon.is_empty or coverage_polygon.area <= 0.05:
                return
        edge_keys = _ring_edge_keys_geometric(
            corners_tuple, vertex_map=vertex_map, grid=grid_m
        )
        samples = _sample_tile_points(
            corners_tuple, normalized_plane, sample_density_per_m2
        )
        out.append(
            CandidateFace(
                face_id=next_face_id,
                plane_id=next_face_id,
                polygon=poly2d,
                edge_keys=edge_keys,
                supporting_points=samples,
                support_density=float(len(samples)) / area,
                confidence_label=label,
                corners=corners_tuple,
                plane=normalized_plane,
                area=area,
                support_score=float(len(samples)),
                coverage_polygon=coverage_polygon,
                domain_area=float(domain.area),
                prism_id=0,
            )
        )
        next_face_id += 1

    # 1. Ceiling tiles (story-filtered by `_ceiling_belongs_to_story`-style
    # heuristic — caller provides `rooms_by_story` for the y-bracket lookup).
    for tile in tier_payload.get("ceiling") or []:
        if not _tile_belongs_to_story(tile, domain_story, rooms_by_story):
            continue
        corners = _corners_from_dict_list(tile.get("corners") or [])
        plane = _plane_from_dict_safe(tile.get("plane"))
        label = _ceiling_source_to_label(tile.get("source"), plane)
        add_tile(corners, plane, label)

    # 2. Per-room walls and floor (filtered to the domain's story).
    if rooms_by_story is not None:
        for room in rooms_by_story.get(domain_story, []):
            for wall in room.get("walls") or []:
                corners = _corners_from_dict_list(wall.get("corners") or [])
                add_tile(corners, None, "wall")
            for floor in room.get("floor") or []:
                corners = _corners_from_dict_list(floor.get("corners") or [])
                add_tile(corners, None, "floor")

    # 3. Visual shells (gable_end, etc.) — story-assign by avg y.
    for shell in tier_payload.get("visual_shells") or []:
        if not _tile_belongs_to_story(shell, domain_story, rooms_by_story):
            continue
        corners = _corners_from_dict_list(shell.get("corners") or [])
        plane = _plane_from_dict_safe(shell.get("plane"))
        # Gable shells are oblique by construction — map to single-oblique.
        add_tile(corners, plane, "single-oblique")

    # 4. Gable closures (small triangular fillers — keep the dedicated label).
    for closure in tier_payload.get("gable_closures") or []:
        if not _tile_belongs_to_story(closure, domain_story, rooms_by_story):
            continue
        corners = _corners_from_dict_list(closure.get("corners") or [])
        add_tile(corners, None, "gable-closure")

    # 5. Synthetic domain-boundary walls — fallback so the polyhedron can
    # close even when rooms[].walls is sparse / empty. Each ring edge of
    # the domain becomes a wall trapezoid from `domain_floor_y` up to the
    # ceiling tile y at that XZ corner (so wall top corners coincide with
    # ceiling tile corners — required for vertex-coincidence edge keys).
    if domain_floor_y is not None and out:
        ceiling_planes = [
            cf.plane
            for cf in out
            if cf.confidence_label
            in {"flat-ceiling", "single-oblique", "gable-closure"}
        ]

        def _top_y_at(x: float, z: float) -> float:
            ys: list[float] = []
            for plane in ceiling_planes:
                if abs(plane.b) <= 1e-9:
                    continue
                ys.append((plane.d - plane.a * x - plane.c * z) / plane.b)
            if not ys:
                return domain_floor_y + 2.5
            return float(max(ys))

        ring = list(domain.exterior.coords)[:-1]
        n = len(ring)
        # Only emit synthetic walls if no real walls exist for this domain
        # (avoids duplicating room walls and creating overlapping faces).
        has_real_walls = any(cf.confidence_label == "wall" for cf in out)
        if not has_real_walls:
            for idx in range(n):
                sx, sz = ring[idx]
                ex, ez = ring[(idx + 1) % n]
                start_top_y = _top_y_at(float(sx), float(sz))
                end_top_y = _top_y_at(float(ex), float(ez))
                if (
                    start_top_y <= domain_floor_y + 0.01
                    or end_top_y <= domain_floor_y + 0.01
                ):
                    continue
                corners = (
                    (float(sx), float(domain_floor_y), float(sz)),
                    (float(sx), float(start_top_y), float(sz)),
                    (float(ex), float(end_top_y), float(ez)),
                    (float(ex), float(domain_floor_y), float(ez)),
                )
                add_tile(corners, None, "wall")

    return out


def _ceiling_source_to_label(source: Any, plane: Plane | None) -> str:
    """Map `tier_payload.ceiling[].source` to the label vocabulary the
    downstream envelope payload + tests expect."""
    if plane is not None:
        norm = (
            float(plane.a) ** 2
            + float(plane.b) ** 2
            + float(plane.c) ** 2
        ) ** 0.5
        b_unit = abs(float(plane.b)) / norm if norm > 1e-12 else 0.0
        if b_unit > 0.996:  # cos(5 deg)
            return "flat-ceiling"
        if b_unit > 0.05:
            return "single-oblique"
    src = str(source or "").lower()
    if "oblique" in src or "computed" in src:
        return "single-oblique"
    if "flat" in src or "merged" in src or "raw" in src:
        return "flat-ceiling"
    return "flat-ceiling"


def _corners_from_dict_list(
    raw: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for c in raw:
        try:
            out.append((float(c["x"]), float(c["y"]), float(c["z"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _plane_from_dict_safe(raw: Mapping[str, Any] | None) -> Plane | None:
    if not raw:
        return None
    try:
        return Plane(
            a=float(raw["a"]),
            b=float(raw["b"]),
            c=float(raw["c"]),
            d=float(raw["d"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _plane_from_oriented_polygon_local(
    corners: Sequence[Sequence[float]],
) -> Plane | None:
    """Compute a plane from polygon corners via Newell's method."""
    if len(corners) < 3:
        return None
    nx = ny = nz = 0.0
    n = len(corners)
    for i in range(n):
        c = corners[i]
        nxt = corners[(i + 1) % n]
        nx += (float(c[1]) - float(nxt[1])) * (float(c[2]) + float(nxt[2]))
        ny += (float(c[2]) - float(nxt[2])) * (float(c[0]) + float(nxt[0]))
        nz += (float(c[0]) - float(nxt[0])) * (float(c[1]) + float(nxt[1]))
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm <= 1e-12:
        return None
    nx /= norm
    ny /= norm
    nz /= norm
    cx = sum(float(c[0]) for c in corners) / n
    cy = sum(float(c[1]) for c in corners) / n
    cz = sum(float(c[2]) for c in corners) / n
    d = nx * cx + ny * cy + nz * cz
    return Plane(a=nx, b=ny, c=nz, d=d)


def _ring_edge_keys_geometric(
    corners: Sequence[Sequence[float]],
    *,
    vertex_map: dict[tuple[int, int, int], int],
    grid: float,
) -> tuple[EdgeKey, ...]:
    n = len(corners)
    if n < 3:
        return tuple()
    vids: list[int] = []
    for c in corners:
        key = (
            int(round(float(c[0]) / grid)),
            int(round(float(c[1]) / grid)),
            int(round(float(c[2]) / grid)),
        )
        vid = vertex_map.get(key)
        if vid is None:
            vid = len(vertex_map)
            vertex_map[key] = vid
        vids.append(vid)
    return tuple(
        tuple(sorted((vids[i], vids[(i + 1) % n]))) for i in range(n)
    )


def _sample_tile_points(
    corners: Sequence[Sequence[float]],
    plane: Plane,
    density_per_m2: float,
) -> np.ndarray:
    """Sample points uniformly across a planar 3D polygon, projected back
    to the plane via barycentric. Density measured in plane-area
    samples/m². Used as scan-evidence proxy for PolyFit's data-fit term."""
    if len(corners) < 3:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(corners, dtype=float)
    # Triangle-fan sampling; each triangle gets samples proportional to area.
    origin = arr[0]
    samples: list[np.ndarray] = []
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - origin
        v2 = arr[i + 1] - origin
        tri_area = 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
        if tri_area <= 1e-9:
            continue
        n_samples = max(1, int(round(tri_area * density_per_m2)))
        # Stratified barycentric sampling.
        for k in range(n_samples):
            r1 = (k + 0.5) / n_samples
            r2 = (k + 0.5) / n_samples
            sqrt_r1 = r1**0.5
            u = 1.0 - sqrt_r1
            v = sqrt_r1 * (1.0 - r2)
            w = sqrt_r1 * r2
            point = u * origin + v * arr[i] + w * arr[i + 1]
            samples.append(point)
    if not samples:
        return np.empty((0, 3), dtype=float)
    return np.asarray(samples, dtype=float)


def _tile_belongs_to_story(
    tile: Mapping[str, Any],
    story_id: Any,
    rooms_by_story: Mapping[Any, list[dict[str, Any]]] | None,
) -> bool:
    """Heuristic: assign tile to the story whose floor sits just below the
    tile's average y. Tiles with explicit `story` use that. Used to bucket
    `tier_payload.ceiling[]` and `visual_shells[]` (which carry no story
    field) into per-story candidate pools."""
    explicit = tile.get("story") if isinstance(tile, Mapping) else None
    if explicit is not None:
        return explicit == story_id
    corners = tile.get("corners") if isinstance(tile, Mapping) else None
    if not corners:
        return False
    ys = [
        float(c.get("y"))
        for c in corners
        if isinstance(c, Mapping) and c.get("y") is not None
    ]
    if not ys:
        return False
    tile_y = sum(ys) / len(ys)
    if not rooms_by_story:
        return story_id is None

    def floor_y(s: Any) -> float | None:
        floors_y: list[float] = []
        for room in rooms_by_story.get(s, []):
            for floor in room.get("floor") or []:
                fcs = floor.get("corners") or []
                ys_f = [
                    float(c.get("y"))
                    for c in fcs
                    if isinstance(c, Mapping) and c.get("y") is not None
                ]
                if ys_f:
                    floors_y.append(sum(ys_f) / len(ys_f))
        if not floors_y:
            return None
        return sum(floors_y) / len(floors_y)

    own_floor = floor_y(story_id)
    if own_floor is None or own_floor > tile_y:
        return False
    best_story = None
    best_gap = float("inf")
    for s in rooms_by_story:
        sf = floor_y(s)
        if sf is None or sf > tile_y:
            continue
        gap = tile_y - sf
        if gap < best_gap:
            best_gap = gap
            best_story = s
    return best_story == story_id


def generate_candidates(
    planes: Sequence[Plane],
    domain_polygon: Polygon,
    bounding_prism: tuple[float, float],
    scan_points: np.ndarray,
    epsilon: float = DEFAULT.epsilon_meters,
    *,
    min_support_points: int = DEFAULT.min_support_points,
    confidence_labels: Sequence[str] | None = None,
    max_intersections: int = DEFAULT_MAX_INTERSECTIONS,
    plane_support_ratios: Mapping[int, float] | None = None,
    plane_footprints: Mapping[int, Polygon] | None = None,
) -> list[CandidateFace]:
    """Generate one bounded polygon candidate for each input plane.

    The current implementation targets the M1/M2 closed-boundary case: input
    planes are outward-oriented boundary planes, so the solid interior satisfies
    ``n·x <= d`` for every plane. Vertices are all valid three-plane
    intersections, and each face polygon is the ordered ring of intersections
    lying on that face.
    """

    if len(planes) < 4:
        return []
    y_min, y_max = (float(bounding_prism[0]), float(bounding_prism[1]))
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    labels = confidence_labels or ()
    normalized = [_normalize_plane(plane) for plane in planes]
    points = _as_points(scan_points)

    if (
        plane_footprints is not None
        and len(plane_footprints) >= 2
        and _planes_have_normal_variation(plane_footprints, normalized)
    ):
        arrangement = _domain_arrangement_candidates(
            normalized,
            domain_polygon=domain_polygon,
            y_min=y_min,
            y_max=y_max,
            points=points,
            epsilon=max(float(epsilon), 1e-9),
            min_support_points=min_support_points,
            labels=labels,
            plane_support_ratios=plane_support_ratios,
            plane_footprints=plane_footprints,
        )
        if arrangement:
            return arrangement

    prism_candidates = _domain_prism_candidates(
        normalized,
        domain_polygon=domain_polygon,
        y_min=y_min,
        y_max=y_max,
        points=points,
        epsilon=max(float(epsilon), 1e-9),
        min_support_points=min_support_points,
        labels=labels,
        plane_support_ratios=plane_support_ratios,
    )
    if prism_candidates:
        return prism_candidates

    _check_halfspace_generation_budget(
        len(normalized),
        max_intersections=max_intersections,
    )

    candidates: list[CandidateFace] = []
    for plane_id, plane in enumerate(normalized):
        vertices = _face_vertices(
            plane_id,
            normalized,
            domain_polygon=domain_polygon,
            y_min=y_min,
            y_max=y_max,
        )
        if len(vertices) < 3:
            continue
        ordered = _order_vertices_on_plane(plane, vertices)
        if len(ordered) < 3:
            continue
        corners = tuple(tuple(float(v) for v in item.point) for item in ordered)
        polygon = Polygon([item.uv for item in ordered])
        if polygon.is_empty or polygon.area <= 1e-9:
            continue
        edge_keys = _edge_keys_for_ring(plane_id, ordered, normalized)
        if len(edge_keys) != len(corners):
            continue
        supporting, support_score = _supporting_points(
            plane,
            points,
            epsilon=max(float(epsilon), 1e-9),
        )
        if len(supporting) < min_support_points:
            continue
        area = _polygon_area_3d(corners)
        if area <= 1e-9:
            continue
        coverage_polygon = _coverage_polygon_xz(corners, domain_polygon)
        candidates.append(
            CandidateFace(
                face_id=len(candidates),
                plane_id=plane_id,
                polygon=polygon,
                edge_keys=edge_keys,
                supporting_points=supporting,
                support_density=float(len(supporting)) / area,
                confidence_label=(
                    str(labels[plane_id]) if plane_id < len(labels) else "plane"
                ),
                corners=corners,
                plane=plane,
                area=area,
                support_score=support_score,
                coverage_polygon=coverage_polygon,
                domain_area=float(domain_polygon.area),
            )
        )
    return candidates


def build_edge_incidence(
    candidates: Sequence[CandidateFace],
) -> dict[EdgeKey, list[int]]:
    incidence: dict[EdgeKey, list[int]] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.face_id if candidate.face_id >= 0 else index
        for edge_key in candidate.edge_keys:
            incidence.setdefault(edge_key, []).append(candidate_id)
    return incidence


def solve_face_selection_ilp(
    candidates: Sequence[CandidateFace],
    edge_incidence: Mapping[EdgeKey, Sequence[int]],
    weights: SelectionWeights | None = None,
    time_budget_seconds: float = 30.0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> FaceSelectionResult:
    """Select faces under the hard constraint that every edge has 0 or 2 faces."""

    start = perf_counter()
    weights = weights or SelectionWeights()
    ordered = tuple(candidates)
    if not ordered:
        return FaceSelectionResult((), 0.0, {}, "empty", perf_counter() - start)
    if len(ordered) > max_candidates:
        raise ValueError(
            f"face selection has {len(ordered)} candidates, above the "
            f"{max_candidates} candidate cap; decompose the support domain "
            "or pre-filter planes before solving the MILP"
        )

    coeffs = _objective_coefficients(ordered, weights)
    id_to_index = {candidate.face_id: idx for idx, candidate in enumerate(ordered)}
    edge_items = _edge_items(edge_incidence, id_to_index)
    edge_coeffs = _edge_complexity_coefficients(ordered, edge_items, weights)
    selected_ids: set[int] | None = None
    objective = 0.0
    status = "not_solved"

    if time_budget_seconds <= 1e-9:
        selected_ids = _round_and_repair(ordered, edge_incidence, coeffs)
        objective = _objective_for_ids(
            ordered,
            coeffs,
            edge_items,
            edge_coeffs,
            selected_ids,
        )
        status = "lp_relaxation_time_budget"
    else:
        try:
            selected_ids, objective, status = _solve_with_scipy_milp(
                ordered,
                coeffs,
                edge_items,
                edge_coeffs,
                time_budget_seconds=time_budget_seconds,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            selected_ids = _round_and_repair(ordered, edge_incidence, coeffs)
            objective = _objective_for_ids(
                ordered,
                coeffs,
                edge_items,
                edge_coeffs,
                selected_ids,
            )
            status = f"lp_relaxation_fallback:{type(exc).__name__}"

    if selected_ids is None:
        selected_ids = _round_and_repair(ordered, edge_incidence, coeffs)
        objective = _objective_for_ids(
            ordered,
            coeffs,
            edge_items,
            edge_coeffs,
            selected_ids,
        )
        status = "lp_relaxation_infeasible"

    selected = tuple(
        candidate for candidate in ordered if candidate.face_id in selected_ids
    )
    elapsed = perf_counter() - start
    return FaceSelectionResult(
        selected=selected,
        objective=objective,
        energy_breakdown=_energy_breakdown(
            selected,
            ordered,
            edge_items,
            weights,
        ),
        solver_status=status,
        elapsed_seconds=elapsed,
    )


def assemble_polyhedron(selection: FaceSelectionResult) -> HalfEdgePolyhedron:
    """Build and validate a half-edge polyhedron from selected candidates."""

    polygons = [
        ([list(corner) for corner in candidate.corners], candidate.plane)
        for candidate in selection.selected
    ]
    polyhedron = build_from_planar_polygons(polygons)
    issues = validate_polyhedron(polyhedron)
    if issues:
        summary = ", ".join(f"{issue.kind}:{issue.ids}" for issue in issues[:5])
        raise ValueError(f"selected faces do not form a valid polyhedron: {summary}")
    return polyhedron


def face_selection_trace(
    candidates: Sequence[CandidateFace],
    edge_incidence: Mapping[EdgeKey, Sequence[int]],
    selection: FaceSelectionResult | None = None,
) -> dict[str, object]:
    """Return a compact JSON-friendly trace for candidate selection debugging."""

    selected_ids = (
        {candidate.face_id for candidate in selection.selected}
        if selection is not None
        else set()
    )
    incidence_sizes = [len(face_ids) for face_ids in edge_incidence.values()]
    return {
        "candidate_count": len(candidates),
        "edge_count": len(edge_incidence),
        "edge_incidence": {
            "min": min(incidence_sizes) if incidence_sizes else 0,
            "max": max(incidence_sizes) if incidence_sizes else 0,
            "open_edges": sum(1 for size in incidence_sizes if size == 1),
            "manifold_edges": sum(1 for size in incidence_sizes if size == 2),
            "overfull_edges": sum(1 for size in incidence_sizes if size > 2),
        },
        "candidates": [
            {
                "face_id": candidate.face_id,
                "plane_id": candidate.plane_id,
                "selected": candidate.face_id in selected_ids,
                "edge_count": len(candidate.edge_keys),
                "area": candidate.area,
                "support_points": len(candidate.supporting_points),
                "support_density": candidate.support_density,
                "support_score": candidate.support_score,
                "coverage_area": float(candidate.coverage_polygon.area),
                "domain_area": candidate.domain_area,
                "confidence_label": candidate.confidence_label,
                "corners": [list(corner) for corner in candidate.corners],
                "plane": {
                    "a": candidate.plane.a,
                    "b": candidate.plane.b,
                    "c": candidate.plane.c,
                    "d": candidate.plane.d,
                },
            }
            for candidate in candidates
        ],
        "selection": (
            {
                "selected_count": len(selection.selected),
                "objective": selection.objective,
                "energy_breakdown": selection.energy_breakdown,
                "solver_status": selection.solver_status,
                "elapsed_seconds": selection.elapsed_seconds,
            }
            if selection is not None
            else None
        ),
    }


def _planes_have_normal_variation(
    plane_footprints: Mapping[int, Polygon],
    planes: Sequence[Plane],
    *,
    cos_tol: float = 0.99,
) -> bool:
    """True if at least two top-plane footprints come from planes whose
    normals differ meaningfully (cos angle ≤ cos_tol). Without this guard,
    near-coplanar slight-y-offset flat ceilings (scan noise grouped as
    distinct planes) trigger the composite arrangement spuriously and the
    ILP fails to pick a coherent envelope."""
    pids = list(plane_footprints.keys())
    for i, a in enumerate(pids):
        na = np.array([planes[a].a, planes[a].b, planes[a].c], dtype=float)
        la = float(np.linalg.norm(na))
        if la <= 1e-9:
            continue
        na /= la
        for b in pids[i + 1 :]:
            nb = np.array([planes[b].a, planes[b].b, planes[b].c], dtype=float)
            lb = float(np.linalg.norm(nb))
            if lb <= 1e-9:
                continue
            nb /= lb
            if abs(float(na @ nb)) <= cos_tol:
                return True
    return False


def _domain_arrangement_candidates(
    planes: Sequence[Plane],
    *,
    domain_polygon: Polygon,
    y_min: float,
    y_max: float,
    points: np.ndarray,
    epsilon: float,
    min_support_points: int,
    labels: Sequence[str],
    plane_support_ratios: Mapping[int, float] | None,
    plane_footprints: Mapping[int, Polygon],
) -> list[CandidateFace]:
    """PolyFit-style multi-top arrangement: each ceiling plane group emits its
    own candidate face on its scan-supported (x,z) region. Walls bridge the
    floor to whichever plane is above each ring sub-segment, with augmented
    top vertices at kink crossings (W ∩ G ∩ H = single 3-plane vertex). The
    floor stays one piece. See IMPLEMENTATION_PLAN.md §11.
    """
    if domain_polygon.is_empty or domain_polygon.area <= 1e-9:
        return []
    domain = orient_polygon(domain_polygon, sign=1.0)
    coords = [(float(x), float(z)) for x, z in domain.exterior.coords[:-1]]
    if len(coords) < 3:
        return []

    floor_id = _find_horizontal_plane(planes, upward=False)
    if floor_id is None:
        return []
    floor = planes[floor_id]
    if abs(floor.b) <= 1e-9:
        return []
    floor_y = floor.d / floor.b
    if floor_y < y_min - _PLANE_TOL:
        return []

    boundary_matches = _match_domain_boundary_planes(coords, planes)
    if len(boundary_matches) != len(coords):
        return []

    # 1. Ownership: highest scan support claims first (architectural truth —
    # the plane that defined the most scanned ceiling area in this domain
    # dominates that area). Greedy subtraction so each (x,z) belongs to one
    # plane only. Tie-break by higher y when supports are equal.
    centroid = domain.centroid
    cx, cz = float(centroid.x), float(centroid.y)
    candidate_ids = [
        pid for pid in plane_footprints if planes[pid].b > _OBLIQUE_TOP_B_MIN
    ]
    sorted_top_ids = sorted(
        candidate_ids,
        key=lambda pid: (
            float(plane_support_ratios.get(pid, 0.0))
            if plane_support_ratios is not None
            else float(plane_footprints[pid].area),
            _y_on_plane(planes[pid], cx, cz),
        ),
        reverse=True,
    )
    if not sorted_top_ids:
        return []
    owned: dict[int, Polygon] = {}
    used = Polygon()
    for pid in sorted_top_ids:
        fp = plane_footprints[pid]
        if fp is None or fp.is_empty:
            continue
        try:
            region = fp.intersection(domain)
            region = region.difference(used)
        except Exception:
            continue
        if region.is_empty or region.area < 0.5:
            continue
        if region.geom_type == "MultiPolygon":
            region = max(region.geoms, key=lambda g: g.area)
        if not isinstance(region, Polygon):
            continue
        owned[pid] = region
        used = used.union(region)
    if len(owned) < 2:
        return []  # let the single-top fallback handle this

    # Gap-fill: walls outside any owned region produce 0-coverage sample
    # bands that bail the arrangement. Assign every uncovered patch of the
    # domain to the geometrically nearest owned region. This is the
    # composite-ceiling Voronoi extension — each plane claims its scan
    # support plus the proximate uncovered area.
    try:
        coverage = unary_union(list(owned.values()))
        gap_geom = domain.difference(coverage)
    except Exception:
        gap_geom = Polygon()
    if not gap_geom.is_empty and gap_geom.area > 1e-3:
        gap_polys: list[Polygon] = []
        if gap_geom.geom_type == "Polygon":
            gap_polys.append(gap_geom)
        elif gap_geom.geom_type == "MultiPolygon":
            gap_polys.extend(gap_geom.geoms)
        for gap in gap_polys:
            if gap.is_empty or gap.area <= 1e-3:
                continue
            best_pid = min(owned, key=lambda pid: owned[pid].distance(gap))
            try:
                merged = unary_union([owned[best_pid], gap])
            except Exception:
                continue
            if merged.is_empty:
                continue
            if merged.geom_type == "MultiPolygon":
                merged = max(merged.geoms, key=lambda g: g.area)
            if isinstance(merged, Polygon):
                owned[best_pid] = merged

    # 2. Sample wall edges at high resolution; record which owned plane covers
    #    each parametric position. Compress to (t_start, t_end, plane_id).
    n = len(coords)
    wall_segments: list[list[tuple[float, float, int]]] = []
    for idx in range(n):
        start = np.array(coords[idx], dtype=float)
        end = np.array(coords[(idx + 1) % n], dtype=float)
        seg_length = float(np.linalg.norm(end - start))
        if seg_length <= 1e-9:
            wall_segments.append([])
            continue
        # Step inward slightly so points sit inside ownership polygons rather
        # than exactly on the domain boundary (which counts as outside in
        # Shapely's contains test). For a CCW-oriented domain in (x,z) (per
        # `orient_polygon(..., sign=1.0)`), the leftward perpendicular to
        # the forward direction (end - start) points into the interior.
        normal_in = np.array([-(end[1] - start[1]), end[0] - start[0]]) / seg_length
        n_samples = max(60, int(seg_length * 20))
        samples = []
        for k in range(n_samples):
            t = (k + 0.5) / n_samples
            point_xz = start + t * (end - start) + normal_in * 0.001
            sample_pid = None
            sample_y = float("-inf")
            for pid, region in owned.items():
                if region.contains(Point(point_xz[0], point_xz[1])):
                    py = _y_on_plane(planes[pid], float(point_xz[0]), float(point_xz[1]))
                    if py > sample_y:
                        sample_y = py
                        sample_pid = pid
            samples.append((t, sample_pid))
        # Compress to segments
        segs: list[tuple[float, float, int]] = []
        cur_pid = samples[0][1]
        cur_start = 0.0
        for t, pid in samples[1:]:
            if pid != cur_pid:
                if cur_pid is not None:
                    segs.append((cur_start, t, cur_pid))
                cur_pid = pid
                cur_start = t
        if cur_pid is not None:
            segs.append((cur_start, 1.0, cur_pid))
        wall_segments.append(segs)

    # If any wall has no owning plane along the entire edge, fall back.
    if any(not segs for segs in wall_segments):
        return []

    # 3. Build wall faces. Each wall is one polygon with augmented top vertices
    #    at kink crossings. Top corners are at y_on_plane(owning_plane, x, z).
    out: list[CandidateFace | None] = []
    next_id = 0
    wall_top_segments_by_plane: dict[
        int, list[tuple[int, tuple[float, float, float], tuple[float, float, float]]]
    ] = {pid: [] for pid in owned}
    for ring_idx, segs in enumerate(wall_segments):
        wall_plane_id = boundary_matches[ring_idx][0]
        prev_plane_id = boundary_matches[(ring_idx - 1) % n][0]
        next_plane_id = boundary_matches[(ring_idx + 1) % n][0]
        start_xz = np.array(coords[ring_idx], dtype=float)
        end_xz = np.array(coords[(ring_idx + 1) % n], dtype=float)
        # Build polygon vertices walking the wall counterclockwise:
        # bottom from start to end at floor_y, then top from end back to start
        # at the appropriate plane y, with intermediate kink vertices.
        bottom_corners = [
            (float(start_xz[0]), floor_y, float(start_xz[1])),
            (float(end_xz[0]), floor_y, float(end_xz[1])),
        ]
        # Top corners walking back (end → start) with K+1 corners for K
        # segments: t=1 (right end), K-1 kinks at internal segment
        # boundaries, t=0 (left end).
        top_corners: list[tuple[float, float, float]] = []
        last_pid = segs[-1][2]
        top_corners.append(
            (
                float(end_xz[0]),
                _y_on_plane(
                    planes[last_pid], float(end_xz[0]), float(end_xz[1])
                ),
                float(end_xz[1]),
            )
        )
        for i in range(len(segs) - 1, 0, -1):
            t_kink = segs[i][0]  # equals segs[i-1][1]
            xz_kink = start_xz + t_kink * (end_xz - start_xz)
            kink_plane = planes[segs[i - 1][2]]
            top_corners.append(
                (
                    float(xz_kink[0]),
                    _y_on_plane(
                        kink_plane, float(xz_kink[0]), float(xz_kink[1])
                    ),
                    float(xz_kink[1]),
                )
            )
        first_pid = segs[0][2]
        top_corners.append(
            (
                float(start_xz[0]),
                _y_on_plane(
                    planes[first_pid], float(start_xz[0]), float(start_xz[1])
                ),
                float(start_xz[1]),
            )
        )

        polygon_corners = tuple(bottom_corners + top_corners)
        # Edge keys, walking the polygon: bottom (W,F), right (W,next_W),
        # then top edges (W, plane_id of segment) one per segment, then
        # left (W, prev_W).
        edge_keys: list[EdgeKey] = []
        edge_keys.append(tuple(sorted((wall_plane_id, floor_id))))
        edge_keys.append(tuple(sorted((wall_plane_id, next_plane_id))))
        # Top edges: as we walked top from t=1 down to t=0 with kink vertices
        # interleaved, consecutive vertex pairs along the top correspond to
        # one segment each. The number of top edges equals len(segs).
        # Top corners list (in walking order) has length = 2*len(segs) (one
        # vertex per segment endpoint t_b plus one per kink + one extra at
        # t=0). Re-derive edge plane ids by stepping segs in reverse.
        for seg_idx in range(len(segs) - 1, -1, -1):
            seg_pid = segs[seg_idx][2]
            edge_keys.append(tuple(sorted((wall_plane_id, seg_pid))))
        edge_keys.append(tuple(sorted((wall_plane_id, prev_plane_id))))

        if len(edge_keys) != len(polygon_corners):
            # Can happen if the bookkeeping above produces an inconsistency;
            # bail out and let the prism fallback take over.
            return []

        wall = _make_candidate(
            face_id=next_id,
            plane_id=wall_plane_id,
            plane=planes[wall_plane_id],
            corners=polygon_corners,
            edge_keys=tuple(edge_keys),
            points=points,
            epsilon=epsilon,
            min_support_points=min_support_points,
            label=_label_for_plane(labels, wall_plane_id),
            domain_polygon=domain,
            prism_id=0,
            plane_support_ratios=plane_support_ratios,
        )
        out.append(wall)
        next_id += 1

        # Record the contribution of each owning plane to this wall: each
        # segment becomes one (W_id, plane_id) "wall-share" entry the top
        # face later uses to know it touches W_id along that segment.
        for seg in segs:
            wall_top_segments_by_plane[seg[2]].append(
                (wall_plane_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            )

    # 4. Top faces: one per owning plane, polygon = owned region.
    for pid, region in owned.items():
        ring = list(region.exterior.coords)[:-1]
        if len(ring) < 3:
            continue
        # Reverse winding so the outward normal points up (consistent with
        # CCW domain orientation viewed from above for the floor face).
        ring_rev = list(reversed(ring))
        corners = tuple(
            (
                float(x),
                _y_on_plane(planes[pid], float(x), float(z)),
                float(z),
            )
            for x, z in ring_rev
        )
        # Edge keys: best-effort match each polygon edge to a wall plane it
        # lies along (tolerance-based) or to a neighbor top plane (kink).
        edge_keys = []
        for k in range(len(corners)):
            ax, _, az = corners[k]
            bx, _, bz = corners[(k + 1) % len(corners)]
            mx, mz = (ax + bx) * 0.5, (az + bz) * 0.5
            # Try wall planes first.
            wall_match = _nearest_wall_plane(
                (mx, mz), boundary_matches, planes
            )
            if wall_match is not None:
                edge_keys.append(tuple(sorted((pid, wall_match))))
                continue
            # Otherwise, this edge lies on a kink with another top plane.
            # Find the other top plane whose owned region also borders this
            # midpoint — the y at midpoint should match between pid and the
            # neighbor (kink line).
            neighbor = _nearest_top_neighbor(
                (mx, mz), pid, owned, planes
            )
            if neighbor is not None:
                edge_keys.append(tuple(sorted((pid, neighbor))))
                continue
            # Fallback: pair against the floor (degenerate but permissive).
            edge_keys.append(tuple(sorted((pid, floor_id))))
        top = _make_candidate(
            face_id=next_id,
            plane_id=pid,
            plane=planes[pid],
            corners=corners,
            edge_keys=tuple(edge_keys),
            points=points,
            epsilon=epsilon,
            min_support_points=min_support_points,
            label=_label_for_plane(labels, pid),
            domain_polygon=domain,
            prism_id=0,
            plane_support_ratios=plane_support_ratios,
        )
        out.append(top)
        next_id += 1

    # 5. Floor face: full domain at floor_y, perimeter edges (F, W_i).
    wall_ids = [m[0] for m in boundary_matches]
    floor_corners = tuple((float(x), floor_y, float(z)) for x, z in coords)
    floor_edge_keys = tuple(
        tuple(sorted((floor_id, w))) for w in wall_ids
    )
    floor_face = _make_candidate(
        face_id=next_id,
        plane_id=floor_id,
        plane=floor,
        corners=floor_corners,
        edge_keys=floor_edge_keys,
        points=points,
        epsilon=epsilon,
        min_support_points=min_support_points,
        label=_label_for_plane(labels, floor_id),
        domain_polygon=domain,
        prism_id=0,
        plane_support_ratios=plane_support_ratios,
    )
    out.append(floor_face)
    next_id += 1

    if any(face is None for face in out):
        return []
    return [face for face in out if face is not None]


def _nearest_wall_plane(
    midpoint_xz: tuple[float, float],
    boundary_matches: Sequence[
        tuple[int, tuple[float, float], tuple[float, float]]
    ],
    planes: Sequence[Plane],
    tol: float = 0.05,
) -> int | None:
    mx, mz = midpoint_xz
    for plane_id, start, end in boundary_matches:
        plane = planes[plane_id]
        residual = abs(plane.a * mx + plane.c * mz - plane.d)
        if residual <= tol:
            # Also check parametric position lies on the segment (with margin).
            sx, sz = start
            ex, ez = end
            seg = np.array([ex - sx, ez - sz])
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-9:
                continue
            t = ((mx - sx) * seg[0] + (mz - sz) * seg[1]) / (seg_len * seg_len)
            if -0.05 <= t <= 1.05:
                return plane_id
    return None


def _nearest_top_neighbor(
    midpoint_xz: tuple[float, float],
    self_pid: int,
    owned: Mapping[int, Polygon],
    planes: Sequence[Plane],
) -> int | None:
    self_y = _y_on_plane(planes[self_pid], midpoint_xz[0], midpoint_xz[1])
    best: tuple[float, int] | None = None
    for pid, region in owned.items():
        if pid == self_pid:
            continue
        other_y = _y_on_plane(planes[pid], midpoint_xz[0], midpoint_xz[1])
        diff = abs(self_y - other_y)
        if best is None or diff < best[0]:
            best = (diff, pid)
    return best[1] if best is not None else None


def _domain_prism_candidates(
    planes: Sequence[Plane],
    *,
    domain_polygon: Polygon,
    y_min: float,
    y_max: float,
    points: np.ndarray,
    epsilon: float,
    min_support_points: int,
    labels: Sequence[str],
    plane_support_ratios: Mapping[int, float] | None = None,
) -> list[CandidateFace]:
    """Enumerate every valid (floor, top) prism over the support domain.

    Paper-aligned (PolyFit / Bauchet–Lafarge) candidate generation: for each
    pair of compatible planes — a downward-facing "floor" plus an upward-facing
    "top" — emit a full prism (floor + walls + top) tagged with a shared
    `prism_id`. The MILP's per-prism binding constraint (`solve_face_selection_ilp`)
    then forces the selected subset to be a coherent set of prisms; the manifold
    edge constraint enforces watertightness across the union. This naturally
    represents stacked stories, gables alongside flat ceilings, and multi-piece
    domain tops as competing or co-existing alternatives the solver picks
    between.
    """

    if domain_polygon.is_empty or domain_polygon.area <= 1e-9:
        return []
    domain = orient_polygon(domain_polygon, sign=1.0)
    coords = [(float(x), float(z)) for x, z in domain.exterior.coords[:-1]]
    if len(coords) < 3:
        return []

    boundary_matches = _match_domain_boundary_planes(coords, planes)
    if len(boundary_matches) != len(coords):
        return []

    floor_ids = _enumerate_floor_planes(planes, coords=coords)
    top_ids = _enumerate_top_planes(planes)
    if not floor_ids or not top_ids:
        return []

    out: list[CandidateFace] = []
    next_face_id = 0
    next_prism_id = 0
    for floor_id in floor_ids:
        floor = planes[floor_id]
        if abs(floor.b) <= 1e-9:
            continue
        floor_y = floor.d / floor.b
        if floor_y < y_min - _PLANE_TOL or floor_y > y_max + _PLANE_TOL:
            continue
        for top_id in top_ids:
            if top_id == floor_id:
                continue
            top = planes[top_id]
            if abs(top.b) <= 1e-9:
                continue
            top_ys = tuple(_y_on_plane(top, x, z) for x, z in coords)
            if min(top_ys) < floor_y + _MIN_PRISM_HEIGHT_M:
                continue
            if max(top_ys) > y_max + _PLANE_TOL:
                continue
            prism_faces = _build_prism_faces(
                planes=planes,
                floor_id=floor_id,
                floor=floor,
                floor_y=floor_y,
                top_id=top_id,
                top=top,
                top_ys=top_ys,
                coords=coords,
                boundary_matches=boundary_matches,
                points=points,
                epsilon=epsilon,
                min_support_points=min_support_points,
                labels=labels,
                domain=domain,
                first_face_id=next_face_id,
                prism_id=next_prism_id,
                plane_support_ratios=plane_support_ratios,
            )
            if prism_faces is None:
                continue
            out.extend(prism_faces)
            next_face_id += len(prism_faces)
            next_prism_id += 1

    # Gable-pair prism is parked: implementation requires adding a vertical
    # `V_ridge` plane to the candidate-face vocabulary so split-wall edges
    # have non-degenerate (plane_a, plane_b) keys. See IMPLEMENTATION_PLAN.md
    # §10b for the rollout. For now only single-oblique prisms are emitted;
    # gables render as shed-roof approximations on per-room domains.
    return out


def _opposing_oblique_pairs(
    planes: Sequence[Plane],
    top_ids: Sequence[int],
) -> list[tuple[int, int]]:
    obliques = [
        idx
        for idx in top_ids
        if _OBLIQUE_TOP_B_MIN <= planes[idx].b < _HORIZONTAL_NORMAL_MIN
    ]
    out: list[tuple[int, int]] = []
    for i, a in enumerate(obliques):
        pa_xz = np.array([planes[a].a, planes[a].c], dtype=float)
        la = float(np.linalg.norm(pa_xz))
        if la <= 1e-9:
            continue
        pa_xz /= la
        for b in obliques[i + 1 :]:
            pb_xz = np.array([planes[b].a, planes[b].c], dtype=float)
            lb = float(np.linalg.norm(pb_xz))
            if lb <= 1e-9:
                continue
            pb_xz /= lb
            if float(pa_xz @ pb_xz) <= _COMPETING_OBLIQUE_AZIMUTH_MIN_COS:
                out.append((a, b))
    return out


def _ridge_xz_line(
    t1: Plane, t2: Plane
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project the ridge line (t1 ∩ t2) onto XZ. Returns (point, direction)
    where direction is a unit XZ vector and point is on the ridge in XZ."""
    # T1 ∩ T2 lifted to XZ: solve (a1, c1)·(x,z) + b1·y = d1 etc. for any y;
    # the (x,z) projection collapses by the b-eliminated combination.
    # Eliminate y by computing b2·T1 - b1·T2: (b2 a1 - b1 a2) x + (b2 c1 - b1 c2) z = b2 d1 - b1 d2
    a = float(t2.b * t1.a - t1.b * t2.a)
    c = float(t2.b * t1.c - t1.b * t2.c)
    d = float(t2.b * t1.d - t1.b * t2.d)
    norm = (a * a + c * c) ** 0.5
    if norm <= 1e-9:
        return None
    # XZ line: a·x + c·z = d. Direction is perpendicular to (a, c).
    direction = np.array([-c, a], dtype=float) / norm
    # Pick a point on the line: project origin onto a·x+c·z = d.
    point = np.array([a * d / (a * a + c * c), c * d / (a * a + c * c)], dtype=float)
    return point, direction


def _ridge_domain_crossings(
    ridge_point: np.ndarray,
    ridge_dir: np.ndarray,
    coords: Sequence[tuple[float, float]],
) -> list[tuple[int, np.ndarray]]:
    """Return list of (edge_index, crossing_point_xz) where the ridge line
    crosses the domain ring. Edge indices are 0..n-1 mapping the ring edge
    coords[i]→coords[(i+1) % n]."""
    n = len(coords)
    out: list[tuple[int, np.ndarray]] = []
    for i in range(n):
        s = np.array(coords[i], dtype=float)
        e = np.array(coords[(i + 1) % n], dtype=float)
        seg = e - s
        # Solve s + t·seg = ridge_point + u·ridge_dir for (t, u); t∈[0,1] crosses
        det = seg[0] * (-ridge_dir[1]) - seg[1] * (-ridge_dir[0])
        if abs(det) <= 1e-9:
            continue
        rhs = ridge_point - s
        t = (rhs[0] * (-ridge_dir[1]) - rhs[1] * (-ridge_dir[0])) / det
        if -1e-9 <= t <= 1.0 + 1e-9:
            out.append((i, s + max(0.0, min(1.0, t)) * seg))
    return out


def _build_gable_prism_faces(
    *,
    planes: Sequence[Plane],
    floor_id: int,
    floor: Plane,
    floor_y: float,
    t1_id: int,
    t2_id: int,
    coords: Sequence[tuple[float, float]],
    domain: Polygon,
    boundary_matches: Sequence[tuple[int, tuple[float, float], tuple[float, float]]],
    points: np.ndarray,
    epsilon: float,
    min_support_points: int,
    labels: Sequence[str],
    first_face_id: int,
    prism_id: int,
    plane_support_ratios: Mapping[int, float] | None,
    y_max: float,
) -> list[CandidateFace] | None:
    t1 = planes[t1_id]
    t2 = planes[t2_id]
    ridge = _ridge_xz_line(t1, t2)
    if ridge is None:
        return None
    ridge_point, ridge_dir = ridge
    crossings = _ridge_domain_crossings(ridge_point, ridge_dir, coords)
    if len(crossings) != 2:
        return None
    n = len(coords)
    side_normal = np.array([ridge_dir[1], -ridge_dir[0]], dtype=float)

    def side_of(point_xz: np.ndarray) -> float:
        return float(side_normal @ (point_xz - ridge_point))

    # Choose which oblique covers each side by checking T1's y at a point
    # away from the ridge: the side where T1's y is HIGHER is T1's side
    # (since obliques meet at the ridge and slope downward away from it).
    test_offset = ridge_point + side_normal
    t1_y_offset = _y_on_plane(t1, float(test_offset[0]), float(test_offset[1]))
    t2_y_offset = _y_on_plane(t2, float(test_offset[0]), float(test_offset[1]))
    if t1_y_offset >= t2_y_offset:
        positive_side_top_id, negative_side_top_id = t1_id, t2_id
    else:
        positive_side_top_id, negative_side_top_id = t2_id, t1_id
    positive_top = planes[positive_side_top_id]
    negative_top = planes[negative_side_top_id]

    # Apex height at each crossing must be ≥ floor + min height and ≤ y_max
    ridge_y_at = lambda px: _y_on_plane(t1, float(px[0]), float(px[1]))
    apex_ys = [ridge_y_at(cross[1]) for cross in crossings]
    if min(apex_ys) < floor_y + _MIN_PRISM_HEIGHT_M:
        return None
    if max(apex_ys) > y_max + _PLANE_TOL:
        return None

    # Walk the ring; insert the two crossing points to split each crossed edge
    # into two sub-edges. Each sub-edge gets a wall face with top tracking the
    # appropriate oblique. Record which sub-edges are the gable-end pair (they
    # contain the apex point at one of their endpoints).
    crossing_by_edge: dict[int, np.ndarray] = {idx: pt for idx, pt in crossings}
    augmented: list[tuple[float, float, int, bool]] = []  # x, z, plane_id, is_apex
    for i in range(n):
        x, z = coords[i]
        plane_id = boundary_matches[i][0]
        augmented.append((x, z, plane_id, False))
        if i in crossing_by_edge:
            cx, cz = crossing_by_edge[i]
            augmented.append((float(cx), float(cz), plane_id, True))
    m = len(augmented)
    if m < 5:  # 4 corners + 2 crossings minimum
        return None

    out: list[CandidateFace | None] = []
    next_id = first_face_id
    wall_ids = [item[2] for item in augmented]
    floor_edge_keys = tuple(
        tuple(sorted((floor_id, plane_id))) for plane_id in wall_ids
    )
    floor_corners = tuple((x, floor_y, z) for x, z, _pid, _ap in augmented)
    out.append(
        _make_candidate(
            face_id=next_id,
            plane_id=floor_id,
            plane=floor,
            corners=floor_corners,
            edge_keys=floor_edge_keys,
            points=points,
            epsilon=epsilon,
            min_support_points=min_support_points,
            label=_label_for_plane(labels, floor_id),
            domain_polygon=domain,
            prism_id=prism_id,
            plane_support_ratios=plane_support_ratios,
        )
    )
    next_id += 1

    # Per-vertex top y: at apex points, ridge height; otherwise, the side's top.
    top_ys = []
    side_signs = []
    for x, z, _pid, is_apex in augmented:
        side = side_of(np.array([x, z], dtype=float))
        side_signs.append(side)
        if is_apex:
            top_ys.append(_y_on_plane(t1, x, z))  # = T2 at apex; pick T1
        elif side >= 0:
            top_ys.append(_y_on_plane(positive_top, x, z))
        else:
            top_ys.append(_y_on_plane(negative_top, x, z))

    # Top faces: one per oblique. Indices on each side + the two apex vertices.
    pos_indices = [
        i
        for i, (_, _, _, is_apex) in enumerate(augmented)
        if is_apex or side_signs[i] >= 0
    ]
    neg_indices = [
        i
        for i, (_, _, _, is_apex) in enumerate(augmented)
        if is_apex or side_signs[i] <= 0
    ]
    if len(pos_indices) < 3 or len(neg_indices) < 3:
        return None

    def _top_face(side_indices: list[int], side_top_id: int, side_top: Plane):
        # Reverse winding so the face's outward normal points UP/OUT.
        ordered = list(reversed(side_indices))
        corners = tuple(
            (augmented[i][0], top_ys[i], augmented[i][1]) for i in ordered
        )
        edge_keys: list[EdgeKey] = []
        for k in range(len(ordered)):
            a_idx = ordered[k]
            b_idx = ordered[(k + 1) % len(ordered)]
            a_aug = augmented[a_idx]
            b_aug = augmented[b_idx]
            # Edge between two consecutive vertices on this top face: it's
            # either along a wall plane (consecutive on ring) or the ridge
            # (between the two apex points).
            if a_aug[3] and b_aug[3]:
                # Ridge edge: T1 ∩ T2
                other_top_id = (
                    t2_id if side_top_id == t1_id else t1_id
                )
                edge_keys.append(tuple(sorted((side_top_id, other_top_id))))
            else:
                # Walls share plane_id when consecutive on same ring edge
                # (apex split in middle); otherwise wall planes differ.
                wall_id = a_aug[2] if a_aug[2] == b_aug[2] else b_aug[2]
                edge_keys.append(tuple(sorted((side_top_id, wall_id))))
        return _make_candidate(
            face_id=next_id,
            plane_id=side_top_id,
            plane=side_top,
            corners=corners,
            edge_keys=tuple(edge_keys),
            points=points,
            epsilon=epsilon,
            min_support_points=min_support_points,
            label=_label_for_plane(labels, side_top_id),
            domain_polygon=domain,
            prism_id=prism_id,
            plane_support_ratios=plane_support_ratios,
        )

    out.append(_top_face(pos_indices, positive_side_top_id, positive_top))
    next_id += 1
    out.append(_top_face(neg_indices, negative_side_top_id, negative_top))
    next_id += 1

    # Walls: one per (i, i+1) pair in `augmented`. Each is a quadrilateral
    # (or degenerates to a triangle when an endpoint is at the apex).
    for k in range(m):
        a_idx = k
        b_idx = (k + 1) % m
        a = augmented[a_idx]
        b = augmented[b_idx]
        # Two consecutive sub-edges on the same ring edge share plane_id;
        # else, this is a real corner with adjacent walls.
        plane_id = a[2] if a[2] == b[2] else b[2]
        prev_plane_id = augmented[(a_idx - 1) % m][2]
        next_plane_id = augmented[(b_idx + 1) % m][2]
        # Determine top plane for this wall: which side does it sit on?
        # Use midpoint of (a, b) — both apex-points lie on the ridge so the
        # midpoint sits on one side or the other.
        midpoint = np.array(
            [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5], dtype=float
        )
        side = side_of(midpoint)
        wall_top_id = (
            positive_side_top_id if side >= 0 else negative_side_top_id
        )
        corners = (
            (a[0], floor_y, a[1]),
            (a[0], top_ys[a_idx], a[1]),
            (b[0], top_ys[b_idx], b[1]),
            (b[0], floor_y, b[1]),
        )
        edge_keys = (
            tuple(sorted((plane_id, prev_plane_id))),
            tuple(sorted((plane_id, wall_top_id))),
            tuple(sorted((plane_id, next_plane_id))),
            tuple(sorted((plane_id, floor_id))),
        )
        wall = _make_candidate(
            face_id=next_id,
            plane_id=plane_id,
            plane=planes[plane_id],
            corners=corners,
            edge_keys=edge_keys,
            points=points,
            epsilon=epsilon,
            min_support_points=min_support_points,
            label=_label_for_plane(labels, plane_id),
            domain_polygon=domain,
            prism_id=prism_id,
            plane_support_ratios=plane_support_ratios,
        )
        out.append(wall)
        next_id += 1

    if any(face is None for face in out):
        return None
    return [face for face in out if face is not None]


def _enumerate_floor_planes(
    planes: Sequence[Plane],
    *,
    coords: Sequence[tuple[float, float]] | None = None,
) -> list[int]:
    """All downward-facing horizontal planes (b ≤ -cos 5°). `coords` is
    accepted for symmetry with `_enumerate_top_planes` but ignored here —
    upward planes do NOT serve as floors in this revision (a stacked-story
    enumeration produced floating prisms in the viewer, 2026-05-10).
    """
    del coords
    out: list[int] = []
    for idx, plane in enumerate(planes):
        if plane.b <= -_HORIZONTAL_NORMAL_MIN:
            out.append(idx)
    return out


def _enumerate_top_planes(planes: Sequence[Plane]) -> list[int]:
    """All upward-facing planes — flat ceilings AND single obliques."""
    out: list[int] = []
    for idx, plane in enumerate(planes):
        if plane.b >= _OBLIQUE_TOP_B_MIN:
            out.append(idx)
    return out


def _avg_y_at(
    plane: Plane, coords: Sequence[tuple[float, float]]
) -> float:
    if not coords:
        return float("nan")
    ys = [_y_on_plane(plane, x, z) for x, z in coords]
    return float(sum(ys) / len(ys))


def _build_prism_faces(
    *,
    planes: Sequence[Plane],
    floor_id: int,
    floor: Plane,
    floor_y: float,
    top_id: int,
    top: Plane,
    top_ys: tuple[float, ...],
    coords: Sequence[tuple[float, float]],
    boundary_matches: Sequence[tuple[int, tuple[float, float], tuple[float, float]]],
    points: np.ndarray,
    epsilon: float,
    min_support_points: int,
    labels: Sequence[str],
    domain: Polygon,
    first_face_id: int,
    prism_id: int,
    plane_support_ratios: Mapping[int, float] | None = None,
) -> list[CandidateFace] | None:
    out: list[CandidateFace | None] = []
    next_id = first_face_id
    wall_ids = [plane_id for plane_id, _s, _e in boundary_matches]
    floor_edge_keys = tuple(
        tuple(sorted((floor_id, plane_id))) for plane_id in wall_ids
    )
    top_edge_keys = tuple(tuple(sorted((top_id, plane_id))) for plane_id in wall_ids)
    floor_corners = tuple((x, floor_y, z) for x, z in coords)
    top_corners = tuple(
        (coords[i][0], top_ys[i], coords[i][1])
        for i in range(len(coords) - 1, -1, -1)
    )
    floor_face = _make_candidate(
        face_id=next_id,
        plane_id=floor_id,
        plane=floor,
        corners=floor_corners,
        edge_keys=floor_edge_keys,
        points=points,
        epsilon=epsilon,
        min_support_points=min_support_points,
        label=_label_for_plane(labels, floor_id),
        domain_polygon=domain,
        prism_id=prism_id,
        plane_support_ratios=plane_support_ratios,
    )
    out.append(floor_face)
    next_id += 1
    top_face = _make_candidate(
        face_id=next_id,
        plane_id=top_id,
        plane=top,
        corners=top_corners,
        edge_keys=tuple(reversed(top_edge_keys)),
        points=points,
        epsilon=epsilon,
        min_support_points=min_support_points,
        label=_label_for_plane(labels, top_id),
        domain_polygon=domain,
        prism_id=prism_id,
        plane_support_ratios=plane_support_ratios,
    )
    out.append(top_face)
    next_id += 1
    n = len(boundary_matches)
    for idx, (plane_id, start, end) in enumerate(boundary_matches):
        prev_plane_id = boundary_matches[(idx - 1) % n][0]
        next_plane_id = boundary_matches[(idx + 1) % n][0]
        start_top_y = top_ys[idx]
        end_top_y = top_ys[(idx + 1) % n]
        corners = (
            (start[0], floor_y, start[1]),
            (start[0], start_top_y, start[1]),
            (end[0], end_top_y, end[1]),
            (end[0], floor_y, end[1]),
        )
        edge_keys = (
            tuple(sorted((plane_id, prev_plane_id))),
            tuple(sorted((plane_id, top_id))),
            tuple(sorted((plane_id, next_plane_id))),
            tuple(sorted((plane_id, floor_id))),
        )
        wall = _make_candidate(
            face_id=next_id,
            plane_id=plane_id,
            plane=planes[plane_id],
            corners=corners,
            edge_keys=edge_keys,
            points=points,
            epsilon=epsilon,
            min_support_points=min_support_points,
            label=_label_for_plane(labels, plane_id),
            domain_polygon=domain,
            prism_id=prism_id,
            plane_support_ratios=plane_support_ratios,
        )
        out.append(wall)
        next_id += 1
    if any(face is None for face in out):
        return None
    return [face for face in out if face is not None]


def _check_halfspace_generation_budget(
    plane_count: int,
    *,
    max_intersections: int,
) -> None:
    intersection_count = plane_count * (plane_count - 1) * (plane_count - 2) // 6
    if intersection_count <= max_intersections:
        return
    raise ValueError(
        f"halfspace candidate generation would evaluate {intersection_count} "
        f"three-plane intersections for {plane_count} planes, above the "
        f"{max_intersections} cap; use domain decomposition or the prism/"
        "partition candidate path first"
    )


def _find_horizontal_plane(
    planes: Sequence[Plane],
    *,
    upward: bool,
) -> int | None:
    candidates: list[tuple[float, int]] = []
    for idx, plane in enumerate(planes):
        horizontal = abs(float(plane.b))
        if horizontal < _HORIZONTAL_NORMAL_MIN:
            continue
        if upward and plane.b <= 0.0:
            continue
        if not upward and plane.b >= 0.0:
            continue
        candidates.append((horizontal, idx))
    if not candidates:
        return None
    return max(candidates)[1]


def _find_top_plane(planes: Sequence[Plane]) -> int | None:
    """Return the best upward-facing top-plane id, oblique permitted.

    Tier 1 — flat ceiling (`|b| ≥ cos 5°`, `b > 0`).
    Tier 2 — single oblique with `b ≥ sin 10°` (mono-pitch / one half-gable).
    """

    flat = _find_horizontal_plane(planes, upward=True)
    if flat is not None:
        return flat
    oblique: list[tuple[float, int]] = []
    for idx, plane in enumerate(planes):
        if plane.b < _OBLIQUE_TOP_B_MIN:
            continue
        oblique.append((float(plane.b), idx))
    if not oblique:
        return None
    return max(oblique)[1]


def _y_on_plane(plane: Plane, x: float, z: float) -> float:
    return float((plane.d - plane.a * x - plane.c * z) / plane.b)


def _has_competing_oblique_top(planes: Sequence[Plane], top_id: int) -> bool:
    """True if a second upward-facing oblique exists with a roughly opposite
    XZ azimuth — the gable-pair signature, which a single-top extrusion would
    silently mis-model."""

    top = planes[top_id]
    if top.b >= _HORIZONTAL_NORMAL_MIN:
        return False
    top_xz = np.array([top.a, top.c], dtype=float)
    top_norm = float(np.linalg.norm(top_xz))
    if top_norm <= 1e-9:
        return False
    top_xz /= top_norm
    for idx, plane in enumerate(planes):
        if idx == top_id or plane.b < _OBLIQUE_TOP_B_MIN:
            continue
        if plane.b >= _HORIZONTAL_NORMAL_MIN:
            continue
        other_xz = np.array([plane.a, plane.c], dtype=float)
        other_norm = float(np.linalg.norm(other_xz))
        if other_norm <= 1e-9:
            continue
        other_xz /= other_norm
        if float(top_xz @ other_xz) <= _COMPETING_OBLIQUE_AZIMUTH_MIN_COS:
            return True
    return False


def _match_domain_boundary_planes(
    coords: Sequence[tuple[float, float]],
    planes: Sequence[Plane],
) -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
    matches: list[tuple[int, tuple[float, float], tuple[float, float]]] = []
    used: set[int] = set()
    for idx, start in enumerate(coords):
        end = coords[(idx + 1) % len(coords)]
        expected = _outward_normal_for_boundary_segment(start, end)
        if expected is None:
            return []
        best: tuple[float, int] | None = None
        for plane_id, plane in enumerate(planes):
            if plane_id in used or abs(float(plane.b)) > 0.05:
                continue
            normal_2d = np.array([plane.a, plane.c], dtype=float)
            normal_len = float(np.linalg.norm(normal_2d))
            if normal_len <= 1e-9:
                continue
            normal_2d /= normal_len
            if float(normal_2d @ expected) < 0.95:
                continue
            residual = max(
                abs(plane.a * start[0] + plane.c * start[1] - plane.d),
                abs(plane.a * end[0] + plane.c * end[1] - plane.d),
            )
            if residual > 1e-5:
                continue
            if best is None or residual < best[0]:
                best = (residual, plane_id)
        if best is None:
            return []
        used.add(best[1])
        matches.append((best[1], start, end))
    return matches


def _outward_normal_for_boundary_segment(
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray | None:
    dx = float(end[0] - start[0])
    dz = float(end[1] - start[1])
    length = float((dx * dx + dz * dz) ** 0.5)
    if length <= 1e-9:
        return None
    return np.array([dz / length, -dx / length], dtype=float)


def _make_candidate(
    *,
    face_id: int,
    plane_id: int,
    plane: Plane,
    corners: tuple[tuple[float, float, float], ...],
    edge_keys: tuple[EdgeKey, ...],
    points: np.ndarray,
    epsilon: float,
    min_support_points: int,
    label: str,
    domain_polygon: Polygon,
    prism_id: int | None = None,
    plane_support_ratios: Mapping[int, float] | None = None,
) -> CandidateFace | None:
    supporting, support_score = _supporting_points(
        plane,
        points,
        epsilon=epsilon,
    )
    if len(supporting) < min_support_points:
        return None
    area = _polygon_area_3d(corners)
    if area <= 1e-9:
        return None
    # Paper-aligned data-fit when per-story plane-support evidence is
    # available: combine plane_support_ratio (segments-defining-this-plane
    # within the domain) with envelope-height-above-floor so the outer
    # envelope wins ties between equally-supported flat and oblique tops.
    # See user feedback 2026-05-10 ("wayyy too many rooms have a flat
    # ceiling") — without the height term, flat ceilings beat gable
    # obliques on raw evidence count.
    if plane_support_ratios is not None and plane_id in plane_support_ratios:
        avg_y = float(np.mean([float(corner[1]) for corner in corners]))
        height_factor = max(0.0, avg_y - _ENVELOPE_HEIGHT_REFERENCE_M)
        support_score = float(plane_support_ratios[plane_id]) * area * (
            1.0 + _ENVELOPE_HEIGHT_WEIGHT * height_factor
        )
    _origin, u, v = _plane_frame(plane)
    polygon = Polygon(
        [
            (
                float(np.asarray(corner, dtype=float) @ u),
                float(np.asarray(corner, dtype=float) @ v),
            )
            for corner in corners
        ]
    )
    if polygon.is_empty or polygon.area <= 1e-9:
        return None
    return CandidateFace(
        face_id=face_id,
        plane_id=plane_id,
        polygon=polygon,
        edge_keys=edge_keys,
        supporting_points=supporting,
        support_density=float(len(supporting)) / area,
        confidence_label=label,
        corners=corners,
        plane=plane,
        area=area,
        support_score=support_score,
        coverage_polygon=_coverage_polygon_xz(corners, domain_polygon),
        domain_area=float(domain_polygon.area),
        prism_id=prism_id,
    )


def _label_for_plane(labels: Sequence[str], plane_id: int) -> str:
    return str(labels[plane_id]) if plane_id < len(labels) else "plane"


def _coverage_polygon_xz(
    corners: Sequence[Sequence[float]],
    domain_polygon: Polygon,
) -> Polygon:
    if len(corners) < 3 or domain_polygon.is_empty:
        return Polygon()
    projected = Polygon([(float(corner[0]), float(corner[2])) for corner in corners])
    if projected.is_empty or projected.area <= 1e-9:
        return Polygon()
    clipped = projected.intersection(domain_polygon)
    if clipped.is_empty or clipped.area <= 1e-9:
        return Polygon()
    return clipped


@dataclass(frozen=True, slots=True)
class _FaceVertex:
    point: np.ndarray
    active_planes: frozenset[int]
    uv: tuple[float, float]


def _normalize_plane(plane: Plane) -> Plane:
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError(f"degenerate plane normal: {plane}")
    return Plane(
        a=float(plane.a) / norm,
        b=float(plane.b) / norm,
        c=float(plane.c) / norm,
        d=float(plane.d) / norm,
    )


def _as_points(scan_points: np.ndarray) -> np.ndarray:
    points = np.asarray(scan_points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("scan_points must have shape (N, 3)")
    return points


def _face_vertices(
    plane_id: int,
    planes: Sequence[Plane],
    *,
    domain_polygon: Polygon,
    y_min: float,
    y_max: float,
) -> list[_FaceVertex]:
    by_key: dict[tuple[int, int, int], _FaceVertex] = {}
    plane = planes[plane_id]
    origin, u, v = _plane_frame(plane)
    for j, k in combinations((idx for idx in range(len(planes)) if idx != plane_id), 2):
        try:
            point = three_plane_intersection(plane, planes[j], planes[k])
        except np.linalg.LinAlgError:
            continue
        if not _inside_halfspaces(point, planes):
            continue
        if point[1] < y_min - _PLANE_TOL or point[1] > y_max + _PLANE_TOL:
            continue
        if not domain_polygon.covers(Point(float(point[0]), float(point[2]))):
            continue
        active = frozenset(
            idx
            for idx, candidate in enumerate(planes)
            if abs(_signed_residual(candidate, point)) <= 1e-5
        )
        if plane_id not in active:
            continue
        rel = point - origin
        uv = (float(rel @ u), float(rel @ v))
        key = tuple(round(float(coord) * _POINT_KEY_SCALE) for coord in point)
        by_key[key] = _FaceVertex(point=point, active_planes=active, uv=uv)
    return list(by_key.values())


def _inside_halfspaces(point: np.ndarray, planes: Sequence[Plane]) -> bool:
    return all(_signed_residual(plane, point) <= 1e-6 for plane in planes)


def _signed_residual(plane: Plane, point: np.ndarray) -> float:
    return float(plane.a * point[0] + plane.b * point[1] + plane.c * point[2] - plane.d)


def _plane_frame(plane: Plane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    origin = normal * float(plane.d)
    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return origin, u, v


def _order_vertices_on_plane(
    plane: Plane,
    vertices: Sequence[_FaceVertex],
) -> list[_FaceVertex]:
    if len(vertices) < 3:
        return []
    centroid = np.mean(np.array([vertex.uv for vertex in vertices]), axis=0)
    ordered = sorted(
        vertices,
        key=lambda vertex: np.arctan2(
            vertex.uv[1] - centroid[1],
            vertex.uv[0] - centroid[0],
        ),
    )
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    corners = [vertex.point for vertex in ordered]
    if len(corners) >= 3:
        ring_normal = np.zeros(3, dtype=float)
        for idx, point in enumerate(corners):
            nxt = corners[(idx + 1) % len(corners)]
            ring_normal += np.cross(point, nxt)
        if float(ring_normal @ normal) < 0.0:
            ordered.reverse()
    return ordered


def _edge_keys_for_ring(
    plane_id: int,
    ring: Sequence[_FaceVertex],
    planes: Sequence[Plane],
) -> tuple[EdgeKey, ...]:
    keys: list[EdgeKey] = []
    for idx, start in enumerate(ring):
        end = ring[(idx + 1) % len(ring)]
        shared = sorted((start.active_planes & end.active_planes) - {plane_id})
        if not shared:
            shared = [_nearest_shared_plane(plane_id, start.point, end.point, planes)]
        keys.append(tuple(sorted((plane_id, shared[0]))))
    return tuple(keys)


def _nearest_shared_plane(
    plane_id: int,
    start: np.ndarray,
    end: np.ndarray,
    planes: Sequence[Plane],
) -> int:
    midpoint = (start + end) * 0.5
    distances = [
        (abs(_signed_residual(plane, midpoint)), idx)
        for idx, plane in enumerate(planes)
        if idx != plane_id
    ]
    distances.sort()
    return distances[0][1]


def _supporting_points(
    plane: Plane,
    points: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    if len(points) == 0:
        return np.empty((0, 3), dtype=float), 0.0
    residuals = np.abs(
        plane.a * points[:, 0]
        + plane.b * points[:, 1]
        + plane.c * points[:, 2]
        - plane.d
    )
    mask = residuals <= epsilon
    supporting = points[mask]
    weights = np.clip(1.0 - residuals[mask] / epsilon, 0.0, 1.0)
    return supporting, float(weights.sum())


def _polygon_area_3d(corners: Sequence[Sequence[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    pts = [np.asarray(corner, dtype=float) for corner in corners]
    normal_sum = np.zeros(3, dtype=float)
    for idx, point in enumerate(pts):
        normal_sum += np.cross(point, pts[(idx + 1) % len(pts)])
    return 0.5 * float(np.linalg.norm(normal_sum))


def _objective_coefficients(
    candidates: Sequence[CandidateFace],
    weights: SelectionWeights,
) -> np.ndarray:
    max_support = max(
        (candidate.support_score for candidate in candidates),
        default=0.0,
    )
    coeffs: list[float] = []
    for candidate in candidates:
        support = candidate.support_score / max_support if max_support > 0.0 else 0.0
        coverage = (
            candidate.coverage_polygon.area / candidate.domain_area
            if candidate.domain_area > 1e-9
            else 0.0
        )
        coeffs.append(
            1e-6
            - float(weights.data_fit) * support
            - float(weights.coverage) * coverage
        )
    return np.asarray(coeffs, dtype=float)


def _edge_items(
    edge_incidence: Mapping[EdgeKey, Sequence[int]],
    id_to_index: Mapping[int, int],
) -> list[tuple[EdgeKey, tuple[int, ...]]]:
    edge_items = [
        (edge, tuple(face_id for face_id in face_ids if face_id in id_to_index))
        for edge, face_ids in edge_incidence.items()
    ]
    return [(edge, face_ids) for edge, face_ids in edge_items if face_ids]


def _edge_complexity_coefficients(
    candidates: Sequence[CandidateFace],
    edge_items: Sequence[tuple[EdgeKey, tuple[int, ...]]],
    weights: SelectionWeights,
) -> np.ndarray:
    if not edge_items:
        return np.empty((0,), dtype=float)
    plane_by_id = {candidate.plane_id: candidate.plane for candidate in candidates}
    unit = float(weights.complexity) / float(len(edge_items))
    return np.asarray(
        [
            unit if _edge_is_sharp(edge, plane_by_id) else 0.0
            for edge, _face_ids in edge_items
        ],
        dtype=float,
    )


def _edge_is_sharp(
    edge: EdgeKey,
    plane_by_id: Mapping[int, Plane],
) -> bool:
    plane_a = plane_by_id.get(edge[0])
    plane_b = plane_by_id.get(edge[1])
    if plane_a is None or plane_b is None:
        return False
    normal_a = np.array([plane_a.a, plane_a.b, plane_a.c], dtype=float)
    normal_b = np.array([plane_b.a, plane_b.b, plane_b.c], dtype=float)
    norm_a = float(np.linalg.norm(normal_a))
    norm_b = float(np.linalg.norm(normal_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return False
    dot = abs(float((normal_a / norm_a) @ (normal_b / norm_b)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.arccos(dot)) >= DEFAULT.sharp_edge_radians


def _solve_with_scipy_milp(
    candidates: Sequence[CandidateFace],
    coeffs: np.ndarray,
    edge_items: Sequence[tuple[EdgeKey, tuple[int, ...]]],
    edge_coeffs: np.ndarray,
    *,
    time_budget_seconds: float,
) -> tuple[set[int] | None, float, str]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    candidate_ids = [candidate.face_id for candidate in candidates]
    id_to_index = {face_id: index for index, face_id in enumerate(candidate_ids)}
    n_faces = len(candidates)
    n_edges = len(edge_items)

    # Per-prism binding: faces sharing a prism_id move as one. Each prism v
    # gets an auxiliary binary z_v; we enforce x_i = z_v for every i in v, so
    # the solver can pick or drop each prism atomically. See PolyFit's
    # "candidate solid" framing — variants of stacked stories or competing
    # tops are alternatives the manifold edge constraint then arbitrates.
    prism_to_faces: dict[int, list[int]] = {}
    for candidate in candidates:
        if candidate.prism_id is None:
            continue
        prism_to_faces.setdefault(candidate.prism_id, []).append(
            id_to_index[candidate.face_id]
        )
    binding_rows = sum(len(faces) for faces in prism_to_faces.values())
    n_prisms = len(prism_to_faces)

    n_vars = n_faces + n_edges + n_prisms
    objective = np.zeros(n_vars, dtype=float)
    objective[:n_faces] = coeffs
    objective[n_faces : n_faces + n_edges] = edge_coeffs

    n_rows = n_edges + 1 + binding_rows
    constraints = lil_matrix((n_rows, n_vars), dtype=float)
    lb = np.zeros(n_rows, dtype=float)
    ub = np.zeros(n_rows, dtype=float)
    for row, (_edge, face_ids) in enumerate(edge_items):
        for face_id in face_ids:
            constraints[row, id_to_index[face_id]] = 1.0
        constraints[row, n_faces + row] = -2.0
    constraints[n_edges, :n_faces] = 1.0
    lb[n_edges] = 1.0
    ub[n_edges] = float(n_faces)
    binding_row = n_edges + 1
    for prism_index, (_prism_id, face_indices) in enumerate(
        sorted(prism_to_faces.items())
    ):
        z_col = n_faces + n_edges + prism_index
        for face_index in face_indices:
            constraints[binding_row, face_index] = 1.0
            constraints[binding_row, z_col] = -1.0
            binding_row += 1

    result = milp(
        c=objective,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(constraints.tocsr(), lb, ub),
        options={"time_limit": max(float(time_budget_seconds), 1e-6)},
    )
    if not result.success or result.x is None:
        return None, float("inf"), f"milp_{result.status}:{result.message}"
    selected = {
        candidates[index].face_id
        for index, value in enumerate(result.x[:n_faces])
        if value >= 0.5
    }
    return selected, float(result.fun), f"milp_{result.status}"


def _round_and_repair(
    candidates: Sequence[CandidateFace],
    edge_incidence: Mapping[EdgeKey, Sequence[int]],
    coeffs: np.ndarray,
) -> set[int]:
    candidate_by_id = {candidate.face_id: candidate for candidate in candidates}
    id_to_index = {candidate.face_id: idx for idx, candidate in enumerate(candidates)}
    selected = {
        candidate.face_id
        for idx, candidate in enumerate(candidates)
        if coeffs[idx] < 0.0
    }
    if not selected:
        selected.add(candidates[int(np.argmin(coeffs))].face_id)

    for _ in range(64):
        changed = False
        for _edge, face_ids_raw in edge_incidence.items():
            face_ids = [
                face_id for face_id in face_ids_raw if face_id in candidate_by_id
            ]
            count = sum(1 for face_id in face_ids if face_id in selected)
            if count in (0, 2):
                continue
            if count == 1:
                additions = [face_id for face_id in face_ids if face_id not in selected]
                if additions:
                    best = min(
                        additions,
                        key=lambda face_id: coeffs[id_to_index[face_id]],
                    )
                    selected.add(best)
                    changed = True
                    continue
            for face_id in face_ids:
                selected.discard(face_id)
                changed = True
        if not changed:
            break
    return selected


def _objective_for_ids(
    candidates: Sequence[CandidateFace],
    coeffs: np.ndarray,
    edge_items: Sequence[tuple[EdgeKey, tuple[int, ...]]],
    edge_coeffs: np.ndarray,
    selected_ids: set[int],
) -> float:
    id_to_index = {candidate.face_id: idx for idx, candidate in enumerate(candidates)}
    face_objective = float(
        sum(
            coeffs[id_to_index[face_id]]
            for face_id in selected_ids
            if face_id in id_to_index
        )
    )
    edge_objective = 0.0
    for idx, (_edge, face_ids) in enumerate(edge_items):
        if sum(1 for face_id in face_ids if face_id in selected_ids) == 2:
            edge_objective += float(edge_coeffs[idx])
    return face_objective + edge_objective


def _energy_breakdown(
    selected: Sequence[CandidateFace],
    candidates: Sequence[CandidateFace],
    edge_items: Sequence[tuple[EdgeKey, tuple[int, ...]]],
    weights: SelectionWeights,
) -> dict[str, float]:
    selected_support = sum(candidate.support_score for candidate in selected)
    total_support = sum(candidate.support_score for candidate in candidates) or 1.0
    fit = selected_support / total_support
    coverage = _selected_coverage_ratio(selected)
    selected_ids = {candidate.face_id for candidate in selected}
    plane_by_id = {candidate.plane_id: candidate.plane for candidate in candidates}
    selected_sharp_edges = sum(
        1
        for edge, face_ids in edge_items
        if sum(1 for face_id in face_ids if face_id in selected_ids) == 2
        and _edge_is_sharp(edge, plane_by_id)
    )
    complexity = (
        float(weights.complexity) * selected_sharp_edges / float(len(edge_items))
        if edge_items
        else 0.0
    )
    return {
        "data_fit": float(weights.data_fit) * (1.0 - fit),
        "complexity": complexity,
        "coverage": float(weights.coverage) * (1.0 - coverage),
        "coverage_ratio": coverage,
    }


def _selected_coverage_ratio(selected: Sequence[CandidateFace]) -> float:
    polygons = [
        candidate.coverage_polygon
        for candidate in selected
        if not candidate.coverage_polygon.is_empty
        and candidate.coverage_polygon.area > 1e-9
    ]
    domain_area = max((candidate.domain_area for candidate in selected), default=0.0)
    if not polygons or domain_area <= 1e-9:
        return 0.0
    return min(1.0, float(unary_union(polygons).area) / domain_area)
