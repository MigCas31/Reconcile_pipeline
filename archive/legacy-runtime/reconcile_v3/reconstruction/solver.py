"""Phase B.1 -- BIP solver for roof-envelope reconstruction.

Given a per-building candidate-face set (from Phase A) plus the
scan-derived footprint, pick a subset of candidates that best explains
the scan evidence under geometric constraints.

Formulation (see the plan file for the full derivation):

    variables:
      x_i in {0, 1}    one per candidate face i
      b_k in {0, 1}    one per azimuth bin k  (auxiliary)

    objective:
      max  Σ_i (w_fit * fit_i + w_prior * gbm_prior_i) * x_i
           - w_complexity * Σ_i x_i

    constraints:
      # 1. Coverage of scan footprint
      Σ_i area(face_i n scan_fp) * x_i  >=  θ_cov * area(scan_fp)

      # 2. Pairwise non-overlap (conflicting pairs)
      x_i + x_j <= 1   for (i, j) with overlap > θ_overlap AND
                                   azimuth_diff < θ_az

      # 3. Azimuth coherence
      Σ_k b_k <= K
      x_i <= b_{bin(i)}                  ∀ i

      # 4. Topology (every selected face shares a ridge with another
      #    selected face -- skipped for single-face buildings)
      Σ_{j in neighbors(i)} x_j >= x_i    ∀ i with neighbors

CBC solver via python-mip; runtime capped per building.

Returns a :class:`SolveResult` with solver status, selected face ids,
objective value, runner-up objective (for ambiguity detection), LP gap,
coverage ratio, and a first-pass decision (``auto_accept`` / ``review``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from math import atan2, degrees, sqrt

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from ..audit import note_geom_skip

Point2D = tuple[float, float]
_UNASSIGNED_MIN_AREA_M2 = 0.10
_BIAS_BASE = 0.35
_BIAS_PERPENDICULARITY_GAIN = 0.65
_RUNNER_UP_TIME_FRACTION = 0.3
_AXIS_HINT_MIN_ASPECT = 1.35
_PIECE_MIN_OVERLAP_M2 = 0.05


# ── Hyperparameters ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SolverConfig:
    """All tunable knobs in one place so the B.5 hyperparam search can
    iterate cleanly. Defaults are starting points -- will be overridden by
    the search output promoted to ``hyperparams.json``."""

    w_fit: float = 1.0
    w_prior: float = 0.0  # off by default until D1 calibration lands
    w_complexity: float = 0.05
    theta_cov: float = 0.85
    theta_overlap: float = 0.10  # min fractional overlap to call a pair conflicting
    theta_az_deg: float = 45.0  # coplanar-ish if azimuth diff below this
    k_azimuth_bins: int = 6
    azimuth_bin_width_deg: float = 45.0  # 360/8 bins by default
    time_limit_s: float = 30.0
    # Auto-accept thresholds -- duplicated in triage.py (C) but checked
    # here too so the solver emits a first-pass decision.
    lp_gap_epsilon: float = 0.02
    runner_up_margin: float = 0.05


# ── I/O dataclasses ─────────────────────────────────────────────────────


@dataclass
class SolveResult:
    building_uuid: str
    status: str  # "solved" | "ambiguous" | "infeasible" | "no_candidates"
    selected_face_ids: list[str] = field(default_factory=list)
    objective_value: float = 0.0
    runner_up_objective: float = 0.0
    lp_gap: float | None = None
    coverage_ratio: float = 0.0
    solve_time_ms: int = 0
    decision: str = "review"
    reason: str = ""
    zone_results: list[dict] = field(default_factory=list)
    zone_confidence_summary: list[dict] = field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────────────


def _poly_from_ring(ring: Iterable[Point2D]) -> ShapelyPolygon | None:
    ring_list = list(ring)
    if len(ring_list) < 3:
        return None
    try:
        poly = ShapelyPolygon(ring_list)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        return None
    return poly


def _expand_polygons(geom) -> list[ShapelyPolygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, ShapelyPolygon):
        return [geom]
    return [
        poly
        for poly in getattr(geom, "geoms", [])
        if isinstance(poly, ShapelyPolygon) and not poly.is_empty
    ]


def _largest_polygon(geom) -> ShapelyPolygon | None:
    polys = _expand_polygons(geom)
    if not polys:
        return None
    return max(polys, key=lambda poly: float(poly.area))


def _azimuth_bin(az_deg: float, bin_width_deg: float) -> int:
    """Map an azimuth [0, 360) into a discrete bin index."""
    normalized = az_deg % 360.0
    return int(normalized // bin_width_deg)


def _azimuth_diff_deg(a: float, b: float) -> float:
    """Circular distance in degrees, range [0, 180]."""
    raw = abs(a - b) % 360.0
    return min(raw, 360.0 - raw)


def _axis_diff_deg(a: float, b: float) -> float:
    """Undirected azimuth-axis distance in degrees, range [0, 90]."""
    raw = abs(a - b) % 180.0
    return min(raw, 180.0 - raw)


def _zone_major_axis_metrics(
    footprint_xz: list[Point2D] | None,
) -> tuple[float | None, float | None]:
    poly = _poly_from_ring(footprint_xz or [])
    if poly is None or poly.area <= 0.0:
        return None, None
    try:
        rect = poly.minimum_rotated_rectangle
    except Exception:
        return None, None
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return None, None
    best_edge: tuple[float, float] | None = None
    shortest = None
    for idx in range(len(coords) - 1):
        x0, z0 = coords[idx]
        x1, z1 = coords[idx + 1]
        dx = float(x1) - float(x0)
        dz = float(z1) - float(z0)
        length = sqrt(dx * dx + dz * dz)
        if length <= 1e-6:
            continue
        azimuth = degrees(atan2(dx, dz)) % 360.0
        if best_edge is None or length > best_edge[0]:
            best_edge = (length, azimuth)
        if shortest is None or length < shortest:
            shortest = length
    if best_edge is None or shortest is None:
        return None, None
    return best_edge[1], max(best_edge[0] / max(shortest, 1e-6), 1.0)


def _apply_fallback_axis_bias(
    candidates: list[dict],
    *,
    major_axis_azimuth_deg: float,
) -> list[dict]:
    biased: list[dict] = []
    for candidate in candidates:
        updated = dict(candidate)
        candidate_axis_azimuth_deg = (
            candidate.get("source_gap_major_axis_azimuth_deg")
            if candidate.get("source_gap_major_axis_azimuth_deg") is not None
            else major_axis_azimuth_deg
        )
        axis_diff = _axis_diff_deg(
            float(candidate.get("azimuth_deg") or 0.0),
            float(candidate_axis_azimuth_deg),
        )
        perpendicularity = max(0.0, 1.0 - abs(axis_diff - 90.0) / 90.0)
        bias_factor = _BIAS_BASE + _BIAS_PERPENDICULARITY_GAIN * perpendicularity
        support = float(candidate.get("support_m2") or 0.0)
        updated["support_m2"] = support * bias_factor
        updated["fallback_axis_bias"] = bias_factor
        biased.append(updated)
    return biased


def _pairwise_conflicts(
    candidates: list[dict],
    polys: list[ShapelyPolygon | None],
    *,
    theta_overlap: float,
    theta_az_deg: float,
) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` where the XZ projections overlap significantly AND
    the azimuths are close -- i.e. two planes trying to claim the same
    footprint region. These are the non-overlap constraints.
    """
    conflicts: list[tuple[int, int]] = []
    n = len(candidates)
    for i in range(n):
        pi = polys[i]
        if pi is None:
            continue
        for j in range(i + 1, n):
            pj = polys[j]
            if pj is None:
                continue
            if not pi.intersects(pj):
                continue
            try:
                inter_area = pi.intersection(pj).area
            except Exception as exc:
                note_geom_skip(exc, "solver.pair_intersection")
                continue
            if inter_area <= 0:
                continue
            min_area = min(pi.area, pj.area)
            if min_area <= 0:
                continue
            frac = inter_area / min_area
            if frac < theta_overlap:
                continue
            az_diff = _azimuth_diff_deg(
                float(candidates[i]["azimuth_deg"]),
                float(candidates[j]["azimuth_deg"]),
            )
            if az_diff >= theta_az_deg:
                continue
            conflicts.append((i, j))
    return conflicts


def _candidate_coverage_area(
    poly: ShapelyPolygon | None, scan_fp: ShapelyPolygon | None
) -> float:
    """Area of a candidate's XZ footprint that falls inside the scan
    footprint. Zero if either polygon is missing.
    """
    if poly is None:
        return 0.0
    if scan_fp is None:
        return float(poly.area)
    try:
        return float(poly.intersection(scan_fp).area)
    except Exception:
        return 0.0


# ── Core solve ──────────────────────────────────────────────────────────


def solve_building(
    building_uuid: str,
    candidates: list[dict],
    scan_footprint_xz: list[Point2D] | None,
    *,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Solve the BIP for one building. Candidates are per-face dicts in
    the same shape as ``build_candidate_faces`` emits (see
    ``candidate_faces.CandidateFace``).

    Returns a :class:`SolveResult`. On ``infeasible`` or ``no_candidates``
    the ``selected_face_ids`` list is empty.
    """
    cfg = config or SolverConfig()
    t0 = time.perf_counter()

    result = SolveResult(
        building_uuid=building_uuid,
        status="no_candidates",
        decision="review",
        reason="",
    )

    if not candidates:
        result.reason = "no candidate faces"
        result.solve_time_ms = int((time.perf_counter() - t0) * 1000)
        return result

    # Defer heavy imports until actually solving -- lets dataclasses above
    # be imported without a mip/CBC runtime.
    from mip import (
        BINARY,
        MAXIMIZE,
        Model,
        OptimizationStatus,
        xsum,
    )

    # Geometry prep
    polys = [_poly_from_ring(c.get("footprint_xz") or []) for c in candidates]
    scan_poly = _poly_from_ring(scan_footprint_xz) if scan_footprint_xz else None
    scan_area = float(scan_poly.area) if scan_poly is not None else 0.0
    coverage_areas = [_candidate_coverage_area(p, scan_poly) for p in polys]
    if not any(a > 0 for a in coverage_areas):
        result.status = "infeasible"
        result.reason = "no candidate covers the scan footprint"
        result.solve_time_ms = int((time.perf_counter() - t0) * 1000)
        return result

    # Build model
    m = Model(sense=MAXIMIZE, solver_name="CBC")
    m.verbose = 0

    x = [m.add_var(var_type=BINARY, name=f"x_{i}") for i in range(len(candidates))]
    n_bins = max(2, round(360.0 / max(cfg.azimuth_bin_width_deg, 1e-6)))
    b = [m.add_var(var_type=BINARY, name=f"b_{k}") for k in range(n_bins)]
    face_bin = [
        _azimuth_bin(float(candidates[i]["azimuth_deg"]), cfg.azimuth_bin_width_deg)
        for i in range(len(candidates))
    ]

    # Objective
    fit = []
    for c in candidates:
        area = max(float(c.get("area_m2") or 0.0), 1e-9)
        support = float(c.get("support_m2") or 0.0)
        fit.append(min(support / area, 1.0))
    prior = [float(c.get("gbm_prior") or 0.0) for c in candidates]
    m.objective = xsum(
        (cfg.w_fit * fit[i] + cfg.w_prior * prior[i]) * x[i]
        for i in range(len(candidates))
    ) - cfg.w_complexity * xsum(x[i] for i in range(len(candidates)))

    # 1. Coverage
    if scan_area > 0:
        m += (
            xsum(coverage_areas[i] * x[i] for i in range(len(candidates)))
            >= cfg.theta_cov * scan_area
        )

    # 2. Non-overlap
    conflicts = _pairwise_conflicts(
        candidates,
        polys,
        theta_overlap=cfg.theta_overlap,
        theta_az_deg=cfg.theta_az_deg,
    )
    for i, j in conflicts:
        m += x[i] + x[j] <= 1

    # 3. Azimuth coherence
    m += xsum(b[k] for k in range(n_bins)) <= cfg.k_azimuth_bins
    for i in range(len(candidates)):
        m += x[i] <= b[face_bin[i]]

    # 4. Topology -- a selected face that has *any* in-set neighbours must
    #    have at least one of them selected too. Candidates with no in-set
    #    neighbours are left free: they're the only face in their connected
    #    component (single-pitch building, or a Phase A segment whose slices
    #    aren't wired to each other), and forcing them to 0 makes those
    #    buildings spuriously infeasible.
    id_to_idx = {c["id"]: i for i, c in enumerate(candidates)}
    for i, c in enumerate(candidates):
        nbrs = [id_to_idx[n] for n in (c.get("neighbors") or []) if n in id_to_idx]
        if nbrs:
            m += xsum(x[j] for j in nbrs) >= x[i]

    # Solve
    status = m.optimize(max_seconds=cfg.time_limit_s)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if status == OptimizationStatus.INFEASIBLE:
        result.status = "infeasible"
        result.reason = "LP infeasible under current hyperparameters"
        result.solve_time_ms = elapsed_ms
        return result
    if status not in (
        OptimizationStatus.OPTIMAL,
        OptimizationStatus.FEASIBLE,
    ):
        result.status = "infeasible"
        result.reason = f"solver returned {status.name}"
        result.solve_time_ms = elapsed_ms
        return result

    selected = [i for i in range(len(candidates)) if x[i].x and x[i].x > 0.5]
    objective_value = float(m.objective_value or 0.0)

    # Runner-up: re-solve with a "forbid this exact selection" constraint
    # so we can see how close the next-best is.
    runner_up_obj = 0.0
    if selected:
        m += (
            xsum(x[i] for i in selected)
            - xsum(x[i] for i in range(len(candidates)) if i not in selected)
            <= len(selected) - 1
        )
        # Reduced budget for the second solve -- ambiguity check need not be
        # as thorough as the primary.
        status2 = m.optimize(
            max_seconds=max(5.0, cfg.time_limit_s * _RUNNER_UP_TIME_FRACTION)
        )
        if status2 in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            runner_up_obj = float(m.objective_value or 0.0)

    # Coverage ratio on the selected set
    if scan_area > 0:
        selected_coverage = sum(coverage_areas[i] for i in selected)
        coverage_ratio = min(selected_coverage / scan_area, 1.0)
    else:
        coverage_ratio = 1.0 if selected else 0.0

    # LP gap proxy: (integer - LP relaxation) / LP relaxation. python-mip
    # exposes ``gap`` on the Model after optimize, but only in newer
    # releases; fall back to None if absent.
    lp_gap: float | None
    try:
        lp_gap = float(m.gap)
    except Exception:
        lp_gap = None

    result.status = "solved"
    result.selected_face_ids = [candidates[i]["id"] for i in selected]
    result.objective_value = objective_value
    result.runner_up_objective = runner_up_obj
    result.lp_gap = lp_gap
    result.coverage_ratio = coverage_ratio
    result.solve_time_ms = elapsed_ms

    # First-pass decision. Ambiguity = runner-up within margin of optimum.
    unambiguous = True
    if objective_value > 0 and runner_up_obj > 0:
        margin = (objective_value - runner_up_obj) / abs(objective_value)
        if margin < cfg.runner_up_margin:
            unambiguous = False
    gap_tight = lp_gap is None or lp_gap <= cfg.lp_gap_epsilon
    if unambiguous and gap_tight and coverage_ratio >= cfg.theta_cov:
        result.decision = "auto_accept"
    else:
        result.status = "ambiguous" if not unambiguous else result.status
        result.decision = "review"
        if not unambiguous:
            result.reason = (
                f"runner-up within {cfg.runner_up_margin:.2f} "
                f"(obj={objective_value:.3f}, runner_up={runner_up_obj:.3f})"
            )
        elif not gap_tight:
            result.reason = f"lp_gap={lp_gap:.3f} > {cfg.lp_gap_epsilon}"
        elif coverage_ratio < cfg.theta_cov:
            result.reason = (
                f"coverage_ratio={coverage_ratio:.3f} < theta_cov={cfg.theta_cov}"
            )

    return result


def solve_building_with_zones(
    building_uuid: str,
    candidates: list[dict],
    scan_footprint_xz: list[Point2D] | None,
    *,
    zones: list[dict] | None = None,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Solve a building either globally or one zone at a time.

    When ``zones`` are present and candidates carry ``zone_id`` tags, each
    zone becomes its own local coverage problem and the final result is the
    aggregate of the per-zone solves. This keeps detached or weakly
    connected roof parts from competing for one whole-building coverage
    target.
    """

    if not zones:
        return solve_building(
            building_uuid, candidates, scan_footprint_xz, config=config
        )

    zone_lookup: dict[str, dict] = {}
    zone_area_by_id: dict[str, float] = {}
    for zone in zones:
        zone_id = str(zone.get("id") or "")
        if not zone_id:
            continue
        ring = zone.get("footprint_xz") or []
        poly = _poly_from_ring(
            [
                (float(pt[0]), float(pt[1]))
                for pt in ring
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
        )
        if poly is None:
            continue
        zone_lookup[zone_id] = zone
        zone_area_by_id[zone_id] = float(poly.area)

    if not zone_lookup:
        return solve_building(
            building_uuid, candidates, scan_footprint_xz, config=config
        )

    grouped_candidates: dict[str, list[dict]] = {}
    unassigned_candidates: list[dict] = []
    for candidate in candidates:
        zone_id = candidate.get("zone_id")
        if isinstance(zone_id, str) and zone_id in zone_lookup:
            grouped_candidates.setdefault(zone_id, []).append(candidate)
        else:
            unassigned_candidates.append(candidate)

    zone_results: list[dict] = []
    zone_confidence_summary: list[dict] = []
    selected_face_ids: list[str] = []
    objective_total = 0.0
    runner_up_total = 0.0
    weighted_coverage_num = 0.0
    weighted_coverage_den = 0.0
    solve_time_total = 0
    scan_poly = _poly_from_ring(scan_footprint_xz) if scan_footprint_xz else None
    zone_union = (
        unary_union(
            [
                _poly_from_ring(zone.get("footprint_xz") or [])
                for zone in zone_lookup.values()
                if _poly_from_ring(zone.get("footprint_xz") or []) is not None
            ]
        )
        if zone_lookup
        else None
    )

    for zone_id, zone in zone_lookup.items():
        zone_candidates = grouped_candidates.get(zone_id) or []
        zone_ring = zone.get("footprint_xz") or scan_footprint_xz
        zone_fp = None
        if zone_ring:
            zone_fp = [(float(pt[0]), float(pt[1])) for pt in zone_ring]
        zone_config = config or SolverConfig()
        zone_candidates_for_solve = zone_candidates
        if zone.get("fallback_kind") and not (zone.get("seed_subpart_ids") or []):
            gap_axis_candidates = [
                float(candidate["source_gap_major_axis_azimuth_deg"])
                for candidate in zone_candidates
                if candidate.get("source_gap_major_axis_azimuth_deg") is not None
            ]
            major_axis_azimuth_deg = (
                gap_axis_candidates[0] if gap_axis_candidates else None
            )
            aspect_ratio = None
            if major_axis_azimuth_deg is None:
                major_axis_azimuth_deg, aspect_ratio = _zone_major_axis_metrics(zone_fp)
            if major_axis_azimuth_deg is not None and (
                gap_axis_candidates
                or (aspect_ratio is not None and aspect_ratio >= _AXIS_HINT_MIN_ASPECT)
            ):
                zone_config = replace(
                    zone_config,
                    k_azimuth_bins=min(zone_config.k_azimuth_bins, 2),
                )
                zone_candidates_for_solve = _apply_fallback_axis_bias(
                    zone_candidates,
                    major_axis_azimuth_deg=major_axis_azimuth_deg,
                )
        zone_result = solve_building(
            building_uuid,
            zone_candidates_for_solve,
            zone_fp,
            config=zone_config,
        )
        zone_result_dict = asdict(zone_result)
        zone_result_dict["zone_id"] = zone_id
        zone_result_dict["part_id"] = zone.get("part_id")
        zone_result_dict["zone_source"] = zone.get("source")
        zone_result_dict["zone_confidence"] = zone.get("confidence")
        zone_result_dict["zone_fallback_kind"] = zone.get("fallback_kind")
        if zone.get("fallback_kind") and len(zone_result.selected_face_ids) > 8:
            zone_result_dict["status"] = "ambiguous"
            zone_result_dict["decision"] = "review"
            reason = str(zone_result_dict.get("reason") or "")
            extra = "fallback_zone_selected_many_slices"
            zone_result_dict["reason"] = f"{reason}; {extra}".strip("; ")
        zone_results.append(zone_result_dict)
        zone_confidence_summary.append(
            {
                "zone_id": zone_id,
                "confidence": zone.get("confidence"),
                "fallback_kind": zone.get("fallback_kind"),
            }
        )
        selected_face_ids.extend(zone_result.selected_face_ids)
        objective_total += zone_result.objective_value
        runner_up_total += zone_result.runner_up_objective
        solve_time_total += zone_result.solve_time_ms
        zone_area = zone_area_by_id.get(zone_id, 0.0)
        weighted_coverage_num += zone_result.coverage_ratio * zone_area
        weighted_coverage_den += zone_area

    unassigned_remainder = None
    if scan_poly is not None:
        try:
            unassigned_remainder = (
                scan_poly.difference(zone_union)
                if zone_union is not None
                else scan_poly
            )
        except Exception:
            unassigned_remainder = scan_poly
    remainder_pieces = [
        poly
        for poly in _expand_polygons(unassigned_remainder)
        if float(poly.area) >= _UNASSIGNED_MIN_AREA_M2
    ]

    if unassigned_candidates and remainder_pieces:
        for piece_index, remainder_piece in enumerate(
            sorted(remainder_pieces, key=lambda poly: (-float(poly.area), poly.bounds))
        ):
            remainder_ring = [
                (float(x), float(z))
                for x, z in list(remainder_piece.exterior.coords)[:-1]
            ]
            piece_candidates: list[dict] = []
            for candidate in unassigned_candidates:
                candidate_poly = _poly_from_ring(candidate.get("footprint_xz") or [])
                if candidate_poly is None:
                    continue
                try:
                    overlap_area = float(
                        candidate_poly.intersection(remainder_piece).area
                    )
                except Exception:
                    overlap_area = 0.0
                if overlap_area < _PIECE_MIN_OVERLAP_M2:
                    continue
                piece_candidates.append(candidate)
            unassigned_zone_id = (
                "__unassigned__"
                if len(remainder_pieces) == 1
                else f"__unassigned__:{piece_index}"
            )
            zone_result = solve_building(
                building_uuid,
                piece_candidates,
                remainder_ring,
                config=config,
            )
            zone_result_dict = asdict(zone_result)
            zone_result_dict["zone_id"] = unassigned_zone_id
            zone_result_dict["part_id"] = None
            zone_result_dict["zone_source"] = "unassigned"
            zone_result_dict["zone_confidence"] = None
            zone_result_dict["zone_fallback_kind"] = None
            zone_results.append(zone_result_dict)
            zone_confidence_summary.append(
                {
                    "zone_id": unassigned_zone_id,
                    "confidence": None,
                    "fallback_kind": None,
                }
            )
            selected_face_ids.extend(zone_result.selected_face_ids)
            objective_total += zone_result.objective_value
            runner_up_total += zone_result.runner_up_objective
            solve_time_total += zone_result.solve_time_ms

    result = SolveResult(
        building_uuid=building_uuid,
        status="no_candidates",
        zone_results=zone_results,
        zone_confidence_summary=zone_confidence_summary,
    )
    result.selected_face_ids = selected_face_ids
    result.objective_value = objective_total
    result.runner_up_objective = runner_up_total
    result.solve_time_ms = solve_time_total
    if weighted_coverage_den > 0.0:
        result.coverage_ratio = min(weighted_coverage_num / weighted_coverage_den, 1.0)
    elif zone_results:
        result.coverage_ratio = min(
            sum(float(row.get("coverage_ratio", 0.0) or 0.0) for row in zone_results)
            / len(zone_results),
            1.0,
        )

    statuses = [str(row.get("status") or "") for row in zone_results]
    decisions = [str(row.get("decision") or "") for row in zone_results]
    low_confidence_zone_ids = [
        str(row.get("zone_id"))
        for row in zone_results
        if row.get("zone_confidence") is not None
        and float(row.get("zone_confidence")) < 0.4
    ]
    reasons = [
        f"{row.get('zone_id')}: {row.get('reason')}"
        for row in zone_results
        if row.get("reason")
    ]
    if statuses and all(status == "no_candidates" for status in statuses):
        result.status = "no_candidates"
        result.decision = "review"
        result.reason = "; ".join(reasons) or "no candidate faces"
        return result
    if any(status == "infeasible" for status in statuses):
        result.status = "infeasible"
    elif any(status == "ambiguous" for status in statuses):
        result.status = "ambiguous"
    elif any(status == "no_candidates" for status in statuses):
        result.status = "ambiguous"
    else:
        result.status = "solved"
    if low_confidence_zone_ids:
        reasons.append(
            "low_confidence_zones=" + ",".join(sorted(low_confidence_zone_ids))
        )
    result.decision = (
        "auto_accept"
        if decisions
        and all(decision == "auto_accept" for decision in decisions)
        and not low_confidence_zone_ids
        else "review"
    )
    result.reason = "; ".join(reasons)
    return result


# ── Batch entrypoint ────────────────────────────────────────────────────


def solve_corpus(
    buildings: list[dict],
    scan_footprints: dict[str, list[Point2D] | None] | None = None,
    *,
    config: SolverConfig | None = None,
) -> list[SolveResult]:
    """Run :func:`solve_building` over a list of per-building records
    (same shape as ``candidates.json`` -- each has ``building_uuid`` and
    ``candidates``). ``scan_footprints`` maps uuid -> XZ ring.
    """
    scan_footprints = scan_footprints or {}
    results: list[SolveResult] = []
    for b in buildings:
        uuid = b["building_uuid"]
        results.append(
            solve_building_with_zones(
                uuid,
                b.get("candidates") or [],
                scan_footprints.get(uuid),
                zones=b.get("zones") or [],
                config=config,
            )
        )
    return results
