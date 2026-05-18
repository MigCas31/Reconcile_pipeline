from __future__ import annotations

from collections.abc import Iterable

from scripts import prototype_raw_ceiling_plane_scorer as legacy

from .config import RelationConfig
from .geometry import safe_intersection_area
from .models import PlaneContext, PlaneRelation, RelationGraph


def _height_residual_m(
    a: legacy.TargetPlaneRecord,
    b: legacy.TargetPlaneRecord,
) -> float | None:
    try:
        overlap = a.poly_xz.intersection(b.poly_xz)
    except Exception:
        return None
    if overlap.is_empty:
        return None
    point = overlap.representative_point()
    ay = legacy._plane_y_at(a, float(point.x), float(point.y))
    by = legacy._plane_y_at(b, float(point.x), float(point.y))
    if ay is None or by is None:
        return None
    return abs(float(ay) - float(by))


def _shared_boundary_len_m(
    a: legacy.TargetPlaneRecord,
    b: legacy.TargetPlaneRecord,
) -> float:
    try:
        shared = a.poly_xz.boundary.intersection(b.poly_xz.boundary)
    except Exception:
        return 0.0
    return float(getattr(shared, "length", 0.0) or 0.0)


def _boundary_height_residual_m(
    a: legacy.TargetPlaneRecord,
    b: legacy.TargetPlaneRecord,
) -> float | None:
    try:
        shared = a.poly_xz.boundary.intersection(b.poly_xz.boundary)
    except Exception:
        return None
    if (
        getattr(shared, "is_empty", True)
        or float(getattr(shared, "length", 0.0) or 0.0) <= 1e-9
    ):
        return None
    point = shared.representative_point()
    ay = legacy._plane_y_at(a, float(point.x), float(point.y))
    by = legacy._plane_y_at(b, float(point.x), float(point.y))
    if ay is None or by is None:
        return None
    return abs(float(ay) - float(by))


def _pairwise_targets(
    targets: list[legacy.TargetPlaneRecord],
) -> Iterable[tuple[legacy.TargetPlaneRecord, legacy.TargetPlaneRecord]]:
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            yield a, b


def build_plane_relation_graph(
    targets: list[legacy.TargetPlaneRecord],
    contexts_by_target: dict[str, PlaneContext],
    cfg: RelationConfig,
) -> RelationGraph:
    relations: list[PlaneRelation] = []

    for a, b in _pairwise_targets(targets):
        overlap_m2 = safe_intersection_area(a.poly_xz, b.poly_xz)
        shared_boundary_len_m = _shared_boundary_len_m(a, b)
        azimuth_delta = legacy._wrapped_angle_delta_deg(a.azimuth_deg, b.azimuth_deg)
        inclination_delta = abs(float(a.inclination_deg) - float(b.inclination_deg))
        height_residual = _height_residual_m(a, b)

        context_a = contexts_by_target.get(a.element_id)
        context_b = contexts_by_target.get(b.element_id)
        parts_a = set(context_a.building_part_ids) if context_a is not None else set()
        parts_b = set(context_b.building_part_ids) if context_b is not None else set()
        same_part = bool(parts_a & parts_b)
        same_story = int(a.story) == int(b.story)

        relation_kind = "disjoint"
        covering_target_id = None
        covered_target_id = None

        if overlap_m2 >= cfg.min_relation_overlap_m2:
            if (
                same_story
                and azimuth_delta is not None
                and azimuth_delta <= cfg.same_face_max_azimuth_delta_deg
                and inclination_delta <= cfg.same_face_max_inclination_delta_deg
                and (
                    height_residual is None
                    or height_residual <= cfg.same_face_max_height_residual_m
                )
            ):
                relation_kind = "same_face"
            elif (
                same_story
                and azimuth_delta is not None
                and cfg.mirror_min_azimuth_delta_deg
                <= azimuth_delta
                <= cfg.mirror_max_azimuth_delta_deg
            ):
                relation_kind = "mirror_pair"
            else:
                frac_a = overlap_m2 / max(float(a.poly_xz.area), 1e-9)
                frac_b = overlap_m2 / max(float(b.poly_xz.area), 1e-9)
                if max(frac_a, frac_b) >= cfg.covering_min_overlap_fraction:
                    relation_kind = "covering"
                    if frac_a >= frac_b:
                        covered_target_id = a.element_id
                        covering_target_id = b.element_id
                    else:
                        covered_target_id = b.element_id
                        covering_target_id = a.element_id
                else:
                    relation_kind = "local_competitor"
        elif (
            same_story
            and same_part
            and azimuth_delta is not None
            and cfg.mirror_min_azimuth_delta_deg
            <= azimuth_delta
            <= cfg.mirror_max_azimuth_delta_deg
            and inclination_delta <= cfg.continuous_run_max_inclination_delta_deg
            and shared_boundary_len_m >= cfg.continuous_run_min_shared_boundary_m
        ):
            boundary_height_residual = _boundary_height_residual_m(a, b)
            if (
                boundary_height_residual is not None
                and boundary_height_residual <= cfg.continuous_run_max_height_residual_m
            ):
                relation_kind = "continuous_run_neighbor"
                height_residual = boundary_height_residual

        relations.append(
            PlaneRelation(
                a_target_id=a.element_id,
                b_target_id=b.element_id,
                relation_kind=relation_kind,
                overlap_m2=float(overlap_m2),
                shared_boundary_len_m=float(shared_boundary_len_m),
                azimuth_delta_deg=float(azimuth_delta)
                if azimuth_delta is not None
                else None,
                inclination_delta_deg=float(inclination_delta),
                height_residual_m=float(height_residual)
                if height_residual is not None
                else None,
                same_story=same_story,
                same_part=same_part,
                covering_target_id=covering_target_id,
                covered_target_id=covered_target_id,
            )
        )

    return RelationGraph(contexts_by_target=contexts_by_target, relations=relations)
