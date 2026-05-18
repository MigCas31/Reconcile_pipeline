from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .config import LayerPolicyConfig
from .geometry import iter_polygons, stored_row_plane_coeffs
from .models import RelationGraph


def _covering_relations_for_target(target_id: str, graph: RelationGraph) -> list:
    return [
        relation
        for relation in graph.relations_by_target.get(target_id, [])
        if relation.relation_kind == "covering"
        and relation.covered_target_id == target_id
    ]


def _is_through_building_owner_candidate(
    row: dict,
    graph: RelationGraph,
    context,
    cfg: LayerPolicyConfig,
) -> bool:
    if context is None:
        return False
    if str(row.get("provenance_relevance_flag") or "") != "suspect_interior_slice":
        return False
    if not bool(context.crosses_main_extension_boundary):
        return False
    if len(context.source_edge_ids) < cfg.through_building_min_source_edge_count:
        return False
    if len(row.get("piece_part_ids") or []) < cfg.through_building_min_piece_part_count:
        return False
    covering_relations = _covering_relations_for_target(
        str(row.get("target_element_id") or ""),
        graph,
    )
    if not covering_relations:
        return False
    azimuth_deltas = [
        float(relation.azimuth_delta_deg)
        for relation in covering_relations
        if relation.azimuth_delta_deg is not None
    ]
    if not azimuth_deltas:
        return False
    return all(
        cfg.through_building_min_covering_azimuth_delta_deg
        <= azimuth_delta
        <= cfg.through_building_max_covering_azimuth_delta_deg
        for azimuth_delta in azimuth_deltas
    )


def _is_anchor_valid(row: dict, context, cfg: LayerPolicyConfig) -> bool:
    if context is None:
        return False
    has_mirror_partner = bool(row.get("mirror_partner_target_ids"))
    piece_anchor_count = row.get("piece_anchor_chain_count")
    piece_min_residual = row.get("piece_anchor_height_residual_min_m")
    # Mirror-partner exemption: when multiple chains all exceed the ARKit-noise
    # threshold (~10 cm) it indicates a systemic plane-fitting offset rather
    # than bad support — typical of through-building planes where the plane is
    # fitted across stories but the eave chains sit at a single story's height.
    # Require ≥ 2 supporting chains to avoid applying the exemption to lone
    # borderline chains that are more likely to be marginal extensions.
    multi_chain_mirror_pair = (
        has_mirror_partner
        and piece_anchor_count is not None
        and int(piece_anchor_count) >= 2
    )
    if (
        piece_anchor_count is not None
        and int(piece_anchor_count) > 0
        and not multi_chain_mirror_pair
    ):
        if piece_min_residual is None:
            return False
        return float(piece_min_residual) <= cfg.max_piece_anchor_height_residual_m
    # Legacy fallback and mirror-partner path: require good XZ overlap and
    # alignment; skip the height-residual gate for confirmed mirror pairs with
    # multiple supporting chains.
    if float(context.source_edge_overlap_m) < cfg.min_source_edge_overlap_m:
        return False
    if (
        context.source_edge_alignment_deg is not None
        and float(context.source_edge_alignment_deg) > cfg.max_source_edge_alignment_deg
    ):
        return False
    if not multi_chain_mirror_pair and (
        context.source_edge_height_residual_m is not None
        and float(context.source_edge_height_residual_m)
        > cfg.max_source_edge_height_residual_m
    ):
        return False
    return True


def _row_xz_polygon(row: dict) -> Any | None:
    corners = row.get("corners") or []
    shell = [(float(point[0]), float(point[2])) for point in corners if len(point) >= 3]
    if len(shell) < 3:
        return None

    hole_rings: list[list[tuple[float, float]]] = []
    for hole in row.get("holes") or []:
        ring = [(float(point[0]), float(point[2])) for point in hole if len(point) >= 3]
        if len(ring) >= 3:
            hole_rings.append(ring)

    poly = Polygon(shell, holes=hole_rings or None)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _row_source_edge_lines_xz(row: dict) -> list[LineString]:
    lines: list[LineString] = []
    for segment in row.get("source_edge_segments_xz") or []:
        if not isinstance(segment, (list, tuple)) or len(segment) < 2:
            continue
        start = segment[0]
        end = segment[1]
        if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)):
            continue
        if len(start) < 2 or len(end) < 2:
            continue
        try:
            line = LineString(
                [
                    (float(start[0]), float(start[1])),
                    (float(end[0]), float(end[1])),
                ]
            )
        except Exception:
            continue
        if line.is_empty or float(line.length) <= 1e-9:
            continue
        lines.append(line)
    return lines


def _mark_ridge_post_clip_unanchored(row: dict) -> None:
    row["overlay_suppressed"] = True
    row["final_layer"] = False
    row["final_layer_reason"] = "ridge_eave_post_clip_unanchored"


def _requires_ridge_post_clip_anchor_gate(row: dict) -> bool:
    if str(row.get("target_kind") or "") != "ridge_eave_plane_group":
        return False
    if str(row.get("piece_role") or "") != "supported":
        return False
    return str(row.get("xy_conflict_clip_reason") or "") == "final_layer_conflict"


def _ridge_anchor_valid_for_poly(
    row: dict,
    poly: Any | None,
    cfg: LayerPolicyConfig,
) -> bool:
    if not _requires_ridge_post_clip_anchor_gate(row):
        return True
    if poly is None or getattr(poly, "is_empty", True):
        return False

    lines = _row_source_edge_lines_xz(row)
    if not lines:
        # Test fixtures and legacy rows may not include explicit source chain
        # geometry; do not hard-fail those rows.
        return True
    line_union = unary_union(lines)
    if line_union is None or getattr(line_union, "is_empty", True):
        return False

    try:
        min_distance = float(poly.distance(line_union))
    except Exception:
        return False
    if min_distance > cfg.ridge_post_clip_max_source_edge_distance_m:
        return False

    if (
        float(getattr(poly, "area", 0.0) or 0.0)
        < cfg.ridge_post_clip_overlap_check_min_area_m2
    ):
        return True

    try:
        chain_mask = line_union.buffer(
            cfg.ridge_post_clip_source_edge_buffer_m, cap_style=2
        )
    except Exception:
        return False
    if chain_mask is None or getattr(chain_mask, "is_empty", True):
        return False
    try:
        overlap_area = float(poly.intersection(chain_mask).area)
    except Exception:
        return False
    if overlap_area < cfg.ridge_post_clip_min_source_edge_overlap_m2:
        return False
    return True


def _row_mean_y(row: dict) -> float:
    ys = [
        float(point[1])
        for point in (row.get("corners") or [])
        if isinstance(point, (list, tuple)) and len(point) >= 3
    ]
    if not ys:
        return 0.0
    return sum(ys) / float(len(ys))


def _rescue_unanchored_ridge_coverage(
    rows: list[dict], cfg: LayerPolicyConfig
) -> list[dict]:
    rescue_indexes = [
        index
        for index, row in enumerate(rows)
        if str(row.get("target_kind") or "") == "ridge_eave_plane_group"
        and str(row.get("piece_role") or "") == "supported"
        and str(row.get("final_layer_reason") or "")
        == "ridge_eave_post_clip_unanchored"
        and bool(row.get("overlay_suppressed"))
    ]
    if not rescue_indexes:
        return rows

    visible_polys = [
        _row_xz_polygon(row)
        for row in rows
        if bool(row.get("final_layer")) and not bool(row.get("overlay_suppressed"))
    ]
    visible_polys = [
        poly
        for poly in visible_polys
        if poly is not None and not getattr(poly, "is_empty", True)
    ]
    visible_union = unary_union(visible_polys) if visible_polys else None
    visible_y_values = [
        _row_mean_y(row)
        for row in rows
        if bool(row.get("final_layer")) and not bool(row.get("overlay_suppressed"))
    ]
    visible_reference_y = float(median(visible_y_values)) if visible_y_values else 0.0

    extra_rows: list[dict] = []
    rescued_indexes: set[int] = set()
    for row_index in rescue_indexes:
        if row_index in rescued_indexes:
            continue
        row = rows[row_index]
        poly = _row_xz_polygon(row)
        if poly is None:
            continue
        if visible_union is not None:
            try:
                missing_geom = poly.difference(visible_union)
            except Exception:
                missing_geom = poly
        else:
            missing_geom = poly
        missing_area = float(getattr(missing_geom, "area", 0.0) or 0.0)
        if missing_area < cfg.xy_conflict_min_residual_area_m2:
            continue

        peer_indexes = []
        for peer_index in rescue_indexes:
            if peer_index in rescued_indexes:
                continue
            peer_poly = _row_xz_polygon(rows[peer_index])
            if peer_poly is None:
                continue
            try:
                overlap_area = float(peer_poly.intersection(missing_geom).area)
            except Exception:
                overlap_area = 0.0
            if overlap_area >= cfg.xy_conflict_min_residual_area_m2:
                peer_indexes.append((overlap_area, peer_index))
        if not peer_indexes:
            continue

        in_band_peer_indexes = []
        for overlap_area, peer_index in peer_indexes:
            peer_y = _row_mean_y(rows[peer_index])
            if (
                abs(peer_y - visible_reference_y)
                <= cfg.ridge_post_clip_rescue_max_visible_y_delta_m
            ):
                in_band_peer_indexes.append((overlap_area, peer_index))
        if in_band_peer_indexes:
            peer_indexes = in_band_peer_indexes

        def _peer_rank(
            item: tuple[float, int],
        ) -> tuple[float, float, float, float, str]:
            overlap_area, peer_index = item
            peer = rows[peer_index]
            peer_y = _row_mean_y(peer)
            return (
                overlap_area,
                -abs(peer_y - visible_reference_y),
                float(peer.get("support_score") or 0.0),
                float(peer.get("source_part_overlap_fraction") or 0.0),
                str(peer.get("piece_id") or ""),
            )

        _best_overlap, best_index = max(peer_indexes, key=_peer_rank)
        best_row = rows[best_index]
        best_row["overlay_suppressed"] = False
        best_row["final_layer"] = True
        best_row["final_layer_reason"] = "ridge_eave_post_clip_rescued_for_coverage"
        if visible_union is not None:
            extras = _apply_xy_overlap_resolution(
                best_row,
                [visible_union],
                cfg,
                clip_reason="ridge_post_clip_rescue_conflict",
            )
            if extras:
                extra_rows.extend(extras)
        if not bool(best_row.get("overlay_suppressed")):
            best_poly = _row_xz_polygon(best_row)
            if best_poly is not None and not best_poly.is_empty:
                try:
                    visible_union = (
                        best_poly
                        if visible_union is None
                        else unary_union([visible_union, best_poly])
                    )
                except Exception:
                    pass
        for _overlap_area, peer_index in peer_indexes:
            rescued_indexes.add(peer_index)

    if extra_rows:
        rows.extend(extra_rows)
    return rows


def _overlap_fraction(subject: Any, cover: Any) -> float:
    subject_area = float(getattr(subject, "area", 0.0) or 0.0)
    if subject_area <= 1e-9:
        return 0.0
    try:
        overlap = subject.intersection(cover)
    except Exception:
        return 0.0
    if getattr(overlap, "is_empty", True):
        return 0.0
    return float(getattr(overlap, "area", 0.0) or 0.0) / subject_area


def _piece_index_value(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return 0


def _row_plane_coeffs(row: dict) -> tuple[float, float, float, float] | None:
    stored = stored_row_plane_coeffs(row)
    if stored is not None:
        return stored

    corners = row.get("corners") or []
    points = [
        (float(point[0]), float(point[1]), float(point[2]))
        for point in corners
        if len(point) >= 3
    ]
    if len(points) < 3:
        return None

    p0 = points[0]
    for idx in range(1, len(points) - 1):
        p1 = points[idx]
        for jdx in range(idx + 1, len(points)):
            p2 = points[jdx]
            v1 = (
                p1[0] - p0[0],
                p1[1] - p0[1],
                p1[2] - p0[2],
            )
            v2 = (
                p2[0] - p0[0],
                p2[1] - p0[1],
                p2[2] - p0[2],
            )
            nx = v1[1] * v2[2] - v1[2] * v2[1]
            ny = v1[2] * v2[0] - v1[0] * v2[2]
            nz = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(nx) <= 1e-9 and abs(ny) <= 1e-9 and abs(nz) <= 1e-9:
                continue
            if abs(ny) <= 1e-9:
                continue
            d = -(nx * p0[0] + ny * p0[1] + nz * p0[2])
            return nx, ny, nz, d
    return None


def _plane_y_at(
    plane_coeffs: tuple[float, float, float, float] | None,
    x: float,
    z: float,
    fallback_y: float,
) -> float:
    if plane_coeffs is None:
        return fallback_y
    nx, ny, nz, d = plane_coeffs
    if abs(ny) <= 1e-9:
        return fallback_y
    return -(nx * x + nz * z + d) / ny


def _ring_to_3d_loop(
    coords: list[tuple[float, float]],
    plane_coeffs: tuple[float, float, float, float] | None,
    fallback_y: float,
) -> list[list[float]]:
    out: list[list[float]] = []
    for x, z in coords:
        y = _plane_y_at(plane_coeffs, float(x), float(z), fallback_y)
        out.append([round(float(x), 6), round(float(y), 6), round(float(z), 6)])
    return out


def _iter_large_polygons(geom: Any, min_area_m2: float) -> list[Polygon]:
    polys = [
        poly
        for poly in iter_polygons(geom)
        if float(getattr(poly, "area", 0.0) or 0.0) >= min_area_m2
    ]
    polys.sort(key=lambda poly: float(poly.area), reverse=True)
    return polys


def _apply_xy_overlap_resolution(
    row: dict,
    cover_polys: list[Any],
    cfg: LayerPolicyConfig,
    *,
    clip_reason: str,
) -> list[dict]:
    subject_poly = _row_xz_polygon(row)
    if subject_poly is None:
        row["overlay_suppressed"] = True
        return []

    valid_cover_polys = [
        poly
        for poly in cover_polys
        if poly is not None and not getattr(poly, "is_empty", True)
    ]
    if not valid_cover_polys:
        return []

    try:
        cover_union = unary_union(valid_cover_polys)
        residual_geom = subject_poly.difference(cover_union)
    except Exception:
        row["overlay_suppressed"] = True
        return []

    subject_area = max(float(subject_poly.area), 1e-9)
    residual_area = float(getattr(residual_geom, "area", 0.0) or 0.0)
    residual_fraction = residual_area / subject_area
    suppressed_area = max(subject_area - residual_area, 0.0)

    row["xy_conflict_clip_reason"] = clip_reason
    row["xy_conflict_residual_area_m2"] = round(residual_area, 6)
    row["xy_conflict_residual_fraction"] = round(residual_fraction, 6)
    row["xy_conflict_suppressed_area_m2"] = round(suppressed_area, 6)

    if (
        residual_area < cfg.xy_conflict_min_residual_area_m2
        or residual_fraction < cfg.xy_conflict_min_residual_fraction
    ):
        row["overlay_suppressed"] = True
        return []

    residual_polys = _iter_large_polygons(
        residual_geom, cfg.xy_conflict_min_residual_area_m2
    )
    if not residual_polys:
        row["overlay_suppressed"] = True
        return []

    source_piece_id = str(row.get("piece_id") or "")
    if not source_piece_id:
        source_piece_id = str(row.get("target_element_id") or "piece")

    plane_coeffs = _row_plane_coeffs(row)
    first_corner = next(
        (corner for corner in (row.get("corners") or []) if len(corner) >= 3), None
    )
    fallback_y = float(first_corner[1]) if first_corner is not None else 0.0
    base_piece_index = _piece_index_value(row.get("piece_index"))

    original_reason = str(row.get("final_layer_reason") or "")
    clipped_reason_by_base_reason = {
        "same_face_shadowed_by_committed": "same_face_shadow_clipped",
        "committed_union_demoted": "committed_union_demoted_clipped",
    }
    clipped_reason = clipped_reason_by_base_reason.get(original_reason, original_reason)

    extras: list[dict] = []
    for poly_index, poly in enumerate(residual_polys):
        shell = list(poly.exterior.coords)[:-1]
        if len(shell) < 3:
            continue
        holes: list[list[list[float]]] = []
        for ring in poly.interiors:
            ring_coords = list(ring.coords)[:-1]
            if len(ring_coords) < 3:
                continue
            try:
                hole_area = float(abs(Polygon(ring_coords).area))
            except Exception:
                hole_area = 0.0
            if hole_area <= 1e-5:
                continue
            holes.append(_ring_to_3d_loop(ring_coords, plane_coeffs, fallback_y))
        corners = _ring_to_3d_loop(shell, plane_coeffs, fallback_y)

        if poly_index == 0:
            target_row = row
            target_row["piece_id"] = source_piece_id
            target_row["piece_index"] = base_piece_index
        else:
            target_row = dict(row)
            target_row["piece_id"] = f"{source_piece_id}:xy-residual:{poly_index}"
            target_row["piece_index"] = base_piece_index + poly_index
            extras.append(target_row)

        target_row["corners"] = corners
        target_row["holes"] = holes
        target_row["area_xz_m2"] = round(float(poly.area), 6)
        target_row["overlay_suppressed"] = False
        target_row["xy_conflict_clipped"] = True
        target_row["xy_conflict_source_piece_id"] = source_piece_id
        target_row["xy_conflict_residual_index"] = poly_index
        target_row["final_layer_reason"] = clipped_reason

    if not row.get("xy_conflict_clipped"):
        row["overlay_suppressed"] = True
        return []

    return extras


def _final_piece_priority(row: dict) -> tuple[int, float, float]:
    target_kind = str(row.get("target_kind") or "")
    support_score = float(row.get("support_score") or 0.0)
    area = float(row.get("area_xz_m2") or 0.0)
    if target_kind == "committed_oblique":
        return (3, support_score, area)
    if target_kind == "candidate_oblique":
        return (2, support_score, area)
    if target_kind == "ridge_eave_plane_group":
        return (1, support_score, area)
    return (0, support_score, area)


def _planes_coplanar(
    plane_a: tuple[float, float, float, float] | None,
    plane_b: tuple[float, float, float, float] | None,
    *,
    max_angle_deg: float,
) -> bool:
    if plane_a is None or plane_b is None:
        return False
    import math

    a1, b1, c1, _ = plane_a
    a2, b2, c2, _ = plane_b
    n1 = math.sqrt(a1 * a1 + b1 * b1 + c1 * c1)
    n2 = math.sqrt(a2 * a2 + b2 * b2 + c2 * c2)
    if n1 <= 0.0 or n2 <= 0.0:
        return False
    cos_ang = (a1 * a2 + b1 * b2 + c1 * c2) / (n1 * n2)
    return abs(cos_ang) >= math.cos(math.radians(max_angle_deg))


# Promote residual pieces of committed_oblique targets whose building-parts are
# a subset of the target's source_part_ids. These represent the same slope
# continuing onto parts the eave chain already observed — not plane-extent
# overshoot past the footprint. Skip if promoting would create a coplanar
# overlap with an existing final piece (same-target supported siblings,
# coplanar candidate/ceiling-oblique peers, ridge-eave mirrors) — those cases
# already have a valid final owner.
_COMMITTED_RESIDUAL_MIN_AREA_M2 = 0.5
_COMMITTED_RESIDUAL_COPLANAR_MAX_ANGLE_DEG = 10.0
_COMMITTED_RESIDUAL_COPLANAR_OVERLAP_MIN_M2 = 0.25


def _promote_committed_residual_part_contained(
    rows: list[dict], cfg: LayerPolicyConfig
) -> list[dict]:
    candidate_indexes: list[int] = []
    for index, row in enumerate(rows):
        if str(row.get("final_layer_reason") or "") != "committed_residual":
            continue
        if float(row.get("area_xz_m2") or 0.0) < _COMMITTED_RESIDUAL_MIN_AREA_M2:
            continue
        source_part_ids = set(row.get("source_part_ids") or [])
        piece_parts_story = set(row.get("piece_part_ids_story") or [])
        if not source_part_ids or not piece_parts_story:
            continue
        if not piece_parts_story.issubset(source_part_ids):
            continue
        candidate_indexes.append(index)

    if not candidate_indexes:
        return rows

    final_polys: list[tuple[Any, tuple[float, float, float, float] | None]] = []
    for row in rows:
        if not bool(row.get("final_layer")) or bool(row.get("overlay_suppressed")):
            continue
        poly = _row_xz_polygon(row)
        if poly is None or float(getattr(poly, "area", 0.0) or 0.0) < 1e-6:
            continue
        final_polys.append((poly, stored_row_plane_coeffs(row)))

    for index in candidate_indexes:
        row = rows[index]
        candidate_poly = _row_xz_polygon(row)
        if candidate_poly is None or float(candidate_poly.area) < 1e-6:
            continue
        cand_plane = stored_row_plane_coeffs(row)

        max_coplanar_overlap = 0.0
        for peer_poly, peer_plane in final_polys:
            if not _planes_coplanar(
                cand_plane,
                peer_plane,
                max_angle_deg=_COMMITTED_RESIDUAL_COPLANAR_MAX_ANGLE_DEG,
            ):
                continue
            try:
                overlap_area = float(candidate_poly.intersection(peer_poly).area)
            except Exception:
                continue
            if overlap_area > max_coplanar_overlap:
                max_coplanar_overlap = overlap_area

        if max_coplanar_overlap >= _COMMITTED_RESIDUAL_COPLANAR_OVERLAP_MIN_M2:
            continue

        row["final_layer"] = True
        row["final_layer_reason"] = "committed_residual_part_contained"

    return rows


_RIDGE_EAVE_ACCEPTED_OWNER_REASONS = frozenset(
    {
        "ridge_eave_relation_owner",
        "ridge_eave_through_building_owner",
        "ridge_eave_disjoint_extension_owner",
        "committed_relation_owner",
    }
)


def _promote_ridge_eave_mirror_rescued_residuals(rows: list[dict]) -> list[dict]:
    # When one slope of a gable has an accepted final (the scan captured eave
    # evidence on that side), its mirror-partner plane often shows up as a
    # `residual` piece with no direct anchor — the opposite slope simply wasn't
    # scanned as thoroughly. A residual that sits above a shared building-part
    # zone represents the other face of the same physical roof and should
    # render alongside its partner.
    final_supported_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("final_layer")):
            continue
        if str(row.get("piece_role") or "") != "supported":
            continue
        if (
            str(row.get("final_layer_reason") or "")
            not in _RIDGE_EAVE_ACCEPTED_OWNER_REASONS
        ):
            continue
        target_id = str(row.get("target_element_id") or "")
        if target_id:
            final_supported_by_target[target_id].append(row)

    if not final_supported_by_target:
        return rows

    for row in rows:
        if bool(row.get("final_layer")):
            continue
        if str(row.get("final_layer_reason") or "") != "ridge_eave_not_supported":
            continue
        if str(row.get("piece_role") or "") != "residual":
            continue
        if str(row.get("target_kind") or "") != "ridge_eave_plane_group":
            continue

        # Topology guard: do not rescue a residual that has no building-part
        # beneath it — a floating polygon is not a physical roof face.
        residual_zones = set(row.get("piece_part_ids_story") or [])
        if not residual_zones:
            continue

        partner_ids = row.get("mirror_partner_target_ids") or []
        if not partner_ids:
            continue

        rescued = False
        for partner_id in partner_ids:
            for partner_row in final_supported_by_target.get(str(partner_id), ()):
                partner_zones = set(partner_row.get("piece_part_ids_story") or [])
                if partner_zones and residual_zones & partner_zones:
                    rescued = True
                    break
            if rescued:
                break

        if rescued:
            row["final_layer"] = True
            row["final_layer_reason"] = "ridge_eave_mirror_rescued_residual"

    return rows


def _enforce_final_xy_disjointness(
    rows: list[dict], cfg: LayerPolicyConfig
) -> list[dict]:
    final_indexes = [
        index
        for index, row in enumerate(rows)
        if bool(row.get("final_layer")) and not bool(row.get("overlay_suppressed"))
    ]
    if len(final_indexes) < 2:
        return rows

    final_indexes.sort(
        key=lambda index: (
            _final_piece_priority(rows[index]),
            str(
                rows[index].get("piece_id")
                or rows[index].get("target_element_id")
                or ""
            ),
        ),
        reverse=True,
    )

    # Zone-scoped occupied tracking: a newcomer is only clipped against already-
    # accepted rows whose `piece_part_ids_story` overlaps its own. Planes that
    # sit over physically distinct building-parts (e.g. opposite wings) project
    # to overlapping 2D footprints when viewed from above but represent
    # different roofs, so they must not compete for the same XY extent.
    # Rows with no zone information fall back to universal competition.
    accepted: list[tuple[frozenset, Any]] = []
    extra_rows: list[dict] = []
    for row_index in final_indexes:
        row = rows[row_index]
        poly = _row_xz_polygon(row)
        if poly is None:
            row["overlay_suppressed"] = True
            continue

        row_zones: frozenset = frozenset(row.get("piece_part_ids_story") or [])
        relevant_polys = []
        for accepted_zones, accepted_poly in accepted:
            if row_zones and accepted_zones and accepted_zones.isdisjoint(row_zones):
                continue
            relevant_polys.append(accepted_poly)

        if relevant_polys:
            try:
                relevant_union = unary_union(relevant_polys)
            except Exception:
                relevant_union = relevant_polys[0]
            extras = _apply_xy_overlap_resolution(
                row,
                [relevant_union],
                cfg,
                clip_reason="final_layer_conflict",
            )
            if extras:
                for extra in extras:
                    extra_poly = _row_xz_polygon(extra)
                    if not _ridge_anchor_valid_for_poly(extra, extra_poly, cfg):
                        _mark_ridge_post_clip_unanchored(extra)
                extra_rows.extend(extras)
            if bool(row.get("overlay_suppressed")):
                continue
            poly = _row_xz_polygon(row)
            if poly is None:
                row["overlay_suppressed"] = True
                continue
            if not _ridge_anchor_valid_for_poly(row, poly, cfg):
                _mark_ridge_post_clip_unanchored(row)
                continue

        accepted.append((row_zones, poly))

    if extra_rows:
        rows.extend(extra_rows)
    if not bool(cfg.ridge_post_clip_enable_rescue):
        return rows
    return _rescue_unanchored_ridge_coverage(rows, cfg)


def classify_split_piece_rows(
    split_piece_rows: list[dict],
    graph: RelationGraph,
    cfg: LayerPolicyConfig,
) -> list[dict]:
    committed_owner_piece_by_target: dict[str, str] = {}
    committed_supported_by_target: dict[str, list[dict]] = {}
    for row in split_piece_rows:
        if str(row.get("target_kind") or "") != "committed_oblique":
            continue
        if str(row.get("piece_role") or "") != "supported":
            continue
        target_id = str(row.get("target_element_id") or "")
        if not target_id:
            continue
        committed_supported_by_target.setdefault(target_id, []).append(row)
    for target_id, rows in committed_supported_by_target.items():
        owner_row = max(
            rows,
            key=lambda row: (
                float(row.get("area_xz_m2") or 0.0),
                float(row.get("support_score") or 0.0),
                str(row.get("piece_id") or ""),
            ),
        )
        committed_owner_piece_by_target[target_id] = str(
            owner_row.get("piece_id") or ""
        )

    out: list[dict] = []
    for row in split_piece_rows:
        annotated = dict(row)
        target_kind = str(row.get("target_kind") or "")
        piece_role = str(row.get("piece_role") or "")
        target_id = str(row.get("target_element_id") or "")
        context = graph.contexts_by_target.get(target_id)
        through_building_owner = _is_through_building_owner_candidate(
            row,
            graph,
            context,
            cfg,
        )

        if target_kind == "committed_oblique":
            if piece_role != "supported":
                annotated["final_layer"] = False
                annotated["final_layer_reason"] = "committed_residual"
                out.append(annotated)
                continue
            owner_piece_id = committed_owner_piece_by_target.get(target_id, "")
            is_owner = str(row.get("piece_id") or "") == owner_piece_id
            annotated["final_layer"] = bool(is_owner)
            if is_owner:
                annotated["final_layer_reason"] = "committed_relation_owner"
            else:
                source_part_overlap = float(
                    row.get("source_part_overlap_fraction") or 0.0
                )
                if source_part_overlap < cfg.min_cross_part_source_overlap_fraction:
                    annotated["final_layer_reason"] = "committed_cross_part_unowned"
                else:
                    annotated["final_layer_reason"] = "committed_union_demoted"
            out.append(annotated)
            continue

        if target_kind == "candidate_oblique":
            annotated["final_layer"] = False
            annotated["final_layer_reason"] = "candidate_oblique"
            out.append(annotated)
            continue

        if target_kind == "ridge_eave_plane_group":
            if piece_role != "supported":
                annotated["final_layer"] = False
                annotated["final_layer_reason"] = "ridge_eave_not_supported"
                out.append(annotated)
                continue

            if not _is_anchor_valid(row, context, cfg) and not through_building_owner:
                annotated["final_layer"] = False
                annotated["final_layer_reason"] = "ridge_eave_unanchored"
                out.append(annotated)
                continue

            # A supported plane with no rain-capable creator segments and no
            # mirror partner sits entirely under overhead coverage (e.g. a flat
            # roof) with no opposite-slope cross-validation — not a real
            # exterior roof face.
            if float(row.get("creator_rain_area_fraction") or 0.0) == 0.0 and not bool(
                row.get("creator_has_partner")
            ):
                annotated["final_layer"] = False
                annotated["final_layer_reason"] = "ridge_eave_covered_unpaired"
                out.append(annotated)
                continue

            source_part_ids = (
                set(context.source_part_ids) if context is not None else set()
            )
            crossed_part_ids = (
                set(context.crossed_part_ids) if context is not None else set()
            )
            disjoint_source_crossed = (
                bool(source_part_ids)
                and bool(crossed_part_ids)
                and source_part_ids.isdisjoint(crossed_part_ids)
            )

            if (
                context is not None
                and context.crosses_main_extension_boundary
                and not through_building_owner
            ):
                source_part_overlap = float(
                    row.get("source_part_overlap_fraction") or 0.0
                )
                if source_part_overlap < cfg.min_cross_part_source_overlap_fraction:
                    annotated["final_layer"] = False
                    annotated["final_layer_reason"] = "ridge_eave_cross_part_unowned"
                    out.append(annotated)
                    continue

            has_covering_relation = bool(
                _covering_relations_for_target(target_id, graph)
            )
            has_local_competitor = bool(row.get("local_competitor_target_ids"))
            has_mirror_partner = bool(row.get("mirror_partner_target_ids"))

            if (
                has_covering_relation
                and not disjoint_source_crossed
                and not through_building_owner
            ):
                annotated["final_layer"] = False
                annotated["final_layer_reason"] = "ridge_eave_covered_by_relation"
                out.append(annotated)
                continue

            if (
                has_local_competitor
                and not has_mirror_partner
                and not disjoint_source_crossed
            ):
                annotated["final_layer"] = False
                annotated["final_layer_reason"] = "ridge_eave_relation_competitor"
                out.append(annotated)
                continue

            annotated["final_layer"] = True
            annotated["final_layer_reason"] = (
                "ridge_eave_through_building_owner"
                if through_building_owner
                else "ridge_eave_disjoint_extension_owner"
                if disjoint_source_crossed
                else "ridge_eave_relation_owner"
            )
            out.append(annotated)
            continue

        annotated["final_layer"] = False
        annotated["final_layer_reason"] = "unknown_target_kind"
        out.append(annotated)

    out = _apply_same_face_source_suppression(out, graph, cfg)
    out = _apply_same_face_chain_subset_suppression(out, graph)
    out = _promote_committed_residual_part_contained(out, cfg)
    out = _promote_ridge_eave_mirror_rescued_residuals(out)
    out = _enforce_final_xy_disjointness(out, cfg)
    return out


def _element_kind(target_id: str) -> str:
    """Extract the element kind from a target_element_id (<uuid>::<kind>::<id>)."""
    parts = target_id.split("::")
    return parts[1] if len(parts) >= 2 else ""


def _apply_same_face_source_suppression(
    rows: list[dict],
    graph: RelationGraph,
    cfg: LayerPolicyConfig,
) -> list[dict]:
    rows_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_target[str(row.get("target_element_id") or "")].append(row)

    committed_winner_rows_by_target: dict[str, list[dict]] = {}
    for target_id, target_rows in rows_by_target.items():
        committed_rows = [
            row
            for row in target_rows
            if str(row.get("target_kind") or "") == "committed_oblique"
            and str(row.get("piece_role") or "") == "supported"
        ]
        if not committed_rows:
            continue
        winners = [row for row in committed_rows if bool(row.get("final_layer"))]
        committed_winner_rows_by_target[target_id] = winners or committed_rows

    candidate_same_face_to_committed: dict[str, set[str]] = defaultdict(set)
    for relation in graph.relations:
        if relation.relation_kind != "same_face":
            continue
        a_rows = rows_by_target.get(relation.a_target_id, [])
        b_rows = rows_by_target.get(relation.b_target_id, [])
        if not a_rows or not b_rows:
            continue
        kinds_a = {str(row.get("target_kind") or "") for row in a_rows}
        kinds_b = {str(row.get("target_kind") or "") for row in b_rows}
        if "candidate_oblique" in kinds_a and "committed_oblique" in kinds_b:
            candidate_same_face_to_committed[relation.a_target_id].add(
                relation.b_target_id
            )
        if "candidate_oblique" in kinds_b and "committed_oblique" in kinds_a:
            candidate_same_face_to_committed[relation.b_target_id].add(
                relation.a_target_id
            )

    for (
        candidate_target_id,
        committed_target_ids,
    ) in candidate_same_face_to_committed.items():
        candidate_rows = rows_by_target.get(candidate_target_id, [])
        if not candidate_rows:
            continue
        for candidate_row in list(candidate_rows):
            if str(candidate_row.get("target_kind") or "") != "candidate_oblique":
                continue
            if str(candidate_row.get("piece_role") or "") != "supported":
                continue
            candidate_poly = _row_xz_polygon(candidate_row)
            if candidate_poly is None:
                continue
            best_overlap = 0.0
            best_committed_target_id = None
            best_committed_piece_id = None
            cover_polys: list[Any] = []
            candidate_kind = _element_kind(candidate_target_id)
            for committed_target_id in committed_target_ids:
                # A committed piece from a different element kind (e.g. roof-oblique)
                # must not shadow a candidate from a different kind (e.g.
                # ceiling-oblique): they are the exterior and interior faces of the
                # same plane and must coexist in the final layer.
                if _element_kind(committed_target_id) != candidate_kind:
                    continue
                for committed_row in committed_winner_rows_by_target.get(
                    committed_target_id, []
                ):
                    committed_poly = _row_xz_polygon(committed_row)
                    if committed_poly is None:
                        continue
                    cover_polys.append(committed_poly)
                    overlap_fraction = _overlap_fraction(candidate_poly, committed_poly)
                    if overlap_fraction > best_overlap:
                        best_overlap = overlap_fraction
                        best_committed_target_id = committed_target_id
                        best_committed_piece_id = str(
                            committed_row.get("piece_id") or ""
                        )
            if best_overlap >= cfg.same_face_shadow_min_overlap_fraction:
                candidate_row["final_layer"] = False
                candidate_row["final_layer_reason"] = "same_face_shadowed_by_committed"
                candidate_row["same_face_shadow_target_id"] = best_committed_target_id
                candidate_row["same_face_shadow_piece_id"] = best_committed_piece_id
                candidate_row["same_face_shadow_overlap_fraction"] = round(
                    best_overlap, 6
                )
                extra_rows = _apply_xy_overlap_resolution(
                    candidate_row,
                    cover_polys,
                    cfg,
                    clip_reason="same_face_shadow",
                )
                for extra_row in extra_rows:
                    rows.append(extra_row)
                    rows_by_target[candidate_target_id].append(extra_row)

    for target_id, committed_rows in rows_by_target.items():
        committed_supported = [
            row
            for row in committed_rows
            if str(row.get("target_kind") or "") == "committed_oblique"
            and str(row.get("piece_role") or "") == "supported"
        ]
        if len(committed_supported) < 2:
            continue
        owner_rows = [
            row for row in committed_supported if bool(row.get("final_layer"))
        ]
        demoted_rows = [
            row
            for row in committed_supported
            if not bool(row.get("final_layer"))
            and str(row.get("final_layer_reason") or "") == "committed_union_demoted"
        ]
        if not owner_rows or not demoted_rows:
            continue
        owner_polys = [(row, _row_xz_polygon(row)) for row in owner_rows]
        owner_polys = [(row, poly) for row, poly in owner_polys if poly is not None]
        if not owner_polys:
            continue
        owner_cover_polys = [poly for _, poly in owner_polys]
        for demoted_row in list(demoted_rows):
            demoted_poly = _row_xz_polygon(demoted_row)
            if demoted_poly is None:
                continue
            best_overlap = 0.0
            best_owner_piece_id = None
            for owner_row, owner_poly in owner_polys:
                overlap_fraction = _overlap_fraction(demoted_poly, owner_poly)
                if overlap_fraction > best_overlap:
                    best_overlap = overlap_fraction
                    best_owner_piece_id = str(owner_row.get("piece_id") or "")
            if best_overlap >= cfg.same_face_shadow_min_overlap_fraction:
                demoted_row["same_face_shadow_target_id"] = target_id
                demoted_row["same_face_shadow_piece_id"] = best_owner_piece_id
                demoted_row["same_face_shadow_overlap_fraction"] = round(
                    best_overlap, 6
                )
                extra_rows = _apply_xy_overlap_resolution(
                    demoted_row,
                    owner_cover_polys,
                    cfg,
                    clip_reason="committed_union_shadow",
                )
                for extra_row in extra_rows:
                    rows.append(extra_row)
                    rows_by_target[target_id].append(extra_row)

    return rows


def _chain_id_set(row: dict) -> set[str]:
    return {str(chain_id) for chain_id in (row.get("chain_ids") or []) if str(chain_id)}


def _apply_same_face_chain_subset_suppression(
    rows: list[dict],
    graph: RelationGraph,
) -> list[dict]:
    rows_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_target[str(row.get("target_element_id") or "")].append(row)

    owner_kinds = {"ridge_eave_plane_group"}
    owner_rows_by_target: dict[str, list[dict]] = {}
    owner_chain_ids_by_target: dict[str, set[str]] = {}
    owner_area_by_target: dict[str, float] = {}
    owner_best_piece_id_by_target: dict[str, str] = {}
    for target_id, target_rows in rows_by_target.items():
        owner_rows = [
            row
            for row in target_rows
            if str(row.get("target_kind") or "") in owner_kinds
            and str(row.get("piece_role") or "") == "supported"
            and bool(row.get("final_layer"))
        ]
        if not owner_rows:
            continue
        owner_chain_ids = set().union(*(_chain_id_set(row) for row in owner_rows))
        if not owner_chain_ids:
            continue
        owner_rows_by_target[target_id] = owner_rows
        owner_chain_ids_by_target[target_id] = owner_chain_ids
        owner_area_by_target[target_id] = sum(
            float(row.get("area_xz_m2") or 0.0) for row in owner_rows
        )
        best_owner_row = max(
            owner_rows,
            key=lambda row: float(row.get("area_xz_m2") or 0.0),
        )
        owner_best_piece_id_by_target[target_id] = str(
            best_owner_row.get("piece_id") or ""
        )

    candidate_rows_by_target: dict[str, list[dict]] = {}
    candidate_chain_ids_by_target: dict[str, set[str]] = {}
    for target_id, target_rows in rows_by_target.items():
        candidate_rows = [
            row
            for row in target_rows
            if str(row.get("target_kind") or "") == "candidate_oblique"
            and str(row.get("piece_role") or "") == "supported"
        ]
        if not candidate_rows:
            continue
        candidate_chain_ids = set().union(
            *(_chain_id_set(row) for row in candidate_rows)
        )
        if not candidate_chain_ids:
            continue
        candidate_rows_by_target[target_id] = candidate_rows
        candidate_chain_ids_by_target[target_id] = candidate_chain_ids

    covering_owner_target_ids_by_candidate: dict[str, set[str]] = defaultdict(set)
    for relation in graph.relations:
        if relation.relation_kind != "covering":
            continue
        covered_target_id = str(relation.covered_target_id or "")
        covering_target_id = str(relation.covering_target_id or "")
        if (
            covered_target_id in candidate_chain_ids_by_target
            and covering_target_id in owner_chain_ids_by_target
        ):
            covering_owner_target_ids_by_candidate[covered_target_id].add(
                covering_target_id
            )

    suppression_by_candidate_target: dict[str, dict[str, Any]] = {}
    for relation in graph.relations:
        if relation.relation_kind != "same_face":
            continue
        a_target_id = str(relation.a_target_id)
        b_target_id = str(relation.b_target_id)
        if (
            a_target_id in candidate_chain_ids_by_target
            and b_target_id in owner_chain_ids_by_target
        ):
            candidate_target_id = a_target_id
            owner_target_id = b_target_id
        elif (
            b_target_id in candidate_chain_ids_by_target
            and a_target_id in owner_chain_ids_by_target
        ):
            candidate_target_id = b_target_id
            owner_target_id = a_target_id
        else:
            continue

        if not covering_owner_target_ids_by_candidate.get(candidate_target_id):
            continue
        candidate_chain_ids = candidate_chain_ids_by_target[candidate_target_id]
        owner_chain_ids = owner_chain_ids_by_target[owner_target_id]
        if not candidate_chain_ids.issubset(owner_chain_ids):
            continue

        strict_subset = candidate_chain_ids != owner_chain_ids
        if not strict_subset:
            continue
        union_count = len(candidate_chain_ids.union(owner_chain_ids))
        jaccard = (
            len(candidate_chain_ids.intersection(owner_chain_ids)) / float(union_count)
            if union_count
            else 0.0
        )
        score = (
            1 if strict_subset else 0,
            len(owner_chain_ids),
            owner_area_by_target.get(owner_target_id, 0.0),
            jaccard,
        )
        previous = suppression_by_candidate_target.get(candidate_target_id)
        if previous is None or score > previous["score"]:
            suppression_by_candidate_target[candidate_target_id] = {
                "owner_target_id": owner_target_id,
                "owner_chain_ids": owner_chain_ids,
                "strict_subset": strict_subset,
                "jaccard": jaccard,
                "score": score,
            }

    for candidate_target_id, payload in suppression_by_candidate_target.items():
        candidate_chain_ids = candidate_chain_ids_by_target[candidate_target_id]
        owner_target_id = str(payload["owner_target_id"])
        owner_chain_ids = set(payload["owner_chain_ids"])
        strict_subset = bool(payload["strict_subset"])
        jaccard = float(payload["jaccard"])
        for candidate_row in candidate_rows_by_target.get(candidate_target_id, []):
            candidate_row["final_layer"] = False
            candidate_row["overlay_suppressed"] = True
            candidate_row["final_layer_reason"] = "same_face_chain_subset_shadowed"
            candidate_row["same_face_shadow_target_id"] = owner_target_id
            candidate_row["same_face_shadow_piece_id"] = (
                owner_best_piece_id_by_target.get(owner_target_id)
            )
            candidate_row["same_face_shadow_chain_subset"] = True
            candidate_row["same_face_shadow_chain_strict_subset"] = strict_subset
            candidate_row["same_face_shadow_candidate_chain_count"] = len(
                candidate_chain_ids
            )
            candidate_row["same_face_shadow_owner_chain_count"] = len(owner_chain_ids)
            candidate_row["same_face_shadow_chain_jaccard"] = round(jaccard, 6)

    return rows
