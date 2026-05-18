from __future__ import annotations

import math
from typing import Any

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union
from shapely.validation import make_valid

from .graph_utils import stable_hash as _stable_hash
from .math_utils import angle_diff, plane_normal, plane_y_at

LATTICE_SCALE_MM = 1000
AREA_EPS = 0.01
SEAM_MIN_LENGTH_M = 0.15
NEAR_TOUCH_BUFFER_M = 0.35
SEAM_EXTENSION_BUFFER_M = 0.45
SEAM_EXTENSION_PASSES = 3
EAVE_GAP_MAX_AREA_M2 = 3.0
EAVE_GAP_MAX_AVG_WIDTH_M = 0.55
EAVE_GAP_MIN_TOUCH_LENGTH_M = 0.45
EAVE_GAP_MIN_EXTENDED_SEGMENT_INTERSECTION_M = 0.20
RAW_OWNERSHIP_MIN_RATIO = 0.30
DOMAIN_OWNERSHIP_MIN_RATIO = 0.20
SEGMENT_SUPPORT_BUFFER_M = 0.18


def _snap(value: float) -> float:
    return round(float(value) * LATTICE_SCALE_MM) / LATTICE_SCALE_MM


def _iter_polygons(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        if not geom.is_valid:
            try:
                return _iter_polygons(make_valid(geom))
            except Exception:
                try:
                    return _iter_polygons(geom.buffer(0))
                except Exception:
                    return []
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty and p.area > AREA_EPS]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for child in geom.geoms:
            out.extend(_iter_polygons(child))
        return out
    return []


def _iter_lines(geom: Any) -> list[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > SEAM_MIN_LENGTH_M else []
    out: list[LineString] = []
    for child in getattr(geom, "geoms", []) or []:
        out.extend(_iter_lines(child))
    return out


def _safe_intersection(left: Any, right: Any) -> Any:
    try:
        return left.intersection(right)
    except Exception:
        try:
            return make_valid(left).intersection(make_valid(right))
        except Exception:
            try:
                return left.buffer(0).intersection(right.buffer(0))
            except Exception:
                return GeometryCollection()


def _safe_difference(left: Any, right: Any) -> Any:
    try:
        return left.difference(right)
    except Exception:
        try:
            return make_valid(left).difference(make_valid(right))
        except Exception:
            try:
                return left.buffer(0).difference(right.buffer(0))
            except Exception:
                return GeometryCollection()


def _dedup_ring(coords: Any) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for x, z, *_ in coords:
        point = (_snap(float(x)), _snap(float(z)))
        if ring and point == ring[-1]:
            continue
        ring.append(point)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def _snapped_polygon_parts(poly: Polygon) -> list[Polygon]:
    exterior = _dedup_ring(poly.exterior.coords)
    if len(exterior) < 3:
        return []
    holes: list[list[tuple[float, float]]] = []
    for interior in poly.interiors:
        hole = _dedup_ring(interior.coords)
        if len(hole) >= 3:
            holes.append(hole)
    try:
        snapped = Polygon(exterior, holes)
    except Exception:
        return []
    return _iter_polygons(snapped)


def _iter_output_polygons(geom: Any) -> list[Polygon]:
    out: list[Polygon] = []
    for poly in _iter_polygons(geom):
        for snapped in _snapped_polygon_parts(poly):
            exterior_len = len(list(snapped.exterior.coords)) - 1
            if not snapped.interiors and exterior_len <= 4:
                out.append(snapped)
                continue
            for tri in triangulate(snapped):
                clipped = _safe_intersection(tri, snapped)
                out.extend(
                    part
                    for part in _iter_polygons(clipped)
                    if not part.interiors and part.area > AREA_EPS
                )
    return out


def _valid_polygon(poly: Polygon) -> Polygon | None:
    if not poly.is_valid:
        try:
            repaired = make_valid(poly)
        except Exception:
            return None
        parts = _iter_polygons(repaired)
        if not parts:
            return None
        poly = max(parts, key=lambda p: float(p.area))
    if poly.is_empty or poly.area <= AREA_EPS:
        return None
    return poly


def _poly_from_corners(corners: list) -> Polygon | None:
    points: list[tuple[float, float]] = []
    for corner in corners or []:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            continue
        points.append((_snap(float(corner[0])), _snap(float(corner[2]))))
    if len(points) < 3:
        return None
    return _valid_polygon(Polygon(points))


def _linework_for_polygon(poly: Polygon) -> list[LineString]:
    lines = [LineString(poly.exterior.coords)]
    for ring in poly.interiors:
        lines.append(LineString(ring.coords))
    return lines


def _plane_from_surface(surface: dict[str, Any]) -> dict[str, Any] | None:
    cluster = surface.get("cluster") or {}
    ref = cluster.get("refPt") or surface.get("center") or {}
    if (
        cluster.get("avgAzimuth") is not None
        and cluster.get("avgIncl") is not None
        and ref
    ):
        return {
            "n": plane_normal(float(cluster["avgAzimuth"]), float(cluster["avgIncl"])),
            "ref": {
                "x": float(ref["x"]),
                "y": float(ref["y"]),
                "z": float(ref["z"]),
            },
        }

    corners = [
        c
        for c in surface.get("corners") or []
        if isinstance(c, (list, tuple)) and len(c) >= 3
    ]
    if len(corners) < 3:
        return None
    a, b, c = corners[0], corners[1], corners[2]
    ab = (
        float(b[0]) - float(a[0]),
        float(b[1]) - float(a[1]),
        float(b[2]) - float(a[2]),
    )
    ac = (
        float(c[0]) - float(a[0]),
        float(c[1]) - float(a[1]),
        float(c[2]) - float(a[2]),
    )
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-9 or abs(ny) <= 1e-9:
        return None
    nx /= norm
    ny /= norm
    nz /= norm
    if ny < 0.0:
        nx *= -1.0
        ny *= -1.0
        nz *= -1.0
    return {
        "n": {"x": nx, "y": ny, "z": nz},
        "ref": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
    }


def _plane_coefficients(plane: dict[str, Any]) -> tuple[float, float, float] | None:
    n = plane["n"]
    ref = plane["ref"]
    if abs(float(n["y"])) <= 1e-9:
        return None
    ax = float(n["x"]) / float(n["y"])
    bz = float(n["z"]) / float(n["y"])
    c = float(ref["y"]) + ax * float(ref["x"]) + bz * float(ref["z"])
    return ax, bz, c


def _surface_y_at(surface_info: dict[str, Any], x: float, z: float) -> float:
    return _snap(plane_y_at(surface_info["plane"], _snap(x), _snap(z)))


def _lift_polygon(poly: Polygon, surface_info: dict[str, Any]) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    return [
        [
            _snap(float(x)),
            _surface_y_at(surface_info, float(x), float(z)),
            _snap(float(z)),
        ]
        for x, z, *_ in coords
    ]


def _building_footprint_polygon(building_footprint: list | None) -> Polygon | None:
    if not building_footprint or len(building_footprint) < 3:
        return None
    points = []
    for p in building_footprint:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x = float(p[0])
        z = float(p[2] if len(p) >= 3 else p[1])
        points.append((_snap(x), _snap(z)))
    if len(points) < 3:
        return None
    return _valid_polygon(Polygon(points))


def _story_envelope_polygons(bldg: dict) -> dict[int, Any]:
    by_story: dict[int, list[Polygon]] = {}
    for room in bldg.get("rooms") or []:
        poly = _poly_from_corners(room.get("floor_polygon") or [])
        if poly is None:
            continue
        story = int(room.get("story", 0) or 0)
        by_story.setdefault(story, []).append(poly)
    out: dict[int, Any] = {}
    for story, polys in by_story.items():
        parts = _iter_polygons(unary_union(polys))
        if parts:
            out[story] = unary_union(parts)
    return out


def _raw_plane_polygon_by_id(bldg: dict) -> dict[str, Polygon]:
    uuid = str(bldg.get("uuid") or "")
    out: dict[str, Polygon] = {}
    for room_idx, room in enumerate(bldg.get("rooms") or []):
        story = int(room.get("story", 0))
        for plane_idx, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) != 4:
                continue
            poly = _poly_from_corners(corners)
            if poly is None:
                continue
            raw_id = f"{uuid}::ceiling-raw::{story}:{room_idx}:{plane_idx}"
            out[raw_id] = poly
    return out


def _raw_evidence_union(
    surface: dict[str, Any], raw_polys_by_id: dict[str, Polygon]
) -> Polygon | None:
    raw_ids = [
        str(raw_id)
        for raw_id in ((surface.get("cluster") or {}).get("raw_plane_ids") or [])
        if raw_id
    ]
    polys = [raw_polys_by_id[raw_id] for raw_id in raw_ids if raw_id in raw_polys_by_id]
    if not polys:
        return None
    geom = unary_union(polys)
    parts = _iter_polygons(geom)
    if not parts:
        return None
    return unary_union(parts)


def _segment_support_union(surface: dict[str, Any]) -> Any:
    lines: list[LineString] = []
    for seg in (surface.get("cluster") or {}).get("segs") or []:
        a = seg.get("a")
        b = seg.get("b")
        if not (
            isinstance(a, (list, tuple))
            and isinstance(b, (list, tuple))
            and len(a) >= 3
            and len(b) >= 3
        ):
            continue
        line = LineString(
            [
                (_snap(float(a[0])), _snap(float(a[2]))),
                (_snap(float(b[0])), _snap(float(b[2]))),
            ]
        )
        if line.length > 0.05:
            lines.append(line)
    if not lines:
        return None
    return unary_union(lines).buffer(SEGMENT_SUPPORT_BUFFER_M, cap_style=2)


def _surface_infos(
    *,
    bldg: dict,
    oblique_surfaces: list[dict[str, Any]],
    building_footprint: list | None,
) -> list[dict[str, Any]]:
    raw_polys_by_id = _raw_plane_polygon_by_id(bldg)
    footprint = _building_footprint_polygon(building_footprint)
    infos: list[dict[str, Any]] = []
    for index, surface in enumerate(oblique_surfaces):
        base_poly = _poly_from_corners(surface.get("corners") or [])
        plane = _plane_from_surface(surface)
        if base_poly is None or plane is None:
            continue
        raw_poly = _raw_evidence_union(surface, raw_polys_by_id)
        domain_parts = [base_poly]
        if raw_poly is not None and not raw_poly.is_empty:
            domain_parts.append(raw_poly)
        domain = unary_union(domain_parts)
        if footprint is not None:
            domain = domain.intersection(footprint)
        parts = _iter_polygons(domain)
        if not parts:
            continue
        domain = unary_union(parts)
        infos.append(
            {
                "surface_index": index,
                "surface": surface,
                "hypothesis_id": surface.get("roof_hypothesis_id")
                or f"roof-hypothesis:oblique:{index}",
                "story": int(
                    surface.get("dominant_story", surface.get("story", 0)) or 0
                ),
                "base_poly": base_poly,
                "raw_poly": raw_poly,
                "segment_poly": _segment_support_union(surface),
                "domain": domain,
                "plane": plane,
                "plane_coefficients": _plane_coefficients(plane),
                "support_score": float(
                    surface.get("roof_hypothesis_support_score") or 0.0
                ),
            }
        )
    return infos


def _equal_height_line(
    left: dict[str, Any],
    right: dict[str, Any],
    domain: Any,
) -> tuple[LineString, tuple[float, float, float]] | None:
    ci = left.get("plane_coefficients")
    cj = right.get("plane_coefficients")
    if ci is None or cj is None:
        return None
    ai, bi, ci0 = ci
    aj, bj, cj0 = cj
    dx, dz, offset = aj - ai, bj - bi, ci0 - cj0
    norm = math.sqrt(dx * dx + dz * dz)
    if norm <= 1e-9:
        return None

    minx, minz, maxx, maxz = domain.bounds
    span = max(maxx - minx, maxz - minz, 1.0) * 4.0
    cx = (minx + maxx) * 0.5
    cz = (minz + maxz) * 0.5
    t = -(dx * cx + dz * cz + offset) / (norm * norm)
    px = cx + dx * t
    pz = cz + dz * t
    ux = -dz / norm
    uz = dx / norm
    line = LineString(
        [
            (px - ux * span, pz - uz * span),
            (px + ux * span, pz + uz * span),
        ]
    )
    return line, (dx, dz, offset)


def _half_plane_polygon(
    *,
    dx: float,
    dz: float,
    offset: float,
    bounds: tuple[float, float, float, float],
    keep_leq: bool,
) -> Polygon | None:
    norm = math.sqrt(dx * dx + dz * dz)
    if norm <= 1e-9:
        return None
    minx, minz, maxx, maxz = bounds
    span = max(maxx - minx, maxz - minz, 1.0) * 8.0
    cx = (minx + maxx) * 0.5
    cz = (minz + maxz) * 0.5
    t = -(dx * cx + dz * cz + offset) / (norm * norm)
    px = cx + dx * t
    pz = cz + dz * t
    ux = -dz / norm
    uz = dx / norm
    nx = dx / norm
    nz = dz / norm
    if keep_leq:
        nx *= -1.0
        nz *= -1.0
    a = (px - ux * span, pz - uz * span)
    b = (px + ux * span, pz + uz * span)
    return Polygon(
        [
            a,
            b,
            (b[0] + nx * span, b[1] + nz * span),
            (a[0] + nx * span, a[1] + nz * span),
        ]
    )


def _classify_seam_kind(
    left: dict[str, Any],
    right: dict[str, Any],
    line: LineString,
) -> str:
    left_cluster = left["surface"].get("cluster") or {}
    right_cluster = right["surface"].get("cluster") or {}
    ad = angle_diff(
        float(left_cluster.get("avgAzimuth", 0.0)),
        float(right_cluster.get("avgAzimuth", 0.0)),
    )
    if ad >= 140.0:
        return "ridge"
    coords = list(line.coords)
    if len(coords) < 2:
        return "hip"
    mx = (coords[0][0] + coords[-1][0]) * 0.5
    mz = (coords[0][1] + coords[-1][1]) * 0.5
    dx = coords[-1][0] - coords[0][0]
    dz = coords[-1][1] - coords[0][1]
    length = math.sqrt(dx * dx + dz * dz)
    if length <= 1e-9:
        return "hip"
    nx = -dz / length
    nz = dx / length
    seam_y = min(_surface_y_at(left, mx, mz), _surface_y_at(right, mx, mz))
    side_a = min(
        _surface_y_at(left, mx + nx * 0.5, mz + nz * 0.5),
        _surface_y_at(right, mx + nx * 0.5, mz + nz * 0.5),
    )
    side_b = min(
        _surface_y_at(left, mx - nx * 0.5, mz - nz * 0.5),
        _surface_y_at(right, mx - nx * 0.5, mz - nz * 0.5),
    )
    if seam_y < min(side_a, side_b) - 0.05:
        return "valley"
    return "hip"


def _seam_edges(
    infos: list[dict[str, Any]], global_domain: Any | None = None
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left_idx, left in enumerate(infos):
        for right in infos[left_idx + 1 :]:
            if left["story"] != right["story"]:
                continue
            pair_window = _safe_intersection(
                left["domain"].buffer(NEAR_TOUCH_BUFFER_M, join_style=2),
                right["domain"].buffer(NEAR_TOUCH_BUFFER_M, join_style=2),
            )
            if global_domain is not None:
                pair_window = _safe_intersection(pair_window, global_domain)
            if pair_window.is_empty or pair_window.area <= AREA_EPS:
                continue
            equal_line = _equal_height_line(left, right, pair_window)
            if equal_line is None:
                continue
            unclipped_line, coefficients = equal_line
            clipped = unclipped_line.intersection(pair_window.buffer(0.02))
            for line in _iter_lines(clipped):
                kind = _classify_seam_kind(left, right, line)
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                endpoints = []
                for x, z in (coords[0], coords[-1]):
                    y = (_surface_y_at(left, x, z) + _surface_y_at(right, x, z)) * 0.5
                    endpoints.append([_snap(float(x)), _snap(y), _snap(float(z))])
                edge_id = _stable_hash(
                    [
                        left["hypothesis_id"],
                        right["hypothesis_id"],
                        kind,
                        str(endpoints),
                    ],
                    20,
                )
                line_xz = [
                    [_snap(float(coords[0][0])), _snap(float(coords[0][1]))],
                    [_snap(float(coords[-1][0])), _snap(float(coords[-1][1]))],
                ]
                edges.append(
                    {
                        "id": f"roofedge:{edge_id}",
                        "kind": kind,
                        "story": left["story"],
                        "surface_indices": [
                            left["surface_index"],
                            right["surface_index"],
                        ],
                        "roof_hypothesis_ids": [
                            left["hypothesis_id"],
                            right["hypothesis_id"],
                        ],
                        "line": endpoints,
                        "line_xz": line_xz,
                        "equal_height_coefficients": {
                            "dx": round(coefficients[0], 9),
                            "dz": round(coefficients[1], 9),
                            "offset": round(coefficients[2], 9),
                        },
                    }
                )
    return edges


def _info_by_surface_index(infos: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(info["surface_index"]): info for info in infos}


def _infos_domain_union(infos: list[dict[str, Any]]) -> Any:
    parts = _iter_polygons(unary_union([info["domain"] for info in infos]))
    if not parts:
        return None
    return unary_union(parts)


def _cluster_segment_lines(info: dict[str, Any]) -> list[LineString]:
    lines: list[LineString] = []
    for seg in (info["surface"].get("cluster") or {}).get("segs") or []:
        a = seg.get("a")
        b = seg.get("b")
        if not (
            isinstance(a, (list, tuple))
            and isinstance(b, (list, tuple))
            and len(a) >= 3
            and len(b) >= 3
        ):
            continue
        line = LineString(
            [
                (_snap(float(a[0])), _snap(float(a[2]))),
                (_snap(float(b[0])), _snap(float(b[2]))),
            ]
        )
        if line.length > 0.05:
            lines.append(line)
    return lines


def _extended_segment_lines(info: dict[str, Any], envelope: Any) -> list[LineString]:
    if envelope is None or getattr(envelope, "is_empty", True):
        return []
    minx, minz, maxx, maxz = envelope.bounds
    span = max(maxx - minx, maxz - minz, 1.0) * 3.0
    out: list[LineString] = []
    for line in _cluster_segment_lines(info):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        ax, az = coords[0]
        bx, bz = coords[-1]
        dx = bx - ax
        dz = bz - az
        length = math.sqrt(dx * dx + dz * dz)
        if length <= 1e-9:
            continue
        ux = dx / length
        uz = dz / length
        out.append(
            LineString(
                [
                    (ax - ux * span, az - uz * span),
                    (bx + ux * span, bz + uz * span),
                ]
            )
        )
    return out


def _line_overlap_length(lines: list[LineString], geom: Any) -> float:
    total = 0.0
    for line in lines:
        for overlap in _iter_lines(_safe_intersection(line, geom)):
            total += float(overlap.length)
    return total


def _boundary_touch_length(left: Polygon, right: Any) -> float:
    return float(_safe_intersection(left.boundary, right.boundary.buffer(0.04)).length)


def _extend_domains_to_segment_supported_gaps(
    infos: list[dict[str, Any]], story_envelopes: dict[int, Any]
) -> None:
    """Fill only narrow eave gaps justified by extended scanned roof segments."""
    by_story: dict[int, list[dict[str, Any]]] = {}
    for info in infos:
        by_story.setdefault(int(info["story"]), []).append(info)

    for story, story_infos in by_story.items():
        envelope = story_envelopes.get(story)
        if envelope is None or getattr(envelope, "is_empty", True):
            continue
        story_domain = unary_union([info["domain"] for info in story_infos])
        missing = _safe_difference(envelope, story_domain)
        for gap in _iter_polygons(missing):
            if gap.area <= AREA_EPS or gap.area > EAVE_GAP_MAX_AREA_M2:
                continue
            candidates: list[tuple[float, float, dict[str, Any]]] = []
            for info in story_infos:
                touch_len = _boundary_touch_length(gap, info["domain"])
                if touch_len < EAVE_GAP_MIN_TOUCH_LENGTH_M:
                    continue
                avg_width = float(gap.area) / max(touch_len, 1e-9)
                if avg_width > EAVE_GAP_MAX_AVG_WIDTH_M:
                    continue
                extended_lines = _extended_segment_lines(info, envelope)
                segment_len = _line_overlap_length(extended_lines, gap)
                domain_len = _line_overlap_length(
                    extended_lines, info["domain"].buffer(0.04, join_style=2)
                )
                raw_near = (
                    info["raw_poly"] is not None
                    and gap.distance(info["raw_poly"]) <= SEGMENT_SUPPORT_BUFFER_M
                )
                if (
                    segment_len < EAVE_GAP_MIN_EXTENDED_SEGMENT_INTERSECTION_M
                    and not raw_near
                ):
                    continue
                if domain_len < EAVE_GAP_MIN_EXTENDED_SEGMENT_INTERSECTION_M:
                    continue
                candidates.append((touch_len, segment_len, info))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_touch, _best_segment, best = candidates[0]
            if len(candidates) > 1 and candidates[1][0] > best_touch * 0.35:
                continue
            merged = unary_union([best["domain"], gap])
            parts = _iter_polygons(merged)
            if parts:
                best["domain"] = unary_union(parts)


def _extend_domains_to_seams(
    *,
    infos: list[dict[str, Any]],
    seam_edges: list[dict[str, Any]],
    footprint: Polygon | None,
) -> None:
    by_index = _info_by_surface_index(infos)
    for edge in seam_edges:
        surface_indices = edge.get("surface_indices") or []
        if len(surface_indices) != 2:
            continue
        left = by_index.get(int(surface_indices[0]))
        right = by_index.get(int(surface_indices[1]))
        if left is None or right is None:
            continue
        coeffs = edge.get("equal_height_coefficients") or {}
        try:
            dx = float(coeffs["dx"])
            dz = float(coeffs["dz"])
            offset = float(coeffs["offset"])
        except Exception:
            continue

        pair_window = (
            left["domain"]
            .buffer(SEAM_EXTENSION_BUFFER_M, join_style=2)
            .intersection(right["domain"].buffer(SEAM_EXTENSION_BUFFER_M, join_style=2))
        )
        if footprint is not None:
            pair_window = pair_window.intersection(footprint)
        if pair_window.is_empty or pair_window.area <= AREA_EPS:
            continue

        for info, keep_leq in ((left, True), (right, False)):
            half_plane = _half_plane_polygon(
                dx=dx,
                dz=dz,
                offset=offset,
                bounds=pair_window.bounds,
                keep_leq=keep_leq,
            )
            if half_plane is None:
                continue
            side_region = pair_window.intersection(half_plane)
            if side_region.is_empty or side_region.area <= AREA_EPS:
                continue
            if side_region.distance(info["domain"]) > SEAM_EXTENSION_BUFFER_M + 0.02:
                continue
            if not side_region.intersects(info["domain"].buffer(0.03)):
                continue
            merged = unary_union([info["domain"], side_region])
            parts = _iter_polygons(merged)
            if parts:
                info["domain"] = unary_union(parts)


def _line_from_xz(coords: list) -> LineString:
    return LineString(
        [
            (float(coords[0][0]), float(coords[0][1])),
            (float(coords[1][0]), float(coords[1][1])),
        ]
    )


def _cell_touches_edge(poly: Polygon, edge: dict[str, Any]) -> bool:
    coords = edge.get("line_xz") or []
    if len(coords) < 2:
        return False
    try:
        line = _line_from_xz(coords)
    except Exception:
        return False
    return poly.boundary.distance(line) <= 0.03 and poly.buffer(0.03).intersects(line)


def _append_split_parts(out: list[Polygon], geom: Any) -> None:
    for part in _iter_polygons(geom):
        if part.area > AREA_EPS:
            out.append(part)


def _split_atoms_by_polygon(atoms: list[Polygon], splitter: Any) -> list[Polygon]:
    if splitter is None or getattr(splitter, "is_empty", True):
        return atoms
    next_atoms: list[Polygon] = []
    for atom in atoms:
        if atom.is_empty or atom.area <= AREA_EPS:
            continue
        if not atom.intersects(splitter.boundary):
            next_atoms.append(atom)
            continue
        _append_split_parts(next_atoms, _safe_intersection(atom, splitter))
        _append_split_parts(next_atoms, _safe_difference(atom, splitter))
    return next_atoms


def _split_atoms_by_seam(atoms: list[Polygon], edge: dict[str, Any]) -> list[Polygon]:
    coeffs = edge.get("equal_height_coefficients") or {}
    coords = edge.get("line_xz") or []
    if len(coords) < 2:
        return atoms
    try:
        dx = float(coeffs["dx"])
        dz = float(coeffs["dz"])
        offset = float(coeffs["offset"])
        line = _line_from_xz(coords)
    except Exception:
        return atoms
    next_atoms: list[Polygon] = []
    for atom in atoms:
        if atom.is_empty or atom.area <= AREA_EPS:
            continue
        if not atom.buffer(0.02, join_style=2).intersects(line):
            next_atoms.append(atom)
            continue
        half_leq = _half_plane_polygon(
            dx=dx,
            dz=dz,
            offset=offset,
            bounds=atom.bounds,
            keep_leq=True,
        )
        half_geq = _half_plane_polygon(
            dx=dx,
            dz=dz,
            offset=offset,
            bounds=atom.bounds,
            keep_leq=False,
        )
        if half_leq is None or half_geq is None:
            next_atoms.append(atom)
            continue
        _append_split_parts(next_atoms, _safe_intersection(atom, half_leq))
        _append_split_parts(next_atoms, _safe_intersection(atom, half_geq))
    return next_atoms


def _arrangement_atoms(
    infos: list[dict[str, Any]],
    global_domain: Any,
    edges: list[dict[str, Any]],
) -> list[Polygon]:
    atoms = _iter_polygons(global_domain)
    for info in infos:
        atoms = _split_atoms_by_polygon(atoms, info["domain"])
    for edge in edges:
        atoms = _split_atoms_by_seam(atoms, edge)
    return atoms


def _choose_owner(
    cell_poly: Polygon,
    infos: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    cell_area = float(cell_poly.area)
    if cell_area <= AREA_EPS:
        return None
    point = cell_poly.representative_point()
    candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any], str]] = []
    for info in infos:
        domain_overlap = float(cell_poly.intersection(info["domain"]).area) / cell_area
        if domain_overlap < DOMAIN_OWNERSHIP_MIN_RATIO and not info["domain"].covers(
            point
        ):
            continue
        raw_overlap = 0.0
        if info["raw_poly"] is not None:
            raw_overlap = (
                float(cell_poly.intersection(info["raw_poly"]).area) / cell_area
            )
        segment_overlap = 0.0
        if info["segment_poly"] is not None and not info["segment_poly"].is_empty:
            segment_overlap = (
                float(cell_poly.intersection(info["segment_poly"]).area) / cell_area
            )
        touching_edges = [
            edge
            for edge in edges
            if info["surface_index"] in (edge.get("surface_indices") or [])
            and _cell_touches_edge(cell_poly, edge)
        ]
        y = _surface_y_at(info, point.x, point.y)
        evidence = {
            "cell_overlap_ratio": round(domain_overlap, 6),
            "raw_rectangle_overlap_ratio": round(raw_overlap, 6),
            "eave_chain_overlap_ratio": round(segment_overlap, 6),
            "lower_envelope_y_m": round(y, 6),
            "support_score": round(float(info["support_score"]), 6),
            "touching_edge_ids": [edge["id"] for edge in touching_edges],
        }
        raw_bucket = 1.0 if raw_overlap >= RAW_OWNERSHIP_MIN_RATIO else 0.0
        segment_bucket = 1.0 if segment_overlap >= 0.02 else 0.0
        seam_bucket = 1.0 if touching_edges else 0.0
        score = (
            raw_bucket,
            segment_bucket,
            seam_bucket,
            -y,
            float(info["support_score"]),
            domain_overlap,
        )
        kind = touching_edges[0]["kind"] if touching_edges else "surface"
        candidates.append((score, info, evidence, kind))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], candidates[0][3]


def _arrangement_cells(
    infos: list[dict[str, Any]],
    global_domain: Any,
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for raw_poly in _arrangement_atoms(infos, global_domain, edges):
        clipped = _safe_intersection(raw_poly, global_domain)
        for cell_poly in _iter_output_polygons(clipped):
            if cell_poly.area <= AREA_EPS:
                continue
            owner = _choose_owner(cell_poly, infos, edges)
            if owner is None:
                continue
            info, evidence, intersection_kind = owner
            corners = _lift_polygon(cell_poly, info)
            if len(corners) < 3:
                continue
            cell_id = (
                f"roofcell:{_stable_hash([info['hypothesis_id'], str(corners)], 20)}"
            )
            cells.append(
                {
                    "id": cell_id,
                    "owner_roof_hypothesis_id": info["hypothesis_id"],
                    "owner_surface_index": info["surface_index"],
                    "intersection_kind": intersection_kind,
                    "story": info["story"],
                    "area_xz_m2": round(float(cell_poly.area), 6),
                    "points_xz": [
                        [_snap(float(x)), _snap(float(z))]
                        for x, z, *_ in list(cell_poly.exterior.coords)[:-1]
                    ],
                    "corners": corners,
                    "evidence": evidence,
                }
            )
    return cells


def _boundary_edges(global_domain: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for poly in _iter_polygons(global_domain):
        coords = list(poly.exterior.coords)
        for idx in range(len(coords) - 1):
            a = coords[idx]
            b = coords[idx + 1]
            line = LineString([a, b])
            if line.length <= SEAM_MIN_LENGTH_M:
                continue
            line_xz = [
                [_snap(float(a[0])), _snap(float(a[1]))],
                [_snap(float(b[0])), _snap(float(b[1]))],
            ]
            out.append(
                {
                    "id": f"roofedge:{_stable_hash(['boundary', str(line_xz)], 20)}",
                    "kind": "boundary",
                    "surface_indices": [],
                    "roof_hypothesis_ids": [],
                    "line_xz": line_xz,
                }
            )
    return out


def build_roof_arrangement(
    *,
    bldg: dict,
    oblique_surfaces: list[dict[str, Any]],
    building_footprint: list | None = None,
) -> dict[str, Any]:
    """Split selected oblique roof faces into backend-owned arrangement cells.

    The arrangement is built from 2D roof-face domains in XZ. Pairwise
    equal-height lines are noded with face/domain boundaries and polygonized
    into cells, then each cell is lifted onto its owner plane.
    """
    infos = _surface_infos(
        bldg=bldg,
        oblique_surfaces=oblique_surfaces,
        building_footprint=building_footprint,
    )
    if not infos:
        return {
            "cells": [],
            "edges": [],
            "oblique_split": [],
            "metadata": {"surface_count": 0},
        }
    footprint = _building_footprint_polygon(building_footprint)
    story_envelopes = _story_envelope_polygons(bldg)

    global_domain = None
    for pass_index in range(SEAM_EXTENSION_PASSES):
        seam_edges_for_pass = _seam_edges(infos, global_domain)
        if not seam_edges_for_pass:
            break
        before_area = sum(float(info["domain"].area) for info in infos)
        _extend_domains_to_seams(
            infos=infos,
            seam_edges=seam_edges_for_pass,
            footprint=footprint,
        )
        global_domain = _infos_domain_union(infos)
        after_area = sum(float(info["domain"].area) for info in infos)
        if pass_index > 0 and abs(after_area - before_area) <= AREA_EPS:
            break

    _extend_domains_to_segment_supported_gaps(infos, story_envelopes)
    global_domain = _infos_domain_union(infos)
    if global_domain is None:
        return {
            "cells": [],
            "edges": [],
            "oblique_split": [],
            "metadata": {"surface_count": len(infos)},
        }

    seam_edges = _seam_edges(infos, global_domain)
    edges = seam_edges + _boundary_edges(global_domain)
    cells = _arrangement_cells(infos, global_domain, seam_edges)
    oblique_split = [
        {
            "kind": "oblique",
            "surface_kind": "oblique",
            "corners": cell["corners"],
            "source_surface_index": cell["owner_surface_index"],
            "roof_hypothesis_id": cell["owner_roof_hypothesis_id"],
            "arrangement_cell_id": cell["id"],
            "intersection_kind": cell["intersection_kind"],
            "evidence": cell["evidence"],
        }
        for cell in cells
    ]
    return {
        "cells": cells,
        "edges": edges,
        "oblique_split": oblique_split,
        "metadata": {
            "surface_count": len(infos),
            "cell_count": len(cells),
            "seam_edge_count": len(seam_edges),
            "boundary_edge_count": len(edges) - len(seam_edges),
        },
    }
