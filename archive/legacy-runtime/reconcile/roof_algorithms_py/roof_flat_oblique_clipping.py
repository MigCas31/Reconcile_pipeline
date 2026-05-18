from __future__ import annotations

import math
from typing import Any, Literal

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import split
from shapely.validation import make_valid

from .math_utils import plane_normal

AREA_EPS = 0.01
FLAT_OBLIQUE_INTERSECTION_TOL_M = 0.03
TOP_FLAT_MIN_RESIDUAL_AREA_M2 = 0.5
TOP_FLAT_MIN_RESIDUAL_RATIO = 0.10
LATTICE_SCALE_MM = 1000


def _snap(value: float) -> float:
    return round(float(value) * LATTICE_SCALE_MM) / LATTICE_SCALE_MM


def _surface_story(surface: dict[str, Any]) -> int:
    return int(surface.get("dominant_story", surface.get("story", 0)) or 0)


def _source_kind(surface: dict[str, Any]) -> str:
    return str(surface.get("kind") or surface.get("source_kind") or "")


def _can_clip_against_oblique(flat: dict[str, Any], oblique: dict[str, Any]) -> bool:
    flat_story = _surface_story(flat)
    oblique_story = _surface_story(oblique)
    if flat_story == oblique_story:
        return True
    return _source_kind(flat) == "top" and oblique_story == flat_story + 1


def _flat_y(surface: dict[str, Any]) -> float | None:
    y = surface.get("y")
    if isinstance(y, (int, float)):
        return float(y)
    ys = [
        float(corner[1])
        for corner in (surface.get("corners") or [])
        if isinstance(corner, (list, tuple)) and len(corner) >= 3
    ]
    if not ys:
        return None
    return sum(ys) / len(ys)


def _surface_poly(surface: dict[str, Any]) -> Polygon | None:
    points: list[tuple[float, float]] = []
    for corner in surface.get("corners") or []:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            continue
        points.append((_snap(float(corner[0])), _snap(float(corner[2]))))
    if len(points) < 3:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
        except Exception:
            return None
    poly = _largest_polygon(poly)
    if poly is None or poly.is_empty or poly.area <= AREA_EPS:
        return None
    return poly


def _largest_polygon(geom: Any) -> Polygon | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(
            (poly for poly in geom.geoms if not poly.is_empty),
            key=lambda poly: float(poly.area),
            default=None,
        )
    if isinstance(geom, GeometryCollection):
        polys = [
            poly
            for poly in geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
        return max(polys, key=lambda poly: float(poly.area), default=None)
    return None


def _decompose_polys(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty]
    if isinstance(geom, GeometryCollection):
        return [
            poly
            for poly in geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
    return [
        poly
        for poly in getattr(geom, "geoms", [])
        if isinstance(poly, Polygon) and not poly.is_empty
    ]


def _surface_plane_coefficients(
    surface: dict[str, Any],
) -> tuple[float, float, float] | None:
    cluster = surface.get("cluster") or {}
    avg_azimuth = cluster.get("avgAzimuth")
    avg_incl = cluster.get("avgIncl")
    ref = cluster.get("refPt") or surface.get("center") or {}
    if avg_azimuth is not None and avg_incl is not None and ref:
        normal = plane_normal(float(avg_azimuth), float(avg_incl))
        return _height_model_from_plane(normal, ref)

    corners = [
        corner
        for corner in (surface.get("corners") or [])
        if isinstance(corner, (list, tuple)) and len(corner) >= 3
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
    normal = {
        "x": ab[1] * ac[2] - ab[2] * ac[1],
        "y": ab[2] * ac[0] - ab[0] * ac[2],
        "z": ab[0] * ac[1] - ab[1] * ac[0],
    }
    return _height_model_from_plane(
        normal,
        {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
    )


def _height_model_from_plane(
    normal: dict[str, float], ref: dict[str, float]
) -> tuple[float, float, float] | None:
    ny = float(normal["y"])
    norm = math.sqrt(
        float(normal["x"]) * float(normal["x"])
        + ny * ny
        + float(normal["z"]) * float(normal["z"])
    )
    if norm <= 1e-9 or abs(ny) <= 1e-9:
        return None
    nx = float(normal["x"]) / norm
    ny /= norm
    nz = float(normal["z"]) / norm
    return (
        -nx / ny,
        -nz / ny,
        float(ref["y"]) + (nx * float(ref["x"]) + nz * float(ref["z"])) / ny,
    )


def _oblique_y(model: tuple[float, float, float], x: float, z: float) -> float:
    a, b, c = model
    return a * float(x) + b * float(z) + c


ClipSide = Literal["above", "below"]


def _keep_signed(
    *,
    flat_y: float,
    model: tuple[float, float, float],
    x: float,
    z: float,
    keep_side: ClipSide,
) -> float:
    oblique_y = _oblique_y(model, x, z)
    if keep_side == "below":
        return oblique_y - float(flat_y) + FLAT_OBLIQUE_INTERSECTION_TOL_M
    return float(flat_y) - oblique_y + FLAT_OBLIQUE_INTERSECTION_TOL_M


def _equal_height_line(
    *,
    flat_y: float,
    model: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    keep_side: ClipSide,
) -> LineString | None:
    a, b, _c = model
    if abs(a) <= 1e-9 and abs(b) <= 1e-9:
        return None
    minx, minz, maxx, maxz = bounds
    span = max(4.0, math.hypot(maxx - minx, maxz - minz) * 3.0)
    cx = (minx + maxx) * 0.5
    cz = (minz + maxz) * 0.5
    signed = _keep_signed(flat_y=flat_y, model=model, x=cx, z=cz, keep_side=keep_side)
    # Project the bounds center onto signed == 0, then extend along the
    # perpendicular to the height gradient.
    grad_x = -a if keep_side == "above" else a
    grad_z = -b if keep_side == "above" else b
    denom = a * a + b * b
    px = cx - signed * grad_x / denom
    pz = cz - signed * grad_z / denom
    dx = -b
    dz = a
    length = math.hypot(dx, dz)
    if length <= 1e-9:
        return None
    dx /= length
    dz /= length
    return LineString(
        [
            (_snap(px - dx * span), _snap(pz - dz * span)),
            (_snap(px + dx * span), _snap(pz + dz * span)),
        ]
    )


def _kept_piece_parts(
    *,
    piece: Polygon,
    flat_y: float,
    model: tuple[float, float, float],
    keep_side: ClipSide,
) -> list[Polygon]:
    rep = piece.representative_point()
    sample_points = [(float(rep.x), float(rep.y))]
    sample_points.extend(piece.exterior.coords[:-1])
    signed_values = [
        _keep_signed(
            flat_y=flat_y,
            model=model,
            x=float(point[0]),
            z=float(point[1]),
            keep_side=keep_side,
        )
        for point in sample_points
    ]
    if all(value >= -1e-9 for value in signed_values):
        return [piece]
    if all(value < -1e-9 for value in signed_values):
        return []

    line = _equal_height_line(
        flat_y=flat_y, model=model, bounds=piece.bounds, keep_side=keep_side
    )
    if line is None:
        return [piece]
    try:
        split_result = split(piece, line)
        pieces = _decompose_polys(split_result)
    except Exception:
        pieces = [piece]

    kept: list[Polygon] = []
    for candidate in pieces:
        if candidate.is_empty or candidate.area <= AREA_EPS:
            continue
        rep = candidate.representative_point()
        signed = _keep_signed(
            flat_y=flat_y,
            model=model,
            x=float(rep.x),
            z=float(rep.y),
            keep_side=keep_side,
        )
        if signed >= -1e-9:
            kept.append(candidate)
    return kept


def _corners_from_poly(poly: Polygon, y: float) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    return [[_snap(float(x)), _snap(float(y)), _snap(float(z))] for x, z in coords]


def _mark_suppressed(
    hypothesis_graph: dict[str, Any], hypothesis_id: str, reason: str
) -> None:
    selected_ids = [
        str(value)
        for value in (hypothesis_graph.get("selected_hypothesis_ids") or [])
        if str(value) != hypothesis_id
    ]
    hypothesis_graph["selected_hypothesis_ids"] = selected_ids

    assignments = hypothesis_graph.get("selected_room_assignments") or {}
    for room_id, ids in list(assignments.items()):
        kept = [str(value) for value in (ids or []) if str(value) != hypothesis_id]
        if kept:
            assignments[room_id] = kept
        else:
            assignments.pop(room_id, None)
    hypothesis_graph["selected_room_assignments"] = assignments

    for node in hypothesis_graph.get("nodes") or []:
        if not isinstance(node, dict) or str(node.get("id")) != hypothesis_id:
            continue
        node["selected"] = False
        node["suppressed"] = True
        node["suppressed_reason"] = reason

    for edge in hypothesis_graph.get("edges") or []:
        if edge.get("type") == "COVERS_ROOM" and str(edge.get("from")) == hypothesis_id:
            edge["selected"] = False

    metadata = hypothesis_graph.get("metadata")
    if isinstance(metadata, dict):
        metadata["selected_hypothesis_count"] = len(selected_ids)
        metadata["selected_room_count"] = len(assignments)


def clip_flat_surfaces_against_obliques(
    *,
    flat_surfaces: list[dict[str, Any]],
    oblique_surfaces: list[dict[str, Any]],
    hypothesis_graph: dict[str, Any] | None = None,
    keep_side: ClipSide = "above",
    suppress_consumed_top_flats: bool = True,
) -> list[dict[str, Any]]:
    clipped_surfaces: list[dict[str, Any]] = []

    for flat in flat_surfaces:
        flat_y = _flat_y(flat)
        flat_poly = _surface_poly(flat)
        if flat_y is None or flat_poly is None:
            clipped_surfaces.append(flat)
            continue

        original_area = float(flat_poly.area)
        pieces = [flat_poly]
        clipped_by: set[str] = set()

        for oblique in oblique_surfaces:
            if not _can_clip_against_oblique(flat, oblique):
                continue
            oblique_poly = _surface_poly(oblique)
            model = _surface_plane_coefficients(oblique)
            if oblique_poly is None or model is None:
                continue

            next_pieces: list[Polygon] = []
            cut_this_oblique = False
            for piece in pieces:
                try:
                    domain = piece.intersection(oblique_poly)
                except Exception:
                    domain = None
                domain_polys = [
                    poly for poly in _decompose_polys(domain) if poly.area > AREA_EPS
                ]
                if not domain_polys:
                    next_pieces.append(piece)
                    continue

                kept = _kept_piece_parts(
                    piece=piece,
                    flat_y=flat_y,
                    model=model,
                    keep_side=keep_side,
                )
                if sum(float(poly.area) for poly in kept) < piece.area - AREA_EPS:
                    cut_this_oblique = True
                next_pieces.extend(poly for poly in kept if poly.area > AREA_EPS)

            pieces = next_pieces
            if cut_this_oblique:
                hypothesis_id = oblique.get("roof_hypothesis_id")
                if hypothesis_id:
                    clipped_by.add(str(hypothesis_id))
            if not pieces:
                break

        residual_polys = [piece for piece in pieces if piece.area > AREA_EPS]
        post_area = sum(float(poly.area) for poly in residual_polys)
        hypothesis_id = str(flat.get("roof_hypothesis_id") or "")
        source_kind = _source_kind(flat)
        suppressed = (
            suppress_consumed_top_flats
            and source_kind == "top"
            and clipped_by
            and (
                post_area < TOP_FLAT_MIN_RESIDUAL_AREA_M2
                or post_area / max(original_area, AREA_EPS)
                < TOP_FLAT_MIN_RESIDUAL_RATIO
            )
        )
        if suppressed:
            if hypothesis_graph is not None and hypothesis_id:
                _mark_suppressed(
                    hypothesis_graph,
                    hypothesis_id,
                    "top_flat_consumed_by_oblique_intersections",
                )
            continue

        if not residual_polys:
            continue

        for index, poly in enumerate(residual_polys):
            surface = dict(flat)
            surface["corners"] = _corners_from_poly(poly, flat_y)
            surface["pre_clip_area_m2"] = round(original_area, 6)
            surface["post_clip_area_m2"] = round(float(poly.area), 6)
            if clipped_by:
                surface["clipped_by_oblique_hypothesis_ids"] = sorted(clipped_by)
            if len(residual_polys) > 1:
                surface["clip_piece_index"] = index
                surface["clip_piece_count"] = len(residual_polys)
            clipped_surfaces.append(surface)

    return clipped_surfaces
