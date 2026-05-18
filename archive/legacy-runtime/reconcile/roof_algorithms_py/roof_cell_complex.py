from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, triangulate, unary_union
from shapely.validation import make_valid

from .graph_utils import stable_hash as _stable_hash
from .math_utils import plane_normal, plane_y_at
from .roof_arrangement_kernel import build_arranged_polyhedral_cell

EPS = 1e-6
AREA_EPS = 0.01
LATTICE_SCALE_MM = 1000
MIN_UPPER_VOID_HEIGHT_M = 0.12
MIN_KNEE_WALL_AREA_M2 = 0.1
MIN_KNEE_WALL_BASE_EDGE_M = 0.15
KNEE_WALL_OCCUPIED_SHELL_MAX_DIST_M = 0.08
KNEE_WALL_OCCUPIED_SHELL_MAX_BOTTOM_DELTA_M = 0.35


def _snap(value: float) -> float:
    return round(float(value) * LATTICE_SCALE_MM) / LATTICE_SCALE_MM


def _decompose_polys(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _largest_polygon_component(geom: Any) -> Polygon | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = [
            poly
            for poly in geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
        return max(polys, key=lambda poly: float(poly.area), default=None)
    if isinstance(geom, GeometryCollection):
        best: Polygon | None = None
        best_area = 0.0
        for child in geom.geoms:
            poly = _largest_polygon_component(child)
            if poly is None:
                continue
            area = float(poly.area)
            if area > best_area:
                best = poly
                best_area = area
        return best
    return None


def _poly_xz_from_3d(corners: list) -> Polygon | None:
    points: list[tuple[float, float]] = []
    for corner in corners or []:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            continue
        points.append((_snap(float(corner[0])), _snap(float(corner[2]))))
    if len(points) < 3:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            poly = _largest_polygon_component(poly)
        except Exception:
            return None
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= AREA_EPS:
        return None
    return poly


def _linework_for_polygon(poly: Polygon) -> list[LineString]:
    lines = [LineString(list(poly.exterior.coords))]
    for ring in poly.interiors:
        lines.append(LineString(list(ring.coords)))
    return lines


def _surface_plane(surface: dict[str, Any]) -> dict[str, Any] | None:
    cluster = surface.get("cluster") or {}
    avg_azimuth = cluster.get("avgAzimuth")
    avg_incl = cluster.get("avgIncl")
    ref = cluster.get("refPt") or surface.get("center") or {}
    if avg_azimuth is not None and avg_incl is not None and ref:
        return {
            "n": plane_normal(float(avg_azimuth), float(avg_incl)),
            "ref": {
                "x": float(ref["x"]),
                "y": float(ref["y"]),
                "z": float(ref["z"]),
            },
        }
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
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= EPS or abs(ny) <= EPS:
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
        "ref": {
            "x": float(a[0]),
            "y": float(a[1]),
            "z": float(a[2]),
        },
    }


def _surface_is_oblique(surface: dict[str, Any]) -> bool:
    kind = str(surface.get("kind") or surface.get("surface_kind") or "")
    if kind == "oblique":
        return True
    if kind == "flat":
        return False
    hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
    if hypothesis_id.startswith("roof-hypothesis:oblique:"):
        return True
    if hypothesis_id.startswith("roof-hypothesis:flat:"):
        return False
    cluster = surface.get("cluster") or {}
    try:
        return abs(float(cluster.get("avgIncl"))) > 5.0
    except Exception:
        return False


def _surface_y_at(surface: dict[str, Any], x: float, z: float) -> float:
    if _surface_is_oblique(surface):
        plane = _surface_plane(surface)
        if plane is not None:
            return _snap(plane_y_at(plane, _snap(x), _snap(z)))
    corners = surface.get("corners") or []
    ys = [float(c[1]) for c in corners if isinstance(c, (list, tuple)) and len(c) >= 3]
    return _snap(sum(ys) / len(ys)) if ys else 0.0


def _avg_y(corners: list) -> float:
    ys = [
        float(c[1])
        for c in corners or []
        if isinstance(c, (list, tuple)) and len(c) >= 3
    ]
    if not ys:
        return 0.0
    return _snap(sum(ys) / len(ys))


def _classify_upper_cell_kind(
    *,
    atom: dict[str, Any],
    room_partition: dict[str, Any],
    room_has_oblique_atom: bool,
    part_family: str,
    top_surface: dict[str, Any],
) -> str:
    if room_partition.get("mixed") or room_has_oblique_atom:
        return "upper_void"
    if str(atom.get("flat_role") or "") == "ceiling_cap":
        return "upper_void"
    if str(top_surface.get("kind", "flat")) == "oblique":
        return "attic"
    if part_family == "gable_or_multi_slope":
        return "attic"
    return "upper_void"


def _lift_polygon(
    polygon: Polygon,
    *,
    y_at,
) -> list[list[float]]:
    coords = list(polygon.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    corners: list[list[float]] = []
    for x, z, *_ in coords:
        sx = _snap(float(x))
        sz = _snap(float(z))
        corners.append([sx, _snap(y_at(sx, sz)), sz])
    return corners


def _triangle_area_2d(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return abs(
        (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) * 0.5
    )


def _polyhedral_volume(base: list[list[float]], top: list[list[float]]) -> float:
    if len(base) < 3 or len(base) != len(top):
        return 0.0
    origin = (float(base[0][0]), float(base[0][2]))
    volume = 0.0
    for idx in range(1, len(base) - 1):
        p0 = origin
        p1 = (float(base[idx][0]), float(base[idx][2]))
        p2 = (float(base[idx + 1][0]), float(base[idx + 1][2]))
        area = _triangle_area_2d(p0, p1, p2)
        h0 = max(float(top[0][1]) - float(base[0][1]), 0.0)
        h1 = max(float(top[idx][1]) - float(base[idx][1]), 0.0)
        h2 = max(float(top[idx + 1][1]) - float(base[idx + 1][1]), 0.0)
        volume += area * ((h0 + h1 + h2) / 3.0)
    return _snap(volume)


def _cell_faces(
    base: list[list[float]], top: list[list[float]]
) -> list[dict[str, Any]]:
    faces = [
        {"kind": "bottom", "corners": base},
        {"kind": "top", "corners": list(reversed(top))},
    ]
    for idx in range(len(base)):
        nxt = (idx + 1) % len(base)
        faces.append(
            {
                "kind": "side",
                "corners": [
                    base[idx],
                    base[nxt],
                    top[nxt],
                    top[idx],
                ],
            }
        )
    return faces


def _convex_subregions(region: Polygon) -> list[Polygon]:
    if region.is_empty or region.area <= AREA_EPS:
        return []
    try:
        hull = region.convex_hull
        if abs(float(hull.area) - float(region.area)) <= AREA_EPS:
            return [region]
    except Exception:
        return [region]
    triangles = triangulate(region)
    if not triangles:
        return [region]
    pieces: list[Polygon] = []
    for triangle in triangles:
        try:
            clipped = region.intersection(triangle)
        except Exception:
            continue
        for piece in _decompose_polys(clipped):
            if piece.is_empty or piece.area <= AREA_EPS:
                continue
            pieces.append(piece)
    if not pieces:
        return [region]
    return pieces


def _perimeter_side_face_indices(region: Polygon, atom_poly: Polygon) -> set[int]:
    coords = list(region.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    boundary = atom_poly.boundary
    keep: set[int] = set()
    for idx in range(len(coords)):
        nxt = (idx + 1) % len(coords)
        edge = LineString([coords[idx], coords[nxt]])
        if edge.length <= EPS:
            continue
        try:
            shared_len = float(edge.intersection(boundary).length)
        except Exception:
            shared_len = 0.0
        if shared_len >= max(0.05, edge.length * 0.7):
            keep.add(idx)
    return keep


def _derive_knee_walls(
    *,
    cell_id: str,
    room_id: str,
    room_index: int | None,
    part_id: str | None,
    base_atom_id: str,
    top_surface_kind: str,
    faces: list[dict[str, Any]],
    perimeter_side_face_indices: set[int],
) -> list[dict[str, Any]]:
    knee_walls: list[dict[str, Any]] = []
    if top_surface_kind != "oblique":
        return knee_walls
    for face_index, face in enumerate(faces):
        if face.get("kind") != "side":
            continue
        metadata = face.get("metadata") or {}
        side_index = metadata.get("side_index")
        if not isinstance(side_index, int):
            side_index = face_index - 2
        if side_index not in perimeter_side_face_indices:
            continue
        corners = face.get("corners") or []
        if len(corners) != 4:
            continue
        ys = sorted(float(c[1]) for c in corners)
        height = ys[-1] - ys[0]
        if height <= MIN_UPPER_VOID_HEIGHT_M:
            continue
        # Minimum face area filter (cross product of two edges of the quad)
        c = corners
        dx1, dy1, dz1 = c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]
        dx2, dy2, dz2 = c[3][0] - c[0][0], c[3][1] - c[0][1], c[3][2] - c[0][2]
        cx = dy1 * dz2 - dz1 * dy2
        cy = dz1 * dx2 - dx1 * dz2
        cz = dx1 * dy2 - dy1 * dx2
        face_area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        if face_area < MIN_KNEE_WALL_AREA_M2:
            continue
        # Minimum base edge length (bottom two corners)
        bottom = sorted(corners, key=lambda p: p[1])[:2]
        bdx = bottom[1][0] - bottom[0][0]
        bdz = bottom[1][2] - bottom[0][2]
        base_len = math.sqrt(bdx * bdx + bdz * bdz)
        if base_len < MIN_KNEE_WALL_BASE_EDGE_M:
            continue
        knee_walls.append(
            {
                "id": f"knee-wall:{
                    _stable_hash(
                        [cell_id, str(face_index), str(corners)],
                        20,
                    )
                }",
                "cell_id": cell_id,
                "room_id": room_id,
                "room_index": room_index,
                "part_id": part_id,
                "base_atom_id": base_atom_id,
                "height_m": _snap(height),
                "face_area_m2": round(face_area, 4),
                "base_edge_m": round(base_len, 4),
                "corners": corners,
            }
        )
    return knee_walls


def _extreme_edge(
    corners: list[list[float]], *, mode: str
) -> tuple[LineString | None, float | None]:
    if mode not in {"top", "bottom"}:
        return None, None
    if len(corners) < 2:
        return None, None
    ordered = sorted(
        (
            [float(point[0]), float(point[1]), float(point[2])]
            for point in corners
            if isinstance(point, (list, tuple)) and len(point) >= 3
        ),
        key=lambda point: point[1],
    )
    if len(ordered) < 2:
        return None, None
    points = ordered[:2] if mode == "bottom" else ordered[-2:]
    edge = LineString([(points[0][0], points[0][2]), (points[1][0], points[1][2])])
    if edge.length <= EPS:
        return None, None
    return edge, _snap((points[0][1] + points[1][1]) * 0.5)


def _line_angle_deg(left: LineString, right: LineString) -> float:
    left_coords = list(left.coords)
    right_coords = list(right.coords)
    if len(left_coords) < 2 or len(right_coords) < 2:
        return 180.0
    lx = float(left_coords[1][0] - left_coords[0][0])
    lz = float(left_coords[1][1] - left_coords[0][1])
    rx = float(right_coords[1][0] - right_coords[0][0])
    rz = float(right_coords[1][1] - right_coords[0][1])
    left_len = math.sqrt(lx * lx + lz * lz)
    right_len = math.sqrt(rx * rx + rz * rz)
    if left_len <= EPS or right_len <= EPS:
        return 180.0
    dot = max(-1.0, min(1.0, (lx * rx + lz * rz) / (left_len * right_len)))
    return math.degrees(math.acos(abs(dot)))


def filter_knee_walls_by_occupied_shell(
    *,
    roof_cell_complex: dict[str, Any],
    occupied_room_cell_complex: dict[str, Any] | None,
) -> dict[str, Any]:
    if not roof_cell_complex:
        return roof_cell_complex
    knee_walls = [
        wall
        for wall in (roof_cell_complex.get("knee_walls") or [])
        if isinstance(wall, dict)
    ]
    if not knee_walls or not occupied_room_cell_complex:
        return roof_cell_complex

    supports_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in occupied_room_cell_complex.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        room_index = cell.get("room_index")
        if not isinstance(room_index, int):
            continue
        for face in cell.get("faces") or []:
            if not isinstance(face, dict):
                continue
            metadata = face.get("metadata") or {}
            if str(metadata.get("boundary_class") or "") != "exterior_wall":
                continue
            top_edge, top_y = _extreme_edge(face.get("corners") or [], mode="top")
            if top_edge is None or top_y is None:
                continue
            supports_by_room[room_index].append(
                {
                    "edge": top_edge,
                    "top_y_m": float(top_y),
                    "cell_id": cell.get("id"),
                    "face_id": face.get("id"),
                }
            )

    kept: list[dict[str, Any]] = []
    dropped_count = 0
    for wall in knee_walls:
        room_index = wall.get("room_index")
        if not isinstance(room_index, int):
            dropped_count += 1
            continue
        bottom_edge, bottom_y = _extreme_edge(wall.get("corners") or [], mode="bottom")
        if bottom_edge is None or bottom_y is None:
            dropped_count += 1
            continue
        base_edge_m = float(wall.get("base_edge_m", 0.0) or 0.0)
        min_supported_length = max(0.15, min(1.0, base_edge_m * 0.25))
        best_support: dict[str, Any] | None = None
        for support in supports_by_room.get(room_index, []):
            support_edge = support["edge"]
            if _line_angle_deg(bottom_edge, support_edge) > 12.0:
                continue
            if (
                abs(float(bottom_y) - float(support["top_y_m"]))
                > KNEE_WALL_OCCUPIED_SHELL_MAX_BOTTOM_DELTA_M
            ):
                continue
            distance = float(bottom_edge.distance(support_edge))
            if distance > KNEE_WALL_OCCUPIED_SHELL_MAX_DIST_M:
                continue
            supported_length = float(
                bottom_edge.intersection(
                    support_edge.buffer(
                        KNEE_WALL_OCCUPIED_SHELL_MAX_DIST_M, cap_style=2
                    )
                ).length
            )
            if (
                best_support is None
                or supported_length > float(best_support["supported_length_m"]) + EPS
            ):
                best_support = {
                    "supported_length_m": round(supported_length, 6),
                    "distance_m": round(distance, 6),
                    "occupied_top_y_m": round(float(support["top_y_m"]), 6),
                    "occupied_cell_id": support.get("cell_id"),
                    "occupied_face_id": support.get("face_id"),
                }
        if (
            best_support is None
            or float(best_support["supported_length_m"]) + EPS < min_supported_length
        ):
            dropped_count += 1
            continue
        enriched = dict(wall)
        enriched["occupied_shell_support"] = best_support
        kept.append(enriched)

    metadata = dict(roof_cell_complex.get("metadata") or {})
    metadata["knee_wall_count"] = len(kept)
    metadata["knee_wall_dropped_by_occupied_shell"] = dropped_count
    return {
        **roof_cell_complex,
        "knee_walls": kept,
        "metadata": metadata,
    }


def build_roof_cell_complex(
    *,
    room_partitions: list[dict[str, Any]],
    selected_oblique_surfaces: list[dict[str, Any]],
    selected_flat_surfaces: list[dict[str, Any]],
    building_part_graph: dict[str, Any],
    roof_coverage_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    part_nodes = {node["id"]: node for node in (building_part_graph.get("nodes") or [])}
    room_membership = building_part_graph.get("room_membership") or {}
    hypothesis_membership = building_part_graph.get("hypothesis_membership") or {}

    surface_records: list[dict[str, Any]] = []
    for kind, surfaces in (("oblique", selected_oblique_surfaces),):
        for surface in surfaces:
            hypothesis_id = surface.get("roof_hypothesis_id")
            if not hypothesis_id:
                continue
            part_ids = hypothesis_membership.get(str(hypothesis_id), [])
            poly = _poly_xz_from_3d(surface.get("corners") or [])
            if poly is None:
                continue
            surface_records.append(
                {
                    "kind": kind,
                    "roof_hypothesis_id": str(hypothesis_id),
                    "part_ids": [str(part_id) for part_id in part_ids],
                    "polygon": poly,
                    "surface": surface,
                }
            )

    cells: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    cells_by_atom: dict[str, list[str]] = defaultdict(list)
    knee_walls: list[dict[str, Any]] = []
    coverage_by_atom = (roof_coverage_graph or {}).get("atom_coverage") or {}

    for room_partition in room_partitions:
        room_id = f"room:{int(room_partition['room_index'])}"
        part_ids = room_membership.get(room_id, [])
        part_id = str(part_ids[0]) if part_ids else None
        part = part_nodes.get(part_id) or {}
        part_family = str(part.get("roof_family_guess", "unknown"))
        room_has_oblique_atom = any(
            partition.get("kind") == "oblique"
            for partition in (room_partition.get("partitions") or [])
        )

        for atom in room_partition.get("partitions") or []:
            if atom.get("kind") != "flat":
                continue
            flat_role = str(atom.get("flat_role") or "")
            coverage = coverage_by_atom.get(str(atom.get("id"))) or {}
            sloped_state = str(coverage.get("sloped_state", "none"))
            atom_is_upper_void_candidate = (
                flat_role
                in {
                    "ceiling_cap",
                    "ambiguous_flat_over_sloped_part",
                }
                or room_has_oblique_atom
                or sloped_state in {"confirmed", "partial"}
            )
            if not atom_is_upper_void_candidate:
                continue
            atom_poly = _poly_xz_from_3d(atom.get("poly") or [])
            if atom_poly is None:
                continue
            base_y = _avg_y(atom.get("poly") or [])

            overlaps: list[dict[str, Any]] = []
            linework = _linework_for_polygon(atom_poly)
            target_sloped_hypothesis_id = coverage.get("sloped_hypothesis_id")
            for surface_record in surface_records:
                if (
                    surface_record["kind"] == "oblique"
                    and target_sloped_hypothesis_id
                    and surface_record["roof_hypothesis_id"]
                    != target_sloped_hypothesis_id
                ):
                    continue
                try:
                    overlap = atom_poly.intersection(surface_record["polygon"])
                except Exception:
                    continue
                for overlap_poly in _decompose_polys(overlap):
                    if overlap_poly.is_empty or overlap_poly.area <= AREA_EPS:
                        continue
                    overlaps.append({**surface_record, "overlap": overlap_poly})
                    linework.extend(_linework_for_polygon(overlap_poly))
            if not overlaps:
                continue

            arrangement = unary_union(linework)
            regions = list(polygonize(arrangement))
            if not regions:
                regions = [atom_poly]

            for region in regions:
                if region.is_empty or region.area <= AREA_EPS:
                    continue
                rep = region.representative_point()
                if not atom_poly.buffer(EPS).covers(rep):
                    continue

                best_surface = None
                best_top_y = None
                for overlap in overlaps:
                    try:
                        if not overlap["overlap"].buffer(EPS).covers(rep):
                            continue
                    except Exception:
                        continue
                    top_y = _surface_y_at(
                        overlap["surface"], float(rep.x), float(rep.y)
                    )
                    if top_y <= base_y + MIN_UPPER_VOID_HEIGHT_M:
                        continue
                    same_part = part_id is not None and part_id in (
                        overlap.get("part_ids") or []
                    )
                    current_same_part = (
                        part_id is not None
                        and best_surface is not None
                        and part_id in (best_surface.get("part_ids") or [])
                    )
                    if (
                        best_top_y is None
                        or top_y < best_top_y - EPS
                        or (
                            abs(top_y - best_top_y) <= EPS
                            and same_part
                            and not current_same_part
                        )
                    ):
                        best_top_y = top_y
                        best_surface = overlap
                if best_surface is None or best_top_y is None:
                    continue

                top_kind = _classify_upper_cell_kind(
                    atom=atom,
                    room_partition=room_partition,
                    room_has_oblique_atom=room_has_oblique_atom,
                    part_family=part_family,
                    top_surface=best_surface,
                )
                convex_regions = _convex_subregions(region)
                for convex_index, convex_region in enumerate(convex_regions):
                    if convex_region.is_empty or convex_region.area <= AREA_EPS:
                        continue
                    coords = list(convex_region.exterior.coords)
                    if coords and coords[-1] == coords[0]:
                        coords = coords[:-1]
                    if len(coords) < 3:
                        continue
                    region_footprint = [
                        (_snap(float(x)), _snap(float(z))) for x, z, *_ in coords
                    ]
                    perimeter_side_face_indices = _perimeter_side_face_indices(
                        convex_region, atom_poly
                    )
                    cell_id_hash = _stable_hash(
                        [
                            atom["id"],
                            best_surface["roof_hypothesis_id"],
                            str(region_footprint),
                            str(convex_index),
                        ],
                        20,
                    )
                    cell_id = f"roof-cell:{top_kind}:{cell_id_hash}"
                    cell = build_arranged_polyhedral_cell(
                        cell_id=cell_id,
                        room_id=room_id,
                        room_index=atom.get("room_index"),
                        part_id=part_id,
                        story=atom.get("story"),
                        base_atom_id=str(atom["id"]),
                        cell_kind=top_kind,
                        region_footprint=region_footprint,
                        base_y=base_y,
                        top_y_at=lambda x, z, surface=best_surface["surface"]: (
                            _surface_y_at(surface, x, z)
                        ),
                        top_surface_kind=best_surface["kind"],
                        roof_hypothesis_id=best_surface["roof_hypothesis_id"],
                        perimeter_side_face_indices=perimeter_side_face_indices,
                    )
                    if cell is None or float(cell.get("volume_m3", 0.0)) <= EPS:
                        continue

                    cells.append(cell)
                    knee_walls.extend(
                        _derive_knee_walls(
                            cell_id=cell_id,
                            room_id=room_id,
                            room_index=atom.get("room_index"),
                            part_id=part_id,
                            base_atom_id=str(atom["id"]),
                            top_surface_kind=best_surface["kind"],
                            faces=cell["faces"],
                            perimeter_side_face_indices=perimeter_side_face_indices,
                        )
                    )
                    cells_by_atom[str(atom["id"])].append(cell_id)
                    edges.append(
                        {
                            "id": f"edge:atom-cell:{
                                _stable_hash(
                                    [str(atom['id']), cell_id],
                                    20,
                                )
                            }",
                            "type": "BOUNDS_CELL",
                            "from": str(atom["id"]),
                            "to": cell_id,
                        }
                    )
                    edges.append(
                        {
                            "id": f"edge:cell-room:{
                                _stable_hash(
                                    [cell_id, room_id],
                                    20,
                                )
                            }",
                            "type": "ABOVE_ROOM",
                            "from": cell_id,
                            "to": room_id,
                        }
                    )
                    edges.append(
                        {
                            "id": f"edge:cell-hypothesis:{
                                _stable_hash(
                                    [cell_id, best_surface['roof_hypothesis_id']],
                                    20,
                                )
                            }",
                            "type": "SUPPORTED_BY",
                            "from": cell_id,
                            "to": best_surface["roof_hypothesis_id"],
                        }
                    )

    return {
        "cells": cells,
        "edges": edges,
        "knee_walls": knee_walls,
        "cells_by_atom": dict(cells_by_atom),
        "metadata": {
            "backend": "exact_lattice_roof_wall_slab_arrangement_v2",
            "numeric_model": "fixed_point_mm",
            "exact_on_lattice": True,
            "polyhedral_kernel": "halfspace_triple_intersection",
            "cell_count": len(cells),
            "attic_cell_count": sum(
                1 for cell in cells if cell.get("cell_kind") == "attic"
            ),
            "upper_void_cell_count": sum(
                1 for cell in cells if cell.get("cell_kind") == "upper_void"
            ),
            "knee_wall_count": len(knee_walls),
        },
    }
