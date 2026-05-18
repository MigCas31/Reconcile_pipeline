from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

from .graph_utils import stable_hash as _stable_hash
from .math_utils import plane_normal, plane_y_at

EPS = 1e-6
AREA_EPS = 0.01
LATTICE_SCALE_MM = 1000
ROOM_TOP_MIN_CLEARANCE_M = 0.15
ROOM_TOP_SHELL_TOL_M = 0.08
# Max y a flat hypothesis may sit above the room's highest wallTop before it
# is rejected as physically outside the room volume. Scan-noise overshoots
# cluster under ~0.15 m; cross-story phantoms start at ~1 m. 0.5 m cleanly
# separates the two without regressing noisy-but-legitimate measurements.
ROOM_TOP_MAX_CLEARANCE_M = 0.5


def _snap(value: float) -> float:
    return round(float(value) * LATTICE_SCALE_MM) / LATTICE_SCALE_MM


def _poly_xz(corners: list) -> Polygon | None:
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
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda geom: geom.area)
            elif poly.geom_type == "GeometryCollection":
                polys = [
                    geom
                    for geom in poly.geoms
                    if isinstance(geom, Polygon)
                    and not geom.is_empty
                    and geom.area > AREA_EPS
                ]
                if polys:
                    poly = max(polys, key=lambda geom: geom.area)
                else:
                    poly = None
        except Exception:
            return None
    if (
        poly is None
        or not isinstance(poly, Polygon)
        or poly.is_empty
        or poly.area <= AREA_EPS
    ):
        try:
            hull = Polygon(points).convex_hull
        except Exception:
            hull = None
        if isinstance(hull, Polygon) and not hull.is_empty and hull.area > AREA_EPS:
            poly = hull
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= AREA_EPS:
        return None
    return poly


def _room_polygon_with_fallback(
    room_data: dict[str, Any],
) -> tuple[Polygon | None, list[list[float]]]:
    raw_corners = room_data.get("fp") or []
    poly = _poly_xz(raw_corners)
    if poly is not None:
        return poly, raw_corners
    graph_fp_xz = room_data.get("graph_fp_xz") or []
    if not isinstance(graph_fp_xz, list) or len(graph_fp_xz) < 3:
        return None, []
    floor_y = float(room_data.get("floorY", 0.0))
    fallback_corners = [
        [_snap(float(point[0])), _snap(floor_y), _snap(float(point[1]))]
        for point in graph_fp_xz
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    poly = _poly_xz(fallback_corners)
    if poly is None:
        return None, []
    return poly, fallback_corners


def _linework_for_polygon(poly) -> list[LineString]:
    if isinstance(poly, Polygon):
        polys = [poly]
    elif isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
    else:
        return []
    lines: list[LineString] = []
    for geom in polys:
        lines.append(LineString(list(geom.exterior.coords)))
        for ring in geom.interiors:
            lines.append(LineString(list(ring.coords)))
    return lines


def _decompose_polys(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [
            item
            for item in geom.geoms
            if isinstance(item, Polygon) and not item.is_empty
        ]
    return [
        item
        for item in getattr(geom, "geoms", [])
        if isinstance(item, Polygon) and not item.is_empty
    ]


def _decompose_lines(geom: Any) -> list[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [
            item
            for item in geom.geoms
            if isinstance(item, LineString) and not item.is_empty
        ]
    return [
        item
        for item in getattr(geom, "geoms", [])
        if isinstance(item, LineString) and not item.is_empty
    ]


def _flat_y(surface: dict[str, Any], room_data: dict[str, Any]) -> float:
    y = surface.get("y")
    if isinstance(y, (float, int)):
        wall_top_y = room_data.get("wallTopY")
        if isinstance(wall_top_y, (float, int)):
            # Flat-segment clusters use a 0.15 m y-tolerance, so the cluster
            # avgY can sit up to 0.15 m below the room's actual wall top.
            # Clamp only within that range; larger gaps indicate a different
            # issue (wrong hypothesis, cross-story confusion) and should not
            # be silently raised.
            gap = float(wall_top_y) - float(y)
            if 0.001 < gap <= 0.15:
                return float(wall_top_y)
        return float(y)
    implicit_top_y = room_data.get("implicit_top_y")
    if isinstance(implicit_top_y, (float, int)):
        return float(implicit_top_y)
    corners = surface.get("corners") or []
    ys = [float(c[1]) for c in corners if isinstance(c, (list, tuple)) and len(c) >= 3]
    if ys:
        return sum(ys) / len(ys)
    return (float(room_data["wallTopY"]) + float(room_data["wallTopMin"])) * 0.5


def _median(values: list[float]) -> float | None:
    ordered = sorted(
        float(value) for value in values if isinstance(value, (float, int))
    )
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def _room_min_valid_top_y(room_data: dict[str, Any]) -> float:
    floor_y = float(room_data.get("floorY", 0.0))
    wall_top_min = float(room_data.get("wallTopMin", floor_y))
    return _snap(
        max(floor_y + ROOM_TOP_MIN_CLEARANCE_M, wall_top_min - ROOM_TOP_SHELL_TOL_M)
    )


def _implicit_flat_y(room_data: dict[str, Any]) -> float:
    median_top = _median(list(room_data.get("wallTopYs") or []))
    if median_top is not None:
        return _snap(max(float(median_top), _room_min_valid_top_y(room_data)))
    return _snap(
        max(
            (float(room_data["wallTopY"]) + float(room_data["wallTopMin"])) * 0.5,
            _room_min_valid_top_y(room_data),
        )
    )


def _room_max_valid_top_y(room_data: dict[str, Any]) -> float:
    wall_top_ys = [
        float(value)
        for value in (room_data.get("wallTopYs") or [])
        if isinstance(value, (float, int))
    ]
    if wall_top_ys:
        wall_top_max = max(wall_top_ys)
    else:
        wall_top_max = float(room_data.get("wallTopY", 0.0))
    return _snap(wall_top_max + ROOM_TOP_MAX_CLEARANCE_M)


def _flat_surface_is_valid_for_room(
    surface: dict[str, Any], room_data: dict[str, Any]
) -> bool:
    y = _flat_y(surface, room_data)
    if y < _room_min_valid_top_y(room_data) - EPS:
        return False
    return y <= _room_max_valid_top_y(room_data) + EPS


def _build_implicit_flat_atom(
    *,
    room_key: str,
    room_index: int,
    story: int,
    atom: Polygon,
    room_data: dict[str, Any],
    supporting_hypothesis_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    top_boundary_mode = str(room_data.get("top_boundary_mode") or "")
    if top_boundary_mode == "ceiling_below_occupied_volume":
        flat_role_reason = "explicit_upper_occupancy_shell_cap"
    else:
        flat_role_reason = "room_shell_top_fallback"
    corners, holes = _atom_corners(
        atom, "flat", None, {**room_data, "implicit_top_y": _implicit_flat_y(room_data)}
    )
    if len(corners) < 3:
        return None
    atom_id = f"ceiling-partition:{
        _stable_hash(
            [room_key, 'implicit-flat', str(corners)],
            20,
        )
    }"
    return {
        "id": atom_id,
        "room_index": room_index,
        "story": story,
        "kind": "flat",
        "roof_hypothesis_id": None,
        "poly": corners,
        "holes": holes,
        "area_m2": round(float(atom.area), 6),
        "supporting_roof_hypothesis_ids": sorted(
            set(str(value) for value in (supporting_hypothesis_ids or []) if value)
        ),
        "flat_role": "implicit_room_shell_cap",
        "flat_role_reason": flat_role_reason,
        "top_y_m": _implicit_flat_y(room_data),
        "top_boundary_mode": top_boundary_mode or "roof_candidate",
        "top_boundary_reason": room_data.get("top_boundary_reason"),
    }


def _surface_plane_from_corners(surface: dict[str, Any]) -> dict[str, Any] | None:
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
    if abs(ny) <= EPS:
        return None
    return {
        "n": {"x": float(nx), "y": float(ny), "z": float(nz)},
        "ref": {
            "x": float(a[0]),
            "y": float(a[1]),
            "z": float(a[2]),
        },
    }


def _surface_plane_from_cluster(surface: dict[str, Any]) -> dict[str, Any] | None:
    cluster = surface.get("cluster") or {}
    avg_azimuth = cluster.get("avgAzimuth")
    avg_incl = cluster.get("avgIncl")
    ref = cluster.get("refPt") or surface.get("center") or {}
    if avg_azimuth is None or avg_incl is None or not ref:
        return None
    return {
        "n": plane_normal(float(avg_azimuth), float(avg_incl)),
        "ref": {
            "x": float(ref["x"]),
            "y": float(ref["y"]),
            "z": float(ref["z"]),
        },
    }


def _surface_plane(surface: dict[str, Any]) -> dict[str, Any] | None:
    plane = _surface_plane_from_cluster(surface)
    if plane is not None:
        return plane
    return _surface_plane_from_corners(surface)


def _height_model(
    kind: str, surface: dict[str, Any], room_data: dict[str, Any]
) -> tuple[float, float, float]:
    if kind == "flat":
        return (0.0, 0.0, _snap(_flat_y(surface, room_data)))
    plane = _surface_plane(surface)
    if plane is None:
        return (0.0, 0.0, _snap(_flat_y(surface, room_data)))
    n = plane["n"]
    ref = plane["ref"]
    ny = float(n["y"])
    if abs(ny) <= EPS:
        return (0.0, 0.0, _snap(_flat_y(surface, room_data)))
    a = -float(n["x"]) / ny
    b = -float(n["z"]) / ny
    c = float(ref["y"]) + (
        (float(n["x"]) * float(ref["x"]) + float(n["z"]) * float(ref["z"])) / ny
    )
    return (_snap(a), _snap(b), _snap(c))


def _height_at(model: tuple[float, float, float], x: float, z: float) -> float:
    a, b, c = model
    return _snap(a * float(x) + b * float(z) + c)


def _equal_height_split_lines(
    left_model: tuple[float, float, float],
    right_model: tuple[float, float, float],
    clip_poly: Polygon,
) -> list[LineString]:
    if clip_poly.is_empty or clip_poly.area <= AREA_EPS:
        return []
    da = float(left_model[0] - right_model[0])
    db = float(left_model[1] - right_model[1])
    dc = float(left_model[2] - right_model[2])
    norm_sq = da * da + db * db
    if norm_sq <= EPS:
        return []
    centroid = clip_poly.representative_point()
    cx = float(centroid.x)
    cz = float(centroid.y)
    signed = da * cx + db * cz + dc
    px = cx - (signed * da / norm_sq)
    pz = cz - (signed * db / norm_sq)
    dx = -db
    dz = da
    dir_norm = (dx * dx + dz * dz) ** 0.5
    if dir_norm <= EPS:
        return []
    dx /= dir_norm
    dz /= dir_norm
    minx, minz, maxx, maxz = clip_poly.bounds
    span = ((maxx - minx) ** 2 + (maxz - minz) ** 2) ** 0.5
    reach = max(4.0, span * 2.0)
    splitter = LineString(
        [
            (_snap(px - dx * reach), _snap(pz - dz * reach)),
            (_snap(px + dx * reach), _snap(pz + dz * reach)),
        ]
    )
    try:
        clipped = clip_poly.intersection(splitter)
    except Exception:
        return []
    return [line for line in _decompose_lines(clipped) if line.length > 0.05]


def _snap_ring_xz(coords: list) -> list[tuple[float, float]]:
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    snapped: list[tuple[float, float]] = []
    for x, z, *_ in coords:
        point = (_snap(float(x)), _snap(float(z)))
        if snapped and point == snapped[-1]:
            continue
        snapped.append(point)
    if len(snapped) >= 2 and snapped[0] == snapped[-1]:
        snapped.pop()
    return snapped


def _best_repaired_polygon(geom: Any, reference: Polygon) -> Polygon | None:
    polys = [
        poly
        for poly in _decompose_polys(geom)
        if isinstance(poly, Polygon) and not poly.is_empty and poly.area > AREA_EPS
    ]
    if not polys:
        return None
    rep = reference.representative_point()
    covering = [poly for poly in polys if poly.buffer(EPS).covers(rep)]
    candidates = covering or polys
    return max(
        candidates,
        key=lambda poly: (
            poly.intersection(reference).area,
            poly.area,
        ),
    )


def _sanitize_snapped_atom_polygon(
    atom: Polygon,
    exterior_xz: list[tuple[float, float]],
    holes_xz: list[list[tuple[float, float]]],
) -> Polygon | None:
    if len(exterior_xz) < 3:
        return None
    snapped = Polygon(exterior_xz, [ring for ring in holes_xz if len(ring) >= 3])
    if snapped.is_valid and not snapped.is_empty and snapped.area > AREA_EPS:
        return snapped
    try:
        repaired = make_valid(snapped)
    except Exception:
        return None
    return _best_repaired_polygon(repaired, atom)


def _atom_corners(
    atom: Polygon,
    kind: str,
    surface: dict[str, Any] | None,
    room_data: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], list[list[tuple[float, float, float]]]]:
    # Shapely's polygonize() can produce polygons with interior rings when the
    # linework nests (e.g. a room atom that encloses sibling partitions). We
    # must carry those holes through to the viewer — storing only the exterior
    # makes the stored vertex ring disagree with atom.area and triggers a
    # "huge triangle" rendering artifact downstream.
    plane = (
        _surface_plane(surface) if kind == "oblique" and surface is not None else None
    )
    flat_y = _snap(_flat_y(surface or {}, room_data))

    snapped_exterior = _snap_ring_xz(list(atom.exterior.coords))
    snapped_holes = [_snap_ring_xz(list(ring.coords)) for ring in atom.interiors]
    sanitized_atom = _sanitize_snapped_atom_polygon(
        atom, snapped_exterior, snapped_holes
    )
    if sanitized_atom is None:
        return [], []

    def _map_ring(
        coords: list[tuple[float, float]],
    ) -> list[tuple[float, float, float]]:
        if plane is not None:
            out: list[tuple[float, float, float]] = []
            for x, z in coords:
                sx = float(x)
                sz = float(z)
                out.append((sx, _snap(plane_y_at(plane, sx, sz)), sz))
            return out
        return [(float(x), flat_y, float(z)) for x, z in coords]

    exterior = _map_ring(list(sanitized_atom.exterior.coords)[:-1])
    holes = [_map_ring(list(ring.coords)[:-1]) for ring in sanitized_atom.interiors]
    holes = [ring for ring in holes if len(ring) >= 3]
    return exterior, holes


def _room_outline_corners(poly: Polygon, floor_y: float) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    return [
        [_snap(float(x)), _snap(float(floor_y)), _snap(float(z))] for x, z, *_ in coords
    ]


def derive_room_ceiling_partitions(
    *,
    room_records: list[dict[str, Any]],
    oblique_roof_surfaces: list[dict[str, Any]],
    flat_roof_surfaces: list[dict[str, Any]],
    hypothesis_graph: dict[str, Any],
) -> dict[str, Any]:
    nodes_by_id = {
        node["id"]: node
        for node in hypothesis_graph.get("nodes") or []
        if node.get("type") == "RoofHypothesis"
    }
    selected_room_assignments = hypothesis_graph.get("selected_room_assignments") or {}
    cover_edges = {
        (edge["from"], edge["to"]): edge
        for edge in hypothesis_graph.get("edges") or []
        if edge.get("type") == "COVERS_ROOM"
    }

    surfaces_by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for surface in oblique_roof_surfaces:
        hypothesis_id = surface.get("roof_hypothesis_id")
        if hypothesis_id:
            surfaces_by_hypothesis[str(hypothesis_id)].append(surface)
    for surface in flat_roof_surfaces:
        hypothesis_id = surface.get("roof_hypothesis_id")
        if hypothesis_id:
            surfaces_by_hypothesis[str(hypothesis_id)].append(surface)

    room_partitions: list[dict[str, Any]] = []
    flat_surfaces: list[dict[str, Any]] = []
    oblique_surfaces: list[dict[str, Any]] = []
    selected_hypothesis_ids = {
        str(hypothesis_id)
        for hypothesis_id in (
            hypothesis_graph.get("selected_hypothesis_ids")
            or [
                node_id for node_id, node in nodes_by_id.items() if node.get("selected")
            ]
        )
    }
    split_line_count = 0
    rejected_flat_candidate_count = 0
    implicit_partition_count = 0

    for room_data in room_records:
        room_index = int(room_data["room_index"])
        room_key = f"room:{room_index}"
        room_polygon, _room_corners = _room_polygon_with_fallback(room_data)
        if room_polygon is None:
            continue

        selected_ids = list(selected_room_assignments.get(room_key) or [])
        candidate_records: list[dict[str, Any]] = []
        linework = _linework_for_polygon(room_polygon)

        fallback_candidate_ids = sorted(selected_hypothesis_ids - set(selected_ids))
        if selected_ids:
            # Let globally-selected oblique hypotheses challenge a room-local
            # flat selection, but do not allow room-unselected flat planes to
            # steal atoms from rooms that already have explicit assignments.
            fallback_candidate_ids = [
                hypothesis_id
                for hypothesis_id in fallback_candidate_ids
                if str(
                    (nodes_by_id.get(hypothesis_id) or {}).get("surface_kind", "flat")
                )
                == "oblique"
            ]
        candidate_ids = list(dict.fromkeys(selected_ids + fallback_candidate_ids))
        for hypothesis_id in candidate_ids:
            surfaces = surfaces_by_hypothesis.get(hypothesis_id) or []
            if not surfaces:
                continue
            node = nodes_by_id.get(hypothesis_id) or {}
            edge = cover_edges.get((hypothesis_id, room_key)) or {}
            for surface_index, surface in enumerate(surfaces):
                if str(
                    node.get("surface_kind", "flat")
                ) == "flat" and not _flat_surface_is_valid_for_room(surface, room_data):
                    rejected_flat_candidate_count += 1
                    continue
                surface_poly = _poly_xz(surface.get("corners") or [])
                if surface_poly is None:
                    continue
                try:
                    overlap = room_polygon.intersection(surface_poly)
                except Exception:
                    continue
                if overlap.is_empty or overlap.area <= AREA_EPS:
                    continue
                room_overlaps = _decompose_polys(overlap)
                if not room_overlaps:
                    continue
                for room_overlap in room_overlaps:
                    linework.extend(_linework_for_polygon(room_overlap))
                candidate_records.append(
                    {
                        "surface_key": f"{hypothesis_id}:{surface_index}",
                        "hypothesis_id": hypothesis_id,
                        "surface": surface,
                        "kind": str(node.get("surface_kind", "flat")),
                        "room_overlap": unary_union(room_overlaps),
                        "selected_for_room": hypothesis_id in selected_ids,
                        "edge_score": float(
                            ((edge.get("evidence") or {}).get("edge_score")) or 0.0
                        ),
                        "height_model": _height_model(
                            str(node.get("surface_kind", "flat")), surface, room_data
                        ),
                    }
                )

        for left_index, left in enumerate(candidate_records):
            for right in candidate_records[left_index + 1 :]:
                try:
                    common = left["room_overlap"].intersection(right["room_overlap"])
                except Exception:
                    continue
                for common_poly in _decompose_polys(common):
                    split_lines = _equal_height_split_lines(
                        left["height_model"],
                        right["height_model"],
                        common_poly,
                    )
                    split_line_count += len(split_lines)
                    linework.extend(split_lines)

        arrangement = unary_union(linework)
        atom_candidates = list(polygonize(arrangement))
        if not atom_candidates:
            atom_candidates = [room_polygon]

        atoms: list[dict[str, Any]] = []
        for atom in atom_candidates:
            if atom.is_empty or atom.area <= AREA_EPS:
                continue
            rep = atom.representative_point()
            if not room_polygon.buffer(EPS).contains(rep):
                continue

            owner_id = None
            owner_top_y = None
            owner_selected = False
            owner_edge_score = -1.0
            owner_kind = "flat"
            owner_surface = None
            supporting_hypothesis_ids: list[str] = []
            for candidate in candidate_records:
                try:
                    overlap_area = atom.intersection(candidate["room_overlap"]).area
                except Exception:
                    overlap_area = 0.0
                if overlap_area <= AREA_EPS:
                    continue
                supporting_hypothesis_ids.append(str(candidate["hypothesis_id"]))
                top_y = _height_at(
                    candidate["height_model"], float(rep.x), float(rep.y)
                )
                selected_for_room = bool(candidate["selected_for_room"])
                edge_score = float(candidate["edge_score"])
                if (
                    owner_top_y is None
                    or top_y < owner_top_y - EPS
                    or (
                        abs(top_y - owner_top_y) <= EPS
                        and selected_for_room
                        and not owner_selected
                    )
                    or (
                        abs(top_y - owner_top_y) <= EPS
                        and selected_for_room == owner_selected
                        and edge_score > owner_edge_score + EPS
                    )
                ):
                    owner_id = str(candidate["hypothesis_id"])
                    owner_top_y = top_y
                    owner_selected = selected_for_room
                    owner_edge_score = edge_score
                    owner_surface = candidate["surface"]
                    owner_kind = str(candidate["kind"])

            if owner_id is None:
                owner_kind = "flat"
                owner_surface = None

            corners, holes = _atom_corners(atom, owner_kind, owner_surface, room_data)
            if len(corners) < 3:
                continue
            partition_id = f"ceiling-partition:{
                _stable_hash(
                    [room_key, owner_id or 'fallback', str(corners)],
                    20,
                )
            }"
            atom_record = {
                "id": partition_id,
                "room_index": room_index,
                "story": int(room_data["story"]),
                "kind": owner_kind,
                "roof_hypothesis_id": owner_id,
                "poly": corners,
                "holes": holes,
                "area_m2": round(float(atom.area), 6),
                "supporting_roof_hypothesis_ids": sorted(
                    set(supporting_hypothesis_ids)
                ),
                "top_boundary_mode": room_data.get("top_boundary_mode"),
                "top_boundary_reason": room_data.get("top_boundary_reason"),
            }
            if isinstance(owner_surface, dict):
                if "flat_role" in owner_surface:
                    atom_record["flat_role"] = owner_surface.get("flat_role")
                    atom_record["flat_role_reason"] = owner_surface.get(
                        "flat_role_reason"
                    )
            elif owner_kind == "flat":
                top_boundary_mode = str(room_data.get("top_boundary_mode") or "")
                atom_record["flat_role"] = "implicit_room_shell_cap"
                atom_record["flat_role_reason"] = (
                    "explicit_upper_occupancy_shell_cap"
                    if top_boundary_mode == "ceiling_below_occupied_volume"
                    else "room_shell_top_fallback"
                )
                atom_record["top_y_m"] = _implicit_flat_y(room_data)
            if owner_top_y is not None:
                atom_record["top_y_m"] = owner_top_y
            atoms.append(atom_record)
            if owner_kind == "oblique":
                oblique_surfaces.append(atom_record)
            else:
                flat_surfaces.append(atom_record)

        atom_polys = [_poly_xz(atom_record.get("poly") or []) for atom_record in atoms]
        atom_polys = [
            poly for poly in atom_polys if poly is not None and poly.area > AREA_EPS
        ]
        if atom_polys:
            try:
                uncovered = room_polygon.difference(unary_union(atom_polys))
            except Exception:
                uncovered = None
            for uncovered_poly in _decompose_polys(uncovered):
                if uncovered_poly.is_empty or uncovered_poly.area <= AREA_EPS:
                    continue
                supporting_hypothesis_ids: list[str] = []
                rep = uncovered_poly.representative_point()
                for candidate in candidate_records:
                    try:
                        overlap_area = uncovered_poly.intersection(
                            candidate["room_overlap"]
                        ).area
                    except Exception:
                        overlap_area = 0.0
                    if overlap_area <= AREA_EPS:
                        continue
                    if candidate["kind"] == "oblique":
                        supporting_hypothesis_ids.append(
                            str(candidate["hypothesis_id"])
                        )
                        continue
                    top_y = _height_at(
                        candidate["height_model"], float(rep.x), float(rep.y)
                    )
                    if top_y >= _room_min_valid_top_y(room_data) - EPS:
                        supporting_hypothesis_ids.append(
                            str(candidate["hypothesis_id"])
                        )
                implicit_atom = _build_implicit_flat_atom(
                    room_key=room_key,
                    room_index=room_index,
                    story=int(room_data["story"]),
                    atom=uncovered_poly,
                    room_data=room_data,
                    supporting_hypothesis_ids=supporting_hypothesis_ids,
                )
                if implicit_atom is None:
                    continue
                atoms.append(implicit_atom)
                flat_surfaces.append(implicit_atom)
                implicit_partition_count += 1

        if not atoms:
            implicit_atom = _build_implicit_flat_atom(
                room_key=room_key,
                room_index=room_index,
                story=int(room_data["story"]),
                atom=room_polygon,
                room_data=room_data,
            )
            if implicit_atom is not None:
                atoms.append(implicit_atom)
                flat_surfaces.append(implicit_atom)
                implicit_partition_count += 1

        room_partitions.append(
            {
                "room_index": room_index,
                "story": int(room_data["story"]),
                "graph_room_id": room_data.get("graph_room_id"),
                "room_outline": _room_outline_corners(
                    room_polygon, float(room_data.get("floorY", 0.0))
                ),
                "partition_count": len(atoms),
                "mixed": len({atom["kind"] for atom in atoms}) > 1
                or len(
                    {
                        atom["roof_hypothesis_id"]
                        for atom in atoms
                        if atom["roof_hypothesis_id"]
                    }
                )
                > 1,
                "partitions": atoms,
            }
        )

    return {
        "room_partitions": room_partitions,
        "flat": flat_surfaces,
        "oblique": oblique_surfaces,
        "metadata": {
            "room_partition_count": len(room_partitions),
            "flat_partition_count": len(flat_surfaces),
            "oblique_partition_count": len(oblique_surfaces),
            "mixed_room_count": sum(1 for room in room_partitions if room["mixed"]),
            "split_line_count": split_line_count,
            "rejected_flat_candidate_count": rejected_flat_candidate_count,
            "implicit_partition_count": implicit_partition_count,
        },
    }


def inject_simple_slant_partitions(
    *,
    partitions: dict[str, Any],
    simple_slant_ceilings: list[dict[str, Any]],
    room_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not simple_slant_ceilings:
        return partitions

    room_data_by_index = {
        int(room_data["room_index"]): room_data
        for room_data in room_records
        if isinstance(room_data, dict) and isinstance(room_data.get("room_index"), int)
    }
    room_partitions = list(partitions.get("room_partitions") or [])
    room_partition_by_index = {
        int(room_partition["room_index"]): room_partition
        for room_partition in room_partitions
        if isinstance(room_partition, dict)
        and isinstance(room_partition.get("room_index"), int)
    }
    flat_surfaces = list(partitions.get("flat") or [])
    oblique_surfaces = list(partitions.get("oblique") or [])
    metadata = dict(partitions.get("metadata") or {})
    injected_count = 0

    for ceiling in simple_slant_ceilings:
        if not isinstance(ceiling, dict):
            continue
        room_index = ceiling.get("room_index")
        if not isinstance(room_index, int):
            continue
        existing = room_partition_by_index.get(room_index)
        if existing is not None and int(existing.get("partition_count", 0) or 0) > 0:
            continue
        room_data = room_data_by_index.get(room_index)
        poly = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in (ceiling.get("poly") or [])
            if isinstance(point, (list, tuple)) and len(point) >= 3
        ]
        poly_xz = _poly_xz(poly)
        if room_data is None or poly_xz is None or len(poly) < 3:
            continue
        partition = {
            "id": f"ceiling-partition:{
                _stable_hash(
                    [f'room:{room_index}', 'simple-slant', str(poly)],
                    20,
                )
            }",
            "room_index": room_index,
            "story": int(room_data.get("story", ceiling.get("story", 0)) or 0),
            "kind": "oblique",
            "roof_hypothesis_id": None,
            "poly": poly,
            "area_m2": round(float(poly_xz.area), 6),
            "supporting_roof_hypothesis_ids": [],
            "top_boundary_mode": room_data.get("top_boundary_mode"),
            "top_boundary_reason": room_data.get("top_boundary_reason")
            or "simple_slant_ceiling_polygon",
        }
        room_partition = {
            "room_index": room_index,
            "story": int(room_data.get("story", ceiling.get("story", 0)) or 0),
            "graph_room_id": room_data.get("graph_room_id"),
            "room_outline": _room_outline_corners(
                poly_xz, float(room_data.get("floorY", 0.0))
            ),
            "partition_count": 1,
            "mixed": False,
            "partitions": [partition],
        }
        room_partition_by_index[room_index] = room_partition
        room_partitions.append(room_partition)
        oblique_surfaces.append(partition)
        injected_count += 1

    if injected_count <= 0:
        return partitions

    metadata["room_partition_count"] = len(room_partitions)
    metadata["oblique_partition_count"] = len(oblique_surfaces)
    metadata["flat_partition_count"] = len(flat_surfaces)
    metadata["mixed_room_count"] = sum(1 for room in room_partitions if room["mixed"])
    metadata["simple_slant_partition_count"] = injected_count

    return {
        "room_partitions": room_partitions,
        "flat": flat_surfaces,
        "oblique": oblique_surfaces,
        "metadata": metadata,
    }
