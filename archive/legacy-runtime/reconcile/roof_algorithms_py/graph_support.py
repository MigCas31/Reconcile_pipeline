from __future__ import annotations

from typing import Any

from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile_v2.decision_logic import classify_roof_room, classify_room_relation_state


def _largest_polygon_component(geom) -> Polygon | None:
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
        best = None
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


def _poly_xz(points3d: list) -> Polygon | None:
    coords = []
    for point in points3d or []:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        coords.append((float(point[0]), float(point[2])))
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            poly = _largest_polygon_component(poly)
        except Exception:
            return None
    if (
        poly is not None
        and not poly.is_empty
        and isinstance(poly, Polygon)
        and poly.area > 1e-6
    ):
        return poly
    try:
        hull = MultiPoint(coords).convex_hull
    except Exception:
        hull = None
    if (
        hull is None
        or hull.is_empty
        or not isinstance(hull, Polygon)
        or hull.area <= 1e-6
    ):
        return None
    return hull


def _graph_room_polygons(graph) -> dict[str, Polygon]:
    cached = getattr(graph, "_room_polygon_cache", None)
    if isinstance(cached, dict):
        return cached
    polygons: dict[str, list[Polygon]] = {}
    cell_complex = (getattr(graph, "geometry_index", None) or {}).get(
        "cell_complex"
    ) or {}
    for cell in cell_complex.get("cells") or []:
        if not isinstance(cell, dict) or str(cell.get("kind") or "") != "room":
            continue
        room_id = str(cell.get("source_id") or "")
        footprint = (cell.get("properties") or {}).get("xz_footprint") or []
        if not room_id or not isinstance(footprint, list) or len(footprint) < 3:
            continue
        coords = []
        for point in footprint:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            coords.append((float(point[0]), float(point[1])))
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            try:
                poly = make_valid(poly)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda geom: geom.area, default=None)
            except Exception:
                continue
        if (
            poly is None
            or poly.is_empty
            or not isinstance(poly, Polygon)
            or poly.area <= 1e-6
        ):
            continue
        polygons.setdefault(room_id, []).append(poly)
    merged: dict[str, Polygon] = {}
    for room_id, polys in polygons.items():
        if not polys:
            continue
        if len(polys) == 1:
            merged[room_id] = polys[0]
            continue
        try:
            unioned = unary_union(polys)
            if unioned.geom_type == "MultiPolygon":
                unioned = max(unioned.geoms, key=lambda geom: geom.area, default=None)
        except Exception:
            unioned = None
        if (
            unioned is not None
            and isinstance(unioned, Polygon)
            and not unioned.is_empty
        ):
            merged[room_id] = unioned
    graph._room_polygon_cache = merged
    return merged


def match_graph_room(graph, story: int, floor_polygon: list) -> object | None:
    if graph is None or not floor_polygon:
        return None
    floor_poly = _poly_xz(floor_polygon)
    room_polys = _graph_room_polygons(graph) if floor_poly is not None else {}
    if floor_poly is not None and room_polys:
        best = None
        best_overlap = 0.0
        best_iou = 0.0
        for room in graph.nodes_by_type("Room"):
            if room.story != story:
                continue
            room_poly = room_polys.get(room.id)
            if room_poly is None:
                continue
            try:
                overlap = float(floor_poly.intersection(room_poly).area)
            except Exception:
                overlap = 0.0
            if overlap <= 1e-6:
                continue
            union_area = max(float(floor_poly.union(room_poly).area), 1e-6)
            iou = overlap / union_area
            if overlap > best_overlap + 1e-6 or (
                abs(overlap - best_overlap) <= 1e-6 and iou > best_iou
            ):
                best = room
                best_overlap = overlap
                best_iou = iou
        if best is not None:
            return best
    fcx = sum(p[0] for p in floor_polygon) / len(floor_polygon)
    fcz = sum(p[2] for p in floor_polygon) / len(floor_polygon)
    best = None
    best_score = float("inf")
    for room in graph.nodes_by_type("Room"):
        if room.story != story or not room.bbox_xz:
            continue
        min_x, min_z, max_x, max_z = [float(v) for v in room.bbox_xz]
        if min_x <= fcx <= max_x and min_z <= fcz <= max_z:
            return room
        cx = (min_x + max_x) * 0.5
        cz = (min_z + max_z) * 0.5
        score = (cx - fcx) ** 2 + (cz - fcz) ** 2
        if score < best_score:
            best_score = score
            best = room
    return best


def classify_graph_roof_room(
    graph,
    story: int,
    floor_polygon: list,
) -> tuple[object | None, dict[str, Any] | None]:
    room = match_graph_room(graph, story, floor_polygon)
    if room is None:
        return None, None
    return room, classify_roof_room(graph, room)


def graph_allows_roof_candidate(decision: dict[str, Any] | None) -> bool:
    if decision is None:
        return False
    return decision.get("mode") != "ceiling_below_occupied_volume"


def graph_requires_geometry_fallback(decision: dict[str, Any] | None) -> bool:
    if decision is None:
        return True
    return decision.get("mode") != "roof_candidate"


def room_surface_bindings(graph, room_id: str) -> dict[str, dict[str, str | None]]:
    if graph is None:
        return {}

    bindings: dict[str, dict[str, str | None]] = {}
    boundary_by_surface: dict[str, str] = {}
    for child in graph.children(room_id):
        if child.type != "Boundary":
            continue
        surface_id = None
        for edge in graph.edges_from(child.id, "BOUNDS"):
            target = graph.get_node(edge.to_id)
            if target is not None and target.type == "Surface":
                surface_id = target.id
                break
        if surface_id:
            boundary_by_surface[surface_id] = child.id

    for child in graph.children(room_id):
        if child.type != "Surface":
            continue
        surface_kind = (child.properties or {}).get("surface_kind", "surface")
        key = child.source_ids[0] if child.source_ids else child.id
        bindings[key] = {
            "graph_surface_id": child.id,
            "boundary_id": boundary_by_surface.get(child.id),
            "surface_kind": surface_kind,
        }
    return bindings


def graph_room_index_map(bldg: dict, graph) -> dict[str, int]:
    if graph is None:
        return {}
    out: dict[str, int] = {}
    for room_idx, room in enumerate(bldg.get("rooms", [])):
        graph_room = match_graph_room(
            graph,
            int(room.get("story", 0)),
            room.get("floor_polygon") or [],
        )
        if graph_room is not None and graph_room.id not in out:
            out[graph_room.id] = room_idx
    return out


def graph_roof_candidate_room_indices(bldg: dict, graph) -> set[int]:
    if graph is None:
        return set()
    out: set[int] = set()
    for room_idx, room in enumerate(bldg.get("rooms", [])):
        graph_room, decision = classify_graph_roof_room(
            graph,
            int(room.get("story", 0)),
            room.get("floor_polygon") or [],
        )
        if graph_room is None or decision is None:
            continue
        if graph_allows_roof_candidate(decision):
            out.add(room_idx)
    return out


def graph_upper_room_ids(graph, room_id: str) -> list[tuple[str, str]]:
    if graph is None or not room_id:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in graph.edges_to(room_id, "ABOVE"):
        state = str((edge.evidence or {}).get("relation_state", "confirmed"))
        key = (edge.from_id, state)
        if key not in seen:
            seen.add(key)
            out.append(key)
    for edge in graph.edges_from(room_id, "BELOW"):
        state = str((edge.evidence or {}).get("relation_state", "confirmed"))
        key = (edge.to_id, state)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def graph_room_relation_profile(graph, room_id: str) -> dict[str, Any] | None:
    if graph is None or not room_id:
        return None
    room = graph.get_node(room_id)
    if room is None:
        return None
    return classify_room_relation_state(graph, room)
