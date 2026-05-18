from __future__ import annotations

from .graph_support import graph_upper_room_ids, match_graph_room


def compute_plane_height_caps(
    *,
    bldg: dict,
    ceiling_planes: list,
    plane_clipped: list,
    top_story: int,
    all_stories: list,
    floors_by_story: dict,
    point_in_poly_xz,
    point_in_poly_2d,
    graph=None,
) -> list:
    def _poly_bbox(poly2d):
        xs = [p[0] for p in poly2d]
        zs = [p[1] for p in poly2d]
        return min(xs), min(zs), max(xs), max(zs)

    def _bbox_overlap(a, b):
        ax0, az0, ax1, az1 = a
        bx0, bz0, bx1, bz1 = b
        return min(ax1, bx1) > max(ax0, bx0) and min(az1, bz1) > max(az0, bz0)

    def _graph_cap_for_plane(plane, poly2d):
        if graph is None or len(poly2d) < 3:
            return None
        try:
            from shapely.geometry import Polygon
        except ImportError:
            return None

        plane_poly = Polygon(poly2d)
        if plane_poly.is_empty or not plane_poly.is_valid:
            return None

        room_cells = {
            cell.source_ids[0]: cell
            for cell in graph.nodes_by_type("Cell")
            if (cell.properties or {}).get("cell_kind") == "room" and cell.source_ids
        }
        seed_room_ids: set[str] = set()
        for room_idx in plane.get("room_indices", []):
            if (
                not isinstance(room_idx, int)
                or room_idx < 0
                or room_idx >= len(bldg.get("rooms", []))
            ):
                continue
            room = bldg["rooms"][room_idx]
            graph_room = match_graph_room(
                graph, int(room.get("story", 0)), room.get("floor_polygon") or []
            )
            if graph_room is not None:
                seed_room_ids.add(graph_room.id)

        best_cap = None
        for room_id in seed_room_ids:
            for upper_room_id, relation_state in graph_upper_room_ids(graph, room_id):
                if relation_state not in {"confirmed", "partial"}:
                    continue
                room_cell = room_cells.get(upper_room_id)
                upper_room = graph.get_node(upper_room_id)
                if upper_room is None:
                    continue
                overlap_ok = True
                if room_cell is not None:
                    footprint = (room_cell.properties or {}).get("xz_footprint")
                    if isinstance(footprint, list) and len(footprint) >= 3:
                        upper_poly = Polygon(
                            [(float(p[0]), float(p[1])) for p in footprint]
                        )
                        overlap_ok = (
                            upper_poly.intersects(plane_poly)
                            and upper_poly.intersection(plane_poly).area > 0.01
                        )
                elif upper_room.bbox_xz:
                    overlap_ok = _bbox_overlap(
                        _poly_bbox(poly2d), tuple(float(v) for v in upper_room.bbox_xz)
                    )
                if not overlap_ok:
                    continue
                cap_y = (upper_room.properties or {}).get("floor_height_y")
                if cap_y is None and room_cell is not None:
                    bbox_xyz = (room_cell.properties or {}).get("bbox_xyz")
                    if isinstance(bbox_xyz, list) and len(bbox_xyz) == 6:
                        cap_y = float(bbox_xyz[1])
                if cap_y is None:
                    continue
                cap_y = float(cap_y)
                if best_cap is None or cap_y < best_cap:
                    best_cap = cap_y
        return best_cap

    plane_max_y = [float("inf")] * len(ceiling_planes)

    for i, plane in enumerate(ceiling_planes):
        if plane["dominantStory"] >= top_story:
            continue
        if len(plane_clipped[i]["clipped"]) < 3:
            continue

        graph_cap = _graph_cap_for_plane(plane, plane_clipped[i]["clipped"])
        if graph_cap is not None:
            plane_max_y[i] = graph_cap
            continue

        cap_y = float("inf")
        for s in all_stories:
            if s <= plane["dominantStory"]:
                continue
            for fp in floors_by_story.get(s, []):
                if len(fp) < 3:
                    continue
                poly2d = plane_clipped[i]["clipped"]
                overlap = any(point_in_poly_xz(p[0], p[1], fp) for p in poly2d)
                if not overlap:
                    overlap = any(point_in_poly_2d(v[0], v[2], poly2d) for v in fp)
                if overlap:
                    cap_y = min(cap_y, fp[0][1])

            wall_max = float("-inf")
            for room in bldg.get("rooms", []):
                if int(room.get("story", 0)) != s:
                    continue
                for w in room.get("walls_computed", []) or []:
                    for c in w.get("corners", []):
                        wall_max = max(wall_max, c[1])
            if wall_max > cap_y:
                cap_y = wall_max

        if cap_y != float("inf"):
            plane_max_y[i] = cap_y

    return plane_max_y
