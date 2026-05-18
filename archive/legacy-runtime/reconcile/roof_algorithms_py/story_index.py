from __future__ import annotations

from .graph_support import match_graph_room
from .math_utils import point_in_poly_xz


def _relation_supports_upper_occupancy(edge) -> bool:
    evidence = edge.evidence or {}
    state = str(evidence.get("relation_state", "unknown"))
    if state in {"confirmed", "partial"}:
        return True
    support = evidence.get("support_ratio")
    if support is None:
        return False
    try:
        return float(support) >= 0.05
    except (TypeError, ValueError):
        return False


def build_story_index(bldg: dict, graph=None) -> dict:
    floors_by_story: dict[int, list[list]] = {}
    for room in bldg.get("rooms", []):
        fp = room.get("floor_polygon")
        if not fp or len(fp) < 3:
            continue
        story = int(room.get("story", 0))
        floors_by_story.setdefault(story, []).append(fp)

    all_stories = sorted(floors_by_story.keys())

    def has_floor_above(px: float, pz: float, edge_story: int) -> bool:
        if graph is not None:
            probe_poly = [
                [px - 0.05, 0.0, pz - 0.05],
                [px + 0.05, 0.0, pz - 0.05],
                [px + 0.05, 0.0, pz + 0.05],
                [px - 0.05, 0.0, pz + 0.05],
            ]
            graph_room = match_graph_room(graph, edge_story, probe_poly)
            if graph_room is not None:
                for edge in graph.edges_to(graph_room.id, "ABOVE"):
                    upper = graph.get_node(edge.from_id)
                    if upper is None:
                        continue
                    if isinstance(upper.story, int) and upper.story <= edge_story:
                        continue
                    if _relation_supports_upper_occupancy(edge):
                        return True
                for edge in graph.edges_from(graph_room.id, "BELOW"):
                    upper = graph.get_node(edge.to_id)
                    if upper is None:
                        continue
                    if isinstance(upper.story, int) and upper.story <= edge_story:
                        continue
                    if _relation_supports_upper_occupancy(edge):
                        return True
        for s in all_stories:
            if s <= edge_story:
                continue
            for fp in floors_by_story[s]:
                if point_in_poly_xz(px, pz, fp):
                    return True
        return False

    return {
        "floors_by_story": floors_by_story,
        "all_stories": all_stories,
        "has_floor_above": has_floor_above,
    }


def build_story_stats(bldg: dict) -> dict:
    story_max_y: dict[int, float] = {}
    story_floor_polys: dict[int, list[list]] = {}
    story_floor_y: dict[int, float] = {}
    story_floor_regions: dict[int, list[tuple[list, float]]] = {}

    for room in bldg.get("rooms", []):
        s = int(room.get("story", 0))

        for w in room.get("walls_computed", []) or []:
            for c in w.get("corners", []):
                y = float(c[1])
                if s not in story_max_y or y > story_max_y[s]:
                    story_max_y[s] = y

        story_floor_polys.setdefault(s, [])
        fp = room.get("floor_polygon")
        if fp:
            story_floor_polys[s].append(fp)
            fy = float(fp[0][1])
            if s not in story_floor_y or fy < story_floor_y[s]:
                story_floor_y[s] = fy
            story_floor_regions.setdefault(s, []).append((fp, fy))

    return {
        "story_max_y": story_max_y,
        "story_floor_polys": story_floor_polys,
        "story_floor_y": story_floor_y,
        "story_floor_regions": story_floor_regions,
    }


def compute_building_y_bounds(bldg: dict) -> dict:
    bldg_min_y = float("inf")
    bldg_max_y = float("-inf")

    for room in bldg.get("rooms", []):
        for w in room.get("walls_computed", []) or []:
            for c in w.get("corners", []):
                y = float(c[1])
                if y < bldg_min_y:
                    bldg_min_y = y
                if y > bldg_max_y:
                    bldg_max_y = y

    return {"bldg_min_y": bldg_min_y, "bldg_max_y": bldg_max_y}
