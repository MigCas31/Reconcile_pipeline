from __future__ import annotations

from math import hypot

from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.geometry import edge_inclination_deg, xz_polygon_to_vec3
from reconcile_tiers.roof.roof import FlatSurface

INTERMEDIATE_MIN_DIM_M = 2.0
TOP_INCL_MAX_DEG = 5.0
TOP_Y_CLUSTER_M = 0.15
TOP_PAD_M = 0.3


def _room_ceiling_y(room) -> float | None:
    ys = [corner[1] for wall in room.walls_computed for corner in wall.corners]
    if not ys:
        return None
    return max(float(y) for y in ys)


def _bbox_dim_xz(corners: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in corners]
    zs = [p[2] for p in corners]
    return max(xs) - min(xs), max(zs) - min(zs)


def _intermediate_flats(model: BuildingModel, top_story: int) -> list[FlatSurface]:
    out: list[FlatSurface] = []
    for room in model.rooms:
        if room.story >= top_story or len(room.floor_polygon) < 3:
            continue
        dx, dz = _bbox_dim_xz(room.floor_polygon)
        if max(dx, dz) < INTERMEDIATE_MIN_DIM_M:
            continue
        y = _room_ceiling_y(room)
        if y is None:
            continue
        out.append(
            FlatSurface(
                corners=xz_polygon_to_vec3(
                    [(p[0], p[2]) for p in room.floor_polygon], y
                ),
                story=room.story,
                dominant_story=top_story,
                source="intermediate",
            )
        )
    return out


def _ceiling_flats(model: BuildingModel, top_story: int) -> list[FlatSurface]:
    out: list[FlatSurface] = []
    for room in model.rooms:
        if room.ceiling_type != "flat" or len(room.ceiling_polygon) < 3:
            continue
        out.append(
            FlatSurface(
                corners=[
                    [float(p[0]), float(p[1]), float(p[2])]
                    for p in room.ceiling_polygon
                ],
                story=room.story,
                dominant_story=top_story,
                source="ceiling",
            )
        )
    return out


def _top_flat_edges(
    model: BuildingModel, top_story: int
) -> list[tuple[list[float], list[float], float]]:
    edges: list[tuple[list[float], list[float], float]] = []
    for room in model.rooms:
        if room.story != top_story:
            continue
        for wall in room.walls_computed:
            if len(wall.corners) < 3:
                continue
            wall_max_y = max(c[1] for c in wall.corners)
            for idx, a in enumerate(wall.corners):
                b = wall.corners[(idx + 1) % len(wall.corners)]
                if edge_inclination_deg(a, b) >= TOP_INCL_MAX_DEG:
                    continue
                if hypot(b[0] - a[0], b[2] - a[2]) < 0.3:
                    continue
                y = (a[1] + b[1]) / 2.0
                if abs(y - wall_max_y) > 0.05:
                    continue
                edges.append((a, b, float(y)))
    return edges


def _top_flats(model: BuildingModel, top_story: int) -> list[FlatSurface]:
    groups: list[list[tuple[list[float], list[float], float]]] = []
    for edge in sorted(_top_flat_edges(model, top_story), key=lambda e: e[2]):
        for group in groups:
            if abs(edge[2] - sum(e[2] for e in group) / len(group)) <= TOP_Y_CLUSTER_M:
                group.append(edge)
                break
        else:
            groups.append([edge])
    out: list[FlatSurface] = []
    for group in groups:
        xs = [point[0] for a, b, _ in group for point in (a, b)]
        zs = [point[2] for a, b, _ in group for point in (a, b)]
        y = sum(e[2] for e in group) / len(group)
        ring = [
            (min(xs) - TOP_PAD_M, min(zs) - TOP_PAD_M),
            (max(xs) + TOP_PAD_M, min(zs) - TOP_PAD_M),
            (max(xs) + TOP_PAD_M, max(zs) + TOP_PAD_M),
            (min(xs) - TOP_PAD_M, max(zs) + TOP_PAD_M),
        ]
        out.append(
            FlatSurface(
                corners=xz_polygon_to_vec3(ring, y),
                story=top_story,
                dominant_story=top_story,
                source="top",
            )
        )
    return out


def build_flat_surfaces(model: BuildingModel) -> list[FlatSurface]:
    top_story = max((room.story for room in model.rooms), default=0)
    ceiling_flats = _ceiling_flats(model, top_story)
    if ceiling_flats:
        return [
            *ceiling_flats,
            *(_top_flats(model, top_story) if top_story == 0 else []),
        ]
    return [*_intermediate_flats(model, top_story), *_top_flats(model, top_story)]
