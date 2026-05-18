"""Step 6 of the assembly: Wing per plan-view component.

A Wing is a connected component of the building plan in XZ. The
component decomposition is computed structurally: take the
shapely union of every Floor polygon, then each disjoint geometry
in the result is one Wing. Each Wing's `footprint` is that
component's actual exterior outline (concave when the plan has
re-entrant corners).

Rooms are assigned to a Wing by point-in-polygon containment of
their floor centroid against each component.
"""

from __future__ import annotations

from reconcile_tiers.payload.schema import Vec3
from reconcile_tiers.twin._geometry import FLOAT_EPS, polygon_area_xz
from reconcile_tiers.twin.types import Room, Story, Wing


def wings_for_rooms(
    rooms: tuple[Room, ...], *, building_uuid: str
) -> tuple[tuple[Wing, ...], tuple[tuple[float, list[tuple[float, float]]], ...]]:
    """Detect plan-view-connected wings from the room set.

    Returns `(wings, hole_polygons)`. Each entry in `hole_polygons` is
    `(ground_y, exterior_xz)` for one interior hole in the union of
    floor polygons — these are unclaimed regions of the building's
    plan view that no room covers, and become `Gap` primitives in the
    residual stream.
    """
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    if not rooms:
        return (), ()

    polys: list[Polygon] = []
    ground_y: float | None = None
    for r in rooms:
        if len(r.floor.polygon) < 3:
            continue
        polys.append(Polygon([(c.x, c.z) for c in r.floor.polygon]).buffer(0))
        for c in r.floor.polygon:
            if ground_y is None or c.y < ground_y:
                ground_y = c.y
    if not polys or ground_y is None:
        return (), ()

    union = unary_union(polys)
    components: list[Polygon]
    if union.geom_type == "Polygon":
        components = [union]
    elif union.geom_type == "MultiPolygon":
        components = list(union.geoms)
    else:
        return (), ()

    holes_out: list[tuple[float, list[tuple[float, float]]]] = []
    wings: list[Wing] = []
    for idx, comp in enumerate(components):
        for interior in comp.interiors:
            ring = list(interior.coords)
            if ring and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) >= 3:
                holes_out.append(
                    (float(ground_y), [(float(x), float(z)) for x, z in ring])
                )
        comp_rooms: list[Room] = []
        for r in rooms:
            poly = Polygon([(c.x, c.z) for c in r.floor.polygon]).buffer(0)
            try:
                centroid = poly.representative_point()
            except Exception:
                continue
            if comp.contains(Point(centroid.x, centroid.y)) or (
                comp.intersects(poly)
                and comp.intersection(poly).area >= 0.5 * poly.area
            ):
                comp_rooms.append(r)
        if not comp_rooms:
            continue

        stories = _stories_for_wing_rooms(
            comp_rooms, wing_idx=idx, building_uuid=building_uuid
        )
        if not stories:
            continue

        footprint = _exterior_to_vec3(comp, ground_y)
        if footprint is None:
            continue
        wings.append(
            Wing(
                id=f"{building_uuid}::wing::{idx}",
                stories=stories,
                footprint=footprint,
            )
        )
    return tuple(wings), tuple(holes_out)


def _stories_for_wing_rooms(
    rooms: list[Room], *, wing_idx: int, building_uuid: str
) -> tuple[Story, ...]:
    by_story: dict[int, list[Room]] = {}
    for r in rooms:
        by_story.setdefault(r.story_index, []).append(r)
    return tuple(
        Story(
            id=f"{building_uuid}::wing::{wing_idx}::story::{idx}",
            rooms=tuple(grouped),
        )
        for idx, grouped in sorted(by_story.items())
    )


def _exterior_to_vec3(component, y: float) -> tuple[Vec3, ...] | None:
    coords = list(component.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None
    footprint = tuple(Vec3(x=float(x), y=float(y), z=float(z)) for x, z in coords)
    if polygon_area_xz(footprint) < FLOAT_EPS:
        return None
    return footprint
