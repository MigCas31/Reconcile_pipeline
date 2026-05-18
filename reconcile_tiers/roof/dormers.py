from __future__ import annotations

from math import hypot, sqrt

from shapely.geometry import LineString, Point, Polygon

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import Plane
from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)
from reconcile_tiers.roof.geometry import (
    edge_inclination_deg,
    wall_normal_xz,
    wall_quads,
)
from reconcile_tiers.roof.roof import DormerCandidate, ObliqueSurface

PERPENDICULAR_DOT = 0.35
# A real dormer-front wall, scanned from inside the room, runs from the room
# floor (well below the slant) up through the slant and tops out as the
# visible dormer face. The bottom routinely sits below the slant; a bottom
# floating well ABOVE the slant indicates a free-standing fragment, not a
# dormer face — keep the upper bound, drop the lower one.
MAX_BOTTOM_ABOVE_SLANT_M = 1.5
# A dormer's front wall protrudes meaningfully above the slant — that's the
# whole point of a dormer. Without this gate, knee walls whose tops barely
# clear the slant get misclassified as dormer faces.
MIN_TOP_ABOVE_SLANT_M = 0.10
MAX_WIDTH_RATIO = 0.70
OPENING_PLANE_TOL_M = 0.05
OPENING_CENTROID_PLANE_TOL_M = 0.20
OPENING_CENTROID_BBOX_PAD_M = 0.25
OPENING_EDGE_MARGIN_M = 0.01
MIN_DORMER_DEPTH_M = 0.20
# RoomPlan often fits the host gable plane through a shallow dormer front,
# leaving only a small measured rise above the plane. Keep the wall-width
# gate at 20 cm, but allow a smaller cutout depth when the front has an
# attached opening and passes the host-slope checks.
MIN_DORMER_CUTOUT_DEPTH_M = 0.10
MAX_WALL_CENTER_FROM_SLANT_XZ_M = 2.0
SLANT_XZ_CUTOUT_BUFFER_M = 0.15
MIN_ROOM_SLANT_INCL_DEG = 5.0
MAX_ROOM_SLANT_INCL_DEG = 80.0
# Dormers are protrusions through a real pitched roof plane. Shallow
# transition roofs and short room-local slants routinely put ordinary vertical
# walls above the fitted plane; treating those as dormers punches destructive
# holes through the roof shell.
MIN_PARENT_INCLINATION_DEG = 25.0
MIN_PARENT_RIDGE_SPAN_M = 4.0
# The dormer front opening should be attached to the sloped host roof. A
# regular facade window can sit near a roof plane in 3D, but if its XZ footprint
# is this far from the selected slant, there is no roof surface to cut into.
MAX_FRONT_OPENING_FROM_SLANT_XZ_M = 0.75
FLOOR_ABOVE_WALL_BUFFER_M = 0.05


def _room_floor_xz(room) -> Polygon | None:
    floor = getattr(room, "floor_polygon", None) or []
    if len(floor) < 3:
        return None
    try:
        poly = Polygon([(float(p[0]), float(p[2])) for p in floor])
    except Exception:
        return None
    poly = make_valid_polygon(poly)
    if poly is None or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _wall_under_higher_floor(
    wall: ExtractedWall, room: ExtractedRoom, rooms: list[ExtractedRoom]
) -> bool:
    center = Point(*_wall_center_xz(wall.corners))
    for other in rooms:
        if other.story <= room.story:
            continue
        poly = _room_floor_xz(other)
        if poly is not None and poly.buffer(FLOOR_ABOVE_WALL_BUFFER_M).contains(center):
            return True
    return False


def _wall_width_along_ridge(
    corners: list[list[float]], ridge_x: float, ridge_z: float
) -> float:
    projections = [p[0] * ridge_x + p[2] * ridge_z for p in corners]
    return max(projections) - min(projections)


def _bottom_edge(corners: list[list[float]]) -> tuple[list[float], list[float]]:
    bottom = sorted(corners, key=lambda p: p[1])[:2]
    return bottom[0], bottom[1]


def _top_edge(corners: list[list[float]]) -> tuple[list[float], list[float]]:
    top = sorted(corners, key=lambda p: p[1], reverse=True)[:2]
    return top[0], top[1]


def _wall_center_xz(corners: list[list[float]]) -> tuple[float, float]:
    return (
        sum(float(p[0]) for p in corners) / len(corners),
        sum(float(p[2]) for p in corners) / len(corners),
    )


def _projection_axes(corners: list[list[float]]) -> tuple[int, int]:
    nx, ny, nz = newell_normal(corners)
    anx, any_, anz = abs(nx), abs(ny), abs(nz)
    if any_ >= anx and any_ >= anz:
        return 0, 2
    if anx >= any_ and anx >= anz:
        return 1, 2
    return 0, 1


def _point_in_polygon_2d(
    point: tuple[float, float], poly: list[tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _distance_point_to_segment_2d(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = point[0] - a[0], point[1] - a[1]
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    px, py = a[0] + t * vx, a[1] + t * vy
    return hypot(point[0] - px, point[1] - py)


def _distance_to_plane(
    normal: tuple[float, float, float], p0: list[float], point: list[float]
) -> float:
    norm = sqrt(sum(value * value for value in normal))
    if norm <= 1e-12:
        return float("inf")
    return (
        abs(sum(normal[idx] * (float(point[idx]) - p0[idx]) for idx in range(3))) / norm
    )


def _opening_inside_wall(
    wall_corners: list[list[float]], opening: ExtractedElement
) -> bool:
    if len(opening.corners) != 4 or len(wall_corners) < 3:
        return False
    normal = newell_normal(wall_corners)
    if sqrt(sum(value * value for value in normal)) <= 1e-12:
        return False
    p0 = [float(value) for value in wall_corners[0]]
    axis0, axis1 = _projection_axes(wall_corners)
    outer = [(float(corner[axis0]), float(corner[axis1])) for corner in wall_corners]
    centroid = [
        sum(float(corner[idx]) for corner in opening.corners) / len(opening.corners)
        for idx in range(3)
    ]

    # Prefer exact containment when the opening quad is cleanly coplanar with
    # the wall, but RoomPlan often reports cutouts with a few corners slightly
    # off-plane. Fall back to centroid-on-pane matching so a front window/door
    # remains the primary dormer signal even when the cutout quad is noisy.
    def centroid_on_pane() -> bool:
        if _distance_to_plane(normal, p0, centroid) > OPENING_CENTROID_PLANE_TOL_M:
            return False
        ys = [float(corner[1]) for corner in wall_corners]
        if (
            not min(ys) - OPENING_CENTROID_BBOX_PAD_M
            <= centroid[1]
            <= max(ys) + OPENING_CENTROID_BBOX_PAD_M
        ):
            return False
        projected = [
            (float(corner[axis0]), float(corner[axis1])) for corner in wall_corners
        ]
        min0, max0 = min(p[0] for p in projected), max(p[0] for p in projected)
        min1, max1 = min(p[1] for p in projected), max(p[1] for p in projected)
        return (
            min0 - OPENING_CENTROID_BBOX_PAD_M
            <= centroid[axis0]
            <= max0 + OPENING_CENTROID_BBOX_PAD_M
            and min1 - OPENING_CENTROID_BBOX_PAD_M
            <= centroid[axis1]
            <= max1 + OPENING_CENTROID_BBOX_PAD_M
        )

    if any(
        _distance_to_plane(normal, p0, point) > OPENING_PLANE_TOL_M
        for point in opening.corners
    ):
        return centroid_on_pane()

    if not _point_in_polygon_2d((centroid[axis0], centroid[axis1]), outer):
        return centroid_on_pane()

    for corner in opening.corners:
        point = (float(corner[axis0]), float(corner[axis1]))
        if not _point_in_polygon_2d(point, outer):
            return centroid_on_pane()
        min_distance = min(
            _distance_point_to_segment_2d(point, outer[idx - 1], outer[idx])
            for idx in range(len(outer))
        )
        if min_distance < OPENING_EDGE_MARGIN_M:
            return centroid_on_pane()
    return True


def _has_front_opening(front_wall: ExtractedWall, room) -> bool:
    return any(
        _opening_inside_wall(front_wall.corners, opening)
        for opening in [*room.windows, *room.doors]
    )


def _front_openings(front_wall: ExtractedWall, room) -> list[ExtractedElement]:
    return [
        opening
        for opening in [*room.windows, *room.doors]
        if _opening_inside_wall(front_wall.corners, opening)
    ]


def _opening_center_xz(opening: ExtractedElement) -> tuple[float, float]:
    return (
        sum(float(p[0]) for p in opening.corners) / len(opening.corners),
        sum(float(p[2]) for p in opening.corners) / len(opening.corners),
    )


def _opening_width_along_ridge(
    opening: ExtractedElement, ridge_x: float, ridge_z: float
) -> float:
    projections = [p[0] * ridge_x + p[2] * ridge_z for p in opening.corners]
    return max(projections) - min(projections)


def _room_has_slanted_wall(room) -> bool:
    for wall in room.walls_computed:
        for corners in wall_quads(wall):
            for idx, first in enumerate(corners):
                second = corners[(idx + 1) % len(corners)]
                inclination = edge_inclination_deg(first, second)
                if MIN_ROOM_SLANT_INCL_DEG < inclination < MAX_ROOM_SLANT_INCL_DEG:
                    return True
    return False


def _clamp_to_slant(
    front_xz: tuple[float, float],
    up_x: float,
    up_z: float,
    depth: float,
    slant_xz: Polygon,
) -> float:
    """Clamp a per-side back-depth so the back point stays inside the slant XZ.

    Casts a ray from `front_xz` along `(up_x, up_z)` for `depth` metres and
    finds where it crosses the slant polygon's boundary. Returns the smaller
    of `depth` and the distance to the first boundary crossing past the front
    point. Used to keep dormer cutouts from extending past the actual slant
    on shallow-pitch roofs where the abstract "rise to top_y" point lies
    well beyond where the slant physically ends.
    """
    if depth <= 0.0:
        return depth
    end = (front_xz[0] + up_x * depth, front_xz[1] + up_z * depth)
    ray = LineString([front_xz, end])
    intersection = ray.intersection(slant_xz.boundary)
    if intersection.is_empty:
        return depth

    crossings: list[tuple[float, float]] = []
    geoms = list(getattr(intersection, "geoms", [intersection]))
    for geom in geoms:
        if geom.geom_type == "Point":
            crossings.append((geom.x, geom.y))
        elif geom.geom_type == "LineString":
            crossings.extend((float(x), float(y)) for x, y in geom.coords)

    distances = [hypot(x - front_xz[0], y - front_xz[1]) for x, y in crossings]
    distances = [d for d in distances if d > 1e-6]
    if not distances:
        return depth
    return min(depth, min(distances))


def cutout_and_trim(
    plane: Plane,
    wall_corners: list[list[float]],
    slant_xz: Polygon | None = None,
    min_cutout_depth_m: float = MIN_DORMER_CUTOUT_DEPTH_M,
) -> tuple[list[list[float]], list[list[list[float]]], list[list[float]]] | None:
    """Reconstruct cutout, cheeks, header for a dormer against a slant plane.

    Returned tuple: (cutout_quad, cheek_quads, header_quad). Cheek quads are
    triangles (3 corners) whose back-apex sits on the slant at the wall apex y
    (or at the slant edge if `slant_xz` is provided and the slant ends sooner);
    the header back edge sits on the slant level line at that same y.

    `slant_xz` is the parent oblique's XZ polygon. When supplied, each side's
    back-depth is clamped to where a ray cast along the slant gradient leaves
    the slant boundary — so on shallow pitches the dormer doesn't extend past
    the actual slant.
    """
    b0, b1 = _bottom_edge(wall_corners)
    top_y = max(p[1] for p in wall_corners)
    # The wall's bottom edge in XZ defines the dormer's eave. If the two
    # bottom corners coincide in XZ, the wall is degenerate (a vertical
    # line, not a face) and there's no dormer to reconstruct.
    if hypot(b0[0] - b1[0], b0[2] - b1[2]) < MIN_DORMER_DEPTH_M:
        return None
    grad_x = -plane.a / plane.b
    grad_z = -plane.c / plane.b
    grad_len = hypot(grad_x, grad_z)
    if grad_len < 1e-9:
        return None
    up_x = grad_x / grad_len
    up_z = grad_z / grad_len

    front = [(b0[0], b0[2]), (b1[0], b1[2])]
    front_ys = [plane.y_at(x, z) for x, z in front]
    if any(y is None for y in front_ys):
        return None

    clearances = [(top_y - float(y)) / grad_len for y in front_ys]
    if slant_xz is not None and not slant_xz.is_empty:
        slant_xz = slant_xz.buffer(SLANT_XZ_CUTOUT_BUFFER_M)
        clearances = [
            _clamp_to_slant(front[i], up_x, up_z, clearances[i], slant_xz)
            for i in (0, 1)
        ]
    if min(clearances) < min_cutout_depth_m:
        return None

    front_on_slant = [
        [front[0][0], float(front_ys[0]), front[0][1]],
        [front[1][0], float(front_ys[1]), front[1][1]],
    ]
    back_xz = [
        (front[i][0] + up_x * clearances[i], front[i][1] + up_z * clearances[i])
        for i in (0, 1)
    ]
    back_y = [plane.y_at(x, z) for x, z in back_xz]
    if any(y is None for y in back_y):
        return None
    # Snap back_y to top_y when within float tolerance so the cutout's back
    # corners and the header's back corners are the same vertex when the
    # slant fully reaches top_y. When clamped, back_y is meaningfully below
    # top_y and the two diverge into the cutout-on-slant + header-at-top_y pair.
    back_y = [
        float(top_y) if abs(float(y) - top_y) < 1e-9 else float(y) for y in back_y
    ]
    back_on_slant = [[back_xz[i][0], back_y[i], back_xz[i][1]] for i in (0, 1)]
    front_top = [
        [front[0][0], top_y, front[0][1]],
        [front[1][0], top_y, front[1][1]],
    ]
    back_top = [[back_xz[i][0], top_y, back_xz[i][1]] for i in (0, 1)]

    cutout = [front_on_slant[0], front_on_slant[1], back_on_slant[1], back_on_slant[0]]
    header = [front_top[0], front_top[1], back_top[1], back_top[0]]
    cheeks = [
        _cheek_polygon(front_on_slant[0], front_top[0], back_top[0], back_on_slant[0]),
        _cheek_polygon(front_on_slant[1], front_top[1], back_top[1], back_on_slant[1]),
    ]
    return cutout, cheeks, header


def _cheek_polygon(
    front_bottom: list[float],
    front_top: list[float],
    back_top: list[float],
    back_bottom: list[float],
) -> list[list[float]]:
    """Return a triangle when the back-top and back-bottom coincide
    (slant fully reached at top_y), otherwise a 4-corner trapezoid."""
    if all(abs(back_top[i] - back_bottom[i]) < 1e-6 for i in range(3)):
        return [front_bottom, front_top, back_top]
    return [front_bottom, front_top, back_top, back_bottom]


def detect_dormers(
    model: BuildingModel, obliques: list[ObliqueSurface]
) -> list[DormerCandidate]:
    candidates: list[DormerCandidate] = []
    for surface_idx, surface in enumerate(obliques):
        ridge_x = surface.ridge["x"]
        ridge_z = surface.ridge["z"]
        ridge_span = surface.ridge["max"] - surface.ridge["min"]
        if surface.cluster.avg_incl < MIN_PARENT_INCLINATION_DEG:
            continue
        if ridge_span < MIN_PARENT_RIDGE_SPAN_M:
            continue
        slant_xz = _surface_xz_polygon(surface)
        for room in model.rooms:
            if room.story != surface.dominant_story:
                continue
            room_walls: list[ExtractedWall] = []
            seen_ids: set[str] = set()
            for wall in [*room.walls_computed, *room.walls_merged]:
                if wall.id in seen_ids:
                    continue
                seen_ids.add(wall.id)
                room_walls.append(wall)
            for wall in room_walls:
                if len(wall.corners) < 3:
                    continue
                normal_x, normal_z = wall_normal_xz(wall.corners)
                if abs(normal_x * ridge_x + normal_z * ridge_z) > PERPENDICULAR_DOT:
                    continue
                front_openings = _front_openings(wall, room)
                if not front_openings:
                    continue
                if _wall_under_higher_floor(wall, room, model.rooms):
                    continue
                opening_distances: list[float] = []
                if slant_xz is not None:
                    center_x, center_z = _wall_center_xz(wall.corners)
                    distances = [slant_xz.distance(Point(center_x, center_z))]
                    opening_distances = [
                        slant_xz.distance(Point(*_opening_center_xz(opening)))
                        for opening in front_openings
                    ]
                    distances.extend(opening_distances)
                    if min(distances) > MAX_WALL_CENTER_FROM_SLANT_XZ_M:
                        continue
                    nearest_opening_distance = min(opening_distances or distances)
                    if nearest_opening_distance > MAX_FRONT_OPENING_FROM_SLANT_XZ_M:
                        continue
                # The wall's bottom edge defines the dormer eave, but top
                # protrusion has to be measured where the top edge actually is.
                # Knee walls often lean or get scanned with a top edge shifted
                # slightly up the slant; measuring at the bottom edge turns the
                # roof rise across that wall into false dormer protrusion.
                b0, b1 = _bottom_edge(wall.corners)
                slant_at_b0 = surface.plane.y_at(b0[0], b0[2])
                slant_at_b1 = surface.plane.y_at(b1[0], b1[2])
                if slant_at_b0 is None or slant_at_b1 is None:
                    continue
                bottom_above_slant = min(b0[1] - slant_at_b0, b1[1] - slant_at_b1)
                if bottom_above_slant > MAX_BOTTOM_ABOVE_SLANT_M:
                    continue
                t0, t1 = _top_edge(wall.corners)
                slant_at_t0 = surface.plane.y_at(t0[0], t0[2])
                slant_at_t1 = surface.plane.y_at(t1[0], t1[2])
                if slant_at_t0 is None or slant_at_t1 is None:
                    continue
                top_above_slant = min(t0[1] - slant_at_t0, t1[1] - slant_at_t1)
                if top_above_slant < MIN_TOP_ABOVE_SLANT_M:
                    continue
                width = _wall_width_along_ridge(wall.corners, ridge_x, ridge_z)
                if cutout_and_trim(surface.plane, wall.corners, slant_xz) is None:
                    continue

                for opening_idx, opening in enumerate(front_openings):
                    if (
                        slant_xz is not None
                        and opening_distances
                        and opening_distances[opening_idx]
                        > MAX_FRONT_OPENING_FROM_SLANT_XZ_M
                    ):
                        continue
                    opening_width = _opening_width_along_ridge(
                        opening, ridge_x, ridge_z
                    )
                    if (
                        ridge_span > 0
                        and width >= MAX_WIDTH_RATIO * ridge_span
                        and opening_width >= MAX_WIDTH_RATIO * ridge_span
                    ):
                        continue
                    if (
                        cutout_and_trim(surface.plane, opening.corners, slant_xz)
                        is None
                    ):
                        continue
                    candidates.append(
                        DormerCandidate(
                            roof_surface_index=surface_idx,
                            room_index=room.index,
                            front_wall_id=wall.id,
                            front_opening_id=opening.id,
                        )
                    )
    return candidates


def _surface_xz_polygon(surface: ObliqueSurface) -> Polygon | None:
    from reconcile_tiers._core.shapely2 import make_valid_polygon

    if len(surface.corners) < 3:
        return None
    poly = make_valid_polygon(
        Polygon([(float(p[0]), float(p[2])) for p in surface.corners])
    )
    if poly is None or poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly
