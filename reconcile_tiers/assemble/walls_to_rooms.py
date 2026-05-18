from __future__ import annotations

import logging
from collections.abc import Sequence
from math import sqrt

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import PlaneKey, fit_plane_any, plane_key
from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers._core.wall_axis import principal_axis_and_coverage
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedElement,
    ExtractedRoom,
)
from reconcile_tiers.payload.schema import HorizontalLid, Quad, Room, Vec3, Wall

PLANE_EPS_M = 0.05
PARENT_OPENING_PLANE_EPS_M = 0.10
ORPHAN_OPENING_PLANE_DRIFT_MAX_M = 0.30
EDGE_MARGIN_M = 0.01
OPENING_AREA_EPS_M2 = 1e-6
OPENING_OUTSIDE_FRACTION = 1e-3
WALL_DEDUP_CROSS_ROOM_FRACTION = 0.5
WALL_DEDUP_SAME_ROOM_FRACTION = 0.1
LOGGER = logging.getLogger(__name__)


def _vec3(corner: Sequence[float]) -> Vec3:
    return Vec3(x=float(corner[0]), y=float(corner[1]), z=float(corner[2]))


def _coords(corners: Sequence[Vec3]) -> list[list[float]]:
    return [[corner.x, corner.y, corner.z] for corner in corners]


def _dedupe_adjacent(corners: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for corner in corners:
        if (
            not out
            or sum((float(corner[idx]) - float(out[-1][idx])) ** 2 for idx in range(3))
            > 1e-10
        ):
            out.append(corner)
    if (
        len(out) >= 2
        and sum((float(out[0][idx]) - float(out[-1][idx])) ** 2 for idx in range(3))
        <= 1e-10
    ):
        out.pop()
    return out


def _merge_uplift_strips(strips: list[list[list[float]]]) -> list[list[float]] | None:
    valid = [strip for strip in strips if len(strip) == 4]
    if not valid:
        return None

    lower: list[list[float]] = []
    for strip in valid:
        if not lower:
            lower.append(strip[0])
        lower.append(strip[1])

    upper = [valid[-1][2]]
    upper.extend(strip[3] for strip in reversed(valid))
    merged = _dedupe_adjacent(lower + upper)
    return merged if len(merged) >= 3 else None


def _merge_descent_strips(strips: list[list[list[float]]]) -> list[list[float]] | None:
    valid = [strip for strip in strips if len(strip) == 4]
    if not valid:
        return None

    upper: list[list[float]] = []
    for strip in valid:
        if not upper:
            upper.append(strip[0])
        upper.append(strip[3])

    lower = [valid[-1][2]]
    lower.extend(strip[1] for strip in reversed(valid))
    merged = _dedupe_adjacent(upper + lower)
    return merged if len(merged) >= 3 else None


def _centroid(corners: Sequence[Sequence[float]]) -> list[float]:
    n = max(1, len(corners))
    return [sum(float(corner[axis]) for corner in corners) / n for axis in range(3)]


def _room_center(room: ExtractedRoom) -> list[float]:
    return _centroid(room.floor_polygon) if room.floor_polygon else [0.0, 0.0, 0.0]


def _orient_floor_up(corners: list[list[float]]) -> list[list[float]]:
    if len(corners) < 3:
        return corners
    return list(reversed(corners)) if newell_normal(corners)[1] <= 0.0 else corners


def _floor_polygon_xz(floor_polygon: list[list[float]]) -> Polygon | None:
    if len(floor_polygon) < 3:
        return None
    poly = Polygon([(float(corner[0]), float(corner[2])) for corner in floor_polygon])
    if not poly.is_valid:
        poly = make_valid(poly)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _orient_wall_outward(
    corners: list[list[float]],
    room_center: list[float],
    floor_polygon: list[list[float]] | None = None,
) -> list[list[float]]:
    if len(corners) < 3:
        return corners
    normal = newell_normal(corners)
    wall_center = _centroid(corners)
    floor_poly = _floor_polygon_xz(floor_polygon or [])
    nxz_len = sqrt(normal[0] * normal[0] + normal[2] * normal[2])
    if floor_poly is not None and nxz_len > 1e-12:
        step_m = 0.10
        nx = normal[0] / nxz_len
        nz = normal[2] / nxz_len
        along_normal = Point(wall_center[0] + nx * step_m, wall_center[2] + nz * step_m)
        opposite_normal = Point(
            wall_center[0] - nx * step_m, wall_center[2] - nz * step_m
        )
        normal_inside = floor_poly.covers(along_normal)
        opposite_inside = floor_poly.covers(opposite_normal)
        if normal_inside != opposite_inside:
            return list(reversed(corners)) if normal_inside else corners

    outward = [wall_center[idx] - room_center[idx] for idx in range(3)]
    if sum(normal[idx] * outward[idx] for idx in range(3)) < 0.0:
        return list(reversed(corners))
    return corners


def _projection_axes(corners: list[list[float]]) -> tuple[int, int]:
    nx, ny, nz = newell_normal(corners)
    anx, any_, anz = abs(nx), abs(ny), abs(nz)
    if any_ >= anx and any_ >= anz:
        return 0, 2
    if anx >= any_ and anx >= anz:
        return 1, 2
    return 0, 1


def _distance_to_wall_plane(
    normal: tuple[float, float, float], p0: list[float], point: list[float]
) -> float:
    norm = sqrt(sum(value * value for value in normal))
    if norm <= 1e-12:
        return float("inf")
    return (
        abs(sum(normal[idx] * (float(point[idx]) - p0[idx]) for idx in range(3))) / norm
    )


def _max_opening_distance_to_wall_plane(
    wall_corners: list[list[float]], opening: ExtractedElement
) -> float:
    normal = newell_normal(wall_corners)
    if sqrt(sum(value * value for value in normal)) <= 1e-12:
        return float("inf")
    p0 = [float(value) for value in wall_corners[0]]
    return max(_distance_to_wall_plane(normal, p0, point) for point in opening.corners)


def _valid_polygon(points: list[tuple[float, float]]) -> Polygon | None:
    poly = Polygon(points)
    if not poly.is_valid:
        poly = make_valid(poly)
    if (
        not isinstance(poly, Polygon)
        or poly.is_empty
        or poly.area <= OPENING_AREA_EPS_M2
    ):
        return None
    return poly


def _normalize(vec: Sequence[float]) -> list[float] | None:
    norm = sqrt(sum(float(value) * float(value) for value in vec))
    if norm <= 1e-12:
        return None
    return [float(value) / norm for value in vec]


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[idx]) * float(b[idx]) for idx in range(3))


def _wall_frame(
    corners: list[list[float]],
) -> tuple[list[float], list[float], list[float]] | None:
    if len(corners) < 3:
        return None
    normal = _normalize(newell_normal(corners))
    if normal is None:
        return None
    origin = [float(value) for value in corners[0]]
    u = None
    for corner in corners[1:]:
        candidate = [float(corner[idx]) - origin[idx] for idx in range(3)]
        u = _normalize(candidate)
        if u is not None:
            break
    if u is None:
        return None
    v = _normalize(_cross(normal, u))
    if v is None:
        return None
    return origin, u, v


def _project_frame_point(
    corner: Sequence[float],
    frame: tuple[list[float], list[float], list[float]],
) -> tuple[float, float]:
    origin, u, v = frame
    delta = [float(corner[idx]) - origin[idx] for idx in range(3)]
    return (_dot(delta, u), _dot(delta, v))


def _unproject_frame_point(
    point: tuple[float, float],
    frame: tuple[list[float], list[float], list[float]],
) -> list[float]:
    origin, u, v = frame
    return [
        origin[idx] + u[idx] * float(point[0]) + v[idx] * float(point[1])
        for idx in range(3)
    ]


def _clamp_opening_to_wall(
    opening: ExtractedElement, wall_corners: list[list[float]]
) -> ExtractedElement | None:
    frame = _wall_frame(wall_corners)
    if frame is None:
        return None
    wall_points = [_project_frame_point(corner, frame) for corner in wall_corners]
    wall_poly = _valid_polygon(wall_points)
    if wall_poly is None:
        return None

    opening_points = [_project_frame_point(corner, frame) for corner in opening.corners]
    opening_poly = _valid_polygon(opening_points)
    if opening_poly is None:
        return None

    outside_area = opening_poly.difference(wall_poly).area
    if outside_area <= OPENING_AREA_EPS_M2:
        clamped_points = opening_points
    else:
        clamped_points = []
        for point in opening_points:
            p = Point(point)
            if wall_poly.covers(p):
                clamped_points.append(point)
                continue
            _src, nearest = nearest_points(p, wall_poly.exterior)
            clamped_points.append((float(nearest.x), float(nearest.y)))

    clamped_poly = _valid_polygon(clamped_points)
    if clamped_poly is None:
        return None
    clamped_outside = clamped_poly.difference(wall_poly).area
    if clamped_outside > max(
        OPENING_AREA_EPS_M2, OPENING_OUTSIDE_FRACTION * clamped_poly.area
    ):
        return None

    return ExtractedElement(
        id=opening.id,
        corners=[_unproject_frame_point(point, frame) for point in clamped_points],
        source=opening.source,
        parent_wall_id=opening.parent_wall_id,
    )


def reclip_cutouts_to_wall(
    wall_corners: list[list[float]], cutouts: list[Quad]
) -> list[Quad]:
    """Reproject cutouts onto a (possibly mutated) wall plane and clamp each
    corner onto the new wall outline. Use after a wall transform has changed
    `wall.corners` so the cutouts no longer track the new outline (e.g. after
    `_clip_wall_to_gable_roof` truncates a top-story wall under a gable).

    Cutouts that already fit inside the wall are reprojected onto the wall
    plane. Cutouts that lie partly outside are replaced by the corner-clamped
    quad — same primitive as `_clamp_opening_to_wall`.
    Cutouts that no longer overlap the wall at all are dropped. The result is
    always a list of 4-corner Quads, matching the payload schema.
    """
    if not cutouts:
        return []
    frame = _wall_frame(wall_corners)
    if frame is None:
        return list(cutouts)
    wall_poly = _valid_polygon([_project_frame_point(c, frame) for c in wall_corners])
    if wall_poly is None:
        return []
    out: list[Quad] = []
    for cutout in cutouts:
        cor3d = _coords(cutout.corners)
        cut_points = [_project_frame_point(c, frame) for c in cor3d]
        cut_poly = _valid_polygon(cut_points)
        if cut_poly is None:
            continue
        outside_area = cut_poly.difference(wall_poly).area
        if outside_area <= max(
            OPENING_AREA_EPS_M2, OPENING_OUTSIDE_FRACTION * cut_poly.area
        ):
            clamped_points = cut_points
        else:
            clamped_points = []
            for point in cut_points:
                p = Point(point)
                if wall_poly.covers(p):
                    clamped_points.append(point)
                    continue
                _src, nearest = nearest_points(p, wall_poly.exterior)
                clamped_points.append((float(nearest.x), float(nearest.y)))
        clamped_poly = _valid_polygon(clamped_points)
        if clamped_poly is None:
            continue
        if clamped_poly.area <= OPENING_AREA_EPS_M2:
            continue
        new_corners = [_unproject_frame_point(p, frame) for p in clamped_points]
        out.append(Quad(corners=[_vec3(p) for p in new_corners]))
    return out


def _opening_inside_wall(
    wall_corners: list[list[float]], opening: ExtractedElement
) -> bool:
    if len(opening.corners) != 4 or len(wall_corners) < 3:
        return False
    normal = newell_normal(wall_corners)
    if sqrt(sum(value * value for value in normal)) <= 1e-12:
        return False
    p0 = [float(value) for value in wall_corners[0]]
    if any(
        _distance_to_wall_plane(normal, p0, point) > PLANE_EPS_M
        for point in opening.corners
    ):
        return False

    axis0, axis1 = _projection_axes(wall_corners)
    outer = [(float(corner[axis0]), float(corner[axis1])) for corner in wall_corners]
    wall_poly = _valid_polygon(outer)
    if wall_poly is None:
        return False

    opening_poly = _valid_polygon(
        [(float(corner[axis0]), float(corner[axis1])) for corner in opening.corners]
    )
    if opening_poly is None:
        return False

    allowed_wall = wall_poly.buffer(EDGE_MARGIN_M)
    outside_area = opening_poly.difference(allowed_wall).area
    return outside_area <= max(
        OPENING_AREA_EPS_M2, OPENING_OUTSIDE_FRACTION * opening_poly.area
    )


def _quad(element: ExtractedElement) -> Quad:
    return Quad(corners=[_vec3(corner) for corner in element.corners])


def _opening_on_wall_plane(
    wall_corners: list[list[float]],
    opening: ExtractedElement,
    *,
    max_distance_m: float = PLANE_EPS_M,
) -> bool:
    if len(opening.corners) != 4 or len(wall_corners) < 3:
        return False
    normal = newell_normal(wall_corners)
    if sqrt(sum(value * value for value in normal)) <= 1e-12:
        return False
    p0 = [float(value) for value in wall_corners[0]]
    return all(
        _distance_to_wall_plane(normal, p0, point) <= max_distance_m
        for point in opening.corners
    )


def _wall_cutout_element(
    wall_corners: list[list[float]],
    opening: ExtractedElement,
    *,
    force_parent_clamp: bool = False,
) -> ExtractedElement | None:
    if force_parent_clamp:
        if not _opening_on_wall_plane(
            wall_corners,
            opening,
            max_distance_m=PARENT_OPENING_PLANE_EPS_M,
        ):
            return None
    elif not _opening_inside_wall(wall_corners, opening):
        return None
    return _clamp_opening_to_wall(opening, wall_corners)


def _wall_cutout(
    wall_corners: list[list[float]], opening: ExtractedElement
) -> Quad | None:
    clamped = _wall_cutout_element(wall_corners, opening)
    return _quad(clamped) if clamped is not None else None


def _wall_id_from_locator(locator_id: str) -> str | None:
    """Extract the original wall id from a tier-wall locator. Returns None for
    perimeter-synth walls (no parent wall id) and walls whose locator doesn't
    follow the expected `{uuid}::tier-wall::{room}:{wall_id}[suffix]` shape.
    """
    tail = locator_id.rsplit("::tier-wall::", 1)[-1]
    parts = tail.split(":", 2)
    if len(parts) < 2:
        return None
    wall_id = parts[1]
    if wall_id.startswith("perimeter-synth"):
        return None
    return wall_id


def _opening_inside_wall_frame(
    wall_corners: list[list[float]], opening: ExtractedElement
) -> bool:
    """Plane-agnostic containment: project both the wall and the opening into
    the wall's 2D frame (the wall plane's local coords) and test that the
    opening fits inside the wall outline. The perpendicular offset is dropped
    by projection, so this works even when the opening's plane has drifted
    past the 5 cm clamp tolerance — the geometric fallback for orphans
    without a `parent_wall_id` link.
    """
    if len(opening.corners) != 4 or len(wall_corners) < 3:
        return False
    frame = _wall_frame(wall_corners)
    if frame is None:
        return False
    wall_poly = _valid_polygon([_project_frame_point(c, frame) for c in wall_corners])
    if wall_poly is None:
        return False
    opening_poly = _valid_polygon(
        [_project_frame_point(c, frame) for c in opening.corners]
    )
    if opening_poly is None:
        return False
    outside = opening_poly.difference(wall_poly.buffer(EDGE_MARGIN_M)).area
    return outside <= max(
        OPENING_AREA_EPS_M2, OPENING_OUTSIDE_FRACTION * opening_poly.area
    )


def _attach_orphaned_openings(
    walls: list[Wall],
    opening_refs: list[tuple[str, int, ExtractedElement]],
    clamped_openings: dict[tuple[str, int], ExtractedElement],
) -> list[Wall]:
    """Second pass for openings that no wall absorbed in the first pass.

    The first pass uses a 5 cm plane-distance gate which fails when wall
    geometry has drifted (clipped, snapped, synth-replaced) past the scan
    window. This pass relinks orphans using ground-truth signals — first the
    `parent_wall_id` if it points at a surviving wall, then a plane-agnostic
    XZ containment check. Both feed into the same `_clamp_opening_to_wall`
    primitive, so the cutout sits on the wall plane and inside its outline.
    """
    if not opening_refs:
        return walls
    out_walls = list(walls)
    for opening_kind, opening_idx, opening in opening_refs:
        if (opening_kind, opening_idx) in clamped_openings:
            continue
        target_idx: int | None = None
        if opening.parent_wall_id:
            for i, wall in enumerate(out_walls):
                if wall.synthetic:
                    continue
                if _wall_id_from_locator(wall.locator_id) == opening.parent_wall_id:
                    target_idx = i
                    break
        if target_idx is None:
            best: tuple[tuple[float, float], int] | None = None
            for i, wall in enumerate(out_walls):
                if wall.synthetic:
                    continue
                wall_corners = _coords(wall.corners)
                if not _opening_inside_wall_frame(wall_corners, opening):
                    continue
                max_plane_distance = _max_opening_distance_to_wall_plane(
                    wall_corners, opening
                )
                if max_plane_distance > ORPHAN_OPENING_PLANE_DRIFT_MAX_M:
                    continue
                wall_normal = newell_normal(wall_corners)
                opening_normal = newell_normal(
                    [[float(c) for c in corner] for corner in opening.corners]
                )
                wnorm = sqrt(sum(v * v for v in wall_normal))
                onorm = sqrt(sum(v * v for v in opening_normal))
                if wnorm <= 1e-9 or onorm <= 1e-9:
                    continue
                cos_ang = abs(
                    sum(wall_normal[k] * opening_normal[k] for k in range(3))
                ) / (wnorm * onorm)
                cos_ang = min(1.0, cos_ang)
                if cos_ang < 0.985:  # ~10 deg
                    continue
                score = (max_plane_distance, -cos_ang)
                if best is None or score < best[0]:
                    best = (score, i)
            if best is not None:
                target_idx = best[1]
        if target_idx is None:
            continue
        wall = out_walls[target_idx]
        wall_corners = _coords(wall.corners)
        clamped = _clamp_opening_to_wall(opening, wall_corners)
        if clamped is None:
            continue
        from dataclasses import replace as _replace

        out_walls[target_idx] = _replace(wall, cutouts=[*wall.cutouts, _quad(clamped)])
        clamped_openings[(opening_kind, opening_idx)] = clamped
    return out_walls


def _quad_openings(
    openings: list[ExtractedElement],
    *,
    model_uuid: str,
    room_index: int,
    kind: str,
) -> list[ExtractedElement]:
    quads: list[ExtractedElement] = []
    for opening in openings:
        if len(opening.corners) == 4:
            quads.append(opening)
            continue
        LOGGER.warning(
            "Skipping non-quad %s opening in tier payload assembly: "
            "uuid=%s room=%s id=%s corner_count=%s",
            kind,
            model_uuid,
            room_index,
            opening.id,
            len(opening.corners),
        )
    return quads


def _project_to_plane(
    corners: Sequence[Sequence[float]], normal: Sequence[float]
) -> Polygon | None:
    import numpy as np  # local import preserved across formatter passes

    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return None
    n = n / norm
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        return None
    u = u / u_norm
    v = np.cross(n, u)
    pts = np.asarray(corners, dtype=float)
    flat = [(float(np.dot(p, u)), float(np.dot(p, v))) for p in pts]
    if len(flat) < 3:
        return None
    poly = Polygon(flat)
    if not poly.is_valid:
        poly = make_valid(poly)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area < 1e-6:
        return None
    return poly


def _wall_has_valid_plane_local_area(wall: Wall) -> bool:
    corners3 = _coords(wall.corners)
    plane = fit_plane_any(corners3)
    if plane is None:
        return False
    return _project_to_plane(corners3, plane[:3]) is not None


def _wall_plane_key(corners: Sequence[Sequence[float]]) -> PlaneKey | None:
    plane = fit_plane_any(corners)
    if plane is None:
        return None
    return plane_key(plane)


def dedup_room_walls(rooms: list[Room], *, cross_room: bool = True) -> list[Room]:
    pending: list[tuple[int, int, int, PlaneKey, Polygon, Wall]] = []
    by_room: dict[tuple[int, int], list[Wall]] = {}
    for story_idx, room in enumerate(rooms):
        by_room[(room.story, story_idx)] = list(room.walls)
        for wall_idx, wall in enumerate(room.walls):
            corners3 = [[c.x, c.y, c.z] for c in wall.corners]
            key = _wall_plane_key(corners3)
            plane = fit_plane_any(corners3)
            if key is None or plane is None:
                continue
            poly2d = _project_to_plane(corners3, plane[:3])
            if poly2d is None:
                continue
            pending.append((room.story, story_idx, wall_idx, key, poly2d, wall))

    pending.sort(
        key=lambda item: (
            -len(item[5].cutouts),
            -item[4].area,
            item[0],
            item[1],
            item[2],
        )
    )
    claimed: dict[PlaneKey, list[tuple[tuple[int, int], Polygon]]] = {}
    drop: set[tuple[int, int, int]] = set()
    for story, room_pos, wall_idx, key, poly2d, _wall in pending:
        owner_room = (story, room_pos)
        prev_list = claimed.get(key, [])
        dropped = False
        for prev_room, prev_poly in prev_list:
            if not cross_room and prev_room != owner_room:
                continue
            try:
                inter = poly2d.intersection(prev_poly).area
            except Exception:
                inter = 0.0
            if inter <= 0.0:
                continue
            threshold = (
                WALL_DEDUP_SAME_ROOM_FRACTION
                if prev_room == owner_room
                else WALL_DEDUP_CROSS_ROOM_FRACTION
            ) * poly2d.area
            if inter >= threshold:
                drop.add((story, room_pos, wall_idx))
                dropped = True
                break
        if dropped:
            continue
        claimed.setdefault(key, []).append((owner_room, poly2d))

    out: list[Room] = []
    for story_idx, room in enumerate(rooms):
        kept_walls = [
            wall
            for wall_idx, wall in enumerate(room.walls)
            if (room.story, story_idx, wall_idx) not in drop
            and _wall_has_valid_plane_local_area(wall)
        ]
        if len(kept_walls) == len(room.walls):
            out.append(room)
        else:
            out.append(
                Room(
                    story=room.story,
                    floor=room.floor,
                    walls=kept_walls,
                    doors=room.doors,
                    windows=room.windows,
                    locator_id=room.locator_id,
                    heating=room.heating,
                )
            )
    return out


def assemble_rooms(model: BuildingModel) -> list[Room]:
    rooms: list[Room] = []
    wall_axis = principal_axis_and_coverage(
        wall.corners for room in model.rooms for wall in room.walls_computed
    )
    building_boundary = _building_footprint_boundary(model)
    all_wall_lines = [
        line
        for room in model.rooms
        for wall in [*room.walls_computed, *room.synthetic_walls]
        if (line := _line_from_extracted_wall(wall)) is not None
    ]
    gap_lines = _gap_context_lines(model)
    boundary_endpoints = (
        _boundary_wall_endpoints(all_wall_lines, building_boundary)
        if building_boundary is not None
        else []
    )
    for room in model.rooms:
        if len(room.floor_polygon) < 3:
            continue
        room_center = _room_center(room)
        windows = _quad_openings(
            room.windows,
            model_uuid=model.uuid,
            room_index=room.index,
            kind="window",
        )
        doors = _quad_openings(
            room.doors,
            model_uuid=model.uuid,
            room_index=room.index,
            kind="door",
        )
        opening_refs = [
            *[("window", idx, window) for idx, window in enumerate(windows)],
            *[("door", idx, door) for idx, door in enumerate(doors)],
        ]
        clamped_openings: dict[tuple[str, int], ExtractedElement] = {}
        walls: list[Wall] = []
        for wall in [*room.walls_computed, *room.synthetic_walls]:
            wall_corners = _orient_wall_outward(
                [
                    [float(corner[0]), float(corner[1]), float(corner[2])]
                    for corner in wall.corners
                ],
                room_center,
                room.floor_polygon,
            )
            cutouts: list[Quad] = []
            if not wall.synthetic:
                for opening_kind, opening_idx, opening in opening_refs:
                    clamped = _wall_cutout_element(
                        wall_corners,
                        opening,
                        force_parent_clamp=opening.parent_wall_id == wall.id,
                    )
                    if clamped is None:
                        continue
                    cutouts.append(_quad(clamped))
                    clamped_openings.setdefault((opening_kind, opening_idx), clamped)
            descent_strip = None
            if wall.descent_strip:
                strip = _merge_descent_strips(wall.descent_strip)
                if strip:
                    descent_strip = [
                        _vec3(corner)
                        for corner in _orient_wall_outward(
                            strip, room_center, room.floor_polygon
                        )
                    ]
            uplift_strip = None
            if wall.uplift_strip:
                strip = _merge_uplift_strips(wall.uplift_strip)
                if strip:
                    uplift_strip = [
                        _vec3(corner)
                        for corner in _orient_wall_outward(
                            strip, room_center, room.floor_polygon
                        )
                    ]
            walls.append(
                Wall(
                    corners=[_vec3(corner) for corner in wall_corners],
                    descent_strip=descent_strip,
                    uplift_strip=uplift_strip,
                    cutouts=cutouts,
                    locator_id=f"{model.uuid}::tier-wall::{room.index}:{wall.id}",
                    synthetic=wall.synthetic,
                )
            )
        walls = _synthesise_missing_perimeter_walls(
            model=model,
            room=room,
            walls=walls,
            room_center=room_center,
            building_boundary=building_boundary,
            all_wall_lines=all_wall_lines,
            gap_lines=gap_lines,
            boundary_endpoints=boundary_endpoints,
        )
        walls = _rectilinearize_short_oblique_notches(
            walls,
            floor_polygon=room.floor_polygon,
            room_center=room_center,
            wall_axis=wall_axis,
        )
        walls = _snap_short_parallel_caps_to_longer_run(walls)
        walls = [wall for wall in walls if _wall_has_valid_plane_local_area(wall)]
        walls = _attach_orphaned_openings(walls, opening_refs, clamped_openings)
        visible_doors = [
            clamped_openings.get(("door", idx), door) for idx, door in enumerate(doors)
        ]
        visible_windows = [
            clamped_openings.get(("window", idx), window)
            for idx, window in enumerate(windows)
        ]
        rooms.append(
            Room(
                story=room.story,
                floor=[
                    HorizontalLid(
                        corners=[
                            _vec3(corner)
                            for corner in _orient_floor_up(room.floor_polygon)
                        ]
                    )
                ],
                walls=walls,
                doors=[_quad(door) for door in visible_doors],
                windows=[_quad(window) for window in visible_windows],
                locator_id=f"{model.uuid}::tier-room::{room.index}",
                heating=room.heating,
            )
        )
    return dedup_room_walls(rooms)


# Re-exports — perimeter completion and rectilinear-wall synthesis live in
# `_wall_perimeter`. `assemble_rooms` calls them through this module so the
# perimeter pass stays a single import in tests and sibling modules.
from reconcile_tiers.assemble._wall_perimeter import (  # noqa: E402, F401
    PERIMETER_SYNTH_BOUNDARY_COVERAGE_MIN,
    PERIMETER_SYNTH_BOUNDARY_TOL_M,
    PERIMETER_SYNTH_COVERAGE_TOL_M,
    PERIMETER_SYNTH_ENDPOINT_TOL_M,
    PERIMETER_SYNTH_GAP_COVERAGE_MAX,
    PERIMETER_SYNTH_MIN_LENGTH_M,
    PERIMETER_SYNTH_SAMPLE_STEP_M,
    PERIMETER_SYNTH_WALL_COVERAGE_MAX,
    RECTILINEAR_AXIS_TOL_DEG,
    RECTILINEAR_CAP_MAX_OFFSET_M,
    RECTILINEAR_CAP_MAX_WALL_M,
    RECTILINEAR_CAP_MIN_LENGTH_RATIO,
    RECTILINEAR_CAP_MIN_OFFSET_M,
    RECTILINEAR_CAP_MIN_OVERLAP_FRACTION,
    RECTILINEAR_CAP_PARALLEL_TOL_DEG,
    RECTILINEAR_COVERAGE_MIN,
    RECTILINEAR_NOTCH_ENDPOINT_TOL_M,
    RECTILINEAR_NOTCH_MAX_LEG_M,
    RECTILINEAR_NOTCH_MAX_WALL_M,
    RECTILINEAR_NOTCH_MIN_LEG_M,
    RECTILINEAR_NOTCH_MIN_SUPPORT_M,
    RECTILINEAR_OFF_AXIS_MIN_DEG,
    _angle_mod90_delta,
    _boundary_coverage_fraction,
    _boundary_wall_endpoints,
    _building_footprint_boundary,
    _coverage_fraction,
    _dist_xz,
    _gap_context_lines,
    _line_from_extracted_wall,
    _line_from_wall,
    _line_intersection_xz,
    _make_rectilinear_wall,
    _polygon_edge_lines,
    _project_point_to_line_xz,
    _rectilinearize_short_oblique_notches,
    _room_wall_y_extent,
    _snap_short_parallel_caps_to_longer_run,
    _support_candidates,
    _synthesise_missing_perimeter_walls,
    _unit_xz,
    _xz_angle_deg,
    _xz_span,
)
