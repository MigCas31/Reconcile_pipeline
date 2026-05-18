from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from reconcile_tiers._core.footprint import (
    is_xz_supported_by_story_footprint,
    story_footprint_context,
)
from reconcile_tiers.extract.building import ExtractedRoom, ExtractedStitchWall
from reconcile_tiers.extract.room_ceiling import RoomCeiling, room_ceiling

MAX_GAP_M = 1.50
MIN_GAP_M = 0.06
T_JUNCTION_MIN_GAP_M = 0.005
T_JUNCTION_MAX_GAP_M = 0.08
T_JUNCTION_MIN_EDGE_MARGIN_M = 0.15
T_JUNCTION_MAX_PARALLEL_COS = 0.50
MIN_WALL_HEIGHT_M = 0.50
HEIGHT_RATIO = 0.65
MUTUAL_THRESH_M = 0.60
DIRECT_QUAD_MAX_GAP_M = 0.15
MIN_LEG_LENGTH_M = 0.06
WALL_OVERLAP_BUFFER_M = 0.03
WALL_OVERLAP_MIN_M = 0.10
WALL_OVERLAP_MAX_RATIO = 0.50
L_STITCH_CAP_MIN_AREA_M2 = 0.01
INFERRED_VOID_ROOM_SOURCE = "inferred-enclosed-void"
L_STITCH_PARALLEL_SOURCE_MIN_COS = 0.95
L_STITCH_BACK_EXTENSION_MIN_M = 0.50
L_STITCH_PARALLEL_OFFSET_MAX_M = 0.30
L_STITCH_PARALLEL_OFFSET_MAX_RATIO = 0.35
WALL_LOCAL_XZ_TOL_M = 1e-4
OWNER_WALL_TOP_SEARCH_M = 0.12


def _select_lower_room_ceiling(
    rooms: list[ExtractedRoom],
    room_idx1: int,
    room_idx2: int,
    x: float,
    z: float,
    cache: dict[int, RoomCeiling | None],
) -> RoomCeiling | None:
    """Pick the lower of the two rooms' ceilings at the given XZ.
    Returns None if neither room has a usable ceiling polygon — caller
    should skip emitting the cap (no scan support, no synthesis)."""

    def get_ceiling(idx: int) -> RoomCeiling | None:
        if idx not in cache:
            cache[idx] = room_ceiling(rooms[idx])
        return cache[idx]

    rc1 = get_ceiling(room_idx1)
    rc2 = get_ceiling(room_idx2)
    y1 = rc1.y_at(x, z) if rc1 is not None else None
    y2 = rc2.y_at(x, z) if rc2 is not None else None
    if y1 is None and y2 is None:
        return None
    if y1 is None:
        return rc2
    if y2 is None:
        return rc1
    return rc1 if y1 <= y2 else rc2


def _wall_height(corners: list[list[float]]) -> float:
    if not corners:
        return 0.0
    ys = [float(corner[1]) for corner in corners]
    return max(ys) - min(ys)


def _wall_endpoint_y_range(
    corners: list[list[float]], endpoint_idx: int
) -> tuple[float, float]:
    ys = [float(corner[1]) for corner in corners]
    if not ys:
        return 0.0, 0.0
    if endpoint_idx >= len(corners):
        return min(ys), max(ys)
    x = float(corners[endpoint_idx][0])
    z = float(corners[endpoint_idx][2])
    local = [
        float(corner[1])
        for corner in corners
        if math.hypot(float(corner[0]) - x, float(corner[2]) - z) <= WALL_LOCAL_XZ_TOL_M
    ]
    if not local:
        return min(ys), max(ys)
    return min(min(local), min(ys)), max(local)


def _wall_y_range_at_xz(
    corners: list[list[float]], x: float, z: float
) -> tuple[float, float] | None:
    if len(corners) < 2:
        return None
    x0, z0 = float(corners[0][0]), float(corners[0][2])
    x1, z1 = float(corners[1][0]), float(corners[1][2])
    dx = x1 - x0
    dz = z1 - z0
    length2 = dx * dx + dz * dz
    if length2 <= 1e-12:
        return None
    t = ((float(x) - x0) * dx + (float(z) - z0) * dz) / length2
    t = max(0.0, min(1.0, t))
    y0_low, y0_high = _wall_endpoint_y_range(corners, 0)
    y1_low, y1_high = _wall_endpoint_y_range(corners, 1)
    return (
        y0_low + t * (y1_low - y0_low),
        y0_high + t * (y1_high - y0_high),
    )


def _wall_y_range_near_xz(
    corners: list[list[float]],
    x: float,
    z: float,
    max_distance_m: float = OWNER_WALL_TOP_SEARCH_M,
) -> tuple[float, float] | None:
    if len(corners) < 2:
        return None
    line = LineString(
        [
            (float(corners[0][0]), float(corners[0][2])),
            (float(corners[1][0]), float(corners[1][2])),
        ]
    )
    if line.length <= 1e-9 or line.distance(Point(float(x), float(z))) > max_distance_m:
        return None
    return _wall_y_range_at_xz(corners, x, z)


def _owner_local_top_y(
    rooms: list[ExtractedRoom],
    owner_rooms: set[int],
    x: float,
    z: float,
    fallback: float,
) -> float:
    tops: list[float] = []
    for room_idx in owner_rooms:
        if room_idx < 0 or room_idx >= len(rooms):
            continue
        for wall in rooms[room_idx].walls_computed:
            if wall.synthetic:
                continue
            y_range = _wall_y_range_near_xz(wall.corners, x, z)
            if y_range is not None:
                tops.append(y_range[1])
    return min([fallback, *tops])


def _stitch_quad(
    a: tuple[float, float],
    b: tuple[float, float],
    y_bot: float,
    y_top_a: float,
    y_top_b: float,
) -> list[list[float]] | None:
    if min(y_top_a, y_top_b) - y_bot < MIN_WALL_HEIGHT_M:
        return None
    ax, az = a
    bx, bz = b
    return [
        [ax, y_bot, az],
        [bx, y_bot, bz],
        [bx, y_top_b, bz],
        [ax, y_top_a, az],
    ]


def _wall_dir_xz(corners: list[list[float]]) -> tuple[float, float]:
    dx = float(corners[1][0]) - float(corners[0][0])
    dz = float(corners[1][2]) - float(corners[0][2])
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return 0.0, 0.0
    return dx / length, dz / length


def _build_l_candidate(
    p_anchor: tuple[float, float],
    p_other: tuple[float, float],
    dir_anchor_out: tuple[float, float],
    dir_other_out: tuple[float, float],
):
    ax, az = p_anchor
    ox, oz = p_other
    along = (ox - ax) * dir_anchor_out[0] + (oz - az) * dir_anchor_out[1]
    if along < 0.0:
        return None
    cx = ax + along * dir_anchor_out[0]
    cz = az + along * dir_anchor_out[1]
    if (cx - ox) * dir_other_out[0] + (cz - oz) * dir_other_out[1] < 0.0:
        return None
    perp_len = math.hypot(
        (ox - ax) - along * dir_anchor_out[0],
        (oz - az) - along * dir_anchor_out[1],
    )
    source_cos = abs(
        dir_anchor_out[0] * dir_other_out[0] + dir_anchor_out[1] * dir_other_out[1]
    )
    if (
        source_cos >= L_STITCH_PARALLEL_SOURCE_MIN_COS
        and along >= L_STITCH_BACK_EXTENSION_MIN_M
        and MIN_LEG_LENGTH_M <= perp_len <= L_STITCH_PARALLEL_OFFSET_MAX_M
        and perp_len / max(along, 1e-9) <= L_STITCH_PARALLEL_OFFSET_MAX_RATIO
    ):
        return None
    legs = []
    if along >= MIN_LEG_LENGTH_M:
        legs.append(((ax, az), (cx, cz), "anchor"))
    if perp_len >= MIN_LEG_LENGTH_M:
        legs.append(((cx, cz), (ox, oz), "other"))
    return {"corner": (cx, cz), "legs": legs, "total_len": along + perp_len}


def _projected_xz_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    area2 = 0.0
    for idx, corner in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        area2 += float(corner[0]) * float(nxt[2]) - float(nxt[0]) * float(corner[2])
    return abs(area2) * 0.5


def _triangle_xz_area(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) * 0.5


def _xz_key(corner: list[float]) -> tuple[int, int]:
    return round(float(corner[0]) * 10000), round(float(corner[2]) * 10000)


def stitch_closure_groups(stitches: list[ExtractedStitchWall]) -> list[set[int]]:
    """Group stitches that belong to the same L-closure (legs + caps).

    Public so downstream stages (e.g. assembly) can drop a whole closure
    atomically when one piece fails validation."""

    groups = [{idx} for idx in range(len(stitches))]
    cap_point_sets: list[tuple[int, frozenset[tuple[int, int]]]] = []
    for idx, stitch in enumerate(stitches):
        if stitch.type not in {"stitch_floor", "stitch_ceiling"}:
            continue
        cap_points = frozenset(_xz_key(corner) for corner in stitch.corners)
        if len(cap_points) < 3:
            continue
        cap_point_sets.append((idx, cap_points))

    for cap_idx, cap_points in cap_point_sets:
        cap = stitches[cap_idx]
        for idx, stitch in enumerate(stitches):
            if stitch.story != cap.story or stitch.room_indices != cap.room_indices:
                continue
            if stitch.type in {"stitch_floor", "stitch_ceiling"}:
                if (
                    frozenset(_xz_key(corner) for corner in stitch.corners)
                    == cap_points
                ):
                    groups[cap_idx].add(idx)
                    groups[idx].add(cap_idx)
                continue
            if stitch.type != "stitch" or len(stitch.corners) < 2:
                continue
            edge_points = {_xz_key(stitch.corners[0]), _xz_key(stitch.corners[1])}
            if len(edge_points) == 2 and edge_points.issubset(cap_points):
                groups[cap_idx].add(idx)
                groups[idx].add(cap_idx)

    changed = True
    while changed:
        changed = False
        for idx, group in enumerate(groups):
            merged = set(group)
            for member in group:
                merged.update(groups[member])
            if merged != group:
                groups[idx] = merged
                changed = True
    return groups


# Back-compat alias for callers that referenced the prior private name.
_stitch_closure_groups = stitch_closure_groups


def _wall_edge_lines(room: ExtractedRoom) -> list[LineString]:
    lines: list[LineString] = []
    for wall in room.walls_computed:
        if wall.synthetic:
            continue
        corners = wall.corners
        if len(corners) < 2:
            continue
        for idx, corner in enumerate(corners):
            nxt = corners[(idx + 1) % len(corners)]
            line = LineString(
                [(float(corner[0]), float(corner[2])), (float(nxt[0]), float(nxt[2]))]
            )
            if line.length > 1e-6:
                lines.append(line)
    return lines


def _inferred_room_floor_polys(
    rooms_on_story: list[tuple[int, ExtractedRoom]],
) -> list[tuple[int, Polygon]]:
    polys: list[tuple[int, Polygon]] = []
    for room_idx, room in rooms_on_story:
        if (
            room.raw_ceiling_source != INFERRED_VOID_ROOM_SOURCE
            or len(room.floor_polygon) < 3
        ):
            continue
        try:
            poly = Polygon(
                [(float(corner[0]), float(corner[2])) for corner in room.floor_polygon]
            )
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-6:
            continue
        polys.append((room_idx, poly))
    return polys


def _segment_crosses_inferred_room(
    a: tuple[float, float],
    b: tuple[float, float],
    inferred_room_polys: list[tuple[int, Polygon]],
    owner_rooms: set[int],
) -> bool:
    if not inferred_room_polys:
        return False
    line = LineString([a, b])
    if line.length <= 1e-9:
        return True
    midpoint = line.interpolate(0.5, normalized=True)
    for room_idx, poly in inferred_room_polys:
        if room_idx in owner_rooms:
            continue
        if poly.covers(midpoint):
            return True
    return False


def _segment_conflicts_with_wall_run(
    a: tuple[float, float],
    b: tuple[float, float],
    wall_lines: list[LineString],
) -> bool:
    """A stitch line conflicts with an existing wall edge if it either
    runs *along* it (collinear overlap, the legacy check) or *crosses
    through* it (perpendicular intersection). Crossing detection
    catches stitches that pierce through a real scan-derived wall — e.g.
    two endpoints on opposite sides of a gable wall on the top story."""

    line = LineString([a, b])
    if line.length <= 1e-9:
        return True
    end_a = Point(*a)
    end_b = Point(*b)
    for wall_line in wall_lines:
        try:
            inter = line.intersection(wall_line)
        except Exception:
            inter = None
        if inter is not None and not inter.is_empty:
            # A crossing point in the interior of the candidate (not at one of
            # its endpoints) means the stitch pierces a wall. Shapely's
            # ``crosses`` is sensitive to floating-point noise when an endpoint
            # sits *exactly* on a wall (e.g. T-junction projections), so we
            # check intersection-to-endpoint distance and treat anything within
            # WALL_OVERLAP_BUFFER_M of an endpoint as a touch.
            points: list = []
            if isinstance(inter, Point):
                points = [inter]
            elif isinstance(inter, MultiPoint):
                points = list(inter.geoms)
            for pt in points:
                if (
                    pt.distance(end_a) > WALL_OVERLAP_BUFFER_M
                    and pt.distance(end_b) > WALL_OVERLAP_BUFFER_M
                ):
                    return True
        overlap_len = line.intersection(
            wall_line.buffer(WALL_OVERLAP_BUFFER_M, cap_style="flat")
        ).length
        if (
            overlap_len > WALL_OVERLAP_MIN_M
            and overlap_len / line.length > WALL_OVERLAP_MAX_RATIO
        ):
            return True
    return False


# Back-compat alias.
_segment_overlaps_wall_run = _segment_conflicts_with_wall_run


def stitch_wall_gaps(rooms: list[ExtractedRoom]) -> list[ExtractedStitchWall]:
    story_rooms: dict[int, list[tuple[int, ExtractedRoom]]] = defaultdict(list)
    for room_idx, room in enumerate(rooms):
        story_rooms[room.story].append((room_idx, room))

    footprint_by_story = story_footprint_context(rooms)

    room_ceiling_cache: dict[int, RoomCeiling | None] = {}
    stitches: list[ExtractedStitchWall] = []
    for story, rooms_on_story in story_rooms.items():
        story_footprint = footprint_by_story.get(story)
        wall_lines = [
            line
            for _room_idx, room in rooms_on_story
            for line in _wall_edge_lines(room)
        ]
        inferred_room_polys = _inferred_room_floor_polys(rooms_on_story)
        heights = [
            _wall_height(wall.corners)
            for _room_idx, room in rooms_on_story
            for wall in room.walls_computed
            if not wall.synthetic
            and len(wall.corners) >= 4
            and _wall_height(wall.corners) >= MIN_WALL_HEIGHT_M
        ]
        if not heights:
            continue
        median_h = float(np.median(heights))
        endpoints = []
        support_walls = []
        for room_idx, room in rooms_on_story:
            for wall_idx, wall in enumerate(room.walls_computed):
                if wall.synthetic:
                    continue
                corners = wall.corners
                if len(corners) < 4:
                    continue
                wall_h = _wall_height(corners)
                if wall_h < MIN_WALL_HEIGHT_M or wall_h < median_h * HEIGHT_RATIO:
                    continue
                y0_low, y0_high = _wall_endpoint_y_range(corners, 0)
                y1_low, y1_high = _wall_endpoint_y_range(corners, 1)
                endpoints.append(
                    (
                        corners[0][0],
                        corners[0][2],
                        y0_low,
                        y0_high,
                        (room_idx, wall_idx),
                        0,
                    )
                )
                endpoints.append(
                    (
                        corners[1][0],
                        corners[1][2],
                        y1_low,
                        y1_high,
                        (room_idx, wall_idx),
                        1,
                    )
                )
                support_walls.append(
                    {
                        "p0": (float(corners[0][0]), float(corners[0][2])),
                        "p1": (float(corners[1][0]), float(corners[1][2])),
                        "corners": corners,
                        "key": (room_idx, wall_idx),
                        "dir": _wall_dir_xz(corners),
                    }
                )

        used_pairs = set()
        stitched_endpoints = set()

        def add(
            stitch_type: str,
            corners: list[list[float]],
            room_idx1: int,
            room_idx2: int,
            story: int = story,
        ) -> None:
            stitches.append(
                ExtractedStitchWall(
                    id=f"{stitch_type}:{story}:{len(stitches)}",
                    corners=corners,
                    type=stitch_type,
                    story=story,
                    room_index=room_idx1,
                    room_indices=[room_idx1, room_idx2],
                )
            )

        for idx, endpoint in enumerate(endpoints):
            x1, z1, yb1, yt1, wk1, end1 = endpoint
            best_dist = MAX_GAP_M
            best_idx = -1
            for jdx, candidate in enumerate(endpoints):
                if idx == jdx:
                    continue
                x2, z2, _yb2, _yt2, wk2, _end2 = candidate
                if wk2 == wk1:
                    continue
                dist = math.hypot(x1 - x2, z1 - z2)
                if MIN_GAP_M <= dist < best_dist:
                    best_dist = dist
                    best_idx = jdx
            if best_idx < 0:
                continue

            x2, z2, yb2, yt2, wk2, end2 = endpoints[best_idx]
            if best_dist <= MUTUAL_THRESH_M:
                reverse_best_dist = MAX_GAP_M
                reverse_best = -1
                for kdx, reverse_candidate in enumerate(endpoints):
                    if kdx == best_idx:
                        continue
                    xk, zk, _ykb, _ykt, wkk, _ek = reverse_candidate
                    if wkk == wk2:
                        continue
                    dist = math.hypot(x2 - xk, z2 - zk)
                    if MIN_GAP_M <= dist < reverse_best_dist:
                        reverse_best_dist = dist
                        reverse_best = kdx
                if reverse_best != idx:
                    continue

            pair_key = tuple(sorted([(*wk1, end1), (*wk2, end2)]))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            y_bot = max(float(yb1), float(yb2))
            if min(float(yt1), float(yt2)) - y_bot < MIN_WALL_HEIGHT_M:
                continue

            room_idx1, wall_idx1 = wk1
            room_idx2, wall_idx2 = wk2
            owner_rooms = {room_idx1, room_idx2}

            if best_dist < DIRECT_QUAD_MAX_GAP_M:
                if _segment_conflicts_with_wall_run((x1, z1), (x2, z2), wall_lines):
                    continue
                if _segment_crosses_inferred_room(
                    (x1, z1),
                    (x2, z2),
                    inferred_room_polys,
                    owner_rooms,
                ):
                    continue
                top1 = _owner_local_top_y(rooms, owner_rooms, x1, z1, float(yt1))
                top2 = _owner_local_top_y(rooms, owner_rooms, x2, z2, float(yt2))
                quad = _stitch_quad((x1, z1), (x2, z2), y_bot, top1, top2)
                if quad is None:
                    continue
                add("stitch", quad, room_idx1, room_idx2)
                stitched_endpoints.add((*wk1, end1))
                stitched_endpoints.add((*wk2, end2))
                continue

            wall1_corners = rooms[room_idx1].walls_computed[wall_idx1].corners
            wall2_corners = rooms[room_idx2].walls_computed[wall_idx2].corners
            dir1 = _wall_dir_xz(wall1_corners)
            dir2 = _wall_dir_xz(wall2_corners)
            out_sign1 = -1.0 if end1 == 0 else 1.0
            out_sign2 = -1.0 if end2 == 0 else 1.0
            u1_out = (out_sign1 * dir1[0], out_sign1 * dir1[1])
            u2_out = (out_sign2 * dir2[0], out_sign2 * dir2[1])

            candidates = [
                candidate
                for candidate in (
                    _build_l_candidate((x1, z1), (x2, z2), u1_out, u2_out),
                    _build_l_candidate((x2, z2), (x1, z1), u2_out, u1_out),
                )
                if candidate is not None and candidate["legs"]
            ]
            # Reject L candidates whose corner falls outside the story footprint.
            # The L corner extends each wall's tangent direction outward; for
            # endpoints at the edge of the building it can land in empty space
            # beyond the scan-derived footprint. Synthesised geometry must
            # follow the scan, not invent space outside it.
            candidates = [
                candidate
                for candidate in candidates
                if is_xz_supported_by_story_footprint(
                    candidate["corner"][0], candidate["corner"][1], story_footprint
                )
            ]
            candidates = [
                candidate
                for candidate in candidates
                if not any(
                    _segment_conflicts_with_wall_run(a, b, wall_lines)
                    for a, b, _source in candidate["legs"]
                )
            ]
            candidates = [
                candidate
                for candidate in candidates
                if not any(
                    _segment_crosses_inferred_room(
                        a, b, inferred_room_polys, owner_rooms
                    )
                    for a, b, _source in candidate["legs"]
                )
            ]
            if not candidates:
                continue
            picked = min(candidates, key=lambda candidate: candidate["total_len"])
            cx, cz = picked["corner"]
            l_leg_quads = []
            for (ax, az), (bx, bz), _source in picked["legs"]:
                # L stitches bridge two room envelopes. At any point on the
                # dogleg, the closure must stop at the lower local owner-wall
                # top; using the taller wall's global top creates panels that
                # punch through a sloped roof/eave.
                ranges_a = [
                    y_range
                    for wall_corners in (wall1_corners, wall2_corners)
                    if (y_range := _wall_y_range_at_xz(wall_corners, ax, az))
                    is not None
                ]
                ranges_b = [
                    y_range
                    for wall_corners in (wall1_corners, wall2_corners)
                    if (y_range := _wall_y_range_at_xz(wall_corners, bx, bz))
                    is not None
                ]
                if not ranges_a or not ranges_b:
                    continue
                quad = _stitch_quad(
                    (ax, az),
                    (bx, bz),
                    y_bot,
                    _owner_local_top_y(
                        rooms,
                        owner_rooms,
                        ax,
                        az,
                        min(y_range[1] for y_range in ranges_a),
                    ),
                    _owner_local_top_y(
                        rooms,
                        owner_rooms,
                        bx,
                        bz,
                        min(y_range[1] for y_range in ranges_b),
                    ),
                )
                if quad is not None:
                    l_leg_quads.append(quad)
            if not l_leg_quads:
                continue
            for quad in l_leg_quads:
                add("stitch", quad, room_idx1, room_idx2)
            # An L-stitch cap is a triangle at (x1,z1) → (cx,cz) → (x2,z2).
            # Short-leg angled junctions are still real ceiling/floor pockets;
            # skip only near-zero projected area so meaningful angles remain covered.
            cap_area = _triangle_xz_area((x1, z1), (cx, cz), (x2, z2))
            cap_is_degenerate = cap_area < L_STITCH_CAP_MIN_AREA_M2
            if not cap_is_degenerate:
                add(
                    "stitch_floor",
                    [[x1, y_bot, z1], [cx, y_bot, cz], [x2, y_bot, z2]],
                    room_idx1,
                    room_idx2,
                )
                chosen_ceiling = _select_lower_room_ceiling(
                    rooms, room_idx1, room_idx2, cx, cz, room_ceiling_cache
                )
                if chosen_ceiling is not None:
                    y1c = chosen_ceiling.y_at(x1, z1)
                    ycc = chosen_ceiling.y_at(cx, cz)
                    y2c = chosen_ceiling.y_at(x2, z2)
                    if y1c is not None and ycc is not None and y2c is not None:
                        add(
                            "stitch_ceiling",
                            [
                                [x1, float(y1c), z1],
                                [cx, float(ycc), cz],
                                [x2, float(y2c), z2],
                            ],
                            room_idx1,
                            room_idx2,
                        )
            stitched_endpoints.add((*wk1, end1))
            stitched_endpoints.add((*wk2, end2))

        t_junctions = set()
        for endpoint in endpoints:
            x, z, yb, yt, wk, end = endpoint
            endpoint_key = (*wk, end)
            if endpoint_key in stitched_endpoints:
                continue
            room_idx, wall_idx = wk
            endpoint_dir = _wall_dir_xz(
                rooms[room_idx].walls_computed[wall_idx].corners
            )
            best = None
            for wall in support_walls:
                support_room_idx, support_wall_idx = wall["key"]
                if support_room_idx == room_idx:
                    continue
                p0 = np.array(wall["p0"], dtype=float)
                p1 = np.array(wall["p1"], dtype=float)
                edge = p1 - p0
                edge_len = float(np.linalg.norm(edge))
                if edge_len <= 1e-9:
                    continue
                edge_unit = edge / edge_len
                endpoint_vec = np.array([x, z], dtype=float)
                t = float(np.dot(endpoint_vec - p0, edge_unit))
                if (
                    t < T_JUNCTION_MIN_EDGE_MARGIN_M
                    or t > edge_len - T_JUNCTION_MIN_EDGE_MARGIN_M
                ):
                    continue
                projection = p0 + t * edge_unit
                dist = float(np.linalg.norm(endpoint_vec - projection))
                if not (T_JUNCTION_MIN_GAP_M <= dist <= T_JUNCTION_MAX_GAP_M):
                    continue
                cos_parallel = abs(
                    float(
                        endpoint_dir[0] * wall["dir"][0]
                        + endpoint_dir[1] * wall["dir"][1]
                    )
                )
                if cos_parallel > T_JUNCTION_MAX_PARALLEL_COS:
                    continue
                projection_tuple = (float(projection[0]), float(projection[1]))
                if _segment_conflicts_with_wall_run(
                    (x, z), projection_tuple, wall_lines
                ):
                    continue
                if _segment_crosses_inferred_room(
                    (x, z),
                    projection_tuple,
                    inferred_room_polys,
                    {room_idx, support_room_idx},
                ):
                    continue
                support_range = _wall_y_range_at_xz(
                    wall["corners"], projection_tuple[0], projection_tuple[1]
                )
                if support_range is None:
                    continue
                y_bot = max(float(yb), support_range[0])
                t_owner_rooms = {room_idx, support_room_idx}
                y_top_endpoint = _owner_local_top_y(
                    rooms, t_owner_rooms, x, z, float(yt)
                )
                y_top_projection = _owner_local_top_y(
                    rooms,
                    t_owner_rooms,
                    projection_tuple[0],
                    projection_tuple[1],
                    support_range[1],
                )
                if min(y_top_endpoint, y_top_projection) - y_bot < MIN_WALL_HEIGHT_M:
                    continue
                key = (
                    room_idx,
                    wall_idx,
                    end,
                    support_room_idx,
                    support_wall_idx,
                    round(projection_tuple[0], 4),
                    round(projection_tuple[1], 4),
                )
                if key in t_junctions:
                    continue
                candidate = {
                    "dist": dist,
                    "key": key,
                    "projection": projection_tuple,
                    "y_bot": y_bot,
                    "y_top_endpoint": y_top_endpoint,
                    "y_top_projection": y_top_projection,
                    "support_room_idx": support_room_idx,
                }
                if best is None or candidate["dist"] < best["dist"]:
                    best = candidate
            if best is None:
                continue
            t_junctions.add(best["key"])
            px, pz = best["projection"]
            add(
                "stitch",
                [
                    [x, best["y_bot"], z],
                    [px, best["y_bot"], pz],
                    [px, best["y_top_projection"], pz],
                    [x, best["y_top_endpoint"], z],
                ],
                room_idx,
                best["support_room_idx"],
            )

    return stitches
