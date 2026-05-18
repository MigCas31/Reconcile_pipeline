"""Perimeter completion and rectilinear-wall synthesis.

Builds the rectilinear walls that fill notches and missing perimeter spans
when the scan misses them. Extracted from `walls_to_rooms.py`; re-exports
in `walls_to_rooms.py` keep the public symbols (`reclip_cutouts_to_wall`
and friends) backwards-compatible.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import atan2, degrees

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from reconcile_tiers.extract.building import BuildingModel, ExtractedRoom
from reconcile_tiers.payload.schema import Vec3, Wall

RECTILINEAR_COVERAGE_MIN = 0.70
RECTILINEAR_AXIS_TOL_DEG = 12.0
RECTILINEAR_OFF_AXIS_MIN_DEG = 20.0
RECTILINEAR_NOTCH_MAX_WALL_M = 1.30
RECTILINEAR_NOTCH_MAX_LEG_M = 1.50
RECTILINEAR_NOTCH_MIN_LEG_M = 0.05
RECTILINEAR_NOTCH_ENDPOINT_TOL_M = 0.20
RECTILINEAR_NOTCH_MIN_SUPPORT_M = 0.25
RECTILINEAR_CAP_MAX_WALL_M = 1.30
RECTILINEAR_CAP_PARALLEL_TOL_DEG = 3.0
RECTILINEAR_CAP_MIN_OFFSET_M = 0.25
RECTILINEAR_CAP_MAX_OFFSET_M = 2.00
RECTILINEAR_CAP_MIN_OVERLAP_FRACTION = 0.80
RECTILINEAR_CAP_MIN_LENGTH_RATIO = 1.50
PERIMETER_SYNTH_MIN_LENGTH_M = 2.0
PERIMETER_SYNTH_BOUNDARY_COVERAGE_MIN = 0.95
PERIMETER_SYNTH_WALL_COVERAGE_MAX = 0.20
PERIMETER_SYNTH_GAP_COVERAGE_MAX = 0.20
PERIMETER_SYNTH_ENDPOINT_TOL_M = 0.35
PERIMETER_SYNTH_SAMPLE_STEP_M = 0.20
PERIMETER_SYNTH_COVERAGE_TOL_M = 0.08
PERIMETER_SYNTH_BOUNDARY_TOL_M = 0.12

LOGGER = logging.getLogger(__name__)


def _xz_span(
    corners: Sequence[Sequence[float] | Vec3],
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    pts: list[tuple[float, float]] = []
    for corner in corners:
        if isinstance(corner, Vec3):
            pts.append((float(corner.x), float(corner.z)))
        else:
            pts.append((float(corner[0]), float(corner[2])))
    best: tuple[tuple[float, float], tuple[float, float], float] | None = None
    for idx, a in enumerate(pts):
        for b in pts[idx + 1 :]:
            length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            if best is None or length > best[2]:
                best = (a, b, length)
    if best is None or best[2] <= 1e-9:
        return None
    return best


def _angle_mod90_delta(a: float, b: float) -> float:
    diff = abs((a - b) % 90.0)
    return min(diff, 90.0 - diff)


def _xz_angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:

    return degrees(atan2(b[1] - a[1], b[0] - a[0])) % 90.0


def _unit_xz(dx: float, dz: float) -> tuple[float, float, float] | None:
    length = (dx * dx + dz * dz) ** 0.5
    if length <= 1e-9:
        return None
    return dx / length, dz / length, length


def _dist_xz(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _line_from_wall(wall: Wall) -> LineString | None:
    span = _xz_span(wall.corners)
    if span is None:
        return None
    p0, p1, length = span
    if length <= 1e-6:
        return None
    return LineString([p0, p1])


def _line_from_extracted_wall(wall) -> LineString | None:
    span = _xz_span(wall.corners)
    if span is None:
        return None
    p0, p1, length = span
    if length <= 1e-6:
        return None
    return LineString([p0, p1])


def _polygon_edge_lines(corners: Sequence[Sequence[float]]) -> list[LineString]:
    points = [(float(corner[0]), float(corner[2])) for corner in corners]
    lines: list[LineString] = []
    for p0, p1 in zip(points, points[1:] + points[:1], strict=False):
        line = LineString([p0, p1])
        if line.length > 1e-6:
            lines.append(line)
    return lines


def _coverage_fraction(
    line: LineString, candidates: Sequence[LineString], *, tol_m: float
) -> float:
    if not candidates or line.length <= 1e-9:
        return 0.0
    sample_count = max(10, int(line.length / PERIMETER_SYNTH_SAMPLE_STEP_M))
    covered = 0
    for idx in range(sample_count + 1):
        point = line.interpolate(idx / sample_count, normalized=True)
        if min(candidate.distance(point) for candidate in candidates) <= tol_m:
            covered += 1
    return covered / (sample_count + 1)


def _boundary_coverage_fraction(line: LineString, boundary) -> float:
    if line.length <= 1e-9:
        return 0.0
    sample_count = max(10, int(line.length / PERIMETER_SYNTH_SAMPLE_STEP_M))
    covered = 0
    for idx in range(sample_count + 1):
        point = line.interpolate(idx / sample_count, normalized=True)
        if point.distance(boundary) <= PERIMETER_SYNTH_BOUNDARY_TOL_M:
            covered += 1
    return covered / (sample_count + 1)


def _building_footprint_boundary(model: BuildingModel):
    from reconcile_tiers.assemble.walls_to_rooms import _floor_polygon_xz

    polys = []
    for room in model.rooms:
        poly = _floor_polygon_xz(room.floor_polygon)
        if poly is not None:
            polys.append(poly)
    if not polys:
        return None
    footprint = unary_union(polys)
    if footprint.is_empty:
        return None
    return footprint.boundary


def _gap_context_lines(model: BuildingModel) -> list[LineString]:
    lines: list[LineString] = []
    for wall in model.gap_walls:
        lines.extend(_polygon_edge_lines(wall.corners))
    for stitch in model.stitch_walls:
        lines.extend(_polygon_edge_lines(stitch.corners))
    for closure in model.gap_closures:
        lines.extend(_polygon_edge_lines(closure.corners))
    return lines


def _boundary_wall_endpoints(lines: Sequence[LineString], boundary) -> list[Point]:
    endpoints: list[Point] = []
    for line in lines:
        if (
            _boundary_coverage_fraction(line, boundary)
            < PERIMETER_SYNTH_BOUNDARY_COVERAGE_MIN
        ):
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        endpoints.append(Point(coords[0]))
        endpoints.append(Point(coords[-1]))
    return endpoints


def _room_wall_y_extent(room: ExtractedRoom) -> tuple[float, float] | None:
    ys = [
        float(corner[1])
        for wall in [*room.walls_computed, *room.synthetic_walls]
        for corner in wall.corners
    ]
    if not ys:
        return None
    return min(ys), max(ys)


def _line_intersection_xz(
    p0: tuple[float, float],
    d0: tuple[float, float],
    p1: tuple[float, float],
    d1: tuple[float, float],
) -> tuple[tuple[float, float], float, float] | None:
    den = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(den) <= 1e-9:
        return None
    rx = p1[0] - p0[0]
    rz = p1[1] - p0[1]
    t = (rx * d1[1] - rz * d1[0]) / den
    u = (rx * d0[1] - rz * d0[0]) / den
    return (p0[0] + t * d0[0], p0[1] + t * d0[1]), t, u


def _project_point_to_line_xz(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_dir: tuple[float, float],
) -> tuple[float, float]:
    vx = point[0] - line_start[0]
    vz = point[1] - line_start[1]
    t = vx * line_dir[0] + vz * line_dir[1]
    return line_start[0] + t * line_dir[0], line_start[1] + t * line_dir[1]


def _support_candidates(
    *,
    point: tuple[float, float],
    walls: list[Wall],
    skip_idx: int,
    floor_polygon: list[list[float]],
    axis_deg: float,
) -> list[tuple[int, float, tuple[float, float], float]]:
    supports: list[tuple[int, float, tuple[float, float], float]] = []

    def add_support(
        priority: int, matched_distance: float, dx: float, dz: float, length: float
    ) -> None:
        if length < RECTILINEAR_NOTCH_MIN_SUPPORT_M:
            return
        unit = _unit_xz(dx, dz)
        if unit is None:
            return
        ux, uz, _length = unit
        angle = _xz_angle_deg((0.0, 0.0), (ux, uz))
        if _angle_mod90_delta(angle, axis_deg) > RECTILINEAR_AXIS_TOL_DEG:
            return
        supports.append((priority, matched_distance, (ux, uz), length))

    for idx, wall in enumerate(walls):
        if idx == skip_idx:
            continue
        span = _xz_span(wall.corners)
        if span is None:
            continue
        a, b, length = span
        da = _dist_xz(point, a)
        db = _dist_xz(point, b)
        if da <= RECTILINEAR_NOTCH_ENDPOINT_TOL_M:
            add_support(0, da, b[0] - a[0], b[1] - a[1], length)
        if db <= RECTILINEAR_NOTCH_ENDPOINT_TOL_M:
            add_support(0, db, a[0] - b[0], a[1] - b[1], length)

    for idx, corner in enumerate(floor_polygon):
        nxt = floor_polygon[(idx + 1) % len(floor_polygon)] if floor_polygon else None
        if nxt is None:
            continue
        a = (float(corner[0]), float(corner[2]))
        b = (float(nxt[0]), float(nxt[2]))
        length = _dist_xz(a, b)
        da = _dist_xz(point, a)
        db = _dist_xz(point, b)
        if da <= RECTILINEAR_NOTCH_ENDPOINT_TOL_M:
            add_support(1, da, b[0] - a[0], b[1] - a[1], length)
        if db <= RECTILINEAR_NOTCH_ENDPOINT_TOL_M:
            add_support(1, db, a[0] - b[0], a[1] - b[1], length)

    return sorted(supports, key=lambda item: (item[0], item[1], -item[3]))


def _make_rectilinear_wall(
    *,
    p0: tuple[float, float],
    p1: tuple[float, float],
    y_min: float,
    y_max: float,
    locator_id: str,
    room_center: list[float],
    floor_polygon: list[list[float]],
) -> Wall | None:
    from reconcile_tiers.assemble.walls_to_rooms import _orient_wall_outward, _vec3

    if _dist_xz(p0, p1) < RECTILINEAR_NOTCH_MIN_LEG_M:
        return None
    corners = [
        [p0[0], y_max, p0[1]],
        [p1[0], y_max, p1[1]],
        [p1[0], y_min, p1[1]],
        [p0[0], y_min, p0[1]],
    ]
    corners = _orient_wall_outward(corners, room_center, floor_polygon)
    return Wall(
        corners=[_vec3(corner) for corner in corners],
        descent_strip=None,
        uplift_strip=None,
        cutouts=[],
        locator_id=locator_id,
        synthetic=True,
    )


def _synthesise_missing_perimeter_walls(
    *,
    model: BuildingModel,
    room: ExtractedRoom,
    walls: list[Wall],
    room_center: list[float],
    building_boundary,
    all_wall_lines: Sequence[LineString],
    gap_lines: Sequence[LineString],
    boundary_endpoints: Sequence[Point],
) -> list[Wall]:
    if (
        building_boundary is None
        or not boundary_endpoints
        or len(room.floor_polygon) < 3
    ):
        return walls
    y_extent = _room_wall_y_extent(room)
    if y_extent is None:
        return walls
    y_min, y_max = y_extent
    room_wall_lines = [
        line for wall in walls if (line := _line_from_wall(wall)) is not None
    ]
    if not room_wall_lines:
        return walls

    out = list(walls)
    for edge_idx, (corner, nxt) in enumerate(
        zip(
            room.floor_polygon,
            room.floor_polygon[1:] + room.floor_polygon[:1],
            strict=False,
        )
    ):
        p0 = (float(corner[0]), float(corner[2]))
        p1 = (float(nxt[0]), float(nxt[2]))
        edge = LineString([p0, p1])
        if edge.length <= PERIMETER_SYNTH_MIN_LENGTH_M:
            continue
        if (
            _boundary_coverage_fraction(edge, building_boundary)
            < PERIMETER_SYNTH_BOUNDARY_COVERAGE_MIN
        ):
            continue
        if (
            _coverage_fraction(
                edge, room_wall_lines, tol_m=PERIMETER_SYNTH_COVERAGE_TOL_M
            )
            >= PERIMETER_SYNTH_WALL_COVERAGE_MAX
        ):
            continue
        if (
            _coverage_fraction(
                edge, all_wall_lines, tol_m=PERIMETER_SYNTH_COVERAGE_TOL_M
            )
            >= PERIMETER_SYNTH_WALL_COVERAGE_MAX
        ):
            continue
        if (
            _coverage_fraction(edge, gap_lines, tol_m=PERIMETER_SYNTH_COVERAGE_TOL_M)
            >= PERIMETER_SYNTH_GAP_COVERAGE_MAX
        ):
            continue
        endpoint_distance_a = min(
            Point(p0).distance(endpoint) for endpoint in boundary_endpoints
        )
        endpoint_distance_b = min(
            Point(p1).distance(endpoint) for endpoint in boundary_endpoints
        )
        if (
            endpoint_distance_a > PERIMETER_SYNTH_ENDPOINT_TOL_M
            or endpoint_distance_b > PERIMETER_SYNTH_ENDPOINT_TOL_M
        ):
            continue

        wall = _make_rectilinear_wall(
            p0=p0,
            p1=p1,
            y_min=y_min,
            y_max=y_max,
            locator_id=(
                f"{model.uuid}::tier-wall::{room.index}:perimeter-synth:{edge_idx}"
            ),
            room_center=room_center,
            floor_polygon=room.floor_polygon,
        )
        if wall is None:
            continue
        LOGGER.debug(
            "Synthesised missing perimeter wall %s length=%.2fm",
            wall.locator_id,
            edge.length,
        )
        out.append(wall)
        room_wall_lines.append(edge)
    return out


def _rectilinearize_short_oblique_notches(
    walls: list[Wall],
    *,
    floor_polygon: list[list[float]],
    room_center: list[float],
    wall_axis: tuple[float, float] | None,
) -> list[Wall]:
    if wall_axis is None or wall_axis[1] < RECTILINEAR_COVERAGE_MIN:
        return walls
    axis_deg = wall_axis[0]
    out: list[Wall] = []
    consumed: set[int] = set()

    for idx, wall in enumerate(walls):
        if idx in consumed:
            continue
        span = _xz_span(wall.corners)
        if span is None:
            out.append(wall)
            continue
        p0, p1, length = span
        if (
            wall.synthetic
            or ":rect-closure:" in wall.locator_id
            or wall.cutouts
            or wall.descent_strip is not None
            or wall.uplift_strip is not None
            or length > RECTILINEAR_NOTCH_MAX_WALL_M
        ):
            out.append(wall)
            continue
        wall_angle = _xz_angle_deg(p0, p1)
        if _angle_mod90_delta(wall_angle, axis_deg) < RECTILINEAR_OFF_AXIS_MIN_DEG:
            out.append(wall)
            continue

        start_supports = _support_candidates(
            point=p0,
            walls=walls,
            skip_idx=idx,
            floor_polygon=floor_polygon,
            axis_deg=axis_deg,
        )
        end_supports = _support_candidates(
            point=p1,
            walls=walls,
            skip_idx=idx,
            floor_polygon=floor_polygon,
            axis_deg=axis_deg,
        )
        best = None
        for s0 in start_supports:
            for s1 in end_supports:
                d0 = s0[2]
                d1 = s1[2]
                # Supports should describe the two perpendicular wall families,
                # not two collinear lines on the same run.
                if abs(d0[0] * d1[0] + d0[1] * d1[1]) > 0.30:
                    continue
                intersection = _line_intersection_xz(p0, d0, p1, d1)
                if intersection is None:
                    continue
                corner, t0, t1 = intersection
                leg0 = abs(t0)
                leg1 = abs(t1)
                if (
                    leg0 < RECTILINEAR_NOTCH_MIN_LEG_M
                    or leg1 < RECTILINEAR_NOTCH_MIN_LEG_M
                    or leg0 > RECTILINEAR_NOTCH_MAX_LEG_M
                    or leg1 > RECTILINEAR_NOTCH_MAX_LEG_M
                ):
                    continue
                score = (s0[0] + s1[0], leg0 + leg1, s0[1] + s1[1])
                if best is None or score < best[0]:
                    best = (score, corner)
        if best is None:
            out.append(wall)
            continue

        _score, corner = best
        ys = [float(c.y) for c in wall.corners]
        y_min = min(ys)
        y_max = max(ys)
        base_id = wall.locator_id.rsplit("::tier-wall::", 1)[-1]
        first = _make_rectilinear_wall(
            p0=p0,
            p1=corner,
            y_min=y_min,
            y_max=y_max,
            locator_id=f"{wall.locator_id}:rect-closure:0",
            room_center=room_center,
            floor_polygon=floor_polygon,
        )
        second = _make_rectilinear_wall(
            p0=corner,
            p1=p1,
            y_min=y_min,
            y_max=y_max,
            locator_id=f"{wall.locator_id}:rect-closure:1",
            room_center=room_center,
            floor_polygon=floor_polygon,
        )
        replacements = [
            candidate for candidate in (first, second) if candidate is not None
        ]
        if len(replacements) != 2:
            out.append(wall)
            continue
        LOGGER.debug(
            "Rectilinearized short off-axis wall %s into two closure walls", base_id
        )
        out.extend(replacements)
        consumed.add(idx)

    return out


def _snap_short_parallel_caps_to_longer_run(walls: list[Wall]) -> list[Wall]:
    from reconcile_tiers.assemble.walls_to_rooms import _vec3

    out: list[Wall] = []
    spans = [_xz_span(wall.corners) for wall in walls]
    for idx, wall in enumerate(walls):
        span = spans[idx]
        if span is None:
            out.append(wall)
            continue
        p0, p1, length = span
        if (
            wall.synthetic
            or ":rect-closure:" in wall.locator_id
            or wall.cutouts
            or wall.descent_strip is not None
            or wall.uplift_strip is not None
            or length > RECTILINEAR_CAP_MAX_WALL_M
        ):
            out.append(wall)
            continue
        axis = _unit_xz(p1[0] - p0[0], p1[1] - p0[1])
        if axis is None:
            out.append(wall)
            continue
        ux, uz, _ = axis
        angle = _xz_angle_deg(p0, p1)
        best = None
        for other_idx, _other in enumerate(walls):
            if other_idx == idx:
                continue
            other_span = spans[other_idx]
            if other_span is None:
                continue
            q0, q1, other_length = other_span
            if other_length < length * RECTILINEAR_CAP_MIN_LENGTH_RATIO:
                continue
            other_axis = _unit_xz(q1[0] - q0[0], q1[1] - q0[1])
            if other_axis is None:
                continue
            oux, ouz, _ = other_axis
            other_angle = _xz_angle_deg(q0, q1)
            if (
                _angle_mod90_delta(angle, other_angle)
                > RECTILINEAR_CAP_PARALLEL_TOL_DEG
            ):
                continue
            # Use the candidate direction that agrees with this wall's axis.
            if ux * oux + uz * ouz < 0.0:
                oux, ouz = -oux, -ouz
            nx, nz = -uz, ux
            offset = (q0[0] - p0[0]) * nx + (q0[1] - p0[1]) * nz
            abs_offset = abs(offset)
            if (
                abs_offset < RECTILINEAR_CAP_MIN_OFFSET_M
                or abs_offset > RECTILINEAR_CAP_MAX_OFFSET_M
            ):
                continue
            other_t = [
                (q0[0] - p0[0]) * ux + (q0[1] - p0[1]) * uz,
                (q1[0] - p0[0]) * ux + (q1[1] - p0[1]) * uz,
            ]
            overlap = max(0.0, min(length, max(other_t)) - max(0.0, min(other_t)))
            overlap_fraction = overlap / max(length, 1e-9)
            if overlap_fraction < RECTILINEAR_CAP_MIN_OVERLAP_FRACTION:
                continue
            score = (-overlap_fraction, abs_offset, -other_length)
            if best is None or score < best[0]:
                best = (score, (q0, (oux, ouz)))
        if best is None:
            out.append(wall)
            continue

        _score, (line_start, line_dir) = best
        snapped_p0 = _project_point_to_line_xz(p0, line_start, line_dir)
        snapped_p1 = _project_point_to_line_xz(p1, line_start, line_dir)
        snapped_corners: list[list[float]] = []
        for corner in wall.corners:
            xz = (float(corner.x), float(corner.z))
            d0 = _dist_xz(xz, p0)
            d1 = _dist_xz(xz, p1)
            snapped = snapped_p0 if d0 <= d1 else snapped_p1
            snapped_corners.append([snapped[0], float(corner.y), snapped[1]])
        out.append(
            Wall(
                corners=[_vec3(corner) for corner in snapped_corners],
                descent_strip=wall.descent_strip,
                uplift_strip=wall.uplift_strip,
                cutouts=wall.cutouts,
                locator_id=wall.locator_id,
                synthetic=wall.synthetic,
            )
        )
    return out
