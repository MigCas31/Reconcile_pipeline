"""Find facing wall pairs between adjacent scanned rooms and emit the
wall-thickness slab quad bounded by them.

Each scanned room captures the *interior* face of every wall it bounds.
Where two rooms share a wall, both rooms' scans capture that wall — once
per side, separated by the wall thickness. The volume between those two
captured-faces is missing from the model. This module finds those pairs
geometrically and emits the slab quad whose 4 corners are projections of
the wall-pair endpoints onto each other along the overlap region.

Vertices of every emitted slab come from wall endpoints of the two
participating rooms. No shapely difference is run on noisy floor
polygons; no ear-clipping; no host-room assignment. The slab is between
exactly two rooms, by construction.

Math references (archived legacy implementations consulted while writing
this module — no runtime dependency on them):
  - archive/legacy-runtime under reconcile/extract3d/gaps.py (snap-to-wall flow)
  - archive/legacy-runtime wall-thickness inference (parallel-edge math)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry import Point, Polygon

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.extract.building import (
    ExtractedRoom,
    ExtractedWall,
)
from reconcile_tiers.extract.room_ceiling import RoomCeiling, room_ceiling

# Corpus-derived bounds (will be confirmed by histogram in tracking_progress).
MIN_WALL_THICKNESS_M = 0.04
MAX_WALL_THICKNESS_M = 0.70
MIN_BOUNDED_RECESS_WIDTH_M = 0.80
MAX_BOUNDED_RECESS_WIDTH_M = 0.85
MIN_WALL_OVERLAP_M = 0.20
ANGLE_TOLERANCE_DEG = 15.0  # how anti-parallel two facing walls must be
ENDPOINT_SNAP_TOL_M = (
    0.05  # snap projected slab corners back to wall endpoints if close
)
CEILING_EXTENT_BUFFER_M = MAX_WALL_THICKNESS_M + 0.05
RECESS_CAP_ALIGNMENT_TOL_M = 0.12
MIN_RECESS_CAP_COVERAGE = 0.60
OPENING_RECESS_END_TOL_M = 0.20


@dataclass(frozen=True, slots=True)
class CeilingFacet:
    ceiling: RoomCeiling
    xz_polygon: Polygon | None

    def y_at(self, x: float, z: float) -> float | None:
        if self.xz_polygon is not None:
            try:
                if not self.xz_polygon.buffer(CEILING_EXTENT_BUFFER_M).covers(
                    Point(float(x), float(z))
                ):
                    return None
            except Exception:
                return None
        return self.ceiling.y_at(float(x), float(z))


@dataclass(frozen=True, slots=True)
class CeilingProfile:
    facets: tuple[CeilingFacet, ...]
    fallback: CeilingFacet | None = None

    def y_at(self, x: float, z: float) -> float | None:
        candidates = []
        if self.fallback is not None:
            y = self.fallback.y_at(x, z)
            if y is not None:
                candidates.append(float(y))
        for facet in self.facets:
            y = facet.y_at(x, z)
            if y is not None:
                candidates.append(float(y))
        if candidates:
            return min(candidates)
        return None


@dataclass(frozen=True, slots=True)
class WallRun:
    """Bottom-edge projection of a wall onto the XZ floor plane, plus the
    top-edge y profile sampled along the same tangent. The profile lets
    a slab follow gable ends and other sloped wall tops instead of being
    capped at a single flat y."""

    room_index: int
    wall_index: int
    p0: tuple[float, float]
    p1: tuple[float, float]
    y_low: float
    y_high: float
    top_profile: tuple[tuple[float, float], ...]  # sorted (t, y) pairs along tangent
    bottom_profile: tuple[tuple[float, float], ...]
    endpoint_bottoms: tuple[float, float]
    ceiling_profile: CeilingProfile | None = None
    has_opening: bool = False
    opening_intervals: tuple[tuple[float, float], ...] = ()

    @property
    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    @property
    def tangent(self) -> tuple[float, float]:
        L = self.length or 1.0
        return ((self.p1[0] - self.p0[0]) / L, (self.p1[1] - self.p0[1]) / L)

    def top_y_at(self, t: float) -> float:
        return _interp_profile(self.top_profile, t)

    def bottom_y_at(self, t: float) -> float:
        return _interp_profile(self.bottom_profile, t)

    def ceiling_y_at(self, x: float, z: float) -> float | None:
        if self.ceiling_profile is None:
            return None
        return self.ceiling_profile.y_at(x, z)


def _interp_profile(profile: tuple[tuple[float, float], ...], t: float) -> float:
    """Piecewise-linear interpolation of (t, y) pairs. Profile is sorted
    by t. Outside the range, clamp to the endpoints (no extrapolation)."""
    if not profile:
        return 0.0
    if len(profile) == 1:
        return profile[0][1]
    if t <= profile[0][0]:
        return profile[0][1]
    if t >= profile[-1][0]:
        return profile[-1][1]
    for i in range(len(profile) - 1):
        t0, y0 = profile[i]
        t1, y1 = profile[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return y0 + f * (y1 - y0)
    return profile[-1][1]


def _dedupe_profile_points(
    points: list[tuple[float, float]],
    *,
    prefer: str,
    tol: float = 1e-6,
) -> tuple[tuple[float, float], ...]:
    if not points:
        return ()
    points = sorted(points)
    out: list[tuple[float, float]] = []
    group_t = points[0][0]
    group_ys = [points[0][1]]
    for t, y in points[1:]:
        if abs(t - group_t) <= tol:
            group_ys.append(y)
            continue
        out.append((group_t, max(group_ys) if prefer == "high" else min(group_ys)))
        group_t = t
        group_ys = [y]
    out.append((group_t, max(group_ys) if prefer == "high" else min(group_ys)))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class SlabQuad:
    """Wall-thickness slab between two facing walls.

    Corners are XZ points: a0, a1 on wall_a's bottom edge; b0, b1 on
    wall_b's bottom edge. The quad is wound a0 -> a1 -> b1 -> b0 (CCW
    when looking from above, assuming wall_b is to the left of wall_a's
    tangent direction).
    """

    wall_a: WallRun
    wall_b: WallRun
    a0: tuple[float, float]
    a1: tuple[float, float]
    b0: tuple[float, float]
    b1: tuple[float, float]
    floor_y: float
    ceiling_y_a: float  # ceiling on wall_a's side (room A)
    ceiling_y_b: float  # ceiling on wall_b's side (room B)
    thickness_m: float  # perpendicular distance between the two walls


def _ceiling_profile_for_room(room: ExtractedRoom) -> CeilingProfile | None:
    facets: list[CeilingFacet] = []
    for raw_plane in room.raw_ceiling_planes:
        corners = raw_plane.corners
        if len(corners) < 3:
            continue
        coords = [(float(c[0]), float(c[2])) for c in corners]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if not poly.is_valid or poly.is_empty:
            continue
        ys = [float(c[1]) for c in corners]
        if max(ys) - min(ys) <= 0.10:
            ceiling = RoomCeiling(plane=None, flat_y=float(sorted(ys)[len(ys) // 2]))
        else:
            plane = Plane.fit(corners)
            if isinstance(plane, FitFailure):
                continue
            ceiling = RoomCeiling(plane=plane, flat_y=None)
        facets.append(CeilingFacet(ceiling=ceiling, xz_polygon=poly))

    fallback = None
    ceiling = room_ceiling(room)
    xz_polygon = None
    if ceiling is not None and room.ceiling_polygon and len(room.ceiling_polygon) >= 3:
        coords = [(float(c[0]), float(c[2])) for c in room.ceiling_polygon]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                xz_polygon = poly
        except Exception:
            xz_polygon = None
        fallback = CeilingFacet(ceiling=ceiling, xz_polygon=xz_polygon)
    if facets or fallback is not None:
        return CeilingProfile(facets=tuple(facets), fallback=fallback)
    return None


def _attached_descent_bottom_y(
    room: ExtractedRoom, endpoint: tuple[float, float]
) -> float | None:
    best: float | None = None
    for wall in room.walls_computed:
        for strip in wall.descent_strip or []:
            if not any(
                math.hypot(
                    float(corner[0]) - endpoint[0], float(corner[2]) - endpoint[1]
                )
                <= ENDPOINT_SNAP_TOL_M
                for corner in strip
            ):
                continue
            strip_min = min(float(corner[1]) for corner in strip)
            best = strip_min if best is None else min(best, strip_min)
    return best


def _wall_run(
    room_index: int, wall_index: int, wall: ExtractedWall, room: ExtractedRoom
) -> WallRun | None:
    if wall.synthetic or len(wall.corners) < 2:
        return None
    ys = [float(c[1]) for c in wall.corners]
    y_low, y_high = min(ys), max(ys)
    base = [c for c in wall.corners if abs(float(c[1]) - y_low) <= 1e-3]
    if len(base) < 2:
        base = wall.corners[:2]
    p0 = (float(base[0][0]), float(base[0][2]))
    p1 = (float(base[1][0]), float(base[1][2]))
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if length < 1e-6:
        return None
    tangent = ((p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length)
    # Separate corners into top and bottom by y; project each onto the
    # tangent to get its (t, y) profile entry. The descent/uplift strips
    # — which capture the actual wall top profile when scanned ceilings
    # disagree with the room's nominal wall-top — are folded in here so a
    # gable-end wall produces a sloped top profile instead of a flat one.
    median_y = 0.5 * (y_low + y_high)
    top_pts: list[tuple[float, float]] = []
    bottom_pts: list[tuple[float, float]] = []
    for c in wall.corners:
        cy = float(c[1])
        t = (float(c[0]) - p0[0]) * tangent[0] + (float(c[2]) - p0[1]) * tangent[1]
        if cy >= median_y:
            top_pts.append((t, cy))
        else:
            bottom_pts.append((t, cy))
    for strip in (wall.uplift_strip or []) + (wall.descent_strip or []):
        for c in strip:
            cy = float(c[1])
            t = (float(c[0]) - p0[0]) * tangent[0] + (float(c[2]) - p0[1]) * tangent[1]
            # Strip points represent additional wall-top samples (uplift)
            # or wall-bottom samples (descent); classify by y vs median.
            if cy >= median_y:
                top_pts.append((t, cy))
            else:
                bottom_pts.append((t, cy))
    if not top_pts:
        top_pts = [(0.0, y_high), (length, y_high)]
    if not bottom_pts:
        bottom_pts = [(0.0, y_low), (length, y_low)]
    top_profile = _dedupe_profile_points(top_pts, prefer="low")
    bottom_profile = _dedupe_profile_points(bottom_pts, prefer="low")
    p0_bottom = _interp_profile(bottom_profile, 0.0)
    p1_bottom = _interp_profile(bottom_profile, length)
    attached_p0 = _attached_descent_bottom_y(room, p0)
    attached_p1 = _attached_descent_bottom_y(room, p1)
    endpoint_bottoms = (
        min(p0_bottom, attached_p0) if attached_p0 is not None else p0_bottom,
        min(p1_bottom, attached_p1) if attached_p1 is not None else p1_bottom,
    )
    ceiling_profile = _ceiling_profile_for_room(room)
    opening_intervals = []
    for element in (*room.doors, *room.openings):
        if element.parent_wall_id != wall.id:
            continue
        ts = [
            (float(corner[0]) - p0[0]) * tangent[0]
            + (float(corner[2]) - p0[1]) * tangent[1]
            for corner in element.corners
        ]
        if ts:
            opening_intervals.append((min(ts), max(ts)))
    has_opening = bool(opening_intervals)
    return WallRun(
        room_index=room_index,
        wall_index=wall_index,
        p0=p0,
        p1=p1,
        y_low=y_low,
        y_high=y_high,
        top_profile=top_profile,
        bottom_profile=bottom_profile,
        endpoint_bottoms=endpoint_bottoms,
        ceiling_profile=ceiling_profile,
        has_opening=has_opening,
        opening_intervals=tuple(opening_intervals),
    )


def collect_runs(rooms: Sequence[ExtractedRoom]) -> dict[int, list[WallRun]]:
    """Group bottom-edge wall runs by story."""
    runs_by_story: dict[int, list[WallRun]] = {}
    for ridx, room in enumerate(rooms):
        for widx, wall in enumerate(room.walls_computed):
            run = _wall_run(ridx, widx, wall, room)
            if run is None:
                continue
            runs_by_story.setdefault(room.story, []).append(run)
    return runs_by_story


def _ceiling_edge_runs(rooms: Sequence[ExtractedRoom]) -> dict[int, list[WallRun]]:
    """Synthesize ``WallRun`` instances from ceiling polygon edges. These act
    as fallback "walls" for cause 5 — when a real shared wall is genuinely
    absent (doorway, opening, or scan defect) but the rooms' ceilings still
    define the seam.

    Each ceiling polygon edge becomes a WallRun whose:
      - p0/p1 are the edge endpoints' XZ coordinates,
      - y_low is the room's floor y,
      - y_high is the ceiling polygon y at the edge,
      - top/bottom profiles are constants (the slab is rectangular, not
        kinked, since it represents synthesized wall-thickness for an
        absent wall).

    Indexed at ``wall_index = -(edge_index + 1)`` so they don't collide
    with real wall_index values."""
    runs_by_story: dict[int, list[WallRun]] = {}
    for ridx, room in enumerate(rooms):
        if not room.ceiling_polygon or len(room.ceiling_polygon) < 3:
            continue
        ceiling_profile = _ceiling_profile_for_room(room)
        floor_y = float(room.floor_polygon[0][1]) if room.floor_polygon else 0.0
        n = len(room.ceiling_polygon)
        for edge_idx in range(n):
            c0 = room.ceiling_polygon[edge_idx]
            c1 = room.ceiling_polygon[(edge_idx + 1) % n]
            p0 = (float(c0[0]), float(c0[2]))
            p1 = (float(c1[0]), float(c1[2]))
            length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if length < MIN_WALL_OVERLAP_M:
                continue
            run = WallRun(
                room_index=ridx,
                wall_index=-(edge_idx + 1),
                p0=p0,
                p1=p1,
                y_low=floor_y,
                y_high=max(float(c0[1]), float(c1[1])),
                top_profile=((0.0, float(c0[1])), (length, float(c1[1]))),
                bottom_profile=((0.0, floor_y), (length, floor_y)),
                endpoint_bottoms=(floor_y, floor_y),
                ceiling_profile=ceiling_profile,
                has_opening=False,
                opening_intervals=(),
            )
            runs_by_story.setdefault(room.story, []).append(run)
    return runs_by_story


def _angle_diff_deg(t_a: tuple[float, float], t_b: tuple[float, float]) -> float:
    """Smallest angle between two tangent directions, ignoring sign."""
    cos = max(-1.0, min(1.0, t_a[0] * t_b[0] + t_a[1] * t_b[1]))
    return math.degrees(math.acos(abs(cos)))


def _project_onto_line(
    p: tuple[float, float], anchor: tuple[float, float], tangent: tuple[float, float]
) -> tuple[float, float, float]:
    """Project p onto the line (anchor, tangent). Returns the foot point and
    the signed parameter along the tangent."""
    rx = p[0] - anchor[0]
    rz = p[1] - anchor[1]
    t = rx * tangent[0] + rz * tangent[1]
    foot = (anchor[0] + t * tangent[0], anchor[1] + t * tangent[1])
    return foot[0], foot[1], t


def _perp_distance(
    a_anchor: tuple[float, float],
    a_tangent: tuple[float, float],
    b: tuple[float, float],
) -> float:
    foot_x, foot_z, _t = _project_onto_line(b, a_anchor, a_tangent)
    return math.hypot(b[0] - foot_x, b[1] - foot_z)


def _segment_overlap_fraction_on_run(
    seg0: tuple[float, float],
    seg1: tuple[float, float],
    run: WallRun,
) -> float:
    seg_len = math.hypot(seg1[0] - seg0[0], seg1[1] - seg0[1])
    if seg_len < 1e-6 or run.length < 1e-6:
        return 0.0
    _x0, _z0, t0 = _project_onto_line(seg0, run.p0, run.tangent)
    _x1, _z1, t1 = _project_onto_line(seg1, run.p0, run.tangent)
    seg_lo, seg_hi = sorted((t0, t1))
    overlap = max(0.0, min(seg_hi, run.length) - max(seg_lo, 0.0))
    return overlap / seg_len


def _run_aligns_with_segment(
    run: WallRun,
    seg0: tuple[float, float],
    seg1: tuple[float, float],
    *,
    angle_tol_deg: float,
) -> bool:
    seg_len = math.hypot(seg1[0] - seg0[0], seg1[1] - seg0[1])
    if seg_len < 1e-6:
        return False
    seg_tangent = ((seg1[0] - seg0[0]) / seg_len, (seg1[1] - seg0[1]) / seg_len)
    if _angle_diff_deg(seg_tangent, run.tangent) > angle_tol_deg:
        return False
    if _perp_distance(run.p0, run.tangent, seg0) > RECESS_CAP_ALIGNMENT_TOL_M:
        return False
    if _perp_distance(run.p0, run.tangent, seg1) > RECESS_CAP_ALIGNMENT_TOL_M:
        return False
    return _segment_overlap_fraction_on_run(seg0, seg1, run) >= MIN_RECESS_CAP_COVERAGE


def _has_bounded_recess_cap(
    runs: Sequence[WallRun],
    ra: WallRun,
    rb: WallRun,
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
    *,
    angle_tol_deg: float,
) -> bool:
    """True when a wider-than-wall-thickness slab is visibly bounded by a
    third wall at either short end. This models a narrow exterior recess, not
    ordinary wall thickness, so the cap-wall signal is required."""

    if not (ra.has_opening or rb.has_opening):
        return False

    def opening_touches(run: WallRun, t: float) -> bool:
        return any(
            min(abs(t - lo), abs(t - hi)) <= OPENING_RECESS_END_TOL_M
            for lo, hi in run.opening_intervals
        )

    start_has_opening = opening_touches(ra, _t_along(a0, ra)) or opening_touches(
        rb, _t_along(b0, rb)
    )
    end_has_opening = opening_touches(ra, _t_along(a1, ra)) or opening_touches(
        rb, _t_along(b1, rb)
    )

    paired = {(ra.room_index, ra.wall_index), (rb.room_index, rb.wall_index)}
    for run in runs:
        if (run.room_index, run.wall_index) in paired:
            continue
        if run.room_index in {ra.room_index, rb.room_index}:
            continue
        if start_has_opening and _run_aligns_with_segment(
            run, a0, b0, angle_tol_deg=angle_tol_deg
        ):
            return True
        if end_has_opening and _run_aligns_with_segment(
            run, a1, b1, angle_tol_deg=angle_tol_deg
        ):
            return True
    return False


def _ceiling_y_for_room(room: ExtractedRoom, fallback: float) -> float:
    """Pick the room's ceiling y. Prefer the median y of its ceiling polygon
    when one exists; otherwise the median wall-top y; otherwise a fallback."""
    if room.ceiling_polygon and len(room.ceiling_polygon) >= 3:
        ys = [float(c[1]) for c in room.ceiling_polygon]
        return float(sorted(ys)[len(ys) // 2])
    wall_tops: list[float] = []
    for wall in room.walls_computed:
        if wall.synthetic:
            continue
        for c in wall.corners:
            wall_tops.append(float(c[1]))
    if not wall_tops:
        return fallback
    wall_tops.sort()
    return wall_tops[len(wall_tops) - len(wall_tops) // 4 - 1]  # rough p75


def find_facing_pairs(
    rooms: Sequence[ExtractedRoom],
    *,
    angle_tol_deg: float = ANGLE_TOLERANCE_DEG,
    min_thickness_m: float = MIN_WALL_THICKNESS_M,
    max_thickness_m: float = MAX_WALL_THICKNESS_M,
    min_bounded_recess_width_m: float = MIN_BOUNDED_RECESS_WIDTH_M,
    max_bounded_recess_width_m: float = MAX_BOUNDED_RECESS_WIDTH_M,
    min_overlap_m: float = MIN_WALL_OVERLAP_M,
) -> list[SlabQuad]:
    """Return slab quads for every facing wall pair across rooms within a
    story. A pair is "facing" when their tangents are roughly parallel
    (within angle_tol_deg of 0° or 180°), the perpendicular distance is in
    [min_thickness_m, max_thickness_m], and the tangent-projected overlap
    is at least min_overlap_m.

    Two-pass detection:
      1. Pair real wall runs with each other (cause 1 — wall-thickness slab
         from two scanned walls).
      2. Where rooms have ceiling polygons that face each other but no real
         wall-pair backs them (e.g., the wall is genuinely missing because
         the rooms connect via a doorway, or the wall wasn't scanned),
         pair their CEILING POLYGON EDGES instead (cause 5 — ceiling-edge
         slab). Vertices come from ceiling polygon corners. Dedupe against
         already-emitted wall slabs by XZ-overlap of the slab quad."""
    runs_by_story = collect_runs(rooms)
    ceiling_runs_by_story = _ceiling_edge_runs(rooms)
    slabs: list[SlabQuad] = []
    for _story, runs in runs_by_story.items():
        for i, ra in enumerate(runs):
            for rb in runs[i + 1 :]:
                if ra.room_index == rb.room_index:
                    continue
                if _angle_diff_deg(ra.tangent, rb.tangent) > angle_tol_deg:
                    continue
                # Perpendicular distance — measured at both endpoints of
                # rb against ra's line, then averaged. Reject if either
                # endpoint falls outside the thickness band.
                d0 = _perp_distance(ra.p0, ra.tangent, rb.p0)
                d1 = _perp_distance(ra.p0, ra.tangent, rb.p1)
                min_d = min(d0, d1)
                max_d = max(d0, d1)
                if min_d < min_thickness_m:
                    continue
                # Project rb endpoints onto ra to find overlap interval.
                _x0, _z0, t_b0 = _project_onto_line(rb.p0, ra.p0, ra.tangent)
                _x1, _z1, t_b1 = _project_onto_line(rb.p1, ra.p0, ra.tangent)
                t_lo, t_hi = sorted((t_b0, t_b1))
                ov_lo = max(0.0, t_lo)
                ov_hi = min(ra.length, t_hi)
                if ov_hi - ov_lo < min_overlap_m:
                    continue
                # Slab corners on wall_a:
                a0 = (
                    ra.p0[0] + ov_lo * ra.tangent[0],
                    ra.p0[1] + ov_lo * ra.tangent[1],
                )
                a1 = (
                    ra.p0[0] + ov_hi * ra.tangent[0],
                    ra.p0[1] + ov_hi * ra.tangent[1],
                )
                # Snap slab corners back to actual wall endpoints when close —
                # avoids the "near-coincident but not equal" floating-point
                # wrinkle that produced phantom slivers in the old pipeline.
                a0 = _snap_to_endpoint(a0, ra)
                a1 = _snap_to_endpoint(a1, ra)
                # Drop perpendicular from a0/a1 to wall_b's line.
                b0 = _drop_perpendicular(a0, rb)
                b1 = _drop_perpendicular(a1, rb)
                b0 = _snap_to_endpoint(b0, rb)
                b1 = _snap_to_endpoint(b1, rb)
                if max_d > max_thickness_m:
                    if (
                        max_d < min_bounded_recess_width_m
                        or max_d > max_bounded_recess_width_m
                    ):
                        continue
                    if not _has_bounded_recess_cap(
                        runs,
                        ra,
                        rb,
                        a0,
                        a1,
                        b0,
                        b1,
                        angle_tol_deg=angle_tol_deg,
                    ):
                        continue
                thickness = 0.5 * (d0 + d1)
                # Floor y from rooms.
                room_a = rooms[ra.room_index]
                room_b = rooms[rb.room_index]
                fa = room_a.floor_polygon[0][1] if room_a.floor_polygon else ra.y_low
                fb = room_b.floor_polygon[0][1] if room_b.floor_polygon else rb.y_low
                floor_y = max(float(fa), float(fb))
                ceil_y_a = _ceiling_y_for_room(room_a, ra.y_high)
                ceil_y_b = _ceiling_y_for_room(room_b, rb.y_high)
                slabs.append(
                    SlabQuad(
                        wall_a=ra,
                        wall_b=rb,
                        a0=a0,
                        a1=a1,
                        b0=b0,
                        b1=b1,
                        floor_y=floor_y,
                        ceiling_y_a=ceil_y_a,
                        ceiling_y_b=ceil_y_b,
                        thickness_m=thickness,
                    )
                )

    # Cause-5 second pass: pair ceiling-polygon edges between rooms that
    # don't already have a wall-pair slab covering the same XZ region.
    # This catches doorway / opening / scan-defect cases where the
    # rooms genuinely lack a shared wall but their ceilings still have a
    # seam that needs filling.
    for story, ceiling_runs in ceiling_runs_by_story.items():
        existing_slab_polys = [
            _slab_xz_polygon(s)
            for s in slabs
            if rooms[s.wall_a.room_index].story == story
        ]
        for i, ra in enumerate(ceiling_runs):
            for rb in ceiling_runs[i + 1 :]:
                if ra.room_index == rb.room_index:
                    continue
                if _angle_diff_deg(ra.tangent, rb.tangent) > angle_tol_deg:
                    continue
                d0 = _perp_distance(ra.p0, ra.tangent, rb.p0)
                d1 = _perp_distance(ra.p0, ra.tangent, rb.p1)
                if min(d0, d1) < min_thickness_m or max(d0, d1) > max_thickness_m:
                    continue
                thickness = 0.5 * (d0 + d1)
                _x0, _z0, t_b0 = _project_onto_line(rb.p0, ra.p0, ra.tangent)
                _x1, _z1, t_b1 = _project_onto_line(rb.p1, ra.p0, ra.tangent)
                t_lo, t_hi = sorted((t_b0, t_b1))
                ov_lo = max(0.0, t_lo)
                ov_hi = min(ra.length, t_hi)
                if ov_hi - ov_lo < min_overlap_m:
                    continue
                a0 = (
                    ra.p0[0] + ov_lo * ra.tangent[0],
                    ra.p0[1] + ov_lo * ra.tangent[1],
                )
                a1 = (
                    ra.p0[0] + ov_hi * ra.tangent[0],
                    ra.p0[1] + ov_hi * ra.tangent[1],
                )
                b0 = _drop_perpendicular(a0, rb)
                b1 = _drop_perpendicular(a1, rb)
                # Skip if a real wall-pair slab already covers this XZ region.
                proposed_xz = (a0, a1, b1, b0)
                if any(
                    _xz_polygon_overlap(proposed_xz, ex) > 0.5
                    for ex in existing_slab_polys
                ):
                    continue
                room_a = rooms[ra.room_index]
                room_b = rooms[rb.room_index]
                fa = room_a.floor_polygon[0][1] if room_a.floor_polygon else ra.y_low
                fb = room_b.floor_polygon[0][1] if room_b.floor_polygon else rb.y_low
                floor_y = max(float(fa), float(fb))
                ceil_y_a = _ceiling_y_for_room(room_a, ra.y_high)
                ceil_y_b = _ceiling_y_for_room(room_b, rb.y_high)
                slabs.append(
                    SlabQuad(
                        wall_a=ra,
                        wall_b=rb,
                        a0=a0,
                        a1=a1,
                        b0=b0,
                        b1=b1,
                        floor_y=floor_y,
                        ceiling_y_a=ceil_y_a,
                        ceiling_y_b=ceil_y_b,
                        thickness_m=thickness,
                    )
                )
    return slabs


def _slab_xz_polygon(slab: SlabQuad) -> tuple[tuple[float, float], ...]:
    return (slab.a0, slab.a1, slab.b1, slab.b0)


def _xz_polygon_overlap(
    a: tuple[tuple[float, float], ...],
    b: tuple[tuple[float, float], ...],
) -> float:
    """Fractional overlap of two XZ quads: |intersection| / |a|. Used only
    to dedupe ceiling-edge slabs against wall-pair slabs that cover the
    same physical seam."""
    from shapely.geometry import Polygon

    try:
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid or not pb.is_valid or pa.is_empty or pb.is_empty:
            return 0.0
        return float(pa.intersection(pb).area / max(pa.area, 1e-9))
    except Exception:
        return 0.0


def _drop_perpendicular(p: tuple[float, float], run: WallRun) -> tuple[float, float]:
    """Foot of the perpendicular from p to run's line, clamped to the run's
    endpoint range when the foot would fall beyond the run."""
    _fx, _fz, t = _project_onto_line(p, run.p0, run.tangent)
    t = max(0.0, min(run.length, t))
    return (run.p0[0] + t * run.tangent[0], run.p0[1] + t * run.tangent[1])


def _t_along(p: tuple[float, float], run: WallRun) -> float:
    """Tangent-direction parameter of point p along the wall run, where
    t=0 is at run.p0 and t=run.length is at run.p1."""
    return (p[0] - run.p0[0]) * run.tangent[0] + (p[1] - run.p0[1]) * run.tangent[1]


def _snap_to_endpoint(p: tuple[float, float], run: WallRun) -> tuple[float, float]:
    """If p is within ENDPOINT_SNAP_TOL_M of either run.p0 or run.p1, return
    the actual endpoint coordinate."""
    if math.hypot(p[0] - run.p0[0], p[1] - run.p0[1]) < ENDPOINT_SNAP_TOL_M:
        return run.p0
    if math.hypot(p[0] - run.p1[0], p[1] - run.p1[1]) < ENDPOINT_SNAP_TOL_M:
        return run.p1
    return p


# Re-exports — slab-quad breakpoint logic and wall emission live in
# `_slab_walls`. `compute_slab_walls` is the main public entry point
# alongside `find_facing_pairs` (above).
from reconcile_tiers.extract._slab_walls import (  # noqa: E402, F401
    compute_slab_walls,
    slab_locator_id,
)
