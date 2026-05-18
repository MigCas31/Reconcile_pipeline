"""Reconstruct stairs between adjacent stories using room geometry + BR18 priors.

A multi-story building physically requires a way between adjacent stories.
For every adjacent story pair (N, N+1) we look for a stair connection. Two
scan signals — RoomPlan `stairs` objects (Signal A) and raw-ceiling/upper-
floor overlap polygons (Signal B) — bias *which* `(L, U)` room pair carries
the stair, but absence of signals is not a reason to skip: room geometry
alone (walls, floors, doors, openings) plus residential building-code priors
yields a synthesised placement.

Output mirrors `pascal-editor`'s `StairNode` + `StairSegmentNode` shape so
the editor / tier viewer can render the steps without further translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reconcile_tiers.payload.schema import (
    Stair,
    StairAttachmentSide,
)

# BR18 §57 residential band
RISE_TYPICAL_M = 0.175
RISE_MIN_M = 0.155
RISE_MAX_M = 0.20
RISE_HARD_MAX_M = 0.22  # try-harder fallback before giving up
GOING_TYPICAL_M = 0.27
WIDTH_DEFAULT_M = 0.9
WIDTH_MIN_M = 0.75
LANDING_DEPTH_MIN_M = 0.90

# Same-story rooms count as walkable-connected when their floor polygons
# come within this distance (≈ DK interior wall thickness).
ROOM_ADJACENCY_M = 0.30
PORTAL_ROOM_PROXIMITY_M = 1.0

# Hard fit threshold: ≥ 92 % of the run rectangle must lie inside the
# reachable polygon.
FIT_MIN_FRACTION = 0.92

# Soft scoring weights
W_SIGNAL_A = 3.0
W_SIGNAL_B = 4.0
W_WALL_HUG = 2.0
W_OPENING = 1.5
W_HEADING_AGREEMENT = 1.5
W_FOOTPRINT_OVERLAP = 2.5

# Confidence tags
CONF_SCAN_CONFIRMED = "scan_confirmed"
CONF_SYNTHESIZED = "synthesized"
CONF_SIGNAL_OVERRULED = "signal_overruled"


# ─────────────── data structures ───────────────


@dataclass
class _RoomGeom:
    room_index: int
    story: int
    floor_xz: Polygon
    floor_y: float
    walls_xz: list[LineString]
    doors_xz: list[Point]
    locator_id: str


@dataclass
class _Signal:
    kind: str  # 'A' or 'B'
    centroid_xz: tuple[float, float]
    heading_rad: float | None
    void_polygon: Polygon | None
    base_story: int
    top_story: int
    fragment_id: str | None


@dataclass
class _Candidate:
    base_story: int
    top_story: int
    lower: _RoomGeom
    upper: _RoomGeom
    signals_a: list[_Signal]
    signals_b: list[_Signal]
    connection_score: float


@dataclass
class _Placement:
    foot_xz: np.ndarray
    heading_rad: float
    width: float
    step_count: int
    riser: float
    flight_run: float
    # 'straight' = single flight; 'L' = 90° turn at a landing; 'U' = 180° turn
    # at a landing (switchback). For 'L' and 'U' the first flight runs along
    # `heading_rad`, the landing turns by `landing_turn`, and the second
    # flight runs at +90° (L) or +180° (U) from the first.
    flight_shape: str
    landing_turn: StairAttachmentSide
    quality: str
    score: float
    diag: dict = field(default_factory=dict)


# ─────────────── public entrypoint ───────────────


def reconstruct_stairs(
    merged: dict[str, Any],
    rooms_payload: list[dict[str, Any]],
    ceiling_payload: list[dict[str, Any]],
    *,
    uuid: str,
    drops_sink: list | None = None,
) -> list[Stair]:
    rooms = _index_rooms(rooms_payload)
    if not rooms:
        return []
    story_rooms: dict[int, list[_RoomGeom]] = {}
    for room in rooms:
        story_rooms.setdefault(room.story, []).append(room)
    story_y = {
        st: sum(r.floor_y for r in lst) / len(lst) for st, lst in story_rooms.items()
    }
    stories = sorted(story_rooms.keys())
    if len(stories) < 2:
        return []

    signals_a = _collect_signal_a(merged, story_y)
    signals_b = _collect_signal_b(ceiling_payload, rooms)
    portals_by_story = _index_portals(merged)

    out: list[Stair] = []
    stair_index = 0
    for i in range(len(stories) - 1):
        n = stories[i]
        n1 = stories[i + 1]
        candidates = _score_room_pairs(
            story_rooms[n], story_rooms[n1], signals_a, signals_b
        )
        if not candidates:
            if drops_sink is not None:
                drops_sink.append(
                    {
                        "kind": "stair_defect",
                        "drop_reason": "no candidate room pair",
                        "base_story": n,
                        "top_story": n1,
                    }
                )
            continue
        emitted = 0
        emitted_pairs: set[tuple[int, int]] = set()
        emitted_feet: list[tuple[float, float]] = []
        for cand in candidates:
            pair_key = (cand.lower.room_index, cand.upper.room_index)
            if pair_key in emitted_pairs:
                continue
            # First (best) pair always tried; subsequent only if they carry
            # an independent scan signal of their own.
            if emitted >= 1 and not (cand.signals_a or cand.signals_b):
                continue
            stair = _try_emit(
                cand,
                story_y=story_y,
                same_story=story_rooms[n],
                portals=portals_by_story.get(n, []),
                upper_story_rooms=story_rooms.get(n1, []),
                upper_portals=portals_by_story.get(n1, []),
                uuid=uuid,
                stair_index=stair_index,
                drops_sink=drops_sink,
            )
            if stair is None:
                continue
            # Spatially distinct check: a second stair on the same transition
            # must sit at least 2 m XZ from any already-emitted stair foot,
            # otherwise it's the same physical stairwell witnessed by a
            # different signal — drop the duplicate.
            foot_xz = (stair.position.x, stair.position.z)
            if any(
                math.hypot(foot_xz[0] - f[0], foot_xz[1] - f[1]) < 2.0
                for f in emitted_feet
            ):
                continue
            out.append(stair)
            emitted_pairs.add(pair_key)
            emitted_feet.append(foot_xz)
            stair_index += 1
            emitted += 1
        if emitted == 0 and drops_sink is not None:
            drops_sink.append(
                {
                    "kind": "stair_defect",
                    "drop_reason": f"no fit between story_{n}_and_story_{n1}",
                    "base_story": n,
                    "top_story": n1,
                    "candidates_tried": len(candidates),
                }
            )
    return out


# ─────────────── room indexing ───────────────


def _index_rooms(rooms_payload: list[dict[str, Any]]) -> list[_RoomGeom]:
    out: list[_RoomGeom] = []
    for ri, r in enumerate(rooms_payload):
        story = r.get("story")
        if story is None:
            continue
        floor_xz = None
        floor_y = None
        for f in r.get("floor") or []:
            poly = _polygon_xz(f.get("corners") or [])
            if poly is not None and poly.area > 1e-6:
                ys = [c["y"] for c in f["corners"]]
                floor_xz = poly
                floor_y = sum(ys) / len(ys)
                break
        if floor_xz is None:
            continue
        walls_xz: list[LineString] = []
        for w in r.get("walls") or []:
            line = _wall_xz_axis(w.get("corners") or [])
            if line is not None and line.length > 0.05:
                walls_xz.append(line)
        doors_xz: list[Point] = []
        for d in r.get("doors") or []:
            pt = _quad_centroid_xz(d.get("corners") or [])
            if pt is not None:
                doors_xz.append(pt)
        out.append(
            _RoomGeom(
                room_index=ri,
                story=int(story),
                floor_xz=floor_xz,
                floor_y=float(floor_y),
                walls_xz=walls_xz,
                doors_xz=doors_xz,
                locator_id=r.get("locator_id", f"r{ri}"),
            )
        )
    return out


def _polygon_xz(corners: list[dict[str, float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    pts = [(c["x"], c["z"]) for c in corners]
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or not isinstance(p, Polygon):
        return None
    return p


def _wall_xz_axis(corners: list[dict[str, float]]) -> LineString | None:
    if len(corners) < 2:
        return None
    pts = [(c["x"], c["z"]) for c in corners]
    # Pick the two most-distant XZ points — the wall's footprint endpoints.
    best = 0.0
    pair = None
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d > best:
                best = d
                pair = (a, b)
    if pair is None or best < 1e-6:
        return None
    return LineString([pair[0], pair[1]])


def _quad_centroid_xz(corners: list[dict[str, float]]) -> Point | None:
    if not corners:
        return None
    cx = sum(c["x"] for c in corners) / len(corners)
    cz = sum(c["z"] for c in corners) / len(corners)
    return Point(cx, cz)


def _try_emit(
    cand: _Candidate,
    *,
    story_y: dict[int, float],
    same_story: list[_RoomGeom],
    portals: list[np.ndarray],
    upper_story_rooms: list[_RoomGeom],
    upper_portals: list[np.ndarray],
    uuid: str,
    stair_index: int,
    drops_sink: list | None,
) -> Stair | None:
    total_rise = story_y[cand.top_story] - story_y[cand.base_story]
    if total_rise < 0.30:
        return None
    reachable = _reachable_polygon(cand.lower, same_story, portals)

    # Side-blockers: doors / openings physically can't sit on the side of a
    # stair flight (the slope blocks them). Collect every door / opening XZ
    # centroid across BOTH connecting stories — even ones in rooms we didn't
    # pick as L or U, since the stair's run rectangle can sweep through
    # adjacent rooms reached via the door-walk.
    side_blockers: list[np.ndarray] = []
    for r in same_story + upper_story_rooms:
        for d_pt in r.doors_xz:
            side_blockers.append(np.array([d_pt.x, d_pt.y]))
    side_blockers.extend(portals)
    side_blockers.extend(upper_portals)

    # Solid walls: a stair's run rectangle can't cross a wall except at a door
    # or opening. We treat every wall on both connecting stories as solid,
    # then carve out segments where a portal sits on it (so the stair can pass
    # through that gap). The hard-constraint check rejects placements whose
    # run rectangle's interior is traversed by any solid wall segment.
    solid_walls = _collect_solid_walls(
        rooms=same_story + upper_story_rooms,
        portals_by_story={
            cand.base_story: portals,
            cand.top_story: upper_portals,
        },
    )

    # Try-harder ladder. Each rung tries placements with progressively
    # relaxed constraints; we pick the first rung that produces validated
    # placements. Real DK residential stairs are mostly L-shape or
    # switchback (split flights save floor space), so we try those before
    # falling back to a long straight flight.
    attempts: list[tuple[str, _Placement]] = []
    for stage in (
        ("L_shape", WIDTH_DEFAULT_M, RISE_MAX_M, "normal", "L"),
        ("U_shape", WIDTH_DEFAULT_M, RISE_MAX_M, "normal", "U"),
        ("straight", WIDTH_DEFAULT_M, RISE_MAX_M, "normal", "straight"),
        ("L_shrink_width", WIDTH_MIN_M, RISE_MAX_M, "normal", "L"),
        ("steepen", WIDTH_DEFAULT_M, RISE_HARD_MAX_M, "low_quality", "straight"),
        ("steepen_U", WIDTH_DEFAULT_M, RISE_HARD_MAX_M, "low_quality", "U"),
    ):
        stage_name, width, max_rise, quality, shape = stage
        step_count, riser = _step_priors(total_rise, max_rise=max_rise)
        if step_count is None:
            continue
        flight_run = (step_count - 1) * GOING_TYPICAL_M
        placements = _generate_placements(
            cand=cand,
            width=width,
            flight_run=flight_run,
            step_count=step_count,
            riser=riser,
            flight_shape=shape,
            reachable=reachable,
            quality=quality,
        )
        for p in placements:
            if not _passes_hard_constraints(p, reachable, side_blockers, solid_walls):
                continue
            attempts.append((stage_name, p))
        if attempts:
            break

    if not attempts:
        return None
    # Pick highest-scored attempt at this rung
    best_stage, best = max(attempts, key=lambda kv: kv[1].score if kv[1] else -1.0)
    if best is None:
        return None

    return _build_stair(
        cand=cand,
        placement=best,
        stage=best_stage,
        story_y=story_y,
        uuid=uuid,
        stair_index=stair_index,
    )


def _reachable_polygon(
    base: _RoomGeom, same_story: list[_RoomGeom], portals: list[np.ndarray]
) -> Polygon:
    polys = [base.floor_xz]
    for r in same_story:
        if r.room_index == base.room_index:
            continue
        if base.floor_xz.distance(r.floor_xz) <= ROOM_ADJACENCY_M:
            polys.append(r.floor_xz)
            continue
        # Portal-mediated: any same-story portal close to both base and r
        for p in portals:
            pt = Point(float(p[0]), float(p[1]))
            if (
                base.floor_xz.distance(pt) <= PORTAL_ROOM_PROXIMITY_M
                and r.floor_xz.distance(pt) <= PORTAL_ROOM_PROXIMITY_M
            ):
                polys.append(r.floor_xz)
                break
    union = unary_union(polys)
    if isinstance(union, Polygon):
        return union
    if union.geom_type == "MultiPolygon":
        return max(union.geoms, key=lambda g: g.area)
    return base.floor_xz


def _step_priors(
    total_rise: float, *, max_rise: float = RISE_MAX_M
) -> tuple[int | None, float]:
    """Pick step count s.t. riser ∈ [RISE_MIN, max_rise], closest to RISE_TYPICAL."""
    n_typ = total_rise / RISE_TYPICAL_M
    cands: list[tuple[float, int, float]] = []
    for n in range(max(2, int(n_typ - 3)), int(n_typ + 3) + 1):
        if n < 2:
            continue
        riser = total_rise / n
        if RISE_MIN_M <= riser <= max_rise:
            cands.append((abs(riser - RISE_TYPICAL_M), n, riser))
    if not cands:
        return (None, 0.0)
    cands.sort()
    _, n, riser = cands[0]
    return (n, riser)


# ─────────────── placement generation ───────────────


def _generate_placements(
    *,
    cand: _Candidate,
    width: float,
    flight_run: float,
    step_count: int,
    riser: float,
    flight_shape: str,
    reachable: Polygon,
    quality: str,
) -> list[_Placement]:
    """Generate placement candidates whose heading is grid-aligned with the
    room's walls. Real residential stairs are always either parallel or
    perpendicular to a wall — never on a free angle. We enumerate:

    * Wall-aligned placements: foot at any of {wall start, wall end, or any
      door centroid that lies on the wall}; heading along the wall (both
      directions); body offset perpendicularly into the room.
    * Wall-perpendicular placements: foot at any door on the wall; heading
      perpendicular to the wall, pointing into the room.

    All headings are taken straight from wall vectors, so no diagonal stairs.
    """
    out: list[_Placement] = []
    L = cand.lower
    # First-flight run for fit checking — split shapes only need half the run
    # along the chosen wall before turning at the landing.
    needed_run = flight_run / 2 if flight_shape != "straight" else flight_run

    for wall in L.walls_xz:
        a, b = wall.coords[0], wall.coords[-1]
        wlen = math.hypot(b[0] - a[0], b[1] - a[1])
        if wlen < 0.05:
            continue
        # Unit vector along the wall
        ux = (b[0] - a[0]) / wlen
        uz = (b[1] - a[1]) / wlen

        # Foot anchor candidates on this wall: the two endpoints and any
        # door centroid whose perpendicular distance to the wall is small.
        anchors_along: list[float] = [0.0, wlen]
        for d_pt in L.doors_xz:
            t = (d_pt.x - a[0]) * ux + (d_pt.y - a[1]) * uz
            if not (0.0 <= t <= wlen):
                continue
            perp = abs(-(d_pt.x - a[0]) * uz + (d_pt.y - a[1]) * ux)
            if perp < 0.30:
                anchors_along.append(t)

        # Wall-aligned (parallel) placements
        if wlen >= needed_run * 0.8:
            for t in anchors_along:
                # Foot at point `a + ux*t along the wall`. Two run directions.
                for sign in (+1, -1):
                    run_ux, run_uz = ux * sign, uz * sign
                    if sign == +1 and t > wlen - needed_run * 0.5:
                        continue  # not enough wall left going +
                    if sign == -1 and t < needed_run * 0.5:
                        continue
                    heading = math.atan2(run_ux, run_uz)
                    base_foot = (a[0] + ux * t, a[1] + uz * t)
                    for perp_sign in (+1, -1):
                        px, pz = -run_uz * perp_sign, run_ux * perp_sign
                        foot = (
                            base_foot[0] + px * width / 2,
                            base_foot[1] + pz * width / 2,
                        )
                        test = (foot[0] + px * 0.4, foot[1] + pz * 0.4)
                        if not L.floor_xz.contains(Point(test[0], test[1])):
                            continue
                        landing_turn = (
                            StairAttachmentSide.LEFT
                            if perp_sign == +1
                            else StairAttachmentSide.RIGHT
                        )
                        out.append(
                            _make_placement(
                                cand=cand,
                                foot=foot,
                                heading=heading,
                                width=width,
                                flight_run=flight_run,
                                step_count=step_count,
                                riser=riser,
                                flight_shape=flight_shape,
                                landing_turn=landing_turn,
                                quality=quality,
                                wall=wall,
                            )
                        )

        # Wall-perpendicular placements (foot at a door, heading away from wall)
        # Skip endpoint anchors here — only doors make sense.
        door_ts = [t for t in anchors_along if 0.05 < t < wlen - 0.05]
        for t in door_ts:
            base_foot = (a[0] + ux * t, a[1] + uz * t)
            # Two perpendicular directions; pick whichever points into the room
            for perp_sign in (+1, -1):
                run_ux, run_uz = -uz * perp_sign, ux * perp_sign
                heading = math.atan2(run_ux, run_uz)
                # Test point ahead in heading direction; must be inside the room
                test = (base_foot[0] + run_ux * 0.4, base_foot[1] + run_uz * 0.4)
                if not L.floor_xz.contains(Point(test[0], test[1])):
                    continue
                # Foot offset slightly into the room from the door
                foot = (base_foot[0] + run_ux * 0.05, base_foot[1] + run_uz * 0.05)
                out.append(
                    _make_placement(
                        cand=cand,
                        foot=foot,
                        heading=heading,
                        width=width,
                        flight_run=flight_run,
                        step_count=step_count,
                        riser=riser,
                        flight_shape=flight_shape,
                        landing_turn=StairAttachmentSide.LEFT,
                        quality=quality,
                        wall=wall,
                    )
                )

    return out


def _make_placement(
    *,
    cand: _Candidate,
    foot: tuple[float, float],
    heading: float,
    width: float,
    flight_run: float,
    step_count: int,
    riser: float,
    flight_shape: str,
    landing_turn: StairAttachmentSide,
    quality: str,
    wall: LineString | None,
) -> _Placement:
    foot_arr = np.array(foot, dtype=float)
    score = 0.0
    diag: dict = {"sources": []}
    # Footprint overlap is implicit in candidate; placement scoring mostly
    # captures "where in L does the stair sit".
    # Wall-hugging bonus
    if wall is not None:
        score += W_WALL_HUG
        diag["sources"].append("wall_hug")
    # Opening alignment bonus: foot within 1 m of any door
    for d in cand.lower.doors_xz:
        if d.distance(Point(foot_arr[0], foot_arr[1])) <= 1.0:
            score += W_OPENING
            diag["sources"].append("opening_aligned")
            break
    # Signal A proximity
    for s in cand.signals_a:
        mid = (
            foot_arr[0] + math.sin(heading) * flight_run / 2,
            foot_arr[1] + math.cos(heading) * flight_run / 2,
        )
        d = math.hypot(mid[0] - s.centroid_xz[0], mid[1] - s.centroid_xz[1])
        # full bonus when within 1 m of OOBB centroid, decays to 0 at 3 m
        if d <= 3.0:
            score += W_SIGNAL_A * max(0.0, 1.0 - d / 3.0)
            diag["sources"].append(f"signal_A:{d:.2f}m")
        # Heading agreement bonus when signal carries a heading
        if s.heading_rad is not None:
            cos = math.cos(heading - s.heading_rad)
            # also accept anti-parallel (switchback might invert); take abs
            cos = max(cos, math.cos(heading - s.heading_rad - math.pi))
            score += W_HEADING_AGREEMENT * max(0.0, cos)
    # Signal B alignment: head end inside void polygon
    head = (
        foot_arr[0] + math.sin(heading) * flight_run,
        foot_arr[1] + math.cos(heading) * flight_run,
    )
    head_pt = Point(head)
    for s in cand.signals_b:
        if s.void_polygon is None:
            continue
        if s.void_polygon.contains(head_pt):
            score += W_SIGNAL_B
            diag["sources"].append("signal_B_inside")
            break
        d = s.void_polygon.distance(head_pt)
        if d <= 2.0:
            score += W_SIGNAL_B * max(0.0, 1.0 - d / 2.0)
            diag["sources"].append(f"signal_B:{d:.2f}m")
            break
    if quality == "low_quality":
        score *= 0.7
    return _Placement(
        foot_xz=foot_arr,
        heading_rad=float(heading),
        width=width,
        step_count=step_count,
        riser=riser,
        flight_run=flight_run,
        flight_shape=flight_shape,
        landing_turn=landing_turn,
        quality=quality,
        score=score,
        diag=diag,
    )


def _passes_hard_constraints(
    p: _Placement,
    reachable: Polygon,
    side_blockers: list[np.ndarray],
    solid_walls: list[LineString],
) -> bool:
    """Each flight segment (one for straight, two for L/U) must:
    * fit ≥ FIT_MIN_FRACTION inside the reachable polygon,
    * not have a portal on its side (slope blocks the door),
    * not be crossed in its interior by any solid wall (a wall with no door
      / opening near it). Walls with doors / openings have those gaps carved
      out of `solid_walls` so the stair can legally pass through them.
    """
    segments = _flight_segments(p)
    if not segments:
        return False
    for seg in segments:
        rect = _segment_rectangle(seg)
        if rect.area <= 0:
            return False
        try:
            inside = rect.intersection(reachable).area / rect.area
        except Exception:
            return False
        if inside < FIT_MIN_FRACTION:
            return False
        if _segment_has_side_blocker(seg, side_blockers):
            return False
        if _segment_crosses_solid_wall(seg, rect, solid_walls):
            return False
    return True


def _segment_crosses_solid_wall(
    seg: _FlightSegment, rect: Polygon, solid_walls: list[LineString]
) -> bool:
    """A solid wall traverses the run rectangle's interior if their shared
    intersection length exceeds 0.20 m (a real cross, not just a corner
    touch)."""
    if not solid_walls:
        return False
    rect_inner = rect.buffer(-0.05)
    if rect_inner.is_empty:
        return False
    for wall in solid_walls:
        try:
            inter = wall.intersection(rect_inner)
        except Exception:
            continue
        if inter.is_empty:
            continue
        length = float(inter.length) if hasattr(inter, "length") else 0.0
        if length > 0.20:
            return True
    return False


def _collect_solid_walls(
    *,
    rooms: list[_RoomGeom],
    portals_by_story: dict[int, list[np.ndarray]],
) -> list[LineString]:
    """Return wall line segments minus a 0.6 m gap centred on every door /
    opening that lies on (or close to) the wall. The output is the set of
    physically solid wall portions that a stair must not cross. Portals are
    matched to walls only on the same story — a door on the lower floor
    can't open a gap in an upper-floor wall.
    """
    door_radius = 0.30
    portal_gap = 0.30
    out: list[LineString] = []
    for r in rooms:
        relevant_portals: list[Point] = []
        for d_pt in r.doors_xz:
            relevant_portals.append(d_pt)
        for pt in portals_by_story.get(r.story, []):
            relevant_portals.append(Point(float(pt[0]), float(pt[1])))
        for wall in r.walls_xz:
            wlen = wall.length
            if wlen < 0.05:
                continue
            # Find portal projections onto this wall
            pieces: list[tuple[float, float]] = [(0.0, wlen)]
            for portal_pt in relevant_portals:
                if wall.distance(portal_pt) > door_radius:
                    continue
                t = wall.project(portal_pt)
                lo = max(0.0, t - portal_gap)
                hi = min(wlen, t + portal_gap)
                # Subtract [lo, hi] from current pieces
                new_pieces: list[tuple[float, float]] = []
                for a, b in pieces:
                    if hi <= a or lo >= b:
                        new_pieces.append((a, b))
                    else:
                        if a < lo:
                            new_pieces.append((a, lo))
                        if hi < b:
                            new_pieces.append((hi, b))
                pieces = new_pieces
                if not pieces:
                    break
            for a, b in pieces:
                if b - a < 0.10:
                    continue
                start = wall.interpolate(a)
                end = wall.interpolate(b)
                out.append(LineString([(start.x, start.y), (end.x, end.y)]))
    return out


# Re-exports — flight-segment geometry, stair construction, and slab-opening
# emission live in `_stair_emission`. `_passes_hard_constraints` /
# `_segment_crosses_solid_wall` (above) call into them; tests and the
# try-harder ladder reach them through this module.
from reconcile_tiers.assemble._stair_emission import (  # noqa: E402, F401
    SIDE_BLOCKER_END_TOLERANCE_M,
    SIDE_BLOCKER_WIDTH_TOLERANCE_M,
    _build_stair,
    _flight_segments,
    _FlightSegment,
    _ring_to_vec3,
    _segment_has_side_blocker,
    _segment_rectangle,
    _slab_opening_polygon,
)

# Re-exports — signal collection and pair scoring live in `_stair_signals`.
# `reconstruct_stairs` (above) calls them as the first phase.
from reconcile_tiers.assemble._stair_signals import (  # noqa: E402
    _collect_signal_a,
    _collect_signal_b,
    _index_portals,
    _score_room_pairs,
)
