"""Stair flight-segment geometry and emission.

Decomposes a `_Placement` into one or two flight rectangles, builds the
final `Stair` payload from a `_Candidate` + `_Placement`, and computes
the upper-floor slab opening polygon. Extracted from `stairs.py` so the
emission step is split from the candidate-scoring / try-harder ladder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from reconcile_tiers.payload.schema import (
    Stair,
    StairAttachmentSide,
    StairSegment,
    StairSegmentType,
    StairType,
    Vec3,
)


@dataclass
class _FlightSegment:
    foot_xz: tuple[float, float]
    heading_rad: float
    length: float
    width: float


def _flight_segments(p) -> list[_FlightSegment]:
    from reconcile_tiers.assemble.stairs import LANDING_DEPTH_MIN_M

    """Decompose a placement into one or two flight rectangles for fit check.

    For L / U shapes the chain is `[STAIR(half) → LANDING(landing_depth at
    ±90°) → STAIR(half)]`. The viewer accumulates each segment's
    `attachment_side` as a ±90° rotation of the cursor's heading, so:

    * L-shape: second flight heading = first heading ±90° (one turn).
    * U-shape: second flight heading = first heading ±180° (two turns).

    In both cases the second flight's foot sits at the END of the landing,
    not at the end of the first flight. We mirror that here.
    """
    half = p.flight_run / 2
    if p.flight_shape == "straight":
        return [
            _FlightSegment(
                foot_xz=(float(p.foot_xz[0]), float(p.foot_xz[1])),
                heading_rad=p.heading_rad,
                length=p.flight_run,
                width=p.width,
            )
        ]
    ux1 = math.sin(p.heading_rad)
    uz1 = math.cos(p.heading_rad)
    after_first = (
        float(p.foot_xz[0]) + ux1 * half,
        float(p.foot_xz[1]) + uz1 * half,
    )
    turn = math.pi / 2 if p.landing_turn == StairAttachmentSide.LEFT else -math.pi / 2
    landing_heading = p.heading_rad + turn
    landing_depth = max(p.width, LANDING_DEPTH_MIN_M)
    landing_end = (
        after_first[0] + math.sin(landing_heading) * landing_depth,
        after_first[1] + math.cos(landing_heading) * landing_depth,
    )
    if p.flight_shape == "U":
        second_heading = landing_heading + turn  # cumulative ±180° from first
    else:  # 'L'
        second_heading = landing_heading  # second flight has no further turn
    return [
        _FlightSegment(
            foot_xz=(float(p.foot_xz[0]), float(p.foot_xz[1])),
            heading_rad=p.heading_rad,
            length=half,
            width=p.width,
        ),
        _FlightSegment(
            foot_xz=landing_end,
            heading_rad=second_heading,
            length=half,
            width=p.width,
        ),
    ]


def _segment_rectangle(seg: _FlightSegment) -> Polygon:
    ux = math.sin(seg.heading_rad)
    uz = math.cos(seg.heading_rad)
    sx, sz = -uz, ux
    fx, fz = seg.foot_xz
    L = seg.length
    w = seg.width
    pts = [
        (fx + sx * w / 2, fz + sz * w / 2),
        (fx - sx * w / 2, fz - sz * w / 2),
        (fx - sx * w / 2 + ux * L, fz - sz * w / 2 + uz * L),
        (fx + sx * w / 2 + ux * L, fz + sz * w / 2 + uz * L),
    ]
    poly = Polygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


def _segment_has_side_blocker(
    seg: _FlightSegment, side_blockers: list[np.ndarray]
) -> bool:
    if not side_blockers:
        return False
    forward_x = math.sin(seg.heading_rad)
    forward_z = math.cos(seg.heading_rad)
    side_x = forward_z
    side_z = -forward_x
    half_width = seg.width / 2 + SIDE_BLOCKER_WIDTH_TOLERANCE_M
    end_min = SIDE_BLOCKER_END_TOLERANCE_M
    end_max = seg.length - SIDE_BLOCKER_END_TOLERANCE_M
    if end_max <= end_min:
        return False
    fx, fz = seg.foot_xz
    for pt in side_blockers:
        dx = float(pt[0]) - fx
        dz = float(pt[1]) - fz
        local_forward = dx * forward_x + dz * forward_z
        local_side = dx * side_x + dz * side_z
        if end_min <= local_forward <= end_max and abs(local_side) <= half_width:
            return True
    return False


# A portal sitting on the long edge of the run rectangle (between the foot
# and the top, perpendicular to the heading) is physically impossible — the
# stair's slope blocks the doorway. Doors at the foot and top landing are
# fine, so we leave a small forgiving zone at each end.
SIDE_BLOCKER_END_TOLERANCE_M = 0.5
SIDE_BLOCKER_WIDTH_TOLERANCE_M = 0.10


# ─────────────── emission ───────────────


def _build_stair(
    *,
    cand,
    placement,
    stage: str,
    story_y: dict[int, float],
    uuid: str,
    stair_index: int,
) -> Stair:
    from reconcile_tiers.assemble.stairs import (
        GOING_TYPICAL_M,
        LANDING_DEPTH_MIN_M,
        _step_priors,
    )

    total_rise = story_y[cand.top_story] - story_y[cand.base_story]
    riser = total_rise / placement.step_count
    width = placement.width
    if placement.flight_shape == "straight":
        flight_run = (placement.step_count - 1) * GOING_TYPICAL_M
        segments = [
            StairSegment(
                segment_type=StairSegmentType.STAIR,
                width=round(width, 4),
                length=round(flight_run, 4),
                height=round(total_rise, 4),
                step_count=placement.step_count,
                attachment_side=StairAttachmentSide.FRONT,
            ),
        ]
        total_steps = placement.step_count
    else:
        # L (90° turn) or U (180° turn) — two equal flights with a landing.
        # The viewer's stair renderer chains segments by accumulating the
        # `attachment_side` as a Y-rotation per segment: LEFT/RIGHT add ±90°.
        # So an L-shape needs the landing to turn ±90° and the second stair
        # to attach FRONT (no further turn). A U-shape needs both the landing
        # and the second stair to turn ±90° (cumulative 180°).
        rise_per = total_rise / 2
        n_per, _riser_per = _step_priors(rise_per)
        if n_per is None:
            n_per = max(2, placement.step_count // 2)
            rise_per / n_per
        run_per = (n_per - 1) * GOING_TYPICAL_M
        landing = max(width, LANDING_DEPTH_MIN_M)
        if placement.flight_shape == "U":
            second_attach = placement.landing_turn
        else:  # 'L'
            second_attach = StairAttachmentSide.FRONT
        segments = [
            StairSegment(
                segment_type=StairSegmentType.STAIR,
                width=round(width, 4),
                length=round(run_per, 4),
                height=round(rise_per, 4),
                step_count=n_per,
                attachment_side=StairAttachmentSide.FRONT,
            ),
            StairSegment(
                segment_type=StairSegmentType.LANDING,
                width=round(width, 4),
                length=round(landing, 4),
                height=0.0,
                step_count=0,
                attachment_side=placement.landing_turn,
            ),
            StairSegment(
                segment_type=StairSegmentType.STAIR,
                width=round(width, 4),
                length=round(run_per, 4),
                height=round(rise_per, 4),
                step_count=n_per,
                attachment_side=second_attach,
            ),
        ]
        total_steps = n_per * 2

    slab_polygon = _slab_opening_polygon(cand, placement)
    fragment_ids = [s.fragment_id for s in cand.signals_a if s.fragment_id]
    fragment_ids += [s.fragment_id for s in cand.signals_b if s.fragment_id]

    locator_id = f"{uuid}::tier-stair::{stair_index}"
    return Stair(
        locator_id=locator_id,
        position=Vec3(
            x=float(placement.foot_xz[0]),
            y=float(story_y[cand.base_story]),
            z=float(placement.foot_xz[1]),
        ),
        rotation=float(placement.heading_rad),
        stair_type=StairType.STRAIGHT,
        width=round(float(width), 4),
        total_rise=round(float(total_rise), 4),
        step_count=int(total_steps),
        from_story=cand.base_story,
        to_story=cand.top_story,
        segments=segments,
        riser_height=round(float(riser), 4),
        source_fragment_ids=fragment_ids,
        capture_coverage=1.0,
        slab_opening_polygon=slab_polygon,
    )


def _slab_opening_polygon(cand, p) -> list[Vec3]:
    upper_y = cand.upper.floor_y
    # Prefer a Signal-B void polygon clipped to the upper room's floor.
    for s in cand.signals_b:
        if s.void_polygon is None:
            continue
        clipped = s.void_polygon.intersection(cand.upper.floor_xz)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.area < 0.05:
            continue
        return _ring_to_vec3(clipped, upper_y)
    # Synthesised: a rectangle covering the upper half of the *final* flight,
    # axis-aligned with that flight's heading. For L/U shapes the final flight
    # turns at the landing, so the head sits offset from the foot.
    segs = _flight_segments(p)
    final = segs[-1]
    ux = math.sin(final.heading_rad)
    uz = math.cos(final.heading_rad)
    sx, sz = -uz, ux
    w = final.width
    fx, fz = final.foot_xz
    head_mid = (fx + ux * final.length * 0.75, fz + uz * final.length * 0.75)
    half_run = final.length / 2
    rect = Polygon(
        [
            (
                head_mid[0] - ux * half_run + sx * w / 2,
                head_mid[1] - uz * half_run + sz * w / 2,
            ),
            (
                head_mid[0] + ux * half_run + sx * w / 2,
                head_mid[1] + uz * half_run + sz * w / 2,
            ),
            (
                head_mid[0] + ux * half_run - sx * w / 2,
                head_mid[1] + uz * half_run - sz * w / 2,
            ),
            (
                head_mid[0] - ux * half_run - sx * w / 2,
                head_mid[1] - uz * half_run - sz * w / 2,
            ),
        ]
    )
    if not rect.is_valid:
        rect = rect.buffer(0)
    clipped = rect.intersection(cand.upper.floor_xz)
    if clipped.is_empty:
        return []
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)
    if clipped.area < 0.05:
        return []
    return _ring_to_vec3(clipped, upper_y)


def _ring_to_vec3(geom, y: float) -> list[Vec3]:
    coords = list(geom.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [Vec3(x=float(x), y=float(y), z=float(z)) for x, z in coords]
