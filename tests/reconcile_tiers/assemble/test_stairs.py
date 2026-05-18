"""Tests for `reconcile_tiers.assemble.stairs.reconstruct_stairs`.

Geometric invariants verified against real `pipeline-outputs/` data so the
assertions track user-visible reconstruction behaviour rather than internal
helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from reconcile_tiers.assemble.stairs import (
    GOING_TYPICAL_M,
    RISE_HARD_MAX_M,
    RISE_MIN_M,
    reconstruct_stairs,
)
from reconcile_tiers.payload.schema import StairSegmentType

PIPELINE_OUTPUTS = Path("pipeline-outputs")


def _load(uuid: str):
    merged = json.loads((PIPELINE_OUTPUTS / uuid / "merged.json").read_text())
    payload = json.loads((PIPELINE_OUTPUTS / uuid / "tier_payload.json").read_text())
    return merged, payload


def _reconstruct(uuid: str, drops_sink: list | None = None):
    merged, payload = _load(uuid)
    return reconstruct_stairs(
        merged,
        payload.get("rooms") or [],
        payload.get("ceiling") or [],
        uuid=uuid,
        drops_sink=drops_sink,
    )


def _segments_xz(stair) -> list[tuple[tuple[float, float], float, float, float]]:
    """Return [(foot_xz, heading_rad, length, width)] for each STAIR segment.
    Chains through landings the same way the viewer does: LEFT/RIGHT
    attachment sides accumulate +/-90 deg on the cursor's heading.
    """
    cursor = (stair.position.x, stair.position.z)
    heading = stair.rotation
    out = []
    for seg in stair.segments:
        if seg.attachment_side == "left":
            heading += math.pi / 2
        elif seg.attachment_side == "right":
            heading -= math.pi / 2
        # else FRONT -- no rotation
        if seg.segment_type == "stair":
            out.append((cursor, heading, seg.length, seg.width))
        # advance cursor along this segment's heading
        cursor = (
            cursor[0] + math.sin(heading) * seg.length,
            cursor[1] + math.cos(heading) * seg.length,
        )
    return out


def _segment_rectangle_xz(foot, heading, length, width) -> Polygon:
    ux, uz = math.sin(heading), math.cos(heading)
    sx, sz = -uz, ux
    pts = [
        (foot[0] + sx * width / 2, foot[1] + sz * width / 2),
        (foot[0] - sx * width / 2, foot[1] - sz * width / 2),
        (
            foot[0] - sx * width / 2 + ux * length,
            foot[1] - sz * width / 2 + uz * length,
        ),
        (
            foot[0] + sx * width / 2 + ux * length,
            foot[1] + sz * width / 2 + uz * length,
        ),
    ]
    rect = Polygon(pts)
    return rect if rect.is_valid else rect.buffer(0)


def _reachable_polygon_for_story(payload, story: int) -> Polygon:
    polys = []
    for room in payload.get("rooms") or []:
        if room.get("story") != story:
            continue
        for f in room.get("floor") or []:
            corners = f.get("corners") or []
            if len(corners) < 3:
                continue
            p = Polygon([(c["x"], c["z"]) for c in corners])
            if p.is_valid and not p.is_empty:
                polys.append(p)
    if not polys:
        return Polygon()
    union = unary_union(polys)
    if union.geom_type == "Polygon":
        return union
    return union.buffer(0.30)  # buffer same-story rooms together


# ─────────────── geometric invariants ───────────────


def test_riser_in_legal_or_fallback_band():
    """Every emitted stair has a riser inside [RISE_MIN, RISE_HARD_MAX]."""
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
        "016980bc-6762-4022-bfbf-17df4112e10c",
        "3a034c99-8986-4749-aedb-0a5a04ea803f",
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    ):
        for s in _reconstruct(uuid):
            assert RISE_MIN_M <= s.riser_height <= RISE_HARD_MAX_M, (
                f"{uuid} {s.locator_id}: riser {s.riser_height} out of band"
            )


def test_run_rectangle_inside_reachable_polygon():
    """Each STAIR segment's rectangle sits well inside the union of all
    same-story room floors (buffered to absorb wall thicknesses + portal
    walks). For L/U-shape stairs the second flight turns at the landing, so
    we validate per-segment rectangles, not a single straight projection.
    """
    # Use a generous buffer to mirror the implementation's reachable polygon
    # (which unions adjacent rooms via wall-thickness adjacency + portals).
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
        "016980bc-6762-4022-bfbf-17df4112e10c",
        "3a034c99-8986-4749-aedb-0a5a04ea803f",
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    ):
        _, payload = _load(uuid)
        for s in _reconstruct(uuid):
            polys = []
            for r in payload.get("rooms") or []:
                if r.get("story") != s.from_story:
                    continue
                for f in r.get("floor") or []:
                    cs = f.get("corners") or []
                    if len(cs) >= 3:
                        polys.append(Polygon([(c["x"], c["z"]) for c in cs]))
            if not polys:
                continue
            reachable = unary_union([p.buffer(0.50) for p in polys])
            for foot, heading, length, width in _segments_xz(s):
                rect = _segment_rectangle_xz(foot, heading, length, width)
                if rect.area <= 0:
                    continue
                inside = rect.intersection(reachable).area / rect.area
                # Loose threshold -- the test's reachable buffer is a proxy for
                # the implementation's door-walk; it's not exact.
                assert inside >= 0.70, (
                    f"{uuid} {s.locator_id}: segment rectangle only "
                    f"{inside:.0%} inside reachable"
                )


def test_one_stair_per_adjacent_story_pair_by_default():
    """When no scan signals discriminate, only one stair per (N, N+1) is
    emitted. 287808db has multiple OOBBs on story 0->1 -> multiple stairs
    allowed; 016980bc only signals one transition -> one per pair.
    """
    stairs = _reconstruct("016980bc-6762-4022-bfbf-17df4112e10c")
    pairs = [(s.from_story, s.to_story) for s in stairs]
    assert len(pairs) == len(set(pairs)), f"016980bc emitted duplicates: {pairs}"


def test_multi_stairwell_stairs_are_spatially_distinct():
    """When a building emits multiple stairs on the same (N, N+1), they must
    be at least 2 m apart in XZ -- otherwise the same stairwell is double-counted.
    """
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    ):
        stairs = _reconstruct(uuid)
        by_pair: dict[tuple[int, int], list] = {}
        for s in stairs:
            by_pair.setdefault((s.from_story, s.to_story), []).append(s)
        for pair, lst in by_pair.items():
            for i, a in enumerate(lst):
                for b in lst[i + 1 :]:
                    d = math.hypot(
                        a.position.x - b.position.x, a.position.z - b.position.z
                    )
                    assert d >= 2.0, f"{uuid} {pair}: two stairs only {d:.2f} m apart"


def test_segments_partition_total_step_count():
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
    ):
        for s in _reconstruct(uuid):
            seg_total = sum(seg.step_count for seg in s.segments)
            assert seg_total == s.step_count, (
                f"{uuid} {s.locator_id}: seg sum {seg_total} != total {s.step_count}"
            )


def test_run_length_matches_step_count():
    """Each `stair`-type segment has length = (step_count - 1) x GOING."""
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
        "016980bc-6762-4022-bfbf-17df4112e10c",
    ):
        for s in _reconstruct(uuid):
            for seg in s.segments:
                if seg.segment_type != StairSegmentType.STAIR:
                    continue
                expected = (seg.step_count - 1) * GOING_TYPICAL_M
                assert abs(seg.length - expected) <= 0.05, (
                    f"{uuid} {s.locator_id}: seg.length {seg.length} != expected "
                    f"{expected:.2f}"
                )


def test_slab_opening_polygon_inside_upper_room_when_present():
    """When a slab opening polygon is emitted, its centroid must land inside
    *some* upper-story room's floor polygon (so build.py can punch the hole)."""
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
    ):
        _, payload = _load(uuid)
        rooms_by_story: dict[int, list[Polygon]] = {}
        for r in payload.get("rooms") or []:
            for f in r.get("floor") or []:
                corners = f.get("corners") or []
                if len(corners) < 3:
                    continue
                rooms_by_story.setdefault(r["story"], []).append(
                    Polygon([(c["x"], c["z"]) for c in corners])
                )
        for s in _reconstruct(uuid):
            if not s.slab_opening_polygon:
                continue
            cx = sum(v.x for v in s.slab_opening_polygon) / len(s.slab_opening_polygon)
            cz = sum(v.z for v in s.slab_opening_polygon) / len(s.slab_opening_polygon)
            pt = Point(cx, cz)
            uppers = rooms_by_story.get(s.to_story, [])
            assert any(p.contains(pt) for p in uppers), (
                f"{uuid} {s.locator_id}: slab opening centroid not inside any "
                f"story-{s.to_story} room"
            )


def test_every_adjacent_story_pair_attempted():
    """Every adjacent story pair gets either a stair OR a defect entry -- never
    silently skipped."""
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
    ):
        _, payload = _load(uuid)
        stories = sorted(
            {
                r["story"]
                for r in payload.get("rooms") or []
                if r.get("story") is not None
            }
        )
        drops: list = []
        stairs = _reconstruct(uuid, drops_sink=drops)
        emitted_pairs = {(s.from_story, s.to_story) for s in stairs}
        defect_pairs = {
            (d["base_story"], d["top_story"])
            for d in drops
            if d.get("kind") == "stair_defect"
        }
        for i in range(len(stories) - 1):
            pair = (stories[i], stories[i + 1])
            assert pair in emitted_pairs or pair in defect_pairs, (
                f"{uuid}: story pair {pair} not attempted"
            )


def test_stair_heading_is_wall_aligned():
    """Real residential stairs run parallel or perpendicular to a wall -- never
    on a free angle. Every emitted stair's heading must match (within 2 deg)
    either a wall direction or its perpendicular on either connecting story.
    """
    tol_deg = 2.0
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
        "016980bc-6762-4022-bfbf-17df4112e10c",
        "3a034c99-8986-4749-aedb-0a5a04ea803f",
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    ):
        _, payload = _load(uuid)
        # Wall heading axes (mod 180 deg) per story.
        wall_axes_by_story: dict[int, set[float]] = {}
        for r in payload.get("rooms") or []:
            story = r.get("story")
            if story is None:
                continue
            for w in r.get("walls") or []:
                cs = w.get("corners") or []
                if len(cs) < 2:
                    continue
                xs = [c["x"] for c in cs]
                zs = [c["z"] for c in cs]
                dx = max(xs) - min(xs)
                dz = max(zs) - min(zs)
                if dx + dz < 0.05:
                    continue
                axis = math.degrees(math.atan2(dx, dz)) % 180
                wall_axes_by_story.setdefault(int(story), set()).add(round(axis, 1))
        for s in _reconstruct(uuid):
            heading_deg = math.degrees(s.rotation) % 180
            axes = wall_axes_by_story.get(s.from_story, set()) | wall_axes_by_story.get(
                s.to_story, set()
            )
            assert axes, (
                f"{uuid} {s.locator_id}: no walls on stories "
                f"{s.from_story},{s.to_story}"
            )
            best = min(
                min(abs(heading_deg - a), abs(heading_deg - (a + 90) % 180))
                for a in axes
            )
            best = min(best, 180 - best)
            assert best <= tol_deg, (
                f"{uuid} {s.locator_id}: heading {heading_deg:.1f} deg "
                f"not wall-aligned "
                f"(closest axis Δ={best:.1f} deg)"
            )


def test_no_segment_crosses_a_solid_wall():
    """A stair flight can't pass through a solid wall -- only through doors
    or openings. For each emitted stair, build the set of walls on both
    connecting stories minus a 0.6 m gap around every door / opening, then
    assert no segment rectangle's interior is traversed by any of them.
    """
    from shapely.geometry import LineString

    door_radius = 0.30
    portal_gap = 0.30
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
        "016980bc-6762-4022-bfbf-17df4112e10c",
        "3a034c99-8986-4749-aedb-0a5a04ea803f",
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    ):
        merged, payload = _load(uuid)
        # Per-story solid walls (with portal gaps carved out)
        solid_by_story: dict[int, list[LineString]] = {}
        for r in payload.get("rooms") or []:
            story = r.get("story")
            if story is None:
                continue
            doors = []
            for d in r.get("doors") or []:
                cs = d.get("corners") or []
                if cs:
                    doors.append(
                        Point(
                            sum(c["x"] for c in cs) / len(cs),
                            sum(c["z"] for c in cs) / len(cs),
                        )
                    )
            for o in (merged.get("openings") or []) + (merged.get("doors") or []):
                t = o.get("transform")
                if isinstance(t, list) and len(t) == 16 and o.get("story") == story:
                    doors.append(Point(t[12], t[14]))
            for w in r.get("walls") or []:
                cs = w.get("corners") or []
                if len(cs) < 2:
                    continue
                [c["x"] for c in cs]
                [c["z"] for c in cs]
                # Wall axis = two most distant XZ points
                pts = [(c["x"], c["z"]) for c in cs]
                best = 0.0
                pair = None
                for i, a in enumerate(pts):
                    for b in pts[i + 1 :]:
                        d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                        if d2 > best:
                            best = d2
                            pair = (a, b)
                if pair is None or best < 1e-6:
                    continue
                wall = LineString([pair[0], pair[1]])
                pieces = [(0.0, wall.length)]
                for d_pt in doors:
                    if wall.distance(d_pt) > door_radius:
                        continue
                    t = wall.project(d_pt)
                    lo = max(0.0, t - portal_gap)
                    hi = min(wall.length, t + portal_gap)
                    new_pieces = []
                    for a, b in pieces:
                        if hi <= a or lo >= b:
                            new_pieces.append((a, b))
                        else:
                            if a < lo:
                                new_pieces.append((a, lo))
                            if hi < b:
                                new_pieces.append((hi, b))
                    pieces = new_pieces
                for a, b in pieces:
                    if b - a < 0.10:
                        continue
                    s = wall.interpolate(a)
                    e = wall.interpolate(b)
                    solid_by_story.setdefault(int(story), []).append(
                        LineString([(s.x, s.y), (e.x, e.y)])
                    )
        for s in _reconstruct(uuid):
            for foot, heading, length, width in _segments_xz(s):
                rect = _segment_rectangle_xz(foot, heading, length, width)
                rect_inner = rect.buffer(-0.05)
                if rect_inner.is_empty:
                    continue
                for st in (s.from_story, s.to_story):
                    for wall in solid_by_story.get(st, []):
                        inter = wall.intersection(rect_inner)
                        ilen = float(inter.length) if hasattr(inter, "length") else 0.0
                        assert ilen <= 0.20, (
                            f"{uuid} {s.locator_id}: segment "
                            f"foot=({foot[0]:.2f},{foot[1]:.2f}) "
                            f"crosses a solid wall on story {st} (intersection length "
                            f"{ilen:.2f} m)"
                        )


def test_no_door_or_opening_on_stair_side():
    """A door / opening can't physically open onto the side of a stair flight.
    For every emitted stair, no door / opening on the connecting stories may
    sit inside the run rectangle except within 0.5 m of the foot or top end.
    Only doors on the from/to stories matter -- doors on unrelated floors are
    physically separated by the slab.
    """
    end_tol = 0.5
    width_tol = 0.10
    for uuid in (
        "287808db-3826-4351-b9a1-6f9831bdc870",
        "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
        "016980bc-6762-4022-bfbf-17df4112e10c",
        "3a034c99-8986-4749-aedb-0a5a04ea803f",
        "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    ):
        merged, payload = _load(uuid)
        portals_by_story: dict[int, list[tuple[float, float]]] = {}
        for r in payload.get("rooms") or []:
            story = r.get("story")
            if story is None:
                continue
            for d in r.get("doors") or []:
                cs = d.get("corners") or []
                if cs:
                    portals_by_story.setdefault(int(story), []).append(
                        (
                            sum(c["x"] for c in cs) / len(cs),
                            sum(c["z"] for c in cs) / len(cs),
                        )
                    )
        for o in (merged.get("openings") or []) + (merged.get("doors") or []):
            t = o.get("transform")
            story = o.get("story")
            if isinstance(t, list) and len(t) == 16 and story is not None:
                portals_by_story.setdefault(int(story), []).append((t[12], t[14]))
        for s in _reconstruct(uuid):
            for foot, heading, length, width in _segments_xz(s):
                forward = (math.sin(heading), math.cos(heading))
                side = (forward[1], -forward[0])
                half_w = width / 2 + width_tol
                end_max = length - end_tol
                if end_max <= end_tol:
                    continue
                for st in (s.from_story, s.to_story):
                    for px, pz in portals_by_story.get(st, []):
                        dx = px - foot[0]
                        dz = pz - foot[1]
                        local_forward = dx * forward[0] + dz * forward[1]
                        local_side = dx * side[0] + dz * side[1]
                        if (
                            end_tol <= local_forward <= end_max
                            and abs(local_side) <= half_w
                        ):
                            pytest.fail(
                                f"{uuid} {s.locator_id}: door/opening at ({px:.2f}, "
                                f"{pz:.2f}) on story {st} "
                                f"sits on stair side (segment "
                                f"foot=({foot[0]:.2f},{foot[1]:.2f}), "
                                f"forward={local_forward:.2f}, side={local_side:.2f})"
                            )


def test_negative_single_story_building():
    base = PIPELINE_OUTPUTS
    chosen = None
    for d in sorted(base.iterdir()):
        tp_path = d / "tier_payload.json"
        if not tp_path.exists():
            continue
        try:
            tp = json.loads(tp_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rooms = tp.get("rooms") or []
        stories = {r.get("story") for r in rooms if r.get("story") is not None}
        if len(stories) == 1:
            chosen = d.name
            break
    if chosen is None:
        pytest.skip("no single-story building in pipeline-outputs/")
    stairs = _reconstruct(chosen)
    assert stairs == []
