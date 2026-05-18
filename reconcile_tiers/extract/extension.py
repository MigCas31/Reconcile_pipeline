"""Wall extensions across the inter-story void.

Two operations replace the old lower-wall-up `extension_strip`:

* :func:`compute_descent_strips` — for every upper-story wall on the
  upper-story footprint perimeter, attach a downward strip from the wall's
  bottom edge down to the lower-story ceiling. Where the perimeter has no
  scanned upper wall, synthesise one.
* :func:`compute_uplift_strips` — for every lower-story interior partition
  whose XZ projection sits inside the upper-story footprint, attach an
  upward strip from the wall's top edge up to the upper-story floor.

The upper-wall plane owns the inter-story face; the lower-wall plane closes
the void above interior partitions only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract.building import ExtractedRoom, ExtractedWall

TOP_EPSILON_M = 0.05
MAX_GAP_TO_SLAB_M = 0.80

# Picked from the corpus distribution (analysis_outputs/perimeter_distance_audit.py):
# upper-story walls bimodally separate into a sharp peak at 0.0 m
# (clearly perimeter, 52% of all walls) and a broad interior cluster
# from 0.5 m onward. The valley between them sits at 0.05 m, well past
# the scan-noise floor and well before the interior cluster begins.
PERIMETER_DISTANCE_TOL_M = 0.05

# Below this, two perimeter walls' XZ projections are treated as
# overlapping when subtracting from the boundary to find missing arcs.
# Same scan-noise scale as the perimeter tolerance.
PERIMETER_COVERAGE_BUFFER_M = 0.05

MIN_SYNTHETIC_ARC_LENGTH_M = 0.10


def _xz_polygon(corners: list[list[float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    poly = make_valid(Polygon([(float(c[0]), float(c[2])) for c in corners]))
    if not isinstance(poly, Polygon) or poly.is_empty or not poly.is_valid:
        return None
    return poly


def _wall_bottom_segment(
    corners: list[list[float]],
) -> tuple[LineString, list[list[float]]] | None:
    if len(corners) < 3:
        return None
    ys = [float(c[1]) for c in corners]
    min_y = min(ys)
    max_y = max(ys)
    if max_y - min_y < 0.10:
        return None
    threshold = min_y + (max_y - min_y) * 0.4
    bottoms = [(idx, c) for idx, c in enumerate(corners) if ys[idx] < threshold]
    if len(bottoms) < 2:
        return None
    by_x = sorted(bottoms, key=lambda item: (float(item[1][0]), float(item[1][2])))
    a = by_x[0][1]
    b = by_x[-1][1]
    seg = LineString([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])
    if seg.length < 1e-6:
        return None
    return seg, [list(c) for c in (a, b)]


def _wall_top_segment(
    corners: list[list[float]],
) -> tuple[LineString, list[list[float]]] | None:
    if len(corners) < 3:
        return None
    ys = [float(c[1]) for c in corners]
    min_y = min(ys)
    max_y = max(ys)
    if max_y - min_y < 0.10:
        return None
    threshold = min_y + (max_y - min_y) * 0.6
    tops = [(idx, c) for idx, c in enumerate(corners) if ys[idx] > threshold]
    if len(tops) < 2:
        return None
    by_x = sorted(tops, key=lambda item: (float(item[1][0]), float(item[1][2])))
    a = by_x[0][1]
    b = by_x[-1][1]
    seg = LineString([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])
    if seg.length < 1e-6:
        return None
    return seg, [list(c) for c in (a, b)]


def _segment_median_distance(seg: LineString, boundary) -> float:
    n = max(8, int(seg.length / 0.05))
    distances = sorted(
        seg.interpolate(i / n, normalized=True).distance(boundary) for i in range(n + 1)
    )
    return distances[len(distances) // 2]


def _room_floor_y(room: ExtractedRoom) -> float | None:
    if len(room.floor_polygon) < 3:
        return None
    return float(np.mean([float(c[1]) for c in room.floor_polygon]))


def _room_ceiling_y(room: ExtractedRoom) -> float | None:
    tops = [
        max(float(c[1]) for c in wall.corners)
        for wall in room.walls_computed
        if len(wall.corners) >= 3
    ]
    if not tops:
        return None
    return float(np.median(tops))


# How close (XZ distance) a lower wall has to be to the upper wall's bottom
# segment to count as "directly below". Wall thickness scale; smaller than
# typical room widths (so we don't pick up walls from across the room) but
# large enough to absorb scan noise + small wall-thickness mismatches.
LOWER_WALL_PROXIMITY_M = 0.30


def _wall_top_edge_3d(wall: ExtractedWall) -> tuple[LineString, float, float] | None:
    """Return the wall top edge as (xz_segment, y_at_start, y_at_end).

    The top edge is the line between the two extreme top corners, so a sloped
    wall top (gable, mansard) preserves its slope: y_at_start ≠ y_at_end.
    """
    if len(wall.corners) < 3:
        return None
    ys = [float(c[1]) for c in wall.corners]
    min_y = min(ys)
    max_y = max(ys)
    if max_y - min_y < 0.10:
        return None
    threshold = min_y + (max_y - min_y) * 0.6
    tops = [c for c in wall.corners if float(c[1]) > threshold]
    if len(tops) < 2:
        return None
    ts = sorted(tops, key=lambda c: (float(c[0]), float(c[2])))
    a = ts[0]
    b = ts[-1]
    seg = LineString([(float(a[0]), float(a[2])), (float(b[0]), float(b[2]))])
    if seg.length < 1e-6:
        return None
    return seg, float(a[1]), float(b[1])


def _lower_wall_tops(
    rooms: list[ExtractedRoom], story: int
) -> list[tuple[LineString, float, float]]:
    """Top edges (with per-endpoint Y) for every wall on `story`."""
    out: list[tuple[LineString, float, float]] = []
    for room in rooms:
        if room.story != story:
            continue
        for wall in room.walls_computed:
            edge = _wall_top_edge_3d(wall)
            if edge is not None:
                out.append(edge)
    return out


def _all_wall_tops_below(
    rooms: list[ExtractedRoom], upper_story: int
) -> list[tuple[LineString, float, float]]:
    """Top edges of every wall that lives on a story strictly *below*
    `upper_story`, regardless of how many stories down. Split-level
    buildings interleave stories vertically (story 1's floor can sit below
    story 0's ceiling), so the relevant lower wall under a story-N wall
    isn't always on story N-1."""
    out: list[tuple[LineString, float, float]] = []
    for room in rooms:
        if room.story >= upper_story:
            continue
        for wall in room.walls_computed:
            edge = _wall_top_edge_3d(wall)
            if edge is not None:
                out.append(edge)
    return out


def _interpolated_top_y_at(pt: Point, edge: tuple[LineString, float, float]) -> float:
    """Interpolate the top edge's Y at the XZ-projection of `pt` onto its line."""
    seg, y_a, y_b = edge
    if seg.length <= 0:
        return y_a
    t = float(seg.project(pt)) / float(seg.length)
    t = max(0.0, min(1.0, t))
    return y_a + t * (y_b - y_a)


def _highest_lower_top_y(
    pt: Point,
    lower_tops: list[tuple[LineString, float, float]],
    proximity_m: float,
) -> float | None:
    """Return the highest interpolated lower-wall top Y at this XZ point,
    among lower walls whose XZ top edge is within `proximity_m` of `pt`.

    Two principles combined:
    * **Highest, not closest** — an upper wall's descent must stop at the
      tallest opaque surface below; otherwise a tall partition next to a
      short one would let the strip overshoot past its top.
    * **Interpolate per-edge Y** — the lower wall's top can slope (gable,
      partial-height transitions). Using the edge's Y at the actual XZ
      projection lets the strip follow the slope instead of clamping to a
      single number."""
    best: float | None = None
    for edge in lower_tops:
        seg = edge[0]
        if seg.distance(pt) <= proximity_m:
            y = _interpolated_top_y_at(pt, edge)
            if best is None or y > best:
                best = y
    return best


def _story_footprints(rooms: list[ExtractedRoom]) -> dict[int, object]:
    polys_by_story: dict[int, list[Polygon]] = defaultdict(list)
    for room in rooms:
        poly = _xz_polygon(room.floor_polygon)
        if poly is not None:
            polys_by_story[room.story].append(poly)
    footprints: dict[int, object] = {}
    for story, polys in polys_by_story.items():
        union = make_valid(unary_union(polys))
        if union is not None and not union.is_empty:
            footprints[story] = union
    return footprints


def _room_at_xz(
    rooms: list[ExtractedRoom],
    polygons: list[Polygon | None],
    story: int,
    point: Point,
) -> ExtractedRoom | None:
    """Return the room on `story` whose floor polygon contains `point`, else None."""
    for room, polygon in zip(rooms, polygons, strict=False):
        if room.story != story or polygon is None:
            continue
        if polygon.contains(point) or polygon.touches(point):
            return room
    return None


def _flat_ceiling_supports(
    rooms: list[ExtractedRoom],
    polygons: list[Polygon | None],
    upper_story: int,
) -> list[tuple[Polygon, float]]:
    """Flat lower-story ceiling slabs that can support an upper-wall descent.

    Wall tops remain the preferred support signal because they preserve local
    scanned height changes. Flat ceilings cover the case where the lower wall
    edge is offset too far away to match, but the horizontal slab under the
    upper wall is still known.
    """
    from reconcile_tiers.extract.ceilings import _classify_should_be_flat

    classifications = _classify_should_be_flat(rooms)
    supports: list[tuple[Polygon, float]] = []
    for idx, room in enumerate(rooms):
        if room.story >= upper_story:
            continue
        should_flat, ceiling_y = classifications.get(idx, (False, None))
        polygon = polygons[idx] if idx < len(polygons) else None
        if not should_flat or ceiling_y is None or polygon is None:
            continue
        supports.append((polygon, float(ceiling_y)))
    return supports


def _highest_flat_ceiling_y(
    pt: Point,
    flat_supports: list[tuple[Polygon, float]],
) -> float | None:
    best: float | None = None
    for polygon, y in flat_supports:
        if polygon.contains(pt) or polygon.touches(pt):
            if best is None or y > best:
                best = y
    return best


def _highest_descent_support_y(
    pt: Point,
    lower_tops: list[tuple[LineString, float, float]],
    flat_supports: list[tuple[Polygon, float]],
    proximity_m: float,
) -> float | None:
    wall_y = _highest_lower_top_y(pt, lower_tops, proximity_m)
    flat_y = _highest_flat_ceiling_y(pt, flat_supports)
    if wall_y is None:
        return flat_y
    if flat_y is None:
        return wall_y
    return max(wall_y, flat_y)


def _descent_quad(
    bottom_endpoints: list[list[float]],
    lower_ceiling_y: float,
) -> list[list[float]]:
    a, b = bottom_endpoints
    return [
        [float(a[0]), float(a[1]), float(a[2])],
        [float(a[0]), lower_ceiling_y, float(a[2])],
        [float(b[0]), lower_ceiling_y, float(b[2])],
        [float(b[0]), float(b[1]), float(b[2])],
    ]


# When splitting a wall by per-room ownership, breakpoints from neighbouring
# rooms can be a few mm-cm apart due to scan noise on the same physical
# boundary. Only merge breakpoints that are within scan-noise distance —
# real boundaries between distinct rooms (typically separated by wall
# thickness, 10+ cm) stay separate, so each sub-segment stays inside its
# owner room and uses that room's actual ceiling Y.
SUBSEG_MERGE_TOL_M = 0.02


def _seg_t(seg: LineString, pt) -> float:
    if seg.length <= 0:
        return 0.0
    return float(seg.project(Point(pt[0], pt[1]))) / float(seg.length)


def _split_segment_by_room(
    seg: LineString,
    rooms: list[ExtractedRoom],
    polygons: list[Polygon | None],
    story: int,
) -> list[tuple[float, float, list[float], list[float], ExtractedRoom]]:
    """Split `seg` into sub-segments of constant room ownership. All transition
    points share a single parameterization along `seg`, so adjacent
    sub-segments end and begin at exactly the same XZ coordinates.

    Breakpoints from different room polygons that fall within
    SUBSEG_MERGE_TOL_M of each other are collapsed to their mean — this
    absorbs the small (~few cm) jitter that comes from two rooms'
    boundaries being independently scanned.

    Returns one tuple per non-empty sub-segment:
        (t_lo, t_hi, xz_lo, xz_hi, room)
    where xz_lo/xz_hi are computed from `seg.interpolate(t)`, and t_lo/t_hi
    are normalised parameter values in [0, 1]. Sub-segments shorter than
    MIN_SYNTHETIC_ARC_LENGTH_M are dropped.
    """
    rooms_polys = [
        (room, polygon)
        for room, polygon in zip(rooms, polygons, strict=False)
        if room.story == story and polygon is not None
    ]
    seg_len = float(seg.length)
    if not rooms_polys or seg_len <= 0:
        return []

    spans: list[tuple[float, float, ExtractedRoom]] = []
    raw_breakpoints: set[float] = {0.0, 1.0}
    for room, polygon in rooms_polys:
        try:
            inter = seg.intersection(polygon)
        except Exception:
            continue
        if inter.is_empty:
            continue
        parts = inter.geoms if hasattr(inter, "geoms") else [inter]
        for part in parts:
            if not isinstance(part, LineString) or part.length <= 0:
                continue
            coords = list(part.coords)
            t_a = _seg_t(seg, coords[0])
            t_b = _seg_t(seg, coords[-1])
            t_lo, t_hi = (t_a, t_b) if t_a < t_b else (t_b, t_a)
            spans.append((t_lo, t_hi, room))
            raw_breakpoints.add(t_lo)
            raw_breakpoints.add(t_hi)

    if not spans:
        return []

    merge_tol_t = SUBSEG_MERGE_TOL_M / seg_len
    sorted_bp = sorted(raw_breakpoints)
    cluster: list[float] = [sorted_bp[0]]
    merged_bp: list[float] = []
    for value in sorted_bp[1:]:
        if value - cluster[-1] <= merge_tol_t:
            cluster.append(value)
        else:
            merged_bp.append(sum(cluster) / len(cluster))
            cluster = [value]
    merged_bp.append(sum(cluster) / len(cluster))
    # Force the segment endpoints to remain at 0 and 1.
    merged_bp[0] = 0.0
    merged_bp[-1] = 1.0

    out: list[tuple[float, float, list[float], list[float], ExtractedRoom]] = []
    for idx in range(len(merged_bp) - 1):
        t_lo = merged_bp[idx]
        t_hi = merged_bp[idx + 1]
        if (t_hi - t_lo) * seg_len < MIN_SYNTHETIC_ARC_LENGTH_M:
            continue
        pt_lo = seg.interpolate(t_lo, normalized=True)
        pt_hi = seg.interpolate(t_hi, normalized=True)
        pt_mid = seg.interpolate((t_lo + t_hi) / 2.0, normalized=True)
        # Owner = polygon that contains the sub-segment midpoint in XZ.
        # Falls back to checking the endpoints if the midpoint sits exactly on a
        # boundary (rare). No tolerance / buffer.
        owner = None
        for room, polygon in rooms_polys:
            if polygon.contains(pt_mid) or polygon.touches(pt_mid):
                owner = room
                break
        if owner is None:
            for room, polygon in rooms_polys:
                if polygon.contains(pt_lo) or polygon.contains(pt_hi):
                    owner = room
                    break
        if owner is None:
            continue
        out.append(
            (
                t_lo,
                t_hi,
                [float(pt_lo.x), float(pt_lo.y)],
                [float(pt_hi.x), float(pt_hi.y)],
                owner,
            )
        )
    return out


def _wall_y_at(endpoints: list[list[float]], t: float) -> float:
    return float(endpoints[0][1]) + max(0.0, min(1.0, t)) * (
        float(endpoints[1][1]) - float(endpoints[0][1])
    )


SAMPLE_STEP_M = 0.05


def _descent_quads_from_lower_walls(
    seg: LineString,
    endpoints: list[list[float]],
    lower_tops: list[tuple[LineString, float, float]],
    flat_supports: list[tuple[Polygon, float]] | None = None,
) -> list[list[list[float]]]:
    """Walk the upper wall's bottom segment in SAMPLE_STEP_M steps. At each
    sample, the strip's bottom Y is the highest support below the wall:
    nearby lower-wall top first, then a flat lower-story ceiling slab under
    the point. The strip is emitted as one narrow quad per sample interval —
    adjacent quads share their transition coordinates exactly, and the bottom
    edge follows the support profile point-by-point (no constant-Y clamping).

    Sample intervals where either endpoint has no support, or where the
    resulting gap falls outside [TOP_EPSILON_M, MAX_GAP_TO_SLAB_M] on both
    sides, are skipped (so the strip can break and resume around
    obstructions)."""
    seg_len = float(seg.length)
    if seg_len <= 0 or (not lower_tops and not flat_supports):
        return []
    flat_supports = flat_supports or []
    n_intervals = max(1, round(seg_len / SAMPLE_STEP_M))
    samples: list[tuple[float, Point, float | None, float]] = []
    for i in range(n_intervals + 1):
        t = i / n_intervals
        pt = seg.interpolate(t, normalized=True)
        wall_y = _wall_y_at(endpoints, t)
        lower_y = _highest_descent_support_y(
            pt,
            lower_tops,
            flat_supports,
            LOWER_WALL_PROXIMITY_M,
        )
        samples.append((t, pt, lower_y, wall_y))

    quads: list[list[list[float]]] = []
    for i in range(n_intervals):
        _t_lo, pt_lo, lower_lo, wall_y_lo = samples[i]
        _t_hi, pt_hi, lower_hi, wall_y_hi = samples[i + 1]
        if lower_lo is None or lower_hi is None:
            continue
        gap_lo = wall_y_lo - lower_lo
        gap_hi = wall_y_hi - lower_hi
        # Both ends must have a real gap. If the lower top is at or above
        # the upper wall's bottom on either side, there's no inter-story
        # void to fill (or the matched lower wall isn't actually below) —
        # skip so we don't emit a self-intersecting / negative-height quad.
        if min(gap_lo, gap_hi) <= TOP_EPSILON_M:
            continue
        if max(gap_lo, gap_hi) > MAX_GAP_TO_SLAB_M:
            continue
        quads.append(
            [
                [float(pt_lo.x), wall_y_lo, float(pt_lo.y)],
                [float(pt_lo.x), lower_lo, float(pt_lo.y)],
                [float(pt_hi.x), lower_hi, float(pt_hi.y)],
                [float(pt_hi.x), wall_y_hi, float(pt_hi.y)],
            ]
        )
    return quads


def _uplift_quads_per_upper_room(
    seg: LineString,
    endpoints: list[list[float]],
    rooms: list[ExtractedRoom],
    polygons: list[Polygon | None],
    upper_story: int,
) -> list[list[list[float]]]:
    """Walk the wall's top segment, emit one uplift quad per sub-segment of
    constant upper-room ownership, using that room's floor Y as the strip's
    top. Sub-segments outside any upper room are skipped. Adjacent quads
    share exact transition coordinates."""
    quads: list[list[list[float]]] = []
    for t_lo, t_hi, xz_lo, xz_hi, upper_room in _split_segment_by_room(
        seg, rooms, polygons, upper_story
    ):
        floor_y = _room_floor_y(upper_room)
        if floor_y is None:
            continue
        wall_y_a = _wall_y_at(endpoints, t_lo)
        wall_y_b = _wall_y_at(endpoints, t_hi)
        gap = floor_y - max(wall_y_a, wall_y_b)
        max_gap = floor_y - min(wall_y_a, wall_y_b)
        if gap <= TOP_EPSILON_M or max_gap > MAX_GAP_TO_SLAB_M:
            continue
        quads.append(
            [
                [xz_lo[0], wall_y_a, xz_lo[1]],
                [xz_hi[0], wall_y_b, xz_hi[1]],
                [xz_hi[0], floor_y, xz_hi[1]],
                [xz_lo[0], floor_y, xz_lo[1]],
            ]
        )
    return quads


def _uplift_quad(
    top_endpoints: list[list[float]],
    upper_floor_y: float,
) -> list[list[float]]:
    a, b = top_endpoints
    return [
        [float(a[0]), float(a[1]), float(a[2])],
        [float(b[0]), float(b[1]), float(b[2])],
        [float(b[0]), upper_floor_y, float(b[2])],
        [float(a[0]), upper_floor_y, float(a[2])],
    ]


def compute_descent_strips(
    rooms: list[ExtractedRoom],
    *,
    wings: list | None = None,
) -> list[ExtractedRoom]:
    """Attach `descent_strip` to perimeter upper-story walls and synthesise
    where missing.

    The strip's bottom Y at any XZ point comes from the highest support below
    it. Nearby lower-story wall tops are preferred so mixed wall heights stay
    local; flat lower-story ceiling slabs fill the case where the wall edge is
    offset but the horizontal slab is known.

    When `wings` is provided, support search is restricted to lower-story
    rooms in the same wing as the upper-story wall's owning room. This
    stops a descent strip on one wing from clipping to a ceiling under a
    different wing. Default `wings=None` preserves the legacy behaviour.
    """
    footprints = _story_footprints(rooms)
    polygons = [_xz_polygon(room.floor_polygon) for room in rooms]
    rooms_by_index: dict[int, ExtractedRoom] = {
        idx: room for idx, room in enumerate(rooms)
    }
    new_walls_by_room: dict[int, list[ExtractedWall]] = {
        idx: [] for idx in rooms_by_index
    }
    new_descent_by_wall: dict[tuple[int, int], list[list[float]]] = {}

    room_to_wing_idx = _build_wing_membership(rooms, wings)
    support_cache: dict[tuple[int, int | None], tuple] = {}

    def _supports_for(upper_story: int, wing_idx: int | None):
        key = (upper_story, wing_idx)
        cached = support_cache.get(key)
        if cached is not None:
            return cached
        if wing_idx is None or not wings:
            wing_rooms = rooms
            wing_polys = polygons
        else:
            wing_rooms = []
            wing_polys = []
            for i in range(len(rooms)):
                if room_to_wing_idx.get(i) == wing_idx:
                    wing_rooms.append(rooms[i])
                    wing_polys.append(polygons[i] if i < len(polygons) else None)
        result = (
            _all_wall_tops_below(wing_rooms, upper_story),
            _flat_ceiling_supports(wing_rooms, wing_polys, upper_story),
        )
        support_cache[key] = result
        return result

    for upper_story in sorted(footprints):
        lower_tops_full = _all_wall_tops_below(rooms, upper_story)
        flat_supports_full = _flat_ceiling_supports(rooms, polygons, upper_story)
        if not lower_tops_full and not flat_supports_full:
            continue
        upper_fp = footprints[upper_story]
        boundary = upper_fp.boundary
        perimeter_segments: list[LineString] = []

        room_indices_in_story = [
            idx for idx, room in rooms_by_index.items() if room.story == upper_story
        ]

        for room_idx in room_indices_in_story:
            room = rooms_by_index[room_idx]
            wing_idx = room_to_wing_idx.get(room_idx)
            lower_tops, flat_supports = _supports_for(upper_story, wing_idx)
            for wall_idx, wall in enumerate(room.walls_computed):
                seg_pair = _wall_bottom_segment(wall.corners)
                if seg_pair is None:
                    continue
                seg, endpoints = seg_pair
                if _segment_median_distance(seg, boundary) > PERIMETER_DISTANCE_TOL_M:
                    continue
                quads = _descent_quads_from_lower_walls(
                    seg,
                    endpoints,
                    lower_tops,
                    flat_supports,
                )
                if not quads:
                    continue
                new_descent_by_wall[(room_idx, wall_idx)] = quads
                perimeter_segments.append(seg)

        rooms_with_polys = [
            (idx, _xz_polygon(rooms_by_index[idx].floor_polygon))
            for idx in room_indices_in_story
        ]
        rooms_with_polys = [
            (idx, poly) for idx, poly in rooms_with_polys if poly is not None
        ]

        for arc_idx, arc in enumerate(
            _missing_perimeter_arcs(boundary, perimeter_segments)
        ):
            owner_idx = _arc_room_owner(arc, rooms_with_polys, wings=wings, rooms=rooms)
            if owner_idx is None:
                continue
            owner_floor_y = _room_floor_y(rooms_by_index[owner_idx])
            if owner_floor_y is None:
                continue
            owner_wing_idx = room_to_wing_idx.get(owner_idx)
            arc_lower_tops, arc_flat_supports = _supports_for(
                upper_story, owner_wing_idx
            )
            coords = list(arc.coords)
            for seg_idx in range(len(coords) - 1):
                a = coords[seg_idx]
                b = coords[seg_idx + 1]
                arc_seg = LineString([a, b])
                if arc_seg.length < MIN_SYNTHETIC_ARC_LENGTH_M:
                    continue
                arc_endpoints = [
                    [float(a[0]), owner_floor_y, float(a[1])],
                    [float(b[0]), owner_floor_y, float(b[1])],
                ]
                quads = _descent_quads_from_lower_walls(
                    arc_seg,
                    arc_endpoints,
                    arc_lower_tops,
                    arc_flat_supports,
                )
                for sub_idx, quad in enumerate(quads):
                    synth = ExtractedWall(
                        id=_stable_synth_id(
                            owner_idx, arc, seg_idx + arc_idx * 1000 + sub_idx * 100000
                        ),
                        corners=quad,
                        source="synthesised",
                        descent_strip=None,
                        uplift_strip=None,
                        synthetic=True,
                    )
                    new_walls_by_room[owner_idx].append(synth)

    out: list[ExtractedRoom] = []
    for idx, room in rooms_by_index.items():
        updated_walls: list[ExtractedWall] = []
        for wall_idx, wall in enumerate(room.walls_computed):
            descent = new_descent_by_wall.get((idx, wall_idx))
            updated_walls.append(replace(wall, descent_strip=descent))
        out.append(
            replace(
                room,
                walls_computed=updated_walls,
                synthetic_walls=[*room.synthetic_walls, *new_walls_by_room[idx]],
            )
        )
    return out


def compute_uplift_strips(
    rooms: list[ExtractedRoom],
    *,
    wings: list | None = None,
) -> list[ExtractedRoom]:
    """Attach `uplift_strip` to lower-story interior walls below the upper slab.

    The strip's top Y is the floor Y of the *specific* upper-story room directly
    above the wall (not a per-story median), so per-room floor differences don't
    leave a gap.

    When `wings` is provided, the upper-room candidates considered for the
    strip's top Y are limited to rooms in the same wing as the lower-story
    wall's owning room. Default `wings=None` preserves legacy behaviour.
    """
    footprints = _story_footprints(rooms)
    polygons = [_xz_polygon(room.floor_polygon) for room in rooms]
    room_to_wing_idx = _build_wing_membership(rooms, wings)

    out: list[ExtractedRoom] = []
    for room_idx, room in enumerate(rooms):
        upper_story = room.story + 1
        upper_fp = footprints.get(upper_story)
        lower_fp = footprints.get(room.story)
        if upper_fp is None or lower_fp is None:
            out.append(room)
            continue
        lower_boundary = lower_fp.boundary
        wing_idx = room_to_wing_idx.get(room_idx)
        if wings and wing_idx is not None:
            wing_rooms = []
            wing_polys = []
            for i in range(len(rooms)):
                if room_to_wing_idx.get(i) == wing_idx:
                    wing_rooms.append(rooms[i])
                    wing_polys.append(polygons[i] if i < len(polygons) else None)
            search_rooms = wing_rooms
            search_polys = wing_polys
        else:
            search_rooms = rooms
            search_polys = polygons

        updated_walls: list[ExtractedWall] = []
        for wall in room.walls_computed:
            seg_pair = _wall_top_segment(wall.corners)
            if seg_pair is None:
                updated_walls.append(replace(wall, uplift_strip=None))
                continue
            seg, endpoints = seg_pair
            if (
                _segment_median_distance(seg, lower_boundary)
                <= PERIMETER_DISTANCE_TOL_M
            ):
                updated_walls.append(replace(wall, uplift_strip=None))
                continue
            quads = _uplift_quads_per_upper_room(
                seg, endpoints, search_rooms, search_polys, upper_story
            )
            updated_walls.append(replace(wall, uplift_strip=quads or None))
        out.append(replace(room, walls_computed=updated_walls))
    return out


# Re-exports — synthetic perimeter-arc helpers live in `_extension_arcs`.
# Both `compute_descent_strips` and `compute_uplift_strips` (above) call
# them.
from reconcile_tiers.extract._extension_arcs import (  # noqa: E402
    _arc_room_owner,
    _build_wing_membership,
    _missing_perimeter_arcs,
    _stable_synth_id,
)
