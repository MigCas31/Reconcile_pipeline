"""Wall-endpoint stitching helpers."""

import math
from collections import defaultdict

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from reconcile_v2.decision_logic import classify_stitch_decision

from .lineage import STEP_STITCH_WALLS, record

_ = Point  # keep formatter from stripping the used-at-runtime import

# --- notch-aware snap (shared by extract_3d.py and extract3d/stitch.py) ------
_SNAP_PARALLEL_DOT_MIN = 0.98
_SNAP_PERP_MIN_M = 0.4
_SNAP_PERP_MAX_M = 2.0
_SNAP_ALONG_OVERLAP_MIN_M = 0.2
_SNAP_PERP_UNIFORM_TOL_M = 0.08  # all corners of an entry must share perp within this
_SNAP_CORNER_XZ_MATCH_M = 0.005  # caps follow via (x,z)-match to a snapped wall
_SNAP_NOTCH_SAMPLES = 21
_SNAP_NOTCH_COVERED_FRAC = 0.60  # covered-outside fraction required
_SNAP_NOTCH_INSIDE_MAX = 0.20  # inside-stitch fraction must stay below this
_SNAP_NOTCH_EXT_MARGIN_M = (
    0.2  # along offset past stitch ends before sampling "outside"
)
_SNAP_NOTCH_EXT_SPAN_M = 1.0  # sampling window size on each side past stitch ends


def _snap_safe_polygon(xz_points):
    if len(xz_points) < 3:
        return None
    poly = Polygon(xz_points)
    if not poly.is_valid:
        fixed = make_valid(poly)
        if hasattr(fixed, "geoms"):
            polys = [g for g in fixed.geoms if isinstance(g, Polygon)]
            poly = max(polys, key=lambda p: p.area) if polys else poly
        elif isinstance(fixed, Polygon):
            poly = fixed
    return poly if poly.is_valid and not poly.is_empty else None


def _snap_wall_frame(corners):
    """Return (origin_xz, dir_xz, length) from a wall's first edge, or None."""
    if not corners or len(corners) < 2:
        return None
    a = (float(corners[0][0]), float(corners[0][2]))
    b = (float(corners[1][0]), float(corners[1][2]))
    dx, dz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dz)
    if length < 0.1:
        return None
    return a, (dx / length, dz / length), length


def _snap_project(origin, direction, point):
    """Return (along, perp) of ``point`` in the wall's local frame."""
    dx, dz = point[0] - origin[0], point[1] - origin[1]
    along = dx * direction[0] + dz * direction[1]
    perp = dx * (-direction[1]) + dz * direction[0]
    return along, perp


def _snap_world(origin, direction, along, perp):
    normal = (-direction[1], direction[0])
    return (
        origin[0] + along * direction[0] + perp * normal[0],
        origin[1] + along * direction[1] + perp * normal[1],
    )


def _snap_notch_ok(
    wall_origin, wall_dir, wall_len, stitch_along, perp_offset, room_poly
):
    """Reproduce the Phase-A notch predicate against a shapely polygon."""
    if room_poly is None:
        return False
    a_lo, a_hi = stitch_along
    strip_mid_perp = perp_offset / 2.0

    inside_hits = 0
    for i in range(_SNAP_NOTCH_SAMPLES):
        t = i / (_SNAP_NOTCH_SAMPLES - 1)
        a = a_lo + t * (a_hi - a_lo)
        p = _snap_world(wall_origin, wall_dir, a, strip_mid_perp)
        if room_poly.covers(Point(p[0], p[1])):
            inside_hits += 1
    inside_frac = inside_hits / _SNAP_NOTCH_SAMPLES

    windows = [
        (
            max(0.0, a_lo - _SNAP_NOTCH_EXT_MARGIN_M - _SNAP_NOTCH_EXT_SPAN_M),
            max(0.0, a_lo - _SNAP_NOTCH_EXT_MARGIN_M),
        ),
        (
            min(wall_len, a_hi + _SNAP_NOTCH_EXT_MARGIN_M),
            min(wall_len, a_hi + _SNAP_NOTCH_EXT_MARGIN_M + _SNAP_NOTCH_EXT_SPAN_M),
        ),
    ]
    outside_hits = 0
    outside_total = 0
    for w_lo, w_hi in windows:
        if w_hi - w_lo < 0.1:
            continue
        for i in range(_SNAP_NOTCH_SAMPLES):
            t = i / (_SNAP_NOTCH_SAMPLES - 1)
            a = w_lo + t * (w_hi - w_lo)
            p = _snap_world(wall_origin, wall_dir, a, strip_mid_perp)
            outside_total += 1
            if room_poly.covers(Point(p[0], p[1])):
                outside_hits += 1
    outside_frac = outside_hits / outside_total if outside_total else 0.0

    return (
        outside_frac >= _SNAP_NOTCH_COVERED_FRAC
        and inside_frac <= _SNAP_NOTCH_INSIDE_MAX
    )


_DEDUP_XZ_TOL_M = 0.15  # xz-bbox bucket size for matching vertical stitch quads
_DEDUP_Y_TOL_M = 0.15  # y-extent bucket size for matching vertical stitch quads
_DEDUP_MIN_SHARED_CORNERS = 2

# A stitch whose corners collapse to fewer than 3 unique 3D points can only
# render as a line/point; the viewer's triangulator either skips it (no mesh)
# or, worse, Earcut on a quad like [A,A,B,B] emits two overlapping zero-area
# triangles that read as a bow-tie X when back-face rendering kicks in.
_DEGENERATE_VERTEX_TOL_M = 0.001  # 1 mm -- two corners closer than this are "same"

# Minimum XZ length of an emitted stitch leg (either branch). Matches
# ``min_gap`` in ``stitch_wall_gaps``: we never pair endpoints closer than
# 6 cm, so a leg shorter than 6 cm has no physical basis and renders as a
# sliver / bow-tie even though its 4 vertices pass the degeneracy test.
_MIN_LEG_LENGTH_M = 0.06

# Endpoint pairs closer than this collapse to a single direct quad
# spanning P1 -> P2 instead of an L-shape with cap triangles. At this scale
# the distinction between "orthogonal L + caps" and "one slightly-oblique
# quad" is indistinguishable in the viewer, and the direct form avoids
# orphaned cap triangles when the L's legs fall below _MIN_LEG_LENGTH_M.
_DIRECT_QUAD_MAX_GAP_M = 0.15

_CROSS_ENDPOINT_TOL_M = 0.03
_CROSS_Y_OVERLAP_TOL_M = 0.05


def _segments_share_endpoint_xz(a0, a1, b0, b1, tol=_CROSS_ENDPOINT_TOL_M):
    for p in (a0, a1):
        for q in (b0, b1):
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= tol:
                return True
    return False


def _orientation_2d(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment_2d(a, b, c, eps=1e-9):
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
    )


def _segments_intersect_xz(a0, a1, b0, b1, eps=1e-9):
    o1 = _orientation_2d(a0, a1, b0)
    o2 = _orientation_2d(a0, a1, b1)
    o3 = _orientation_2d(b0, b1, a0)
    o4 = _orientation_2d(b0, b1, a1)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and _point_on_segment_2d(a0, b0, a1, eps):
        return True
    if abs(o2) <= eps and _point_on_segment_2d(a0, b1, a1, eps):
        return True
    if abs(o3) <= eps and _point_on_segment_2d(b0, a0, b1, eps):
        return True
    if abs(o4) <= eps and _point_on_segment_2d(b0, a1, b1, eps):
        return True
    return False


def _candidate_segments_cross_existing(candidate_segments, accepted_segments):
    for a0, a1, ay0, ay1 in candidate_segments:
        for b0, b1, by0, by1 in accepted_segments:
            if min(ay1, by1) - max(ay0, by0) <= _CROSS_Y_OVERLAP_TOL_M:
                continue
            if _segments_share_endpoint_xz(a0, a1, b0, b1):
                continue
            if _segments_intersect_xz(a0, a1, b0, b1):
                return True
    return False


def dedup_duplicate_vertical_stitches(stitch_walls):
    """Drop vertical ``type='stitch'`` quads that duplicate an earlier one.

    Two different wall-endpoint pairs can independently emit a stitch on the
    same physical wall plane -- e.g. when a single wall separates room 1 from
    both room 0 and room 5, the pair (wall_a_end0 ↔ wall_AEAB_end0) and the
    pair (wall_b_end0 ↔ wall_AEAB_end1) each extrude an outward L-leg along
    AEAB. ``used_pairs`` does not catch this because the input endpoint
    tuples differ. Without dedup, the viewer overlays two near-identical
    translucent quads and their outline edges cross visually as a bow-tie.

    Only ``type='stitch'`` entries are compared. Cap triangles
    (``stitch_floor`` / ``stitch_ceiling``) fan into genuinely different
    wedges per pair and are left untouched.
    """

    def plane_normal(corners):
        nx = ny = nz = 0.0
        n = len(corners)
        for k in range(n):
            a, b = corners[k], corners[(k + 1) % n]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        L = math.sqrt(nx * nx + ny * ny + nz * nz)
        if L < 1e-6:
            return None
        return (nx / L, ny / L, nz / L)

    def parallel(
        n1, n2, tol=0.02
    ):  # ~8 degrees -- well under 90 deg between adjacent walls
        if n1 is None or n2 is None:
            return False
        d = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
        return abs(abs(d) - 1.0) < tol

    kept = []
    accepted = []  # list of (corners, plane_normal)
    for entry in stitch_walls:
        if entry.get("type") != "stitch":
            kept.append(entry)
            continue
        corners = entry.get("corners") or []
        if len(corners) < 3:
            kept.append(entry)
            continue
        n_self = plane_normal(corners)
        is_dup = False
        for prev_corners, prev_normal in accepted:
            # Two walls that share 2 corners but lie on different planes (adjacent
            # walls meeting at an edge) are NOT duplicates; skip them early.
            if not parallel(n_self, prev_normal):
                continue
            shared = 0
            for ca in corners:
                for cb in prev_corners:
                    if (
                        math.hypot(ca[0] - cb[0], ca[2] - cb[2]) < _DEDUP_XZ_TOL_M
                        and abs(ca[1] - cb[1]) < _DEDUP_Y_TOL_M
                    ):
                        shared += 1
                        break
                if shared >= _DEDUP_MIN_SHARED_CORNERS:
                    break
            if shared >= _DEDUP_MIN_SHARED_CORNERS:
                is_dup = True
                break
        if is_dup:
            record(entry, STEP_STITCH_WALLS, "dropped_duplicate")
            continue
        accepted.append((corners, n_self))
        kept.append(entry)
    return kept


def prune_crossing_vertical_stitches(stitch_walls):
    """Drop vertical stitch quads whose base segment crosses an earlier stitch."""
    kept = []
    accepted_segments = []
    for entry in stitch_walls:
        if entry.get("type") != "stitch":
            kept.append(entry)
            continue
        corners = entry.get("corners") or []
        if len(corners) < 4:
            kept.append(entry)
            continue
        candidate = [
            (
                (float(corners[0][0]), float(corners[0][2])),
                (float(corners[1][0]), float(corners[1][2])),
                float(min(corners[0][1], corners[2][1])),
                float(max(corners[0][1], corners[2][1])),
            )
        ]
        if _candidate_segments_cross_existing(candidate, accepted_segments):
            continue
        accepted_segments.extend(candidate)
        kept.append(entry)
    return kept


def snap_stitches_to_non_owner_walls(rooms_out, stitch_walls):
    """Post-process ``stitch_walls`` to move scan-boundary gaps onto the adjacent wall.

    When a synthetic stitch runs parallel to a non-owner room's wall with a ~1 m
    perpendicular offset, AND that non-owner room's floor polygon reaches the
    wall just past the stitch's ends while notching inward inside the stitch
    zone, the gap is almost always a scan-registration artifact rather than a
    real void. The snap translates the stitch's (x, z) onto the wall's plane,
    preserving its y range. See ``scripts/simulate_stitch_snap.py`` for the
    cohort simulation that validated the predicate (248 repaired / 0 regressed
    across 62 buildings).

    Mutates ``stitch_walls`` in place. Entries that are snapped gain:
      - ``snapped_to_wall``: the non-owner wall id.
      - ``id`` (if absent): ``"snap:<wall_id>"`` so viewer locators stop falling
        back to the synthetic ``sw:<index>``.

    Caps (``stitch_floor`` / ``stitch_ceiling``) follow their wall by matching
    corner (x, z) positions to the snapped wall's original endpoints.
    """
    if not stitch_walls:
        return stitch_walls

    room_polys = {}
    for ri, room in enumerate(rooms_out):
        fp = room.get("floor_polygon") or []
        room_polys[ri] = _snap_safe_polygon([(float(p[0]), float(p[2])) for p in fp])

    walls_by_story = defaultdict(list)
    for ri, room in enumerate(rooms_out):
        story = int(room.get("story", 0))
        for wall in room.get("walls_computed", []):
            frame = _snap_wall_frame(wall.get("corners", []))
            if frame is None:
                continue
            walls_by_story[story].append((ri, wall, frame))

    # Pass 1: identify stitch walls to snap, collect (x,z) displacements.
    displacements = []  # list of (orig_xz, new_xz)
    for entry in stitch_walls:
        if entry.get("type") != "stitch":
            continue
        corners = entry.get("corners") or []
        if len(corners) < 4:
            continue
        frame = _snap_wall_frame(corners)
        if frame is None:
            continue
        _s_origin, s_dir, _s_len = frame
        story = int(entry.get("story", 0))
        owner_rooms = set()
        owners = entry.get("room_indices")
        if owners:
            owner_rooms.update(r for r in owners if r is not None)
        if entry.get("room_index") is not None:
            owner_rooms.add(entry["room_index"])

        best = None  # (wall_id, new_xz_by_corner_idx)
        for ri, wall, (w_origin, w_dir, w_len) in walls_by_story.get(story, ()):
            if ri in owner_rooms:
                continue
            dot = s_dir[0] * w_dir[0] + s_dir[1] * w_dir[1]
            if abs(dot) < _SNAP_PARALLEL_DOT_MIN:
                continue

            perps = []
            alongs = []
            for c in corners:
                along, perp = _snap_project(w_origin, w_dir, (float(c[0]), float(c[2])))
                alongs.append(along)
                perps.append(perp)
            perp_mean = sum(perps) / len(perps)
            if max(abs(p - perp_mean) for p in perps) > _SNAP_PERP_UNIFORM_TOL_M:
                continue
            if not (_SNAP_PERP_MIN_M <= abs(perp_mean) <= _SNAP_PERP_MAX_M):
                continue

            a_lo, a_hi = min(alongs[:2]), max(alongs[:2])
            overlap = max(0.0, min(a_hi, w_len) - max(a_lo, 0.0))
            if overlap < _SNAP_ALONG_OVERLAP_MIN_M:
                continue

            notch = False
            for owner_ri in owner_rooms:
                if _snap_notch_ok(
                    w_origin,
                    w_dir,
                    w_len,
                    (a_lo, a_hi),
                    perp_mean,
                    room_polys.get(owner_ri),
                ):
                    notch = True
                    break
            if not notch:
                continue

            new_corners_xz = []
            for along, _perp in zip(alongs, perps, strict=False):
                new_xz = _snap_world(w_origin, w_dir, along, 0.0)
                new_corners_xz.append(new_xz)
            best = (wall.get("id"), new_corners_xz)
            break

        if best is None:
            continue

        wall_id, new_corners_xz = best
        for idx, c in enumerate(corners):
            orig_xz = (float(c[0]), float(c[2]))
            new_xz = new_corners_xz[idx]
            displacements.append((orig_xz, new_xz))
            c[0] = new_xz[0]
            c[2] = new_xz[1]
        if wall_id:
            entry["snapped_to_wall"] = wall_id
            entry.setdefault("id", f"snap:{wall_id}")
        record(entry, STEP_STITCH_WALLS, "snapped_to_non_owner_wall")

    # Pass 2: drag any stitch entry corners (caps AND L-shape partner walls) that
    # share (x,z) with a snapped wall's original endpoints. Unsnapped partner walls
    # get their shared corner C dragged so the L-shape stays connected; caps
    # triangulating (P1, C, P2) likewise track the projected positions.
    if displacements:
        for entry in stitch_walls:
            if entry.get("snapped_to_wall"):
                continue
            if entry.get("type") not in ("stitch", "stitch_floor", "stitch_ceiling"):
                continue
            for c in entry.get("corners") or []:
                cx, cz = float(c[0]), float(c[2])
                for orig_xz, new_xz in displacements:
                    if (
                        abs(cx - orig_xz[0]) <= _SNAP_CORNER_XZ_MATCH_M
                        and abs(cz - orig_xz[1]) <= _SNAP_CORNER_XZ_MATCH_M
                    ):
                        c[0] = new_xz[0]
                        c[2] = new_xz[1]
                        break

    return stitch_walls


def recommend_stitch_actions(graph):
    """Return graph-driven wall stitching candidates.

    Legacy stitching still performs the geometry synthesis. This helper exposes
    a deterministic migration target based on `CONNECTS_TO` and wall role.
    """
    actions = []
    for edge in graph.edges:
        if edge.type != "CONNECTS_TO":
            continue
        left = graph.get_node(edge.from_id)
        right = graph.get_node(edge.to_id)
        if left is None or right is None:
            continue
        if left.type != "Surface" or right.type != "Surface":
            continue
        actions.append(
            classify_stitch_decision(graph, left, right, edge.evidence or {})
        )
    return actions


def stitch_wall_gaps(rooms_out):
    max_gap = 1.50
    min_gap = 0.06
    min_wall_height = 0.5
    height_ratio = 0.65
    mutual_thresh = 0.60

    story_rooms = defaultdict(list)
    for room_idx, room in enumerate(rooms_out):
        story_rooms[room["story"]].append((room_idx, room))

    stitch_walls = []
    for story, rooms in story_rooms.items():
        accepted_segments = []
        heights = []
        for _room_idx, room in rooms:
            for wall in room["walls_computed"]:
                corners = wall["corners"]
                if len(corners) < 4:
                    continue
                wall_h = abs(corners[2][1] - corners[0][1])
                if wall_h >= min_wall_height:
                    heights.append(wall_h)
        if not heights:
            continue

        median_h = float(np.median(heights))
        endpoints = []
        for room_idx, room in rooms:
            for wall_idx, wall in enumerate(room["walls_computed"]):
                corners = wall["corners"]
                if len(corners) < 4:
                    continue
                wall_h = abs(corners[2][1] - corners[0][1])
                if wall_h < min_wall_height or wall_h < median_h * height_ratio:
                    continue
                # Robust to either corner-y ordering: upstream pipelines
                # have emitted walls with c[0] as the bottom or the top in
                # different branches. Stitch pairing only cares about the
                # wall's y-range, so derive bottom/top from min/max of all
                # four corner y-values rather than trusting index order.
                ys = [c[1] for c in corners]
                y_low = min(ys)
                y_high = max(ys)
                extension = wall.get("extension_strip")
                ext_top = (
                    max(max(point[1] for point in strip) for strip in extension)
                    if extension
                    else None
                )
                top_y = ext_top if ext_top is not None else y_high
                key = (room_idx, wall_idx)
                endpoints.append((corners[0][0], corners[0][2], y_low, top_y, key, 0))
                endpoints.append((corners[1][0], corners[1][2], y_low, top_y, key, 1))

        used_pairs = set()
        for idx in range(len(endpoints)):
            x1, z1, yb1, yt1, wk1, end1 = endpoints[idx]
            best_dist = max_gap
            best_idx = -1
            for jdx in range(len(endpoints)):
                if idx == jdx:
                    continue
                x2, z2, _yb2, _yt2, wk2, _end2 = endpoints[jdx]
                if wk2 == wk1:
                    continue
                dist = math.hypot(x1 - x2, z1 - z2)
                if min_gap <= dist < best_dist:
                    best_dist = dist
                    best_idx = jdx

            if best_idx < 0:
                continue
            x2, z2, yb2, yt2, wk2, end2 = endpoints[best_idx]

            if best_dist <= mutual_thresh:
                reverse_best_dist = max_gap
                reverse_best = -1
                for kdx in range(len(endpoints)):
                    if kdx == best_idx:
                        continue
                    xk, zk, _ykb, _ykt, wkk, _ek = endpoints[kdx]
                    if wkk == wk2:
                        continue
                    d_val = math.hypot(x2 - xk, z2 - zk)
                    if min_gap <= d_val < reverse_best_dist:
                        reverse_best_dist = d_val
                        reverse_best = kdx
                if reverse_best != idx:
                    continue

            pair_key = tuple(sorted([(*wk1, end1), (*wk2, end2)]))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            y_bot = max(yb1, yb2)
            y_top = min(yt1, yt2)
            if y_top - y_bot < min_wall_height:
                continue

            room_idx1, wall_idx1 = wk1
            room_idx2, wall_idx2 = wk2

            # Short-gap fast path: for sub-_DIRECT_QUAD_MAX_GAP_M endpoint
            # pairs, emit one direct quad P1->P2 instead of an L-closure plus
            # cap triangles. Prevents orphan caps whenever both L-legs would
            # fall below _MIN_LEG_LENGTH_M, and matches what the user sees as
            # a "closed rectangular piece" -- at sub-decimeter gaps the
            # orthogonal/oblique distinction is below scan noise anyway.
            if best_dist < _DIRECT_QUAD_MAX_GAP_M:
                direct_entry = {
                    "corners": [
                        [x1, y_bot, z1],
                        [x2, y_bot, z2],
                        [x2, y_top, z2],
                        [x1, y_top, z1],
                    ],
                    "story": story,
                    "type": "stitch",
                    "room_index": room_idx1,
                    "room_indices": [room_idx1, room_idx2],
                    "id": f"stitch:{story}:{len(stitch_walls)}",
                }
                record(direct_entry, STEP_STITCH_WALLS, "created_direct_quad")
                stitch_walls.append(direct_entry)
                accepted_segments.append(((x1, z1), (x2, z2), y_bot, y_top))
                continue

            wall1_corners = rooms_out[room_idx1]["walls_computed"][wall_idx1]["corners"]
            wall2_corners = rooms_out[room_idx2]["walls_computed"][wall_idx2]["corners"]

            # Direction along each parent wall (from endpoint 0 to endpoint 1)
            def _wall_dir_xz(corners):
                dx = corners[1][0] - corners[0][0]
                dz = corners[1][2] - corners[0][2]
                length = math.hypot(dx, dz)
                if length < 1e-6:
                    return (0.0, 0.0)
                return (dx / length, dz / length)

            dir1 = _wall_dir_xz(wall1_corners)
            dir2 = _wall_dir_xz(wall2_corners)

            # Outward direction along each parent wall -- the axis past the
            # matched endpoint, pointing AWAY from the wall's interior. This
            # is the only valid half-space for the missing corner: projecting
            # onto the inward direction would place the L inside a room and
            # get dropped by ``_drop_stitches_crossing_rooms``.
            out_sign1 = -1.0 if end1 == 0 else 1.0
            out_sign2 = -1.0 if end2 == 0 else 1.0
            u1_out = (out_sign1 * dir1[0], out_sign1 * dir1[1])
            u2_out = (out_sign2 * dir2[0], out_sign2 * dir2[1])

            # Build an orthogonal L by projecting one endpoint onto the other
            # wall's outward axis. Two candidates: anchor on wall 1 or wall 2.
            # For each candidate the corner must lie in BOTH walls' outward
            # half-planes (otherwise the L is guaranteed to cut through a
            # room). Pick whichever valid L has the shorter total leg length
            # and doesn't cross an existing segment. When parent walls are
            # already near-perpendicular one leg collapses below
            # ``_MIN_LEG_LENGTH_M`` and we emit a single perpendicular stitch.
            # Oblique (non-right-angle) closures are never emitted -- real
            # buildings don't have oblique walls.
            def _build_l_candidate(p_anchor, p_other, dir_anchor_out, dir_other_out):
                ax, az = p_anchor
                ox, oz = p_other
                along = (ox - ax) * dir_anchor_out[0] + (oz - az) * dir_anchor_out[1]
                if along < 0:
                    return None
                cx_c = ax + along * dir_anchor_out[0]
                cz_c = az + along * dir_anchor_out[1]
                # Corner must lie in the other wall's outward half-plane too.
                if (cx_c - ox) * dir_other_out[0] + (cz_c - oz) * dir_other_out[1] < 0:
                    return None
                perp_len = math.hypot(
                    (ox - ax) - along * dir_anchor_out[0],
                    (oz - az) - along * dir_anchor_out[1],
                )
                legs = []
                if along >= _MIN_LEG_LENGTH_M:
                    legs.append(((ax, az), (cx_c, cz_c)))
                if perp_len >= _MIN_LEG_LENGTH_M:
                    legs.append(((cx_c, cz_c), (ox, oz)))
                return {
                    "corner": (cx_c, cz_c),
                    "legs": legs,
                    "total_len": along + perp_len,
                }

            cand_a = _build_l_candidate((x1, z1), (x2, z2), u1_out, u2_out)
            cand_b = _build_l_candidate((x2, z2), (x1, z1), u2_out, u1_out)
            candidates = [c for c in (cand_a, cand_b) if c is not None]
            if not candidates:
                continue
            candidates.sort(key=lambda c: c["total_len"])

            picked = None
            for cand in candidates:
                if not cand["legs"]:
                    continue
                segs = [
                    ((ax, az), (bx, bz), y_bot, y_top)
                    for (ax, az), (bx, bz) in cand["legs"]
                ]
                if _candidate_segments_cross_existing(segs, accepted_segments):
                    continue
                picked = (cand, segs)
                break

            if picked is None:
                continue

            cand, candidate_segments = picked
            cx, cz = cand["corner"]
            for (ax, az), (bx, bz) in cand["legs"]:
                leg_entry = {
                    "corners": [
                        [ax, y_bot, az],
                        [bx, y_bot, bz],
                        [bx, y_top, bz],
                        [ax, y_top, az],
                    ],
                    "story": story,
                    "type": "stitch",
                    "room_index": room_idx1,
                    "room_indices": [room_idx1, room_idx2],
                    "id": f"stitch:{story}:{len(stitch_walls)}",
                }
                record(leg_entry, STEP_STITCH_WALLS, "created")
                stitch_walls.append(leg_entry)
            accepted_segments.extend(candidate_segments)

            # Triangular floor/ceiling caps fill P1 - C - P2.
            floor_cap = {
                "corners": [
                    [x1, y_bot, z1],
                    [cx, y_bot, cz],
                    [x2, y_bot, z2],
                ],
                "story": story,
                "type": "stitch_floor",
                "room_index": room_idx1,
                "room_indices": [room_idx1, room_idx2],
                "id": f"stitch_floor:{story}:{len(stitch_walls)}",
            }
            record(floor_cap, STEP_STITCH_WALLS, "created")
            stitch_walls.append(floor_cap)

            ceiling_cap = {
                "corners": [
                    [x1, y_top, z1],
                    [cx, y_top, cz],
                    [x2, y_top, z2],
                ],
                "story": story,
                "type": "stitch_ceiling",
                "room_index": room_idx1,
                "room_indices": [room_idx1, room_idx2],
                "id": f"stitch_ceiling:{story}:{len(stitch_walls)}",
            }
            record(ceiling_cap, STEP_STITCH_WALLS, "created")
            stitch_walls.append(ceiling_cap)

    stitch_walls = snap_stitches_to_non_owner_walls(rooms_out, stitch_walls)
    # Snap/cap-reconciliation can force corners to coincide; drop any
    # quad that has collapsed before pruning/dedup sees it.
    stitch_walls = _drop_degenerate_stitches(stitch_walls)
    stitch_walls = prune_crossing_vertical_stitches(stitch_walls)
    stitch_walls = dedup_duplicate_vertical_stitches(stitch_walls)
    stitch_walls = _drop_stitches_crossing_rooms(stitch_walls, rooms_out)
    return _drop_degenerate_stitches(stitch_walls)


def _drop_stitches_crossing_rooms(stitch_walls, rooms_out):
    """Drop stitches whose xz base lies substantially inside a room's interior.

    Stitches are meant to fill the gap BETWEEN rooms (inter-room void), not
    cut through the middle of a room. But the snap/candidate pass can emit a
    stitch that connects two opposite corners of a small room via a diagonal
    -- visible from above as an X cutting across the room's floor polygon.
    Seen on b4f7407a room 3 (SE corner to NW corner diagonal + the other
    diagonal from the adjacent pair).

    Heuristic: project the base edge (vertices at min-y) to xz, compute its
    intersection length with every room's floor polygon interior; if it lies
    >= 60% inside some room, drop the stitch. 60% lets a stitch legitimately
    graze a room corner (where the connected wall ends) without being caught.
    """
    try:
        from shapely.geometry import LineString
        from shapely.geometry import Polygon as _SP
    except Exception:
        return stitch_walls

    room_polys = []
    for room in rooms_out:
        fp = room.get("floor_polygon") or []
        if len(fp) < 3:
            continue
        try:
            p = _SP([(c[0], c[2]) for c in fp])
            if p.is_valid and not p.is_empty and p.area > 0:
                room_polys.append(p)
        except Exception:
            continue
    if not room_polys:
        return stitch_walls

    INSIDE_FRACTION_MAX = 0.60
    kept = []
    for entry in stitch_walls:
        corners = entry.get("corners") or []
        if len(corners) < 2:
            kept.append(entry)
            continue
        ys = [c[1] for c in corners]
        y_min = min(ys)
        base = [c for c in corners if abs(c[1] - y_min) < 0.05]
        if len(base) < 2:
            kept.append(entry)
            continue
        seg = LineString([(base[0][0], base[0][2]), (base[-1][0], base[-1][2])])
        total = seg.length
        if total <= 1e-6:
            kept.append(entry)
            continue
        inside = 0.0
        for p in room_polys:
            inter = p.intersection(seg)
            if inter.is_empty or not hasattr(inter, "length"):
                continue
            inside = max(inside, inter.length)
        if inside / total >= INSIDE_FRACTION_MAX:
            record(
                entry,
                STEP_STITCH_WALLS,
                "dropped_crosses_room",
                f"inside_fraction={inside / total:.2f}",
            )
            continue
        kept.append(entry)
    return kept


def _drop_degenerate_stitches(stitch_walls):
    """Drop stitch entries whose corners collapse to fewer than 3 unique points.

    These appear when upstream snapping forces two wall endpoints onto the
    same 3D point -- the quad becomes ``[A, A, B, B]`` (collapses to a line)
    or the cap triangle becomes ``[A, A, B]`` (collapses to a segment). In
    b4f7407a room 3 this produces ~13 such stitches that render as visible
    bow-tie X artefacts on the tier and full-model views (see tracking entry
    for 2026-04-24 bow-tie repair).

    Also rejects entries whose XZ-plane bounding-box diagonal falls below
    ``_MIN_LEG_LENGTH_M``: a 6-mm-wide x 2-m-tall quad has 4 unique 3D
    vertices (it differs vertically) but still reads as a sliver / line
    from any side view. Catches slivers that branch gates missed because
    a snap displacement moved a corner post-emission.
    """
    kept = []
    for entry in stitch_walls:
        corners = entry.get("corners") or []
        unique = []
        for c in corners:
            same = False
            for u in unique:
                if (
                    abs(c[0] - u[0]) <= _DEGENERATE_VERTEX_TOL_M
                    and abs(c[1] - u[1]) <= _DEGENERATE_VERTEX_TOL_M
                    and abs(c[2] - u[2]) <= _DEGENERATE_VERTEX_TOL_M
                ):
                    same = True
                    break
            if not same:
                unique.append(c)
        if len(unique) < 3:
            record(
                entry,
                STEP_STITCH_WALLS,
                "dropped_degenerate",
                f"unique_vertices={len(unique)}",
            )
            continue

        if entry.get("type") == "stitch" and corners:
            xs = [c[0] for c in corners]
            zs = [c[2] for c in corners]
            xz_diag = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
            if xz_diag < _MIN_LEG_LENGTH_M:
                record(
                    entry,
                    STEP_STITCH_WALLS,
                    "dropped_sliver_base",
                    f"xz_diag_m={xz_diag:.4f}",
                )
                continue

        kept.append(entry)
    return kept
