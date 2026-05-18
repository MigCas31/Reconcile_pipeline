"""Ceiling inference and vertical wall extension helpers."""

import math
from collections import defaultdict

import numpy as np

SLOPED_CEILING_QUERY_MARGIN_M = 0.30


def build_sloped_ceiling_lookup(rooms_out):
    """Per-story registry of sloped-ceiling perimeter edges.

    Each room with ``ceiling_type == "sloped"`` contributes its
    ``ceiling_polygon`` (3D, traced from wall tops). At query time we project
    a (x, z) point onto each ceiling edge and linearly interpolate Y on the
    edge — gap-wall vertices live on or near a wall plane (= a ceiling
    perimeter edge), so projecting onto edges captures the actual roof Y at
    the eave/ridge end of the slope without needing an interior triangulation.
    Avoiding triangulation also sidesteps the common case where ridge
    vertices project onto the eave line in XZ (gable / hip shapes), which
    collapses any ear-clip output to a flat-Y triangulation.

    Returns ``{story: [{"edges": [(ax, az, ay, bx, bz, by), ...],
    "y_min": float, "y_max": float}, ...]}``.
    """
    lookup = defaultdict(list)
    for room in rooms_out:
        if room.get("ceiling_type") != "sloped":
            continue
        ceiling_poly = room.get("ceiling_polygon") or []
        if len(ceiling_poly) < 3:
            continue
        edges = []
        ys = []
        n = len(ceiling_poly)
        for i in range(n):
            a = ceiling_poly[i]
            b = ceiling_poly[(i + 1) % n]
            ax, ay, az = float(a[0]), float(a[1]), float(a[2])
            bx, by, bz = float(b[0]), float(b[1]), float(b[2])
            ys.append(ay)
            if (ax - bx) ** 2 + (az - bz) ** 2 < 1e-9:
                continue
            edges.append((ax, az, ay, bx, bz, by))
        if not edges:
            continue
        lookup[room.get("story", 0)].append(
            {
                "edges": edges,
                "y_min": min(ys),
                "y_max": max(ys),
            }
        )
    return lookup


def sloped_ceiling_y_at(lookup, xz_pt, story, margin=SLOPED_CEILING_QUERY_MARGIN_M):
    """Return the sloped-ceiling Y at (x, z) on ``story``, or None.

    For each room with a sloped ceiling on ``story``, find the closest
    perimeter edge (in XZ) and, if the query point is within ``margin`` m
    of that edge, return the linearly-interpolated Y at the projection.
    Across rooms, return the minimum — a gap vertex in the eave strip
    shared between two slopes should clip to whichever slope drops lowest.

    ``margin`` is set wide enough (0.30 m) to catch gap-wall vertices sitting
    in the wall-thickness strip between two adjacent rooms, where the gap
    edge sits a few cm outside both rooms' floor footprints.
    """
    entries = lookup.get(story)
    if not entries:
        return None
    px = float(xz_pt[0])
    pz = float(xz_pt[1])
    best = None
    for entry in entries:
        y = _edge_y_at(px, pz, entry["edges"], margin)
        if y is None:
            continue
        if best is None or y < best:
            best = y
    return best


def _edge_y_at(px, pz, edges, margin):
    """Y at (px, pz) projected onto the closest ceiling-perimeter edge.

    Returns None if no edge is within ``margin`` m. Otherwise the Y at the
    foot of the perpendicular from (px, pz) onto the closest edge,
    interpolated between the edge's endpoint Ys.
    """
    margin2 = margin * margin
    best_dist2 = margin2
    best_y = None
    for ax, az, ay, bx, bz, by in edges:
        ex = bx - ax
        ez = bz - az
        edge_len2 = ex * ex + ez * ez
        if edge_len2 < 1e-12:
            continue
        t = ((px - ax) * ex + (pz - az) * ez) / edge_len2
        t_c = max(0.0, min(1.0, t))
        cx = ax + t_c * ex
        cz = az + t_c * ez
        dx = px - cx
        dz = pz - cz
        dist2 = dx * dx + dz * dz
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_y = ay + t_c * (by - ay)
    return best_y


def reassign_raw_ceiling_planes_spatially(rooms_out):
    """Move each raw ceiling plane to the room whose floor it actually sits over.

    The per-room SVD matches a raw ceiling *file* to the best merged room via
    shared wall IDs, but Apple RoomPlan's captured ceiling mesh can span
    multiple rooms — a vaulted hallway, a shared pitched attic, etc. Without a
    spatial reassignment, every plane in the file lands on the single keyed
    room, which visually bleeds neighbouring rooms' ceilings onto the wrong
    footprint.

    For each plane, take its XZ centroid and pick the room (same story,
    building-local) whose floor polygon contains it. If no floor contains the
    centroid, fall back to the room with the largest 2D overlap with the plane.
    If nothing matches at all (should be rare — scan mesh drifting beyond every
    modelled room), keep the plane where it was so we don't silently drop data.
    """
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        return

    room_polys = []
    for room in rooms_out:
        fp = room.get("floor_polygon") or []
        if len(fp) < 3:
            room_polys.append(None)
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in fp])
            if not poly.is_valid:
                poly = poly.buffer(0)
            room_polys.append(poly if poly.is_valid and not poly.is_empty else None)
        except Exception:
            room_polys.append(None)

    reassignments = [[] for _ in rooms_out]
    for src_idx, room in enumerate(rooms_out):
        src_story = room.get("story", 0)
        for plane in room.get("raw_ceiling_planes") or []:
            corners = plane.get("corners") or []
            if len(corners) < 3:
                reassignments[src_idx].append(plane)
                continue
            xs = [c[0] for c in corners]
            zs = [c[2] for c in corners]
            cx = sum(xs) / len(xs)
            cz = sum(zs) / len(zs)
            target_idx = None
            centroid = Point(cx, cz)
            for ridx, poly in enumerate(room_polys):
                if poly is None or rooms_out[ridx].get("story", 0) != src_story:
                    continue
                if poly.contains(centroid):
                    target_idx = ridx
                    break
            if target_idx is None:
                try:
                    plane_poly = Polygon([(c[0], c[2]) for c in corners])
                    if not plane_poly.is_valid:
                        plane_poly = plane_poly.buffer(0)
                except Exception:
                    plane_poly = None
                if (
                    plane_poly is not None
                    and plane_poly.is_valid
                    and not plane_poly.is_empty
                ):
                    best_area = 0.0
                    for ridx, poly in enumerate(room_polys):
                        if poly is None or rooms_out[ridx].get("story", 0) != src_story:
                            continue
                        try:
                            inter = poly.intersection(plane_poly).area
                        except Exception:
                            inter = 0.0
                        if inter > best_area:
                            best_area = inter
                            target_idx = ridx
            if target_idx is None:
                target_idx = src_idx
            reassignments[target_idx].append(plane)

    for ridx, room in enumerate(rooms_out):
        room["raw_ceiling_planes"] = reassignments[ridx]


def extend_wall_to_slab(corners, slab_y_above, epsilon=0.05, max_gap=0.80):
    if len(corners) < 3:
        return None

    ys = [corner[1] for corner in corners]
    max_y = max(ys)
    min_y = min(ys)
    wall_height = max_y - min_y
    if wall_height < 0.1:
        return None

    y_thresh = min_y + wall_height * 0.4
    top_indices = [idx for idx, y_val in enumerate(ys) if y_val > y_thresh]
    if not top_indices:
        return None

    need_ext = [idx for idx in top_indices if ys[idx] < slab_y_above - epsilon]
    if not need_ext:
        return None

    max_top_y = max(ys[idx] for idx in top_indices)
    if slab_y_above - max_top_y > max_gap:
        return None

    extended = [list(corner) for corner in corners]
    need_ext_set = set(need_ext)
    for idx in need_ext:
        extended[idx][1] = slab_y_above

    extension_strips = []
    top_indices_sorted = sorted(top_indices)
    for idx in range(len(top_indices_sorted) - 1):
        i0 = top_indices_sorted[idx]
        i1 = top_indices_sorted[idx + 1]
        orig_y0 = corners[i0][1]
        orig_y1 = corners[i1][1]
        ext_y0 = slab_y_above if i0 in need_ext_set else orig_y0
        ext_y1 = slab_y_above if i1 in need_ext_set else orig_y1
        if abs(ext_y0 - orig_y0) < 1e-4 and abs(ext_y1 - orig_y1) < 1e-4:
            continue
        extension_strips.append(
            [
                [corners[i0][0], orig_y0, corners[i0][2]],
                [corners[i1][0], orig_y1, corners[i1][2]],
                [corners[i1][0], ext_y1, corners[i1][2]],
                [corners[i0][0], ext_y0, corners[i0][2]],
            ]
        )

    if not extension_strips:
        return None

    return {"extended_corners": extended, "extension_strip": extension_strips}


# --- Should-be-flat classifier ---------------------------------------------
# Wall-top consensus (P10/P50/P90 spread) plus a guard that no raw-ceiling
# point reaches above the wall tops, plus a same-story neighbour vote. Runs
# *independently* of raw_ceiling_planes' shape, so wall polygons stored as
# ceiling planes (the noMesh ingest path on iOS) cannot contaminate the
# classification — that bug previously left 81% of rooms with ceiling_type=None.
WALL_TOP_SPREAD_MAX_M = 0.20
CEILING_OVERSHOOT_MAX_M = 0.30
RIDGE_EAVE_MAX_DELTA_M = 0.10
NEIGHBOUR_TOLERANCE_M = 0.30
NOISY_PLANE_DROP_TAU_M = 0.30


def _wall_top_percentiles(walls):
    tops = []
    for w in walls:
        cs = w.get("corners") or []
        ys = [c[1] for c in cs if isinstance(c, (list, tuple)) and len(c) >= 2]
        if ys:
            tops.append(max(ys))
    if len(tops) < 2:
        return None
    return {
        "p10": float(np.percentile(tops, 10)),
        "p50": float(np.percentile(tops, 50)),
        "p90": float(np.percentile(tops, 90)),
    }


def _max_raw_ceiling_corner_y(room):
    best = None
    for plane in room.get("raw_ceiling_planes") or []:
        for corner in plane.get("corners") or []:
            if len(corner) >= 2 and (best is None or corner[1] > best):
                best = corner[1]
    return best


def _ridge_eave_consistent(room):
    eave = room.get("ceiling_eave_height")
    ridge = room.get("ceiling_ridge_height")
    if eave is None or ridge is None:
        return None
    return abs(ridge - eave) < RIDGE_EAVE_MAX_DELTA_M


def classify_should_be_flat(rooms_out):
    """Return {room_idx: (should_be_flat, expected_y)}.

    A room "should be flat" when its walls cluster at one height (low P90-P10
    spread) AND no raw-ceiling point pokes above the wall-top consensus. A
    same-story neighbour vote reinforces edge cases (rooms with ≤ 2 walls).
    Existing ridge/eave fields, when present, can veto ("ridge - eave > 0.10m"
    means a real slope was already detected).
    """
    per_room = []
    for room in rooms_out:
        walls = room.get("walls_computed") or room.get("walls_merged") or []
        per_room.append(
            {
                "room": room,
                "ws": _wall_top_percentiles(walls),
                "max_raw_y": _max_raw_ceiling_corner_y(room),
            }
        )

    by_story_p50 = {}
    for entry in per_room:
        ws = entry["ws"]
        if ws is None:
            continue
        if (ws["p90"] - ws["p10"]) >= WALL_TOP_SPREAD_MAX_M:
            continue
        max_raw = entry["max_raw_y"]
        if max_raw is not None and (max_raw - ws["p50"]) >= CEILING_OVERSHOOT_MAX_M:
            continue
        story = entry["room"].get("story", 0)
        by_story_p50.setdefault(story, []).append(ws["p50"])

    neighbour_median = {
        story: float(np.median(p50s)) for story, p50s in by_story_p50.items() if p50s
    }

    out = {}
    for ri, entry in enumerate(per_room):
        ws = entry["ws"]
        room = entry["room"]
        if ws is None:
            out[ri] = (False, None)
            continue
        if _ridge_eave_consistent(room) is False:
            out[ri] = (False, None)
            continue
        max_raw = entry["max_raw_y"]
        ceiling_consistent = (max_raw is None) or (
            (max_raw - ws["p50"]) < CEILING_OVERSHOOT_MAX_M
        )
        if not ceiling_consistent:
            out[ri] = (False, None)
            continue
        internal_flat = (ws["p90"] - ws["p10"]) < WALL_TOP_SPREAD_MAX_M
        nbm = neighbour_median.get(room.get("story", 0))
        nb_consensus = (nbm is not None) and (
            abs(ws["p50"] - nbm) < NEIGHBOUR_TOLERANCE_M
        )
        if internal_flat:
            out[ri] = (True, ws["p50"])
        elif nb_consensus:
            out[ri] = (True, nbm)
        else:
            out[ri] = (False, None)
    return out


def drop_noisy_raw_ceiling_planes(rooms_out, classifications):
    """For should-be-flat rooms, drop raw planes with corners far below the consensus.

    A real ceiling plane never reaches more than ~0.3 m below the consensus
    ceiling height. Wall polygons stored as ceiling planes by the noMesh ingest
    path span floor-to-ceiling (or upper-half-of-wall for spandrels) — their
    `min(corner_y)` is well below `expected_y`.
    """
    for ri, room in enumerate(rooms_out):
        should_flat, expected_y = classifications.get(ri, (False, None))
        if not should_flat or expected_y is None:
            continue
        kept = []
        for plane in room.get("raw_ceiling_planes") or []:
            corners = plane.get("corners") or []
            if not corners:
                continue
            if (expected_y - min(c[1] for c in corners)) > NOISY_PLANE_DROP_TAU_M:
                continue
            kept.append(plane)
        room["raw_ceiling_planes"] = kept


def apply_flat_classification(rooms_out, classifications):
    """Set ceiling_type="flat" + heights for should-be-flat rooms.

    When the room already has a `ceiling_polygon` (built by the slant pipeline),
    keep it — it follows the wall tops more faithfully than a floor-lift. When
    not, lift the floor polygon to the ceiling height. Either way, set
    ceiling_type/eave/ridge so downstream consumers see the flat conclusion.
    """
    for ri, room in enumerate(rooms_out):
        should_flat, expected_y = classifications.get(ri, (False, None))
        if not should_flat or expected_y is None:
            continue
        kept_max_y = _max_raw_ceiling_corner_y(room)
        ceil_y = max(expected_y, kept_max_y) if kept_max_y is not None else expected_y
        ceil_y = round(float(ceil_y), 4)
        existing_poly = room.get("ceiling_polygon") or []
        existing_is_flat = (
            len(existing_poly) >= 3
            and (max(p[1] for p in existing_poly) - min(p[1] for p in existing_poly))
            < RIDGE_EAVE_MAX_DELTA_M
        )
        if existing_is_flat:
            existing_y = round(float(existing_poly[0][1]), 4)
            room["ceiling_type"] = "flat"
            room["ceiling_eave_height"] = existing_y
            room["ceiling_ridge_height"] = existing_y
        else:
            fp = room.get("floor_polygon") or []
            if len(fp) >= 3:
                room["ceiling_polygon"] = [
                    [round(v[0], 4), ceil_y, round(v[2], 4)] for v in fp
                ]
            room["ceiling_type"] = "flat"
            room["ceiling_eave_height"] = ceil_y
            room["ceiling_ridge_height"] = ceil_y


def infer_ceilings(rooms_out):
    classifications = classify_should_be_flat(rooms_out)
    drop_noisy_raw_ceiling_planes(rooms_out, classifications)
    apply_flat_classification(rooms_out, classifications)

    slant_thresh = 0.15
    slope_thresh = 0.15
    xz_match_tol = 0.20

    for ri, room in enumerate(rooms_out):
        if classifications.get(ri, (False, None))[0]:
            continue
        floor_polygon = room.get("floor_polygon", [])
        walls = room.get("walls_computed") or room.get("walls_merged", [])
        if len(floor_polygon) < 3 or not walls:
            continue

        has_slant = False
        for wall in walls:
            corners = wall["corners"]
            if len(corners) < 3:
                continue
            ys = [corner[1] for corner in corners]
            mid_y = (max(ys) + min(ys)) / 2.0
            top_corners = [corner for corner in corners if corner[1] > mid_y - 0.01]
            if len(top_corners) >= 2:
                top_range = max(corner[1] for corner in top_corners) - min(
                    corner[1] for corner in top_corners
                )
                if top_range > slant_thresh:
                    has_slant = True
                    break

        if not has_slant:
            continue

        ceiling_polygon = build_ceiling_from_wall_tops(
            floor_polygon, walls, xz_match_tol
        )
        if ceiling_polygon is None or len(ceiling_polygon) < 3:
            continue

        ceiling_ys = [point[1] for point in ceiling_polygon]
        y_range = max(ceiling_ys) - min(ceiling_ys)
        ceiling_type = "flat" if y_range < slope_thresh else "sloped"
        room["ceiling_polygon"] = ceiling_polygon
        room["ceiling_type"] = ceiling_type
        room["ceiling_ridge_height"] = round(max(ceiling_ys), 4)
        room["ceiling_eave_height"] = round(min(ceiling_ys), 4)


def build_ceiling_from_wall_tops(floor_polygon, walls, xz_tol):
    def xz_dist(a, b):
        return math.hypot(a[0] - b[0], a[2] - b[2])

    wall_info = []
    for wall in walls:
        corners = wall["corners"]
        if len(corners) < 3:
            continue
        ys = [corner[1] for corner in corners]
        mid_y = (max(ys) + min(ys)) / 2.0
        bot = [corner for corner in corners if corner[1] <= mid_y + 0.01]
        top = [corner for corner in corners if corner[1] > mid_y - 0.01]
        if len(bot) >= 2 and len(top) >= 2:
            wall_info.append({"bot": bot, "top": top})

    if not wall_info:
        return None

    def nearest_corner(corner, corners):
        best_d = float("inf")
        best_corner = None
        for candidate in corners:
            dist = xz_dist(corner, candidate)
            if dist < best_d:
                best_d = dist
                best_corner = candidate
        return best_d, best_corner

    n = len(floor_polygon)
    edge_walls = [None] * n
    used = set()
    for edge_idx in range(n):
        v0 = floor_polygon[edge_idx]
        v1 = floor_polygon[(edge_idx + 1) % n]
        best_score = float("inf")
        best_idx = None
        for wi, wi_info in enumerate(wall_info):
            if wi in used:
                continue
            d0, _ = nearest_corner(v0, wi_info["bot"])
            d1, _ = nearest_corner(v1, wi_info["bot"])
            score = d0 + d1
            if score < best_score:
                best_score = score
                best_idx = wi
        if best_idx is not None and best_score < xz_tol * 2:
            edge_walls[edge_idx] = best_idx
            used.add(best_idx)

    ceiling_pts = []
    for edge_idx in range(n):
        wi = edge_walls[edge_idx]
        if wi is None:
            continue

        v0 = floor_polygon[edge_idx]
        v1 = floor_polygon[(edge_idx + 1) % n]
        top_corners = wall_info[wi]["top"]
        bot_corners = wall_info[wi]["bot"]

        start_bot = sorted((xz_dist(v0, bc), bc) for bc in bot_corners)[0][1]
        end_bot = sorted((xz_dist(v1, bc), bc) for bc in bot_corners)[0][1]

        def find_top_at_xz(bot_corner, _top_corners=top_corners):
            return sorted((xz_dist(bot_corner, tc), tc) for tc in _top_corners)[0][1]

        start_top = find_top_at_xz(start_bot)
        end_top = find_top_at_xz(end_bot)

        if len(top_corners) <= 2:
            ordered_top = [start_top, end_top]
        else:
            interior = []
            for corner in top_corners:
                if (
                    xz_dist(corner, start_bot) > xz_tol
                    and xz_dist(corner, end_bot) > xz_tol
                ):
                    interior.append(corner)

            edge_dir = np.array([v1[0] - v0[0], v1[2] - v0[2]])
            edge_len = np.linalg.norm(edge_dir)
            if edge_len > 1e-6:
                unit = edge_dir / edge_len
                interior.sort(
                    key=lambda c: np.dot(np.array([c[0] - v0[0], c[2] - v0[2]]), unit)
                )

            ordered_top = [start_top, *interior, end_top]

        for corner in ordered_top:
            point = [round(corner[0], 4), round(corner[1], 4), round(corner[2], 4)]
            if ceiling_pts:
                last = ceiling_pts[-1]
                if abs(point[0] - last[0]) < 0.01 and abs(point[2] - last[2]) < 0.01:
                    if point[1] > last[1]:
                        ceiling_pts[-1] = point
                    continue
            ceiling_pts.append(point)

    if len(ceiling_pts) >= 2:
        first = ceiling_pts[0]
        last = ceiling_pts[-1]
        if abs(first[0] - last[0]) < 0.01 and abs(first[2] - last[2]) < 0.01:
            if last[1] > first[1]:
                ceiling_pts[0] = last
            ceiling_pts.pop()

    return ceiling_pts if len(ceiling_pts) >= 3 else None


def flat_ceiling_fallback(rooms, slab_y):
    _ = slab_y
    wall_tops = []
    for room in rooms:
        for wall in room.get("walls_computed") or room.get("walls_merged", []):
            ys = [corner[1] for corner in wall["corners"]]
            if ys:
                wall_tops.append(max(ys))

    if not wall_tops:
        return

    ceil_y = round(float(np.median(wall_tops)), 4)
    for room in rooms:
        floor_polygon = room.get("floor_polygon", [])
        if len(floor_polygon) < 3:
            continue
        room["ceiling_polygon"] = [
            [round(v[0], 4), ceil_y, round(v[2], 4)] for v in floor_polygon
        ]
        room["ceiling_type"] = "flat"
        room["ceiling_ridge_height"] = ceil_y
        room["ceiling_eave_height"] = ceil_y


def find_closest_slab_y(wall_corners, slabs_above):
    if not slabs_above:
        return None
    wall_x = np.mean([corner[0] for corner in wall_corners])
    wall_z = np.mean([corner[2] for corner in wall_corners])
    best_dist = float("inf")
    best_y = None
    for slab_x, slab_z, slab_y in slabs_above:
        dist = math.hypot(wall_x - slab_x, wall_z - slab_z)
        if dist < best_dist:
            best_dist = dist
            best_y = slab_y
    return best_y


COHORT_TOLERANCE_M = 0.15
MIN_COHORT_COVERAGE = 0.70
MAX_OUTLIER_SPAN_FRAC = 0.25
MAX_DOMINANT_DELTA_M = 0.40
MIN_DOMINANT_DELTA_M = 0.10
FLOOR_MATCH_TOLERANCE_M = 0.10
COLINEAR_ANGLE_DEG = 8.0
COLINEAR_OFFSET_M = 0.15
COLINEAR_GAP_M = 0.80


def _wall_bottom_chord_xz(corners):
    """Return the longest chord among bottom corners as (p0, p1, length).

    Robust to non-canonical walls (5+ corners, arbitrary ordering) — selects
    corners within 1 cm of min(y) and picks the widest pair in XZ.
    """
    if len(corners) < 2:
        return None
    ys = [c[1] for c in corners]
    min_y = min(ys)
    bot = [c for c in corners if c[1] < min_y + 0.01]
    if len(bot) < 2:
        return None
    best_len = 0.0
    best_pair = None
    for i in range(len(bot)):
        for j in range(i + 1, len(bot)):
            dx = bot[i][0] - bot[j][0]
            dz = bot[i][2] - bot[j][2]
            d = math.hypot(dx, dz)
            if d > best_len:
                best_len = d
                best_pair = (bot[i], bot[j])
    if best_pair is None or best_len < 1e-4:
        return None
    return best_pair[0], best_pair[1], best_len


def _wall_record(corners):
    """Pre-compute top_y, bot_y, span, axis, mid for a wall — used by the cohort."""
    chord = _wall_bottom_chord_xz(corners)
    if chord is None:
        return None
    p0, p1, span = chord
    ys = [c[1] for c in corners]
    axis = ((p1[0] - p0[0]) / span, (p1[2] - p0[2]) / span)
    mid = ((p0[0] + p1[0]) * 0.5, (p0[2] + p1[2]) * 0.5)
    return {
        "top_y": float(max(ys)),
        "bot_y": float(min(ys)),
        "span": float(span),
        "axis": axis,
        "mid": mid,
    }


def _weighted_median(items):
    items = sorted(items, key=lambda t: t[0])
    total = sum(w for _, w in items)
    if total <= 0:
        return items[len(items) // 2][0]
    half = total / 2.0
    acc = 0.0
    for val, w in items:
        acc += w
        if acc >= half:
            return val
    return items[-1][0]


def compute_story_wall_top_cohort(rooms_out, story):
    """Find the dominant top-Y cohort for walls_computed on the given story.

    Returns None if no single cluster (tolerance ``COHORT_TOLERANCE_M``) covers
    at least ``MIN_COHORT_COVERAGE`` of the total span-weighted perimeter. The
    returned dict is the only input the per-wall gates need — it includes
    pre-computed records for every eligible wall on the story so that the
    colinearity check does not rebuild them.
    """
    records = []
    for room in rooms_out:
        if room.get("story", 0) != story:
            continue
        for wall in room.get("walls_computed") or []:
            rec = _wall_record(wall.get("corners") or [])
            if rec is None or rec["span"] < 0.05 or rec["top_y"] - rec["bot_y"] < 0.10:
                continue
            records.append(rec)
    if not records:
        return None

    total_perim = sum(r["span"] for r in records)
    if total_perim <= 0:
        return None

    sorted_by_top = sorted(records, key=lambda r: r["top_y"])
    clusters = [[sorted_by_top[0]]]
    for rec in sorted_by_top[1:]:
        if rec["top_y"] - clusters[-1][-1]["top_y"] <= COHORT_TOLERANCE_M:
            clusters[-1].append(rec)
        else:
            clusters.append([rec])

    best_cluster = max(
        clusters,
        key=lambda cl: sum(r["span"] for r in cl),
    )
    best_w = sum(r["span"] for r in best_cluster)
    coverage = best_w / total_perim
    if coverage < MIN_COHORT_COVERAGE:
        return None

    dominant_y = sum(r["top_y"] * r["span"] for r in best_cluster) / best_w
    cohort_floor_y = _weighted_median([(r["bot_y"], r["span"]) for r in best_cluster])

    return {
        "dominant_y": dominant_y,
        "coverage_frac": coverage,
        "total_perimeter": total_perim,
        "cohort_floor_y": cohort_floor_y,
        "records": records,
    }


def _is_colinear_neighbour(target_rec, other_rec):
    ax, az = target_rec["axis"]
    bx, bz = other_rec["axis"]
    cos_a = min(1.0, max(-1.0, abs(ax * bx + az * bz)))
    if math.degrees(math.acos(cos_a)) > COLINEAR_ANGLE_DEG:
        return False
    nx, nz = -az, ax
    tx, tz = target_rec["mid"]
    ox, oz = other_rec["mid"]
    perp = abs((ox - tx) * nx + (oz - tz) * nz)
    if perp > COLINEAR_OFFSET_M:
        return False
    along = (ox - tx) * ax + (oz - tz) * az
    gap = abs(along) - (target_rec["span"] + other_rec["span"]) * 0.5
    return gap <= COLINEAR_GAP_M


def should_extend_wall_to_dominant(corners, cohort):
    """Return the target top-Y if this wall passes every dominant-height gate.

    Gates applied in order (fast-fail):
      * wall top-Y is below dominant by at least ``MIN_DOMINANT_DELTA_M``
        and by no more than ``MAX_DOMINANT_DELTA_M``
      * wall span is at most ``MAX_OUTLIER_SPAN_FRAC`` of cohort perimeter
      * wall bottom-Y matches the cohort floor-Y within ``FLOOR_MATCH_TOLERANCE_M``
      * wall is colinear with at least one dominant-cohort neighbour
    """
    if cohort is None:
        return None
    rec = _wall_record(corners)
    if rec is None:
        return None
    dominant_y = cohort["dominant_y"]
    delta = dominant_y - rec["top_y"]
    if delta < MIN_DOMINANT_DELTA_M or delta > MAX_DOMINANT_DELTA_M:
        return None
    if rec["span"] > MAX_OUTLIER_SPAN_FRAC * cohort["total_perimeter"]:
        return None
    cohort_floor_y = cohort.get("cohort_floor_y")
    if cohort_floor_y is None:
        return None
    if abs(rec["bot_y"] - cohort_floor_y) > FLOOR_MATCH_TOLERANCE_M:
        return None
    for other in cohort["records"]:
        if other is rec:
            continue
        if abs(other["top_y"] - dominant_y) > COHORT_TOLERANCE_M:
            continue
        if _is_colinear_neighbour(rec, other):
            return dominant_y
    return None


def extend_wall_to_dominant(corners, dominant_y, epsilon=0.05):
    """Lift the wall's top corners to ``dominant_y``.

    Mirrors :func:`extend_wall_to_slab` so the resulting record has the same
    ``extended_corners``/``extension_strip`` shape the viewer already renders.
    ``should_extend_wall_to_dominant`` must have already approved the wall —
    this function performs only the geometric lift.
    """
    if len(corners) < 3:
        return None
    ys = [c[1] for c in corners]
    max_y = max(ys)
    min_y = min(ys)
    if max_y - min_y < 0.1:
        return None
    y_thresh = min_y + (max_y - min_y) * 0.4
    top_indices = [idx for idx, y in enumerate(ys) if y > y_thresh]
    if not top_indices:
        return None
    need_ext = [idx for idx in top_indices if ys[idx] < dominant_y - epsilon]
    if not need_ext:
        return None

    extended = [list(c) for c in corners]
    need_ext_set = set(need_ext)
    for idx in need_ext:
        extended[idx][1] = dominant_y

    extension_strips = []
    top_sorted = sorted(top_indices)
    for a in range(len(top_sorted) - 1):
        i0, i1 = top_sorted[a], top_sorted[a + 1]
        orig_y0, orig_y1 = corners[i0][1], corners[i1][1]
        ext_y0 = dominant_y if i0 in need_ext_set else orig_y0
        ext_y1 = dominant_y if i1 in need_ext_set else orig_y1
        if abs(ext_y0 - orig_y0) < 1e-4 and abs(ext_y1 - orig_y1) < 1e-4:
            continue
        extension_strips.append(
            [
                [corners[i0][0], orig_y0, corners[i0][2]],
                [corners[i1][0], orig_y1, corners[i1][2]],
                [corners[i1][0], ext_y1, corners[i1][2]],
                [corners[i0][0], ext_y0, corners[i0][2]],
            ]
        )
    if not extension_strips:
        return None
    return {"extended_corners": extended, "extension_strip": extension_strips}


def find_best_slab_above(
    wall_corners,
    wall_top_y,
    slabs_above,
    min_margin=0.05,
    max_gap=None,
    stack_tol=0.10,
):
    """Pick the slab above whose XZ footprint is spatially closest to the wall midpoint.

    slabs_above: list of (shapely_polygon_xz, floor_y). Polygon coords are (x, z).
    Returns the chosen slab's floor_y, or None if no slab is strictly above
    `wall_top_y + min_margin` (and within `max_gap` if provided).

    `max_gap` mirrors `extend_wall_to_slab`'s gate: candidates with
    `slab_y - wall_top_y > max_gap` are skipped, so the picker can't prefer an
    unreachable high slab over a closer-but-less-central viable slab.

    `stack_tol` enforces the stacking constraint: only extend a wall when its
    midpoint sits inside (or within `stack_tol` metres of) an upper-story room
    floor polygon. Walls whose midpoint is farther than `stack_tol` from every
    candidate slab are air-facing — the user supplies those thicknesses from
    the envelope, so no auto-extension is needed. Prevents extensions in
    detached wings / outbuildings that have no room overhead.
    """
    if not slabs_above:
        return None
    from shapely.geometry import Point  # local import to keep module load light

    wx = float(np.mean([c[0] for c in wall_corners]))
    wz = float(np.mean([c[2] for c in wall_corners]))
    pt = Point(wx, wz)
    best_dist = float("inf")
    best_y = None
    for poly, slab_y in slabs_above:
        if slab_y <= wall_top_y + min_margin:
            continue
        if max_gap is not None and slab_y - wall_top_y > max_gap:
            continue
        dist = float(poly.distance(pt))
        if stack_tol is not None and dist > stack_tol:
            continue
        if dist < best_dist:
            best_dist = dist
            best_y = slab_y
    return best_y
