"""Exterior gap indicators and closure geometry."""

import math
from collections import defaultdict

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon
from shapely.validation import make_valid


def wall_xz_length(corners):
    """Compute the XZ-plane length of a wall/element from its corners."""
    if len(corners) < 2:
        return 0.0
    c = np.array(corners)
    dx = c[1][0] - c[0][0]
    dz = c[1][2] - c[0][2]
    return math.hypot(dx, dz)


def _canonicalize_wall_quad(corners):
    """Normalize a wall with arbitrary corner count/order to a 4-corner
    quad ``[bl, br, tr, tl]`` whose bottom edge is the wall's length axis.

    Walls produced by scan-cache / extension-strip code paths can have
    5+ corners or non-canonical ordering, which breaks downstream code
    that assumes ``corners[0..1]`` is the bottom edge (the gap-closure
    construction in particular).

    For canonical 4-corner walls already ordered ``[bl, br, tr, tl]`` the
    return value is bit-identical to the input.
    """
    if len(corners) < 4:
        return [list(c) for c in corners]
    arr = [list(c) for c in corners]
    ys = [c[1] for c in arr]
    if max(ys) - min(ys) < 1e-3:
        return arr
    mid_y = (max(ys) + min(ys)) / 2.0
    bot_cs = [c for c in arr if c[1] < mid_y + 0.01]
    top_cs = [c for c in arr if c[1] > mid_y - 0.01]
    if len(bot_cs) < 2 or len(top_cs) < 2:
        return arr

    bl, br = bot_cs[0], bot_cs[1]
    best_d2 = -1.0
    for i in range(len(bot_cs)):
        for j in range(i + 1, len(bot_cs)):
            dx = bot_cs[i][0] - bot_cs[j][0]
            dz = bot_cs[i][2] - bot_cs[j][2]
            d2 = dx * dx + dz * dz
            if d2 > best_d2:
                best_d2 = d2
                bl, br = bot_cs[i], bot_cs[j]

    # Preserve the incoming corners[0]→corners[1] xz direction so canonical
    # walls round-trip unchanged and the pc/wc axes stay consistent.
    ref_dx = arr[1][0] - arr[0][0]
    ref_dz = arr[1][2] - arr[0][2]
    if (br[0] - bl[0]) * ref_dx + (br[2] - bl[2]) * ref_dz < 0:
        bl, br = br, bl

    def _nearest(x, z, candidates):
        best = candidates[0]
        best_d2 = float("inf")
        for c in candidates:
            dx = c[0] - x
            dz = c[2] - z
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best = c
        return best

    tl = _nearest(bl[0], bl[2], top_cs)
    tr = _nearest(br[0], br[2], top_cs)
    return [list(bl), list(br), list(tr), list(tl)]


def _wall_top_bottom_profiles(corners, axis2d, origin_xz):
    """Project a wall polygon to (t, y) and split into top/bottom polylines.

    `t` = signed projection of corner XZ onto ``axis2d`` (relative to
    ``origin_xz``). The polygon's vertex order is partitioned into two
    monotonic-in-t runs; vertical (zero-dt) edges are absorbed into the
    surrounding run. The run with higher mean y becomes the top profile.

    Each profile is returned as a list of (t, y) sorted by t and
    deduplicated per t — top keeps the maximum y, bottom keeps the
    minimum. Returns ([], []) for degenerate inputs.

    Preserves kinks in the upper outline (e.g. an attic wall whose top
    follows a sloped roof eave) that a Y-mid split would discard.
    """
    if len(corners) < 3:
        return [], []
    pts = []
    for c in corners:
        dx = float(c[0]) - float(origin_xz[0])
        dz = float(c[2]) - float(origin_xz[1])
        t = dx * axis2d[0] + dz * axis2d[1]
        pts.append((t, float(c[1])))
    n = len(pts)

    eps = 1e-9
    t_min = min(p[0] for p in pts)
    t_max = max(p[0] for p in pts)
    if t_max - t_min < 1e-6:
        return [], []

    runs = []
    current = [pts[0]]
    current_sign = 0
    for i in range(1, n + 1):
        a = pts[(i - 1) % n]
        b = pts[i % n]
        dt = b[0] - a[0]
        if abs(dt) < eps:
            current.append(b)
            continue
        sign = 1 if dt > 0 else -1
        if current_sign == 0 or current_sign == sign:
            current.append(b)
            current_sign = sign
        else:
            runs.append((current_sign, current))
            current = [a, b]
            current_sign = sign
    if current:
        runs.append((current_sign, current))
    if len(runs) >= 2 and runs[0][0] == runs[-1][0]:
        merged = runs[-1][1] + runs[0][1]
        runs = [(runs[0][0], merged), *runs[1:-1]]

    fwd = next((r for s, r in runs if s == 1), None)
    bwd = next((r for s, r in runs if s == -1), None)
    if not fwd or not bwd:
        return [], []

    def _consolidate(profile, take_max):
        s = sorted(profile, key=lambda p: p[0])
        out = []
        for t, y in s:
            if out and abs(out[-1][0] - t) < 1e-6:
                out[-1] = (
                    out[-1][0],
                    max(out[-1][1], y) if take_max else min(out[-1][1], y),
                )
            else:
                out.append((t, y))
        return out

    fwd_my = sum(p[1] for p in fwd) / len(fwd)
    bwd_my = sum(p[1] for p in bwd) / len(bwd)
    if fwd_my >= bwd_my:
        top_raw, bot_raw = fwd, bwd
    else:
        top_raw, bot_raw = bwd, fwd
    return _consolidate(top_raw, take_max=True), _consolidate(bot_raw, take_max=False)


def _interp_profile_y(profile, t):
    """Piecewise-linear interpolation of y at parameter t along profile.

    Profile is a list of (t, y) sorted by t. Out-of-range t clamps to ends.
    """
    if not profile:
        return 0.0
    if t <= profile[0][0]:
        return profile[0][1]
    if t >= profile[-1][0]:
        return profile[-1][1]
    for k in range(len(profile) - 1):
        t0, y0 = profile[k]
        t1, y1 = profile[k + 1]
        if t0 <= t <= t1:
            span = t1 - t0
            if span < 1e-9:
                return y0
            return y0 + (t - t0) / span * (y1 - y0)
    return profile[-1][1]


def detect_exterior_gap_indicators(rooms_out):
    """Detect gaps in front of doors/openings/storages.

    Looks for a parallel wall on the other side.
    """
    min_width = 0.50
    min_dist = 0.20
    max_dist = 1.0
    max_angle = 15.0
    width_tolerance = 0.9

    story_walls = defaultdict(list)
    for room_idx, room in enumerate(rooms_out):
        for wall in room["walls_computed"]:
            story_walls[room["story"]].append((wall, room_idx))

    indicators = []

    def find_parallel_wall(corners, elem_width, story, room_idx, require_other_room):
        c = np.array(corners)
        e1 = c[1] - c[0]
        e2 = c[2] - c[1]
        normal = np.cross(e1, e2)
        normal_xz = np.array([normal[0], normal[2]])
        nlen = np.linalg.norm(normal_xz)
        if nlen < 1e-6:
            return None
        normal_xz = normal_xz / nlen

        elem_cx = float(np.mean([co[0] for co in corners]))
        elem_cz = float(np.mean([co[2] for co in corners]))

        best_match = None
        best_dist = float("inf")

        for sign in [1.0, -1.0]:
            direction = normal_xz * sign

            for wall, wall_room_idx in story_walls.get(story, []):
                if require_other_room and wall_room_idx == room_idx:
                    continue

                wall_corners = wall["corners"]
                if len(wall_corners) < 3:
                    continue

                wall_width = wall_xz_length(wall_corners)
                if wall_width < elem_width * width_tolerance:
                    continue

                wall_cx = float(np.mean([co[0] for co in wall_corners]))
                wall_cz = float(np.mean([co[2] for co in wall_corners]))

                to_wall = np.array([wall_cx - elem_cx, wall_cz - elem_cz])
                dist_along = float(np.dot(to_wall, direction))

                if dist_along < min_dist or dist_along > max_dist:
                    continue

                perp = float(
                    np.abs(np.dot(to_wall, np.array([-direction[1], direction[0]])))
                )
                if perp > max(elem_width, wall_width) * 0.5:
                    continue

                wall_arr = np.array(wall_corners)
                we1 = wall_arr[1] - wall_arr[0]
                we2 = wall_arr[2] - wall_arr[1]
                wnormal = np.cross(we1, we2)
                wnormal_xz = np.array([wnormal[0], wnormal[2]])
                wnlen = np.linalg.norm(wnormal_xz)
                if wnlen < 1e-6:
                    continue
                wnormal_xz = wnormal_xz / wnlen

                cos_angle = abs(float(np.dot(normal_xz, wnormal_xz)))
                cos_angle = min(cos_angle, 1.0)
                angle_deg = math.degrees(math.acos(cos_angle))

                if angle_deg > max_angle:
                    continue

                if dist_along < best_dist:
                    best_dist = dist_along
                    best_match = (wall, wall_corners, dist_along, angle_deg)

        return best_match

    for room_idx, room in enumerate(rooms_out):
        story = room["story"]

        for elem_type, elems in [
            ("door", room.get("doors", [])),
            ("opening", room.get("openings", [])),
        ]:
            for elem in elems:
                corners = elem.get("corners", [])
                if len(corners) < 3:
                    continue
                elem_width = wall_xz_length(corners)
                if elem_width < min_width:
                    continue
                match = find_parallel_wall(
                    corners, elem_width, story, room_idx, require_other_room=False
                )
                if match:
                    wall, wall_corners, dist_along, angle_deg = match
                    indicators.append(
                        {
                            "story": story,
                            "element_type": elem_type,
                            "element_id": elem["id"],
                            "element_corners": corners,
                            "element_width_m": round(elem_width, 2),
                            "wall_id": wall["id"],
                            "wall_corners": wall_corners,
                            "wall_distance_m": round(dist_along, 2),
                            "angle_deg": round(angle_deg, 1),
                            "confidence": (
                                "high"
                                if angle_deg < 5 and dist_along < 0.3
                                else "medium"
                                if angle_deg < 10 and dist_along < 0.6
                                else "low"
                            ),
                        }
                    )

        storage_wall_proximity = 0.05
        room_wall_ids = {wall["id"] for wall in room["walls_computed"]}
        for storage in room.get("storages", []):
            corners = storage.get("corners", [])
            if len(corners) < 3:
                continue
            elem_width = wall_xz_length(corners)
            if elem_width < min_width:
                continue

            sc = np.array(corners)
            s_cx = float(sc[:, 0].mean())
            s_cz = float(sc[:, 2].mean())
            flush = False
            for wall, _w_ri in story_walls.get(story, []):
                if wall["id"] not in room_wall_ids:
                    continue
                wc = wall["corners"]
                if len(wc) < 2:
                    continue
                w0 = np.array([wc[0][0], wc[0][2]])
                w1 = np.array([wc[1][0], wc[1][2]])
                sp = np.array([s_cx, s_cz])
                edge = w1 - w0
                elen = np.linalg.norm(edge)
                if elen < 1e-6:
                    continue
                t = float(np.dot(sp - w0, edge)) / (elen * elen)
                t = max(0.0, min(1.0, t))
                closest = w0 + t * edge
                dist = float(np.linalg.norm(sp - closest))
                if (
                    dist
                    < storage_wall_proximity
                    + elem_width * 0.5
                    + wall_xz_length(wc) * 0.0
                ):
                    perp_dist = (
                        float(abs(edge[0] * (sp - w0)[1] - edge[1] * (sp - w0)[0]))
                        / elen
                    )
                    if perp_dist < storage_wall_proximity:
                        flush = True
                        break
            if not flush:
                continue

            match = find_parallel_wall(
                corners, elem_width, story, room_idx, require_other_room=True
            )
            if match:
                wall, wall_corners, dist_along, angle_deg = match
                indicators.append(
                    {
                        "story": story,
                        "element_type": "storage",
                        "element_id": storage["id"],
                        "element_corners": corners,
                        "element_width_m": round(elem_width, 2),
                        "wall_id": wall["id"],
                        "wall_corners": wall_corners,
                        "wall_distance_m": round(dist_along, 2),
                        "angle_deg": round(angle_deg, 1),
                        "confidence": (
                            "high"
                            if angle_deg < 5 and dist_along < 0.3
                            else "medium"
                            if angle_deg < 10 and dist_along < 0.6
                            else "low"
                        ),
                    }
                )

    return indicators


def _attach_ontology_gap_ids(closures, graph):
    by_story = defaultdict(list)
    for node in graph.nodes_by_type("Gap"):
        region_wkt = (node.properties or {}).get("region_wkt")
        if not isinstance(region_wkt, str) or not region_wkt:
            continue
        try:
            poly = shapely_wkt.loads(region_wkt)
            if not poly.is_valid:
                poly = make_valid(poly)
        except Exception:
            continue
        if poly.is_empty or poly.area <= 0.0:
            continue
        by_story[node.story].append((node, poly))

    for idx, closure in enumerate(closures):
        corners = closure.get("corners") or []
        if len(corners) < 3:
            continue
        try:
            poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
            if not poly.is_valid:
                poly = make_valid(poly)
        except Exception:
            continue
        if poly.is_empty or poly.area <= 0.0:
            continue
        best_id = None
        best_area = 0.0
        for node, node_poly in by_story.get(closure.get("story"), []):
            try:
                area = float(poly.intersection(node_poly).area)
            except Exception:
                continue
            if area > best_area:
                best_area = area
                best_id = node.id
        if best_id and best_area > 1e-5:
            closure["ontology_gap_id"] = best_id
            closure["id"] = (
                closure.get("id")
                or f"gc:{best_id}:{closure.get('type', 'closure')}:{idx}"
            )


def compute_gap_closures(indicators, rooms_out, graph=None):
    """Create wall/floor/ceiling quads that close the gaps between parallel walls."""
    closures = []

    all_walls_by_id = {}
    element_parent_wall = {}
    for room in rooms_out:
        for wall in room["walls_computed"]:
            all_walls_by_id[wall["id"]] = wall["corners"]
        parent_lookup = room.get("_parent_lookup", {})
        for elem_list in [
            room.get("doors", []),
            room.get("openings", []),
            room.get("storages", []),
        ]:
            for elem in elem_list:
                parent_id = parent_lookup.get(elem["id"])
                if parent_id:
                    element_parent_wall[elem["id"]] = parent_id

    for ind in indicators:
        wc_raw = ind["wall_corners"]
        if len(wc_raw) < 4:
            continue
        wc = np.array(_canonicalize_wall_quad(wc_raw))

        parent_id = element_parent_wall.get(ind["element_id"])
        if parent_id and parent_id in all_walls_by_id:
            pc_raw = all_walls_by_id[parent_id]
        else:
            pc_raw = ind["element_corners"]
        if len(pc_raw) < 4:
            continue
        pc = np.array(_canonicalize_wall_quad(pc_raw))

        wall_dir = wc[1] - wc[0]
        wall_dir_xz = np.array([wall_dir[0], 0.0, wall_dir[2]])
        wall_len = np.linalg.norm(wall_dir_xz)
        if wall_len < 1e-6:
            continue
        axis = wall_dir_xz / wall_len

        origin = np.array([wc[0][0], 0.0, wc[0][2]])
        axis2d = np.array([axis[0], axis[2]])

        def proj_xz(pts, _origin=origin, _axis2d=axis2d):
            return [
                float(
                    np.dot(
                        np.array([p[0], p[2]]) - np.array([_origin[0], _origin[2]]),
                        _axis2d,
                    )
                )
                for p in pts
            ]

        p_proj = proj_xz(pc)
        w_proj = proj_xz(wc)

        p_min, p_max = min(p_proj), max(p_proj)
        w_min, w_max = min(w_proj), max(w_proj)

        ov_min = max(p_min, w_min)
        ov_max = min(p_max, w_max)
        if ov_max - ov_min < 0.05:
            continue

        p0_xz = np.array([pc[0][0], pc[0][2]])
        p1_xz = np.array([pc[1][0], pc[1][2]])
        p_edge = p1_xz - p0_xz
        p_elen = np.linalg.norm(p_edge)

        w0_xz = np.array([wc[0][0], wc[0][2]])
        w1_xz = np.array([wc[1][0], wc[1][2]])
        w_edge = w1_xz - w0_xz
        w_elen = np.linalg.norm(w_edge)

        def edge_t(xz_pt, p0, edge, elen):
            if elen > 1e-6:
                return float(np.clip(np.dot(xz_pt - p0, edge) / (elen**2), 0, 1))
            return 0.0

        origin_xz = (float(origin[0]), float(origin[2]))
        p_top_profile, p_bot_profile = _wall_top_bottom_profiles(
            pc_raw, axis2d, origin_xz
        )
        w_top_profile, w_bot_profile = _wall_top_bottom_profiles(
            wc_raw, axis2d, origin_xz
        )
        if not (p_top_profile and p_bot_profile and w_top_profile and w_bot_profile):
            continue

        side_pts = []
        for t_val in [ov_min, ov_max]:
            xz_pt = np.array([origin[0], origin[2]]) + axis2d * t_val

            p_t = edge_t(xz_pt, p0_xz, p_edge, p_elen)
            p_xz = p0_xz + p_t * p_edge if p_elen > 1e-6 else p0_xz
            p_ybot = _interp_profile_y(p_bot_profile, t_val)
            p_ytop = _interp_profile_y(p_top_profile, t_val)

            w_t = edge_t(xz_pt, w0_xz, w_edge, w_elen)
            w_xz = w0_xz + w_t * w_edge if w_elen > 1e-6 else w0_xz
            w_ybot = _interp_profile_y(w_bot_profile, t_val)
            w_ytop = _interp_profile_y(w_top_profile, t_val)

            side_pts.append((p_xz, p_ybot, p_ytop, w_xz, w_ybot, w_ytop))

            closures.append(
                {
                    "corners": [
                        [float(p_xz[0]), p_ybot, float(p_xz[1])],
                        [float(w_xz[0]), w_ybot, float(w_xz[1])],
                        [float(w_xz[0]), w_ytop, float(w_xz[1])],
                        [float(p_xz[0]), p_ytop, float(p_xz[1])],
                    ],
                    "type": "side",
                    "indicator_element_id": ind["element_id"],
                    "indicator_wall_id": ind["wall_id"],
                    "story": ind["story"],
                }
            )

        left, right = side_pts
        p_xz_l, p_ybot_l, p_ytop_l, w_xz_l, w_ybot_l, w_ytop_l = left
        p_xz_r, p_ybot_r, p_ytop_r, w_xz_r, w_ybot_r, w_ytop_r = right
        closures.append(
            {
                "corners": [
                    [float(p_xz_l[0]), p_ybot_l, float(p_xz_l[1])],
                    [float(p_xz_r[0]), p_ybot_r, float(p_xz_r[1])],
                    [float(w_xz_r[0]), w_ybot_r, float(w_xz_r[1])],
                    [float(w_xz_l[0]), w_ybot_l, float(w_xz_l[1])],
                ],
                "type": "floor",
                "indicator_element_id": ind["element_id"],
                "indicator_wall_id": ind["wall_id"],
                "story": ind["story"],
            }
        )

        closures.append(
            {
                "corners": [
                    [float(p_xz_l[0]), p_ytop_l, float(p_xz_l[1])],
                    [float(p_xz_r[0]), p_ytop_r, float(p_xz_r[1])],
                    [float(w_xz_r[0]), w_ytop_r, float(w_xz_r[1])],
                    [float(w_xz_l[0]), w_ytop_l, float(w_xz_l[1])],
                ],
                "type": "ceiling",
                "indicator_element_id": ind["element_id"],
                "indicator_wall_id": ind["wall_id"],
                "story": ind["story"],
            }
        )

    if graph is not None:
        _attach_ontology_gap_ids(closures, graph)
    return closures
