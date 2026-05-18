"""Cross-floor gap detection and gap wall synthesis."""

import math
from collections import defaultdict

import numpy as np
from shapely import STRtree
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile_v2.decision_logic import classify_gap_decision

from ..gap_ids import stable_gap_wall_id
from .ceilings import build_sloped_ceiling_lookup, sloped_ceiling_y_at
from .lineage import STEP_GAP_WALLS, record
from .overlaps import decompose_polys, floor_polygon_to_shapely


def _ytop_at_xz(xz_pt, snapped, eps=1e-3):
    """Return the per-vertex `ytop` for an arbitrary `(x, z)` point relative to
    the snapped gap polygon. Used when a clip introduces new vertices on a
    room boundary that don't exist in the original `snapped` list.

    If `xz_pt` matches an existing snapped vertex within `eps`, return that
    vertex's `ytop` directly. Otherwise project onto the closest snapped
    polygon edge and linearly interpolate `ytop` along that edge — the new
    vertex always lies on an original gap edge (Shapely's `difference` only
    introduces vertices at intersections with the clip boundary).
    """
    eps2 = eps * eps
    for s in snapped:
        dx = float(xz_pt[0]) - float(s["xz"][0])
        dz = float(xz_pt[1]) - float(s["xz"][1])
        if dx * dx + dz * dz < eps2:
            return float(s["ytop"])
    n = len(snapped)
    best = None
    for i in range(n):
        j = (i + 1) % n
        p0 = snapped[i]["xz"]
        p1 = snapped[j]["xz"]
        ex = float(p1[0]) - float(p0[0])
        ez = float(p1[1]) - float(p0[1])
        L2 = ex * ex + ez * ez
        if L2 < 1e-12:
            continue
        t = (
            (float(xz_pt[0]) - float(p0[0])) * ex
            + (float(xz_pt[1]) - float(p0[1])) * ez
        ) / L2
        t_c = max(0.0, min(1.0, t))
        proj_x = float(p0[0]) + t_c * ex
        proj_z = float(p0[1]) + t_c * ez
        d = (float(xz_pt[0]) - proj_x) ** 2 + (float(xz_pt[1]) - proj_z) ** 2
        if best is None or d < best[0]:
            best = (d, t_c, float(snapped[i]["ytop"]), float(snapped[j]["ytop"]))
    if best is None:
        return float(snapped[0]["ytop"])
    _, t, y0, y1 = best
    return y0 + t * (y1 - y0)


def _edge_on_room_boundary(p0, p1, room_boundary, eps=0.02):
    """True if the midpoint of an edge lies within `eps` of the room-union
    boundary — i.e. the edge was introduced by the clip operation and
    coincides with an existing room wall, so a synthetic gap wall would
    duplicate the room's wall geometry."""
    if room_boundary is None:
        return False
    mx = (float(p0[0]) + float(p1[0])) / 2.0
    mz = (float(p0[1]) + float(p1[1])) / 2.0
    try:
        return Point(mx, mz).distance(room_boundary) < eps
    except Exception:
        return False


def _piece_index(piece_idx, n_pieces, element_idx=None):
    """Encode piece index into a `stable_gap_wall_id` index parameter.

    For single-piece gaps and the first piece of a multi-piece gap, return
    `element_idx` unchanged so existing IDs remain stable. For subsequent
    pieces, offset by `piece_idx * 10000` (or use `piece_idx` for `None`-
    indexed roles like `polygon`) so each piece's elements get unique IDs
    while still hashing under the same gap anchor.
    """
    if n_pieces <= 1 or piece_idx == 0:
        return element_idx
    if element_idx is None:
        return piece_idx
    return piece_idx * 10000 + element_idx


def earclip_2d(coords, eps=1e-3):
    """Ear-clipping triangulation of a simple 2D polygon (no holes).

    `coords` is a list of (x, z) pairs without the closing duplicate. Returns
    a list of (i, j, k) index triplets into the original `coords` array.
    Pre-cleans the input by dropping near-duplicate consecutive vertices and
    collinear interior vertices (within `eps`) — the snapped gap polygon
    often contains both, which would otherwise produce a swarm of zero-area
    sliver triangles. Handles concave polygons by skipping reflex/non-empty
    ears. O(n^3) — fine for n < 100. Falls back to a fan from the first
    surviving vertex if no ear can be found.

    Used for `gap_ceiling` so each emitted triangle carries the actual
    snapped ytops at its three polygon vertices instead of being flattened
    to Newell's best-fit plane in `viewer-modules/geometry.js`
    `createPolygonMesh` (which projects an n-vertex polygon onto a single
    plane and discards the per-vertex height variation).
    """
    n = len(coords)
    if n < 3:
        return []

    # Drop near-duplicate consecutive points (including wrap-around).
    cleaned = []
    src_idx = []
    for i, c in enumerate(coords):
        if cleaned:
            dx = c[0] - cleaned[-1][0]
            dy = c[1] - cleaned[-1][1]
            if dx * dx + dy * dy < eps * eps:
                continue
        cleaned.append(c)
        src_idx.append(i)
    if len(cleaned) >= 2:
        dx = cleaned[0][0] - cleaned[-1][0]
        dy = cleaned[0][1] - cleaned[-1][1]
        if dx * dx + dy * dy < eps * eps:
            cleaned.pop()
            src_idx.pop()

    # Drop collinear interior points iteratively.
    changed = True
    while changed and len(cleaned) > 3:
        changed = False
        m = len(cleaned)
        nxt_c, nxt_i = [], []
        for i in range(m):
            prev = cleaned[(i - 1) % m]
            curr = cleaned[i]
            nxt = cleaned[(i + 1) % m]
            cross = (curr[0] - prev[0]) * (nxt[1] - prev[1]) - (curr[1] - prev[1]) * (
                nxt[0] - prev[0]
            )
            if abs(cross) > eps:
                nxt_c.append(curr)
                nxt_i.append(src_idx[i])
            else:
                changed = True
        if len(nxt_c) < 3:
            break
        cleaned, src_idx = nxt_c, nxt_i

    n = len(cleaned)
    if n < 3:
        return []
    if n == 3:
        return [(src_idx[0], src_idx[1], src_idx[2])]

    area2 = sum(
        cleaned[i][0] * cleaned[(i + 1) % n][1]
        - cleaned[(i + 1) % n][0] * cleaned[i][1]
        for i in range(n)
    )
    if area2 < 0:
        cleaned = list(reversed(cleaned))
        src_idx = list(reversed(src_idx))

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _point_in_tri(p, a, b, c):
        d1 = _cross(p, a, b)
        d2 = _cross(p, b, c)
        d3 = _cross(p, c, a)
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (has_neg and has_pos)

    work = list(range(n))
    triangles = []
    while len(work) > 3:
        m = len(work)
        ear_found = False
        for i in range(m):
            ip, ic, iN = work[(i - 1) % m], work[i], work[(i + 1) % m]
            a, b, c = cleaned[ip], cleaned[ic], cleaned[iN]
            if _cross(a, b, c) <= 0:
                continue
            inside = False
            for j in range(m):
                if j in ((i - 1) % m, i, (i + 1) % m):
                    continue
                if _point_in_tri(cleaned[work[j]], a, b, c):
                    inside = True
                    break
            if inside:
                continue
            triangles.append((src_idx[ip], src_idx[ic], src_idx[iN]))
            work.pop(i)
            ear_found = True
            break
        if not ear_found:
            for k in range(1, len(work) - 1):
                triangles.append(
                    (
                        src_idx[work[0]],
                        src_idx[work[k]],
                        src_idx[work[k + 1]],
                    )
                )
            return triangles
    triangles.append((src_idx[work[0]], src_idx[work[1]], src_idx[work[2]]))
    return triangles


def recommend_gap_actions(graph):
    """Return graph-driven closure recommendations keyed by gap node id.

    This does not replace legacy geometry generation yet. It provides a stable
    decision layer that callers can use in hybrid mode while the exact 3D
    backend matures.
    """
    recommendations = {}
    for gap in graph.nodes_by_type("Gap"):
        decision = classify_gap_decision(graph, gap)
        recommendations[gap.id] = {**decision, "story": gap.story}
    return recommendations


def _normalize_gap_kind(kind):
    if kind == "within_story":
        return "intra_story"
    return kind


def _map_gaps_to_ontology_ids(gaps, graph):
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

    for gap in gaps:
        corners = gap.get("corners") or []
        if len(corners) < 3:
            continue
        try:
            gap_poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
            if not gap_poly.is_valid:
                gap_poly = make_valid(gap_poly)
        except Exception:
            continue
        if gap_poly.is_empty:
            continue
        target_kind = _normalize_gap_kind(gap.get("type"))
        best_node = None
        best_area = 0.0
        for node, node_poly in by_story.get(gap.get("story"), []):
            node_kind = (node.properties or {}).get("gap_kind")
            if node_kind != target_kind:
                continue
            try:
                overlap = float(gap_poly.intersection(node_poly).area)
            except Exception:
                continue
            if overlap > best_area:
                best_area = overlap
                best_node = node
        if best_node is not None and best_area > 1e-5:
            gap["_ontology_gap_id"] = best_node.id


def compute_cross_floor_gaps(rooms_out):
    """Detect gaps between floor polygons: within-story and cross-story."""
    wall_half = 0.25
    pair_half = 0.50
    max_gap = 1.00
    min_area = 0.005
    max_half_floor = 1.50

    story_rooms_raw = defaultdict(list)
    for room in rooms_out:
        story = room["story"]
        poly = floor_polygon_to_shapely(room["floor_polygon"])
        if poly is not None and poly.is_valid and poly.area > 0.01:
            ys = [c[1] for c in room["floor_polygon"]]
            story_rooms_raw[story].append((poly, float(np.mean(ys))))

    story_rooms = defaultdict(list)
    story_floor_ys = defaultdict(list)
    for story, entries in story_rooms_raw.items():
        floor_ys_all = [fy for _, fy in entries]
        median_y = float(np.median(floor_ys_all))
        for poly, fy in entries:
            if abs(fy - median_y) <= max_half_floor:
                story_rooms[story].append(poly)
                story_floor_ys[story].append(fy)

    story_footprints = {}
    story_y_map = {}
    for story, polys in sorted(story_rooms.items()):
        fp = make_valid(unary_union(polys))
        if fp.area > 0.01:
            story_footprints[story] = fp
            story_y_map[story] = float(np.mean(story_floor_ys[story]))

    # A story is a "half-floor" if it sits between two other stories
    # AND both adjacent Δy are within `max_half_floor`. Half-floors
    # have their own walls + slabs, so cross-storey gap lids over their
    # XZ would slice through the existing room envelope. Below we
    # subtract `half_floor_fp` from each full story's `missing` so the
    # gap lid stops at the half-floor's outline. Requiring BOTH
    # neighbours within max_half_floor is what disambiguates the
    # half-floor from the full stories that flank it (each full story
    # only has one close neighbour — the half-floor itself).
    stories_by_y = sorted(story_y_map.keys(), key=lambda s: story_y_map[s])
    half_floor_stories: set[int] = set()
    for i in range(1, len(stories_by_y) - 1):
        story = stories_by_y[i]
        dy_below = abs(story_y_map[story] - story_y_map[stories_by_y[i - 1]])
        dy_above = abs(story_y_map[story] - story_y_map[stories_by_y[i + 1]])
        if dy_below < max_half_floor and dy_above < max_half_floor:
            half_floor_stories.add(story)
    if half_floor_stories:
        half_floor_fp = make_valid(
            unary_union([story_footprints[s] for s in half_floor_stories])
        )
    else:
        half_floor_fp = Polygon()

    gaps = []

    def emit_single_gap(part, story, floor_y, gap_type):
        area = part.area
        compactness = 4 * math.pi * area / (part.length**2) if part.length > 0 else 0
        if compactness < 0.15:
            confidence = "high"
        elif compactness < 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        coords_2d = list(part.exterior.coords)
        corners_3d = [[c[0], floor_y, c[1]] for c in coords_2d]
        centroid = part.centroid
        gaps.append(
            {
                "story": story,
                "type": gap_type,
                "corners": corners_3d,
                "area_m2": round(area, 3),
                "compactness": round(compactness, 3),
                "confidence": confidence,
                "centroid": [
                    round(centroid.x, 3),
                    floor_y,
                    round(centroid.y, 3),
                ],
            }
        )

    def emit_gaps(regions, story, floor_y, gap_type, clip_to=None):
        for region in regions:
            if clip_to is not None:
                try:
                    region = make_valid(region.intersection(clip_to))
                except Exception:
                    continue
            for part in decompose_polys(region):
                if part.area < min_area:
                    continue
                emit_single_gap(part, story, floor_y, gap_type)

    for story, polys in sorted(story_rooms.items()):
        if len(polys) < 2:
            continue

        footprint = story_footprints[story]
        floor_y = story_y_map[story]

        closed = make_valid(
            footprint.buffer(wall_half, join_style=2).buffer(-wall_half, join_style=2)
        )
        morph_gaps = make_valid(closed.difference(footprint))

        hole_gap_parts = []
        for poly_part in decompose_polys(closed):
            for interior in poly_part.interiors:
                hole = Polygon(interior)
                if hole.is_valid and hole.area > min_area:
                    hole_gap_parts.append(hole)

        tree = STRtree(polys)
        buffered = [p.buffer(pair_half, join_style=2) for p in polys]
        pair_gap_parts = []
        seen_pairs = set()

        for i, _poly_i in enumerate(polys):
            candidates = tree.query(buffered[i])
            for j in candidates:
                if j <= i:
                    continue
                pair_key = (i, j)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if polys[i].distance(polys[j]) > max_gap:
                    continue

                try:
                    intersection = buffered[i].intersection(buffered[j])
                    gap = make_valid(intersection.difference(footprint))
                    pair_gap_parts.extend(decompose_polys(gap))
                except Exception:
                    continue

        wide_closed = make_valid(
            footprint.buffer(pair_half, join_style=2).buffer(-pair_half, join_style=2)
        )

        # Phase 1's `close.difference(footprint)` can produce ONE ring-
        # shaped polygon threading every inter-room void when rooms are
        # tightly packed. Three.js ear-clips that ring into building-
        # spanning spike triangles. Cut the morph-gap output into per-
        # pair segments by intersecting with each adjacent pair's
        # buffered neighborhood — each segment is a local simple strip
        # between exactly two rooms. See the legacy function for the
        # full rationale.
        if not morph_gaps.is_empty:
            pair_neighborhoods = []
            for i in range(len(polys)):
                for j in range(i + 1, len(polys)):
                    if polys[i].distance(polys[j]) > max_gap:
                        continue
                    try:
                        nbhd = buffered[i].intersection(buffered[j])
                    except Exception:
                        continue
                    if not nbhd.is_empty:
                        pair_neighborhoods.append(nbhd)
            covered = []
            for nbhd in pair_neighborhoods:
                try:
                    chunk = make_valid(morph_gaps.intersection(nbhd))
                except Exception:
                    continue
                if chunk.is_empty or chunk.area < min_area:
                    continue
                covered.append(chunk)
                emit_gaps([chunk], story, floor_y, "within_story", clip_to=closed)
            if covered:
                try:
                    leftover = make_valid(morph_gaps.difference(unary_union(covered)))
                except Exception:
                    leftover = None
                if leftover is not None and not leftover.is_empty:
                    for part in decompose_polys(leftover):
                        if part.area >= min_area:
                            emit_gaps(
                                [part], story, floor_y, "within_story", clip_to=closed
                            )
            else:
                for part in decompose_polys(morph_gaps):
                    emit_gaps([part], story, floor_y, "within_story", clip_to=closed)

        # Interior hole gaps — enclosed voids under the morph-closed
        # footprint (thick walls, closets). Emit per-part; they're not
        # spanned by pair neighborhoods because each is local.
        for hole in hole_gap_parts:
            emit_gaps([hole], story, floor_y, "within_story")

        # Precompute Phase 1 coverage once so Phase 2 can subtract without
        # recomputing per iteration.
        phase1_parts = list(decompose_polys(morph_gaps)) + list(hole_gap_parts)
        phase1_cover = make_valid(unary_union(phase1_parts)) if phase1_parts else None

        for pair_gap in pair_gap_parts:
            if phase1_cover is not None:
                try:
                    pair_gap = make_valid(pair_gap.difference(phase1_cover))
                except Exception:
                    pass
            if pair_gap.is_empty:
                continue
            emit_gaps([pair_gap], story, floor_y, "within_story", clip_to=wide_closed)

    sorted_stories = sorted(story_footprints.keys())
    if len(sorted_stories) >= 2:
        all_footprints = [story_footprints[s] for s in sorted_stories]
        full_envelope = make_valid(unary_union(all_footprints))

        for story in sorted_stories:
            fp = story_footprints[story]
            floor_y = story_y_map[story]
            try:
                missing = make_valid(full_envelope.difference(fp))
                if not half_floor_fp.is_empty and story not in half_floor_stories:
                    missing = make_valid(missing.difference(half_floor_fp))
            except Exception:
                continue
            if missing.is_empty:
                continue
            other_room_buffs = [
                p.buffer(pair_half, join_style=2)
                for other, polys_other in story_rooms.items()
                if other != story
                for p in polys_other
            ]
            covered = []
            covered_union = None
            for nbhd in other_room_buffs:
                try:
                    chunk = make_valid(missing.intersection(nbhd))
                    if covered_union is not None:
                        chunk = make_valid(chunk.difference(covered_union))
                except Exception:
                    continue
                if chunk.is_empty or chunk.area < min_area:
                    continue
                covered.append(chunk)
                covered_union = (
                    chunk
                    if covered_union is None
                    else make_valid(unary_union([covered_union, chunk]))
                )
                emit_gaps([chunk], story, floor_y, "cross_story")
            if covered:
                try:
                    leftover = make_valid(missing.difference(covered_union))
                except Exception:
                    leftover = None
                if leftover is not None and not leftover.is_empty:
                    for part in decompose_polys(leftover):
                        if part.area >= min_area:
                            emit_gaps([part], story, floor_y, "cross_story")
            else:
                emit_gaps(decompose_polys(missing), story, floor_y, "cross_story")

    return gaps


def assign_gaps_to_rooms(gaps, rooms_out, graph=None):
    """Assign closeable within-story gaps to nearest room and expand floor polygons.

    When a topology graph is provided, only gaps with graph decision `close`
    are merged into room floors. This prevents intentional openings from being
    indiscriminately sealed while ensuring enclosed inferred voids are filled.
    """
    room_shapely = []
    for ri, room in enumerate(rooms_out):
        poly = floor_polygon_to_shapely(room.get("floor_polygon", []))
        room_shapely.append((ri, poly))

    if graph is not None:
        _map_gaps_to_ontology_ids(gaps, graph)
        gap_decisions = {
            node.id: classify_gap_decision(graph, node)
            for node in graph.nodes_by_type("Gap")
        }
    else:
        gap_decisions = {}

    for gap in gaps:
        if gap.get("type") != "within_story":
            continue
        ontology_gap_id = gap.get("_ontology_gap_id")
        if ontology_gap_id and graph is not None:
            decision = gap_decisions.get(ontology_gap_id, {})
            if decision.get("action") != "close":
                continue

        corners_3d = gap.get("corners") or []
        if len(corners_3d) < 3:
            continue

        gap_poly = floor_polygon_to_shapely(corners_3d)
        if gap_poly is None:
            continue

        story = gap.get("story")
        gap_centroid = gap_poly.centroid

        best_ri = None
        best_dist = float("inf")
        for ri, rpoly in room_shapely:
            if rpoly is None or rooms_out[ri].get("story") != story:
                continue
            d = rpoly.distance(gap_centroid)
            if d < best_dist:
                best_dist = d
                best_ri = ri

        if best_ri is None:
            continue

        gap["room_index"] = best_ri

        assigned_room = rooms_out[best_ri]
        wall_top_ys = []
        for w in assigned_room.get("walls_computed") or assigned_room.get(
            "walls_merged", []
        ):
            cs = w.get("corners", [])
            if cs:
                wall_top_ys.append(max(c[1] for c in cs))
        if wall_top_ys:
            ceiling_y = round(float(np.median(wall_top_ys)), 4)
        else:
            ceiling_y = corners_3d[0][1]
        gap["ceiling_corners"] = [
            [round(c[0], 4), ceiling_y, round(c[2], 4)] for c in corners_3d
        ]

        room = assigned_room
        room_poly = room_shapely[best_ri][1]
        if room_poly is None:
            continue

        floor_y = (
            room["floor_polygon"][0][1]
            if room.get("floor_polygon")
            else corners_3d[0][1]
        )
        merged = make_valid(unary_union([room_poly, gap_poly]))
        if getattr(merged, "geom_type", "") == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        if merged.is_empty:
            continue

        coords_2d = list(merged.exterior.coords)
        if coords_2d and coords_2d[0] == coords_2d[-1]:
            coords_2d = coords_2d[:-1]
        room["floor_polygon"] = [
            [round(c[0], 4), floor_y, round(c[1], 4)] for c in coords_2d
        ]
        room_shapely[best_ri] = (best_ri, merged)


def compute_gap_walls(
    gaps,
    rooms_out,
    story_y_map,
    gap_closures=None,
    graph=None,
    pre_absorption_floor_polygons=None,
):
    """Create wall quads along each edge of cross-floor gap polygons.

    `pre_absorption_floor_polygons`, when provided, is a list index-aligned
    with ``rooms_out`` giving each room's floor polygon **before**
    ``assign_gaps_to_rooms`` merged within-story gaps into it. Using the
    pre-absorption boundary here matters: a within-story gap sits in the
    wall-thickness strip between two rooms, and after absorption the merged
    boundary swallows one side of the gap — causing the post-absorption
    room-union clip to drop the entire gap (walls, floor, ceiling) for the
    ~90% of gaps that get absorbed. The pre-absorption boundary keeps the
    scan-derived wall planes as the reference, so gap edges that coincide
    with real walls are skipped (no duplicate wall) while the "free" end
    edges of the gap still emit synthetic quads.
    """
    default_wall_height = 2.50
    min_wall_height = 0.5
    max_snap_dist = 1.0
    max_y_dist = 0.75
    min_snap_dist = 1e-6
    gap_policies = {}
    if graph is not None:
        _map_gaps_to_ontology_ids(gaps, graph)
        recommendations = recommend_gap_actions(graph)
        for gap in graph.nodes_by_type("Gap"):
            decision = recommendations.get(gap.id)
            if decision is None:
                continue
            gap_policies.setdefault(
                (
                    gap.story,
                    _normalize_gap_kind((gap.properties or {}).get("gap_kind")),
                ),
                [],
            ).append((gap.id, decision))

    def add_wall_edge(story, wc, room_index=None, wall_id=None):
        if len(wc) < 4:
            return
        ys = [c[1] for c in wc]
        h = max(ys) - min(ys)
        if h < min_wall_height:
            return
        p0_xz = np.array([wc[0][0], wc[0][2]])
        p1_xz = np.array([wc[1][0], wc[1][2]])
        edge = p1_xz - p0_xz
        elen = np.linalg.norm(edge)
        if elen < 1e-6:
            return
        ybot_avg = (wc[0][1] + wc[1][1]) / 2
        mid_y = (max(ys) + min(ys)) / 2.0
        top_cs = [c for c in wc if c[1] > mid_y - 0.01]
        if len(top_cs) < 2:
            top_cs = [wc[3], wc[2]]
        top_profile = []
        for c in top_cs:
            cxz = np.array([c[0], c[2]])
            t = float(np.clip(np.dot(cxz - p0_xz, edge) / (elen**2), 0, 1))
            top_profile.append((t, c[1]))
        top_profile.sort(key=lambda p: p[0])
        edge_unit = edge / elen
        story_walls[story].append(
            {
                "corners": wc,
                "p0_xz": p0_xz,
                "edge": edge,
                "edge_unit": edge_unit,
                "elen": elen,
                "ybot_avg": ybot_avg,
                "top_profile": top_profile,
                "room_index": room_index,
                "wall_id": wall_id,
            }
        )

    story_walls = defaultdict(list)
    for room in rooms_out:
        story = room["story"]
        for wall in room["walls_computed"]:
            wc = wall["corners"]
            if len(wc) < 4:
                continue
            ext = wall.get("extension_strip")
            if ext and len(ext) >= 1:
                ext_top_y = max(c[1] for quad in ext for c in quad)
                ec = [list(c) for c in wc]
                ys = [c[1] for c in wc]
                mid_y = (max(ys) + min(ys)) / 2.0
                for i, c in enumerate(ec):
                    if c[1] > mid_y - 0.01:
                        ec[i] = [c[0], ext_top_y, c[2]]
            else:
                ec = wc
            add_wall_edge(
                story, ec, room_index=room.get("room_index"), wall_id=wall.get("id")
            )

    for gc in gap_closures or []:
        if gc.get("type") == "side" and len(gc.get("corners", [])) >= 4:
            add_wall_edge(gc["story"], gc["corners"])

    sorted_stories = sorted(story_y_map.keys())
    ceiling_y_map = {}
    for i, story in enumerate(sorted_stories):
        if i + 1 < len(sorted_stories):
            ceiling_y_map[story] = story_y_map[sorted_stories[i + 1]]
        else:
            heights = []
            for wall in story_walls.get(story, []):
                ys = [c[1] for c in wall["corners"]]
                heights.append(max(ys) - min(ys))
            median_h = float(np.median(heights)) if heights else default_wall_height
            ceiling_y_map[story] = story_y_map[story] + median_h

    sloped_ceilings = build_sloped_ceiling_lookup(rooms_out)

    def clamp_to_sloped_ceiling(xz_pt, story, floor_y, ytop):
        slant_y = sloped_ceiling_y_at(sloped_ceilings, xz_pt, story)
        if slant_y is None or slant_y >= ytop:
            return ytop
        return max(slant_y, floor_y + min_wall_height)

    def interp_top_profile(top_profile, t):
        if len(top_profile) == 1:
            return top_profile[0][1]
        if t <= top_profile[0][0]:
            return top_profile[0][1]
        if t >= top_profile[-1][0]:
            return top_profile[-1][1]
        for k in range(len(top_profile) - 1):
            t0, y0 = top_profile[k]
            t1, y1 = top_profile[k + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return y0 + frac * (y1 - y0)
        return top_profile[-1][1]

    def project_to_wall_line(xz_pt, wall):
        rel = xz_pt - wall["p0_xz"]
        t_raw = float(np.dot(rel, wall["edge_unit"]))
        proj = wall["p0_xz"] + t_raw * wall["edge_unit"]
        dist = float(np.linalg.norm(xz_pt - proj))
        t_profile = float(np.clip(t_raw / wall["elen"], 0.0, 1.0))
        return proj, dist, t_profile

    def snap_vertex_y(xz_pt, story, floor_y):
        """Snap to the closest wall — used for wall quad heights."""
        fallback_top = ceiling_y_map.get(story, floor_y + default_wall_height)
        best_dist = max_snap_dist
        best_ybot = floor_y
        best_ytop = fallback_top

        for wall in story_walls.get(story, []):
            if abs(wall["ybot_avg"] - floor_y) > max_y_dist:
                continue
            _proj, dist, t = project_to_wall_line(xz_pt, wall)
            if dist < best_dist:
                best_dist = dist
                best_ybot = float(wall["ybot_avg"])
                best_ytop = interp_top_profile(wall["top_profile"], t)

        best_ytop = clamp_to_sloped_ceiling(xz_pt, story, floor_y, best_ytop)
        return best_ybot, best_ytop

    def pick_support_wall(xz_pt, prev_xz, next_xz, story, floor_y):
        tangent = None
        edge_ref = next_xz - prev_xz
        edge_len = float(np.linalg.norm(edge_ref))
        if edge_len > min_snap_dist:
            tangent = edge_ref / edge_len
        best = None
        for wall in story_walls.get(story, []):
            if abs(wall["ybot_avg"] - floor_y) > max_y_dist:
                continue
            proj, dist, t_profile = project_to_wall_line(xz_pt, wall)
            if dist > max_snap_dist:
                continue
            cos_parallel = 1.0
            if tangent is not None:
                cos_parallel = abs(float(np.dot(tangent, wall["edge_unit"])))
            # Prioritize near-coplanar fit; orientation is a secondary signal.
            score = dist + 0.35 * (1.0 - cos_parallel)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "wall": wall,
                    "proj": proj,
                    "dist": dist,
                    "t_profile": t_profile,
                }
        return best

    def build_snapped_vertices(edge_verts, story, floor_y):
        snapped = []
        n_edges = len(edge_verts)
        for i, c in enumerate(edge_verts):
            prev_c = edge_verts[(i - 1) % n_edges]
            next_c = edge_verts[(i + 1) % n_edges]
            xz = np.array([c[0], c[2]], dtype=float)
            prev_xz = np.array([prev_c[0], prev_c[2]], dtype=float)
            next_xz = np.array([next_c[0], next_c[2]], dtype=float)
            picked = pick_support_wall(xz, prev_xz, next_xz, story, floor_y)
            if picked is None:
                ybot, ytop = snap_vertex_y(xz, story, floor_y)
                snapped.append(
                    {
                        "xz": xz,
                        "ybot": ybot,
                        "ytop": ytop,
                        "wall": None,
                    }
                )
                continue
            ybot = float(picked["wall"]["ybot_avg"])
            ytop = interp_top_profile(
                picked["wall"]["top_profile"], picked["t_profile"]
            )
            ytop = clamp_to_sloped_ceiling(picked["proj"], story, floor_y, ytop)
            snapped.append(
                {
                    "xz": picked["proj"],
                    "ybot": ybot,
                    "ytop": ytop,
                    "wall": picked["wall"],
                }
            )

        # Degeneracy guard: if snapping collapses polygon, fallback to original.
        coords = [(s["xz"][0], s["xz"][1]) for s in snapped]
        try:
            poly = Polygon(coords)
            if (not poly.is_valid) or poly.area <= 1e-6:
                raise ValueError("invalid snapped polygon")
        except Exception:
            snapped = []
            for c in edge_verts:
                xz = np.array([c[0], c[2]], dtype=float)
                ybot, ytop = snap_vertex_y(xz, story, floor_y)
                snapped.append(
                    {
                        "xz": xz,
                        "ybot": ybot,
                        "ytop": ytop,
                        "wall": None,
                    }
                )
        return snapped

    # Per-story union of room floor polygons + their boundary, used below to
    # drop synthetic floor/ceiling caps whose area already lies under a room
    # floor, and to skip side-wall quads whose edges coincide with a real
    # room wall. We prefer the pre-absorption polygons when the caller
    # supplies them: `assign_gaps_to_rooms` grows each room to swallow its
    # assigned within-story gap, and using the post-absorption boundary here
    # collapses most gaps to zero area (fix for 90% of within-story gaps
    # being emitted without wall geometry — see the docstring).
    story_room_union = {}
    story_room_boundary = {}
    rooms_by_story = defaultdict(list)
    for idx, room in enumerate(rooms_out):
        s = room.get("story")
        if s is None:
            continue
        if pre_absorption_floor_polygons is not None and idx < len(
            pre_absorption_floor_polygons
        ):
            fp_source = pre_absorption_floor_polygons[idx]
        else:
            fp_source = room.get("floor_polygon")
        rp = floor_polygon_to_shapely(fp_source)
        if rp is None or not rp.is_valid or rp.area <= 0.0:
            continue
        rooms_by_story[s].append(rp)
    for s, polys in rooms_by_story.items():
        try:
            u = make_valid(unary_union(polys))
        except Exception:
            u = None
        story_room_union[s] = u
        try:
            story_room_boundary[s] = (
                u.boundary if (u is not None and not u.is_empty) else None
            )
        except Exception:
            story_room_boundary[s] = None

    walls = []
    for gap in gaps:
        gap_type = gap["type"]
        if gap_type == "cross_story":
            corners_3d = gap.get("corners") or []
            if len(corners_3d) < 3:
                continue
            story = gap["story"]
            below = story - 1
            if below not in story_y_map:
                continue
            closed = len(corners_3d) >= 4 and corners_3d[0] == corners_3d[-1]
            edge_verts = corners_3d[:-1] if closed else corners_3d
            if len(edge_verts) < 3:
                continue
            snapped_below = build_snapped_vertices(
                edge_verts, below, story_y_map[below]
            )
            draped = [
                [float(s["xz"][0]), float(s["ytop"]), float(s["xz"][1])]
                for s in snapped_below
            ]
            new_corners = [list(p) for p in draped]
            if closed:
                new_corners.append(list(draped[0]))
            gap["corners"] = new_corners
            gap["ceiling_corners"] = [list(p) for p in draped]
            centroid = gap.get("centroid")
            if isinstance(centroid, list) and len(centroid) == 3:
                centroid[1] = float(np.mean([p[1] for p in draped]))
            continue
        if gap_type != "within_story":
            continue
        if graph is not None:
            gap_kind = _normalize_gap_kind(gap["type"])
            decisions = gap_policies.get((gap.get("story"), gap_kind), [])
            if decisions and not any(
                decision.get("action") == "close" for _, decision in decisions
            ):
                continue
        corners_3d = gap["corners"]
        if len(corners_3d) < 3:
            continue
        story = gap["story"]
        floor_y = corners_3d[0][1]

        vertex_ys = []
        n = len(corners_3d)
        if n >= 4 and corners_3d[0] == corners_3d[-1]:
            edge_verts = corners_3d[:-1]
        else:
            edge_verts = corners_3d
        n_edges = len(edge_verts)
        if n_edges < 3:
            continue

        snapped = build_snapped_vertices(edge_verts, story, floor_y)
        for sv in snapped:
            vertex_ys.append((sv["ybot"], sv["ytop"]))

        gap_floor_y = max(yb for yb, _ in vertex_ys)
        for c in corners_3d:
            c[1] = gap_floor_y
        gap["centroid"][1] = gap_floor_y

        # Clip the snapped polygon by the room-union for this story, so any
        # part of the gap that ended up inside a room (after vertices were
        # snapped to wall planes) is removed from the emitted geometry. This
        # gives provably 100% coverage with zero overlap: room floors cover
        # the room interiors, gap pieces cover only what's outside rooms.
        snapped_xz_2d = [(float(s["xz"][0]), float(s["xz"][1])) for s in snapped]
        try:
            snapped_poly = Polygon(snapped_xz_2d)
            if not snapped_poly.is_valid:
                snapped_poly = make_valid(snapped_poly)
        except Exception:
            snapped_poly = None

        room_union = story_room_union.get(story)
        room_boundary = story_room_boundary.get(story)
        pieces_for_caps = [snapped]
        caps_were_clipped = False
        if (
            room_union is not None
            and not room_union.is_empty
            and snapped_poly is not None
            and not snapped_poly.is_empty
        ):
            try:
                clipped = make_valid(snapped_poly.difference(room_union))
            except Exception:
                clipped = None
            if clipped is None or clipped.is_empty:
                pieces_for_caps = []
                caps_were_clipped = True
            elif abs(snapped_poly.area - clipped.area) > 1e-6:
                all_pieces = sorted(
                    (p for p in decompose_polys(clipped) if p.area > 0.01),
                    key=lambda p: -p.area,
                )
                pieces_for_caps = []
                for piece in all_pieces:
                    piece_coords = list(piece.exterior.coords)
                    if piece_coords and piece_coords[0] == piece_coords[-1]:
                        piece_coords = piece_coords[:-1]
                    if len(piece_coords) < 3:
                        continue
                    ps = []
                    for x, z in piece_coords:
                        ytop = _ytop_at_xz((x, z), snapped)
                        ps.append(
                            {
                                "xz": np.array([x, z], dtype=float),
                                "ybot": gap_floor_y,
                                "ytop": ytop,
                                "wall": None,
                            }
                        )
                    pieces_for_caps.append(ps)
                caps_were_clipped = True

        # Side walls always trace the original snapped polygon so the gap's
        # end edges emit even when the snapped footprint lies entirely inside
        # the room union (typical for within-story gaps that `assign_gaps_to_rooms`
        # absorbed into a room floor). Edges that coincide with a real wall —
        # detected via `room_boundary` derived from pre-absorption room
        # polygons — are skipped so we don't stack a synthetic quad on top of
        # an existing `walls_computed` wall.
        n_edges_sides = len(snapped)
        for ei in range(n_edges_sides):
            j = (ei + 1) % n_edges_sides
            c0 = snapped[ei]["xz"]
            c1 = snapped[j]["xz"]
            if _edge_on_room_boundary(c0, c1, room_boundary):
                continue
            ytop0 = float(snapped[ei]["ytop"])
            ytop1 = float(snapped[j]["ytop"])
            entry = {
                "id": stable_gap_wall_id(gap, gap["type"], "edge", ei),
                "corners": [
                    [float(c0[0]), gap_floor_y, float(c0[1])],
                    [float(c1[0]), gap_floor_y, float(c1[1])],
                    [float(c1[0]), ytop1, float(c1[1])],
                    [float(c0[0]), ytop0, float(c0[1])],
                ],
                "type": gap["type"],
                "story": story,
                "confidence": gap["confidence"],
                "ontology_gap_id": gap.get("_ontology_gap_id"),
            }
            record(
                entry,
                STEP_GAP_WALLS,
                "created",
                f"type={gap['type']}, confidence={gap['confidence']}",
            )
            walls.append(entry)

        n_pieces = len(pieces_for_caps)
        for piece_idx, piece_snapped in enumerate(pieces_for_caps):
            n_edges_p = len(piece_snapped)

            polygon_floor = {
                "id": stable_gap_wall_id(
                    gap,
                    "gap_floor",
                    "polygon",
                    _piece_index(piece_idx, n_pieces, None),
                ),
                "corners": [
                    [float(s["xz"][0]), gap_floor_y, float(s["xz"][1])]
                    for s in piece_snapped
                ],
                "type": "gap_floor",
                "story": story,
                "confidence": gap["confidence"],
                "ontology_gap_id": gap.get("_ontology_gap_id"),
            }
            record(
                polygon_floor,
                STEP_GAP_WALLS,
                "created",
                f"type=gap_floor_polygon, confidence={gap['confidence']}",
            )
            walls.append(polygon_floor)

            xz_2d_p = [(float(s["xz"][0]), float(s["xz"][1])) for s in piece_snapped]
            ti = 0
            for ia, ib, ic in earclip_2d(xz_2d_p):
                sa, sb, sc = piece_snapped[ia], piece_snapped[ib], piece_snapped[ic]
                x0, z0 = sa["xz"]
                x1, z1 = sb["xz"]
                x2, z2 = sc["xz"]
                area_xz = abs((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) * 0.5
                if area_xz < 1e-5:
                    continue
                ceiling_tri = {
                    "id": stable_gap_wall_id(
                        gap,
                        "gap_ceiling",
                        "tri",
                        _piece_index(piece_idx, n_pieces, ti),
                    ),
                    "corners": [
                        [float(sa["xz"][0]), float(sa["ytop"]), float(sa["xz"][1])],
                        [float(sb["xz"][0]), float(sb["ytop"]), float(sb["xz"][1])],
                        [float(sc["xz"][0]), float(sc["ytop"]), float(sc["xz"][1])],
                    ],
                    "type": "gap_ceiling",
                    "story": story,
                    "confidence": gap["confidence"],
                    "ontology_gap_id": gap.get("_ontology_gap_id"),
                }
                record(
                    ceiling_tri,
                    STEP_GAP_WALLS,
                    "created",
                    f"type=gap_ceiling_tri, confidence={gap['confidence']}",
                )
                walls.append(ceiling_tri)
                ti += 1

            # Short-edge caps extend slightly past the perimeter; skip them
            # when the piece came from a clip so the extension doesn't push
            # into an adjacent room and defeat the no-overlap guarantee.
            if caps_were_clipped:
                continue

            short_edge_threshold = 0.25  # metres — wall-thickness edges
            max_cap_reach = 0.30  # don't extend further along run edges

            def _clamp_towards(origin, target, max_dist):
                ddx = target[0] - origin[0]
                ddz = target[1] - origin[1]
                d = math.sqrt(ddx * ddx + ddz * ddz)
                if d <= max_dist or d < 1e-6:
                    return target
                ratio = max_dist / d
                return np.array(
                    [origin[0] + ddx * ratio, origin[1] + ddz * ratio],
                    dtype=float,
                )

            for ei in range(n_edges_p):
                j = (ei + 1) % n_edges_p
                c0 = piece_snapped[ei]["xz"]
                c1 = piece_snapped[j]["xz"]
                dx = c1[0] - c0[0]
                dz = c1[1] - c0[1]
                edge_len = math.sqrt(dx * dx + dz * dz)
                if edge_len > short_edge_threshold:
                    continue
                ip = (ei - 1) % n_edges_p
                jn = (j + 1) % n_edges_p
                cp = piece_snapped[ip]["xz"]
                cn = piece_snapped[jn]["xz"]
                cp_clamped = _clamp_towards(c0, cp, max_cap_reach)
                cn_clamped = _clamp_towards(c1, cn, max_cap_reach)

                floor_quad = {
                    "id": stable_gap_wall_id(gap, "gap_floor", "cap", ei),
                    "corners": [
                        [float(cp_clamped[0]), gap_floor_y, float(cp_clamped[1])],
                        [float(c0[0]), gap_floor_y, float(c0[1])],
                        [float(c1[0]), gap_floor_y, float(c1[1])],
                        [float(cn_clamped[0]), gap_floor_y, float(cn_clamped[1])],
                    ],
                    "type": "gap_floor",
                    "story": story,
                    "confidence": gap["confidence"],
                    "ontology_gap_id": gap.get("_ontology_gap_id"),
                }
                record(
                    floor_quad,
                    STEP_GAP_WALLS,
                    "created",
                    f"type=gap_floor, confidence={gap['confidence']}",
                )
                walls.append(floor_quad)

                quad_verts = [cp_clamped, c0, c1, cn_clamped]
                ceil_ys = [
                    snap_vertex_y(np.array([v[0], v[1]]), story, gap_floor_y)[1]
                    for v in quad_verts
                ]
                ceiling_quad = {
                    "id": stable_gap_wall_id(gap, "gap_ceiling", "cap", ei),
                    "corners": [
                        [float(quad_verts[k][0]), ceil_ys[k], float(quad_verts[k][1])]
                        for k in range(4)
                    ],
                    "type": "gap_ceiling",
                    "story": story,
                    "confidence": gap["confidence"],
                    "ontology_gap_id": gap.get("_ontology_gap_id"),
                }
                record(
                    ceiling_quad,
                    STEP_GAP_WALLS,
                    "created",
                    f"type=gap_ceiling, confidence={gap['confidence']}",
                )
                walls.append(ceiling_quad)

    return walls
