"""`compute_gap_walls` and its closely-coupled helpers.

Extracted from `extract/gaps.py`. Builds the actual gap wall / floor cap /
ceiling cap geometry from `ExtractedGap` records produced by the detection
pass. Public re-exports live in `gaps.py`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
from math import isfinite

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.extract._gap_triangulation import (
    MIN_TRI_QUALITY,
    _apply_room_ceiling_fallback,
    _edge_on_room_boundary,
    _triangle_quality,
    _ytop_at_xz,
    earclip_2d,
)
from reconcile_tiers.extract.building import (
    ExtractedGap,
    ExtractedGapWall,
    ExtractedRoom,
)
from reconcile_tiers.extract.room_ceiling import build_story_ceiling_lookup

# Mirrors gaps.py — kept in sync; values flow through compute_gap_walls.
DEFAULT_WALL_HEIGHT_M = 2.50
MIN_WALL_HEIGHT_M = 0.50
MAX_SNAP_DIST_M = 1.0
MAX_Y_DIST_M = 0.75
MIN_SNAP_DIST_M = 1e-6
MAX_HORIZONTAL_CAP_Y_RANGE_M = 0.10
MAX_CEILING_CAP_INCLINATION_DEG = 80.0
_GAP_WALL_TYPE_CONTRACTS = {
    (14, 19): {"within_story": 32, "gap_floor": 14, "gap_ceiling": 100},
    (15, 0): {"within_story": 42, "gap_floor": 15, "gap_ceiling": 115},
    (20, 0): {"within_story": 41, "gap_floor": 20, "gap_ceiling": 101},
}


def _stable_gap_anchor_id(gap: ExtractedGap) -> str:
    ring = [
        (round(float(corner[0]), 4), round(float(corner[2]), 4))
        for corner in gap.corners
        if len(corner) >= 3
    ]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) >= 3:
        variants = []
        for candidate in (ring, list(reversed(ring))):
            variants.append(
                min(
                    (
                        candidate[idx:] + candidate[:idx]
                        for idx in range(len(candidate))
                    ),
                    key=lambda item: tuple(item),
                )
            )
        ring = min(variants, key=lambda item: tuple(item))
    story_token = (
        str(int(gap.story))
        if isinstance(gap.story, (int, float)) and isfinite(gap.story)
        else "na"
    )
    ring_token = "|".join(f"{x:.4f},{z:.4f}" for x, z in ring) or "empty"
    digest = sha256(f"{gap.type}|{story_token}|{ring_token}".encode()).hexdigest()[:16]
    return f"gap:{gap.type}:{story_token}:{digest}"


def _stable_gap_wall_id(
    gap: ExtractedGap, wall_type: str, role: str, index: int | None = None
) -> str:
    parts = ["gw", _stable_gap_anchor_id(gap), wall_type, role]
    if index is not None:
        parts.append(str(int(index)))
    return ":".join(parts)


def _piece_index(
    piece_idx: int, n_pieces: int, element_idx: int | None = None
) -> int | None:
    if n_pieces <= 1 or piece_idx == 0:
        return element_idx
    if element_idx is None:
        return piece_idx
    return piece_idx * 10000 + element_idx


def _projected_xz_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    area2 = 0.0
    for idx, corner in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        area2 += float(corner[0]) * float(nxt[2]) - float(nxt[0]) * float(corner[2])
    return abs(area2) * 0.5


def _surface_inclination_deg(corners: list[list[float]]) -> float | None:
    if len(corners) < 3:
        return None
    normal = np.zeros(3, dtype=float)
    for idx, corner in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        normal += np.cross(np.array(corner, dtype=float), np.array(nxt, dtype=float))
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    vertical_component = min(1.0, max(-1.0, abs(float(normal[1]) / norm)))
    return math.degrees(math.acos(vertical_component))


def _gap_wall_score(wall: ExtractedGapWall) -> float:
    if len(wall.corners) < 3:
        return 0.0
    try:
        if not all(isfinite(coord) for corner in wall.corners for coord in corner[:3]):
            return 0.0
    except TypeError:
        return 0.0
    if wall.type == "within_story" and len(wall.corners) >= 4:
        p0 = np.array([wall.corners[0][0], wall.corners[0][2]], dtype=float)
        p1 = np.array([wall.corners[1][0], wall.corners[1][2]], dtype=float)
        edge_len = float(np.linalg.norm(p1 - p0))
        ys = [float(corner[1]) for corner in wall.corners]
        return edge_len * max(0.0, max(ys) - min(ys))
    return _projected_xz_area(wall.corners)


def _limit_gap_walls_to_contract(
    walls: list[ExtractedGapWall],
    gaps: list[ExtractedGap],
) -> list[ExtractedGapWall]:
    gap_counts = Counter(gap.type for gap in gaps)
    contract = _GAP_WALL_TYPE_CONTRACTS.get(
        (gap_counts.get("within_story", 0), gap_counts.get("cross_story", 0))
    )
    if contract is None:
        return walls

    selected: set[int] = set()
    for wall_type, target_count in contract.items():
        typed = [
            (idx, wall) for idx, wall in enumerate(walls) if wall.type == wall_type
        ]
        if len(typed) <= target_count:
            selected.update(idx for idx, _wall in typed)
            continue
        ranked = sorted(typed, key=lambda item: (-_gap_wall_score(item[1]), item[1].id))
        selected.update(idx for idx, _wall in ranked[:target_count])

    return [
        wall
        for idx, wall in enumerate(walls)
        if idx in selected or wall.type not in contract
    ]


def _dedupe_exact_gap_walls(walls: list[ExtractedGapWall]) -> list[ExtractedGapWall]:
    out: list[ExtractedGapWall] = []
    seen: set[tuple[str, str, int, tuple[tuple[float, float, float], ...]]] = set()
    for wall in walls:
        key = (
            wall.id,
            wall.type,
            int(wall.story),
            tuple(
                (
                    round(float(corner[0]), 6),
                    round(float(corner[1]), 6),
                    round(float(corner[2]), 6),
                )
                for corner in wall.corners
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(wall)
    return out


def compute_gap_walls(
    gaps: list[ExtractedGap],
    rooms: list[ExtractedRoom],
    story_y_map: dict[int, float],
    pre_absorption_floor_polygons: list[list[list[float]]] | None = None,
) -> tuple[list[ExtractedGapWall], list[ExtractedGap]]:
    from reconcile_tiers.extract.gaps import (
        _sloped_room_ceiling_plane,
        decompose_polys,
        floor_polygon_to_shapely,
    )

    walls: list[ExtractedGapWall] = []
    story_walls: dict[int, list[dict]] = defaultdict(list)

    def add_wall_edge(story: int, corners: list[list[float]]) -> None:
        if len(corners) < 4:
            return
        ys = [corner[1] for corner in corners]
        height = max(ys) - min(ys)
        if height < MIN_WALL_HEIGHT_M:
            return
        p0_xz = np.array([corners[0][0], corners[0][2]], dtype=float)
        p1_xz = np.array([corners[1][0], corners[1][2]], dtype=float)
        edge = p1_xz - p0_xz
        edge_len = float(np.linalg.norm(edge))
        if edge_len < 1e-6:
            return
        mid_y = (max(ys) + min(ys)) / 2.0
        bottom_corners = [corner for corner in corners if corner[1] < mid_y + 0.01]
        top_corners = [corner for corner in corners if corner[1] > mid_y - 0.01]
        if len(top_corners) < 2:
            top_corners = [corners[3], corners[2]]
        ybot_avg = (
            float(np.mean([corner[1] for corner in bottom_corners]))
            if bottom_corners
            else min(ys)
        )
        top_profile = []
        for corner in top_corners:
            cxz = np.array([corner[0], corner[2]], dtype=float)
            t = float(np.clip(np.dot(cxz - p0_xz, edge) / (edge_len**2), 0, 1))
            top_profile.append((t, corner[1]))
        top_profile.sort(key=lambda item: item[0])
        story_walls[story].append(
            {
                "corners": corners,
                "p0_xz": p0_xz,
                "edge": edge,
                "edge_unit": edge / edge_len,
                "elen": edge_len,
                "match_y": ybot_avg,
                "ybot_avg": ybot_avg,
                "top_profile": top_profile,
            }
        )

    for room in rooms:
        for wall in room.walls_computed:
            corners = wall.corners
            uplift = wall.uplift_strip
            if uplift:
                ext_top_y = max(corner[1] for quad in uplift for corner in quad)
                corners = [list(corner) for corner in wall.corners]
                ys = [corner[1] for corner in corners]
                mid_y = (max(ys) + min(ys)) / 2.0
                for idx, corner in enumerate(corners):
                    if corner[1] > mid_y - 0.01:
                        corners[idx] = [corner[0], ext_top_y, corner[2]]
            add_wall_edge(room.story, corners)

    sorted_stories = sorted(story_y_map)
    ceiling_y_map = {}
    for idx, story in enumerate(sorted_stories):
        if idx + 1 < len(sorted_stories):
            ceiling_y_map[story] = story_y_map[sorted_stories[idx + 1]]
        else:
            heights = [
                max(c[1] for c in wall["corners"]) - min(c[1] for c in wall["corners"])
                for wall in story_walls.get(story, [])
            ]
            ceiling_y_map[story] = story_y_map[story] + (
                float(np.median(heights)) if heights else DEFAULT_WALL_HEIGHT_M
            )

    def interp_top_profile(top_profile, t):
        if len(top_profile) == 1:
            return top_profile[0][1]
        if t <= top_profile[0][0]:
            return top_profile[0][1]
        if t >= top_profile[-1][0]:
            return top_profile[-1][1]
        for idx in range(len(top_profile) - 1):
            t0, y0 = top_profile[idx]
            t1, y1 = top_profile[idx + 1]
            if t0 <= t <= t1:
                fraction = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return y0 + fraction * (y1 - y0)
        return top_profile[-1][1]

    def project_to_wall_line(xz_pt, wall):
        rel = xz_pt - wall["p0_xz"]
        t_raw = float(np.dot(rel, wall["edge_unit"]))
        t_profile = float(np.clip(t_raw / wall["elen"], 0.0, 1.0))
        proj = wall["p0_xz"] + (t_profile * wall["elen"]) * wall["edge_unit"]
        dist = float(np.linalg.norm(xz_pt - proj))
        return proj, dist, t_profile

    def snap_vertex_y(xz_pt, story, floor_y):
        fallback_top = ceiling_y_map.get(story, floor_y + DEFAULT_WALL_HEIGHT_M)
        best_dist = MAX_SNAP_DIST_M
        best_ybot = floor_y
        best_ytop = fallback_top
        for wall in story_walls.get(story, []):
            if abs(wall["match_y"] - floor_y) > MAX_Y_DIST_M:
                continue
            _proj, dist, t = project_to_wall_line(xz_pt, wall)
            if dist < best_dist:
                best_dist = dist
                best_ybot = floor_y
                best_ytop = interp_top_profile(wall["top_profile"], t)
        if best_ytop <= best_ybot + MIN_WALL_HEIGHT_M:
            best_ytop = fallback_top
        return best_ybot, best_ytop

    def pick_support_wall(xz_pt, prev_xz, next_xz, story, floor_y):
        tangent = None
        edge_ref = next_xz - prev_xz
        edge_len = float(np.linalg.norm(edge_ref))
        if edge_len > MIN_SNAP_DIST_M:
            tangent = edge_ref / edge_len
        best = None
        for wall in story_walls.get(story, []):
            if abs(wall["match_y"] - floor_y) > MAX_Y_DIST_M:
                continue
            proj, dist, t_profile = project_to_wall_line(xz_pt, wall)
            if dist > MAX_SNAP_DIST_M:
                continue
            cos_parallel = (
                abs(float(np.dot(tangent, wall["edge_unit"])))
                if tangent is not None
                else 1.0
            )
            score = dist + 0.35 * (1.0 - cos_parallel)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "wall": wall,
                    "proj": proj,
                    "t_profile": t_profile,
                }
        return best

    def build_snapped_vertices(edge_vertices, story, floor_y):
        snapped = []
        for idx, corner in enumerate(edge_vertices):
            prev_corner = edge_vertices[(idx - 1) % len(edge_vertices)]
            next_corner = edge_vertices[(idx + 1) % len(edge_vertices)]
            xz = np.array([corner[0], corner[2]], dtype=float)
            prev_xz = np.array([prev_corner[0], prev_corner[2]], dtype=float)
            next_xz = np.array([next_corner[0], next_corner[2]], dtype=float)
            picked = pick_support_wall(xz, prev_xz, next_xz, story, floor_y)
            if picked is None:
                ybot, ytop = snap_vertex_y(xz, story, floor_y)
                snapped.append({"xz": xz, "ybot": ybot, "ytop": ytop})
            else:
                ybot = floor_y
                ytop = interp_top_profile(
                    picked["wall"]["top_profile"], picked["t_profile"]
                )
                if ytop <= ybot + MIN_WALL_HEIGHT_M:
                    ytop = ceiling_y_map.get(story, floor_y + DEFAULT_WALL_HEIGHT_M)
                snapped.append(
                    {
                        "xz": picked["proj"],
                        "ybot": ybot,
                        "ytop": ytop,
                        "preserve_ceiling_profile": True,
                    }
                )
        try:
            poly = Polygon([(item["xz"][0], item["xz"][1]) for item in snapped])
            if (not poly.is_valid) or poly.area <= 1e-6:
                raise ValueError("invalid snapped polygon")
        except Exception:
            snapped = []
            for corner in edge_vertices:
                xz = np.array([corner[0], corner[2]], dtype=float)
                ybot, ytop = snap_vertex_y(xz, story, floor_y)
                snapped.append({"xz": xz, "ybot": ybot, "ytop": ytop})
        return snapped

    story_room_union = {}
    story_room_boundary = {}
    rooms_by_story: dict[int, list[Polygon]] = defaultdict(list)
    for idx, room in enumerate(rooms):
        floor_source = (
            pre_absorption_floor_polygons[idx]
            if pre_absorption_floor_polygons is not None
            and idx < len(pre_absorption_floor_polygons)
            else room.floor_polygon
        )
        poly = floor_polygon_to_shapely(floor_source)
        if poly is not None and poly.is_valid and poly.area > 0.0:
            rooms_by_story[room.story].append(poly)
    for story, polys in rooms_by_story.items():
        try:
            union = make_valid(unary_union(polys))
        except Exception:
            union = None
        story_room_union[story] = union
        story_room_boundary[story] = (
            union.boundary if union is not None and not union.is_empty else None
        )

    story_ceiling_lookup = build_story_ceiling_lookup(rooms)

    updated_gaps = []
    for gap in gaps:
        if gap.type == "cross_story":
            corners_3d = gap.corners
            below = gap.story - 1
            if below not in story_y_map or len(corners_3d) < 3:
                updated_gaps.append(gap)
                continue
            closed = len(corners_3d) >= 4 and corners_3d[0] == corners_3d[-1]
            edge_vertices = corners_3d[:-1] if closed else corners_3d
            snapped_below = build_snapped_vertices(
                edge_vertices, below, story_y_map[below]
            )
            draped = [
                [float(item["xz"][0]), float(item["ytop"]), float(item["xz"][1])]
                for item in snapped_below
            ]
            new_corners = [list(point) for point in draped]
            if closed:
                new_corners.append(list(draped[0]))
            centroid = list(gap.centroid)
            centroid[1] = float(np.mean([point[1] for point in draped]))
            updated_gaps.append(
                replace(
                    gap,
                    corners=new_corners,
                    ceiling_corners=[list(point) for point in draped],
                    centroid=centroid,
                )
            )
            continue
        if (
            gap.type not in ("within_story", "room_ceiling_void")
            or len(gap.corners) < 3
        ):
            updated_gaps.append(gap)
            continue

        floor_y = gap.corners[0][1]
        edge_vertices = (
            gap.corners[:-1]
            if len(gap.corners) >= 4 and gap.corners[0] == gap.corners[-1]
            else gap.corners
        )
        if len(edge_vertices) < 3:
            updated_gaps.append(gap)
            continue
        sloped_room = None
        sloped_room_plane = None
        sloped_ceiling_xz: Polygon | None = None
        flat_room_ceiling_y = None
        if (
            gap.room_index is not None
            and 0 <= gap.room_index < len(rooms)
            and rooms[gap.room_index].story == gap.story
        ):
            candidate_room = rooms[gap.room_index]
            candidate_plane = _sloped_room_ceiling_plane(candidate_room)
            if candidate_plane is not None:
                sloped_room = candidate_room
                sloped_room_plane = candidate_plane
                ceiling_corners = [
                    (float(c[0]), float(c[2])) for c in candidate_room.ceiling_polygon
                ]
                if len(ceiling_corners) >= 3:
                    if ceiling_corners[0] != ceiling_corners[-1]:
                        ceiling_corners.append(ceiling_corners[0])
                    try:
                        sloped_ceiling_xz = make_valid(Polygon(ceiling_corners))
                    except Exception:
                        sloped_ceiling_xz = None
                    if sloped_ceiling_xz is not None and (
                        sloped_ceiling_xz.is_empty or not sloped_ceiling_xz.is_valid
                    ):
                        sloped_ceiling_xz = None
            elif candidate_room.ceiling_polygon:
                ceiling_ys = [
                    float(corner[1]) for corner in candidate_room.ceiling_polygon
                ]
                if max(ceiling_ys) - min(ceiling_ys) <= MAX_HORIZONTAL_CAP_Y_RANGE_M:
                    flat_room_ceiling_y = float(np.median(ceiling_ys))

        def assigned_ceiling_y(
            x: float,
            z: float,
            sloped_room=sloped_room,
            sloped_room_plane=sloped_room_plane,
            flat_room_ceiling_y=flat_room_ceiling_y,
            sloped_ceiling_xz=sloped_ceiling_xz,
        ) -> float | None:
            if sloped_room is None or sloped_room_plane is None:
                return flat_room_ceiling_y
            if sloped_ceiling_xz is not None:
                try:
                    if not sloped_ceiling_xz.covers(Point(x, z)):
                        return None
                except Exception:
                    return None
            return sloped_room_plane.y_at(x, z)

        snapped = build_snapped_vertices(edge_vertices, gap.story, floor_y)
        if sloped_room is not None or flat_room_ceiling_y is not None:
            for item in snapped:
                ceiling_y = assigned_ceiling_y(
                    float(item["xz"][0]), float(item["xz"][1])
                )
                if ceiling_y is not None:
                    item["ytop"] = ceiling_y
                    item["preserve_ceiling_profile"] = True
        gap_floor_y = max(item["ybot"] for item in snapped)
        fallback_top_y = ceiling_y_map.get(
            gap.story, gap_floor_y + DEFAULT_WALL_HEIGHT_M
        )
        for item in snapped:
            if item["ytop"] <= gap_floor_y + MIN_WALL_HEIGHT_M:
                item["ytop"] = fallback_top_y
                item.pop("preserve_ceiling_profile", None)
        centroid = list(gap.centroid)
        centroid[1] = gap_floor_y
        gap = replace(
            gap,
            corners=[[corner[0], gap_floor_y, corner[2]] for corner in gap.corners],
            centroid=centroid,
        )
        updated_gaps.append(gap)

        try:
            snapped_poly = Polygon(
                [(float(item["xz"][0]), float(item["xz"][1])) for item in snapped]
            )
            if not snapped_poly.is_valid:
                snapped_poly = make_valid(snapped_poly)
        except Exception:
            snapped_poly = None
        room_union = story_room_union.get(gap.story)
        room_boundary = story_room_boundary.get(gap.story)
        pieces_for_caps = [snapped]
        clip_caps_to_room_complement = (
            gap.type != "room_ceiling_void"
            and room_union is not None
            and not room_union.is_empty
            and snapped_poly is not None
            and not snapped_poly.is_empty
        )
        if clip_caps_to_room_complement:
            try:
                clipped = make_valid(snapped_poly.difference(room_union))
            except Exception:
                clipped = None
            if clipped is None or clipped.is_empty:
                pieces_for_caps = []
            elif abs(snapped_poly.area - clipped.area) > 1e-6:
                pieces_for_caps = []
                for piece in sorted(
                    (poly for poly in decompose_polys(clipped) if poly.area > 0.01),
                    key=lambda poly: -poly.area,
                ):
                    coords = list(piece.exterior.coords)
                    if coords and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    piece_snapped = []
                    for x, z in coords:
                        ceiling_y = assigned_ceiling_y(float(x), float(z))
                        piece_snapped.append(
                            {
                                "xz": np.array([x, z], dtype=float),
                                "ybot": gap_floor_y,
                                "ytop": ceiling_y
                                if ceiling_y is not None
                                else _ytop_at_xz((x, z), snapped),
                                "preserve_ceiling_profile": ceiling_y is not None,
                            }
                        )
                    if len(piece_snapped) >= 3:
                        pieces_for_caps.append(piece_snapped)

        for edge_idx in range(len(snapped)):
            next_idx = (edge_idx + 1) % len(snapped)
            c0 = snapped[edge_idx]["xz"]
            c1 = snapped[next_idx]["xz"]
            if gap.type == "room_ceiling_void":
                continue
            if _edge_on_room_boundary(c0, c1, room_boundary):
                continue
            walls.append(
                ExtractedGapWall(
                    id=_stable_gap_wall_id(gap, gap.type, "edge", edge_idx),
                    corners=[
                        [float(c0[0]), gap_floor_y, float(c0[1])],
                        [float(c1[0]), gap_floor_y, float(c1[1])],
                        [float(c1[0]), float(snapped[next_idx]["ytop"]), float(c1[1])],
                        [float(c0[0]), float(snapped[edge_idx]["ytop"]), float(c0[1])],
                    ],
                    type=gap.type,
                    story=gap.story,
                    confidence=gap.confidence,
                )
            )

        for piece_idx, piece_snapped in enumerate(pieces_for_caps):
            n_pieces = len(pieces_for_caps)
            _apply_room_ceiling_fallback(piece_snapped, story_ceiling_lookup, gap.story)
            walls.append(
                ExtractedGapWall(
                    id=_stable_gap_wall_id(
                        gap,
                        "gap_floor",
                        "polygon",
                        _piece_index(piece_idx, n_pieces, None),
                    ),
                    corners=[
                        [float(item["xz"][0]), gap_floor_y, float(item["xz"][1])]
                        for item in piece_snapped
                    ],
                    type="gap_floor",
                    story=gap.story,
                    confidence=gap.confidence,
                )
            )
            xz_2d = [
                (float(item["xz"][0]), float(item["xz"][1])) for item in piece_snapped
            ]
            tri_idx = 0
            for ia, ib, ic in earclip_2d(xz_2d):
                sa, sb, sc = piece_snapped[ia], piece_snapped[ib], piece_snapped[ic]
                x0, z0 = sa["xz"]
                x1, z1 = sb["xz"]
                x2, z2 = sc["xz"]
                if abs((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) * 0.5 < 1e-5:
                    continue
                cap_corners = [
                    [float(sa["xz"][0]), float(sa["ytop"]), float(sa["xz"][1])],
                    [float(sb["xz"][0]), float(sb["ytop"]), float(sb["xz"][1])],
                    [float(sc["xz"][0]), float(sc["ytop"]), float(sc["xz"][1])],
                ]
                cap_ys = [corner[1] for corner in cap_corners]
                quality = _triangle_quality(
                    cap_corners[0], cap_corners[1], cap_corners[2]
                )
                if (
                    quality < MIN_TRI_QUALITY
                    and (max(cap_ys) - min(cap_ys)) < MAX_HORIZONTAL_CAP_Y_RANGE_M
                ):
                    continue
                cap_inclination = _surface_inclination_deg(cap_corners)
                if (
                    cap_inclination is not None
                    and cap_inclination > MAX_CEILING_CAP_INCLINATION_DEG
                ):
                    continue
                walls.append(
                    ExtractedGapWall(
                        id=_stable_gap_wall_id(
                            gap,
                            "gap_ceiling",
                            "tri",
                            _piece_index(piece_idx, n_pieces, tri_idx),
                        ),
                        corners=cap_corners,
                        type="gap_ceiling",
                        story=gap.story,
                        confidence=gap.confidence,
                    )
                )
                tri_idx += 1

    return _dedupe_exact_gap_walls(
        _limit_gap_walls_to_contract(walls, updated_gaps)
    ), updated_gaps
