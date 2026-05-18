from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers.extract.building import (
    ExteriorGapIndicator,
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
    GapClosure,
)

_DEBUG_ENABLED = os.environ.get("EXTERIOR_DEBUG") == "1"
_REJECTION_LOG: list[dict] = []
_ACCEPTED_LOG: list[dict] = []


def drain_rejection_log() -> tuple[list[dict], list[dict]]:
    """Return and clear the in-memory diagnostic logs used by EXTERIOR_DEBUG."""

    rejections = list(_REJECTION_LOG)
    accepted = list(_ACCEPTED_LOG)
    _REJECTION_LOG.clear()
    _ACCEPTED_LOG.clear()
    return rejections, accepted


def _log_reject(
    *,
    story: int,
    room_idx: int,
    element_type: str,
    element_id: str,
    elem_width: float,
    candidate_wall_id: str,
    gate: str,
    value: float,
    threshold: float | tuple[float, float],
) -> None:
    if not _DEBUG_ENABLED:
        return
    _REJECTION_LOG.append(
        {
            "story": story,
            "room_idx": room_idx,
            "element_type": element_type,
            "element_id": element_id,
            "elem_width": round(elem_width, 4),
            "candidate_wall_id": candidate_wall_id,
            "gate": gate,
            "value": round(float(value), 4),
            "threshold": threshold
            if not isinstance(threshold, tuple)
            else [threshold[0], threshold[1]],
        }
    )


def _log_accept(
    *,
    story: int,
    room_idx: int,
    element_type: str,
    element_id: str,
    elem_width: float,
    wall_id: str,
    wall_width: float,
    dist_along: float,
    perp: float,
    angle_deg: float,
) -> None:
    if not _DEBUG_ENABLED:
        return
    _ACCEPTED_LOG.append(
        {
            "story": story,
            "room_idx": room_idx,
            "element_type": element_type,
            "element_id": element_id,
            "elem_width": round(elem_width, 4),
            "wall_id": wall_id,
            "wall_width": round(wall_width, 4),
            "dist_along": round(dist_along, 4),
            "perp": round(perp, 4),
            "angle_deg": round(angle_deg, 4),
        }
    )


def wall_xz_length(corners: list[list[float]]) -> float:
    if len(corners) < 2:
        return 0.0
    dx = float(corners[1][0]) - float(corners[0][0])
    dz = float(corners[1][2]) - float(corners[0][2])
    return math.hypot(dx, dz)


def _canonicalize_wall_quad(corners: list[list[float]]) -> list[list[float]]:
    if len(corners) < 4:
        return [list(corner) for corner in corners]
    arr = [list(corner) for corner in corners]
    ys = [corner[1] for corner in arr]
    if max(ys) - min(ys) < 1e-3:
        return arr
    mid_y = (max(ys) + min(ys)) / 2.0
    bottom = [corner for corner in arr if corner[1] < mid_y + 0.01]
    top = [corner for corner in arr if corner[1] > mid_y - 0.01]
    if len(bottom) < 2 or len(top) < 2:
        return arr

    bl, br = bottom[0], bottom[1]
    best_d2 = -1.0
    for idx in range(len(bottom)):
        for jdx in range(idx + 1, len(bottom)):
            dx = bottom[idx][0] - bottom[jdx][0]
            dz = bottom[idx][2] - bottom[jdx][2]
            d2 = dx * dx + dz * dz
            if d2 > best_d2:
                best_d2 = d2
                bl, br = bottom[idx], bottom[jdx]

    ref_dx = arr[1][0] - arr[0][0]
    ref_dz = arr[1][2] - arr[0][2]
    if (br[0] - bl[0]) * ref_dx + (br[2] - bl[2]) * ref_dz < 0:
        bl, br = br, bl

    def nearest(x: float, z: float, candidates: list[list[float]]) -> list[float]:
        return min(
            candidates, key=lambda corner: (corner[0] - x) ** 2 + (corner[2] - z) ** 2
        )

    return [
        list(bl),
        list(br),
        list(nearest(br[0], br[2], top)),
        list(nearest(bl[0], bl[2], top)),
    ]


def _wall_top_bottom_profiles(
    corners: list[list[float]],
    axis2d: np.ndarray,
    origin_xz: tuple[float, float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(corners) < 3:
        return [], []
    pts = []
    for corner in corners:
        dx = float(corner[0]) - origin_xz[0]
        dz = float(corner[2]) - origin_xz[1]
        pts.append((dx * axis2d[0] + dz * axis2d[1], float(corner[1])))
    t_min = min(point[0] for point in pts)
    t_max = max(point[0] for point in pts)
    if t_max - t_min < 1e-6:
        return [], []

    runs = []
    current = [pts[0]]
    current_sign = 0
    for idx in range(1, len(pts) + 1):
        a = pts[(idx - 1) % len(pts)]
        b = pts[idx % len(pts)]
        dt = b[0] - a[0]
        if abs(dt) < 1e-9:
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
        runs = [(runs[0][0], runs[-1][1] + runs[0][1]), *runs[1:-1]]

    forward = next((run for sign, run in runs if sign == 1), None)
    backward = next((run for sign, run in runs if sign == -1), None)
    if not forward or not backward:
        return [], []

    def consolidate(profile, take_max: bool) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for t, y in sorted(profile, key=lambda point: point[0]):
            if out and abs(out[-1][0] - t) < 1e-6:
                out[-1] = (
                    out[-1][0],
                    max(out[-1][1], y) if take_max else min(out[-1][1], y),
                )
            else:
                out.append((t, y))
        return out

    if sum(point[1] for point in forward) / len(forward) >= sum(
        point[1] for point in backward
    ) / len(backward):
        return consolidate(forward, True), consolidate(backward, False)
    return consolidate(backward, True), consolidate(forward, False)


def _interp_profile_y(profile: list[tuple[float, float]], t: float) -> float:
    if not profile:
        return 0.0
    if t <= profile[0][0]:
        return profile[0][1]
    if t >= profile[-1][0]:
        return profile[-1][1]
    for idx in range(len(profile) - 1):
        t0, y0 = profile[idx]
        t1, y1 = profile[idx + 1]
        if t0 <= t <= t1:
            return y0 + (t - t0) / (t1 - t0) * (y1 - y0) if t1 > t0 else y0
    return profile[-1][1]


def _normal_xz(corners: list[list[float]]) -> np.ndarray | None:
    if len(corners) < 3:
        return None
    normal = newell_normal(corners)
    normal_xz = np.array([normal[0], normal[2]], dtype=float)
    norm = float(np.linalg.norm(normal_xz))
    if norm < 1e-6:
        return None
    return normal_xz / norm


def _projection_coverage(
    elem_corners: list[list[float]],
    wall_corners: list[list[float]],
) -> float:
    """Fraction of the element's XZ extent that overlaps the wall when both are
    projected onto the wall's axis. Required physical signal that the door /
    opening actually faces the candidate wall (rather than being off to one
    end), independent of the symmetric `perp` center-to-center check.
    """

    canon = _canonicalize_wall_quad(wall_corners)
    if len(canon) < 2:
        return 0.0
    axis = np.array([canon[1][0] - canon[0][0], canon[1][2] - canon[0][2]], dtype=float)
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-6:
        return 0.0
    axis /= axis_len
    origin = np.array([canon[0][0], canon[0][2]], dtype=float)

    def proj(corners: list[list[float]]) -> tuple[float, float]:
        ts = [float(np.dot(np.array([c[0], c[2]]) - origin, axis)) for c in corners]
        return min(ts), max(ts)

    elem_min, elem_max = proj(elem_corners)
    wall_min, wall_max = proj(canon)
    overlap = max(0.0, min(elem_max, wall_max) - max(elem_min, wall_min))
    elem_extent = elem_max - elem_min
    if elem_extent < 1e-6:
        return 0.0
    return overlap / elem_extent


def _find_parallel_wall(
    *,
    corners: list[list[float]],
    elem_width: float,
    story: int,
    room_idx: int,
    require_other_room: bool,
    story_walls: dict[int, list[tuple[ExtractedWall, int]]],
    exclude_wall_id: str | None = None,
    element_type: str = "",
    element_id: str = "",
):
    normal_xz = _normal_xz(corners)
    if normal_xz is None:
        return None
    elem_cx = float(np.mean([corner[0] for corner in corners]))
    elem_cz = float(np.mean([corner[2] for corner in corners]))
    best_match = None
    best_dist = float("inf")

    def _reject(wall_id: str, gate: str, value: float, threshold) -> None:
        _log_reject(
            story=story,
            room_idx=room_idx,
            element_type=element_type,
            element_id=element_id,
            elem_width=elem_width,
            candidate_wall_id=wall_id,
            gate=gate,
            value=value,
            threshold=threshold,
        )

    for sign in (1.0, -1.0):
        direction = normal_xz * sign
        for wall, wall_room_idx in story_walls.get(story, []):
            if exclude_wall_id is not None and wall.id == exclude_wall_id:
                continue
            if require_other_room and wall_room_idx == room_idx:
                _reject(wall.id, "same_room", float(room_idx), float(wall_room_idx))
                continue
            wall_corners = wall.corners
            if len(wall_corners) < 3:
                continue
            wall_width = wall_xz_length(wall_corners)
            if wall_width < elem_width * 0.9:
                _reject(
                    wall.id, "wall_width_ratio", wall_width / max(elem_width, 1e-6), 0.9
                )
                continue
            wall_cx = float(np.mean([corner[0] for corner in wall_corners]))
            wall_cz = float(np.mean([corner[2] for corner in wall_corners]))
            to_wall = np.array([wall_cx - elem_cx, wall_cz - elem_cz], dtype=float)
            dist_along = float(np.dot(to_wall, direction))
            if dist_along < 0.20 or dist_along > 1.30:
                _reject(wall.id, "dist_along", dist_along, (0.20, 1.30))
                continue
            perp = abs(float(np.dot(to_wall, np.array([-direction[1], direction[0]]))))
            perp_threshold = max(elem_width, wall_width) * 0.5
            if perp > perp_threshold:
                _reject(wall.id, "perp", perp, perp_threshold)
                continue
            wall_normal = _normal_xz(wall_corners)
            if wall_normal is None:
                continue
            angle_deg = math.degrees(
                math.acos(min(abs(float(np.dot(normal_xz, wall_normal))), 1.0))
            )
            if angle_deg > 15.0:
                _reject(wall.id, "angle_deg", angle_deg, 15.0)
                continue
            coverage = _projection_coverage(corners, wall_corners)
            if coverage < 0.5:
                _reject(wall.id, "projection_coverage", coverage, 0.5)
                continue
            if dist_along < best_dist:
                best_dist = dist_along
                best_match = (
                    wall,
                    wall_corners,
                    dist_along,
                    angle_deg,
                    perp,
                    wall_width,
                )
    if best_match is not None and _DEBUG_ENABLED:
        wall, _wc, dist_along, angle_deg, perp, wall_width = best_match
        _log_accept(
            story=story,
            room_idx=room_idx,
            element_type=element_type,
            element_id=element_id,
            elem_width=elem_width,
            wall_id=wall.id,
            wall_width=wall_width,
            dist_along=dist_along,
            perp=perp,
            angle_deg=angle_deg,
        )
    if best_match is not None:
        wall, wc, dist_along, angle_deg, _perp, _ww = best_match
        return (wall, wc, dist_along, angle_deg)
    return None


def _confidence(angle_deg: float, dist_along: float) -> str:
    if angle_deg < 5 and dist_along < 0.3:
        return "high"
    if angle_deg < 10 and dist_along < 0.6:
        return "medium"
    return "low"


def _indicator_from_match(
    *,
    story: int,
    element_type: str,
    element: ExtractedElement,
    elem_width: float,
    match,
    parent_wall: ExtractedWall | None = None,
) -> ExteriorGapIndicator:
    wall, wall_corners, dist_along, angle_deg = match
    return ExteriorGapIndicator(
        story=story,
        element_type=element_type,
        element_id=element.id,
        element_corners=element.corners,
        element_width_m=round(elem_width, 2),
        wall_id=wall.id,
        wall_corners=wall_corners,
        wall_distance_m=round(dist_along, 2),
        angle_deg=round(angle_deg, 1),
        confidence=_confidence(angle_deg, dist_along),
        parent_wall_id=parent_wall.id if parent_wall is not None else None,
        parent_wall_corners=parent_wall.corners if parent_wall is not None else None,
    )


def detect_exterior_gap_indicators(
    rooms: list[ExtractedRoom],
) -> list[ExteriorGapIndicator]:
    story_walls: dict[int, list[tuple[ExtractedWall, int]]] = defaultdict(list)
    for room_idx, room in enumerate(rooms):
        for wall in room.walls_computed:
            story_walls[room.story].append((wall, room_idx))

    indicators: list[ExteriorGapIndicator] = []
    for room_idx, room in enumerate(rooms):
        room_walls_by_id = {wall.id: wall for wall in room.walls_computed}
        for element_type, elements in (
            ("door", room.doors),
            ("opening", room.openings),
        ):
            for element in elements:
                elem_width = wall_xz_length(element.corners)
                if elem_width < 0.50:
                    continue
                parent_wall = (
                    room_walls_by_id.get(element.parent_wall_id)
                    if element.parent_wall_id
                    else None
                )
                match = _find_parallel_wall(
                    corners=element.corners,
                    elem_width=elem_width,
                    story=room.story,
                    room_idx=room_idx,
                    require_other_room=False,
                    story_walls=story_walls,
                    exclude_wall_id=element.parent_wall_id,
                    element_type=element_type,
                    element_id=element.id,
                )
                if match:
                    indicators.append(
                        _indicator_from_match(
                            story=room.story,
                            element_type=element_type,
                            element=element,
                            elem_width=elem_width,
                            match=match,
                            parent_wall=parent_wall,
                        )
                    )

        room_wall_ids = {wall.id for wall in room.walls_computed}
        for storage in room.storages:
            elem_width = wall_xz_length(storage.corners)
            if elem_width < 0.50:
                continue
            sc = np.asarray(storage.corners, dtype=float)
            storage_xz = np.array([float(sc[:, 0].mean()), float(sc[:, 2].mean())])
            flush_wall = None
            for wall, _wall_room_idx in story_walls.get(room.story, []):
                if wall.id not in room_wall_ids or len(wall.corners) < 2:
                    continue
                w0 = np.array([wall.corners[0][0], wall.corners[0][2]], dtype=float)
                w1 = np.array([wall.corners[1][0], wall.corners[1][2]], dtype=float)
                edge = w1 - w0
                edge_len = float(np.linalg.norm(edge))
                if edge_len < 1e-6:
                    continue
                perp_dist = (
                    abs(
                        float(
                            edge[0] * (storage_xz - w0)[1]
                            - edge[1] * (storage_xz - w0)[0]
                        )
                    )
                    / edge_len
                )
                if perp_dist < 0.05:
                    flush_wall = wall
                    break
            if flush_wall is None:
                continue
            match = _find_parallel_wall(
                corners=storage.corners,
                elem_width=elem_width,
                story=room.story,
                room_idx=room_idx,
                require_other_room=True,
                story_walls=story_walls,
                element_type="storage",
                element_id=storage.id,
            )
            if match:
                indicators.append(
                    _indicator_from_match(
                        story=room.story,
                        element_type="storage",
                        element=storage,
                        elem_width=elem_width,
                        match=match,
                        parent_wall=flush_wall,
                    )
                )
    return indicators


def compute_gap_closures(indicators: list[ExteriorGapIndicator]) -> list[GapClosure]:
    closures: list[GapClosure] = []
    for indicator in indicators:
        if len(indicator.wall_corners) < 4 or len(indicator.element_corners) < 4:
            continue
        wall_corners = np.asarray(
            _canonicalize_wall_quad(indicator.wall_corners), dtype=float
        )
        parent_source_corners = (
            indicator.parent_wall_corners or indicator.element_corners
        )
        parent_corners = np.asarray(
            _canonicalize_wall_quad(parent_source_corners), dtype=float
        )
        wall_dir = wall_corners[1] - wall_corners[0]
        wall_dir_xz = np.array([wall_dir[0], 0.0, wall_dir[2]], dtype=float)
        wall_len = float(np.linalg.norm(wall_dir_xz))
        if wall_len < 1e-6:
            continue
        axis = wall_dir_xz / wall_len
        origin = np.array([wall_corners[0][0], 0.0, wall_corners[0][2]], dtype=float)
        axis2d = np.array([axis[0], axis[2]], dtype=float)

        def project(points, origin=origin, axis2d=axis2d):
            return [
                float(
                    np.dot(
                        np.array([point[0], point[2]])
                        - np.array([origin[0], origin[2]]),
                        axis2d,
                    )
                )
                for point in points
            ]

        parent_proj = project(parent_corners)
        wall_proj = project(wall_corners)
        overlap_min = max(min(parent_proj), min(wall_proj))
        overlap_max = min(max(parent_proj), max(wall_proj))
        if overlap_max - overlap_min < 0.05:
            continue

        parent_raw = _canonicalize_wall_quad(parent_source_corners)
        wall_raw = _canonicalize_wall_quad(indicator.wall_corners)
        origin_xz = (float(origin[0]), float(origin[2]))
        parent_top, parent_bottom = _wall_top_bottom_profiles(
            parent_raw, axis2d, origin_xz
        )
        wall_top, wall_bottom = _wall_top_bottom_profiles(wall_raw, axis2d, origin_xz)
        if not (parent_top and parent_bottom and wall_top and wall_bottom):
            continue

        parent_xz0 = np.array([parent_corners[0][0], parent_corners[0][2]], dtype=float)
        parent_xz1 = np.array([parent_corners[1][0], parent_corners[1][2]], dtype=float)
        parent_edge = parent_xz1 - parent_xz0
        parent_len = float(np.linalg.norm(parent_edge))
        wall_xz0 = np.array([wall_corners[0][0], wall_corners[0][2]], dtype=float)
        wall_xz1 = np.array([wall_corners[1][0], wall_corners[1][2]], dtype=float)
        wall_edge = wall_xz1 - wall_xz0
        wall_edge_len = float(np.linalg.norm(wall_edge))

        def edge_point(xz_pt, start, edge, length):
            if length < 1e-6:
                return start
            t = float(np.clip(np.dot(xz_pt - start, edge) / (length**2), 0, 1))
            return start + t * edge

        side_pts = []
        for t_val in (overlap_min, overlap_max):
            xz_pt = np.array([origin[0], origin[2]], dtype=float) + axis2d * t_val
            parent_xz = edge_point(xz_pt, parent_xz0, parent_edge, parent_len)
            wall_xz = edge_point(xz_pt, wall_xz0, wall_edge, wall_edge_len)
            parent_ybot = _interp_profile_y(parent_bottom, t_val)
            parent_ytop = _interp_profile_y(parent_top, t_val)
            wall_ybot = _interp_profile_y(wall_bottom, t_val)
            wall_ytop = _interp_profile_y(wall_top, t_val)
            side_pts.append(
                (parent_xz, parent_ybot, parent_ytop, wall_xz, wall_ybot, wall_ytop)
            )
            closures.append(
                GapClosure(
                    corners=[
                        [float(parent_xz[0]), parent_ybot, float(parent_xz[1])],
                        [float(wall_xz[0]), wall_ybot, float(wall_xz[1])],
                        [float(wall_xz[0]), wall_ytop, float(wall_xz[1])],
                        [float(parent_xz[0]), parent_ytop, float(parent_xz[1])],
                    ],
                    type="side",
                    indicator_element_id=indicator.element_id,
                    indicator_wall_id=indicator.wall_id,
                    story=indicator.story,
                )
            )

        left, right = side_pts
        p_xz_l, p_ybot_l, p_ytop_l, w_xz_l, w_ybot_l, w_ytop_l = left
        p_xz_r, p_ybot_r, p_ytop_r, w_xz_r, w_ybot_r, w_ytop_r = right
        closures.append(
            GapClosure(
                corners=[
                    [float(p_xz_l[0]), p_ybot_l, float(p_xz_l[1])],
                    [float(p_xz_r[0]), p_ybot_r, float(p_xz_r[1])],
                    [float(w_xz_r[0]), w_ybot_r, float(w_xz_r[1])],
                    [float(w_xz_l[0]), w_ybot_l, float(w_xz_l[1])],
                ],
                type="floor",
                indicator_element_id=indicator.element_id,
                indicator_wall_id=indicator.wall_id,
                story=indicator.story,
            )
        )
        closures.append(
            GapClosure(
                corners=[
                    [float(p_xz_l[0]), p_ytop_l, float(p_xz_l[1])],
                    [float(p_xz_r[0]), p_ytop_r, float(p_xz_r[1])],
                    [float(w_xz_r[0]), w_ytop_r, float(w_xz_r[1])],
                    [float(w_xz_l[0]), w_ytop_l, float(w_xz_l[1])],
                ],
                type="ceiling",
                indicator_element_id=indicator.element_id,
                indicator_wall_id=indicator.wall_id,
                story=indicator.story,
            )
        )
    return closures
