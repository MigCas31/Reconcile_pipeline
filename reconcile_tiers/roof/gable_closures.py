from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin

from reconcile_tiers._core.newell import newell_normal, polygon_area_3d
from reconcile_tiers._core.plane import Plane
from reconcile_tiers.extract.building import BuildingModel, ExtractedRoom, ExtractedWall
from reconcile_tiers.payload.schema import GableClosureKind
from reconcile_tiers.roof.geometry import angle_diff_deg
from reconcile_tiers.roof.roof import ObliqueSurface, RoofModel

GABLE_PERPENDICULAR_TOLERANCE_DEG = 30.0
GABLE_PAIR_AZIMUTH_TOLERANCE_DEG = 40.0
GABLE_PAIR_INCL_TOLERANCE_DEG = 10.0
GABLE_END_RIDGE_PROJ_TOLERANCE_M = 1.0
MIN_PANEL_AREA_M2 = 0.05
MIN_PANEL_HEIGHT_M = 0.05
ROOF_CLIP_EPS_M = 1e-5


@dataclass(frozen=True, slots=True)
class GableClosureSurface:
    corners: list[list[float]]
    kind: GableClosureKind
    room_index: int | None
    source: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    room: ExtractedRoom
    wall: ExtractedWall
    fp: tuple[tuple[float, float], tuple[float, float]]
    ridge_proj: float
    eave_y: float | None
    attic_lid_y: float | None


def build_gable_closures(
    model: BuildingModel, roof: RoofModel
) -> list[GableClosureSurface]:
    if roof.kinks.ridge_y is None:
        return []
    pair = _select_gable_pair(roof.oblique)
    if pair is None:
        return []
    plane_a, plane_b = pair[0].plane, pair[1].plane
    ridge_axis_az = _ridge_axis_azimuth(pair[0], pair[1])
    ridge_axis_rad = radians(ridge_axis_az)
    ridge_axis = (sin(ridge_axis_rad), cos(ridge_axis_rad))
    gable_axis = (cos(ridge_axis_rad), -sin(ridge_axis_rad))
    ridge_y = float(roof.kinks.ridge_y)

    candidates = _collect_candidates(model, roof, ridge_axis_az, ridge_axis)
    if not candidates:
        return []

    min_proj = min(c.ridge_proj for c in candidates)
    max_proj = max(c.ridge_proj for c in candidates)
    if max_proj - min_proj < 2 * GABLE_END_RIDGE_PROJ_TOLERANCE_M:
        return []

    out: list[GableClosureSurface] = []
    for extreme_proj in (min_proj, max_proj):
        group = [
            c
            for c in candidates
            if abs(c.ridge_proj - extreme_proj) <= GABLE_END_RIDGE_PROJ_TOLERANCE_M
        ]
        if not group:
            continue
        eave_ys = [c.eave_y for c in group if c.eave_y is not None]
        attic_lid_ys = [c.attic_lid_y for c in group if c.attic_lid_y is not None]
        if not eave_ys or not attic_lid_ys:
            continue
        eave_y = min(eave_ys)
        attic_lid_y = max(attic_lid_ys)

        merged_fp = _merge_footprints(group, ridge_axis, gable_axis)
        if merged_fp is None:
            continue

        avg_proj = sum(c.ridge_proj for c in candidates) / len(candidates)
        outward_sign = 1.0 if extreme_proj >= avg_proj else -1.0
        outward_xz = (ridge_axis[0] * outward_sign, ridge_axis[1] * outward_sign)
        room_index = group[0].room.index

        if attic_lid_y - eave_y >= MIN_PANEL_HEIGHT_M:
            lower = _build_lower_panel(
                merged_fp, eave_y, attic_lid_y, plane_a, plane_b, outward_xz
            )
            if lower is not None and polygon_area_3d(lower) >= MIN_PANEL_AREA_M2:
                out.append(
                    GableClosureSurface(
                        corners=lower,
                        kind=GableClosureKind.LOWER,
                        room_index=room_index,
                        source="gable_end",
                    )
                )

        if ridge_y - attic_lid_y >= MIN_PANEL_HEIGHT_M:
            upper = _build_upper_panel(
                merged_fp, attic_lid_y, ridge_y, plane_a, plane_b, outward_xz
            )
            if upper is not None and polygon_area_3d(upper) >= MIN_PANEL_AREA_M2:
                out.append(
                    GableClosureSurface(
                        corners=upper,
                        kind=GableClosureKind.UPPER,
                        room_index=room_index,
                        source="gable_end",
                    )
                )
    return out


def _collect_candidates(
    model: BuildingModel,
    roof: RoofModel,
    ridge_axis_az: float,
    ridge_axis: tuple[float, float],
) -> list[_Candidate]:
    top_story = max((r.story for r in model.rooms), default=0)
    out: list[_Candidate] = []
    for room in model.rooms:
        if room.story != top_story:
            continue
        eave_y = roof.kinks.eave_y(room.index)
        attic_lid_y = roof.kinks.attic_lid_y(room.index)
        for wall in room.walls_computed:
            fp = _wall_footprint_segment(wall)
            if fp is None:
                continue
            if not _is_perpendicular_to_ridge(fp, ridge_axis_az):
                continue
            mid_x = (fp[0][0] + fp[1][0]) / 2.0
            mid_z = (fp[0][1] + fp[1][1]) / 2.0
            ridge_proj = mid_x * ridge_axis[0] + mid_z * ridge_axis[1]
            out.append(
                _Candidate(
                    room=room,
                    wall=wall,
                    fp=fp,
                    ridge_proj=ridge_proj,
                    eave_y=eave_y,
                    attic_lid_y=attic_lid_y,
                )
            )
    return out


def _merge_footprints(
    group: list[_Candidate],
    ridge_axis: tuple[float, float],
    gable_axis: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    points = [pt for c in group for pt in c.fp]
    if len(points) < 2:
        return None
    avg_ridge_proj = sum(
        p[0] * ridge_axis[0] + p[1] * ridge_axis[1] for p in points
    ) / len(points)
    gable_projs = [p[0] * gable_axis[0] + p[1] * gable_axis[1] for p in points]
    g_min = min(gable_projs)
    g_max = max(gable_projs)
    if g_max - g_min < 1e-6:
        return None
    p1 = (
        avg_ridge_proj * ridge_axis[0] + g_min * gable_axis[0],
        avg_ridge_proj * ridge_axis[1] + g_min * gable_axis[1],
    )
    p2 = (
        avg_ridge_proj * ridge_axis[0] + g_max * gable_axis[0],
        avg_ridge_proj * ridge_axis[1] + g_max * gable_axis[1],
    )
    return (p1, p2)


def _select_gable_pair(
    obliques: list[ObliqueSurface],
) -> tuple[ObliqueSurface, ObliqueSurface] | None:
    best: tuple[float, ObliqueSurface, ObliqueSurface] | None = None
    for i, a in enumerate(obliques):
        for b in obliques[i + 1 :]:
            diff = angle_diff_deg(a.cluster.avg_azimuth, b.cluster.avg_azimuth)
            if (
                not (180.0 - GABLE_PAIR_AZIMUTH_TOLERANCE_DEG)
                <= diff
                <= (180.0 + GABLE_PAIR_AZIMUTH_TOLERANCE_DEG)
            ):
                continue
            if (
                abs(a.cluster.avg_incl - b.cluster.avg_incl)
                > GABLE_PAIR_INCL_TOLERANCE_DEG
            ):
                continue
            score = polygon_area_3d(a.corners) + polygon_area_3d(b.corners)
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return None
    return (best[1], best[2])


def _ridge_axis_azimuth(a: ObliqueSurface, b: ObliqueSurface) -> float:
    """Return the ridge-direction azimuth (degrees) for an opposing gable pair.

    The ridge runs perpendicular to the slope direction of either plane. Averaging
    the two slope azimuths after collapsing them onto the same half-circle gives
    a stable slope direction; the ridge sits 90° from it.
    """
    az_a = a.cluster.avg_azimuth % 360.0
    az_b_collapsed = (b.cluster.avg_azimuth + 180.0) % 360.0
    if abs(az_a - az_b_collapsed) > 180.0:
        if az_b_collapsed > az_a:
            az_b_collapsed -= 360.0
        else:
            az_b_collapsed += 360.0
    slope_az = (az_a + az_b_collapsed) / 2.0
    return (slope_az + 90.0) % 180.0


def _wall_footprint_segment(
    wall: ExtractedWall,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    corners = wall.corners
    if len(corners) < 3:
        return None
    sorted_corners = sorted(corners, key=lambda c: float(c[1]))
    p1 = (float(sorted_corners[0][0]), float(sorted_corners[0][2]))
    p2 = (float(sorted_corners[1][0]), float(sorted_corners[1][2]))
    if (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 < 1e-9:
        return None
    return (p1, p2)


def _is_perpendicular_to_ridge(
    footprint: tuple[tuple[float, float], tuple[float, float]],
    ridge_axis_az: float,
) -> bool:
    p1, p2 = footprint
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    wall_az = degrees(atan2(dx, dz)) % 180.0
    diff = abs(wall_az - (ridge_axis_az % 180.0))
    diff = min(diff, 180.0 - diff)
    return abs(diff - 90.0) <= GABLE_PERPENDICULAR_TOLERANCE_DEG


def _build_lower_panel(
    footprint: tuple[tuple[float, float], tuple[float, float]],
    eave_y: float,
    attic_lid_y: float,
    plane_a: Plane,
    plane_b: Plane,
    outward_xz: tuple[float, float],
) -> list[list[float]] | None:
    p1, p2 = footprint
    rectangle = [
        [p1[0], eave_y, p1[1]],
        [p2[0], eave_y, p2[1]],
        [p2[0], attic_lid_y, p2[1]],
        [p1[0], attic_lid_y, p1[1]],
    ]
    clipped = _clip_below_plane(rectangle, plane_a)
    if len(clipped) < 3:
        return None
    clipped = _clip_below_plane(clipped, plane_b)
    if len(clipped) < 3:
        return None
    return _orient_outward(clipped, outward_xz)


def _build_upper_panel(
    footprint: tuple[tuple[float, float], tuple[float, float]],
    attic_lid_y: float,
    ridge_y: float,
    plane_a: Plane,
    plane_b: Plane,
    outward_xz: tuple[float, float],
) -> list[list[float]] | None:
    p1, p2 = footprint
    yA0 = plane_a.y_at(p1[0], p1[1])
    yA1 = plane_a.y_at(p2[0], p2[1])
    yB0 = plane_b.y_at(p1[0], p1[1])
    yB1 = plane_b.y_at(p2[0], p2[1])
    if yA0 is None or yA1 is None or yB0 is None or yB1 is None:
        return None
    denom_peak = (yB1 - yB0) - (yA1 - yA0)
    if abs(denom_peak) < 1e-9:
        return None
    t_peak = (yA0 - yB0) / denom_peak
    if not (0.0 <= t_peak <= 1.0):
        return None

    a_lower_at_t0 = yA0 < yB0
    if a_lower_at_t0:
        slope_left, base_left = (yA1 - yA0), yA0
        slope_right, base_right = (yB1 - yB0), yB0
    else:
        slope_left, base_left = (yB1 - yB0), yB0
        slope_right, base_right = (yA1 - yA0), yA0
    if abs(slope_left) < 1e-9 or abs(slope_right) < 1e-9:
        return None
    t_left = (attic_lid_y - base_left) / slope_left
    t_right = (attic_lid_y - base_right) / slope_right
    t_left = max(0.0, min(1.0, t_left))
    t_right = max(0.0, min(1.0, t_right))
    if t_left >= t_right:
        return None
    if not (t_left <= t_peak <= t_right):
        return None

    def at_t(t: float) -> tuple[float, float]:
        return (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
        )

    left_xz = at_t(t_left)
    right_xz = at_t(t_right)
    peak_xz = at_t(t_peak)
    triangle = [
        [left_xz[0], attic_lid_y, left_xz[1]],
        [right_xz[0], attic_lid_y, right_xz[1]],
        [peak_xz[0], ridge_y, peak_xz[1]],
    ]
    return _orient_outward(triangle, outward_xz)


def _clip_below_plane(corners: list[list[float]], plane: Plane) -> list[list[float]]:
    if len(corners) < 3:
        return []
    a, b, c, d = float(plane.a), float(plane.b), float(plane.c), float(plane.d)

    def signed(p: list[float]) -> float:
        return a * p[0] + b * p[1] + c * p[2] - d

    def intersect(
        p0: list[float], p1: list[float], s0: float, s1: float
    ) -> list[float]:
        denom = s0 - s1
        if abs(denom) <= 1e-12:
            return list(p0)
        t = max(0.0, min(1.0, s0 / denom))
        return [p0[i] + t * (p1[i] - p0[i]) for i in range(3)]

    out: list[list[float]] = []
    for idx, cur in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)]
        s_cur = signed(cur)
        s_nxt = signed(nxt)
        cur_inside = s_cur <= ROOF_CLIP_EPS_M
        nxt_inside = s_nxt <= ROOF_CLIP_EPS_M
        if cur_inside and nxt_inside:
            out.append(list(nxt))
        elif cur_inside and not nxt_inside:
            out.append(intersect(cur, nxt, s_cur, s_nxt))
        elif not cur_inside and nxt_inside:
            out.append(intersect(cur, nxt, s_cur, s_nxt))
            out.append(list(nxt))
    return _dedupe_3d(out)


def _dedupe_3d(corners: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for corner in corners:
        if not out or sum((corner[i] - out[-1][i]) ** 2 for i in range(3)) > 1e-10:
            out.append(corner)
    if len(out) >= 2 and sum((out[0][i] - out[-1][i]) ** 2 for i in range(3)) <= 1e-10:
        out.pop()
    return out


def _orient_outward(
    corners: list[list[float]], outward_xz: tuple[float, float]
) -> list[list[float]]:
    nx, _ny, nz = newell_normal(corners)
    dot = nx * outward_xz[0] + nz * outward_xz[1]
    if dot < 0.0:
        return list(reversed(corners))
    return corners
