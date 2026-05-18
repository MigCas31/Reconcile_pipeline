"""Raw-ceiling-plane snapping helpers used by `reconcile_tiers.build`.

Snaps raw scan ceilings onto reconciled oblique gables (so the room's
sloped ceiling sits coplanar with the shell), detects raws that cross an
upper-floor slab (so they don't poke through), and applies axis-aligned
priors when enabled.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.assemble.ceiling_painter import CeilingCandidate
from reconcile_tiers.build_internals.constants import (
    FLAT_HORIZONTAL_Y_SPAN_M,
    RAW_TO_OBLIQUE_MAX_AZIMUTH_DELTA_DEG,
    RAW_TO_OBLIQUE_MAX_INCL_DELTA_DEG,
    RAW_TO_OBLIQUE_MAX_Y_DELTA_M,
    RAW_UPPER_SLAB_MIN_OVERLAP_M2,
    RAW_UPPER_SLAB_MIN_OVERLAP_RATIO,
    RAW_UPPER_SLAB_VERTICAL_EPS_M,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingSource


def _snap_raw_to_oblique(corners: list[list[float]], obliques) -> list[list[float]]:
    """If the raw plane closely matches a reconciled oblique, lift the corners onto
    that oblique's plane so the room's sloped ceiling is co-planar with the gable
    shell. Otherwise return the original corners unchanged.
    """
    from math import atan2, degrees, hypot

    from reconcile_tiers.roof.geometry import angle_diff_deg

    if len(corners) < 3 or not obliques:
        return corners
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return corners
    raw_incl = degrees(atan2(hypot(plane.a, plane.c), abs(plane.b)))
    raw_azimuth = degrees(atan2(-plane.a, -plane.c)) % 360.0

    best: tuple[float, object] | None = None
    for ob in obliques:
        az_delta = angle_diff_deg(raw_azimuth, ob.cluster.avg_azimuth)
        if az_delta > RAW_TO_OBLIQUE_MAX_AZIMUTH_DELTA_DEG:
            continue
        incl_delta = abs(raw_incl - ob.cluster.avg_incl)
        if incl_delta > RAW_TO_OBLIQUE_MAX_INCL_DELTA_DEG:
            continue
        score = az_delta + incl_delta
        if best is None or score < best[0]:
            best = (score, ob)
    if best is None:
        return corners

    target_plane = best[1].plane
    snapped: list[list[float]] = []
    deltas: list[float] = []
    for x, _y, z in corners:
        y = target_plane.y_at(float(x), float(z))
        if y is None:
            return corners
        deltas.append(abs(float(y) - float(_y)))
        snapped.append([float(x), float(y), float(z)])
    if deltas and max(deltas) > RAW_TO_OBLIQUE_MAX_Y_DELTA_M:
        return corners
    return snapped


def _is_horizontal(corners: list[list[float]]) -> bool:
    if len(corners) < 3:
        return False
    ys = [float(p[1]) for p in corners]
    return max(ys) - min(ys) <= FLAT_HORIZONTAL_Y_SPAN_M


def _raw_ceiling_crosses_upper_floor_slab(
    room,
    raw_corners: list[list[float]],
    model: BuildingModel,
) -> bool:
    if len(raw_corners) < 3:
        return False

    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon

    raw_plane = Plane.fit(raw_corners)
    if isinstance(raw_plane, FitFailure):
        return False
    raw_poly = make_valid_polygon(
        Polygon([(float(p[0]), float(p[2])) for p in raw_corners])
    )
    if raw_poly is None or raw_poly.is_empty or raw_poly.area <= 0.0:
        return False

    slab_polys = []
    slab_ys: list[float] = []
    for other in model.rooms:
        if other.story <= room.story or len(other.floor_polygon) < 3:
            continue
        slab_poly = make_valid_polygon(
            Polygon([(float(p[0]), float(p[2])) for p in other.floor_polygon])
        )
        if slab_poly is None or slab_poly.is_empty:
            continue
        slab_polys.append(slab_poly)
        slab_ys.append(
            sum(float(p[1]) for p in other.floor_polygon) / len(other.floor_polygon)
        )
    if not slab_polys:
        return False

    for slab_poly, slab_y in zip(slab_polys, slab_ys, strict=True):
        overlap = raw_poly.intersection(slab_poly)
        if overlap.is_empty:
            continue
        overlap_area = float(overlap.area)
        if overlap_area < RAW_UPPER_SLAB_MIN_OVERLAP_M2:
            continue
        if overlap_area / float(raw_poly.area) < RAW_UPPER_SLAB_MIN_OVERLAP_RATIO:
            continue

        parts = (
            [overlap]
            if overlap.geom_type == "Polygon"
            else list(getattr(overlap, "geoms", []))
        )
        sample_points: list[tuple[float, float]] = []
        for part in parts:
            if part.is_empty or part.geom_type != "Polygon":
                continue
            sample_points.extend((float(x), float(z)) for x, z in part.exterior.coords)
            rp = part.representative_point()
            sample_points.append((float(rp.x), float(rp.y)))
        if not sample_points:
            continue

        for x, z in sample_points:
            raw_y = raw_plane.y_at(x, z)
            if raw_y is not None and raw_y >= slab_y - RAW_UPPER_SLAB_VERTICAL_EPS_M:
                return True

    return False


def _snap_raw_ceiling_corners(
    corners: list[list[float]],
    wall_axis_math: float,
) -> list[list[float]]:
    """Layer 3 emission-time prior for `tier-ceiling-raw` pieces.

    Fits a plane to the raw scan corners; if its slope direction is close to
    an axis-aligned target, rotates the plane onto that target and recomputes
    corner Y. Falls back to the original corners on near-flat / out-of-tol /
    fit-failure cases.
    """
    if len(corners) < 3:
        return corners
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return corners
    from reconcile_tiers._core.wall_axis import snap_corners_and_plane_to_axis

    snapped = snap_corners_and_plane_to_axis(
        corners, plane, wall_axis_math=wall_axis_math
    )
    if snapped is None:
        return corners
    new_corners, _new_plane = snapped
    return new_corners


def _fit_candidate(
    corners: list[list[float]],
    source: CeilingSource,
    locator_id: str,
    arrangement_cell_id: str | None = None,
    story: int | None = None,
) -> CeilingCandidate | None:
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return None
    return CeilingCandidate(
        corners=[[float(p[0]), float(p[1]), float(p[2])] for p in corners],
        plane=plane,
        source=source,
        locator_id=locator_id,
        arrangement_cell_id=arrangement_cell_id,
        story=story,
    )
