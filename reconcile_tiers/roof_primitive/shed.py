"""Axis-snapped oblique primitive over one building part."""

from __future__ import annotations

from math import atan2, cos, degrees, hypot, radians, sin, sqrt, tan

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers._core.shapely2 import make_valid_polygon
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster
from reconcile_tiers.roof_primitive.types import ObliqueParams, ShedParams

MIN_OBLIQUE_INCL_DEG = 5.0
MAX_OBLIQUE_INCL_DEG = 75.0
MIN_WING_COVERAGE_RATIO = 0.55
SAME_NORMAL_TOL = 0.06
SAME_HEIGHT_TOL_M = 0.35


def classify_oblique(
    wing,
    wing_index: int,
    model: BuildingModel,
    *,
    exclude_room_indices: set[int] | None = None,
) -> ObliqueParams | None:
    """Return ObliqueParams when ceiling evidence supports one wing plane.

    The classifier is deliberately conservative: it only fires when oblique
    ceiling polygons/raw planes with a common plane cover most of the wing in
    XZ. It keeps the observed height and inclination, but snaps the plane's
    horizontal gradient to the wing's long or short axis so raw scan yaw does
    not leak into the synthesised roof.
    """
    excluded = exclude_room_indices or set()
    candidates = []
    story_votes: dict[int, int] = {}
    wing_poly = wing.polygon
    wing_area = max(float(wing_poly.area), 1e-9)

    for room in model.rooms:
        if room.index in excluded or len(room.floor_polygon) < 3:
            continue
        room_poly = _room_xz_polygon(room.floor_polygon)
        if room_poly is None or not wing_poly.intersects(room_poly):
            continue
        if wing_poly.intersection(room_poly).area / max(room_poly.area, 1e-9) < 0.5:
            continue
        story_votes[room.story] = story_votes.get(room.story, 0) + 1

        if room.ceiling_type == "sloped" and len(room.ceiling_polygon) >= 3:
            _append_candidate(candidates, room.ceiling_polygon)
        for raw in room.raw_ceiling_planes:
            _append_candidate(candidates, raw.corners)

    if not candidates:
        return None

    best_plane: Plane | None = None
    best_geom = None
    best_area = 0.0
    for plane, _poly in candidates:
        same_polys = [
            other_poly
            for other_plane, other_poly in candidates
            if _same_plane_family(plane, other_plane, wing_poly)
        ]
        if not same_polys:
            continue
        try:
            union = unary_union(same_polys)
            clipped = union.intersection(wing_poly)
        except Exception:
            continue
        area = (
            float(clipped.area) if clipped is not None and not clipped.is_empty else 0.0
        )
        if area > best_area:
            best_area = area
            best_plane = plane
            best_geom = clipped

    if best_plane is None or best_geom is None:
        return None
    if best_area / wing_area < MIN_WING_COVERAGE_RATIO:
        return None

    dominant_story = (
        max(story_votes, key=lambda s: (story_votes[s], s)) if story_votes else 0
    )
    coords = _ring_without_close(wing_poly.exterior.coords)
    if len(coords) < 3:
        return None
    snapped = _snap_plane_to_wing_axis(best_plane, best_geom, wing)
    if snapped is None:
        return None
    snapped_plane, slope_axis_math_deg, eave_axis_math_deg, incl_deg = snapped
    return ObliqueParams(
        wing_index=wing_index,
        plane=snapped_plane,
        polygon_xz=tuple((float(x), float(z)) for x, z in coords),
        dominant_story=dominant_story,
        slope_axis_math_deg=slope_axis_math_deg,
        eave_axis_math_deg=eave_axis_math_deg,
        incl_deg=incl_deg,
    )


def classify_shed(
    wing,
    wing_index: int,
    model: BuildingModel,
    *,
    exclude_room_indices: set[int] | None = None,
) -> ShedParams | None:
    """Compatibility wrapper for the old named shed primitive."""

    return classify_oblique(
        wing,
        wing_index,
        model,
        exclude_room_indices=exclude_room_indices,
    )


def synthesise_oblique(params: ObliqueParams) -> list[ObliqueSurface]:
    corners = []
    for x, z in params.polygon_xz:
        y = params.plane.y_at(float(x), float(z))
        if y is None:
            return []
        corners.append([float(x), float(y), float(z)])
    if len(corners) < 3:
        return []

    incl = _plane_inclination(params.plane)
    az = _plane_azimuth(params.plane)
    ref_pt = [
        sum(c[0] for c in corners) / len(corners),
        sum(c[1] for c in corners) / len(corners),
        sum(c[2] for c in corners) / len(corners),
    ]
    return [
        ObliqueSurface(
            corners=corners,
            plane=params.plane,
            cluster=RoofCluster(
                segments=[],
                avg_incl=incl,
                avg_azimuth=az,
                ref_pt=ref_pt,
            ),
            dominant_story=params.dominant_story,
            ridge=_ridge_for(corners, az),
        )
    ]


def synthesise_shed(params: ShedParams) -> list[ObliqueSurface]:
    """Compatibility wrapper for the old named shed primitive."""

    return synthesise_oblique(params)


def _append_candidate(
    candidates: list[tuple[Plane, Polygon]],
    corners: list[list[float]],
) -> None:
    if len(corners) < 3:
        return
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return
    incl = _plane_inclination(plane)
    if incl < MIN_OBLIQUE_INCL_DEG or incl > MAX_OBLIQUE_INCL_DEG:
        return
    poly = _room_xz_polygon(corners)
    if poly is not None:
        candidates.append((plane, poly))


def _room_xz_polygon(corners: list[list[float]]) -> Polygon | None:
    try:
        return make_valid_polygon(
            Polygon([(float(p[0]), float(p[2])) for p in corners])
        )
    except Exception:
        return None


def _same_plane_family(a: Plane, b: Plane, sample_poly: Polygon) -> bool:
    dot = a.a * b.a + a.b * b.b + a.c * b.c
    if abs(1.0 - dot) > SAME_NORMAL_TOL:
        return False
    point = sample_poly.representative_point()
    ay = a.y_at(float(point.x), float(point.y))
    by = b.y_at(float(point.x), float(point.y))
    if ay is None or by is None:
        return False
    return abs(ay - by) <= SAME_HEIGHT_TOL_M


def _snap_plane_to_wing_axis(
    plane: Plane,
    observed_geom,
    wing,
) -> tuple[Plane, float, float, float] | None:
    grad_x = -plane.a / plane.b
    grad_z = -plane.c / plane.b
    grad_len = hypot(grad_x, grad_z)
    if grad_len <= 1e-9:
        return None

    long_axis = _canonical_axis_deg(float(getattr(wing, "long_axis_math", 0.0)))
    slope_axis = _nearest_axis_deg(grad_x, grad_z, (long_axis, long_axis + 90.0))
    axis_x, axis_z = _axis_unit(slope_axis)
    sign = 1.0 if grad_x * axis_x + grad_z * axis_z >= 0.0 else -1.0
    incl_deg = _plane_inclination(plane)
    snapped_slope = tan(radians(incl_deg))
    snapped_grad_x = sign * snapped_slope * axis_x
    snapped_grad_z = sign * snapped_slope * axis_z

    ref = observed_geom.representative_point()
    ref_x = float(ref.x)
    ref_z = float(ref.y)
    ref_y = plane.y_at(ref_x, ref_z)
    if ref_y is None:
        return None

    snapped_plane = _plane_from_gradient(
        ref_x=ref_x,
        ref_y=float(ref_y),
        ref_z=ref_z,
        grad_x=snapped_grad_x,
        grad_z=snapped_grad_z,
    )
    return (
        snapped_plane,
        _canonical_axis_deg(slope_axis),
        _canonical_axis_deg(slope_axis + 90.0),
        incl_deg,
    )


def _nearest_axis_deg(
    grad_x: float,
    grad_z: float,
    axes_deg: tuple[float, ...],
) -> float:
    grad_len = hypot(grad_x, grad_z)
    if grad_len <= 1e-9:
        return _canonical_axis_deg(axes_deg[0])
    ux = grad_x / grad_len
    uz = grad_z / grad_len
    return _canonical_axis_deg(
        max(
            axes_deg,
            key=lambda angle: abs(
                ux * _axis_unit(angle)[0] + uz * _axis_unit(angle)[1]
            ),
        )
    )


def _axis_unit(axis_math_deg: float) -> tuple[float, float]:
    angle = radians(axis_math_deg)
    return (cos(angle), sin(angle))


def _canonical_axis_deg(axis_math_deg: float) -> float:
    return float(axis_math_deg % 180.0)


def _plane_from_gradient(
    *,
    ref_x: float,
    ref_y: float,
    ref_z: float,
    grad_x: float,
    grad_z: float,
) -> Plane:
    # Plane form is a*x + b*y + c*z = d, with y = d - a*x - c*z when b=1.
    a = -grad_x
    b = 1.0
    c = -grad_z
    d = ref_y - grad_x * ref_x - grad_z * ref_z
    norm = sqrt(a * a + b * b + c * c)
    return Plane(a=a / norm, b=b / norm, c=c / norm, d=d / norm)


def _ring_without_close(coords) -> list[tuple[float, float]]:
    ring = list(coords)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return [(float(x), float(z)) for x, z in ring]


def _plane_azimuth(plane: Plane) -> float:
    return degrees(atan2(-plane.a, -plane.c)) % 360.0


def _plane_inclination(plane: Plane) -> float:
    return degrees(atan2(hypot(plane.a, plane.c), abs(plane.b)))


def _ridge_for(corners: list[list[float]], avg_azimuth: float) -> dict[str, float]:
    from math import cos, radians, sin

    angle = radians(avg_azimuth + 90.0)
    rx = sin(angle)
    rz = cos(angle)
    projections = [float(point[0]) * rx + float(point[2]) * rz for point in corners]
    return {"x": rx, "z": rz, "min": min(projections), "max": max(projections)}
