"""Cross-cutting utilities used across the ceiling_helpers sub-modules.

Small helpers that don't fit any of the other thematic modules and that are
needed by more than one sibling. Keeps the dependency graph between the
themed sub-modules acyclic.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.assemble.synthesis import _half_plane_polygon
from reconcile_tiers.build_internals.constants import (
    _CEILING_POLYGON_PLANARITY_TOL_M,
    ROOM_OBLIQUE_FLAT_LID_TOL_M,
)


def _ceiling_polygon_is_planar(corners: list[list[float]]) -> bool:
    """True if the wall-derived ceiling_polygon corners are coplanar within
    `_CEILING_POLYGON_PLANARITY_TOL_M`."""
    plane = Plane.fit(corners)
    if isinstance(plane, FitFailure):
        return False
    max_residual = max(
        abs(
            plane.a * float(p[0])
            + plane.b * float(p[1])
            + plane.c * float(p[2])
            - plane.d
        )
        for p in corners
    )
    return max_residual <= _CEILING_POLYGON_PLANARITY_TOL_M


def _raw_plane_xz_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    from shapely.geometry import Polygon as _ShPoly

    try:
        poly = _ShPoly([(float(p[0]), float(p[2])) for p in corners])
    except Exception:
        return 0.0
    if not poly.is_valid:
        poly = poly.buffer(0)
    return float(poly.area) if not poly.is_empty else 0.0


def _room_oblique_raw_coverage(room) -> tuple[float, float]:
    from math import atan2, degrees, hypot

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    floor_area = _raw_plane_xz_area(room.floor_polygon)
    if floor_area <= 1e-6:
        return 0.0, 0.0
    oblique_polys = []
    oblique_area_sum = 0.0
    for raw in room.raw_ceiling_planes:
        plane = Plane.fit(raw.corners)
        if isinstance(plane, FitFailure):
            continue
        incl = degrees(
            atan2(hypot(float(plane.a), float(plane.c)), abs(float(plane.b)))
        )
        if not 5.0 <= incl <= 75.0:
            continue
        oblique_area_sum += _raw_plane_xz_area(raw.corners)
        try:
            poly = Polygon([(float(p[0]), float(p[2])) for p in raw.corners])
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 1e-9:
            oblique_polys.append(poly)
    if not oblique_polys:
        return 0.0, 0.0
    try:
        oblique_area = float(unary_union(oblique_polys).area)
    except Exception:
        oblique_area = oblique_area_sum
    return oblique_area, oblique_area / floor_area


def _story_labels(has_basement: bool, story_mean_ys: list[float]) -> list[str]:
    from reconcile_tiers.extract.stories import SPLIT_LEVEL_DY_M

    n = len(story_mean_ys)
    if n == 0:
        return []
    gaps = [story_mean_ys[i + 1] - story_mean_ys[i] for i in range(n - 1)]
    labels: list[str] = []
    full_floor_counter = 0
    full_floor_names = ["Stueetage"] + [f"{i}. sal" for i in range(1, n)]
    for i in range(n):
        if i == 0 and has_basement:
            labels.append("Kælder")
        elif i == 1 and has_basement:
            labels.append(full_floor_names[full_floor_counter])
            full_floor_counter += 1
        elif i > 0 and gaps[i - 1] < SPLIT_LEVEL_DY_M:
            labels.append("Halvplan")
        else:
            labels.append(
                full_floor_names[min(full_floor_counter, len(full_floor_names) - 1)]
            )
            full_floor_counter += 1
    return labels


def _parse_room_idx(arrangement_cell_id: str | None) -> int | None:
    if not arrangement_cell_id:
        return None
    parts = arrangement_cell_id.split(":")
    if len(parts) >= 5 and parts[2] == "room":
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


def _flat_lid_y(room) -> float | None:
    if room.ceiling_type not in {"flat", None}:
        return None
    y_values: list[float] = []
    if len(room.ceiling_flat_polygon) >= 3:
        y_values.extend(float(point[1]) for point in room.ceiling_flat_polygon)
    if len(room.ceiling_polygon) >= 3:
        ys = [float(point[1]) for point in room.ceiling_polygon]
        if max(ys) - min(ys) <= ROOM_OBLIQUE_FLAT_LID_TOL_M:
            y_values.extend(ys)
    return max(y_values) if y_values else None


def _hybrid_room_lid_y(room) -> float | None:
    """Best-available flat ceiling Y for a hybrid room.

    Prefers the eave height from `extract/ceilings.py` (already authoritative
    for hybrid kneewall rooms); falls back to the mean Y of `ceiling_polygon`
    so legacy rooms without `ceiling_eave_height` still resolve a lid.
    """
    eave = getattr(room, "ceiling_eave_height", None)
    if eave is not None:
        return float(eave)
    if len(room.ceiling_polygon) >= 3:
        return sum(float(p[1]) for p in room.ceiling_polygon) / len(
            room.ceiling_polygon
        )
    return None


def _upper_floor_coverage_xz(room, model):
    from reconcile_tiers.build_internals.polygon_utils import _room_floor_xz_polygon

    room_poly = _room_floor_xz_polygon(room)
    if room_poly is None:
        return None, None

    from shapely.ops import unary_union

    upper_polys = [
        poly
        for other in model.rooms
        if other.story > room.story
        if (poly := _room_floor_xz_polygon(other)) is not None
    ]
    if not upper_polys:
        return room_poly, None
    coverage = unary_union(upper_polys).intersection(room_poly)
    if coverage.is_empty:
        return room_poly, None
    return room_poly, coverage


def _vec3_at_y_from_polygon(poly, y: float) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(x), float(y), float(z)] for x, z in coords]


def _lower_plane_half(room_poly, plane, other_plane, wing_poly=None):
    """Half-plane in XZ where ``plane`` is below ``other_plane``, intersected
    with ``room_poly`` (and optionally ``wing_poly``)."""
    from shapely.geometry import box

    if abs(float(plane.b)) < 1e-9 or abs(float(other_plane.b)) < 1e-9:
        return None
    lhs_a = -float(plane.a) / float(plane.b) + float(other_plane.a) / float(
        other_plane.b
    )
    lhs_c = -float(plane.c) / float(plane.b) + float(other_plane.c) / float(
        other_plane.b
    )
    lhs_k = float(plane.d) / float(plane.b) - float(other_plane.d) / float(
        other_plane.b
    )
    min_x, min_z, max_x, max_z = room_poly.bounds
    bbox = box(min_x - 1.0, min_z - 1.0, max_x + 1.0, max_z + 1.0)
    half = _half_plane_polygon(bbox, lhs_a, lhs_c, -lhs_k)
    if half is None or half.is_empty:
        return None
    geom = room_poly.intersection(half)
    if wing_poly is not None and not wing_poly.is_empty and not geom.is_empty:
        geom = geom.intersection(wing_poly)
    return geom
