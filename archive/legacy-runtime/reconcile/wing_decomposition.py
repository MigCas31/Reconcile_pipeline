"""Decompose a building footprint into rectangular wings.

Lifted from `reconcile_ext/stages/units.py` so the same logic can be used
both in `reconcile_v3` zones (to split L/T/U buildings into per-wing zones
before roof subparts are consulted) and in the roof pipeline
(`reconcile/roof_algorithms_py`) to scope cluster_oblique_segments per wing.

Algorithm (residential rectilinear assumption):
  1. Detect principal azimuth (longest-edge-dominated orientation, folded into
     [0, 90) so two perpendicular wall families reinforce each other).
  2. Rotate footprint so the principal azimuth aligns with the X axis.
  3. Build a grid from unique X / Y coords of rotated vertices; each grid cell
     whose centre lies inside the polygon is a primitive rectangle.
  4. Greedy-merge adjacent primitive rectangles sharing a full edge.
  5. Rotate rectangles back into world frame.

If the footprint isn't sufficiently axis-aligned (axis-coverage < 70%), the
function returns the footprint as a single rectangle: rectangles are the
wrong primitive for that shape.

This module has no dependency on V3 or roof results — only Shapely + math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon

_GRID_SNAP_TOL_M = 0.12
_CELL_CENTRE_EPS_M = 0.02
_AXIS_TOLERANCE_DEG = 12.0
_MIN_RECT_AREA_M2 = 4.0
_SIDE_BY_SIDE_MIN_SHARED_RATIO = 0.55
_THIN_CAP_MAX_WIDTH_M = 0.75
_THIN_CAP_MIN_OVERLAP_RATIO = 0.45


@dataclass(frozen=True)
class Wing:
    index: int
    polygon: Polygon
    area_m2: float
    role: str


def _principal_azimuth_deg(poly: Polygon) -> float:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    bins: dict[int, float] = {}
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.05:
            continue
        az = math.degrees(math.atan2(dy, dx)) % 90.0
        bin_key = round(az / 2.0) % 45
        for offset, weight in ((0, 1.0), (1, 0.5), (-1, 0.5)):
            key = (bin_key + offset) % 45
            bins[key] = bins.get(key, 0.0) + weight * length
    if not bins:
        return 0.0
    return float(max(bins.items(), key=lambda kv: kv[1])[0]) * 2.0


def _rectilinearity_coverage(poly: Polygon) -> float:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    axis_len = 0.0
    total_len = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-9:
            continue
        total_len += length
        az = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 90.0
        if az <= _AXIS_TOLERANCE_DEG or az >= 90.0 - _AXIS_TOLERANCE_DEG:
            axis_len += length
    if total_len < 1e-9:
        return 0.0
    return axis_len / total_len


def _snap_coords(values: list[float], tol: float) -> list[float]:
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _grid_decompose(poly: Polygon) -> list[tuple[float, float, float, float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    xs = _snap_coords([c[0] for c in coords], _GRID_SNAP_TOL_M)
    ys = _snap_coords([c[1] for c in coords], _GRID_SNAP_TOL_M)
    boxes: list[tuple[float, float, float, float]] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[j], ys[j + 1]
            if (x2 - x1) < 2 * _CELL_CENTRE_EPS_M or (y2 - y1) < 2 * _CELL_CENTRE_EPS_M:
                continue
            centre = Point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if poly.contains(centre):
                boxes.append((x1, y1, x2, y2))
    return boxes


def _merge_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    tol = _GRID_SNAP_TOL_M

    def eq(a: float, b: float) -> bool:
        return abs(a - b) <= tol

    current = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            merged_with = -1
            for j in range(i + 1, len(current)):
                a = current[i]
                b = current[j]
                if eq(a[1], b[1]) and eq(a[3], b[3]):
                    if eq(a[2], b[0]):
                        current[i] = (a[0], a[1], b[2], a[3])
                        merged_with = j
                        break
                    if eq(b[2], a[0]):
                        current[i] = (b[0], b[1], a[2], b[3])
                        merged_with = j
                        break
                if eq(a[0], b[0]) and eq(a[2], b[2]):
                    if eq(a[3], b[1]):
                        current[i] = (a[0], a[1], a[2], b[3])
                        merged_with = j
                        break
                    if eq(b[3], a[1]):
                        current[i] = (b[0], b[1], b[2], a[3])
                        merged_with = j
                        break
            if merged_with >= 0:
                current.pop(merged_with)
                changed = True
                break
    return current


def _box_to_polygon(box: tuple[float, float, float, float]) -> Polygon:
    x1, y1, x2, y2 = box
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _box_width(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0]))


def _box_height(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def _interval_overlap(
    left_min: float, left_max: float, right_min: float, right_max: float
) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def _boxes_same_macro_wing(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    tol = _GRID_SNAP_TOL_M
    left_width = _box_width(left)
    right_width = _box_width(right)
    left_height = _box_height(left)
    right_height = _box_height(right)
    if min(left_width, right_width, left_height, right_height) <= 1e-9:
        return False

    vertical_touch = abs(left[2] - right[0]) <= tol or abs(right[2] - left[0]) <= tol
    if vertical_touch:
        shared = _interval_overlap(left[1], left[3], right[1], right[3])
        if shared / max(left_height, right_height) >= _SIDE_BY_SIDE_MIN_SHARED_RATIO:
            return True

    horizontal_touch = abs(left[3] - right[1]) <= tol or abs(right[3] - left[1]) <= tol
    if horizontal_touch:
        shared = _interval_overlap(left[0], left[2], right[0], right[2])
        thinner = min(left_height, right_height)
        narrower_width = min(left_width, right_width)
        if (
            thinner <= _THIN_CAP_MAX_WIDTH_M
            and shared / narrower_width >= _THIN_CAP_MIN_OVERLAP_RATIO
        ):
            return True

    return False


def _coalesce_macro_wing_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    min_seed_area_m2: float = _MIN_RECT_AREA_M2,
) -> list[Polygon]:
    """Coalesce disjoint rectangle tiles into larger architectural zones.

    `_merge_boxes()` produces an exact disjoint tiling.  That is useful as a
    primitive, but it over-splits cross/intersection footprints: one continuous
    top mass becomes several rectangles around the intersection with another
    mass.  This pass keeps broad side-by-side fragments together and absorbs
    thin cap strips, while leaving deep perpendicular extensions separate.
    """
    if len(boxes) <= 1:
        return [_box_to_polygon(box) for box in boxes]

    adjacency: list[set[int]] = [set() for _ in boxes]
    for i, left in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            right = boxes[j]
            if _boxes_same_macro_wing(left, right):
                adjacency[i].add(j)
                adjacency[j].add(i)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(boxes)):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        components.append(component)

    out: list[Polygon] = []
    for component in components:
        if not any(
            _box_width(boxes[index]) * _box_height(boxes[index]) >= min_seed_area_m2
            for index in component
        ):
            continue
        polys = [_box_to_polygon(boxes[index]) for index in component]
        geom = polys[0]
        for poly in polys[1:]:
            geom = geom.union(poly)
        if geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        if isinstance(geom, Polygon) and geom.area > 0:
            out.append(geom)
    return out


def decompose_to_wings(
    footprint: Polygon, *, min_area_m2: float = _MIN_RECT_AREA_M2
) -> list[Wing]:
    """Decompose ``footprint`` into rectangular wings sorted by area (desc).

    Returns a single-wing list when the footprint isn't rectilinear enough or
    the rectangle decomposition collapses to one wing — callers can rely on
    a non-empty result whenever the input is a valid polygon.
    """
    if footprint.is_empty or footprint.area <= 0:
        return []
    azimuth = _principal_azimuth_deg(footprint)
    rotated = shp_rotate(footprint, -azimuth, origin=(0, 0), use_radians=False)
    coverage_hint = _rectilinearity_coverage(rotated)
    if coverage_hint < 0.70:
        return [Wing(index=0, polygon=footprint, area_m2=footprint.area, role="main")]
    boxes = _grid_decompose(rotated)
    if not boxes:
        return [Wing(index=0, polygon=footprint, area_m2=footprint.area, role="main")]
    merged_boxes = _merge_boxes(boxes)
    rectangles_rotated = _coalesce_macro_wing_boxes(
        merged_boxes, min_seed_area_m2=min_area_m2
    )
    rectangles = [
        shp_rotate(rect, azimuth, origin=(0, 0), use_radians=False)
        for rect in rectangles_rotated
    ]
    qualified = sorted(
        (rect for rect in rectangles if rect.area >= min_area_m2),
        key=lambda r: -r.area,
    )
    if not qualified:
        return [Wing(index=0, polygon=footprint, area_m2=footprint.area, role="main")]
    return [
        Wing(
            index=index,
            polygon=rect,
            area_m2=float(rect.area),
            role="main" if index == 0 else "extension",
        )
        for index, rect in enumerate(qualified)
    ]


def assign_xz_points_to_wings(
    wings: list[Wing], points_xz: list[tuple[float, float]]
) -> list[int | None]:
    """For each (x, z) point return the index of the wing containing it, or
    None when no wing's polygon contains the point. Falls back to the wing
    with the smallest XZ distance when the point is just outside all wings.
    """
    out: list[int | None] = []
    for x, z in points_xz:
        pt = Point(float(x), float(z))
        inside_idx: int | None = None
        for i, wing in enumerate(wings):
            if wing.polygon.contains(pt):
                inside_idx = i
                break
        if inside_idx is not None:
            out.append(inside_idx)
            continue
        best_idx: int | None = None
        best_dist = float("inf")
        for i, wing in enumerate(wings):
            d = wing.polygon.distance(pt)
            if d < best_dist:
                best_dist = d
                best_idx = i
        out.append(best_idx if best_dist <= 0.5 else None)
    return out
