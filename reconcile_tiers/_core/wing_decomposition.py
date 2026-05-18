"""Decompose a building footprint into rectangular wings.

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

Pure Shapely + math; no other dependencies.
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
# Two large boxes touching with similar shared-edge length used to merge
# unconditionally — that engulfed clear extensions whose width perpendicular
# to the shared edge was much smaller than the main mass. Require the
# perpendicular-dimension ratio to also be reasonably balanced; otherwise the
# narrower box is an extension, not a co-equal fragment of the same wing.
_FRAGMENT_MIN_PERP_RATIO = 0.4


@dataclass(frozen=True)
class Wing:
    index: int
    polygon: Polygon
    area_m2: float
    role: str
    # Long axis (math convention, mod 180°): the direction the wing's
    # ridge would run for a gable parallel to the long edge. Derived from
    # the wing rectangle's longest edge in `decompose_to_wings`. Used by
    # the per-part roof primitive classifier in Phase 5.
    long_axis_math: float = 0.0
    # Multi-signal fusion provenance (populated by `decompose_to_wings_v2`;
    # v1 leaves these at their defaults so legacy readers ignore them).
    confidence: str = "high"
    disagreement: tuple[str, ...] = ()


def _long_axis_math_deg(poly: Polygon) -> float:
    """Math-convention angle of the polygon's longest edge, mod 180.

    For a Wing rectangle this is the direction of the longer side (= the
    natural ridge axis for a gable parallel to that side). Falls back to
    `_principal_azimuth_deg` when the polygon has no clear longest edge.
    """
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    best_len = 0.0
    best_angle = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= best_len:
            continue
        best_len = length
        best_angle = math.degrees(math.atan2(dy, dx)) % 180.0
    if best_len <= 0.0:
        return _principal_azimuth_deg(poly)
    return best_angle


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
            # Perpendicular dimension to the shared edge is the box width.
            perp_ratio = min(left_width, right_width) / max(left_width, right_width)
            if perp_ratio >= _FRAGMENT_MIN_PERP_RATIO:
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
        # Side-by-side fragments that share their longer edge: keep them
        # together only when the perpendicular height ratio is balanced.
        if shared / max(left_width, right_width) >= _SIDE_BY_SIDE_MIN_SHARED_RATIO:
            perp_ratio = min(left_height, right_height) / max(left_height, right_height)
            if perp_ratio >= _FRAGMENT_MIN_PERP_RATIO:
                return True

    return False


def _coalesce_macro_wing_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    min_seed_area_m2: float = _MIN_RECT_AREA_M2,
) -> list[Polygon]:
    """Coalesce disjoint rectangle tiles into larger architectural zones.

    `_merge_boxes()` produces an exact disjoint tiling. That is useful as a
    primitive, but it over-splits cross/intersection footprints: one continuous
    top mass becomes several rectangles around the intersection with another
    mass. This pass keeps broad side-by-side fragments together and absorbs
    thin cap strips, while leaving deep perpendicular extensions separate.

    Thin connectors that bridge two large boxes (e.g. a 0.4 m strip between two
    full-storey wings of a + or T-shaped building) are NOT used to merge the
    two large boxes — instead the connector is assigned to whichever large box
    it touches with the largest shared edge, and the two large boxes stay
    separate wings.
    """
    if len(boxes) <= 1:
        return [_box_to_polygon(box) for box in boxes]

    is_thin = [
        min(_box_width(box), _box_height(box)) <= _THIN_CAP_MAX_WIDTH_M for box in boxes
    ]

    adjacency: list[set[int]] = [set() for _ in boxes]
    for i, left in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            right = boxes[j]
            if not _boxes_same_macro_wing(left, right):
                continue
            # If both ends are thin, treat them as the cap-of-cap case and
            # merge as before. Otherwise a thin connector between two large
            # boxes must NOT bridge them.
            if is_thin[i] and not is_thin[j]:
                continue
            if is_thin[j] and not is_thin[i]:
                continue
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

    # Assign each thin singleton box to the large component it touches most.
    # `_boxes_same_macro_wing` uses an asymmetric thin-cap rule that fires when
    # the thinner box is ≤ 0.75 m wide, so any thin box that touches a large
    # one has a defined connection — pick the largest shared overlap.
    def _shared_edge_length(thin_box, other_box) -> float:
        tol = _GRID_SNAP_TOL_M
        # Vertical edge match
        if (
            abs(thin_box[2] - other_box[0]) <= tol
            or abs(other_box[2] - thin_box[0]) <= tol
        ):
            return _interval_overlap(
                thin_box[1], thin_box[3], other_box[1], other_box[3]
            )
        # Horizontal edge match
        if (
            abs(thin_box[3] - other_box[1]) <= tol
            or abs(other_box[3] - thin_box[1]) <= tol
        ):
            return _interval_overlap(
                thin_box[0], thin_box[2], other_box[0], other_box[2]
            )
        return 0.0

    box_to_component = {
        idx: c_idx for c_idx, comp in enumerate(components) for idx in comp
    }
    moved = True
    while moved:
        moved = False
        for c_idx, comp in enumerate(components):
            if len(comp) != 1:
                continue
            only = comp[0]
            if not is_thin[only]:
                continue
            best_target = -1
            best_overlap = 0.0
            for other_c_idx, other_comp in enumerate(components):
                if other_c_idx == c_idx or not other_comp:
                    continue
                # Prefer absorbing into a large-box component.
                if all(is_thin[idx] for idx in other_comp):
                    continue
                overlap = max(
                    _shared_edge_length(boxes[only], boxes[idx]) for idx in other_comp
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_target = other_c_idx
            if best_target >= 0 and best_overlap > _GRID_SNAP_TOL_M:
                components[best_target].append(only)
                components[c_idx] = []
                box_to_component[only] = best_target
                moved = True
                break
    components = [comp for comp in components if comp]

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
        return [
            Wing(
                index=0,
                polygon=footprint,
                area_m2=footprint.area,
                role="main",
                long_axis_math=_long_axis_math_deg(footprint),
            )
        ]
    boxes = _grid_decompose(rotated)
    if not boxes:
        return [
            Wing(
                index=0,
                polygon=footprint,
                area_m2=footprint.area,
                role="main",
                long_axis_math=_long_axis_math_deg(footprint),
            )
        ]
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
        return [
            Wing(
                index=0,
                polygon=footprint,
                area_m2=footprint.area,
                role="main",
                long_axis_math=_long_axis_math_deg(footprint),
            )
        ]
    return [
        Wing(
            index=index,
            polygon=rect,
            area_m2=float(rect.area),
            role="main" if index == 0 else "extension",
            long_axis_math=_long_axis_math_deg(rect),
        )
        for index, rect in enumerate(qualified)
    ]


WING_OVERLAP_EPS_M2 = 1e-6


_WINGS_CACHE: dict[str, list[Wing]] = {}


def compute_wings_for_model(model) -> list[Wing]:
    """Decompose ``model``'s building footprint into wings (cached per uuid).

    Returns [] when the footprint can't be derived. A single-wing list when the
    footprint isn't rectilinear enough — callers can treat that as "no wing
    constraint" (a single-wing intersection is a no-op).
    """
    cached = _WINGS_CACHE.get(model.uuid)
    if cached is not None:
        return cached
    from reconcile_tiers._core.shapely2 import make_valid_polygon
    from reconcile_tiers.roof.footprint import build_building_footprint

    footprint = build_building_footprint(model)
    if footprint is None or len(footprint.polygon_xz) < 3:
        _WINGS_CACHE[model.uuid] = []
        return []
    poly = make_valid_polygon(Polygon(footprint.polygon_xz))
    if poly is None or poly.is_empty:
        _WINGS_CACHE[model.uuid] = []
        return []
    wings = decompose_to_wings(poly)
    _WINGS_CACHE[model.uuid] = wings
    return wings


def wing_polygon_for_xz_polygon(xz_poly: Polygon, wings: list[Wing]) -> Polygon | None:
    """Pick the wing polygon with the largest XZ overlap with ``xz_poly``.

    Returns the single wing polygon directly when only one wing exists.
    Returns None when no wing overlaps the input.
    """
    if not wings or xz_poly is None or xz_poly.is_empty:
        return None
    if len(wings) == 1:
        return wings[0].polygon
    best_poly = None
    best_overlap = 0.0
    for wing in wings:
        overlap = wing.polygon.intersection(xz_poly).area
        if overlap > best_overlap:
            best_overlap = overlap
            best_poly = wing.polygon
    return best_poly


def wing_polygon_for_room(room, wings: list[Wing]) -> Polygon | None:
    """Pick the wing polygon with the largest XZ overlap with ``room.floor_polygon``."""
    if not wings or len(room.floor_polygon) < 3:
        return None
    if len(wings) == 1:
        return wings[0].polygon
    from reconcile_tiers._core.shapely2 import make_valid_polygon

    room_poly = make_valid_polygon(
        Polygon([(float(p[0]), float(p[2])) for p in room.floor_polygon])
    )
    if room_poly is None or room_poly.is_empty:
        return None
    return wing_polygon_for_xz_polygon(room_poly, wings)


def filter_by_wing_xz(items, wing_poly: Polygon | None, xz_polygon_fn):
    """Keep items whose XZ polygon (via ``xz_polygon_fn``) overlaps ``wing_poly``.

    Returns the input list unchanged when no wing constraint applies.
    """
    if wing_poly is None or wing_poly.is_empty:
        return list(items)
    out = []
    for item in items:
        poly = xz_polygon_fn(item)
        if poly is None or poly.is_empty:
            continue
        if poly.intersection(wing_poly).area > WING_OVERLAP_EPS_M2:
            out.append(item)
    return out


def clip_3d_polygon_to_wing(
    corners: list, plane, wing_poly: Polygon | None
) -> list[list[list[float]]]:
    """Intersect a 3D polygon's XZ projection with ``wing_poly`` and re-emit 3D corners
    by evaluating ``plane.y_at(x, z)`` for each new vertex.

    Returns a list of polygons (each a list of [x, y, z] corner triplets). The
    intersection can split a polygon into several pieces. Returns the original
    corners (in a single-element list) when no wing constraint applies, the
    plane is degenerate, or any vertex falls outside the plane domain.
    """
    if wing_poly is None or wing_poly.is_empty or len(corners) < 3:
        return [[list(c) for c in corners]] if len(corners) >= 3 else []
    from reconcile_tiers._core.shapely2 import make_valid_polygon

    base = make_valid_polygon(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    if base is None or base.is_empty:
        return []
    intersected = base.intersection(wing_poly)
    if intersected.is_empty:
        return []
    parts: list[Polygon] = []
    if intersected.geom_type == "Polygon":
        parts = [intersected]
    elif intersected.geom_type == "MultiPolygon":
        parts = [
            p for p in intersected.geoms if isinstance(p, Polygon) and p.area > 1e-9
        ]
    else:
        return []
    out: list[list[list[float]]] = []
    for part in parts:
        ring = list(part.exterior.coords)
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        new_corners: list[list[float]] = []
        ok = True
        for x, z in ring:
            y = plane.y_at(float(x), float(z)) if plane is not None else None
            if y is None:
                ok = False
                break
            new_corners.append([float(x), float(y), float(z)])
        if ok and len(new_corners) >= 3:
            out.append(new_corners)
    return out


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
