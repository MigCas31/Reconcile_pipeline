"""R2 prototype — decompose the merged footprint into rectangular units.

The primitive is a **rectangle in the building's own frame**, not an arbitrary
sub-polygon from reflex-vertex cuts. Residential buildings are overwhelmingly
assembled from rectangular masses (main rectangle + rectangular extension(s)),
so forcing the output to be rectangles prevents oblique cuts and suppresses
noise from cosmetic wall jogs that don't constitute real part boundaries.

Algorithm:
    1. Detect principal azimuth (longest-edge-dominated orientation).
    2. Rotate footprint so principal azimuth aligns with the X axis.
    3. Build a grid from the unique X and Y coords of rotated vertices.
       Each grid cell that falls inside the polygon is a primitive rectangle.
    4. Greedy-merge adjacent primitive rectangles that share a full edge
       into larger rectangles (shrinks the count without loss of coverage).
    5. Rotate rectangles back to world frame.
    6. Assign rooms by floor-centroid containment; drop zero-room slivers.

Diagnostic only — does not alter V3 geometry.
"""

from __future__ import annotations

import math

from shapely.affinity import rotate as shp_rotate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..constants import FOOTPRINT_SIMPLIFY_TOL_M
from ..models import ExtReport, ExtSnapshot, ExtUnit

# Units smaller than this are dropped (likely artefacts of near-collinear
# vertices after rotation/snapping, or grid-edge slivers).
_MIN_UNIT_AREA_M2 = 4.0

# Tolerance when collapsing near-equal X/Y coordinates onto a shared grid
# line. Larger than numeric noise, smaller than wall-thickness (~0.3m).
_GRID_SNAP_TOL_M = 0.12

# When testing whether a grid cell's centre lies inside the polygon, shrink
# the cell a touch so cells perched exactly on a polygon edge don't flicker.
_CELL_CENTRE_EPS_M = 0.02

# For rectilinearity check: if more than this fraction of edge length is
# non-axis-aligned after rotation, we flag low coverage (the building isn't
# a rectilinear assembly in any single frame — split-wing / curved cases).
_AXIS_TOLERANCE_DEG = 12.0


def _merged_footprint(snapshot: ExtSnapshot) -> Polygon | None:
    polys = []
    for part in snapshot.parts:
        if len(part.footprint_xz) < 3:
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in part.footprint_xz])
        except Exception:
            continue
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)
    for gf in snapshot.gap_fill_footprints:
        if len(gf) < 3:
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in gf])
        except Exception:
            continue
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    simplified = merged.simplify(FOOTPRINT_SIMPLIFY_TOL_M, preserve_topology=True)
    if simplified.is_valid and not simplified.is_empty:
        if simplified.geom_type == "MultiPolygon":
            simplified = max(simplified.geoms, key=lambda g: g.area)
        merged = simplified
    return merged


def _principal_azimuth_deg(poly: Polygon) -> float:
    """Edge-length-weighted dominant orientation, folded into [0, 90).

    Residential footprints have two perpendicular wall families; folding at
    90° makes the two families reinforce each other instead of splitting the
    evidence. Returns the angle (degrees) that, applied as a negative rotation
    to the polygon, aligns the dominant family with the X axis.
    """
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
        # 2°-wide bins; contribute to the bin itself and both neighbours for
        # a smoother peak (avoids knife-edge flips near a bin boundary).
        bin_key = round(az / 2.0) % 45
        for offset, weight in ((0, 1.0), (1, 0.5), (-1, 0.5)):
            bins[(bin_key + offset) % 45] = (
                bins.get((bin_key + offset) % 45, 0.0) + weight * length
            )
    if not bins:
        return 0.0
    return float(max(bins.items(), key=lambda kv: kv[1])[0]) * 2.0


def _rectilinearity_coverage(poly: Polygon) -> float:
    """Fraction of edge length within ``_AXIS_TOLERANCE_DEG`` of an axis.

    Use in the rotated frame as a sanity check: if only 60% of edge length is
    axis-aligned, rectangle decomposition will miss a lot. Falling back to
    treating the whole footprint as one unit is safer.
    """
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
    """Collapse coordinate values within ``tol`` of each other onto a shared mean."""
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
    """Break ``poly`` into axis-aligned rectangles via vertex-coord grid.

    Returns a list of ``(xmin, ymin, xmax, ymax)`` boxes that tile the parts
    of ``poly`` whose grid-cell centre is interior. No two boxes overlap.
    """
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
    """Greedy pairwise merge of adjacent rectangles sharing a full edge.

    Repeats until no pair can merge. Not guaranteed globally minimal, but in
    practice finds the obvious rectangles (main + extension(s)) for typical
    residential shapes.
    """
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
                # Horizontal merge: share a full vertical edge.
                if eq(a[1], b[1]) and eq(a[3], b[3]):
                    if eq(a[2], b[0]):
                        current[i] = (a[0], a[1], b[2], a[3])
                        merged_with = j
                        break
                    if eq(b[2], a[0]):
                        current[i] = (b[0], b[1], a[2], b[3])
                        merged_with = j
                        break
                # Vertical merge: share a full horizontal edge.
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


def _rectangular_faces(footprint: Polygon) -> tuple[list[Polygon], float]:
    """Return (faces_in_world_frame, coverage_fraction).

    Coverage is the ratio ``sum(rect areas) / footprint area`` — callers can
    use it to detect when the rectangle assumption fails badly.
    """
    azimuth = _principal_azimuth_deg(footprint)
    rotated = shp_rotate(footprint, -azimuth, origin=(0, 0), use_radians=False)
    coverage_hint = _rectilinearity_coverage(rotated)
    boxes = _grid_decompose(rotated)
    if not boxes:
        return [footprint], 1.0
    merged = _merge_boxes(boxes)
    covered = sum((b[2] - b[0]) * (b[3] - b[1]) for b in merged)
    coverage = covered / footprint.area if footprint.area > 0 else 0.0
    faces_rotated = [_box_to_polygon(b) for b in merged]
    faces = [
        shp_rotate(f, azimuth, origin=(0, 0), use_radians=False) for f in faces_rotated
    ]
    # Low axis-aligned edge fraction → rectangles can't describe this shape;
    # back off to a single unit rather than emit misleading decomposition.
    if coverage_hint < 0.70:
        return [footprint], 1.0
    return faces, coverage


def _cut_lines_from_faces(faces: list[Polygon], footprint: Polygon) -> list[LineString]:
    """Shared interior edges between rectangles — used as viewer cut overlays."""
    if len(faces) < 2:
        return []
    all_boundaries = unary_union([f.boundary for f in faces])
    # Any boundary length that lies on the footprint's exterior isn't a cut.
    footprint_boundary_buffered = footprint.boundary.buffer(0.02)
    interior = all_boundaries.difference(footprint_boundary_buffered)
    if interior.is_empty:
        return []
    out: list[LineString] = []
    if interior.geom_type == "LineString":
        out.append(interior)
    elif interior.geom_type == "MultiLineString":
        out.extend(interior.geoms)
    elif interior.geom_type == "GeometryCollection":
        for g in interior.geoms:
            if g.geom_type == "LineString":
                out.append(g)
    return [ln for ln in out if ln.length >= 0.2]


def _assign_rooms_to_units(
    faces: list[Polygon], rooms: list
) -> list[tuple[Polygon, list[str]]]:
    assignments: list[tuple[Polygon, list[str]]] = [(f, []) for f in faces]
    for room in rooms:
        fp = room.floor_polygon
        if len(fp) < 3:
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in fp])
        except Exception:
            continue
        if not poly.is_valid or poly.is_empty:
            continue
        centroid = poly.centroid
        best_idx = None
        for i, (face, _) in enumerate(assignments):
            if face.contains(centroid):
                best_idx = i
                break
        if best_idx is None:
            best_overlap = 0.0
            for i, (face, _) in enumerate(assignments):
                ov = face.intersection(poly).area
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = i
        if best_idx is None:
            best_dist = float("inf")
            for i, (face, _) in enumerate(assignments):
                d = face.distance(centroid)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        if best_idx is not None:
            assignments[best_idx][1].append(room.id)
    return assignments


def detect_units(
    snapshot: ExtSnapshot, report: ExtReport
) -> tuple[list[ExtUnit], list[tuple[list[tuple[float, float, float]], str]]]:
    """Produce rectangular unit partition + viewer-ready cut-line records."""
    footprint = _merged_footprint(snapshot)
    if footprint is None:
        return [], []

    faces, _coverage = _rectangular_faces(footprint)
    # Largest rectangle = main; remaining descending by area = extensions.
    faces = [f for f in faces if f.area >= _MIN_UNIT_AREA_M2] or [footprint]
    faces.sort(key=lambda f: -f.area)

    assignments = _assign_rooms_to_units(faces, snapshot.rooms)
    # Drop zero-room rectangles, keep at least the largest so every building
    # ends up with a "main".
    trimmed = [(f, rids) for f, rids in assignments if rids]
    if not trimmed:
        trimmed = assignments[:1]
    assignments = trimmed

    # Keep rectangles' shared edges available as viewer cut overlays. The
    # seam_id tag is synthesised since rectangle cuts don't trace back to a
    # specific reflex-vertex seam one-to-one.
    kept_faces = [f for f, _ in assignments]
    cut_lines = _cut_lines_from_faces(kept_faces, footprint)

    units: list[ExtUnit] = []
    for idx, (face, room_ids) in enumerate(assignments):
        role = "main" if idx == 0 else "extension"
        unit_id = f"unit::{role}::{idx}"
        ring = list(face.exterior.coords)
        if ring and ring[0] == ring[-1]:
            ring = ring[:-1]
        y = snapshot.rooms[0].floor_y if snapshot.rooms else 0.0
        footprint_xz = [(float(x), float(y), float(z)) for x, z in ring]
        rationale = f"{role} rectangle — {len(room_ids)} room(s), {face.area:.1f}m²"
        units.append(
            ExtUnit(
                id=unit_id,
                role=role,
                index=idx,
                room_ids=tuple(room_ids),
                footprint_xz=footprint_xz,
                area_m2=float(face.area),
                seam_ids=(),
                rationale=rationale,
            )
        )

    y = snapshot.rooms[0].floor_y if snapshot.rooms else 0.0
    cut_lines_out: list[tuple[list[tuple[float, float, float]], str]] = []
    for i, line in enumerate(cut_lines):
        (x1, z1) = line.coords[0]
        (x2, z2) = line.coords[-1]
        cut_lines_out.append(
            (
                [(float(x1), float(y), float(z1)), (float(x2), float(y), float(z2))],
                f"rect-cut::{i}",
            )
        )

    return units, cut_lines_out
