"""Step 7c of the assembly: Ridges from RoofSurface pairs.

A `Ridge` is the mathematical intersection of two non-parallel
RoofSurface planes, clipped to the XZ overlap of the two surface
polygons. No tolerance: the plane intersection is exact (float
precision); the clipping is shapely intersection of the two XZ
polygons against the line.

Phase C-3c emits one Ridge per pair of non-parallel surfaces whose
intersection line passes through both surfaces' XZ extents. Coplanar
fragmentation in the input (multiple RoofSurfaces sharing a plane,
left over from Phase C-3b) survives unchanged; merging by Ridge
sharing is Phase C-3d.
"""

from __future__ import annotations

from itertools import combinations

from reconcile_tiers.payload.schema import Vec3
from reconcile_tiers.twin._geometry import (
    FLOAT_EPS,
    plane_plane_intersection,
)
from reconcile_tiers.twin.types import (
    Evidence,
    Provenance,
    Ridge,
    Roof,
    RoofSurface,
)


def ridges_for_roof(roof: Roof, *, building_uuid: str) -> tuple[Ridge, ...]:
    """Compute Ridges between every pair of non-parallel RoofSurfaces."""
    out: list[Ridge] = []
    for idx, (a, b) in enumerate(combinations(roof.surfaces, 2)):
        ridge = _ridge_between(a, b, idx=idx, roof_id=roof.id)
        if ridge is not None:
            out.append(ridge)
    return tuple(out)


def _ridge_between(
    a: RoofSurface, b: RoofSurface, *, idx: int, roof_id: str
) -> Ridge | None:
    from shapely.geometry import LineString

    intersection = plane_plane_intersection(a.plane, b.plane)
    if intersection is None:
        return None
    point, direction = intersection
    dx, _, dz = direction

    poly_a = _xz_polygon(a)
    poly_b = _xz_polygon(b)
    if poly_a is None or poly_b is None:
        return None

    # Build a 2D line in XZ that extends well beyond either polygon's
    # bounding box. Shapely will clip it to the polygon overlap.
    bounds = poly_a.bounds[:4] + poly_b.bounds[:4]
    span = max(
        abs(bounds[2] - bounds[0]),
        abs(bounds[3] - bounds[1]),
        abs(bounds[6] - bounds[4]),
        abs(bounds[7] - bounds[5]),
    )
    if span < FLOAT_EPS:
        return None
    t = span * 4.0
    line_xz = LineString(
        [
            (point.x - dx * t, point.z - dz * t),
            (point.x + dx * t, point.z + dz * t),
        ]
    )
    # The ridge is where the planes meet *and* the surfaces are
    # spatially adjacent. Clip the line to the union of the two surface
    # XZ polygons; the longest piece that intersects BOTH of them is
    # the ridge. Two truly adjacent surfaces share their boundary, so
    # the line touches both. Two surfaces that don't meet in plan view
    # yield a segment that's entirely inside one or neither.
    union_poly = poly_a.union(poly_b)
    if union_poly.is_empty:
        return None
    seg_in_union = line_xz.intersection(union_poly)
    if seg_in_union.is_empty or seg_in_union.geom_type not in (
        "LineString",
        "MultiLineString",
    ):
        return None
    seg = _longest_segment_touching_both(seg_in_union, poly_a, poly_b)
    if seg is None or seg.is_empty:
        return None

    if seg.geom_type == "MultiLineString":
        # Pick the longest contiguous piece.
        seg = max(seg.geoms, key=lambda g: g.length)
    coords = list(seg.coords)
    if len(coords) < 2:
        return None
    (x1, z1), (x2, z2) = coords[0], coords[-1]
    if abs(x2 - x1) < FLOAT_EPS and abs(z2 - z1) < FLOAT_EPS:
        return None

    # Lift the 2D endpoints back to 3D using either plane (they lie on
    # both planes by construction, so either gives the same Y at that XZ).
    y1 = _y_at_xz(a, x1, z1)
    y2 = _y_at_xz(a, x2, z2)
    if y1 is None or y2 is None:
        return None

    member_ids = (a.id, b.id)
    evidence = Evidence(
        provenance=Provenance(kind="computed", source="plane_plane_intersection"),
        geometry=(Vec3(x=x1, y=y1, z=z1), Vec3(x=x2, y=y2, z=z2)),
        parents=member_ids,
    )
    return Ridge(
        id=f"{roof_id}::ridge::{idx}",
        endpoint_a=Vec3(x=float(x1), y=float(y1), z=float(z1)),
        endpoint_b=Vec3(x=float(x2), y=float(y2), z=float(z2)),
        member_ids=member_ids,
        evidence=(evidence,),
    )


def _longest_segment_touching_both(seg, poly_a, poly_b):
    """From `seg` (a LineString or MultiLineString clipped to A or B), pick
    the longest piece whose endpoints lie within both A and B (allowing
    a sliver overlap that means each part touches both surfaces)."""
    geoms = list(getattr(seg, "geoms", [seg]))
    if not geoms:
        return None
    # Filter to pieces that actually intersect both polygons.
    candidates = []
    for g in geoms:
        if g.is_empty:
            continue
        if g.intersects(poly_a) and g.intersects(poly_b):
            candidates.append(g)
    if not candidates:
        return None
    return max(candidates, key=lambda g: g.length)


def _xz_polygon(surface: RoofSurface):
    from shapely.geometry import Polygon

    poly = Polygon([(c.x, c.z) for c in surface.polygon]).buffer(0)
    if poly.is_empty:
        return None
    return poly


def _y_at_xz(surface: RoofSurface, x: float, z: float) -> float | None:
    plane = surface.plane
    if abs(plane.b) < FLOAT_EPS:
        return None
    return -(plane.a * x + plane.c * z + plane.d) / plane.b
