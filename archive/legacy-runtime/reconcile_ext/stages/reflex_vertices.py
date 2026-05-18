"""R1.1 — enumerate reflex vertices on the merged building footprint.

A reflex vertex is a corner of the footprint where the interior angle
exceeds 180°. Every extension produces at least two such vertices at the
seam where the extension meets the main house. Not every reflex vertex is
an extension seam (bay windows, bow windows, scan noise all produce them)
so each vertex is scored by:

  - interior angle (> ``REFLEX_MIN_INTERIOR_ANGLE_DEG``)
  - edge lengths on both sides (> ``REFLEX_MIN_EDGE_M``)
  - notch depth — distance from vertex to the convex hull (>
    ``REFLEX_MIN_NOTCH_DEPTH_M``)

Only vertices clearing all three bars end up in the report.
"""

from __future__ import annotations

import math

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..constants import (
    FOOTPRINT_SIMPLIFY_TOL_M,
    REFLEX_MIN_EDGE_M,
    REFLEX_MIN_INTERIOR_ANGLE_DEG,
    REFLEX_MIN_NOTCH_DEPTH_M,
)
from ..models import ExtReflexVertex, ExtSnapshot


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
    # Heal notches that V3 already closed with gap-fill slabs. Without this,
    # scan-registration gaps (room-to-room alignment noise V3 bridged) would
    # reintroduce concave corners here and misfire as extension seams.
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
    # Pick the largest polygon if the union produced a MultiPolygon — reflex
    # detection operates on one ring at a time. (We could iterate all rings
    # later if multi-component buildings show up.)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    # Douglas-Peucker collapses sub-15cm jogs produced by the union of many
    # slightly-offset room polygons. Without this, every scan-noise zigzag
    # becomes a false-positive reflex vertex.
    simplified = merged.simplify(FOOTPRINT_SIMPLIFY_TOL_M, preserve_topology=True)
    if simplified.is_valid and not simplified.is_empty:
        if simplified.geom_type == "MultiPolygon":
            simplified = max(simplified.geoms, key=lambda g: g.area)
        merged = simplified
    return merged


def _part_id_for_vertex(
    snapshot: ExtSnapshot, x: float, z: float, tol: float = 0.2
) -> str | None:
    p = Point(x, z)
    best_id = None
    best_dist = float("inf")
    for part in snapshot.parts:
        if len(part.footprint_xz) < 3:
            continue
        poly = Polygon([(c[0], c[2]) for c in part.footprint_xz])
        if not poly.is_valid:
            continue
        dist = poly.exterior.distance(p)
        if dist < best_dist:
            best_dist = dist
            best_id = part.id
    if best_dist <= tol:
        return best_id
    return None


def _signed_angle_cw(
    ax: float, az: float, bx: float, bz: float, cx: float, cz: float
) -> float:
    """Interior angle at B on path A->B->C, measured through the polygon's
    interior when the ring is traversed in standard Shapely (CCW) order.

    Returns degrees. For a CCW ring, a reflex vertex has interior angle > 180°.
    """
    # Turn is computed from incoming edge direction (a->b) to outgoing edge
    # direction (b->c). A left turn (cross>0) means the CCW ring swung left,
    # which at this vertex means the interior angle is < 180° (convex).
    ux, uz = bx - ax, bz - az
    vx, vz = cx - bx, cz - bz
    cross = ux * vz - uz * vx
    dot = ux * vx + uz * vz
    turn = math.degrees(math.atan2(cross, dot))
    interior = 180.0 - turn
    # Normalise to [0°, 360°).
    if interior < 0:
        interior += 360.0
    if interior >= 360.0:
        interior -= 360.0
    return interior


def _edge_azimuth_deg(ax: float, az: float, bx: float, bz: float) -> float:
    dx, dz = bx - ax, bz - az
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, dz)) % 360.0


def detect_reflex_vertices(snapshot: ExtSnapshot) -> list[ExtReflexVertex]:
    footprint = _merged_footprint(snapshot)
    if footprint is None:
        return []

    coords = list(footprint.exterior.coords)
    # Shapely closes the ring — strip the duplicate last vertex.
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return []

    # Ensure CCW — Shapely exterior is CCW by convention, but validate.
    hull = footprint.convex_hull
    # Compute a signed area to be safe.
    signed_area = 0.0
    for i in range(len(coords)):
        x1, z1 = coords[i]
        x2, z2 = coords[(i + 1) % len(coords)]
        signed_area += (x2 - x1) * (z2 + z1)
    if signed_area > 0:
        coords = list(reversed(coords))

    results: list[ExtReflexVertex] = []
    for i, (bx, bz) in enumerate(coords):
        ax, az = coords[(i - 1) % len(coords)]
        cx, cz = coords[(i + 1) % len(coords)]

        interior_angle = _signed_angle_cw(ax, az, bx, bz, cx, cz)
        if interior_angle < REFLEX_MIN_INTERIOR_ANGLE_DEG:
            continue

        left_len = math.hypot(bx - ax, bz - az)
        right_len = math.hypot(cx - bx, cz - bz)
        if left_len < REFLEX_MIN_EDGE_M or right_len < REFLEX_MIN_EDGE_M:
            continue

        vertex = Point(bx, bz)
        notch_depth = vertex.distance(hull.exterior) if hull.exterior else 0.0
        if notch_depth < REFLEX_MIN_NOTCH_DEPTH_M:
            continue

        results.append(
            ExtReflexVertex(
                id=f"reflex::{i}",
                x=float(bx),
                z=float(bz),
                interior_angle_deg=float(interior_angle),
                notch_depth_m=float(notch_depth),
                left_edge_len_m=float(left_len),
                right_edge_len_m=float(right_len),
                left_edge_azimuth_deg=_edge_azimuth_deg(bx, bz, ax, az),
                right_edge_azimuth_deg=_edge_azimuth_deg(bx, bz, cx, cz),
                part_id=_part_id_for_vertex(snapshot, bx, bz),
                rationale=(
                    f"Interior angle {interior_angle:.1f}°, notch depth "
                    f"{notch_depth:.2f}m; "
                    f"adjacent edges {left_len:.2f}m and {right_len:.2f}m"
                ),
            )
        )

    return results
