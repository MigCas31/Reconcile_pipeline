"""Phase 5 Step 6: pairwise valley resolution between adjacent building parts.

When two wings share a boundary, their slopes must compose along that
boundary. The 3D line where two oblique planes meet projects to a line in
XZ -- this is the valley (or step-up). Each slope keeps the side of the
valley where its OWN plane is higher; the other side belongs to the
neighbouring wing.

Algorithm:
1. Build wing adjacency: pairs of wings whose buffered footprints overlap
   along an edge >= MIN_ADJACENCY_EDGE_M long.
2. For each adjacent pair (i, j), pairwise over slopes (a in part_i, b in part_j):
   - Compute the half-plane in XZ where pa.y_at > pb.y_at:
     `A*x + B*z + D > 0`
     with A = pa.b·pb.a - pb.b·pa.a, B = pa.b·pb.c - pb.b·pa.c,
     D = pb.b·pa.d - pa.b·pb.d.
   - Clip a's XZ ring to that half-plane; clip b's XZ ring to the opposite.
3. Re-lift each clipped XZ ring onto its slope's own plane (corners stay on
   the plane by construction since `Plane.y_at(x, z)` is exact).
4. Drop slopes whose remaining area falls below MIN_REMAINING_AREA_M2.

This produces clean valleys at L/T/U-junctions where two adjacent wings'
gable slopes face each other, and step-ups where one wing's ridge is
higher than its neighbour's.

Cases handled by the algorithm in step 2 by construction:
- Two equal-height gables at L-junction -> diagonal valley line from
  inner corner up to ridge intersection.
- Taller wing eats the shorter -- lower slope's plane meets the higher
  plane *above* the lower's `ridge_y`; the lower slope's polygon shrinks
  until it terminates at that intersection.
- Y-junction (3+ adjacent wings) -- pairwise clipping is associative;
  the central vertex emerges as the common endpoint of three valley lines
  within snap tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers._core.shapely2 import make_valid_polygon

if TYPE_CHECKING:
    from reconcile_tiers.roof.part_record import PartRecord
    from reconcile_tiers.roof.roof import ObliqueSurface

# Wings must share at least this much edge length to be considered adjacent
# for valley resolution. Below this, the shared boundary is noise and any
# valley would be too short to matter visually.
MIN_ADJACENCY_EDGE_M = 0.5

# Buffer applied to wing polygons when checking adjacency, to tolerate the
# axis-snap noise on shared edges from the wing decomposition.
ADJACENCY_BUFFER_M = 0.05

# Drop a slope after clipping if its remaining XZ area falls below this.
# A slope shrunk to ~nothing means the partner's plane fully dominated.
MIN_REMAINING_AREA_M2 = 0.10

# Floating-point epsilon for the half-plane line coefficient sanity check.
PLANE_PARALLEL_EPS = 1e-9


def resolve_valleys(part_records: list[PartRecord]) -> list[PartRecord]:
    """Pairwise valley clipping between slopes of adjacent wings.

    Returns a new list of `PartRecord`s with their `surfaces` rebuilt to
    reflect any valley clipping. Records whose wings have no adjacent
    neighbours pass through unchanged. Surfaces clipped to less than
    `MIN_REMAINING_AREA_M2` are dropped.
    """
    from reconcile_tiers.roof.part_record import PartRecord

    if len(part_records) < 2:
        return list(part_records)

    adjacency = _build_wing_adjacency(part_records)
    if not adjacency:
        return list(part_records)

    # State: per-(record_idx, surface_idx) -> mutable XZ geometry. Geometry
    # may be a Polygon or MultiPolygon -- cross-gable clipping naturally
    # disconnects a single slope into the arms it covers (e.g., the north
    # slope of an X-axis gable splits into a left arm and a right arm where
    # a perpendicular Z-axis gable's higher plane wins the central region).
    from shapely.geometry.base import BaseGeometry

    state: dict[tuple[int, int], tuple[ObliqueSurface, BaseGeometry]] = {}
    for ri, pr in enumerate(part_records):
        for si, surf in enumerate(pr.surfaces):
            xz_poly = _surface_xz_polygon(surf)
            if xz_poly is not None:
                state[(ri, si)] = (surf, xz_poly)

    # Pairwise resolve each surface from wing i against each surface from wing j
    # for every adjacent (i, j). The half-plane clip is applied ONLY within
    # the XZ overlap of the two surfaces -- outside the overlap, each slope
    # owns its footprint unconditionally.
    for ri, rj in adjacency:
        pr_a, pr_b = part_records[ri], part_records[rj]
        for si in range(len(pr_a.surfaces)):
            entry_a = state.get((ri, si))
            if entry_a is None:
                continue
            for sj in range(len(pr_b.surfaces)):
                entry_b = state.get((rj, sj))
                if entry_b is None:
                    continue
                surf_a, geom_a = entry_a
                surf_b, geom_b = entry_b
                resolved_a, resolved_b = _resolve_surface_pair(
                    surf_a.plane, geom_a, surf_b.plane, geom_b
                )
                if resolved_a is not None:
                    state[(ri, si)] = (surf_a, resolved_a)
                    entry_a = state[(ri, si)]
                if resolved_b is not None:
                    state[(rj, sj)] = (surf_b, resolved_b)

    # Rebuild PartRecords with clipped surfaces. Each input slope may produce
    # multiple output ObliqueSurfaces if the cross-gable clipping disconnected
    # its footprint into separate pieces.
    new_records: list[PartRecord] = []
    for ri, pr in enumerate(part_records):
        new_surfaces: list[ObliqueSurface] = []
        for si, surf in enumerate(pr.surfaces):
            entry = state.get((ri, si))
            if entry is None:
                new_surfaces.append(surf)
                continue
            _, clipped_geom = entry
            for piece in _polygon_pieces(clipped_geom):
                if piece.area < MIN_REMAINING_AREA_M2:
                    continue
                new_surf = _surface_with_xz_polygon(surf, piece)
                if new_surf is not None:
                    new_surfaces.append(new_surf)
        new_records.append(
            PartRecord(
                wing=pr.wing,
                kind=pr.kind,
                surfaces=new_surfaces,
                params=pr.params,
            )
        )
    return new_records


def _polygon_pieces(geom) -> list[Polygon]:
    """Decompose a Polygon or MultiPolygon into a list of polygon pieces.

    Cross-gable clipping naturally splits a slope into multiple disjoint
    polygons (one per arm of the cross). This helper exposes them so each
    can become its own ObliqueSurface.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    parts = []
    for g in getattr(geom, "geoms", []):
        if isinstance(g, Polygon) and not g.is_empty and g.area > 0:
            parts.append(g)
    return parts


# ---- adjacency --------------------------------------------------------------


def _build_wing_adjacency(part_records: list[PartRecord]) -> set[tuple[int, int]]:
    """Return set of (i, j) tuples (i<j) where wings share enough boundary.

    Adjacency check: buffer both wing polygons by ADJACENCY_BUFFER_M and
    intersect. The resulting overlap is roughly a thin strip along the
    shared edge of width ~2*ADJACENCY_BUFFER_M; we estimate the shared
    edge length as `area / (2 * buffer)` and require >= MIN_ADJACENCY_EDGE_M.
    """
    n = len(part_records)
    adj: set[tuple[int, int]] = set()
    for i in range(n):
        wi = part_records[i].wing
        if wi is None or wi.polygon.is_empty:
            continue
        wi_buf = wi.polygon.buffer(ADJACENCY_BUFFER_M)
        for j in range(i + 1, n):
            wj = part_records[j].wing
            if wj is None or wj.polygon.is_empty:
                continue
            try:
                shared = wi_buf.intersection(wj.polygon.buffer(ADJACENCY_BUFFER_M))
            except Exception:
                continue
            if shared.is_empty:
                continue
            edge_estimate = shared.area / (2.0 * ADJACENCY_BUFFER_M)
            if edge_estimate >= MIN_ADJACENCY_EDGE_M:
                adj.add((i, j))
    return adj


# ---- plane intersection in XZ ----------------------------------------------


def _planes_higher_halfplane(
    plane_a: Plane, plane_b: Plane
) -> tuple[float, float, float] | None:
    """Return (A, B, D) such that `pa.y_at(x,z) > pb.y_at(x,z)` iff
    `A*x + B*z + D > 0`.

    Derivation:
        pa.y_at - pb.y_at > 0
        (pa.d - pa.a*x - pa.c*z)/pa.b - (pb.d - pb.a*x - pb.c*z)/pb.b > 0
    Multiply by pa.b * pb.b (both > 0 by convention -- `Plane.fit` sets
    normal[1] >= 0 and `MIN_NY` rejects near-vertical):
        pb.b*(pa.d - pa.a*x - pa.c*z) - pa.b*(pb.d - pb.a*x - pb.c*z) > 0
        x*(pa.b*pb.a - pb.b*pa.a) + z*(pa.b*pb.c - pb.b*pa.c)
            + (pb.b*pa.d - pa.b*pb.d) > 0

    Returns None when the planes are nearly parallel (no meaningful line).
    """
    if plane_a.b <= 0 or plane_b.b <= 0:
        return None
    A = plane_a.b * plane_b.a - plane_b.b * plane_a.a
    B = plane_a.b * plane_b.c - plane_b.b * plane_a.c
    D = plane_b.b * plane_a.d - plane_a.b * plane_b.d
    if abs(A) < PLANE_PARALLEL_EPS and abs(B) < PLANE_PARALLEL_EPS:
        return None
    return (A, B, D)


# ---- pairwise resolution within XZ overlap ---------------------------------


# Below this fraction of either polygon's area, the XZ overlap is treated as
# noise from the buffered adjacency check rather than a real shared region --
# we skip resolution for that pair to avoid clipping a slope that doesn't
# actually fight its neighbour.
MIN_OVERLAP_AREA_FRACTION = 0.02
MIN_OVERLAP_AREA_M2 = 0.05


def _resolve_surface_pair(
    plane_a: Plane,
    geom_a,
    plane_b: Plane,
    geom_b,
):
    """Resolve two slopes whose XZ projections may overlap.

    In the overlap region, the slope whose plane is higher wins; outside the
    overlap, each slope keeps its footprint untouched. Inputs and outputs
    may be Polygon or MultiPolygon -- cross-gable clipping naturally splits
    a slope into disjoint pieces (left arm + right arm of a + roof, etc.).
    """
    if geom_a.is_empty or geom_b.is_empty:
        return (None, None)
    try:
        overlap = geom_a.intersection(geom_b)
    except Exception:
        return (None, None)
    if overlap.is_empty:
        return (None, None)
    overlap_area = float(getattr(overlap, "area", 0.0))
    if overlap_area < MIN_OVERLAP_AREA_M2:
        return (None, None)
    min_self = min(float(geom_a.area), float(geom_b.area))
    if min_self > 0 and overlap_area / min_self < MIN_OVERLAP_AREA_FRACTION:
        return (None, None)

    line = _planes_higher_halfplane(plane_a, plane_b)
    if line is None:
        return (None, None)
    A, B, D = line

    keep_a = _clip_geom_to_halfplane(overlap, A, B, D)
    keep_b = _clip_geom_to_halfplane(overlap, -A, -B, -D)

    new_a = _stitch_outside_overlap(geom_a, overlap, keep_a)
    new_b = _stitch_outside_overlap(geom_b, overlap, keep_b)
    return (new_a, new_b)


def _stitch_outside_overlap(geom, overlap, keep_in_overlap):
    """Return `geom` with `overlap` subtracted, optionally re-adding the
    `keep_in_overlap` slice. May return a Polygon or MultiPolygon.
    """
    try:
        outside = geom.difference(overlap)
    except Exception:
        return None
    if (keep_in_overlap is None or keep_in_overlap.is_empty) and outside.is_empty:
        return None
    if keep_in_overlap is None or keep_in_overlap.is_empty:
        return outside if not outside.is_empty else None
    if outside.is_empty:
        return keep_in_overlap
    try:
        from shapely.ops import unary_union

        merged = unary_union([outside, keep_in_overlap])
    except Exception:
        return None
    return None if merged.is_empty else merged


def _clip_geom_to_halfplane(geom, A: float, B: float, D: float):
    """Half-plane-clip a Polygon or MultiPolygon. Returns clipped geometry
    (Polygon, MultiPolygon, or None if everything was clipped away).
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return _clip_polygon_to_halfplane(geom, A, B, D)
    pieces = []
    for g in getattr(geom, "geoms", []):
        if not isinstance(g, Polygon) or g.is_empty:
            continue
        clipped = _clip_polygon_to_halfplane(g, A, B, D)
        if clipped is not None and not clipped.is_empty:
            pieces.append(clipped)
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    try:
        from shapely.ops import unary_union

        return unary_union(pieces)
    except Exception:
        return pieces[0]


# ---- half-plane clipping ----------------------------------------------------


def _clip_polygon_to_halfplane(
    poly: Polygon, A: float, B: float, D: float
) -> Polygon | None:
    """Sutherland-Hodgman clip of `poly` to the half-plane
    `A*x + B*z + D >= 0`.

    Returns None if the result is empty or invalid.
    """
    if poly.is_empty:
        return None
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None

    def side(pt: tuple[float, float]) -> float:
        return A * pt[0] + B * pt[1] + D

    out: list[tuple[float, float]] = []
    n = len(coords)
    for i in range(n):
        cur = (float(coords[i][0]), float(coords[i][1]))
        nxt = (float(coords[(i + 1) % n][0]), float(coords[(i + 1) % n][1]))
        cur_side = side(cur)
        nxt_side = side(nxt)
        cur_in = cur_side >= 0.0
        nxt_in = nxt_side >= 0.0
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            denom = A * (nxt[0] - cur[0]) + B * (nxt[1] - cur[1])
            if abs(denom) < 1e-12:
                continue
            t = -side(cur) / denom
            ix = cur[0] + t * (nxt[0] - cur[0])
            iz = cur[1] + t * (nxt[1] - cur[1])
            out.append((ix, iz))
    if len(out) < 3:
        return None
    try:
        clipped = Polygon(out)
        if not clipped.is_valid:
            clipped = make_valid_polygon(clipped)
            if clipped is None:
                return None
        if clipped.is_empty or clipped.area < 1e-9:
            return None
        return clipped
    except Exception:
        return None


# ---- surface ↔ XZ polygon ---------------------------------------------------


def _surface_xz_polygon(surf: ObliqueSurface) -> Polygon | None:
    if len(surf.corners) < 3:
        return None
    pts = [(float(c[0]), float(c[2])) for c in surf.corners]
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = make_valid_polygon(poly)
        if poly is None:
            return None
    if poly.is_empty or poly.area < 1e-9:
        return None
    return poly


def _surface_with_xz_polygon(
    surf: ObliqueSurface, xz_poly: Polygon
) -> ObliqueSurface | None:
    """Reconstruct an ObliqueSurface with a new XZ ring lifted onto its
    plane. Corners land exactly on `surf.plane` by construction."""
    from reconcile_tiers.roof.roof import ObliqueSurface

    coords = list(xz_poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None
    new_corners: list[list[float]] = []
    for x, z in coords:
        y = surf.plane.y_at(float(x), float(z))
        if y is None:
            return None
        new_corners.append([float(x), float(y), float(z)])
    return ObliqueSurface(
        corners=new_corners,
        plane=surf.plane,
        cluster=surf.cluster,
        dominant_story=surf.dominant_story,
        ridge=surf.ridge,
        source_index=surf.source_index,
    )
