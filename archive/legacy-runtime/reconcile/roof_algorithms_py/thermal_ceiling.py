"""Compute thermal-envelope ceiling surfaces.

For each exposed room, determines what the ceiling looks like from above:
- Flat ceilings stay as-is at room top-Y
- Oblique roof surfaces get clipped to room footprint and capped at top-Y
- Flat caps fill the gap where slant meets horizontal plane
- Knee-wall gaps (floor below extends >30cm beyond current floor) get ceilings
"""

from __future__ import annotations

from .math_utils import (
    clip_by_half_plane_xz,
    clip_by_max_y,
    point_in_poly_2d,
    point_in_poly_xz,
)


def _robust_wall_y(tops: list[float], fallback: float, *, use_max: bool) -> float:
    """Pick a robust wall-top height, ignoring outliers when most walls agree.

    If >= 2/3 of walls cluster within 20 cm of the median, return the median
    (outlier pattern: e.g. one stairwell wall much higher/lower than the
    rest).  Otherwise the spread is real (sloped ceiling) -> return the
    original max or min.
    """
    if not tops or len(tops) < 3:
        return fallback

    n = len(tops)
    sorted_tops = sorted(tops) if tops is not sorted else tops
    median = (
        sorted_tops[n // 2]
        if n % 2 == 1
        else (sorted_tops[n // 2 - 1] + sorted_tops[n // 2]) / 2
    )
    cluster_count = sum(1 for y in sorted_tops if abs(y - median) <= 0.20)

    if cluster_count >= n * 2 / 3:
        return median
    return fallback


def _robust_ceiling_y(er: dict) -> float:
    """Robust wall-top Y for flat ceiling fallback (ignores high outliers)."""
    return _robust_wall_y(er.get("wallTopYs", []), er["wallTopY"], use_max=True)


def _robust_eave_y(er: dict) -> float:
    """Robust wall-top-min Y for eave clipping (ignores low outliers).

    Only applies the outlier filter when the spread is small (< 0.5m),
    indicating a single anomalous wall (e.g. stairwell).  When the spread
    is large, the low wall is a genuine eave (sloped ceiling) and must be
    kept -- clipping at the median would remove the entire slant.
    """
    tops = er.get("wallTopYs", [])
    if tops and (max(tops) - min(tops)) >= 0.5:
        return er["wallTopMin"]
    return _robust_wall_y(tops, er["wallTopMin"], use_max=False)


def _clip_above_y(
    poly: list[tuple[float, float, float]], min_y: float
) -> list[tuple[float, float, float]]:
    """Sutherland-Hodgman clip: keep portion of poly at or above min_y."""
    out: list[tuple[float, float, float]] = []
    n = len(poly)
    for k in range(n):
        curr, nxt = poly[k], poly[(k + 1) % n]
        c_in, n_in = curr[1] >= min_y, nxt[1] >= min_y
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = (min_y - curr[1]) / (nxt[1] - curr[1])
            out.append(
                (
                    curr[0] + t * (nxt[0] - curr[0]),
                    min_y,
                    curr[2] + t * (nxt[2] - curr[2]),
                )
            )
    return out


def _xz_projection(poly_3d: list) -> list[tuple[float, float]]:
    """Extract (x, z) from 3D points."""
    return [(p[0], p[2]) for p in poly_3d]


def _shapely_poly(points_2d: list[tuple[float, float]]):
    """Create a Shapely Polygon, return None if invalid or degenerate."""
    if len(points_2d) < 3:
        return None
    try:
        from shapely.geometry import Polygon

        p = Polygon(points_2d)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area < 0.01:
            return None
        return p
    except Exception:
        return None


# Wall top within BARRIER_REACH of a cap plane is treated as reaching the cap
# and gets its XZ strip subtracted from the cap polygon. Scan noise routinely
# leaves walls 10-25 cm short of the true ceiling; 30 cm captures that cohort
# without snapping truly shorter partitions.
BARRIER_REACH = 0.30

# Half-width of the buffer applied to a wall's XZ projection when building a
# barrier strip. Walls project to near-degenerate quads (essentially a line),
# so buffering gives a thin valid polygon that can cleanly subtract from caps.
# Matched to the within_story gap detection scale in extract3d/gaps.py.
WALL_STRIP_HALF_WIDTH = 0.03


def _wall_xz_strip(corners: list) -> object | None:
    """Build a Shapely polygon representing a wall's XZ footprint strip.

    Walls are thin vertical quads whose XZ projection collapses to a line;
    we build the line from the unique XZ points and buffer it out slightly
    so downstream `difference` operations can cut caps cleanly.
    """
    from shapely.geometry import LineString, Polygon

    if not corners:
        return None
    seen: list[tuple[float, float]] = []
    for c in corners:
        pt = (round(float(c[0]), 6), round(float(c[2]), 6))
        if pt not in seen:
            seen.append(pt)
    if len(seen) < 2:
        return None
    try:
        if len(seen) == 2:
            geom = LineString(seen).buffer(
                WALL_STRIP_HALF_WIDTH, cap_style=2, join_style=2
            )
        else:
            poly = Polygon(seen)
            if not poly.is_valid or poly.area < 1e-6:
                geom = LineString([*seen, seen[0]]).buffer(
                    WALL_STRIP_HALF_WIDTH, cap_style=2, join_style=2
                )
            else:
                geom = poly.buffer(WALL_STRIP_HALF_WIDTH, cap_style=2, join_style=2)
    except Exception:
        return None
    if geom.is_empty or geom.area < 1e-6:
        return None
    return geom


def _wall_top_y(corners: list) -> float | None:
    """Return the highest Y ordinate across a wall's 3D corners."""
    if not corners:
        return None
    ys = [float(c[1]) for c in corners if len(c) >= 2]
    if not ys:
        return None
    return max(ys)


def _build_story_thermal_context(
    exposed_rooms: list,
    gap_walls: list | None,
    rooms: list | None,
) -> dict:
    """Per-story precomputation for widened cap domains and wall barriers.

    Returns a dict keyed by story with:
      - attached_gaps_by_room: {room_index -> Shapely geometry of gap polygons}
      - walls_info: list of (xz_strip_shapely, wall_top_y) for every wall on the story
    """
    context: dict[int, dict] = {}

    stories = sorted({er["story"] for er in exposed_rooms})
    for story in stories:
        context[story] = {
            "attached_gaps_by_room": {},
            "walls_info": [],
        }

    if gap_walls:
        from shapely.ops import unary_union

        per_story_per_room: dict[int, dict[int, list]] = {}
        for gap in gap_walls:
            if gap.get("type") != "within_story":
                continue
            story = gap.get("story")
            ri = gap.get("room_index")
            if story is None or ri is None or story not in context:
                continue
            corners = gap.get("corners") or []
            if len(corners) < 3:
                continue
            gap_xz = _xz_projection(corners)
            poly = _shapely_poly(gap_xz)
            if poly is None:
                continue
            per_story_per_room.setdefault(story, {}).setdefault(ri, []).append(poly)

        for story, by_room in per_story_per_room.items():
            for ri, polys in by_room.items():
                if len(polys) == 1:
                    context[story]["attached_gaps_by_room"][ri] = polys[0]
                else:
                    context[story]["attached_gaps_by_room"][ri] = unary_union(polys)

    if rooms:
        for room in rooms:
            story = room.get("story")
            if story is None or story not in context:
                continue
            walls = list(room.get("walls_merged") or []) + list(
                room.get("walls_computed") or []
            )
            for wall in walls:
                corners = wall.get("corners") or []
                if len(corners) < 2:
                    continue
                strip = _wall_xz_strip(corners)
                top_y = _wall_top_y(corners)
                if strip is None or top_y is None:
                    continue
                context[story]["walls_info"].append((strip, top_y))

    return context


def _barrier_union_for_cap(
    walls_info: list[tuple[object, float]],
    cap_y: float,
):
    """Union of wall XZ strips whose top Y reaches the cap plane (cap_y - REACH)."""
    from shapely.ops import unary_union

    strips = [s for s, top_y in walls_info if top_y >= cap_y - BARRIER_REACH]
    if not strips:
        return None
    if len(strips) == 1:
        return strips[0]
    return unary_union(strips)


def _remove_collinear_xz(
    poly_3d: list[tuple[float, float, float]], eps: float = 1e-4
) -> list[tuple[float, float, float]]:
    """Remove collinear vertices (in XZ) that create zero-width arms.

    These arise from Shapely difference operations where the remainder
    polygon boundary doubles back along a shared edge, producing
    degenerate features that break viewer triangulation.
    """
    pts = list(poly_3d)
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float, float]] = []
        n = len(pts)
        for i in range(n):
            prev, curr, nxt = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            cross = (curr[0] - prev[0]) * (nxt[2] - prev[2]) - (curr[2] - prev[2]) * (
                nxt[0] - prev[0]
            )
            if abs(cross) > eps:
                out.append(curr)
            else:
                changed = True
        if len(out) < 3:
            return pts
        pts = out
    return pts


def _poly_3d_from_shapely(geom, y: float) -> list[list[tuple[float, float, float]]]:
    """Convert Shapely geometry to list of 3D polygon(s) at given Y."""
    from shapely.geometry import MultiPolygon, Polygon

    results = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        return results

    for poly in geoms:
        if poly.is_empty or poly.area < 0.01:
            continue
        coords = list(poly.exterior.coords)[:-1]  # drop closing duplicate
        poly_3d = [(x, y, z) for x, z in coords]
        poly_3d = _remove_collinear_xz(poly_3d)
        if len(poly_3d) >= 3:
            results.append(poly_3d)
    return results


def _clip_3d_to_floor_xz(
    poly_3d: list[tuple[float, float, float]],
    floor_poly: list,
) -> list[tuple[float, float, float]]:
    """Sutherland-Hodgman clip of 3D polygon against floor polygon edges in XZ."""
    result = list(poly_3d)
    n = len(floor_poly)
    for i in range(n):
        if len(result) < 3:
            return []
        ax, _, az = floor_poly[i][0], floor_poly[i][1], floor_poly[i][2]
        bx, _, bz = (
            floor_poly[(i + 1) % n][0],
            floor_poly[(i + 1) % n][1],
            floor_poly[(i + 1) % n][2],
        )
        # Inward normal of edge a->b
        nx = -(bz - az)
        nz = bx - ax
        result = clip_by_half_plane_xz(result, ax, az, nx, nz)
    return result


def _surfaces_overlapping_room(
    oblique_surfaces: list[dict], floor_poly: list
) -> list[int]:
    """Return indices of oblique surfaces that overlap the room floor polygon in XZ."""
    hits = []
    fp_xz = _xz_projection(floor_poly)
    fp_shapely = _shapely_poly(fp_xz)
    if fp_shapely is None:
        return hits

    for si, surf in enumerate(oblique_surfaces):
        corners = surf.get("corners", [])
        if len(corners) < 3:
            continue
        # Check if any vertex of the surface falls inside the floor poly
        for c in corners:
            if point_in_poly_xz(c[0], c[2], floor_poly):
                hits.append(si)
                break
        else:
            # Check if floor centroid falls inside surface XZ projection
            surf_xz = _xz_projection(corners)
            surf_shapely = _shapely_poly(surf_xz)
            if surf_shapely is not None and fp_shapely.intersects(surf_shapely):
                if fp_shapely.intersection(surf_shapely).area > 0.05:
                    hits.append(si)
    return hits


def _find_covering_planes(room_data: dict, ceiling_planes: list) -> list[int]:
    """Find ceiling planes that cover (project onto) the room."""
    from .math_utils import plane_y_at

    room_data["fp"]
    fcx, fcz = room_data["fcx"], room_data["fcz"]
    floor_y = room_data["floorY"]
    top_y = room_data["wallTopY"]
    story = room_data["story"]

    hits = []
    for pi, plane in enumerate(ceiling_planes):
        # Check dominant story matches
        if plane["dominantStory"] != story:
            continue
        # Check that the plane projects to reasonable Y at room centroid
        y_at_center = plane_y_at(plane, fcx, fcz)
        if y_at_center < floor_y or y_at_center > top_y + 3.0:
            continue
        hits.append(pi)
    return hits


def _build_thermal_for_oblique_room(
    room_data: dict,
    oblique_surfaces: list[dict],
    surface_indices: list[int],
    ceiling_planes: list,
    story_context: dict | None = None,
) -> list[dict]:
    """Build thermal ceiling surfaces for a room under oblique roof surfaces.

    Projects ceiling planes onto the room floor polygon to get full coverage,
    then clips each plane surface at:
    1. The opposing plane (ridge line) -- lower envelope
    2. wallTopY from above
    3. wallTopMin from below (eave height)

    Ridge/eave caps use a widened footprint (room fp plus within-story gap
    polygons assigned to this room) and are split by walls whose top Y
    reaches the cap plane within BARRIER_REACH.
    """
    from .math_utils import plane_y_at

    fp = room_data["fp"]
    top_y = room_data["wallTopY"]
    bottom_y = _robust_eave_y(room_data)
    story = room_data["story"]
    room_idx = room_data.get("room_index")
    out: list[dict] = []

    # Find ceiling planes that cover this room
    plane_indices = _find_covering_planes(room_data, ceiling_planes)

    if not plane_indices:
        # Fallback: use pre-clipped oblique surfaces
        return _build_thermal_from_oblique_surfaces(
            room_data, oblique_surfaces, surface_indices
        )

    # For each ceiling plane, project the room floor polygon onto it,
    # then clip by the lower envelope of all other planes (keep lower Y).
    # This correctly handles mansard roofs, kinks, and multi-slope ceilings.
    slant_polys = []  # XZ footprints of pre-bottom-clip slants (for ridge cap)
    rendered_slant_polys = []  # XZ footprints of rendered slants (for eave cap)
    for pi in plane_indices:
        plane = ceiling_planes[pi]

        # Project floor polygon onto this plane
        projected = [(p[0], plane_y_at(plane, p[0], p[2]), p[2]) for p in fp]

        # Clip by lower envelope: for each other covering plane, keep the
        # portion where THIS plane is lower (closer to the room)
        for oi in plane_indices:
            if oi == pi:
                continue
            other = ceiling_planes[oi]
            projected = _clip_by_lower_plane(projected, plane, other)
            if len(projected) < 3:
                break

        if len(projected) < 3:
            continue

        # Clip at top-Y
        projected = clip_by_max_y(projected, top_y)
        if len(projected) < 3:
            continue

        slant_polys.append(_xz_projection(projected))

        # Clip at bottom-Y (eave)
        clipped_for_render = _clip_above_y(projected, bottom_y)

        if len(clipped_for_render) >= 3:
            # Find dormer cutout holes that overlap this surface
            clipped_holes = _find_dormer_holes_for_surface(
                clipped_for_render,
                oblique_surfaces,
                surface_indices,
                fp,
                top_y,
                bottom_y,
            )
            entry: dict = {
                "kind": "thermal-slant",
                "story": story,
                "poly": clipped_for_render,
            }
            if clipped_holes:
                entry["cutout_holes"] = clipped_holes
            out.append(entry)
            rendered_slant_polys.append(_xz_projection(clipped_for_render))

    from shapely.ops import unary_union

    fp_xz = _xz_projection(fp)
    fp_shapely = _shapely_poly(fp_xz)

    # Widen the cap footprint with within-story gap polygons assigned to this
    # room so ridge/eave caps cover the building envelope under the roof, not
    # just the room's own floor slab.
    attached_gaps = None
    walls_info: list = []
    if story_context is not None:
        attached_gaps = story_context.get("attached_gaps_by_room", {}).get(room_idx)
        walls_info = story_context.get("walls_info", [])
    if fp_shapely is not None and attached_gaps is not None:
        try:
            fp_cap_shapely = unary_union([fp_shapely, attached_gaps])
        except Exception:
            fp_cap_shapely = fp_shapely
    else:
        fp_cap_shapely = fp_shapely

    # Ridge cap: gap between pre-bottom-clip slant XZ projections at top_y
    if fp_cap_shapely is not None and slant_polys:
        parts = [_shapely_poly(p) for p in slant_polys]
        parts = [p for p in parts if p is not None]
        if parts:
            slant_union = unary_union(parts)
            slant_hull = slant_union.convex_hull
            bounded = fp_cap_shapely.intersection(slant_hull)
            try:
                remainder = bounded.difference(slant_union)
            except Exception:
                remainder = None
            if remainder is not None and not remainder.is_empty:
                barrier = _barrier_union_for_cap(walls_info, top_y)
                if barrier is not None:
                    try:
                        remainder = remainder.difference(barrier)
                    except Exception:
                        pass
                cap_polys = _poly_3d_from_shapely(remainder, top_y)
                for poly in cap_polys:
                    if len(poly) >= 3:
                        out.append(
                            {
                                "kind": "thermal-cap",
                                "story": story,
                                "poly": poly,
                            }
                        )

    # Eave cap: gap between all emitted thermal surfaces and the cap footprint.
    # Covers the area below the eave line where slants were clipped at wallTopMin.
    if fp_cap_shapely is not None:
        all_emitted_xz = []
        for entry in out:
            exz = _shapely_poly(_xz_projection(entry["poly"]))
            if exz is not None:
                all_emitted_xz.append(exz)
        if all_emitted_xz:
            emitted_union = unary_union(all_emitted_xz)
            try:
                eave_gap = fp_cap_shapely.difference(emitted_union)
            except Exception:
                eave_gap = None
            if eave_gap is not None and not eave_gap.is_empty:
                barrier = _barrier_union_for_cap(walls_info, bottom_y)
                if barrier is not None:
                    try:
                        eave_gap = eave_gap.difference(barrier)
                    except Exception:
                        pass
                eave_polys = _poly_3d_from_shapely(eave_gap, bottom_y)
                for poly in eave_polys:
                    if len(poly) >= 3:
                        out.append(
                            {
                                "kind": "thermal-cap",
                                "story": story,
                                "poly": poly,
                            }
                        )

    return out


def _clip_by_lower_plane(
    poly: list[tuple[float, float, float]],
    this_plane: dict,
    other_plane: dict,
) -> list[tuple[float, float, float]]:
    """Sutherland-Hodgman: keep portion of poly where this_plane Y <= other_plane Y."""
    from .math_utils import plane_y_at

    if len(poly) < 3:
        return poly
    out: list[tuple[float, float, float]] = []
    n = len(poly)
    for k in range(n):
        curr, nxt = poly[k], poly[(k + 1) % n]
        # Value: this_y - other_y; keep where <= 0 (this plane is lower)
        cv = plane_y_at(this_plane, curr[0], curr[2]) - plane_y_at(
            other_plane, curr[0], curr[2]
        )
        nv = plane_y_at(this_plane, nxt[0], nxt[2]) - plane_y_at(
            other_plane, nxt[0], nxt[2]
        )
        c_in = cv <= 0.0
        n_in = nv <= 0.0
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = cv / (cv - nv) if abs(cv - nv) > 1e-12 else 0.5
            ix = curr[0] + t * (nxt[0] - curr[0])
            iz = curr[2] + t * (nxt[2] - curr[2])
            iy = plane_y_at(this_plane, ix, iz)
            out.append((ix, iy, iz))
    return out


def _find_dormer_holes_for_surface(
    surface_poly: list[tuple[float, float, float]],
    oblique_surfaces: list[dict],
    surface_indices: list[int],
    fp: list,
    top_y: float,
    bottom_y: float,
) -> list[list[tuple[float, float, float]]]:
    """
    Find dormer cutout holes from oblique surfaces that overlap this thermal
    surface.
    """
    clipped_holes = []
    surf_xz = _xz_projection(surface_poly)
    surf_shapely = _shapely_poly(surf_xz)
    if surf_shapely is None:
        return clipped_holes

    for si in surface_indices:
        surf = oblique_surfaces[si]
        for hole in surf.get("cutout_holes", []):
            hole_xz = _xz_projection(hole)
            hole_shapely = _shapely_poly(hole_xz)
            if hole_shapely is None:
                continue
            if not surf_shapely.intersects(hole_shapely):
                continue
            # Clip hole to this surface's bounds
            h = _clip_3d_to_floor_xz(list(hole), fp)
            if len(h) >= 3:
                h = clip_by_max_y(h, top_y)
                if len(h) >= 3:
                    h = _clip_above_y(h, bottom_y)
                    if len(h) >= 3:
                        clipped_holes.append(h)
    return clipped_holes


def _build_thermal_from_oblique_surfaces(
    room_data: dict,
    oblique_surfaces: list[dict],
    surface_indices: list[int],
) -> list[dict]:
    """Fallback: build thermal surfaces from pre-clipped oblique surfaces."""
    fp = room_data["fp"]
    top_y = room_data["wallTopY"]
    bottom_y = _robust_eave_y(room_data)
    story = room_data["story"]
    out: list[dict] = []

    full_xz_polys = []
    for si in surface_indices:
        surf = oblique_surfaces[si]
        corners = list(surf["corners"])
        clipped = _clip_3d_to_floor_xz(corners, fp)
        if len(clipped) < 3:
            continue
        clipped = clip_by_max_y(clipped, top_y)
        if len(clipped) < 3:
            continue
        full_xz_polys.append(_xz_projection(clipped))
        clipped = _clip_above_y(clipped, bottom_y)
        if len(clipped) < 3:
            continue

        clipped_holes = []
        for hole in surf.get("cutout_holes", []):
            h = _clip_3d_to_floor_xz(list(hole), fp)
            if len(h) >= 3:
                h = clip_by_max_y(h, top_y)
                if len(h) >= 3:
                    h = _clip_above_y(h, bottom_y)
                    if len(h) >= 3:
                        clipped_holes.append(h)

        entry: dict = {"kind": "thermal-slant", "story": story, "poly": clipped}
        if clipped_holes:
            entry["cutout_holes"] = clipped_holes
        out.append(entry)

    fp_xz = _xz_projection(fp)
    fp_shapely = _shapely_poly(fp_xz)
    if fp_shapely is not None and full_xz_polys:
        from shapely.ops import unary_union

        parts = [_shapely_poly(p) for p in full_xz_polys]
        parts = [p for p in parts if p is not None]
        if parts:
            oblique_union = unary_union(parts)
            # Constrain to convex hull of oblique projections so cap doesn't
            # extend past slanted walls into uncovered floor area
            oblique_hull = oblique_union.convex_hull
            bounded = fp_shapely.intersection(oblique_hull)
            try:
                remainder = bounded.difference(oblique_union)
            except Exception:
                remainder = None
            if remainder is not None:
                cap_polys = _poly_3d_from_shapely(remainder, top_y)
                for poly in cap_polys:
                    if len(poly) >= 3:
                        out.append(
                            {
                                "kind": "thermal-cap",
                                "story": story,
                                "poly": poly,
                            }
                        )

    return out


def _floor_y_at_xz(
    story_floor_regions: dict[int, list[tuple[list, float]]] | None,
    story: int,
    x: float,
    z: float,
    fallback_y: float,
) -> float:
    """Find the floor Y of the room whose XZ footprint contains (x, z)."""
    if not story_floor_regions:
        return fallback_y
    regions = story_floor_regions.get(story, [])
    for fp, fy in regions:
        xz_pts = [(p[0], p[2]) for p in fp]
        if point_in_poly_2d(x, z, xz_pts):
            return fy
    return fallback_y


def _build_knee_wall_ceilings(
    floors_by_story: dict[int, list[list]],
    story_floor_y: dict[int, float],
    story_floor_regions: dict[int, list[tuple[list, float]]] | None = None,
    oblique_roof_surfaces: list[dict] | None = None,
) -> list[dict]:
    """Detect knee-wall gaps: where floor below extends >30cm beyond floor above.

    The 30cm is a threshold: if the lower floor extends more than 30cm past
    the upper floor boundary, the ENTIRE difference area becomes a knee-wall
    ceiling (not shrunk by 30cm).

    For half-level buildings, the ceiling Y is looked up per gap polygon using
    story_floor_regions so each zone gets the correct height.

    When ``oblique_roof_surfaces`` is provided, the knee polygon is clipped to
    the XZ union of those surfaces (with a 0.3m eave buffer). This prevents
    emitting a knee cap over regions with no sloped roof above -- those regions
    have their actual ceiling set by a flat roof or independent structure, and
    a knee at the upper-floor height would float above an air gap.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    THRESHOLD_M = 0.30
    MIN_AREA = 0.1
    OBLIQUE_BUFFER_M = 0.30
    out: list[dict] = []

    oblique_union = None
    if oblique_roof_surfaces:
        oblique_polys = []
        for s in oblique_roof_surfaces:
            xz = _xz_projection(s.get("corners") or [])
            shp = _shapely_poly(xz)
            if shp is not None:
                oblique_polys.append(shp)
        if oblique_polys:
            oblique_union = unary_union(oblique_polys).buffer(OBLIQUE_BUFFER_M)

    stories = sorted(floors_by_story.keys())
    for i in range(1, len(stories)):
        upper_story = stories[i]
        lower_story = stories[i - 1]

        upper_fps = floors_by_story.get(upper_story, [])
        lower_fps = floors_by_story.get(lower_story, [])
        if not upper_fps or not lower_fps:
            continue

        upper_polys = [_shapely_poly(_xz_projection(fp)) for fp in upper_fps]
        upper_polys = [p for p in upper_polys if p is not None]
        lower_polys = [_shapely_poly(_xz_projection(fp)) for fp in lower_fps]
        lower_polys = [p for p in lower_polys if p is not None]

        if not upper_polys or not lower_polys:
            continue

        upper_union = unary_union(upper_polys)
        lower_union = unary_union(lower_polys)

        # Full difference (no buffer shrink)
        try:
            full_gap = lower_union.difference(upper_union)
        except Exception:
            continue

        if full_gap.is_empty:
            continue

        # Only include if the gap extends more than 30cm from the upper boundary
        # Check by buffering upper inward -- if the unbuffered diff has area but
        # the buffered diff doesn't, the gap is < 30cm and we skip it.
        buffered_upper = upper_union.buffer(THRESHOLD_M)
        try:
            test_gap = lower_union.difference(buffered_upper)
        except Exception:
            continue
        if test_gap.is_empty or test_gap.area < MIN_AREA:
            continue

        # Threshold met -- use the full difference (not the buffered one)
        gap = full_gap

        # Clip to oblique roof footprint: a knee only exists where a sloped
        # roof descends onto the lower story.
        if oblique_union is not None:
            try:
                gap = gap.intersection(oblique_union)
            except Exception:
                continue

        if gap.is_empty:
            continue

        fallback_y = story_floor_y.get(upper_story)
        if fallback_y is None:
            continue

        geoms = (
            [gap]
            if isinstance(gap, Polygon)
            else list(gap.geoms)
            if isinstance(gap, MultiPolygon)
            else []
        )
        for g in geoms:
            if g.is_empty or g.area < MIN_AREA:
                continue
            # Only keep gap polygons adjacent to the upper floor -- these are
            # true knee walls. Polygons separated from the upper floor are
            # building extensions with their own roof, not knee wall zones.
            if g.distance(upper_union) > 0.1:
                continue
            coords = list(g.exterior.coords)[:-1]
            # Per-zone height lookup for half-level support
            centroid = g.centroid
            y = _floor_y_at_xz(
                story_floor_regions,
                upper_story,
                centroid.x,
                centroid.y,  # shapely 2D: x=X, y=Z
                fallback_y,
            )
            out.append(
                {
                    "kind": "thermal-knee",
                    "story": upper_story,
                    "poly": [(x, y, z) for x, z in coords],
                }
            )

    return out


def _find_best_matching_room(slant_ceiling: dict, exposed_rooms: list) -> dict | None:
    """Find the exposed room whose floor polygon best overlaps a slant ceiling."""
    story = slant_ceiling["story"]
    poly = slant_ceiling.get("poly", [])
    if not poly:
        return None

    slant_xz = _xz_projection(poly)
    slant_shapely = _shapely_poly(slant_xz)

    best_room = None
    best_overlap = 0.0

    for er in exposed_rooms:
        if er["story"] != story:
            continue
        fp_xz = _xz_projection(er["fp"])
        fp_shapely = _shapely_poly(fp_xz)
        if fp_shapely is None:
            continue

        if slant_shapely is not None:
            try:
                overlap = fp_shapely.intersection(slant_shapely).area
            except Exception:
                overlap = 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_room = er
        elif best_room is None:
            # Fallback: first room on same story
            best_room = er

    return best_room


def _subtract_knee_zones_from_surfaces(
    surfaces: list[dict],
    knee_zones_by_story: dict[int, object],
) -> list[dict]:
    """Remove knee-wall zone coverage from lower-story thermal surfaces.

    When a knee wall exists at story N, the thermal boundary in that zone is
    the knee wall ceiling (at story N floor level), not the room ceiling below.
    Subtract knee zone from story N-1 thermal-flat/slant/cap surfaces.
    """
    if not knee_zones_by_story:
        return surfaces

    result = []
    for surf in surfaces:
        kind = surf["kind"]
        story = surf["story"]

        # Only subtract from flat/slant/cap on the story BELOW the knee wall
        if kind not in ("thermal-flat", "thermal-slant", "thermal-cap"):
            result.append(surf)
            continue

        # Knee walls at story N cover areas on story N-1
        knee_zone = knee_zones_by_story.get(story + 1)
        if knee_zone is None:
            result.append(surf)
            continue

        poly = surf["poly"]
        surf_xz = _xz_projection(poly)
        surf_shapely = _shapely_poly(surf_xz)
        if surf_shapely is None:
            result.append(surf)
            continue

        try:
            remainder = surf_shapely.difference(knee_zone)
        except Exception:
            result.append(surf)
            continue

        if remainder.is_empty or remainder.area < 0.01:
            continue  # entirely covered by knee zone -- drop this surface

        # Check if the surface was barely affected -- keep original if > 95% remains
        if remainder.area > surf_shapely.area * 0.95:
            result.append(surf)
            continue

        if kind == "thermal-slant":
            # For slant surfaces, clip the 3D polygon against remainder boundary
            clipped = _clip_3d_to_remainder(poly, remainder)
            if clipped and len(clipped) >= 3:
                new_surf = dict(surf)
                new_surf["poly"] = clipped
                result.append(new_surf)
        else:
            # Flat/cap surfaces: constant Y
            y_val = poly[0][1]
            rebuilt = _poly_3d_from_shapely(remainder, y_val)
            for rp in rebuilt:
                if len(rp) >= 3:
                    new_surf = dict(surf)
                    new_surf["poly"] = rp
                    result.append(new_surf)

    return result


def _clip_3d_to_remainder(
    poly_3d: list[tuple[float, float, float]],
    remainder_2d,
) -> list[tuple[float, float, float]]:
    """Clip a 3D polygon to a 2D XZ remainder, interpolating Y from original."""
    from shapely.geometry import MultiPolygon, Polygon

    poly_xz = [(p[0], p[2]) for p in poly_3d]
    poly_shp = _shapely_poly(poly_xz)
    if poly_shp is None:
        return []

    try:
        clipped = poly_shp.intersection(remainder_2d)
    except Exception:
        return []

    if clipped.is_empty or clipped.area < 0.01:
        return []

    if isinstance(clipped, Polygon):
        geoms = [clipped]
    elif isinstance(clipped, MultiPolygon):
        geoms = list(clipped.geoms)
    else:
        return []

    geoms = [g for g in geoms if not g.is_empty and g.area >= 0.01]
    if not geoms:
        return []
    geom = max(geoms, key=lambda g: g.area)

    coords_2d = list(geom.exterior.coords)[:-1]
    result = []
    for x, z in coords_2d:
        y = _interpolate_y_on_poly(x, z, poly_3d)
        result.append((x, y, z))
    return result


def _interpolate_y_on_poly(
    x: float, z: float, poly_3d: list[tuple[float, float, float]]
) -> float:
    """Interpolate Y at (x, z) by projecting onto the nearest edge of poly_3d."""
    import math

    best_y = poly_3d[0][1]
    best_dist = float("inf")

    n = len(poly_3d)
    for i in range(n):
        p0 = poly_3d[i]
        p1 = poly_3d[(i + 1) % n]
        dx, dz = p1[0] - p0[0], p1[2] - p0[2]
        edge_len_sq = dx * dx + dz * dz
        if edge_len_sq < 1e-12:
            continue
        t = ((x - p0[0]) * dx + (z - p0[2]) * dz) / edge_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = p0[0] + t * dx
        proj_z = p0[2] + t * dz
        dist = math.sqrt((x - proj_x) ** 2 + (z - proj_z) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_y = p0[1] + t * (p1[1] - p0[1])

    return best_y


def build_thermal_ceilings(
    *,
    exposed_rooms: list,
    oblique_roof_surfaces: list[dict],
    ceiling_planes: list,
    flat_ceilings: list[dict],
    simple_slant_ceilings: list[dict],
    floors_by_story: dict[int, list[list]],
    story_floor_y: dict[int, float],
    story_floor_regions: dict[int, list[tuple[list, float]]] | None = None,
    dormers: list[dict] | None = None,
    gap_walls: list | None = None,
    rooms: list | None = None,
) -> list[dict]:
    """Compute thermal-envelope ceiling surfaces for all exposed rooms."""
    out: list[dict] = []

    story_thermal_context = _build_story_thermal_context(
        exposed_rooms, gap_walls, rooms
    )

    # Compute knee wall gaps first so we can exclude their zones from
    # lower-story thermal surfaces (prevents double ceilings).
    knee_ceilings = _build_knee_wall_ceilings(
        floors_by_story,
        story_floor_y,
        story_floor_regions,
        oblique_roof_surfaces=oblique_roof_surfaces,
    )

    # Build knee zone lookup: for each upper story, the XZ union polygon
    # where the lower floor extends beyond the upper floor.
    knee_zones_by_story: dict[int, object] = {}
    if knee_ceilings:
        from shapely.ops import unary_union

        by_story: dict[int, list] = {}
        for kc in knee_ceilings:
            s = kc["story"]
            xz = _xz_projection(kc["poly"])
            shp = _shapely_poly(xz)
            if shp is not None:
                by_story.setdefault(s, []).append(shp)
        for s, polys in by_story.items():
            knee_zones_by_story[s] = unary_union(polys)

    # 1. Flat ceilings -> thermal-flat
    for fc in flat_ceilings:
        out.append(
            {
                "kind": "thermal-flat",
                "story": fc["story"],
                "poly": list(fc["poly"]),
            }
        )

    # 2. Simple slant ceilings -> thermal-slant + cap
    for sc in simple_slant_ceilings:
        poly = list(sc["poly"])
        story = sc["story"]

        # Find the room whose floor polygon best overlaps this slant surface
        matching_room = _find_best_matching_room(sc, exposed_rooms)

        if matching_room:
            top_y = matching_room["wallTopY"]
            clipped = clip_by_max_y(poly, top_y)
            if len(clipped) >= 3:
                out.append(
                    {
                        "kind": "thermal-slant",
                        "story": story,
                        "poly": clipped,
                    }
                )

                # Flat cap: floor poly minus slant XZ projection
                fp_xz = _xz_projection(matching_room["fp"])
                slant_xz = _xz_projection(clipped)
                fp_shapely = _shapely_poly(fp_xz)
                slant_shapely = _shapely_poly(slant_xz)
                if fp_shapely and slant_shapely:
                    try:
                        remainder = fp_shapely.difference(slant_shapely)
                        cap_polys = _poly_3d_from_shapely(remainder, top_y)
                        for cp in cap_polys:
                            if len(cp) >= 3:
                                out.append(
                                    {
                                        "kind": "thermal-cap",
                                        "story": story,
                                        "poly": cp,
                                    }
                                )
                    except Exception:
                        pass
        else:
            # No matching room found, include as-is
            out.append(
                {
                    "kind": "thermal-slant",
                    "story": story,
                    "poly": poly,
                }
            )

    # 3. Rooms under oblique roof surfaces (not flat, not simple-slant)
    for er in exposed_rooms:
        story = er["story"]
        # Skip flat rooms -- already handled in step 1
        if (er["wallTopY"] - er["wallTopMin"]) < 0.3:
            continue

        # Skip rooms already handled by simple slant ceilings
        is_simple_slant = False
        for sc in simple_slant_ceilings:
            if sc["story"] == story:
                sc_fp = sc.get("poly", [])
                if sc_fp and er["fp"]:
                    sc_cx = sum(p[0] for p in sc_fp) / len(sc_fp)
                    sc_cz = sum(p[2] for p in sc_fp) / len(sc_fp)
                    if abs(sc_cx - er["fcx"]) < 0.5 and abs(sc_cz - er["fcz"]) < 0.5:
                        is_simple_slant = True
                        break
        if is_simple_slant:
            continue

        # Find overlapping oblique surfaces
        surface_indices = _surfaces_overlapping_room(oblique_roof_surfaces, er["fp"])
        if surface_indices:
            thermal = _build_thermal_for_oblique_room(
                er,
                oblique_roof_surfaces,
                surface_indices,
                ceiling_planes,
                story_context=story_thermal_context.get(story),
            )
            out.extend(thermal)
        else:
            # No oblique surface found -- emit flat ceiling as fallback.
            ceiling_y = _robust_ceiling_y(er)
            poly = [(p[0], ceiling_y, p[2]) for p in er["fp"]]
            out.append(
                {
                    "kind": "thermal-flat",
                    "story": story,
                    "poly": poly,
                }
            )

    # 4. Subtract knee-wall zones from lower-story surfaces to prevent
    #    double ceilings (knee wall IS the thermal boundary, not room ceiling)
    out = _subtract_knee_zones_from_surfaces(out, knee_zones_by_story)

    # 5. Add knee wall ceilings
    out.extend(knee_ceilings)

    # 6. Dormer cheeks and headers
    for di, d in enumerate(dormers or []):
        for _ci, cheek in enumerate(d.get("cheeks", [])):
            corners = cheek.get("corners", [])
            if len(corners) >= 3:
                out.append(
                    {
                        "kind": "thermal-dormer-cheek",
                        "story": 0,
                        "poly": [(c[0], c[1], c[2]) for c in corners],
                        "dormer_index": di,
                    }
                )
        header = d.get("header")
        if header:
            corners = header.get("corners", [])
            if len(corners) >= 3:
                out.append(
                    {
                        "kind": "thermal-dormer-header",
                        "story": 0,
                        "poly": [(c[0], c[1], c[2]) for c in corners],
                        "dormer_index": di,
                    }
                )

    # 7. Clean up all surfaces: remove collinear vertices that create
    #    zero-width arms and break viewer triangulation.
    for entry in out:
        entry["poly"] = _remove_collinear_xz(entry["poly"])

    return [e for e in out if len(e["poly"]) >= 3]
