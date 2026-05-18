from __future__ import annotations

from .ceiling_clipping_caps import compute_plane_height_caps
from .ceiling_clipping_initial import build_initial_plane_clips
from .ceiling_clipping_opposing import apply_lower_envelope_cuts
from .math_utils import angle_diff, clip_poly_by_half_plane_2d, clip_poly_by_ridge

# Douglas-Peucker tolerance for snap-robust polygon output. Smaller than any
# meaningful architectural detail (5 mm) but large enough to remove the
# sub-millimetre vertex clusters that polygon-cut operations leave behind.
# Without this the 1 mm lattice snap in `boundary_model._snapped_corners`
# turns near-degenerate corners into self-intersecting polygons that the
# Three.js triangulator silently drops.
_SNAP_ROBUST_SIMPLIFY_TOL_M = 0.005


def _simplify_for_snap(coords: list, tol: float = _SNAP_ROBUST_SIMPLIFY_TOL_M) -> list:
    """Pre-snap and repair the polygon to be lattice-stable.

    The downstream ``boundary_model._snapped_corners`` rounds every vertex to
    a 1 mm lattice. Cut polygons here often have sub-mm vertex clusters and
    sliver-thin features that the snap turns into self-intersections — once
    that happens, ``derive_roof_surfaces_from_boundary_model`` carries the
    invalid corners through to the viewer, which silently drops the face.

    Pipeline here:
      1. Snap each corner to the 1 mm lattice ourselves.
      2. Drop consecutive duplicates produced by the snap.
      3. Simplify with Douglas-Peucker at ``tol`` (5 mm) to remove the
         remaining slivers.
      4. Repair any topological invalidity via ``make_valid`` and keep the
         largest polygon part.
    The result is at mm precision, valid, and re-snapping downstream is
    therefore a no-op.
    """
    if not coords or len(coords) < 3:
        return list(coords)
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
    except ImportError:
        return list(coords)

    snapped: list[tuple[float, float]] = []
    for c in coords:
        sx = round(float(c[0]) * 1000.0) / 1000.0
        sz = round(float(c[1]) * 1000.0) / 1000.0
        if not snapped or snapped[-1] != (sx, sz):
            snapped.append((sx, sz))
    if len(snapped) >= 2 and snapped[-1] == snapped[0]:
        snapped.pop()
    if len(snapped) < 3:
        return list(coords)

    try:
        sp = Polygon(snapped)
        if not sp.is_valid:
            sp = make_valid(sp)
        if sp.geom_type == "GeometryCollection":
            polys = [g for g in sp.geoms if g.geom_type == "Polygon"]
            sp = max(polys, key=lambda g: g.area) if polys else None
        elif sp.geom_type == "MultiPolygon":
            sp = max(sp.geoms, key=lambda g: g.area)
        if sp is None or sp.is_empty or sp.geom_type != "Polygon":
            return list(coords)
        simplified = sp.simplify(tol, preserve_topology=True)
        if simplified.is_empty or simplified.geom_type != "Polygon":
            simplified = sp
        ring = list(simplified.exterior.coords)
        if ring and ring[-1] == ring[0]:
            ring = ring[:-1]
        if len(ring) < 3:
            return list(coords)
        # Re-snap after DP — DP can introduce sub-mm coordinates.
        out: list[tuple[float, float]] = []
        for x, z in ring:
            sx = round(float(x) * 1000.0) / 1000.0
            sz = round(float(z) * 1000.0) / 1000.0
            if not out or out[-1] != (sx, sz):
                out.append((sx, sz))
        if len(out) >= 2 and out[-1] == out[0]:
            out.pop()
        return out if len(out) >= 3 else list(coords)
    except Exception:
        return list(coords)


def _shared_partition_for_gable_pair(
    plane_i: dict,
    plane_j: dict,
    ridge_min_i: float,
    ridge_max_i: float,
    ridge_min_j: float,
    ridge_max_j: float,
    building_footprint: list,
) -> tuple[list, list] | None:
    """Partition the **building footprint** along the gable pair's ridge.

    Both opposing planes share a single building. The natural shared
    envelope is therefore the building footprint, clipped along the ridge
    axis to the pair's combined ridge range, then split by the analytic
    half-plane line (where the two plane equations have equal y).

    Using the building footprint — rather than the union of the two
    individually-clipped polygons — is what makes the partition symmetric.
    Otherwise per-plane room evidence asymmetries (one plane covers more
    rooms than the other) propagate straight into the partition and the
    facade gap survives.

    Returns ``(poly_i, poly_j)`` or ``None`` if the geometry is degenerate.
    """
    if not building_footprint or len(building_footprint) < 3:
        return None

    # Use plane_i's ridge axis as the parametric reference. plane_i and plane_j
    # have nearly opposite ridge axes (180° apart) so plane_j's ridge range is
    # the negation of plane_i's. We project both into plane_i's frame.
    rx, rz = plane_i["ridgeX"], plane_i["ridgeZ"]
    refx, refz = plane_i["ref"]["x"], plane_i["ref"]["z"]

    def _proj_i(t: float, plane_other: dict) -> float:
        # plane_j's ridge axis dotted with plane_i's gives roughly +1 or -1.
        # Reproject plane_j's ridge_min/max into plane_i's parameter space.
        dot = plane_other["ridgeX"] * rx + plane_other["ridgeZ"] * rz
        # Offset between the two ref points along plane_i's ridge axis.
        offset = (plane_other["ref"]["x"] - refx) * rx + (
            plane_other["ref"]["z"] - refz
        ) * rz
        return offset + t * dot

    rj_min_in_i = _proj_i(ridge_min_j, plane_j)
    rj_max_in_i = _proj_i(ridge_max_j, plane_j)
    j_lo, j_hi = sorted((rj_min_in_i, rj_max_in_i))
    combined_min = min(ridge_min_i, j_lo)
    combined_max = max(ridge_max_i, j_hi)

    envelope = list(building_footprint)
    envelope = clip_poly_by_ridge(envelope, rx, rz, refx, refz, combined_min, True)
    if len(envelope) < 3:
        return None
    envelope = clip_poly_by_ridge(envelope, rx, rz, refx, refz, combined_max, False)
    if len(envelope) < 3:
        return None

    # Half-plane line where the two plane equations have equal y. Same formula
    # as in `apply_lower_envelope_cuts`'s opposing branch.
    ai, bi = (
        plane_i["n"]["x"] / plane_i["n"]["y"],
        plane_i["n"]["z"] / plane_i["n"]["y"],
    )
    ci = plane_i["ref"]["y"] + ai * plane_i["ref"]["x"] + bi * plane_i["ref"]["z"]
    aj, bj = (
        plane_j["n"]["x"] / plane_j["n"]["y"],
        plane_j["n"]["z"] / plane_j["n"]["y"],
    )
    cj = plane_j["ref"]["y"] + aj * plane_j["ref"]["x"] + bj * plane_j["ref"]["z"]
    dx, dz, offset = aj - ai, bj - bi, ci - cj
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None

    side_i = clip_poly_by_half_plane_2d(envelope, dx, dz, offset)
    side_j = clip_poly_by_half_plane_2d(envelope, -dx, -dz, -offset)
    if len(side_i) < 3 or len(side_j) < 3:
        return None
    return _simplify_for_snap(side_i), _simplify_for_snap(side_j)


def _partition_gable_pairs(
    ceiling_planes: list,
    plane_clipped: list,
    building_footprint: list,
) -> None:
    """Replace each opposing pair's polygons with a building-footprint partition.

    Tier-6 gables would otherwise leave footprint patches uncovered (one
    plane reached but the opposing plane didn't) or over-covered (both
    reached the same (x,z)). Partitioning the building footprint along the
    analytic ridge line guarantees a clean per-(x,z) winner.
    """
    if not building_footprint or len(building_footprint) < 3:
        return
    n = len(ceiling_planes)
    paired: set[int] = set()
    for i in range(n):
        if i in paired:
            continue
        pi = ceiling_planes[i]
        if len(plane_clipped[i]["clipped"]) < 3:
            continue
        for j in range(i + 1, n):
            if j in paired:
                continue
            pj = ceiling_planes[j]
            if pi["dominantStory"] != pj["dominantStory"]:
                continue
            if len(plane_clipped[j]["clipped"]) < 3:
                continue
            ad = angle_diff(pi["cl"]["avgAzimuth"], pj["cl"]["avgAzimuth"])
            if ad < 140.0:
                continue
            partitioned = _shared_partition_for_gable_pair(
                pi,
                pj,
                plane_clipped[i]["ridgeMin"],
                plane_clipped[i]["ridgeMax"],
                plane_clipped[j]["ridgeMin"],
                plane_clipped[j]["ridgeMax"],
                building_footprint,
            )
            if partitioned is None:
                continue
            plane_clipped[i]["clipped"] = list(partitioned[0])
            plane_clipped[j]["clipped"] = list(partitioned[1])
            paired.add(i)
            paired.add(j)
            break


def clip_ceiling_planes(
    *,
    bldg: dict,
    ceiling_planes: list,
    building_footprint: list,
    exposed_rooms: list,
    all_rooms: list[dict] | None = None,
    top_story: int,
    all_stories: list,
    floors_by_story: dict,
    point_in_poly_xz,
    point_in_poly_2d,
    graph=None,
    wing_polygons: list | None = None,
):
    plane_clipped = build_initial_plane_clips(
        ceiling_planes=ceiling_planes,
        building_footprint=building_footprint,
        exposed_rooms=exposed_rooms,
        all_rooms=all_rooms,
        wing_polygons=wing_polygons,
    )

    plane_max_y = compute_plane_height_caps(
        bldg=bldg,
        ceiling_planes=ceiling_planes,
        plane_clipped=plane_clipped,
        top_story=top_story,
        all_stories=all_stories,
        floors_by_story=floors_by_story,
        point_in_poly_xz=point_in_poly_xz,
        point_in_poly_2d=point_in_poly_2d,
        graph=graph,
    )

    # Replace each opposing-plane pair's polygons with a ridge partition of
    # the building footprint. This is what guarantees a clean per-(x,z)
    # winner across the gable; the subsequent lower-envelope cut then
    # becomes a no-op for those pairs and only acts on L-junctions and
    # incidental overlaps.
    _partition_gable_pairs(ceiling_planes, plane_clipped, building_footprint)

    apply_lower_envelope_cuts(
        ceiling_planes=ceiling_planes,
        plane_clipped=plane_clipped,
    )

    # Final pass: simplify every clipped polygon to a 5 mm Douglas-Peucker
    # tolerance. Polygon-cut operations (footprint clip, ridge clip,
    # half-plane cut, lower-envelope cut) leave sub-millimetre vertex
    # clusters that the downstream `boundary_model._snapped_corners` 1 mm
    # lattice snap turns into self-intersecting polygons.
    for entry in plane_clipped:
        cs = entry.get("clipped") or []
        if len(cs) >= 3:
            entry["clipped"] = _simplify_for_snap(cs)

    return {
        "plane_clipped": plane_clipped,
        "plane_max_y": plane_max_y,
        "junction_patches": [],
        "l_junctions": [],
    }
