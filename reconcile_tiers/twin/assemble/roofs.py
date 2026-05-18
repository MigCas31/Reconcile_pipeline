"""Step 7 of the assembly: Roof per Wing.

Each Wing's roof is composed of architectural-face RoofSurfaces. Per-room
oblique sub-Ceilings are clustered by horizontal-normal direction (sign
of the dot product between projected normals): two sub-ceilings whose
horizontal normals point the same way belong to the same roof face;
opposite-pointing normals are different faces. This collapses the
fragmentation that the per-room emission produces — a 4-room gable
wing with 8 oblique sub-ceilings becomes 2 RoofSurfaces (one per gable
side), not 8.

The clustering is a sign test on horizontal-projection dot products:
no scan-precision tolerance, no plane-similarity threshold. Each
cluster's canonical plane is fitted (SVD) through the union of its
members' corners; the cluster's polygon is the shapely union of its
members' XZ extents, lifted onto the canonical plane.

A Wing with zero oblique top-story sub-Ceilings (e.g. a flat-roof
building) yields no Roof; `Wing.roof` stays `None`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from reconcile_tiers.payload.schema import Plane, Vec3
from reconcile_tiers.twin._geometry import (
    FLOAT_EPS,
    fit_plane,
    lift_xz_to_plane,
    plane_is_horizontal,
    plane_is_vertical,
)
from reconcile_tiers.twin.assemble.ridges import ridges_for_roof
from reconcile_tiers.twin.types import (
    Ceiling,
    CeilingKind,
    Evidence,
    Provenance,
    Roof,
    RoofKind,
    RoofSurface,
    Wing,
)


def roof_for_wing(
    wing: Wing,
    *,
    classification_roof_type: str | None,
    building_uuid: str,
) -> Roof | None:
    """Build a Roof for `wing` from the top-story oblique sub-Ceilings.

    `classification_roof_type` is the building-level roof classification
    string (e.g. "gable", "hip", "shed", "complex", "none"); it
    informs `Roof.kind` only — assignment of fragments to surfaces is
    structural, not classification-driven.
    """
    top_story_rooms = _top_story_rooms(wing)
    if not top_story_rooms:
        return None

    obliques = list(_oblique_subceilings(top_story_rooms))
    if not obliques:
        return None

    # Collect wall peak corners across the wing's top-story rooms.
    # A peak is a corner whose Y is ≥30cm above the wall's other corners
    # (a gable end's apex).
    wall_peaks = _collect_wall_peaks(top_story_rooms)

    clusters = _cluster_by_horizontal_normal(obliques)

    # Derive face regions structurally:
    # - Preferred: ridge axis from gable wall peaks (most reliable when
    #   extraction surfaces gable end walls as pentagons).
    # - Fallback: ridge axis perpendicular to cluster normal direction
    #   through the wing centroid (works for kinked-ceiling gables
    #   where wall peaks are absent).
    face_regions = None
    if len(wall_peaks) >= 2:
        face_regions = _face_regions_from_peaks(wing, wall_peaks)
    if face_regions is None and clusters:
        face_regions = _face_regions_from_cluster_normals(wing, clusters)

    surfaces: list[RoofSurface] = []
    next_idx = 0
    for cluster in clusters:
        for surface in _build_clustered_surfaces(
            cluster,
            wing_id=wing.id,
            start_idx=next_idx,
            wall_peaks=wall_peaks,
            face_regions=face_regions,
        ):
            surfaces.append(surface)
            next_idx += 1
    if not surfaces:
        return None

    roof_skeleton = Roof(
        id=f"{wing.id}::roof",
        kind=_roof_kind_from_classification(classification_roof_type),
        surfaces=tuple(surfaces),
    )
    ridges = ridges_for_roof(roof_skeleton, building_uuid=building_uuid)
    if not ridges:
        return roof_skeleton
    return Roof(
        id=roof_skeleton.id,
        kind=roof_skeleton.kind,
        surfaces=roof_skeleton.surfaces,
        ridges=ridges,
    )


def _top_story_rooms(wing: Wing) -> tuple:
    if not wing.stories:
        return ()
    top_story_index = max(s.rooms[0].story_index for s in wing.stories if s.rooms)
    rooms: list = []
    for story in wing.stories:
        for room in story.rooms:
            if room.story_index == top_story_index:
                rooms.append(room)
    return tuple(rooms)


def _oblique_subceilings(rooms: Iterable) -> Iterable[Ceiling]:
    for room in rooms:
        ceiling = room.ceiling
        if ceiling.kind is CeilingKind.OBLIQUE and ceiling.plane is not None:
            yield ceiling
        elif ceiling.kind is CeilingKind.COMPOSITE:
            for part in ceiling.parts:
                if part.kind is CeilingKind.OBLIQUE and part.plane is not None:
                    yield part


def _face_regions_from_cluster_normals(wing: Wing, clusters: list):
    """Fallback face-region split: when no wall peaks are present
    (typical for kinked-ceiling gables — slopes are in ceilings, not
    in pentagon walls), derive the ridge axis from cluster normals.

    For a gable with 2 opposing clusters, the cluster normal direction
    is the slope axis; the ridge axis is its perpendicular, passing
    through the wing centroid. The wing footprint splits along that
    axis into 2 face regions, one per cluster.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import split as shapely_split

    if not clusters or len(wing.footprint) < 3:
        return None

    # Use the first cluster's normal as the slope-axis direction.
    h_norm = _horizontal_normal(clusters[0][0].plane)
    if h_norm is None:
        return None
    # Ridge axis is perpendicular to the slope direction in XZ.
    perp_x, perp_z = -h_norm[1], h_norm[0]

    try:
        wing_poly = Polygon([(c.x, c.z) for c in wing.footprint]).buffer(0)
    except Exception:
        return None
    if wing_poly.is_empty:
        return None
    cx, cz = wing_poly.centroid.x, wing_poly.centroid.y

    minx, minz, maxx, maxz = wing_poly.bounds
    extent = math.hypot(maxx - minx, maxz - minz) * 4.0
    splitter = LineString(
        [
            (cx - perp_x * extent, cz - perp_z * extent),
            (cx + perp_x * extent, cz + perp_z * extent),
        ]
    )
    try:
        result = shapely_split(wing_poly, splitter)
    except Exception:
        return None
    components = [
        g for g in result.geoms
        if g.geom_type == "Polygon" and not g.is_empty
    ]
    if len(components) < 2:
        return None
    return components


def _face_regions_from_peaks(wing: Wing, peaks: list):
    """Split the wing's top-story footprint along the ridge axis to
    produce face-region polygons.

    Ridge axis: the line connecting the two most-distant peak corners
    in plan view. For a gable building this is the building's main
    ridge. For complex roofs with multiple ridges, this only handles
    the dominant axis (Phase 2 of this redesign would split iteratively).

    Returns a list of shapely Polygons (XZ projection) — one per face
    region. Each region is one side of the ridge axis intersected
    with the wing footprint.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import split as shapely_split

    if len(peaks) < 2 or len(wing.footprint) < 3:
        return None
    # Pick the two peaks that are furthest apart in XZ — that's the
    # main ridge axis of the wing.
    best: tuple[Vec3, Vec3, float] | None = None
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            a, b = peaks[i], peaks[j]
            d = math.hypot(a.x - b.x, a.z - b.z)
            if best is None or d > best[2]:
                best = (a, b, d)
    if best is None or best[2] < FLOAT_EPS:
        return None
    pa, pb, _ = best

    try:
        wing_poly = Polygon(
            [(c.x, c.z) for c in wing.footprint]
        ).buffer(0)
    except Exception:
        return None
    if wing_poly.is_empty:
        return None

    # Extend the ridge line well beyond the wing's bounding box so
    # `split` has clean intersections with the footprint boundary.
    minx, minz, maxx, maxz = wing_poly.bounds
    extent = math.hypot(maxx - minx, maxz - minz) * 4.0
    dx = pb.x - pa.x
    dz = pb.z - pa.z
    edge = math.hypot(dx, dz)
    if edge < FLOAT_EPS:
        return None
    ux, uz = dx / edge, dz / edge
    cx, cz = 0.5 * (pa.x + pb.x), 0.5 * (pa.z + pb.z)
    splitter = LineString(
        [(cx - ux * extent, cz - uz * extent), (cx + ux * extent, cz + uz * extent)]
    )

    try:
        result = shapely_split(wing_poly, splitter)
    except Exception:
        return None
    components = [g for g in result.geoms if g.geom_type == "Polygon" and not g.is_empty]
    if len(components) < 2:
        return None
    return components


def _collect_wall_peaks(rooms) -> list:
    """Find peak corners of top-story walls — corners whose Y is at
    least 30 cm above the wall's other corners. These are architectural
    apex points (gable ends) that should anchor cluster plane fits.
    """
    peaks: list = []
    for room in rooms:
        for wall in room.walls:
            ys = sorted(c.y for c in wall.polygon)
            if len(ys) < 3:
                continue
            if ys[-1] - ys[-2] > 0.3:
                # The single highest corner is a peak.
                peak_corner = max(wall.polygon, key=lambda c: c.y)
                peaks.append(peak_corner)
    return peaks


def _cluster_by_horizontal_normal(
    obliques: list[Ceiling],
) -> list[list[Ceiling]]:
    """Group sub-ceilings whose horizontal-projection normals point the
    same way. Structural sign test: two sub-ceilings A and B are in the
    same cluster iff `A.h_normal · B.h_normal > 0` (positive dot
    product → same architectural face). A and B are in different
    clusters iff the dot product is negative; iff zero, they're treated
    as different faces (perpendicular slopes; safer to keep apart).
    """
    clusters: list[list[Ceiling]] = []
    for sub in obliques:
        h_norm = _horizontal_normal(sub.plane)
        if h_norm is None:
            clusters.append([sub])
            continue
        placed = False
        for cluster in clusters:
            rep = _horizontal_normal(cluster[0].plane)
            if rep is None:
                continue
            dot = h_norm[0] * rep[0] + h_norm[1] * rep[1]
            if dot > FLOAT_EPS:
                cluster.append(sub)
                placed = True
                break
        if not placed:
            clusters.append([sub])
    return clusters


def _horizontal_normal(plane: Plane) -> tuple[float, float] | None:
    """Unit-length XZ projection of the plane normal, or None if the
    plane is exactly horizontal (its normal has no XZ component)."""
    mag = math.hypot(plane.a, plane.c)
    if mag < FLOAT_EPS:
        return None
    return (plane.a / mag, plane.c / mag)


def _build_clustered_surfaces(
    cluster: list[Ceiling],
    *,
    wing_id: str,
    start_idx: int,
    wall_peaks: list[Vec3] = (),
    face_regions=None,
) -> list[RoofSurface]:
    """Emit one RoofSurface per *connected component* of the cluster's
    plan-view union. The cluster shares a single architectural face
    family (same horizontal-normal direction); each connected component
    is one face within that family. The canonical plane is fitted
    through every member's corners *plus any gable wall peak* whose
    XZ falls inside the cluster's union (architectural anchors)."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    all_corners: list[Vec3] = []
    member_polys = []
    member_ids: list[str] = []
    for member in cluster:
        all_corners.extend(member.polygon)
        try:
            poly = Polygon([(c.x, c.z) for c in member.polygon]).buffer(0)
        except Exception:
            continue
        if poly.is_empty:
            continue
        member_polys.append(poly)
        member_ids.append(member.id)
    if len(all_corners) < 3 or not member_polys:
        return []

    cluster_union = unary_union(member_polys)
    # Find wall peaks inside the cluster's plan-view union. A peak is
    # an architectural ridge anchor — the cluster plane MUST pass
    # through it. To enforce this, we replicate each peak many times
    # in the SVD fit data so it dominates the noisy sub-ceiling
    # corners (~hundreds of those, ~1-3 peaks per cluster).
    peak_anchors: list[Vec3] = []
    for peak in wall_peaks:
        if cluster_union.contains(Point(peak.x, peak.z)):
            peak_anchors.append(peak)

    if peak_anchors:
        # Heavy weight: each peak counted as many times as there are
        # sub-ceiling corners. The SVD fit then balances "all peaks"
        # against "all sub-ceiling corners" with equal total weight.
        per_peak_weight = max(1, len(all_corners) // max(1, len(peak_anchors)))
        for peak in peak_anchors:
            for _ in range(per_peak_weight):
                all_corners.append(peak)

    plane = fit_plane(all_corners)
    if plane is None or plane_is_horizontal(plane) or plane_is_vertical(plane):
        return []

    # If we have face regions from the ridge-axis split, the cluster's
    # polygon = the face region matching the cluster's normal direction.
    # This replaces the noisy slope_polygon union with the architectural
    # face's actual extent.
    if face_regions:
        cluster_h_normal = _horizontal_normal(plane)
        cluster_centroid = cluster_union.centroid
        if cluster_h_normal is not None and not cluster_centroid.is_empty:
            best_region = None
            best_score = float("-inf")
            for region in face_regions:
                region_centroid = region.centroid
                if region_centroid.is_empty:
                    continue
                # The cluster's normal points OUTWARD from the ridge
                # toward this face. The region whose centroid is in
                # the same direction (positive dot product with the
                # cluster normal, measured from the cluster centroid)
                # is the matching face.
                vx = region_centroid.x - cluster_centroid.x
                vz = region_centroid.y - cluster_centroid.y
                dot = vx * cluster_h_normal[0] + vz * cluster_h_normal[1]
                if dot > best_score:
                    best_score = dot
                    best_region = region
            if best_region is not None:
                # Replace the cluster polygon with the face region —
                # but keep the plane fitted from sub-ceiling corners
                # plus weighted peaks (already done above).
                cluster_union = best_region

    union = cluster_union
    # Snap near-coincident vertices to remove the millimetre-scale
    # zigzags that shapely emits when input polygons almost-touch.
    # This is a *float-precision* cleanup, not an architectural
    # tolerance: the snap distance (1 mm) is well below any real
    # building feature.
    try:
        union = union.simplify(0.001, preserve_topology=True)
    except Exception:
        pass
    if union.geom_type == "Polygon":
        components = [union]
    elif union.geom_type == "MultiPolygon":
        components = list(union.geoms)
    else:
        return []

    surfaces: list[RoofSurface] = []
    for offset, comp in enumerate(components):
        coords = list(comp.exterior.coords)
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        # Drop consecutive near-duplicate corners that simplify() may
        # have left behind.
        cleaned: list[tuple[float, float]] = []
        for x, z in coords:
            if not cleaned or (
                abs(cleaned[-1][0] - x) > 1e-3 or abs(cleaned[-1][1] - z) > 1e-3
            ):
                cleaned.append((x, z))
        if len(cleaned) >= 2 and (
            abs(cleaned[0][0] - cleaned[-1][0]) < 1e-3
            and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-3
        ):
            cleaned.pop()
        coords = cleaned
        if len(coords) < 3:
            continue
        flat_corners = tuple(Vec3(x=float(x), y=0.0, z=float(z)) for x, z in coords)
        polygon = lift_xz_to_plane(flat_corners, plane)
        idx = start_idx + offset
        parent_ids = tuple(member_ids)
        evidence = Evidence(
            provenance=Provenance(kind="computed", source="roof_face_cluster"),
            geometry=polygon,
            parents=parent_ids,
        )
        surfaces.append(
            RoofSurface(
                id=f"{wing_id}::roof-surface::{idx}",
                polygon=polygon,
                plane=plane,
                evidence=(evidence,),
            )
        )
    return surfaces


_KIND_MAP = {
    "gable": RoofKind.GABLE,
    "hip": RoofKind.HIP,
    "mansard": RoofKind.MANSARD,
    "shed": RoofKind.SHED,
    "flat": RoofKind.FLAT,
}


def _roof_kind_from_classification(roof_type: str | None) -> RoofKind:
    if roof_type is None:
        return RoofKind.COMPLEX
    return _KIND_MAP.get(roof_type, RoofKind.COMPLEX)
