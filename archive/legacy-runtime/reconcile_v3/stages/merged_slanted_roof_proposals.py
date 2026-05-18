"""merged_slanted_roof_proposals -- labelable pieces of merged slanted-roof planes.

A "merged plane" is the cluster of near-coplanar ``V3RoofProposal``s (same
normal direction and offset within tolerance). A "segment" is one connected
piece of that merged plane's footprint after:

  1. Clipping to the **building boundary** (XZ union of all floor slabs).
  2. Splitting at **opposing-plane seams** -- lines where the merged plane
     intersects another merged plane in 3D. We keep only the half where
     this plane is the highest (i.e. rain hits this plane, not the other).
  3. Splitting along **XZ boundaries of rain-exposed room/part/gap pieces**
     sitting under the plane -- every segment corresponds to exactly one
     rain-exposed floor piece.

Each segment is the unit the labeler accepts or rejects. The payload carries
the full provenance needed to reverse-engineer "why was this segment a roof /
why was it not" without re-running the proposer: contributing raw proposals,
merge thresholds, opposing merged planes and their seam lines, the room/part
boundaries that split it, and the building boundary polygon used for the
outer clip.
"""

from __future__ import annotations

from math import hypot
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from ..audit import HypothesisTrace, note_geom_skip
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Gap, V3MergedRoofSegment, V3RoofProposal, V3Slab
from .slanted_roof_proposals import (
    _exposed_pieces,
    _footprint_polygon_3d_to_shapely,
    _rain_exposure,
)
from .slanted_roofs import project_xz_onto_plane

#: Plane-clustering thresholds -- proposals join a cluster when their
#: upward-normalized normal dot >= this AND |Δd| <= ``D_ABS_MAX``. Matched to
#: the canonical ``roof_algorithms_py/oblique_clustering.py`` gate
#: (``azimuth<30 deg``, ``incl<15 deg``, ``COPLANAR_TOL=0.5m``): a dot of ~0.94
#: approximates the 30 degx15 deg angular cone, and 0.5m matches ``COPLANAR_TOL``.
NORMAL_DOT_MIN = 0.94
D_ABS_MAX = 0.50
#: Azimuth cosine below which two clusters are considered "opposing" (>30 deg
#: apart in horizontal direction). Matches the viewer threshold.
OPPOSING_COS_AZIMUTH_MAX = 0.866
#: Minimum segment area in m^2 -- below this we treat the piece as a split
#: artefact (slivers produced by intersecting the seam line with a polygon
#: vertex) and drop it rather than emit a degenerate label target.
MIN_SEGMENT_AREA_M2 = 0.05
#: Minimum ribbon thickness in metres (``2 * area / perimeter``) -- drops
#: thin slivers that survive the area gate (long + narrow polygons can have
#: area >= 0.05 m^2 but width ≪ 1 cm). Mirrors ``EXPOSED_PIECE_MIN_WIDTH_M``
#: from ``slanted_roof_proposals.py``.
MIN_SEGMENT_WIDTH_M = 0.10


def _normalized_plane(
    plane: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    a, b, c, d = plane
    n = hypot(a, hypot(b, c))
    if not (n > 1e-9):
        return None
    s = 1.0 / n if b >= 0 else -1.0 / n
    return (a * s, b * s, c * s, d * s)


def _cluster_proposals_by_plane(
    proposals: list[V3RoofProposal],
) -> list[dict[str, Any]]:
    """Group proposals whose upward-normalized planes match within tolerance.

    The running mean of each cluster's plane is re-normalized after every
    add so the cluster plane stays a valid unit-normal (a, b, c, d).
    """

    clusters: list[dict[str, Any]] = []
    for proposal in proposals:
        plane_n = _normalized_plane(proposal.plane)
        if plane_n is None:
            continue
        assigned = False
        for cluster in clusters:
            ax, ay, az, ad = cluster["plane"]
            dot = ax * plane_n[0] + ay * plane_n[1] + az * plane_n[2]
            if dot >= NORMAL_DOT_MIN and abs(ad - plane_n[3]) <= D_ABS_MAX:
                cluster["members"].append(proposal)
                n = len(cluster["members"])
                bx, by, bz, bd = cluster["plane"]
                mx = ((n - 1) * bx + plane_n[0]) / n
                my = ((n - 1) * by + plane_n[1]) / n
                mz = ((n - 1) * bz + plane_n[2]) / n
                md = ((n - 1) * bd + plane_n[3]) / n
                norm = hypot(mx, hypot(my, mz)) or 1.0
                cluster["plane"] = (mx / norm, my / norm, mz / norm, md / norm)
                assigned = True
                break
        if not assigned:
            clusters.append({"plane": plane_n, "members": [proposal]})
    for idx, cluster in enumerate(clusters):
        canonical = min(
            (m.id for m in cluster["members"] if isinstance(m.id, str)),
            default=f"cluster-{idx}",
        )
        cluster["canonical_id"] = canonical
    return clusters


def _cos_azimuth(
    p: tuple[float, float, float, float],
    q: tuple[float, float, float, float],
) -> float:
    ha = hypot(p[0], p[2])
    hb = hypot(q[0], q[2])
    if ha < 1e-4 or hb < 1e-4:
        return 1.0
    return (p[0] * q[0] + p[2] * q[2]) / (ha * hb)


def _half_plane_for_opposing(
    own: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Coefficients (A, B, C) such that ``A*x + B*z + C >= 0`` marks where the
    **own** plane is covered (own.y <= other.y). The rain-hitting half is
    therefore ``A*x + B*z + C <= 0``.
    """

    a_i, b_i, _, d_i = own
    a_j, b_j, _, d_j = other
    # Using full plane equation, z-coefficient becomes bJ*cI - bI*cJ.
    c_i = own[2]
    c_j = other[2]
    return (b_j * a_i - b_i * a_j, b_j * c_i - b_i * c_j, b_j * d_i - b_i * d_j)


def _half_plane_polygon(
    coeffs: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    keep_negative: bool = True,
) -> Polygon | None:
    """Return a polygon covering ``{(x,z) : A*x + B*z + C <= 0}`` (or >=0 if
    ``keep_negative`` is False) intersected with the given XZ bounding box.

    Falls back to None if the half-plane is degenerate or doesn't touch the
    box.
    """

    A, B, C = coeffs
    if abs(A) < 1e-12 and abs(B) < 1e-12:
        # Degenerate -- whole plane trivially on one side.
        trivial = C >= 0
        return (
            box(*bounds)
            if (trivial and not keep_negative) or (not trivial and keep_negative)
            else None
        )

    minx, minz, maxx, maxz = bounds
    pad = max(maxx - minx, maxz - minz, 1.0) * 4.0
    minx -= pad
    minz -= pad
    maxx += pad
    maxz += pad

    corners = [(minx, minz), (maxx, minz), (maxx, maxz), (minx, maxz)]
    values = [A * x + B * z + C for x, z in corners]

    kept: list[tuple[float, float]] = []
    n = len(corners)
    for i in range(n):
        vi = values[i]
        vj = values[(i + 1) % n]
        pi = corners[i]
        pj = corners[(i + 1) % n]
        inside_i = (vi <= 0) if keep_negative else (vi >= 0)
        inside_j = (vj <= 0) if keep_negative else (vj >= 0)
        if inside_i:
            kept.append(pi)
        if inside_i != inside_j:
            t = vi / (vi - vj) if abs(vi - vj) > 1e-18 else 0.0
            kept.append((pi[0] + t * (pj[0] - pi[0]), pi[1] + t * (pj[1] - pi[1])))
    if len(kept) < 3:
        return None
    try:
        poly = Polygon(kept)
    except Exception:
        return None
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _building_boundary(slabs: list[V3Slab]) -> Polygon | MultiPolygon | None:
    polys: list[Polygon] = []
    for slab in slabs:
        p = _footprint_polygon_3d_to_shapely(slab.polygon)
        if p is not None:
            polys.append(p)
    if not polys:
        return None
    union = unary_union(polys)
    if union.is_empty:
        return None
    # Shapely's unary_union can return a MultiPolygon even when the result is
    # a single connected region -- unwrap so downstream Polygon-only consumers
    # (e.g. building_boundary_xz ring extraction) see the actual polygon.
    if isinstance(union, MultiPolygon) and len(union.geoms) == 1:
        return union.geoms[0]
    return union


def _collect_rain_exposed_pieces(
    rooms: list[V3Room],
    gaps: list[V3Gap],
) -> list[dict[str, Any]]:
    """One record per connected rain-exposed piece per room/part/gap.

    Gaps that are closed contribute their footprint as a distinct "piece" so
    that a roof plane spanning a gap gets split at the room/gap boundary.
    Rooms contribute the connected components of their floor polygon after
    subtracting higher-floor overlap (same ``_rain_exposure`` the proposer
    uses).
    """

    pieces: list[dict[str, Any]] = []
    for room in rooms:
        exposure = _rain_exposure(room, rooms, gaps)
        connected = _exposed_pieces(exposure)
        floor_y = float(room.floor_polygon[0][1]) if room.floor_polygon else 0.0
        for idx, poly in enumerate(connected):
            pieces.append(
                {
                    "kind": "room",
                    "room_id": room.identifier,
                    "room_index": room.index,
                    "story": room.story,
                    "piece_index": idx,
                    "polygon": poly,
                    "floor_y": floor_y,
                }
            )
    for gap in gaps:
        if gap.status != "closed":
            continue
        gap_poly = _footprint_polygon_3d_to_shapely(gap.footprint_xz)
        if gap_poly is None or gap_poly.is_empty:
            continue
        pieces.append(
            {
                "kind": "gap",
                "gap_id": gap.id,
                "floor_y": gap.floor_y,
                "ceiling_y": gap.ceiling_y,
                "polygon": gap_poly,
            }
        )
    return pieces


def _polygon_min_width(poly: Polygon) -> float:
    """Ribbon-thickness metric: 2 * area / perimeter. A 1 m x 1 m square gets
    0.5 (inscribed circle radius); a very long thin ribbon approaches 0."""

    perim = float(poly.length)
    if perim <= 0.0:
        return 0.0
    return 2.0 * float(poly.area) / perim


def _polygon_coords(poly: Polygon) -> list[tuple[float, float]]:
    """Return the exterior ring as (x, z) tuples with the closing vertex
    stripped and sub-cm noise removed via Douglas-Peucker simplify.

    Shapely intersections emit near-duplicate vertices and collinear runs
    (sub-mm to cm-scale geometric noise). Three.js's ``createPolygonMesh``
    builds the mesh's normal frame from only the first three vertices, so a
    near-duplicate pair at positions 0 and 1 produces a cross-product of
    floating-point noise -- the mesh is then flattened onto an arbitrary
    plane that does not match the merged-plane inclination. A 1 cm
    ``simplify`` tolerance keeps the polygon shape intact at the physical
    resolution we care about while guaranteeing each kept vertex is a real
    corner the normal frame can rely on."""

    simplified = poly.simplify(0.01, preserve_topology=True)
    if simplified.is_empty or simplified.geom_type != "Polygon":
        return []
    coords = list(simplified.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(z)) for x, z in coords]


def _expand_multipolygon(
    geom: Polygon | MultiPolygon | None,
) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    if isinstance(geom, Polygon):
        return [geom]
    # GeometryCollection -- pick polygon components.
    out: list[Polygon] = []
    for g in getattr(geom, "geoms", []):
        if isinstance(g, Polygon) and not g.is_empty:
            out.append(g)
    return out


def _member_snapshots(members: list[V3RoofProposal]) -> list[dict[str, Any]]:
    snapshots = []
    for m in members:
        snapshots.append(
            {
                "id": m.id,
                "plane": list(m.plane),
                "corners": [list(c) for c in m.corners],
                "features": dict(m.features or {}),
                "heuristic_label": m.heuristic_label,
                "segment_index": m.segment_index,
                "source_room_id": m.source_room_id,
                "source_wall_id": m.source_wall_id,
                "slab_room_id": m.slab_room_id,
                "slab_id": m.slab_id,
            }
        )
    return snapshots


def _piece_ref(piece: dict[str, Any]) -> dict[str, Any]:
    if piece["kind"] == "room":
        return {
            "kind": "room",
            "room_id": piece["room_id"],
            "room_index": piece["room_index"],
            "story": piece["story"],
            "piece_index": piece["piece_index"],
        }
    return {
        "kind": "gap",
        "gap_id": piece["gap_id"],
        "floor_y": piece["floor_y"],
        "ceiling_y": piece["ceiling_y"],
    }


def _cluster_heuristic_label(members: list[V3RoofProposal]) -> str:
    labels = {m.heuristic_label for m in members}
    if "accepted" in labels:
        return "accepted"
    if "rejected" in labels:
        return "rejected"
    return "not_evaluated"


def build_merged_roof_segments(
    building_uuid: str,
    rooms: list[V3Room],
    gaps: list[V3Gap],
    slabs: list[V3Slab],
    roof_proposals: list[V3RoofProposal],
) -> list[V3MergedRoofSegment]:
    """Cluster ``roof_proposals`` by plane, clip + split their unioned
    footprints into rain-exposed-piece-sized labelable segments, and attach
    full provenance for downstream reverse-engineering.
    """

    if not roof_proposals:
        return []

    clusters = _cluster_proposals_by_plane(roof_proposals)
    if not clusters:
        return []

    # Pre-compute per-cluster member XZ union, bbox, and serializable boundary.
    for cluster in clusters:
        member_polys: list[Polygon] = []
        for m in cluster["members"]:
            mp = _footprint_polygon_3d_to_shapely(m.corners)
            if mp is not None:
                member_polys.append(mp)
        cluster["member_polys"] = member_polys
        cluster["union_xz"] = unary_union(member_polys) if member_polys else None

    boundary = _building_boundary(slabs)
    boundary_xz_ring = (
        _polygon_coords(boundary) if isinstance(boundary, Polygon) else None
    )

    rain_pieces = _collect_rain_exposed_pieces(rooms, gaps)

    cluster_params = {
        "normal_dot_min": NORMAL_DOT_MIN,
        "d_abs_max": D_ABS_MAX,
        "opposing_cos_azimuth_max": OPPOSING_COS_AZIMUTH_MAX,
    }

    segments: list[V3MergedRoofSegment] = []
    for cluster in clusters:
        union_xz = cluster["union_xz"]
        if union_xz is None or union_xz.is_empty:
            continue

        # 1. Clip to building boundary
        if boundary is not None and not boundary.is_empty:
            try:
                union_xz = union_xz.intersection(boundary)
            except Exception as exc:
                note_geom_skip(exc, "merged_roof.boundary_clip")
        if union_xz is None or union_xz.is_empty:
            continue

        # 2. Determine opposing clusters + split by their seams
        own_plane = cluster["plane"]
        own_bounds = union_xz.bounds
        opposing: list[dict[str, Any]] = []
        for other in clusters:
            if other is cluster:
                continue
            if other["union_xz"] is None or other["union_xz"].is_empty:
                continue
            other_bounds = other["union_xz"].bounds
            if (
                own_bounds[2] < other_bounds[0]
                or other_bounds[2] < own_bounds[0]
                or own_bounds[3] < other_bounds[1]
                or other_bounds[3] < own_bounds[1]
            ):
                continue
            if _cos_azimuth(own_plane, other["plane"]) >= OPPOSING_COS_AZIMUTH_MAX:
                continue
            opposing.append(other)

        # Split (not clip) along each opposing-plane seam. Each seam line
        # divides the cluster footprint into "rain-hitting" and "covered"
        # halves; BOTH are kept, each becoming an independently-labelable
        # segment. ``side_bits`` records on which side of each seam the
        # region lies ("rain" = this plane is highest, "covered" = the
        # opposing plane sits above this one), so a downstream
        # reverse-engineering pass can tell a ridge front from its back.
        regions: list[dict[str, Any]] = [{"poly": union_xz, "sides": {}}]
        for opp in opposing:
            coeffs = _half_plane_for_opposing(own_plane, opp["plane"])
            next_regions: list[dict[str, Any]] = []
            for region in regions:
                region_poly = region["poly"]
                if region_poly is None or region_poly.is_empty:
                    continue
                rain_half = _half_plane_polygon(
                    coeffs, region_poly.bounds, keep_negative=True
                )
                covered_half = _half_plane_polygon(
                    coeffs, region_poly.bounds, keep_negative=False
                )
                for half, side in ((rain_half, "rain"), (covered_half, "covered")):
                    if half is None:
                        continue
                    try:
                        piece = region_poly.intersection(half)
                    except Exception as exc:
                        note_geom_skip(exc, "merged_roof.half_plane_clip")
                        continue
                    if piece.is_empty:
                        continue
                    next_regions.append(
                        {
                            "poly": piece,
                            "sides": {**region["sides"], opp["canonical_id"]: side},
                        }
                    )
            regions = next_regions
            if not regions:
                break
        if not regions:
            continue

        opposing_canonicals = tuple(opp["canonical_id"] for opp in opposing)
        opposing_planes = tuple(
            tuple(float(x) for x in opp["plane"]) for opp in opposing
        )
        member_snapshots = _member_snapshots(cluster["members"])
        member_ids = tuple(m.id for m in cluster["members"])
        heuristic_label = _cluster_heuristic_label(cluster["members"])

        # Intersect each region with each rain-exposed piece -- one segment
        # per (cluster x opposing-sides x piece) connected component.
        segment_counter = 0
        for region in regions:
            region_poly = region["poly"]
            region_sides = region["sides"]
            for piece in rain_pieces:
                piece_poly = piece["polygon"]
                try:
                    sub = region_poly.intersection(piece_poly)
                except Exception as exc:
                    note_geom_skip(exc, "merged_roof.rain_piece_clip")
                    continue
                for connected in _expand_multipolygon(sub):
                    if connected.area < MIN_SEGMENT_AREA_M2:
                        continue
                    if _polygon_min_width(connected) < MIN_SEGMENT_WIDTH_M:
                        continue
                    coords_xz = _polygon_coords(connected)
                    if len(coords_xz) < 3:
                        continue
                    footprint_xz: list[tuple[float, float, float]] = [
                        (float(x), 0.0, float(z)) for x, z in coords_xz
                    ]
                    corners = project_xz_onto_plane(footprint_xz, own_plane)
                    inner = f"{
                        cluster['canonical_id'].rsplit(
                            '::',
                            1,
                        )[-1]
                    }:seg-{segment_counter}"
                    seg_id = make_element_id(
                        building_uuid, "v3-merged-roof-segment", inner
                    )
                    piece_ref = _piece_ref(piece)
                    features = {
                        "area_m2": round(float(connected.area), 4),
                        "perimeter_m": round(float(connected.length), 4),
                        "member_count": len(member_ids),
                        "opposing_cluster_count": len(opposing_canonicals),
                        "piece_kind": piece_ref["kind"],
                        "clipped_by_building_boundary": boundary is not None,
                        "rain_hitting_side_count": sum(
                            1 for v in region_sides.values() if v == "rain"
                        ),
                        "covered_side_count": sum(
                            1 for v in region_sides.values() if v == "covered"
                        ),
                    }
                    opposing_sides = tuple(
                        (canonical, region_sides.get(canonical, "unknown"))
                        for canonical in opposing_canonicals
                    )
                    segments.append(
                        V3MergedRoofSegment(
                            id=seg_id,
                            cluster_canonical_id=cluster["canonical_id"],
                            merged_plane=tuple(float(x) for x in own_plane),
                            corners=corners,
                            footprint_xz=footprint_xz,
                            member_proposal_ids=member_ids,
                            member_snapshots=tuple(member_snapshots),
                            cluster_params=cluster_params,
                            building_boundary_xz=boundary_xz_ring,
                            opposing_cluster_canonicals=opposing_canonicals,
                            opposing_planes=opposing_planes,
                            room_boundary_refs=(piece_ref,),
                            features=features,
                            heuristic_label=heuristic_label,  # type: ignore[arg-type]
                            trace=HypothesisTrace(
                                stage="merged_slanted_roof_proposals",
                                rule="cluster-split-by-opposing-seams-and-piece-boundaries",
                                inputs={
                                    "cluster_canonical_id": cluster["canonical_id"],
                                    "cluster_member_count": len(member_ids),
                                    "opposing_canonicals": list(opposing_canonicals),
                                    "opposing_sides": list(opposing_sides),
                                    "piece_ref": piece_ref,
                                    "heuristic_label": heuristic_label,
                                },
                                decision_reason=(
                                    "Merged-plane footprint clipped to the XZ union of "
                                    "floor slabs (building boundary), split along "
                                    "opposing-merged-plane seams into rain-hitting AND "
                                    "covered sides (both kept as independent labelable "
                                    f"segments, azimuth cos < "
                                    f"{OPPOSING_COS_AZIMUTH_MAX}), "
                                    "then intersected with each rain-exposed "
                                    "room/part/gap piece. Each connected component "
                                    "above the minimum area becomes one labelable "
                                    "segment carrying full cluster + opposing + piece "
                                    "provenance for reverse-engineering."
                                ),
                            ),
                        )
                    )
                segment_counter += 1

    return segments
