"""slanted_roofs — slanted roof detection + SlopeHypothesis fusion.

Three evidence sources are fused into a ``SlopeHypothesis`` per room:

1. Oblique scan-segment clusters (direct, high confidence) — reuses
   ``reconcile.roof_algorithms_py.oblique_clustering.cluster_oblique_segments``.
2. Wall-height asymmetry: one exterior wall far taller than its opposite.
3. Missing exterior wall: a part-boundary floor edge with no wall on it.

Sources 2 and 3 either corroborate an existing cluster (boost confidence)
or fire on their own and emit an *inferred* hypothesis. Weak hypotheses
become ``V3UnresolvedRegion(reason="slope-evidence-weak")`` — never
silently dropped, never upgraded to geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from reconcile.extract3d.overlaps import floor_polygon_to_shapely
from reconcile.roof_algorithms_py.math_utils import angle_diff, plane_normal
from reconcile.roof_algorithms_py.oblique_clustering import cluster_oblique_segments

from ..audit import HypothesisTrace
from ..constants import (
    CLUSTER_AZIMUTH_TOL_DEG,
    CONFIDENCE_ASYMMETRIC_ONLY,
    CONFIDENCE_BOOST_ASYMMETRIC,
    CONFIDENCE_BOOST_MISSING_WALL,
    CONFIDENCE_CLUSTER,
    CONFIDENCE_MISSING_WALL_ONLY,
    KNEE_DELTA_M,
    MIN_OBLIQUE_SEGMENT_LEN_M,
    PART_BOUNDARY_SEARCH_M,
    SCAN_NOISE_M,
    SLOPE_HYPOTHESIS_MIN_CONFIDENCE,
)
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Part, V3SlantedRoof, V3UnresolvedRegion


@dataclass(frozen=True)
class SlopeHypothesis:
    room_id: str
    direction_xz: tuple[float, float]
    confidence: float
    sources: tuple[str, ...]
    plane: tuple[float, float, float, float] | None
    cluster_index: int | None


def _collect_room_oblique_segments(room: V3Room) -> list[dict]:
    segments: list[dict] = []
    for wall in room.walls:
        corners = wall.get("corners") or []
        n = len(corners)
        if n < 3:
            continue
        for i in range(n):
            pa = corners[i]
            pb = corners[(i + 1) % n]
            if pa[1] < pb[1]:
                pa, pb = pb, pa
            dx, dy, dz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
            length = sqrt(dx * dx + dy * dy + dz * dz)
            if length < MIN_OBLIQUE_SEGMENT_LEN_M:
                continue
            horiz = sqrt(dx * dx + dz * dz)
            incl = atan2(abs(dy), horiz) * 180.0 / 3.141592653589793
            if incl <= 5.0 or incl >= 80.0:
                continue
            azimuth = (atan2(dx, dz) * 180.0 / 3.141592653589793) % 360.0
            segments.append(
                {
                    "a": list(pa),
                    "b": list(pb),
                    "incl": incl,
                    "azimuth": azimuth,
                    "len": length,
                    "story": room.story,
                    "room_idx": room.index,
                    "room_id": room.identifier,
                    "wall_id": wall.get("id"),
                }
            )
    return segments


def _plane_tuple_from_cluster(cl: dict) -> tuple[float, float, float, float]:
    n = plane_normal(cl["avgAzimuth"], cl["avgIncl"])
    ref = cl["refPt"]
    d = -(n["x"] * ref["x"] + n["y"] * ref["y"] + n["z"] * ref["z"])
    return (float(n["x"]), float(n["y"]), float(n["z"]), float(d))


def project_xz_onto_plane(
    corners_xz: list[tuple[float, float, float]],
    plane: tuple[float, float, float, float],
) -> list[tuple[float, float, float]]:
    """Return corners with Y solved from the plane a·x + b·y + c·z + d = 0.

    Falls back to the supplied Y when the plane is effectively horizontal
    (|b| below a tiny epsilon), which means the caller doesn't know a real
    slope and must not invent one.
    """
    a, b, c, d = plane
    if abs(b) < 1e-6:
        return list(corners_xz)
    return [
        (float(x), float(-(a * x + c * z + d) / b), float(z)) for x, _, z in corners_xz
    ]


def _dedupe_and_despike(
    coords: list[tuple[float, float]],
    *,
    eps_dist: float = 1e-3,
    eps_cos: float = 1e-3,
) -> list[tuple[float, float]]:
    """Sanitize an XZ ring: drop zero-length edges and colinear-reversal spikes.

    Both arise when a convex hull is intersected with a non-convex room
    polygon — Shapely re-visits pinch points, producing zero-area detours
    that the viewer's edge-loop renderer would otherwise draw as visible
    lines sticking out of the slanted roof.

    Accepts an open or closed ring; returns a closed ring (first == last)
    of length ``n_distinct + 1`` when ``n_distinct >= 3``, else ``[]``.
    """
    if len(coords) < 2:
        return []
    pts = list(coords[:-1]) if coords[0] == coords[-1] else list(coords)

    pruned: list[tuple[float, float]] = []
    for p in pts:
        if not pruned or _xz_dist(pruned[-1], p) > eps_dist:
            pruned.append(p)
    while len(pruned) >= 2 and _xz_dist(pruned[-1], pruned[0]) <= eps_dist:
        pruned.pop()

    changed = True
    while changed and len(pruned) >= 3:
        changed = False
        i = 0
        while i < len(pruned) and len(pruned) >= 3:
            a = pruned[(i - 1) % len(pruned)]
            b = pruned[i]
            c = pruned[(i + 1) % len(pruned)]
            ax, az = b[0] - a[0], b[1] - a[1]
            cx, cz = c[0] - b[0], c[1] - b[1]
            la = sqrt(ax * ax + az * az)
            lc = sqrt(cx * cx + cz * cz)
            if la < eps_dist or lc < eps_dist:
                pruned.pop(i)
                changed = True
                continue
            cos_t = (ax * cx + az * cz) / (la * lc)
            if cos_t <= -1.0 + eps_cos:
                pruned.pop(i)
                changed = True
                continue
            i += 1

    if len(pruned) < 3:
        return []
    pruned.append(pruned[0])
    return pruned


def _xz_dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    dx = p[0] - q[0]
    dz = p[1] - q[1]
    return sqrt(dx * dx + dz * dz)


def _cluster_corners(
    cl: dict,
    plane: tuple[float, float, float, float],
    clip_poly: Polygon | MultiPolygon | None,
) -> list[tuple[float, float, float]]:
    from shapely.geometry import MultiPoint

    pts: list[tuple[float, float]] = []
    for seg in cl["segs"]:
        for end in (seg["a"], seg["b"]):
            pts.append((float(end[0]), float(end[2])))
    if not pts:
        return []

    hull = MultiPoint(pts).convex_hull
    if clip_poly is not None and not clip_poly.is_empty:
        hull = hull.intersection(clip_poly)

    if isinstance(hull, Polygon) and not hull.is_empty:
        coords = list(hull.exterior.coords)
    elif isinstance(hull, MultiPolygon) and not hull.is_empty:
        largest = max(hull.geoms, key=lambda g: g.area)
        coords = list(largest.exterior.coords)
    elif isinstance(hull, LineString):
        coords = list(hull.coords)
    elif hasattr(hull, "x") and hasattr(hull, "y"):
        coords = [(hull.x, hull.y)]
    else:
        coords = []
    coords = _dedupe_and_despike([(float(x), float(z)) for x, z in coords])
    if len(coords) < 4:
        return []
    corners_xz = [(x, 0.0, z) for x, z in coords]
    return project_xz_onto_plane(corners_xz, plane)


def _cluster_centroid_xz(cl: dict) -> tuple[float, float]:
    xs = [float(e) for seg in cl["segs"] for e in (seg["a"][0], seg["b"][0])]
    zs = [float(e) for seg in cl["segs"] for e in (seg["a"][2], seg["b"][2])]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


def _room_centroid_xz(room: V3Room) -> tuple[float, float]:
    fp = room.floor_polygon
    if not fp:
        return (0.0, 0.0)
    return (sum(c[0] for c in fp) / len(fp), sum(c[2] for c in fp) / len(fp))


def _wall_xz_midpoint(wall: dict) -> tuple[float, float]:
    corners = wall.get("corners") or []
    if not corners:
        return (0.0, 0.0)
    return (
        sum(c[0] for c in corners) / len(corners),
        sum(c[2] for c in corners) / len(corners),
    )


def _wall_top_y(wall: dict) -> float | None:
    corners = wall.get("corners") or []
    if not corners:
        return None
    return float(max(c[1] for c in corners))


def _outward_direction(
    wall_xz: tuple[float, float], centroid: tuple[float, float]
) -> tuple[float, float]:
    dx, dz = wall_xz[0] - centroid[0], wall_xz[1] - centroid[1]
    n = sqrt(dx * dx + dz * dz)
    return (dx / n, dz / n) if n > 1e-6 else (1.0, 0.0)


def _azimuth_from_direction(direction_xz: tuple[float, float]) -> float:
    return (atan2(direction_xz[0], direction_xz[1]) * 180.0 / 3.141592653589793) % 360.0


def _wall_near_part_boundary(wall: dict, part_boundary: LineString) -> bool:
    mid = _wall_xz_midpoint(wall)
    return part_boundary.distance(Point(mid)) <= SCAN_NOISE_M


def _missing_exterior_edges(
    room: V3Room, room_poly: Polygon, part_boundary: LineString
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    missing: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if not isinstance(room_poly, Polygon) or room_poly.is_empty:
        return missing
    coords = list(room_poly.exterior.coords)
    wall_mids = [_wall_xz_midpoint(w) for w in room.walls]
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        edge = LineString([a, b])
        if edge.length < SCAN_NOISE_M:
            continue
        if (
            part_boundary.distance(edge.interpolate(0.5, normalized=True))
            > SCAN_NOISE_M
        ):
            continue
        covered = False
        for wm in wall_mids:
            if edge.distance(Point(wm)) <= SCAN_NOISE_M:
                covered = True
                break
        if not covered:
            missing.append((a, b))
    return missing


def cluster_oblique_to_roof_faces(
    building_uuid: str, rooms: list[V3Room], parts: list[V3Part]
) -> tuple[list[V3SlantedRoof], dict[str, SlopeHypothesis], list[V3UnresolvedRegion]]:
    all_segments: list[dict] = []
    for room in rooms:
        all_segments.extend(_collect_room_oblique_segments(room))

    clusters = cluster_oblique_segments(all_segments) if all_segments else []

    room_floor_polys: dict[str, Polygon] = {}
    for room in rooms:
        poly = floor_polygon_to_shapely(room.floor_polygon)
        if isinstance(poly, Polygon) and not poly.is_empty and poly.is_valid:
            room_floor_polys[room.identifier] = poly

    slanted_roofs: list[V3SlantedRoof] = []
    for idx, cl in enumerate(clusters):
        plane = _plane_tuple_from_cluster(cl)
        room_ids = sorted({s.get("room_id") for s in cl["segs"] if s.get("room_id")})
        ratifying_polys = [
            room_floor_polys[rid] for rid in room_ids if rid in room_floor_polys
        ]
        clip_poly = unary_union(ratifying_polys) if ratifying_polys else None
        corners = _cluster_corners(cl, plane, clip_poly)
        if len(corners) < 3:
            continue
        segment_ids = tuple(
            f"{s.get('room_id', '?')}::{s.get('wall_id', '?')}::{i}"
            for i, s in enumerate(cl["segs"])
        )
        slanted_roofs.append(
            V3SlantedRoof(
                id=make_element_id(building_uuid, "v3-slanted-roof", f"cluster-{idx}"),
                plane=plane,
                corners=corners,
                source_segment_ids=segment_ids,
                trace=HypothesisTrace(
                    stage="slanted_roofs",
                    rule="oblique-cluster",
                    inputs={
                        "cluster_index": idx,
                        "segment_count": len(cl["segs"]),
                        "avg_azimuth_deg": round(float(cl["avgAzimuth"]), 2),
                        "avg_incl_deg": round(float(cl["avgIncl"]), 2),
                        "room_ids": room_ids,
                    },
                    decision_reason=(
                        "≥2 oblique wall-top segments cluster within 30° azimuth / "
                        "15° inclination and a 0.5 m coplanarity tolerance — treat "
                        "as one roof face."
                    ),
                ),
            )
        )

    room_polys: dict[str, Polygon] = {}
    for room in rooms:
        poly = floor_polygon_to_shapely(room.floor_polygon)
        if isinstance(poly, Polygon) and not poly.is_empty:
            room_polys[room.identifier] = poly

    part_boundaries: list[LineString] = []
    for part in parts:
        if len(part.footprint_xz) >= 3:
            poly = Polygon([(c[0], c[2]) for c in part.footprint_xz])
            if poly.is_valid and poly.area > 0:
                part_boundaries.append(poly.boundary)

    hypotheses: dict[str, SlopeHypothesis] = {}
    unresolved: list[V3UnresolvedRegion] = []

    for room in rooms:
        sources: list[str] = []
        confidence = 0.0
        direction: tuple[float, float] | None = None
        cluster_index: int | None = None
        plane: tuple[float, float, float, float] | None = None

        matching_cluster = None
        matching_idx = -1
        for idx, cl in enumerate(clusters):
            if any(s.get("room_id") == room.identifier for s in cl["segs"]):
                matching_cluster = cl
                matching_idx = idx
                break
        if matching_cluster is not None:
            sources.append("oblique-cluster")
            az_rad = float(matching_cluster["avgAzimuth"]) * 3.141592653589793 / 180.0
            direction = (sin(az_rad), cos(az_rad))
            confidence = CONFIDENCE_CLUSTER
            cluster_index = matching_idx
            plane = _plane_tuple_from_cluster(matching_cluster)

        wall_tops = [(w, _wall_top_y(w)) for w in room.walls]
        wall_tops = [(w, y) for w, y in wall_tops if y is not None]
        centroid = _room_centroid_xz(room)
        nearest_part_boundary: LineString | None = None
        for b in part_boundaries:
            if b.distance(Point(centroid)) <= PART_BOUNDARY_SEARCH_M:
                nearest_part_boundary = b
                break

        if len(wall_tops) >= 2:
            wall_tops_sorted = sorted(wall_tops, key=lambda p: p[1])
            short_wall, short_y = wall_tops_sorted[0]
            tall_wall, tall_y = wall_tops_sorted[-1]
            if tall_y - short_y >= KNEE_DELTA_M and nearest_part_boundary is not None:
                if _wall_near_part_boundary(short_wall, nearest_part_boundary):
                    short_mid = _wall_xz_midpoint(short_wall)
                    tall_mid = _wall_xz_midpoint(tall_wall)
                    asym_dir = (short_mid[0] - tall_mid[0], short_mid[1] - tall_mid[1])
                    n = sqrt(asym_dir[0] ** 2 + asym_dir[1] ** 2)
                    if n > 1e-6:
                        asym_dir = (asym_dir[0] / n, asym_dir[1] / n)
                        sources.append("wall-height-asymmetry")
                        if direction is not None:
                            cluster_az = _azimuth_from_direction(direction)
                            if (
                                angle_diff(
                                    cluster_az, _azimuth_from_direction(asym_dir)
                                )
                                < CLUSTER_AZIMUTH_TOL_DEG
                            ):
                                confidence = min(
                                    1.0, confidence + CONFIDENCE_BOOST_ASYMMETRIC
                                )
                        else:
                            direction = asym_dir
                            confidence = max(confidence, CONFIDENCE_ASYMMETRIC_ONLY)

        if room.identifier in room_polys and nearest_part_boundary is not None:
            missing = _missing_exterior_edges(
                room, room_polys[room.identifier], nearest_part_boundary
            )
            if missing:
                edge = missing[0]
                mid = (
                    (edge[0][0] + edge[1][0]) / 2,
                    (edge[0][1] + edge[1][1]) / 2,
                )
                out_dir = _outward_direction(mid, centroid)
                sources.append("missing-exterior-wall")
                if direction is not None:
                    dir_az = _azimuth_from_direction(direction)
                    out_az = _azimuth_from_direction(out_dir)
                    if angle_diff(dir_az, out_az) < CLUSTER_AZIMUTH_TOL_DEG:
                        confidence = min(
                            1.0, confidence + CONFIDENCE_BOOST_MISSING_WALL
                        )
                else:
                    direction = out_dir
                    confidence = max(confidence, CONFIDENCE_MISSING_WALL_ONLY)

        if not sources:
            continue

        if confidence < SLOPE_HYPOTHESIS_MIN_CONFIDENCE:
            fp_xz = (
                [(float(x), 0.0, float(z)) for x, _, z in room.floor_polygon]
                if room.floor_polygon
                else []
            )
            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(
                        building_uuid, "v3-unresolved", f"slope-weak-{room.index}"
                    ),
                    room_id=room.identifier,
                    footprint_xz=fp_xz,
                    y_range=None,
                    reason="slope-evidence-weak",
                    context={
                        "sources": list(sources),
                        "confidence": round(confidence, 3),
                    },
                    trace=HypothesisTrace(
                        stage="slanted_roofs",
                        rule="slope-hypothesis-below-threshold",
                        inputs={
                            "sources": list(sources),
                            "confidence": round(confidence, 3),
                            "threshold": SLOPE_HYPOTHESIS_MIN_CONFIDENCE,
                        },
                        decision_reason=(
                            "Slope evidence accumulated but confidence stayed below "
                            "threshold — do NOT silently cap this room."
                        ),
                    ),
                )
            )
            continue

        assert direction is not None
        hypotheses[room.identifier] = SlopeHypothesis(
            room_id=room.identifier,
            direction_xz=direction,
            confidence=round(confidence, 3),
            sources=tuple(sources),
            plane=plane,
            cluster_index=cluster_index,
        )

    return slanted_roofs, hypotheses, unresolved
