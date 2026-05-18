"""Phase 2: derive ~300 features per labeled merged-roof segment.

Every feature comes from fields already embedded in a label record (see
``reports/slanted_roof_feature_catalogue.md`` Parts I + II). We never
re-open ``reconcile_v3_results.json`` here — that file is ~2GB and the
label snapshot is authoritative at label time.

Categories (match catalogue order):

    A. Plane descriptors
    B. Edge geometry (2D XZ and 3D)
    C. Vertex / corner geometry
    D. Face / polygon geometry (XZ area, compactness, hull, etc.)
    E. 3D position / building-relative
    F. Adjacency & neighbors (opposing planes, side pieces)
    G. Cluster-member aggregates (over ``cluster_members``)
    H. Cluster quality (spread over the cluster)
    I. Physics / drainage
    J. Record metadata

Outputs a flat ``dict[str, float|int|str|bool|None]`` per record; the caller
collects these into a pandas DataFrame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean, pstdev
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Small numeric helpers (kept local to avoid importing the whole pipeline).
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _safe(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    xs = [v for v in (_safe(x) for x in values) if v is not None]
    if not xs:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_range": None,
            f"{prefix}_median": None,
        }
    arr = np.asarray(xs, dtype=float)
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_range": float(arr.max() - arr.min()),
        f"{prefix}_median": float(np.median(arr)),
    }


def _azimuth_diff_deg(a: float, b: float) -> float:
    """Smallest angular distance in degrees (0..180)."""
    d = abs((a - b) % 360.0)
    return float(d if d <= 180.0 else 360.0 - d)


def _plane_to_azimuth_incl(plane: Sequence[float]) -> tuple[float, float]:
    """Return (azimuth_deg ∈ [0, 360), incl_deg ∈ [0, 90]) for a 4-plane.

    Plane is ``[a, b, c, d]`` where ``(a, b, c)`` is the normal and Y is up.
    Inclination is the tilt of the surface from horizontal
    (0 = flat roof, 90 = vertical wall). Azimuth is the compass direction
    of the horizontal projection of the normal: ``atan2(n.x, n.z)``
    wrapped to [0, 360).
    """
    if plane is None or len(plane) < 3:
        return (float("nan"), float("nan"))
    nx, ny, nz = float(plane[0]), float(plane[1]), float(plane[2])
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < _EPS:
        return (float("nan"), float("nan"))
    nx, ny, nz = nx / length, ny / length, nz / length
    incl = math.degrees(math.acos(max(-1.0, min(1.0, abs(ny)))))
    az = math.degrees(math.atan2(nx, nz))
    if az < 0:
        az += 360.0
    return (az, incl)


def _polygon_xz(corners_xyz: Sequence[Sequence[float]]) -> Polygon | None:
    if not corners_xyz or len(corners_xyz) < 3:
        return None
    xz = [(float(c[0]), float(c[2])) for c in corners_xyz]
    try:
        poly = Polygon(xz)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


def _polygon_2d(points: Sequence[Sequence[float]]) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    try:
        poly = Polygon([(float(p[0]), float(p[1])) for p in points])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


def _edges_3d(
    corners_xyz: Sequence[Sequence[float]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Closed ring of (a, b, length_m) edges."""
    if not corners_xyz or len(corners_xyz) < 2:
        return []
    pts = [np.asarray(c, dtype=float) for c in corners_xyz]
    edges = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        edges.append((a, b, float(np.linalg.norm(b - a))))
    return edges


def _plane_residual_rms(
    corners_xyz: Sequence[Sequence[float]], plane: Sequence[float]
) -> float | None:
    if not corners_xyz or plane is None or len(plane) < 4:
        return None
    n = np.asarray(plane[:3], dtype=float)
    norm = np.linalg.norm(n)
    if norm < _EPS:
        return None
    n = n / norm
    d = float(plane[3]) / norm
    pts = np.asarray(corners_xyz, dtype=float)
    dists = pts @ n + d
    return float(np.sqrt(np.mean(dists * dists)))


# ---------------------------------------------------------------------------
# Category A — plane descriptors.
# ---------------------------------------------------------------------------


def plane_features(plane: Sequence[float]) -> dict[str, Any]:
    if plane is None or len(plane) < 4:
        return {
            "plane_a": None,
            "plane_b": None,
            "plane_c": None,
            "plane_d": None,
            "plane_azimuth_deg": None,
            "plane_incl_deg": None,
            "plane_rise_over_run": None,
            "plane_pitch_is_shallow": None,
            "plane_pitch_is_architectural": None,
            "plane_pitch_is_steep": None,
            "plane_pitch_is_nearly_flat": None,
            "plane_pitch_is_nearly_vertical": None,
            "plane_water_flow_azimuth_deg": None,
        }
    a, b, c, d = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
    az, incl = _plane_to_azimuth_incl(plane)
    # Water-flow direction = horizontal projection of the downslope vector.
    # Downslope = negative of the in-plane gradient of y. For a plane with
    # normal n, the downslope vector in XZ is (-n.x, -n.z) — opposite of the
    # normal's horizontal projection. Flat planes return NaN.
    if math.isnan(incl) or incl < 1e-3:
        flow_az: float | None = None
    else:
        flow_az = math.degrees(math.atan2(-a, -c))
        if flow_az < 0:
            flow_az += 360.0
    tan_incl = math.tan(math.radians(incl)) if not math.isnan(incl) else None
    return {
        "plane_a": a,
        "plane_b": b,
        "plane_c": c,
        "plane_d": d,
        "plane_azimuth_deg": None if math.isnan(az) else float(az),
        "plane_incl_deg": None if math.isnan(incl) else float(incl),
        "plane_rise_over_run": None if tan_incl is None else float(tan_incl),
        "plane_pitch_is_nearly_flat": None if math.isnan(incl) else bool(incl < 5.0),
        "plane_pitch_is_shallow": None
        if math.isnan(incl)
        else bool(5.0 <= incl < 15.0),
        "plane_pitch_is_architectural": None
        if math.isnan(incl)
        else bool(15.0 <= incl < 60.0),
        "plane_pitch_is_steep": None if math.isnan(incl) else bool(incl >= 60.0),
        "plane_pitch_is_nearly_vertical": None
        if math.isnan(incl)
        else bool(incl >= 80.0),
        "plane_water_flow_azimuth_deg": flow_az,
    }


# ---------------------------------------------------------------------------
# Category B — edge geometry (3D + XZ).
# ---------------------------------------------------------------------------


def edge_features(corners_xyz: Sequence[Sequence[float]]) -> dict[str, Any]:
    edges = _edges_3d(corners_xyz)
    if not edges:
        return {
            "edge_count": 0,
            "edge_longest_m": None,
            "edge_shortest_m": None,
            "edge_mean_m": None,
            "edge_std_m": None,
            "edge_length_cv": None,
            "edge_total_m": None,
            "ridge_edge_length_m": None,
            "eave_edge_length_m": None,
            "edges_horizontal_fraction": None,
            "edges_vertical_fraction": None,
            "edge_longest_azimuth_deg": None,
        }
    lengths = [length for _, _, length in edges]
    if not lengths:
        return {"edge_count": 0}
    mean_len = mean(lengths)
    std_len = pstdev(lengths) if len(lengths) > 1 else 0.0
    cv = std_len / mean_len if mean_len > _EPS else None

    # Ridge = near-horizontal edge at or near top Y. Eave = near-horizontal
    # edge at or near bottom Y. Compute per corner-Y span.
    ys = [float(c[1]) for c in corners_xyz]
    y_min, y_max = min(ys), max(ys)
    y_span = max(y_max - y_min, _EPS)

    ridge_len = 0.0
    eave_len = 0.0
    horiz = 0
    vert = 0
    longest_az = None
    longest_len = -1.0
    for a, b, length in edges:
        dy = abs(float(b[1] - a[1]))
        dxz = math.hypot(float(b[0] - a[0]), float(b[2] - a[2]))
        if length <= _EPS:
            continue
        slope = dy / length  # sin(tilt)
        if slope < 0.1:  # near-horizontal
            horiz += 1
            mid_y = float((a[1] + b[1]) / 2.0)
            # top 20% of Y-span → ridge; bottom 20% → eave.
            if mid_y >= y_max - 0.2 * y_span:
                ridge_len += length
            if mid_y <= y_min + 0.2 * y_span:
                eave_len += length
        elif slope > 0.9:  # near-vertical
            vert += 1
        if length > longest_len:
            longest_len = length
            if dxz > _EPS:
                az = math.degrees(math.atan2(b[0] - a[0], b[2] - a[2]))
                if az < 0:
                    az += 360.0
                longest_az = float(az)
    return {
        "edge_count": len(edges),
        "edge_longest_m": max(lengths),
        "edge_shortest_m": min(lengths),
        "edge_mean_m": mean_len,
        "edge_std_m": std_len,
        "edge_length_cv": cv,
        "edge_total_m": sum(lengths),
        "ridge_edge_length_m": ridge_len,
        "eave_edge_length_m": eave_len,
        "edges_horizontal_fraction": horiz / len(edges),
        "edges_vertical_fraction": vert / len(edges),
        "edge_longest_azimuth_deg": longest_az,
    }


# ---------------------------------------------------------------------------
# Category C — vertex / corner geometry.
# ---------------------------------------------------------------------------


def vertex_features(
    corners_xyz: Sequence[Sequence[float]], plane: Sequence[float] | None
) -> dict[str, Any]:
    if not corners_xyz:
        return {
            "vertex_count": 0,
            "y_min_m": None,
            "y_max_m": None,
            "y_range_m": None,
            "y_mean_m": None,
            "y_std_m": None,
            "x_range_m": None,
            "z_range_m": None,
            "slant_residual_rms_m": None,
        }
    pts = np.asarray(corners_xyz, dtype=float)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    return {
        "vertex_count": len(corners_xyz),
        "y_min_m": float(ys.min()),
        "y_max_m": float(ys.max()),
        "y_range_m": float(ys.max() - ys.min()),
        "y_mean_m": float(ys.mean()),
        "y_std_m": float(ys.std(ddof=0)),
        "x_range_m": float(xs.max() - xs.min()),
        "z_range_m": float(zs.max() - zs.min()),
        "slant_residual_rms_m": _plane_residual_rms(corners_xyz, plane)
        if plane is not None
        else None,
    }


# ---------------------------------------------------------------------------
# Category D — face / polygon geometry (XZ projection).
# ---------------------------------------------------------------------------


def polygon_features(
    corners_xyz: Sequence[Sequence[float]],
) -> dict[str, Any]:
    poly = _polygon_xz(corners_xyz)
    if poly is None:
        return {
            "poly_area_xz_m2": None,
            "poly_perimeter_xz_m": None,
            "poly_compactness": None,
            "poly_convex_hull_area_m2": None,
            "poly_convex_hull_ratio": None,
            "poly_bbox_aspect": None,
            "poly_bbox_area_m2": None,
            "poly_min_rect_major_m": None,
            "poly_min_rect_minor_m": None,
            "poly_min_rect_aspect": None,
            "poly_min_rect_azimuth_deg": None,
            "poly_min_rect_fill_ratio": None,
            "poly_min_width_m": None,
            "poly_simplify_vertex_ratio": None,
            "poly_is_convex": None,
            "poly_centroid_x": None,
            "poly_centroid_z": None,
        }
    area = float(poly.area)
    perimeter = float(poly.length)
    compactness = (
        (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > _EPS else None
    )
    hull = poly.convex_hull
    hull_area = float(hull.area) if hull.area > 0 else None
    hull_ratio = area / hull_area if hull_area else None
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    bbox_area = w * h
    bbox_aspect = max(w, h) / min(w, h) if min(w, h) > _EPS else None
    # Minimum rotated rectangle. Shapely warns (and returns NaN vertices) on
    # degenerate polygons; silence the warning and treat it as unavailable.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            mrr = poly.minimum_rotated_rectangle
            mrr_coords = (
                list(mrr.exterior.coords)[:-1] if hasattr(mrr, "exterior") else []
            )
        except Exception:
            mrr_coords = []
    major = minor = mrr_az = None
    mrr_area = None
    if len(mrr_coords) >= 4:
        p0, p1, p2 = (np.asarray(mrr_coords[i], dtype=float) for i in range(3))
        e1 = float(np.linalg.norm(p1 - p0))
        e2 = float(np.linalg.norm(p2 - p1))
        major = max(e1, e2)
        minor = min(e1, e2)
        mrr_area = major * minor
        long_edge = (p1 - p0) if e1 >= e2 else (p2 - p1)
        mrr_az = math.degrees(math.atan2(long_edge[0], long_edge[1]))
        if mrr_az < 0:
            mrr_az += 360.0
    mrr_aspect = major / minor if (major and minor and minor > _EPS) else None
    mrr_fill = area / mrr_area if (mrr_area and mrr_area > _EPS) else None
    min_width = (2.0 * area / perimeter) if perimeter > _EPS else None
    try:
        simplified = poly.simplify(0.05, preserve_topology=True)
        n_orig = len(poly.exterior.coords) - 1
        n_simp = (
            len(simplified.exterior.coords) - 1
            if hasattr(simplified, "exterior")
            else n_orig
        )
        simplify_ratio = n_simp / n_orig if n_orig > 0 else None
    except Exception:
        simplify_ratio = None
    try:
        is_convex = bool(abs(hull.area - area) < 1e-3)
    except Exception:
        is_convex = None
    cx, cz = poly.centroid.x, poly.centroid.y
    return {
        "poly_area_xz_m2": area,
        "poly_perimeter_xz_m": perimeter,
        "poly_compactness": compactness,
        "poly_convex_hull_area_m2": hull_area,
        "poly_convex_hull_ratio": hull_ratio,
        "poly_bbox_aspect": bbox_aspect,
        "poly_bbox_area_m2": bbox_area,
        "poly_min_rect_major_m": major,
        "poly_min_rect_minor_m": minor,
        "poly_min_rect_aspect": mrr_aspect,
        "poly_min_rect_azimuth_deg": mrr_az,
        "poly_min_rect_fill_ratio": mrr_fill,
        "poly_min_width_m": min_width,
        "poly_simplify_vertex_ratio": simplify_ratio,
        "poly_is_convex": is_convex,
        "poly_centroid_x": float(cx),
        "poly_centroid_z": float(cz),
    }


# ---------------------------------------------------------------------------
# Category E — 3D position / building-relative.
# ---------------------------------------------------------------------------


def position_features(
    corners_xyz: Sequence[Sequence[float]],
    building_boundary_xz: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    poly = _polygon_xz(corners_xyz)
    boundary = _polygon_2d(building_boundary_xz or [])
    if poly is None or boundary is None:
        return {
            "inside_building_footprint": None,
            "distance_to_footprint_edge_m": None,
            "fraction_inside_footprint": None,
            "touches_footprint_boundary_length_m": None,
            "centroid_distance_to_boundary_m": None,
            "overshoot_area_m2": None,
        }
    centroid = poly.centroid
    try:
        inter = poly.intersection(boundary)
        inter_area = float(inter.area) if not inter.is_empty else 0.0
    except Exception:
        inter_area = 0.0
    poly_area = float(poly.area)
    frac_inside = inter_area / poly_area if poly_area > _EPS else None
    overshoot = max(poly_area - inter_area, 0.0)
    try:
        boundary_line = boundary.boundary
        touches_len = float(poly.boundary.intersection(boundary_line).length)
        dist_boundary = float(centroid.distance(boundary_line))
    except Exception:
        touches_len = None
        dist_boundary = None
    try:
        contains = bool(boundary.contains(centroid))
    except Exception:
        contains = None
    # Centroid→nearest footprint-edge distance (outside = negative convention
    # isn't meaningful here because the boundary is itself the edge; we
    # report signed distance as positive inside, 0 on boundary, positive
    # outside too — keep it simple with distance-to-boundary as-is).
    return {
        "inside_building_footprint": contains,
        "distance_to_footprint_edge_m": dist_boundary,
        "fraction_inside_footprint": frac_inside,
        "touches_footprint_boundary_length_m": touches_len,
        "centroid_distance_to_boundary_m": dist_boundary,
        "overshoot_area_m2": overshoot,
    }


# ---------------------------------------------------------------------------
# Category F — adjacency & neighbors.
# ---------------------------------------------------------------------------


def neighbor_features(
    plane: Sequence[float] | None,
    opposing_planes: Sequence[Sequence[float]] | None,
    opposing_cluster_canonicals: Sequence[str] | None,
    side_pieces: Sequence[dict] | None,
    corners_xyz: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "opposing_count": 0,
        "opposing_azimuth_mean_deg": None,
        "opposing_azimuth_std_deg": None,
        "opposing_incl_mean_deg": None,
        "opposing_incl_std_deg": None,
        "opposing_azimuth_diff_max_deg": None,
        "opposing_azimuth_diff_min_deg": None,
        "opposing_azimuth_diff_mean_deg": None,
        "opposing_incl_diff_max_deg": None,
        "opposing_cos_min": None,
        "side_piece_count": 0,
        "side_piece_area_sum_m2": None,
        "side_piece_has_parent": None,
    }
    if opposing_planes:
        az_mine, incl_mine = (
            _plane_to_azimuth_incl(plane) if plane else (float("nan"), float("nan"))
        )
        azs, incls = [], []
        az_diffs, incl_diffs = [], []
        cos_mins = []
        nm = None
        if plane is not None and len(plane) >= 3:
            nm = np.asarray(plane[:3], dtype=float)
            nm_norm = np.linalg.norm(nm)
            nm = nm / nm_norm if nm_norm > _EPS else None
        for op in opposing_planes:
            if op is None or len(op) < 3:
                continue
            az, incl = _plane_to_azimuth_incl(op)
            if not math.isnan(az):
                azs.append(az)
                if not math.isnan(az_mine):
                    az_diffs.append(_azimuth_diff_deg(az, az_mine))
            if not math.isnan(incl):
                incls.append(incl)
                if not math.isnan(incl_mine):
                    incl_diffs.append(abs(incl - incl_mine))
            if nm is not None:
                op_n = np.asarray(op[:3], dtype=float)
                on = np.linalg.norm(op_n)
                if on > _EPS:
                    cos_mins.append(float(abs(np.dot(nm, op_n / on))))
        out["opposing_count"] = len(opposing_planes)
        if azs:
            out["opposing_azimuth_mean_deg"] = float(np.mean(azs))
            out["opposing_azimuth_std_deg"] = float(np.std(azs))
        if incls:
            out["opposing_incl_mean_deg"] = float(np.mean(incls))
            out["opposing_incl_std_deg"] = float(np.std(incls))
        if az_diffs:
            out["opposing_azimuth_diff_max_deg"] = float(max(az_diffs))
            out["opposing_azimuth_diff_min_deg"] = float(min(az_diffs))
            out["opposing_azimuth_diff_mean_deg"] = float(np.mean(az_diffs))
        if incl_diffs:
            out["opposing_incl_diff_max_deg"] = float(max(incl_diffs))
        if cos_mins:
            out["opposing_cos_min"] = float(min(cos_mins))

    if side_pieces:
        areas = []
        any_parent = False
        for sp in side_pieces:
            poly = _polygon_xz(sp.get("piece_corners_xyz") or [])
            if poly is not None:
                areas.append(float(poly.area))
            if sp.get("parent_proposal_id"):
                any_parent = True
        out["side_piece_count"] = len(side_pieces)
        out["side_piece_area_sum_m2"] = float(sum(areas)) if areas else 0.0
        out["side_piece_has_parent"] = any_parent

    # opposing cluster canonical count (may differ from opposing_planes count
    # when planes come from the same cluster).
    if opposing_cluster_canonicals is not None:
        out["opposing_cluster_count"] = len(opposing_cluster_canonicals)
    return out


# ---------------------------------------------------------------------------
# Category G — cluster-member aggregates.
# ---------------------------------------------------------------------------


_MEMBER_NUMERIC_FEATURES = [
    "segment_azimuth_deg",
    "segment_incl_deg",
    "segment_length_m",
    "segment_mid_y_m",
    "plane_height_above_slab_m",
    "plane_y_at_piece_centroid_m",
    "piece_area_m2",
    "piece_perimeter_m",
    "piece_compactness",
    "piece_bbox_aspect",
    "piece_min_width_m",
    "piece_vertex_count",
    "rain_exposure_ratio",
    "slant_delta_over_piece_m",
    "seg_mid_to_piece_centroid_xz_m",
    "slab_area_m2",
    "slab_floor_y_m",
    "slab_vertex_count",
    "story_delta",
]

_MEMBER_BOOL_FEATURES = [
    "is_same_room",
    "is_top_story_slab",
]


def cluster_member_features(cluster_members: Sequence[dict] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"cluster_member_count": 0}
    if not cluster_members:
        for f in _MEMBER_NUMERIC_FEATURES:
            out.update(_stats([], f"member_{f}"))
        for f in _MEMBER_BOOL_FEATURES:
            out[f"member_{f}_fraction"] = None
        out.update(
            {
                "member_heuristic_accepted_fraction": None,
                "member_unique_source_rooms": 0,
                "member_unique_source_walls": 0,
                "member_unique_slab_rooms": 0,
                "member_unique_stories": 0,
                "member_plane_d_spread_m": None,
                "member_plane_azimuth_spread_deg": None,
                "member_plane_incl_spread_deg": None,
                "member_plane_dot_min": None,
                "member_piece_kind_room_fraction": None,
                "member_piece_kind_gap_fraction": None,
            }
        )
        return out

    out["cluster_member_count"] = len(cluster_members)
    feats = [m.get("features") or {} for m in cluster_members]
    for f in _MEMBER_NUMERIC_FEATURES:
        out.update(_stats([fs.get(f) for fs in feats], f"member_{f}"))
    for f in _MEMBER_BOOL_FEATURES:
        vals = [bool(fs.get(f)) for fs in feats if f in fs]
        out[f"member_{f}_fraction"] = float(sum(vals) / len(vals)) if vals else None
    heur = [m.get("heuristic_label") for m in cluster_members]
    out["member_heuristic_accepted_fraction"] = (
        float(sum(1 for x in heur if x == "accepted") / len(heur)) if heur else None
    )
    out["member_unique_source_rooms"] = len(
        {m.get("source_room_id") for m in cluster_members if m.get("source_room_id")}
    )
    out["member_unique_source_walls"] = len(
        {m.get("source_wall_id") for m in cluster_members if m.get("source_wall_id")}
    )
    out["member_unique_slab_rooms"] = len(
        {m.get("slab_room_id") for m in cluster_members if m.get("slab_room_id")}
    )
    stories = set()
    for m in cluster_members:
        fs = m.get("features") or {}
        if "segment_story" in fs and fs["segment_story"] is not None:
            stories.add(int(fs["segment_story"]))
    out["member_unique_stories"] = len(stories)

    # Cluster spread on plane coefficients.
    planes = [m.get("plane") for m in cluster_members if m.get("plane")]
    if planes:
        azs = []
        incls = []
        ds = []
        ns = []
        for p in planes:
            az, incl = _plane_to_azimuth_incl(p)
            if not math.isnan(az):
                azs.append(az)
            if not math.isnan(incl):
                incls.append(incl)
            if len(p) >= 4:
                ds.append(float(p[3]))
            if len(p) >= 3:
                nv = np.asarray(p[:3], dtype=float)
                nn = np.linalg.norm(nv)
                if nn > _EPS:
                    ns.append(nv / nn)
        if azs:
            # circular std via cos/sin
            rad = np.radians(azs)
            c, s = np.mean(np.cos(rad)), np.mean(np.sin(rad))
            r = math.sqrt(c * c + s * s)
            out["member_plane_azimuth_spread_deg"] = float(
                math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(r, _EPS)))))
            )
        if incls:
            out["member_plane_incl_spread_deg"] = float(max(incls) - min(incls))
        if ds:
            out["member_plane_d_spread_m"] = float(max(ds) - min(ds))
        if len(ns) >= 2:
            stack = np.stack(ns)
            dots = stack @ stack.T
            # min off-diagonal abs
            off = dots[~np.eye(len(ns), dtype=bool)]
            out["member_plane_dot_min"] = float(np.min(np.abs(off)))

    piece_kinds = [fs.get("piece_kind") for fs in feats if "piece_kind" in fs]
    # NOTE: merged-level piece_kind lives on the record itself; cluster-member
    # features don't include it — fall back to None if missing.
    if piece_kinds:
        n = len(piece_kinds)
        out["member_piece_kind_room_fraction"] = piece_kinds.count("room") / n
        out["member_piece_kind_gap_fraction"] = piece_kinds.count("gap") / n

    return out


# ---------------------------------------------------------------------------
# Category H — cluster quality (over the merged XZ geometry).
# ---------------------------------------------------------------------------


def cluster_quality_features(
    cluster_members: Sequence[dict] | None,
    merged_corners_xyz: Sequence[Sequence[float]] | None,
    cluster_params: dict | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cluster_pre_clip_area_sum_m2": None,
        "cluster_post_clip_area_m2": None,
        "cluster_clip_ratio": None,
        "cluster_param_d_abs_max": None,
        "cluster_param_normal_dot_min": None,
        "cluster_param_opposing_cos_az_max": None,
    }
    if cluster_params:
        out["cluster_param_d_abs_max"] = _safe(cluster_params.get("d_abs_max"))
        out["cluster_param_normal_dot_min"] = _safe(
            cluster_params.get("normal_dot_min")
        )
        out["cluster_param_opposing_cos_az_max"] = _safe(
            cluster_params.get("opposing_cos_azimuth_max")
        )
    if cluster_members:
        pre_polys = []
        for m in cluster_members:
            p = _polygon_xz(m.get("corners") or [])
            if p is not None:
                pre_polys.append(p)
        if pre_polys:
            try:
                union = unary_union(pre_polys)
                out["cluster_pre_clip_area_sum_m2"] = float(union.area)
            except Exception:
                out["cluster_pre_clip_area_sum_m2"] = float(
                    sum(p.area for p in pre_polys)
                )
    merged_poly = _polygon_xz(merged_corners_xyz or [])
    if merged_poly is not None:
        out["cluster_post_clip_area_m2"] = float(merged_poly.area)
        if (
            out["cluster_pre_clip_area_sum_m2"]
            and out["cluster_pre_clip_area_sum_m2"] > _EPS
        ):
            out["cluster_clip_ratio"] = float(
                out["cluster_post_clip_area_m2"] / out["cluster_pre_clip_area_sum_m2"]
            )
    return out


# ---------------------------------------------------------------------------
# Category I — physics / drainage.
# ---------------------------------------------------------------------------


def physics_features(
    plane: Sequence[float] | None,
    corners_xyz: Sequence[Sequence[float]] | None,
    building_boundary_xz: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    poly = _polygon_xz(corners_xyz or [])
    boundary = _polygon_2d(building_boundary_xz or [])
    out: dict[str, Any] = {
        "drainage_flow_azimuth_deg": None,
        "drainage_to_building_center_cos": None,
        "drainage_sheds_away_from_center": None,
        "eave_y_m": None,
        "ridge_y_m": None,
    }
    if corners_xyz:
        ys = [float(c[1]) for c in corners_xyz]
        out["eave_y_m"] = float(min(ys))
        out["ridge_y_m"] = float(max(ys))
    if plane is None or len(plane) < 3:
        return out
    a, b, c, _ = plane[0], plane[1], plane[2], plane[3] if len(plane) >= 4 else 0.0
    if abs(float(b)) < 1e-3:  # near-vertical normal → no drainage
        return out
    flow_vec = np.array([-float(a), -float(c)])
    flow_norm = np.linalg.norm(flow_vec)
    if flow_norm < _EPS:
        return out
    flow_vec = flow_vec / flow_norm
    az = math.degrees(math.atan2(flow_vec[0], flow_vec[1]))
    if az < 0:
        az += 360.0
    out["drainage_flow_azimuth_deg"] = float(az)
    if poly is not None and boundary is not None:
        pc = poly.centroid
        bc = boundary.centroid
        to_out = np.array([pc.x - bc.x, pc.y - bc.y])
        to_out_norm = np.linalg.norm(to_out)
        if to_out_norm > _EPS:
            to_out = to_out / to_out_norm
            dot = float(np.dot(flow_vec, to_out))
            out["drainage_to_building_center_cos"] = dot
            out["drainage_sheds_away_from_center"] = bool(dot > 0)
    return out


# ---------------------------------------------------------------------------
# Category J — record metadata.
# ---------------------------------------------------------------------------


def metadata_features(record: dict) -> dict[str, Any]:
    fs = record.get("features_snapshot") or {}
    piece_kind = fs.get("piece_kind")
    return {
        "building_uuid": record.get("building_uuid"),
        "proposal_id": record.get("proposal_id"),
        "cluster_canonical_id": record.get("cluster_canonical_id"),
        "label": record.get("label"),
        "heuristic_label": record.get("heuristic_label"),
        "labeler": record.get("labeler"),
        "ts": record.get("ts"),
        "merge_mode": record.get("merge_mode"),
        "part_index": record.get("part_index"),
        "part_count": record.get("part_count"),
        "is_split_child": record.get("is_split_child"),
        "parent_proposal_id": record.get("parent_proposal_id"),
        "piece_kind": piece_kind,
        "piece_kind_is_room": (piece_kind == "room") if piece_kind else None,
        "piece_kind_is_gap": (piece_kind == "gap") if piece_kind else None,
        "rain_hitting_side_count": fs.get("rain_hitting_side_count"),
        "covered_side_count": fs.get("covered_side_count"),
        "clipped_by_building_boundary": fs.get("clipped_by_building_boundary"),
        "snapshot_member_count": fs.get("member_count"),
        "snapshot_opposing_cluster_count": fs.get("opposing_cluster_count"),
        "snapshot_area_m2": fs.get("area_m2"),
        "snapshot_perimeter_m": fs.get("perimeter_m"),
        "n_room_boundary_refs": len(record.get("room_boundary_refs") or []),
        "heuristic_disagrees_with_user": (
            record.get("heuristic_label") != record.get("label")
            if record.get("heuristic_label") and record.get("label")
            else None
        ),
        "label_is_accept": (
            record.get("label") == "accepted" if record.get("label") else None
        ),
    }


# ---------------------------------------------------------------------------
# Top-level assembly.
# ---------------------------------------------------------------------------


def expand(record: dict) -> dict[str, Any]:
    """Return a flat feature dict for one labeled record.

    The record must have the embedded fields documented in
    ``reports/slanted_roof_feature_catalogue.md`` Part I (merged plane,
    segment corners, cluster members, side pieces, building boundary,
    opposing planes, features snapshot).
    """
    corners = record.get("segment_corners_xyz") or []
    plane = record.get("merged_plane")
    boundary = record.get("building_boundary_xz") or []
    out: dict[str, Any] = {}
    out.update(metadata_features(record))
    out.update(plane_features(plane))
    out.update(edge_features(corners))
    out.update(vertex_features(corners, plane))
    out.update(polygon_features(corners))
    out.update(position_features(corners, boundary))
    out.update(
        neighbor_features(
            plane,
            record.get("opposing_planes"),
            record.get("opposing_cluster_canonicals"),
            record.get("side_pieces"),
            corners,
        )
    )
    out.update(cluster_member_features(record.get("cluster_members")))
    out.update(
        cluster_quality_features(
            record.get("cluster_members"),
            corners,
            record.get("cluster_params"),
        )
    )
    out.update(physics_features(plane, corners, boundary))
    return out


def expand_all(records: Sequence[dict]) -> list[dict[str, Any]]:
    return [expand(r) for r in records]
