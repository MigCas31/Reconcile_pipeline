from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from .graph_support import (
    classify_graph_roof_room,
    graph_allows_roof_candidate,
    graph_requires_geometry_fallback,
)
from .math_utils import angle_diff, plane_normal, seg_midpoint

MIN_RAW_RECTANGLE_AREA_M2 = 5.0
MIN_RAW_RECTANGLE_INCL_DEG = 10.0
MAX_RAW_RECTANGLE_INCL_DEG = 75.0
MAX_PLANE_RESIDUAL_M = 0.08
MIN_RIDGE_EDGE_LEN_M = 2.0
MAX_RIDGE_EDGE_ANGLE_DELTA_DEG = 15.0
MAX_DUPLICATE_PLANE_DISTANCE_M = 0.5


def _fit_plane(corners: list) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(corners) != 4:
        return None
    pts = np.array(corners, dtype=float)
    centroid = pts.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(pts - centroid)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal = normal / norm
    if normal[1] < 0:
        normal = -normal
    residuals = np.abs((pts - centroid) @ normal)
    return centroid, normal, float(residuals.max())


def _polygon_xz(corners: list) -> Polygon | None:
    points = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
    if len(points) != 4 or len(set(points)) != 4:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        try:
            repaired = make_valid(poly)
            if isinstance(repaired, Polygon):
                poly = repaired
            else:
                return None
        except Exception:
            return None
    if poly.is_empty or poly.area < MIN_RAW_RECTANGLE_AREA_M2:
        return None
    return poly


def _inclination_and_downslope(normal: np.ndarray) -> tuple[float, float] | None:
    if abs(float(normal[1])) <= 1e-9:
        return None
    grad_x = -float(normal[0]) / float(normal[1])
    grad_z = -float(normal[2]) / float(normal[1])
    incl = degrees(atan2(sqrt(grad_x * grad_x + grad_z * grad_z), 1.0))
    # Segment azimuths in the wall-derived pipeline point downhill.
    azimuth = degrees(atan2(-grad_x, -grad_z)) % 360.0
    return incl, azimuth


def _ridge_like_edges(
    corners: list,
    *,
    ridge_azimuth: float,
) -> list[tuple[list[float], list[float], float]]:
    edges: list[tuple[list[float], list[float], float]] = []
    for idx, start in enumerate(corners):
        end = corners[(idx + 1) % len(corners)]
        dx = float(end[0]) - float(start[0])
        dz = float(end[2]) - float(start[2])
        length_xz = sqrt(dx * dx + dz * dz)
        if length_xz < MIN_RIDGE_EDGE_LEN_M:
            continue
        edge_az = degrees(atan2(dx, dz)) % 180.0
        delta = min(
            angle_diff(edge_az, ridge_azimuth % 180.0),
            angle_diff((edge_az + 180.0) % 360.0, ridge_azimuth),
        )
        if delta > MAX_RIDGE_EDGE_ANGLE_DELTA_DEG:
            continue
        edges.append((start, end, length_xz))
    return edges


def _is_duplicate_of_wall_cluster(
    raw_cluster: dict[str, Any], clusters: list[dict]
) -> bool:
    raw_mid = seg_midpoint(raw_cluster["segs"][0])
    for cluster in clusters:
        if (
            angle_diff(float(raw_cluster["avgAzimuth"]), float(cluster["avgAzimuth"]))
            > 10.0
        ):
            continue
        if abs(float(raw_cluster["avgIncl"]) - float(cluster["avgIncl"])) > 10.0:
            continue
        n = plane_normal(float(cluster["avgAzimuth"]), float(cluster["avgIncl"]))
        n_len = sqrt(n["x"] ** 2 + n["y"] ** 2 + n["z"] ** 2)
        if n_len <= 1e-9:
            continue
        ref = cluster.get("refPt") or seg_midpoint(cluster["segs"][0])
        dist = (
            abs(
                (raw_mid["x"] - ref["x"]) * n["x"]
                + (raw_mid["y"] - ref["y"]) * n["y"]
                + (raw_mid["z"] - ref["z"]) * n["z"]
            )
            / n_len
        )
        if dist <= MAX_DUPLICATE_PLANE_DISTANCE_M:
            return True
    return False


def collect_raw_oblique_rectangle_clusters(
    bldg: dict,
    existing_clusters: list[dict],
    has_floor_above,
    exclude_room_indices: set[int] | None = None,
    graph=None,
) -> list[dict]:
    clusters: list[dict] = []

    for room_idx, room in enumerate(bldg.get("rooms", []) or []):
        if exclude_room_indices and room_idx in exclude_room_indices:
            continue
        fp = room.get("floor_polygon") or []
        if len(fp) < 3:
            continue
        story = int(room.get("story", 0))
        graph_room, roof_decision = classify_graph_roof_room(graph, story, fp)
        if roof_decision is not None and not graph_allows_roof_candidate(roof_decision):
            continue
        fcx = sum(float(p[0]) for p in fp) / len(fp)
        fcz = sum(float(p[2]) for p in fp) / len(fp)
        if graph_requires_geometry_fallback(roof_decision) and has_floor_above(
            fcx, fcz, story
        ):
            continue

        for plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) != 4:
                continue
            poly = _polygon_xz(corners)
            if poly is None:
                continue
            fitted = _fit_plane(corners)
            if fitted is None:
                continue
            centroid, normal, max_residual = fitted
            if max_residual > MAX_PLANE_RESIDUAL_M:
                continue
            orientation = _inclination_and_downslope(normal)
            if orientation is None:
                continue
            incl, azimuth = orientation
            if incl < MIN_RAW_RECTANGLE_INCL_DEG or incl > MAX_RAW_RECTANGLE_INCL_DEG:
                continue

            ridge_edges = _ridge_like_edges(
                corners, ridge_azimuth=(azimuth + 90.0) % 360.0
            )
            if len(ridge_edges) < 2:
                continue
            ridge_edges = sorted(ridge_edges, key=lambda edge: edge[2], reverse=True)[
                :2
            ]
            raw_plane_id = (
                f"{bldg.get('uuid', '')}::ceiling-raw::{story}:{room_idx}:{plane_index}"
            )
            segs = [
                {
                    "a": list(start),
                    "b": list(end),
                    "incl": incl,
                    "azimuth": azimuth,
                    "len": length,
                    "story": story,
                    "room_idx": room_idx,
                    "graph_room_id": graph_room.id if graph_room is not None else None,
                    "source": "raw_ceiling_rectangle",
                    "raw_plane_id": raw_plane_id,
                }
                for start, end, length in ridge_edges
            ]
            cluster = {
                "segs": segs,
                "avgIncl": incl,
                "avgAzimuth": azimuth,
                "refPt": {
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]),
                },
                "room_indices": [room_idx],
                "source": "raw_ceiling_rectangle",
                "raw_plane_ids": [raw_plane_id],
                "raw_area_xz_m2": round(float(poly.area), 6),
                "raw_plane_max_residual_m": round(max_residual, 6),
            }
            if _is_duplicate_of_wall_cluster(cluster, existing_clusters + clusters):
                continue
            clusters.append(cluster)

    return clusters
