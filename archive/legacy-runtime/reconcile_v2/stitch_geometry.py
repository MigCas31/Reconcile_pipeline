"""Planar stitching for V2 surfaces.

Builds a stitched geometry layer with:
- coplanar clustering (by story + kind + plane similarity)
- 2D union in plane space
- shared vertex index across stitched faces
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.ops import polygonize, unary_union


@dataclass
class _SurfaceWorld:
    surface_id: str
    story: int | None
    kind: str
    points: np.ndarray  # (N, 3)


@dataclass
class _PlaneCluster:
    story: int | None
    kind: str
    normal: np.ndarray  # (3,)
    offset: float
    members: list[_SurfaceWorld]


def _transform_point(flat: list[float], p: list[float]) -> np.ndarray:
    x, y, z = p
    tx = flat[0] * x + flat[4] * y + flat[8] * z + flat[12]
    ty = flat[1] * x + flat[5] * y + flat[9] * z + flat[13]
    tz = flat[2] * x + flat[6] * y + flat[10] * z + flat[14]
    tw = flat[3] * x + flat[7] * y + flat[11] * z + flat[15]
    if abs(tw) > 1e-8 and abs(tw - 1.0) > 1e-8:
        return np.array([tx / tw, ty / tw, tz / tw], dtype=float)
    return np.array([tx, ty, tz], dtype=float)


def _stable_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    p0 = points[0]
    for i in range(1, points.shape[0] - 1):
        v1 = points[i] - p0
        v2 = points[i + 1] - p0
        n = np.cross(v1, v2)
        ln = np.linalg.norm(n)
        if ln > 1e-8:
            n = n / ln
            # Deterministic sign to improve clustering stability.
            idx = int(np.argmax(np.abs(n)))
            if n[idx] < 0:
                n = -n
            return n
    return None


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(normal, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(up, normal)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(normal, u)
    v = v / max(np.linalg.norm(v), 1e-12)
    return u, v


def _project(
    points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    d = points - origin
    return np.column_stack((d @ u, d @ v))


def _unproject(
    pts2: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    return origin + np.outer(pts2[:, 0], u) + np.outer(pts2[:, 1], v)


def _to_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _arrangement_faces(
    polys2: list[Polygon],
    *,
    kind: str,
    merge_gap_bridge_m: float,
) -> tuple[list[Polygon], Polygon | MultiPolygon | None]:
    """Build planar faces from noded segment arrangement.

    1) Compute target union area (with optional conservative bridge)
    2) Node all source polygon boundaries
    3) Polygonize into candidate faces
    4) Keep only faces whose representative point is inside union area
    """
    if not polys2:
        return [], None

    area_union = unary_union(polys2)
    if kind in ("wall", "floor", "opening") and merge_gap_bridge_m > 0:
        area_union = area_union.buffer(merge_gap_bridge_m).buffer(-merge_gap_bridge_m)
    area_union = area_union.buffer(0)
    if area_union.is_empty:
        return [], None

    lines = []
    for p in polys2:
        if p.is_empty:
            continue
        try:
            lines.append(LineString(list(p.exterior.coords)))
            for h in p.interiors:
                lines.append(LineString(list(h.coords)))
        except Exception:
            continue
    if not lines:
        return _to_polygons(area_union), area_union

    noded = unary_union(lines)
    candidates = list(polygonize(noded))
    if not candidates:
        return _to_polygons(area_union), area_union

    keep: list[Polygon] = []
    for face in candidates:
        if face.is_empty or face.area < 1e-8:
            continue
        rp = face.representative_point()
        try:
            inside = area_union.covers(rp)
        except Exception:
            inside = False
        if inside:
            keep.append(face)

    if not keep:
        return _to_polygons(area_union), area_union
    return keep, area_union


def _dissolve_faces_by_source_sets(
    face_records: list[tuple[Polygon, list[str]]],
) -> list[tuple[Polygon, list[str]]]:
    """Merge adjacent arrangement faces that share the same source-coverage set."""
    grouped: dict[tuple[str, ...], list[Polygon]] = {}
    for poly, src_ids in face_records:
        if poly.is_empty or poly.area < 1e-8:
            continue
        key = tuple(sorted(set(src_ids)))
        grouped.setdefault(key, []).append(poly)

    out: list[tuple[Polygon, list[str]]] = []
    for key, polys in grouped.items():
        geom = unary_union(polys)
        for p in _to_polygons(geom):
            if p.is_empty or p.area < 1e-8:
                continue
            out.append((p, list(key)))
    return out


def _arrangement_graph(polys2: list[Polygon], *, snap_2d_m: float) -> dict[str, Any]:
    """Build a planar graph (vertices/edges) from noded polygon boundaries."""
    lines = []
    for p in polys2:
        if p.is_empty:
            continue
        try:
            lines.append(LineString(list(p.exterior.coords)))
            for h in p.interiors:
                lines.append(LineString(list(h.coords)))
        except Exception:
            continue

    if not lines:
        return {"vertices_2d": {}, "edges_2d": [], "stats": {"vertices": 0, "edges": 0}}

    noded = unary_union(lines)
    segments: list[LineString] = []

    def collect_segments(g) -> None:
        if g is None or g.is_empty:
            return
        if isinstance(g, LineString):
            segments.append(g)
            return
        if isinstance(g, MultiLineString):
            for lg in g.geoms:
                segments.append(lg)
            return
        if isinstance(g, GeometryCollection):
            for gg in g.geoms:
                collect_segments(gg)

    collect_segments(noded)

    vertices: dict[tuple[int, int], str] = {}
    vertex_out: dict[str, list[float]] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}

    def intern_vertex(x: float, y: float) -> str:
        qx = round(float(x) / snap_2d_m)
        qy = round(float(y) / snap_2d_m)
        k = (qx, qy)
        vid = vertices.get(k)
        if vid is None:
            vid = f"pv:{len(vertices)}"
            vertices[k] = vid
            vertex_out[vid] = [round(qx * snap_2d_m, 6), round(qy * snap_2d_m, 6)]
        return vid

    for seg in segments:
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            a = coords[i]
            b = coords[i + 1]
            if (
                abs(float(a[0]) - float(b[0])) < 1e-12
                and abs(float(a[1]) - float(b[1])) < 1e-12
            ):
                continue
            va = intern_vertex(float(a[0]), float(a[1]))
            vb = intern_vertex(float(b[0]), float(b[1]))
            if va == vb:
                continue
            ek = (va, vb) if va < vb else (vb, va)
            rec = edges.get(ek)
            if rec is None:
                rec = {
                    "id": f"pe:{len(edges)}",
                    "vertex_a": ek[0],
                    "vertex_b": ek[1],
                    "count": 0,
                }
                edges[ek] = rec
            rec["count"] += 1

    return {
        "vertices_2d": vertex_out,
        "edges_2d": list(edges.values()),
        "stats": {"vertices": len(vertex_out), "edges": len(edges)},
    }


def _bbox_xz_from_members(
    members: list[_SurfaceWorld],
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    zs: list[float] = []
    for m in members:
        if not isinstance(m.points, np.ndarray) or m.points.shape[0] < 1:
            continue
        xs.extend(float(v) for v in m.points[:, 0].tolist())
        zs.extend(float(v) for v in m.points[:, 2].tolist())
    if not xs or not zs:
        return None
    return (min(xs), min(zs), max(xs), max(zs))


def _bbox_overlap_area_xz(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return 0.0
    ax0, az0, ax1, az1 = a
    bx0, bz0, bx1, bz1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oz = min(az1, bz1) - max(az0, bz0)
    if ox <= 0 or oz <= 0:
        return 0.0
    return float(ox * oz)


def _estimate_wall_thickness_by_cluster(
    clusters: list[_PlaneCluster],
    *,
    parallel_tolerance_deg: float = 5.0,
    min_thickness_m: float = 0.03,
    max_thickness_m: float = 1.0,
    min_overlap_area_m2: float = 0.02,
) -> dict[str, float]:
    wall_infos: list[dict[str, Any]] = []
    cos_tol = float(np.cos(np.deg2rad(parallel_tolerance_deg)))

    for ci, c in enumerate(clusters):
        if c.kind != "wall":
            continue
        ln = float(np.linalg.norm(c.normal))
        if ln < 1e-9:
            continue
        wall_infos.append(
            {
                "cluster_id": f"cluster:{c.kind}:{c.story}:{ci}",
                "story": c.story,
                "normal": c.normal / ln,
                "offset": float(c.offset),
                "bbox_xz": _bbox_xz_from_members(c.members),
            }
        )

    out: dict[str, float] = {}
    for i, a in enumerate(wall_infos):
        best = None
        for j, b in enumerate(wall_infos):
            if i == j:
                continue
            if a["story"] != b["story"]:
                continue
            if abs(float(np.dot(a["normal"], b["normal"]))) < cos_tol:
                continue
            overlap = _bbox_overlap_area_xz(a["bbox_xz"], b["bbox_xz"])
            if overlap < min_overlap_area_m2:
                continue
            d = abs(float(a["offset"]) - float(b["offset"]))
            if d < min_thickness_m or d > max_thickness_m:
                continue
            if best is None or d < best:
                best = d
        if best is not None:
            out[a["cluster_id"]] = float(best)
    return out


def _polygon_from_node_loop_xz(
    node_ids: list[str],
    shared_nodes: dict[str, list[float]],
    holes: list[dict[str, Any]] | None = None,
) -> Polygon | None:
    ext = []
    for nid in node_ids or []:
        p = shared_nodes.get(str(nid))
        if not isinstance(p, list) or len(p) < 3:
            continue
        ext.append((float(p[0]), float(p[2])))
    if len(ext) < 3:
        return None

    int_rings = []
    for h in holes or []:
        ring = []
        for nid in h.get("node_ids", []):
            p = shared_nodes.get(str(nid))
            if not isinstance(p, list) or len(p) < 3:
                continue
            ring.append((float(p[0]), float(p[2])))
        if len(ring) >= 3:
            int_rings.append(ring)
    try:
        poly = Polygon(ext, int_rings)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-8:
            return None
        return poly
    except Exception:
        return None


def _wall_length_xz(node_ids: list[str], shared_nodes: dict[str, list[float]]) -> float:
    pts = []
    for nid in node_ids or []:
        p = shared_nodes.get(str(nid))
        if not isinstance(p, list) or len(p) < 3:
            continue
        pts.append((float(p[0]), float(p[2])))
    if len(pts) < 2:
        return 0.0
    segs = []
    for i in range(len(pts)):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % len(pts)]
        segs.append(float(np.hypot(x1 - x0, z1 - z0)))
    return max(segs) if segs else 0.0


def _story_footprint_ledger(
    nodes: list[dict[str, Any]],
    stitched_surfaces: list[dict[str, Any]],
    shared_nodes: dict[str, list[float]],
    geometry_index: dict[str, Any],
) -> dict[str, Any]:
    room_story = {
        str(n.get("id")): int(n.get("story"))
        for n in nodes
        if n.get("type") == "Room" and isinstance(n.get("story"), int)
    }

    target_by_story: dict[int, list[Any]] = {}
    for rid, wkt in (geometry_index.get("room_floor_polygons") or {}).items():
        st = room_story.get(str(rid))
        if st is None:
            continue
        try:
            g = shapely_wkt.loads(wkt)
            if g is None or g.is_empty:
                continue
            target_by_story.setdefault(st, []).append(g)
        except Exception:
            continue

    stitched_by_story: dict[int, list[Polygon]] = {}
    wall_proxy_by_story: dict[int, float] = {}
    for s in stitched_surfaces:
        st = s.get("story")
        if not isinstance(st, int):
            continue
        kind = str(s.get("kind") or "")
        if kind == "floor":
            p = _polygon_from_node_loop_xz(
                list(s.get("node_ids", [])), shared_nodes, holes=s.get("holes", [])
            )
            if p is not None:
                stitched_by_story.setdefault(st, []).append(p)
        if kind == "wall" and s.get("thickness_m") is not None:
            length = _wall_length_xz(list(s.get("node_ids", [])), shared_nodes)
            wall_proxy_by_story[st] = wall_proxy_by_story.get(st, 0.0) + (
                float(s.get("thickness_m")) * max(length, 0.0)
            )

    stories = sorted(
        set(target_by_story.keys())
        | set(stitched_by_story.keys())
        | set(wall_proxy_by_story.keys())
    )
    out = []
    max_abs_drift_pct = 0.0
    for st in stories:
        target_geom = (
            unary_union(target_by_story.get(st, []))
            if target_by_story.get(st)
            else None
        )
        stitched_geom = (
            unary_union(stitched_by_story.get(st, []))
            if stitched_by_story.get(st)
            else None
        )
        target_area = float(target_geom.area) if target_geom is not None else 0.0
        stitched_area = float(stitched_geom.area) if stitched_geom is not None else 0.0
        drift = stitched_area - target_area
        drift_pct = (drift / target_area * 100.0) if target_area > 1e-8 else 0.0
        max_abs_drift_pct = max(max_abs_drift_pct, abs(drift_pct))
        out.append(
            {
                "story": st,
                "target_footprint_area_m2": round(target_area, 6),
                "stitched_floor_area_m2": round(stitched_area, 6),
                "area_drift_m2": round(drift, 6),
                "area_drift_pct": round(drift_pct, 4),
                "wall_thickness_length_proxy_m2": round(
                    float(wall_proxy_by_story.get(st, 0.0)), 6
                ),
            }
        )

    return {
        "stories": out,
        "max_abs_area_drift_pct": round(max_abs_drift_pct, 4),
    }


def _quant_key3(p: np.ndarray, snap: float) -> tuple[int, int, int]:
    return (
        round(float(p[0]) / snap),
        round(float(p[1]) / snap),
        round(float(p[2]) / snap),
    )


def _node_point(
    shared_nodes: dict[str, list[float]], nid: str
) -> tuple[float, float, float] | None:
    p = shared_nodes.get(nid)
    if not isinstance(p, list) or len(p) < 3:
        return None
    return (float(p[0]), float(p[1]), float(p[2]))


def _point_segment_distance(
    p: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    abz = b[2] - a[2]
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 < 1e-12:
        dx = p[0] - a[0]
        dy = p[1] - a[1]
        dz = p[2] - a[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz), 0.0
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    apz = p[2] - a[2]
    t = (apx * abx + apy * aby + apz * abz) / ab2
    t_clamped = max(0.0, min(1.0, t))
    proj_x = a[0] + t_clamped * abx
    proj_y = a[1] + t_clamped * aby
    proj_z = a[2] + t_clamped * abz
    dx = p[0] - proj_x
    dy = p[1] - proj_y
    dz = p[2] - proj_z
    return math.sqrt(dx * dx + dy * dy + dz * dz), t_clamped


def _point_distance_to_plane(
    p: tuple[float, float, float],
    plane_origin: tuple[float, float, float] | None,
    plane_normal: np.ndarray | None,
) -> float:
    if plane_origin is None or plane_normal is None:
        return 0.0
    dx = p[0] - plane_origin[0]
    dy = p[1] - plane_origin[1]
    dz = p[2] - plane_origin[2]
    return abs(
        float(plane_normal[0]) * dx
        + float(plane_normal[1]) * dy
        + float(plane_normal[2]) * dz
    )


def _build_edge_topology(
    stitched_surfaces: list[dict[str, Any]],
    shared_nodes: dict[str, list[float]],
    *,
    t_junction_tol_m: float = 0.01,
) -> dict[str, Any]:
    """Construct explicit shared-edge constraints + topology validation."""
    undirected: dict[tuple[str, str], dict[str, Any]] = {}
    face_edges: dict[str, list[tuple[str, str]]] = {}
    issues: list[dict[str, Any]] = []

    def add_loop(
        surface_id: str,
        loop_node_ids: list[str],
        loop_kind: str,
        loop_id: str | None = None,
    ) -> None:
        if not isinstance(loop_node_ids, list) or len(loop_node_ids) < 3:
            return
        seq = loop_node_ids
        n = len(seq)
        local = []
        for i in range(n):
            a = str(seq[i])
            b = str(seq[(i + 1) % n])
            if a == b:
                continue
            k = (a, b) if a < b else (b, a)
            rec = undirected.get(k)
            if rec is None:
                rec = {
                    "id": f"edge:{len(undirected)}",
                    "node_a": k[0],
                    "node_b": k[1],
                    "incident_surfaces": set(),
                    "oriented_counts": {},
                    "loop_refs": [],
                }
                undirected[k] = rec
            rec["incident_surfaces"].add(surface_id)
            oriented_key = f"{a}->{b}"
            rec["oriented_counts"][oriented_key] = (
                rec["oriented_counts"].get(oriented_key, 0) + 1
            )
            rec["loop_refs"].append(
                {
                    "surface_id": surface_id,
                    "loop_kind": loop_kind,
                    "loop_id": loop_id,
                    "oriented": oriented_key,
                }
            )
            local.append(k)
        face_edges[surface_id] = face_edges.get(surface_id, []) + local

    for s in stitched_surfaces:
        sid = s["id"]
        add_loop(sid, s.get("node_ids", []), "exterior", None)
        for h in s.get("holes", []):
            add_loop(sid, h.get("node_ids", []), "hole", h.get("id"))

    edges_out = []
    face_adjacency_set: set[tuple[str, str, str]] = set()
    non_manifold_edges = 0
    duplicate_oriented_edges = 0

    for rec in undirected.values():
        inc = sorted(list(rec["incident_surfaces"]))
        orient = rec["oriented_counts"]
        total_oriented = sum(int(v) for v in orient.values())
        dup_oriented = any(int(v) > 1 for v in orient.values())
        if dup_oriented:
            duplicate_oriented_edges += 1

        if len(inc) == 1:
            cls = "boundary"
        elif len(inc) == 2:
            cls = "interior_manifold"
        else:
            cls = "non_manifold"
            non_manifold_edges += 1
            issues.append(
                {
                    "code": "NON_MANIFOLD_EDGE",
                    "edge_id": rec["id"],
                    "node_a": rec["node_a"],
                    "node_b": rec["node_b"],
                    "incident_surfaces": inc,
                }
            )

        if dup_oriented:
            issues.append(
                {
                    "code": "DUPLICATE_ORIENTED_EDGE",
                    "edge_id": rec["id"],
                    "node_a": rec["node_a"],
                    "node_b": rec["node_b"],
                    "oriented_counts": orient,
                }
            )

        if len(inc) == 2:
            a, b = sorted(inc)
            face_adjacency_set.add((a, b, rec["id"]))

        edges_out.append(
            {
                "id": rec["id"],
                "node_a": rec["node_a"],
                "node_b": rec["node_b"],
                "incident_surface_ids": inc,
                "classification": cls,
                "oriented_counts": orient,
                "loop_refs": rec["loop_refs"],
                "oriented_total": total_oriented,
            }
        )

    # T-junction detection: node near interior of edge
    # without being one of its endpoints.
    t_junctions = []
    active_node_ids: set[str] = set()
    surface_cluster_by_id = {
        str(surface.get("id")): str(surface.get("cluster_id") or "")
        for surface in stitched_surfaces
    }
    node_cluster_ids: dict[str, set[str]] = {}
    for s in stitched_surfaces:
        cluster_id = str(s.get("cluster_id") or "")
        for nid in s.get("node_ids", []):
            active_node_ids.add(str(nid))
            if cluster_id:
                node_cluster_ids.setdefault(str(nid), set()).add(cluster_id)
        for h in s.get("holes", []):
            for nid in h.get("node_ids", []):
                active_node_ids.add(str(nid))
                if cluster_id:
                    node_cluster_ids.setdefault(str(nid), set()).add(cluster_id)
    node_ids = sorted(active_node_ids)
    edge_pairs: list[tuple[str, str, str, tuple[str, ...]]] = []
    edge_indices_by_cluster: dict[str, list[int]] = {}
    for edge in edges_out:
        cluster_ids = tuple(
            sorted(
                {
                    cluster_id
                    for ref in (edge.get("loop_refs") or [])
                    if (
                        cluster_id := surface_cluster_by_id.get(
                            str(ref.get("surface_id") or "")
                        )
                    )
                }
            )
        )
        edge_index = len(edge_pairs)
        edge_pairs.append((edge["id"], edge["node_a"], edge["node_b"], cluster_ids))
        for cluster_id in cluster_ids:
            edge_indices_by_cluster.setdefault(cluster_id, []).append(edge_index)
    node_points = {
        nid: p for nid in node_ids if (p := _node_point(shared_nodes, nid)) is not None
    }

    for nid in node_ids:
        p = node_points.get(nid)
        if p is None:
            continue
        candidate_indices: list[int] = []
        seen_indices: set[int] = set()
        for cluster_id in sorted(node_cluster_ids.get(nid, set())):
            for edge_index in edge_indices_by_cluster.get(cluster_id, []):
                if edge_index in seen_indices:
                    continue
                seen_indices.add(edge_index)
                candidate_indices.append(edge_index)
        if not candidate_indices:
            candidate_indices = list(range(len(edge_pairs)))
        for edge_index in candidate_indices:
            eid, na, nb, _cluster_ids = edge_pairs[edge_index]
            if nid == na or nid == nb:
                continue
            a = node_points.get(na)
            b = node_points.get(nb)
            if a is None or b is None:
                continue
            dist, t = _point_segment_distance(p, a, b)
            if dist <= t_junction_tol_m and t_junction_tol_m < t < (
                1.0 - t_junction_tol_m
            ):
                t_junctions.append(
                    {
                        "code": "T_JUNCTION",
                        "node_id": nid,
                        "edge_id": eid,
                        "distance_m": round(dist, 6),
                        "edge_param": round(t, 6),
                    }
                )
                if len(t_junctions) <= 200:
                    issues.append(t_junctions[-1])

    boundary_edges = sum(1 for e in edges_out if e["classification"] == "boundary")
    interior_edges = sum(
        1 for e in edges_out if e["classification"] == "interior_manifold"
    )

    manifold_pass = (
        non_manifold_edges == 0
        and duplicate_oriented_edges == 0
        and len(t_junctions) == 0
    )

    return {
        "undirected_edges": edges_out,
        "face_adjacency": [
            {"surface_a": a, "surface_b": b, "shared_edge_id": eid}
            for a, b, eid in sorted(face_adjacency_set)
        ],
        "validation": {
            "manifold_pass": manifold_pass,
            "total_edges": len(edges_out),
            "boundary_edges": boundary_edges,
            "interior_edges": interior_edges,
            "non_manifold_edges": non_manifold_edges,
            "duplicate_oriented_edges": duplicate_oriented_edges,
            "t_junction_count": len(t_junctions),
            "issue_count": len(issues),
            "issues": issues,
        },
    }


def _dedupe_loop_nodes(loop_ids: list[str]) -> list[str]:
    out: list[str] = []
    for nid in loop_ids:
        if not out or out[-1] != nid:
            out.append(nid)
    if len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    return out


def _simplify_loop_nodes(
    loop_ids: list[str],
    shared_nodes: dict[str, list[float]],
    *,
    tol_m: float,
) -> tuple[list[str], int]:
    """Remove degenerate vertices (duplicate, spikes, colinear points) from a loop."""
    ids = _dedupe_loop_nodes(loop_ids)
    if len(ids) < 3:
        return ids, 0
    node_points = {
        nid: p for nid in ids if (p := _node_point(shared_nodes, nid)) is not None
    }

    removed = 0
    changed = True
    while changed and len(ids) >= 3:
        changed = False
        n = len(ids)
        i = 0
        while i < n and len(ids) >= 3:
            a_id = ids[(i - 1) % len(ids)]
            b_id = ids[i % len(ids)]
            c_id = ids[(i + 1) % len(ids)]

            # Remove immediate spikes: a->b->a
            if a_id == c_id:
                ids.pop(i % len(ids))
                removed += 1
                changed = True
                n = len(ids)
                continue

            a = node_points.get(a_id)
            b = node_points.get(b_id)
            c = node_points.get(c_id)
            if a is None or b is None or c is None:
                i += 1
                continue

            # Remove b when nearly colinear with segment a-c.
            dist, t = _point_segment_distance(b, a, c)
            if dist <= tol_m and tol_m < t < (1.0 - tol_m):
                ids.pop(i % len(ids))
                removed += 1
                changed = True
                n = len(ids)
                continue
            i += 1

    return _dedupe_loop_nodes(ids), removed


def _canonical_loop_key(loop_ids: list[str]) -> tuple[str, ...]:
    ids = _dedupe_loop_nodes(loop_ids)
    if len(ids) < 3:
        return tuple()

    n = len(ids)
    rots_fwd = [tuple(ids[i:] + ids[:i]) for i in range(n)]
    rev = list(reversed(ids))
    rots_rev = [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(rots_fwd + rots_rev)


def _dedupe_stitched_surfaces(
    stitched_surfaces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    index: dict[tuple[Any, ...], int] = {}
    merged = 0

    for s in stitched_surfaces:
        ext_key = _canonical_loop_key(list(s.get("node_ids", [])))
        if not ext_key:
            continue

        holes = [h for h in (s.get("holes") or []) if isinstance(h, dict)]
        hole_keys = sorted(
            [_canonical_loop_key(list(h.get("node_ids", []))) for h in holes]
        )
        hole_keys = [hk for hk in hole_keys if hk]

        normal = s.get("normal") or [0.0, 0.0, 0.0]
        if not isinstance(normal, list) or len(normal) < 3:
            normal = [0.0, 0.0, 0.0]
        normal_q = (
            round(float(normal[0]), 4),
            round(float(normal[1]), 4),
            round(float(normal[2]), 4),
        )
        offset_q = round(float(s.get("offset_m", 0.0)), 4)

        key = (
            s.get("story"),
            s.get("kind"),
            normal_q,
            offset_q,
            ext_key,
            tuple(hole_keys),
        )

        at = index.get(key)
        if at is None:
            c = dict(s)
            c["source_surface_ids"] = sorted(set(list(s.get("source_surface_ids", []))))
            deduped.append(c)
            index[key] = len(deduped) - 1
            continue

        merged += 1
        c = deduped[at]
        c["source_surface_ids"] = sorted(
            set(list(c.get("source_surface_ids", [])))
            | set(list(s.get("source_surface_ids", [])))
        )

    return deduped, merged


def _split_loop_edges_at_nodes(
    loop_ids: list[str],
    shared_nodes: dict[str, list[float]],
    *,
    tol_m: float,
    candidate_node_ids: list[str] | None = None,
    plane_origin: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
) -> tuple[list[str], int]:
    """Insert existing nodes that lie on loop edges (splitting)."""
    ids = _dedupe_loop_nodes(loop_ids)
    if len(ids) < 3:
        return ids, 0

    all_nodes = (
        candidate_node_ids
        if candidate_node_ids is not None
        else list(shared_nodes.keys())
    )
    node_points = {
        nid: p
        for nid in set(list(ids) + [str(nid) for nid in all_nodes])
        if (p := _node_point(shared_nodes, str(nid))) is not None
    }

    inserted = 0
    result: list[str] = []
    n = len(ids)
    for i in range(n):
        a_id = ids[i]
        b_id = ids[(i + 1) % n]
        a = node_points.get(a_id)
        b = node_points.get(b_id)
        if a is None or b is None:
            continue

        splits: list[tuple[float, str]] = []
        for nid in all_nodes:
            if nid == a_id or nid == b_id:
                continue
            p = node_points.get(str(nid))
            if p is None:
                continue
            if plane_origin is not None and plane_normal is not None:
                plane_dist = _point_distance_to_plane(p, plane_origin, plane_normal)
                if plane_dist > (tol_m * 1.5):
                    continue
            dist, t = _point_segment_distance(p, a, b)
            if dist <= tol_m and tol_m < t < (1.0 - tol_m):
                splits.append((t, nid))

        splits.sort(key=lambda x: x[0])
        # Keep unique node insertions per edge by node id and roughly by t.
        filtered: list[str] = []
        last_t = -1.0
        seen = set()
        for t, nid in splits:
            if nid in seen:
                continue
            if last_t >= 0 and abs(t - last_t) < 1e-6:
                continue
            seen.add(nid)
            filtered.append(nid)
            last_t = t

        result.append(a_id)
        for nid in filtered:
            if nid != a_id and nid != b_id:
                result.append(nid)
                inserted += 1

    result = _dedupe_loop_nodes(result)
    if len(result) < 3:
        return ids, 0
    return result, inserted


def _split_surface_loops_at_existing_nodes(
    stitched_surfaces: list[dict[str, Any]],
    shared_nodes: dict[str, list[float]],
    *,
    tol_m: float,
    candidate_nodes_by_cluster: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Split stitched surface loops by inserting colinear existing nodes."""
    out = []
    total_inserted = 0
    for s in stitched_surfaces:
        s2 = dict(s)
        n = s.get("normal") or [0.0, 0.0, 0.0]
        normal = None
        if isinstance(n, list) and len(n) >= 3:
            nn = np.array([float(n[0]), float(n[1]), float(n[2])], dtype=float)
            ln = float(np.linalg.norm(nn))
            if ln > 1e-8:
                normal = nn / ln
        origin = (
            _node_point(shared_nodes, str((s.get("node_ids") or [""])[0]))
            if s.get("node_ids")
            else None
        )
        ext, nins = _split_loop_edges_at_nodes(
            list(s.get("node_ids", [])),
            shared_nodes,
            tol_m=tol_m,
            candidate_node_ids=(
                (candidate_nodes_by_cluster or {}).get(str(s.get("cluster_id") or ""))
                if candidate_nodes_by_cluster
                else None
            ),
            plane_origin=origin,
            plane_normal=normal,
        )
        s2["node_ids"] = ext
        total_inserted += nins

        holes = []
        for h in s.get("holes", []):
            h2 = dict(h)
            hid, hins = _split_loop_edges_at_nodes(
                list(h.get("node_ids", [])),
                shared_nodes,
                tol_m=tol_m,
                candidate_node_ids=(
                    (candidate_nodes_by_cluster or {}).get(
                        str(s.get("cluster_id") or "")
                    )
                    if candidate_nodes_by_cluster
                    else None
                ),
                plane_origin=origin,
                plane_normal=normal,
            )
            h2["node_ids"] = hid
            total_inserted += hins
            holes.append(h2)
        s2["holes"] = holes
        out.append(s2)
    return out, total_inserted


def _simplify_surface_loops(
    stitched_surfaces: list[dict[str, Any]],
    shared_nodes: dict[str, list[float]],
    *,
    tol_m: float,
) -> tuple[list[dict[str, Any]], int]:
    out = []
    removed_vertices = 0

    for s in stitched_surfaces:
        s2 = dict(s)
        ext, r = _simplify_loop_nodes(
            list(s.get("node_ids", [])), shared_nodes, tol_m=tol_m
        )
        removed_vertices += r
        if len(ext) < 3:
            continue
        s2["node_ids"] = ext

        holes = []
        seen_holes: set[tuple[str, ...]] = set()
        for h in s.get("holes", []):
            h2 = dict(h)
            hid, hr = _simplify_loop_nodes(
                list(h.get("node_ids", [])), shared_nodes, tol_m=tol_m
            )
            removed_vertices += hr
            if len(hid) < 3:
                continue
            hk = _canonical_loop_key(hid)
            if not hk or hk in seen_holes:
                continue
            seen_holes.add(hk)
            h2["node_ids"] = hid
            holes.append(h2)
        s2["holes"] = holes
        out.append(s2)

    return out, removed_vertices


def build_stitched_geometry(
    nodes: list[dict[str, Any]],
    geometry_index: dict[str, Any],
    *,
    angle_tolerance_deg: float = 3.0,
    distance_tolerance_m: float = 0.03,
    snap_m: float = 0.01,
    merge_gap_bridge_m: float = 0.015,
    partial_coplanar_split: bool = True,
    conservative_mode: bool = True,
    kinds: tuple[str, ...] = ("wall", "floor", "door", "window", "opening", "storage"),
) -> dict[str, Any]:
    """Generate stitched surfaces and shared nodes from V2 surfaces."""
    effective_merge_gap_bridge_m = 0.0 if conservative_mode else merge_gap_bridge_m

    surfaces: list[_SurfaceWorld] = []
    synthetic_surface_ids = {
        str(n.get("id"))
        for n in nodes
        if n.get("type") == "Surface"
        and isinstance((n.get("properties") or {}).get("synthetic_kind"), str)
    }
    transforms = geometry_index.get("transforms", {})
    local_polys = geometry_index.get("surface_polygons", {})

    for n in nodes:
        if n.get("type") != "Surface":
            continue
        kind = (n.get("properties") or {}).get("surface_kind", "surface")
        if kind not in kinds:
            continue
        sid = n.get("id")
        tf_ref = n.get("transform_ref")
        poly = local_polys.get(sid)
        tf = transforms.get(tf_ref)
        if (
            not sid
            or not isinstance(poly, list)
            or len(poly) < 3
            or not isinstance(tf, list)
            or len(tf) != 16
        ):
            continue
        world = np.array([_transform_point(tf, p) for p in poly], dtype=float)
        surfaces.append(
            _SurfaceWorld(surface_id=sid, story=n.get("story"), kind=kind, points=world)
        )

    clusters: list[_PlaneCluster] = []
    cos_tol = float(np.cos(np.deg2rad(angle_tolerance_deg)))

    for s in surfaces:
        n = _stable_normal(s.points)
        if n is None:
            continue
        offset = float(np.dot(n, s.points.mean(axis=0)))

        assigned = False
        for c in clusters:
            if c.story != s.story or c.kind != s.kind:
                continue
            if abs(float(np.dot(c.normal, n))) < cos_tol:
                continue
            if abs(c.offset - offset) > distance_tolerance_m:
                continue
            c.members.append(s)
            c.normal = c.normal + n
            c.normal = c.normal / max(np.linalg.norm(c.normal), 1e-12)
            c.offset = float(
                (c.offset * (len(c.members) - 1) + offset) / len(c.members)
            )
            assigned = True
            break

        if not assigned:
            clusters.append(
                _PlaneCluster(
                    story=s.story, kind=s.kind, normal=n, offset=offset, members=[s]
                )
            )

    shared_nodes: dict[str, list[float]] = {}
    shared_node_index: dict[tuple[int, int, int], str] = {}
    stitched_surfaces: list[dict[str, Any]] = []
    render_surfaces: list[dict[str, Any]] = []
    wall_thickness_by_cluster = _estimate_wall_thickness_by_cluster(clusters)
    cluster_split_candidates: dict[str, set[str]] = {}
    arrangement_clusters: list[dict[str, Any]] = []

    def intern_shared_node(p: np.ndarray) -> str:
        key = _quant_key3(p, snap_m)
        nid = shared_node_index.get(key)
        if nid is None:
            nid = f"v:{len(shared_nodes)}"
            shared_node_index[key] = nid
            shared_nodes[nid] = [
                float(round(p[0], 6)),
                float(round(p[1], 6)),
                float(round(p[2], 6)),
            ]
        return nid

    for ci, cluster in enumerate(clusters):
        if not cluster.members:
            continue
        normal = cluster.normal / max(np.linalg.norm(cluster.normal), 1e-12)
        u, v = _plane_basis(normal)
        origin = cluster.members[0].points.mean(axis=0)

        member_polys2: list[tuple[str, Polygon]] = []
        for m in cluster.members:
            p2 = _project(m.points, origin, u, v)
            # Snap in plane space to reduce micro gaps.
            p2s = np.round(p2 / snap_m) * snap_m
            poly = Polygon([(float(x), float(y)) for x, y in p2s])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-6:
                continue
            member_polys2.append((m.surface_id, poly))

        if not member_polys2:
            continue

        arrangement_faces, _union_area = _arrangement_faces(
            [p for _, p in member_polys2],
            kind=cluster.kind,
            merge_gap_bridge_m=effective_merge_gap_bridge_m,
        )
        cluster_id = f"cluster:{cluster.kind}:{cluster.story}:{ci}"
        arr_graph = _arrangement_graph([p for _, p in member_polys2], snap_2d_m=snap_m)
        arrangement_clusters.append(
            {
                "cluster_id": cluster_id,
                "story": cluster.story,
                "kind": cluster.kind,
                "vertex_count": arr_graph["stats"]["vertices"],
                "edge_count": arr_graph["stats"]["edges"],
            }
        )
        split_nodes = cluster_split_candidates.setdefault(cluster_id, set())
        for p2 in arr_graph["vertices_2d"].values():
            p3 = _unproject(
                np.array([[float(p2[0]), float(p2[1])]], dtype=float), origin, u, v
            )[0]
            split_nodes.add(intern_shared_node(p3))

        face_records: list[tuple[Polygon, list[str]]] = []
        for poly in arrangement_faces:
            if poly.is_empty or poly.area < 1e-6:
                continue
            source_ids = []
            area_eps = max(1e-8, float(poly.area) * 1e-3)
            for sid, src_poly in member_polys2:
                try:
                    if src_poly.intersection(poly).area > area_eps:
                        source_ids.append(sid)
                except Exception:
                    continue
            if not source_ids:
                source_ids = [sid for sid, _ in member_polys2]
            face_records.append((poly, sorted(source_ids)))

        if partial_coplanar_split and cluster.kind in ("wall", "floor"):
            face_records = _dissolve_faces_by_source_sets(face_records)

        for pi, (poly, source_ids) in enumerate(face_records):
            exterior = np.array(poly.exterior.coords[:-1], dtype=float)
            ext3 = _unproject(exterior, origin, u, v)

            node_ids = []
            for p in ext3:
                nid = intern_shared_node(p)
                node_ids.append(nid)
                split_nodes.add(nid)

            hole_loops = []
            for hi, hole in enumerate(poly.interiors):
                h2 = np.array(hole.coords[:-1], dtype=float)
                h3 = _unproject(h2, origin, u, v)
                h_ids = []
                for p in h3:
                    nid = intern_shared_node(p)
                    h_ids.append(nid)
                    split_nodes.add(nid)
                hole_loops.append({"id": f"hole:{ci}:{pi}:{hi}", "node_ids": h_ids})

            stitched_surfaces.append(
                {
                    "id": f"stitched:{cluster.kind}:{cluster.story}:{ci}:{pi}",
                    "story": cluster.story,
                    "kind": cluster.kind,
                    "cluster_id": cluster_id,
                    "normal": [
                        float(round(normal[0], 6)),
                        float(round(normal[1], 6)),
                        float(round(normal[2], 6)),
                    ],
                    "offset_m": float(round(cluster.offset, 6)),
                    "source_surface_ids": sorted(source_ids),
                    "node_ids": node_ids,
                    "holes": hole_loops,
                    "area_m2": float(round(poly.area, 6)),
                    **(
                        {
                            "thickness_m": float(
                                round(wall_thickness_by_cluster.get(cluster_id, 0.0), 6)
                            ),
                            "thickness_cm": float(
                                round(
                                    wall_thickness_by_cluster.get(cluster_id, 0.0)
                                    * 100.0,
                                    2,
                                )
                            ),
                        }
                        if cluster.kind == "wall"
                        and cluster_id in wall_thickness_by_cluster
                        else {}
                    ),
                }
            )

        # Conservative merged render planes: dissolve whole
        # coplanar cluster for wall/floor.
        if cluster.kind in ("wall", "floor"):
            base_member_polys2 = [
                (sid, p) for sid, p in member_polys2 if sid not in synthetic_surface_ids
            ]
            if base_member_polys2:
                render_union = unary_union([p for _, p in base_member_polys2]).buffer(0)
                render_polys = _to_polygons(render_union)
                render_source_ids = sorted([sid for sid, _ in base_member_polys2])
            else:
                render_polys = (
                    _to_polygons(_union_area)
                    if _union_area is not None
                    else [fr[0] for fr in face_records]
                )
                render_source_ids = sorted([sid for sid, _ in member_polys2])
        else:
            render_polys = [fr[0] for fr in face_records]
            render_source_ids = sorted([sid for sid, _ in member_polys2])
        for rpi, rpoly in enumerate(render_polys):
            if rpoly.is_empty or rpoly.area < 1e-6:
                continue
            rexterior = np.array(rpoly.exterior.coords[:-1], dtype=float)
            rext3 = _unproject(rexterior, origin, u, v)
            rnode_ids = []
            for p in rext3:
                nid = intern_shared_node(p)
                rnode_ids.append(nid)
                split_nodes.add(nid)
            rhole_loops = []
            for hi, hole in enumerate(rpoly.interiors):
                h2 = np.array(hole.coords[:-1], dtype=float)
                h3 = _unproject(h2, origin, u, v)
                h_ids = []
                for p in h3:
                    nid = intern_shared_node(p)
                    h_ids.append(nid)
                    split_nodes.add(nid)
                rhole_loops.append(
                    {
                        "id": f"render-hole:{ci}:{rpi}:{hi}",
                        "node_ids": h_ids,
                    }
                )

            render_surfaces.append(
                {
                    "id": f"render:{cluster.kind}:{cluster.story}:{ci}:{rpi}",
                    "story": cluster.story,
                    "kind": cluster.kind,
                    "cluster_id": cluster_id,
                    "normal": [
                        float(round(normal[0], 6)),
                        float(round(normal[1], 6)),
                        float(round(normal[2], 6)),
                    ],
                    "offset_m": float(round(cluster.offset, 6)),
                    "source_surface_ids": render_source_ids,
                    "node_ids": rnode_ids,
                    "holes": rhole_loops,
                    "area_m2": float(round(rpoly.area, 6)),
                    **(
                        {
                            "thickness_m": float(
                                round(wall_thickness_by_cluster.get(cluster_id, 0.0), 6)
                            ),
                            "thickness_cm": float(
                                round(
                                    wall_thickness_by_cluster.get(cluster_id, 0.0)
                                    * 100.0,
                                    2,
                                )
                            ),
                        }
                        if cluster.kind == "wall"
                        and cluster_id in wall_thickness_by_cluster
                        else {}
                    ),
                }
            )

    stitched_surfaces, split_insertions = _split_surface_loops_at_existing_nodes(
        stitched_surfaces,
        shared_nodes,
        tol_m=max(snap_m * 1.25, 0.005),
        candidate_nodes_by_cluster={
            k: sorted(v) for k, v in cluster_split_candidates.items()
        },
    )
    stitched_surfaces, simplified_vertices = _simplify_surface_loops(
        stitched_surfaces,
        shared_nodes,
        tol_m=max(snap_m * 0.5, 0.003),
    )
    stitched_surfaces, duplicate_faces_merged = _dedupe_stitched_surfaces(
        stitched_surfaces
    )

    edge_topology = _build_edge_topology(
        stitched_surfaces, shared_nodes, t_junction_tol_m=snap_m
    )
    validation = edge_topology["validation"]
    footprint_ledger = _story_footprint_ledger(
        nodes, stitched_surfaces, shared_nodes, geometry_index
    )

    if conservative_mode:
        max_drift = float(footprint_ledger.get("max_abs_area_drift_pct", 0.0))
        if max_drift > 1.0:
            validation["issues"].append(
                {
                    "code": "FOOTPRINT_AREA_DRIFT",
                    "max_abs_area_drift_pct": round(max_drift, 4),
                }
            )
            validation["issue_count"] = len(validation["issues"])

    return {
        "shared_nodes": shared_nodes,
        "stitched_surfaces": stitched_surfaces,
        "render_surfaces": render_surfaces,
        "edge_topology": edge_topology,
        "story_footprint_ledger": footprint_ledger,
        "planar_arrangement": {
            "clusters": arrangement_clusters,
        },
        "validation": validation,
        "stats": {
            "input_surfaces": len(surfaces),
            "clusters": len(clusters),
            "stitched_surfaces": len(stitched_surfaces),
            "render_surfaces": len(render_surfaces),
            "shared_nodes": len(shared_nodes),
            "shared_edges": len(edge_topology["undirected_edges"]),
            "face_adjacency": len(edge_topology["face_adjacency"]),
            "manifold_pass": validation["manifold_pass"],
            "t_junction_count": validation["t_junction_count"],
            "non_manifold_edges": validation["non_manifold_edges"],
            "duplicate_oriented_edges": validation["duplicate_oriented_edges"],
            "split_insertions": split_insertions,
            "simplified_vertices_removed": simplified_vertices,
            "duplicate_faces_merged": duplicate_faces_merged,
            "wall_thickness_estimated_clusters": len(wall_thickness_by_cluster),
            "max_story_footprint_area_drift_pct": footprint_ledger.get(
                "max_abs_area_drift_pct", 0.0
            ),
            "merge_gap_bridge_m": effective_merge_gap_bridge_m,
            "snap_m": snap_m,
            "angle_tolerance_deg": angle_tolerance_deg,
            "distance_tolerance_m": distance_tolerance_m,
            "conservative_mode": conservative_mode,
        },
    }
