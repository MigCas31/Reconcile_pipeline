"""Build topology V2 graph from merged + scan inputs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely import wkt as shapely_wkt

from reconcile.loader import load_merged, load_scan_cache
from reconcile.models import Room, Surface, Vec3
from reconcile.transform import room_floor_polygon

from .models import GraphEdge, GraphNode, TopologyGraph
from .stitch_geometry import build_stitched_geometry
from .topology import (
    GapRecord,
    IntraStoryAdjacency,
    infer_gap_records,
    infer_intra_story_adjacency,
)


def _stable_hash(parts: list[str], length: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:length]


def _edge_id(edge_type: str, from_id: str, to_id: str, salt: str = "") -> str:
    return f"edge:{edge_type.lower()}:{_stable_hash([edge_type, from_id, to_id, salt])}"


def _polygon_bbox_xz(room: Room) -> list[float] | None:
    poly = room_floor_polygon(room)
    if poly is None or poly.is_empty:
        return None
    minx, minz, maxx, maxz = poly.bounds
    return [round(minx, 4), round(minz, 4), round(maxx, 4), round(maxz, 4)]


def _surface_local_polygon_from_dimensions(
    kind: str, dims_xyz: tuple[float, float, float]
) -> list[list[float]]:
    """Build a conservative local polygon when polygonCorners are absent.

    RoomPlan surfaces are typically represented in local space and
    positioned by transform.  For vertical surfaces, we model a
    width/height rectangle in local XY.  For floor-like surfaces,
    we model a width/depth rectangle in local XZ.
    """
    w = max(float(dims_xyz[0]), 0.001)
    h = max(float(dims_xyz[1]), 0.001)
    d = max(float(dims_xyz[2]), 0.001)

    if kind == "floor":
        hx = w / 2.0
        hz = d / 2.0
        return [
            [-hx, 0.0, -hz],
            [hx, 0.0, -hz],
            [hx, 0.0, hz],
            [-hx, 0.0, hz],
        ]

    hw = w / 2.0
    hh = h / 2.0
    return [
        [-hw, -hh, 0.0],
        [hw, -hh, 0.0],
        [hw, hh, 0.0],
        [-hw, hh, 0.0],
    ]


def _surface_local_polygon(surface: Surface, kind: str) -> list[list[float]]:
    if surface.polygon_corners:
        return [[float(c.x), float(c.y), float(c.z)] for c in surface.polygon_corners]
    dims = (surface.dimensions.x, surface.dimensions.y, surface.dimensions.z)
    return _surface_local_polygon_from_dimensions(kind, dims)


def _raw_surface_local_polygon(raw: dict, kind: str) -> list[list[float]]:
    corners = raw.get("polygonCorners") or []
    if isinstance(corners, list) and len(corners) >= 3:
        out = []
        for c in corners:
            if isinstance(c, list) and len(c) >= 3:
                out.append([float(c[0]), float(c[1]), float(c[2])])
        if len(out) >= 3:
            return out
    dims = raw.get("dimensions") or [0.1, 0.1, 0.1]
    if not isinstance(dims, list) or len(dims) < 3:
        dims = [0.1, 0.1, 0.1]
    return _surface_local_polygon_from_dimensions(
        kind, (float(dims[0]), float(dims[1]), float(dims[2]))
    )


def _raw_apply_transform(
    local_polygon: list[list[float]], tf_flat: list[float]
) -> list[list[float]]:
    world: list[list[float]] = []
    if not isinstance(tf_flat, list) or len(tf_flat) != 16:
        return world
    for p in local_polygon:
        if not isinstance(p, list) or len(p) < 3:
            continue
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        tx = tf_flat[0] * x + tf_flat[4] * y + tf_flat[8] * z + tf_flat[12]
        ty = tf_flat[1] * x + tf_flat[5] * y + tf_flat[9] * z + tf_flat[13]
        tz = tf_flat[2] * x + tf_flat[6] * y + tf_flat[10] * z + tf_flat[14]
        tw = tf_flat[3] * x + tf_flat[7] * y + tf_flat[11] * z + tf_flat[15]
        if abs(tw) > 1e-8 and abs(tw - 1.0) > 1e-8:
            world.append([tx / tw, ty / tw, tz / tw])
        else:
            world.append([tx, ty, tz])
    return world


def _floor_local_polygon(floor_obj) -> list[list[float]]:
    if floor_obj.polygon_corners:
        return [[float(c.x), float(c.y), float(c.z)] for c in floor_obj.polygon_corners]
    dims = (floor_obj.dimensions.x, floor_obj.dimensions.y, floor_obj.dimensions.z)
    return _surface_local_polygon_from_dimensions("floor", dims)


def _apply_transform_to_local_polygon(
    local_polygon: list[list[float]], transform
) -> list[list[float]]:
    world = []
    for x, y, z in local_polygon:
        p = transform.apply(Vec3(x=x, y=y, z=z))
        world.append([float(p.x), float(p.y), float(p.z)])
    return world


def _surface_bbox_xz(surface: Surface, kind: str) -> list[float] | None:
    local_polygon = _surface_local_polygon(surface, kind)
    if not local_polygon:
        t = surface.transform.translation
        return [round(t.x, 4), round(t.z, 4), round(t.x, 4), round(t.z, 4)]
    world = _apply_transform_to_local_polygon(local_polygon, surface.transform)
    xs = [p[0] for p in world]
    zs = [p[2] for p in world]
    return [round(min(xs), 4), round(min(zs), 4), round(max(xs), 4), round(max(zs), 4)]


def _surface_kind(surface: Surface, fallback: str = "surface") -> str:
    return next(iter(surface.category.keys()), fallback)


def _ifc_class_for_surface(kind: str) -> str:
    if kind == "wall":
        return "IfcWall"
    if kind == "door":
        return "IfcDoor"
    if kind == "window":
        return "IfcWindow"
    if kind == "opening":
        return "IfcOpeningElement"
    if kind == "storage":
        return "IfcFurniture"
    if kind == "floor":
        return "IfcSlab"
    return "IfcBuildingElementProxy"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _scan_manifest_hash(scan_dir: Path) -> str:
    rows: list[str] = []
    for p in sorted(scan_dir.glob("*.json")):
        st = p.stat()
        rows.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _source_node_id(kind: str, value: str) -> str:
    return f"src:{kind}:{_stable_hash([kind, value])}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    idx = (len(arr) - 1) * q
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return arr[lo]
    t = idx - lo
    return arr[lo] * (1.0 - t) + arr[hi] * t


def _parse_linestring_coords(line_wkt: str) -> list[tuple[float, float]]:
    try:
        g = shapely_wkt.loads(line_wkt)
    except Exception:
        return []
    if g is None or g.is_empty or g.geom_type != "LineString":
        return []
    out = []
    for c in list(g.coords):
        if len(c) < 2:
            continue
        out.append((float(c[0]), float(c[1])))
    return out


def _story_wall_height_targets(nodes: dict[str, GraphNode]) -> dict[int, float]:
    by_story: dict[int, list[float]] = defaultdict(list)
    for n in nodes.values():
        if n.type != "Surface":
            continue
        if (n.properties or {}).get("surface_kind") != "wall":
            continue
        if not isinstance(n.story, int):
            continue
        dims = (n.properties or {}).get("dimensions_m", [])
        if not isinstance(dims, list) or len(dims) < 2:
            continue
        h = float(dims[1])
        if h > 0.1:
            by_story[n.story].append(h)
    return {s: max(2.2, _percentile(vals, 0.9)) for s, vals in by_story.items() if vals}


def _story_floor_y_map(
    nodes: dict[str, GraphNode], geometry_index: dict[str, Any]
) -> dict[int, float]:
    out: dict[int, list[float]] = defaultdict(list)
    transforms = geometry_index.get("transforms", {})
    polys = geometry_index.get("surface_polygons", {})
    for n in nodes.values():
        if n.type != "Surface":
            continue
        if (n.properties or {}).get("surface_kind") != "floor":
            continue
        if not isinstance(n.story, int):
            continue
        sid = n.id
        tf = transforms.get(n.transform_ref or "")
        poly = polys.get(sid)
        if (
            not isinstance(tf, list)
            or len(tf) != 16
            or not isinstance(poly, list)
            or len(poly) < 3
        ):
            continue
        ys = []
        for p in poly:
            if not isinstance(p, list) or len(p) < 3:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            wy = tf[1] * x + tf[5] * y + tf[9] * z + tf[13]
            ys.append(float(wy))
        if ys:
            out[n.story].append(sum(ys) / len(ys))
    return {s: _percentile(vals, 0.5) for s, vals in out.items() if vals}


def _augment_synthetic_surfaces(
    *,
    nodes: dict[str, GraphNode],
    edges: dict[str, GraphEdge],
    story_ids: dict[int, str],
    geometry_index: dict[str, Any],
    room_to_story: dict[str, int],
    adjacencies: list[IntraStoryAdjacency],
    gaps: list[GapRecord],
    pipeline_overlays: dict[str, Any] | None = None,
) -> None:
    """Inject extension and gap-closure surfaces before stitching."""
    wall_targets = _story_wall_height_targets(nodes)
    story_floor_y = _story_floor_y_map(nodes, geometry_index)

    room_contains_surface: dict[str, set[str]] = defaultdict(set)
    for e in edges.values():
        if (
            e.type == "CONTAINS"
            and str(e.from_id).startswith("room:")
            and str(e.to_id).startswith("surface:")
        ):
            room_contains_surface[e.from_id].add(e.to_id)

    def add_world_quad_surface(
        *,
        quad: list[list[float]],
        story: int,
        source_tag: str,
        synthetic_kind: str,
        surface_kind: str = "wall",
        ifc_class: str = "IfcWall",
        source_ids: list[str] | None = None,
        room_id: str | None = None,
        extra_properties: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(quad, list) or len(quad) < 3:
            return
        pts = []
        for p in quad:
            if not isinstance(p, list) or len(p) < 3:
                return
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        if len(pts) < 3:
            return

        tf_ref = "transform:identity"
        if tf_ref not in geometry_index["transforms"]:
            geometry_index["transforms"][tf_ref] = (
                np.eye(4, dtype=float).flatten(order="F").tolist()
            )

        p0 = np.array(pts[0], dtype=float)
        p1 = np.array(pts[1], dtype=float)
        p2 = np.array(pts[2], dtype=float) if len(pts) > 2 else p1
        p3 = np.array(pts[-1], dtype=float)
        e01 = float(np.linalg.norm(p1 - p0))
        e12 = float(np.linalg.norm(p2 - p1))
        e03 = float(np.linalg.norm(p3 - p0))

        hash_val = _stable_hash([str(story), json.dumps(pts)], 20)
        sid = f"surface:{synthetic_kind}:{hash_val}"
        if sid in nodes:
            return
        if surface_kind == "wall":
            dims = [max(e01, 0.01), max(e03, 0.01), 0.1]
        else:
            # For floor/ceiling-like quads, map as slab-like dimensions.
            dims = [max(e01, 0.01), 0.02, max(e12, e03, 0.01)]
        nodes[sid] = GraphNode(
            id=sid,
            type="Surface",
            story=int(story),
            source=source_tag,
            source_ids=list(source_ids or []),
            transform_ref=tf_ref,
            ifc_class=ifc_class,
            properties={
                "surface_kind": surface_kind,
                "synthetic_kind": synthetic_kind,
                "dimensions_m": dims,
                **(extra_properties or {}),
            },
        )
        geometry_index["surface_polygons"][sid] = [
            [round(p[0], 5), round(p[1], 5), round(p[2], 5)] for p in pts
        ]
        if story in story_ids:
            ce = GraphEdge(
                id=_edge_id("CONTAINS", story_ids[int(story)], sid, synthetic_kind),
                type="CONTAINS",
                from_id=story_ids[int(story)],
                to_id=sid,
                source=source_tag,
                ifc_relation="IfcRelContainedInSpatialStructure",
            )
            edges[ce.id] = ce
        if room_id and room_id in nodes:
            ce = GraphEdge(
                id=_edge_id("CONTAINS", room_id, sid, synthetic_kind),
                type="CONTAINS",
                from_id=room_id,
                to_id=sid,
                source=source_tag,
                ifc_relation="IfcRelContainedInSpatialStructure",
            )
            edges[ce.id] = ce

    # 1) Bring in reconciliation-pipeline geometry overlays first.
    pipeline_ext_count = 0
    if pipeline_overlays:
        for ext in pipeline_overlays.get("extension_strips", []):
            add_world_quad_surface(
                quad=ext["corners"],
                story=int(ext["story"]),
                source_tag="reconcile_extract_3d",
                synthetic_kind="pipeline_wall_extension_strip",
                surface_kind="wall",
                ifc_class="IfcWall",
                source_ids=[str(ext.get("wall_id", ""))],
                room_id=ext.get("room_id"),
            )
            pipeline_ext_count += 1

        for gc in pipeline_overlays.get("gap_closures", []):
            gc_type = str(gc.get("type", "unknown"))
            if gc_type == "side":
                sk = "wall"
                ifc = "IfcWall"
            elif gc_type == "floor":
                sk = "floor"
                ifc = "IfcSlab"
            elif gc_type == "ceiling":
                sk = "floor"
                ifc = "IfcCovering"
            else:
                sk = "wall"
                ifc = "IfcBuildingElementProxy"
            add_world_quad_surface(
                quad=gc["corners"],
                story=int(gc.get("story", 0)),
                source_tag="reconcile_extract_3d",
                synthetic_kind=f"pipeline_gap_closure_{gc_type}",
                surface_kind=sk,
                ifc_class=ifc,
                source_ids=[
                    str(gc.get("indicator_element_id", "")),
                    str(gc.get("indicator_wall_id", "")),
                ],
                extra_properties={"closure_role": gc_type},
            )
        for sw in pipeline_overlays.get("stitch_walls", []):
            add_world_quad_surface(
                quad=sw["corners"],
                story=int(sw.get("story", 0)),
                source_tag="reconcile_extract_3d",
                synthetic_kind="pipeline_stitch_wall",
                surface_kind="wall",
                ifc_class="IfcWall",
            )
        for gw in pipeline_overlays.get("gap_walls", []):
            gw_type = str(gw.get("type", "unknown"))
            add_world_quad_surface(
                quad=gw["corners"],
                story=int(gw.get("story", 0)),
                source_tag="reconcile_extract_3d",
                synthetic_kind=f"pipeline_gap_wall_{gw_type}",
                surface_kind="wall",
                ifc_class="IfcWall",
                extra_properties={"gap_wall_role": gw_type},
            )

    # 1b) Fallback wall-extension strips only when pipeline extensions are unavailable.
    if pipeline_ext_count == 0:
        for n in list(nodes.values()):
            if n.type != "Surface":
                continue
            if (n.properties or {}).get("surface_kind") != "wall":
                continue
            if (n.properties or {}).get("synthetic_kind"):
                continue
            story = n.story
            if not isinstance(story, int):
                continue
            target_h = wall_targets.get(story)
            dims = (n.properties or {}).get("dimensions_m", [])
            if target_h is None or not isinstance(dims, list) or len(dims) < 2:
                continue
            cur_h = float(dims[1])
            ext_h = target_h - cur_h
            if ext_h < 0.08 or ext_h > 0.8:
                continue
            poly = geometry_index["surface_polygons"].get(n.id)
            if not isinstance(poly, list) or len(poly) < 3:
                continue
            xs = [float(p[0]) for p in poly if isinstance(p, list) and len(p) >= 3]
            ys = [float(p[1]) for p in poly if isinstance(p, list) and len(p) >= 3]
            if not xs or not ys:
                continue
            min_x, max_x = min(xs), max(xs)
            top_y = max(ys)
            if (max_x - min_x) < 0.05:
                continue
            add_world_quad_surface(
                quad=[
                    [min_x, top_y, 0.0],
                    [max_x, top_y, 0.0],
                    [max_x, top_y + ext_h, 0.0],
                    [min_x, top_y + ext_h, 0.0],
                ],
                story=story,
                source_tag="derived_extension",
                synthetic_kind="wall_extension_strip",
                surface_kind="wall",
                ifc_class="IfcWall",
                source_ids=[n.id],
            )

    # 2) Intra-story wall-thickness gaps => synthetic wall closure surfaces.
    for a in adjacencies:
        story = int(a.story)
        line = _parse_linestring_coords(a.centerline_wkt)
        if len(line) < 2:
            continue
        x0, z0 = line[0]
        x1, z1 = line[-1]
        dx, dz = (x1 - x0), (z1 - z0)
        seg_len = float(np.hypot(dx, dz))
        if seg_len < 0.15:
            continue
        thickness_m = max(0.04, float(a.thickness_cm) / 100.0)
        if thickness_m > 0.9:
            continue

        wall_h = wall_targets.get(story, 2.4)
        base_y = story_floor_y.get(story, 0.0)
        mx, mz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
        ux, uz = dx / seg_len, dz / seg_len
        nx, nz = -uz, ux

        m = np.eye(4, dtype=float)
        m[0, 0], m[1, 0], m[2, 0] = ux, 0.0, uz
        m[0, 1], m[1, 1], m[2, 1] = 0.0, 1.0, 0.0
        m[0, 2], m[1, 2], m[2, 2] = nx, 0.0, nz
        m[0, 3], m[1, 3], m[2, 3] = mx, base_y + wall_h * 0.5, mz
        gap_hash = _stable_hash([str(story), a.room_a, a.room_b, a.centerline_wkt], 20)
        tf_ref = f"transform:gap:{gap_hash}"
        geometry_index["transforms"][tf_ref] = m.flatten(order="F").tolist()

        sid = f"surface:gapwall:{gap_hash}"
        if sid in nodes:
            continue
        nodes[sid] = GraphNode(
            id=sid,
            type="Surface",
            story=story,
            source="gap_closure",
            source_ids=[a.room_a, a.room_b],
            transform_ref=tf_ref,
            ifc_class="IfcWall",
            properties={
                "surface_kind": "wall",
                "synthetic_kind": "intra_story_gap_thickness_wall",
                "dimensions_m": [seg_len, wall_h, thickness_m],
                "thickness_cm": round(a.thickness_cm, 2),
                "centerline_wkt": a.centerline_wkt,
            },
        )
        geometry_index["surface_polygons"][sid] = [
            [-round(seg_len * 0.5, 5), -round(wall_h * 0.5, 5), 0.0],
            [round(seg_len * 0.5, 5), -round(wall_h * 0.5, 5), 0.0],
            [round(seg_len * 0.5, 5), round(wall_h * 0.5, 5), 0.0],
            [-round(seg_len * 0.5, 5), round(wall_h * 0.5, 5), 0.0],
        ]
        if story in story_ids:
            ce = GraphEdge(
                id=_edge_id("CONTAINS", story_ids[story], sid, "gap-wall"),
                type="CONTAINS",
                from_id=story_ids[story],
                to_id=sid,
                source="gap_closure",
                ifc_relation="IfcRelContainedInSpatialStructure",
            )
            edges[ce.id] = ce
        for rr in (a.room_a, a.room_b):
            rid = f"room:{rr}"
            if rid in nodes and room_to_story.get(rid) == story:
                ce = GraphEdge(
                    id=_edge_id("CONTAINS", rid, sid, rr),
                    type="CONTAINS",
                    from_id=rid,
                    to_id=sid,
                    source="gap_closure",
                    ifc_relation="IfcRelContainedInSpatialStructure",
                )
                edges[ce.id] = ce

    # 3) Cross-story polygon gaps => synthetic floor closure surfaces.
    for g in gaps:
        if g.kind != "cross_story" or not g.region_wkt:
            continue
        try:
            poly = shapely_wkt.loads(g.region_wkt)
        except Exception:
            continue
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            continue
        coords = [(float(x), float(z)) for x, z, *_ in list(poly.exterior.coords)]
        if len(coords) < 4:
            continue
        story = int(g.story)
        y = float(story_floor_y.get(story, 0.0))
        sid = f"surface:gapfloor:{_stable_hash([str(story), g.region_wkt], 20)}"
        if sid in nodes:
            continue
        tf_ref = f"transform:gapfloor:{_stable_hash([sid], 16)}"
        geometry_index["transforms"][tf_ref] = (
            np.eye(4, dtype=float).flatten(order="F").tolist()
        )
        nodes[sid] = GraphNode(
            id=sid,
            type="Surface",
            story=story,
            source="gap_closure",
            source_ids=[sid],
            transform_ref=tf_ref,
            ifc_class="IfcSlab",
            properties={
                "surface_kind": "floor",
                "synthetic_kind": "cross_story_gap_floor_closure",
                "dimensions_m": [0.0, 0.0, 0.0],
            },
        )
        geometry_index["surface_polygons"][sid] = [
            [round(x, 5), round(y, 5), round(z, 5)] for x, z in coords[:-1]
        ]
        if story in story_ids:
            ce = GraphEdge(
                id=_edge_id("CONTAINS", story_ids[story], sid, "gap-floor"),
                type="CONTAINS",
                from_id=story_ids[story],
                to_id=sid,
                source="gap_closure",
                ifc_relation="IfcRelContainedInSpatialStructure",
            )
            edges[ce.id] = ce
        if g.affected_room:
            rid = f"room:{g.affected_room}"
            if rid in nodes:
                ce = GraphEdge(
                    id=_edge_id("CONTAINS", rid, sid, "gap-floor"),
                    type="CONTAINS",
                    from_id=rid,
                    to_id=sid,
                    source="gap_closure",
                    ifc_relation="IfcRelContainedInSpatialStructure",
                )
                edges[ce.id] = ce


def _load_pipeline_overlays(
    merged_path: Path, scan_dir: Path, uuid: str
) -> dict[str, Any] | None:
    """Load wall extensions and gap closures from extract_3d."""
    run_uuid = uuid or merged_path.parent.name
    pipeline_root = merged_path.parent.parent
    scan_root = scan_dir.parent
    if not run_uuid:
        return None
    if not pipeline_root.exists() or not scan_root.exists():
        return None

    try:
        from reconcile.extract_3d import extract_building
    except Exception:
        return None

    try:
        result = extract_building(run_uuid, pipeline_root, scan_root)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None

    overlays: dict[str, Any] = {
        "extension_strips": [],
        "gap_closures": list(result.get("gap_closures") or []),
        "stitch_walls": list(result.get("stitch_walls") or []),
        "gap_walls": list(result.get("gap_walls") or []),
    }
    rooms = list(result.get("rooms") or [])
    for i, room in enumerate(rooms):
        rid = f"room:merged_room_{i}"
        story = int(room.get("story", 0))
        for w in list(room.get("walls_computed") or []):
            ext = w.get("extension_strip")
            if not ext:
                continue
            for strip in ext:
                if not isinstance(strip, list) or len(strip) < 3:
                    continue
                overlays["extension_strips"].append(
                    {
                        "corners": strip,
                        "story": story,
                        "room_id": rid,
                        "wall_id": w.get("id"),
                    }
                )
    return overlays


def build_topology_graph(
    merged_path: Path,
    scan_dir: Path,
    uuid: str = "",
    pipeline_version: str = "v2.0.0",
) -> TopologyGraph:
    """Build a property-graph artifact from merged and scan inputs."""
    merged_raw = json.loads(merged_path.read_text())
    building = load_merged(merged_path)
    raw_rooms = load_scan_cache(scan_dir)

    building_ref = uuid or merged_path.stem
    building_id = f"building:{_stable_hash([building_ref], 12)}"

    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    geometry_index: dict[str, Any] = {
        "transforms": {},
        "surface_polygons": {},
        "room_floor_polygons": {},
    }

    # Building node
    nodes[building_id] = GraphNode(
        id=building_id,
        type="Building",
        source="merged",
        source_ids=[str(merged_path)],
        ifc_class="IfcBuilding",
        properties={"uuid": uuid, "merged_version": building.version},
    )

    # Source nodes for provenance
    merged_source_id = _source_node_id("merged_file", str(merged_path.resolve()))
    nodes[merged_source_id] = GraphNode(
        id=merged_source_id,
        type="Source",
        source="system",
        source_ids=[str(merged_path.resolve())],
        properties={"kind": "merged_file", "sha256": _file_sha256(merged_path)},
    )
    e = GraphEdge(
        id=_edge_id("DERIVED_FROM", building_id, merged_source_id),
        type="DERIVED_FROM",
        from_id=building_id,
        to_id=merged_source_id,
        source="builder",
    )
    edges[e.id] = e

    scan_source_id = _source_node_id("scan_cache", str(scan_dir.resolve()))
    nodes[scan_source_id] = GraphNode(
        id=scan_source_id,
        type="Source",
        source="system",
        source_ids=[str(scan_dir.resolve())],
        properties={
            "kind": "scan_cache",
            "manifest_sha256": _scan_manifest_hash(scan_dir),
        },
    )
    e = GraphEdge(
        id=_edge_id("DERIVED_FROM", building_id, scan_source_id),
        type="DERIVED_FROM",
        from_id=building_id,
        to_id=scan_source_id,
        source="builder",
    )
    edges[e.id] = e

    story_ids: dict[int, str] = {}
    room_to_story: dict[str, int] = {}
    surface_to_rooms: dict[str, list[str]] = defaultdict(list)

    # Collect story values from rooms + top-level surfaces/floors.
    story_values = {r.story for r in building.rooms}
    for s in (
        building.top_level_walls + building.top_level_doors + building.top_level_windows
    ):
        story_values.add(s.story)
    for f in building.top_level_floors:
        story_values.add(f.story)

    for story in sorted(story_values):
        sid = f"story:{story}"
        story_ids[story] = sid
        nodes[sid] = GraphNode(
            id=sid,
            type="Story",
            story=story,
            source="merged",
            source_ids=[str(story)],
            ifc_class="IfcBuildingStorey",
            properties={"story_index": story},
        )
        ce = GraphEdge(
            id=_edge_id("CONTAINS", building_id, sid),
            type="CONTAINS",
            from_id=building_id,
            to_id=sid,
            source="builder",
            ifc_relation="IfcRelContainedInSpatialStructure",
        )
        edges[ce.id] = ce

    sorted_stories = sorted(story_ids)
    for i in range(len(sorted_stories) - 1):
        a = story_ids[sorted_stories[i]]
        b = story_ids[sorted_stories[i + 1]]
        fe = GraphEdge(
            id=_edge_id("FOLLOWS", a, b),
            type="FOLLOWS",
            from_id=a,
            to_id=b,
            source="builder",
            evidence={"delta_story": sorted_stories[i + 1] - sorted_stories[i]},
        )
        edges[fe.id] = fe

    # Build raw surface -> raw room mapping for DERIVED_FROM provenance
    raw_surface_to_room_ids: dict[str, set[str]] = defaultdict(set)
    for rr in raw_rooms:
        for s in rr.walls + rr.doors + rr.windows:
            raw_surface_to_room_ids[s.identifier].add(rr.identifier or "")

    # Room + surface nodes and boundaries
    room_iter = sorted(enumerate(building.rooms), key=lambda it: it[1].identifier or "")
    for room_i, room in room_iter:
        raw_room = (
            (merged_raw.get("rooms") or [])[room_i]
            if room_i < len(merged_raw.get("rooms") or [])
            else {}
        )
        rid = f"room:{room.identifier}"
        room_to_story[rid] = room.story
        bbox = _polygon_bbox_xz(room)

        room_node = GraphNode(
            id=rid,
            type="Room",
            story=room.story,
            confidence=1.0,
            source="merged",
            source_ids=[room.identifier or ""],
            bbox_xz=bbox,
            ifc_class="IfcSpace",
            legacy_refs={"merged_room_id": room.identifier or ""},
            properties={},
        )
        nodes[rid] = room_node

        if room.reference_origin_transform is not None:
            tx = room.reference_origin_transform.to_flat()
            t_ref = f"transform:{_stable_hash([rid, json.dumps(tx)])}"
            room_node.transform_ref = t_ref
            geometry_index["transforms"][t_ref] = tx

        re = GraphEdge(
            id=_edge_id("CONTAINS", story_ids[room.story], rid),
            type="CONTAINS",
            from_id=story_ids[room.story],
            to_id=rid,
            source="builder",
            ifc_relation="IfcRelContainedInSpatialStructure",
        )
        edges[re.id] = re

        poly = room_floor_polygon(room)
        if poly is not None and not poly.is_empty:
            geometry_index["room_floor_polygons"][rid] = poly.wkt

        for kind, collection in (
            ("wall", room.walls),
            ("door", room.doors),
            ("window", room.windows),
        ):
            for s in sorted(collection, key=lambda x: x.identifier):
                sid = f"surface:{s.identifier}"
                if sid not in nodes:
                    s_kind = _surface_kind(s, kind)
                    local_polygon = _surface_local_polygon(s, s_kind)
                    s_node = GraphNode(
                        id=sid,
                        type="Surface",
                        story=s.story,
                        source="merged",
                        source_ids=[s.identifier],
                        transform_ref=f"transform:{_stable_hash([sid])}",
                        bbox_xz=_surface_bbox_xz(s, s_kind),
                        ifc_class=_ifc_class_for_surface(s_kind),
                        legacy_refs={"merged_surface_id": s.identifier},
                        properties={
                            "surface_kind": s_kind,
                            "dimensions_m": [
                                float(s.dimensions.x),
                                float(s.dimensions.y),
                                float(s.dimensions.z),
                            ],
                        },
                    )
                    nodes[sid] = s_node
                    geometry_index["transforms"][s_node.transform_ref] = (
                        s.transform.to_flat()
                    )
                    geometry_index["surface_polygons"][sid] = [
                        [round(c[0], 5), round(c[1], 5), round(c[2], 5)]
                        for c in local_polygon
                    ]

                surface_to_rooms[sid].append(rid)

                ce = GraphEdge(
                    id=_edge_id("CONTAINS", rid, sid),
                    type="CONTAINS",
                    from_id=rid,
                    to_id=sid,
                    source="builder",
                    ifc_relation="IfcRelContainedInSpatialStructure",
                )
                edges[ce.id] = ce

                boundary_id = f"boundary:{_stable_hash([rid, sid], 20)}"
                nodes[boundary_id] = GraphNode(
                    id=boundary_id,
                    type="Boundary",
                    story=room.story,
                    source="derived",
                    source_ids=[room.identifier or "", s.identifier],
                    confidence=1.0,
                    ifc_class="IfcRelSpaceBoundary",
                    legacy_refs={
                        "room": room.identifier or "",
                        "surface": s.identifier,
                    },
                    properties={"boundary_level": "L1"},
                )

                # Room contains boundary
                be_contains = GraphEdge(
                    id=_edge_id("CONTAINS", rid, boundary_id),
                    type="CONTAINS",
                    from_id=rid,
                    to_id=boundary_id,
                    source="builder",
                )
                edges[be_contains.id] = be_contains

                # Boundary references room and boundary element surface
                be_room = GraphEdge(
                    id=_edge_id("BOUNDS", boundary_id, rid, "room"),
                    type="BOUNDS",
                    from_id=boundary_id,
                    to_id=rid,
                    source="builder",
                    ifc_relation="IfcRelSpaceBoundary",
                )
                edges[be_room.id] = be_room

                be_surface = GraphEdge(
                    id=_edge_id("BOUNDS", boundary_id, sid, "surface"),
                    type="BOUNDS",
                    from_id=boundary_id,
                    to_id=sid,
                    source="builder",
                    ifc_relation="IfcRelSpaceBoundary",
                )
                edges[be_surface.id] = be_surface

                # Provenance: surface -> raw rooms containing same surface UUID.
                raw_room_ids = sorted(raw_surface_to_room_ids.get(s.identifier, set()))
                for rrid in raw_room_ids:
                    src_id = _source_node_id("scan_room", rrid)
                    if src_id not in nodes:
                        nodes[src_id] = GraphNode(
                            id=src_id,
                            type="Source",
                            source="scan-cache",
                            source_ids=[rrid],
                            properties={"kind": "scan_room", "room_id": rrid},
                        )
                    de = GraphEdge(
                        id=_edge_id("DERIVED_FROM", sid, src_id),
                        type="DERIVED_FROM",
                        from_id=sid,
                        to_id=src_id,
                        source="builder",
                        evidence={"via_surface_identifier": s.identifier},
                    )
                    edges[de.id] = de

        # Openings/storages from full merged model
        # (not currently represented by loader models).
        raw_openings = raw_room.get("openings") or []
        raw_objects = raw_room.get("objects") or []
        raw_storages = [
            o
            for o in raw_objects
            if isinstance(o, dict)
            and isinstance(o.get("category"), dict)
            and "storage" in o.get("category", {})
        ]
        for kind, collection in (("opening", raw_openings), ("storage", raw_storages)):
            for raw_s in sorted(collection, key=lambda x: str(x.get("identifier", ""))):
                sid_raw = str(raw_s.get("identifier", ""))
                if not sid_raw:
                    continue
                sid = f"surface:{sid_raw}"
                tf = raw_s.get("transform")
                if not isinstance(tf, list) or len(tf) != 16:
                    continue
                local_polygon = _raw_surface_local_polygon(raw_s, kind)
                if len(local_polygon) < 3:
                    continue
                world = _raw_apply_transform(local_polygon, tf)
                xs = [p[0] for p in world] if world else []
                zs = [p[2] for p in world] if world else []
                bbox_xz = (
                    [
                        round(min(xs), 4),
                        round(min(zs), 4),
                        round(max(xs), 4),
                        round(max(zs), 4),
                    ]
                    if xs and zs
                    else None
                )
                dims = raw_s.get("dimensions") or [0.1, 0.1, 0.1]
                if sid not in nodes:
                    nodes[sid] = GraphNode(
                        id=sid,
                        type="Surface",
                        story=int(raw_s.get("story", room.story)),
                        source="merged",
                        source_ids=[sid_raw],
                        transform_ref=f"transform:{_stable_hash([sid])}",
                        bbox_xz=bbox_xz,
                        ifc_class=_ifc_class_for_surface(kind),
                        legacy_refs={f"merged_{kind}_id": sid_raw},
                        properties={
                            "surface_kind": kind,
                            "dimensions_m": [
                                float(dims[0]),
                                float(dims[1]),
                                float(dims[2]),
                            ],
                        },
                    )
                    geometry_index["transforms"][f"transform:{_stable_hash([sid])}"] = [
                        float(v) for v in tf
                    ]
                    geometry_index["surface_polygons"][sid] = [
                        [round(c[0], 5), round(c[1], 5), round(c[2], 5)]
                        for c in local_polygon
                    ]
                ce = GraphEdge(
                    id=_edge_id("CONTAINS", rid, sid, kind),
                    type="CONTAINS",
                    from_id=rid,
                    to_id=sid,
                    source="builder",
                    ifc_relation="IfcRelContainedInSpatialStructure",
                )
                edges[ce.id] = ce

        # Floors as surfaces in V2
        for f in sorted(room.floors, key=lambda x: x.identifier):
            sid = f"surface:{f.identifier}"
            if sid not in nodes:
                local_polygon = _floor_local_polygon(f)
                floor_bbox = None
                if local_polygon:
                    world = _apply_transform_to_local_polygon(
                        local_polygon, f.transform
                    )
                    xs = [p[0] for p in world]
                    zs = [p[2] for p in world]
                    floor_bbox = [
                        round(min(xs), 4),
                        round(min(zs), 4),
                        round(max(xs), 4),
                        round(max(zs), 4),
                    ]
                s_node = GraphNode(
                    id=sid,
                    type="Surface",
                    story=f.story,
                    source="merged",
                    source_ids=[f.identifier],
                    transform_ref=f"transform:{_stable_hash([sid])}",
                    bbox_xz=floor_bbox,
                    ifc_class=_ifc_class_for_surface("floor"),
                    legacy_refs={"merged_floor_id": f.identifier},
                    properties={
                        "surface_kind": "floor",
                        "dimensions_m": [
                            float(f.dimensions.x),
                            float(f.dimensions.y),
                            float(f.dimensions.z),
                        ],
                    },
                )
                nodes[sid] = s_node
                geometry_index["transforms"][s_node.transform_ref] = (
                    f.transform.to_flat()
                )
                geometry_index["surface_polygons"][sid] = [
                    [round(c[0], 5), round(c[1], 5), round(c[2], 5)]
                    for c in local_polygon
                ]
            surface_to_rooms[sid].append(rid)

            ce = GraphEdge(
                id=_edge_id("CONTAINS", rid, sid),
                type="CONTAINS",
                from_id=rid,
                to_id=sid,
                source="builder",
                ifc_relation="IfcRelContainedInSpatialStructure",
            )
            edges[ce.id] = ce

    # Explicit top-level surfaces that may not be assigned to rooms.
    for kind, collection in (
        ("wall", building.top_level_walls),
        ("door", building.top_level_doors),
        ("window", building.top_level_windows),
    ):
        for s in sorted(collection, key=lambda x: x.identifier):
            sid = f"surface:{s.identifier}"
            if sid not in nodes:
                s_kind = _surface_kind(s, kind)
                local_polygon = _surface_local_polygon(s, s_kind)
                nodes[sid] = GraphNode(
                    id=sid,
                    type="Surface",
                    story=s.story,
                    source="merged_top_level",
                    source_ids=[s.identifier],
                    transform_ref=f"transform:{_stable_hash([sid])}",
                    bbox_xz=_surface_bbox_xz(s, s_kind),
                    ifc_class=_ifc_class_for_surface(s_kind),
                    legacy_refs={"merged_surface_id": s.identifier},
                    properties={
                        "surface_kind": s_kind,
                        "top_level": True,
                        "dimensions_m": [
                            float(s.dimensions.x),
                            float(s.dimensions.y),
                            float(s.dimensions.z),
                        ],
                    },
                )
                geometry_index["transforms"][f"transform:{_stable_hash([sid])}"] = (
                    s.transform.to_flat()
                )
                geometry_index["surface_polygons"][sid] = [
                    [round(c[0], 5), round(c[1], 5), round(c[2], 5)]
                    for c in local_polygon
                ]

            ce = GraphEdge(
                id=_edge_id("CONTAINS", story_ids[s.story], sid, "top-level"),
                type="CONTAINS",
                from_id=story_ids[s.story],
                to_id=sid,
                source="builder",
                ifc_relation="IfcRelContainedInSpatialStructure",
            )
            edges[ce.id] = ce

    # Top-level openings and storages from merged full model.
    top_level_openings = merged_raw.get("openings") or []
    top_level_objects = merged_raw.get("objects") or []
    top_level_storages = [
        o
        for o in top_level_objects
        if isinstance(o, dict)
        and isinstance(o.get("category"), dict)
        and "storage" in o.get("category", {})
    ]
    for kind, collection in (
        ("opening", top_level_openings),
        ("storage", top_level_storages),
    ):
        for raw_s in sorted(collection, key=lambda x: str(x.get("identifier", ""))):
            sid_raw = str(raw_s.get("identifier", ""))
            if not sid_raw:
                continue
            sid = f"surface:{sid_raw}"
            tf = raw_s.get("transform")
            if not isinstance(tf, list) or len(tf) != 16:
                continue
            story = int(raw_s.get("story", 0))
            local_polygon = _raw_surface_local_polygon(raw_s, kind)
            if len(local_polygon) < 3:
                continue
            world = _raw_apply_transform(local_polygon, tf)
            xs = [p[0] for p in world] if world else []
            zs = [p[2] for p in world] if world else []
            bbox_xz = (
                [
                    round(min(xs), 4),
                    round(min(zs), 4),
                    round(max(xs), 4),
                    round(max(zs), 4),
                ]
                if xs and zs
                else None
            )
            dims = raw_s.get("dimensions") or [0.1, 0.1, 0.1]
            if sid not in nodes:
                nodes[sid] = GraphNode(
                    id=sid,
                    type="Surface",
                    story=story,
                    source="merged_top_level",
                    source_ids=[sid_raw],
                    transform_ref=f"transform:{_stable_hash([sid])}",
                    bbox_xz=bbox_xz,
                    ifc_class=_ifc_class_for_surface(kind),
                    legacy_refs={f"merged_{kind}_id": sid_raw},
                    properties={
                        "surface_kind": kind,
                        "top_level": True,
                        "dimensions_m": [
                            float(dims[0]),
                            float(dims[1]),
                            float(dims[2]),
                        ],
                    },
                )
                geometry_index["transforms"][f"transform:{_stable_hash([sid])}"] = [
                    float(v) for v in tf
                ]
                geometry_index["surface_polygons"][sid] = [
                    [round(c[0], 5), round(c[1], 5), round(c[2], 5)]
                    for c in local_polygon
                ]
            if story in story_ids:
                ce = GraphEdge(
                    id=_edge_id("CONTAINS", story_ids[story], sid, f"top-level-{kind}"),
                    type="CONTAINS",
                    from_id=story_ids[story],
                    to_id=sid,
                    source="builder",
                    ifc_relation="IfcRelContainedInSpatialStructure",
                )
                edges[ce.id] = ce

    # Compute topology evidence once so it can feed synthetic
    # closure geometry and graph edges.
    adjacencies = infer_intra_story_adjacency(building)
    gaps, story_footprints = infer_gap_records(building)
    pipeline_overlays = _load_pipeline_overlays(merged_path, scan_dir, uuid)

    # Inject synthetic extension + gap-closure surfaces before stitching.
    _augment_synthetic_surfaces(
        nodes=nodes,
        edges=edges,
        story_ids=story_ids,
        geometry_index=geometry_index,
        room_to_story=room_to_story,
        adjacencies=adjacencies,
        gaps=gaps,
        pipeline_overlays=pipeline_overlays,
    )

    # Adjacency from floor-gap evidence.
    for a in adjacencies:
        room_a = f"room:{a.room_a}"
        room_b = f"room:{a.room_b}"
        if room_a not in nodes or room_b not in nodes:
            continue

        ae = GraphEdge(
            id=_edge_id("ADJACENT_TO", room_a, room_b),
            type="ADJACENT_TO",
            from_id=room_a,
            to_id=room_b,
            source="floor_gaps",
            confidence=a.confidence_score,
            evidence={
                "method": "wall_thickness_inference",
                "story": a.story,
                "thickness_cm": round(a.thickness_cm, 2),
                "thickness_p05_cm": round(a.thickness_p05_cm, 2),
                "thickness_p95_cm": round(a.thickness_p95_cm, 2),
                "thickness_std_cm": round(a.thickness_std_cm, 2),
                "overlap_ratio": round(a.overlap_ratio, 3),
                "angle_delta_deg": round(a.angle_delta_deg, 2),
                "opening_proximity_count": a.opening_proximity_count,
                "centerline_wkt": a.centerline_wkt,
                "confidence_label": a.confidence,
                "support_ratio": round(a.support_ratio, 6),
                "relation_state": a.relation_state,
                "floor_delta_m": round(a.floor_delta_m, 6),
            },
        )
        edges[ae.id] = ae

    # Gap nodes and HAS_GAP edges.
    for idx, g in enumerate(gaps):
        gid = (
            f"gap:{g.kind}:{_stable_hash([str(idx), str(g.story), g.region_wkt or ''])}"
        )
        nodes[gid] = GraphNode(
            id=gid,
            type="Gap",
            story=g.story,
            confidence=g.confidence_score,
            source="derived",
            source_ids=[str(idx)],
            ifc_class="IfcOpeningElement" if g.kind == "cross_story" else None,
            properties={
                "gap_kind": g.kind,
                "confidence_label": g.confidence,
                "reference_story": g.reference_story,
                "region_wkt": g.region_wkt,
                "relation_state": g.relation_state,
            },
            metrics={
                "area_m2": g.area_m2,
                "thickness_cm": g.thickness_cm,
                "support_ratio": g.support_ratio,
                **g.metrics,
            },
        )

        # Story owns the gap
        sid = story_ids.get(g.story)
        if sid:
            he = GraphEdge(
                id=_edge_id("HAS_GAP", sid, gid),
                type="HAS_GAP",
                from_id=sid,
                to_id=gid,
                source="gap_detection",
                confidence=g.confidence_score,
            )
            edges[he.id] = he

        if g.affected_room:
            rid = f"room:{g.affected_room}"
            if rid in nodes:
                he = GraphEdge(
                    id=_edge_id("HAS_GAP", rid, gid, "affected"),
                    type="HAS_GAP",
                    from_id=rid,
                    to_id=gid,
                    source="gap_detection",
                    confidence=g.confidence_score,
                )
                edges[he.id] = he

        # Intra-story link to both rooms when available.
        if g.kind == "intra_story":
            ra = g.metrics.get("room_a")
            rb = g.metrics.get("room_b")
            for rr in (ra, rb):
                if rr:
                    rid = f"room:{rr}"
                    if rid in nodes:
                        he = GraphEdge(
                            id=_edge_id("HAS_GAP", rid, gid, rr),
                            type="HAS_GAP",
                            from_id=rid,
                            to_id=gid,
                            source="floor_gaps",
                            confidence=g.confidence_score,
                        )
                        edges[he.id] = he

    metadata = {
        "uuid": uuid,
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline_version": pipeline_version,
        "source_hashes": {
            "merged_sha256": _file_sha256(merged_path),
            "scan_manifest_sha256": _scan_manifest_hash(scan_dir),
        },
    }

    quality = {
        "story_footprints": story_footprints,
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "rooms": sum(1 for n in nodes.values() if n.type == "Room"),
            "stories": sum(1 for n in nodes.values() if n.type == "Story"),
            "surfaces": sum(1 for n in nodes.values() if n.type == "Surface"),
            "gaps": sum(1 for n in nodes.values() if n.type == "Gap"),
            "boundaries": sum(1 for n in nodes.values() if n.type == "Boundary"),
        },
        "synthetic_surfaces": {
            "wall_extension_strip": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and (n.properties or {}).get("synthetic_kind") == "wall_extension_strip"
            ),
            "pipeline_wall_extension_strip": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and (n.properties or {}).get("synthetic_kind")
                == "pipeline_wall_extension_strip"
            ),
            "intra_story_gap_thickness_wall": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and (n.properties or {}).get("synthetic_kind")
                == "intra_story_gap_thickness_wall"
            ),
            "cross_story_gap_floor_closure": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and (n.properties or {}).get("synthetic_kind")
                == "cross_story_gap_floor_closure"
            ),
            "pipeline_gap_closure": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and str((n.properties or {}).get("synthetic_kind", "")).startswith(
                    "pipeline_gap_closure_"
                )
            ),
            "pipeline_stitch_wall": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and (n.properties or {}).get("synthetic_kind") == "pipeline_stitch_wall"
            ),
            "pipeline_gap_wall": sum(
                1
                for n in nodes.values()
                if n.type == "Surface"
                and str((n.properties or {}).get("synthetic_kind", "")).startswith(
                    "pipeline_gap_wall_"
                )
            ),
        },
    }

    stitched = build_stitched_geometry(
        nodes=[
            {
                "id": n.id,
                "type": n.type,
                "story": n.story,
                "properties": n.properties,
                "transform_ref": n.transform_ref,
            }
            for n in nodes.values()
        ],
        geometry_index=geometry_index,
    )
    geometry_index["stitched_geometry"] = stitched
    quality["stitched_geometry"] = stitched.get("stats", {})

    graph = TopologyGraph(
        version="2.0.0",
        metadata=metadata,
        nodes=sorted(nodes.values(), key=lambda n: n.id),
        edges=sorted(edges.values(), key=lambda e: e.id),
        geometry_index=geometry_index,
        quality=quality,
    )

    return graph
