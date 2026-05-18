"""Portable exact-on-lattice 3D cell-complex contract and helpers.

The kernel is defined on a canonical fixed-point millimetre lattice. All
topology predicates operate on snapped integer coordinates, portable across
Swift/Go/C++/Python. The builder emits exact polyhedral cells from halfspace
arrangements via polyhedral_kernel.build_arranged_polyhedral_cell().
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, triangulate, unary_union
from shapely.validation import make_valid

from .polyhedral_kernel import build_arranged_polyhedral_cell

EPS = 1e-6
LATTICE_SCALE_MM = 1000


def _stable_hash(parts: list[str], length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:length]


def _round3(value: float) -> float:
    return round(float(value), 6)


def _snap_to_lattice(value: float) -> int:
    return round(float(value) * LATTICE_SCALE_MM)


def _from_lattice(value: int) -> float:
    return float(Fraction(value, LATTICE_SCALE_MM))


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [_round3(self.x), _round3(self.y), _round3(self.z)]

    def as_lattice(self) -> list[int]:
        return [
            _snap_to_lattice(self.x),
            _snap_to_lattice(self.y),
            _snap_to_lattice(self.z),
        ]


@dataclass(frozen=True)
class Face3D:
    id: str
    boundary_kind: str
    vertices: tuple[Vec3, ...]
    normal: Vec3
    area: float
    plane: tuple[float, float, float, float]
    role: str | None = None
    source_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "boundary_kind": self.boundary_kind,
            "vertices": [v.as_list() for v in self.vertices],
            "vertices_lattice": [v.as_lattice() for v in self.vertices],
            "normal": self.normal.as_list(),
            "area": _round3(self.area),
            "plane": [_round3(v) for v in self.plane],
            "role": self.role,
            "source_kind": self.source_kind,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Cell3D:
    id: str
    kind: str
    story: int | None
    source_id: str | None
    centroid: Vec3
    volume_m3: float
    bbox_xyz: tuple[float, float, float, float, float, float]
    faces: tuple[Face3D, ...]
    properties: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "story": self.story,
            "source_id": self.source_id,
            "centroid": self.centroid.as_list(),
            "volume_m3": _round3(self.volume_m3),
            "bbox_xyz": [_round3(v) for v in self.bbox_xyz],
            "bbox_lattice": [_snap_to_lattice(v) for v in self.bbox_xyz],
            "faces": [f.as_dict() for f in self.faces],
            "properties": self.properties,
        }


@dataclass(frozen=True)
class CellAdjacency:
    from_cell_id: str
    to_cell_id: str
    relation: str
    shared_face_area: float
    axis: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_cell_id": self.from_cell_id,
            "to_cell_id": self.to_cell_id,
            "relation": self.relation,
            "shared_face_area": _round3(self.shared_face_area),
            "axis": self.axis,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CellComplex:
    cells: tuple[Cell3D, ...]
    adjacency: tuple[CellAdjacency, ...]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [c.as_dict() for c in self.cells],
            "adjacency": [a.as_dict() for a in self.adjacency],
            "metadata": self.metadata,
        }


def _plane_from_points(a: Vec3, b: Vec3, c: Vec3) -> tuple[float, float, float, float]:
    ax, ay, az = a.as_lattice()
    bx, by, bz = b.as_lattice()
    cx, cy, cz = c.as_lattice()
    ab = (bx - ax, by - ay, bz - az)
    ac = (cx - ax, cy - ay, cz - az)
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    norm_sq = nx * nx + ny * ny + nz * nz
    if norm_sq == 0:
        return (0.0, 0.0, 0.0, 0.0)
    norm = math.sqrt(float(norm_sq))
    nx /= norm
    ny /= norm
    nz /= norm
    d = -(nx * a.x + ny * a.y + nz * a.z)
    return (_round3(nx), _round3(ny), _round3(nz), _round3(d))


def _polygon_area_xz(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, z1 = points[i]
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return abs(area) * 0.5


def _snap_poly_coords(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    snapped: list[tuple[float, float]] = []
    for x, z in points:
        sx = _from_lattice(_snap_to_lattice(x))
        sz = _from_lattice(_snap_to_lattice(z))
        if (
            snapped
            and abs(snapped[-1][0] - sx) <= EPS
            and abs(snapped[-1][1] - sz) <= EPS
        ):
            continue
        snapped.append((sx, sz))
    if (
        len(snapped) >= 2
        and abs(snapped[0][0] - snapped[-1][0]) <= EPS
        and abs(snapped[0][1] - snapped[-1][1]) <= EPS
    ):
        snapped.pop()
    return snapped


def _build_face(cell_id: str, face_kind: str, vertices: list[Vec3]) -> Face3D:
    if len(vertices) < 3:
        raise ValueError("face requires at least three vertices")
    plane = _plane_from_points(vertices[0], vertices[1], vertices[2])
    area = 0.0
    if abs(plane[1]) > 1.0 - 1e-3:
        area = _polygon_area_xz(
            [
                (
                    _from_lattice(_snap_to_lattice(v.x)),
                    _from_lattice(_snap_to_lattice(v.z)),
                )
                for v in vertices
            ]
        )
    else:
        a, b, c, d = vertices[0], vertices[1], vertices[2], vertices[3]
        u = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
        v = math.dist((b.x, b.y, b.z), (c.x, c.y, c.z))
        area = u * v
        if area <= EPS:
            area = math.dist((a.x, a.y, a.z), (d.x, d.y, d.z)) * u
    normal = Vec3(plane[0], plane[1], plane[2])
    fid = f"face:{
        _stable_hash(
            [cell_id, face_kind, str([v.as_list() for v in vertices])],
            20,
        )
    }"
    return Face3D(
        id=fid,
        boundary_kind=face_kind,
        vertices=tuple(vertices),
        normal=normal,
        area=area,
        plane=plane,
    )


def _convex_subpolygons(polygon: Polygon) -> list[Polygon]:
    if polygon.is_empty or polygon.area <= EPS:
        return []
    try:
        hull = polygon.convex_hull
        if abs(float(hull.area) - float(polygon.area)) <= EPS:
            return [polygon]
    except Exception:
        return [polygon]
    out: list[Polygon] = []
    for triangle in triangulate(polygon):
        try:
            clipped = polygon.intersection(triangle)
        except Exception:
            continue
        if clipped.is_empty:
            continue
        geoms = (
            [clipped]
            if isinstance(clipped, Polygon)
            else list(getattr(clipped, "geoms", []))
        )
        for geom in geoms:
            if not isinstance(geom, Polygon) or geom.is_empty or geom.area <= EPS:
                continue
            out.append(geom)
    return out or [polygon]


def _face_from_arranged_record(face: dict[str, Any]) -> Face3D | None:
    corners = face.get("corners") or []
    vertices = [
        Vec3(float(c[0]), float(c[1]), float(c[2]))
        for c in corners
        if isinstance(c, list) and len(c) >= 3
    ]
    if len(vertices) < 3:
        return None
    plane_raw = face.get("plane_coeffs")
    if isinstance(plane_raw, list) and len(plane_raw) == 4:
        plane = tuple(float(v) for v in plane_raw)
    else:
        plane = _plane_from_points(vertices[0], vertices[1], vertices[2])
    normal = Vec3(plane[0], plane[1], plane[2])
    return Face3D(
        id=str(face.get("id") or ""),
        boundary_kind=str(face.get("kind") or "face"),
        vertices=tuple(vertices),
        normal=normal,
        area=float(face.get("area_m2", 0.0) or 0.0),
        plane=plane,
        role=str(face.get("role")) if face.get("role") is not None else None,
        source_kind=str(face.get("source_kind"))
        if face.get("source_kind") is not None
        else None,
        metadata=dict(face.get("metadata") or {}),
    )


def _story_heights(graph) -> tuple[dict[int, float], dict[int, float]]:
    floor_map: dict[int, list[float]] = {}
    ceiling_map: dict[int, list[float]] = {}
    transforms = (graph.geometry_index or {}).get("transforms", {})
    surface_polygons = (graph.geometry_index or {}).get("surface_polygons", {})

    for node in graph.nodes:
        if node.type != "Surface" or not isinstance(node.story, int):
            continue
        kind = (node.properties or {}).get("surface_kind")
        tf = transforms.get(node.transform_ref or "")
        poly = surface_polygons.get(node.id)
        if not isinstance(tf, list) or len(tf) != 16 or not isinstance(poly, list):
            continue

        ys: list[float] = []
        for p in poly:
            if not isinstance(p, list) or len(p) < 3:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            wy = tf[1] * x + tf[5] * y + tf[9] * z + tf[13]
            ys.append(wy)
        if not ys:
            continue

        if kind == "floor":
            floor_map.setdefault(node.story, []).append(sum(ys) / len(ys))
        elif kind == "wall":
            ceiling_map.setdefault(node.story, []).append(max(ys))

    sorted_stories = sorted({*floor_map.keys(), *ceiling_map.keys()})
    floor_y = {story: min(vals) for story, vals in floor_map.items() if vals}
    top_y: dict[int, float] = {}
    for story in sorted_stories:
        candidates = ceiling_map.get(story, [])
        if candidates:
            top_y[story] = max(candidates)
    for idx, story in enumerate(sorted_stories[:-1]):
        next_story = sorted_stories[idx + 1]
        if story in floor_y and next_story in floor_y:
            top_y[story] = min(
                top_y.get(story, floor_y[next_story]), floor_y[next_story]
            )
    return floor_y, top_y


def _box_faces(
    *,
    cell_id: str,
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    boundary_kind: str,
) -> tuple[Face3D, ...]:
    bottom = [
        Vec3(min_x, min_y, min_z),
        Vec3(max_x, min_y, min_z),
        Vec3(max_x, min_y, max_z),
        Vec3(min_x, min_y, max_z),
    ]
    top = [
        Vec3(min_x, max_y, min_z),
        Vec3(min_x, max_y, max_z),
        Vec3(max_x, max_y, max_z),
        Vec3(max_x, max_y, min_z),
    ]
    west = [
        Vec3(min_x, min_y, min_z),
        Vec3(min_x, min_y, max_z),
        Vec3(min_x, max_y, max_z),
        Vec3(min_x, max_y, min_z),
    ]
    east = [
        Vec3(max_x, min_y, min_z),
        Vec3(max_x, max_y, min_z),
        Vec3(max_x, max_y, max_z),
        Vec3(max_x, min_y, max_z),
    ]
    north = [
        Vec3(min_x, min_y, min_z),
        Vec3(min_x, max_y, min_z),
        Vec3(max_x, max_y, min_z),
        Vec3(max_x, min_y, min_z),
    ]
    south = [
        Vec3(min_x, min_y, max_z),
        Vec3(max_x, min_y, max_z),
        Vec3(max_x, max_y, max_z),
        Vec3(min_x, max_y, max_z),
    ]
    return (
        _build_face(cell_id, boundary_kind, bottom),
        _build_face(cell_id, boundary_kind, top),
        _build_face(cell_id, boundary_kind, west),
        _build_face(cell_id, boundary_kind, east),
        _build_face(cell_id, boundary_kind, north),
        _build_face(cell_id, boundary_kind, south),
    )


def _prism_faces(
    *,
    cell_id: str,
    footprint: list[tuple[float, float]],
    floor_y: float,
    ceiling_y: float,
    boundary_kind: str,
) -> tuple[Face3D, ...]:
    base = [Vec3(x, floor_y, z) for x, z in footprint]
    top = [Vec3(x, ceiling_y, z) for x, z in footprint]
    faces = [
        _build_face(cell_id, boundary_kind, base),
        _build_face(cell_id, boundary_kind, list(reversed(top))),
    ]
    for i in range(len(footprint)):
        x0, z0 = footprint[i]
        x1, z1 = footprint[(i + 1) % len(footprint)]
        quad = [
            Vec3(x0, floor_y, z0),
            Vec3(x1, floor_y, z1),
            Vec3(x1, ceiling_y, z1),
            Vec3(x0, ceiling_y, z0),
        ]
        edge_len = math.dist((x0, z0), (x1, z1))
        if edge_len <= EPS:
            continue
        faces.append(_build_face(cell_id, boundary_kind, quad))
    return tuple(faces)


def _room_footprint(graph, node) -> list[tuple[float, float]] | None:
    room_floor_polygons = (graph.geometry_index or {}).get("room_floor_polygons", {})
    room_wkt = room_floor_polygons.get(node.id)
    if isinstance(room_wkt, str) and room_wkt:
        try:
            poly = shapely_wkt.loads(room_wkt)
        except Exception:
            poly = None
        if isinstance(poly, Polygon):
            try:
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
            except Exception:
                poly = None
        if isinstance(poly, Polygon) and not poly.is_empty and poly.area > EPS:
            coords = list(poly.exterior.coords)
            if coords and coords[-1] == coords[0]:
                coords = coords[:-1]
            snapped = _snap_poly_coords([(float(x), float(z)) for x, z, *_ in coords])
            if len(snapped) >= 3:
                return snapped
    bbox = node.bbox_xz
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    min_x, min_z, max_x, max_z = [
        _from_lattice(_snap_to_lattice(float(v))) for v in bbox
    ]
    return [(min_x, min_z), (max_x, min_z), (max_x, max_z), (min_x, max_z)]


def _story_room_polygons(graph) -> dict[int, list[tuple[Any, Polygon]]]:
    out: dict[int, list[tuple[Any, Polygon]]] = {}
    for node in graph.nodes:
        if node.type != "Room" or not isinstance(node.story, int):
            continue
        footprint = _room_footprint(graph, node)
        if not footprint or len(footprint) < 3:
            continue
        poly = Polygon(footprint)
        if not poly.is_valid:
            try:
                poly = make_valid(poly)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
            except Exception:
                continue
        if poly.is_empty or poly.area <= EPS:
            continue
        out.setdefault(node.story, []).append((node, poly))
    return out


def _story_gap_polygons(graph) -> dict[int, list[tuple[Any, Polygon]]]:
    out: dict[int, list[tuple[Any, Polygon]]] = {}
    for node in graph.nodes:
        if node.type != "Gap" or not isinstance(node.story, int):
            continue
        region_wkt = (node.properties or {}).get("region_wkt")
        if not isinstance(region_wkt, str) or not region_wkt:
            continue
        try:
            poly = shapely_wkt.loads(region_wkt)
        except Exception:
            continue
        if not isinstance(poly, Polygon):
            continue
        if not poly.is_valid:
            try:
                poly = make_valid(poly)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
            except Exception:
                continue
        if poly.is_empty or poly.area <= EPS:
            continue
        coords = list(poly.exterior.coords)
        if coords and coords[-1] == coords[0]:
            coords = coords[:-1]
        snapped = _snap_poly_coords([(float(x), float(z)) for x, z, *_ in coords])
        if len(snapped) < 3:
            continue
        out.setdefault(node.story, []).append((node, Polygon(snapped)))
    return out


def _cell_from_polygon(
    *,
    cell_id: str,
    kind: str,
    story: int | None,
    source_id: str | None,
    polygon: Polygon,
    floor_y: float,
    ceiling_y: float,
    boundary_kind: str,
    properties: dict[str, Any] | None = None,
) -> list[Cell3D]:
    if polygon.is_empty or polygon.area <= EPS:
        return []
    floor_y = _from_lattice(_snap_to_lattice(floor_y))
    ceiling_y = _from_lattice(_snap_to_lattice(ceiling_y))
    if ceiling_y - floor_y <= EPS:
        return []
    out: list[Cell3D] = []
    for part_index, subpoly in enumerate(_convex_subpolygons(polygon)):
        if subpoly.is_empty or subpoly.area <= EPS:
            continue
        coords = list(subpoly.exterior.coords)
        if coords and coords[-1] == coords[0]:
            coords = coords[:-1]
        footprint = _snap_poly_coords([(float(x), float(z)) for x, z, *_ in coords])
        if len(footprint) < 3:
            continue
        perimeter = set(range(len(footprint)))
        poly_cell_id = cell_id if part_index == 0 else f"{cell_id}:part:{part_index}"
        arranged = build_arranged_polyhedral_cell(
            cell_id=poly_cell_id,
            room_id=source_id or poly_cell_id,
            room_index=None,
            part_id=None,
            story=story,
            base_atom_id=source_id or poly_cell_id,
            cell_kind=kind,
            region_footprint=footprint,
            base_y=floor_y,
            top_y_at=lambda _x, _z, y=ceiling_y: y,
            top_surface_kind="flat",
            roof_hypothesis_id=None,
            perimeter_side_face_indices=perimeter,
        )
        if arranged is None:
            continue
        faces = tuple(
            face_obj
            for face_obj in (
                _face_from_arranged_record(face)
                for face in (arranged.get("faces") or [])
            )
            if face_obj is not None
        )
        if not faces:
            continue
        bbox_raw = arranged.get("bbox_xyz") or [0.0] * 6
        centroid_raw = arranged.get("centroid_xyz") or [0.0, 0.0, 0.0]
        out.append(
            Cell3D(
                id=poly_cell_id,
                kind=kind,
                story=story,
                source_id=source_id,
                centroid=Vec3(
                    float(centroid_raw[0]),
                    float(centroid_raw[1]),
                    float(centroid_raw[2]),
                ),
                volume_m3=float(arranged.get("volume_m3", 0.0) or 0.0),
                bbox_xyz=tuple(float(v) for v in bbox_raw),
                faces=faces,
                properties={
                    "xz_footprint": [[_round3(x), _round3(z)] for x, z in footprint],
                    "arrangement": arranged.get("arrangement") or {},
                    **(properties or {}),
                },
            )
        )
    return out


def _face_polygon_xz(corners: list[list[float]]) -> Polygon | None:
    pts = _snap_poly_coords(
        [
            (float(c[0]), float(c[2]))
            for c in corners
            if isinstance(c, (list, tuple)) and len(c) >= 3
        ]
    )
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        except Exception:
            return None
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= EPS:
        return None
    return poly


def _plane_y_at_xz(
    plane: tuple[float, float, float, float], x: float, z: float, fallback_y: float
) -> float:
    nx, ny, nz, d = plane
    if abs(ny) <= EPS:
        return fallback_y
    return _from_lattice(_snap_to_lattice((-(nx * x + nz * z + d)) / ny))


def split_planar_faces_exact_on_lattice(
    face_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Split a coplanar-ish face group into exact-on-lattice planar atoms.

    Faces are partitioned on the XZ lattice using polygon arrangement and then
    lifted back onto their source planes. The returned adjacency is driven by
    exact shared boundary incidence between partition atoms.
    """
    prepared: list[dict[str, Any]] = []
    for face in face_records:
        corners = face.get("corners") or []
        poly = _face_polygon_xz(corners)
        if poly is None:
            continue
        snapped_corners = [
            Vec3(
                _from_lattice(_snap_to_lattice(float(c[0]))),
                _from_lattice(_snap_to_lattice(float(c[1]))),
                _from_lattice(_snap_to_lattice(float(c[2]))),
            )
            for c in corners
            if isinstance(c, (list, tuple)) and len(c) >= 3
        ]
        if len(snapped_corners) < 3:
            continue
        plane = _plane_from_points(
            snapped_corners[0], snapped_corners[1], snapped_corners[2]
        )
        prepared.append(
            {
                "id": face["id"],
                "boundary_id": face.get("boundary_id"),
                "story": face.get("story"),
                "kind": face.get("kind"),
                "role": face.get("role"),
                "source": face.get("source") or {},
                "properties": face.get("properties") or {},
                "polygon": poly,
                "plane": plane,
                "fallback_y": sum(v.y for v in snapped_corners) / len(snapped_corners),
            }
        )

    if not prepared:
        return {"atoms": [], "adjacency": []}

    linework = [LineString(list(rec["polygon"].exterior.coords)) for rec in prepared]
    arrangement = unary_union(linework)
    geoms = (
        [arrangement]
        if getattr(arrangement, "geom_type", "") != "MultiLineString"
        else list(arrangement.geoms)
    )

    atoms: list[dict[str, Any]] = []
    atom_polys: dict[str, Polygon] = {}
    for atom in polygonize(geoms):
        if atom.is_empty or atom.area <= EPS:
            continue
        if not atom.is_valid:
            try:
                atom = make_valid(atom)
                if atom.geom_type == "MultiPolygon":
                    atom = max(atom.geoms, key=lambda g: g.area)
            except Exception:
                continue
        if not isinstance(atom, Polygon) or atom.is_empty or atom.area <= EPS:
            continue

        owner = None
        best_area = 0.0
        for rec in prepared:
            try:
                overlap_area = atom.intersection(rec["polygon"]).area
            except Exception:
                overlap_area = 0.0
            if overlap_area > best_area + EPS:
                best_area = overlap_area
                owner = rec
        if owner is None or best_area <= EPS:
            continue

        coords = list(atom.exterior.coords)
        if coords and coords[-1] == coords[0]:
            coords = coords[:-1]
        footprint = _snap_poly_coords([(float(x), float(z)) for x, z, *_ in coords])
        if len(footprint) < 3:
            continue

        corners3d = [
            [
                _round3(x),
                _round3(_plane_y_at_xz(owner["plane"], x, z, owner["fallback_y"])),
                _round3(z),
            ]
            for x, z in footprint
        ]
        atom_id = f"roofatom:{_stable_hash([owner['id'], str(corners3d)], 20)}"
        atom_record = {
            "id": atom_id,
            "parent_face_id": owner["id"],
            "parent_boundary_id": owner.get("boundary_id"),
            "story": owner.get("story"),
            "kind": owner.get("kind"),
            "role": owner.get("role"),
            "corners": corners3d,
            "source": owner.get("source") or {},
            "properties": owner.get("properties") or {},
            "plane": [_round3(v) for v in owner["plane"]],
            "area": _round3(atom.area),
        }
        atoms.append(atom_record)
        atom_polys[atom_id] = atom

    adjacency: list[dict[str, Any]] = []
    for i, left in enumerate(atoms):
        left_poly = atom_polys.get(left["id"])
        if left_poly is None:
            continue
        for right in atoms[i + 1 :]:
            if left["parent_face_id"] == right["parent_face_id"]:
                continue
            right_poly = atom_polys.get(right["id"])
            if right_poly is None:
                continue
            shared = left_poly.boundary.intersection(right_poly.boundary)
            shared_len = shared.length if not shared.is_empty else 0.0
            if shared_len <= EPS:
                continue
            adjacency.append(
                {
                    "from_atom_id": left["id"],
                    "to_atom_id": right["id"],
                    "from_boundary_id": left.get("parent_boundary_id"),
                    "to_boundary_id": right.get("parent_boundary_id"),
                    "from_face_id": left["parent_face_id"],
                    "to_face_id": right["parent_face_id"],
                    "shared_edge_length": _round3(shared_len),
                }
            )

    return {"atoms": atoms, "adjacency": adjacency}


def _layered_story_cells(
    graph, floor_y: dict[int, float], top_y: dict[int, float]
) -> list[Cell3D]:
    cells: list[Cell3D] = []
    story_rooms = _story_room_polygons(graph)
    story_gaps = _story_gap_polygons(graph)

    for story, room_entries in story_rooms.items():
        if not room_entries:
            continue
        fy = _from_lattice(_snap_to_lattice(floor_y.get(story, 0.0)))
        cy = _from_lattice(_snap_to_lattice(top_y.get(story, fy + 2.4)))
        if cy - fy <= EPS:
            continue

        room_polys = [poly for _node, poly in room_entries]
        union_poly = unary_union(room_polys)
        try:
            if not union_poly.is_valid:
                union_poly = make_valid(union_poly)
        except Exception:
            pass
        if union_poly.geom_type == "MultiPolygon":
            union_poly = unary_union([geom for geom in union_poly.geoms])
        if union_poly.is_empty:
            continue

        envelope = union_poly.envelope.buffer(1.0, join_style=2)
        linework = [LineString(list(poly.exterior.coords)) for poly in room_polys]
        for _gap_node, gap_poly in story_gaps.get(story, []):
            linework.append(LineString(list(gap_poly.exterior.coords)))
        linework.append(LineString(list(envelope.exterior.coords)))
        arrangement = unary_union(linework)
        geoms = (
            [arrangement]
            if getattr(arrangement, "geom_type", "") != "MultiLineString"
            else list(arrangement.geoms)
        )

        for idx, atom in enumerate(polygonize(geoms)):
            if atom.is_empty or atom.area <= EPS:
                continue
            if not atom.is_valid:
                try:
                    atom = make_valid(atom)
                    if atom.geom_type == "MultiPolygon":
                        atom = max(atom.geoms, key=lambda g: g.area)
                except Exception:
                    continue
            rep = atom.representative_point()
            best_room = None
            best_room_area = 0.0
            for room_node, room_poly in room_entries:
                area = atom.intersection(room_poly).area
                if area > best_room_area + EPS:
                    best_room = room_node
                    best_room_area = area

            if best_room is not None and best_room_area > EPS:
                room_cells = _cell_from_polygon(
                    cell_id=f"cell:room:{best_room.id.split(':', 1)[-1]}:{idx}",
                    kind="room",
                    story=story,
                    source_id=best_room.id,
                    polygon=atom,
                    floor_y=fy,
                    ceiling_y=cy,
                    boundary_kind="physical",
                    properties={"partition_story": story},
                )
                cells.extend(room_cells)
                continue

            best_gap = None
            best_gap_area = 0.0
            for gap_node, gap_poly in story_gaps.get(story, []):
                try:
                    area = atom.intersection(gap_poly).area
                except Exception:
                    continue
                if area > best_gap_area + EPS:
                    best_gap = gap_node
                    best_gap_area = area

            touches_envelope = (
                atom.boundary.intersection(envelope.boundary).length > EPS
            )
            if best_gap is not None and best_gap_area > EPS:
                gap_cells = _cell_from_polygon(
                    cell_id=f"cell:gap:{best_gap.id.split(':', 1)[-1]}:{idx}",
                    kind="gap",
                    story=story,
                    source_id=best_gap.id,
                    polygon=atom,
                    floor_y=fy,
                    ceiling_y=cy,
                    boundary_kind="closure"
                    if (best_gap.properties or {}).get("gap_kind") == "cross_story"
                    else "virtual",
                    properties={
                        "gap_kind": (best_gap.properties or {}).get("gap_kind")
                    },
                )
                cells.extend(gap_cells)
            elif not touches_envelope and union_poly.envelope.contains(rep):
                auto_gap_cells = _cell_from_polygon(
                    cell_id=f"cell:gap:auto:{story}:{idx}",
                    kind="gap",
                    story=story,
                    source_id=None,
                    polygon=atom,
                    floor_y=fy,
                    ceiling_y=cy,
                    boundary_kind="virtual",
                    properties={"gap_kind": "intra_story", "derived": True},
                )
                cells.extend(auto_gap_cells)
    return cells


def _room_cell(graph, node, floor_y: float, ceiling_y: float) -> Cell3D | None:
    footprint = _room_footprint(graph, node)
    if not footprint or len(footprint) < 3:
        return None
    floor_y = _from_lattice(_snap_to_lattice(floor_y))
    ceiling_y = _from_lattice(_snap_to_lattice(ceiling_y))
    xs = [p[0] for p in footprint]
    zs = [p[1] for p in footprint]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    if max_x - min_x <= EPS or max_z - min_z <= EPS or ceiling_y - floor_y <= EPS:
        return None
    area = _polygon_area_xz(footprint)
    volume = area * (ceiling_y - floor_y)
    centroid_x = sum(x for x, _ in footprint) / len(footprint)
    centroid_z = sum(z for _, z in footprint) / len(footprint)
    centroid = Vec3(
        centroid_x,
        (floor_y + ceiling_y) * 0.5,
        centroid_z,
    )
    cid = f"cell:room:{node.id.split(':', 1)[-1]}"
    return Cell3D(
        id=cid,
        kind="room",
        story=node.story,
        source_id=node.id,
        centroid=centroid,
        volume_m3=volume,
        bbox_xyz=(min_x, floor_y, min_z, max_x, ceiling_y, max_z),
        faces=_prism_faces(
            cell_id=cid,
            footprint=footprint,
            floor_y=floor_y,
            ceiling_y=ceiling_y,
            boundary_kind="physical",
        ),
        properties={"xz_footprint": [[_round3(x), _round3(z)] for x, z in footprint]},
    )


def _gap_cell(node, floor_y: float, ceiling_y: float) -> Cell3D | None:
    region_wkt = (node.properties or {}).get("region_wkt")
    if not isinstance(region_wkt, str) or not region_wkt:
        return None
    try:
        poly = shapely_wkt.loads(region_wkt)
    except Exception:
        return None
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        except Exception:
            return None
    min_x, min_z, max_x, max_z = [
        _from_lattice(_snap_to_lattice(v)) for v in poly.bounds
    ]
    floor_y = _from_lattice(_snap_to_lattice(floor_y))
    ceiling_y = _from_lattice(_snap_to_lattice(ceiling_y))
    if max_x - min_x <= EPS or max_z - min_z <= EPS or ceiling_y - floor_y <= EPS:
        return None
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    footprint = _snap_poly_coords([(float(x), float(z)) for x, z, *_ in coords])
    volume = _polygon_area_xz(footprint) * (ceiling_y - floor_y)
    centroid2d = poly.representative_point()
    centroid = Vec3(
        _from_lattice(_snap_to_lattice(float(centroid2d.x))),
        _from_lattice(_snap_to_lattice((floor_y + ceiling_y) * 0.5)),
        _from_lattice(_snap_to_lattice(float(centroid2d.y))),
    )
    cid = f"cell:gap:{node.id.split(':', 1)[-1]}"
    return Cell3D(
        id=cid,
        kind="gap",
        story=node.story,
        source_id=node.id,
        centroid=centroid,
        volume_m3=volume,
        bbox_xyz=(min_x, floor_y, min_z, max_x, ceiling_y, max_z),
        faces=_prism_faces(
            cell_id=cid,
            footprint=footprint,
            floor_y=floor_y,
            ceiling_y=ceiling_y,
            boundary_kind="closure"
            if (node.properties or {}).get("gap_kind") == "cross_story"
            else "virtual",
        ),
        properties={
            "gap_kind": (node.properties or {}).get("gap_kind"),
            "xz_footprint": [[_round3(x), _round3(z)] for x, z in footprint],
        },
    )


def _outside_cell(cells: list[Cell3D]) -> Cell3D | None:
    if not cells:
        return None
    min_x = _from_lattice(
        min(_snap_to_lattice(c.bbox_xyz[0]) for c in cells) - LATTICE_SCALE_MM
    )
    min_y = _from_lattice(
        min(_snap_to_lattice(c.bbox_xyz[1]) for c in cells) - LATTICE_SCALE_MM
    )
    min_z = _from_lattice(
        min(_snap_to_lattice(c.bbox_xyz[2]) for c in cells) - LATTICE_SCALE_MM
    )
    max_x = _from_lattice(
        max(_snap_to_lattice(c.bbox_xyz[3]) for c in cells) + LATTICE_SCALE_MM
    )
    max_y = _from_lattice(
        max(_snap_to_lattice(c.bbox_xyz[4]) for c in cells) + LATTICE_SCALE_MM
    )
    max_z = _from_lattice(
        max(_snap_to_lattice(c.bbox_xyz[5]) for c in cells) + LATTICE_SCALE_MM
    )
    cid = "cell:outside"
    return Cell3D(
        id=cid,
        kind="outside",
        story=None,
        source_id=None,
        centroid=Vec3(
            (min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5
        ),
        volume_m3=max((max_x - min_x) * (max_y - min_y) * (max_z - min_z), 0.0),
        bbox_xyz=(min_x, min_y, min_z, max_x, max_y, max_z),
        faces=_box_faces(
            cell_id=cid,
            min_x=min_x,
            min_y=min_y,
            min_z=min_z,
            max_x=max_x,
            max_y=max_y,
            max_z=max_z,
            boundary_kind="virtual",
        ),
        properties={},
    )


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(_snap_to_lattice(a0), _snap_to_lattice(b0))
    hi = min(_snap_to_lattice(a1), _snap_to_lattice(b1))
    return _from_lattice(max(0, hi - lo))


def _infer_adjacency(cells: list[Cell3D]) -> list[CellAdjacency]:
    out: list[CellAdjacency] = []
    same_story_polys: dict[str, Polygon] = {}
    story_polys: dict[int, list[Polygon]] = {}
    for cell in cells:
        footprint = (cell.properties or {}).get("xz_footprint")
        if (
            isinstance(footprint, list)
            and len(footprint) >= 3
            and cell.kind != "outside"
        ):
            poly = Polygon([(float(p[0]), float(p[1])) for p in footprint])
            if not poly.is_valid:
                try:
                    poly = make_valid(poly)
                    if poly.geom_type == "MultiPolygon":
                        poly = max(poly.geoms, key=lambda g: g.area)
                except Exception:
                    continue
            if not isinstance(poly, Polygon):
                continue
            if not poly.is_empty and poly.area > EPS:
                same_story_polys[cell.id] = poly
                if isinstance(cell.story, int):
                    story_polys.setdefault(cell.story, []).append(poly)
    building_union_by_story: dict[int, Polygon] = {}
    for story, polys in story_polys.items():
        try:
            merged = unary_union(polys)
            if merged.geom_type == "MultiPolygon":
                merged = unary_union([max(merged.geoms, key=lambda g: g.area)])
            if not merged.is_valid:
                merged = make_valid(merged)
            if merged.geom_type == "MultiPolygon":
                merged = max(merged.geoms, key=lambda g: g.area)
            if isinstance(merged, Polygon) and not merged.is_empty:
                building_union_by_story[story] = merged
        except Exception:
            continue
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            ax0, ay0, az0, ax1, ay1, az1 = a.bbox_xyz
            bx0, by0, bz0, bx1, by1, bz1 = b.bbox_xyz

            x_ov = _overlap_1d(ax0, ax1, bx0, bx1)
            y_ov = _overlap_1d(ay0, ay1, by0, by1)
            z_ov = _overlap_1d(az0, az1, bz0, bz1)

            poly_a = same_story_polys.get(a.id)
            poly_b = same_story_polys.get(b.id)
            if (
                poly_a is not None
                and poly_b is not None
                and a.story == b.story
                and y_ov > EPS
            ):
                boundary = poly_a.boundary.intersection(poly_b.boundary)
                shared_len = boundary.length if not boundary.is_empty else 0.0
                if shared_len > EPS:
                    out.append(
                        CellAdjacency(
                            from_cell_id=a.id,
                            to_cell_id=b.id,
                            relation="adjacent_to",
                            shared_face_area=shared_len * y_ov,
                            axis="xz",
                        )
                    )
                    continue
            if y_ov <= EPS and x_ov > EPS and z_ov > EPS:
                if _snap_to_lattice(ay1) == _snap_to_lattice(by0):
                    area = x_ov * z_ov
                    out.append(
                        CellAdjacency(
                            from_cell_id=a.id,
                            to_cell_id=b.id,
                            relation="below",
                            shared_face_area=area,
                            axis="y",
                        )
                    )
                elif _snap_to_lattice(by1) == _snap_to_lattice(ay0):
                    area = x_ov * z_ov
                    out.append(
                        CellAdjacency(
                            from_cell_id=a.id,
                            to_cell_id=b.id,
                            relation="above",
                            shared_face_area=area,
                            axis="y",
                        )
                    )
    for cell in cells:
        if cell.kind == "outside" or not isinstance(cell.story, int):
            continue
        poly = same_story_polys.get(cell.id)
        if poly is None:
            continue
        union = building_union_by_story.get(cell.story)
        if union is None:
            continue
        exposed = poly.boundary.intersection(union.boundary)
        exposed_len = exposed.length if not exposed.is_empty else 0.0
        if exposed_len > EPS:
            out.append(
                CellAdjacency(
                    from_cell_id=cell.id,
                    to_cell_id="cell:outside",
                    relation="adjacent_to",
                    shared_face_area=exposed_len
                    * (cell.bbox_xyz[4] - cell.bbox_xyz[1]),
                    axis="outside",
                )
            )
    return out


def build_cell_complex(graph) -> CellComplex:
    """Build a deterministic 3D cell complex from an existing topology graph."""
    floor_y, top_y = _story_heights(graph)
    cells = _layered_story_cells(graph, floor_y, top_y)
    if not cells:
        for node in graph.nodes:
            if node.type == "Room" and isinstance(node.story, int):
                fy = floor_y.get(node.story, 0.0)
                cy = top_y.get(node.story, fy + 2.4)
                cell = _room_cell(graph, node, fy, cy)
                if cell is not None:
                    cells.append(cell)
            elif node.type == "Gap" and isinstance(node.story, int):
                fy = floor_y.get(node.story, 0.0)
                cy = top_y.get(node.story, fy + 2.4)
                cell = _gap_cell(node, fy, cy)
                if cell is not None:
                    cells.append(cell)

    outside = _outside_cell(cells)
    if outside is not None:
        cells.append(outside)

    adjacency = _infer_adjacency(cells)
    metadata = {
        "backend": "exact_lattice_polyhedral_arrangement_v3",
        "numeric_model": "fixed_point_mm",
        "topology_exact_on_lattice": True,
        "polyhedral_kernel": "halfspace_triple_intersection",
        "cell_count": len(cells),
        "adjacency_count": len(adjacency),
    }
    return CellComplex(
        cells=tuple(cells), adjacency=tuple(adjacency), metadata=metadata
    )
