"""Stage B kinetic-style convex cell partitioning.

This is the reconstruction-front-end counterpart to ``face_selection``.  It
builds a bounded convex-cell partition by incrementally propagating input
planes through an axis-aligned prism.  The small closed-building cases are the
same cells produced by a full plane arrangement; the incremental form keeps the
API compatible with the kinetic propagation path described in Bauchet &
Lafarge without requiring the legacy selector to know about the cell graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.face_selection import CandidateFace, EdgeKey

_GEOM_TOL = 1e-7
_POINT_KEY_SCALE = 1_000_000
_MAX_CELLS = 5_000


@dataclass(frozen=True, slots=True)
class BoundingPrism:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        if not (
            self.x_min < self.x_max
            and self.y_min < self.y_max
            and self.z_min < self.z_max
        ):
            raise ValueError(f"invalid bounding prism: {self}")

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        *,
        margin: float = 1.0,
    ) -> BoundingPrism:
        pts = _as_points(points)
        if len(pts) == 0:
            raise ValueError("cannot infer BoundingPrism from empty points")
        lo = pts[:, :3].min(axis=0) - float(margin)
        hi = pts[:, :3].max(axis=0) + float(margin)
        return cls(
            x_min=float(lo[0]),
            x_max=float(hi[0]),
            y_min=float(lo[1]),
            y_max=float(hi[1]),
            z_min=float(lo[2]),
            z_max=float(hi[2]),
        )


@dataclass(frozen=True, slots=True)
class KineticEvent:
    time: float
    kind: str
    polygon_a: int
    polygon_b: int
    intersection_geometry: Any


@dataclass(frozen=True, slots=True)
class ConvexCell:
    cell_id: int
    signs: tuple[int, ...]
    vertices: tuple[tuple[float, float, float], ...]
    centroid: tuple[float, float, float]
    volume: float
    boundary_planes: tuple[_BoundaryPlane, ...]
    source_plane_count: int


@dataclass(frozen=True, slots=True)
class _BoundaryPlane:
    plane: Plane
    source_kind: Literal["input", "bbox"]
    source_id: int
    side: int


@dataclass(frozen=True, slots=True)
class _CellVertex:
    point: np.ndarray
    active_boundaries: frozenset[int]


def kinetic_partition(
    planes: Sequence[Plane],
    scan_points: np.ndarray,
    bounding_prism: BoundingPrism | Sequence[float],
    priority: str = "size",
) -> list[ConvexCell]:
    """Partition a bounding prism into convex cells induced by ``planes``.

    Each input plane splits cells that straddle it.  The cell ``signs`` tuple
    uses ``-1`` for the side satisfying ``n.x <= d``, ``+1`` for
    ``n.x >= d``, and ``0`` for planes not propagated because the safety cell
    cap was reached.  ``priority="size"`` processes larger cells first when a
    split wave would exceed the cap.
    """

    if priority != "size":
        raise ValueError(f"unsupported kinetic partition priority: {priority!r}")

    _as_points(scan_points)
    prism = _coerce_bounding_prism(bounding_prism)
    normalized = _dedupe_planes([_normalize_plane(plane) for plane in planes])
    bbox_boundaries = _bbox_boundaries(prism)
    initial_vertices = _vertices_for_boundaries(bbox_boundaries)
    if not initial_vertices:
        return []

    cells = [
        _make_cell(
            cell_id=0,
            signs=(),
            boundaries=tuple(bbox_boundaries),
            source_plane_count=len(normalized),
        )
    ]
    cells = [cell for cell in cells if cell is not None]

    for plane_id, plane in enumerate(normalized):
        next_cells: list[ConvexCell] = []
        ordered = sorted(cells, key=lambda cell: cell.volume, reverse=True)
        for cell in ordered:
            vertices = np.asarray(cell.vertices, dtype=float)
            residuals = _residuals(plane, vertices)
            if np.all(residuals <= _GEOM_TOL):
                next_cells.append(_replace_signs(cell, (*cell.signs, -1)))
                continue
            if np.all(residuals >= -_GEOM_TOL):
                next_cells.append(_replace_signs(cell, (*cell.signs, 1)))
                continue
            if len(next_cells) + 2 + (len(ordered) - len(next_cells)) > _MAX_CELLS:
                next_cells.append(_replace_signs(cell, (*cell.signs, 0)))
                continue
            negative = _split_cell(cell, plane, plane_id, side=-1)
            positive = _split_cell(cell, plane, plane_id, side=1)
            if negative is not None:
                next_cells.append(negative)
            if positive is not None:
                next_cells.append(positive)
        cells = [
            _replace_signs(cell, _pad_signs(cell.signs, plane_id + 1))
            for cell in next_cells
        ]

    return [
        _replace_cell_id(
            _replace_signs(cell, _pad_signs(cell.signs, len(normalized))),
            idx,
        )
        for idx, cell in enumerate(cells)
    ]


def label_cells_inside_outside(
    cells: Sequence[ConvexCell],
    scan_points: np.ndarray,
) -> dict[int, Literal["inside", "outside"]]:
    """Label cells by scan support.

    Interior scan points vote for the cell containing them.  If normals are
    supplied as ``(N, 6)`` rows ``x,y,z,nx,ny,nz``, surface samples also vote
    for the adjacent side opposite the outward normal as inside.
    """

    points = _as_points(scan_points, allow_normals=True)
    inside_votes = {cell.cell_id: 0 for cell in cells}
    outside_votes = {cell.cell_id: 0 for cell in cells}
    if len(points) == 0:
        return {cell.cell_id: "outside" for cell in cells}

    xyz = points[:, :3]
    normals = points[:, 3:6] if points.shape[1] >= 6 else None
    for cell in cells:
        contained = _points_in_cell(xyz, cell)
        inside_votes[cell.cell_id] += int(np.count_nonzero(contained))
        if normals is None:
            continue
        for point, normal in zip(xyz, normals, strict=True):
            nearest = _nearest_input_boundary(cell, point)
            if nearest is None:
                continue
            boundary, sign = nearest
            plane_normal = np.array(
                [boundary.plane.a, boundary.plane.b, boundary.plane.c],
                dtype=float,
            )
            points_into_cell = float(normal @ plane_normal) * float(sign) < 0.0
            if points_into_cell:
                outside_votes[cell.cell_id] += 1
            else:
                inside_votes[cell.cell_id] += 1

    return {
        cell.cell_id: (
            "inside"
            if inside_votes[cell.cell_id] > outside_votes[cell.cell_id]
            and inside_votes[cell.cell_id] > 0
            else "outside"
        )
        for cell in cells
    }


def cells_to_candidate_faces(
    cells: Sequence[ConvexCell],
    labels: Mapping[int, str],
) -> list[CandidateFace]:
    """Return boundary faces between inside and outside cells."""

    cell_by_signature = {cell.signs: cell for cell in cells}
    candidates: list[CandidateFace] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for cell in cells:
        if labels.get(cell.cell_id) != "inside":
            continue
        for plane_id, sign in enumerate(cell.signs):
            if sign == 0:
                continue
            neighbor_signs = list(cell.signs)
            neighbor_signs[plane_id] = -sign
            neighbor = cell_by_signature.get(tuple(neighbor_signs))
            if neighbor is None or labels.get(neighbor.cell_id) == "inside":
                continue
            key = (plane_id, tuple(sorted((cell.cell_id, neighbor.cell_id))))
            if key in seen:
                continue
            seen.add(key)
            candidate = _candidate_for_cell_face(
                face_id=len(candidates),
                cell=cell,
                plane_id=plane_id,
                side=sign,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _split_cell(
    cell: ConvexCell,
    plane: Plane,
    plane_id: int,
    *,
    side: int,
) -> ConvexCell | None:
    if side == -1:
        boundary = _BoundaryPlane(
            plane=plane,
            source_kind="input",
            source_id=plane_id,
            side=-1,
        )
    else:
        boundary = _BoundaryPlane(
            plane=_flip_plane(plane),
            source_kind="input",
            source_id=plane_id,
            side=1,
        )
    return _make_cell(
        cell_id=cell.cell_id,
        signs=(*cell.signs, side),
        boundaries=(*cell.boundary_planes, boundary),
        source_plane_count=cell.source_plane_count,
    )


def _make_cell(
    *,
    cell_id: int,
    signs: tuple[int, ...],
    boundaries: tuple[_BoundaryPlane, ...],
    source_plane_count: int,
) -> ConvexCell | None:
    vertices = _vertices_for_boundaries(boundaries)
    if len(vertices) < 4:
        return None
    coords = tuple(tuple(float(coord) for coord in vertex.point) for vertex in vertices)
    centroid = tuple(float(value) for value in np.mean(np.asarray(coords), axis=0))
    return ConvexCell(
        cell_id=cell_id,
        signs=signs,
        vertices=coords,
        centroid=centroid,
        volume=_convex_volume(coords),
        boundary_planes=boundaries,
        source_plane_count=source_plane_count,
    )


def _vertices_for_boundaries(
    boundaries: Sequence[_BoundaryPlane],
) -> list[_CellVertex]:
    by_key: dict[tuple[int, int, int], _CellVertex] = {}
    for i, j, k in combinations(range(len(boundaries)), 3):
        try:
            point = _three_plane_intersection(
                boundaries[i].plane,
                boundaries[j].plane,
                boundaries[k].plane,
            )
        except np.linalg.LinAlgError:
            continue
        if not _inside_boundaries(point, boundaries):
            continue
        active = frozenset(
            idx
            for idx, boundary in enumerate(boundaries)
            if abs(_signed_residual(boundary.plane, point)) <= 1e-5
        )
        key = tuple(round(float(coord) * _POINT_KEY_SCALE) for coord in point)
        by_key[key] = _CellVertex(point=point, active_boundaries=active)
    return list(by_key.values())


def _candidate_for_cell_face(
    *,
    face_id: int,
    cell: ConvexCell,
    plane_id: int,
    side: int,
) -> CandidateFace | None:
    boundary_index = _boundary_index_for_input(cell, plane_id)
    if boundary_index is None:
        return None
    face_vertices = [
        vertex
        for vertex in _vertices_for_boundaries(cell.boundary_planes)
        if boundary_index in vertex.active_boundaries
    ]
    if len(face_vertices) < 3:
        return None
    boundary = cell.boundary_planes[boundary_index]
    plane = boundary.plane
    ordered = _order_face_vertices(plane, face_vertices)
    corners = tuple(tuple(float(coord) for coord in vertex.point) for vertex in ordered)
    area = _polygon_area_3d(corners)
    if area <= 1e-9:
        return None
    edge_keys = _edge_keys_for_face(
        ordered,
        cell.boundary_planes,
        boundary_index,
        source_plane_count=cell.source_plane_count,
    )
    origin, u, v = _plane_frame(plane)
    polygon = Polygon(
        [
            (
                float((np.asarray(corner, dtype=float) - origin) @ u),
                float((np.asarray(corner, dtype=float) - origin) @ v),
            )
            for corner in corners
        ]
    )
    if polygon.is_empty or polygon.area <= 1e-9:
        return None
    return CandidateFace(
        face_id=face_id,
        plane_id=plane_id,
        polygon=polygon,
        edge_keys=edge_keys,
        supporting_points=np.empty((0, 3), dtype=float),
        support_density=0.0,
        confidence_label="kinetic-boundary",
        corners=corners,
        plane=plane,
        area=area,
        support_score=area,
        coverage_polygon=_coverage_polygon_xz(corners),
        domain_area=max(float(_coverage_polygon_xz(corners).area), 1e-9),
    )


def _edge_keys_for_face(
    ring: Sequence[_CellVertex],
    boundaries: Sequence[_BoundaryPlane],
    face_boundary_index: int,
    *,
    source_plane_count: int,
) -> tuple[EdgeKey, ...]:
    keys: list[EdgeKey] = []
    face_boundary = boundaries[face_boundary_index]
    face_source = _edge_source_id(face_boundary, source_plane_count)
    for idx, start in enumerate(ring):
        end = ring[(idx + 1) % len(ring)]
        shared = sorted(
            (start.active_boundaries & end.active_boundaries) - {face_boundary_index}
        )
        if not shared:
            continue
        other_source = _edge_source_id(boundaries[shared[0]], source_plane_count)
        keys.append(tuple(sorted((face_source, other_source))))
    return tuple(keys)


def _edge_source_id(boundary: _BoundaryPlane, source_plane_count: int) -> int:
    if boundary.source_kind == "input":
        return boundary.source_id
    return source_plane_count + boundary.source_id


def _boundary_index_for_input(cell: ConvexCell, plane_id: int) -> int | None:
    for idx, boundary in enumerate(cell.boundary_planes):
        if boundary.source_kind == "input" and boundary.source_id == plane_id:
            return idx
    return None


def _nearest_input_boundary(
    cell: ConvexCell,
    point: np.ndarray,
) -> tuple[_BoundaryPlane, int] | None:
    best: tuple[float, _BoundaryPlane, int] | None = None
    for idx, sign in enumerate(cell.signs):
        boundary_index = _boundary_index_for_input(cell, idx)
        if boundary_index is None:
            continue
        boundary = cell.boundary_planes[boundary_index]
        distance = abs(_signed_residual(boundary.plane, point))
        if best is None or distance < best[0]:
            best = (distance, boundary, sign)
    if best is None:
        return None
    return best[1], best[2]


def _points_in_cell(points: np.ndarray, cell: ConvexCell) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    mask = np.ones((len(points),), dtype=bool)
    for boundary in cell.boundary_planes:
        residuals = (
            boundary.plane.a * points[:, 0]
            + boundary.plane.b * points[:, 1]
            + boundary.plane.c * points[:, 2]
            - boundary.plane.d
        )
        mask &= residuals <= 1e-6
    return mask


def _replace_signs(cell: ConvexCell, signs: tuple[int, ...]) -> ConvexCell:
    return ConvexCell(
        cell_id=cell.cell_id,
        signs=signs,
        vertices=cell.vertices,
        centroid=cell.centroid,
        volume=cell.volume,
        boundary_planes=cell.boundary_planes,
        source_plane_count=cell.source_plane_count,
    )


def _replace_cell_id(cell: ConvexCell, cell_id: int) -> ConvexCell:
    return ConvexCell(
        cell_id=cell_id,
        signs=cell.signs,
        vertices=cell.vertices,
        centroid=cell.centroid,
        volume=cell.volume,
        boundary_planes=cell.boundary_planes,
        source_plane_count=cell.source_plane_count,
    )


def _pad_signs(signs: tuple[int, ...], size: int) -> tuple[int, ...]:
    if len(signs) >= size:
        return signs
    return signs + (0,) * (size - len(signs))


def _bbox_boundaries(prism: BoundingPrism) -> list[_BoundaryPlane]:
    specs = [
        (Plane(a=1.0, b=0.0, c=0.0, d=prism.x_max), 0),
        (Plane(a=-1.0, b=0.0, c=0.0, d=-prism.x_min), 1),
        (Plane(a=0.0, b=1.0, c=0.0, d=prism.y_max), 2),
        (Plane(a=0.0, b=-1.0, c=0.0, d=-prism.y_min), 3),
        (Plane(a=0.0, b=0.0, c=1.0, d=prism.z_max), 4),
        (Plane(a=0.0, b=0.0, c=-1.0, d=-prism.z_min), 5),
    ]
    return [
        _BoundaryPlane(plane=plane, source_kind="bbox", source_id=idx, side=0)
        for plane, idx in specs
    ]


def _coerce_bounding_prism(value: BoundingPrism | Sequence[float]) -> BoundingPrism:
    if isinstance(value, BoundingPrism):
        return value
    raw = tuple(float(item) for item in value)
    if len(raw) != 6:
        raise ValueError("bounding_prism must be BoundingPrism or six bounds")
    return BoundingPrism(
        x_min=raw[0],
        x_max=raw[1],
        y_min=raw[2],
        y_max=raw[3],
        z_min=raw[4],
        z_max=raw[5],
    )


def _dedupe_planes(planes: Sequence[Plane]) -> list[Plane]:
    out: list[Plane] = []
    for plane in planes:
        if any(_planes_same_or_opposite(plane, existing) for existing in out):
            continue
        out.append(plane)
    return out


def _planes_same_or_opposite(first: Plane, second: Plane) -> bool:
    n1 = np.array([first.a, first.b, first.c], dtype=float)
    n2 = np.array([second.a, second.b, second.c], dtype=float)
    if abs(float(n1 @ n2) - 1.0) <= 1e-6:
        return abs(first.d - second.d) <= 1e-6
    if abs(float(n1 @ n2) + 1.0) <= 1e-6:
        return abs(first.d + second.d) <= 1e-6
    return False


def _normalize_plane(plane: Plane) -> Plane:
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError(f"degenerate plane normal: {plane}")
    return Plane(
        a=float(plane.a) / norm,
        b=float(plane.b) / norm,
        c=float(plane.c) / norm,
        d=float(plane.d) / norm,
    )


def _flip_plane(plane: Plane) -> Plane:
    return Plane(a=-plane.a, b=-plane.b, c=-plane.c, d=-plane.d)


def _as_points(scan_points: np.ndarray, *, allow_normals: bool = False) -> np.ndarray:
    points = np.asarray(scan_points, dtype=float)
    if points.size == 0:
        return np.empty((0, 6 if allow_normals else 3), dtype=float)
    if points.ndim != 2 or points.shape[1] not in ((3, 6) if allow_normals else (3,)):
        expected = "(N, 3) or (N, 6)" if allow_normals else "(N, 3)"
        raise ValueError(f"scan_points must have shape {expected}")
    return points


def _residuals(plane: Plane, points: np.ndarray) -> np.ndarray:
    return (
        plane.a * points[:, 0]
        + plane.b * points[:, 1]
        + plane.c * points[:, 2]
        - plane.d
    )


def _signed_residual(plane: Plane, point: np.ndarray) -> float:
    return float(plane.a * point[0] + plane.b * point[1] + plane.c * point[2] - plane.d)


def _inside_boundaries(
    point: np.ndarray,
    boundaries: Sequence[_BoundaryPlane],
) -> bool:
    return all(
        _signed_residual(boundary.plane, point) <= 1e-6
        for boundary in boundaries
    )


def _three_plane_intersection(p1: Plane, p2: Plane, p3: Plane) -> np.ndarray:
    matrix = np.array(
        [[p1.a, p1.b, p1.c], [p2.a, p2.b, p2.c], [p3.a, p3.b, p3.c]],
        dtype=float,
    )
    rhs = np.array([p1.d, p2.d, p3.d], dtype=float)
    return np.linalg.solve(matrix, rhs)


def _plane_frame(plane: Plane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    origin = normal * float(plane.d)
    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return origin, u, v


def _order_face_vertices(
    plane: Plane,
    vertices: Sequence[_CellVertex],
) -> list[_CellVertex]:
    origin, u, v = _plane_frame(plane)
    uv = [
        (
            float((vertex.point - origin) @ u),
            float((vertex.point - origin) @ v),
        )
        for vertex in vertices
    ]
    centroid = np.mean(np.asarray(uv, dtype=float), axis=0)
    ordered = [
        vertex
        for _angle, vertex in sorted(
            (
                np.arctan2(coord[1] - centroid[1], coord[0] - centroid[0]),
                vertex,
            )
            for coord, vertex in zip(uv, vertices, strict=True)
        )
    ]
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    ring_normal = np.zeros(3, dtype=float)
    for idx, vertex in enumerate(ordered):
        nxt = ordered[(idx + 1) % len(ordered)]
        ring_normal += np.cross(vertex.point, nxt.point)
    if float(ring_normal @ normal) < 0.0:
        ordered.reverse()
    return ordered


def _polygon_area_3d(corners: Sequence[Sequence[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    pts = [np.asarray(corner, dtype=float) for corner in corners]
    normal_sum = np.zeros(3, dtype=float)
    for idx, point in enumerate(pts):
        normal_sum += np.cross(point, pts[(idx + 1) % len(pts)])
    return 0.5 * float(np.linalg.norm(normal_sum))


def _convex_volume(corners: Sequence[Sequence[float]]) -> float:
    if len(corners) < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(np.asarray(corners, dtype=float)).volume)
    except Exception:
        return 0.0


def _coverage_polygon_xz(corners: Sequence[Sequence[float]]) -> Polygon:
    if len(corners) < 3:
        return Polygon()
    polygon = Polygon([(float(corner[0]), float(corner[2])) for corner in corners])
    if polygon.is_empty or not polygon.is_valid:
        return Polygon()
    return polygon
