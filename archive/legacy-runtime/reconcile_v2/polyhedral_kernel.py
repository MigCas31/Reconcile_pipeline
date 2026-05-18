"""Portable exact halfspace arrangement kernel on integer lattice coordinates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

LATTICE_SCALE_MM = 1000
EPS = 1e-6
MIN_TOP_CLEARANCE_M = 1.0 / LATTICE_SCALE_MM


def _stable_hash(parts: list[str], length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:length]


def _snap(value: float) -> int:
    return round(float(value) * LATTICE_SCALE_MM)


def _unsnap(value: int | Fraction) -> float:
    return float(Fraction(value, 1) / LATTICE_SCALE_MM)


def _round6(value: float) -> float:
    return round(float(value), 6)


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2])]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: list[float]) -> list[float]:
    length = _norm(a)
    if length <= EPS:
        return [0.0, 0.0, 0.0]
    return [float(v / length) for v in a]


@dataclass(frozen=True)
class HalfspacePlane:
    id: str
    role: str
    source_kind: str
    coeffs: tuple[int, int, int, int]
    support_corners: tuple[tuple[float, float, float], ...]
    metadata: dict[str, Any]

    def eval_lattice(self, point: tuple[Fraction, Fraction, Fraction]) -> Fraction:
        ax, by, cz, d = self.coeffs
        return (
            Fraction(ax, 1) * point[0]
            + Fraction(by, 1) * point[1]
            + Fraction(cz, 1) * point[2]
            + Fraction(d, 1)
        )


def _plane_from_points(
    *,
    plane_id: str,
    role: str,
    source_kind: str,
    points: list[tuple[float, float, float]],
    seed_point: tuple[float, float, float],
    metadata: dict[str, Any] | None = None,
) -> HalfspacePlane | None:
    if len(points) < 3:
        return None
    p0 = p1 = p2 = None
    for start in range(len(points) - 2):
        a = (_snap(points[start][0]), _snap(points[start][1]), _snap(points[start][2]))
        for middle in range(start + 1, len(points) - 1):
            b = (
                _snap(points[middle][0]),
                _snap(points[middle][1]),
                _snap(points[middle][2]),
            )
            for end in range(middle + 1, len(points)):
                c = (
                    _snap(points[end][0]),
                    _snap(points[end][1]),
                    _snap(points[end][2]),
                )
                u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                cross = (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
                if cross != (0, 0, 0):
                    p0, p1, p2 = a, b, c
                    break
            if p0 is not None:
                break
        if p0 is not None:
            break
    if p0 is None or p1 is None or p2 is None:
        return None
    u = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    a = u[1] * v[2] - u[2] * v[1]
    b = u[2] * v[0] - u[0] * v[2]
    c = u[0] * v[1] - u[1] * v[0]
    if a == 0 and b == 0 and c == 0:
        return None
    d = -(a * p0[0] + b * p0[1] + c * p0[2])
    sx, sy, sz = (_snap(seed_point[0]), _snap(seed_point[1]), _snap(seed_point[2]))
    seed_eval = a * sx + b * sy + c * sz + d
    if seed_eval > 0:
        a *= -1
        b *= -1
        c *= -1
        d *= -1
    return HalfspacePlane(
        id=plane_id,
        role=role,
        source_kind=source_kind,
        coeffs=(a, b, c, d),
        support_corners=tuple((float(x), float(y), float(z)) for x, y, z in points),
        metadata=metadata or {},
    )


def _vertical_plane_from_edge(
    *,
    plane_id: str,
    role: str,
    source_kind: str,
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    base_y: float,
    top_y: float,
    seed_point: tuple[float, float, float],
    metadata: dict[str, Any] | None = None,
) -> HalfspacePlane | None:
    y_top = top_y if top_y > base_y + EPS else base_y + 1.0
    points = [
        (float(edge_start[0]), float(base_y), float(edge_start[1])),
        (float(edge_end[0]), float(base_y), float(edge_end[1])),
        (float(edge_start[0]), float(y_top), float(edge_start[1])),
    ]
    return _plane_from_points(
        plane_id=plane_id,
        role=role,
        source_kind=source_kind,
        points=points,
        seed_point=seed_point,
        metadata=metadata,
    )


def _det3(
    a1: int,
    a2: int,
    a3: int,
    b1: int,
    b2: int,
    b3: int,
    c1: int,
    c2: int,
    c3: int,
) -> int:
    return (
        a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)
    )


def _intersect_three_planes(
    left: HalfspacePlane,
    middle: HalfspacePlane,
    right: HalfspacePlane,
) -> tuple[Fraction, Fraction, Fraction] | None:
    a1, b1, c1, d1 = left.coeffs
    a2, b2, c2, d2 = middle.coeffs
    a3, b3, c3, d3 = right.coeffs
    denom = _det3(a1, b1, c1, a2, b2, c2, a3, b3, c3)
    if denom == 0:
        return None
    x_num = _det3(-d1, b1, c1, -d2, b2, c2, -d3, b3, c3)
    y_num = _det3(a1, -d1, c1, a2, -d2, c2, a3, -d3, c3)
    z_num = _det3(a1, b1, -d1, a2, b2, -d2, a3, b3, -d3)
    return (
        Fraction(x_num, denom),
        Fraction(y_num, denom),
        Fraction(z_num, denom),
    )


def _dedupe_vertices(
    vertices: list[tuple[Fraction, Fraction, Fraction]],
) -> list[tuple[Fraction, Fraction, Fraction]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[Fraction, Fraction, Fraction]] = []
    for vx, vy, vz in vertices:
        key = (
            round(float(vx)),
            round(float(vy)),
            round(float(vz)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append((Fraction(key[0], 1), Fraction(key[1], 1), Fraction(key[2], 1)))
    return out


def _order_face_vertices(
    vertices: list[tuple[Fraction, Fraction, Fraction]],
    plane: HalfspacePlane,
) -> list[list[float]]:
    points = [[_unsnap(v[0]), _unsnap(v[1]), _unsnap(v[2])] for v in vertices]
    centroid = [
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    ]
    normal = _normalize(
        [
            -float(plane.coeffs[0]),
            -float(plane.coeffs[1]),
            -float(plane.coeffs[2]),
        ]
    )
    reference = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.85 else [0.0, 0.0, 1.0]
    u = _normalize(_cross(reference, normal))
    if _norm(u) <= EPS:
        reference = [0.0, 1.0, 0.0]
        u = _normalize(_cross(reference, normal))
    v = _normalize(_cross(normal, u))
    scored = []
    for point in points:
        rel = _vec_sub(point, centroid)
        px = _dot(rel, u)
        py = _dot(rel, v)
        scored.append((math.atan2(py, px), point))
    scored.sort(key=lambda item: item[0])
    ordered = [[_round6(coord) for coord in point] for _, point in scored]
    if len(ordered) >= 3:
        face_normal = _cross(
            _vec_sub(ordered[1], ordered[0]),
            _vec_sub(ordered[2], ordered[1]),
        )
        if _dot(face_normal, normal) < 0.0:
            ordered.reverse()
    return ordered


def _polygon_area_3d(corners: list[list[float]], plane: HalfspacePlane) -> float:
    if len(corners) < 3:
        return 0.0
    normal = _normalize(
        [
            -float(plane.coeffs[0]),
            -float(plane.coeffs[1]),
            -float(plane.coeffs[2]),
        ]
    )
    area_vec = [0.0, 0.0, 0.0]
    for index, point in enumerate(corners):
        nxt = corners[(index + 1) % len(corners)]
        cross = _cross(point, nxt)
        area_vec = [area_vec[i] + cross[i] for i in range(3)]
    return abs(_dot(area_vec, normal)) * 0.5


def _polyhedron_volume(faces: list[dict[str, Any]]) -> float:
    volume = 0.0
    for face in faces:
        corners = face.get("corners") or []
        if len(corners) < 3:
            continue
        origin = corners[0]
        for index in range(1, len(corners) - 1):
            a = origin
            b = corners[index]
            c = corners[index + 1]
            volume += _dot(a, _cross(b, c)) / 6.0
    return abs(volume)


def _polygon_area_xz(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) * 0.5


def _clip_footprint_to_min_clearance(
    region_footprint: list[tuple[float, float]],
    *,
    base_y: float,
    top_y_at,
    min_clearance: float,
) -> list[tuple[float, float]]:
    if len(region_footprint) < 3:
        return []

    clipped = [(float(x), float(z)) for x, z in region_footprint]
    for _ in range(1):
        output: list[tuple[float, float]] = []
        for index, end in enumerate(clipped):
            start = clipped[index - 1]
            start_h = float(top_y_at(start[0], start[1])) - float(base_y)
            end_h = float(top_y_at(end[0], end[1])) - float(base_y)
            start_inside = start_h >= min_clearance - EPS
            end_inside = end_h >= min_clearance - EPS
            if start_inside and end_inside:
                output.append(end)
                continue
            if start_inside != end_inside:
                denom = end_h - start_h
                if abs(denom) > EPS:
                    t = (min_clearance - start_h) / denom
                    ix = start[0] + (end[0] - start[0]) * t
                    iz = start[1] + (end[1] - start[1]) * t
                    output.append((_round6(ix), _round6(iz)))
            if not start_inside and end_inside:
                output.append(end)
        clipped = output
        if len(clipped) < 3:
            return []

    deduped: list[tuple[float, float]] = []
    for point in clipped:
        if (
            deduped
            and abs(point[0] - deduped[-1][0]) <= EPS
            and abs(point[1] - deduped[-1][1]) <= EPS
        ):
            continue
        deduped.append(point)
    if (
        len(deduped) >= 2
        and abs(deduped[0][0] - deduped[-1][0]) <= EPS
        and abs(deduped[0][1] - deduped[-1][1]) <= EPS
    ):
        deduped.pop()
    if len(deduped) < 3 or _polygon_area_xz(deduped) <= EPS:
        return []
    return deduped


def _point_on_segment_xz(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    px = point[0] - start[0]
    pz = point[1] - start[1]
    cross = dx * pz - dz * px
    if abs(cross) > 1e-5:
        return False
    dot = px * dx + pz * dz
    if dot < -1e-5:
        return False
    length_sq = dx * dx + dz * dz
    if dot - length_sq > 1e-5:
        return False
    return True


def _perimeter_indices_from_original(
    original: list[tuple[float, float]],
    clipped: list[tuple[float, float]],
) -> set[int]:
    keep: set[int] = set()
    if len(original) < 2 or len(clipped) < 2:
        return keep
    original_edges = [
        (original[index], original[(index + 1) % len(original)])
        for index in range(len(original))
    ]
    for index, start in enumerate(clipped):
        end = clipped[(index + 1) % len(clipped)]
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        if any(
            _point_on_segment_xz(midpoint, edge_start, edge_end)
            for edge_start, edge_end in original_edges
        ):
            keep.add(index)
    return keep


def _expected_face_count(footprint_count: int) -> int:
    return footprint_count + 2 if footprint_count >= 3 else 0


def _direct_prism_faces(
    *,
    cell_id: str,
    base_corners: list[tuple[float, float, float]],
    top_corners: list[tuple[float, float, float]],
    top_surface_kind: str,
    roof_hypothesis_id: str | None,
    side_face_indices: set[int],
) -> list[dict[str, Any]]:
    if len(base_corners) < 3 or len(base_corners) != len(top_corners):
        return []

    faces: list[dict[str, Any]] = []
    bottom_plane = _plane_from_points(
        plane_id=f"arr-plane:bottom:{cell_id}",
        role="slab",
        source_kind="bottom_cap",
        points=[base_corners[0], base_corners[1], base_corners[2]],
        seed_point=base_corners[0],
        metadata={"face_kind": "bottom"},
    )
    top_plane = _plane_from_points(
        plane_id=f"arr-plane:top:{cell_id}",
        role="roof" if top_surface_kind == "oblique" else "slab",
        source_kind=top_surface_kind,
        points=[top_corners[0], top_corners[1], top_corners[2]],
        seed_point=top_corners[0],
        metadata={"face_kind": "top", "roof_hypothesis_id": roof_hypothesis_id},
    )
    if bottom_plane is None or top_plane is None:
        return []

    bottom_ordered = [[_round6(x), _round6(y), _round6(z)] for x, y, z in base_corners]
    top_ordered = [[_round6(x), _round6(y), _round6(z)] for x, y, z in top_corners]
    faces.append(
        {
            "id": f"arr-face:{
                _stable_hash(
                    [cell_id, 'direct-bottom', str(bottom_ordered)],
                    20,
                )
            }",
            "kind": "bottom",
            "role": "slab",
            "source_kind": "bottom_cap",
            "corners": bottom_ordered,
            "corners_lattice": [
                [_snap(c[0]), _snap(c[1]), _snap(c[2])] for c in bottom_ordered
            ],
            "plane_coeffs": list(bottom_plane.coeffs),
            "area_m2": _round6(_polygon_area_3d(bottom_ordered, bottom_plane)),
            "metadata": {"face_kind": "bottom"},
        }
    )
    faces.append(
        {
            "id": f"arr-face:{
                _stable_hash(
                    [cell_id, 'direct-top', str(top_ordered)],
                    20,
                )
            }",
            "kind": "top",
            "role": "roof" if top_surface_kind == "oblique" else "slab",
            "source_kind": top_surface_kind,
            "corners": list(reversed(top_ordered)),
            "corners_lattice": [
                [_snap(c[0]), _snap(c[1]), _snap(c[2])] for c in reversed(top_ordered)
            ],
            "plane_coeffs": list(top_plane.coeffs),
            "area_m2": _round6(_polygon_area_3d(top_ordered, top_plane)),
            "metadata": {"face_kind": "top", "roof_hypothesis_id": roof_hypothesis_id},
        }
    )
    for index in range(len(base_corners)):
        nxt = (index + 1) % len(base_corners)
        corners = [
            [
                _round6(base_corners[index][0]),
                _round6(base_corners[index][1]),
                _round6(base_corners[index][2]),
            ],
            [
                _round6(base_corners[nxt][0]),
                _round6(base_corners[nxt][1]),
                _round6(base_corners[nxt][2]),
            ],
            [
                _round6(top_corners[nxt][0]),
                _round6(top_corners[nxt][1]),
                _round6(top_corners[nxt][2]),
            ],
            [
                _round6(top_corners[index][0]),
                _round6(top_corners[index][1]),
                _round6(top_corners[index][2]),
            ],
        ]
        side_plane = _plane_from_points(
            plane_id=f"arr-plane:side:{cell_id}:{index}",
            role="wall" if index in side_face_indices else "splitter",
            source_kind="wall" if index in side_face_indices else "splitter",
            points=[base_corners[index], base_corners[nxt], top_corners[index]],
            seed_point=top_corners[index],
            metadata={
                "face_kind": "side",
                "side_index": index,
                "perimeter_facing": index in side_face_indices,
            },
        )
        if side_plane is None:
            continue
        faces.append(
            {
                "id": f"arr-face:{
                    _stable_hash(
                        [cell_id, f'direct-side:{index}', str(corners)],
                        20,
                    )
                }",
                "kind": "side",
                "role": side_plane.role,
                "source_kind": side_plane.source_kind,
                "corners": corners,
                "corners_lattice": [
                    [_snap(c[0]), _snap(c[1]), _snap(c[2])] for c in corners
                ],
                "plane_coeffs": list(side_plane.coeffs),
                "area_m2": _round6(_polygon_area_3d(corners, side_plane)),
                "metadata": side_plane.metadata,
            }
        )
    return faces


def build_arranged_polyhedral_cell(
    *,
    cell_id: str,
    room_id: str,
    room_index: int | None,
    part_id: str | None,
    story: int | None,
    base_atom_id: str,
    cell_kind: str,
    region_footprint: list[tuple[float, float]],
    base_y: float,
    top_y_at,
    top_surface_kind: str,
    roof_hypothesis_id: str | None,
    perimeter_side_face_indices: set[int],
) -> dict[str, Any] | None:
    if len(region_footprint) < 3:
        return None
    clipped_footprint = [(float(x), float(z)) for x, z in region_footprint]
    side_face_indices = set(perimeter_side_face_indices)
    if top_surface_kind == "oblique":
        clipped_footprint = _clip_footprint_to_min_clearance(
            clipped_footprint,
            base_y=base_y,
            top_y_at=top_y_at,
            min_clearance=MIN_TOP_CLEARANCE_M,
        )
        if len(clipped_footprint) < 3:
            return None
        side_face_indices = _perimeter_indices_from_original(
            region_footprint, clipped_footprint
        )

    seed_x = sum(point[0] for point in clipped_footprint) / len(clipped_footprint)
    seed_z = sum(point[1] for point in clipped_footprint) / len(clipped_footprint)
    seed_top_y = float(top_y_at(seed_x, seed_z))
    seed_y = (float(base_y) + seed_top_y) * 0.5
    seed_point = (seed_x, seed_y, seed_z)

    top_corners = [
        (float(x), float(top_y_at(x, z)), float(z)) for x, z in clipped_footprint
    ]
    base_corners = [(float(x), float(base_y), float(z)) for x, z in clipped_footprint]
    planes: list[HalfspacePlane] = []
    bottom_plane = _plane_from_points(
        plane_id=f"arr-plane:bottom:{cell_id}",
        role="slab",
        source_kind="bottom_cap",
        points=[base_corners[0], base_corners[1], base_corners[2]],
        seed_point=seed_point,
        metadata={"face_kind": "bottom"},
    )
    if bottom_plane is None:
        return None
    planes.append(bottom_plane)
    top_plane = _plane_from_points(
        plane_id=f"arr-plane:top:{cell_id}",
        role="roof" if top_surface_kind == "oblique" else "slab",
        source_kind=top_surface_kind,
        points=[top_corners[0], top_corners[1], top_corners[2]],
        seed_point=seed_point,
        metadata={"face_kind": "top", "roof_hypothesis_id": roof_hypothesis_id},
    )
    if top_plane is None:
        return None
    planes.append(top_plane)

    max_top_y = max(point[1] for point in top_corners)
    for index, start in enumerate(clipped_footprint):
        end = clipped_footprint[(index + 1) % len(clipped_footprint)]
        perimeter_facing = index in side_face_indices
        plane = _vertical_plane_from_edge(
            plane_id=f"arr-plane:side:{cell_id}:{index}",
            role="wall" if perimeter_facing else "splitter",
            source_kind="wall" if perimeter_facing else "splitter",
            edge_start=start,
            edge_end=end,
            base_y=base_y,
            top_y=max_top_y,
            seed_point=seed_point,
            metadata={
                "face_kind": "side",
                "side_index": index,
                "perimeter_facing": perimeter_facing,
            },
        )
        if plane is not None:
            planes.append(plane)

    vertices: list[tuple[Fraction, Fraction, Fraction]] = []
    for i, left in enumerate(planes):
        for j in range(i + 1, len(planes)):
            middle = planes[j]
            for right in planes[j + 1 :]:
                point = _intersect_three_planes(left, middle, right)
                if point is None:
                    continue
                if any(
                    plane.eval_lattice(point) > Fraction(1, LATTICE_SCALE_MM * 10)
                    for plane in planes
                ):
                    continue
                vertices.append(point)
    vertices = _dedupe_vertices(vertices)
    if len(vertices) < 4:
        return None

    faces: list[dict[str, Any]] = []
    face_vertices_by_role: dict[str, list[list[float]]] = {}
    for plane in planes:
        on_plane = [vertex for vertex in vertices if plane.eval_lattice(vertex) == 0]
        if len(on_plane) < 3:
            continue
        ordered = _order_face_vertices(on_plane, plane)
        if len(ordered) < 3:
            continue
        face_id = f"arr-face:{_stable_hash([cell_id, plane.id, str(ordered)], 20)}"
        face_record = {
            "id": face_id,
            "kind": str((plane.metadata or {}).get("face_kind", "side")),
            "role": plane.role,
            "source_kind": plane.source_kind,
            "corners": ordered,
            "corners_lattice": [
                [_snap(c[0]), _snap(c[1]), _snap(c[2])] for c in ordered
            ],
            "plane_coeffs": list(plane.coeffs),
            "area_m2": _round6(_polygon_area_3d(ordered, plane)),
            "metadata": plane.metadata,
        }
        faces.append(face_record)
        face_vertices_by_role.setdefault(face_record["kind"], []).extend(ordered)

    top_faces = [face for face in faces if str(face.get("kind") or "") == "top"]
    top_face_vertex_count = max(
        (len(face.get("corners") or []) for face in top_faces), default=0
    )
    if len(clipped_footprint) >= 3 and (
        not top_faces
        or top_face_vertex_count < len(clipped_footprint)
        or len(faces) < _expected_face_count(len(clipped_footprint))
    ):
        direct_faces = _direct_prism_faces(
            cell_id=cell_id,
            base_corners=base_corners,
            top_corners=top_corners,
            top_surface_kind=top_surface_kind,
            roof_hypothesis_id=roof_hypothesis_id,
            side_face_indices=side_face_indices,
        )
        if direct_faces:
            faces = direct_faces

    if not faces:
        return None

    all_points = [corner for face in faces for corner in (face.get("corners") or [])]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    zs = [point[2] for point in all_points]
    volume = _round6(_polyhedron_volume(faces))
    centroid = [
        _round6(sum(_unsnap(vertex[i]) for vertex in vertices) / len(vertices))
        for i in range(3)
    ]
    return {
        "id": cell_id,
        "type": "Cell",
        "cell_kind": cell_kind,
        "story": story,
        "room_id": room_id,
        "room_index": room_index,
        "part_id": part_id,
        "base_atom_id": base_atom_id,
        "roof_hypothesis_id": roof_hypothesis_id,
        "roof_surface_kind": top_surface_kind,
        "volume_m3": volume,
        "centroid_xyz": centroid,
        "bbox_xyz": [
            _round6(min(xs)),
            _round6(min(ys)),
            _round6(min(zs)),
            _round6(max(xs)),
            _round6(max(ys)),
            _round6(max(zs)),
        ],
        "faces": faces,
        "arrangement": {
            "plane_count": len(planes),
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "plane_roles": [plane.role for plane in planes],
            "plane_source_kinds": [plane.source_kind for plane in planes],
        },
    }
