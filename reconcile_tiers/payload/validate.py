from __future__ import annotations

import math

from shapely import set_precision
from shapely.geometry import Polygon

from reconcile_tiers._core.newell import is_planar, newell_normal
from reconcile_tiers._core.plane import Plane as CorePlane
from reconcile_tiers._core.plane import PlaneKey, fit_plane_any, plane_key
from reconcile_tiers._core.shapely2 import oriented_rectangle_side_lengths
from reconcile_tiers.payload.schema import (
    CeilingPiece,
    CeilingSource,
    DormerFace,
    GableClosure,
    GapKind,
    GapPiece,
    HorizontalLid,
    KneeWall,
    Plane,
    Quad,
    Room,
    TierPayload,
    Vec3,
    Wall,
)

CEILING_OVERLAP_TOLERANCE_M2 = 1e-2
WALL_OVERLAP_TOLERANCE_M2 = 1.0
WALL_RELATIVE_OVERLAP_TOLERANCE = 0.50
WALL_VERTICAL_SEPARATION_TOLERANCE_M = 0.05
FLOOR_ADJACENCY_TOLERANCE_M = 0.05
FLOOR_OVERLAP_TOLERANCE_M2 = 0.30
GAP_MIN_THICKNESS_M = 0.01
OVERLAY_GRID_SIZE_M = 1e-8

_HORIZONTAL_GAP_KINDS = {
    GapKind.GAP_FLOOR,
    GapKind.GAP_CEILING,
    GapKind.STITCH_FLOOR,
    GapKind.STITCH_CEIL,
    GapKind.EXTERIOR_FLOOR,
    GapKind.EXTERIOR_CEIL,
}


class PayloadInvariantError(ValueError):
    def __init__(self, path: str, expected: str, actual: object) -> None:
        super().__init__(f"{path}: expected {expected}, got {actual!r}")
        self.path = path
        self.expected = expected
        self.actual = actual


def _coords(corners: list[Vec3]) -> list[list[float]]:
    return [[corner.x, corner.y, corner.z] for corner in corners]


def _validate_horizontal_lid(lid: HorizontalLid, path: str) -> None:
    if len(lid.corners) < 3:
        raise PayloadInvariantError(path, "at least 3 corners", len(lid.corners))
    ys = [corner.y for corner in lid.corners]
    spread = max(ys) - min(ys)
    if spread > 1e-3:
        raise PayloadInvariantError(path, "planar Y spread <= 0.001", spread)
    normal_y = newell_normal(_coords(lid.corners))[1]
    if normal_y <= 0.0:
        raise PayloadInvariantError(path, "Newell normal +Y", normal_y)
    _validate_xz_polygon(lid.corners, path)
    if lid.holes:
        outer_y_mean = sum(ys) / len(ys)
        outer_xz = _polygon_xz(lid.corners)
        for h_idx, hole in enumerate(lid.holes):
            hole_path = f"{path}.holes[{h_idx}]"
            if len(hole) < 3:
                raise PayloadInvariantError(hole_path, "at least 3 corners", len(hole))
            hole_ys = [c.y for c in hole]
            if max(hole_ys) - min(hole_ys) > 1e-3:
                raise PayloadInvariantError(
                    hole_path,
                    "planar hole Y spread <= 0.001",
                    max(hole_ys) - min(hole_ys),
                )
            if abs(sum(hole_ys) / len(hole_ys) - outer_y_mean) > 0.05:
                raise PayloadInvariantError(
                    hole_path,
                    f"hole Y close to lid Y ({outer_y_mean:.3f})",
                    sum(hole_ys) / len(hole_ys),
                )
            hole_xz = _polygon_xz(hole)
            if not outer_xz.contains(hole_xz.buffer(-1e-6)):
                raise PayloadInvariantError(
                    hole_path, "hole inside outer ring", "hole crosses outer ring"
                )


def _validate_quad(quad: Quad, path: str) -> None:
    if len(quad.corners) != 4:
        raise PayloadInvariantError(path, "exactly 4 corners", len(quad.corners))
    if not is_planar(_coords(quad.corners), tol=0.05):
        raise PayloadInvariantError(
            path, "coplanar within 0.05 m", _coords(quad.corners)
        )
    _validate_plane_local_polygon(quad.corners, path)


def _validate_plane(plane: Plane, path: str) -> None:
    coeffs = (plane.a, plane.b, plane.c, plane.d)
    if not all(math.isfinite(value) for value in coeffs):
        raise PayloadInvariantError(path, "finite plane coefficients", coeffs)
    if abs(plane.b) < CorePlane.MIN_NY:
        raise PayloadInvariantError(path, f"|b| >= {CorePlane.MIN_NY}", plane.b)


def _plane_y_at(plane: Plane, x: float, z: float) -> float:
    return (plane.d - plane.a * x - plane.c * z) / plane.b


def _polygon_xz(corners: list[Vec3]) -> Polygon:
    return Polygon([(corner.x, corner.z) for corner in corners])


def _ceiling_piece_xz_polygon(piece: CeilingPiece) -> Polygon:
    holes = [
        [(corner.x, corner.z) for corner in hole]
        for hole in piece.holes
        if len(hole) >= 3
    ]
    return set_precision(
        Polygon([(corner.x, corner.z) for corner in piece.corners], holes),
        OVERLAY_GRID_SIZE_M,
    )


def _room_floor_polygons(rooms: list[Room]) -> list[Polygon | None]:
    """Return the union XZ polygon per room (across split floor pieces)."""
    floors: list[Polygon | None] = []
    for room in rooms:
        polys: list[Polygon] = []
        for piece in room.floor:
            try:
                poly = _polygon_xz(piece.corners)
            except Exception:
                continue
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        if not polys:
            floors.append(None)
            continue
        merged = polys[0]
        for p in polys[1:]:
            try:
                merged = merged.union(p)
            except Exception:
                pass
        floors.append(merged if isinstance(merged, Polygon) else polys[0])
    return floors


def _floors_are_nested(
    floors: list[Polygon | None], idx_a: int | None, idx_b: int | None
) -> bool:
    if idx_a is None or idx_b is None or idx_a == idx_b:
        return False
    if idx_a >= len(floors) or idx_b >= len(floors):
        return False
    poly_a = floors[idx_a]
    poly_b = floors[idx_b]
    if poly_a is None or poly_b is None:
        return False
    try:
        return bool(poly_a.covers(poly_b) or poly_b.covers(poly_a))
    except Exception:
        return False


def _floors_are_adjacent_without_overlap(
    rooms: list[Room],
    floors: list[Polygon | None],
    idx_a: int | None,
    idx_b: int | None,
) -> bool:
    if idx_a is None or idx_b is None or idx_a == idx_b:
        return False
    if idx_a >= len(rooms) or idx_b >= len(rooms):
        return False
    if rooms[idx_a].story != rooms[idx_b].story:
        return False
    poly_a = floors[idx_a]
    poly_b = floors[idx_b]
    if poly_a is None or poly_b is None:
        return False
    try:
        if poly_a.intersection(poly_b).area > FLOOR_OVERLAP_TOLERANCE_M2:
            return False
        return poly_a.distance(poly_b) <= FLOOR_ADJACENCY_TOLERANCE_M
    except Exception:
        return False


def _ceiling_room_index(piece: CeilingPiece) -> int | None:
    markers = (
        "::tier-ceiling-flat::",
        "::tier-ceiling-raw::",
        "::tier-ceiling-computed-oblique-room::",
        "::tier-ceiling-roof-arrangement-room::",
    )
    for marker in markers:
        if marker not in piece.locator_id:
            continue
        suffix = piece.locator_id.rsplit(marker, 1)[1]
        room_token = suffix.split("/", 1)[0].split(":", 1)[0]
        try:
            return int(room_token)
        except ValueError:
            return None
    return None


def _validate_xz_polygon(corners: list[Vec3], path: str) -> Polygon:
    poly = _polygon_xz(corners)
    if not poly.is_valid or poly.is_empty or poly.area <= 0.0:
        raise PayloadInvariantError(path, "valid non-empty XZ polygon", poly.wkt)
    return poly


def _validate_plane_local_polygon(corners: list[Vec3], path: str) -> Polygon:
    plane = fit_plane_any(_coords(corners))
    if plane is None:
        raise PayloadInvariantError(path, "fittable plane", _coords(corners))
    poly = _project_corners_to_plane(corners, plane[:3])
    if poly is None or not poly.is_valid or poly.is_empty or poly.area <= 0.0:
        raise PayloadInvariantError(
            path, "valid non-empty plane-local polygon", _coords(corners)
        )
    return poly


def _validate_ceiling_piece(piece: CeilingPiece, path: str) -> None:
    _validate_plane(piece.plane, f"{path}.plane")
    if len(piece.corners) < 3:
        raise PayloadInvariantError(
            f"{path}.corners", "at least 3 corners", len(piece.corners)
        )
    if newell_normal(_coords(piece.corners))[1] <= 0.0:
        raise PayloadInvariantError(
            f"{path}.corners",
            "Newell normal +Y",
            newell_normal(_coords(piece.corners))[1],
        )
    for idx, corner in enumerate(piece.corners):
        expected_y = _plane_y_at(piece.plane, corner.x, corner.z)
        if abs(expected_y - corner.y) > 0.01:
            raise PayloadInvariantError(
                f"{path}.corners[{idx}]",
                "corner.y within 0.01 m of plane.y_at",
                corner.y - expected_y,
            )
    main_poly = _validate_xz_polygon(piece.corners, f"{path}.corners")
    for hole_idx, hole in enumerate(piece.holes):
        if len(hole) < 3:
            raise PayloadInvariantError(
                f"{path}.holes[{hole_idx}]", "at least 3 corners", len(hole)
            )
        hole_poly = _polygon_xz(hole)
        if not main_poly.contains(hole_poly):
            raise PayloadInvariantError(
                f"{path}.holes[{hole_idx}]",
                "hole contained by main polygon",
                hole_poly.wkt,
            )


def _validate_wall(wall: Wall, room: Room, wall_idx: int, room_path: str) -> None:
    if len(wall.corners) < 3:
        raise PayloadInvariantError(
            f"{room_path}.walls[{wall_idx}].corners",
            "at least 3 corners",
            len(wall.corners),
        )
    _validate_plane_local_polygon(
        wall.corners, f"{room_path}.walls[{wall_idx}].corners"
    )
    for cutout_idx, cutout in enumerate(wall.cutouts):
        _validate_quad(cutout, f"{room_path}.walls[{wall_idx}].cutouts[{cutout_idx}]")


def _validate_vertical_roof_face(face: KneeWall | DormerFace, path: str) -> None:
    if len(face.corners) < 3:
        raise PayloadInvariantError(
            f"{path}.corners", "at least 3 corners", len(face.corners)
        )
    if not is_planar(_coords(face.corners), tol=0.05):
        raise PayloadInvariantError(
            f"{path}.corners", "coplanar within 0.05 m", _coords(face.corners)
        )
    _validate_plane_local_polygon(face.corners, f"{path}.corners")
    for cutout_idx, cutout in enumerate(getattr(face, "cutouts", [])):
        _validate_quad(cutout, f"{path}.cutouts[{cutout_idx}]")


def _validate_gable_closure(closure: GableClosure, path: str) -> None:
    if len(closure.corners) < 3:
        raise PayloadInvariantError(
            f"{path}.corners", "at least 3 corners", len(closure.corners)
        )
    if not is_planar(_coords(closure.corners), tol=0.05):
        raise PayloadInvariantError(
            f"{path}.corners", "coplanar within 0.05 m", _coords(closure.corners)
        )
    _validate_plane_local_polygon(closure.corners, f"{path}.corners")


def _validate_gap_piece(piece: GapPiece, path: str) -> None:
    if len(piece.corners) < 3:
        raise PayloadInvariantError(
            f"{path}.corners", "at least 3 corners", len(piece.corners)
        )
    if piece.kind in _HORIZONTAL_GAP_KINDS:
        normal_y = newell_normal(_coords(piece.corners))[1]
        if normal_y <= 0.0:
            raise PayloadInvariantError(f"{path}.corners", "Newell normal +Y", normal_y)
        _validate_xz_polygon(piece.corners, f"{path}.corners")


def _project_corners_to_plane(
    corners: list[Vec3], normal: tuple[float, float, float]
) -> Polygon | None:
    import numpy as np

    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return None
    n = n / norm
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        return None
    u = u / u_norm
    v = np.cross(n, u)
    pts = np.asarray([[c.x, c.y, c.z] for c in corners], dtype=float)
    flat = [(float(np.dot(p, u)), float(np.dot(p, v))) for p in pts]
    if len(flat) < 3:
        return None
    poly = Polygon(flat)
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _validate_ceiling_partition(pieces: list[CeilingPiece], rooms: list[Room]) -> None:
    room_floors = _room_floor_polygons(rooms)
    by_key: dict[PlaneKey, list[tuple[int, Polygon, int | None, CeilingSource]]] = {}
    for idx, piece in enumerate(pieces):
        key = plane_key(piece.plane)
        poly = _ceiling_piece_xz_polygon(piece)
        if not poly.is_valid or poly.is_empty:
            continue
        room_idx = _ceiling_room_index(piece)
        for jdx, prev, prev_room_idx, prev_source in by_key.get(key, []):
            if (
                piece.source == CeilingSource.ROOF_ARRANGEMENT_ATTIC
                or prev_source == CeilingSource.ROOF_ARRANGEMENT_ATTIC
            ):
                continue
            try:
                inter = poly.intersection(prev).area
            except Exception:
                inter = 0.0
            if inter > CEILING_OVERLAP_TOLERANCE_M2:
                if _floors_are_nested(room_floors, room_idx, prev_room_idx):
                    continue
                raise PayloadInvariantError(
                    f"ceiling[{idx}]",
                    f"no XZ overlap with ceiling[{jdx}] on same plane",
                    f"{inter:.4f} m^2",
                )
        by_key.setdefault(key, []).append((idx, poly, room_idx, piece.source))


def _vertically_separated(
    corners_a: list[Vec3],
    corners_b: list[Vec3],
    tolerance: float = WALL_VERTICAL_SEPARATION_TOLERANCE_M,
) -> bool:
    a_min = min(corner.y for corner in corners_a)
    a_max = max(corner.y for corner in corners_a)
    b_min = min(corner.y for corner in corners_b)
    b_max = max(corner.y for corner in corners_b)
    return a_max <= b_min + tolerance or b_max <= a_min + tolerance


def _wall_overlap_allowed_by_size(
    intersection_area: float, poly_a: Polygon, poly_b: Polygon
) -> bool:
    relative_limit = WALL_RELATIVE_OVERLAP_TOLERANCE * min(poly_a.area, poly_b.area)
    return intersection_area <= max(WALL_OVERLAP_TOLERANCE_M2, relative_limit)


def _validate_wall_partition(
    rooms: list[Room],
    knee_walls: list[KneeWall],
    dormer_faces: list[DormerFace],
) -> None:
    room_floors = _room_floor_polygons(rooms)
    by_key: dict[PlaneKey, list[tuple[str, Polygon, int | None, list[Vec3]]]] = {}
    sources: list[tuple[str, list[Vec3], int | None]] = []
    for room_idx, room in enumerate(rooms):
        for wall_idx, wall in enumerate(room.walls):
            sources.append(
                (f"rooms[{room_idx}].walls[{wall_idx}]", wall.corners, room_idx)
            )
    for kw_idx, knee in enumerate(knee_walls):
        sources.append((f"knee_walls[{kw_idx}]", knee.corners, None))
    for face_idx, face in enumerate(dormer_faces):
        sources.append((f"dormer_faces[{face_idx}]", face.corners, None))
    for path, corners, room_idx in sources:
        plane = fit_plane_any([[c.x, c.y, c.z] for c in corners])
        if plane is None:
            continue
        key = plane_key(plane)
        poly = _project_corners_to_plane(corners, plane[:3])
        if poly is None:
            continue
        poly = set_precision(poly, OVERLAY_GRID_SIZE_M)
        for prev_path, prev_poly, prev_room_idx, prev_corners in by_key.get(key, []):
            try:
                inter = poly.intersection(prev_poly).area
            except Exception:
                inter = 0.0
            if inter > WALL_OVERLAP_TOLERANCE_M2:
                if _vertically_separated(corners, prev_corners):
                    continue
                if _floors_are_nested(room_floors, room_idx, prev_room_idx):
                    continue
                if _floors_are_adjacent_without_overlap(
                    rooms, room_floors, room_idx, prev_room_idx
                ):
                    continue
                if _wall_overlap_allowed_by_size(inter, poly, prev_poly):
                    continue
                raise PayloadInvariantError(
                    path,
                    f"no plane-local overlap with {prev_path} on same plane",
                    f"{inter:.4f} m^2",
                )
        by_key.setdefault(key, []).append((path, poly, room_idx, corners))


def _validate_no_sliver_gaps(gaps: list[GapPiece]) -> None:
    for idx, piece in enumerate(gaps):
        plane = fit_plane_any([[c.x, c.y, c.z] for c in piece.corners])
        if plane is None:
            continue
        poly = _project_corners_to_plane(piece.corners, plane[:3])
        if poly is None:
            continue
        sides = oriented_rectangle_side_lengths(poly)
        if sides:
            min_width = min(sides)
        else:
            minx, miny, maxx, maxy = poly.bounds
            min_width = min(maxx - minx, maxy - miny)
        if min_width < GAP_MIN_THICKNESS_M:
            raise PayloadInvariantError(
                f"gaps[{idx}]",
                f"in-plane min width (oriented) >= {GAP_MIN_THICKNESS_M}",
                f"{min_width:.4f} m",
            )


def validate_payload(payload: TierPayload) -> None:
    if payload.schema_version != "1":
        raise PayloadInvariantError("schema_version", "'1'", payload.schema_version)
    if not 1 <= payload.classification.tier <= 8:
        raise PayloadInvariantError(
            "classification.tier", "1..8", payload.classification.tier
        )
    for room_idx, room in enumerate(payload.rooms):
        room_path = f"rooms[{room_idx}]"
        for floor_idx, floor_piece in enumerate(room.floor):
            _validate_horizontal_lid(floor_piece, f"{room_path}.floor[{floor_idx}]")
        for wall_idx, wall in enumerate(room.walls):
            _validate_wall(wall, room, wall_idx, room_path)
        for door_idx, door in enumerate(room.doors):
            _validate_quad(door, f"{room_path}.doors[{door_idx}]")
        for window_idx, window in enumerate(room.windows):
            _validate_quad(window, f"{room_path}.windows[{window_idx}]")
    for knee_idx, knee in enumerate(payload.knee_walls):
        _validate_vertical_roof_face(knee, f"knee_walls[{knee_idx}]")
    for face_idx, face in enumerate(payload.dormer_faces):
        _validate_vertical_roof_face(face, f"dormer_faces[{face_idx}]")
    for closure_idx, closure in enumerate(payload.gable_closures):
        _validate_gable_closure(closure, f"gable_closures[{closure_idx}]")
    for ceiling_idx, piece in enumerate(payload.ceiling):
        _validate_ceiling_piece(piece, f"ceiling[{ceiling_idx}]")
    for gap_idx, piece in enumerate(payload.gaps):
        _validate_gap_piece(piece, f"gaps[{gap_idx}]")
    _validate_ceiling_partition(payload.ceiling, payload.rooms)
    _validate_wall_partition(payload.rooms, payload.knee_walls, payload.dormer_faces)
    _validate_no_sliver_gaps(payload.gaps)
