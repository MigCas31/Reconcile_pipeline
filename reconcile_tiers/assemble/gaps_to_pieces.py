from __future__ import annotations

import math

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.footprint import (
    is_stitch_edge_unsupported_by_story_footprint,
    story_footprint_context,
)
from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers._core.plane import fit_plane_any
from reconcile_tiers._core.shapely2 import oriented_rectangle_side_lengths
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.extract.stitches import stitch_closure_groups
from reconcile_tiers.payload.schema import GapKind, GapPiece, GapScope, Vec3

GAP_PIECE_MIN_THICKNESS_M = 0.01
GAP_PIECE_OVERLAP_M2 = 0.01
CAP_DUPLICATE_OVERLAP_FRACTION = 0.50
SIDE_DUPLICATE_OVERLAP_FRACTION = 0.50
HORIZONTAL_PLANE_BUCKET_M = 0.005
MAX_HORIZONTAL_CAP_Y_RANGE_M = 0.10
MIN_SLOPED_CAP_Y_RANGE_M = HORIZONTAL_PLANE_BUCKET_M
MIN_SLOPED_CAP_INCL_DEG = 5.0
WALL_OVERLAP_BUFFER_M = 0.03
WALL_OVERLAP_MIN_M = 0.10
WALL_OVERLAP_MAX_RATIO = 0.50
EXTERIOR_INTERNAL_OVERLAP_FRACTION = 0.95
EXTERIOR_INTERNAL_MIN_BOUNDARY_DISTANCE_M = 0.25

_GAP_KIND_BY_TYPE = {
    "within_story": GapKind.SIDE,
    "room_ceiling_void": GapKind.SIDE,
    "gap_floor": GapKind.GAP_FLOOR,
    "gap_ceiling": GapKind.GAP_CEILING,
}
_STITCH_KIND_BY_TYPE = {
    "stitch": GapKind.STITCH,
    "stitch_floor": GapKind.STITCH_FLOOR,
    "stitch_ceiling": GapKind.STITCH_CEIL,
}
_EXTERIOR_KIND_BY_TYPE = {
    "side": GapKind.EXTERIOR_SIDE,
    "floor": GapKind.EXTERIOR_FLOOR,
    "ceiling": GapKind.EXTERIOR_CEIL,
}
_HORIZONTAL_KINDS = {
    GapKind.GAP_FLOOR,
    GapKind.GAP_CEILING,
    GapKind.STITCH_FLOOR,
    GapKind.STITCH_CEIL,
    GapKind.EXTERIOR_FLOOR,
    GapKind.EXTERIOR_CEIL,
}
_UPWARD_KINDS = {GapKind.GAP_CEILING, GapKind.STITCH_CEIL, GapKind.EXTERIOR_CEIL}
_UPWARD_KINDS.update({GapKind.GAP_FLOOR, GapKind.STITCH_FLOOR, GapKind.EXTERIOR_FLOOR})
_SIDE_OWNERSHIP_KINDS = {GapKind.SIDE, GapKind.STITCH, GapKind.EXTERIOR_SIDE}


def _normal_y(corners: list[list[float]]) -> float:
    return newell_normal(corners)[1]


def _planarize_horizontal(
    corners: list[list[float]], kind: GapKind
) -> list[list[float]]:
    mean_y = sum(float(corner[1]) for corner in corners) / len(corners)
    out = [[float(corner[0]), mean_y, float(corner[2])] for corner in corners]
    normal_y = _normal_y(out)
    if kind in _UPWARD_KINDS and normal_y < 0.0:
        out.reverse()
    return out


def _wind_upward(corners: list[list[float]]) -> list[list[float]]:
    out = [[float(corner[0]), float(corner[1]), float(corner[2])] for corner in corners]
    if _normal_y(out) < 0.0:
        out.reverse()
    return out


def _y_range(corners: list[list[float]]) -> float:
    ys = [float(corner[1]) for corner in corners]
    return max(ys) - min(ys) if ys else 0.0


def _inclination_from_horizontal_deg(corners: list[list[float]]) -> float:
    import numpy as _np

    plane = fit_plane_any(corners)
    if plane is None:
        return 0.0
    normal = _np.asarray(plane[:3], dtype=float)
    norm = float(_np.linalg.norm(normal))
    if norm <= 1e-12:
        return 0.0
    ny = abs(float(normal[1]) / norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, ny))))


def _is_sloped_ceiling_cap(corners: list[list[float]]) -> bool:
    return (
        _y_range(corners) > MIN_SLOPED_CAP_Y_RANGE_M
        and _inclination_from_horizontal_deg(corners) > MIN_SLOPED_CAP_INCL_DEG
    )


def _to_vec3(corners: list[list[float]]) -> list[Vec3]:
    return [
        Vec3(x=float(corner[0]), y=float(corner[1]), z=float(corner[2]))
        for corner in corners
    ]


def _xz_polygon(corners: list[list[float]]):
    if len(corners) < 3:
        return None
    poly = Polygon([(float(corner[0]), float(corner[2])) for corner in corners])
    if not poly.is_valid or poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _story_floor_unions(model: BuildingModel) -> dict[int, object]:
    floors_by_story: dict[int, list] = {}
    for room in model.rooms:
        floor = _xz_polygon(room.floor_polygon)
        if floor is None:
            continue
        floors_by_story.setdefault(room.story, []).append(floor)
    out: dict[int, object] = {}
    for story, floors in floors_by_story.items():
        try:
            union = unary_union(floors)
        except Exception:
            continue
        if not union.is_empty and union.area > 1e-9:
            out[story] = union
    return out


def _exterior_piece_is_internal(
    corners: list[list[float]],
    kind: GapKind,
    story: int | None,
    floor_unions_by_story: dict[int, object],
) -> bool:
    if story is None or kind not in _EXTERIOR_KIND_BY_TYPE.values():
        return False
    floor_union = floor_unions_by_story.get(story)
    if floor_union is None or floor_union.is_empty:
        return False
    if kind == GapKind.EXTERIOR_SIDE:
        if len(corners) < 2:
            return False
        geom = LineString(
            [
                (float(corners[0][0]), float(corners[0][2])),
                (float(corners[1][0]), float(corners[1][2])),
            ]
        )
        if geom.length <= 1e-9:
            return False
        try:
            overlap_fraction = (
                float(geom.intersection(floor_union).length) / geom.length
            )
            sample = geom.interpolate(0.5, normalized=True)
            boundary_distance = float(floor_union.boundary.distance(sample))
        except Exception:
            return False
    else:
        geom = _xz_polygon(corners)
        if geom is None:
            return False
        try:
            overlap_fraction = float(geom.intersection(floor_union).area) / max(
                float(geom.area), 1e-9
            )
            boundary_distance = float(floor_union.boundary.distance(geom.centroid))
        except Exception:
            return False
    return (
        overlap_fraction >= EXTERIOR_INTERNAL_OVERLAP_FRACTION
        and boundary_distance >= EXTERIOR_INTERNAL_MIN_BOUNDARY_DISTANCE_M
    )


def _horizontal_plane_key(corners: list[list[float]]) -> int:
    mean_y = sum(float(corner[1]) for corner in corners) / len(corners)
    return round(mean_y / HORIZONTAL_PLANE_BUCKET_M)


def _project_to_global_plane_polygon(corners: list[list[float]]):
    import numpy as _np
    from shapely.geometry import Polygon as _Polygon

    plane = fit_plane_any(corners)
    if plane is None:
        return None, None
    n = _np.asarray(plane[:3], dtype=float)
    norm = float(_np.linalg.norm(n))
    if norm <= 1e-12:
        return None, None
    n = n / norm
    d = float(plane[3]) / norm
    dominant = int(_np.argmax(_np.abs(n)))
    if n[dominant] < 0.0:
        n = -n
        d = -d
    axes = [axis for axis in range(3) if axis != dominant]
    pts = _np.asarray(corners, dtype=float)
    flat = [(float(p[axes[0]]), float(p[axes[1]])) for p in pts]
    try:
        poly = _Polygon(flat)
    except Exception:
        return None, None
    if not poly.is_valid or poly.is_empty or poly.area <= 1e-9:
        return None, None
    key = (
        round(float(n[0]) / 0.005),
        round(float(n[1]) / 0.005),
        round(float(n[2]) / 0.005),
        round(d / 0.005),
        dominant,
    )
    return key, poly


def _has_substantial_overlap(poly, claimed) -> bool:
    for prev in claimed:
        try:
            overlap = float(poly.intersection(prev).area)
        except Exception:
            continue
        if overlap > GAP_PIECE_OVERLAP_M2:
            return True
    return False


def _is_mostly_covered_by_claimed_cap(poly, claimed) -> bool:
    for prev in claimed:
        try:
            overlap = float(poly.intersection(prev).area)
        except Exception:
            continue
        if (
            overlap > GAP_PIECE_OVERLAP_M2
            and overlap / max(float(poly.area), 1e-9) >= CAP_DUPLICATE_OVERLAP_FRACTION
        ):
            return True
    return False


def _wall_edge_lines(model: BuildingModel) -> list[LineString]:
    lines: list[LineString] = []
    for room in model.rooms:
        for wall in room.walls_computed:
            corners = wall.corners
            if len(corners) < 2:
                continue
            for idx, corner in enumerate(corners):
                nxt = corners[(idx + 1) % len(corners)]
                line = LineString(
                    [
                        (float(corner[0]), float(corner[2])),
                        (float(nxt[0]), float(nxt[2])),
                    ]
                )
                if line.length > 1e-6:
                    lines.append(line)
    return lines


def _stitch_overlaps_wall_run(
    corners: list[list[float]], wall_lines: list[LineString]
) -> bool:
    if len(corners) < 2:
        return True
    line = LineString(
        [
            (float(corners[0][0]), float(corners[0][2])),
            (float(corners[1][0]), float(corners[1][2])),
        ]
    )
    if line.length <= 1e-9:
        return True
    for wall_line in wall_lines:
        overlap_len = line.intersection(
            wall_line.buffer(WALL_OVERLAP_BUFFER_M, cap_style="flat")
        ).length
        if (
            overlap_len > WALL_OVERLAP_MIN_M
            and overlap_len / line.length > WALL_OVERLAP_MAX_RATIO
        ):
            return True
    return False


def _is_sliver(corners: list[list[float]]) -> bool:
    if len(corners) < 3:
        return True
    import numpy as _np
    from shapely.geometry import Polygon as _Polygon

    plane = fit_plane_any(corners)
    if plane is None:
        return True
    n = _np.asarray(plane[:3], dtype=float)
    norm = float(_np.linalg.norm(n))
    if norm <= 1e-12:
        return True
    n = n / norm
    helper = (
        _np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else _np.array([0.0, 1.0, 0.0])
    )
    u = _np.cross(n, helper)
    u_norm = float(_np.linalg.norm(u))
    if u_norm <= 1e-12:
        return True
    u = u / u_norm
    v = _np.cross(n, u)
    pts = _np.asarray(corners, dtype=float)
    flat = [(float(_np.dot(p, u)), float(_np.dot(p, v))) for p in pts]
    try:
        poly = _Polygon(flat)
        if not poly.is_valid or poly.is_empty:
            return True
        sides = oriented_rectangle_side_lengths(poly)
        if not sides:
            return True
        return min(sides) < GAP_PIECE_MIN_THICKNESS_M
    except Exception:
        return True


def _stitches_covered_by_side_closures(
    model: BuildingModel, tol_m: float = 0.20
) -> set[int]:
    """Stitches whose XZ endpoints coincide with an exterior side-closure's
    bottom edge are redundant: the closure already bridges that vertical gap."""

    sides_by_story: dict[
        int, list[tuple[tuple[float, float], tuple[float, float]]]
    ] = {}
    for closure in model.gap_closures:
        if closure.type != "side" or len(closure.corners) < 2:
            continue
        e0 = (float(closure.corners[0][0]), float(closure.corners[0][2]))
        e1 = (float(closure.corners[1][0]), float(closure.corners[1][2]))
        sides_by_story.setdefault(closure.story, []).append((e0, e1))
    if not sides_by_story:
        return set()

    covered: set[int] = set()
    tol2 = tol_m * tol_m
    for idx, stitch in enumerate(model.stitch_walls):
        if stitch.type != "stitch" or len(stitch.corners) < 2:
            continue
        sx0 = (float(stitch.corners[0][0]), float(stitch.corners[0][2]))
        sx1 = (float(stitch.corners[1][0]), float(stitch.corners[1][2]))
        for e0, e1 in sides_by_story.get(stitch.story, []):
            d_aa = (sx0[0] - e0[0]) ** 2 + (sx0[1] - e0[1]) ** 2
            d_bb = (sx1[0] - e1[0]) ** 2 + (sx1[1] - e1[1]) ** 2
            d_ab = (sx0[0] - e1[0]) ** 2 + (sx0[1] - e1[1]) ** 2
            d_ba = (sx1[0] - e0[0]) ** 2 + (sx1[1] - e0[1]) ** 2
            if (d_aa < tol2 and d_bb < tol2) or (d_ab < tol2 and d_ba < tol2):
                covered.add(idx)
                break
    return covered


def _dropped_stitch_indices(
    model: BuildingModel, footprint_by_story: dict[int, dict]
) -> set[int]:
    """Pre-compute which stitch indices to drop, expanded along closure groups.

    The L-stitch closure (two legs + floor cap + ceiling cap) is atomic for
    rendering: keeping a cap or single leg without its siblings produces a
    half-open L hanging in space. If any leg fails footprint support, drop the
    entire closure group."""

    stitches = model.stitch_walls
    if not stitches:
        return set()
    failed: set[int] = set()
    for idx, stitch in enumerate(stitches):
        if stitch.type != "stitch":
            continue
        ctx = footprint_by_story.get(stitch.story)
        if is_stitch_edge_unsupported_by_story_footprint(stitch.corners, ctx):
            failed.add(idx)
    failed.update(_stitches_covered_by_side_closures(model))
    if not failed:
        return set()
    groups = stitch_closure_groups(stitches)
    dropped: set[int] = set()
    for idx in failed:
        dropped.update(groups[idx])
    return dropped


def assemble_gap_pieces(model: BuildingModel) -> list[GapPiece]:
    pieces: list[GapPiece] = []
    room_floors_by_y: dict[float, list] = {}
    for room in model.rooms:
        floor = _xz_polygon(room.floor_polygon)
        if floor is None:
            continue
        floor_y = _horizontal_plane_key(room.floor_polygon)
        room_floors_by_y.setdefault(floor_y, []).append(floor)

    claimed_side_polys: dict[tuple[GapKind, object], list[tuple[object, int]]] = {}
    claimed_cap_polys: dict[float, list] = {}
    dropped_piece_indices: set[int] = set()
    wall_lines = _wall_edge_lines(model)
    footprint_by_story = story_footprint_context(model)
    floor_unions_by_story = _story_floor_unions(model)
    dropped_stitch_indices = _dropped_stitch_indices(model, footprint_by_story)

    def append_piece(
        corners: list[list[float]],
        kind: GapKind,
        scope: GapScope,
        locator_id: str,
        story: int | None = None,
    ) -> None:
        if _is_sliver(corners):
            return
        if _exterior_piece_is_internal(corners, kind, story, floor_unions_by_story):
            return
        is_sloped_ceiling = kind in {
            GapKind.GAP_CEILING,
            GapKind.EXTERIOR_CEIL,
        } and _is_sloped_ceiling_cap(corners)
        if kind in _HORIZONTAL_KINDS and not is_sloped_ceiling:
            cap_poly = _xz_polygon(corners)
            cap_y = _horizontal_plane_key(corners)
            if (
                kind != GapKind.EXTERIOR_FLOOR
                and cap_poly is not None
                and _has_substantial_overlap(cap_poly, room_floors_by_y.get(cap_y, []))
            ):
                return
            if cap_poly is not None:
                if _is_mostly_covered_by_claimed_cap(
                    cap_poly, claimed_cap_polys.get(cap_y, [])
                ):
                    return
                claimed_cap_polys.setdefault(cap_y, []).append(cap_poly)
        elif kind in _SIDE_OWNERSHIP_KINDS:
            if kind == GapKind.STITCH and _stitch_overlaps_wall_run(
                corners, wall_lines
            ):
                return
            key, side_poly = _project_to_global_plane_polygon(corners)
            if key is not None and side_poly is not None:
                claim_key = (GapKind.SIDE, key)
                retained_claims = []
                for claimed_poly, piece_idx in claimed_side_polys.get(claim_key, []):
                    try:
                        overlap = float(side_poly.intersection(claimed_poly).area)
                    except Exception:
                        retained_claims.append((claimed_poly, piece_idx))
                        continue
                    if overlap <= GAP_PIECE_OVERLAP_M2:
                        retained_claims.append((claimed_poly, piece_idx))
                        continue
                    current_is_duplicate = (
                        overlap / max(side_poly.area, 1e-9)
                        >= SIDE_DUPLICATE_OVERLAP_FRACTION
                    )
                    claimed_is_duplicate = (
                        overlap / max(claimed_poly.area, 1e-9)
                        >= SIDE_DUPLICATE_OVERLAP_FRACTION
                    )
                    if (
                        current_is_duplicate
                        and claimed_is_duplicate
                        and kind == GapKind.EXTERIOR_SIDE
                        and pieces[piece_idx].kind != GapKind.EXTERIOR_SIDE
                        and side_poly.area >= claimed_poly.area
                    ):
                        dropped_piece_indices.add(piece_idx)
                        continue
                    if current_is_duplicate:
                        return
                    if claimed_is_duplicate:
                        dropped_piece_indices.add(piece_idx)
                        continue
                    retained_claims.append((claimed_poly, piece_idx))
                claimed_side_polys[claim_key] = retained_claims
                claimed_side_polys[claim_key].append((side_poly, len(pieces)))
        pieces.append(
            GapPiece(
                corners=_to_vec3(corners),
                kind=kind,
                scope=scope,
                locator_id=locator_id,
            )
        )

    for wall in model.gap_walls:
        kind = _GAP_KIND_BY_TYPE.get(wall.type)
        if kind is None:
            continue
        corners = wall.corners
        if kind == GapKind.GAP_CEILING and _is_sloped_ceiling_cap(corners):
            corners = _wind_upward(corners)
        elif kind in _HORIZONTAL_KINDS:
            corners = _planarize_horizontal(corners, kind)
        append_piece(
            corners,
            kind,
            GapScope.INTRA_STORY,
            f"{model.uuid}::tier-gap::{wall.id}",
            wall.story,
        )

    for idx, stitch in enumerate(model.stitch_walls):
        if idx in dropped_stitch_indices:
            continue
        kind = _STITCH_KIND_BY_TYPE.get(stitch.type)
        if kind is None:
            continue
        corners = stitch.corners
        if kind in _HORIZONTAL_KINDS:
            corners = _planarize_horizontal(corners, kind)
        append_piece(
            corners,
            kind,
            GapScope.JUNCTION,
            f"{model.uuid}::tier-gap::{stitch.id}",
            stitch.story,
        )

    for idx, closure in enumerate(model.gap_closures):
        kind = _EXTERIOR_KIND_BY_TYPE.get(closure.type)
        if kind is None:
            continue
        corners = closure.corners
        if kind == GapKind.EXTERIOR_CEIL and _is_sloped_ceiling_cap(corners):
            corners = _wind_upward(corners)
        elif kind in _HORIZONTAL_KINDS:
            corners = _planarize_horizontal(corners, kind)
        append_piece(
            corners,
            kind,
            GapScope.EXTERIOR,
            (
                f"{model.uuid}::tier-gap::"
                f"{closure.indicator_element_id}:{closure.indicator_wall_id}:{closure.type}:{idx}"
            ),
            closure.story,
        )
    return [
        piece for idx, piece in enumerate(pieces) if idx not in dropped_piece_indices
    ]
