"""Wall / floor / ceiling splitting at adjacency boundaries.

`tag_payload` slices walls at terrain_y / deep-basement thresholds, splits
floors at the basement footprint boundary, and splits ceilings at oblique
outlines. Extracted from `adjacency.py`; the orchestrator + eval helpers
stay there. Public re-exports in `adjacency.py` keep imports stable.
"""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers.payload.schema import (
    AdjacencyKind,
    CeilingPiece,
    HorizontalLid,
    Quad,
    Room,
    Vec3,
    Wall,
)


def _is_unheated(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _is_unheated as _impl

    return _impl(*args, **kwargs)


def _mean_y(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _mean_y as _impl

    return _impl(*args, **kwargs)


def _eval_wall_tag_at(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _eval_wall_tag_at as _impl

    return _impl(*args, **kwargs)


def _eval_floor_tag_at(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _eval_floor_tag_at as _impl

    return _impl(*args, **kwargs)


def _eval_ceiling_tag_at(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _eval_ceiling_tag_at as _impl

    return _impl(*args, **kwargs)


def _xz_polygon(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _xz_polygon as _impl

    return _impl(*args, **kwargs)


def _xz_polygon_from_xyz(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _xz_polygon_from_xyz as _impl

    return _impl(*args, **kwargs)


def _xz_centroid(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _xz_centroid as _impl

    return _impl(*args, **kwargs)


def _polys_to_polygon_list(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _polys_to_polygon_list as _impl

    return _impl(*args, **kwargs)


def _polygon_to_xz_vec3(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _polygon_to_xz_vec3 as _impl

    return _impl(*args, **kwargs)


def _polygon_to_plane_vec3(*args, **kwargs):
    from reconcile_tiers.payload.adjacency import _polygon_to_plane_vec3 as _impl

    return _impl(*args, **kwargs)


def _split_wall(
    wall: Wall,
    host_room: Room,
    *,
    has_basement: bool,
    terrain_y: float,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
) -> list[tuple[list[Vec3], AdjacencyKind, tuple[float, float] | None]]:
    """Split a wall horizontally at terrain_y / terrain_y - 2 m for basement
    walls. Other splits (e.g. partition end) are deferred to v2 -- the
    centroid-based partition test in `_eval_wall_tag_at` already handles the
    common case where one side of the wall faces another room along its full
    length.

    Returns a list of `(corners_3d, adjacency, y_band)` tuples, one per
    uniform-tag horizontal slice. `y_band` is `(y_lo, y_hi)` for actual Y
    splits and `None` when the whole wall is emitted unchanged -- the caller
    uses this to clip cutouts to the piece's band.
    """
    if _is_unheated(host_room):
        return [(list(wall.corners), AdjacencyKind.INTERNAL_TO_UNHEATED_HOST, None)]

    if len(wall.corners) < 3:
        return [(list(wall.corners), AdjacencyKind.UNKNOWN, None)]

    if not (host_room.story == 0 and has_basement):
        return [
            _tag_whole_wall(
                wall,
                host_room,
                has_basement=has_basement,
                terrain_y=terrain_y,
                room_polys_by_story=room_polys_by_story,
            )
        ]

    from reconcile_tiers.payload.adjacency import BASEMENT_DEEP_THRESHOLD_M

    y_min = min(c.y for c in wall.corners)
    y_max = max(c.y for c in wall.corners)
    deep_threshold = terrain_y - BASEMENT_DEEP_THRESHOLD_M

    cuts = sorted(
        {
            cut
            for cut in (terrain_y, deep_threshold)
            if y_min + 1e-6 < cut < y_max - 1e-6
        }
    )
    if not cuts:
        return [
            _tag_whole_wall(
                wall,
                host_room,
                has_basement=has_basement,
                terrain_y=terrain_y,
                room_polys_by_story=room_polys_by_story,
            )
        ]

    coords = [[c.x, c.y, c.z] for c in wall.corners]
    nx, _ny, nz = newell_normal(coords)
    nxz_len = (nx * nx + nz * nz) ** 0.5
    if nxz_len <= 1e-9:
        return [
            _tag_whole_wall(
                wall,
                host_room,
                has_basement=has_basement,
                terrain_y=terrain_y,
                room_polys_by_story=room_polys_by_story,
            )
        ]

    # Slice the wall by horizontal Y planes only. The polygon stays in its
    # original 3D plane -- we just intersect with horizontal half-spaces.
    bands = [(y_min, cuts[0])]
    for i in range(len(cuts) - 1):
        bands.append((cuts[i], cuts[i + 1]))
    bands.append((cuts[-1], y_max))

    pieces: list[tuple[list[Vec3], AdjacencyKind, tuple[float, float] | None]] = []
    for y_lo, y_hi in bands:
        if y_hi - y_lo < 1e-3:
            continue
        sliced = _slice_wall_by_y_band(wall.corners, y_lo, y_hi)
        if sliced is None or len(sliced) < 3:
            continue
        cx = sum(c.x for c in sliced) / len(sliced)
        cz = sum(c.z for c in sliced) / len(sliced)
        cy = (y_lo + y_hi) / 2.0
        tag = _eval_wall_tag_at(
            x=cx,
            z=cz,
            y=cy,
            host_room=host_room,
            has_basement=has_basement,
            terrain_y=terrain_y,
            room_polys_by_story=room_polys_by_story,
        )
        pieces.append((sliced, tag, (y_lo, y_hi)))

    if not pieces:
        return [
            _tag_whole_wall(
                wall,
                host_room,
                has_basement=has_basement,
                terrain_y=terrain_y,
                room_polys_by_story=room_polys_by_story,
            )
        ]
    return pieces


def _slice_wall_by_y_band(
    corners: list[Vec3], y_lo: float, y_hi: float
) -> list[Vec3] | None:
    """Sutherland-Hodgman clip a wall polygon to the horizontal band
    [y_lo, y_hi]. Preserves the wall's 3D plane (X and Z are interpolated
    from the original edge). Returns None if the result is degenerate.
    """
    if not corners:
        return None
    poly = list(corners)
    poly = _clip_against_y(poly, y_lo, keep_above=True)
    if not poly:
        return None
    poly = _clip_against_y(poly, y_hi, keep_above=False)
    return poly if len(poly) >= 3 else None


def _clip_cutouts_to_band(
    cutouts: list[Quad], band: tuple[float, float] | None
) -> list[Quad]:
    """Clip each cutout quad to the horizontal Y band of its host wall piece.
    Drops cutouts that fall fully outside the band; clips the rest with the
    same plane-preserving Sutherland-Hodgman as the wall outline. `band=None`
    means the wall wasn't Y-split, so cutouts are passed through.

    Cutout corners can differ from the band boundary by a few ULPs (one corner
    is exactly `y_lo`, its sibling is off by ~1e-15) -- Sutherland-Hodgman then
    inserts a near-degenerate edge and produces a duplicate vertex. The result
    must be a 4-corner Quad to satisfy the schema, so we dedup consecutive
    duplicates and drop cutouts that don't end up as proper quads.
    """
    if band is None:
        return list(cutouts)
    y_lo, y_hi = band
    tol = 1e-3
    clipped: list[Quad] = []
    for quad in cutouts:
        cy_min = min(c.y for c in quad.corners)
        cy_max = max(c.y for c in quad.corners)
        if cy_max <= y_lo + tol or cy_min >= y_hi - tol:
            continue
        if cy_min >= y_lo - tol and cy_max <= y_hi + tol:
            clipped.append(quad)
            continue
        sliced = _slice_wall_by_y_band(quad.corners, y_lo, y_hi)
        if sliced is None or len(sliced) < 3:
            continue
        sliced = _dedup_consecutive_vec3(sliced)
        if len(sliced) != 4:
            continue
        clipped.append(Quad(corners=sliced))
    return clipped


def _dedup_consecutive_vec3(corners: list[Vec3], *, eps: float = 1e-6) -> list[Vec3]:
    """Remove consecutive (and wrap-around) duplicate vertices within eps."""
    if not corners:
        return []
    out: list[Vec3] = [corners[0]]
    for c in corners[1:]:
        last = out[-1]
        if (
            abs(c.x - last.x) < eps
            and abs(c.y - last.y) < eps
            and abs(c.z - last.z) < eps
        ):
            continue
        out.append(c)
    if len(out) > 1:
        first = out[0]
        last = out[-1]
        if (
            abs(first.x - last.x) < eps
            and abs(first.y - last.y) < eps
            and abs(first.z - last.z) < eps
        ):
            out.pop()
    return out


def _clip_against_y(
    corners: list[Vec3], y_cut: float, *, keep_above: bool
) -> list[Vec3]:
    if not corners:
        return []
    out: list[Vec3] = []
    n = len(corners)
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        a_in = (a.y >= y_cut) if keep_above else (a.y <= y_cut)
        b_in = (b.y >= y_cut) if keep_above else (b.y <= y_cut)
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (y_cut - a.y) / (b.y - a.y) if abs(b.y - a.y) > 1e-12 else 0.0
            out.append(
                Vec3(
                    a.x + t * (b.x - a.x),
                    y_cut,
                    a.z + t * (b.z - a.z),
                )
            )
    return out


def _tag_whole_wall(
    wall: Wall,
    host_room: Room,
    *,
    has_basement: bool,
    terrain_y: float,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
) -> tuple[list[Vec3], AdjacencyKind, None]:
    cx = sum(c.x for c in wall.corners) / len(wall.corners)
    cz = sum(c.z for c in wall.corners) / len(wall.corners)
    cy = _mean_y(wall.corners)
    tag = _eval_wall_tag_at(
        x=cx,
        z=cz,
        y=cy,
        host_room=host_room,
        has_basement=has_basement,
        terrain_y=terrain_y,
        room_polys_by_story=room_polys_by_story,
    )
    return list(wall.corners), tag, None


def _split_floor(
    floor: HorizontalLid,
    host_room: Room,
    *,
    has_basement: bool,
    room_polys_by_story: dict[int, list[tuple[Polygon, Room]]],
) -> list[tuple[list[Vec3], AdjacencyKind]]:
    """Split a floor lid where its XZ footprint crosses an exposure boundary.
    Currently the only boundary is "above an unheated basement vs not" for
    story-1 floors of buildings with a basement.
    """
    if _is_unheated(host_room):
        return [(list(floor.corners), AdjacencyKind.INTERNAL_TO_UNHEATED_HOST)]

    floor_poly = _xz_polygon(floor.corners)
    if floor_poly is None:
        return [(list(floor.corners), AdjacencyKind.UNKNOWN)]
    y = _mean_y(floor.corners)

    # Only story-1 floors above a basement need geometric splitting; every
    # other case is uniform.
    if not (host_room.story == 1 and has_basement):
        rep = floor_poly.representative_point()
        tag = _eval_floor_tag_at(
            x=rep.x,
            z=rep.y,
            host_room=host_room,
            has_basement=has_basement,
            room_polys_by_story=room_polys_by_story,
        )
        return [(list(floor.corners), tag)]

    basement_polys = [poly for poly, _room in room_polys_by_story.get(0, [])]
    if not basement_polys:
        rep = floor_poly.representative_point()
        tag = _eval_floor_tag_at(
            x=rep.x,
            z=rep.y,
            host_room=host_room,
            has_basement=has_basement,
            room_polys_by_story=room_polys_by_story,
        )
        return [(list(floor.corners), tag)]

    basement_union = unary_union(basement_polys)

    pieces: list[tuple[list[Vec3], AdjacencyKind]] = []
    try:
        over_basement = floor_poly.intersection(basement_union)
        outside_basement = floor_poly.difference(basement_union)
    except Exception:
        rep = floor_poly.representative_point()
        tag = _eval_floor_tag_at(
            x=rep.x,
            z=rep.y,
            host_room=host_room,
            has_basement=has_basement,
            room_polys_by_story=room_polys_by_story,
        )
        return [(list(floor.corners), tag)]

    for region in (over_basement, outside_basement):
        for sub in _polys_to_polygon_list(region):
            rep = sub.representative_point()
            tag = _eval_floor_tag_at(
                x=rep.x,
                z=rep.y,
                host_room=host_room,
                has_basement=has_basement,
                room_polys_by_story=room_polys_by_story,
            )
            pieces.append((_polygon_to_xz_vec3(sub, y), tag))

    if not pieces:
        rep = floor_poly.representative_point()
        tag = _eval_floor_tag_at(
            x=rep.x,
            z=rep.y,
            host_room=host_room,
            has_basement=has_basement,
            room_polys_by_story=room_polys_by_story,
        )
        return [(list(floor.corners), tag)]
    return pieces


def _split_ceiling(
    piece: CeilingPiece,
    host_room: Room | None,
    *,
    top_story_idx: int,
    oblique_xz_polys: list[Polygon],
) -> list[tuple[list[Vec3], AdjacencyKind]]:
    """Split a ceiling piece where its XZ footprint crosses an oblique outline,
    so the under-sky portion gets `EXTERNAL_AIR` and the under-attic portion
    gets `UNHEATED_ATTIC` for flat-lid sources.

    Lower-story ceilings and sloped-source ceilings have a uniform tag, so we
    return a single piece for those.
    """
    from reconcile_tiers.payload.adjacency import FLAT_LID_SOURCES

    if host_room is not None and _is_unheated(host_room):
        return [(list(piece.corners), AdjacencyKind.INTERNAL_TO_UNHEATED_HOST)]
    if host_room is not None and host_room.story < top_story_idx:
        return [(list(piece.corners), AdjacencyKind.INTERNAL_TO_HEATED)]
    if piece.source not in FLAT_LID_SOURCES:
        # Sloped or unknown sources are uniformly external air at top story.
        return [(list(piece.corners), AdjacencyKind.EXTERNAL_AIR)]
    if not oblique_xz_polys:
        return [(list(piece.corners), AdjacencyKind.EXTERNAL_AIR)]

    ceiling_poly = _xz_polygon(piece.corners)
    if ceiling_poly is None:
        return [(list(piece.corners), AdjacencyKind.UNKNOWN)]

    oblique_union = unary_union(oblique_xz_polys)

    try:
        under_attic = ceiling_poly.intersection(oblique_union)
        under_sky = ceiling_poly.difference(oblique_union)
    except Exception:
        return [(list(piece.corners), AdjacencyKind.EXTERNAL_AIR)]

    pieces: list[tuple[list[Vec3], AdjacencyKind]] = []
    for sub in _polys_to_polygon_list(under_attic):
        pieces.append(
            (_polygon_to_plane_vec3(sub, piece.plane), AdjacencyKind.UNHEATED_ATTIC)
        )
    for sub in _polys_to_polygon_list(under_sky):
        pieces.append(
            (_polygon_to_plane_vec3(sub, piece.plane), AdjacencyKind.EXTERNAL_AIR)
        )
    if not pieces:
        return [(list(piece.corners), AdjacencyKind.EXTERNAL_AIR)]
    return pieces


def _holes_contained_by_corners(
    piece: CeilingPiece, corners: list[Vec3]
) -> list[list[Vec3]]:
    main_poly = _xz_polygon(corners)
    if main_poly is None:
        return []
    holes: list[list[Vec3]] = []
    for hole in piece.holes:
        if len(hole) < 3:
            continue
        hole_poly = _xz_polygon(hole)
        if hole_poly is not None and main_poly.contains(hole_poly):
            holes.append(hole)
    return holes
