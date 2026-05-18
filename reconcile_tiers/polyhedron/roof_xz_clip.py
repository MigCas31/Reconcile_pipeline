"""Clip ceiling/roof tiles to the room floor footprint in XZ.

Mis-assigned roof planes often extend beyond the room's horizontal boundary
while their centroid still lies inside the floor polygon. Intersect each
ceiling tile's XZ projection with the floor footprint, then lift the clipped
ring back onto the tile's supporting plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.manifold_repair import TileFace

CEILING_SOURCES = frozenset({"ceiling", "visual_shell", "gable_closure"})

# Area change below this ratio keeps original corners (numerical stability).
_UNCHANGED_AREA_RATIO = 0.98


@dataclass(frozen=True, slots=True)
class RoofXzClipResult:
    tiles: tuple[TileFace, ...]
    clipped_locator_ids: tuple[str, ...]
    dropped_locator_ids: tuple[str, ...]
    floor_area_m2: float | None


def floor_polygon_from_tiles(tiles: Sequence[TileFace]) -> Polygon | None:
    """Union of floor-tile footprints in XZ."""
    polys: list[Polygon] = []
    for tile in tiles:
        if tile.source != "floor":
            continue
        poly = _polygon_xz(tile.corners)
        if poly is not None:
            polys.append(poly)
    if not polys:
        return None
    merged = unary_union(polys)
    return _largest_polygon(merged)


def clip_roof_tiles_to_floor_xz(
    tiles: Sequence[TileFace],
    *,
    min_clip_area_m2: float = 0.02,
    unchanged_area_ratio: float = _UNCHANGED_AREA_RATIO,
) -> RoofXzClipResult:
    """Return tiles with ceiling/roof pieces clipped to the room floor XZ polygon."""
    floor_poly = floor_polygon_from_tiles(tiles)
    if floor_poly is None:
        return RoofXzClipResult(
            tiles=tuple(tiles),
            clipped_locator_ids=(),
            dropped_locator_ids=(),
            floor_area_m2=None,
        )

    out: list[TileFace] = []
    clipped_ids: list[str] = []
    dropped_ids: list[str] = []
    next_id = max((t.face_id for t in tiles), default=-1) + 1

    for tile in tiles:
        if tile.source not in CEILING_SOURCES:
            out.append(tile)
            continue

        tile_xz = _polygon_xz(tile.corners)
        if tile_xz is None:
            out.append(tile)
            continue

        orig_area = float(tile_xz.area)
        if orig_area <= 1e-9:
            dropped_ids.append(tile.locator_id)
            continue

        clipped_xz = tile_xz.intersection(floor_poly)
        clipped_poly = _largest_polygon(clipped_xz)
        if clipped_poly is None or float(clipped_poly.area) < min_clip_area_m2:
            dropped_ids.append(tile.locator_id)
            continue

        if float(clipped_poly.area) >= orig_area * unchanged_area_ratio:
            out.append(tile)
            continue

        ring = list(clipped_poly.exterior.coords)[:-1]
        new_corners = _corners_from_xz_ring(ring, tile.plane)
        if new_corners is None:
            out.append(tile)
            continue

        new_plane = _plane_from_corners(new_corners) or tile.plane
        out.append(
            TileFace(
                face_id=next_id,
                corners=new_corners,
                plane=new_plane,
                source=tile.source,
                locator_id=tile.locator_id,
                story=tile.story,
                room_index=tile.room_index,
            )
        )
        next_id += 1
        clipped_ids.append(tile.locator_id)

    return RoofXzClipResult(
        tiles=tuple(out),
        clipped_locator_ids=tuple(clipped_ids),
        dropped_locator_ids=tuple(dropped_ids),
        floor_area_m2=float(floor_poly.area),
    )


def footprint_edges_for_viewer(
    tiles: Sequence[TileFace],
    *,
    y: float | None = None,
) -> list[dict[str, list[float]]]:
    """Horizontal floor-outline segments for trace viewer overlays."""
    floor_poly = floor_polygon_from_tiles(tiles)
    if floor_poly is None:
        return []
    if y is None:
        wall_ys = [
            float(c[1])
            for tile in tiles
            if tile.source == "wall"
            for c in tile.corners
        ]
        floor_ys = [
            float(c[1])
            for tile in tiles
            if tile.source == "floor"
            for c in tile.corners
        ]
        if wall_ys:
            y = max(wall_ys)
        elif floor_ys:
            y = sum(floor_ys) / len(floor_ys)
        else:
            y = 0.0

    edges: list[dict[str, list[float]]] = []
    coords = list(floor_poly.exterior.coords)
    for i in range(len(coords) - 1):
        x0, z0 = coords[i]
        x1, z1 = coords[i + 1]
        edges.append(
            {
                "a": [float(x0), float(y), float(z0)],
                "b": [float(x1), float(y), float(z1)],
            }
        )
    return edges


def _polygon_xz(corners: Sequence[Sequence[float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-9:
        return None
    return poly


def _largest_polygon(geom: Any) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 1e-9 else None
    if isinstance(geom, MultiPolygon):
        best: Polygon | None = None
        best_area = 0.0
        for part in geom.geoms:
            if isinstance(part, Polygon) and part.area > best_area:
                best = part
                best_area = float(part.area)
        return best
    return None


def _y_on_plane(plane: Plane, x: float, z: float) -> float | None:
    b = float(plane.b)
    if abs(b) < 1e-6:
        return None
    return (float(plane.d) - float(plane.a) * x - float(plane.c) * z) / b


def _corners_from_xz_ring(
    ring_xz: Sequence[Sequence[float]],
    plane: Plane,
) -> tuple[tuple[float, float, float], ...] | None:
    corners: list[tuple[float, float, float]] = []
    for pt in ring_xz:
        x = float(pt[0])
        z = float(pt[1])
        y = _y_on_plane(plane, x, z)
        if y is None:
            return None
        corners.append((x, y, z))
    cleaned = _dedupe_ring(corners)
    if len(cleaned) < 3:
        return None
    return tuple(cleaned)


def _dedupe_ring(
    corners: list[tuple[float, float, float]],
    tol: float = 1e-4,
) -> list[tuple[float, float, float]]:
    if not corners:
        return []
    out: list[tuple[float, float, float]] = [corners[0]]
    for c in corners[1:]:
        prev = out[-1]
        if (
            abs(c[0] - prev[0]) <= tol
            and abs(c[1] - prev[1]) <= tol
            and abs(c[2] - prev[2]) <= tol
        ):
            continue
        out.append(c)
    if len(out) >= 2 and _coord_dist(out[0], out[-1]) <= tol:
        out.pop()
    return out


def _coord_dist(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return float(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
    )


def _plane_from_corners(
    corners: Sequence[Sequence[float]],
) -> Plane | None:
    if len(corners) < 3:
        return None
    nx = ny = nz = 0.0
    n = len(corners)
    for i in range(n):
        c = corners[i]
        nxt = corners[(i + 1) % n]
        nx += (float(c[1]) - float(nxt[1])) * (float(c[2]) + float(nxt[2]))
        ny += (float(c[2]) - float(nxt[2])) * (float(c[0]) + float(nxt[0]))
        nz += (float(c[0]) - float(nxt[0])) * (float(c[1]) + float(nxt[1]))
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm <= 1e-12:
        return None
    nx /= norm
    ny /= norm
    nz /= norm
    cx = sum(float(c[0]) for c in corners) / n
    cy = sum(float(c[1]) for c in corners) / n
    cz = sum(float(c[2]) for c in corners) / n
    return Plane(a=nx, b=ny, c=nz, d=nx * cx + ny * cy + nz * cz)
