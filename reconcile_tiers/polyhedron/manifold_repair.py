"""Tile-based watertight envelope reconstruction from tier_payload.

Builds a half-edge polyhedron per room from tier_payload tiles, detects
orphan half-edges (manifold holes), and uses a PolyFit-style ILP to pick
the minimal filler set that closes the manifold via Geniet's
``FACE_CREATION_FROM_EDGE`` operator.

This is the v2 envelope path; replaces the synthesis-based logic that
lived in ``face_selection.py``. See ``IMPLEMENTATION_PLAN.md`` §3 for
the per-room and building-level flows, and §5 for the file-by-file
contract.

Increment 1 (this file's first version): the data classes, tile
collection, and the permissive half-edge builder. No filler logic yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdge,
    HalfEdgePolyhedron,
    Vertex,
)

__all__ = [
    "FillerCandidate",
    "FillerSelection",
    "HoleChain",
    "HoleChainExtraction",
    "RoomPolyhedronBuild",
    "RoomRepairResult",
    "TileFace",
    "apply_fillers",
    "build_room_polyhedron",
    "collect_room_tiles",
    "envelope_candidate_from_repair",
    "extract_hole_chains",
    "hypothesise_fillers",
    "hypothesise_neighbor_plane_extension_fillers",
    "hypothesise_original_tile_fillers",
    "hypothesise_single_face_fillers",
    "prepare_room_tiles",
    "repair_room",
    "select_fillers",
]


@dataclass(frozen=True, slots=True)
class TileFace:
    """A face anchored to a tier_payload tile.

    Tile faces are *forced selected* by the downstream ILP — they're the
    ground truth from `tier_payload`. Only synthetic fillers added later
    are variables in the ILP.
    """

    face_id: int
    corners: tuple[tuple[float, float, float], ...]
    plane: Plane
    source: str  # "floor" | "wall" | "ceiling" | "visual_shell" | "gable_closure"
    locator_id: str
    story: int | None = None
    room_index: int | None = None


@dataclass(slots=True)
class RoomPolyhedronBuild:
    """Result of building a half-edge polyhedron from a room's tiles."""

    poly: HalfEdgePolyhedron
    # Map polyhedron Face.id → originating TileFace (or None for filler).
    tile_face_by_id: dict[int, TileFace] = field(default_factory=dict)
    orphan_half_edges: list[HalfEdge] = field(default_factory=list)
    skipped_tiles: list[tuple[TileFace, str]] = field(default_factory=list)
    # Vertex_id → original (x, y, z) from the tile corner cluster centroid.
    # Needed because `HalfEdgePolyhedron.vertex_position` requires 3 incident
    # faces (3-plane intersection); hole-boundary vertices may have only 2.
    vertex_coords: dict[int, tuple[float, float, float]] = field(
        default_factory=dict
    )


def collect_room_tiles(
    payload: Mapping[str, Any],
    room: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    next_face_id_start: int = 0,
) -> list[TileFace]:
    """Collect tier_payload tiles belonging to one room.

    Returns: the room's floor + walls (per-room) + ceiling tiles whose XZ
    centroid lies within the room's floor footprint (filtered from
    `payload.ceiling[]`, `payload.visual_shells[]`, `payload.gable_closures[]`).

    The room's `story` (`room.get("story")`) is propagated to each tile.

    Per ``IMPLEMENTATION_PLAN.md`` §Objective: visualization stays anchored
    to tier_payload's tiles — we do NOT invent new ceiling geometry. The
    tile corners are emitted verbatim; downstream consumers handle topology
    repair via `apply_fillers`, never by reshaping these polygons.
    """
    out: list[TileFace] = []
    next_id = next_face_id_start
    story = room.get("story")
    room_index = _room_index(room)

    floor_corners_xz: list[tuple[float, float]] = []
    for floor in room.get("floor", []) or []:
        corners = _corners_3d(floor.get("corners") or [])
        if len(corners) < 3:
            continue
        plane = _plane_from_dict(floor.get("plane")) or _plane_from_corners(corners)
        if plane is None:
            continue
        out.append(
            TileFace(
                face_id=next_id,
                corners=corners,
                plane=plane,
                source="floor",
                locator_id=str(
                    floor.get("locator_id")
                    or f"room:{room_index}:floor:{len(out)}"
                ),
                story=story,
                room_index=room_index,
            )
        )
        next_id += 1
        floor_corners_xz.extend((c[0], c[2]) for c in corners)

    if not floor_corners_xz:
        return out

    try:
        floor_poly = Polygon(floor_corners_xz)
        if not floor_poly.is_valid:
            floor_poly = floor_poly.buffer(0)
    except Exception:
        floor_poly = None

    wall_y_max: float | None = None
    wall_y_min: float | None = None
    for w_idx, wall in enumerate(room.get("walls") or []):
        corners = _corners_3d(wall.get("corners") or [])
        if len(corners) < 3:
            continue
        plane = _plane_from_dict(wall.get("plane")) or _plane_from_corners(corners)
        if plane is None:
            continue
        for c in corners:
            y = c[1]
            wall_y_max = y if wall_y_max is None else max(wall_y_max, y)
            wall_y_min = y if wall_y_min is None else min(wall_y_min, y)
        out.append(
            TileFace(
                face_id=next_id,
                corners=corners,
                plane=plane,
                source="wall",
                locator_id=str(
                    wall.get("locator_id")
                    or f"room:{room_index}:wall:{w_idx}"
                ),
                story=story,
                room_index=room_index,
            )
        )
        next_id += 1

    # Y-range that a ceiling for THIS room may sit in. Without a Y filter,
    # `payload.ceiling[]` tiles whose XZ centroid falls inside the room
    # footprint are assigned to EVERY storey beneath them — story 0
    # inherits story 2's roof slopes etc. Constrain to ceilings near or
    # slightly above the room's wall tops (with a half-storey margin for
    # gable roofs that rise above eave height).
    ceiling_y_low = (wall_y_max - 0.20) if wall_y_max is not None else None
    ceiling_y_high = (wall_y_max + 4.00) if wall_y_max is not None else None

    for kind, source_label in (
        ("ceiling", "ceiling"),
        ("visual_shells", "visual_shell"),
        ("gable_closures", "gable_closure"),
    ):
        for tile in payload.get(kind) or []:
            corners = _corners_3d(tile.get("corners") or [])
            if len(corners) < 3:
                continue
            if floor_poly is not None and not _xz_centroid_in_polygon(
                corners, floor_poly
            ):
                continue
            if ceiling_y_low is not None and ceiling_y_high is not None:
                ys = [c[1] for c in corners]
                tile_y_min = min(ys)
                tile_y_max = max(ys)
                if tile_y_max < ceiling_y_low or tile_y_min > ceiling_y_high:
                    continue
            plane = _plane_from_dict(tile.get("plane")) or _plane_from_corners(
                corners
            )
            if plane is None:
                continue
            out.append(
                TileFace(
                    face_id=next_id,
                    corners=corners,
                    plane=plane,
                    source=source_label,
                    locator_id=str(
                        tile.get("locator_id")
                        or f"room:{room_index}:{source_label}:{len(out)}"
                    ),
                    story=story,
                    room_index=room_index,
                )
            )
            next_id += 1

    return out


def build_room_polyhedron(
    tiles: Sequence[TileFace],
    *,
    coord_tol: float = 1e-3,
    snap_tol: float = 0.0,
    merge_coplanar: bool = True,
) -> RoomPolyhedronBuild:
    """Permissive half-edge builder.

    Like `half_edge.build_from_planar_polygons` but:
    - Does NOT raise on unpaired half-edges. Orphans are returned in the
      result for downstream hole detection.
    - Skips and reports degenerate tiles (zero-area, fewer than 3 corners,
      duplicate consecutive corners) instead of failing the whole build.
    - Skips the "≥3 incident faces per vertex" check (tile-built polyhedra
      can have 2-incidence vertices at hole boundaries before fillers
      are added).
    - When ``merge_coplanar=True`` (default), adjacent coplanar tiles are
      merged into one face before building. This is required by the
      Geniet half-edge framework: every vertex must be a true 3-plane
      corner. Without merging, ``tier_payload``'s multi-tile flat
      ceilings produce 2-incidence vertices that fail strict validation.
    - Per plan §7 risk mitigation, a union-find pre-pass snaps tile
      corners that drift between 1 mm and ``snap_tol`` (default 5 cm)
      to a shared cluster centroid. tier_payload's floor polygons
      typically include collinear-on-wall scan-noise corners that
      drift several cm from the wall's actual endpoint. Without
      snapping they end up as separate vertices and the half-edge
      build cannot twin-pair the wall edge with the floor's
      subdivided edge — yielding huge orphan-edge chains and oversized
      fillers (e.g., 60+ m² fillers in a 9 m² room).
    """
    if snap_tol > 0:
        tiles = _snap_tile_corners(tiles, snap_tol=snap_tol)
    if merge_coplanar:
        tiles = list(_merge_coplanar_tiles(tiles, coord_tol=coord_tol))
    poly = HalfEdgePolyhedron()
    result = RoomPolyhedronBuild(poly=poly)

    centroids: list[np.ndarray] = []

    def assign_vertex(point: tuple[float, float, float]) -> int:
        coord = np.asarray(point, dtype=float)
        for vid, c in enumerate(centroids):
            if float(np.linalg.norm(coord - c)) <= coord_tol:
                return vid
        centroids.append(coord)
        return len(centroids) - 1

    valid_tiles: list[TileFace] = []
    polygon_vids: list[list[int]] = []
    for tile in tiles:
        if len(tile.corners) < 3:
            result.skipped_tiles.append((tile, "fewer_than_3_corners"))
            continue
        vids = [assign_vertex(c) for c in tile.corners]
        if any(vids[i] == vids[(i + 1) % len(vids)] for i in range(len(vids))):
            result.skipped_tiles.append((tile, "duplicate_consecutive_corners"))
            continue
        if _polygon_area_3d(tile.corners) <= 1e-9:
            result.skipped_tiles.append((tile, "zero_area"))
            continue
        valid_tiles.append(tile)
        polygon_vids.append(vids)

    for vid in range(len(centroids)):
        poly.vertices.append(Vertex(id=vid))
        c = centroids[vid]
        result.vertex_coords[vid] = (float(c[0]), float(c[1]), float(c[2]))

    he_by_directed: dict[tuple[int, int], HalfEdge] = {}
    he_id = 0
    for face_idx, (tile, vids) in enumerate(
        zip(valid_tiles, polygon_vids, strict=True)
    ):
        # tier_payload winding isn't always consistent ("CCW from outside" is
        # not enforced upstream). Try the tile's given orientation first; if
        # any directed edge already exists in that direction, retry with the
        # ring reversed. Drop only if both orientations conflict.
        applied_he_id, success = _try_add_face(
            poly,
            tile,
            face_idx,
            vids,
            he_id,
            he_by_directed,
            result.tile_face_by_id,
        )
        if success:
            he_id = applied_he_id
            continue
        applied_he_id, success = _try_add_face(
            poly,
            tile,
            face_idx,
            list(reversed(vids)),
            he_id,
            he_by_directed,
            result.tile_face_by_id,
        )
        if success:
            he_id = applied_he_id
            continue
        result.skipped_tiles.append((tile, "duplicate_directed_edge_both_windings"))

    for (src, dst), he in he_by_directed.items():
        twin = he_by_directed.get((dst, src))
        if twin is None:
            result.orphan_half_edges.append(he)
            continue
        he.opposite = twin

    by_origin: dict[int, list[HalfEdge]] = {}
    for he in poly.half_edges:
        by_origin.setdefault(he.origin.id, []).append(he)
    for v in poly.vertices:
        outs = by_origin.get(v.id, [])
        if outs:
            v.outgoing = min(outs, key=lambda h: h.id)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _room_index(room: Mapping[str, Any]) -> int | None:
    locator = room.get("locator_id") or ""
    if isinstance(locator, str):
        marker = "::tier-room::"
        idx = locator.rfind(marker)
        if idx >= 0:
            tail = locator[idx + len(marker) :]
            try:
                return int(tail.split(":")[0])
            except (ValueError, IndexError):
                pass
    return None


def _corners_3d(
    raw: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float, float], ...]:
    out: list[tuple[float, float, float]] = []
    for c in raw:
        try:
            out.append((float(c["x"]), float(c["y"]), float(c["z"])))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def _plane_from_dict(raw: Mapping[str, Any] | None) -> Plane | None:
    if not raw:
        return None
    try:
        return Plane(
            a=float(raw["a"]),
            b=float(raw["b"]),
            c=float(raw["c"]),
            d=float(raw["d"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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


def _polygon_area_3d(corners: Sequence[Sequence[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    arr = np.asarray(corners, dtype=float)
    origin = arr[0]
    total = 0.0
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - origin
        v2 = arr[i + 1] - origin
        total += 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
    return total


def _xz_centroid_in_polygon(
    corners: Sequence[Sequence[float]],
    polygon: Polygon,
) -> bool:
    if not corners:
        return False
    cx = sum(float(c[0]) for c in corners) / len(corners)
    cz = sum(float(c[2]) for c in corners) / len(corners)
    try:
        from shapely.geometry import Point

        return polygon.contains(Point(cx, cz)) or polygon.touches(Point(cx, cz))
    except Exception:
        return False


def _clean_polygon_corners(
    corners: Sequence[Sequence[float]],
    *,
    coord_tol: float,
    collinear_tol: float = 1e-4,
) -> tuple[tuple[float, float, float], ...]:
    """Drop near-duplicate consecutive corners and collinear-with-neighbour
    corners from a polygon's corner ring.

    tier_payload polygons (especially auto-generated room floors) often
    have many near-duplicate vertices and collinear "intermediate"
    points along straight edges. They produce vertex_underconstrained
    defects in the half-edge framework because each interior vertex is
    incident to only 1-2 faces. Cleaning before build solves this.
    """
    if len(corners) < 3:
        return tuple(tuple(float(x) for x in c) for c in corners)
    pts = [tuple(float(x) for x in c) for c in corners]
    deduped: list[tuple[float, float, float]] = []
    for c in pts:
        if deduped and _coord_distance(c, deduped[-1]) <= coord_tol:
            continue
        deduped.append(c)
    while len(deduped) >= 2 and _coord_distance(deduped[0], deduped[-1]) <= coord_tol:
        deduped.pop()
    if len(deduped) < 3:
        return tuple(deduped)

    final: list[tuple[float, float, float]] = []
    n = len(deduped)
    for i in range(n):
        prev_p = deduped[(i - 1) % n]
        cur_p = deduped[i]
        next_p = deduped[(i + 1) % n]
        a = np.asarray(prev_p, dtype=float)
        b = np.asarray(cur_p, dtype=float)
        c = np.asarray(next_p, dtype=float)
        ab = b - a
        bc = c - b
        len_ab = float(np.linalg.norm(ab))
        len_bc = float(np.linalg.norm(bc))
        if len_ab <= 1e-12 or len_bc <= 1e-12:
            continue
        cross = float(np.linalg.norm(np.cross(ab, bc)))
        if cross / (len_ab * len_bc) <= collinear_tol:
            continue
        final.append(cur_p)
    if len(final) < 3:
        return tuple(deduped)
    return tuple(final)


def _coord_distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return float(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
    )


def _snap_tile_corners(
    tiles: Sequence[TileFace],
    *,
    snap_tol: float,
) -> list[TileFace]:
    """Union-find snap of corners across all tiles in a room.

    Collects every corner from every tile, clusters them using the given
    tolerance, and rewrites each tile's corner ring using the cluster's
    centroid as the canonical position. This makes corners that drifted
    1–50 mm between tiles share a single vertex in the half-edge build
    instead of producing parallel orphan edges.

    Subsequent ``_clean_polygon_corners`` removes any ring entries that
    collapse to consecutive duplicates after snapping.
    """
    if not tiles:
        return list(tiles)

    pts: list[tuple[float, float, float]] = []
    for tile in tiles:
        pts.extend(tile.corners)
    n = len(pts)
    if n == 0:
        return list(tiles)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    arr = np.array(pts, dtype=float)
    tol_sq = snap_tol * snap_tol
    for i in range(n):
        diffs = arr - arr[i]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        for j in np.flatnonzero(d2 <= tol_sq):
            j_int = int(j)
            if j_int != i:
                union(i, j_int)

    cluster_pts: dict[int, list[int]] = {}
    for i in range(n):
        cluster_pts.setdefault(find(i), []).append(i)
    centroid_by_root: dict[int, tuple[float, float, float]] = {}
    for root, members in cluster_pts.items():
        c = arr[members].mean(axis=0)
        centroid_by_root[root] = (float(c[0]), float(c[1]), float(c[2]))

    out: list[TileFace] = []
    cursor = 0
    for tile in tiles:
        new_corners: list[tuple[float, float, float]] = []
        for _ in tile.corners:
            new_corners.append(centroid_by_root[find(cursor)])
            cursor += 1
        out.append(
            TileFace(
                face_id=tile.face_id,
                corners=tuple(new_corners),
                plane=tile.plane,
                source=tile.source,
                locator_id=tile.locator_id,
                story=tile.story,
                room_index=tile.room_index,
            )
        )
    return out


def _merge_coplanar_tiles(
    tiles: Sequence[TileFace],
    *,
    coord_tol: float,
    normal_tol: float = 0.01,
    distance_tol: float = 0.01,
) -> list[TileFace]:
    """Group tiles by their supporting plane (within tolerance) and merge
    each group's polygons into one face per plane.

    Two tiles share a plane when their normals point the same way (within
    ``normal_tol``) AND the offsets `d` agree (within ``distance_tol``).
    Within a plane group we project corners onto the plane's 2D frame,
    union the polygons (Shapely), and emit one merged ``TileFace`` per
    connected component.

    Required to satisfy Geniet's 3-manifold invariant: each vertex must
    be the intersection of ≥3 distinct planes. tier_payload encodes
    flat ceilings as multiple coplanar tiles; without merging, interior
    vertices are on only 2 faces.
    """
    from shapely.geometry import Polygon as ShPoly
    from shapely.ops import unary_union

    if not tiles:
        return []

    # Per-tile polygon cleanup: drop near-duplicate consecutive corners
    # and collinear-with-neighbour corners. tier_payload polygons can
    # have 15+ vertices for a rectangular floor with many near-duplicates;
    # those produce vertex_underconstrained defects in the half-edge
    # framework because each near-duplicate vertex is incident to only
    # 1-2 distinct faces.
    cleaned: list[TileFace] = []
    for tile in tiles:
        clean_corners = _clean_polygon_corners(
            tile.corners,
            coord_tol=max(coord_tol, 5e-3),
        )
        if len(clean_corners) < 3:
            cleaned.append(tile)  # let build's degenerate check handle it
            continue
        cleaned.append(
            TileFace(
                face_id=tile.face_id,
                corners=clean_corners,
                plane=tile.plane,
                source=tile.source,
                locator_id=tile.locator_id,
                story=tile.story,
                room_index=tile.room_index,
            )
        )
    tiles = cleaned

    # Pass-through tiles that the merge step can't process (too few
    # corners). The downstream build will reject them with the proper
    # reason ("fewer_than_3_corners") instead of silently dropping.
    passthrough: list[TileFace] = []
    mergeable: list[TileFace] = []
    for tile in tiles:
        if len(tile.corners) < 3:
            passthrough.append(tile)
        else:
            mergeable.append(tile)

    groups: list[list[TileFace]] = []
    for tile in mergeable:
        normal = np.array(
            [tile.plane.a, tile.plane.b, tile.plane.c], dtype=float
        )
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            groups.append([tile])
            continue
        normal_unit = normal / norm
        d_unit = float(tile.plane.d) / norm
        attached = False
        for group in groups:
            head = group[0]
            head_n = np.array(
                [head.plane.a, head.plane.b, head.plane.c], dtype=float
            )
            head_norm = float(np.linalg.norm(head_n))
            if head_norm <= 1e-12:
                continue
            head_unit = head_n / head_norm
            head_d = float(head.plane.d) / head_norm
            # Aligned-or-opposite normals both count as the same plane.
            cos = float(normal_unit @ head_unit)
            if abs(abs(cos) - 1.0) > normal_tol:
                continue
            # Offsets compared in the same orientation.
            d_compare = d_unit if cos > 0 else -d_unit
            if abs(d_compare - head_d) > distance_tol:
                continue
            group.append(tile)
            attached = True
            break
        if not attached:
            groups.append([tile])

    out: list[TileFace] = []
    next_face_id = max((t.face_id for t in tiles), default=-1) + 1
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
            continue
        # Pick a 2D frame for this plane.
        head = group[0]
        normal = np.array(
            [head.plane.a, head.plane.b, head.plane.c], dtype=float
        )
        normal /= np.linalg.norm(normal)
        # Build a 2D basis (u, v) on the plane.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(normal @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        origin = np.asarray(group[0].corners[0], dtype=float)

        def to_2d(p):
            arr = np.asarray(p, dtype=float) - origin
            return (float(arr @ u), float(arr @ v))

        def to_3d(uv):
            arr = origin + uv[0] * u + uv[1] * v
            return (float(arr[0]), float(arr[1]), float(arr[2]))

        polys_2d: list[ShPoly] = []
        for tile in group:
            ring = [to_2d(c) for c in tile.corners]
            try:
                p = ShPoly(ring)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty:
                    polys_2d.append(p)
            except Exception:
                continue
        if not polys_2d:
            out.extend(group)
            continue
        # Buffer slightly to absorb sub-millimetre gaps between adjacent
        # tiles, then de-buffer to recover sharp boundaries.
        buf = max(coord_tol, 5e-4)
        buffered = [p.buffer(buf, join_style=2) for p in polys_2d]
        try:
            merged_2d = unary_union(buffered).buffer(-buf, join_style=2)
        except Exception:
            merged_2d = unary_union(polys_2d)
        if merged_2d.is_empty:
            out.extend(group)
            continue
        components = (
            list(merged_2d.geoms)
            if merged_2d.geom_type == "MultiPolygon"
            else [merged_2d]
        )
        plane_label = group[0].source
        plane = group[0].plane
        story = group[0].story
        room_index = group[0].room_index
        locator_seed = ",".join(t.locator_id for t in group[:3])
        for comp_idx, comp in enumerate(components):
            if comp.is_empty or comp.area < 1e-6:
                continue
            ring = list(comp.exterior.coords)[:-1]
            corners3d = tuple(to_3d(p) for p in ring)
            if len(corners3d) < 3:
                continue
            out.append(
                TileFace(
                    face_id=next_face_id,
                    corners=corners3d,
                    plane=plane,
                    source=plane_label,
                    locator_id=f"merged::{locator_seed}::{comp_idx}",
                    story=story,
                    room_index=room_index,
                )
            )
            next_face_id += 1
    out.extend(passthrough)
    return out


def prepare_room_tiles(
    tiles: Sequence[TileFace],
    *,
    coord_tol: float = 1e-3,
    snap_tol: float = 0.05,
    merge_coplanar: bool = True,
) -> list[TileFace]:
    """Snap shared corners and optionally merge coplanar tiles before build."""
    snapped = (
        _snap_tile_corners(tiles, snap_tol=snap_tol)
        if snap_tol > 0.0
        else list(tiles)
    )
    if merge_coplanar:
        return _merge_coplanar_tiles(snapped, coord_tol=coord_tol)
    return snapped


def _try_add_face(
    poly: HalfEdgePolyhedron,
    tile: TileFace,
    face_idx: int,
    vids: list[int],
    next_he_id: int,
    he_by_directed: dict[tuple[int, int], HalfEdge],
    tile_face_by_id: dict[int, TileFace],
) -> tuple[int, bool]:
    """Attempt to add a face with the given vertex ring. Returns (next_he_id_after, success).

    On failure (any directed edge already claimed), rolls back partial state
    cleanly so the caller can retry with the reversed ring.
    """
    face = Face(id=face_idx, plane=tile.plane)
    ring: list[HalfEdge] = []
    he_id = next_he_id
    for vid in vids:
        he = HalfEdge(id=he_id, origin=poly.vertices[vid], face=face)
        he_id += 1
        ring.append(he)
    for i, he in enumerate(ring):
        he.next = ring[(i + 1) % len(ring)]
    keys: list[tuple[int, int]] = []
    for he in ring:
        dst_vid = he.next.origin.id
        key = (he.origin.id, dst_vid)
        if key in he_by_directed:
            return next_he_id, False
        keys.append(key)
    # Commit: append face, half-edges, register directed edges.
    face.half_edge = ring[0]
    poly.faces.append(face)
    poly.half_edges.extend(ring)
    for he, key in zip(ring, keys, strict=True):
        he_by_directed[key] = he
    tile_face_by_id[face_idx] = tile
    return he_id, True


def _drop_face_from_partial(
    poly: HalfEdgePolyhedron,
    face: Face,
    ring: list[HalfEdge],
    he_by_directed: dict[tuple[int, int], HalfEdge],
) -> None:
    """Remove a face that turned out to be inconsistent during build.

    Pops its half-edges from the polyhedron and the directed-edge index.
    The face object itself stays in `poly.faces` but with no half-edge
    pointer; downstream consumers detect this via `face.half_edge is None`.
    """
    if poly.faces and poly.faces[-1] is face:
        poly.faces.pop()
    keys_to_drop: list[tuple[int, int]] = []
    for key, he in he_by_directed.items():
        if he.face is face:
            keys_to_drop.append(key)
    for key in keys_to_drop:
        he_by_directed.pop(key, None)
    for he in ring:
        if he in poly.half_edges:
            poly.half_edges.remove(he)


# ---------------------------------------------------------------------------
# Increment 2: orphan chain extraction
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HoleChain:
    """A closed loop of orphan half-edges around one manifold hole.

    The half-edges are ordered: each one's destination vertex is the next
    one's origin vertex. The loop closes (last.next.origin == first.origin).
    """

    half_edges: tuple[HalfEdge, ...]
    vertex_ids: tuple[int, ...]  # ordered vertex IDs around the hole


@dataclass(slots=True)
class HoleChainExtraction:
    """Result of walking orphan half-edges to extract closed hole chains."""

    closed_chains: list[HoleChain] = field(default_factory=list)
    open_chains: list[tuple[HalfEdge, ...]] = field(default_factory=list)
    ambiguous_vertices: list[int] = field(default_factory=list)


def extract_hole_chains(
    build: RoomPolyhedronBuild,
) -> HoleChainExtraction:
    """Group orphan half-edges into closed boundary loops.

    Each closed loop is the boundary of one manifold hole. Open chains
    indicate genuine non-manifold defects (e.g., a hole that exits the
    geometry). Ambiguous vertices (multiple orphans starting from the
    same vertex — figure-8 / pinched topology) are flagged so callers
    can fall back gracefully.
    """

    extraction = HoleChainExtraction()
    orphans = list(build.orphan_half_edges)
    if not orphans:
        return extraction

    # Index orphans by their origin vertex ID. Vertices with multiple
    # outgoing orphans are "ambiguous" (figure-8 / multiple holes sharing
    # a vertex). DFS handles these by exploring all continuations.
    by_origin: dict[int, list[HalfEdge]] = {}
    for he in orphans:
        by_origin.setdefault(he.origin.id, []).append(he)
    extraction.ambiguous_vertices = [
        vid for vid, hes in by_origin.items() if len(hes) > 1
    ]

    # DFS chain extraction. Pop each unvisited orphan as a candidate
    # start; explore all forward continuations; commit a closed chain
    # whenever we return to the start. This recovers many holes that
    # the greedy walker abandons at ambiguous vertices.
    consumed: set[int] = set()
    for start in orphans:
        if id(start) in consumed:
            continue
        path = _find_closed_chain(start, by_origin, consumed)
        if path is not None:
            extraction.closed_chains.append(
                HoleChain(
                    half_edges=tuple(path),
                    vertex_ids=tuple(he.origin.id for he in path),
                )
            )
            for he in path:
                consumed.add(id(he))

    # Whatever remains is genuinely open — record it.
    for he in orphans:
        if id(he) not in consumed:
            extraction.open_chains.append((he,))
            consumed.add(id(he))

    return extraction


def _find_closed_chain(
    start: HalfEdge,
    by_origin: dict[int, list[HalfEdge]],
    already_consumed: set[int],
) -> list[HalfEdge] | None:
    """DFS for a closed orphan-edge cycle that starts at ``start``.

    Returns the list of half-edges forming the cycle, in walk order, or
    ``None`` if no cycle exists. Half-edges already consumed by an
    earlier closed chain are skipped — they cannot belong to two holes.
    """
    if id(start) in already_consumed:
        return None
    if start.next is None:
        return None
    target_vid = start.origin.id
    stack: list[tuple[HalfEdge, list[HalfEdge], set[int]]] = [
        (start, [start], {id(start)})
    ]
    while stack:
        cur, path, used = stack.pop()
        if cur.next is None:
            continue
        dst_vid = cur.next.origin.id
        if dst_vid == target_vid and len(path) >= 3:
            return path
        outgoing = by_origin.get(dst_vid, [])
        for nxt in outgoing:
            if id(nxt) in used or id(nxt) in already_consumed:
                continue
            if nxt.next is None:
                continue
            stack.append((nxt, [*path, nxt], used | {id(nxt)}))
    return None


# ---------------------------------------------------------------------------
# Increment 3: filler hypothesis + ILP picking
# ---------------------------------------------------------------------------

CEILING_TILE_SOURCES = frozenset({"ceiling", "visual_shell", "gable_closure"})


@dataclass(frozen=True, slots=True)
class FillerCandidate:
    """A hypothesised face that closes a hole.

    Variable in the ILP — picked or dropped. ``parent_hole_id`` indexes
    into ``HoleChainExtraction.closed_chains``.
    """

    face_id: int  # global face id, allocated after the tiles
    parent_hole_id: int
    corners: tuple[tuple[float, float, float], ...]
    plane: Plane
    derivation: str  # "best_fit_plane" | "triangle_fan_apex" | "neighbor_extension"
    area: float


@dataclass(slots=True)
class FillerSelection:
    """ILP outcome for filler picking on one room."""

    selected: list[FillerCandidate] = field(default_factory=list)
    rejected: list[FillerCandidate] = field(default_factory=list)
    unfilled_hole_ids: list[int] = field(default_factory=list)
    objective: float = 0.0
    solver_status: str = "not_solved"


def _filler_from_plane_on_hole(
    *,
    hole_id: int,
    coords: Sequence[tuple[float, float, float]],
    plane: Plane,
    face_id: int,
    derivation: str,
) -> FillerCandidate | None:
    """Project a hole vertex ring onto ``plane``; return None if degenerate."""
    if len(coords) < 3:
        return None
    snapped = tuple(_project_onto_plane(c, plane) for c in coords)
    snapped = tuple(reversed(snapped))
    area = _polygon_area_3d(snapped)
    if area <= 1e-9:
        return None
    return FillerCandidate(
        face_id=face_id,
        parent_hole_id=hole_id,
        corners=snapped,
        plane=plane,
        derivation=derivation,
        area=float(area),
    )


def _catalog_tiles_by_locator(
    all_tiles: Sequence[TileFace],
    skipped_tiles: Sequence[tuple[TileFace, str]] = (),
) -> dict[str, TileFace]:
    """Deduplicate input tiles by ``locator_id`` (skipped entries win)."""
    catalog: dict[str, TileFace] = {tile.locator_id: tile for tile in all_tiles}
    for tile, _reason in skipped_tiles:
        catalog[tile.locator_id] = tile
    return catalog


def _ceiling_tiles_lost_before_build(
    *,
    tiles_raw: Sequence[TileFace],
    tiles_clipped: Sequence[TileFace],
    build_tiles: Sequence[TileFace],
) -> list[TileFace]:
    """Ceiling tiles dropped by roof XZ clip or step-4 filter before build.

    These never enter ``build_tiles`` but should still propose ``original_tile``
    filler hypotheses so scan ceilings can close holes.
    """
    in_build = {
        tile.locator_id
        for tile in build_tiles
        if tile.source in CEILING_TILE_SOURCES
    }
    out: dict[str, TileFace] = {}
    for tile in tiles_clipped:
        if tile.source not in CEILING_TILE_SOURCES:
            continue
        if tile.locator_id in in_build:
            continue
        out[tile.locator_id] = tile
    for tile in tiles_raw:
        if tile.source not in CEILING_TILE_SOURCES:
            continue
        if tile.locator_id in in_build:
            continue
        out.setdefault(tile.locator_id, tile)
    return list(out.values())


def _filler_ceiling_catalog(
    all_tiles: Sequence[TileFace],
    skipped_tiles: Sequence[tuple[TileFace, str]],
    *,
    pre_filter_ceilings: Sequence[TileFace],
) -> list[TileFace]:
    """Merge ceiling tiles for ``original_tile`` hypotheses (build tiles win)."""
    catalog: dict[str, TileFace] = {}
    for tile in pre_filter_ceilings:
        if tile.source in CEILING_TILE_SOURCES:
            catalog[tile.locator_id] = tile
    for tile in all_tiles:
        if tile.source in CEILING_TILE_SOURCES:
            catalog[tile.locator_id] = tile
    for tile, _reason in skipped_tiles:
        if tile.source in CEILING_TILE_SOURCES:
            catalog[tile.locator_id] = tile
    return list(catalog.values())


def hypothesise_single_face_fillers(
    build: RoomPolyhedronBuild,
    extraction: HoleChainExtraction,
    *,
    first_face_id: int,
) -> list[FillerCandidate]:
    """One best-fit-plane candidate per closed hole chain.

    The plane is the least-squares fit through the hole's vertex
    coordinates. The polygon is the hole's vertex ring projected onto
    that plane (in the order of the chain).

    This is the simplest hypothesis. Use ``hypothesise_fillers`` to also
    get neighbor-plane-extension candidates.
    """
    out: list[FillerCandidate] = []
    next_id = first_face_id
    for hole_id, chain in enumerate(extraction.closed_chains):
        coords = [build.vertex_coords[vid] for vid in chain.vertex_ids]
        if len(coords) < 3:
            continue
        plane = _best_fit_plane(coords)
        if plane is None:
            continue
        snapped = tuple(_project_onto_plane(c, plane) for c in coords)
        snapped = tuple(reversed(snapped))
        area = _polygon_area_3d(snapped)
        if area <= 1e-9:
            continue
        out.append(
            FillerCandidate(
                face_id=next_id,
                parent_hole_id=hole_id,
                corners=snapped,
                plane=plane,
                derivation="best_fit_plane",
                area=float(area),
            )
        )
        next_id += 1
    return out


def hypothesise_original_tile_fillers(
    build: RoomPolyhedronBuild,
    extraction: HoleChainExtraction,
    *,
    first_face_id: int,
    all_tiles: Sequence[TileFace],
    skipped_tiles: Sequence[tuple[TileFace, str]] = (),
    pre_filter_ceilings: Sequence[TileFace] = (),
) -> list[FillerCandidate]:
    """tier_payload ceiling tiles as filler candidates per hole.

    Includes tiles that failed half-edge insertion (``skipped_tiles``)
    and ceilings removed by roof clip / connectivity filter
    (``pre_filter_ceilings``) so scan lids can close holes even when
    they are not mesh faces.
    """
    catalog = _filler_ceiling_catalog(
        all_tiles,
        skipped_tiles,
        pre_filter_ceilings=pre_filter_ceilings,
    )
    out: list[FillerCandidate] = []
    next_id = first_face_id
    for hole_id, chain in enumerate(extraction.closed_chains):
        coords = [build.vertex_coords[vid] for vid in chain.vertex_ids]
        for tile in catalog:
            cand = _filler_from_plane_on_hole(
                hole_id=hole_id,
                coords=coords,
                plane=tile.plane,
                face_id=next_id,
                derivation=f"original_tile:{tile.locator_id}",
            )
            if cand is None:
                continue
            out.append(cand)
            next_id += 1
    return out


def hypothesise_neighbor_plane_extension_fillers(
    build: RoomPolyhedronBuild,
    extraction: HoleChainExtraction,
    *,
    first_face_id: int,
    all_tiles: Sequence[TileFace] | None = None,
    skipped_tiles: Sequence[tuple[TileFace, str]] = (),
) -> list[FillerCandidate]:
    """Neighbor-plane-extension candidates per closed hole chain.

    Per plan §3 step 5: for each distinct face touching the hole's
    orphan boundary, emit a hypothesis where the filler's plane equals
    that face's plane. The corners are the chain's vertex ring projected
    onto the chosen plane.

    When ``all_tiles`` is provided, also emit hypotheses from every
    non-ceiling input tile plane (walls/floors). Ceiling planes are
    handled by ``hypothesise_original_tile_fillers``.

    These hypotheses produce fillers that are coplanar with an
    existing tile — a wall-floor seam hole gets a filler at the floor's
    plane (visually merges into the floor), or at the wall's plane
    (visually merges into the wall). The ILP picks based on energy.
    """
    input_planes: dict[str, Plane] = {}
    if all_tiles is not None:
        for tile in _catalog_tiles_by_locator(all_tiles, skipped_tiles).values():
            if tile.source in CEILING_TILE_SOURCES:
                continue
            input_planes[f"input_tile:{tile.locator_id}"] = tile.plane

    out: list[FillerCandidate] = []
    next_id = first_face_id
    for hole_id, chain in enumerate(extraction.closed_chains):
        coords = [build.vertex_coords[vid] for vid in chain.vertex_ids]
        if len(coords) < 3:
            continue
        plane_by_derivation: dict[str, Plane] = dict(input_planes)
        for he in chain.half_edges:
            if he.face is None:
                continue
            plane_by_derivation[f"neighbor_extension:face_{he.face.id}"] = (
                he.face.plane
            )
        for derivation, plane in plane_by_derivation.items():
            cand = _filler_from_plane_on_hole(
                hole_id=hole_id,
                coords=coords,
                plane=plane,
                face_id=next_id,
                derivation=derivation,
            )
            if cand is None:
                continue
            out.append(cand)
            next_id += 1
    return out


def hypothesise_fillers(
    build: RoomPolyhedronBuild,
    extraction: HoleChainExtraction,
    *,
    first_face_id: int,
    all_tiles: Sequence[TileFace] | None = None,
    skipped_tiles: Sequence[tuple[TileFace, str]] = (),
    pre_filter_ceilings: Sequence[TileFace] = (),
) -> list[FillerCandidate]:
    """Generate filler candidates per hole for the ILP.

    Union of best-fit plane, tier_payload ceiling tiles (including build
    skips and ceilings dropped before build), and neighbor / input-tile
    plane extensions. The ILP picks exactly one candidate per hole.
    """
    single = hypothesise_single_face_fillers(
        build, extraction, first_face_id=first_face_id
    )
    next_id = first_face_id + len(single)
    original: list[FillerCandidate] = []
    if all_tiles is not None:
        original = hypothesise_original_tile_fillers(
            build,
            extraction,
            first_face_id=next_id,
            all_tiles=all_tiles,
            skipped_tiles=skipped_tiles,
            pre_filter_ceilings=pre_filter_ceilings,
        )
        next_id += len(original)
    neighbour = hypothesise_neighbor_plane_extension_fillers(
        build,
        extraction,
        first_face_id=next_id,
        all_tiles=all_tiles,
        skipped_tiles=skipped_tiles,
    )
    return single + original + neighbour


def select_fillers(
    build: RoomPolyhedronBuild,
    extraction: HoleChainExtraction,
    fillers: Sequence[FillerCandidate],
    *,
    weights: tuple[float, float, float] = (0.43, 0.27, 0.30),
) -> FillerSelection:
    """Pick a watertight filler subset via PolyFit-style ILP.

    Tiles are forced selected; only fillers are variables. For each hole
    we require *exactly one* filler from that hole's hypothesis pool.
    The ILP minimises a PolyFit-style energy:

        λ_f · plane_residual_normalised
        + λ_c · area_normalised

    plus a structural bias that prefers neighbor-plane-extension over
    best-fit-plane (no new plane added → simpler complexity term).
    """
    selection = FillerSelection()
    if not fillers:
        return selection

    by_hole: dict[int, list[FillerCandidate]] = {}
    for cand in fillers:
        by_hole.setdefault(cand.parent_hole_id, []).append(cand)

    # Per-candidate energy: prefer (a) lower plane residual against the
    # chain's vertices (data-fit) and (b) coplanar-with-neighbour over
    # synthesised best-fit (complexity term).
    def cand_residual(cand: FillerCandidate) -> float:
        chain = extraction.closed_chains[cand.parent_hole_id]
        coords = [build.vertex_coords[vid] for vid in chain.vertex_ids]
        if not coords:
            return 0.0
        n = np.array([cand.plane.a, cand.plane.b, cand.plane.c], dtype=float)
        norm = float(np.linalg.norm(n))
        if norm <= 1e-12:
            return float("inf")
        n_unit = n / norm
        d_unit = float(cand.plane.d) / norm
        residuals = [abs(float(n_unit @ np.asarray(c)) - d_unit) for c in coords]
        return sum(residuals) / len(residuals)

    def cand_score(cand: FillerCandidate, total_area: float) -> float:
        lambda_f, _lambda_m, lambda_c = weights
        residual = cand_residual(cand)
        area_norm = cand.area / total_area if total_area > 0 else 0.0
        # Prefer reusing tier_payload / mesh planes over synthesising a
        # new best-fit plane.
        if cand.derivation.startswith("original_tile:"):
            complexity_bonus = 0.0
        elif cand.derivation.startswith(
            ("neighbor_extension", "input_tile:", "skipped_tile:")
        ):
            complexity_bonus = 0.15
        else:
            complexity_bonus = 1.0
        return (
            float(lambda_f) * residual
            + float(lambda_c) * area_norm
            + 0.5 * complexity_bonus
        )

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        for cands in by_hole.values():
            best = min(
                cands, key=lambda c: cand_score(c, sum(x.area for x in cands) or 1.0)
            )
            selection.selected.append(best)
            for c in cands:
                if c is not best:
                    selection.rejected.append(c)
        selection.solver_status = "scipy_unavailable_greedy_fallback"
        return selection

    n_vars = len(fillers)
    total_area = sum(c.area for c in fillers) or 1.0
    coeffs = np.array(
        [cand_score(c, total_area) for c in fillers], dtype=float
    )

    # Exactly-one constraint per hole.
    n_holes = len(by_hole)
    constraints = lil_matrix((n_holes, n_vars), dtype=float)
    lb = np.ones(n_holes, dtype=float)
    ub = np.ones(n_holes, dtype=float)
    filler_index = {id(cand): i for i, cand in enumerate(fillers)}
    for row, (_hole_id, cands) in enumerate(sorted(by_hole.items())):
        for c in cands:
            constraints[row, filler_index[id(c)]] = 1.0

    result = milp(
        c=coeffs,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(constraints.tocsr(), lb, ub),
    )

    if not result.success or result.x is None:
        for cands in by_hole.values():
            best = min(
                cands, key=lambda c: cand_score(c, total_area)
            )
            selection.selected.append(best)
            for c in cands:
                if c is not best:
                    selection.rejected.append(c)
        selection.solver_status = f"milp_failed_fallback:{result.message}"
        return selection

    chosen_ids: set[int] = set()
    for i, cand in enumerate(fillers):
        if float(result.x[i]) >= 0.5:
            selection.selected.append(cand)
            chosen_ids.add(cand.face_id)
        else:
            selection.rejected.append(cand)
    selection.objective = float(result.fun)
    selection.solver_status = f"milp_{result.status}"

    selected_holes = {c.parent_hole_id for c in selection.selected}
    selection.unfilled_hole_ids = [
        hole_id for hole_id in by_hole if hole_id not in selected_holes
    ]
    return selection


def _best_fit_plane(
    points: Sequence[Sequence[float]],
) -> Plane | None:
    """Least-squares plane through ≥3 points. Returns None if degenerate."""
    if len(points) < 3:
        return None
    arr = np.asarray(points, dtype=float)
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal = normal / norm
    d = float(normal @ centroid)
    return Plane(
        a=float(normal[0]),
        b=float(normal[1]),
        c=float(normal[2]),
        d=d,
    )


def _project_onto_plane(
    point: Sequence[float],
    plane: Plane,
) -> tuple[float, float, float]:
    n = np.array([plane.a, plane.b, plane.c], dtype=float)
    n_norm_sq = float(n @ n)
    if n_norm_sq <= 1e-12:
        return (float(point[0]), float(point[1]), float(point[2]))
    p = np.asarray(point, dtype=float)
    distance = float(p @ n - plane.d) / n_norm_sq
    proj = p - distance * n
    return (float(proj[0]), float(proj[1]), float(proj[2]))


# ---------------------------------------------------------------------------
# Increment 4: Geniet FACE_CREATION_FROM_EDGE — atomic filler insertion
# ---------------------------------------------------------------------------


def apply_fillers(
    build: RoomPolyhedronBuild,
    extraction: HoleChainExtraction,
    selected: Sequence[FillerCandidate],
) -> int:
    """Atomically insert each selected filler along its hole-boundary chain.

    Implements Geniet 2024 §3.2.3 ``FACE_CREATION_FROM_EDGE`` for the
    specific case where the new face's boundary IS the orphan-edge chain.
    For each filler:

    1. A new ``Face`` is created with ``filler.plane``.
    2. For each orphan half-edge in the chain, a new half-edge is created
       in the opposite direction (the twin), with ``face`` = the new face.
    3. The orphan and its twin are paired (``opposite`` set on both).
    4. The new half-edges' ``next`` pointers are wired to form the
       face's CCW ring (viewed from the filler's outward side, which is
       the reverse of the chain's traversal).
    5. The face is appended to ``poly.faces``.

    Mutates ``build.poly`` and ``build.tile_face_by_id`` in place; updates
    ``build.orphan_half_edges`` to remove the now-paired half-edges.
    Returns the number of fillers successfully applied.
    """
    if not selected:
        return 0
    fillers_by_hole: dict[int, FillerCandidate] = {
        f.parent_hole_id: f for f in selected
    }
    applied = 0
    for hole_id, chain in enumerate(extraction.closed_chains):
        filler = fillers_by_hole.get(hole_id)
        if filler is None:
            continue
        if _apply_filler_from_chain(build, chain, filler):
            applied += 1
    # Refresh the orphan list.
    build.orphan_half_edges = [
        he for he in build.poly.half_edges if he.opposite is None
    ]
    return applied


def _apply_filler_from_chain(
    build: RoomPolyhedronBuild,
    chain: HoleChain,
    filler: FillerCandidate,
) -> bool:
    """Create the filler face and twin-pair its half-edges with the chain.

    Returns True on success, False if the chain or build is malformed
    (e.g., orphan ``next`` pointer broken). Failure leaves the polyhedron
    unmodified.
    """
    poly = build.poly
    n = len(chain.half_edges)
    if n < 3:
        return False
    # Validate: each orphan must still be unpaired and have a next pointer,
    # and consecutive chain edges must share a vertex (orphan_i.next.origin
    # == orphan_{i+1}.origin). The DFS walker constructs chains this way, but
    # an earlier filler insertion could have repointed `next` on a vertex that
    # was incidentally shared between two holes — in that rare case the chain
    # is corrupt and we must abort rather than wire a bogus filler.
    for orphan in chain.half_edges:
        if orphan.opposite is not None:
            return False
        if orphan.next is None:
            return False
    for i in range(n):
        cur = chain.half_edges[i]
        nxt = chain.half_edges[(i + 1) % n]
        if cur.next.origin.id != nxt.origin.id:
            return False

    # Use the filler's pre-allocated face_id rather than len(poly.faces).
    # Tile face ids may be sparse (some _try_add_face attempts skip ids when
    # both windings clash); len(poly.faces) collides with such gaps.
    new_face = Face(id=filler.face_id, plane=filler.plane)
    next_he_id = max((he.id for he in poly.half_edges), default=-1) + 1
    new_hes: list[HalfEdge] = []
    # Walk the chain in REVERSE order. For orphan_i (v_i → v_{i+1}), its
    # twin goes v_{i+1} → v_i. The filler face's CCW ring traverses
    # v_n_1 → v_n_2 → ... → v_1 → v_0 → v_n_1, i.e., the reverse vertex
    # order. Iterating chain.half_edges in reverse and taking each
    # orphan's destination as the twin's origin yields exactly that ring.
    for i in range(n - 1, -1, -1):
        orphan = chain.half_edges[i]
        twin_origin_vid = orphan.next.origin.id
        twin = HalfEdge(
            id=next_he_id,
            origin=poly.vertices[twin_origin_vid],
            face=new_face,
            opposite=orphan,
        )
        next_he_id += 1
        new_hes.append(twin)

    # Pair orphans with their newly-created twins.
    for i, twin in enumerate(new_hes):
        # The reversed iteration above means new_hes[k] is the twin of
        # chain.half_edges[n - 1 - k].
        orphan_index = n - 1 - i
        chain.half_edges[orphan_index].opposite = twin

    # Wire next pointers around the filler's CCW ring.
    for i, he in enumerate(new_hes):
        he.next = new_hes[(i + 1) % n]

    new_face.half_edge = new_hes[0]
    poly.faces.append(new_face)
    poly.half_edges.extend(new_hes)

    # Refresh outgoing pointers for every vertex that gained an edge from this
    # filler. We don't always need to switch — but if a vertex's previous
    # outgoing was an old orphan, the orbit walk now goes through
    # orphan.opposite.next which lives on the filler face. That's fine. The
    # important invariant: outgoing must be a half-edge whose `face` is one
    # the orbit can walk back from. Picking the lowest-id outgoing is the same
    # rule used in `build_room_polyhedron`, so apply it consistently here.
    affected_vids = {twin.origin.id for twin in new_hes}
    for vid in affected_vids:
        outs = [he for he in poly.half_edges if he.origin.id == vid]
        if outs:
            poly.vertices[vid].outgoing = min(outs, key=lambda h: h.id)
    return True


# ---------------------------------------------------------------------------
# Increment 5: per-room high-level pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RoomRepairResult:
    """End-to-end manifold repair output for one room."""

    room_index: int | None
    story: int | None
    build: RoomPolyhedronBuild
    extraction: HoleChainExtraction
    selection: FillerSelection
    fillers_applied: int
    status: str  # "watertight" | "holes_remaining" | "skipped"
    skip_reason: str | None = None
    tile_count: int = 0


def repair_room(
    payload: Mapping[str, Any],
    room: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
    reconcile_planes_after_apply: bool = False,
    plane_reconciliation_tolerance_m: float = 0.005,
) -> RoomRepairResult:
    """Run the full tile→build→detect→hypothesise→ILP→apply pipeline.

    The result's ``build.poly`` is the (ideally) watertight polyhedron
    after fillers are applied. ``status`` is ``"watertight"`` when no
    orphan half-edges remain, ``"holes_remaining"`` when some couldn't
    be filled, ``"skipped"`` when the room had too few tiles to start.

    ``reconcile_planes_after_apply`` (opt-in helper): runs Geniet
    FACE_SHIFT iterations to nudge plane offsets so derived 3-plane
    intersections align with explicit corners (sub-mm). Off by default —
    visualization is anchored to tier_payload's geometry, not to derived
    vertex positions.
    """
    story = room.get("story")
    room_idx = _room_index(room)
    tiles_raw = collect_room_tiles(payload, room, corner_tol=corner_tol)
    pre_filter_ceilings: list[TileFace] = []
    tiles = tiles_raw
    if len(tiles_raw) >= 4:
        from reconcile_tiers.polyhedron.roof_xz_clip import clip_roof_tiles_to_floor_xz
        from reconcile_tiers.polyhedron.tile_coherence import (
            filter_unconnected_ceiling_tiles,
        )

        clip = clip_roof_tiles_to_floor_xz(tiles_raw)
        tiles_clipped = list(clip.tiles)
        tiles_filtered, _dropped = filter_unconnected_ceiling_tiles(
            tiles_clipped, corner_tol=corner_tol
        )
        tiles = prepare_room_tiles(
            tiles_filtered,
            coord_tol=coord_tol,
            snap_tol=corner_tol,
            merge_coplanar=True,
        )
        pre_filter_ceilings = _ceiling_tiles_lost_before_build(
            tiles_raw=tiles_raw,
            tiles_clipped=tiles_clipped,
            build_tiles=tiles,
        )
    if len(tiles) < 4:  # need at least 4 faces to bound a 3D region
        empty_build = RoomPolyhedronBuild(poly=HalfEdgePolyhedron())
        return RoomRepairResult(
            room_index=room_idx,
            story=story,
            build=empty_build,
            extraction=HoleChainExtraction(),
            selection=FillerSelection(),
            fillers_applied=0,
            status="skipped",
            skip_reason="insufficient_tiles",
            tile_count=len(tiles),
        )

    build = build_room_polyhedron(
        tiles, coord_tol=coord_tol, snap_tol=0.0, merge_coplanar=False
    )
    extraction = extract_hole_chains(build)
    next_face_id = max((f.id for f in build.poly.faces), default=-1) + 1
    fillers = hypothesise_fillers(
        build,
        extraction,
        first_face_id=next_face_id,
        all_tiles=tiles,
        skipped_tiles=build.skipped_tiles,
        pre_filter_ceilings=pre_filter_ceilings,
    )
    selection = select_fillers(build, extraction, fillers)
    fillers_applied = apply_fillers(build, extraction, selection.selected)

    if not build.orphan_half_edges:
        status = "watertight"
    else:
        status = "holes_remaining"

    if reconcile_planes_after_apply and status == "watertight":
        # Late import: plane_reconciliation pulls in numpy heavy paths and
        # is opt-in for the strict-validation pipeline.
        from reconcile_tiers.polyhedron.plane_reconciliation import (
            reconcile_planes,
        )

        reconcile_planes(
            build.poly,
            build.vertex_coords,
            tolerance_m=plane_reconciliation_tolerance_m,
        )

    return RoomRepairResult(
        room_index=room_idx,
        story=story,
        build=build,
        extraction=extraction,
        selection=selection,
        fillers_applied=fillers_applied,
        status=status,
        tile_count=len(tiles),
    )


# ---------------------------------------------------------------------------
# Increment 5b: emit EnvelopeCandidate from a RoomRepairResult
# ---------------------------------------------------------------------------


def envelope_candidate_from_repair(
    repair: RoomRepairResult,
) -> object | None:
    """Convert a RoomRepairResult into a `pa.EnvelopeCandidate` for the
    existing v2 corpus exporter / viewer pipeline.

    Faces are emitted in iteration order. Filler faces (no entry in
    ``build.tile_face_by_id``) are tagged source ``polyhedron_v3_filler``
    and rendered with ``kind="ceiling"`` so the viewer's ceiling material
    paints them. Tile-derived faces inherit their tile's source.

    Returns None when the build has no faces (skipped rooms or empty
    polyhedra).
    """
    from reconcile_tiers.polyhedron import payload_adapter as pa

    poly = repair.build.poly
    if not poly.faces:
        return None
    out_faces: list[pa.PayloadFace] = []
    top_sources: list[str] = []
    for face in poly.faces:
        if face.half_edge is None:
            continue
        # Walk the face's CCW ring to recover its corner sequence using
        # vertex_coords (the original tile-corner cluster centroids).
        corners: list[list[float]] = []
        he = face.half_edge
        guard = 0
        while True:
            guard += 1
            if guard > 1024:
                break  # malformed ring; skip face
            corners.append(list(repair.build.vertex_coords[he.origin.id]))
            nxt = he.next
            if nxt is None or nxt is face.half_edge:
                break
            he = nxt
        if len(corners) < 3:
            continue

        tile = repair.build.tile_face_by_id.get(face.id)
        if tile is None:
            # Filler face. If its plane matches an existing tile's plane
            # (neighbor-extension hypothesis was selected), inherit that
            # tile's kind + source so the viewer paints it with the same
            # material — visually merging the filler into the surface it
            # extends instead of showing it as a separate "synthetic"
            # patch.
            inherited = _filler_inherited_source(face.plane, repair.build.tile_face_by_id)
            if inherited is not None:
                kind = _tile_source_to_kind(inherited)
                label = f"polyhedron_v3_filler:{inherited}"
            else:
                kind = "ceiling"
                label = "polyhedron_v3_filler"
            locator = (
                f"envelope-v3-room:{repair.room_index}::filler::{face.id}"
            )
        else:
            kind = _tile_source_to_kind(tile.source)
            label = tile.source
            locator = tile.locator_id
            if kind == "ceiling":
                top_sources.append(f"{tile.source}:face:{face.id}")
        out_faces.append(
            pa.PayloadFace(
                kind=kind,
                locator_id=str(locator),
                corners=corners,
                plane=face.plane,
                source=label,
                room_index=repair.room_index,
                story=repair.story,
            )
        )
    if not out_faces:
        return None

    floor_corners: list[list[float]] = []
    for f in out_faces:
        if f.kind == "floor":
            floor_corners = f.corners
            break
    footprint_area = _xz_polygon_area(floor_corners) if floor_corners else 0.0

    return pa.EnvelopeCandidate(
        locator_id=f"envelope-v3-room:{repair.room_index}",
        faces=out_faces,
        footprint_area_m2=float(footprint_area),
        top_source=" + ".join(top_sources) or "polyhedron_v3",
        top_overlap_ratio=1.0 if repair.status == "watertight" else 0.0,
        selector="manifold_repair_v3",
    )


def _filler_inherited_source(
    plane: Plane,
    tile_face_by_id: Mapping[int, TileFace],
) -> str | None:
    """Find an existing tile whose plane equals this filler's plane (within
    a generous tolerance). If found, return its source so the renderer
    can label the filler as an extension of that surface."""
    n = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return None
    n_unit = n / norm
    d_unit = float(plane.d) / norm
    best_source: str | None = None
    best_err = float("inf")
    for tile in tile_face_by_id.values():
        tn = np.array(
            [tile.plane.a, tile.plane.b, tile.plane.c], dtype=float
        )
        t_norm = float(np.linalg.norm(tn))
        if t_norm <= 1e-12:
            continue
        tn_unit = tn / t_norm
        td_unit = float(tile.plane.d) / t_norm
        cos = float(n_unit @ tn_unit)
        if abs(abs(cos) - 1.0) > 0.02:
            continue
        d_compare = td_unit if cos > 0 else -td_unit
        err = abs(d_compare - d_unit)
        if err < best_err and err < 0.05:
            best_err = err
            best_source = tile.source
    return best_source


def _tile_source_to_kind(source: str) -> pa.FaceKind:
    if source == "floor":
        return "floor"
    if source == "wall":
        return "wall"
    return "ceiling"


def _xz_polygon_area(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    pts = [(float(c[0]), float(c[2])) for c in corners]
    area = 0.0
    n = len(pts)
    for i in range(n):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % n]
        area += x0 * z1 - x1 * z0
    return abs(area) * 0.5
