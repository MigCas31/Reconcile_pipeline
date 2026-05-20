"""Pre-build checks that room tiles share corners/edges (floor–wall–ceiling shell).

Runs on collected ``TileFace`` lists before snap/merge and half-edge construction.
Mis-assigned roof pieces often overlap a room footprint in XZ but fail rim
contact with wall tops; this module surfaces that before manifold repair.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from reconcile_tiers.polyhedron.manifold_repair import TileFace

IssueKind = Literal[
    "missing_floor",
    "missing_ceiling",
    "missing_walls",
    "tile_graph_disconnected",
    "floor_wall_gap",
    "wall_ceiling_gap",
    "ceiling_far_from_wall_tops",
    "isolated_tile",
]

CEILING_SOURCES = frozenset({"ceiling", "visual_shell", "gable_closure"})


@dataclass(frozen=True, slots=True)
class TileCoherenceIssue:
    kind: IssueKind
    message: str
    tile_locator_ids: tuple[str, ...] = ()
    edge: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None


@dataclass(frozen=True, slots=True)
class RoomTileCoherenceResult:
    ok: bool
    issues: tuple[TileCoherenceIssue, ...]
    component_count: int
    shared_edge_count: int
    ceiling_clearance_m: float | None


def cluster_tile_corners(
    tiles: Sequence[TileFace],
    tol: float,
) -> list[list[int]]:
    """Union-find cluster of tile corners; returns per-tile vertex cluster ids."""

    if not tiles:
        return []

    pts: list[tuple[float, float, float]] = []
    for tile in tiles:
        pts.extend(tile.corners)
    n = len(pts)
    if n == 0:
        return [[] for _ in tiles]

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
    tol_sq = tol * tol
    for i in range(n):
        diffs = arr - arr[i]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        for j in np.flatnonzero(d2 <= tol_sq):
            j_int = int(j)
            if j_int != i:
                union(i, j_int)

    per_tile_vids: list[list[int]] = []
    cursor = 0
    for tile in tiles:
        vids = [find(cursor + k) for k in range(len(tile.corners))]
        cursor += len(tile.corners)
        per_tile_vids.append(vids)

    return per_tile_vids


def tile_undirected_edges(corner_vids: list[int]) -> set[frozenset[int]]:
    n = len(corner_vids)
    if n < 2:
        return set()
    return {
        frozenset((corner_vids[i], corner_vids[(i + 1) % n])) for i in range(n)
    }


def _coords_near(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    tol: float,
) -> bool:
    tol_sq = tol * tol
    return (
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    ) <= tol_sq


def _wall_top_corner_coords(
    wall: TileFace,
    *,
    rim_y_tol: float,
) -> list[tuple[float, float, float]]:
    ys = [float(c[1]) for c in wall.corners]
    if not ys:
        return []
    y_ref = max(ys)
    return [
        (float(c[0]), float(c[1]), float(c[2]))
        for c in wall.corners
        if abs(float(c[1]) - y_ref) <= rim_y_tol
    ]


def _any_corner_near_points(
    corners: Sequence[tuple[float, float, float]],
    targets: Sequence[tuple[float, float, float]],
    tol: float,
) -> bool:
    if not targets:
        return False
    for corner in corners:
        for target in targets:
            if _coords_near(corner, target, tol):
                return True
    return False


def ceiling_corner_near_wall_tops(
    ceiling: TileFace,
    walls: Sequence[TileFace],
    *,
    corner_tol: float,
    rim_y_tol: float,
) -> bool:
    """Fallback: any ceiling corner within corner_tol of a wall top corner."""

    wall_tops: list[tuple[float, float, float]] = []
    for wall in walls:
        wall_tops.extend(_wall_top_corner_coords(wall, rim_y_tol=rim_y_tol))
    if not wall_tops:
        return False
    ceiling_corners = [
        (float(c[0]), float(c[1]), float(c[2])) for c in ceiling.corners
    ]
    return _any_corner_near_points(ceiling_corners, wall_tops, corner_tol)


def ceiling_connects_to_walls(
    ceiling: TileFace,
    walls: Sequence[TileFace],
    *,
    corner_tol: float,
    rim_y_tol: float = 0.08,
) -> bool:
    """True when the ceiling meets wall tops by shared rim/edge or corner proximity."""

    if not walls:
        return False

    bundle = [*walls, ceiling]
    per_tile_vids = cluster_tile_corners(bundle, corner_tol)
    ceiling_vids = per_tile_vids[-1]
    ceiling_rim = _rim_edges(
        ceiling, ceiling_vids, rim_y_tol=rim_y_tol, prefer_top=True
    )
    ceiling_edges = tile_undirected_edges(ceiling_vids)

    wall_top_rims: set[frozenset[int]] = set()
    for wi in range(len(walls)):
        wall_top_rims |= _rim_edges(
            walls[wi],
            per_tile_vids[wi],
            rim_y_tol=rim_y_tol,
            prefer_top=True,
        )
        if ceiling_edges.intersection(tile_undirected_edges(per_tile_vids[wi])):
            return True

    if ceiling_rim.intersection(wall_top_rims):
        return True

    return ceiling_corner_near_wall_tops(
        ceiling, walls, corner_tol=corner_tol, rim_y_tol=rim_y_tol
    )


def _ceiling_indices_anchored_to_walls(
    ceilings: Sequence[TileFace],
    walls: Sequence[TileFace],
    *,
    corner_tol: float,
    rim_y_tol: float,
) -> set[int]:
    """Ceiling tile indices that connect to walls, plus ceiling-only neighbours."""

    if not ceilings:
        return set()

    wall_anchored = {
        i
        for i, ceiling in enumerate(ceilings)
        if ceiling_connects_to_walls(
            ceiling, walls, corner_tol=corner_tol, rim_y_tol=rim_y_tol
        )
    }
    if not wall_anchored:
        return set()

    per_tile_vids = cluster_tile_corners(ceilings, corner_tol)
    edge_sets = [tile_undirected_edges(vids) for vids in per_tile_vids]
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(ceilings))}
    for i in range(len(ceilings)):
        for j in range(i + 1, len(ceilings)):
            if edge_sets[i].intersection(edge_sets[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    kept = set(wall_anchored)
    stack = list(wall_anchored)
    while stack:
        node = stack.pop()
        for nb in adjacency.get(node, ()):
            if nb not in kept:
                kept.add(nb)
                stack.append(nb)
    return kept


def filter_unconnected_ceiling_tiles(
    tiles: Sequence[TileFace],
    *,
    corner_tol: float = 0.05,
    rim_y_tol: float = 0.08,
    max_ceiling_clearance_m: float = 1.0,
) -> tuple[list[TileFace], tuple[str, ...]]:
    """Drop ceiling tiles not anchored to wall tops.

    A ceiling is kept when it meets wall tops directly, or shares an edge with
    another ceiling that is (transitively) wall-anchored.
    """

    walls = [t for t in tiles if t.source == "wall"]
    ceilings = [t for t in tiles if t.source in CEILING_SOURCES]
    anchored = _ceiling_indices_anchored_to_walls(
        ceilings, walls, corner_tol=corner_tol, rim_y_tol=rim_y_tol
    )
    anchored_locators = {ceilings[i].locator_id for i in anchored}

    kept: list[TileFace] = []
    dropped: list[str] = []

    for tile in tiles:
        if tile.source not in CEILING_SOURCES:
            kept.append(tile)
            continue
        if tile.locator_id not in anchored_locators:
            dropped.append(tile.locator_id)
            continue
        clearance = _min_ceiling_clearance_m(walls, [tile], rim_y_tol=rim_y_tol)
        if clearance is not None and clearance > max_ceiling_clearance_m:
            dropped.append(tile.locator_id)
            continue
        kept.append(tile)

    return kept, tuple(dropped)


def coherence_issues_to_segments(
    issues: Sequence[TileCoherenceIssue],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for issue in issues:
        if issue.edge is None:
            continue
        a, b = issue.edge
        segments.append(
            {
                "a": [float(a[0]), float(a[1]), float(a[2])],
                "b": [float(b[0]), float(b[1]), float(b[2])],
                "kind": issue.kind,
            }
        )
    return segments


def audit_room_tile_coherence(
    tiles: Sequence[TileFace],
    *,
    corner_tol: float = 0.05,
    rim_y_tol: float = 0.08,
    max_ceiling_clearance_m: float = 1.0,
) -> RoomTileCoherenceResult:
    """Return whether floor, walls, and ceiling form one connected corner shell."""

    issues: list[TileCoherenceIssue] = []
    floors = [t for t in tiles if t.source == "floor"]
    walls = [t for t in tiles if t.source == "wall"]
    ceilings = [t for t in tiles if t.source in CEILING_SOURCES]

    if not floors:
        issues.append(
            TileCoherenceIssue(kind="missing_floor", message="room has no floor tile")
        )
    if not walls:
        issues.append(
            TileCoherenceIssue(kind="missing_walls", message="room has no wall tiles")
        )
    if not ceilings:
        issues.append(
            TileCoherenceIssue(
                kind="missing_ceiling", message="room has no ceiling tile"
            )
        )

    if not tiles:
        return RoomTileCoherenceResult(
            ok=False,
            issues=tuple(issues),
            component_count=0,
            shared_edge_count=0,
            ceiling_clearance_m=None,
        )

    per_tile_vids = cluster_tile_corners(tiles, corner_tol)
    edge_sets = [tile_undirected_edges(vids) for vids in per_tile_vids]

    floor_indices = [i for i, t in enumerate(tiles) if t.source == "floor"]
    wall_indices = [i for i, t in enumerate(tiles) if t.source == "wall"]
    ceiling_indices = [i for i, t in enumerate(tiles) if t.source in CEILING_SOURCES]

    shared_edge_count = 0
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(tiles))}
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            shared = edge_sets[i].intersection(edge_sets[j])
            if shared:
                shared_edge_count += len(shared)
                adjacency[i].add(j)
                adjacency[j].add(i)

    for wi in wall_indices:
        wall = tiles[wi]
        for ci in ceiling_indices:
            ceiling = tiles[ci]
            if adjacency[wi].intersection({ci}):
                continue
            if ceiling_corner_near_wall_tops(
                ceiling,
                [wall],
                corner_tol=corner_tol,
                rim_y_tol=rim_y_tol,
            ):
                adjacency[wi].add(ci)
                adjacency[ci].add(wi)

    component_count = _connected_components(len(tiles), adjacency)
    if component_count > 1:
        issues.append(
            TileCoherenceIssue(
                kind="tile_graph_disconnected",
                message=(
                    f"tiles form {component_count} disconnected components "
                    f"({shared_edge_count} shared edges)"
                ),
            )
        )

    for idx, degree in enumerate(len(adjacency[i]) for i in range(len(tiles))):
        if degree == 0 and len(tiles) > 1:
            issues.append(
                TileCoherenceIssue(
                    kind="isolated_tile",
                    message="tile shares no edge with any neighbour",
                    tile_locator_ids=(tiles[idx].locator_id,),
                )
            )

    if floors and walls:
        issues.extend(
            _check_floor_wall_rims(
                tiles,
                floor_indices,
                wall_indices,
                per_tile_vids,
                rim_y_tol=rim_y_tol,
            )
        )

    if walls and ceilings:
        issues.extend(
            _check_wall_ceiling_rims(
                tiles,
                wall_indices,
                ceiling_indices,
                per_tile_vids,
                corner_tol=corner_tol,
                rim_y_tol=rim_y_tol,
            )
        )

    ceiling_clearance_m: float | None = None
    if walls and ceilings:
        ceiling_clearance_m = _min_ceiling_clearance_m(walls, ceilings, rim_y_tol=rim_y_tol)
        if (
            ceiling_clearance_m is not None
            and ceiling_clearance_m > max_ceiling_clearance_m
        ):
            issues.append(
                TileCoherenceIssue(
                    kind="ceiling_far_from_wall_tops",
                    message=(
                        f"ceiling sits {ceiling_clearance_m:.2f} m above wall tops "
                        f"(max {max_ceiling_clearance_m:.2f} m)"
                    ),
                    tile_locator_ids=tuple(c.locator_id for c in ceilings),
                )
            )

    if floors and walls and ceilings and not _shell_reachable(
        floor_indices, wall_indices, ceiling_indices, adjacency
    ):
        issues.append(
            TileCoherenceIssue(
                kind="tile_graph_disconnected",
                message="floor, walls, and ceiling are not in one connected tile graph",
            )
        )

    return RoomTileCoherenceResult(
        ok=not issues,
        issues=tuple(issues),
        component_count=component_count,
        shared_edge_count=shared_edge_count,
        ceiling_clearance_m=ceiling_clearance_m,
    )


def _connected_components(n: int, adjacency: dict[int, set[int]]) -> int:
    seen: set[int] = set()
    count = 0
    for start in range(n):
        if start in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(nb for nb in adjacency.get(node, ()) if nb not in seen)
    return count


def _shell_reachable(
    floor_indices: list[int],
    wall_indices: list[int],
    ceiling_indices: list[int],
    adjacency: dict[int, set[int]],
) -> bool:
    if not floor_indices or not wall_indices or not ceiling_indices:
        return True

    def bfs_from(start: int, targets: set[int]) -> set[int]:
        seen = {start}
        stack = [start]
        hit: set[int] = set()
        while stack:
            node = stack.pop()
            if node in targets:
                hit.add(node)
            for nb in adjacency.get(node, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return hit

    walls_hit = bfs_from(floor_indices[0], set(wall_indices))
    if not walls_hit:
        return False
    ceilings_hit: set[int] = set()
    for w in walls_hit:
        ceilings_hit |= bfs_from(w, set(ceiling_indices))
    return bool(ceilings_hit)


def _rim_cluster_ids(
    tile: TileFace,
    corner_vids: list[int],
    *,
    rim_y_tol: float,
    prefer_top: bool,
) -> set[int]:
    ys = [float(c[1]) for c in tile.corners]
    if not ys:
        return set()
    y_ref = max(ys) if prefer_top else min(ys)
    return {
        corner_vids[i]
        for i, y in enumerate(ys)
        if abs(y - y_ref) <= rim_y_tol
    }


def _rim_edges(
    tile: TileFace,
    corner_vids: list[int],
    *,
    rim_y_tol: float,
    prefer_top: bool,
) -> set[frozenset[int]]:
    rim_vids = _rim_cluster_ids(tile, corner_vids, rim_y_tol=rim_y_tol, prefer_top=prefer_top)
    n = len(corner_vids)
    out: set[frozenset[int]] = set()
    for i in range(n):
        v0 = corner_vids[i]
        v1 = corner_vids[(i + 1) % n]
        if v0 in rim_vids and v1 in rim_vids:
            out.add(frozenset((v0, v1)))
    return out


def _segment_for_edge(
    tile: TileFace,
    corner_vids: list[int],
    edge: frozenset[int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    n = len(corner_vids)
    for i in range(n):
        pair = frozenset((corner_vids[i], corner_vids[(i + 1) % n]))
        if pair == edge:
            return tile.corners[i], tile.corners[(i + 1) % n]
    verts = sorted(edge)
    return tile.corners[0], tile.corners[min(1, n - 1)]


def _collect_rim_edges(
    indices: list[int],
    tiles: Sequence[TileFace],
    per_tile_vids: list[list[int]],
    *,
    rim_y_tol: float,
    prefer_top: bool,
) -> set[frozenset[int]]:
    out: set[frozenset[int]] = set()
    for idx in indices:
        out |= _rim_edges(
            tiles[idx],
            per_tile_vids[idx],
            rim_y_tol=rim_y_tol,
            prefer_top=prefer_top,
        )
    return out


def _check_floor_wall_rims(
    tiles: Sequence[TileFace],
    floor_indices: list[int],
    wall_indices: list[int],
    per_tile_vids: list[list[int]],
    *,
    rim_y_tol: float,
) -> list[TileCoherenceIssue]:
    issues: list[TileCoherenceIssue] = []
    wall_bottom = _collect_rim_edges(
        wall_indices, tiles, per_tile_vids, rim_y_tol=rim_y_tol, prefer_top=False
    )
    for fi in floor_indices:
        floor = tiles[fi]
        vids = per_tile_vids[fi]
        for edge in tile_undirected_edges(vids):
            if edge in wall_bottom:
                continue
            issues.append(
                TileCoherenceIssue(
                    kind="floor_wall_gap",
                    message="floor edge has no matching wall bottom edge",
                    tile_locator_ids=(floor.locator_id,),
                    edge=_segment_for_edge(floor, vids, edge),
                )
            )
    return issues


def _check_wall_ceiling_rims(
    tiles: Sequence[TileFace],
    wall_indices: list[int],
    ceiling_indices: list[int],
    per_tile_vids: list[list[int]],
    *,
    corner_tol: float,
    rim_y_tol: float,
) -> list[TileCoherenceIssue]:
    issues: list[TileCoherenceIssue] = []
    ceiling_rims = _collect_rim_edges(
        ceiling_indices, tiles, per_tile_vids, rim_y_tol=rim_y_tol, prefer_top=True
    )
    ceiling_corners: list[tuple[float, float, float]] = []
    for ci in ceiling_indices:
        for c in tiles[ci].corners:
            ceiling_corners.append((float(c[0]), float(c[1]), float(c[2])))

    for wi in wall_indices:
        wall = tiles[wi]
        vids = per_tile_vids[wi]
        for edge in _rim_edges(wall, vids, rim_y_tol=rim_y_tol, prefer_top=True):
            if edge in ceiling_rims:
                continue
            a, b = _segment_for_edge(wall, vids, edge)
            if _any_corner_near_points(ceiling_corners, [a, b], corner_tol):
                continue
            issues.append(
                TileCoherenceIssue(
                    kind="wall_ceiling_gap",
                    message="wall top edge has no matching ceiling edge",
                    tile_locator_ids=(wall.locator_id,),
                    edge=(a, b),
                )
            )
    return issues


def _min_ceiling_clearance_m(
    walls: Sequence[TileFace],
    ceilings: Sequence[TileFace],
    *,
    rim_y_tol: float,
) -> float | None:
    wall_top_y: float | None = None
    for wall in walls:
        rim_ys = [
            float(c[1])
            for c in wall.corners
            if abs(float(c[1]) - max(float(x[1]) for x in wall.corners)) <= rim_y_tol
        ]
        if rim_ys:
            local_max = max(rim_ys)
            wall_top_y = local_max if wall_top_y is None else max(wall_top_y, local_max)
    if wall_top_y is None:
        return None

    min_clearance: float | None = None
    for ceiling in ceilings:
        for c in ceiling.corners:
            clearance = float(c[1]) - wall_top_y
            if min_clearance is None or clearance < min_clearance:
                min_clearance = clearance
    return min_clearance
