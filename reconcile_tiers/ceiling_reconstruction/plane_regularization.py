"""Detect and regularize planar shapes from room tiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from reconcile_tiers._core.plane import Plane, planes_equivalent
from reconcile_tiers.polyhedron.manifold_repair import TileFace

_HORIZONTAL_MIN_NY = 0.996  # cos(~5 deg)
_VERTICAL_MAX_NY = 0.174  # sin(~10 deg)
_COPLANAR_OFFSET_RATIO = 0.01


@dataclass(slots=True)
class PlaneGroup:
    plane: Plane
    tiles: list[TileFace] = field(default_factory=list)
    kind: str = "unknown"  # floor | wall | ceiling | other

    @property
    def tile_count(self) -> int:
        return len(self.tiles)


def _classify_plane(plane: Plane) -> str:
    ny = abs(float(plane.b))
    if ny >= _HORIZONTAL_MIN_NY:
        return "ceiling" if plane.b > 0 else "floor"
    if ny <= _VERTICAL_MAX_NY:
        return "wall"
    return "other"


def _bbox_diagonal(tiles: Sequence[TileFace]) -> float:
    points = [c for tile in tiles for c in tile.corners]
    if not points:
        return 1.0
    arr = np.asarray(points, dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _normalize_plane(plane: Plane) -> Plane:
    normal = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return plane
    normal /= norm
    d = float(plane.d) / norm
    return Plane(a=float(normal[0]), b=float(normal[1]), c=float(normal[2]), d=d)


def detect_plane_groups(tiles: Sequence[TileFace]) -> list[PlaneGroup]:
    """Group tiles by equivalent plane equation."""
    groups: list[PlaneGroup] = []
    for tile in tiles:
        plane = _normalize_plane(tile.plane)
        matched: PlaneGroup | None = None
        for group in groups:
            if planes_equivalent(group.plane, plane):
                matched = group
                break
        if matched is None:
            kind = _classify_plane(plane)
            if tile.source == "floor":
                kind = "floor"
            elif tile.source in ("ceiling", "visual_shell", "gable_closure"):
                kind = "ceiling"
            elif tile.source == "wall":
                kind = "wall"
            groups.append(PlaneGroup(plane=plane, tiles=[tile], kind=kind))
        else:
            matched.tiles.append(tile)
    return groups


def _merge_coplanar_groups(
    groups: list[PlaneGroup],
    *,
    max_offset_m: float,
) -> list[PlaneGroup]:
    merged: list[PlaneGroup] = []
    for group in groups:
        target: PlaneGroup | None = None
        for existing in merged:
            if existing.kind != group.kind:
                continue
            n1 = np.array(
                [existing.plane.a, existing.plane.b, existing.plane.c], dtype=float
            )
            n2 = np.array([group.plane.a, group.plane.b, group.plane.c], dtype=float)
            if abs(float(n1 @ n2) - 1.0) > 0.05:
                continue
            if abs(existing.plane.d - group.plane.d) <= max_offset_m:
                target = existing
                break
        if target is None:
            merged.append(
                PlaneGroup(plane=group.plane, tiles=list(group.tiles), kind=group.kind)
            )
        else:
            target.tiles.extend(group.tiles)
            target.plane = _average_plane([target.plane, group.plane])
    return merged


def _average_plane(planes: Sequence[Plane]) -> Plane:
    normals = np.asarray([[p.a, p.b, p.c] for p in planes], dtype=float)
    avg_n = normals.mean(axis=0)
    norm = float(np.linalg.norm(avg_n))
    if norm <= 1e-12:
        return planes[0]
    avg_n /= norm
    avg_d = float(np.mean([p.d for p in planes]))
    return Plane(
        a=float(avg_n[0]),
        b=float(avg_n[1]),
        c=float(avg_n[2]),
        d=avg_d,
    )


def _snap_walls_to_floor_orthogonal(
    groups: list[PlaneGroup],
) -> list[PlaneGroup]:
    """Adjust wall normals to be perpendicular to the floor plane (Y-up)."""
    floors = [g for g in groups if g.kind == "floor"]
    walls = [g for g in groups if g.kind == "wall"]
    if not floors or not walls:
        return groups
    dominant = max(floors, key=lambda g: g.tile_count)
    up = np.array(
        [dominant.plane.a, dominant.plane.b, dominant.plane.c], dtype=float
    )
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-12:
        return groups
    up /= up_norm
    if up[1] < 0:
        up *= -1.0

    out: list[PlaneGroup] = []
    for group in groups:
        if group.kind != "wall":
            out.append(group)
            continue
        n = np.array([group.plane.a, group.plane.b, group.plane.c], dtype=float)
        n -= up * float(n @ up)
        norm = float(np.linalg.norm(n))
        if norm <= 1e-12:
            out.append(group)
            continue
        n /= norm
        d = float(np.mean([group.plane.d]))
        out.append(
            PlaneGroup(
                plane=Plane(a=float(n[0]), b=float(n[1]), c=float(n[2]), d=d),
                tiles=list(group.tiles),
                kind=group.kind,
            )
        )
    return out


def regularize_planes(
    tiles: Sequence[TileFace],
    *,
    coplanar_offset_ratio: float = _COPLANAR_OFFSET_RATIO,
) -> tuple[list[PlaneGroup], list[Plane]]:
    """Detect, merge coplanar, and regularize planes from room tiles."""
    groups = detect_plane_groups(tiles)
    diag = _bbox_diagonal(tiles)
    max_offset = max(0.05, coplanar_offset_ratio * diag)
    groups = _merge_coplanar_groups(groups, max_offset_m=max_offset)
    groups = _snap_walls_to_floor_orthogonal(groups)
    planes = [_normalize_plane(g.plane) for g in groups]
    return groups, planes
