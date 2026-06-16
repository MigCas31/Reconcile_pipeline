"""KSR-specific tile collection: structure (wall+floor) vs evidence (ceiling/shell)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from reconcile_tiers.polyhedron.kinetic_partition import BoundingPrism
from reconcile_tiers.polyhedron.manifold_repair import TileFace, collect_room_tiles

_STRUCTURE_SOURCES = frozenset({"wall", "floor"})
_EVIDENCE_SOURCES = frozenset({"ceiling", "visual_shell", "gable_closure"})


@dataclass(slots=True)
class KsrRoomTiles:
    structure: list[TileFace]
    evidence: list[TileFace]

    @property
    def all_tiles(self) -> list[TileFace]:
        return list(self.structure) + list(self.evidence)


def collect_ksr_room_tiles(
    payload: Mapping[str, object],
    room: Mapping[str, object],
    *,
    corner_tol: float = 0.02,
) -> KsrRoomTiles:
    """Split segment-tier room tiles into structure and graph-cut evidence."""
    all_tiles = collect_room_tiles(payload, room, corner_tol=corner_tol)
    structure = [t for t in all_tiles if t.source in _STRUCTURE_SOURCES]
    evidence = [t for t in all_tiles if t.source in _EVIDENCE_SOURCES]
    return KsrRoomTiles(structure=structure, evidence=evidence)


def has_minimum_structure(tiles: KsrRoomTiles, *, min_walls: int = 3) -> bool:
    wall_count = sum(1 for t in tiles.structure if t.source == "wall")
    return wall_count >= min_walls


def room_bounding_prism(
    structure_tiles: Sequence[TileFace],
    *,
    margin: float = 1.0,
) -> BoundingPrism:
    """Tight axis-aligned prism from floor footprint and wall height only."""
    if not structure_tiles:
        raise ValueError("cannot infer room bounding prism from empty structure tiles")

    floor_corners: list[tuple[float, float, float]] = []
    wall_corners: list[tuple[float, float, float]] = []
    for tile in structure_tiles:
        if tile.source == "floor":
            floor_corners.extend(tile.corners)
        elif tile.source == "wall":
            wall_corners.extend(tile.corners)

    footprint = floor_corners if floor_corners else wall_corners
    if not footprint:
        raise ValueError("structure tiles contain no floor or wall corners")

    fp = np.asarray(footprint, dtype=float)
    x_min, x_max = float(fp[:, 0].min()), float(fp[:, 0].max())
    z_min, z_max = float(fp[:, 2].min()), float(fp[:, 2].max())

    if wall_corners:
        wc = np.asarray(wall_corners, dtype=float)
        y_min = float(wc[:, 1].min())
        y_max = float(wc[:, 1].max())
    else:
        y_min = float(fp[:, 1].min())
        y_max = float(fp[:, 1].max())

    xz_pad = max(0.05, margin * 0.1)
    y_pad = float(margin)
    return BoundingPrism(
        x_min=x_min - xz_pad,
        x_max=x_max + xz_pad,
        y_min=y_min - y_pad,
        y_max=y_max + y_pad,
        z_min=z_min - xz_pad,
        z_max=z_max + xz_pad,
    )
