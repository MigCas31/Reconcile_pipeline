"""Domain models for room postprocessing (corner-sharing graph)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BuildingElement:
    """One renderable polygon from tier_payload (floor, wall, ceiling, shell, …)."""

    id: str
    kind: str
    locator_id: str | None
    corners: tuple[tuple[float, float, float], ...]
    room_index: int | None = None
    story: int | None = None
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class CornerGraphExport:
    """JSON-serializable corner graph (API / viewer)."""

    building_uuid: str
    corner_tol: float
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    wall_edge_segments: list[dict[str, Any]]
