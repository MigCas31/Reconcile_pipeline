"""Data classes for RoomPlan geometry reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_list(cls, lst: list[float]) -> Vec3:
        return cls(x=lst[0], y=lst[1], z=lst[2])


@dataclass
class Transform:
    matrix: np.ndarray  # (4,4) column-major convention

    def apply(self, point: Vec3) -> Vec3:
        """Transform a local-space point to world coordinates."""
        p = np.array([point.x, point.y, point.z, 1.0])
        result = self.matrix @ p
        return Vec3(result[0], result[1], result[2])

    def apply_corners(self, corners: list[Vec3]) -> list[Vec3]:
        return [self.apply(c) for c in corners]

    @property
    def translation(self) -> Vec3:
        return Vec3(self.matrix[0, 3], self.matrix[1, 3], self.matrix[2, 3])

    @classmethod
    def from_flat(cls, flat: list[float]) -> Transform:
        """Parse Apple's column-major flat 16-element array into 4x4 matrix."""
        return cls(matrix=np.array(flat).reshape(4, 4, order="F"))

    @classmethod
    def identity(cls) -> Transform:
        return cls(matrix=np.eye(4))

    def to_flat(self) -> list[float]:
        """Serialize back to Apple's column-major flat array."""
        return self.matrix.flatten(order="F").tolist()


@dataclass
class Surface:
    identifier: str
    category: dict
    confidence: dict
    dimensions: Vec3  # width (x), height (y), depth (z)
    transform: Transform
    polygon_corners: list[Vec3]
    story: int
    completed_edges: list
    parent_identifier: str | None = None
    curve: Any | None = None

    @property
    def surface_type(self) -> str:
        """Return 'wall', 'door', 'window', etc."""
        return next(iter(self.category.keys()), "unknown")


@dataclass
class Floor:
    identifier: str
    category: dict
    confidence: dict
    dimensions: Vec3
    transform: Transform
    polygon_corners: list[Vec3]
    story: int
    completed_edges: list
    parent_identifier: str | None = None
    curve: Any | None = None


@dataclass
class Room:
    identifier: str | None
    walls: list[Surface] = field(default_factory=list)
    doors: list[Surface] = field(default_factory=list)
    windows: list[Surface] = field(default_factory=list)
    floors: list[Floor] = field(default_factory=list)
    story: int = 0
    reference_origin_transform: Transform | None = None


@dataclass
class Building:
    rooms: list[Room] = field(default_factory=list)
    top_level_walls: list[Surface] = field(default_factory=list)
    top_level_doors: list[Surface] = field(default_factory=list)
    top_level_windows: list[Surface] = field(default_factory=list)
    top_level_floors: list[Floor] = field(default_factory=list)
    version: int = 2


@dataclass
class MatchResult:
    merged_surface: Surface
    raw_surface: Surface | None  # None if only in merged (synthesized)
    room_id: str | None
    surface_type: str  # wall, door, window


@dataclass
class MergeReport:
    surface_id: str
    surface_type: str
    fields_from_raw: list[str] = field(default_factory=list)
    fields_from_merged: list[str] = field(default_factory=list)
    divergences: dict[str, float] = field(default_factory=dict)  # field -> delta in cm
    flags: list[str] = field(default_factory=list)
