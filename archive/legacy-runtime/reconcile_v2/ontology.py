"""Ontology vocabulary for topology-v2 graph enrichment."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypedDict


class NodeType(StrEnum):
    BUILDING = "building"
    STORY = "story"
    ROOM = "room"
    WALL = "wall"
    SLAB = "slab"
    OPENING = "opening"
    ROOF_SURFACE = "roof_surface"
    CEILING_SURFACE = "ceiling_surface"
    GAP = "gap"
    CLOSURE = "closure"
    SEGMENT = "segment"
    CELL = "cell"


class SurfaceRole(StrEnum):
    EXTERIOR_WALL = "exterior_wall"
    INTERIOR_WALL = "interior_wall"
    PARTY_WALL = "party_wall"
    FLOOR_SLAB = "floor_slab"
    CEILING_SLAB = "ceiling_slab"
    ROOF_FLAT = "roof_flat"
    ROOF_OBLIQUE = "roof_oblique"
    GROUND = "ground"
    CLOSURE = "closure"
    VIRTUAL = "virtual"


class EdgeType(StrEnum):
    CONTAINS = "contains"
    BOUNDED_BY = "bounded_by"
    ADJACENT_TO = "adjacent_to"
    ABOVE = "above"
    BELOW = "below"
    CONNECTS_TO = "connects_to"
    VOIDS = "voids"
    FILLS = "fills"
    EXPOSES_TO = "exposes_to"
    INTERFERES = "interferes"
    CONTINUES_AS = "continues_as"


class BoundaryKind(StrEnum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    CLOSURE = "closure"


class JunctionType(StrEnum):
    L = "L"
    T = "T"
    CROSS = "cross"
    BUTT = "butt"


class Exposure(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNDERGROUND = "underground"
    SEMI_EXPOSED = "semi_exposed"


class RoomProps(TypedDict, total=False):
    floor_height_y: float
    ceiling_height_y: float
    is_top_story: bool
    is_ground_story: bool
    area_m2: float
    volume_m3: float


class WallProps(TypedDict, total=False):
    azimuth_deg: float
    inclination_deg: float
    height_m: float
    thickness_m: float
    surface_role: str
    length_m: float


class StoryProps(TypedDict, total=False):
    story_index: int
    floor_y: float
    ceiling_y: float
    height_m: float
    room_count: int


class GapProps(TypedDict, total=False):
    gap_kind: Literal["intra_story", "cross_story"]
    width_m: float
    area_m2: float
    recommended_action: str
